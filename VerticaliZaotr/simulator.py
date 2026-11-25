"""V5 standing wheelchair simulator: MuJoCo window + control panel in the browser.

    python simulator.py              everything in one browser page (video + control panel)
    mjpython simulator.py --viewer   separate MuJoCo window + panel (old mode)

The page (http://127.0.0.1:8765) shows the simulation video with the joystick, seat
up/down, brake, emergency stop, speed level, person/chair settings, floor slope, pushes,
the test track, live telemetry and plots.
"""
import argparse, io, json, math, os, pathlib, struct, sys, threading, time, webbrowser, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import mujoco

import build_human as bh
from parts import Parts

HERE = pathlib.Path(__file__).resolve().parent
MODEL = HERE / "chair_human.xml"
PORT = 8765

WHEELS = ["drive_left", "drive_right", "front_left", "front_right"]
TRACK = 0.525                     # m, drive wheel track
BATTERY_V, BATTERY_AH = 25.6, 20.0
LIFT_IDLE_A, ETA_LIFT = 1.25, 0.30
LEVEL_FRACTIONS = [0.25, 0.42, 0.6, 0.8, 1.0]  # speed levels 1..5 as a share of the top speed
TOP_SPEED_CAP = 1.67              # m/s = 6 km/h, usual limit for power wheelchairs
PEAK_TIME = 5.0                   # s, how long the controller may give its peak current
PEAK_RECOVERY = 0.25              # peak budget refills at 1/4 of real time below rated torque
TURN_RATE = 1.6                   # rad/s at full joystick, scaled with level
OBSTACLE_STALL_TIME = 0.35        # s below 45% of ramped command before traction mode starts
OBSTACLE_TORQUE_MULT = 2.0        # short peak for quasi-static obstacle climbing
OBSTACLE_KV = 6400.0             # low-speed response; motor current limits still apply
# Longitudinal grip in obstacle mode versus scrub grip for ordinary skid steering.
# Both were 0.5, so the point-contact tyres kept slipping against the step.
ROAD_FRICTION, SKID_FRICTION = 0.8, 0.5
SKID_YAW_GAIN = 180.0            # N*m per rad/s error; represents lateral track/tyre forces
SKID_YAW_TORQUE = 250.0          # N*m cap for the simplified skid-steer contact model
ACCEL, DECEL, EBRAKE_DECEL = 0.6, 1.0, 2.5   # m/s^2
STANDING_SPEED = 0.35             # m/s, speed limit when the seat is raised > 5 deg


# ------------------------------------------------------------------------------- telemetry
class Telemetry:
    def __init__(self, m):
        self.reload(m)

    def reload(self, m):
        self.m = m
        J = mujoco.mjtObj.mjOBJ_JOINT
        jid = lambda n: mujoco.mj_name2id(m, J, n)
        self.frame = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "frame")
        self.free = m.jnt_type[m.body_jntadr[self.frame]] == mujoco.mjtJoint.mjJNT_FREE
        self.dof = {n: m.jnt_dofadr[jid(n)] for n in
                    ["drive_left_spin", "drive_right_spin", "front_left_spin", "front_right_spin",
                     "actuator_extension"]}
        self.qadr = {n: m.jnt_qposadr[jid(n)] for n in ["lift_angle", "actuator_extension"]}
        self.lift_joints = ["actuator_extension"]
        if jid("actuator_extension_right") >= 0:
            self.lift_joints.append("actuator_extension_right")
        for name in self.lift_joints:
            self.dof[name], self.qadr[name] = int(m.jnt_dofadr[jid(name)]), int(m.jnt_qposadr[jid(name)])
        self.hip_dof = m.jnt_dofadr[jid("hip")] if jid("hip") >= 0 else None
        self.wheel_geom = {}
        for g in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for w in WHEELS:
                if n == f"{w}_contact":
                    self.wheel_geom[g] = w
        ground_names = ["floor", "threshold_30mm", "ramp_up_1_12", "ramp_down_1_8", "platform", "long_ramp_15_up", "long_ramp_15_top"]
        ground_names += [f"hill_{deg}_{part}" for deg in (10, 15, 30) for part in ("up", "top", "down")]
        self.ground = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ground_names]
        self.ground = [g for g in self.ground if g >= 0]
        self.foot_geoms = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("h_foot_L", "h_foot_R")} - {-1}
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        self.wheel_xy = {}
        for g, w in self.wheel_geom.items():
            p = d0.geom_xpos[g] - d0.xpos[self.frame]
            r = float(m.geom_size[g][0])
            self.wheel_xy[w] = [round(float(p[0]), 3), round(float(p[1]), 3), round(r, 3)]
        hb = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("human")]
        self.human_mass = float(m.body_mass[hb].sum())
        self.total_mass = float(m.body_mass.sum())
        self._f6 = np.zeros(6)
        self._display_com_local = None
        self._display_hull = []
        self._display_support = []
        self._display_support_time = -1.0

    def compute(self, d):
        m, t = self.m, {}
        R = d.xmat[self.frame].reshape(3, 3)
        p0 = d.xpos[self.frame]
        v6 = np.zeros(6)
        # world-frame velocity (the body's inertial frame may be rotated), projected on the chassis x axis
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.frame, v6, 0)
        t["speed"] = float(np.dot(v6[3:], R[:, 0])) if self.free else 0.0
        t["yaw_rate"] = math.degrees(v6[2]) if self.free else 0.0
        t["pitch"] = math.degrees(math.asin(max(-1, min(1, -R[2, 0]))))
        t["roll"] = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        t["wheels"] = {}
        for wheel in WHEELS:
            dof = self.dof[f"{wheel}_spin"]
            omega = float(d.qvel[dof])
            torque = float(d.qfrc_actuator[dof])
            t["wheels"][wheel] = dict(rpm=omega * 60 / (2 * math.pi),
                                      torque=torque, power=torque * omega)
        for side in ("left", "right"):
            rear_dof = self.dof[f"drive_{side}_spin"]
            front_dof = self.dof[f"front_{side}_spin"]
            w, wf = float(d.qvel[rear_dof]), float(d.qvel[front_dof])
            tau = float(d.qfrc_actuator[rear_dof])
            tau_f = float(d.qfrc_actuator[front_dof])
            t[f"rpm_{side}"] = w * 60 / (2 * math.pi)
            # Total side effort expressed as equivalent torque on the 320 mm rear wheel.
            t[f"torque_{side}"] = tau + tau_f
            t[f"power_{side}"] = tau * w + tau_f * wf
        t["seat_angle"] = math.degrees(d.qpos[self.qadr["lift_angle"]])
        t["lifts"] = []
        for joint in self.lift_joints:
            force = float(d.qfrc_actuator[self.dof[joint]])
            velocity = float(d.qvel[self.dof[joint]])
            t["lifts"].append(dict(force=force, speed=velocity,
                                   stroke=float(d.qpos[self.qadr[joint]]), power=force * velocity))
        t["stroke"] = sum(a["stroke"] for a in t["lifts"]) / len(t["lifts"])
        t["stroke_speed"] = sum(a["speed"] for a in t["lifts"]) / len(t["lifts"])
        t["lift_force"] = sum(a["force"] for a in t["lifts"])
        t["lift_power"] = sum(a["power"] for a in t["lifts"])
        t["lift_skew_mm"] = (max(a["stroke"] for a in t["lifts"]) - min(a["stroke"] for a in t["lifts"])) * 1000
        # contact loads
        gravity_dir = m.opt.gravity / np.linalg.norm(m.opt.gravity)
        loads = {w: 0.0 for w in WHEELS}
        contact_pts = {w: [] for w in WHEELS}
        feet = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 not in self.wheel_geom and g2 not in self.wheel_geom and g1 not in self.foot_geoms and g2 not in self.foot_geoms:
                continue
            mujoco.mj_contactForce(m, d, i, self._f6)
            fn = float(self._f6[0])
            for g in (g1, g2):
                if g in self.wheel_geom:
                    loads[self.wheel_geom[g]] += fn
                    contact_pts[self.wheel_geom[g]].append(c.pos.copy())
                if g in self.foot_geoms:
                    feet += fn
        # Use one point per wheel.  MuJoCo can omit a resting rigid-wheel contact for
        # an individual solver step; a wheel within 6 mm of the ground still belongs
        # to the support polygon.  This removes 4->0->4 display flicker without hiding
        # a real wheel lift (reported only after a 10 mm gap below).
        pts = []
        for gid, w in self.wheel_geom.items():
            if contact_pts[w]:
                pts.append(np.mean(contact_pts[w], axis=0))
            elif w.startswith(("drive", "front")):
                gap = min(mujoco.mj_geomDistance(m, d, gid, gg, 0.02, None) for gg in self.ground)
                if gap <= 0.006:
                    pts.append(d.geom_xpos[gid] + float(m.geom_size[gid][0]) * gravity_dir)
        t.update({f"load_{w}": loads[w] for w in WHEELS})
        t["feet_load"] = feet
        t["backrest_moment"] = abs(float(d.qfrc_constraint[self.hip_dof])) if self.hip_dof is not None else 0.0
        com = d.subtree_com[self.frame].copy()
        t["com"] = com.tolist()
        # stability: support polygon and CoM in a gravity-aligned frame
        g = gravity_dir
        t.update(self._stability(com, pts, g, R, p0))
        # backup supports: anti-tip wheels and footplate rollers catch the chair after a small lean;
        # the chair really falls over only when the CoM passes them
        touching = {self.wheel_geom.get(c.geom1) for c in d.contact[:d.ncon]} | {self.wheel_geom.get(c.geom2) for c in d.contact[:d.ncon]}
        extra = [d.geom_xpos[gid] + float(m.geom_size[gid][0]) * g for gid, w in self.wheel_geom.items()
                 if w.startswith(("antitip", "footroller")) and w not in touching]
        t["lean_fwd"], t["lean_back"] = t["tip_fwd"], t["tip_back"]
        if extra and len(pts) >= 3:
            st = self._stability(com, pts + extra, g, R, p0)
            for k in ("fwd", "back"):
                if st[f"tip_{k}"] is not None and t[f"tip_{k}"] is not None:
                    t[f"tip_{k}"] = max(t[f"tip_{k}"], st[f"tip_{k}"])
        # a wheel is "off the ground" only if there is a real gap (zero force alone happens with 4 rigid wheels)
        t["lifted"] = []
        if self.free:
            for g, w in self.wheel_geom.items():
                if w.startswith(("drive", "front")) and loads[w] < 1.0:
                    gap = min(mujoco.mj_geomDistance(m, d, g, gg, 0.05, None) for gg in self.ground)
                    if gap > 0.01:
                        t["lifted"].append(w)
        t["total_mass"], t["human_mass"] = self.total_mass, self.human_mass
        t["wheel_xy"] = self.wheel_xy
        # Smoothed coordinates are for drawing only.  Loads, tipping angles and
        # warnings above continue to use the instantaneous physical state.
        com_local = R.T @ (com - p0)
        if self._display_com_local is None or np.linalg.norm(com_local - self._display_com_local) > 0.5:
            self._display_com_local = com_local.copy()
        else:
            self._display_com_local += 0.18 * (com_local - self._display_com_local)
        display_com = p0 + R @ self._display_com_local
        visual = self._stability(display_com, pts, gravity_dir, R, p0)
        if visual["hull"]:
            self._display_hull = visual["hull"]
            self._display_support = visual["support3d"]
            self._display_support_time = d.time
        elif d.time - self._display_support_time > 0.2:
            self._display_hull = []
            self._display_support = []
        t["com_display"] = display_com.tolist()
        t["com_xy_display"] = visual["com_xy"]
        t["hull_display"] = self._display_hull
        t["support3d_display"] = self._display_support
        return t

    @staticmethod
    def _hull(P):
        P = sorted(set(map(tuple, np.round(P, 4))))
        if len(P) < 3:
            return P
        cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, up = [], []
        for p in P:
            while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0: lo.pop()
            lo.append(p)
        for p in reversed(P):
            while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0: up.pop()
            up.append(p)
        return lo[:-1] + up[:-1]

    def _stability(self, com, pts, g, R, p0):
        out = {"tip_fwd": None, "tip_back": None, "tip_side": None, "hull": [], "com_xy": None, "support3d": []}
        # chassis-aligned axes projected onto the plane normal to gravity
        fx = R[:, 0] - np.dot(R[:, 0], g) * g; fx /= np.linalg.norm(fx)
        fy = np.cross(-g, fx)
        to2 = lambda p: [float(np.dot(p - p0, fx)), float(np.dot(p - p0, fy))]
        c2 = np.array(to2(com))
        out["com_xy"] = c2.tolist()
        if len(pts) < 3:
            return out
        H = self._hull([to2(p) for p in pts])
        if len(H) < 3:
            return out
        out["hull"] = [list(h) for h in H]
        zm = float(np.mean([p[2] for p in pts]))
        out["support3d"] = [[x, y, zm] for x, y in self._hull([p[:2] for p in pts])]
        hgt = float(np.dot(com - np.mean(pts, axis=0), -g))
        best = {"fwd": 1e9, "back": 1e9, "side": 1e9}
        for i in range(len(H)):
            a, b = np.array(H[i]), np.array(H[(i + 1) % len(H)])
            e = b - a; n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
            dist = float(np.dot(a - c2, n))
            ang = math.degrees(math.atan2(dist, hgt))
            key = "fwd" if n[0] > 0.7 else "back" if n[0] < -0.7 else "side"
            best[key] = min(best[key], ang)
        for k, v in best.items():
            out[f"tip_{k}"] = v if v < 1e8 else None
        return out


# ------------------------------------------------------------------------------ simulator
class Sim:
    def __init__(self, mass, height, frame_mat=bh.DEFAULT_FRAME, act=bh.DEFAULT_ACTUATOR,
                 motor=bh.DEFAULT_MOTOR, motor_params=None, peak=1.0, chair=bh.DEFAULT_CHAIR):
        self.lock = threading.Lock()
        self.chair = chair
        self.model_version = 0
        self.mass, self.height, self.frame_mat, self.act = mass, height, frame_mat, act
        self.motor, self.motor_params, self.peak = motor, dict(motor_params or {}), float(peak)
        self.mp = bh.motor_profile(motor, self.motor_params)
        self.peak_budget = [PEAK_TIME, PEAK_TIME]
        self.torque_limit = [self.mp["wheel_torque"]] * 2
        self._write_model()
        self.m = mujoco.MjModel.from_xml_path(str(MODEL))
        self.d = mujoco.MjData(self.m)
        self.tel = Telemetry(self.m)
        self._ids()
        self.cmd = dict(jx=0.0, jy=0.0, lift=0, brake=False, park=False, estop=False, level=2,
                        timescale=1, slope_fwd=0.0, slope_side=0.0)
        self.last_input = time.time()
        self.events = []
        self.v_cmd = self.w_cmd = 0.0
        self.neutral_since = 0.0
        self.state = {}
        self.history = []
        self.energy_Wh = 0.0
        self.msg = ""
        self.push = None
        self.sat_since = 0
        self.fast_target = None          # stroke target of the quick seat slider (m)
        self.reset()

    # model handling --------------------------------------------------------------------
    def _write_model(self):
        bh.build(self.mass, self.height, self.frame_mat, self.act, self.motor, self.motor_params, chair=self.chair).write(
            MODEL, encoding="utf-8", xml_declaration=True)

    def _ids(self):
        m = self.m
        A = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.a_lift = A("lift_actuator")
        self.lift_acts = [self.a_lift]
        if A("lift_actuator_right") >= 0:
            self.lift_acts.append(A("lift_actuator_right"))
        self.lift_states = [int(m.actuator_actadr[a]) for a in self.lift_acts]
        self.drive_acts = [[A("drive_left"), A("drive_front_left")],
                           [A("drive_right"), A("drive_front_right")]]
        self.brake_acts = [A(n) for n in ("brake_left", "brake_front_left",
                                          "brake_right", "brake_front_right")]
        self.wdofs = [[self.tel.dof["drive_left_spin"], self.tel.dof["front_left_spin"]],
                      [self.tel.dof["drive_right_spin"], self.tel.dof["front_right_spin"]]]
        J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.wq = [J(n) for n in ("drive_left_spin", "front_left_spin",
                                  "drive_right_spin", "front_right_spin")]
        self.locks = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n) for n in
                      ("brake_lock_left", "brake_lock_front_left",
                       "brake_lock_right", "brake_lock_front_right")]
        self.locked = False
        self.frame = self.tel.frame
        self.lift_act = m.actuator_actadr[self.a_lift]
        self.lift_gear = float(m.actuator_gear[self.a_lift, 0])
        self.parts = Parts(m, self.frame_mat, bh.ACTUATORS[self.act], self.mp)
        self.selected_part = "chair"
        self.seat_pivot = m.body("seat").pos.copy()
        self._body_vel = np.zeros(6)
        self.drive_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{w}_contact")
                            for w in ("drive_left", "front_left", "drive_right", "front_right")]
        self.drive_geom_set = set(self.drive_geoms)
        self.obstacle_geoms = set(self.tel.ground) - {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")}

    def set_config(self, mass, height, frame_mat, act, motor=None, motor_params=None, peak=None, chair=None):
        """Rebuild physics and inertials; topology can change with actuator count."""
        self.mass, self.height, self.frame_mat, self.act = mass, height, frame_mat, act
        if chair:
            self.chair = chair
        if motor:
            self.motor = motor
        if motor_params is not None:
            self.motor_params = {k: float(v) for k, v in motor_params.items() if k in bh.MOTOR_FIELDS and v not in ("", None)}
        if peak is not None:
            self.peak = float(peak)
        self.mp = bh.motor_profile(self.motor, self.motor_params)
        self.peak_budget = [PEAK_TIME, PEAK_TIME]
        self._write_model()
        m2 = mujoco.MjModel.from_xml_path(str(MODEL))
        # Changes in the number of actuators alter joints, constraints and state sizes.
        self.m, self.d = m2, mujoco.MjData(m2)
        self.model_version += 1
        self.tel.reload(self.m)
        self._ids()
        self.reset()
        self.msg = (f"Человек {mass:.0f} кг / {height:.2f} м · рама: {bh.FRAMES[frame_mat]['name']} · "
                    f"привод: {bh.ACTUATORS[act]['name']} · мотор: {self.mp['name']}")

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[0], d.qpos[1] = x, y
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        self.settle_until = 0.5
        self.locked = False
        self.fast_target = None
        self.lift_velocity_cmd = 0.0
        self.lift_running = False
        self.v_cmd = self.w_cmd = 0.0
        self.peak_budget = [PEAK_TIME, PEAK_TIME]
        self.stall_time = self.recovered_time = 0.0
        self.obstacle_mode = False
        self.history.clear()
        self.energy_Wh = 0.0

    def teleport(self, x, y=0.0, yaw=0.0, z=0.0):
        d = self.d
        d.qpos[0], d.qpos[1], d.qpos[2] = x, y, z + 0.003
        self.settle_until = d.time + 0.5
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[:] = 0
        self.v_cmd = self.w_cmd = 0.0
        self.stall_time = self.recovered_time = 0.0
        self.obstacle_mode = False
        mujoco.mj_forward(self.m, d)

    # control ---------------------------------------------------------------------------
    def apply_controls(self, dt):
        m, d, c = self.m, self.d, self.cmd
        # dead-man: if the panel stops sending, centre the joystick and stop the seat
        if time.time() - self.last_input > 0.7:
            c["jx"] = c["jy"] = 0.0; c["lift"] = 0
        # floor slope = tilted gravity
        a, b = math.radians(c["slope_fwd"]), math.radians(c["slope_side"])
        gdir = np.array([math.sin(a) * math.cos(b), -math.sin(b), -math.cos(a) * math.cos(b)])
        m.opt.gravity[:] = 9.80665 * gdir
        seat = math.degrees(d.qpos[self.tel.qadr["lift_angle"]])
        stroke = d.qpos[self.tel.qadr["actuator_extension"]]
        # Control uses the current physics step, independent of video frame rate.
        R = d.xmat[self.frame].reshape(3, 3)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.frame, self._body_vel, 0)
        forward_speed = float(np.dot(self._body_vel[3:], R[:, 0]))
        speed = abs(forward_speed)
        msg = ""
        # --- driving ---
        top = min(self.mp["v_rated"], TOP_SPEED_CAP)
        vmax = top * LEVEL_FRACTIONS[int(c["level"]) - 1]
        if seat > 5:
            vmax = min(vmax, STANDING_SPEED)
            msg = "Сиденье поднято: скорость ограничена"
        jx, jy = float(c["jx"]), float(c["jy"])
        if abs(jx) < 0.08: jx = 0.0
        if abs(jy) < 0.08: jy = 0.0
        hard_stop = c["estop"] or c["brake"] or c["park"]
        v_t = 0.0 if hard_stop else jy * vmax
        w_t = 0.0 if hard_stop else -jx * TURN_RATE * (0.5 + 0.5 * vmax / top)
        acc = EBRAKE_DECEL if (c["brake"] or c["estop"]) else (ACCEL if abs(v_t) > abs(self.v_cmd) else DECEL)
        self.v_cmd += float(np.clip(v_t - self.v_cmd, -acc * dt, acc * dt))
        self.w_cmd += float(np.clip(w_t - self.w_cmd, -3 * dt, 3 * dt))
        # Detect an obstruction at any selected speed. Compare with the acceleration
        # ramp, so ordinary starts on level ground do not request peak current.
        straight = (v_t != 0 and abs(w_t) < 0.08 and not hard_stop)
        changing_direction = self.v_cmd * v_t <= 0
        progress_speed = math.copysign(1.0, v_t) * forward_speed
        stalled = (straight and not changing_direction and abs(self.v_cmd) >= 0.015
                   and progress_speed < 0.45 * abs(self.v_cmd))
        self.stall_time = self.stall_time + dt if stalled else max(0.0, self.stall_time - 2 * dt)
        if self.stall_time >= OBSTACLE_STALL_TIME:
            self.obstacle_mode = True
        touching_obstacle = self.obstacle_mode and any(
            (contact.geom1 in self.drive_geom_set and contact.geom2 in self.obstacle_geoms)
            or (contact.geom2 in self.drive_geom_set and contact.geom1 in self.obstacle_geoms)
            for contact in d.contact)
        # Retain grip until the loaded axle actually leaves the obstacle. A brief
        # speed recovery on the corner used to drop grip halfway over the step.
        if self.obstacle_mode and straight and not touching_obstacle and progress_speed > 0.8 * abs(v_t):
            self.recovered_time += dt
        else:
            self.recovered_time = 0.0
        if not straight or changing_direction or self.recovered_time > 0.5:
            self.obstacle_mode = False
            self.stall_time = 0.0
        tyre_mu = ROAD_FRICTION if self.obstacle_mode else SKID_FRICTION
        m.geom_friction[self.drive_geoms, 0] = tyre_mu
        drive_kv = 400.0
        if self.obstacle_mode:
            # At tiny joystick inputs a fixed gain cannot deliver the allowed
            # current, even with stationary wheels. Reach the same torque reserve
            # at every crawl speed; the force ranges below still enforce its cap.
            peak_force = self.mp["wheel_torque"] * max(self.peak, OBSTACLE_TORQUE_MULT) / (2 * bh.WHEEL_R)
            drive_kv = max(OBSTACLE_KV, 4 * peak_force / max(abs(self.v_cmd), 0.015))
        for acts in self.drive_acts:
            for a_id in acts:
                m.actuator_gainprm[a_id, 0] = drive_kv
                m.actuator_biasprm[a_id, 2] = -drive_kv
        neutral = (v_t == 0 and w_t == 0 and abs(self.v_cmd) < 0.02 and abs(self.w_cmd) < 0.05)
        self.neutral_since = self.neutral_since + dt if neutral else 0.0
        em_brake = c["estop"] or c["park"] or self.neutral_since > 0.3
        if c["estop"]:
            for acts in self.drive_acts:
                d.ctrl[acts] = 0.0
            self.v_cmd = self.w_cmd = 0.0
        else:
            side_speed = (self.v_cmd - self.w_cmd * TRACK / 2,
                          self.v_cmd + self.w_cmd * TRACK / 2)
            for acts, target in zip(self.drive_acts, side_speed):
                d.ctrl[acts] = target
        d.ctrl[self.brake_acts] = 1.0 if em_brake else 0.0
        # --- motor torque limit: controller current (with peak) and back-EMF above rated speed ---
        mp = self.mp
        for k, (acts, dofs) in enumerate(zip(self.drive_acts, self.wdofs)):
            w = max(abs(d.qvel[dofs[0]]), abs(d.qvel[dofs[1]]))
            requested_peak = max(self.peak, OBSTACLE_TORQUE_MULT if self.obstacle_mode else 1.0)
            t_cur = mp["wheel_torque"] * (requested_peak if self.peak_budget[k] > 0 else 1.0)
            motoring = abs(d.ctrl[acts[0]]) >= w * bh.WHEEL_R - 1e-3      # driving, not braking
            if motoring and w > mp["w_rated"]:
                t_emf = mp["wheel_torque"] * max(0.0, (mp["w0"] - w) / (mp["w0"] - mp["w_rated"]))
                t_lim = min(t_cur, t_emf)
            else:
                t_lim = t_cur
            t_lim = max(t_lim, 0.02 * mp["wheel_torque"])
            self.torque_limit[k] = t_lim
            # Split the motor's available longitudinal rim force equally between the
            # identical front and rear wheels of each side.
            half_rim_force = t_lim / (2 * bh.WHEEL_R)
            for a_id in acts:
                m.actuator_forcerange[a_id] = [-half_rim_force, half_rim_force]
            used = abs(d.qfrc_actuator[dofs[0]]) + abs(d.qfrc_actuator[dofs[1]])
            if used > mp["wheel_torque"] * 1.02:
                self.peak_budget[k] = max(0.0, self.peak_budget[k] - dt)
            else:
                self.peak_budget[k] = min(PEAK_TIME, self.peak_budget[k] + PEAK_RECOVERY * dt)
        if em_brake != self.locked:                       # electromagnetic brake: lock / release
            for eq, q in zip(self.locks, self.wq):
                m.eq_data[eq, 0] = d.qpos[q]
                d.eq_active[eq] = 1 if em_brake else 0
            self.locked = em_brake
        # One position reference controls both real kinematic chains. It models
        # a feedback controller; each actuator keeps its own force limit.
        A = bh.ACTUATORS[self.act]
        requested = int(c["lift"])
        allowed = speed <= 0.05 and neutral and not c["estop"]
        if not allowed and (requested or self.fast_target is not None):
            msg = "Подъём только на стоящем кресле" if not c["estop"] else msg
            self.fast_target = None
        lift = requested if allowed else 0
        if lift:
            self.fast_target = None
        target = float(d.act[self.lift_act] / self.lift_gear)
        if self.fast_target is not None:
            error = self.fast_target - target
            lift = 1 if error > 1e-7 else -1 if error < -1e-7 else 0
            if not lift:
                self.fast_target = None
            else:
                msg = "Сиденье движется к выбранному углу"
        endpoint = None
        if lift > 0 and stroke >= bh.ACT_STROKE - 0.0002:
            endpoint = bh.ACT_STROKE
        elif lift < 0 and stroke <= 0.0002:
            endpoint = 0.0
        if endpoint is not None:
            # Snap the reference, rather than leaving a random sub-millimetre
            # preload after every cycle at the lower limit.
            target = endpoint
            self.fast_target = None
            lift = 0
        forces = [abs(float(d.qfrc_actuator[self.tel.dof[j]])) for j in self.tel.lift_joints]
        force_limit = A["unit_force"] if lift >= 0 else A["unit_retract_force"]
        desired_velocity = lift * A["speed"] * max(0.2, 1 - bh.LOAD_SPEED_DROP * max(forces) / force_limit)
        ramp = 0.08 * dt  # modest shaft acceleration avoids a force spike on startup
        if lift:
            self.lift_velocity_cmd += float(np.clip(desired_velocity - self.lift_velocity_cmd, -ramp, ramp))
        else:
            self.lift_velocity_cmd = 0.0
        if self.fast_target is not None:
            if abs(self.fast_target - target) < abs(self.lift_velocity_cmd * dt):
                self.lift_velocity_cmd = (self.fast_target - target) / dt
        # Limit servo windup on a stalled actuator to its elastic tracking error.
        tracking_error = (A["unit_force"] + 2e5 * abs(self.lift_velocity_cmd)) / 1e7
        if lift:
            target = float(np.clip(target, stroke - tracking_error, stroke + tracking_error))
        for a, state in zip(self.lift_acts, self.lift_states):
            d.act[state] = target * self.lift_gear
            d.ctrl[a] = self.lift_velocity_cmd * self.lift_gear
        self.lift_running = bool(abs(self.lift_velocity_cmd) > 1e-9)
        if c["estop"]:
            msg = "АВАРИЙНЫЙ СТОП"
        elif c["park"]:
            msg = "Стояночный тормоз включён"
        if self.obstacle_mode and not msg:
            msg = "Тяговый режим: препятствие"
        self.status = dict(em_brake=bool(em_brake), lift_dir=lift, vmax=vmax,
                           obstacle_mode=self.obstacle_mode, msg=msg or self.msg)
        # --- skid-steer yaw moment ---
        # Four fixed point-contact wheels otherwise resist sideways scrub so strongly
        # that opposing left/right wheel speeds cannot rotate the chassis.  This bounded
        # moment represents the distributed lateral forces of tyres/tracks while the
        # visible wheel actuators still provide longitudinal drive and consume power.
        d.xfrc_applied[self.frame] = 0
        yaw_rate = float(np.dot(self._body_vel[:3], R[:, 2]))
        yaw_torque = float(np.clip(SKID_YAW_GAIN * (self.w_cmd - yaw_rate),
                                   -SKID_YAW_TORQUE, SKID_YAW_TORQUE))
        d.xfrc_applied[self.frame, 3:] = R[:, 2] * yaw_torque
        # --- pushes ---
        if self.push and d.time < self.push[0]:
            F = self.push[1]
            Fw = R @ F
            r = R @ np.array([0.0, 0, 0.6]) + d.xpos[self.frame] - d.xipos[self.frame]
            d.xfrc_applied[self.frame, :3] += Fw
            d.xfrc_applied[self.frame, 3:] += np.cross(r, Fw)

    def _threshold_start_x(self, clearance=0.002):
        """Start 2 mm before actual tyre-to-30-mm-step contact (not tyre's top edge)."""
        m = self.m
        step_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "threshold_30mm")
        wheel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "front_left_contact")
        if step_id < 0 or wheel_id < 0:
            raise RuntimeError("Не найдены threshold_30mm или front_left_contact")
        step_front = float(m.geom_pos[step_id, 0] - m.geom_size[step_id, 0])
        step_height = float(m.geom_pos[step_id, 2] + m.geom_size[step_id, 2])
        radius = float(m.geom_size[wheel_id, 0])
        # A spherical tyre first touches the step's upper front corner at
        # horizontal offset sqrt(r² - (r-h)²), not at the outermost tyre point.
        contact_offset = math.sqrt(max(0.0, radius * radius - (radius - step_height) ** 2))
        front_axle_x = float(self.tel.wheel_xy["front_left"][0])
        return step_front - contact_offset - front_axle_x - clearance

    def handle_events(self):
        while self.events:
            ev = self.events.pop(0)
            name = ev.get("name")
            if name == "push":
                dirs = {"fwd": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0), "right": (0, -1, 0)}
                F = np.array(dirs[ev["dir"]], float) * float(ev.get("force", 250))
                self.push = (self.d.time + 0.3, F)
                self.msg = f"Толчок {ev.get('force', 250)} Н на высоте 0,6 м"
            elif name == "goto":
                spots = {"start": (0.0, 0.0), "threshold": (self._threshold_start_x(), 0.0), "ramp": (3.8, 0.0),
                         "platform": (7.15, 2.0 / 12), "hill10": (11.2, 0.0),
                         "hill15": (19.2, 0.0), "hill30": (25.2, 0.0),
                         "long15": (32.8, 0.0)}
                x, z = spots[ev["where"]]
                self.teleport(x, z=z)
                self.msg = "Перенесено"
            elif name == "reset":
                self.reset(); self.msg = "Сброс"
            elif name == "config":
                self.set_config(float(ev.get("mass", self.mass)), float(ev.get("height", self.height)),
                                ev.get("frame", self.frame_mat), ev.get("act", self.act),
                                ev.get("motor", self.motor), ev.get("motor_params"), ev.get("peak"),
                                ev.get("chair"))
            elif name == "seat_fast":
                th = math.radians(min(75.0, max(0.0, float(ev["angle"]))))
                G = bh.geometry(self.chair, self.act)
                sx, sz = G["SEAT_PIN_LOCAL"]
                c_, s_ = math.cos(th), math.sin(th)
                tip = self.seat_pivot + np.array([sx * c_ + sz * s_, 0, -sx * s_ + sz * c_])
                P = np.array([G["FRAME_PIN"][0], 0, G["FRAME_PIN"][1]])
                self.fast_target = float(np.clip(np.linalg.norm(tip - P) - bh.ACT_RETRACTED, 0, bh.ACT_STROKE))

    def _battery_current(self):
        """Estimated electrical current, sampled at each physics step.

        Screw holding is assumed passive. Numerical settling with a stopped
        controller therefore consumes no simulated energy. No regeneration.
        """
        d, mp = self.d, self.mp
        amps = 0.0
        if not self.status["em_brake"]:
            for dofs in self.wdofs:
                tau = sum(float(d.qfrc_actuator[i]) for i in dofs)
                power = sum(float(d.qfrc_actuator[i] * d.qvel[i]) for i in dofs)
                turning = any(abs(float(d.qvel[i])) > 2 * math.pi / 60 for i in dofs)
                if turning:
                    amps += mp["I0"]
                if power > 0.01:
                    amps += abs(tau) / (mp["ratio"] * mp["eta"]) / mp["kt"]
        if self.lift_running:
            for joint in self.tel.lift_joints:
                i = self.tel.dof[joint]
                power = float(d.qfrc_actuator[i] * d.qvel[i])
                amps += max(LIFT_IDLE_A, max(power, 0) / ETA_LIFT / BATTERY_V)
        return amps

    # main step -------------------------------------------------------------------------
    def advance(self, frame_dt):
        with self.lock:
            self.handle_events()
            n = max(1, int(round(frame_dt * self.cmd["timescale"] / self.m.opt.timestep)))
            for _ in range(n):
                self.apply_controls(self.m.opt.timestep)
                mujoco.mj_step(self.m, self.d)
                self.energy_Wh += self._battery_current() * BATTERY_V * self.m.opt.timestep / 3600
                if not np.isfinite(self.d.qpos).all():
                    self.reset(); self.msg = "Симуляция разошлась, сброс"
                    break
            t = self.tel.compute(self.d)
            t.update(self.status)
            mp = self.mp
            t["battery_A"] = self._battery_current()
            t["motor_info"] = {k: mp[k] for k in ("name", "note", "wheel_torque", "v_rated", "v0", "I", "mass", "key")}
            t["motor_params"] = {k: mp[k] for k in bh.MOTOR_FIELDS}
            t["peak"] = self.peak
            t["torque_limit"] = list(self.torque_limit)
            t["peak_budget"] = list(self.peak_budget)
            t["time"] = self.d.time
            t["energy_Wh"] = self.energy_Wh
            t["battery_pct"] = max(0.0, 100 - 100 * self.energy_Wh / (BATTERY_V * BATTERY_AH))
            t["mass_set"], t["height_set"] = self.mass, self.height
            t["frame_mat"], t["act"] = self.frame_mat, self.act
            t["chair"] = self.chair
            t["act_info"] = bh.ACTUATORS[self.act]
            t["chair_mass"] = t["total_mass"] - t["human_mass"]
            t["fast"] = self.fast_target is not None
            t["cmd"] = dict(self.cmd)
            t["warnings"] = self._warnings(t)
            self.state = t
            if not self.history or self.d.time - self.history[-1][0] >= 0.1:
                self.history.append((round(self.d.time, 2), round(t["speed"], 3), round(t["lift_force"]),
                                     round(t["battery_A"], 2),
                                     round(min(t["load_drive_left"], t["load_drive_right"])),
                                     round(min(t["load_front_left"], t["load_front_right"])),
                                     round(t["seat_angle"], 1)))
                if len(self.history) > 600:
                    del self.history[:len(self.history) - 600]
        return t

    def _warnings(self, t):
        w = []
        if self.d.time < self.settle_until:
            return w
        if t["lifted"]:
            ru = {"drive_left": "ведущее левое", "drive_right": "ведущее правое",
                  "front_left": "переднее левое", "front_right": "переднее правое"}
            w.append("Колесо оторвалось: " + ", ".join(ru[x] for x in t["lifted"]))
        sat = (not self.status["em_brake"]) and any(abs(t[f"torque_{s_}"]) > 0.97 * self.torque_limit[k]
                                                     for k, s_ in enumerate(("left", "right")))
        if not sat:
            self.sat_since = self.d.time
        if self.d.time - self.sat_since > 1.5:     # at the limit for 1.5 s (normal starts are shorter)
            w.append(f"Моторы на пределе момента ({self.mp['wheel_torque']:.1f} Н·м на колесе)")
        if self.peak > 1 and min(self.peak_budget) <= 0:
            w.append("Пиковый ток исчерпан, моторы перегреты — только номинальный момент")
        profile = bh.ACTUATORS[self.act]
        if any(abs(a["force"]) > (profile["unit_force"] if a["force"] >= 0 else profile["unit_retract_force"]) * 0.95 for a in t["lifts"]):
            w.append(f"Актуатор на пределе усилия ({bh.ACTUATORS[self.act]['unit_force']:.0f} Н/шт)")
        if t["lift_skew_mm"] > 1.0:
            w.append("Рассогласование актуаторов больше 1 мм")
        for k, name in (("tip_fwd", "вперёд"), ("tip_back", "назад"), ("tip_side", "вбок")):
            if t[k] is not None and t[k] < 6:
                w.append(f"Запас до опрокидывания {name} меньше 6°")
        if abs(t["pitch"]) > 25 or abs(t["roll"]) > 25:
            w.append("КРЕСЛО ОПРОКИНУЛОСЬ")
        return w

    def snapshot(self):
        with self.lock:
            s = dict(self.state)
            s["history"] = list(self.history)
            s["model_version"] = self.model_version
            s["parts"] = self.parts.list()
            s["selection"] = self.parts.inspect(self.d, self.selected_part) if self.selected_part else None
        return s

    def command(self, data):
        with self.lock:
            for k, v in data.items():
                if k in self.cmd:
                    self.cmd[k] = type(self.cmd[k])(v) if not isinstance(self.cmd[k], bool) else bool(v)
            self.last_input = time.time()

    def event(self, ev):
        with self.lock:
            self.events.append(ev)


# ------------------------------------------------------------------------------ web panel
# ------------------------------------------------------------------------- video in the page
class Video:
    """Renders the scene off-screen and serves it as an MJPEG stream into the control panel."""

    MAX_SIZE = (3840, 2400)

    @classmethod
    def fit_size(cls, width, height):
        width, height = float(width), float(height)
        if not all(math.isfinite(v) and v >= 2 for v in (width, height)):
            raise ValueError("Video size must contain two finite dimensions >= 2")
        scale = min(1.0, cls.MAX_SIZE[0] / width, cls.MAX_SIZE[1] / height)
        return tuple(max(2, min(limit, 2 * math.ceil(v * scale / 2)))
                     for v, limit in zip((width, height), cls.MAX_SIZE))

    def __init__(self, sim, width, height):
        self.sim = sim
        width, height = self.fit_size(width, height)
        self.requested_size = (width, height)
        self._version = sim.model_version
        sim.m.vis.global_.offwidth = width
        sim.m.vis.global_.offheight = height
        self.renderer = mujoco.Renderer(sim.m, height, width)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 140.0, -18.0, 2.6
        self.follow = True
        self.pan = np.zeros(3)
        self.show_com = True
        self.lookat = None
        self.cond = threading.Condition()
        self.frame = b""
        self.ctype = "image/jpeg"
        self.seq = 0
        try:
            from PIL import Image                     # noqa: F401
            self.pil = True
        except ImportError:
            self.pil = False
            self.ctype = "image/png"

    def control(self, c):
        if "size" in c:
            # HTTP workers queue a request. OpenGL resources belong to render().
            self.requested_size = self.fit_size(*c["size"])
        cam = self.cam
        if "orbit" in c:
            cam.azimuth -= 0.4 * float(c["orbit"][0])
            cam.elevation = float(np.clip(cam.elevation - 0.4 * float(c["orbit"][1]), -89, 5))
        if "zoom" in c:
            cam.distance = float(np.clip(cam.distance * math.exp(0.0015 * float(c["zoom"])), 0.6, 15))
        if "pan" in c:
            az = math.radians(cam.azimuth)
            right = np.array([math.sin(az), -math.cos(az), 0.0])
            k = 0.0015 * cam.distance
            self.pan += -right * float(c["pan"][0]) * k + np.array([0, 0, 1.0]) * float(c["pan"][1]) * k
        if "preset" in c:
            yaw = self._yaw()
            p = c["preset"]
            self.pan[:] = 0
            self.follow = True
            rel = {"follow": (140, -18, 2.6), "back": (0, -15, 2.6), "side": (90, -10, 2.6),
                   "front": (180, -12, 2.6), "top": (90, -89, 3.0)}[p]
            cam.azimuth = (yaw + rel[0]) if p != "follow" else rel[0]
            cam.elevation, cam.distance = rel[1], rel[2]
        if "human" in c:
            self.opt.geomgroup[2] = 1 if c["human"] else 0
        if "forces" in c:
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1 if c["forces"] else 0
            self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 if c["forces"] else 0
        if "com" in c:
            self.show_com = bool(c["com"])

    def state(self):
        return dict(human=bool(self.opt.geomgroup[2]), com=self.show_com,
                    size=list(self.requested_size), max_size=list(self.MAX_SIZE),
                    forces=bool(self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]))

    def pick(self, x, y, select=True):
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError("Invalid selection coordinates")
        if self._version != self.sim.model_version:
            return None
        point = np.zeros(3)
        geom, flex, skin = (np.full(1, -1, dtype=np.int32) for _ in range(3))
        mujoco.mjv_select(self.sim.m, self.sim.d, self.opt,
                          self.renderer.width / self.renderer.height, x, 1-y,
                          self.renderer.scene, point, geom, flex, skin)
        key = self.sim.parts.geom_part.get(int(geom[0]))
        # Empty scene/floor clicks dismiss the selected assembly. Hover only reads.
        if key == "track_floor":
            key = None
        if select:
            key = None if key == self.sim.selected_part else key
            self.sim.selected_part = key
        return self.sim.parts.inspect(self.sim.d, key) if key else None

    def _yaw(self):
        R = self.sim.d.xmat[self.sim.frame].reshape(3, 3)
        return math.degrees(math.atan2(R[1, 0], R[0, 0]))

    def render(self):
        sim = self.sim
        model_changed = self._version != sim.model_version
        if model_changed or (self.renderer.width, self.renderer.height) != self.requested_size:
            w, h = self.requested_size
            self.renderer.close()
            sim.m.vis.global_.offwidth = w
            sim.m.vis.global_.offheight = h
            self.renderer = mujoco.Renderer(sim.m, h, w)
            self._version = sim.model_version
            if model_changed:
                self.lookat = None
        d = sim.d
        R = d.xmat[sim.frame].reshape(3, 3)
        target = d.xpos[sim.frame] + R @ np.array([0.35, 0, 0]) + np.array([0, 0, 0.8])
        self.lookat = target if self.lookat is None else self.lookat + 0.25 * (target - self.lookat)
        self.cam.lookat[:] = self.lookat + self.pan
        self.renderer.update_scene(d, self.cam, self.opt)
        selected = sim.parts.groups.get(sim.selected_part)
        if selected and sim.selected_part != "chair":
            selected_geoms = set(selected["geoms"])
            for g in self.renderer.scene.geoms[:self.renderer.scene.ngeom]:
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in selected_geoms:
                    g.rgba[:3] = 0.55 * g.rgba[:3] + 0.45 * np.array([1.0, 0.65, 0.15])
        if self.show_com and sim.state:
            self._overlay(self.renderer.scene, sim.state, d)
        img = self.renderer.render()
        buf = io.BytesIO()
        if self.pil:
            from PIL import Image
            Image.fromarray(img).save(buf, "JPEG", quality=94, subsampling=0)
        else:
            buf.write(_png(img))
        with self.cond:
            self.frame = buf.getvalue()
            self.seq += 1
            self.cond.notify_all()

    @staticmethod
    def _overlay(scn, t, d):
        def add(type_, size, pos, rgba):
            if scn.ngeom >= scn.maxgeom:
                return None
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, type_, np.asarray(size, float), np.asarray(pos, float),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            scn.ngeom += 1
            return g
        def line(a, b, w, rgba):
            g = add(mujoco.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], [0, 0, 0], rgba)
            if g is not None:
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(a, float), np.asarray(b, float))
        c = np.array(t.get("com_display", t["com"]))
        add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.035, 0, 0], c, [1, 0.15, 0.15, 1])
        support = t.get("support3d_display", t.get("support3d", []))
        z0 = float(np.mean([p[2] for p in support])) if support else 0.0
        line(c, [c[0], c[1], z0 + 0.003], 0.005, [1, 0.15, 0.15, 0.9])
        add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.03, 0.002, 0], [c[0], c[1], z0 + 0.003], [1, 0.15, 0.15, 1])
        H = support
        for i in range(len(H)):
            a, b = H[i], H[(i + 1) % len(H)]
            line([a[0], a[1], a[2] + 0.004], [b[0], b[1], b[2] + 0.004], 0.006, [1, 0.65, 0.1, 0.95])

    def stream(self, wfile):
        last = -1
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.seq != last, timeout=2.0)
                frame, last = self.frame, self.seq
            if not frame:
                continue
            wfile.write(b"--frame\r\nContent-Type: " + self.ctype.encode() +
                        b"\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            wfile.flush()


def _png(img):
    """Minimal PNG encoder (fallback when Pillow is not installed)."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(t, data):
        c = struct.pack(">I", len(data)) + t + data
        return c + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def make_handler(sim, video=None):
    page = (HERE / "panel.html").read_bytes()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/options"):
                self._send(json.dumps(dict(
                    actuators={k: v for k, v in bh.ACTUATORS.items()},
                    frames={k: v["name"] for k, v in bh.FRAMES.items()},
                    motors={k: v for k, v in bh.MOTORS.items()}, motor_fields=list(bh.MOTOR_FIELDS),
                    chairs=bh.CHAIRS,
                    video=video is not None, view=video.state() if video else None)).encode())
            elif self.path.startswith("/stream"):
                if video is None:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    video.stream(self.wfile)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            elif self.path.startswith("/api/state"):
                self._send(json.dumps(sim.snapshot(), default=float).encode())
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            if self.path.startswith("/api/inspect"):
                try:
                    with sim.lock:
                        if data.get("clear") is True:
                            sim.selected_part = None
                            selected = None
                        elif "id" in data:
                            key = data["id"]
                            if key not in sim.parts.groups:
                                raise ValueError("Unknown part")
                            sim.selected_part = key
                            selected = sim.parts.inspect(sim.d, key)
                        elif video is not None:
                            selected = video.pick(*map(float, data["point"]), select=not data.get("preview", False))
                        else:
                            raise ValueError("Video unavailable")
                    self._send(json.dumps(selected).encode()); return
                except (KeyError, TypeError, ValueError):
                    self.send_error(400, "Invalid part selection"); return
            elif self.path.startswith("/api/cmd"):
                sim.command(data)
            elif self.path.startswith("/api/event"):
                sim.event(data)
            elif self.path.startswith("/api/cam") and video is not None:
                try:
                    with sim.lock:
                        video.control(data)
                        state = video.state()
                except (TypeError, ValueError):
                    self.send_error(400, "Invalid camera parameters"); return
                self._send(json.dumps(state).encode()); return
            self._send(b"{}")
    return H


def overlay(t):
    def f(x, nd=0):
        return "-" if x is None else f"{x:.{nd}f}"
    left = "Speed\nSeat\nLift force\nBrake\nStatus"
    right = (f"{f(t.get('speed'), 2)} m/s\n{f(t.get('seat_angle'), 1)} deg\n{f(t.get('lift_force'))} N\n"
             f"{'ON' if t.get('em_brake') else 'off'}\n"
             f"{'WARNING' if t.get('warnings') else 'OK'}   (panel: http://127.0.0.1:{PORT})")
    return left, right


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=80.0)
    ap.add_argument("--height", type=float, default=1.75)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--viewer", action="store_true", help="separate MuJoCo window instead of video in the page (mjpython)")
    ap.add_argument("--video", default="1120x700", help="initial video size, WxH; browser adapts to its viewport")
    ap.add_argument("--headless", type=float, default=0, help="run N seconds without a window (test)")
    a = ap.parse_args()

    sim = Sim(a.mass, a.height)
    video = None
    if not a.viewer:
        try:
            w, h = (int(x) for x in a.video.lower().split("x"))
            video = Video(sim, w, h)
        except Exception as e:                       # no off-screen OpenGL: fall back to the MuJoCo window
            print(f"Видео в странице недоступно ({e}). Открываю отдельное окно MuJoCo.")
            mjp = pathlib.Path(sys.executable).with_name("mjpython")
            if sys.platform == "darwin" and mjp.exists() and not os.environ.get("V5_MJPYTHON"):
                os.environ["V5_MJPYTHON"] = "1"
                os.execv(str(mjp), [str(mjp), str(HERE / "simulator.py"), "--viewer"])
            if a.headless == 0:
                a.viewer = True

    global PORT
    srv = None
    for port in range(PORT, PORT + 20):          # another copy may still be running
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sim, video))
            PORT = port
            break
        except OSError:
            continue
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    print(f"Пульт управления: {url}")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if not a.viewer:
        # one window: simulation + video stream into the browser page
        t_end = time.time() + a.headless if a.headless else float("inf")
        fps = 30
        print("Окно симуляции — в браузере. Закрыть: Ctrl+C в этом окне.")
        last = time.time()
        try:
            while time.time() < t_end:
                t0 = time.time()
                sim.advance(min(t0 - last, 0.1) or 1 / fps)   # keep real time even if drawing is slow
                last = t0
                if video:
                    with sim.lock:
                        video.render()
                time.sleep(max(0, 1 / fps - (time.time() - t0)))
        except KeyboardInterrupt:
            pass
        srv.shutdown()
        os._exit(0)                                  # skip noisy OpenGL teardown at interpreter exit

    from mujoco import viewer as mjviewer
    while True:
        version = sim.model_version
        with mjviewer.launch_passive(sim.m, sim.d) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = sim.frame
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.2, 140, -18
            last = 0
            while v.is_running() and version == sim.model_version:
                t0 = time.time()
                t = sim.advance(1 / 60)
                if version != sim.model_version:
                    break  # reopen the viewer with the rebuilt model and state
                with v.lock():
                    if time.time() - last > 0.2:
                        l, r = overlay(t)
                        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, l, r))
                        last = time.time()
                v.sync()
                time.sleep(max(0, 1 / 60 - (time.time() - t0)))
        if version == sim.model_version:
            break
    srv.shutdown()


if __name__ == "__main__":
    main()
