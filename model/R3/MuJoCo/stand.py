"""Load and drive the R3 / Gruzdy MuJoCo load bench. Units are SI unless stated."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "R3_Gruzdy.xml"
RATIO = 48 / 12              # pinion -> 48-tooth idler -> 48-tooth screw gear
PULSES_PER_SECOND = 100      # Gruzdy/src/main.cpp, full-step mode
STEPS_PER_REV = 48
WHEEL_TORQUE = 6 * 0.0980665  # kgf cm -> N m, one JGY-370
WHEEL_FREE = 55 * math.tau / 60
WHEEL_RATED = 40 * math.tau / 60
LINKS = {"lift": 1, "back_hinge": -1, "lower_hinge": 1, "hip_compensation": -1}
GEARS = {"shaft": 1, "driven": 1, "idler": -1, "pinion": RATIO}


class Bench:
    def __init__(self, mass=6.0, torque=.08, lead=1.5, eff=.25, mobile=False):
        if not all(math.isfinite(v) and v > 0 for v in (mass, torque, lead, eff)) or eff > 1:
            raise ValueError("Mass, torque and lead must be positive; efficiency must be in (0, 1].")
        self.mass = mass
        self.lead = lead / 1000
        self.force = math.tau * torque * RATIO * eff / self.lead
        self.speed = PULSES_PER_SECOND * self.lead / (STEPS_PER_REV * RATIO)
        root = ET.parse(MODEL).getroot()
        # Group 2 contains precisely the massive dummy segments. XML reference mass is 6 kg.
        for segment in root.findall(".//geom[@group='2']"):
            segment.set("mass", str(float(segment.get("mass")) * mass / 6))
        root.find("actuator/position").set("forcerange", f"{-self.force} {self.force}")
        if mobile:
            frame = root.find(".//body[@name='frame']")
            frame.insert(0, ET.Element("freejoint", name="base"))
            frame.set("pos", "0 0 0.0001")
            for wheel in root.findall(".//geom[@group='3']"):
                wheel.set("contype", "1")
                wheel.set("conaffinity", "1")
        assets = {p.name: p.read_bytes() for p in (HERE / "assets").glob("*.stl")}
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
        self.d = mj.MjData(self.model)
        self.joints = {
            name: (self.model.joint(name).qposadr.item(), self.model.joint(name).dofadr.item())
            for name in (*LINKS, "extension", "wheelL", "wheelR")
        }
        self.gear_joints = {name: self.model.joint(name + "_angle").qposadr.item() for name in GEARS}
        mj.mj_forward(self.model, self.d)
        self.home = self.d.site("slider_pin").xpos.copy()
        delta = self.d.site("rocker_pin").xpos - self.home
        self.link_length = float(np.linalg.norm(delta))
        self.home_pitch = math.atan2(delta[2], -delta[0])
        sample = mj.MjData(self.model)
        sample.qpos[self.joints["lift"][0]] = math.radians(75)
        mj.mj_forward(self.model, sample)
        self.travel = self.linkage(sample)[0]
        self.target = 0.
        self.peak = 0.
        self.max_error = 0.

    def q(self, name):
        return self.d.qpos[self.joints[name][0]].item()

    def linkage(self, sample):
        """Inverse slider–crank kinematics and derivatives w.r.t. seat angle."""
        tip = sample.site("rocker_pin").xpos
        height = tip[2] - self.home[2]
        horizontal = math.sqrt(self.link_length**2 - height**2)
        stroke = self.home[0] - tip[0] - horizontal
        pitch = math.atan2(height, horizontal) - self.home_pitch
        joint = self.model.joint("lift").id
        velocity = np.cross(sample.xaxis[joint], tip - sample.xanchor[joint])
        stroke_rate = -velocity[0] + height * velocity[2] / horizontal
        pitch_rate = velocity[2] / horizontal
        return stroke, pitch, stroke_rate, pitch_rate

    def sync_gears(self):
        # Positive lift retracts the rod. Angles are for visualizing the actual STEP gears.
        screw_angle = -self.q("extension") * math.tau / self.lead
        for name, address in self.gear_joints.items():
            self.d.qpos[address] = math.remainder(screw_angle * GEARS[name], math.tau)
        mj.mj_forward(self.model, self.d)

    def step(self, direction=1, wheels=(0., 0.)):
        dt = self.model.opt.timestep
        self.target = min(self.travel, max(0., self.target + direction * self.speed * dt))
        self.d.ctrl[0] = self.target
        for index, (wheel, command) in enumerate(zip(("wheelL", "wheelR"), wheels), 1):
            omega = self.d.qvel[self.joints[wheel][1]]
            torque = (command * WHEEL_FREE - omega) * WHEEL_TORQUE / (WHEEL_FREE - WHEEL_RATED)
            self.d.ctrl[index] = min(WHEEL_TORQUE, max(-WHEEL_TORQUE, torque))
        mj.mj_step(self.model, self.d)
        self.peak = max(self.peak, abs(self.d.actuator_force[0].item()))
        error = self.d.site("rocker_pin").xpos - self.d.site("link_tip").xpos
        self.max_error = max(self.max_error, float(np.linalg.norm(error)))

    def gravity_curve(self):
        """Project gravity and slider-crank Jacobian onto the four-bar's single coordinate."""
        sample = mj.MjData(self.model)
        curve = []
        for deg in np.linspace(0, 75, 301):
            angle = math.radians(deg)
            for joint, ratio in LINKS.items():
                sample.qpos[self.joints[joint][0]] = angle * ratio
            mj.mj_forward(self.model, sample)
            stroke, pitch, lever, pitch_rate = self.linkage(sample)
            sample.qpos[self.joints["extension"][0]] = stroke
            pushrod = self.model.joint("pushrod_pitch")
            sample.qpos[pushrod.qposadr.item()] = pitch
            mj.mj_forward(self.model, sample)
            moment = sum(sample.qfrc_bias[self.joints[j][1]] * r for j, r in LINKS.items())
            moment += sample.qfrc_bias[pushrod.dofadr.item()] * pitch_rate
            moment += sample.qfrc_bias[self.joints["extension"][1]] * lever
            curve.append((float(deg), float(moment / lever)))
        return curve

    def trial(self, callback=None):
        curve = self.gravity_curve()
        critical, gravity_peak = max(curve, key=lambda point: point[1])
        first = None
        hold = stalled = 0.
        highest = 0.
        dt = self.model.opt.timestep
        next_sample = 0.
        trace = []
        while self.d.time < self.travel / self.speed + 5:
            self.step()
            angle = math.degrees(self.q("lift"))
            highest = max(angle, highest)
            extension = self.q("extension")
            force = self.d.actuator_force[0].item()
            if angle >= 74.5 and first is None:
                first = self.d.time
            hold = hold + dt if angle >= 74.5 and self.target >= self.travel - 1e-10 else 0.
            blocked = self.target - extension > .0008 and force > .99 * self.force
            stalled = stalled + dt if blocked else 0.
            if self.d.time >= next_sample or hold >= 1 or stalled >= 1:
                trace.append([self.d.time, angle, extension * 1000, force])
                if callback:
                    callback(self)
                next_sample += .20
            if hold >= 1 or stalled >= 1:
                break
        if np.any(self.d.warning.number) or not np.isfinite(self.d.qpos).all() or self.max_error > .0004:
            raise RuntimeError("MuJoCo numerical warning or excessive transmission residual")
        return {
            "mass_kg": self.mass, "passed": hold >= 1, "max_angle_deg": round(highest, 2),
            "end_angle_deg": round(angle, 2), "time_to_74_5_s": None if first is None else round(first, 2),
            "peak_motor_force_N": round(self.peak, 2), "gravity_peak_N": round(gravity_peak, 2),
            "critical_deg": critical, "max_constraint_error_mm": round(self.max_error * 1000, 4),
            "time_s": round(self.d.time, 2),
        }, trace

    def drive_trial(self):
        """Settle, drive forward, then counter-rotate on the same floor."""
        def yaw():
            matrix = self.d.body("frame").xmat.reshape(3, 3)
            return math.atan2(matrix[1, 0], matrix[0, 0])
        def run(seconds, wheels):
            rotation = 0.
            previous = yaw()
            for _ in range(round(seconds / self.model.opt.timestep)):
                self.step(0, wheels)
                current = yaw()
                rotation += math.remainder(current - previous, math.tau)
                previous = current
            return math.degrees(rotation)
        run(2, (0, 0))
        start = self.d.body("frame").xpos.copy()
        run(2, (.6, .6))
        displacement = self.d.body("frame").xpos - start
        turn = run(2, (-.55, .55))
        if np.any(self.d.warning.number) or displacement[0] <= 0 or abs(turn) < 10:
            raise RuntimeError("Drive trial failed")
        return {"mass_kg": self.mass, "forward_2s_mm": round(1000 * displacement[0], 1),
                "turn_2s_deg": round(turn, 1), "wheel_commands": [[.6, .6], [-.55, .55]]}


def camera(cam):
    cam.lookat[:] = (.015, -.014, .153)
    cam.azimuth, cam.elevation, cam.distance = 123, -15, .58


def viewer(bench, mobile, speedup):
    import mujoco.viewer
    from queue import SimpleQueue
    keys = SimpleQueue()
    direction = 0 if mobile else 1
    wheels = (0., 0.)
    print("U/J/H — подъём / опускание / удержание. W/S/A/D/X — движение / разворот / стоп.", flush=True)
    with mujoco.viewer.launch_passive(bench.model, bench.d, key_callback=keys.put) as window:
        camera(window.cam)
        while window.is_running():
            start = time.perf_counter()
            while not keys.empty():
                key = keys.get()
                direction = {ord("U"): 1, ord("J"): -1, ord("H"): 0}.get(key, direction)
                if mobile:
                    wheels = {ord("W"): (.6, .6), ord("S"): (-.6, -.6), ord("A"): (-.55, .55),
                              ord("D"): (.55, -.55), ord("X"): (0, 0)}.get(key, wheels)
            for _ in range(max(1, round(speedup / 50 / bench.model.opt.timestep))):
                bench.step(direction, wheels)
            bench.sync_gears()
            if mobile:
                window.cam.lookat[:] = bench.d.body("frame").xpos + (.07, 0, .14)
            window.sync()
            time.sleep(max(0, .02 - (time.perf_counter() - start)))


def run_suite():
    result = {"mujoco": mj.__version__, "scenarios": {}}
    for name, torque, efficiency, lead, masses in [
        ("reference", .080, .25, 1.5, (5, 6, 8, 29, 29.5)),
        ("higher_losses", .050, .15, 1.5, (5, 6, 8)),
        ("higher_losses_long_lead", .050, .15, 3.0, (5, 6, 8)),
    ]:
        runs = []
        for mass in masses:
            bench = Bench(mass, torque=torque, eff=efficiency, lead=lead)
            record, _ = bench.trial()
            runs.append(record)
        result["scenarios"][name] = {
            "torque_Nm": torque, "efficiency": efficiency, "lead_mm_per_rev": bench.lead * 1000,
            "force_limit_N": bench.force, "stroke_mm": bench.travel * 1000,
            "speed_mm_per_s": bench.speed * 1000, "trials": runs,
        }
    result["on_wheels"] = []
    for mass in (5, 6, 8):
        loaded = Bench(mass, mobile=True)
        max_pin = max_vertical_pin = max_tilt = 0.
        min_normal = math.inf
        def sample_wheels(b):
            nonlocal max_pin, max_vertical_pin, max_tilt, min_normal
            if b.d.time < 1:
                return
            rows = (b.d.efc_type == int(mj.mjtConstraint.mjCNSTR_EQUALITY)) & (
                b.d.efc_id == b.model.equality("link_pin").id)
            pin = b.d.efc_force[rows]
            max_pin = max(max_pin, float(np.linalg.norm(pin)))
            max_vertical_pin = max(max_vertical_pin, abs(float(pin[2])))
            max_tilt = max(max_tilt, math.degrees(math.acos(np.clip(b.d.body("frame").xmat[8], -1, 1))))
            normal = {b.model.geom("contact_" + n).id: 0. for n in ("frontL","frontR","driveL","driveR")}
            for i, contact in enumerate(b.d.contact):
                for geom in contact.geom:
                    if int(geom) in normal:
                        wrench = np.zeros(6)
                        mj.mj_contactForce(b.model, b.d, i, wrench)
                        normal[int(geom)] += wrench[0]
            min_normal = min(min_normal, *normal.values())
        record, _ = loaded.trial(callback=sample_wheels)
        record.update(min_sampled_wheel_load_N=round(min_normal, 2), max_sampled_tilt_deg=round(max_tilt, 3),
                      max_sampled_pin_force_N=round(max_pin, 2), max_sampled_vertical_pin_force_N=round(max_vertical_pin, 2))
        result["on_wheels"].append(record)
    refined = Bench(8)
    refined.model.opt.timestep /= 2
    result["half_timestep_8kg"] = refined.trial()[0]
    result["drive"] = Bench(6, mobile=True).drive_trial()
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=6, help="Mannequin mass, kg")
    parser.add_argument("--torque", type=float, default=.08, help="Running stepper torque, N m")
    parser.add_argument("--lead", type=float, default=1.5, help="Screw travel per revolution, mm")
    parser.add_argument("--eff", type=float, default=.25, help="Combined screw and gear efficiency")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--drive", action="store_true", help="Release base and enable wheel-floor contact")
    parser.add_argument("--on-wheels", action="store_true", help="Run the lift with the frame free on all four wheels")
    parser.add_argument("--suite", action="store_true", help="Rerun published load and drive cases")
    parser.add_argument("--speedup", type=float, default=4, help="Viewer playback rate")
    args = parser.parse_args()
    if args.suite:
        print(json.dumps(run_suite(), indent=2))
    else:
        bench = Bench(args.mass, args.torque, args.lead, args.eff, args.drive or args.on_wheels)
        if args.viewer:
            viewer(bench, args.drive, args.speedup)
        else:
            print(json.dumps(bench.drive_trial() if args.drive else bench.trial()[0], indent=2))
