"""Builds chair_human.xml: the V5 chair (free chassis) + a seated human + load sensors.

    python build_human.py --mass 80 --height 1.75

Human = 14 rigid segments, masses and centres of mass from de Leva (1996, male).
Pelvis and thighs are fixed to the seat, the trunk hinges at the hip and is kept parallel
to the backrest, shanks hinge at the knees and the feet stand (and can slide) on a footplate.
"""
import argparse, math, pathlib
import xml.etree.ElementTree as ET
import numpy as np

import chair_esleh as ce

HERE = pathlib.Path(__file__).resolve().parent

CHAIRS = {"esleh": "V6 — как ESLEH Rester (примитивы)", "v5": "V5.2 — из CAD SolidWorks"}
DEFAULT_CHAIR = "esleh"

# de Leva 1996, male: mass fraction, CoM position from proximal end (fraction of length)
DELEVA = dict(head=(0.0694,), upper_trunk=(0.1596,), mid_trunk=(0.1633,), pelvis=(0.1117,),
              upper_arm=(0.0271, 0.5772), forearm=(0.0162, 0.4574), hand=(0.0061, 0.79),
              thigh=(0.1416, 0.4095), shank=(0.0433, 0.4459), foot=(0.0137, 0.4415))

SEAT_TOP = 0.03          # seat surface above the seat pivot (seat body frame), m
KNEE_FLEX = 0.20         # shank slightly inclined (never a dead-centre strut); knee sits on the seat pivot
BACK_FACE_X = -0.42      # backrest front face in seat frame (back pivot -0.44 + 0.02), m


def f(*v):
    return " ".join(f"{x:.6g}" for x in np.concatenate([np.ravel(np.asarray(x, float)) for x in v]))


def cyl_inertia(m, p0, p1, r):
    """Inertia of a solid cylinder (mass m, axis p0->p1, radius r) about its own centre."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = p1 - p0; L = np.linalg.norm(a); a = a / L
    ia, it = 0.5 * m * r * r, m * (3 * r * r + L * L) / 12
    return it * np.eye(3) + (ia - it) * np.outer(a, a)


def combine(parts):
    """parts: list of (mass, com, inertia_about_com). Returns mass, com, inertia about com."""
    M = sum(p[0] for p in parts)
    c = sum(p[0] * np.asarray(p[1]) for p in parts) / M
    I = np.zeros((3, 3))
    for m, x, Ic in parts:
        d = np.asarray(x) - c
        I += Ic + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, c, I


def inertial(parent, parts):
    M, c, I = combine(parts)
    ET.SubElement(parent, "inertial", mass=f"{M:.6g}", pos=f(c),
                  fullinertia=f(I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
    return M, c


def seg(name, length, mass_total, p0, p1, r):
    frac, comf = DELEVA[name]
    m = frac * mass_total
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return (m, p0 + comf * (p1 - p0), cyl_inertia(m, p0, p1, r))

# ---------------------------------------------------------------------------------------
# Hardware
#   Linear actuators (all 24 V, stroke 300 mm, built-in limit switches, IP54):
#     DTL  DC 24V 300 mm 10 mm/s 3000 N, 475 mm retracted (duxe.ru)          <- default
#     AL01-24-A1-475-300-C11      4 mm/s 6000 N, 475 mm retracted (42unita.ru)
#     PTL  DC 24V 300 mm 36 mm/s 2000 N, retracted length not published -> 475 assumed
#   Drive motors Norton ZW9012L3A-S 150 W 24 V: 1.0 N*m rated, 1500 rpm rated, 2000 rpm
#   no load, gear 16.75 -> 13.4 N*m at the wheel with eta 0.8, ~1.5 m/s at rated speed.
# ---------------------------------------------------------------------------------------
ACTUATORS = {
    "DTL":  dict(name="DTL 24 В · 3000 Н · 10 мм/с", force=3000.0, speed=0.010, retracted=0.475, mass=3.8,
                 note="duxe.ru, 5 170 ₽, 475 мм во втянутом"),
    "DTL6000": dict(name="DTL 24 В · 6000 Н · 5 мм/с", force=6000.0, speed=0.005, retracted=0.475, mass=4.0,
                    note="duxe.ru, 5 170 ₽; длина во втянутом как у DTL 3000 (475 мм)"),
    "2xDTL": dict(name="2× DTL 24 В · 3000 Н · 10 мм/с (параллельно)", force=6000.0, speed=0.010, retracted=0.475,
                  mass=7.6, note="два привода рядом, 2×5 170 ₽; нужен контроллер с синхронизацией"),
    "2xDTL6000": dict(name="2× DTL 24 В · 6000 Н · 5 мм/с (параллельно)", force=12000.0, speed=0.005,
                      retracted=0.475, mass=8.0, note="два DTL 6000 рядом, 2×5 170 ₽: запас для 120–150 кг"),
    "AL01": dict(name="AL01 24 В · 6000 Н · 4 мм/с", force=6000.0, speed=0.004, retracted=0.475, mass=4.0,
                 note="42unita.ru, 3 900 ₽, 475 мм во втянутом"),
    "PTL":  dict(name="PTL 24 В · 2000 Н · 36 мм/с", force=2000.0, speed=0.036, retracted=0.475, mass=2.6,
                 note="длина во втянутом не указана, принята 475 мм"),
}
DEFAULT_ACTUATOR = "DTL6000"
ACT_STROKE = 0.300               # m (all three)
ACT_RETRACTED = 0.475
FAST_SEAT_SPEED = 0.08          # m/s, quick seat slider in the simulator (not a real actuator speed)
LOAD_SPEED_DROP = 0.3            # speed falls ~30 % from no load to rated force (typical DC actuator)
# Pivot positions so that 475 mm <-> seat 0 deg and 775 mm <-> seat 75 deg exactly
# (the actuator runs limit switch to limit switch). Old ChatGPT model: 0.300 / z 0.100.
# ESLEH-style layout (V5.2): the seat pivot is moved up/forward to the user's knee (like standing
# chairs with knee pads), the actuator leans forward from a low pin on the base frame to a bracket
# under the seat. Old CAD pivot: (0.65, 0.55); old pins: seat 0.316 behind pivot, frame (0.45, 0.089).
PIVOT_SHIFT = (0.10875, 0.0, 0.098)   # m, from the CAD seat pivot to the knee of a 175 cm user
SEAT_PIN_LOCAL = (-0.215, -0.130)     # m, actuator pin in the seat frame (x back, z down from the pivot)
FRAME_PIN = (0.36, 0.08)              # m, actuator pin on the base frame (x, z)
BATTERY_OFFSET = (-0.32, 0.0, 0.205)  # m, battery box moved to the rear rack above the drive wheels
RACK_KG = 1.2                         # steel tube rack around the battery
KNEEPAD_KG = 0.8                      # two knee pads with brackets
SEAT_PIN_BEHIND_PIVOT = -SEAT_PIN_LOCAL[0]   # kept for older scripts


def geometry(chair=None):
    """Seat-frame geometry the person and the actuator are fitted to, per chair design."""
    chair = chair or DEFAULT_CHAIR
    if chair == "esleh":
        g = dict(ce.GEOM)
        g["PIVOT"] = ce.PIVOT
        return g
    return dict(SEAT_TOP=SEAT_TOP, BACK_FACE_X=BACK_FACE_X, KNEE_TARGET_X=PIVOT_SHIFT[0], KNEE_FLEX=KNEE_FLEX,
                SEAT_PIN_LOCAL=SEAT_PIN_LOCAL, FRAME_PIN=FRAME_PIN,
                PIVOT=(0.65 + PIVOT_SHIFT[0], 0.55 + PIVOT_SHIFT[2]))

# ---------------------------------------------------------------------------------------
# Masses. The CAD frame is 40x40x2 mm tube, 4277 cm3 of material (V5_Frame_03 mesh):
# steel 7.85 g/cm3 -> 33.6 kg (that is why the CAD chair weighed 86 kg), Al 6061 -> 11.5 kg.
# Reference analogue: ESLEH Rester standing power wheelchair, net 80.4 kg with 2x250 W motors
# and a 38 Ah battery (lead-acid size) — our motors and LiFePO4 battery are much lighter.
# ---------------------------------------------------------------------------------------
FRAME_TUBE_CM3 = 4277.1 + 98.4   # frame tubes + front axle bars
FRAMES = {
    "al": dict(name="Алюминий 6061-T6 (V6: труба Ø32×1,5; V5: 40×40×2)", rho=2.70),
    "steel": dict(name="Сталь (V6: труба Ø32×1,5 как у ESLEH; V5: 40×40×2)", rho=7.85),
}
DEFAULT_FRAME = "steel"          # ESLEH: steel frame
FOOTPLATE_KG = 1.0               # aluminium footplate (not in CAD)
ELECTRONICS_KG = 1.5             # controller, joystick, wiring (not in CAD)


def part_masses(frame_mat, act, motor=None):
    al = frame_mat == "al"
    a = ACTUATORS[act]
    motor = motor or motor_profile()
    return {                       # kg, reason
        "Motor150_Left": motor["mass"], "Motor150_Right": motor["mass"],   # datasheet
        "MotorMount_Left": 0.30 if al else 0.84, "MotorMount_Right": 0.30 if al else 0.84,   # 107 cm3
        "MotorFasteners_Left": 0.12, "MotorFasteners_Right": 0.12,
        "BatteryTray": 1.03 if al else 3.0,                        # 383 cm3
        "BatteryHoldDown": 0.14 if al else 0.41,                   # 53 cm3
        "Battery_24V_20Ah": 5.5,                                   # LiFePO4 25.6 V 20 Ah
        "seat": 3.8 if al else 5.0,                                # frame + 12 mm plywood + foam + cover
        "back": 4.0 if al else 5.2,                                # frame + 10 mm plywood + foam + cover
        "lower_link": 1.75,                                        # 2 steel links, 223 cm3
        "actuator_cylinder": a["mass"] * 0.7, "actuator_rod": a["mass"] * 0.3,
        "drive_left": 3.0, "drive_right": 3.0,                     # 320 mm solid tyre, Al rim, hub
        "front_left": 3.0, "front_right": 3.0,                     # same 320 mm wheel as at the rear
        "antitip_left": 0.2, "antitip_right": 0.2,
    }


def apply_masses(root, frame, frame_mat, act, footplate_pos, motor=None):
    """Scale CAD masses/inertias to real materials; frame body gets tubes + footplate + electronics."""
    for name, m_new in part_masses(frame_mat, act, motor).items():
        b = root.find(f".//body[@name='{name}']")
        inr = b.find("inertial")
        k = m_new / float(inr.get("mass"))
        inr.set("mass", f"{m_new:.4g}")
        inr.set("diaginertia", f(*[float(x) * k for x in inr.get("diaginertia").split()]))
    inr = frame.find("inertial")
    m0, c0 = float(inr.get("mass")), np.array([float(x) for x in inr.get("pos").split()])
    I0 = np.array([float(x) for x in inr.get("diaginertia").split()])
    m_tube = FRAME_TUBE_CM3 * FRAMES[frame_mat]["rho"] / 1000
    parts = [(m_tube, c0, np.diag(I0 * m_tube / m0)),
             (FOOTPLATE_KG, np.asarray(footplate_pos), np.diag([0.01, 0.01, 0.02])),
             (ELECTRONICS_KG, np.array([0.25, 0.0, 0.22]), np.diag([0.003, 0.003, 0.003])),
             (RACK_KG, np.array([0.22 + BATTERY_OFFSET[0], 0.0, 0.3]), np.diag([0.01, 0.01, 0.01])),
             (KNEEPAD_KG, np.array([0.82, 0.0, 0.6]), np.diag([0.003, 0.003, 0.003]))]
    frame.remove(inr)
    M, c, I = combine(parts)
    el = ET.Element("inertial", mass=f"{M:.5g}", pos=f(c),
                    fullinertia=f(I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
    frame.insert(0, el)

WHEEL_R = 0.16                   # m, 320 mm drive wheels

# Drive motors (24 V brushless with gearbox, wheel mounted on the output shaft). Datasheet values:
#   T  - rated torque on the motor shaft, N*m      n_rated - rated speed, rpm (motor)
#   n0 - no-load speed, rpm (motor)                ratio   - gearbox ratio
#   eta - gearbox efficiency (assumed 0.8)         I0 / I  - no-load / rated current, A
MOTORS = {
    "N150": dict(name="Norton ZW9012L3A-S · 150 Вт", T=1.0, n_rated=1500, n0=2000, ratio=30.1, eta=0.8,
                 I0=1.5, I=9.0, mass=1.7, note="стоит сейчас · dcmotorfactory.com"),
    "N1502": dict(name="Norton ZW9012L3A-S · 150 Вт", T=1.0, n_rated=1500, n0=2000, ratio=30.1, eta=0.8,
                 I0=1.5, I=9.0, mass=1.7, note="стоит сейчас · dcmotorfactory.com"),
    "N200": dict(name="Norton ZW9012L3A-S · 200 Вт", T=1.3, n_rated=1500, n0=2000, ratio=16.75, eta=0.8,
                 I0=2.0, I=11.0, mass=1.8, note="та же серия, мощнее · dcmotorfactory.com"),
    "N2002": dict(name="Norton ZW9012L3A-S · 200 Вт", T=1.3, n_rated=1500, n0=2000, ratio=30.1, eta=0.8,
                 I0=2.0, I=11.0, mass=1.8, note="та же серия, мощнее · dcmotorfactory.com"),
    "N200F": dict(name="Norton ZW1012L4F-S · 200 Вт", T=0.95, n_rated=2000, n0=2600, ratio=16.75, eta=0.8,
                  I0=1.8, I=10.9, mass=3.0, note="быстрее, тяжелее · dcmotorfactory.com"),
    "custom": dict(name="Свой мотор", T=1.0, n_rated=1500, n0=2000, ratio=16.75, eta=0.8,
                   I0=1.5, I=9.0, mass=1.7, note="параметры задаются ниже"),
}
DEFAULT_MOTOR = "N150"
MOTOR_FIELDS = ("T", "n_rated", "n0", "ratio", "eta", "I0", "I", "mass")


def motor_profile(key=DEFAULT_MOTOR, params=None):
    """Datasheet values + derived wheel numbers."""
    m = dict(MOTORS[key])
    if key == "custom" and params:
        for k in MOTOR_FIELDS:
            if k in params and params[k] not in (None, ""):
                m[k] = float(params[k])
    m["n0"] = max(m["n0"], m["n_rated"] * 1.05)
    w = lambda rpm: rpm / m["ratio"] * 2 * math.pi / 60          # wheel rad/s
    m["wheel_torque"] = m["T"] * m["ratio"] * m["eta"]         # N*m continuous at the wheel
    m["w_rated"], m["w0"] = w(m["n_rated"]), w(m["n0"])
    m["v_rated"], m["v0"] = m["w_rated"] * WHEEL_R, m["w0"] * WHEEL_R
    m["kt"] = m["T"] / max(m["I"] - m["I0"], 0.1)              # N*m per A on the motor shaft
    m["key"] = key
    return m


WHEEL_TORQUE = motor_profile()["wheel_torque"]                 # 13.4 N*m for the current motor
WHEEL_FORCE = WHEEL_TORQUE / WHEEL_R
CASTER_TRAIL = 0.045             # m, retained fork offset from the original caster layout
ANTITIP_GAP = 0.025              # m, anti-tip wheels lifted off the floor (they scrub in turns otherwise)


def apply_hardware(root, frame, seat, act=DEFAULT_ACTUATOR, motor=None, G=None, chair="v5"):
    """Actuator geometry/speed/force, motor limits, fixed four-wheel drive, brakes."""
    world = root.find("worldbody")
    A = ACTUATORS[act]
    M = motor or motor_profile()
    G = G or geometry("v5")
    pin_l, frame_pin = G["SEAT_PIN_LOCAL"], G["FRAME_PIN"]
    wheel_force = M["wheel_torque"] / WHEEL_R
    gear = 1.0 / A["speed"]                                  # ctrl 1.0 = full speed out
    vis = root.find("visual")
    glob = vis.find("global")
    if glob is None:
        glob = ET.SubElement(vis, "global")
    glob.set("offwidth", "1920"); glob.set("offheight", "1200")     # video in the control panel
    root.find("option").set("noslip_iterations", "4")      # no creeping of locked tyres on slopes
    # --- actuator geometry ---
    seat.find("site[@name='seat_actuator_tip']").set("pos", f(pin_l[0], 0, pin_l[1]))
    seat_pos = np.array([float(v) for v in seat.get("pos").split()])
    P = np.array([frame_pin[0], 0, frame_pin[1]])
    tip0 = seat_pos + np.array([pin_l[0], 0, pin_l[1]])
    # bracket under the seat plate holding the upper pin (CAD variant)
    if chair != "esleh":
      ET.SubElement(seat, "geom", type="box", pos=f(SEAT_PIN_LOCAL[0], 0, (SEAT_PIN_LOCAL[1] - PIVOT_SHIFT[2] - 0.015) / 2),
                  size=f(0.02, 0.03, abs(SEAT_PIN_LOCAL[1] + PIVOT_SHIFT[2] + 0.015) / 2 + 0.01),
                  rgba="0.1 0.1 0.12 1", contype="0", conaffinity="0", group="1")
    v = tip0 - P
    L0 = np.linalg.norm(v); u = v / L0
    assert abs(L0 - ACT_RETRACTED) < 1e-3, L0
    nrm = np.array([u[2], 0, -u[0]])                         # perpendicular, pointing forward
    cyl = frame.find("body[@name='actuator_cylinder']")
    cyl.set("pos", f(P))
    for g in cyl.findall("geom"):
        cyl.remove(g)
    body = dict(rgba="0.15 0.15 0.16 1", contype="0", conaffinity="0", group="1")
    ET.SubElement(cyl, "geom", name="act_tube", type="cylinder", size="0.022",
                  fromto=f(u * 0.03, u * (L0 - 0.045)), rgba="0.75 0.76 0.78 1", contype="0", conaffinity="0", group="1")
    ET.SubElement(cyl, "geom", name="act_motor", type="cylinder", size="0.028",
                  fromto=f(u * 0.02 + nrm * 0.052, u * 0.19 + nrm * 0.052), **body)
    ET.SubElement(cyl, "geom", name="act_gearbox", type="box", size=f(0.03, 0.03, 0.035),
                  pos=f(u * 0.045 + nrm * 0.026), **body)
    ET.SubElement(cyl, "geom", name="act_rear_eye", type="cylinder", size="0.012 0.02",
                  quat="0.7071068 0.7071068 0 0", **body)
    rod = cyl.find("body[@name='actuator_rod']")
    rod.set("pos", f(v))
    for g in rod.findall("geom"):
        rod.remove(g)
    ET.SubElement(rod, "geom", name="act_rod", type="cylinder", size="0.01",
                  fromto=f(-u * 0.36, -u * 0.01), rgba="0.85 0.86 0.88 1", contype="0", conaffinity="0", group="1")
    ET.SubElement(rod, "geom", name="act_front_eye", type="cylinder", size="0.011 0.018",
                  quat="0.7071068 0.7071068 0 0", **body)
    j = rod.find("joint[@name='actuator_extension']")
    j.set("axis", f(u)); j.set("range", f(0, ACT_STROKE))
    j.set("damping", "50")
    # reflected motor+screw inertia (lead screw, not back-drivable); scales with the gear ratio squared
    j.set("armature", f"{max(200.0, 2000.0 * (0.004 / A['speed']) ** 2):.4g}")

    # --- actuators ---
    acts = root.find("actuator")
    for a in list(acts):
        acts.remove(a)
    # Linear actuator: lead-screw drive = constant speed, self-locking when stopped
    # (integrated-velocity servo: the slider sets the speed, the position is held at 0).
    # ctrl +1 = extend (stand up) at 4 mm/s, -1 = retract, 0 = hold. Force limited to 6000 N.
    ET.SubElement(acts, "intvelocity", name="lift_actuator", joint="actuator_extension", gear=f(gear),
                  kp=f(1e7 / gear ** 2), kv=f(2e5 / gear ** 2), ctrlrange="-1 1",
                  actrange=f(0, ACT_STROKE * gear),
                  forcerange=f(-A["force"] / gear, A["force"] / gear))
    # Four-wheel skid steering.  Each physical side motor drives both wheels on its
    # side, so its available rim force is split between the front and rear wheel.
    # Keeping the sum unchanged avoids inventing twice the motor power/torque.
    for side in ("left", "right"):
        ET.SubElement(acts, "velocity", name=f"drive_{side}", joint=f"drive_{side}_spin", gear=f(WHEEL_R),
                      kv="400", ctrlrange=f(-M["v0"], M["v0"]),
                      forcerange=f(-wheel_force / 2, wheel_force / 2))
        ET.SubElement(acts, "velocity", name=f"drive_front_{side}", joint=f"front_{side}_spin", gear=f(WHEEL_R),
                      kv="400", ctrlrange=f(-M["v0"], M["v0"]),
                      forcerange=f(-wheel_force / 2, wheel_force / 2))
    # Electromagnetic parking brakes lock all four driven wheels.
    for axle, prefix in (("rear", "drive"), ("front", "front")):
        for side in ("left", "right"):
            name = f"brake_lock_{side}" if axle == "rear" else f"brake_lock_front_{side}"
            ET.SubElement(root.find("equality"), "joint", name=name, joint1=f"{prefix}_{side}_spin",
                          polycoef="0 0 0 0 0", active="false", solref="0.004 1")
            brake = f"brake_{side}" if axle == "rear" else f"brake_front_{side}"
            ET.SubElement(acts, "damper", name=brake, joint=f"{prefix}_{side}_spin", kv="60", ctrlrange="0 1")

    # --- fixed powered front forks for true four-wheel skid/tank steering ---
    for side in ("left", "right"):
        wb = frame.find(f"body[@name='front_{side}']")
        pos = np.array([float(x) for x in wb.get("pos").split()])
        pos[1] = math.copysign(ce.WHEEL_TRACK / 2, pos[1])
        pos[2] = WHEEL_R
        contact = wb.find(f"geom[@name='front_{side}_contact']")
        contact.set("size", f"{WHEEL_R} 0.0425")
        rear_inertial = frame.find(f"body[@name='drive_{side}']/inertial")
        front_inertial = wb.find("inertial")
        front_inertial.attrib.update(rear_inertial.attrib)
        hub = next((g for g in wb.findall("geom") if g is not contact), None)
        if hub is not None and hub.get("type") == "cylinder":
            hub.set("size", "0.105 0.0455")
        frame.remove(wb)
        fork = ET.SubElement(frame, "body", name=f"front_fork_{side}", pos=f(pos + [CASTER_TRAIL, 0, 0]))
        ET.SubElement(fork, "inertial", mass="0.3", pos=f(-CASTER_TRAIL / 2, 0, 0.04), diaginertia="0.0004 0.0004 0.0002")
        ET.SubElement(fork, "geom", type="box", size=f(CASTER_TRAIL / 2 + 0.01, 0.03, 0.006), pos=f(-CASTER_TRAIL / 2, 0, 0.115),
                      rgba="0.1 0.1 0.12 1", contype="0", conaffinity="0", group="1")
        for sgn in (1, -1):
            ET.SubElement(fork, "geom", type="box", size=f(0.012, 0.004, 0.06), pos=f(-CASTER_TRAIL, sgn * 0.028, 0.055),
                          rgba="0.1 0.1 0.12 1", contype="0", conaffinity="0", group="1")
        wb.set("pos", f(-CASTER_TRAIL, 0, 0))
        fork.append(wb)
    # --- tyres: one stable spherical contact point + visible cylinder ---
    for w in ("drive_left", "drive_right", "front_left", "front_right"):
        g = root.find(f".//geom[@name='{w}_contact']")
        r, hw_ = [float(x) for x in g.get("size").split()]
        body = root.find(f".//body[@name='{w}']")
        ET.SubElement(body, "geom", type="cylinder", size=f(r, hw_), quat=g.get("quat"), rgba=g.get("rgba"),
                      contype="0", conaffinity="0", group="1", mass="0")
        g.attrib.pop("quat", None)
        g.set("type", "sphere"); g.set("size", f(r)); g.set("rgba", "0 0 0 0"); g.set("group", "3")

        # Более мягкий контакт шин
        if w.startswith(("drive", "front")):
            g.set("solref", "0.06 1")
            g.set("solimp", "0.7 0.9 0.005 0.5 2")
        if w.startswith(("drive", "front")):
            # The higher priority makes this tyre value override the floor's generic
            # friction.  Low scrub friction is required for a fixed-wheel skid turn.
            g.set("priority", "1")
            g.set("friction", "0.5 0.005 0.0001")
    # Auxiliary rear wheels removed from the ESLEH model.


def _shift(el, d):
    p = np.array([float(x) for x in el.get("pos", "0 0 0").split()])
    el.set("pos", f(p - d))


def relocate_pivot(root, frame, seat):
    """Move the seat pivot to the knee; keep the backrest parallelogram (back stays vertical)."""
    d = np.array(PIVOT_SHIFT)
    p0 = np.array([float(x) for x in seat.get("pos").split()])
    p1 = p0 + d
    seat.set("pos", f(p1))
    for el in list(seat):
        if el.tag in ("geom", "site", "body", "inertial"):
            _shift(el, d)
    # lower link of the parallelogram: pivot 0.25 m below the seat pivot, same vector to the back pin
    link = frame.find("body[@name='lower_link']")
    link.set("pos", f(p1 - np.array([0, 0, 0.25])))
    _shift(link.find("inertial"), d)
    pin = link.find("site[@name='lower_back_pin']")
    _shift(pin, d)
    pin_p = np.array([float(x) for x in pin.get("pos").split()])
    for g in link.findall("geom"):
        link.remove(g)
    for y in (0.274, -0.274):
        ET.SubElement(link, "geom", type="capsule", size="0.012", fromto=f(0, y, 0, pin_p[0], y, pin_p[2]),
                      rgba="0.5 0.53 0.57 1", contype="0", conaffinity="0", group="1")
    # side brackets carrying the raised pivot + pivot axle
    for y in (0.29, -0.29):
        ET.SubElement(frame, "geom", type="capsule", size="0.014",
                      fromto=f(p0[0] - 0.02, y, 0.50, p1[0], y, p1[2]), rgba="0.1 0.38 0.8 1",
                      contype="0", conaffinity="0", group="1")
        ET.SubElement(frame, "geom", type="capsule", size="0.014",
                      fromto=f(p1[0], y, p1[2], p1[0], y, p1[2] - 0.25), rgba="0.1 0.38 0.8 1",
                      contype="0", conaffinity="0", group="1")
    ET.SubElement(frame, "geom", type="cylinder", size="0.012", fromto=f(p1[0], -0.3, p1[2], p1[0], 0.3, p1[2]),
                  rgba="0.6 0.62 0.66 1", contype="0", conaffinity="0", group="1")


def move_battery_to_rack(root, frame):
    """Battery box on a rear rack above the drive wheels (as on the ESLEH Rester)."""
    o = np.array(BATTERY_OFFSET)
    for n in ("Battery_24V_20Ah", "BatteryTray", "BatteryHoldDown"):
        frame.find(f"body[@name='{n}']").set("pos", f(o))
    g = frame.find("geom[@name='battery_contact']")
    g.set("pos", f(np.array([float(x) for x in g.get("pos").split()]) + o))
    root.find("asset/mesh[@name='BatteryTray_00']").set("scale", "1 0.8 1")   # fits between the wheels
    tube = dict(type="capsule", size="0.011", rgba="0.08 0.08 0.09 1", contype="0", conaffinity="0", group="1")
    xa, xb, y, zt = 0.1 + o[0] - 0.03, 0.34 + o[0] + 0.03, 0.15, 0.17 + o[2] + 0.04
    for a_, b_ in (((xa, y), (xb, y)), ((xa, -y), (xb, -y)), ((xa, -y), (xa, y)), ((xb, -y), (xb, y)),
                   (((xa + xb) / 2, -y), ((xa + xb) / 2, y))):
        ET.SubElement(frame, "geom", fromto=f(a_[0], a_[1], zt, b_[0], b_[1], zt), **tube)
    for x_ in (xa, xb):
        for y_ in (y, -y):
            ET.SubElement(frame, "geom", fromto=f(x_, y_, zt, x_, y_, 0.17), **tube)


def add_test_track(world):
    """Threshold, accessibility ramp and three separate 10/15/30 degree hills."""
    st = dict(contype="2", conaffinity="1", rgba="0.72 0.74 0.78 1", group="0")

    def slope_box(name, xa, xb, za, zb, width=1.2, rgba=None):
        a = math.atan2(zb - za, xb - xa)
        L = math.hypot(xb - xa, zb - za)
        n = np.array([-math.sin(a), 0, math.cos(a)])
        c = np.array([(xa + xb) / 2, 0, (za + zb) / 2]) - n * 0.025
        attrs = dict(st)
        if rgba is not None:
            attrs["rgba"] = rgba
        ET.SubElement(world, "geom", name=name, type="box", pos=f(c), size=f(L / 2, width, 0.025),
                      euler=f(0, -a, 0), **attrs)

    ET.SubElement(world, "geom", name="threshold_30mm", type="box", pos="2.5 0 0.015", size="0.03 1.2 0.015",
                  rgba="0.85 0.55 0.2 1", contype="2", conaffinity="1")
    h = 2.0 / 12
    x0 = 5.0
    for name, xa, xb, za, zb in (("ramp_up_1_12", x0, x0 + 2.0, 0.0, h),
                                 ("ramp_down_1_8", x0 + 4.0, x0 + 4.0 + 8 * h, h, 0.0)):
        slope_box(name, xa, xb, za, zb)
    ET.SubElement(world, "geom", name="platform", type="box", pos=f(x0 + 3.0, 0, h / 2), size=f(1.0, 1.2, h / 2), **st)

    # Independent up-and-down obstacle hills along +x.  Their 0.1 m rise keeps a
    # 0.5 km/h ascent inside the five-second automatic traction window.
    hill_starts = {10: 12.5, 15: 20.5, 30: 26.5}
    colours = {10: "0.35 0.65 0.38 1", 15: "0.88 0.65 0.22 1", 30: "0.82 0.32 0.25 1"}
    height, plateau = 0.1, 0.8
    for deg, start in hill_starts.items():
        run = height / math.tan(math.radians(deg))
        top0, top1, end = start + run, start + run + plateau, start + 2 * run + plateau
        slope_box(f"hill_{deg}_up", start, top0, 0, height, rgba=colours[deg])
        ET.SubElement(world, "geom", name=f"hill_{deg}_top", type="box",
                      pos=f((top0 + top1) / 2, 0, height / 2), size=f(plateau / 2, 1.2, height / 2),
                      rgba=colours[deg], contype="2", conaffinity="1", group="0")
        slope_box(f"hill_{deg}_down", top1, end, height, 0, rgba=colours[deg])


    # Separate 20 m ascent at 15 degrees, followed by a 2 m landing.
    long_angle = math.radians(15.0)
    long_length = 20.0  # measured along the slope, not horizontally
    long_run = long_length * math.cos(long_angle)
    long_rise = long_length * math.sin(long_angle)
    long_start = 34.0
    slope_box("long_ramp_15_up", long_start, long_start + long_run,
              0.0, long_rise, width=1.2, rgba="0.23 0.58 0.83 1")
    ET.SubElement(world, "geom", name="long_ramp_15_top", type="box",
                  pos=f(long_start + long_run + 1.0, 0, long_rise - 0.025),
                  size="1 1.2 0.025", rgba="0.23 0.58 0.83 1",
                  contype="2", conaffinity="1", group="0")


def build(mass=80.0, height=1.75, frame_mat=DEFAULT_FRAME, act=DEFAULT_ACTUATOR, motor=DEFAULT_MOTOR,
          motor_params=None, chair=DEFAULT_CHAIR, src="chair_drive.xml"):
    G = geometry(chair)
    M = motor_profile(motor, motor_params)
    if chair == "esleh":
        tree = ce.base_tree(frame_mat, M["mass"], ACTUATORS[act]["mass"])
    else:
        tree = ET.parse(HERE / src)
    root = tree.getroot()
    root.set("model", "Vert V5 + human")          # fixed: the simulator hot-swaps models of equal layout
    root.insert(0, ET.Comment(f" chair {chair}, frame {frame_mat}, actuator {act}, motor {motor}, human {mass:g} kg / {height:g} m "))
    world = root.find("worldbody")
    frame = world.find("body[@name='frame']")
    seat = frame.find("body[@name='seat']")
    back = seat.find("body[@name='back']")

    s = height / 1.75                                   # length scale
    k = math.sqrt((mass / height) / (80 / 1.75))        # girth scale
    skin = dict(rgba="0.93 0.72 0.58 1", contype="0", conaffinity="0", group="2")
    cloth = dict(rgba="0.25 0.32 0.45 1", contype="0", conaffinity="0", group="2")

    # ---- anthropometry (Winter / de Leva proportions) ----
    Lt, Ls = 0.245 * height, 0.246 * height             # thigh, shank
    Ltr = 0.288 * height                                # hip joint -> shoulder
    Lua, Lfa, Lh = 0.186 * height, 0.146 * height, 0.108 * height
    ankle_h, foot_L = 0.039 * height, 0.152 * height
    r_th, r_sh, r_ar = 0.068 * k, 0.045 * k, 0.040 * k
    hw = 0.09 * s                                       # half distance between hips
    sw = 0.16 * s                                       # torso half width
    td = 0.10 * k                                       # torso half depth
    buttock = td                                        # hip joint ahead of backrest face

    # seat depth "adjusted" so the knee lands on the seat pivot (buttocks may compress up to 4 cm)
    hip_x = G["KNEE_TARGET_X"] - Lt
    kf = G["KNEE_FLEX"]
    # adjustable backrest: for long thighs the backrest pad moves back so the buttocks fit
    back_adj = min(0.0, hip_x - buttock - G["BACK_FACE_X"])
    for g in back.findall("geom"):
        if g.get("fromto"):
            ft_ = np.array([float(x) for x in g.get("fromto").split()])
            g.set("fromto", f(ft_ + np.array([back_adj, 0, 0, back_adj, 0, 0])))
        else:
            p_ = np.array([float(x) for x in g.get("pos", "0 0 0").split()])
            g.set("pos", f(p_ + np.array([back_adj, 0, 0])))
    hip = np.array([hip_x, 0, G["SEAT_TOP"] + r_th])             # in the seat frame
    if chair == "esleh":
        # knee exactly on the pivot axis: the cushion is fitted to the thigh (like setting the knee pad / seat
        # height on the real chair). Otherwise the knee moves while standing, the near-vertical shank acts as a
        # self-locking strut against the footplate and the actuator stalls.
        hip[2] = 0.0
        dz = (G["SEAT_TOP"] + r_th) - 0.0          # how much higher the cushion top must be
        ce.fit_cushion(seat, -dz)
    knee_x = Lt

    # ---- pelvis + thighs: rigidly strapped to the seat ----
    pel = ET.SubElement(seat, "body", name="human_pelvis", pos=f(hip))
    parts = [(DELEVA["pelvis"][0] * mass, np.array([-0.01, 0, 0.04 * s]),
              cyl_inertia(DELEVA["pelvis"][0] * mass, [0, -hw - r_th, 0], [0, hw + r_th, 0], 0.11 * k))]
    ET.SubElement(pel, "geom", name="h_pelvis", type="capsule", size=f(0.085 * k),
                  fromto=f(0, -hw, 0.02, 0, hw, 0.02), **cloth)
    for side, y in (("L", hw), ("R", -hw)):
        parts.append(seg("thigh", Lt, mass, [0, y, 0], [knee_x, y, 0], r_th))
        ET.SubElement(pel, "geom", name=f"h_thigh_{side}", type="capsule", size=f(r_th),
                      fromto=f(0, y, 0, knee_x, y, 0), **cloth)
    inertial(pel, parts)
    ET.SubElement(pel, "site", name="human_seat_site", pos="0 0 0", size="0.01", group="4")

    # ---- trunk, head, arms: hinge at the hip, kept parallel to the backrest ----
    tor = ET.SubElement(pel, "body", name="human_torso", pos="0 0 0")
    ET.SubElement(tor, "joint", name="hip", axis="0 1 0", range="-1.75 0.35", damping="15")
    sh_z = Ltr
    head_c = np.array([0.01, 0, sh_z + 0.07 * s + 0.11 * s])
    parts = [
        (DELEVA["mid_trunk"][0] * mass, np.array([-0.015, 0, 0.30 * Ltr]),
         cyl_inertia(DELEVA["mid_trunk"][0] * mass, [0, 0, 0.05], [0, 0, 0.55 * Ltr], 0.13 * k)),
        (DELEVA["upper_trunk"][0] * mass, np.array([-0.015, 0, 0.75 * Ltr]),
         cyl_inertia(DELEVA["upper_trunk"][0] * mass, [0, 0, 0.55 * Ltr], [0, 0, Ltr], 0.14 * k)),
        (DELEVA["head"][0] * mass, head_c, 0.4 * DELEVA["head"][0] * mass * (0.1 * s) ** 2 * np.eye(3)),
    ]
    ET.SubElement(tor, "geom", name="h_torso", type="box", pos=f(0, 0, Ltr / 2 + 0.01),
                  size=f(td, sw, Ltr / 2), **cloth)
    ET.SubElement(tor, "geom", name="h_neck", type="capsule", size=f(0.05 * k),
                  fromto=f(0.01, 0, sh_z, 0.01, 0, sh_z + 0.08 * s), **skin)
    ET.SubElement(tor, "geom", name="h_head", type="sphere", size=f(0.105 * s), pos=f(head_c), **skin)
    for side, sg in (("L", 1), ("R", -1)):
        y = sg * (sw + r_ar + 0.005)
        shoulder = [0, y, sh_z - 0.03 * s]
        elbow = [0.02, y, sh_z - 0.03 * s - Lua]
        wrist = [0.02 + Lfa, y, elbow[2]]
        tip = [wrist[0] + Lh, y, elbow[2]]
        parts += [seg("upper_arm", Lua, mass, shoulder, elbow, r_ar),
                  seg("forearm", Lfa, mass, elbow, wrist, r_ar * 0.85),
                  seg("hand", Lh, mass, wrist, tip, r_ar * 0.7)]
        ET.SubElement(tor, "geom", name=f"h_uarm_{side}", type="capsule", size=f(r_ar),
                      fromto=f(shoulder, elbow), **cloth)
        ET.SubElement(tor, "geom", name=f"h_farm_{side}", type="capsule", size=f(r_ar * 0.85),
                      fromto=f(elbow, wrist), **cloth)
        ET.SubElement(tor, "geom", name=f"h_hand_{side}", type="capsule", size=f(r_ar * 0.7),
                      fromto=f(wrist, tip), **skin)
    inertial(tor, parts)

    # ---- shanks + feet: hinge at the knees, feet stand on the footplate ----
    for side, y in (("L", hw), ("R", -hw)):
        shk = ET.SubElement(pel, "body", name=f"human_shank_{side}", pos=f(knee_x, y, 0),
                            euler=f(0, kf, 0))
        ET.SubElement(shk, "joint", name=f"knee_{side}", axis="0 1 0",
                      range=f(-math.pi / 2 - kf, 1.2), damping="4")
        ankle = [0, 0, -Ls]
        inertial(shk, [seg("shank", Ls, mass, [0, 0, 0], ankle, r_sh)])
        ET.SubElement(shk, "geom", name=f"h_shank_{side}", type="capsule", size=f(r_sh),
                      fromto=f(0, 0, 0, ankle), **cloth)
        # foot: own body with an ankle hinge so it stays flat on the footplate
        ft = ET.SubElement(shk, "body", name=f"human_foot_{side}", pos=f(ankle), euler=f(0, -kf, 0))
        ET.SubElement(ft, "joint", name=f"ankle_{side}", axis="0 1 0", range="-0.9 0.9", damping="1")
        heel, toe = [-0.05 * s, 0, -ankle_h / 2], [foot_L - 0.05 * s, 0, -ankle_h / 2]
        inertial(ft, [seg("foot", foot_L, mass, heel, toe, 0.04 * k)])
        ET.SubElement(ft, "geom", name=f"h_foot_{side}", type="box",
                      pos=f((heel[0] + toe[0]) / 2, 0, -ankle_h / 2), size=f(foot_L / 2, 0.045 * k, ankle_h / 2),
                      contype="8", conaffinity="0", condim="3", friction="0.5 0.005 0.0001",
                      rgba="0.15 0.15 0.17 1", group="2")

    # ---- supports on the chair (contact surfaces only, no mass) ----
    seat_pos = np.array([float(v) for v in seat.get("pos").split()])
    ankle_x = knee_x - Ls * math.sin(kf)                          # seat frame, sitting
    sole_z = seat_pos[2] + hip[2] - Ls * math.cos(kf) - ankle_h - 0.001   # frame coords
    fp_x = seat_pos[0] + hip[0] + ankle_x + foot_L / 2 - 0.05 * s
    # visible plate under the feet + a longer invisible contact surface the feet can slide back on
    if chair != "esleh":
        ET.SubElement(frame, "geom", name="footplate", type="box", pos=f(fp_x, 0, sole_z - 0.008),
                      size=f(0.15, 0.2, 0.008), rgba="0.35 0.37 0.4 1", contype="0", conaffinity="0", group="1")
    heel_x = seat_pos[0] + hip[0] + ankle_x - 0.05 * s
    x0, x1 = heel_x - 0.06, fp_x + 0.15
    ET.SubElement(frame, "geom", name="foot_support", type="box", pos=f((x0 + x1) / 2, 0, sole_z - 0.008),
                  size=f((x1 - x0) / 2, 0.2, 0.008), rgba="0 0 0 0", contype="0", conaffinity="8",
                  group="3", friction="0.5 0.005 0.0001")
    # heel loops: a stop right behind each heel, the feet cannot slide back under the seat
    for y in (hw, -hw):
        ET.SubElement(frame, "geom", name=f"heel_loop_{'L' if y > 0 else 'R'}", type="box",
                      pos=f(heel_x - 0.05, y, sole_z + 0.03), size=f(0.008, 0.06, 0.03),
                      rgba="0.12 0.12 0.14 1", contype="0", conaffinity="8", group="1")
    # knee pads in front of the knees (on the seat pivot line) - CAD variant; V6 has a shin pad
    kx = seat_pos[0] + hip[0] + knee_x + r_sh + 0.022
    kz = seat_pos[2] + hip[2] - 0.03
    for y in ((hw, -hw) if chair != "esleh" else ()):
        ET.SubElement(frame, "geom", type="box", pos=f(kx, y, kz), size=f(0.018, 0.055, 0.065),
                      rgba="0.18 0.18 0.2 1", contype="0", conaffinity="0", group="1")
        ET.SubElement(frame, "geom", type="capsule", size="0.01",
                      fromto=f(kx + 0.02, y, kz, kx + 0.05, y, sole_z + 0.02), rgba="0.1 0.1 0.12 1",
                      contype="0", conaffinity="0", group="1")
    ET.SubElement(frame, "site", name="footplate_site", type="box", pos=f((x0 + x1) / 2, 0, sole_z),
                  size=f((x1 - x0) / 2 + 0.03, 0.23, 0.05), rgba="0 1 0 0.15", group="4")
    ET.SubElement(frame, "site", name="chassis_site", pos="0 0 0.16", size="0.01", group="4")
    if chair == "esleh":
        knee_w = seat_pos + hip + np.array([knee_x, 0, 0])
        ankle_w = seat_pos + hip + np.array([ankle_x, 0, -Ls * math.cos(kf)])
        lr_mass, lr_pos = ce.add_legrest(root, frame, sole_z, heel_x, heel_x + foot_L, knee_w, ankle_w, hw, r_sh, frame_mat)
        apply_hardware(root, frame, seat, act, M, G, chair)
        ce.finish_frame(frame, [(lr_mass, lr_pos)])
    else:
        relocate_pivot(root, frame, seat)
        move_battery_to_rack(root, frame)
        apply_hardware(root, frame, seat, act, M, G, chair)
        apply_masses(root, frame, frame_mat, act, [fp_x, 0, sole_z - 0.008], M)
    # Удаляем дополнительные колёса вместе с их телами
    EXTRA_WHEELS = (
        "antitip_left",
        "antitip_right",
        "footroller_left",
        "footroller_right",
    )

    for name in EXTRA_WHEELS:
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "body" and child.get("name") == name:
                    parent.remove(child)
    add_test_track(root.find("worldbody"))
    wheels = ["drive_left", "drive_right", "front_left", "front_right"]
    for w in wheels:
        b = root.find(f".//body[@name='{w}']")
        r = float(b.find("geom").get("size").split()[0])
        ET.SubElement(b, "site", name=f"{w}_site", type="sphere", size=f(r + 0.03), rgba="1 0 0 0.1", group="4")

    # ---- actuators: slider-friendly speed command instead of raw torques ----
    tendon = root.find("tendon")
    if tendon is None:
        tendon = ET.Element("tendon")
        root.insert(list(root).index(root.find("equality")), tendon)

    # Trunk stays parallel to the backrest: the hip angle is tied to the backrest angle.
    # The trunk weight goes through the hip into the seat; only the trunk's tipping moment
    # is passed to the backrest (as it would be by a person leaning on it / a chest belt).
    ET.SubElement(root.find("equality"), "joint", name="trunk_on_backrest", joint1="hip",
                  joint2="back_compensation", polycoef="0 1 0 0 0", solref="0.01 1")

    # ---- sensors ----
    sen = root.find("sensor")
    for el in [
        ("jointactuatorfrc", dict(name="lift_force_N", joint="actuator_extension")),
        ("jointvel", dict(name="lift_speed_mps", joint="actuator_extension")),
        ("jointactuatorfrc", dict(name="wheel_left_torque_Nm", joint="drive_left_spin")),
        ("jointactuatorfrc", dict(name="wheel_right_torque_Nm", joint="drive_right_spin")),
        ("velocimeter", dict(name="chassis_velocity", site="chassis_site")),
        ("jointvel", dict(name="wheel_left_radps", joint="drive_left_spin")),
        ("jointvel", dict(name="wheel_right_radps", joint="drive_right_spin")),
        ("subtreecom", dict(name="total_com", body="frame")),
        ("force", dict(name="seat_load_local", site="human_seat_site")),
        ("touch", dict(name="footplate_load_N", site="footplate_site")),
        ("jointpos", dict(name="hip_angle", joint="hip")),
    ] + [("touch", dict(name=f"load_{w}_N", site=f"{w}_site")) for w in wheels]:
        ET.SubElement(sen, el[0], **el[1])

    ET.indent(tree, "  ")
    return tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mass", type=float, default=80.0, help="human mass, kg")
    ap.add_argument("--height", type=float, default=1.75, help="human height, m")
    ap.add_argument("--frame", choices=list(FRAMES), default=DEFAULT_FRAME, help="frame material")
    ap.add_argument("--actuator", choices=list(ACTUATORS), default=DEFAULT_ACTUATOR)
    ap.add_argument("--motor", choices=[k for k in MOTORS if k != "custom"], default=DEFAULT_MOTOR)
    ap.add_argument("--chair", choices=list(CHAIRS), default=DEFAULT_CHAIR)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "chair_human.xml")
    a = ap.parse_args()
    build(a.mass, a.height, a.frame, a.actuator, a.motor, chair=a.chair).write(a.out, encoding="utf-8", xml_declaration=True)
    print(f"written {a.out}  (human {a.mass:g} kg, {a.height:g} m, frame {a.frame}, actuator {a.actuator})")


if __name__ == "__main__":
    main()
