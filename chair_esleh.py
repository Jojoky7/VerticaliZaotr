"""V6: standing power wheelchair modelled after the ESLEH Rester (esleh.com), built from primitives.

Dimensions from the ESLEH datasheet: width 61 cm, height 107 cm, seat 46 x 44 cm, seat height 58 cm,
back height 48 cm, track 52.5 cm and standing angle 75 deg.  This simulator variant uses four equal
320 mm driven wheels instead of the original 200 mm front wheels.
Layout from the product photos: two blue side tubes with cross tubes, motors along the chair on
right-angle gearboxes in front of the rear axle, battery bag with a luggage rack behind the backrest,
actuator leaning forward from the rear of the base to the middle of the seat, thick seat cushion with
grey lateral hip pads, mesh backrest with push handles, armrests (joystick on the right), legrest
with a shin pad and a one-piece footplate with no auxiliary rollers.

The seat pivot sits at the front of the seat at knee height on two posts; the backrest is kept
vertical by a parallelogram (lower link). Body, joint and site names match the CAD-based model so the
same simulator, sensors and person model work with both.

Coordinates: metres, x forward from the rear axle, y to the left, z up (sitting position).
"""
import math
import xml.etree.ElementTree as ET
import numpy as np

# ---------------------------------------------------------------- geometry used by the builder
KNEE_UP = 0.05                       # pivot raised to the knee axis (thigh centre line above the cushion)
PIVOT = (0.70, 0.60 + KNEE_UP)       # seat pivot (x, z), on the front posts, on the user's knee axis
BACK_PIVOT_LOCAL = (-0.52, -0.10 - KNEE_UP)   # backrest hinge at the rear of the seat frame (seat frame coords)
GEOM = dict(
    SEAT_TOP=-0.02 - KNEE_UP,        # seat cushion top in the seat frame (world 0.58 m)
    BACK_FACE_X=BACK_PIVOT_LOCAL[0] + 0.03,   # backrest cushion front face in the seat frame
    KNEE_TARGET_X=0.0,               # the knee is placed on the pivot line
    KNEE_FLEX=-0.17,                 # shins inclined forward 10 deg: standing (+75 deg) leaves the knee 5 deg bent
    SEAT_PIN_LOCAL=(-0.2176, -0.1515), # single-drive legacy layout; builder supplies geometry for a pair
    FRAME_PIN=(0.24, 0.09),
)
WHEEL_TRACK = 0.525
FRONT_X, FRONT_Y = 0.70, WHEEL_TRACK / 2   # front axle, same track and wheels as at the rear
HEAD_X = FRONT_X + 0.045             # caster head tubes / front cross tube
FOOTROLLER_R, FOOTROLLER_GAP = 0.028, 0.035   # rollers clear a 30 mm threshold

BLUE = "0.13 0.22 0.58 1"
BLACK = "0.07 0.07 0.08 1"
DARK = "0.16 0.17 0.19 1"
GREY_PAD = "0.40 0.41 0.43 1"
SILVER = "0.72 0.74 0.77 1"
TYRE = "0.16 0.16 0.17 1"
VIS = dict(contype="0", conaffinity="0", group="1")

# tube mass per metre: (steel 7.85, aluminium 2.70) for wall 1.5 mm
def tube_kg_per_m(d_mm, rho):
    return math.pi * ((d_mm / 2) ** 2 - (d_mm / 2 - 1.5) ** 2) * 1e-6 * rho * 1000


def f(*v):
    return " ".join(f"{x:.6g}" for x in np.concatenate([np.ravel(np.asarray(x, float)) for x in v]))


class Tubes:
    """Collects frame tubes: draws them and sums their length per diameter."""
    def __init__(self, parent):
        self.parent, self.len = parent, {}

    def add(self, a, b, d_mm=32, rgba=BLUE, mirror=False):
        pts = [(a, b)] + ([((a[0], -a[1], a[2]), (b[0], -b[1], b[2]))] if mirror else [])
        for p, q in pts:
            ET.SubElement(self.parent, "geom", type="capsule", size=f(d_mm / 2000), fromto=f(p, q), rgba=rgba, **VIS)
            self.len[d_mm] = self.len.get(d_mm, 0) + float(np.linalg.norm(np.subtract(q, p)))

    def poly(self, pts, d_mm=32, rgba=BLUE, mirror=False):
        for a, b in zip(pts[:-1], pts[1:]):
            self.add(a, b, d_mm, rgba, mirror)

    def mass(self, rho):
        return sum(L * tube_kg_per_m(d, rho) for d, L in self.len.items())


def box_inertia(m, sx, sy, sz):
    return f(m * (sy ** 2 + sz ** 2) / 3, m * (sx ** 2 + sz ** 2) / 3, m * (sx ** 2 + sy ** 2) / 3)


def base_tree(frame_mat="steel", motor_mass=1.7, act_mass=4.0, act_count=1, geom=None):
    G = geom or GEOM
    rho = 7.85 if frame_mat == "steel" else 2.70
    al = frame_mat != "steel"
    root = ET.Element("mujoco", model="V6 ESLEH-like")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.80665", integrator="implicitfast",
                  iterations="100", tolerance="1e-10")
    dflt = ET.SubElement(root, "default")
    ET.SubElement(dflt, "joint", damping="0.15", armature="0.002")
    ET.SubElement(dflt, "geom", condim="3", friction="0.8 0.005 0.0001", solref="0.008 1")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", azimuth="140", elevation="-20")
    ET.SubElement(vis, "headlight", diffuse="0.8 0.8 0.8", ambient="0.35 0.35 0.35")
    ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -3 5", dir="0 1 -2")
    ET.SubElement(world, "geom", name="floor", type="plane", size="8 8 0.1", rgba="0.89 0.9 0.92 1",
                  contype="2", conaffinity="1")

    fr = ET.SubElement(world, "body", name="frame", pos="0 0 0.003")
    ET.SubElement(fr, "freejoint", name="chassis_free")
    T = Tubes(fr)
    # ---- base frame (blue, 32 mm) ----
    T.add((-0.16, 0.21, 0.37), (HEAD_X, 0.22, 0.33), 32, mirror=True)                 # upper side tubes
    T.poly([(0.0, 0.205, 0.22), (0.30, 0.21, 0.17), (HEAD_X, 0.22, 0.21)], 32, mirror=True)   # lower bent tubes
    T.add((-0.11, 0.21, 0.36), (0.0, 0.205, 0.22), 32, mirror=True)                  # rear uprights
    T.add((HEAD_X, 0.22, 0.20), (HEAD_X, 0.22, 0.345), 38, rgba=BLUE, mirror=True)     # caster head tubes
    T.add((-0.14, -0.21, 0.37), (-0.14, 0.21, 0.37), 25)                             # rear cross tube
    T.add((0.0, -0.205, 0.22), (0.0, 0.205, 0.22), 25)                               # motor cross tube
    T.add((0.30, -0.21, 0.17), (0.30, 0.21, 0.17), 25)                               # middle low cross tube
    T.add((0.60, -0.214, 0.335), (0.60, 0.214, 0.335), 25)                             # front cross tube (behind the calves)
    # actuator bracket (rear-lower) and front pivot posts (black)
    pin = G["FRAME_PIN"]
    if act_count == 2:
        # Welded supports connect the two clevises to the existing low cross tube.
        # The old central bracket would leave the side actuators suspended in air.
        for y in (-0.25, 0.25):
            T.add((0.30, math.copysign(0.21, y), 0.17), (0.30, y, 0.17), 25, rgba=BLACK)
            T.add((0.30, y, 0.17), (pin[0] - 0.06, y, pin[1]), 25, rgba=BLACK)
            T.add((pin[0] - 0.06, y, pin[1]), (pin[0], y, pin[1]), 25, rgba=BLACK)
    else:
        T.add((pin[0] - 0.09, -0.06, 0.19), (pin[0] - 0.09, 0.06, 0.19), 25, rgba=BLACK)
        T.add((0.0, 0.06, 0.22), (pin[0] - 0.09, 0.06, 0.19), 25, rgba=BLACK, mirror=True)
        for y in (0.035, -0.035):
            ET.SubElement(fr, "geom", type="box", pos=f(pin[0] - 0.045, y, (pin[1] + 0.19) / 2), size=f(0.05, 0.004, 0.05),
                          euler=f(0, 0.7, 0), rgba=BLACK, **VIS)
    px, pz = PIVOT
    # vertical pivot posts outside the frame: welded to the lower and upper side tubes by short stubs,
    # carry the seat pivot (top) and the lower parallelogram link pivot (250 mm lower)
    POST_Y = 0.272
    z_low, z_up = 0.206, 0.332
    T.add((px, POST_Y, z_low - 0.01), (px, POST_Y, pz + 0.015), 30, rgba=BLACK, mirror=True)       # pivot posts
    T.add((px, 0.215, z_low), (px, POST_Y, z_low), 25, rgba=BLACK, mirror=True)                     # stubs
    T.add((px, 0.222, z_up), (px, POST_Y, z_up), 25, rgba=BLACK, mirror=True)
    for z_ in (pz, pz - 0.25):              # pivot bolts through the posts (no through-axle: the legs are there)
        for s_ in (1, -1):
            ET.SubElement(fr, "geom", type="cylinder", size="0.012", fromto=f(px, s_ * 0.235, z_, px, s_ * 0.295, z_),
                          rgba=SILVER, **VIS)
    # Rear anti-tip mounting arms removed.
    # luggage rack around the battery bag
    xa, xb, yr, zt = -0.15, 0.14, 0.16, 0.585
    T.poly([(xa, yr, zt), (xb, yr, zt)], 16, rgba=BLACK, mirror=True)
    for x_ in (xa, xb, (xa + xb) / 2):
        T.add((x_, -yr, zt), (x_, yr, zt), 16, rgba=BLACK)
    T.add((xa, yr, zt), (xa, yr, 0.36), 16, rgba=BLACK, mirror=True)
    T.add((xb, yr, zt), (xb, 0.2, 0.36), 16, rgba=BLACK, mirror=True)
    # under-body guard (only part that may touch an obstacle)
    ET.SubElement(fr, "geom", name="belly_guard", type="box", pos="0.30 0 0.17", size="0.12 0.18 0.012",
                  rgba="0 0 0 0", contype="1", conaffinity="2", group="3")
    frame_parts = [(T.mass(rho), (0.25, 0, 0.27)), (1.5, (0.25, 0, 0.22)),          # tubes, electronics
                   (1.0, (0.3, 0, 0.2))]                                            # brackets, bolts
    frame_inertial = ET.SubElement(fr, "inertial", mass="1", pos="0 0 0", diaginertia="1 1 1")   # set by finish_frame

    # ---- motors along the chair on right-angle gearboxes ----
    for side, s in (("Left", 1), ("Right", -1)):
        b = ET.SubElement(fr, "body", name=f"Motor150_{side}")
        ET.SubElement(b, "inertial", mass=f"{motor_mass:.4g}", pos=f(0.10, s * 0.175, 0.15),
                      diaginertia="0.0015 0.006 0.006")
        ET.SubElement(b, "geom", type="box", pos=f(0.0, s * 0.175, 0.16), size="0.05 0.035 0.05", rgba=SILVER, **VIS)
        ET.SubElement(b, "geom", type="cylinder", size="0.042", fromto=f(0.04, s * 0.175, 0.15, 0.21, s * 0.175, 0.15),
                      rgba=BLACK, **VIS)
        ET.SubElement(b, "geom", type="cylinder", size="0.044", fromto=f(0.11, s * 0.175, 0.15, 0.15, s * 0.175, 0.15),
                      rgba=SILVER, **VIS)

    # ---- battery bag behind the backrest (LiFePO4 25.6 V 20 Ah inside) ----
    b = ET.SubElement(fr, "body", name="Battery_24V_20Ah")
    ET.SubElement(b, "inertial", mass="5.9", pos="-0.005 0 0.42", diaginertia=box_inertia(5.9, 0.12, 0.093, 0.045))
    ET.SubElement(b, "geom", type="box", pos="-0.005 0 0.455", size="0.14 0.15 0.115", rgba="0.06 0.06 0.07 1", **VIS)
    ET.SubElement(b, "geom", type="box", pos="-0.15 0.09 0.42", size="0.004 0.03 0.012", rgba="0.8 0.1 0.1 1", **VIS)

    # ---- seat (rotates about the pivot) ----
    seat_m = 4.5 if al else 6.0
    seat = ET.SubElement(fr, "body", name="seat", pos=f(px, 0, pz))
    ET.SubElement(seat, "joint", name="lift_angle", axis="0 1 0", range="0 1.308996939", damping="5")
    ET.SubElement(seat, "inertial", mass=f"{seat_m}", pos="-0.29 0 -0.06", diaginertia=box_inertia(seat_m, 0.25, 0.25, 0.08))
    S = Tubes(seat)
    S.add((-0.53, 0.235, -0.10), (-0.02, 0.235, -0.10), 25, rgba=BLACK, mirror=True)  # seat side rails
    for x_ in (-0.51, G["SEAT_PIN_LOCAL"][0], -0.05):
        S.add((x_, -0.235, -0.10), (x_, 0.235, -0.10), 20, rgba=BLACK)
    ET.SubElement(seat, "geom", name="seat_cushion", type="box", pos="-0.29 0 -0.055", size="0.215 0.23 0.035",
                  rgba=DARK, **VIS)
    ET.SubElement(seat, "geom", name="seat_roll", type="capsule", size="0.035", fromto="-0.075 -0.23 -0.055 -0.075 0.23 -0.055",
                  rgba=DARK, **VIS)
    ET.SubElement(seat, "geom", name="seat_cover", type="box", pos="-0.29 0 -0.02", size="0.2 0.21 0.002", rgba="0.22 0.23 0.25 1", **VIS)
    for s in (1, -1):                                                                   # lateral hip pads
        ET.SubElement(seat, "geom", type="box", pos=f(-0.34, s * 0.258, 0.06), size="0.12 0.022 0.085",
                      rgba=GREY_PAD, **VIS)
        ET.SubElement(seat, "geom", type="box", pos=f(-0.34, s * 0.258, -0.06), size="0.015 0.012 0.05",
                      rgba=BLACK, **VIS)
    # everything above was drawn for a pivot at the cushion level: lower it to under the raised pivot
    for el in seat:
        for key in ("pos", "fromto"):
            if el.get(key):
                v = [float(x) for x in el.get(key).split()]
                for i in range(2, len(v), 3):
                    v[i] -= KNEE_UP
                el.set(key, f(v))
    for s_ in (1, -1):        # hinge plates: from the seat rail up to the pivot bolt at the knee (like ESLEH)
        ET.SubElement(seat, "geom", type="box", pos=f(-0.045, s_ * 0.247, -0.075), size="0.065 0.005 0.085",
                      rgba=BLACK, **VIS)
        ET.SubElement(seat, "geom", type="cylinder", size="0.03", fromto=f(0, s_ * 0.241, 0, 0, s_ * 0.253, 0),
                      rgba=BLACK, **VIS)
    ET.SubElement(seat, "site", name="seat_actuator_tip", pos=f(G["SEAT_PIN_LOCAL"][0], 0, G["SEAT_PIN_LOCAL"][1]),
                  size="0.005", rgba="1 0.5 0 1", group="4")

    # ---- backrest, armrests, joystick (kept vertical by the parallelogram) ----
    back_m = 5.0 if al else 6.5
    bx, bz = BACK_PIVOT_LOCAL
    back = ET.SubElement(seat, "body", name="back", pos=f(bx, 0, bz))
    ET.SubElement(back, "joint", name="back_compensation", axis="0 1 0", range="-1.308996939 0", damping="0.8")
    ET.SubElement(back, "inertial", mass=f"{back_m}", pos="0.04 0 0.33", diaginertia=box_inertia(back_m, 0.1, 0.25, 0.3))
    B = Tubes(back)
    B.add((0, 0.235, 0), (-0.03, 0.235, 0.58), 22, rgba=BLACK, mirror=True)             # uprights
    B.add((0, 0.235, 0.02), (0, 0.25, -0.25), 22, rgba=BLACK, mirror=True)               # levers down to the lower link
    for s_ in (1, -1):
        ET.SubElement(back, "geom", type="cylinder", size="0.011", fromto=f(0, s_ * 0.235, -0.25, 0, s_ * 0.265, -0.25),
                      rgba=SILVER, **VIS)
    B.add((-0.03, 0.235, 0.58), (-0.13, 0.235, 0.61), 22, rgba=BLACK, mirror=True)       # push handles
    for s in (1, -1):
        ET.SubElement(back, "geom", type="capsule", size="0.016", fromto=f(-0.08, s * 0.235, 0.60, -0.14, s * 0.235, 0.612),
                      rgba="0.05 0.05 0.05 1", **VIS)
    B.add((-0.01, -0.235, 0.08), (-0.01, 0.235, 0.08), 20, rgba=BLACK)
    ET.SubElement(back, "geom", name="back_cushion", type="box", pos="0.008 0 0.335", size="0.022 0.225 0.225",
                  rgba="0.10 0.10 0.11 1", **VIS)
    ET.SubElement(back, "geom", type="box", pos="0.02 0 0.47", size="0.016 0.2 0.08", rgba="0.15 0.15 0.16 1", **VIS)
    ET.SubElement(back, "geom", type="box", pos="-0.035 0 0.24", size="0.02 0.18 0.11", rgba="0.07 0.07 0.08 1", **VIS)
    for s in (1, -1):                                                                     # armrests
        B.add((-0.005, s * 0.272, 0.10), (-0.005, s * 0.272, 0.265), 20, rgba=BLACK)
        B.add((-0.005, s * 0.235, 0.10), (-0.005, s * 0.272, 0.10), 20, rgba=BLACK)
        ET.SubElement(back, "geom", type="box", pos=f(0.16, s * 0.272, 0.28), size="0.17 0.035 0.018",
                      rgba="0.09 0.09 0.1 1", **VIS)
    ET.SubElement(back, "geom", type="box", pos="0.37 -0.272 0.305", size="0.055 0.04 0.025", rgba="0.12 0.12 0.13 1", **VIS)
    ET.SubElement(back, "geom", type="capsule", size="0.006", fromto="0.385 -0.272 0.33 0.385 -0.272 0.375", rgba=BLACK, **VIS)
    ET.SubElement(back, "geom", type="sphere", size="0.014", pos="0.385 -0.272 0.38", rgba=BLACK, **VIS)
    ET.SubElement(back, "site", name="back_lower_pin", pos="0 0 -0.25", size="0.005", group="4")

    # ---- lower link of the parallelogram ----
    link = ET.SubElement(fr, "body", name="lower_link", pos=f(px, 0, pz - 0.25))
    ET.SubElement(link, "joint", name="lower_angle", axis="0 1 0", range="0 1.308996939", damping="1")
    ET.SubElement(link, "inertial", mass="1.75", pos=f(bx / 2, 0, bz / 2), diaginertia="0.04 0.03 0.07")
    for s in (1, -1):
        ET.SubElement(link, "geom", type="capsule", size="0.011", fromto=f(0, s * 0.25, 0, bx, s * 0.25, bz),
                      rgba="0.5 0.53 0.57 1", **VIS)
    ET.SubElement(link, "site", name="lower_back_pin", pos=f(bx, 0, bz), size="0.005", group="4")

    # ---- linear actuator (geometry and visuals are set by the builder) ----
    cyl = ET.SubElement(fr, "body", name="actuator_cylinder", pos=f(pin[0], 0, pin[1]))
    ET.SubElement(cyl, "joint", name="actuator_pivot", axis="0 1 0", range="-1 1", damping="0.5")
    ET.SubElement(cyl, "inertial", mass=f"{act_mass * 0.7:.4g}", pos="0.05 0 0.18", diaginertia="0.05 0.05 0.005")
    rod = ET.SubElement(cyl, "body", name="actuator_rod", pos="0 0 0.475")
    ET.SubElement(rod, "joint", name="actuator_extension", type="slide", axis="0 0 1", range="0 0.3", damping="20")
    ET.SubElement(rod, "inertial", mass=f"{act_mass * 0.3:.4g}", pos="0 0 -0.15", diaginertia="0.01 0.01 0.001")
    ET.SubElement(rod, "site", name="rod_tip", pos="0 0 0", size="0.005", group="4")

    # ---- wheels ----
    def wheel(name, pos, r, hw, mass, hub_r):
        wb = ET.SubElement(fr, "body", name=name, pos=f(pos))
        ET.SubElement(wb, "joint", name=f"{name}_spin", axis="0 1 0", damping="0.015")
        ET.SubElement(wb, "inertial", mass=f"{mass}", pos="0 0 0",
                      diaginertia=f(mass * (3 * r * r + 4 * hw * hw) / 12, mass * r * r / 2, mass * (3 * r * r + 4 * hw * hw) / 12))
        ET.SubElement(wb, "geom", name=f"{name}_contact", type="cylinder", size=f(r, hw),
                      quat="0.7071067812 0.7071067812 0 0", rgba=TYRE, contype="1", conaffinity="2")
        ET.SubElement(wb, "geom", type="cylinder", size=f(hub_r, hw + 0.003), quat="0.7071067812 0.7071067812 0 0",
                      rgba=SILVER, **VIS)
    for side, s in (("left", 1), ("right", -1)):
        wheel(f"drive_{side}", (0, s * WHEEL_TRACK / 2, 0.16), 0.16, 0.0425, 3.0, 0.105)
        wheel(f"front_{side}", (FRONT_X, s * FRONT_Y, 0.16), 0.16, 0.0425, 3.0, 0.105)

    ET.SubElement(root, "tendon")
    eq = ET.SubElement(root, "equality")
    ET.SubElement(eq, "connect", name="four_bar_loop", site1="back_lower_pin", site2="lower_back_pin", solref="0.006 1")
    ET.SubElement(eq, "connect", name="actuator_loop", site1="rod_tip", site2="seat_actuator_tip", solref="0.006 1")
    ET.SubElement(root, "actuator")
    sen = ET.SubElement(root, "sensor")
    ET.SubElement(sen, "jointpos", name="seat_angle", joint="lift_angle")
    ET.SubElement(sen, "jointpos", name="actuator_stroke", joint="actuator_extension")
    fr.set("_parts", repr(frame_parts))
    return ET.ElementTree(root)


def fit_cushion(seat, dz):
    """Move the cushion top by dz (negative = lower) so the thigh axis lies on the pivot axis."""
    cush = seat.find("geom[@name='seat_cushion']")
    v = [float(x) for x in cush.get("pos").split()]
    sz = [float(x) for x in cush.get("size").split()]
    sz[2] = max(0.015, sz[2] + dz / 2)
    v[2] += dz / 2
    cush.set("pos", f(v)); cush.set("size", f(sz))
    for g in seat.findall("geom[@name='seat_cover']") + seat.findall("geom[@name='seat_roll']"):
        if g.get("pos"):
            p = [float(x) for x in g.get("pos").split()]; p[2] += dz; g.set("pos", f(p))
        else:
            p = [float(x) for x in g.get("fromto").split()]; p[2] += dz; p[5] += dz; g.set("fromto", f(p))


def add_legrest(root, frame, sole_z, heel_x, toe_x, knee, ankle, hw, r_sh, frame_mat="steel"):
    """Legrest frame from the front cross tube, shin pad, one-piece footplate, no auxiliary rollers."""
    T = Tubes(frame)
    top = (HEAD_X, 0.205, 0.33)
    bot = (heel_x + 0.03, 0.19, sole_z - 0.025)          # cross tube under the footplate, not under the heels
    T.add(top, bot, 22, rgba=BLACK, mirror=True)
    T.add((bot[0], -0.19, bot[2]), (bot[0], 0.19, bot[2]), 20, rgba=BLACK)
    # footplate
    x0, x1 = heel_x - 0.07, toe_x + 0.03
    ET.SubElement(frame, "geom", name="footplate", type="box", pos=f((x0 + x1) / 2, 0, sole_z - 0.006),
                  size=f((x1 - x0) / 2, 0.19, 0.006), rgba="0.12 0.12 0.13 1", **VIS)
    # Front footplate anti-tip rollers and their mounting struts removed.
    # shin pad in front of the shins (holds the legs when standing)
    k, a = np.asarray(knee, float), np.asarray(ankle, float)
    axis = (a - k) / np.linalg.norm(a - k)
    fwd = np.array([-axis[2], 0, axis[0]])
    if fwd[0] < 0:
        fwd = -fwd
    Ls = float(np.linalg.norm(a - k))
    pad_h = min(0.12, 0.27 * Ls)                     # shorter pad for short shins (clear of the knee)
    c = k + 0.5 * (a - k) + fwd * (r_sh + 0.035)
    ang = math.atan2(axis[0], -axis[2])
    ET.SubElement(frame, "geom", name="shin_pad", type="box", pos=f(c[0], 0, c[2]), size=f(0.03, hw + 0.08, pad_h),
                  euler=f(0, -ang, 0), rgba="0.06 0.06 0.07 1", **VIS)
    for s in (1, -1):
        T.add((c[0] + 0.03, s * (hw + 0.08), c[2]), (top[0] + (c[2] - top[2]) / (bot[2] - top[2]) * (bot[0] - top[0]),
                                                   s * 0.19, c[2]), 14, rgba=BLACK)
    rho = 7.85 if frame_mat == "steel" else 2.70
    return T.mass(rho) + 1.2, ((x0 + x1) / 2, 0, sole_z)      # tubes + plate and pad (rollers removed)


def finish_frame(frame, extra_parts):
    """Write the frame inertial from all collected parts: [(mass, (x, y, z)), ...]."""
    parts = eval(frame.attrib.pop("_parts")) + list(extra_parts)
    M = sum(m for m, _ in parts)
    c = sum(m * np.asarray(p, float) for m, p in parts) / M
    I = np.zeros(3)
    for m, p in parts:
        d = np.asarray(p, float) - c
        I += m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2]) + m * 0.01
    inr = frame.find("inertial")
    inr.set("mass", f"{M:.5g}"); inr.set("pos", f(c)); inr.set("diaginertia", f(I))
    return M
