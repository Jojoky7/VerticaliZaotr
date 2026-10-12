"""Physics and video regressions. Models are built in memory, without rewriting XML.

Run: python -B -X utf8 -m unittest test_simulator -v
"""
import contextlib
import io
import math
import threading
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image

import simulator as s


def make_sim(**kwargs):
    config = dict(mass=80, height=1.75)
    config.update(kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        tree = s.bh.build(**config)
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
    with patch.object(s.Sim, "_write_model"), patch.object(mujoco.MjModel, "from_xml_path", return_value=model):
        return s.Sim(**config)


def advance_for(sim, seconds, dt=1/30, **command):
    states = []
    for _ in range(math.ceil(seconds/dt)):
        sim.command(command)
        states.append(sim.advance(dt))
    return states


class ThresholdTests(unittest.TestCase):
    def assert_crosses(self, level=2, dt=1/30, clearance=.002, settle=1, jy=1, **config):
        sim = make_sim(**config)
        sim.teleport(sim._threshold_start_x(clearance))
        advance_for(sim, settle, dt, jy=0)
        target_speed = min(sim.mp["v_rated"], s.TOP_SPEED_CAP) * s.LEVEL_FRACTIONS[level-1] * jy
        duration = (2.8-sim.d.qpos[0])/target_speed + 5
        for _ in range(math.ceil(duration/dt)):
            sim.command(dict(jy=jy, level=level))
            state = sim.advance(dt)
            self.assertLess(abs(state["pitch"]), 20)
            self.assertLess(abs(state["roll"]), 20)
            self.assertTrue(np.isfinite(sim.d.qpos).all())
            self.assertLessEqual(max(sim.torque_limit), sim.mp["wheel_torque"] * 2 + 1e-9)
            if sim.d.qpos[0] > 2.8:
                return
        self.fail(f"Stopped at x={sim.d.qpos[0]:.4f}; level={level}, dt={dt}, jy={jy}, config={config}")

    def test_all_levels_contact_and_approach_at_different_frame_rates(self):
        scenarios = [(.01, 0, .5), (1/30, .002, 1), (.066, .25, 1.3), (.1, .5, .2)]
        for level in range(1, 6):
            for dt, clearance, settle in scenarios:
                with self.subTest(level=level, dt=dt, clearance=clearance):
                    self.assert_crosses(level, dt, clearance, settle)

    def test_crawl_speeds(self):
        for level, jy in [(1,.1), (1,.3), (2,.3), (5,.12)]:
            with self.subTest(level=level, jy=jy):
                self.assert_crosses(level=level, jy=jy)

    def test_person_and_motor_configurations(self):
        for config in [dict(mass=50, height=1.45), dict(mass=120, height=1.9),
                       dict(mass=150, height=2, motor="N2002"), dict(motor="N2002"),
                       dict(motor="custom", motor_params=dict(T=2,n_rated=2000,n0=5000,ratio=40,
                                                             eta=.8,I0=3,I=15,mass=3))]:
            with self.subTest(config=config):
                self.assert_crosses(**config)

    def test_insufficient_motor_reaches_its_limit_and_reports_it(self):
        sim = make_sim(mass=150, height=2)
        sim.teleport(sim._threshold_start_x())
        advance_for(sim, 1, jy=0)
        states = advance_for(sim, 10, jy=1, level=2)
        self.assertLess(sim.d.qpos[0], 2.5)
        self.assertTrue(any("Моторы на пределе" in warning for state in states for warning in state["warnings"]))
        self.assertLess(min(sim.peak_budget), .01)

    def test_flat_starts_do_not_trigger_obstacle_mode(self):
        for level in (1, 5):
            sim = make_sim()
            advance_for(sim, 1, jy=0)
            states = advance_for(sim, 1.5, jy=1, level=level)
            self.assertFalse(any(state["obstacle_mode"] for state in states))
            self.assertGreater(sim.d.qpos[0], .2)

    def test_telemetry_cannot_change_drive_control(self):
        normal, stale = make_sim(), make_sim()
        for sim in (normal, stale):
            sim.teleport(sim._threshold_start_x())
            advance_for(sim, 1, jy=0)
        for _ in range(160):           # ~5.3 s at level 2 (2.5 km/h top speed)
            stale.state["speed"] = 1000.0
            for sim in (normal, stale):
                sim.command(dict(jy=1, level=2))
                sim.advance(1/30)
        np.testing.assert_allclose(normal.d.qpos, stale.d.qpos, atol=1e-12, rtol=0)
        self.assertGreater(normal.d.qpos[0], 2.8)

    def test_stop_turn_reverse_and_reset_release_assistance(self):
        for command in [dict(jy=0), dict(jy=1, brake=True), dict(jy=1, park=True),
                        dict(jy=1, estop=True), dict(jy=1, jx=1), dict(jy=-1)]:
            with self.subTest(command=command):
                sim = make_sim()
                sim.obstacle_mode = True
                sim.stall_time = 1
                sim.v_cmd = .1
                sim.command(command)
                sim.advance(.01)
                self.assertFalse(sim.obstacle_mode)
                if command.get("park") or command.get("estop"):
                    self.assertTrue(sim.locked)
        sim.obstacle_mode, sim.stall_time, sim.recovered_time = True, 1, .3
        sim.teleport(0)
        self.assertFalse(sim.obstacle_mode)
        self.assertEqual(sim.stall_time, 0)
        sim.peak_budget = [0,0]
        sim.reset()
        self.assertEqual(sim.peak_budget, [s.PEAK_TIME]*2)

    def test_peak_limits_still_apply(self):
        sim = make_sim()
        sim.obstacle_mode = True
        sim.peak_budget = [0,0]
        sim.v_cmd = .1
        sim.command(dict(jy=1))
        sim.apply_controls(sim.m.opt.timestep)
        self.assertEqual(sim.torque_limit, [sim.mp["wheel_torque"]]*2)
        for acts in sim.drive_acts:
            total = sum(sim.m.actuator_forcerange[a,1]*s.bh.WHEEL_R for a in acts)
            self.assertAlmostEqual(total, sim.mp["wheel_torque"])


class SpeedLimitTests(unittest.TestCase):
    """The gearbox holds the 2.5 km/h limit downhill; the slope cannot overrun the wheels."""

    def test_top_speed_is_project_limit(self):
        self.assertAlmostEqual(s.TOP_SPEED_CAP * 3.6, 2.5)
        sim = make_sim()
        sim.teleport(0, y=6)
        advance_for(sim, 1, jy=0)
        top = max(abs(st["speed"]) for st in advance_for(sim, 6, jy=1, level=5))
        self.assertLess(top * 3.6, 2.55)
        self.assertGreater(top * 3.6, 2.3)

    def test_downhill_does_not_overrun_gearbox(self):
        for motor, mass in [("N150", 80), ("N150", 150), ("N200", 80)]:
            for slope, jy in [(10, 1), (15, 1), (15, -1)]:
                with self.subTest(motor=motor, mass=mass, slope=slope, jy=jy):
                    sim = make_sim(motor=motor, mass=mass)
                    sim.teleport(0, y=6)
                    advance_for(sim, 1, jy=0, slope_fwd=slope * jy)
                    states = advance_for(sim, 8, jy=jy, level=5, slope_fwd=slope * jy)
                    top = max(abs(st["speed"]) for st in states[30:])
                    self.assertLess(top * 3.6, 2.7)
                    self.assertFalse(any("Моторы на пределе" in w for st in states for w in st["warnings"]))
                    advance_for(sim, 3, jy=0, slope_fwd=slope * jy)
                    self.assertTrue(sim.locked)

    def test_long_15_degree_descent(self):
        sim = make_sim(mass=150)
        sim.teleport(54.6, yaw=math.pi, z=20 * math.sin(math.radians(15)))
        advance_for(sim, 1, jy=0)
        speeds = []
        for _ in range(40 * 30):
            sim.command(dict(jy=1, level=5))
            state = sim.advance(1 / 30)
            if 36 < sim.d.qpos[0] < 51:
                speeds.append(abs(state["speed"]))
            if sim.d.qpos[0] < 34:
                break
        self.assertGreater(len(speeds), 100)
        self.assertLess(sum(speeds) / len(speeds) * 3.6, 2.6)
        self.assertLess(max(speeds) * 3.6, 3.0)


class MotorTests(unittest.TestCase):
    """Different motors behave differently on a hill; a weak one stalls instead of rolling back."""

    def climb(self, motor, slope, seconds=8):
        sim = make_sim(motor=motor)
        sim.teleport(0, y=6)
        advance_for(sim, 1, jy=0, slope_fwd=-slope)
        x0 = sim.d.qpos[0]
        states = advance_for(sim, seconds, jy=1, level=5, slope_fwd=-slope)
        return sim, states, sim.d.qpos[0] - x0

    def test_uphill_speed_held_until_motor_limit(self):
        _, states, _ = self.climb("N150", 6)
        self.assertGreater(states[-1]["speed"] * 3.6, 2.3)

    def test_weak_motor_stalls_and_does_not_roll_back(self):
        for motor in ("B57_50", "B57_69"):
            with self.subTest(motor=motor):
                sim, states, dx = self.climb(motor, 10)
                self.assertGreater(dx, -0.2)
                self.assertLess(max(abs(st["speed"]) for st in states) * 3.6, 2.6)
                self.assertTrue(any("не тянет" in w for st in states for w in st["warnings"]))

    def test_weaker_catalogue_motors(self):
        torques = []
        for key in ("N150", "B57_138", "B57_103", "B57_69", "B57_50"):
            mp = s.bh.motor_profile(key)
            torques.append(mp["wheel_torque"])
            if key.startswith("B57"):
                self.assertEqual(mp["ratio"], 30.0)
                self.assertTrue(mp["source"].startswith("https://"))
        self.assertEqual(torques, sorted(torques, reverse=True))
        _, _, slow = self.climb("B57_103", 6, seconds=12)
        _, _, fast = self.climb("N150", 6, seconds=12)
        self.assertLess(slow, fast - 1.0)


class VideoTests(unittest.TestCase):
    def test_resize_is_queued_then_applied_to_encoded_frames(self):
        sim = make_sim()
        sim.advance(.02)
        video = s.Video(sim, 800, 500)
        try:
            worker = threading.Thread(target=video.control, args=({"size":[1920,1200]},))
            worker.start()
            worker.join()
            self.assertEqual((video.renderer.width, video.renderer.height), (800,500))
            for size in [(1920,1200), (800,500)]:
                video.control(dict(size=size))
                video.render()
                with Image.open(io.BytesIO(video.frame)) as img:
                    self.assertEqual(img.size, size)
                self.assertEqual((sim.m.vis.global_.offwidth, sim.m.vis.global_.offheight), size)
        finally:
            video.renderer.close()

    def test_size_validation_and_limit(self):
        video = s.Video.__new__(s.Video)
        video.requested_size = (800,500)
        for size in ([0,500], [float("nan"),500], [float("inf"),500], [100], [100,100,100]):
            with self.subTest(size=size), self.assertRaises((ValueError,TypeError)):
                video.control(dict(size=size))
            self.assertEqual(video.requested_size, (800,500))
        self.assertEqual(s.Video.fit_size(7680,4800), s.Video.MAX_SIZE)
        self.assertEqual(s.Video.fit_size(801,501), (802,502))


if __name__ == "__main__":
    unittest.main()
