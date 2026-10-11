"""Physics and inspection regressions for independent wheels and a real actuator pair."""
import contextlib
import io
import math
import json
import threading
import unittest
import urllib.request
import xml.etree.ElementTree as ET
from unittest.mock import patch

import mujoco
import numpy as np

import simulator as s
from test_simulator import make_sim, advance_for


class WheelMetricsTests(unittest.TestCase):
    def test_each_wheel_power_and_side_totals(self):
        sim = make_sim()
        advance_for(sim, 1)
        for command in (dict(jy=1), dict(jy=.5, jx=.3), dict(jy=-1)):
            state = advance_for(sim, 1, **command)[-1]
            for name in s.WHEELS:
                dof = sim.tel.dof[name + "_spin"]
                wheel = state["wheels"][name]
                self.assertAlmostEqual(wheel["rpm"], sim.d.qvel[dof] * 60 / (2 * math.pi))
                self.assertAlmostEqual(wheel["power"], sim.d.qfrc_actuator[dof] * sim.d.qvel[dof])
            for side in ("left", "right"):
                for key in ("power", "torque"):
                    total = sum(state["wheels"][name][key] for name in ("drive_" + side, "front_" + side))
                    self.assertAlmostEqual(state[key + "_" + side], total)


class LiftTests(unittest.TestCase):
    def assert_closure(self, sim):
        for suffix in ("", "_right"):
            a = sim.d.site_xpos[sim.m.site("rod_tip" + suffix).id]
            b = sim.d.site_xpos[sim.m.site("seat_actuator_tip" + suffix).id]
            self.assertLess(np.linalg.norm(a-b), .002)

    def test_complete_up_down_with_individual_limits(self):
        for mass, height, dt in ((40, 1.45, .02), (80, 1.75, 1/30), (120, 1.9, .1), (150, 2.0, .05)):
            with self.subTest(mass=mass, height=height):
                sim = make_sim(mass=mass, height=height)
                self.assertEqual(len(sim.lift_acts), 2)
                self.assertNotEqual(sim.lift_states[0], sim.lift_states[1])
                advance_for(sim, 1, dt)
                for direction in (1, -1):
                    states = []
                    for _ in range(math.ceil(15/dt)):
                        sim.command(dict(lift=direction))
                        state = sim.advance(dt)
                        states.append(state)
                        self.assert_closure(sim)
                        self.assertLess(state["lift_skew_mm"], .1)
                        for actuator in state["lifts"]:
                            self.assertLessEqual(actuator["force"], 2000.005)
                            self.assertGreaterEqual(actuator["force"], -1600.005)
                            self.assertLess(abs(actuator["speed"]), .038)
                    if direction == 1:
                        self.assertGreater(states[-1]["seat_angle"], 74.5)
                    else:
                        self.assertLess(abs(states[-1]["seat_angle"]), .15)
                    self.assertTrue(np.isfinite(sim.d.qpos).all())

    def test_lower_stop_repeatable_and_idle_energy_zero(self):
        sim = make_sim()
        initial = advance_for(sim, 2)[-1]
        self.assertEqual(initial["energy_Wh"], 0)
        for _ in range(2):
            advance_for(sim, 12, lift=1)
            advance_for(sim, 12, lift=-1)
            idle = advance_for(sim, 2, lift=0)
            self.assertAlmostEqual(idle[-1]["lift_force"], initial["lift_force"], delta=5)
            self.assertEqual(idle[-1]["battery_A"], 0)
            self.assertEqual(idle[0]["energy_Wh"], idle[-1]["energy_Wh"])
            for a in sim.lift_states:
                self.assertEqual(sim.d.act[a], 0)

    def test_angle_target_uses_real_force_and_speed_and_estop(self):
        sim = make_sim()
        advance_for(sim, 1)
        sim.event(dict(name="seat_fast", angle=75))
        states = advance_for(sim, 4)
        self.assertGreater(states[-1]["seat_angle"], 20)
        self.assertLess(states[-1]["seat_angle"], 50)
        self.assertLessEqual(max(abs(a["force"]) for t in states for a in t["lifts"]), 2000.005)
        angle = states[-1]["seat_angle"]
        stopped = advance_for(sim, 1, estop=True)[-1]
        self.assertFalse(sim.lift_running)
        self.assertIsNone(sim.fast_target)
        self.assertEqual(stopped["battery_A"], 0)
        self.assertAlmostEqual(stopped["seat_angle"], angle, delta=.3)

    def test_all_hardware_profiles_compile_and_change_topology(self):
        sim = make_sim()
        for chair in s.bh.CHAIRS:
            for act in s.bh.ACTUATORS:
                with self.subTest(chair=chair, act=act):
                    with contextlib.redirect_stdout(io.StringIO()):
                        tree = s.bh.build(act=act, chair=chair)
                    model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"), assets=None)
                    with patch.object(sim, "_write_model"), patch.object(mujoco.MjModel, "from_xml_path", return_value=model):
                        sim.set_config(80, 1.75, "steel", act, chair=chair)
                    self.assertEqual(len(sim.lift_acts), s.bh.ACTUATORS[act]["count"])
                    self.assertEqual(sim.m.nq, model.nq)
                    advance_for(sim, .1)
                    self.assertEqual(len(sim.snapshot()["lifts"]), len(sim.lift_acts))


class InspectionTests(unittest.TestCase):
    def test_sizes_and_assembly_masses_are_from_model(self):
        sim = make_sim()
        advance_for(sim, 1)
        wheel = sim.parts.inspect(sim.d, "front_left")
        # The new face markings are 2 mm proud of each hub face. Tyre diameter is
        # still 320 mm; visual tread corners can extend its envelope by <1 mm.
        self.assertAlmostEqual(wheel["dimensions_mm"][1], 95)
        self.assertTrue(all(320 <= wheel["dimensions_mm"][i] < 321 for i in (0, 2)))
        self.assertEqual(wheel["mass_kg"], 3)
        threshold = sim.parts.inspect(sim.d, "track_threshold_30mm")
        np.testing.assert_allclose(threshold["dimensions_mm"], [60, 2400, 30])
        self.assertIsNone(threshold["mass_kg"])
        self.assertIsNone(sim.parts.inspect(sim.d, "track_floor")["dimensions_mm"])
        self.assertAlmostEqual(sim.parts.inspect(sim.d, "chair")["mass_kg"], sim.state["chair_mass"])
        for name in ("actuator", "actuator_right"):
            self.assertAlmostEqual(sim.parts.inspect(sim.d, name)["mass_kg"], 2.6)

    def test_catalogue_design_and_live_wheel_specs(self):
        sim = make_sim()
        advance_for(sim, .8, jy=1)
        motor = sim.parts.inspect(sim.d, "Motor150_Left")
        specs = {v["label"]: v for v in motor["specs"]}
        self.assertEqual(specs["Номинальная мощность"]["value"], "150.0 Вт")
        self.assertEqual(specs["Редуктор в модели"]["value"], "30.1:1")
        self.assertIn("16.75:1", specs["Редуктор в каталоге"]["value"])
        self.assertIn("dcmotorfactory.com", motor["sources"][0]["url"])
        for name in s.WHEELS:
            wheel = sim.parts.inspect(sim.d, name)
            values = {v["label"]: v["value"] for v in wheel["specs"]}
            self.assertEqual(values["Диаметр шины"], "320 мм")
            self.assertEqual(values["Ширина шины"], "85 мм")
            self.assertIn("Сейчас", values)
            self.assertIn("подшипники", wheel["build_note"])
        custom = make_sim(motor="custom", motor_params=dict(T=2, n_rated=900, ratio=20))
        motor = custom.parts.inspect(custom.d, "Motor150_Right")
        self.assertFalse(motor["sources"])
        values = {v["label"]: v["value"] for v in motor["specs"]}
        self.assertEqual(values["Редуктор в модели"], "20:1")
        self.assertIn("188.5", values["Мощность на валу"])

    def test_preview_toggle_background_and_clear_endpoint(self):
        sim = make_sim()
        advance_for(sim, 1)
        video = s.Video(sim, 800, 500)
        server = None
        try:
            video.control(dict(human=False, com=False, preset="side"))
            video.render()
            wheel_point = None
            for x in np.linspace(.1, .9, 24):
                for y in np.linspace(.2, .9, 24):
                    p = video.pick(float(x), float(y), select=False)
                    if p and p["id"] in s.WHEELS:
                        wheel_point = (float(x), float(y));break
                if wheel_point:
                    break
            self.assertIsNotNone(wheel_point)
            self.assertEqual(sim.selected_part, "chair")
            selected = video.pick(*wheel_point)
            self.assertEqual(sim.selected_part, selected["id"])
            self.assertEqual(video.pick(*wheel_point, select=False)["id"], selected["id"])
            self.assertEqual(sim.selected_part, selected["id"])
            self.assertIsNone(video.pick(*wheel_point))
            self.assertIsNone(sim.selected_part)
            sim.selected_part = selected["id"]
            self.assertIsNone(video.pick(.02, .02))
            self.assertIsNone(sim.selected_part)
            server = s.ThreadingHTTPServer(("127.0.0.1", 0), s.make_handler(sim, video))
            worker = threading.Thread(target=server.serve_forever, daemon=True);worker.start()
            def request(data):
                req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/api/inspect", data=json.dumps(data).encode(), headers={"Content-Type":"application/json"})
                with urllib.request.urlopen(req, timeout=3) as response:
                    return json.load(response)
            self.assertEqual(request(dict(id="Motor150_Left"))["id"], "Motor150_Left")
            self.assertIn(request(dict(point=wheel_point, preview=True))["id"], s.WHEELS)
            self.assertEqual(sim.selected_part, "Motor150_Left")
            self.assertIsNone(request(dict(clear=True)))
            self.assertIsNone(sim.snapshot()["selection"])
        finally:
            if server:
                server.shutdown();server.server_close()
            video.renderer.close()

    def test_wheel_markers_rotate_without_changing_contacts(self):
        sim = make_sim()
        for wheel in s.WHEELS:
            marker = sim.m.geom(wheel + "_rotation_1_0").id
            contact = sim.m.geom(wheel + "_contact").id
            self.assertEqual(sim.m.geom_contype[marker], 0)
            self.assertEqual(sim.m.geom_conaffinity[marker], 0)
            self.assertAlmostEqual(sim.m.geom_size[contact, 0], .16)
            body = sim.m.body(wheel).id
            self.assertAlmostEqual(sim.m.body_mass[body], 3)
            before = sim.d.geom_xpos[marker]-sim.d.xpos[body]
            q = sim.m.joint(wheel + "_spin").qposadr[0]
            sim.d.qpos[q] += .6
            mujoco.mj_forward(sim.m, sim.d)
            after = sim.d.geom_xpos[marker]-sim.d.xpos[body]
            self.assertGreater(np.linalg.norm(after-before), .02)

    def test_pick_real_surface_and_model_replacement(self):
        sim = make_sim()
        advance_for(sim, 1)
        video = s.Video(sim, 800, 500)
        try:
            video.control(dict(human=False, com=False, preset="side"))
            video.render()
            found = set()
            for x in np.linspace(.1, .9, 32):
                for y in np.linspace(.2, .9, 32):
                    part = video.pick(float(x), float(y))
                    if part:
                        found.add(part["id"])
            self.assertTrue(found & {"front_left", "front_right", "drive_left", "drive_right"})
            self.assertTrue(found & {"actuator", "actuator_right"})
            self.assertIn("seat", found)
            with self.assertRaises(ValueError):
                video.pick(float("nan"), .5)
            old_renderer = video.renderer
            with contextlib.redirect_stdout(io.StringIO()):
                tree = s.bh.build(act="DTL6000")
            model = mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding="unicode"))
            with patch.object(sim, "_write_model"), patch.object(mujoco.MjModel, "from_xml_path", return_value=model):
                sim.set_config(80, 1.75, "steel", "DTL6000")
            advance_for(sim, .02)
            video.render()
            self.assertIsNot(video.renderer, old_renderer)
            self.assertEqual(video._version, sim.model_version)
        finally:
            video.renderer.close()


if __name__ == "__main__":
    unittest.main()
