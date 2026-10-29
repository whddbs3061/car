"""Isolated signal profile: real fusion callbacks, ROS transport stubbed."""
import unittest
import xml.etree.ElementTree as ET

import test_maneuver_fusion as fixtures
import test_no_turn_signals as no_lamps


class CurvatureSignalLaunchTest(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(fixtures.SOURCE / "bringup/morai_bringup/launch/final_ws_curvature_signal.launch").getroot()
        # With enable_highway the fusion runs through purepursuit_mgeo's wrapper.
        self.fusion = self.root.find("node[@name='curvature_signal_controller']")

    def param(self, element, name):
        return element.find("param[@name='%s']" % name).get("value")

    def test_single_signal_overlay_and_fixed_path_controller(self):
        types = {n.get("type") for n in self.root.findall("node")}
        self.assertNotIn("stopline_controller.py", types)  # Must not veto a permitted left arrow.
        self.assertNotIn("adaptive_curvature_purepursuit_node.py", types)
        pp = self.root.find("node[@type='curvature_speed_purepursuit_node.py']")
        # The shared mux forwards it when either highway or avoidance is enabled.
        self.assertEqual(self.param(pp, "command_topic"),
                         "$(eval '/control/ctrl_cmd_main' if (arg('enable_highway') or arg('enable_obstacle_avoidance')) else '/control/ctrl_cmd')")
        self.assertEqual(self.param(pp, "stopline_speed_cap_enabled"), "false")
        self.assertEqual(self.param(pp, "stopline_approach_speed_kph"), "$(arg stopline_approach_speed_kph)")
        self.assertEqual(self.param(self.fusion, "nominal_command_topic"), "/control/ctrl_cmd")
        self.assertEqual(self.param(self.fusion, "output_command_topic"), "/ctrl_cmd")
        self.assertEqual(self.param(pp, "path_file"), self.param(self.fusion, "path_file"))
        self.assertEqual(self.param(self.fusion, "require_reference_path_match"), "true")
        self.assertEqual(self.param(self.fusion, "require_route_signal_context"), "false")
        self.assertEqual(self.param(self.fusion, "stopline_requires_detected_signal"), "true")
        self.assertEqual(self.param(self.fusion, "signal_mgeo_path"), "$(arg signal_mgeo_path)")
        self.assertEqual(self.fusion.find("rosparam[@param='maneuvers']").text, "[]")

    def test_monitor_computes_commands_but_disables_udp_by_default(self):
        pp = self.root.find("node[@type='curvature_speed_purepursuit_node.py']")
        self.assertEqual(self.param(pp, "publish_command"), "true")
        self.assertEqual(self.root.find("arg[@name='enable_control']").get("default"), "false")
        base = self.root.find("include")
        self.assertEqual(base.find("arg[@name='enable_control']").get("value"), "$(arg enable_control)")
        self.assertEqual(base.find("arg[@name='enable_purepursuit']").get("value"), "false")

    def test_roundabout_v61_caps_the_main_controller_without_a_second_driver(self):
        pp = self.root.find("node[@type='curvature_speed_purepursuit_node.py']")
        gate = self.root.find("node[@type='roundabout_v61_gate_node.py']")
        self.assertIsNotNone(gate)
        self.assertEqual(gate.get("name"), "roundabout_merge_gate")
        self.assertEqual(self.param(pp, "enable_roundabout_mission"),
                         "$(arg enable_roundabout_mission)")
        self.assertEqual(self.param(pp, "roundabout_status_topic"),
                         "/roundabout_merge_gate/status")
        self.assertFalse(any(node.get("type") == "roundabout_test_driver.py"
                             for node in self.root.findall("node")))
        lidar = next(include for include in self.root.findall("include")
                     if "lidar_algorithm_pipeline.launch" in include.get("file", ""))
        self.assertEqual(lidar.find("arg[@name='map_obstacle_topic']").get("value"),
                         "/perception/lidar/tracked_obstacles_map")

    def test_external_stops_and_camera_stream_watchdog_are_disabled(self):
        for key in ("require_sensor_quality", "require_fresh_safety", "allow_blackout_lane_corridor", "lamp_output_enabled"):
            self.assertEqual(self.param(self.fusion, key), "false")
        self.assertEqual(self.param(self.fusion, "require_fresh_camera_stream"), "false")
        self.assertEqual(self.param(self.fusion, "test_without_turn_signals"), "true")
        self.assertEqual(self.param(self.fusion, "safety_topic"), "/curvature_signal/unused_external_safety")
        camera = next(include for include in self.root.findall("include")
                      if "camera_perception.launch" in include.get("file", ""))
        for key in ("enable_lane", "enable_highway_gate", "enable_pedestrian_crossing", "enable_intersection_detection"):
            self.assertEqual(camera.find("arg[@name='%s']" % key).get("value"), "false")


class CurvatureSignalBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.case = no_lamps.NoTurnSignalsTest()
        self.case.setUp()
        self.node = self.case.node
        self.node.require_quality = self.node.require_safety = False
        self.node.allow_blackout_lane = False
        self.node.manual = []
        sample = self.case.c.sample
        # Simulate this launch: no LiDAR/safety/quality/fallback publishers exist.
        self.case.c.sample = lambda key, *a, **kw: None if key in ("safety", "quality") else sample(key, *a, **kw)

    def test_matching_arrow_drives_without_lidar_quality_or_fallback(self):
        output, status = self.case.frames()
        self.assertTrue(status["permission"])
        self.assertEqual(output.brake, 0.)
        self.assertGreater(output.accel, 0.)
        self.assertEqual(output.steering, fixtures.Command().steering)
        for key in ("safety", "quality", "fallback"):
            self.assertNotIn(key, self.node.samples)

    def test_straight_green_cannot_authorize_a_left_route(self):
        output, status = self.case.frames(state="GREEN")
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)
        self.assertEqual(output.steering, fixtures.Command().steering)

    def test_red_revokes_previously_allowed_entry(self):
        self.case.frames()
        output, status = self.case.case.frame(state="RED")
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)

    def test_uncalibrated_signal_camera_never_releases_entry(self):
        self.node.camera["calibrated"] = False
        output, status = self.case.frames()
        self.assertFalse(status["permission"])
        self.assertEqual(output.brake, 1.)
        self.assertEqual(status["signal_selection_reason"], "signal_camera_uncalibrated")

if __name__ == "__main__":
    unittest.main()
