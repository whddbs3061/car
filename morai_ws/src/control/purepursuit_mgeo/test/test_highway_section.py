#!/usr/bin/env python3
"""The escape4 highway section in final_ws_curvature_signal.launch, without ROS.

Run from morai_ws (tools/validate_new_final.py runs this folder):
    PYTHONPATH=src/control/purepursuit_mgeo/src python -m unittest discover \
        -s src/control/purepursuit_mgeo/test -p test_highway_section.py -v
"""
from pathlib import Path
import re
import sys
import unittest
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
WS = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

from purepursuit_mgeo import highway_overtake_escape4 as escape4  # noqa: E402
from purepursuit_mgeo.highway_section import (  # noqa: E402
    merged_mux_choice, mux_choice, route_window,
)

LAUNCH = WS / "src/bringup/morai_bringup/launch/final_ws_curvature_signal.launch"
HIGHWAY_CONTROLLER = (WS / "src/experimental/curvature_speed_purepursuit/scripts"
                      / "adaptive_curvature_purepursuit_node.py")


class MuxChoiceTest(unittest.TestCase):
    END = 1741.721

    def choice(self, phase, route_s=1200.0, state_age=0.1, command_age=0.05):
        state = None if phase is None else {"phase": phase, "route_s": route_s}
        return mux_choice(state, state_age, command_age, self.END, 0.5)

    def test_main_controller_outside_the_highway(self):
        self.assertEqual(self.choice("OFF", 1000.0), (False, "outside_highway"))
        self.assertEqual(self.choice(None), (False, "highway_state_stale"))

    def test_highway_controller_from_the_zone_entry_to_the_highway_end(self):
        for phase in ("WAIT_LANE", "CHASE", "CHANGE", "REACQUIRE", "FOLLOW"):
            self.assertEqual(self.choice(phase), (True, phase))
        self.assertEqual(self.choice("DONE", 1700.0), (True, "DONE"))
        self.assertEqual(self.choice("DONE", self.END), (False, "past_highway_end"))

    def test_stale_highway_inputs_fall_back_to_the_main_controller(self):
        self.assertEqual(self.choice("CHASE", state_age=0.8), (False, "highway_state_stale"))
        self.assertEqual(self.choice("CHASE", command_age=0.8), (False, "highway_command_stale"))

    def merged(self, phase="OFF", avoidance="NORMAL", highway_age=0.1,
               avoidance_age=0.1, main_age=0.1, highway_command_age=0.1,
               avoidance_command_age=0.1,
               highway_enabled=True, avoidance_enabled=True):
        return merged_mux_choice(
            {"phase": phase, "route_s": 1200.0}, highway_age,
            {"state": avoidance}, avoidance_age, main_age, highway_command_age,
            avoidance_command_age,
            self.END, 0.5, highway_enabled, avoidance_enabled)

    def test_merged_mux_keeps_nominal_authority_outside_missions(self):
        self.assertEqual(self.merged(), ("main", "nominal"))

    def test_merged_mux_keeps_highway_and_avoidance_authority_separate(self):
        self.assertEqual(self.merged(phase="CHASE"),
                         ("highway", "highway_CHASE"))
        self.assertEqual(self.merged(avoidance="AVOIDING_BYPASS"),
                         ("avoidance", "avoidance_AVOIDING_BYPASS"))
        self.assertEqual(self.merged(phase="CHASE", avoidance="AVOIDING_BYPASS"),
                         ("highway", "highway_CHASE"))

    def test_merged_mux_fails_closed_on_required_stale_inputs(self):
        self.assertEqual(self.merged(highway_age=0.8),
                         ("stop", "highway_state_stale"))
        self.assertEqual(self.merged(avoidance_age=0.8),
                         ("stop", "avoidance_state_stale"))
        self.assertEqual(self.merged(phase="CHASE", highway_command_age=0.8),
                         ("stop", "highway_command_stale"))
        self.assertEqual(self.merged(avoidance="AVOIDING_BYPASS",
                                     avoidance_command_age=0.8),
                         ("stop", "avoidance_command_stale"))
        self.assertEqual(self.merged(main_age=0.8),
                         ("stop", "main_command_stale"))


class RouteWindowTest(unittest.TestCase):
    def test_window_covers_back_and_ahead(self):
        points = [(float(x), 0.0) for x in range(101)]
        s_values = [p[0] for p in points]
        window = route_window(points, s_values, 50.0, 10.0, 30.0)
        self.assertLessEqual(window[0][0], 40.0)
        self.assertGreaterEqual(window[-1][0], 80.0)
        self.assertLessEqual(len(window), 44)
        self.assertEqual(route_window(points, s_values, 95.0, 10.0, 30.0)[-1], (100.0, 0.0))


class LaunchWiringTest(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(str(LAUNCH)).getroot()
        self.nodes = {n.get("name"): n for n in self.root.iter("node")}

    def param(self, node, name):
        return node.find("param[@name='%s']" % name).get("value")

    def test_highway_and_avoidance_nodes_use_explicit_feature_gates(self):
        self.assertEqual(self.root.find("arg[@name='enable_highway']").get("default"), "true")
        self.assertEqual(self.root.find("arg[@name='enable_obstacle_avoidance']").get("default"),
                         "true")
        for name in ("highway_base_path", "highway_lane_strategy",
                     "highway_curvature_purepursuit"):
            self.assertEqual(self.nodes[name].get("if"), "$(arg enable_highway)", name)
        self.assertEqual(self.nodes["highway_command_mux"].get("if"),
                         "$(eval arg('enable_highway') or arg('enable_obstacle_avoidance'))")
        for name in ("bypass_lane_guard", "avoidance_path_manager"):
            self.assertEqual(self.nodes[name].get("if"), "$(arg enable_obstacle_avoidance)")
        self.assertEqual(self.nodes["avoidance_curvature_purepursuit"].get("if"),
                         "$(arg enable_obstacle_avoidance)")

    def test_one_command_reaches_the_fusion(self):
        main = self.nodes["curvature_speed_purepursuit"]
        highway = self.nodes["highway_curvature_purepursuit"]
        avoidance = self.nodes["avoidance_curvature_purepursuit"]
        mux = self.nodes["highway_command_mux"]
        fusion = self.nodes["curvature_signal_controller"]
        self.assertEqual(self.param(main, "command_topic"),
                         "$(eval '/control/ctrl_cmd_main' if (arg('enable_highway') or arg('enable_obstacle_avoidance')) else '/control/ctrl_cmd')")
        self.assertEqual(self.param(mux, "main_command_topic"), "/control/ctrl_cmd_main")
        self.assertEqual(self.param(highway, "command_topic"), "/control/ctrl_cmd_highway")
        self.assertEqual(self.param(mux, "highway_command_topic"), "/control/ctrl_cmd_highway")
        self.assertEqual(self.param(avoidance, "command_topic"), "/control/ctrl_cmd_avoidance")
        self.assertEqual(self.param(mux, "avoidance_command_topic"),
                         "/control/ctrl_cmd_avoidance")
        self.assertEqual(self.param(mux, "output_topic"), "/control/ctrl_cmd")
        self.assertEqual(self.param(fusion, "nominal_command_topic"), "/control/ctrl_cmd")
        self.assertIn("maneuver_fusion_overtake_node.py", fusion.get("type"))
        self.assertIn("maneuver_fusion_node.py", fusion.get("type"))

    def test_highway_controller_follows_highway_lane_strategy(self):
        highway = self.nodes["highway_curvature_purepursuit"]
        self.assertEqual(self.param(highway, "active_path_topic"),
                         "/highway_lane_strategy/active_path")
        self.assertEqual(self.param(highway, "stop_required_topic"),
                         "/highway_lane_strategy/stop_required")
        self.assertEqual(self.param(highway, "trajectory_topic"),
                         "/highway_lane_strategy/trajectory")
        self.assertEqual(self.param(highway, "target_speed_override_topic"),
                         "/highway_lane_strategy/target_speed_mps")
        self.assertEqual(self.param(highway, "require_route_mission_ready"), "false")
        # Its debug topics keep off the main controller's (the fusion reads
        # /experimental/curvature_reference_path).
        remapped = {r.get("from"): r.get("to") for r in highway.findall("remap")}
        published = set(re.findall(r'Publisher\(\s*"(/[^"]+)"',
                                   HIGHWAY_CONTROLLER.read_text(encoding="utf-8")))
        self.assertIn("/experimental/curvature_reference_path", published)
        self.assertEqual(published - set(remapped), set())
        self.assertTrue(all(to.startswith("/highway_control/") for to in remapped.values()))

    def test_avoidance_controller_is_independent_from_escape4(self):
        avoidance = self.nodes["avoidance_curvature_purepursuit"]
        self.assertEqual(avoidance.get("pkg"), "curvature_speed_purepursuit")
        self.assertEqual(self.param(avoidance, "active_path_topic"),
                         "/avoidance_path_manager/active_path")
        self.assertEqual(self.param(avoidance, "trajectory_topic"),
                         "/avoidance_path_manager/trajectory")
        self.assertEqual(self.param(avoidance, "use_target_speed_override"), "false")
        self.assertIsNone(avoidance.find("param[@name='highway_max_speed_kph']"))
        remapped = {r.get("from"): r.get("to") for r in avoidance.findall("remap")}
        published = set(re.findall(r'Publisher\(\s*"(/[^"]+)"',
                                   HIGHWAY_CONTROLLER.read_text(encoding="utf-8")))
        self.assertEqual(published - set(remapped), set())
        self.assertTrue(all(to.startswith("/avoidance_control/")
                            for to in remapped.values()))

    def test_base_path_and_highway_end(self):
        planner = self.nodes["highway_lane_strategy"]
        base = self.nodes["highway_base_path"]
        mux = self.nodes["highway_command_mux"]
        self.assertEqual(self.param(planner, "base_path_topic"),
                         self.param(base, "base_path_topic"))
        self.assertEqual(self.param(planner, "base_stop_topic"),
                         self.param(base, "base_stop_topic"))
        self.assertEqual(self.param(planner, "base_trajectory_topic"), "")
        self.assertEqual(self.param(planner, "highway_end_s"), "$(arg highway_end_s)")
        self.assertEqual(self.param(mux, "highway_end_s"), "$(arg highway_end_s)")
        self.assertEqual(self.param(mux, "avoidance_state_topic"),
                         "/avoidance_path_manager/state")
        self.assertEqual(self.root.find("arg[@name='highway_end_s']").get("default"),
                         str(escape4.OvertakeConfig().highway_end_s))

    def test_lidar_also_runs_for_the_highway(self):
        lidar = next(i for i in self.root.findall("include")
                     if "lidar_algorithm_pipeline.launch" in i.get("file", ""))
        self.assertEqual(lidar.get("if"),
                         "$(eval arg('enable_roundabout_mission') or arg('enable_highway') or arg('enable_obstacle_avoidance'))")
        self.assertEqual(self.param(self.nodes["highway_lane_strategy"], "obstacle_topic"),
                         lidar.find("arg[@name='map_obstacle_topic']").get("value"))

    def test_highway_configs_exist(self):
        for name in ("overtake_config_file", "highway_config_file"):
            default = self.root.find("arg[@name='%s']" % name).get("default")
            self.assertTrue((PACKAGE / default.replace("$(find purepursuit_mgeo)/", "")).is_file(),
                            default)


class PlannerTest(unittest.TestCase):
    def test_escape4_change_lengths(self):
        cfg = escape4.OvertakeConfig()
        self.assertAlmostEqual(escape4.change_geometry(3.5, 15.0, cfg)[0], 85.0, delta=1.0)
        self.assertAlmostEqual(escape4.change_geometry(3.5, 22.0, cfg)[0], 123.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
