#!/usr/bin/env python3
"""The overtaking planner in the team's v61 obstacle integration, ROS stubbed.

* integrated_obstacle_v61_overtake.launch differs from integrated_obstacle_v61.launch
  only in the highway node and the two zone-only wrappers;
* outside the zone highway_overtake_node passes the avoidance path through
  exactly like the team's highway node (uploaded_highway_lane_strategy_node);
* the wrappers change the team's limits only while the zone flag is fresh.

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_overtake_v61.py" -v
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_highway_overtake_node import (  # noqa: E402
    Clock, Feeder, Msg, PoseStamped, Stamp, load_node, overtake_params,
)
from purepursuit_mgeo import highway_zone  # noqa: E402

LAUNCH_DIR = REPO / "src/bringup/morai_bringup/launch"
SOURCE = LAUNCH_DIR / "integrated_obstacle_v61.launch"
OUTPUT = LAUNCH_DIR / "integrated_obstacle_v61_overtake.launch"
WRAPPED = {"highway_lane_strategy", "curvature_speed_purepursuit", "curvature_signal_controller"}
NEW_ARGS = {"overtake_config_file", "highway_cruise_kph", "highway_max_speed_kph",
            "highway_lateral_accel_mps2"}


def generator():
    spec = importlib.util.spec_from_file_location(
        "make_overtake_v61_launch", REPO / "docker/final_ws/make_overtake_v61_launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(element):
    return (element.tag, tuple(sorted(element.attrib.items())),
            tuple(canonical(child) for child in element))


def node(root, name):
    return next(n for n in root.iter("node") if n.get("name") == name)


class GeneratedV61LaunchTest(unittest.TestCase):
    def setUp(self):
        self.source = ET.parse(str(SOURCE)).getroot()
        self.output = ET.parse(str(OUTPUT)).getroot()

    def test_written_file_is_current(self):
        self.assertEqual(generator().main(["--check"]), 0, "run make_overtake_v61_launch.py")

    def test_only_the_three_nodes_differ(self):
        def kept(root):
            return [e for e in root.iter() if not (
                e.tag == "arg" and e.get("name") in NEW_ARGS)]
        others_a = [canonical(n) for n in self.source if not (
            n.tag == "node" and n.get("name") in WRAPPED)]
        others_b = [canonical(n) for n in self.output if not (
            (n.tag == "node" and n.get("name") in WRAPPED)
            or (n.tag == "arg" and n.get("name") in NEW_ARGS))]
        self.assertEqual(others_a, others_b)
        self.assertGreater(len(kept(self.output)), 0)

    def test_wrappers_keep_every_team_parameter(self):
        for name, script, added in (
                ("curvature_speed_purepursuit", "curvature_pp_overtake_node.py",
                 ["highway_max_speed_kph", "highway_lateral_accel_mps2"]),
                ("curvature_signal_controller", "maneuver_fusion_overtake_node.py",
                 ["highway_max_speed_kph"])):
            a, b = node(self.source, name), node(self.output, name)
            self.assertEqual((b.get("pkg"), b.get("type")), ("purepursuit_mgeo", script))
            self.assertEqual([canonical(c) for c in a], [canonical(c) for c in b][:len(a)], name)
            self.assertEqual([c.get("name") for c in b][len(a):], added, name)

    def test_highway_node_keeps_the_team_wiring(self):
        a, b = node(self.source, "highway_lane_strategy"), node(self.output, "highway_lane_strategy")
        self.assertEqual(b.get("type"), "highway_overtake_node.py")
        old = {p.get("name"): p.get("value") for p in a.findall("param")}
        new = {p.get("name"): p.get("value") for p in b.findall("param")}
        for name in ("path_file", "odom_topic", "obstacle_topic", "lane_info_topic",
                     "base_path_topic", "base_stop_topic", "cruise_speed_mps"):
            self.assertEqual(new[name], old[name], name)
        self.assertEqual(new["route_gate_required"], "false")
        controller = node(self.output, "curvature_speed_purepursuit")
        topics = {p.get("name"): p.get("value") for p in controller.findall("param")}
        self.assertEqual(topics["active_path_topic"], "/highway_lane_strategy/active_path")
        self.assertEqual(topics["stop_required_topic"], "/highway_lane_strategy/stop_required")


# The team's highway node as integrated_obstacle_v61.launch configures it.
V61_INTERFACE = {
    "~base_path_topic": "/integration/avoidance/base_path",
    "~base_stop_topic": "/integration/avoidance/base_stop",
    "~cruise_speed_mps": 12.0,          # max_speed_kph 43.2
}


def v61_outputs(publishers):
    path = publishers["~active_path"].publish.call_args
    return {
        "points": None if path is None else [(p.pose.position.x, p.pose.position.y)
                                             for p in path[0][0].poses],
        "stop": publishers["~stop_required"].publish.call_args[0][0].data,
        "speed": round(publishers["~target_speed_mps"].publish.call_args[0][0].data, 9),
        "active": publishers["~active"].publish.call_args[0][0].data,
    }


class PassThroughMatchesV61Test(unittest.TestCase):
    """Outside the zone the output equals the team's v61 highway node, tick by tick."""

    def test_same_path_stop_and_speed(self):
        clock = Clock()
        feed = Feeder(clock)
        team_module, team_pubs = load_node(
            "uploaded_highway_lane_strategy_node",
            dict(V61_INTERFACE, **{"~path_file": str(feed_route())}), clock)
        ours_module, ours_pubs = load_node(
            "highway_overtake_node",
            overtake_params(**{"~base_trajectory_topic": "", "~route_gate_required": False,
                               **V61_INTERFACE}), clock)
        team = team_module.HighwayLaneStrategyNode()
        ours = ours_module.HighwayOvertakeNode()
        s, stops, published = 600.0, 0, 0
        for k in range(240):                       # 12 s at 20 Hz, route s 600 -> ~700
            clock.now = 100.0 + 0.05*k
            s += 0.4
            stop = 50 <= k < 80                    # avoidance stop for 1.5 s
            stale = 120 <= k < 150                 # avoidance output missing for 1.5 s
            for n in (team, ours):
                n._odom_cb(feed.odom(s, 8.0))
                if not stale and k % 2 == 0:
                    n._base_path_cb(base_path(feed, s, clock))
                    n._base_stop_cb(Msg(data=stop))
                n._obstacles_cb(Msg(obstacles=[]))
            team._tick(None)
            ours._tick(None)
            a, b = v61_outputs(team_pubs), v61_outputs(ours_pubs)
            self.assertEqual(a, b, "tick %d" % k)
            stops += a["stop"]
            published += a["points"] is not None
        self.assertGreater(stops, 40)              # the stop and stale phases were exercised
        self.assertGreater(published, 200)


class TeamStateReadersTest(unittest.TestCase):
    """/highway_lane_strategy/state as the team's v61 nodes read it.

    v61_obstacle_avoidance_node mission_owner: a state without a fresh ROS
    "timestamp" or a bool "active" is "mission_status_unavailable" and no
    avoidance starts.  maneuver_fusion_node accepts an offset from the global
    path only with a fresh "timestamp", "stop" false and either an active
    LANE_CHANGE/INNER_HOLD/REJOIN (highway_route_matches) or an inactive
    OFF/DONE with a matching avoidance plan (frenet_route_matches).
    """

    def setUp(self):
        from test_highway_overtake_node import OvertakeNodeTest
        self.harness = OvertakeNodeTest("test_zone_resolution_and_regions_handoff")
        self.harness.setUp()

    def published(self):
        return self.harness.state()

    def assert_readable(self, state):
        now = self.harness.clock.now
        self.assertIs(type(state["active"]), bool)
        self.assertIs(type(state["stop"]), bool)
        self.assertTrue(-0.05 <= now - float(state["timestamp"]) <= 0.7)    # avoidance: 0.7 s

    def test_outside_the_zone_avoidance_may_own_the_path(self):
        self.harness.drive(600.0, 20)
        state = self.published()
        self.assert_readable(state)
        self.assertEqual((state["state"], state["active"], state["stop"]), ("OFF", False, False))

    def test_inside_the_zone_the_offset_is_accepted(self):
        h = self.harness
        h.drive(h.zone[0] + 20.0, 60)
        state = self.published()
        self.assert_readable(state)
        self.assertEqual((state["state"], state["active"], state["stop"]), ("LANE_CHANGE", True, False))

    def test_waiting_for_the_next_gap_is_inner_hold(self):
        planner = self.harness.node.planner
        planner.state = planner.CHASE
        self.assertEqual(planner.integration_state(planner.CHASE), "WAIT_GAP")   # first change
        planner.lane_index = 1
        self.assertEqual(planner.integration_state(planner.CHASE), "INNER_HOLD")


def feed_route():
    return REPO / "data" / "routes" / "2026_molit_comp_global_path.txt"


def base_path(feed, s, clock, count=160):
    _, _, _, i = feed.pose_at(s)
    path = Msg()
    path.header.stamp = Stamp(clock.now)
    for px, py in feed.points[max(0, i-6):i+count]:
        pose = PoseStamped()
        pose.pose.position.x, pose.pose.position.y = px, py
        path.poses.append(pose)
    return path


def load_wrapper(script, base_class_name, base):
    """Import a wrapper script with rospy stubbed and the team node replaced by `base`."""
    rospy = Mock()
    params = {}
    rospy.get_param.side_effect = lambda name, default=None: params.get(name, default)
    team = ModuleType("team")
    setattr(team, base_class_name, base)
    team.MPS_TO_KPH = 3.6
    std = ModuleType("std_msgs.msg")
    std.Bool = Msg
    spec = importlib.util.spec_from_file_location("_test_" + script, PACKAGE / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"rospy": rospy, "std_msgs.msg": std, spec.name: module}), \
            patch.object(highway_zone, "load_script", return_value=team):
        spec.loader.exec_module(module)
    return module, params


class ZoneWrapperTest(unittest.TestCase):
    def test_zone_flag_is_fresh_and_true_only(self):
        flag = highway_zone.ZoneFlag(timeout_s=0.5)
        self.assertFalse(flag.active())
        with patch.object(highway_zone.time, "monotonic", return_value=10.0):
            flag.on_msg(Msg(data=True))
        with patch.object(highway_zone.time, "monotonic", return_value=10.4):
            self.assertTrue(flag.active())
        with patch.object(highway_zone.time, "monotonic", return_value=10.6):
            self.assertFalse(flag.active())             # stale
        with patch.object(highway_zone.time, "monotonic", return_value=11.0):
            flag.on_msg(Msg(data=False))
            self.assertFalse(flag.active())

    def test_fusion_cap_only_inside_the_zone(self):
        class Fusion:
            def __init__(self):
                import threading
                self.lock = threading.RLock()
                self.turn_motion = NS(max_speed_kph=43.2)

            def constrain_route_speed(self, decision, speed, released_context_id=None):
                return self.turn_motion.max_speed_kph

        module, params = load_wrapper("maneuver_fusion_overtake_node.py", "ManeuverFusionNode", Fusion)
        params["~highway_max_speed_kph"] = 79.2
        fusion = module.HighwayManeuverFusion()
        self.assertEqual(fusion.constrain_route_speed(None, 10.0), 43.2)      # no zone flag
        fusion.highway_zone.on_msg(Msg(data=True))
        self.assertEqual(fusion.constrain_route_speed(None, 10.0), 79.2)
        self.assertEqual(fusion.turn_motion.max_speed_kph, 43.2)               # restored
        fusion.highway_zone.on_msg(Msg(data=False))
        self.assertEqual(fusion.constrain_route_speed(None, 10.0), 43.2)

    def test_controller_profile_only_inside_the_zone(self):
        class Controller:
            def __init__(self):
                import threading
                self._lock = threading.RLock()
                self.max_speed_kph, self.lateral_accel_limit = 43.2, 0.45

            def _make_speed_profile(self, s_values, curvatures, final_speed_mps):
                return self.max_speed_kph, self.lateral_accel_limit, round(final_speed_mps*3.6, 6)

        module, params = load_wrapper("curvature_pp_overtake_node.py",
                                      "AdaptiveCurvaturePurePursuit", Controller)
        params.update({"~highway_max_speed_kph": 79.2, "~highway_lateral_accel_mps2": 2.0})
        controller = module.HighwayCurvaturePurePursuit()
        self.assertEqual(controller._make_speed_profile([], [], 12.0), (43.2, 0.45, 43.2))
        controller.highway_zone.on_msg(Msg(data=True))
        self.assertEqual(controller._make_speed_profile([], [], 12.0), (79.2, 2.0, 79.2))
        self.assertEqual((controller.max_speed_kph, controller.lateral_accel_limit), (43.2, 0.45))
        self.assertEqual(controller._make_speed_profile([], [], 5.0)[2], 18.0)  # explicit end kept


if __name__ == "__main__":
    unittest.main()
