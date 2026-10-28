#!/usr/bin/env python3
"""scripts/highway_overtake_node.py against the integration contract, ROS stubbed.

The outside-zone output is compared message by message with the existing
highway_lane_strategy_node.py fed the same inputs.

Run from src/control/purepursuit_mgeo:
    PYTHONPATH=src python -m unittest discover -s test -p "test_highway_overtake_node.py" -v
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import yaml

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

from purepursuit_mgeo.highway_overtake import Route  # noqa: E402
from purepursuit_mgeo.path import load_mgeo_path  # noqa: E402
from purepursuit_mgeo.plan_transport import read_trajectory, trajectory_payload  # noqa: E402

ROUTE_FILE = REPO / "data" / "routes" / "2026_molit_comp_global_path.txt"
CONFIG = PACKAGE / "config" / "highway_overtake.yaml"
REGIONS = PACKAGE / "config" / "route_mission_regions_overtake.json"


class Stamp:
    def __init__(self, value=0.0):
        self.value = float(value)

    @classmethod
    def from_sec(cls, value):
        return cls(value)

    def to_sec(self):
        return self.value

    def __sub__(self, other):
        return Stamp(self.value - other.value)


class Msg:
    """Attribute bag standing in for any ROS message."""
    def __init__(self, data=None, **kwargs):
        self.data = data
        self.header = NS(stamp=Stamp(), frame_id="map", seq=0)
        self.poses = []
        self.obstacles = []
        self.__dict__.update(kwargs)


class PoseStamped(Msg):
    def __init__(self):
        super().__init__(pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                                 orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))


class Marker(Msg):
    CUBE, TEXT_VIEW_FACING, ADD = 1, 9, 0

    def __init__(self):
        super().__init__(ns="", id=0, type=0, action=0, text="",
                         pose=NS(position=NS(x=0.0, y=0.0, z=0.0),
                                 orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0)),
                         scale=NS(x=0.0, y=0.0, z=0.0), color=NS(r=0.0, g=0.0, b=0.0, a=0.0))


class MarkerArray(Msg):
    def __init__(self):
        super().__init__(markers=[])


class Clock:
    def __init__(self):
        self.now = 100.0


def load_node(script, params, clock):
    """Import a node script with rospy and the message packages stubbed."""
    rospy = Mock()
    rospy.get_param.side_effect = lambda name, default=None: params.get(name, default)
    rospy.has_param.side_effect = lambda name: name in params
    rospy.get_time.side_effect = lambda: clock.now
    rospy.Time.now.side_effect = lambda: Stamp(clock.now)
    rospy.Time.from_sec.side_effect = Stamp
    publishers = {}

    def publisher(name, *_args, **_kwargs):
        publishers[name] = Mock()
        return publishers[name]

    rospy.Publisher.side_effect = publisher
    modules = {"rospy": rospy}
    for name, values in {
        "geometry_msgs.msg": {"PoseStamped": PoseStamped},
        "nav_msgs.msg": {"Odometry": Msg, "Path": Msg},
        "std_msgs.msg": {"Bool": Msg, "Float64": Msg, "String": Msg},
        "visualization_msgs.msg": {"Marker": Marker, "MarkerArray": MarkerArray},
        "lidar_perception.msg": {"LidarObstacleArray": Msg},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(values)
        modules[name] = module
    spec = importlib.util.spec_from_file_location("_test_" + script, PACKAGE / "scripts" / (script + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, spec.name: module}):
        spec.loader.exec_module(module)
    module.time = NS(time=lambda: clock.now)          # wall clock of the planner inputs
    return module, publishers


# Interface parameters of both nodes as final_ws_highway_*bringup.launch sets them.
INTERFACE = {
    "~base_trajectory_topic": "/avoidance_path_manager/trajectory",
    "~route_gate_required": True, "~handoff_gate_required": True,
    "~cruise_speed_mps": 8.0,            # MAX_SPEED_KPH 28.8
}


def overtake_params(**extra):
    params = {"~" + k: v for k, v in yaml.safe_load(CONFIG.read_text(encoding="utf-8")).items()}
    params.update(INTERFACE)
    params.update({"~path_file": str(ROUTE_FILE), "~mission_regions_file": str(REGIONS),
                   "~highway_cruise_speed_mps": 15.0, "~highway_max_speed_mps": 22.0})
    params.update(extra)
    return params


class Feeder:
    """Real-route inputs shared by both nodes."""

    def __init__(self, clock):
        self.clock = clock
        self.points = [(p.x, p.y) for p in load_mgeo_path(str(ROUTE_FILE))]
        self.route = Route(self.points)

    def pose_at(self, s):
        i = next(k for k, sk in enumerate(self.route.s) if sk >= s)
        (ax, ay), (bx, by) = self.points[i-1], self.points[i]
        return bx, by, math.atan2(by-ay, bx-ax), i

    def odom(self, s, v):
        x, y, yaw, _ = self.pose_at(s)
        return Msg(pose=NS(pose=NS(position=NS(x=x, y=y, z=0.0),
                                   orientation=NS(x=0.0, y=0.0, z=math.sin(0.5*yaw),
                                                  w=math.cos(0.5*yaw)))),
                   twist=NS(twist=NS(linear=NS(x=v, y=0.0, z=0.0))))

    def base_trajectory(self, s, stop, count=160):
        _, _, _, i = self.pose_at(s)
        path = Msg()
        path.header.stamp = Stamp(self.clock.now)
        path.header.seq = 7
        for px, py in self.points[max(0, i-6):i+count]:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = px, py, 28.5
            path.poses.append(pose)
        return Msg(data=trajectory_payload(path, stop, 2.0, "base"))

    def lane_info(self):
        def lane(c, kind):
            return {"detected": True, "type": kind, "dashed": kind == "white_dashed",
                    "coef": [0.0, 0.0, c], "x_range_m": [4.0, 40.0], "age": 5,
                    "from_guide": False, "coasted": False}
        return Msg(data=json.dumps({"timestamp": self.clock.now, "frame_id": "base_link",
                                    "output_status": "FRESH", "lane_valid": True,
                                    "left_lane": lane(1.75, "white_dashed"),
                                    "right_lane": lane(-1.75, "white_solid"),
                                    "straddling_lane": None}))


def outputs(publishers):
    trajectory = json.loads(publishers["~trajectory"].publish.call_args[0][0].data)
    return {
        "stop": trajectory["stop_required"],
        "speed": round(trajectory["target_speed_mps"], 9),
        "points": trajectory["path"]["points"],
        "seq": trajectory["seq"],
        "reason": trajectory["reason"],
        "stop_topic": publishers["~stop_required"].publish.call_args[0][0].data,
        "speed_topic": round(publishers["~target_speed_mps"].publish.call_args[0][0].data, 9),
        "active": publishers["~active"].publish.call_args[0][0].data,
        "fast": publishers["~fast_change_active"].publish.call_args[0][0].data,
    }


class PassThroughMatchesIntegrationTest(unittest.TestCase):
    """Outside the zone the output equals highway_lane_strategy's, message by message."""

    def test_same_trajectory_stop_and_speed(self):
        clock = Clock()
        feed = Feeder(clock)
        legacy_module, legacy_pubs = load_node("highway_lane_strategy_node",
                                               dict(INTERFACE, **{"~path_file": str(ROUTE_FILE)}), clock)
        ours_module, ours_pubs = load_node("highway_overtake_node", overtake_params(), clock)
        legacy = legacy_module.HighwayLaneStrategyNode()
        ours = ours_module.HighwayOvertakeNode()
        s, compared, stops = 600.0, 0, 0
        for k in range(240):                       # 12 s at 20 Hz, route s 600 -> ~700
            clock.now = 100.0 + 0.05*k
            s += 0.4
            stop = 50 <= k < 80                    # avoidance stop for 1.5 s
            stale = 120 <= k < 150                 # no base trajectory for 1.5 s
            short = 180 <= k < 185                 # a one-point (invalid) base path
            for node in (legacy, ours):
                node._odom_cb(feed.odom(s, 8.0))
                if not stale and k % 2 == 0:
                    node._base_trajectory_cb(feed.base_trajectory(s, stop, 1 if short else 160))
                if hasattr(node, "_obstacles_cb"):
                    node._obstacles_cb(Msg(obstacles=[]))
            legacy._tick(None)
            ours._tick(None)
            a, b = outputs(legacy_pubs), outputs(ours_pubs)
            self.assertEqual(a, b, "tick %d" % k)
            compared += 1
            stops += a["stop"]
        self.assertEqual(compared, 240)
        self.assertGreater(stops, 40)              # the stop and stale phases were exercised


class OvertakeNodeTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.feed = Feeder(self.clock)
        self.module, self.pubs = load_node("highway_overtake_node", overtake_params(), self.clock)
        self.node = self.module.HighwayOvertakeNode()
        cfg = self.node.planner.cfg
        self.zone = (cfg.zone_start_s, cfg.zone_end_s)

    def drive(self, s0, ticks, gate=True):
        s = s0
        for k in range(ticks):
            self.clock.now += 0.05
            s += 0.75
            self.node._odom_cb(self.feed.odom(s, 15.0))
            self.node._base_trajectory_cb(self.feed.base_trajectory(s, False))
            self.node._obstacles_cb(Msg(obstacles=[]))
            self.node._lane_cb(self.feed.lane_info())
            if gate is not None:
                self.node._route_gate_cb(Msg(data=gate))
                self.node._request_cb(Msg(data=gate))
            self.node._tick(None)
        return s

    def state(self):
        return json.loads(self.pubs["~state"].publish.call_args[0][0].data)

    def trajectory(self):
        data = self.pubs["~trajectory"].publish.call_args[0][0].data
        return read_trajectory(data, "map", self.clock.now, 1.0,
                               path_type=Msg, pose_type=PoseStamped, stamp_type=Stamp)

    def test_zone_resolution_and_regions_handoff(self):
        self.assertAlmostEqual(self.zone[0], 1160.24, delta=0.05)
        self.assertAlmostEqual(self.zone[1], 1590.25, delta=0.05)
        regions = json.loads(REGIONS.read_text(encoding="utf-8"))["highway"]
        self.assertTrue(regions["handoff_enabled"])
        self.assertAlmostEqual(regions["handoff_s_m"], self.zone[1], delta=0.05)

    def test_lane_change_publishes_integration_state_and_trajectory(self):
        self.drive(self.zone[0] + 5.0, 4)
        state = self.state()
        self.assertEqual((state["state"], state["phase"]), ("LANE_CHANGE", "CHANGE"))
        self.assertTrue(state["active"] and state["fast_change_active"])
        path, stop, speed, reason = self.trajectory()
        self.assertFalse(stop)
        self.assertTrue(reason.startswith("changing"))
        # The highway speed rises from the pass-through cruise, it does not jump.
        self.assertGreater(speed, 8.0)
        self.assertLess(speed, 8.0 + 1.5*0.2 + 1e-6)
        x, y, yaw, _ = self.feed.pose_at(self.zone[0] + 8.0)
        far = path.poses[-1].pose.position
        self.assertGreater(-math.sin(yaw)*(far.x - x) + math.cos(yaw)*(far.y - y), 3.0)
        seqs = [json.loads(c[0][0].data)["seq"] for c in self.pubs["~trajectory"].publish.call_args_list]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))

    def test_no_route_gate_means_pass_through(self):
        self.drive(self.zone[0] + 5.0, 6, gate=None)
        state = self.state()
        self.assertEqual((state["state"], state["active"]), ("OFF", False))
        self.assertEqual(state["reason"], "base_pass")
        self.assertEqual(self.trajectory()[2], 8.0)          # integration cruise speed

    def test_request_off_holds_the_lane(self):
        self.node.planner.on_route_gate(self.clock.now, True)
        s = self.zone[0] + 5.0
        for _ in range(6):
            self.clock.now += 0.05
            s += 0.75
            self.node._odom_cb(self.feed.odom(s, 15.0))
            self.node._base_trajectory_cb(self.feed.base_trajectory(s, False))
            self.node._obstacles_cb(Msg(obstacles=[]))
            self.node._lane_cb(self.feed.lane_info())
            self.node._route_gate_cb(Msg(data=True))
            self.node._request_cb(Msg(data=False))
            self.node._tick(None)
        state = self.state()
        self.assertEqual((state["state"], state["reason"]), ("WAIT_GAP", "lane_change_not_requested"))

    def test_base_link_obstacles_are_ignored(self):
        obstacle = NS(id=1, center_x_map=0.0, center_y_map=0.0, velocity_x_map=0.0,
                      velocity_y_map=0.0, length=4.6, width=1.9, yaw=0.0)
        message = Msg(obstacles=[obstacle])
        message.header.frame_id = "base_link"
        self.node._obstacles_cb(message)
        self.assertIsNone(self.node.planner.obstacles)

    def test_planner_error_publishes_a_stop(self):
        self.node.planner.step = Mock(side_effect=RuntimeError("boom"))
        self.node._tick(None)
        path, stop, speed, reason = self.trajectory()
        self.assertEqual((stop, speed, reason, len(path.poses)), (True, 0.0, "planner_error", 0))

    def test_controller_limits_lower_the_planner(self):
        module, _ = load_node("highway_overtake_node", overtake_params(**{
            "~controller_lateral_accel_limit_mps2": 1.0, "~controller_max_accel_mps2": 1.0,
            "~controller_max_decel_mps2": 1.5}), self.clock)
        cfg = module.HighwayOvertakeNode().planner.cfg
        self.assertAlmostEqual(cfg.max_lateral_accel_mps2, 0.9)
        self.assertEqual((cfg.target_accel_mps2, cfg.target_decel_mps2), (1.0, 1.5))


if __name__ == "__main__":
    unittest.main()
