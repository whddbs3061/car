"""Exercise the real adapter gate without a ROS installation or vehicle control."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

from purepursuit_mgeo.path import PathPoint
from purepursuit_mgeo.route_avoidance_gate import RouteAvoidanceGate


def stamp(value):
    return types.SimpleNamespace(to_sec=lambda: value)


class Message:
    def __init__(self, data=None):
        self.data = data


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, value):
        self.messages.append(value)


class FakeParent:
    def _control_callback(self, event):
        self.plan_calls += 1

    def _update_tunnel_mode(self, now):
        self.tunnel_mode = self.test_blackout

    def _vehicle_state(self, odom):
        p = odom.pose.pose.position
        return p.x, p.y, 0., 5.

    def _transition(self, state, reason):
        self.state = state
        self.active_candidate = self.local_controller = None


def load_adapter():
    rospy = types.ModuleType("rospy")
    rospy.get_time = lambda: 100.
    rospy.Time = types.SimpleNamespace(now=lambda: stamp(100.))
    rospy.logwarn_throttle = lambda *args: None
    fsm = types.ModuleType("purepursuit_mgeo.tunnel_clearance_fsm")
    fsm.FrenetFsmNode = FakeParent
    fsm.NORMAL, fsm.AVOIDANCE, fsm.RETURN = "NORMAL", "AVOIDANCE", "RETURN"
    fsm.RecentThreatMemory = lambda hold: types.SimpleNamespace(hold_seconds=hold)
    nav = types.ModuleType("nav_msgs.msg")
    nav.Path = object
    std = types.ModuleType("std_msgs.msg")
    std.Bool = std.String = Message
    modules = {"rospy": rospy, "nav_msgs.msg": nav, "std_msgs.msg": std,
               "purepursuit_mgeo.tunnel_clearance_fsm": fsm}
    script = Path(__file__).resolve().parents[1] / "src/control/purepursuit_mgeo/scripts/v61_obstacle_avoidance_node.py"
    spec = importlib.util.spec_from_file_location("route_gate_adapter_under_test", script)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class AdapterGateTests(unittest.TestCase):
    def setUp(self):
        self.module = load_adapter()
        n = self.module.V61ObstacleAvoidance.__new__(self.module.V61ObstacleAvoidance)
        self.node = n
        n.map_frame = "map"
        n.region_gate_enabled = True
        n.region_gate = RouteAvoidanceGate(
            [PathPoint(float(i), 0, 0) for i in range(101)], [
                dict(name="static", start_xyz=[10,0,0], end_xyz=[20,0,0]),
                dict(name="dynamic", start_xyz=[30,0,0], end_xyz=[40,0,0])])
        n.base = types.SimpleNamespace(header=types.SimpleNamespace(stamp=stamp(100.), frame_id="map"),
                                      poses=[object(), object()])
        n.base_at = n.base_stop_at = n.odom_received_at = 100.
        n.base_stop = False
        n.odom_stale_timeout = .5
        n.highway = n.roundabout = dict(active=False, timestamp=100.)
        n.was_suspended = True
        n.test_blackout = n.tunnel_mode = False
        n.plan_calls = 0
        n.lock = threading.RLock()
        n.threat_memory = types.SimpleNamespace(hold_seconds=1.5)
        n.planner = types.SimpleNamespace(last_blocker_diagnostics=[])
        n.state = "NORMAL"
        n._last_published_path = None
        n.active_candidate = n.local_controller = None
        n._region_status = {}
        self.pose(5)

    def pose(self, x):
        self.node.latest_odom = types.SimpleNamespace(
            header=types.SimpleNamespace(frame_id="map", stamp=stamp(100.)),
            pose=types.SimpleNamespace(pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=x, y=0., z=0.))))

    def tick(self):
        self.node._control_locked(None)

    def test_constructor_initializes_enabled_gate_from_private_parameters(self):
        module=load_adapter()
        def parent_init(node):
            node.enable_control=False
            node.global_points=[PathPoint(float(i),0,0) for i in range(101)]
        params={"~avoidance_region_gate_enabled":True,
                "~avoidance_regions":[dict(name="static",start_xyz=[10,0,0],end_xyz=[20,0,0])]}
        module.rospy.get_param=lambda name,default=None:params.get(name,default)
        module.rospy.Publisher=lambda *a,**k:Publisher()
        module.rospy.Subscriber=lambda *a,**k:None
        module.rospy.loginfo=lambda *a,**k:None
        with patch.object(FakeParent,"__init__",parent_init):
            node=module.V61ObstacleAvoidance()
        self.assertTrue(node.ready)
        self.assertTrue(node.region_gate_enabled)
        self.assertEqual(node.region_gate.regions[0]["start_index"],10)

    def test_constructor_default_keeps_other_launches_ungated(self):
        module=load_adapter()
        def parent_init(node):node.enable_control=False
        module.rospy.get_param=lambda name,default=None:default
        module.rospy.Publisher=lambda *a,**k:Publisher()
        module.rospy.Subscriber=lambda *a,**k:None
        with patch.object(FakeParent,"__init__",parent_init):
            node=module.V61ObstacleAvoidance()
        self.assertTrue(node.ready)
        self.assertFalse(node.region_gate_enabled)

    def test_outside_region_passes_base_without_planning_or_stop(self):
        self.tick()
        self.assertEqual(self.node.plan_calls, 0)
        pending = self.node._pending_output
        self.assertIs(pending[0], self.node.base)
        self.assertFalse(pending[1])
        self.assertFalse(pending[2])
        self.assertFalse(pending[-1]["planning_enabled"])

    def test_static_and_dynamic_regions_call_existing_planner(self):
        for x, region in [(10, "static"), (30, "dynamic")]:
            self.pose(x)
            self.tick()
            self.assertTrue(self.node._region_status["planning_enabled"])
            self.assertEqual(self.node._region_status["regions"], [region])
        self.assertEqual(self.node.plan_calls, 2)

    def start_maneuver(self, state="AVOIDANCE"):
        self.pose(15)
        self.tick()
        self.node.state = state
        self.node.active_candidate = object()
        self.node.local_controller = object()
        self.node._last_published_path = object()

    def test_boundary_keeps_avoidance_and_return(self):
        self.start_maneuver()
        candidate = self.node.active_candidate
        self.pose(21)
        self.tick()
        self.assertIs(self.node.active_candidate, candidate)
        self.assertTrue(self.node._region_status["continuing"])
        self.assertFalse(self.node._region_status["entry_allowed"])
        self.assertEqual(self.node.plan_calls, 2)
        self.node.state = "RETURN"
        self.tick()
        self.assertEqual(self.node.plan_calls, 3)
        self.node.state = "NORMAL"
        self.tick()
        self.assertEqual(self.node.plan_calls, 3)
        self.assertFalse(self.node._pending_output[2])

    def test_blackout_allows_outside_then_recovery_blocks_new_entry(self):
        self.node.test_blackout = True
        self.tick()
        self.assertEqual(self.node.plan_calls, 1)
        self.node.test_blackout = False
        self.tick()
        self.assertEqual(self.node.plan_calls, 1)
        self.assertFalse(self.node._region_status["planning_enabled"])

    def test_gps_recovery_keeps_return_outside_region(self):
        self.start_maneuver("RETURN")
        self.node.test_blackout = True
        self.pose(21)
        self.tick()
        self.node.test_blackout = False
        self.tick()
        self.assertEqual(self.node.plan_calls, 3)
        self.assertTrue(self.node._region_status["continuing"])

    def test_missing_odometry_keeps_existing_stop_behavior(self):
        self.node.test_blackout = True
        self.node.latest_odom = None
        self.tick()
        self.assertEqual(self.node.plan_calls, 0)
        self.assertTrue(self.node._pending_output[1])
        self.assertEqual(self.node._pending_output[3], "v61_input_stop")

    def test_highway_and_roundabout_ownership_keep_priority(self):
        self.pose(15)
        for owner in ("highway", "roundabout"):
            self.node.highway = self.node.roundabout = dict(active=False,timestamp=100.)
            setattr(self.node,owner,dict(active=True,timestamp=100.))
            self.tick()
            self.assertEqual(self.node.plan_calls,0)
            self.assertEqual(self.node._pending_output[3],owner)
            self.assertFalse(self.node._region_status["planning_enabled"])

    def test_missing_mission_status_keeps_original_priority(self):
        self.pose(15)
        self.node.highway = None
        self.tick()
        self.assertEqual(self.node.plan_calls,0)
        self.assertEqual(self.node._pending_output[3],"mission_status_unavailable")

    def test_dynamic_zone_wait_mode_bypasses_new_frenet_detours(self):
        self.node.dynamic_stop_wait=True
        self.node.region_gate.regions[1]["name"]="dynamic_obstacle"
        self.pose(35)
        self.tick()
        self.assertEqual(self.node.plan_calls,0)
        self.assertFalse(self.node._pending_output[2])
        self.assertEqual(self.node._pending_output[3],"dynamic_zone_reference_wait_mode")
        self.assertFalse(self.node._region_status["planning_enabled"])

    def test_static_region_planning_and_ongoing_return_are_preserved(self):
        self.node.dynamic_stop_wait=True
        self.node.region_gate.regions[1]["name"]="dynamic_obstacle"
        self.pose(15);self.tick()
        self.assertEqual(self.node.plan_calls,1)
        self.node.state="RETURN"
        self.pose(35);self.tick()
        self.assertEqual(self.node.plan_calls,2)
        self.assertTrue(self.node._region_status["continuing"])

    def test_region_disabled_matches_original_entry_behavior(self):
        self.node.region_gate_enabled=False
        self.tick()
        self.assertEqual(self.node.plan_calls,1)
        self.assertTrue(self.node._region_status["planning_enabled"])

    def test_real_status_adds_gate_without_changing_existing_keys(self):
        self.tick()
        n=self.node
        for name in ("output_stop","output_active","output_status"):
            setattr(n,name,Publisher())
        pending=list(n._pending_output)
        pending[0]=None
        n._flush_output(tuple(pending))
        status=json.loads(n.output_status.messages[-1].data)
        self.assertFalse(status["avoidance_enabled"])
        self.assertFalse(status["active"])
        self.assertEqual(status["version"],"TUNNEL_MOTION_V3")
        self.assertEqual(status["source"],"v61_region_gate_outside_avoidance_regions")


if __name__=="__main__":unittest.main()
