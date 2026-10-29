#!/usr/bin/env python3
"""Tunnel-clearance planning only; v61 retains all vehicle command authority."""
import copy
import json
import math

import rospy
from purepursuit_mgeo.route_avoidance_gate import RouteAvoidanceGate
from purepursuit_mgeo.plan_transport import trajectory_payload
from nav_msgs.msg import Path
from std_msgs.msg import Bool, String
from purepursuit_mgeo.tunnel_clearance_fsm import (
    FrenetFsmNode, NORMAL, AVOIDANCE, RETURN, RecentThreatMemory,
)


def fresh(now, stamp, timeout):
    return stamp is not None and math.isfinite(stamp) and -0.05 <= now-stamp <= timeout


def mission_owner(now, highway, roundabout, timeout=0.7):
    """Unknown ownership must never start an independent avoidance maneuver."""
    for name, sample in (("highway", highway), ("roundabout", roundabout)):
        if sample is None or not fresh(now, sample["timestamp"], timeout):
            return "mission_status_unavailable"
        if sample["active"]:
            return name
    return None


class V61ObstacleAvoidance(FrenetFsmNode):
    def __init__(self):
        self.ready = False
        self.base = self.base_at = self.base_stop_at = None
        self.base_stop = True
        self.highway = self.roundabout = None
        self.was_suspended = True
        self._pending_output = None
        self._input_health = {}
        super().__init__()
        if self.enable_control:
            raise ValueError("This adapter must run with enable_control=false")
        self.region_gate_enabled = bool(rospy.get_param("~avoidance_region_gate_enabled", False))
        self.dynamic_stop_wait = bool(rospy.get_param("~dynamic_zone_stop_wait", False))
        self.region_gate = None
        self.dynamic_guard_state = None
        self._region_status = {"allowed": False, "planning_enabled": False, "reason": "startup"}
        if self.region_gate_enabled:
            self.region_gate = RouteAvoidanceGate(
                self.global_points, rospy.get_param("~avoidance_regions"),
                expected_fingerprint=rospy.get_param("~avoidance_route_fingerprint", ""),
                max_match_distance_m=float(rospy.get_param("~avoidance_max_route_distance_m", 12.)),
                search_ahead_m=float(rospy.get_param("~avoidance_progress_search_ahead_m", 60.)),
                search_behind_m=float(rospy.get_param("~avoidance_progress_search_behind_m", 20.)),
                gps_blackout_override=bool(rospy.get_param("~avoidance_gps_blackout_override", True)))
            rospy.loginfo("Integrated avoidance regions: %s", json.dumps(self.region_gate.regions))
        self.output_path = rospy.Publisher("/integration/avoidance/base_path", Path, queue_size=1)
        self.output_stop = rospy.Publisher("/integration/avoidance/base_stop", Bool, queue_size=1)
        self.output_active = rospy.Publisher("/integration/avoidance/active", Bool, queue_size=1)
        self.output_trajectory = rospy.Publisher("/integration/avoidance/trajectory", String, queue_size=1)
        self.output_status = rospy.Publisher("/integration/avoidance/status", String, queue_size=1)
        rospy.Subscriber("/integration/base_path", Path, self._base_cb, queue_size=1)
        rospy.Subscriber("/integration/base_stop", Bool, self._base_stop_cb, queue_size=1)
        rospy.Subscriber("/highway_lane_strategy/state", String,
                         lambda m: self._mission_cb(m, "highway"), queue_size=1)
        rospy.Subscriber("/roundabout_merge_gate/status", String,
                         lambda m: self._mission_cb(m, "roundabout"), queue_size=1)
        rospy.Subscriber("/integration/dynamic_guard/state", String, self._dynamic_guard_cb, queue_size=1)
        self.ready = True

    def _base_cb(self, msg):
        now = rospy.get_time()
        if (msg.header.frame_id != self.map_frame or len(msg.poses) < 2
                or not fresh(now, msg.header.stamp.to_sec(), 0.5)
                or any(not all(math.isfinite(v) for v in
                    (p.pose.position.x, p.pose.position.y, p.pose.position.z))
                    for p in msg.poses)):
            return
        with self.lock:
            self.base, self.base_at = msg, now

    def _base_stop_cb(self, msg):
        with self.lock:
            self.base_stop, self.base_stop_at = bool(msg.data), rospy.get_time()

    def _mission_cb(self, msg, name):
        try:
            value = json.loads(msg.data)
            if type(value["active"]) is not bool or not fresh(
                    rospy.get_time(), float(value["timestamp"]), 0.7):
                return
            value["timestamp"] = float(value["timestamp"])
            with self.lock:
                old = getattr(self, name)
                if old is None or value["timestamp"] > old["timestamp"]:
                    setattr(self, name, value)
        except (ValueError, TypeError, KeyError):
            return

    def _dynamic_guard_cb(self, msg):
        try:
            data=json.loads(msg.data)
            if (data.get('dynamic_guard_version')!='DYNAMIC_WAIT5_TUNNEL_V1'
                    or type(data.get('request_static_avoidance')) is not bool
                    or not isinstance(data.get('waiting_ids'),list)
                    or not fresh(rospy.get_time(),float(data['timestamp']),.5)):
                return
            with self.lock:
                old=self.dynamic_guard_state
                if old is None or data['timestamp']>old['timestamp']:
                    self.dynamic_guard_state=data
        except (ValueError,TypeError,KeyError):
            return

    def _reset_maneuver(self):
        self._transition(NORMAL, "v61 mission owns path")
        self.reference_fallback_active = False
        self._last_published_path = None
        self.path_stamp = 0.0
        self.last_plan_at = float('-inf')
        self.threat_memory = RecentThreatMemory(self.threat_memory.hold_seconds)
        self._held_obstacles = []

    def _output(self, path, stop, active, reason):
        # Plan under the input lock, but never serialize the full route there.
        # rospy publishers serialize synchronously even with queue_size=1.
        region=dict(self._region_status)
        diagnostics=getattr(self.planner,"last_diagnostics",{})
        planned_at=getattr(self,"last_plan_at",None)
        region.update(
            planner_last_plan_at=planned_at if planned_at is not None and math.isfinite(planned_at) else None,
            planner_last_attempted=diagnostics.get("attempted"),
            planner_last_valid=diagnostics.get("valid"),
            planner_last_reasons=diagnostics.get("reasons",{}),
            planner_last_early_exit=diagnostics.get("early_exit"),
            planner_state_reason=getattr(self,"state_reason",None))
        self._pending_output = (path, stop, active, reason, dict(self._input_health),
                                self.state, self.tunnel_mode,
                                len(self.planner.last_blocker_diagnostics), region)

    def _flush_output(self, pending):
        if pending is None:
            return
        path, stop, active, reason, health, state, tunnel, blockers, region = pending
        # A kept, recently revalidated candidate may have an old creation stamp.
        # Preserve its geometry/seq but stamp this execution heartbeat afresh.
        if path is not None:
            # ROS poses are immutable here. Only this outgoing header changes.
            # Deep-copying thousands of route poses at 20 Hz starves inputs.
            path = copy.copy(path)
            path.header = copy.copy(path.header)
            path.header.stamp = rospy.Time.now()
            self.output_path.publish(path)
            payload = json.loads(trajectory_payload(path, stop, self.normal_speed, reason))
            payload.update(active=bool(active), state=state, tunnel_mode=bool(tunnel),
                           version="TUNNEL_MOTION_V3", avoidance_gate=region,
                           avoidance_enabled=region["planning_enabled"],
                           dynamic_static_requested=region.get("dynamic_static_requested",False))
            self.output_trajectory.publish(String(data=json.dumps(payload, separators=(",", ":"), allow_nan=False)))
        self.output_stop.publish(Bool(data=bool(stop)))
        self.output_active.publish(Bool(data=bool(active)))
        self.output_status.publish(String(data=json.dumps({
            "timestamp": rospy.get_time(), "active": bool(active),
            "version": "TUNNEL_MOTION_V3",
            "state": state, "source": reason, "stop": bool(stop),
            "tunnel_mode": tunnel, "blocker_count": blockers,
            "avoidance_enabled": region["planning_enabled"], "avoidance_gate": region,
            "inputs": health,
        }, separators=(",", ":"))))

    def _control_callback(self, event):
        if not self.ready:
            return
        with self.lock:
            self._pending_output = None
            self._control_locked(event)
            pending = self._pending_output
            self._pending_output = None
        self._flush_output(pending)

    def _control_locked(self, event):
        now = rospy.get_time()
        odom = self.latest_odom
        base_stamp = None if self.base is None else self.base.header.stamp.to_sec()
        odom_stamp = None if odom is None else odom.header.stamp.to_sec()
        checks = {
            "base_path_missing": self.base is not None,
            "base_path_receive_stale": fresh(now, self.base_at, 0.5),
            "base_path_stamp_stale": fresh(now, base_stamp, 0.5),
            "odometry_missing": odom is not None,
            "odometry_frame": odom is not None and odom.header.frame_id == self.map_frame,
            "odometry_receive_stale": fresh(now, self.odom_received_at, self.odom_stale_timeout),
            "odometry_stamp_stale": fresh(now, odom_stamp, self.odom_stale_timeout),
            "odometry_nonfinite": odom is not None and all(math.isfinite(v) for v in self._vehicle_state(odom)),
            "upstream_stop": not self.base_stop,
            "base_stop_receive_stale": fresh(now, self.base_stop_at, 0.5),
        }
        failed = [key for key, ok in checks.items() if not ok]
        def age(stamp):
            return None if stamp is None or not math.isfinite(stamp) else round(now-stamp, 4)
        self._input_health = dict(
            failed=failed, upstream_stop=self.base_stop,
            base_path_receive_age_s=age(self.base_at), base_path_stamp_age_s=age(base_stamp),
            odometry_receive_age_s=age(self.odom_received_at), odometry_stamp_age_s=age(odom_stamp),
            base_stop_receive_age_s=age(self.base_stop_at),
            base_path_points=0 if self.base is None else len(self.base.poses))
        base_ok = all(checks[k] for k in checks if k.startswith('base_path_'))
        stopped = bool(failed)
        owner = mission_owner(now, self.highway, self.roundabout)
        self._update_tunnel_mode(now)
        pose_ok = all(ok for key, ok in checks.items() if key.startswith("odometry_"))
        if self.region_gate_enabled:
            if pose_ok:
                position = odom.pose.pose.position
                region = self.region_gate.update(position.x, position.y, self.tunnel_mode)
            else:
                region = {"allowed": False, "reason": "odometry_unavailable",
                          "gps_blackout": bool(self.tunnel_mode)}
        else:
            region = {"allowed": True, "reason": "region_gate_disabled"}
        # The gate controls new entry, never cancels an ongoing detour/return.
        # Existing input stops and other mission ownership keep their priority.
        continuing = self.state in (AVOIDANCE, RETURN) and not self.was_suspended
        region.update(configured=self.region_gate_enabled, continuing=continuing,
                      entry_allowed=bool(region["allowed"] and not stopped and owner is None),
                      static_planner_called=False,
                      planning_enabled=bool((region["allowed"] or continuing)
                                            and not stopped and owner is None))
        self._region_status = region
        if stopped or owner is not None:
            if not self.was_suspended:
                self._reset_maneuver()
            self.was_suspended = True
            if stopped:
                rospy.logwarn_throttle(1.0, "V61_INPUT_FIX1 stop inputs: %s", json.dumps(self._input_health))
            self._output(self.base if base_ok else None, stopped, False,
                         "v61_input_stop" if stopped else owner)
            return
        feedback=getattr(self,'dynamic_guard_state',None)
        feedback_ok=bool(feedback is not None and fresh(now,feedback.get('timestamp'),.5))
        request=bool(feedback_ok and feedback.get('request_static_avoidance') is True
                     and feedback.get('waiting_ids'))
        waiting=bool(feedback_ok and feedback.get('waiting_ids'))
        # Keep the new-entry permission for a latched wait even after crossing
        # a region boundary. Input/mission ownership checks above still win.
        if request:
            region.update(allowed=True,entry_allowed=True,planning_enabled=True,
                          dynamic_static_requested=True)
        dynamic_region='dynamic_obstacle' in region.get('regions',[])
        if (getattr(self, "dynamic_stop_wait", False) and not request
                and dynamic_region and not self.tunnel_mode
                and (waiting or not continuing)):
            # This mission waits on the base route; final command authority
            # applies speed/stop limits from the same map-tracker observations.
            self._region_status.update(entry_allowed=False, planning_enabled=False,
                                       dynamic_stop_wait=True)
            if not self.was_suspended:
                self._reset_maneuver()
            self.was_suspended = True
            self._output(self.base if base_ok else None, False, False,
                         "v61_reference" if self.tunnel_mode else "dynamic_zone_reference_wait_mode")
            return
        if not region["allowed"] and not continuing:
            if not self.was_suspended:
                self._reset_maneuver()
            self.was_suspended = True
            self._output(self.base if base_ok else None, False, False,
                         "v61_region_gate_" + region["reason"])
            return
        if self.was_suspended:
            self._reset_maneuver()
            self.was_suspended = False
        self._region_status["static_planner_called"] = True
        super()._control_callback(event)

    def _follow(self, controller, vehicle, desired_speed, blocker_count):
        # Only preview steering/state. The inherited control publisher is disabled.
        ended = super()._follow(controller, vehicle, desired_speed, blocker_count)
        active = (self.state in (AVOIDANCE, RETURN)
                  and self.active_candidate is not None and self.local_controller is not None)
        self._output(self._last_published_path if active else self.base,
                     False, active, "frenet_"+self.state.lower() if active else "v61_reference")
        return ended


if __name__ == "__main__":
    V61ObstacleAvoidance()
    rospy.spin()
