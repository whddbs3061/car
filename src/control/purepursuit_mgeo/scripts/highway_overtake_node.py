#!/usr/bin/env python3
"""Camera-lane highway overtaking planner as a highway_lane_strategy drop-in.

final_ws_highway_overtake_bringup.launch starts this node with
name="highway_lane_strategy", so every topic, the atomic trajectory and the
state names match what adaptive_curvature_purepursuit, route_mission_gate and
stop_cause_monitor already read.  All decisions live in
purepursuit_mgeo.highway_overtake (also exercised by tools/highway_overtake_sim.py);
this node only moves data in and out.

Inputs
  ~base_trajectory_topic  /avoidance_path_manager/trajectory (plan_transport JSON);
                          when empty: ~base_path_topic + ~base_stop_topic
  ~odom_topic             /localization/odometry
  ~obstacle_topic         /perception/lidar/tracked_obstacles_map (map frame)
  ~lane_info_topic        /perception/camera/lane_info
  ~route_gate_topic       /planning/highway_route_active          zone
  ~highway_request_topic  /planning/highway_lane_change_request   permission
  ~handoff_topic          /planning/highway_handoff_due           reported in ~state

Outputs (~ = /highway_lane_strategy/)
  ~trajectory   path + stop + speed in one message (plan_transport schema 1)
  ~active_path ~stop_required ~target_speed_mps ~active ~fast_change_active
  ~lead_brake_required ~lead_emergency_brake ~zone_active
  ~state        JSON; "state" uses the integration names, "phase" the planner's
  ~markers      RViz: nearby vehicles coloured by their role

While the planner is inactive (OFF, DONE) the output is the existing
highway_lane_strategy pass-through: the base path, its stop flag and
cruise_speed_mps rate-limited by speed_rise_mps2 / speed_fall_mps2.
"""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from dataclasses import fields

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as RosPath
from std_msgs.msg import Bool, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

from lidar_perception.msg import LidarObstacleArray
from purepursuit_mgeo.highway_overtake import (
    HighwayOvertakePlanner, Obstacle, OvertakeConfig, Route, align_with_controller,
)
from purepursuit_mgeo.mission_regions import load_mission_regions
from purepursuit_mgeo.path import load_mgeo_path
from purepursuit_mgeo.plan_transport import read_trajectory, trajectory_payload

# Planner fields set through integration-named launch parameters.  The
# integration's own cruise_speed_mps is the pass-through speed outside the zone.
PARAM_ALIASES = {
    "highway_cruise_speed_mps": "cruise_speed_mps",
    "highway_max_speed_mps": "max_speed_mps",
    "route_gate_required": "gate_required",
    "route_gate_timeout_s": "gate_timeout_s",
}
INACTIVE = (HighwayOvertakePlanner.OFF, HighwayOvertakePlanner.DONE)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))


class HighwayOvertakeNode:
    def __init__(self) -> None:
        rospy.init_node("highway_lane_strategy", anonymous=False)
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.lane_info_frame = rospy.get_param("~lane_info_frame", "base_link")
        path_file = rospy.get_param("~path_file")
        route = [(p.x, p.y) for p in load_mgeo_path(path_file)]

        values = {}
        for item in fields(OvertakeConfig):
            if item.name not in PARAM_ALIASES.values() and rospy.has_param("~" + item.name):
                values[item.name] = rospy.get_param("~" + item.name)
        for param, name in PARAM_ALIASES.items():
            if rospy.has_param("~" + param):
                values[name] = rospy.get_param("~" + param)
        cfg = OvertakeConfig.from_mapping(values)
        resolver = Route(route)
        for key in ("zone_start", "zone_end", "entry_taper_start", "entry_taper_end"):
            xy = rospy.get_param("~%s_xy" % key, None)
            if xy:
                setattr(cfg, key + "_s", resolver.s_of_xy(float(xy[0]), float(xy[1])))
        if cfg.zone_end_s <= cfg.zone_start_s:
            raise ValueError("zone_end_xy must lie after zone_start_xy on the route")
        if bool(rospy.get_param("~align_with_controller", True)):
            for name, (old, new) in align_with_controller(
                    cfg,
                    float(rospy.get_param("~controller_lateral_accel_limit_mps2", 0.0)),
                    float(rospy.get_param("~controller_max_accel_mps2", 0.0)),
                    float(rospy.get_param("~controller_max_decel_mps2", 0.0))).items():
                rospy.logwarn("highway_overtake: %s %.2f -> %.2f (controller limit)", name, old, new)
        self._check_regions(rospy.get_param("~mission_regions_file", ""), path_file,
                            resolver.s[-1], cfg)
        self.planner = HighwayOvertakePlanner(cfg, route)
        self.lock = threading.Lock()

        # Pass-through (identical to highway_lane_strategy outside its zone).
        self.cruise_speed_mps = float(rospy.get_param("~cruise_speed_mps", 6.0))
        self.speed_rise_mps2 = float(rospy.get_param("~speed_rise_mps2", 0.8))
        self.speed_fall_mps2 = float(rospy.get_param("~speed_fall_mps2", 1.8))
        self.base_timeout_s = float(rospy.get_param("~base_timeout_s", 0.8))
        self.handoff_timeout_s = float(rospy.get_param("~handoff_timeout_s", 0.6))
        self.last_output_speed = self.cruise_speed_mps
        self.base_path = None
        self.base_path_at = None
        self.base_stop = True
        self.base_stop_at = None
        self.handoff = None
        self.trajectory_seq = 0
        self.last_tick = None

        self.path_pub = rospy.Publisher("~active_path", RosPath, queue_size=1)
        self.trajectory_pub = rospy.Publisher("~trajectory", String, queue_size=1)
        self.stop_pub = rospy.Publisher("~stop_required", Bool, queue_size=1)
        self.speed_pub = rospy.Publisher("~target_speed_mps", Float64, queue_size=1)
        self.active_pub = rospy.Publisher("~active", Bool, queue_size=1)
        self.fast_pub = rospy.Publisher("~fast_change_active", Bool, queue_size=1)
        self.lead_brake_pub = rospy.Publisher("~lead_brake_required", Bool, queue_size=1)
        self.lead_emergency_pub = rospy.Publisher("~lead_emergency_brake", Bool, queue_size=1)
        self.zone_pub = rospy.Publisher("~zone_active", Bool, queue_size=1)
        self.state_pub = rospy.Publisher("~state", String, queue_size=1)
        self.publish_markers = bool(rospy.get_param("~publish_markers", True))
        self.marker_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1)

        self.base_trajectory_topic = rospy.get_param("~base_trajectory_topic", "")
        if self.base_trajectory_topic:
            rospy.Subscriber(self.base_trajectory_topic, String, self._base_trajectory_cb, queue_size=1)
        else:
            rospy.Subscriber(rospy.get_param("~base_path_topic", "/avoidance_path_manager/active_path"),
                             RosPath, self._base_path_cb, queue_size=1)
            rospy.Subscriber(rospy.get_param("~base_stop_topic", "/avoidance_path_manager/stop_required"),
                             Bool, self._base_stop_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~odom_topic", "/localization/odometry"),
                         Odometry, self._odom_cb, queue_size=20)
        rospy.Subscriber(rospy.get_param("~obstacle_topic", "/perception/lidar/tracked_obstacles_map"),
                         LidarObstacleArray, self._obstacles_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~lane_info_topic", "/perception/camera/lane_info"),
                         String, self._lane_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~route_gate_topic", "/planning/highway_route_active"),
                         Bool, self._route_gate_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~highway_request_topic", "/planning/highway_lane_change_request"),
                         Bool, self._request_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~handoff_topic", "/planning/highway_handoff_due"),
                         Bool, self._handoff_cb, queue_size=1)

        rospy.logwarn(
            "highway_overtake as highway_lane_strategy: zone s=%.1f..%.1f, route gate %s, "
            "%d changes, highway %.1f..%.1f m/s, pass-through %.1f m/s, entry %.1f deg, "
            "lateral %.2f, accel %.2f, decel %.2f",
            cfg.zone_start_s, cfg.zone_end_s, "required" if cfg.gate_required else "ignored",
            cfg.max_changes, cfg.cruise_speed_mps, cfg.max_speed_mps, self.cruise_speed_mps,
            cfg.max_entry_angle_deg, cfg.max_lateral_accel_mps2, cfg.gap_check_accel_mps2,
            cfg.target_decel_mps2)
        rate = float(rospy.get_param("~rate_hz", 20.0))
        self.rate_hz = max(rate, 1.0)
        self.timer = rospy.Timer(rospy.Duration(1.0/self.rate_hz), self._tick)

    @staticmethod
    def _check_regions(config_file, path_file, route_length, cfg) -> None:
        """Report how route_mission_gate's regions line up with the planner zone."""
        if not config_file:
            return
        try:
            highway = load_mission_regions(config_file, path_file, route_length)["highway"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            rospy.logerr("highway_overtake: cannot read %s: %s", config_file, exc)
            return
        if highway["start_s_m"] > cfg.zone_start_s:
            rospy.logwarn("highway_overtake: route gate opens at s=%.1f, after the zone start %.1f",
                          highway["start_s_m"], cfg.zone_start_s)
        if not highway["handoff_enabled"]:
            rospy.logwarn("highway_overtake: %s has no handoff; route_mission_gate stops the "
                          "vehicle at s=%.1f", config_file, highway["end_s_m"])
        elif abs(highway["handoff_s_m"] - cfg.zone_end_s) > 2.0:
            rospy.logwarn("highway_overtake: route handoff s=%.1f differs from the zone end %.1f",
                          highway["handoff_s_m"], cfg.zone_end_s)

    # ----- callbacks (planner inputs carry wall-clock stamps) ----------------
    def _base_trajectory_cb(self, msg: String) -> None:
        try:
            path, stop, _speed, _reason = read_trajectory(
                msg.data, self.map_frame, rospy.get_time(), self.base_timeout_s,
                path_type=RosPath, pose_type=PoseStamped, stamp_type=rospy.Time,
            )
        except (KeyError, TypeError, ValueError) as exc:
            with self.lock:
                self.base_path = self.base_path_at = self.base_stop_at = None
                self.base_stop = True
                self.planner.on_base_path(time.time(), None, True)
            rospy.logwarn_throttle(1.0, "invalid_base_trajectory:%s", exc)
            return
        age = max(0.0, rospy.get_time() - path.header.stamp.to_sec())
        points = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
        with self.lock:
            self.base_path, self.base_path_at = path, path.header.stamp
            self.base_stop, self.base_stop_at = bool(stop), path.header.stamp
            self.planner.on_base_path(time.time() - age, points if len(points) >= 2 else None,
                                      bool(stop))

    def _base_path_cb(self, msg: RosPath) -> None:
        points = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        with self.lock:
            self.base_path, self.base_path_at = msg, rospy.Time.now()
            self.planner.on_base_path(time.time(), points if len(points) >= 2 else None)

    def _base_stop_cb(self, msg: Bool) -> None:
        with self.lock:
            self.base_stop, self.base_stop_at = bool(msg.data), rospy.Time.now()
            self.planner.on_base_path(time.time(), None, bool(msg.data))

    def _odom_cb(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        values = (pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation),
                  math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))
        if not all(math.isfinite(v) for v in values):
            return
        with self.lock:
            self.planner.on_odom(time.time(), *values)

    def _obstacles_cb(self, msg: LidarObstacleArray) -> None:
        # Collision checks run in the map frame; a base_link array or a
        # malformed track is treated as missing LiDAR, like the existing node.
        if getattr(getattr(msg, "header", None), "frame_id", None) != self.map_frame:
            return
        try:
            obstacles = [Obstacle(int(o.id), float(o.center_x_map), float(o.center_y_map),
                                  float(o.velocity_x_map), float(o.velocity_y_map),
                                  float(o.length), float(o.width)) for o in msg.obstacles]
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        if not all(math.isfinite(v) for o in obstacles
                   for v in (o.x, o.y, o.vx, o.vy, o.length, o.width)):
            return
        with self.lock:
            self.planner.on_obstacles(time.time(), obstacles)

    def _lane_cb(self, msg: String) -> None:
        try:
            info = json.loads(msg.data)
        except ValueError as exc:
            rospy.logwarn_throttle(2.0, "highway_overtake: lane_info JSON error: %s", exc)
            return
        frame = info.get("frame_id") if isinstance(info, dict) else None
        if frame is not None and str(frame) != self.lane_info_frame:
            rospy.logwarn_throttle(2.0, "lane_info ignored: frame=%s expected=%s",
                                   frame, self.lane_info_frame)
            return
        with self.lock:
            self.planner.on_lane_info(time.time(), info)

    def _route_gate_cb(self, msg: Bool) -> None:
        with self.lock:
            self.planner.on_route_gate(time.time(), bool(msg.data))

    def _request_cb(self, msg: Bool) -> None:
        with self.lock:
            self.planner.on_lane_change_request(time.time(), bool(msg.data))

    def _handoff_cb(self, msg: Bool) -> None:
        self.handoff = (rospy.Time.now(), bool(msg.data))

    # ----- output -------------------------------------------------------------
    def _fresh(self, stamp, now) -> bool:
        return stamp is not None and (now - stamp).to_sec() <= self.base_timeout_s

    def _limit_speed_rate(self, target: float, dt: float) -> float:
        """highway_lane_strategy._limit_speed_rate for the pass-through."""
        target = clamp(target, 0.0, self.cruise_speed_mps)
        if target > self.last_output_speed:
            out = min(target, self.last_output_speed + self.speed_rise_mps2*dt)
        else:
            out = max(target, self.last_output_speed - self.speed_fall_mps2*dt)
        self.last_output_speed = out
        return out

    def _handoff_permitted(self, now) -> bool:
        return (self.handoff is not None and self.handoff[1]
                and 0.0 <= (now - self.handoff[0]).to_sec() <= self.handoff_timeout_s)

    def _ros_path(self, points, stamp):
        points = [(float(x), float(y)) for x, y in points
                  if math.isfinite(x) and math.isfinite(y)]
        if len(points) < 2:
            return None
        msg = RosPath()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        for i, (x, y) in enumerate(points):
            a = points[max(0, i-1)]
            b = points[min(len(points)-1, i+1)]
            yaw = math.atan2(b[1]-a[1], b[0]-a[0])
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = x, y
            pose.pose.orientation.z = math.sin(0.5*yaw)
            pose.pose.orientation.w = math.cos(0.5*yaw)
            msg.poses.append(pose)
        return msg

    def _markers(self, status, stamp) -> MarkerArray:
        array = MarkerArray()
        clear = Marker()
        clear.action = 3        # DELETEALL
        array.markers.append(clear)
        gap = status.get("gap") or {}
        blocker = None if gap.get("ok", True) else gap.get("blocker")
        lead = (status.get("follow") or {}).get("lead")
        for k, trk in enumerate(status.get("tracks", [])):
            color = ((1.0, 0.0, 0.0) if trk["id"] == blocker else
                     (1.0, 0.55, 0.0) if trk["id"] == lead else
                     (0.0, 0.8, 1.0) if trk["lane"] == 1 else (0.6, 0.6, 0.6))
            box = Marker()
            box.header.frame_id, box.header.stamp = self.map_frame, stamp
            box.ns, box.id, box.type, box.action = "tracks", 2*k, Marker.CUBE, Marker.ADD
            box.pose.position.x, box.pose.position.y, box.pose.position.z = trk["x"], trk["y"], 0.8
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = 4.6, 1.9, 1.6
            box.color.r, box.color.g, box.color.b = color
            box.color.a = 0.7
            text = Marker()
            text.header = box.header
            text.ns, text.id, text.type, text.action = "tracks", 2*k+1, Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = trk["x"], trk["y"], 2.6
            text.pose.orientation.w = 1.0
            text.scale.z = 1.0
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = "%d L%+d %.0fkm/h" % (trk["id"], trk["lane"], trk["v"]*3.6)
            array.markers.extend([box, text])
        return array

    def _tick(self, event) -> None:
        # An exception would end the rospy timer thread silently; publish a
        # stop instead and keep ticking.
        try:
            self._step(event)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr_throttle(1.0, "highway_overtake: step failed, stopping: %r", exc)
            output = RosPath()
            output.header.stamp = rospy.Time.now()
            output.header.frame_id = self.map_frame
            self.trajectory_seq = (self.trajectory_seq + 1) & 0xFFFFFFFF
            output.header.seq = self.trajectory_seq
            self.trajectory_pub.publish(String(data=trajectory_payload(
                output, True, 0.0, "planner_error")))
            self.stop_pub.publish(Bool(data=True))

    def _step(self, _event) -> None:
        now = rospy.Time.now()
        dt = (1.0/self.rate_hz if self.last_tick is None
              else clamp((now - self.last_tick).to_sec(), 0.001, 0.2))
        self.last_tick = now
        with self.lock:
            if self.planner.state in INACTIVE:
                # An activation continues from the speed published so far.
                self.planner.speed_out = self.last_output_speed
            out = self.planner.step(time.time())
            base_path, base_path_at = self.base_path, self.base_path_at
            base_stop, base_stop_at = self.base_stop, self.base_stop_at
        status = dict(out.status)
        if out.active:
            path = self._ros_path(out.path, now) if out.path else None
            stop = bool(out.stop) or path is None
            speed = max(0.0, float(out.target_speed))
            self.last_output_speed = speed
        else:
            base_fresh = base_path is not None and self._fresh(base_path_at, now)
            stop = (not base_fresh) or (not self._fresh(base_stop_at, now)) or base_stop
            stop = stop or status.get("reason") == "odom_stale"
            path = copy.deepcopy(base_path) if base_fresh else None
            if path is not None and len(path.poses) < 2:
                path, stop = None, True
            if path is None:
                # Same label as the team's V9 highway_lane_strategy_node.
                status["reason"] = "trajectory_path_missing_or_short"
            speed = self._limit_speed_rate(0.0 if stop else self.cruise_speed_mps, dt)
        reason = str(status.get("reason", status.get("state", "")))

        output = path if path is not None else RosPath()
        output.header.stamp = now
        output.header.frame_id = self.map_frame
        self.trajectory_seq = (self.trajectory_seq + 1) & 0xFFFFFFFF
        output.header.seq = self.trajectory_seq
        self.trajectory_pub.publish(String(data=trajectory_payload(output, stop, speed, reason)))
        if path is not None:
            self.path_pub.publish(output)
        self.stop_pub.publish(Bool(data=bool(stop)))
        self.speed_pub.publish(Float64(data=float(speed)))
        fast = bool(out.active and out.fast_change)
        lead_brake = bool(out.active and out.lead_brake)
        lead_emergency = bool(out.active and out.lead_emergency)
        self.active_pub.publish(Bool(data=bool(out.active)))
        self.fast_pub.publish(Bool(data=fast))
        self.lead_brake_pub.publish(Bool(data=lead_brake))
        self.lead_emergency_pub.publish(Bool(data=lead_emergency))
        self.zone_pub.publish(Bool(data=bool(out.zone_active)))
        status.update({
            # ROS time, like highway_lane_strategy: the v61 avoidance node and
            # fusion drop a state without a fresh timestamp.
            "timestamp": now.to_sec(),
            "active": bool(out.active), "stop": bool(stop), "reason": reason,
            "target_speed_mps": round(speed, 2), "fast_change_active": fast,
            "lead_brake_required": lead_brake, "lead_emergency_brake": lead_emergency,
            "handoff_permitted": self._handoff_permitted(now),
            "trajectory_seq": self.trajectory_seq,
        })
        self.state_pub.publish(String(data=json.dumps(status, separators=(",", ":"))))
        if self.publish_markers and out.active:
            self.marker_pub.publish(self._markers(status, now))
        for event in out.events:
            rospy.logwarn("HIGHWAY_OVERTAKE %s", event)
        rospy.loginfo_throttle(
            1.0, "HIGHWAY state=%s(%s) reason=%s lanes=%s target=%.1fm/s stop=%s gap=%s lead=%s",
            status.get("state"), status.get("phase"), reason, status.get("lane_changes_done"),
            speed, stop, status.get("gap"), (status.get("follow") or {}).get("lead"))


if __name__ == "__main__":
    try:
        HighwayOvertakeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
