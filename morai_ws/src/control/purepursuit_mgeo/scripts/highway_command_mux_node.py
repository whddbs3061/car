#!/usr/bin/env python3
"""Fail-closed authority for separate nominal, avoidance and highway control.

final_ws_curvature_signal_highway.launch: the team's curvature controller
drives the whole route on ~main_command_topic and the highway controller
follows highway_lane_strategy on ~highway_command_topic.  This node publishes
one of them on ~output_topic, the fusion's nominal command, as
ZIP avoidance and Escape4 keep independent paths and controllers.  A timer
publishes exactly one command; stale required state or commands produce a
full-brake command.
"""
import json
import threading

import rospy
from morai_msgs.msg import CtrlCmd
from std_msgs.msg import String

from purepursuit_mgeo.highway_section import merged_mux_choice


class HighwayCommandMux:
    def __init__(self):
        self.highway_end_s = float(rospy.get_param("~highway_end_s", 1741.721))
        self.timeout_s = float(rospy.get_param("~timeout_s", 0.5))
        self.highway_enabled = bool(rospy.get_param("~highway_enabled", True))
        self.avoidance_enabled = bool(rospy.get_param("~avoidance_enabled", False))
        self.lock = threading.Lock()
        self.highway_state = None
        self.highway_state_at = None
        self.avoidance_state = None
        self.avoidance_state_at = None
        self.main_command = None
        self.main_at = None
        self.highway_command = None
        self.highway_at = None
        self.avoidance_command = None
        self.avoidance_at = None
        self.selected_source = None
        self.output = rospy.Publisher(rospy.get_param("~output_topic", "/control/ctrl_cmd"),
                                      CtrlCmd, queue_size=1)
        if self.highway_enabled:
            rospy.Subscriber(rospy.get_param("~state_topic", "/highway_lane_strategy/state"),
                             String, self._highway_state_cb, queue_size=1)
        if self.avoidance_enabled:
            rospy.Subscriber(rospy.get_param("~avoidance_state_topic",
                                             "/avoidance_path_manager/state"),
                             String, self._avoidance_state_cb, queue_size=1)
        rospy.Subscriber(rospy.get_param("~main_command_topic", "/control/ctrl_cmd_main"),
                         CtrlCmd, self._main_cb, queue_size=1)
        if self.highway_enabled:
            rospy.Subscriber(rospy.get_param("~highway_command_topic",
                                             "/control/ctrl_cmd_highway"),
                             CtrlCmd, self._highway_cb, queue_size=1)
        if self.avoidance_enabled:
            rospy.Subscriber(rospy.get_param("~avoidance_command_topic",
                                             "/control/ctrl_cmd_avoidance"),
                             CtrlCmd, self._avoidance_cb, queue_size=1)
        rate_hz = max(1.0, float(rospy.get_param("~rate_hz", 20.0)))
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self._timer_cb)

    @staticmethod
    def _parse_state(msg):
        try:
            return json.loads(msg.data)
        except ValueError:
            return None

    def _highway_state_cb(self, msg):
        state = self._parse_state(msg)
        if state is None:
            return
        with self.lock:
            self.highway_state, self.highway_state_at = state, rospy.get_time()

    def _avoidance_state_cb(self, msg):
        state = self._parse_state(msg)
        if state is None:
            return
        with self.lock:
            self.avoidance_state, self.avoidance_state_at = state, rospy.get_time()

    @staticmethod
    def _age(now, stamp):
        return float("inf") if stamp is None else now - stamp

    def _decide(self, now):
        return merged_mux_choice(
            self.highway_state, self._age(now, self.highway_state_at),
            self.avoidance_state, self._age(now, self.avoidance_state_at),
            self._age(now, self.main_at), self._age(now, self.highway_at),
            self._age(now, self.avoidance_at),
            self.highway_end_s, self.timeout_s,
            highway_enabled=self.highway_enabled,
            avoidance_enabled=self.avoidance_enabled)

    @staticmethod
    def _stop_command():
        command = CtrlCmd()
        if hasattr(command, "longlCmdType"):
            command.longlCmdType = 1
        if hasattr(command, "accel"):
            command.accel = 0.0
        if hasattr(command, "brake"):
            command.brake = 1.0
        if hasattr(command, "steering"):
            command.steering = 0.0
        if hasattr(command, "velocity"):
            command.velocity = 0.0
        if hasattr(command, "acceleration"):
            command.acceleration = 0.0
        return command

    def _main_cb(self, msg):
        with self.lock:
            self.main_command, self.main_at = msg, rospy.get_time()

    def _highway_cb(self, msg):
        with self.lock:
            self.highway_command, self.highway_at = msg, rospy.get_time()

    def _avoidance_cb(self, msg):
        with self.lock:
            self.avoidance_command, self.avoidance_at = msg, rospy.get_time()

    def _timer_cb(self, _event):
        with self.lock:
            source, reason = self._decide(rospy.get_time())
            command = (self.main_command if source == "main" else
                       self.highway_command if source == "highway" else
                       self.avoidance_command if source == "avoidance" else None)
            if command is None:
                source = "stop"
                command = self._stop_command()
            if source != self.selected_source:
                rospy.logwarn("DRIVE_MUX %s (%s)", source, reason)
                self.selected_source = source
        self.output.publish(command)


if __name__ == "__main__":
    rospy.init_node("highway_command_mux")
    HighwayCommandMux()
    rospy.spin()
