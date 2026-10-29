#!/usr/bin/env python3
"""turn_signal_controller maneuver_fusion_node.py, unchanged outside the highway zone.

Inside the overtaking planner's zone (fresh /highway_lane_strategy/zone_active)
the speed governor caps the measured speed at highway_max_speed_kph instead of
max_speed_kph: the planner owns the highway speed.  The team's file is loaded
as is, so its updates apply here too.
"""
import rospy
from std_msgs.msg import Bool

from purepursuit_mgeo.highway_zone import ZoneFlag, load_script

module = load_script("turn_signal_controller", "maneuver_fusion_node.py", "overtake_maneuver_fusion")


class HighwayManeuverFusion(module.ManeuverFusionNode):
    def __init__(self):
        self.highway_zone = ZoneFlag()
        self.highway_max_speed_kph = 0.0
        super().__init__()
        self.highway_max_speed_kph = float(rospy.get_param("~highway_max_speed_kph", 0.0))
        self.highway_zone.timeout_s = float(rospy.get_param("~highway_zone_timeout_s", 0.5))
        rospy.Subscriber(rospy.get_param("~highway_zone_topic", "/highway_lane_strategy/zone_active"),
                         Bool, self.highway_zone.on_msg, queue_size=1)

    def constrain_route_speed(self, decision, speed, released_context_id=None):
        if self.highway_max_speed_kph <= 0.0 or not self.highway_zone.active():
            return super().constrain_route_speed(decision, speed, released_context_id)
        with self.lock:
            base = self.turn_motion.max_speed_kph
            self.turn_motion.max_speed_kph = max(base, self.highway_max_speed_kph)
            try:
                return super().constrain_route_speed(decision, speed, released_context_id)
            finally:
                self.turn_motion.max_speed_kph = base


if __name__ == "__main__":
    HighwayManeuverFusion()
    rospy.spin()
