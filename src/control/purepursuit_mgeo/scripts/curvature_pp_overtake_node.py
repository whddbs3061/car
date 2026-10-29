#!/usr/bin/env python3
"""curvature_speed_purepursuit adaptive_curvature_purepursuit_node.py,
unchanged outside the highway zone.

A path received while the overtaking planner's zone is active (fresh
/highway_lane_strategy/zone_active) gets its speed profile from
highway_max_speed_kph and highway_lateral_accel_mps2 instead of the
course-wide max_speed_kph and lateral limit; the next path outside the zone
uses the team's values again.  The planner keeps its manoeuvres within these
limits.  The team's file is loaded as is, so its updates apply here too.
"""
import rospy
from std_msgs.msg import Bool

from purepursuit_mgeo.highway_zone import ZoneFlag, load_script

module = load_script("curvature_speed_purepursuit", "adaptive_curvature_purepursuit_node.py",
                     "overtake_adaptive_curvature_purepursuit")


class HighwayCurvaturePurePursuit(module.AdaptiveCurvaturePurePursuit):
    def __init__(self):
        self.highway_zone = ZoneFlag()
        self.highway_max_speed_kph = 0.0
        self.highway_lateral_accel_mps2 = 0.0
        super().__init__()
        self.highway_max_speed_kph = float(rospy.get_param("~highway_max_speed_kph", 0.0))
        self.highway_lateral_accel_mps2 = float(rospy.get_param("~highway_lateral_accel_mps2", 0.0))
        self.highway_zone.timeout_s = float(rospy.get_param("~highway_zone_timeout_s", 0.5))
        rospy.Subscriber(rospy.get_param("~highway_zone_topic", "/highway_lane_strategy/zone_active"),
                         Bool, self.highway_zone.on_msg, queue_size=1)

    def _make_speed_profile(self, s_values, curvatures, final_speed_mps):
        if not self.highway_zone.active():
            return super()._make_speed_profile(s_values, curvatures, final_speed_mps)
        with self._lock:
            base = (self.max_speed_kph, self.lateral_accel_limit)
            self.max_speed_kph = max(base[0], self.highway_max_speed_kph)
            self.lateral_accel_limit = max(base[1], self.highway_lateral_accel_mps2)
            # A rolling path ends at the course ceiling; raise it with the cap.
            if abs(final_speed_mps - base[0]/module.MPS_TO_KPH) < 1e-9:
                final_speed_mps = self.max_speed_kph/module.MPS_TO_KPH
            try:
                return super()._make_speed_profile(s_values, curvatures, final_speed_mps)
            finally:
                self.max_speed_kph, self.lateral_accel_limit = base


if __name__ == "__main__":
    try:
        HighwayCurvaturePurePursuit()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
