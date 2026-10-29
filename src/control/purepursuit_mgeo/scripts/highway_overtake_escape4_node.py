#!/usr/bin/env python3
"""highway_overtake_node.py with the escape4 planner (highway_overtake_escape4).

The node script is loaded unchanged; only the planner module it uses is the
escape4 variant.  Topics, parameters and the state message are those of
highway_overtake_node.py; config/highway_overtake_escape4.yaml, loaded over
highway_overtake.yaml, sets the highway end speed, the change geometry and the
sideways speed limit.  The pass-through speed also stays below the planner's
cap toward the highway end.
"""
import importlib.util
import os
import sys

import rospy

from purepursuit_mgeo import highway_overtake_escape4 as planner

_spec = importlib.util.spec_from_file_location(
    "highway_overtake_node_base",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "highway_overtake_node.py"))
node = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = node
_spec.loader.exec_module(node)
for _name in ("HighwayOvertakePlanner", "Obstacle", "OvertakeConfig", "Route",
              "align_with_controller"):
    setattr(node, _name, getattr(planner, _name))
node.INACTIVE = (planner.HighwayOvertakePlanner.OFF, planner.HighwayOvertakePlanner.DONE)
_limit_speed_rate = node.HighwayOvertakeNode._limit_speed_rate


def _limit_speed_rate_capped(self, target, dt):
    """The pass-through speed, also below the planner's cap toward the highway end."""
    cap = getattr(self.planner, "pass_through_cap_mps", None)
    return _limit_speed_rate(self, target if cap is None else min(target, cap), dt)


node.HighwayOvertakeNode._limit_speed_rate = _limit_speed_rate_capped


if __name__ == "__main__":
    try:
        node.HighwayOvertakeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
