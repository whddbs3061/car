#!/usr/bin/env python3
"""highway_overtake_node.py with the escape3 planner (highway_overtake_escape3).

The node script is loaded unchanged; only the planner module it uses is the
escape3 variant.  Topics, parameters and the state message are those of
highway_overtake_node.py; config/highway_overtake_escape3.yaml, loaded over
highway_overtake.yaml, sets the gentler change geometry.
"""
import importlib.util
import os
import sys

import rospy

from purepursuit_mgeo import highway_overtake_escape3 as planner

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


if __name__ == "__main__":
    try:
        node.HighwayOvertakeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
