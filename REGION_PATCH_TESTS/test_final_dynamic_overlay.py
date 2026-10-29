"""Real final command node callbacks, with fixture ROS boundary only mocked."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT=Path(__file__).resolve().parents[1]
FIXTURES=ROOT/'src/control/turn_signal_controller/test'
sys.path.insert(0,str(FIXTURES))
import test_maneuver_fusion as fixtures
from purepursuit_mgeo.dynamic_zone_guard import DynamicZoneGuard


class FinalDynamicOverlayTests(unittest.TestCase):
    def setUp(self):
        self.c=fixtures.FusionNodeTest();self.c.setUp();self.node=self.c.node
        region=dict(name='dynamic_obstacle',start_xyz=[40,0,0],end_xyz=[70,0,0])
        self.node.dynamic_zone_guard=DynamicZoneGuard(self.node.points,[region],approach_m=10.)

    def tick(self,objects,speed=0.,x=45.):
        self.c.now+=.2
        self.node.dynamic_obstacles_callback(NS(data=json.dumps(dict(timestamp=self.c.now,objects=objects,
                                  obstacles=objects,obstacle_count=len(objects)))))
        return self.c.tick(signal=None,line=None,x=x,speed=speed)

    def obstacle(self,y=0.):
        return dict(id=7,center_x_map=50.,center_y_map=y,length=1.,width=1.,
                    velocity_x_map=0.,velocity_y_map=0.,yaw=0.,yaw_valid=True,motion_state='MOVING')

    def test_real_constructor_loads_dynamic_private_parameters(self):
        self.c.params.update({"~dynamic_zone_enabled":True,
             "~avoidance_regions":[dict(name="dynamic_obstacle",start_xyz=[40,0,0],end_xyz=[70,0,0])]})
        node=self.c.module.ManeuverFusionNode()
        self.assertIsInstance(node.dynamic_zone_guard,DynamicZoneGuard)
        self.assertAlmostEqual(node.dynamic_zone_guard.speed_mps,20/3.6)

    def test_speed_cap_reaches_only_existing_final_writer(self):
        output,status=self.tick([],speed=12.)
        self.assertEqual(output.accel,0.)
        self.assertGreater(output.brake,0.)
        self.assertAlmostEqual(status['dynamic_zone_guard']['cap_mps'],20/3.6)
        self.assertAlmostEqual(status['effective_speed_target_kph'],20.)

    def test_stop_reaches_final_command_and_mode_reason(self):
        output,status=self.tick([self.obstacle()],speed=5.)
        self.assertEqual(output.brake,1.);self.assertEqual(output.accel,0.)
        self.assertEqual(status['mode'],'SAFE_STOP')
        self.assertIn('dynamic_wait_for_path_clear',status['reason'])
        self.assertEqual(status['dynamic_zone_guard']['waiting_ids'],[7])

    def test_observed_clear_releases_guard_within_region(self):
        self.tick([self.obstacle()])
        for _ in range(6):output,status=self.tick([self.obstacle(y=4.)])
        self.assertFalse(status['dynamic_zone_guard']['stop'])
        self.assertEqual(status['dynamic_zone_guard']['waiting_ids'],[])
        self.assertAlmostEqual(status['dynamic_zone_guard']['cap_mps'],20/3.6)

    def test_other_upstream_stop_stays_braking_after_guard_clear(self):
        self.tick([])
        self.node.command_callback(fixtures.Command(accel=0.,brake=1.))
        self.node.tick(None)
        command=self.node.output_pub.publish.call_args.args[0]
        self.assertEqual(command.brake,1.);self.assertEqual(command.accel,0.)

    def test_outside_dynamic_zone_has_no_new_limit(self):
        output,status=self.tick([],x=10.,speed=8.)
        self.assertIsNone(status['dynamic_zone_guard']['cap_mps'])
        self.assertFalse(status['dynamic_zone_guard']['stop'])


if __name__=='__main__':unittest.main()
