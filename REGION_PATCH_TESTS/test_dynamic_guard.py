import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from purepursuit_mgeo.path import PathPoint,load_mgeo_path
from purepursuit_mgeo.dynamic_zone_guard import DynamicZoneGuard,constrain_pedals,segment_box_distance,segment_polygon_distance,convex_hull


def obstacle(key=1,x=50.,y=0.,vx=0.,vy=0.,state='MOVING',length=1.,width=1.,yaw=0.):
    return dict(id=key,center_x_map=x,center_y_map=y,length=length,width=width,
                velocity_x_map=vx,velocity_y_map=vy,speed_mps=math.hypot(vx,vy),
                motion_state=state,yaw=yaw,yaw_valid=True)


class DynamicGuardTests(unittest.TestCase):
    def setUp(self):
        self.points=[PathPoint(float(i),0.,0.) for i in range(201)]
        self.regions=[dict(name='dynamic_obstacle',start_xyz=[40.,0.,0.],end_xyz=[80.,0.,0.])]
        self.g=DynamicZoneGuard(self.points,self.regions,approach_m=10.)
        self.t=100.

    def tick(self,obs,x=45.,speed=0.,dt=.2,stamp=None):
        self.t+=dt
        stamp=self.t if stamp is None else stamp
        sample=(dict(timestamp=stamp,obstacle_count=len(obs),obstacles=obs),self.t)
        return self.g.step(self.t,self.t,(x,0.),speed,sample)

    def test_outside_unchanged_and_approach_and_zone_cap_20(self):
        self.assertIsNone(self.tick([],x=20.,speed=12.)['cap_mps'])
        self.assertAlmostEqual(self.tick([],x=31.,speed=12.)['cap_mps'],20/3.6)
        self.assertAlmostEqual(self.tick([],x=40.,speed=12.)['cap_mps'],20/3.6)
        self.assertIsNone(self.tick([],x=80.,speed=5.)['cap_mps'])

    def test_current_route_blocker_stops_immediately(self):
        result=self.tick([obstacle()])
        self.assertTrue(result['stop']);self.assertEqual(result['waiting_ids'],[1])

    def test_adjacent_obstacle_does_not_stop(self):
        self.assertFalse(self.tick([obstacle(y=4.)])['stop'])

    def test_predictive_crossing_stops_before_center_enters_route(self):
        self.assertTrue(self.tick([obstacle(y=4.,vy=-4.)])['stop'])

    def test_same_id_moves_out_and_stays_out_then_resume(self):
        self.assertTrue(self.tick([obstacle()])['stop'])
        self.assertTrue(self.tick([obstacle(y=4.)])['stop'])
        for _ in range(5):result=self.tick([obstacle(y=4.)])
        self.assertFalse(result['stop']);self.assertAlmostEqual(result['cap_mps'],20/3.6)

    def test_stopped_or_static_reclassification_inside_keeps_stop(self):
        self.tick([obstacle()])
        for state in ('STOPPED','STATIC','UNKNOWN'):
            self.assertTrue(self.tick([obstacle(state=state)])['stop'])

    def test_initial_static_collision_is_not_driven_through(self):
        self.assertTrue(self.tick([obstacle(state='STATIC')])['stop'])

    def test_disappearance_and_id_switch_do_not_authorize_departure(self):
        self.tick([obstacle()])
        for _ in range(5):self.assertTrue(self.tick([])['stop'])
        result=self.tick([obstacle(key=2,y=4.)])
        self.assertTrue(result['stop']);self.assertIn(1,result['unobserved_ids'])

    def test_multiple_blockers_all_must_be_seen_clear(self):
        self.tick([obstacle(),obstacle(2,x=55.)])
        for _ in range(6):r=self.tick([obstacle(y=4.),obstacle(2,x=55.)])
        self.assertTrue(r['stop']);self.assertEqual(r['waiting_ids'],[2])
        for _ in range(6):r=self.tick([obstacle(y=4.),obstacle(2,x=55.,y=-4.)])
        self.assertFalse(r['stop'])

    def test_repeated_source_frames_do_not_add_clear_votes(self):
        self.tick([obstacle()])
        self.tick([obstacle(y=4.)]);source=self.t
        for _ in range(5):self.assertTrue(self.tick([obstacle(y=4.)],stamp=source)['stop'])

    def test_input_loss_inside_stops_outside_does_not(self):
        self.assertFalse(self.g.step(100.,100.,(20.,0.),12.,None)['stop'])
        self.assertTrue(self.g.step(100.1,100.1,(45.,0.),5.,None)['stop'])

    def test_stop_survives_region_exit_until_observed_clear(self):
        self.tick([obstacle()])
        self.assertTrue(self.tick([],x=81.)['stop'])

    def test_vehicle_motion_does_not_count_as_obstacle_clearance(self):
        self.tick([obstacle()])
        for _ in range(5):self.assertTrue(self.tick([obstacle(y=4.)],speed=1.)['stop'])

    def test_old_or_malformed_input_cannot_clear(self):
        self.tick([obstacle()])
        self.assertTrue(self.tick([obstacle(y=4.)],stamp=self.t-10)['stop'])
        self.t+=.2
        self.assertTrue(self.g.step(self.t,self.t,(45.,0.),0.,({'timestamp':self.t,'obstacles':None},self.t))['stop'])

    def test_footprint_corner_not_center_controls_blocking(self):
        # Center is outside the corridor, but a rotated long box intrudes.
        self.assertTrue(self.tick([obstacle(y=2.5,length=5.,width=1.,yaw=math.pi/4)])['stop'])

    def test_clearance_hysteresis_prevents_border_jitter_release(self):
        self.tick([obstacle()])
        for _ in range(7):self.assertTrue(self.tick([obstacle(y=1.9)])['stop'])

    def test_live_executed_path_is_also_protected(self):
        self.t+=.2
        o=obstacle(y=4.)
        sample=(dict(timestamp=self.t,obstacles=[o]),self.t)
        live=[(45.,4.),(60.,4.)]
        self.assertTrue(self.g.step(self.t,self.t,(45.,0.),0.,sample,live)['stop'])

    def test_speed_overlay_never_overrides_existing_brake_or_steering(self):
        cmd=NS(accel=.8,brake=.4,steering=.25)
        constrain_pedals(cmd,dict(stop=False,cap_mps=20/3.6),3.)
        self.assertEqual(cmd.brake,.4);self.assertEqual(cmd.accel,0.);self.assertEqual(cmd.steering,.25)
        cmd=NS(accel=.8,brake=0.,steering=-.1)
        constrain_pedals(cmd,dict(stop=False,cap_mps=20/3.6),12.)
        self.assertEqual(cmd.accel,0.);self.assertGreater(cmd.brake,0.)

    def test_sensor_missing_after_stop_resets_clear_confirmation(self):
        self.tick([obstacle()]);self.tick([obstacle(y=4.)]);self.tick([obstacle(y=4.)])
        self.t+=1.
        self.assertTrue(self.g.step(self.t,self.t,(45.,0.),0.,None)['stop'])
        self.assertTrue(self.tick([obstacle(y=4.)])['stop'])

    def test_actual_highway_position_adds_no_speed_or_stop_limit(self):
        root=Path(__file__).resolve().parents[1]
        points=load_mgeo_path(str(root/'data/routes/2026_molit_comp_global_path.txt'))
        region=dict(name='dynamic_obstacle',start_xyz=[-59.36,-63.85,29.86],end_xyz=[-60.07,16.88,29.87])
        g=DynamicZoneGuard(points,[region])
        p=points[3118]
        result=g.step(100.,100.,(p.x,p.y),12.,None)
        self.assertFalse(result['stop']);self.assertIsNone(result['cap_mps'])

    def test_actual_route_dynamic_interval(self):
        root=Path(__file__).resolve().parents[1]
        points=load_mgeo_path(str(root/'data/routes/2026_molit_comp_global_path.txt'))
        region=dict(name='dynamic_obstacle',start_xyz=[-59.36,-63.85,29.86],end_xyz=[-60.07,16.88,29.87])
        g=DynamicZoneGuard(points,[region])
        self.assertEqual(g.gate.regions[0]['start_index'],828)
        self.assertEqual(g.gate.regions[0]['end_index'],998)


if __name__=='__main__':unittest.main()
