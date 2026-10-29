#!/usr/bin/env python3
import copy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parents[2]
sys.path.insert(0,str(PACKAGE/"src"))

from mission_evaluator.course import Route, Course, LaneMap, circle_entry
from mission_evaluator.competition import CompetitionRun, best_attempt
from mission_evaluator.collision_udp import HEADER, RECORD, PACKET_SIZE, parse_collision_packet


def rules():
    return json.loads((PACKAGE/"config"/"competition_rules_v1_1.json").read_text(encoding="utf-8"))


def small_course():
    config = rules()
    config["checkpoint_radius_m"] = 0.5
    config["practice"]["initial_search_m"] = 2.0
    route = Route([(0,0,0),(20,0,0),(20,20,0),(0,20,0),(0,0,0)])
    cp = [{"id":str(i),"xyz":list(route.point_at(s)),"s_m":s} for i,s in enumerate((10,30,50,70),1)]
    return SimpleNamespace(route=route,rules=config,radius=0.5,start=(0,0,0),checkpoints=cp,
                           highway=(30,50),highway_active=lambda s:30 <= s < 50)


def started():
    run = CompetitionRun(small_course())
    run.observe_pose(0,0,0,0,36)
    return run


class LapTests(unittest.TestCase):
    def test_same_start_end_does_not_finish(self):
        run = started()
        for i in range(1,20):
            run.observe_pose(i/10,0,0,0,0)
        self.assertEqual(run.state,"RUNNING")
        self.assertEqual(run.passed,[])

    def test_complete_lap_and_frozen_time(self):
        run = started()
        for i in range(1,161):
            p = run.course.route.point_at(i/2)
            run.observe_pose(i/20,p.x,p.y,0,36)
        self.assertEqual(run.state,"FINISHED")
        self.assertEqual(run.passed,["1","2","3","4"])
        before = run.snapshot(8)
        run.tick(500)
        run.observe_collision(500,["1:2"])
        after = run.snapshot(500)
        self.assertEqual(before,after)
        self.assertAlmostEqual(after["lap_time_sec"],7.95)

    def test_no_start_outside_start_disc(self):
        run = CompetitionRun(small_course())
        run.observe_pose(0,20,20,0,36)
        self.assertEqual(run.state,"WAITING")

    def test_manual_start(self):
        run = CompetitionRun(small_course(),auto_start=False)
        self.assertFalse(run.start(0)[0])
        run.observe_pose(0,0,0,0,0)
        self.assertTrue(run.start(0.1)[0])
        self.assertEqual(run.start_time,0.1)

    def test_start_pose_must_be_fresh(self):
        run = CompetitionRun(small_course(),auto_start=False)
        run.observe_pose(0,0,0,0,0)
        self.assertFalse(run.start(1)[0])

    def test_start_deadline_late_progress_not_passed(self):
        run = started()
        run.observe_pose(60.01,5,0,0,1)
        self.assertEqual(run.state,"FAILED")
        self.assertEqual(run.start_mission,"FAILED")

    def test_time_limit_not_adjusted_time(self):
        run = started()
        run.start_mission = "SUCCESS"
        run.event(1,"test","penalty",1000)
        run.tick(899)
        self.assertEqual(run.state,"RUNNING")
        run.tick(900)
        self.assertEqual(run.state,"RUNNING")
        run.tick(900.01)
        self.assertEqual(run.state,"FAILED")

    def test_clock_reset_invalidates(self):
        run = started()
        run.tick(1)
        run.tick(0)
        self.assertEqual(run.state,"INVALID")
        self.assertEqual(run.snapshot(0)["lap_time_sec"],1)

    def test_pause_same_ros_time_no_elapsed(self):
        run = started()
        for _ in range(100):
            run.tick(0)
        self.assertEqual(run.snapshot(0)["lap_time_sec"],0)

    def test_teleport_cannot_finish(self):
        run = started()
        run.observe_pose(.1,0,1,0,0)
        run.observe_pose(.2,20,20,0,36)
        self.assertEqual(run.state,"NEEDS_REVIEW")
        self.assertFalse(run.snapshot(.2)["completed"])

    def test_gap_cannot_bridge_checkpoints(self):
        run = started()
        run.observe_pose(1,10,0,0,36)
        self.assertEqual(run.state,"NEEDS_REVIEW")
        self.assertEqual(run.passed,[])

    def test_missing_speed_not_used_for_continuity(self):
        run = started()
        self.assertFalse(run.observe_pose(.1,1,0,0,None))
        self.assertEqual(run.progress,0)

    def test_checkpoint_skipping_requires_review(self):
        run = started()
        for i in range(1,151):
            x = i/10
            # Detour misses the 0.5m disc but remains in the route corridor.
            y = min(2,max(0,(x-3)/2)) if x < 11 else max(0,2-(x-11))
            run.observe_pose(i/100,x,y,0,72)
            if run.state=="NEEDS_REVIEW":
                break
        self.assertEqual(run.reason,"checkpoint_skipped")
        self.assertEqual(run.passed,[])

    def test_recovery_requires_last_checkpoint_and_costs_15(self):
        run = started()
        run.observe_pose(.1,20,20,0,36)
        self.assertFalse(run.approve_recovery(.1)[0])
        run.observe_pose(.2,0,0,0,0)
        self.assertTrue(run.approve_recovery(.2)[0])
        self.assertEqual(run.penalty,15)
        self.assertEqual(run.start_time,0)

    def test_speed_exact_limit_no_penalty(self):
        run = started()
        run.observe_speed(0,60,False)
        self.assertEqual(run.penalty,0)

    def test_speed_immediate_and_three_second_repeat(self):
        run = started()
        for i in range(61):
            run.observe_speed(i/10,61,False)
        self.assertEqual(run.penalty,45)

    def test_highway_exemption_resets_speed_duration(self):
        run = started()
        run.observe_speed(0,100,False)
        run.observe_speed(.1,100,True)
        run.observe_speed(.2,100,False)
        self.assertEqual(run.penalty,30)

    def test_stale_speed_never_accrues_gap_penalties(self):
        run = started()
        run.observe_speed(0,61,False)
        run.tick(30)
        self.assertEqual(run.penalty,15)
        run.observe_speed(30,61,False)
        self.assertEqual(run.penalty,30)
        self.assertIn("speed_input_gap",run.unassessed)

    def test_unknown_region_is_not_free_speed_pass(self):
        run = started()
        run.observe_speed(0,100,None)
        self.assertIn("speed_region_unavailable",run.unassessed)
        self.assertFalse(run.snapshot(0)["coverage_complete"])

    def test_lane_three_seconds_and_blackout(self):
        run = started()
        for i in range(61):
            run.observe_lane(i/10,True,False)
        self.assertEqual(run.penalty,10)
        for i in range(61,121):
            run.observe_lane(i/10,True,True)
        self.assertEqual(run.penalty,10)

    def test_lane_gap_resets_contact_duration(self):
        run = started()
        for i in range(21):
            run.observe_lane(i/10,True,False)
        run.observe_lane(5,True,False)
        self.assertEqual(run.penalty,0)
        self.assertIn("lane_input_gap",run.unassessed)

    def test_unknown_lane_blackout_is_unassessed(self):
        run = started()
        run.observe_lane(0,True,None)
        self.assertIn("lane_contact_or_blackout_unavailable",run.unassessed)

    def test_collision_continuous_release_recontact(self):
        run = started()
        run.observe_collision(0,["1:7"])
        run.observe_collision(.1,["1:7"])
        run.observe_collision(.2,[])
        run.observe_collision(.3,["1:7"])
        self.assertEqual(run.penalty,30)

    def test_collision_gap_is_not_release(self):
        run = started()
        run.observe_collision(0,["1:7"])
        run.observe_collision(5,["1:7"])
        self.assertEqual(run.penalty,15)
        self.assertIn("collision_input_gap",run.unassessed)

    def test_collision_same_id_different_type(self):
        run = started()
        run.observe_collision(0,["1:7","2:7"])
        self.assertEqual(run.penalty,30)

    def test_signal_once_and_unknown(self):
        run = started()
        run.observe_signal(0,"junction1",False)
        run.observe_signal(.1,"junction1",False)
        run.observe_signal(.2,"junction2",None)
        self.assertEqual(run.penalty,15)
        self.assertIn("signal_crossing_unassessed:junction2",run.unassessed)

    def test_invalid_numbers_rejected(self):
        run = started()
        with self.assertRaises(ValueError):
            run.observe_speed(.1,float("nan"),False)
        with self.assertRaises(ValueError):
            run.observe_pose(.1,float("inf"),0,0,1)

    def test_best_attempt_prioritizes_completion_then_checkpoints(self):
        def attempt(state,checkpoints,t):
            return dict(state=state,completed=state=="FINISHED",checkpoints_passed=list(range(checkpoints)),adjusted_time_sec=t)
        a,b,c = attempt("FAILED",4,200),attempt("FINISHED",4,500),attempt("FINISHED",4,400)
        self.assertIs(best_attempt([a,b,c]),c)
        d = attempt("FAILED",3,100)
        self.assertIs(best_attempt([a,d]),a)
        self.assertIsNone(best_attempt([attempt("INVALID",4,10)]))


class GeometryTests(unittest.TestCase):
    def test_circle_entry_and_tangent(self):
        self.assertAlmostEqual(circle_entry((-2,0),(2,0),(0,0),1),.25)
        self.assertEqual(circle_entry((-2,1),(2,1),(0,0),1),.5)
        self.assertIsNone(circle_entry((-2,2),(2,2),(0,0),1))

    def test_duplicate_route_points_removed(self):
        route = Route([(0,0,0),(0,0,0),(3,4,0)])
        self.assertEqual(route.length,5)
        self.assertAlmostEqual(route.project((1.5,2))["s_m"],2.5)

    def test_projection_window_does_not_choose_closed_finish(self):
        route = small_course().route
        self.assertLessEqual(route.project((0,0),upper=2)["s_m"],2)
        self.assertGreater(route.project((0,0),lower=75)["s_m"],75)
        self.assertEqual(route.project((0,0),lower=route.length)["s_m"],route.length)

    def test_invalid_route_rejected(self):
        for points in ([(0,0,0)],[(0,0,0),(float("nan"),0,0)]):
            with self.assertRaises(ValueError):
                Route(points)

    def test_wheel_solid_contact_excludes_stopline_and_broken(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/"lanes.json"
            record = dict(idx="solid",lane_type=[503],lane_shape=["solid"],lane_width=.15,points=[[-5,.8,0],[5,.8,0]])
            stop = dict(record,idx="stopline",lane_type=[530],points=[[-5,-.8,0],[5,-.8,0]])
            broken = dict(record,idx="broken",lane_shape=["broken"])
            file.write_text(json.dumps([record,stop,broken]),encoding="utf-8")
            lane = LaneMap(file,rules()["practice"])
            self.assertEqual(lane.contact(0,0,0),["solid"])
            self.assertEqual(lane.contact(0,4,0),[])


class CollisionPacketTests(unittest.TestCase):
    def packet(self,records=None,sec=1,nsec=0):
        entries = list(records or [])
        entries += [(0,0,0,0,0,0,0,0)]*(5-len(entries))
        return HEADER.pack(b"#CollisionData$",148,0,0,0,sec,nsec)+b"".join(RECORD.pack(*r) for r in entries)+b"\r\n"

    def test_empty_snapshot(self):
        packet = self.packet()
        self.assertEqual(len(packet),181)
        self.assertEqual(PACKET_SIZE,181)
        self.assertEqual(parse_collision_packet(packet)["keys"],[])

    def test_identity_preserves_type_and_zero_id(self):
        data = parse_collision_packet(self.packet([(1,0,1,2,3,0,0,0),(2,0,1,2,3,0,0,0)]))
        self.assertEqual(data["keys"],["1:0","2:0"])

    def test_truncated_or_wrong_header_rejected(self):
        for packet in (self.packet()[:-1],b"!"+self.packet()[1:],self.packet()[:-2]+b"xx"):
            with self.assertRaises(ValueError):
                parse_collision_packet(packet)

    def test_nonfinite_and_bad_timestamp_rejected(self):
        for packet in (self.packet([(1,3,float("nan"),0,0,0,0,0)]),self.packet(nsec=1_000_000_000)):
            with self.assertRaises(ValueError):
                parse_collision_packet(packet)


class LaunchTests(unittest.TestCase):
    def test_launch_files_parse_and_control_stays_disabled(self):
        observer = ET.parse(PACKAGE/"launch"/"competition_lap.launch").getroot()
        wrapper = ET.parse(PACKAGE/"launch"/"competition_practice.launch").getroot()
        self.assertEqual(wrapper.find("arg[@name='enable_control']").get("default"),"false")
        self.assertEqual(observer.find("arg[@name='collision_udp_enabled']").get("default"),"false")
        self.assertEqual(observer.find("arg[@name='collision_source_port']").get("default"),"9091")
        self.assertEqual(observer.find("arg[@name='collision_port']").get("default"),"9092")
        bridge = observer.find("node[@name='collision_evaluation_bridge']")
        self.assertEqual(bridge.find("param[@name='allowed_sender_port']").get("value"),
                         "$(arg collision_source_port)")
        self.assertFalse(any("ctrl_cmd" in ET.tostring(n,encoding="unicode") for n in observer.iter("node")))

    def test_production_launch_not_changed_to_include_scoring(self):
        for filename in ("final_ws_bringup.launch","perception_control_bringup.launch"):
            document = ET.parse(WORKSPACE/"src"/"bringup"/"morai_bringup"/"launch"/filename)
            self.assertFalse(any("mission_evaluator" in n.get("file","") for n in document.iter("include")))

    def test_scripts_registered_for_catkin_install(self):
        cmake = (PACKAGE/"CMakeLists.txt").read_text()
        for name in ("competition_lap_node.py","collision_evaluation_bridge.py","inspect_competition_course.py"):
            self.assertIn("scripts/"+name,cmake)
            compile((PACKAGE/"scripts"/name).read_text(encoding="utf-8"),name,"exec")


class RealCourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.course = Course(Route.load(WORKSPACE/"data"/"routes"/"2026_molit_comp_global_path.txt"),rules(),
                            WORKSPACE/"data"/"mgeo"/"R_KR_PR_K-city_2025")

    def test_official_course_and_highway(self):
        course = self.course
        self.assertAlmostEqual(course.route.length,2184.611723,places=5)
        self.assertEqual(len(course.checkpoints),15)
        self.assertAlmostEqual(course.highway[0],1118.741751,places=5)
        self.assertAlmostEqual(course.highway[1],1741.720989,places=5)
        self.assertTrue(course.highway_active(course.highway[0]))
        self.assertFalse(course.highway_active(course.highway[1]))
        self.assertIsNone(course.blackout(1500))

    def test_all_real_checkpoints_pass_and_finish(self):
        run = CompetitionRun(self.course)
        start = self.course.start
        run.observe_pose(0,start[0],start[1],0,36)
        steps = math.ceil(self.course.route.length/.5)
        for i in range(1,steps+1):
            p = self.course.route.point_at(i*.5)
            run.observe_pose(i*.05,p.x,p.y,0,36)
            if run.state in run.TERMINAL:
                break
        self.assertEqual(run.state,"FINISHED",run.snapshot(i*.05))
        self.assertEqual(run.passed,[str(i) for i in range(1,16)])
        self.assertEqual(run.start_mission,"SUCCESS")
        self.assertLess(run.snapshot(i*.05)["lap_time_sec"],219)


if __name__ == "__main__":
    unittest.main()
