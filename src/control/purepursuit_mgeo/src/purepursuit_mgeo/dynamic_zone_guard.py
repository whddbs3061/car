"""20 km/h stop/wait mission on a calibrated route interval; ROS independent.

New blockers must intersect the swept path and have map speed >= the configured
threshold. Once held, an actor must leave the path even if it slows or stops.
Missing IDs are never treated as observed clearance.
"""
import bisect
import math
from purepursuit_mgeo.route_avoidance_gate import RouteAvoidanceGate


def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)


def point_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    l2=dx*dx+dy*dy
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/l2)) if l2>1e-12 else 0.
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def segment_distance(a,b,c,d):
    def cross(u,v,w):return (v[0]-u[0])*(w[1]-u[1])-(v[1]-u[1])*(w[0]-u[0])
    if (cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0
            and max(min(a[0],b[0]),min(c[0],d[0]))<=min(max(a[0],b[0]),max(c[0],d[0]))
            and max(min(a[1],b[1]),min(c[1],d[1]))<=min(max(a[1],b[1]),max(c[1],d[1]))):
        return 0.
    return min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))


def segment_box_distance(a,b,o):
    x,y=o['center_x_map'],o['center_y_map']
    if not o['yaw_valid']:
        return max(0.,point_segment((x,y),a,b)-math.hypot(o['length'],o['width'])/2)
    c,s=math.cos(o['yaw']),math.sin(o['yaw'])
    def local(p):
        dx,dy=p[0]-x,p[1]-y
        return (c*dx+s*dy,-s*dx+c*dy)
    a,b=local(a),local(b)
    l,w=o['length']/2,o['width']/2
    if any(abs(p[0])<=l and abs(p[1])<=w for p in (a,b)):return 0.
    corners=[(-l,-w),(-l,w),(l,w),(l,-w)]
    return min(segment_distance(a,b,corners[i],corners[(i+1)%4]) for i in range(4))


def convex_hull(points):
    points=sorted(set(points))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def half(sequence):
        h=[]
        for p in sequence:
            while len(h)>=2 and cross(h[-2],h[-1],p)<=0:h.pop()
            h.append(p)
        return h
    return half(points)[:-1]+half(reversed(points))[:-1]


def segment_polygon_distance(a,b,polygon):
    def inside(p):
        signs=[(polygon[(i+1)%len(polygon)][0]-q[0])*(p[1]-q[1])-
               (polygon[(i+1)%len(polygon)][1]-q[1])*(p[0]-q[0]) for i,q in enumerate(polygon)]
        return all(v>=-1e-9 for v in signs) or all(v<=1e-9 for v in signs)
    if inside(a) or inside(b):return 0.
    return min(segment_distance(a,b,q,polygon[(i+1)%len(polygon)]) for i,q in enumerate(polygon))


def path_prefix(points,x,y,length):
    """Clip a fresh executed path around the closest segment, then forward."""
    if not points or len(points)<2:return []
    i=min(range(len(points)-1),key=lambda i:point_segment((x,y),points[i],points[i+1]))
    out=[];remaining=length
    for j in range(max(0,i-1),len(points)-1):
        a,b=points[j],points[j+1]
        step=math.hypot(b[0]-a[0],b[1]-a[1])
        if step<=1e-9:continue
        if step>remaining:
            ratio=remaining/step
            b=(a[0]+ratio*(b[0]-a[0]),a[1]+ratio*(b[1]-a[1]))
        out.append((a,b));remaining-=step
        if remaining<=0:break
    return out


class DynamicZoneGuard:
    def __init__(self,points,regions,expected_fingerprint='',speed_kph=20.,
                 approach_m=40.,lookahead_m=30.,vehicle_width_m=1.892,
                 margin_m=.3,clearance_extra_m=.3,clear_confirm_s=.6,
                 timeout_s=.6,prediction_s=2.,ego_front_m=3.845,min_obstacle_speed_mps=2.,wait_before_static_s=5.,tunnel_enabled=True):
        chosen=[r for r in regions if r['name']=='dynamic_obstacle']
        if len(chosen)!=1:raise ValueError('exactly one dynamic_obstacle region required')
        values=(speed_kph,approach_m,lookahead_m,vehicle_width_m,margin_m,
                clearance_extra_m,clear_confirm_s,timeout_s,prediction_s,ego_front_m)
        if not all(number(v) and v>0 for v in values):raise ValueError('invalid dynamic mission parameters')
        if not number(min_obstacle_speed_mps) or min_obstacle_speed_mps<0:
            raise ValueError("invalid minimum obstacle speed")
        if not number(wait_before_static_s) or wait_before_static_s<=0:
            raise ValueError("invalid static handoff wait")
        self.wait_before_static_s=wait_before_static_s
        self.tunnel_enabled=bool(tunnel_enabled)
        self.stopped_since=None;self.wait_elapsed_s=0.;self.handoff_requested=False
        self.detour_driving=False;self.tunnel_active=False
        self.min_obstacle_speed_mps=min_obstacle_speed_mps
        self.gate=RouteAvoidanceGate(points,chosen,expected_fingerprint,gps_blackout_override=False)
        self.start=self.gate.regions[0]['start_s_m'];self.end=self.gate.regions[0]['end_s_m']
        self.points=[(p.x,p.y) for p in points]
        self.speed_mps=speed_kph/3.6;self.approach=approach_m
        self.lookahead=lookahead_m;self.front=ego_front_m
        self.half_width=vehicle_width_m/2+margin_m
        self.clear_extra=clearance_extra_m;self.confirm=clear_confirm_s
        self.timeout=timeout_s;self.prediction=prediction_s
        self.held={};self.last_stamp=None;self.valid=False;self.last_received=None
        self.input_error="not_received"
        self.latest={};self.last_active=False;self.last_now=None

    def engaged(self):return self.last_active or bool(self.held)

    def _segments(self,progress,live_path,pose):
        s=self.gate.s;first=max(0,bisect.bisect_left(s,progress-1.)-1)
        last=min(len(s)-1,bisect.bisect_right(s,progress+self.lookahead+self.front))
        segments=[(self.points[i],self.points[i+1]) for i in range(first,last)]
        if live_path:
            segments+=path_prefix(live_path,pose[0],pose[1],self.lookahead+self.front)
        return segments

    def _near(self,o,segments,extra=0.):
        # Exact continuous translation of an oriented box over prediction_s,
        # rather than testing a few timestamps and missing a fast crossing.
        vx,vy=o['velocity_x_map'],o['velocity_y_map']
        if o['motion_state']=='STATIC':vx=vy=0.
        x,y=o['center_x_map'],o['center_y_map']
        end=(x+vx*self.prediction,y+vy*self.prediction)
        margin=self.half_width+extra
        if not o['yaw_valid']:
            radius=math.hypot(o['length'],o['width'])/2+margin
            return any(segment_distance(a,b,(x,y),end)<=radius for a,b in segments)
        c,s=math.cos(o['yaw']),math.sin(o['yaw'])
        l,w=o['length']/2,o['width']/2
        polygon=convex_hull([(cx+c*dx-s*dy,cy+s*dx+c*dy)
                             for cx,cy in ((x,y),end) for dx,dy in ((-l,-w),(-l,w),(l,-w),(l,w))])
        xmin=min(p[0] for p in polygon)-margin;xmax=max(p[0] for p in polygon)+margin
        ymin=min(p[1] for p in polygon)-margin;ymax=max(p[1] for p in polygon)+margin
        for a,b in segments:
            if (max(a[0],b[0])<xmin or min(a[0],b[0])>xmax
                    or max(a[1],b[1])<ymin or min(a[1],b[1])>ymax):continue
            if segment_polygon_distance(a,b,polygon)<=margin:return True
        return False

    def _ingest(self,sample,now,ros_now):
        if sample is None:return False
        payload,received=sample
        try:
            stamp=payload['timestamp'];raw=payload['obstacles']
            if (not number(stamp) or not -.05<=ros_now-stamp<=self.timeout
                    or not 0<=now-received<=self.timeout or not isinstance(raw,list)):
                raise ValueError('stale or invalid input')
            if self.last_stamp is not None and stamp<=self.last_stamp:return False
            parsed={}
            for raw_o in raw:
                o=dict(raw_o);key=o['id']
                if type(key) is not int or key<0 or key in parsed:raise ValueError('ambiguous ID')
                for field in ('center_x_map','center_y_map','length','width','velocity_x_map','velocity_y_map','yaw'):
                    if not number(o.get(field)):raise ValueError('invalid geometry')
                if o['length']<=0 or o['width']<=0:raise ValueError('empty footprint')
                o['yaw_valid']=o.get('yaw_valid') is True
                o['motion_state']=str(o.get('motion_state','UNKNOWN')).upper()
                parsed[key]=o
            if 'obstacle_count' in payload and payload['obstacle_count']!=len(raw):raise ValueError('count mismatch')
            self.latest=parsed;self.valid=True;self.last_stamp=stamp;self.last_received=received
            self.input_error=""
            return True
        except (ValueError,KeyError,TypeError) as error:
            self.valid=False
            self.input_error=str(error)
            return False

    def step(self,now,ros_now,pose,speed,sample,live_path=None,tunnel_mode=False,detour_path=None):
        if not number(now) or not number(ros_now):return self.result(True,'dynamic_clock_invalid',True)
        if self.last_now is not None and now<self.last_now:
            self.valid=False
            return self.result(True,'dynamic_clock_reset_restart_required',True)
        self.last_now=now
        self.tunnel_active=bool(self.tunnel_enabled and tunnel_mode)
        self.detour_driving=False
        if pose is None or not all(number(v) for v in pose):
            self.stopped_since=None;self.wait_elapsed_s=0.
            return self.result(self.engaged(),'dynamic_pose_unavailable',self.last_active)
        region=self.gate.update(pose[0],pose[1])
        s=region['progress_s_m']
        matched=region['route_matched']
        active=bool(matched and (region['regions'] or self.tunnel_active))
        approach=bool(matched and s is not None and self.start-self.approach<=s<self.start)
        if not matched and (self.last_active or self.tunnel_active):
            self.stopped_since=None;self.wait_elapsed_s=0.
            return self.result(True,'dynamic_route_unmatched',True)
        self.last_active=bool(active or approach)
        new=self._ingest(sample,now,ros_now)
        # Away from this mission, stale obstacle input does not change other driving.
        if not active and not self.held:
            self._clear_wait()
            return self.result(False,'dynamic_approach_20kph' if approach else 'outside_dynamic_zone',approach)
        if not number(speed) or speed<0:
            self.stopped_since=None;self.wait_elapsed_s=0.
            return self.result(True,'dynamic_speed_unavailable',True)
        fresh=(self.valid and self.last_stamp is not None and -.05<=ros_now-self.last_stamp<=self.timeout
               and self.last_received is not None and 0<=now-self.last_received<=self.timeout)
        if not fresh:
            self.stopped_since=None;self.wait_elapsed_s=0.
            for track in self.held.values():track['clear_since']=None;track['clear_frames']=0
            return self.result(True,'dynamic_obstacle_input_unavailable',True)
        if callable(live_path):live_path=live_path()
        segments=self._segments(s if s is not None else self.start,live_path,pose)
        if new:
            for key,o in self.latest.items():
                if (key in self.held or math.hypot(o["velocity_x_map"],o["velocity_y_map"])>=self.min_obstacle_speed_mps) and self._near(o,segments):
                    if key not in self.held:
                        self._clear_wait()
                        self.held[key]=dict(origin=(o['center_x_map'],o['center_y_map']),
                                            last=(o['center_x_map'],o['center_y_map']),seen=now,
                                            clear_since=None,clear_frames=0,identity_lost=False)
                    self.held[key]['clear_since']=None;self.held[key]['clear_frames']=0
            released=[]
            for key,t in self.held.items():
                o=self.latest.get(key)
                if o is None:
                    t['clear_since']=None;t['clear_frames']=0
                    continue
                dt=max(0.,now-t['seen'])
                position=(o['center_x_map'],o['center_y_map'])
                gap=math.hypot(position[0]-t['last'][0],position[1]-t['last'][1])
                if dt>5. or gap>2.+20.*dt:t['identity_lost']=True
                moved=math.hypot(position[0]-t['origin'][0],position[1]-t['origin'][1])>=.2
                passed=False
                if self.handoff_requested and s is not None:
                    index=self.gate._nearest(position[0],position[1],0,len(self.gate.points))
                    passed=s>self.gate.s[index]+math.hypot(o['length'],o['width'])/2+1.5
                clear=(not t['identity_lost'] and ((moved and speed<=.2) or passed)
                       and not self._near(o,segments,self.clear_extra))
                if clear:
                    if t['clear_since'] is None:t['clear_since']=now
                    t['clear_frames']+=1
                    if now-t['clear_since']>=self.confirm and t['clear_frames']>=3:released.append(key)
                else:t['clear_since']=None;t['clear_frames']=0
                t.update(last=position,seen=now)
            for key in released:del self.held[key]
        if not self.held:
            self._clear_wait()
            return self.result(False,'dynamic_path_clear',active)
        # Count actual continuous standstill, only with observed path blockage.
        blocking=any(key in self.latest and self._near(self.latest[key],segments)
                     for key in self.held)
        identities_ok=all(key in self.latest and not t['identity_lost']
                          for key,t in self.held.items())
        if not self.handoff_requested:
            if speed<=.2 and blocking and identities_ok:
                if self.stopped_since is None:self.stopped_since=now
                self.wait_elapsed_s=now-self.stopped_since
                if self.wait_elapsed_s>=self.wait_before_static_s:
                    self.handoff_requested=True
            else:
                self.stopped_since=None;self.wait_elapsed_s=0.
        # An authorization/request alone cannot release the brake. The final
        # node must supply a fresh active plan matching the actual executed path.
        path=(detour_path() if callable(detour_path) else detour_path) if self.handoff_requested else None
        if self.handoff_requested and identities_ok and path and len(path)>=2:
            detour=path_prefix(path,pose[0],pose[1],self.lookahead+self.front)
            if detour and not any(self._near(o,detour) for o in self.latest.values()):
                self.detour_driving=True
                return self.result(False,'dynamic_static_detour_follow',True)
        return self.result(True,'dynamic_static_plan_wait' if self.handoff_requested
                           else 'dynamic_wait_for_path_clear',True)

    def _clear_wait(self):
        self.stopped_since=None;self.wait_elapsed_s=0.;self.handoff_requested=False

    def result(self,stop,reason,limit):
        return dict(stop=bool(stop),reason=reason,
                    dynamic_guard_version='DYNAMIC_WAIT5_TUNNEL_V1',
                    tunnel_dynamic_active=self.tunnel_active,
                    wait_elapsed_s=self.wait_elapsed_s,
                    wait_before_static_s=self.wait_before_static_s,
                    request_static_avoidance=self.handoff_requested and bool(self.held),
                    detour_driving=self.detour_driving,cap_mps=0. if stop else self.speed_mps if limit else None,
                    min_obstacle_speed_mps=self.min_obstacle_speed_mps,
                    eligible_obstacle_ids=sorted(key for key,o in self.latest.items()
                        if math.hypot(o["velocity_x_map"],o["velocity_y_map"])>=self.min_obstacle_speed_mps),
                    waiting_ids=sorted(self.held),zone_or_approach=bool(self.last_active),
                    input_error=self.input_error,
                    clearance_frames={str(key):track['clear_frames'] for key,track in self.held.items()},
                    zone_start_s_m=self.start,zone_end_s_m=self.end,
                    progress_s_m=self.gate.progress_s,
                    clear_confirm_s=self.confirm,
                    unobserved_ids=sorted(key for key in self.held if key not in self.latest),
                    identity_uncertain_ids=sorted(key for key,t in self.held.items() if t['identity_lost']))


def constrain_pedals(command,status,speed):
    if status['stop']:
        command.accel=0.;command.brake=max(command.brake,1.)
    elif status['cap_mps'] is not None:
        error=status['cap_mps']-speed
        command.accel=min(command.accel,max(0.,min(1.,error*.45)))
        command.brake=max(command.brake,max(0.,min(1.,(-error-.05)*.4)))
        if command.brake>0:command.accel=0.
    return command
