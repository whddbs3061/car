"""Learn XY distance scale from valid GPS and unscaled inertial displacement.

Ego truth positions are never inputs. A GPS gap freezes the last learned scale;
neither GPS recovery jumps nor the filter's GPS corrections enter training.
"""
import bisect
import math
import statistics
from collections import deque


class DistanceScale:
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.scale = 1.0
        self.ready = False
        self.window_s = 2.0
        self.max_gps_gap_s = .3
        self.minimum_motion_m = 4.0
        self.bounds = (.6, 1.2)
        self.motion = []
        self.pending = deque()
        self.fixes = []
        self.candidates = deque(maxlen=5)
        self.last_gps_stamp = None
        self.last_candidate_stamp = None
        self.last_update_stamp = None
        self.last_learned_stamp = None
        self.reason = 'warming_up' if self.enabled else 'disabled'
        self.rejected = 0

    @property
    def factor(self):
        return self.scale if self.enabled and self.ready else 1.0

    def pause(self, reason):
        """Forget training continuity while retaining the calibrated factor."""
        self.pending.clear()
        self.fixes.clear()
        self.candidates.clear()
        self.last_gps_stamp = None
        self.last_candidate_stamp = None
        self.last_update_stamp = None
        self.reason = reason

    def add_motion(self, stamp, dx, dy, dt, valid=True):
        if not self.enabled:
            return
        if (not valid or not all(math.isfinite(v) for v in (stamp,dx,dy,dt))
                or not 0 < dt <= .25):
            self.pause('motion_unavailable')
            self.motion.clear()
            return
        if self.motion and stamp <= self.motion[-1][0]:
            return
        if self.motion and stamp-self.motion[-1][0] > .25:
            self.pause('motion_gap')
            self.motion.clear()
        if not self.motion:
            self.motion.append((stamp-dt,0.,0.,0.))
        prev=self.motion[-1]
        self.motion.append((stamp,prev[1]+dx,prev[2]+dy,prev[3]+math.hypot(dx,dy)))
        while len(self.motion)>2 and self.motion[1][0]<stamp-8.:
            self.motion.pop(0)
        if self.last_gps_stamp is not None and stamp-self.last_gps_stamp>self.max_gps_gap_s:
            self.pause('gps_gap_frozen' if self.ready else 'gps_gap_uncalibrated')
        while self.pending and self.pending[0][0]<=stamp:
            self._evaluate(*self.pending.popleft())

    def observe_gps(self, stamp, x, y):
        if not self.enabled:
            return
        if not all(math.isfinite(v) for v in (stamp,x,y)):
            self.pause('invalid_gps')
            return
        if self.last_gps_stamp is not None:
            dt=stamp-self.last_gps_stamp
            if dt<=0:
                return
            if dt>self.max_gps_gap_s:
                self.pause('gps_gap_frozen' if self.ready else 'gps_gap_uncalibrated')
        self.last_gps_stamp=stamp
        self.pending.append((stamp,x,y))
        if len(self.pending)>50:
            self.pause('motion_unavailable')
            return
        while self.pending and self.motion and self.pending[0][0]<=self.motion[-1][0]:
            self._evaluate(*self.pending.popleft())

    def _motion_at(self, stamp):
        if not self.motion or not self.motion[0][0]<=stamp<=self.motion[-1][0]:
            return None
        i=bisect.bisect_left([p[0] for p in self.motion],stamp)
        if i==0:
            return self.motion[0][1:]
        a,b=self.motion[i-1],self.motion[i]
        f=(stamp-a[0])/(b[0]-a[0])
        return tuple(a[k]+f*(b[k]-a[k]) for k in (1,2,3))

    def _evaluate(self, stamp, x, y):
        raw=self._motion_at(stamp)
        if raw is None:
            return
        fix=(stamp,x,y,*raw)
        self.fixes.append(fix)
        self.fixes=[p for p in self.fixes if p[0]>=stamp-self.window_s-.4]
        eligible=[p for p in self.fixes if p[0]<=stamp-self.window_s]
        if not eligible or (self.last_candidate_stamp is not None and stamp-self.last_candidate_stamp<.5):
            return
        self.last_candidate_stamp=stamp
        base=eligible[-1]
        gx,gy=x-base[1],y-base[2]
        rx,ry=raw[0]-base[3],raw[1]-base[4]
        raw_length=math.hypot(rx,ry)
        arc=raw[2]-base[5]
        if raw_length<self.minimum_motion_m or arc<=0 or raw_length/arc<.97:
            self.reason='insufficient_straight_motion'
            return
        gps_length=math.hypot(gx,gy)
        ratio=gps_length/raw_length
        alignment=(gx*rx+gy*ry)/max(gps_length*raw_length,1e-12)
        if not self.bounds[0]<=ratio<=self.bounds[1] or alignment<math.cos(math.radians(10.)):
            self.rejected+=1
            self.pause('gps_window_rejected')
            return
        self.candidates.append(ratio)
        if len(self.candidates)<3:
            self.reason='confirming_scale'
            return
        if max(self.candidates)-min(self.candidates)>.05:
            self.reason='unstable_scale'
            return
        target=statistics.median(self.candidates)
        if self.ready and self.last_update_stamp is not None:
            step=.03*max(0.,stamp-self.last_update_stamp)
            target=max(self.scale-step,min(self.scale+step,target))
        self.scale=target
        self.ready=True
        self.last_update_stamp=stamp
        self.last_learned_stamp=stamp
        self.reason='gps_scale_learned'

    def status(self):
        return dict(distance_scale_version='GPS_DISTANCE_SCALE_V1',
                    distance_scale_enabled=self.enabled,
                    distance_scale_ready=self.ready,
                    distance_scale=self.factor,
                    distance_scale_reason=self.reason,
                    distance_scale_samples=len(self.candidates),
                    distance_scale_last_learned_stamp=self.last_learned_stamp,
                    distance_scale_rejected=self.rejected)
