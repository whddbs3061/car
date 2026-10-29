"""Route-progress permission for avoidance; independent of ROS and lookahead.

One forward lap per node lifetime. Restart the node after resetting the scene.
Landmarks snap to the nearest global waypoint in map XY. Regions are half-open:
crossing the start enables planning, crossing the end disables it.
"""

import bisect
import hashlib
import math
import struct


def route_fingerprint(points):
    digest = hashlib.sha256()
    for point in points:
        digest.update(struct.pack("<ddd", point.x, point.y, point.z))
    return digest.hexdigest()


class RouteAvoidanceGate:
    def __init__(self, points, regions, expected_fingerprint="",
                 max_match_distance_m=12.0, search_ahead_m=60.0,
                 search_behind_m=20.0, max_landmark_distance_m=3.0,
                 gps_blackout_override=True):
        self.gps_blackout_override = bool(gps_blackout_override)
        self.points = list(points)
        if len(self.points) < 2 or any(
                not all(math.isfinite(v) for v in (p.x, p.y, p.z))
                for p in self.points):
            raise ValueError("Avoidance gate requires a finite global route")
        if expected_fingerprint and route_fingerprint(self.points) != expected_fingerprint:
            raise ValueError("Avoidance region route does not match the calibrated global path")
        limits = (max_match_distance_m, search_ahead_m, search_behind_m,
                  max_landmark_distance_m)
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in limits):
            raise ValueError("Avoidance gate distances must be finite and positive")
        self.max_match_distance = float(max_match_distance_m)
        self.search_ahead = float(search_ahead_m)
        self.search_behind = float(search_behind_m)
        self.s = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.s.append(self.s[-1] + math.hypot(b.x-a.x, b.y-a.y))
        if self.s[-1] <= 0.0:
            raise ValueError("Avoidance route must have nonzero length")
        self.regions = []
        names = set()
        for region in regions:
            name = region["name"]
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Avoidance region names must be unique nonempty strings")
            names.add(name)
            mapped = {"name": name}
            for boundary in ("start", "end"):
                xyz = region[boundary + "_xyz"]
                if len(xyz) != 3 or not all(math.isfinite(float(v)) for v in xyz):
                    raise ValueError("Avoidance landmarks must contain finite [x,y,z]")
                index = self._nearest(float(xyz[0]), float(xyz[1]), 0, len(self.points))
                p = self.points[index]
                distance = math.hypot(p.x-float(xyz[0]), p.y-float(xyz[1]))
                if distance > max_landmark_distance_m:
                    raise ValueError("Avoidance landmark too far from route: " + name)
                mapped[boundary + "_index"] = index
                mapped[boundary + "_s_m"] = self.s[index]
                mapped[boundary + "_distance_xy_m"] = distance
            if mapped["start_s_m"] >= mapped["end_s_m"]:
                raise ValueError("Avoidance start must precede end: " + name)
            self.regions.append(mapped)
        if not self.regions:
            raise ValueError("At least one avoidance region is required")
        self.progress_s = None

    def _nearest(self, x, y, first, last):
        return min(range(first, last), key=lambda i:
                   (self.points[i].x-x)**2 + (self.points[i].y-y)**2)

    def update(self, x, y, gps_blackout=False):
        """Observe the current vehicle position, never the Pure Pursuit target."""
        first, last = 0, len(self.points)
        if self.progress_s is not None:
            first = max(0, bisect.bisect_left(self.s, self.progress_s-self.search_behind)-1)
            last = min(len(self.points), bisect.bisect_right(
                self.s, self.progress_s+self.search_ahead)+1)
        nearest = None
        distance = None
        matched = False
        if math.isfinite(x) and math.isfinite(y):
            nearest = self._nearest(x, y, first, last)
            # Project onto the segments beside the nearest waypoint so the gate
            # changes at the waypoint itself, not halfway to the previous point.
            candidates = []
            for i in range(max(first, nearest-1), min(last-1, nearest+1)):
                a, b = self.points[i], self.points[i+1]
                dx, dy = b.x-a.x, b.y-a.y
                length2 = dx*dx + dy*dy
                if length2 <= 1e-12:
                    continue
                ratio = max(0.0, min(1.0, ((x-a.x)*dx+(y-a.y)*dy)/length2))
                gap = math.hypot(x-a.x-ratio*dx, y-a.y-ratio*dy)
                candidates.append((gap, self.s[i]+ratio*math.sqrt(length2)))
            if candidates:
                distance, observed_s = min(candidates)
                point = self.points[nearest]
                if math.hypot(point.x-x, point.y-y) <= 1e-8:
                    observed_s = self.s[nearest]
                matched = distance <= self.max_match_distance
                if matched:
                    self.progress_s = (observed_s if self.progress_s is None
                                       else max(self.progress_s, observed_s))
        active_regions = [r["name"] for r in self.regions
                          if matched and r["start_s_m"] <= self.progress_s < r["end_s_m"]]
        override = bool(self.gps_blackout_override and gps_blackout)
        allowed = bool(active_regions or override)
        return {
            "allowed": allowed,
            "regions": active_regions,
            "gps_blackout": bool(gps_blackout),
            "gps_override": override,
            "route_matched": matched,
            "nearest_waypoint_index": nearest,
            "progress_waypoint_index": (None if self.progress_s is None else
                                        bisect.bisect_right(self.s, self.progress_s)-1),
            "progress_s_m": self.progress_s,
            "route_distance_m": distance,
            "reason": ("gps_blackout_override" if override else
                       "+".join(active_regions) if active_regions else
                       "outside_avoidance_regions" if matched else "route_unmatched"),
        }
