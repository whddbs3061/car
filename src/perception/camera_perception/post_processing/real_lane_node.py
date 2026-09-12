#!/usr/bin/env python3
"""`real_lane.py` 를 돌려서 제어가 쓸 값을 내보낸다.

    roslaunch 로:   rosrun ... real_lane_node.py _publish_boundaries:=true
    ROS 없이:       python3 real_lane_node.py --no-ros            (표준출력)
                    python3 real_lane_node.py --no-ros --udp 127.0.0.1:7600

===========================================================================
기존 노드를 그대로 대체한다 - JSON 계약이 같다
===========================================================================
통합 런처(`morai_avoidance_highway_roundabout_final.launch`)는 이렇게 물려 있다.

    lane_info_runner.py  ->  live_lane_info_publisher_v2.py   (옛 BEV 경로)
                             /perception/camera/lane_info  (std_msgs/String, JSON)
                                      |
                             lane_info_semantic_adapter.py
                                      |
                             dashed_lane_detected / left_solid_lane_detected /
                             left_yellow_solid_lane_detected /
                             right_solid_lane_detected /
                             stopline_detected / stopline_distance_m

어댑터가 **실제로 읽는 키는 5개뿐**이다.

    left_lane.detected   left_lane.type   left_lane.dashed
    stopline_detected    stopline_distance_m        (right_lane 도 같은 셋)

이 노드는 그 키를 포함해 v2 가 내보내던 구조를 그대로 낸다. **런처도 어댑터도
회피 로직도 고칠 필요가 없다** - `lane_info_runner.py` 가 가리키는 스크립트만
이것으로 바꾸면 된다.

---------------------------------------------------------------------------
좌우 부호가 뒤집혔다는 점만 주의한다
---------------------------------------------------------------------------
옛 `lane_detection.py` 는 **왼쪽이 음수**(-1)였고, 새 파이프라인은 자차 좌표계
y 부호를 따라 **왼쪽이 +1** 이다. JSON 의 `left_lane` / `right_lane` 은 둘 다
"내 왼쪽/오른쪽 경계" 라는 뜻이므로 여기서 맞춰 담는다.

    left_lane   <-  lane_id == +1
    right_lane  <-  lane_id == -1

새로 생긴 `lane_id == 0` 은 **지금 밟고 있는 선**이다 (차선 변경 중에만 나온다).
계약에는 없는 값이라 `straddling_lane` 으로 따로 담는다 - 기존 소비자는 무시하고,
필요한 쪽만 읽으면 된다.

===========================================================================
필요한 것만 켜서 받는다
===========================================================================
계약 키는 **항상** 나간다. 나머지는 파라미터로 켠다. 기본값을 보수적으로 둔
이유는 크기 때문이다 (300프레임 실측, 프레임당):

    최종 차선 + 정지선 + 곡선 계수     ~3.5 KB      기본 ON
    경계 점열 (6단계)                  ~9 KB        기본 OFF
    레인 픽셀 지면점 (3단계)           ~9 KB        기본 OFF
    2단계 마스크를 점으로              **183 KB**   지원하지 않음

마지막 것은 12Hz 에서 2.24 MB/s 라 JSON 토픽으로 낼 수 없다. 마스크가 필요하면
`~publish_mask_image` 로 이미지 토픽을 쓴다 (자차 좌표가 필요하면 받는 쪽에서
`real_lane.unproject()` 를 부르면 된다 - 같은 파일에 들어 있다).

**차선을 장애물처럼 쓰려면 마스크가 아니라 `left_boundary_points` /
`right_boundary_points` 를 쓴다.** 0.5m 간격으로 샘플링한 경계 폴리라인이라
계획기가 그대로 벽으로 쓸 수 있고, 경계당 60점 남짓이라 가볍다.

===========================================================================
스스로 디버깅할 수 있게
===========================================================================
`~publish_diag` (기본 ON) 이면 `<topic>_diag` 로 단계별 진단이 나간다.

    단계마다 무엇이 몇 개 나왔는지, 몇 ms 걸렸는지
    왜 안 나왔는지 (`픽셀 부족` / `정면 미포함` / `연속성 통과 없음` ...)
    교차로 점수, 트랙별 신뢰도, 유도선 링크 상태

`~stage` 로 N단계까지만 돌릴 수도 있다. 어디서 무너지는지 격리할 때 쓴다.
"""

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import real_lane as rl                                          # noqa: E402
from morai_camera import DEFAULT_IP, DEFAULT_PORT, CameraStream  # noqa: E402

FRAME_ID = "base_link"
DEFAULT_TOPIC = "/perception/camera/lane_info"

# 경계/중심선 점열을 샘플링할 간격과 상한. 계획기가 폴리라인으로 쓰기 좋은 값이다.
POINT_STEP_M = 0.5
POINT_MAX = 80


# --------------------------------------------------------------------------
# JSON 조립
# --------------------------------------------------------------------------
def _lane_meta(c):
    """차선 하나를 계약 형식으로. 없으면 detected=False."""
    if c is None:
        return {"detected": False, "type": None, "dashed": None,
                "track_id": None, "age": 0, "coef": None,
                "x_range_m": None, "n_points": 0, "confidence": 0.0,
                "from_guide": False, "coasted": False}
    return {
        "detected": True,
        "type": rl.CLASS_NAMES[c.cls],
        "dashed": bool(c.cls == rl.CLASS_WHITE_DASHED),
        "track_id": int(c.track_id),
        "age": int(c.age),
        "coef": [round(float(v), 8) for v in c.coef],
        "x_range_m": [round(float(c.x_range[0]), 3), round(float(c.x_range[1]), 3)],
        "n_points": int(c.x.size),
        # 새 파이프라인이 추가로 주는 것들. 기존 소비자는 무시한다.
        "confidence": round(float(c.confidence), 3),
        "inlier_ratio": round(float(c.inlier_ratio), 3),
        # **도색이 아니라 유도선이 이 자리를 채웠다는 뜻.** 출력의 left/right 는
        # "여기까지 비켜도 된다" 는 의미인데 유도선은 넘으면 안 되는 선이 아니라
        # 지나갈 길 힌트다. 회피 계획이 벽으로 오해하면 안 된다.
        "from_guide": bool(c.from_guide),
        # 이 프레임에 관측이 없어 예측만으로 낸 값
        "coasted": bool(c.coasted),
    }


def _sample(c, step=POINT_STEP_M, cap=POINT_MAX):
    """곡선을 자차 좌표 점열로. [[x, y], ...]"""
    if c is None:
        return None
    lo, hi = c.x_range
    n = min(int((hi - lo) / step) + 1, cap)
    xs = np.linspace(lo, hi, max(n, 2))
    ys = np.polyval(c.coef, xs)
    return [[round(float(a), 3), round(float(b), 3)] for a, b in zip(xs, ys)]


def build_payload(res, opt, timing):
    """LaneResult -> 계약 JSON (+ 선택 블록)."""
    lanes = res.lanes or []
    left = next((c for c in lanes if c.lane_id == 1), None)
    right = next((c for c in lanes if c.lane_id == -1), None)
    straddle = next((c for c in lanes if c.lane_id == 0), None)
    sl = res.stopline

    lp, rp = _sample(left), _sample(right)
    center = None
    width = None
    lat = head = None
    if lp and rp:
        n = min(len(lp), len(rp))
        center = [[lp[i][0], round((lp[i][1] + rp[i][1]) / 2.0, 3)] for i in range(n)]
        width = round(abs(left.y_at(rl.ORDER_X_M) - right.y_at(rl.ORDER_X_M)), 3)
    if center and len(center) >= 2:
        # 횡오차는 가장 가까운 중심점의 y, 방위오차는 중심선 기울기
        lat = round(-float(center[0][1]), 3)
        dx = center[-1][0] - center[0][0]
        head = round(float(np.arctan2(center[-1][1] - center[0][1], dx)), 4) \
            if dx > 1e-6 else None

    valid = left is not None or right is not None
    reasons = []
    if left is None:
        reasons.append("NO_LEFT")
    if right is None:
        reasons.append("NO_RIGHT")
    if left is not None and left.from_guide:
        reasons.append("LEFT_FROM_GUIDE")

    out = {
        "timestamp": time.time(),
        "frame_id": FRAME_ID,
        "coordinate_convention": {"x": "forward_m", "y": "left_m"},

        "lane_valid": bool(valid),
        "output_status": "FRESH" if valid else "INVALID",
        "lane_state": ("both" if (left is not None and right is not None)
                       else "left" if left is not None
                       else "right" if right is not None else None),
        "reasons": reasons,

        "left_lane": _lane_meta(left),
        "right_lane": _lane_meta(right),
        # 계약에 없는 추가 값. 차선 변경 중 밟고 있는 선이다.
        "straddling_lane": _lane_meta(straddle) if straddle is not None else None,

        "lane_width_m": width,
        "left_boundary_points": lp,
        "right_boundary_points": rp,
        "centerline_points": center,

        "lateral_error_m": lat,
        "heading_error_rad": head,

        "stopline_detected": sl is not None,
        "stopline_distance_m": None if sl is None else round(float(sl.dist), 3),

        "n_lanes": len(lanes),
        "infer_ms": round(timing.get("1", 0.0), 1),
        "post_ms": round(sum(v for k, v in timing.items() if k != "1"), 1),
    }

    if sl is not None:
        out["stopline"] = {
            "distance_m": round(float(sl.dist), 3),
            "coef": [round(float(v), 6) for v in sl.coef],
            "y_range_m": [round(float(sl.y_range[0]), 3),
                          round(float(sl.y_range[1]), 3)],
            "inlier_ratio": round(float(sl.inlier_ratio), 3),
            "covers_front": bool(sl.covers_front),
            "n_blobs": int(sl.n_blobs),
        }

    # --- 선택 블록 -------------------------------------------------------
    if opt.get("curves"):
        out["curves"] = [{
            "cls": rl.CLASS_NAMES[c.cls], "lane_id": int(c.lane_id),
            "track_id": int(c.track_id),
            "coef": [round(float(v), 8) for v in c.coef],
            "x_range_m": [round(float(c.x_range[0]), 3),
                          round(float(c.x_range[1]), 3)],
            "inlier_ratio": round(float(c.inlier_ratio), 3),
            "confidence": round(float(c.confidence), 3),
        } for c in (res.curves or [])]

    if opt.get("boundaries"):
        out["boundaries"] = [{
            "cls": rl.CLASS_NAMES[b.cls],
            "points": [[round(float(x), 3), round(float(y), 3)]
                       for x, y in zip(b.x, b.y)],
        } for b in (res.boundaries or [])]

    if opt.get("lane_pixels"):
        # **3단계는 이미지 좌표라 그대로 내보내면 규격 위반이다.** 4단계를 태운
        # 자차 좌표(`res.ground`)를 내보낸다.
        out["lane_points"] = {
            rl.CLASS_NAMES[c]: [[round(float(x), 3), round(float(y), 3)]
                                for x, y in zip(v[0], v[1])]
            for c, v in (res.ground or {}).items()}

    return out


def build_diag(res, timing, tracker):
    """왜 그렇게 나왔는지. 화면 없이 원인을 좁히라고 내보낸다."""
    st = res.stats
    d = {"timestamp": time.time(),
         "ms": {k: round(v, 1) for k, v in timing.items()},
         "counts": {
             "boundaries": len(res.boundaries or []),
             "curves": len(res.curves or []),
             "lanes": len(res.lanes or []),
         }}
    if "s04" in st:
        d["ground_kept"] = {rl.CLASS_NAMES[c]: v["kept"] for c, v in st["s04"].items()}
    if "s06" in st:
        d["boundary"] = {rl.CLASS_NAMES[c]: {k: v[k] for k in ("seeds", "grown",
                                                               "short", "kept")}
                         for c, v in st["s06"].items()}
    if "s09" in st:
        d["fit"] = {rl.CLASS_NAMES[c]: {k: v[k] for k in ("failed", "curv", "fitted")}
                    for c, v in st["s09"].items()}
    if "s10" in st:
        d["track"] = st["s10"]
    if "s12" in st:
        d["lane_id"] = {k: st["s12"][k] for k in
                        ("order_x", "off_axis", "straddling", "left", "right",
                         "ego_left", "ego_right", "left_from_guide",
                         "guide_reject", "guide_link") if k in st["s12"]}
    if "stop" in st:
        d["stopline"] = st["stop"]
    if tracker is not None:
        d["tracks"] = [{"id": t.id, "cls": rl.CLASS_NAMES[t.cls],
                        "conf": round(float(t.conf), 3), "miss": t.misses,
                        "age": t.age} for t in tracker.tracks]
    return d


# --------------------------------------------------------------------------
# 실행
# --------------------------------------------------------------------------
class Runner:
    """카메라 -> real_lane -> JSON. ROS 유무와 무관하게 같은 코드를 쓴다."""

    def __init__(self, args):
        self.args = args
        self.seg = rl.Segmenter(checkpoint=args.checkpoint, cam_set=args.cam_set,
                                device=(None if args.device in (None, "auto")
                                        else args.device))
        self.tracker = None if args.no_track else rl.Tracker()
        self.link = None if args.no_track else rl.GuideLink()
        self.rng = np.random.default_rng(0)
        self.opt = {"curves": args.publish_curves,
                    "boundaries": args.publish_boundaries,
                    "lane_pixels": args.publish_lane_pixels}
        self.last_stamp = 0.0

    def process(self, frame, stamp):
        r, t = rl.LaneResult(), {}
        dt = (stamp - self.last_stamp) if self.last_stamp else 0.1
        self.last_stamp = stamp
        dt = float(np.clip(dt, 1e-3, 1.0))

        t0 = time.perf_counter()
        r.mask, r.crop = self.seg.apply(frame)
        t["1"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        r.clean, r.stats["s02"] = rl.morphology(r.mask, self.seg.bonnet)
        t["2"] = (time.perf_counter() - t0) * 1e3
        if self.args.stage < 3:
            return r, t

        t0 = time.perf_counter()
        r.pixels, r.stats["s03"] = rl.lane_pixels(r.clean, occluded=self.seg.bonnet)
        t["3"] = (time.perf_counter() - t0) * 1e3
        if self.args.stage < 4:
            return r, t

        t0 = time.perf_counter()
        r.ground, r.stats["s04"] = rl.to_ground(r.pixels, self.seg.cam, r.attitude)
        r.stopline, r.stats["stop"] = rl.detect_stopline(r.clean, self.seg.cam)
        t["4"] = (time.perf_counter() - t0) * 1e3
        if self.args.stage < 6:
            return r, t

        t0 = time.perf_counter()
        r.boundaries, r.stats["s06"] = rl.group_boundaries(r.ground)
        t["6"] = (time.perf_counter() - t0) * 1e3
        if self.args.stage < 9:
            return r, t

        t0 = time.perf_counter()
        r.curves, r.stats["s09"] = rl.fit_curves(r.boundaries, rng=self.rng)
        t["9"] = (time.perf_counter() - t0) * 1e3
        if self.args.stage < 12:
            return r, t

        used = r.curves
        if self.tracker is not None:
            t0 = time.perf_counter()
            used, r.stats["s10"] = self.tracker.update(
                r.curves, dt=dt, context={"s02": r.stats.get("s02")})
            t["10"] = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        r.lanes, r.stats["s12"] = rl.assign_lane_ids(used, guide_link=self.link)
        t["12"] = (time.perf_counter() - t0) * 1e3
        return r, t


def main(argv=None):
    ap = argparse.ArgumentParser(description="real_lane 퍼블리셔")
    ap.add_argument("--no-ros", action="store_true", help="ROS 없이 표준출력/UDP 로")
    ap.add_argument("--udp", default=None, help="ip:port 로도 보낸다 (--no-ros 용)")
    ap.add_argument("--quiet", action="store_true", help="표준출력을 끈다")
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cam-set", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--stage", type=int, default=12, help="N단계까지만 (디버깅)")
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--publish-curves", action="store_true", default=True)
    ap.add_argument("--publish-boundaries", action="store_true")
    ap.add_argument("--publish-lane-pixels", action="store_true")
    ap.add_argument("--publish-diag", action="store_true", default=True)
    args = ap.parse_args(argv)

    pub = pub_diag = None
    use_ros = not args.no_ros
    if use_ros:
        import rospy
        from std_msgs.msg import String
        rospy.init_node("real_lane_node", anonymous=False)
        # ROS 파라미터가 있으면 그쪽이 이긴다 (roslaunch 에서 넘기는 값)
        for name, attr, cast in (("~ip", "ip", str), ("~port", "port", int),
                                 ("~checkpoint", "checkpoint", str),
                                 ("~cam_set", "cam_set", str),
                                 ("~device", "device", str),
                                 ("~topic", "topic", str),
                                 ("~every", "every", int),
                                 ("~stage", "stage", int),
                                 ("~no_track", "no_track", bool),
                                 ("~publish_curves", "publish_curves", bool),
                                 ("~publish_boundaries", "publish_boundaries", bool),
                                 ("~publish_lane_pixels", "publish_lane_pixels", bool),
                                 ("~publish_diag", "publish_diag", bool)):
            if rospy.has_param(name):
                setattr(args, attr, cast(rospy.get_param(name)))
        pub = rospy.Publisher(args.topic, String, queue_size=1)
        if args.publish_diag:
            pub_diag = rospy.Publisher(args.topic + "_diag", String, queue_size=1)

    sock = None
    if args.udp:
        host, _, port = args.udp.partition(":")
        sock = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))

    runner = Runner(args)
    info = runner.seg.info
    print(f"[real_lane] {os.path.basename(info['path'])}  epoch {info['epoch']} "
          f"{info['backbone']}  {info['num_classes']}클래스 {info['scheme']}  "
          f"device={runner.seg.device}")
    print(f"[real_lane] 단계 {args.stage}  추적 {'끔' if args.no_track else '켬'}  "
          f"선택출력 curves={args.publish_curves} boundaries={args.publish_boundaries} "
          f"lane_pixels={args.publish_lane_pixels}")

    cam = CameraStream(args.ip, args.port).start()
    print(f"[real_lane] {args.ip}:{args.port} 대기 중...")
    if not cam.wait_first(timeout=15.0):
        raise SystemExit("카메라 프레임이 안 옵니다. 시뮬레이터와 IP/포트를 확인하세요.")
    print("[real_lane] 수신 시작.")

    last_seq, n_since, n = -1, 0, 0
    t_log = time.time()
    try:
        while True:
            if use_ros:
                import rospy
                if rospy.is_shutdown():
                    break
            frame, seq, stamp = cam.latest(with_stamp=True)
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            n_since += 1
            if n_since < args.every:
                continue
            n_since = 0

            res, timing = runner.process(frame, stamp)
            payload = build_payload(res, runner.opt, timing)
            text = json.dumps(payload, ensure_ascii=False)
            n += 1

            if pub is not None:
                from std_msgs.msg import String
                pub.publish(String(data=text))
            if sock is not None:
                sock[0].sendto(text.encode("utf-8"), sock[1])
            if not args.quiet and not use_ros:
                print(text, flush=True)

            if args.publish_diag:
                diag = json.dumps(build_diag(res, timing, runner.tracker),
                                  ensure_ascii=False)
                if pub_diag is not None:
                    from std_msgs.msg import String
                    pub_diag.publish(String(data=diag))

            if time.time() - t_log > 2.0:
                t_log = time.time()
                ll = payload["left_lane"]["detected"]
                rr = payload["right_lane"]["detected"]
                print(f"[real_lane] {n}프레임  좌{'O' if ll else 'X'} "
                      f"우{'O' if rr else 'X'}  "
                      f"정지선 {payload['stopline_distance_m']}  "
                      f"{payload['infer_ms']}+{payload['post_ms']}ms")
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        print("[real_lane] 종료")


if __name__ == "__main__":
    main()
