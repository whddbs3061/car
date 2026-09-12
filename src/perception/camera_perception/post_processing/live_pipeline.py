#!/usr/bin/env python3
"""새 파이프라인(pipeline/s01~s12)을 **시뮬레이터로 실시간 확인**한다.

    python3 live_pipeline.py                # 12단계까지 (전체)
    python3 live_pipeline.py --stage 6      # 6단계까지만 보고 싶을 때

===========================================================================
이 파일이 pipeline/ 안에 있지 않은 이유
===========================================================================
`pipeline/` 은 프로젝트 내부 import 가 0 이어야 한다 (`_common.py` 머리말).
이 러너는 카메라 수신을 `morai_camera.py` 에서 가져다 쓰므로 그 규칙을 깨게
된다. 그래서 한 단계 바깥에 둔다. `pipeline/` 은 여전히 torch/cv2/numpy 만
바라본다.

`morai_camera.py` 를 복사하지 않은 이유는, 그 파일이 **대체 대상이 아니기
때문**이다. 대체하려는 것은 후처리(`lane_detection.py`)지 UDP 수신이 아니다.

---------------------------------------------------------------------------
키
---------------------------------------------------------------------------
    1 2 3 4 6 9 0    그 단계까지만 실행 (0 = 12단계)
    k                추적 모드 순환  off -> greedy -> hungarian -> kalman
    t                조감 패널 토글
    p                일시정지
    s                지금 프레임을 png 로 저장
    q / ESC          종료

단계를 **실시간으로 갈아 끼울 수 있게** 한 이유는, 어느 단계에서 무너지는지
보려면 같은 장면을 단계만 바꿔 가며 봐야 하기 때문이다. 껐다 켜면 그 사이에
차가 움직여서 다른 장면이 된다.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# **합본을 쓴다.** pipeline/ 모듈이 아니라 real_lane.py 가 정본이다.
import real_lane as rl                                          # noqa: E402
from real_lane import CLASS_GUIDE, CLASS_NAMES, LaneResult      # noqa: E402
try:
    from morai_imu import ImuStream                              # noqa: E402
except SystemExit:
    ImuStream = None
from morai_camera import DEFAULT_IP, DEFAULT_PORT, CameraStream  # noqa: E402

STAGES = (1, 2, 3, 4, 6, 9, 12)
STAGE_NAME = {1: "Segmentation", 2: "Morphology", 3: "Lane pixels",
              4: "Calibration -> ground", 6: "Lane boundary",
              9: "Curve fitting", 12: "Lane ID"}

# 마스크 색은 **클래스**다 (도색과 같은 색을 쓰면 원래 도색 위에서 안 보인다)
MASK_COLORS = {1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 0),
               4: (0, 0, 255), 5: (0, 165, 255)}
# 경계/곡선 색은 **순서**다. 6단계에는 아직 lane_id 가 없어서 등장 순으로 준다
SEQ_COLORS = ((0, 255, 0), (255, 200, 0), (0, 165, 255), (255, 0, 255),
              (255, 255, 0), (128, 0, 255), (200, 200, 200))
ID_COLORS = {0: (255, 255, 255),        # 밟고 있는 선. 흰색으로 눈에 띄게
             1: (0, 255, 0), -1: (255, 200, 0), 2: (0, 165, 255),
             -2: (255, 0, 255), 3: (255, 255, 0), -3: (128, 0, 255)}
GUIDE_COLOR = (0, 165, 255)
# 채택 안 된 유도선. **지우지 않고 흐리게** 남긴다 - 선택이 틀렸을 때
# 무엇 중에서 골랐는지를 볼 수 없으면 원인을 못 찾는다.
GUIDE_DIM = (0, 70, 110)
STOP_COLOR = (0, 0, 255)          # 정지선. 관측이 정면을 덮은 경우
STOP_EXTRAP = (80, 80, 220)       # 옆에서 본 것을 정면까지 외삽한 경우

# 조감 패널 (워프가 아니라 그냥 미터 좌표 산점도다 - BEV 래스터는 이제 없다)
TOP_X_MAX, TOP_Y_ABS, TOP_PX_PER_M = 40.0, 10.0, 11


def run_stages(seg, frame, upto, rng, tracker=None, dt=0.1, guide_link=None,
               ego=None):
    """프레임 하나를 `upto` 단계까지 태운다. -> (LaneResult, 단계별 ms)"""
    r, t = LaneResult(), {}
    r.tracked = None

    t0 = time.perf_counter()
    r.mask, r.crop = seg.apply(frame)
    t["1"] = (time.perf_counter() - t0) * 1e3
    if upto < 2:
        return r, t

    t0 = time.perf_counter()
    r.clean, r.stats["s02"] = rl.morphology(r.mask, seg.bonnet)
    t["2"] = (time.perf_counter() - t0) * 1e3
    if upto < 3:
        return r, t

    t0 = time.perf_counter()
    r.pixels, r.stats["s03"] = rl.lane_pixels(r.clean, occluded=seg.bonnet)
    t["3"] = (time.perf_counter() - t0) * 1e3
    if upto < 4:
        return r, t

    t0 = time.perf_counter()
    r.ground, r.stats["s04"] = rl.to_ground(r.pixels, seg.cam, r.attitude)
    t["4"] = (time.perf_counter() - t0) * 1e3

    # 정지선은 **12단계 번호 밖의 별도 가지**다. s04 만 공유한다.
    r.stopline, r.stats["stop"] = rl.detect_stopline(r.clean, seg.cam, r.attitude)

    if upto < 6:
        return r, t

    t0 = time.perf_counter()
    r.boundaries, r.stats["s06"] = rl.group_boundaries(r.ground)
    t["6"] = (time.perf_counter() - t0) * 1e3
    if upto < 9:
        return r, t

    t0 = time.perf_counter()
    r.curves, r.stats["s09"] = rl.fit_curves(r.boundaries, rng=rng)
    t["9"] = (time.perf_counter() - t0) * 1e3
    if upto < 12:
        return r, t

    # 10~11단계. **추적은 상태를 갖는다** - tracker 를 호출부가 들고 있어야 한다.
    used = r.curves
    if tracker is not None:
        t0 = time.perf_counter()
        used, r.stats["s10"] = tracker.update(
            r.curves, dt=dt, ego=ego, context={"s02": r.stats.get("s02")})
        t["10"] = (time.perf_counter() - t0) * 1e3
        r.tracked = used

    t0 = time.perf_counter()
    # **GuideLink 는 프레임을 넘는 상태다.** 호출부가 들고 있어야 한다
    # (좌측 도색이 보이는 동안 이어지는 유도선을 기억해 두는 구조).
    r.lanes, r.stats["s12"] = rl.assign_lane_ids(used, guide_link=guide_link)
    t["12"] = (time.perf_counter() - t0) * 1e3
    return r, t


# --------------------------------------------------------------------------
# 그리기
# --------------------------------------------------------------------------
def _overlay_mask(vis, mask, alpha=0.45):
    color = np.zeros_like(vis)
    hit = np.zeros(vis.shape[:2], bool)
    for c, bgr in MASK_COLORS.items():
        m = mask == c
        color[m] = bgr
        hit |= m
    vis[hit] = (vis[hit] * (1 - alpha) + color[hit] * alpha).astype(np.uint8)
    return vis


def _draw_curve_on_image(vis, cam, c, color, tag, thickness=3):
    xs = np.arange(c.x_range[0], c.x_range[1] + 1e-6, 0.5)
    pts3 = np.column_stack([xs, np.polyval(c.coef, xs),
                            np.full(xs.size, -0.35)])
    uv, ok = cam.project(pts3)
    uv = uv[ok]
    if len(uv) < 2:
        return
    uv = uv.astype(np.int32)
    cv2.polylines(vis, [uv], False, color, thickness)
    u, v = uv[len(uv) // 2]
    cv2.putText(vis, tag, (int(u) + 6, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3)
    cv2.putText(vis, tag, (int(u) + 6, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)


def draw_image(r, seg, stage):
    """원본 위에 그 단계까지의 결과를 얹는다."""
    vis = r.crop.copy()

    if stage == 1:
        _overlay_mask(vis, r.mask)
    elif stage >= 2:
        _overlay_mask(vis, r.clean if r.clean is not None else r.mask, 0.30)

    if stage == 3 and r.pixels:
        for c, (u, v, w) in r.pixels.items():
            col = MASK_COLORS.get(c, (255, 255, 255))
            for uu, vv in zip(u.astype(int), v.astype(int)):
                cv2.circle(vis, (uu, vv), 2, col, -1)

    if stage == 6:
        for i, b in enumerate(r.boundaries):
            col = SEQ_COLORS[i % len(SEQ_COLORS)]
            uv, ok = seg.cam.project(np.column_stack(
                [b.x, b.y, np.full(b.x.size, -0.35)]))
            for uu, vv in uv[ok].astype(int):
                cv2.circle(vis, (uu, vv), 3, col, -1)

    if stage == 9:
        for i, c in enumerate(r.curves):
            _draw_curve_on_image(vis, seg.cam, c, SEQ_COLORS[i % len(SEQ_COLORS)],
                                 f"{CLASS_NAMES[c.cls][:6]} {c.inlier_ratio:.0%}")

    if stage >= 4 and r.stopline is not None:
        sl = r.stopline
        col = STOP_COLOR if sl.covers_front else STOP_EXTRAP
        # 관측 구간은 굵게, 정면까지 외삽한 부분은 가늘게 - 눈으로 구분되게
        for ys, th in ((np.linspace(sl.y_range[0], sl.y_range[1], 24), 3),
                       (np.linspace(min(sl.y_range[0], 0.0),
                                    max(sl.y_range[1], 0.0), 24), 1)):
            pts3 = np.column_stack([np.polyval(sl.coef, ys), ys,
                                    np.full(ys.size, -0.35)])
            uv, ok = seg.cam.project(pts3)
            uv = uv[ok]
            if len(uv) >= 2:
                cv2.polylines(vis, [uv.astype(np.int32)], False, col, th)
        tag = f"STOP {sl.dist:.1f}m {sl.inlier_ratio:.0%}"
        if not sl.covers_front:
            tag += f" ~{sl.extrap_m:.1f}m"      # 정면까지 외삽한 거리
        if sl.n_blobs > 1:
            tag += f" x{sl.n_blobs}"            # 덩어리 여럿 = 횡단보도 의심
        uvm, okm = seg.cam.project(np.array([[sl.dist, 0.0, -0.35]]))
        if okm[0]:
            u, v = uvm[0].astype(int)
            for c_, t_ in (((0, 0, 0), 3), (col, 1)):
                cv2.putText(vis, tag, (int(u) - 60, int(v) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_, t_)

    if stage == 12:
        shown = r.tracked if r.tracked is not None else r.curves
        # lane_id 0 은 **미할당이 아니라 "밟고 있는 선"** 이다. falsy 라고
        # 건너뛰면 안 된다. 그래서 "번호를 실제로 받은 곡선" 집합을 따로 만든다.
        #
        # **`in` 으로 비교하면 안 된다.** Curve 는 numpy 배열을 담은 dataclass
        # 라 `==` 가 배열을 돌려주고, bool() 에서 ValueError 로 죽는다.
        # 같은 클래스의 곡선이 비교 대상에 걸릴 때만 터져서 잘 안 드러난다.
        assigned = {id(c) for c in r.lanes}
        for c in shown:
            if id(c) not in assigned:
                if c.cls == CLASS_GUIDE:
                    # 채택된 유도선은 lane_id 를 받으므로 여기 오는 것은 전부
                    # 탈락한 것이다. 흐리고 가늘게 남긴다.
                    _draw_curve_on_image(vis, seg.cam, c, GUIDE_DIM, "", 1)
                continue
            col = ID_COLORS.get(c.lane_id, (150, 150, 150))
            tag = f"{c.lane_id:+d} {CLASS_NAMES[c.cls][:6]}"
            if c.from_guide:
                tag += " G"                      # 도색이 아니라 유도선이 채운 것
            if c.confidence:
                tag += f" {c.confidence:.2f}"
            if c.track_id:
                tag += f" #{c.track_id}"
            # 관성(관측 없이 예측만)인 곡선은 가늘게 그려 눈으로 구분되게 한다
            _draw_curve_on_image(vis, seg.cam, c, col, tag, 1 if c.coasted else 3)
    return vis


def draw_top(r, stage):
    """조감 패널. **워프가 아니라 미터 좌표를 그대로 찍은 것**이다."""
    h = int(TOP_X_MAX * TOP_PX_PER_M)
    w = int(2 * TOP_Y_ABS * TOP_PX_PER_M)
    img = np.zeros((h, w, 3), np.uint8)

    def to_px(x, y):
        return (np.round((TOP_Y_ABS - np.asarray(y)) * TOP_PX_PER_M).astype(int),
                np.round((TOP_X_MAX - np.asarray(x)) * TOP_PX_PER_M).astype(int))

    for m in range(0, int(TOP_X_MAX) + 1, 5):
        yy = int((TOP_X_MAX - m) * TOP_PX_PER_M)
        cv2.line(img, (0, yy), (w, yy), (55, 55, 55), 1)
        cv2.putText(img, f"{m}m", (4, yy - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (110, 110, 110), 1)
    cv2.line(img, (w // 2, 0), (w // 2, h), (55, 55, 55), 1)

    if stage >= 4 and r.ground:
        for c, (x, y, _) in r.ground.items():
            cu, cv_ = to_px(x, y)
            col = MASK_COLORS.get(c, (255, 255, 255))
            for a, b in zip(cu, cv_):
                if 0 <= a < w and 0 <= b < h:
                    img[b, a] = col

    if stage == 6:
        for i, b in enumerate(r.boundaries):
            col = SEQ_COLORS[i % len(SEQ_COLORS)]
            cu, cv_ = to_px(b.x, b.y)
            for a, bb in zip(cu, cv_):
                if 0 <= a < w and 0 <= bb < h:
                    cv2.circle(img, (a, bb), 2, col, -1)

    if stage >= 9:
        src = r.tracked if (stage == 12 and r.tracked is not None) else r.curves
        for i, c in enumerate(src):
            xs = np.arange(c.x_range[0], c.x_range[1] + 1e-6, 0.5)
            cu, cv_ = to_px(xs, np.polyval(c.coef, xs))
            pts = np.stack([cu, cv_], 1).astype(np.int32)
            if stage == 12:
                col = (GUIDE_COLOR if c.cls == CLASS_GUIDE
                       else ID_COLORS.get(c.lane_id, (150, 150, 150)))
            else:
                col = SEQ_COLORS[i % len(SEQ_COLORS)]
            cv2.polylines(img, [pts], False, col, 2)

    if stage >= 4 and getattr(r, "stopline", None) is not None:
        sl = r.stopline
        ys = np.linspace(min(sl.y_range[0], 0.0), max(sl.y_range[1], 0.0), 24)
        cu, cv_ = to_px(np.polyval(sl.coef, ys), ys)
        cv2.polylines(img, [np.stack([cu, cv_], 1).astype(np.int32)], False,
                      STOP_COLOR if sl.covers_front else STOP_EXTRAP, 2)

    cv2.circle(img, (w // 2, h - 1), 4, (0, 255, 255), -1)      # 자차
    return img


def stage_line(r, stage):
    """터미널에 한 줄. 그 단계에서 **무엇이 몇 개 나왔는지**만."""
    if stage <= 2:
        s = r.stats.get("s02")
        if not s:
            return f"mask {int((r.mask > 0).sum())} px"
        return "  ".join(f"{CLASS_NAMES[c][:6]} {v['after']}" for c, v in s.items()
                         if v["before"])
    if stage == 3:
        return "  ".join(f"{CLASS_NAMES[c][:6]} {v['pts']}점"
                         for c, v in r.stats["s03"].items() if v["pts"])
    if stage == 4:
        return "  ".join(f"{CLASS_NAMES[c][:6]} {v['kept']}점"
                         for c, v in r.stats["s04"].items() if v["kept"])
    if stage == 6:
        return (f"경계 {len(r.boundaries)}개  " + "  ".join(
            f"{CLASS_NAMES[b.cls][:6]}({b.x.size}점 {b.x_range[1]-b.x_range[0]:.0f}m)"
            for b in r.boundaries))
    if stage == 9:
        return (f"곡선 {len(r.curves)}개  " + "  ".join(
            f"{CLASS_NAMES[c.cls][:6]}({c.inlier_ratio:.0%})" for c in r.curves))
    lanes = [c for c in r.curves if c.lane_id]
    return (rl.format_lane_id_stats(r.stats["s12"])[0] + "   " +
            " ".join(f"{c.lane_id:+d}:{CLASS_NAMES[c.cls][:6]}" for c in lanes))


def main(argv=None):
    ap = argparse.ArgumentParser(description="새 파이프라인 실시간 확인")
    ap.add_argument("--stage", type=int, default=12, choices=STAGES)
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-top", action="store_true", help="조감 패널을 끈다")
    ap.add_argument("--no-imu", dest="imu", action="store_false", default=True,
                    help="IMU 요 변화를 칼만 예측에 안 쓴다")
    ap.add_argument("--track", default="kalman",
                    choices=("off", "greedy", "hungarian", "kalman"),
                    help="추적 모드. 실행 중 k 키로도 바꾼다")
    ap.add_argument("--every", type=int, default=1, help="N 프레임마다 처리")
    ap.add_argument("--save-dir", default=".", help="s 키로 저장할 폴더")
    args = ap.parse_args(argv)

    seg = rl.Segmenter(checkpoint=args.checkpoint, device=args.device)
    print(f"[pipe] {os.path.basename(seg.info['path'])}  epoch {seg.info['epoch']} "
          f"{seg.info['backbone']}  {seg.info['num_classes']}클래스 "
          f"{seg.info['scheme']}  device={seg.device}")
    print(f"[pipe] 단계 {args.stage} = {STAGE_NAME[args.stage]}   "
          f"키: 1 2 3 4 6 9 0(=12) k t p s q")

    rng = np.random.default_rng(0)
    imu = ImuStream().start() if (args.imu and ImuStream is not None) else None
    if imu is not None:
        print(f"[pipe] IMU {'연결' if imu.wait_first(timeout=3.0) else '**안 옴**'}"
              f"  (요 변화를 칼만 예측에)")
    prev_stamp = [0.0]

    TRACK_MODES = ("off", "greedy", "hungarian", "kalman")
    TRACK_CFG = {"greedy":    dict(assoc="greedy",    kalman=False),
                 "hungarian": dict(assoc="hungarian", kalman=False),
                 "kalman":    dict(assoc="hungarian", kalman=True)}

    def make_tracker(mode):
        return None if mode == "off" else rl.Tracker(**TRACK_CFG[mode])

    # 유도선 링크는 track_id 를 쓰므로 추적이 켜져 있을 때만 의미가 있다
    def make_link(mode):
        return None if mode == "off" else rl.GuideLink()

    track_mode = args.track
    tracker = make_tracker(track_mode)
    guide_link = make_link(track_mode)
    print(f"[pipe] 추적 {track_mode}   (k 키로 off/greedy/hungarian/kalman 순환)")

    cam = CameraStream(args.ip, args.port).start()
    print(f"[pipe] {args.ip}:{args.port} 대기 중...")
    if not cam.wait_first(timeout=15.0):
        raise SystemExit("카메라 프레임이 안 옵니다. 시뮬레이터와 IP/포트를 확인하세요.")
    print("[pipe] 수신 시작.")

    stage = args.stage
    show_top = not args.no_top
    top_open = False
    paused = False
    last_seq, n_since = -1, 0
    last_stamp, dt_cam = 0.0, 0.1
    last_ego = None
    r, frame, fps, t_prev = None, None, 0.0, time.time()
    n_saved, t_log = 0, 0.0

    try:
        while True:
            if not paused:
                f, seq, stamp = cam.latest(with_stamp=True)
                if f is not None and seq != last_seq:
                    # **촬영 시각 기준 dt** 를 쓴다. 수신 시각은 지연이 흔들려서
                    # (실측 중앙값 128ms, p90 145ms) 예측 구간이 들쭉날쭉해진다.
                    dt_cam = (stamp - last_stamp) if last_stamp else 0.1
                    last_stamp = stamp
                    last_seq = seq
                    n_since += 1
                    if n_since >= args.every:
                        n_since = 0
                        frame = f
                        ego = None
                        if imu is not None and prev_stamp[0]:
                            dp = imu.delta_yaw(prev_stamp[0], stamp)
                            if dp is not None:
                                ego = (0.0, 0.0, float(dp))
                        prev_stamp[0] = stamp
                        r, tms = run_stages(seg, frame, stage, rng,
                                            tracker=tracker, dt=max(dt_cam, 1e-3),
                                            guide_link=guide_link, ego=ego)
                        last_ego = ego
                        now = time.time()
                        dt = now - t_prev
                        t_prev = now
                        fps = (0.9 * fps + 0.1 / dt) if dt > 0 and fps else \
                            (1.0 / dt if dt > 0 else 0.0)
                        if now - t_log > 1.0:
                            t_log = now
                            j = r.stats.get("s10", {}).get("junction", 0.0)
                            gl = r.stats.get("s12", {}).get("guide_link")
                            print(f"[{stage:2d}] {fps:4.1f}fps "
                                  f"{sum(tms.values()):5.1f}ms "
                                  + (f"J{j:.2f} " if stage == 12 and tracker else "")
                                  + (f"link#{gl} " if gl else "")
                                  + (f"dyaw{np.degrees(last_ego[2]):+5.2f}deg "
                                     if last_ego else "")
                                  + (f"STOP{r.stopline.dist:5.1f}m"
                                     f"{'' if r.stopline.covers_front else '~'} "
                                     if r.stopline is not None else "")
                                  + stage_line(r, stage))

            if r is not None:
                vis = draw_image(r, seg, stage)
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(vis, f"[{stage}] {STAGE_NAME[stage]}"
                            + (f"   track={track_mode}" if stage == 12 else "")
                            + ("   PAUSED" if paused else ""),
                            (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1)
                cv2.putText(vis, f"{fps:.1f} FPS", (vis.shape[1] - 95, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                if args.scale != 1.0:
                    vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
                cv2.imshow("pipeline", vis)

                if show_top and stage >= 4:
                    cv2.imshow("top (m)", draw_top(r, stage))
                    top_open = True
                elif top_open:
                    # 만든 적 없는 창에 getWindowProperty 를 부르면 예외가 난다.
                    # 그래서 플래그로 기억한다.
                    cv2.destroyWindow("top (m)")
                    top_open = False

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k in [ord(c) for c in "123469"]:
                stage = int(chr(k))
                print(f"[pipe] -> 단계 {stage} = {STAGE_NAME[stage]}")
            elif k == ord("0"):
                stage = 12
                print(f"[pipe] -> 단계 12 = {STAGE_NAME[12]}")
            elif k == ord("k"):
                track_mode = TRACK_MODES[(TRACK_MODES.index(track_mode) + 1)
                                         % len(TRACK_MODES)]
                tracker = make_tracker(track_mode)     # 상태를 새로 시작한다
                guide_link = make_link(track_mode)
                print(f"[pipe] 추적 -> {track_mode}")
            elif k == ord("t"):
                show_top = not show_top
            elif k == ord("p"):
                paused = not paused
            elif k == ord("s") and r is not None:
                p = os.path.join(args.save_dir, f"stage{stage}_{n_saved:03d}.png")
                cv2.imwrite(p, draw_image(r, seg, stage))
                print(f"[pipe] 저장 {p}")
                n_saved += 1
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if imu is not None:
            imu.stop()
        cv2.destroyAllWindows()
        print("[pipe] 종료")


if __name__ == "__main__":
    main()
