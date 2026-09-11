"""차선 검출 결과를 그림으로 그린다 — **실주행 경로는 이 파일을 import 하지 않는다.**

    offline_test.py   녹화 -> png / mp4
    live_overlay.py   실시간 -> 화면

live_output.py (실주행에서 제어로 값만 보내는 경로) 는 이 파일을 쓰지 않는다.
자율주행에 필요 없는 것(그리기, 저장)을 검출 코드에서 떼어 놓은 이유는,
실시간 루프에 시각화가 끼면 프레임을 놓치고 그러면 "화면에서 본 지연"이 실제
지연과 달라져 판단이 흐려지기 때문이다.

--------------------------------------------------------------------------
색이 두 가지 뜻으로 쓰인다는 점에 주의
--------------------------------------------------------------------------
    반투명 마스크  = 모델 출력 그대로. 색은 **클래스**
    굵은 곡선      = 후처리 결과. 색은 **차선 ID**
지금은 두 팔레트가 색을 공유한다 (초록이 마스크에서는 황색선, 선에서는 -1).
읽기 어려우면 MASK_VIEW_COLORS 를 무채색으로 바꾸면 색이 차선 ID 하나만
뜻하게 된다 - 클래스는 선 옆 글자에 이미 나온다.
"""

import cv2
import numpy as np

from lane_detection import (BEV_H, BEV_W, BEV_X_MAX, CLASS_GUIDE, CLASS_NAMES,
                            CLASS_STOPLINE, CLASS_WHITE_DASHED, CLASS_WHITE_SOLID,
                            CLASS_YELLOW, ROAD_Z_EGO, ego_to_bev)

ID_COLORS = {-1: (0, 255, 0), 1: (255, 200, 0), -2: (0, 165, 255), 2: (255, 0, 255),
             -3: (255, 255, 0), 3: (128, 0, 255)}
# 유도선은 차선이 아니라 lane_id 가 없다. **따라가기로 고른 것만 밝게** 칠하고
# 나머지는 어둡게 둔다 - 교차로에는 방향별 유도선이 여럿 보여서, 어느 것을
# 쓰기로 했는지가 화면에서 바로 보여야 한다.
GUIDE_COLOR = (0, 165, 255)
GUIDE_COLOR_DIM = (0, 90, 140)
# **도색과 같은 색을 쓰면 안 된다.** 백색 실선을 흰색으로 칠하면 원래 흰 도색
# 위에서 구분이 안 된다 (실제로 정지선만 보였다).
MASK_VIEW_COLORS = {CLASS_WHITE_SOLID: (255, 0, 255), CLASS_WHITE_DASHED: (255, 255, 0),
                    CLASS_GUIDE: (0, 165, 255),
                    CLASS_YELLOW: (0, 255, 0), CLASS_STOPLINE: (0, 0, 255)}
MASK_VIEW_LABEL = ("solid=magenta  dashed=cyan  yellow=green  stop=red  "
                   "guide=orange")


def _lane_color(l):
    return ID_COLORS.get(l.lane_id, (150, 150, 150))


def _draw_curve(vis, detector, l, color, tag, thickness=3):
    """자차 좌표 곡선 하나를 원본 영상에 투영해 그린다."""
    pts = l.sample()
    pts3 = np.column_stack([pts, np.full(len(pts), ROAD_Z_EGO)])
    uv, valid = detector.cam.project(pts3)
    uv = uv[valid]
    inb = (uv[:, 0] > -2000) & (uv[:, 0] < vis.shape[1] + 2000)
    uv = uv[inb].astype(np.int32)
    if len(uv) < 2:
        return
    cv2.polylines(vis, [uv], False, color, thickness)
    u, v = uv[len(uv) // 2]
    cv2.putText(vis, tag, (int(u) + 6, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3)
    cv2.putText(vis, tag, (int(u) + 6, int(v)), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1)


def draw(det, frame_bgr, detector, alpha=0.35,
         show_mask=True, show_lanes=True):
    """오버레이 한 장. show_mask / show_lanes 는 live_overlay.py 의
    m / l 키 토글용이다 - 껐을 때도 상단 HUD 는 그대로 남긴다."""
    frame = frame_bgr[detector.crop_top:] if frame_bgr.shape[0] > det.mask.shape[0] \
        else frame_bgr
    vis = frame.copy()

    if show_mask:
        color = np.zeros_like(vis); hit = np.zeros(vis.shape[:2], bool)
        for c, bgr in MASK_VIEW_COLORS.items():
            m = det.mask == c
            color[m] = bgr; hit |= m
        vis[hit] = (vis[hit] * (1 - alpha) + color[hit] * alpha).astype(np.uint8)

    for l in (det.lanes if show_lanes else ()):
        _draw_curve(vis, detector, l, _lane_color(l), f"{l.lane_id:+d} {l.name[:6]}")

    picked = det.guide
    for g in (det.guides if show_lanes else ()):
        on = picked is not None and g is picked
        _draw_curve(vis, detector, g, GUIDE_COLOR if on else GUIDE_COLOR_DIM,
                    "guide*" if on else "guide", 3 if on else 2)

    txt = "  ".join(f"{l.lane_id:+d}:{l.name[:6]}({l.n_points})" for l in det.lanes) \
        or "(검출 없음)"
    if det.guides:
        txt += f"   guide x{len(det.guides)}"
    # **주행선이 어디서 나왔는지를 같이 띄운다.** 유도선으로 넘어간 프레임을
    # 화면에서 못 가리면, 교차로에서 이상하게 도는 원인을 나중에 못 찾는다.
    src = det.center_source()
    txt += f"   center={src or '-'}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(vis, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GUIDE_COLOR if src == "guide" else (255, 255, 255), 1)
    cv2.putText(vis, MASK_VIEW_LABEL + f"    {det.infer_ms:.0f}+{det.post_ms:.0f}ms",
                (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return vis


def draw_bev(det, scale=0.5):
    if det.bev is None:
        return None
    img = np.zeros((BEV_H, BEV_W, 3), np.uint8)
    for c, bgr in MASK_VIEW_COLORS.items():
        img[det.bev == c] = bgr
    def poly(l, color, thickness):
        pts = l.sample()
        r, c = ego_to_bev(pts[:, 0], pts[:, 1])
        cv2.polylines(img, [np.stack([c, r], axis=1).astype(np.int32)],
                      False, color, thickness)

    for l in det.lanes:
        poly(l, _lane_color(l), 2)
    picked = det.guide
    for g in det.guides:
        on = picked is not None and g is picked
        poly(g, GUIDE_COLOR if on else GUIDE_COLOR_DIM, 2 if on else 1)
    # 유도선을 쓰기로 한 프레임에서는 **실제로 따라갈 선**도 그린다. 유도선과
    # 주행선이 GUIDE_OFFSET_M 만큼 떨어져 있는지 눈으로 확인하는 용도다.
    if picked is not None and det.center_source() == "guide":
        xs = np.arange(max(picked.x_range[0] - 4.0, 0.0),
                       picked.x_range[1] + 1e-6, 0.5)
        cs = [(x, det.guide_center(x)) for x in xs]
        cs = [(x, y) for x, y in cs if y is not None]
        if len(cs) >= 2:
            r, c = ego_to_bev([x for x, _ in cs], [y for _, y in cs])
            cv2.polylines(img, [np.stack([c, r], axis=1).astype(np.int32)],
                          False, (255, 255, 255), 2)
    for x in range(0, int(BEV_X_MAX) + 1, 5):
        r = int(ego_to_bev(x, 0)[0])
        cv2.line(img, (0, r), (BEV_W, r), (60, 60, 60), 1)
        cv2.putText(img, f"{x}m", (4, r - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (120, 120, 120), 1)
    cv2.circle(img, (int(ego_to_bev(0, 0)[1]), int(ego_to_bev(0, 0)[0])), 5,
               (0, 255, 255), -1)
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def draw_mask_only(det, frame_bgr, crop_top, alpha=0.8, dilate=1):
    """**후처리 없이 모델 출력만** 본다. 위=원본, 아래=세그멘테이션 오버레이."""
    frame = frame_bgr[crop_top:] if frame_bgr.shape[0] > det.mask.shape[0] else frame_bgr
    vis = frame.copy()
    color = np.zeros_like(vis); hit = np.zeros(vis.shape[:2], bool)
    k = np.ones((2 * dilate + 1,) * 2, np.uint8)
    for c, bgr in MASK_VIEW_COLORS.items():
        m = det.mask == c
        if dilate and m.any():
            m = cv2.dilate(m.astype(np.uint8), k) > 0
        color[m] = bgr; hit |= m
    vis[hit] = (vis[hit] * (1 - alpha) + color[hit] * alpha).astype(np.uint8)

    n = np.bincount(det.mask.ravel(), minlength=len(CLASS_NAMES))
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (0, 0, 0), -1)
    cv2.putText(vis, "  ".join(f"{CLASS_NAMES[c][:9]} {n[c]}px"
                               for c in range(1, len(CLASS_NAMES))),
                (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, MASK_VIEW_LABEL, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1)
    top = frame.copy()
    cv2.rectangle(top, (0, 0), (top.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(top, "original", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    return np.vstack([top, vis])

