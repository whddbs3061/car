#!/usr/bin/env python3
"""후처리 단계를 **한 장씩 사진으로** 떨어뜨린다. 창도 재생도 없다.

===========================================================================
실행 (그대로 복사해서 붙여넣기)
===========================================================================
cd C:/MSC/AutoMobility/car/src/perception/camera_perception/post_processing
C:/Users/user/anaconda3/envs/vision_env/python.exe -u StageShots.py

  일부만       --frames 000042,000071,000313
  다른 녹화     --recording ../recordings/lap4_full --until 000471

기본값은 `recordings/last_test` 의 **사진 전부**다. 그 폴더는 png 가 하위폴더
없이 바로 들어 있고 meta.jsonl 도 없다 - lap4_full 프레임과 바이트 단위로
같은 선별본이라, 차량 피치는 `--meta` 로 lap4_full 것을 가져다 쓴다.
===========================================================================

파이프라인 - **BEV 래스터를 쓰지 않는다.**

    1. Segmentation              best.pt 원본 출력 (6클래스)
    2. Morphology                보닛 제거 + CLOSE + 작은 성분 제거
    3. Lane Pixel Extraction     행별 런 중심점 -> 얇은 점열
    4. Calibration -> ground     픽셀을 지면 평면으로 역투영, 자차 좌표 (m)
    5. RANSAC / Poly fitting     자차 좌표에서 2차식 적합

Kalman / Hungarian / Lane ID 는 여기서 하지 않는다 - 프레임 간 상태가 필요한
단계라 정지 사진으로 검증할 수 있는 것이 없다. 실시간 시뮬에서 붙인다.

---------------------------------------------------------------------------
왜 BEV 래스터를 빼는가
---------------------------------------------------------------------------
`lane_detection.py` 는 마스크 전체를 800x400 조감도로 워프한 뒤 그 위에서
찾는다. 픽셀을 지면으로 바로 쏘면 세 가지가 없어진다.

  - 워프 보간에서 얇은 차선이 깨지는 문제 (먼 쪽은 원래 1~2px 다)
  - 0.05m/px 격자에 갇히는 문제 - 점열은 실수 좌표 그대로 남는다
  - 쓰지도 않는 배경까지 32만 픽셀을 채우고 버리는 낭비

대신 **지면이 평평하다는 가정**이 더 직접적으로 드러난다. 오르막·내리막에서
먼 쪽 거리가 틀어지는 것은 두 방식 모두 같지만, 이쪽은 그게 어디서 오는지가
`ground_project.py` 한 곳에 모여 있다.

---------------------------------------------------------------------------
행별 런 중심점을 쓰는 이유 (3단계)
---------------------------------------------------------------------------
마스크 픽셀을 그대로 쓰면 **근거리가 적합을 지배한다.** 같은 차선이 6m 에서
20px 두께, 30m 에서 1px 이라 가까운 쪽 점이 수십 배 많다. 2차식은 점이 많은
쪽에 맞춰지고 먼 쪽이 휜다.

행마다 런(가로로 이어진 픽셀 덩어리)의 중심 하나만 남기면 차선 하나가 행당
점 하나가 되어 거리에 따른 가중이 사라진다. 런이 `EXTRACT_MAX_RUN_PX` 보다
두꺼우면 그 행에서는 차선이 아니라 화살표·정지선 같은 면으로 보고 버린다.
"""

import argparse
import copy
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import lane_detection as LD
from ground_project import unproject_to_ground

# ==========================================================================
# 튜닝 지점 - 여기 숫자만 바꿔 가며 사진을 다시 뽑으면 된다
# ==========================================================================

# --- 2. Morphology --------------------------------------------------------
# 커널이 크면 안 된다. 먼 쪽 차선은 두께가 1~2px 라 OPEN 을 걸면 통째로 사라진다.
# CLOSE 만 작게 걸어 세그멘테이션이 만든 톱니와 1px 구멍을 메운다.
# 점선의 대시 사이 간격(수십 px)은 여기서 메우지 않는다 - 그건 적합이 할 일이고,
# 여기서 이으면 점선과 실선을 구분할 근거가 사라진다.
MORPH_CLOSE_K = (3, 3)
MORPH_MIN_BLOB = 40         # 이보다 작은 연결 성분은 노이즈로 버린다 (px)

# --- 3. Lane Pixel Extraction --------------------------------------------
# 차선을 뽑을 클래스. 정지선(4)은 진행방향과 직각이라 y=f(x) 로 표현할 수 없어
# 여기서 빼고 따로 다룬다.
EXTRACT_CLASSES = (LD.CLASS_WHITE_SOLID, LD.CLASS_WHITE_DASHED,
                   LD.CLASS_YELLOW, LD.CLASS_GUIDE)
EXTRACT_MAX_RUN_PX = 90     # 한 행에서 이보다 두꺼운 런은 차선이 아니다

# --- 4. Calibration -> ground --------------------------------------------
GROUND_X_MIN, GROUND_X_MAX = 3.0, 40.0      # 이 밖은 쓰지 않는다 (m)
GROUND_Y_ABS = 10.0                          # |y| 가 이보다 크면 버린다 (m)

# **카메라 자체(초점거리·주점·장착 pitch 2.0도)는 cam_set.json 의 고정값이고
# 여기서 건드리지 않는다.** 움직이는 것은 차체 자세다 - 도로 경사·뱅크·
# 서스펜션 때문에 정지 중에도 0 이 아니고(GenerateLabels 실측 pitch +0.68도),
# lap4_full 은 pitch -0.86~+1.62도, roll -1.9도까지 나온다.
#
# 지면 교점 거리는 d ~ h/theta 라 delta_d ~ -d^2*delta/h 로 터진다. 실측
# (차로폭 3.30m 기준, 자세를 안 넣었을 때):
#
#        차량 피치      6m     10m     15m     20m     25m     30m
#          -0.8도     3.18    3.08    2.95    2.84    2.73    2.63
#          +0.8도     3.43    3.56    3.74    3.94    4.17    4.42
#
# 10m 까지는 ±0.26m(8%) 라 넘어갈 만하지만 25m 에서는 ±0.87m(26%) 다.
#
# 규약은 `GenerateLabels` 것을 그대로 쓴다 (EGO_PITCH_SIGN / EGO_ROLL_SIGN).
# 학습 라벨이 그 규약으로 만들어졌으니 추론이 다른 규약을 쓰면 어긋난다.
# 거기 실측표에도 roll/pitch 를 넣은 것이 차로폭 3.38 vs 3.36m 로 가장 좋다.
ATTITUDE_SOURCE_DEFAULT = "meta"    # meta / none

# --- 5. RANSAC / Poly fitting --------------------------------------------
# 시드: 가장 가까운 점에서 SEED_X_SPAN 안의 y 히스토그램에서 봉우리를 찾는다.
SEED_X_SPAN = 4.0           # 시드를 찾을 전방 구간 (m)
SEED_BIN_M = 0.25           # y 히스토그램 칸 크기 (m)
SEED_MIN_GAP_M = 1.5        # 서로 다른 차선으로 볼 최소 간격 (차로 폭 3.3m)
SEED_MIN_COUNT = 4          # 봉우리에 점이 이만큼은 있어야 시드로 본다
SEED_MAX_COUNT = 6

# 시드에서 전방으로 걸어 올라가며 점을 모은다 (BEV 슬라이딩 윈도우의 점열 판)
GROW_STEP_M = 1.0           # 한 걸음 (m)
GROW_HALF_M = 0.9           # 걸음마다 y 로 이만큼 안의 점을 가져간다
GROW_MAX_MISS = 5           # 연속으로 이만큼 비면 그 차선은 끝
GROW_DRIFT_GAIN = 0.5       # 커브 추종 - 직전 이동량을 얼마나 이어받을지

FIT_MAX_SPAN_M = 22.0       # 2차식 하나로 덮을 구간 (lane_detection 과 같은 값)
FIT_MIN_POINTS = 8
FIT_MIN_SPAN_M = 3.0

# ==========================================================================

CLASS_COLORS = {
    LD.CLASS_WHITE_SOLID: (255, 0, 255),      # magenta
    LD.CLASS_WHITE_DASHED: (255, 255, 0),     # cyan
    LD.CLASS_YELLOW: (0, 255, 0),             # green
    LD.CLASS_STOPLINE: (0, 0, 255),           # red
    LD.CLASS_GUIDE: (0, 165, 255),            # orange
}
CLASS_SHORT = {LD.CLASS_WHITE_SOLID: "solid", LD.CLASS_WHITE_DASHED: "dashed",
               LD.CLASS_YELLOW: "yellow", LD.CLASS_STOPLINE: "stop",
               LD.CLASS_GUIDE: "guide"}
LEGEND = "solid=magenta dashed=cyan yellow=green stop=red guide=orange"


# ==========================================================================
# 1~3 단계
# ==========================================================================

def overlay(frame, mask, alpha=0.75, dilate=1):
    """마스크를 원본 위에 색으로 얹는다. 얇은 선은 부풀려야 보인다."""
    vis = frame.copy()
    color = np.zeros_like(vis)
    hit = np.zeros(vis.shape[:2], bool)
    k = np.ones((2 * dilate + 1,) * 2, np.uint8)
    for c, bgr in CLASS_COLORS.items():
        m = mask == c
        if dilate and m.any():
            m = cv2.dilate(m.astype(np.uint8), k) > 0
        color[m] = bgr
        hit |= m
    vis[hit] = (vis[hit] * (1 - alpha) + color[hit] * alpha).astype(np.uint8)
    return vis


def morphology(mask, bonnet):
    """2단계. 보닛 제거 -> 클래스별 CLOSE -> 작은 성분 제거.

    클래스마다 따로 도는 이유는 1단계와 같다 - 황색 중앙선과 백색 실선이
    붙어 있어도 다른 차선이라, 한 통에 넣고 모폴로지를 걸면 둘이 이어진다.
    """
    out = mask.copy()
    if bonnet is not None:
        out[bonnet] = LD.CLASS_BG
    k = np.ones(MORPH_CLOSE_K, np.uint8)
    for c in CLASS_COLORS:
        m = (out == c).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            keep[i] = stats[i, cv2.CC_STAT_AREA] >= MORPH_MIN_BLOB
        m = keep[lab]
        out[(out == c) & ~m] = LD.CLASS_BG      # 원래 이 클래스였는데 버려진 것
        out[m & (out == LD.CLASS_BG)] = c       # CLOSE 로 새로 채워진 것
    return out


def extract_pixels(mask):
    """3단계. 클래스별로 행마다 런 중심점을 뽑는다.

    돌려주는 것: {cls: (u, v)}  - 둘 다 float 배열 (부분 픽셀 중심)
    """
    out = {}
    h, w = mask.shape
    for c in EXTRACT_CLASSES:
        m = mask == c
        if not m.any():
            continue
        us, vs = [], []
        rows = np.flatnonzero(m.any(axis=1))
        for r in rows:
            cols = np.flatnonzero(m[r])
            # 끊긴 곳에서 쪼갠다 -> 한 행에 차선이 여럿이면 각각 중심을 낸다
            for run in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1):
                if run.size == 0 or run.size > EXTRACT_MAX_RUN_PX:
                    continue
                us.append(run.mean())
                vs.append(float(r))
        if us:
            out[c] = (np.asarray(us), np.asarray(vs))
    return out


# ==========================================================================
# 4 단계 - Calibration -> vehicle/ground coordinate
# ==========================================================================

def to_ground(cam, pts, attitude=None):
    """{cls: (u,v)} -> {cls: (x,y)} 자차 좌표 (m). 통계도 같이 돌려준다."""
    out, stat = {}, {}
    for c, (u, v) in pts.items():
        uv = np.stack([u, v], axis=1)
        xy, valid = unproject_to_ground(cam, uv, LD.ROAD_Z_EGO, attitude)
        x, y = xy[:, 0], xy[:, 1]
        keep = valid & (x >= GROUND_X_MIN) & (x <= GROUND_X_MAX) \
            & (np.abs(y) <= GROUND_Y_ABS)
        stat[c] = {"in": int(len(u)), "above_horizon": int((~valid).sum()),
                   "out_of_range": int((valid & ~keep).sum()), "kept": int(keep.sum())}
        if keep.any():
            out[c] = (x[keep], y[keep])
    return out, stat


# ==========================================================================
# 5 단계 - RANSAC / Polynomial fitting
# ==========================================================================

def find_seeds(x, y):
    """가장 가까운 구간의 y 히스토그램에서 차선 시작 위치를 찾는다."""
    near = x <= x.min() + SEED_X_SPAN
    if near.sum() < SEED_MIN_COUNT:
        return []
    yy = y[near]
    edges = np.arange(-GROUND_Y_ABS, GROUND_Y_ABS + SEED_BIN_M, SEED_BIN_M)
    hist, _ = np.histogram(yy, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    seeds, work = [], hist.astype(float).copy()
    gap = int(round(SEED_MIN_GAP_M / SEED_BIN_M))
    for _ in range(SEED_MAX_COUNT):
        i = int(work.argmax())
        if work[i] < SEED_MIN_COUNT:
            break
        seeds.append(float(centers[i]))
        work[max(0, i - gap):i + gap + 1] = 0
    return sorted(seeds)


def grow(x, y, y0):
    """시드에서 전방으로 걸어 올라가며 점을 모은다. 인덱스 배열을 돌려준다."""
    order = np.argsort(x)
    sx, sy = x[order], y[order]
    cur, drift, miss = float(y0), 0.0, 0
    picked = []
    lo = sx.min()
    while lo <= sx.max():
        hi = lo + GROW_STEP_M
        band = np.flatnonzero((sx >= lo) & (sx < hi))
        if band.size:
            pred = cur + drift
            take = band[np.abs(sy[band] - pred) <= GROW_HALF_M]
        else:
            take = np.empty(0, int)
        if take.size:
            new = float(sy[take].mean())
            drift = (1 - GROW_DRIFT_GAIN) * drift + GROW_DRIFT_GAIN * (new - cur)
            cur, miss = new, 0
            picked.append(take)
        else:
            miss += 1
            cur += drift            # 빈 구간에서도 추세는 이어 간다 (점선)
            if miss > GROW_MAX_MISS:
                break
        lo = hi
    if not picked:
        return np.empty(0, int)
    return order[np.concatenate(picked)]


def fit_lanes(ground, rng):
    """{cls: (x,y)} -> [ {cls, coef, x_range, x, y, inlier} ]"""
    out = []
    for c, (x, y) in ground.items():
        for y0 in find_seeds(x, y):
            idx = grow(x, y, y0)
            if idx.size < FIT_MIN_POINTS:
                continue
            gx, gy = x[idx], y[idx]
            near = gx <= gx.min() + FIT_MAX_SPAN_M     # 2차식이 감당할 구간만
            gx, gy = gx[near], gy[near]
            if gx.size < FIT_MIN_POINTS or (gx.max() - gx.min()) < FIT_MIN_SPAN_M:
                continue
            # RANSAC 은 lane_detection 의 것을 그대로 쓴다 - 표본을 x 구간별로
            # 뽑는 부분이 여기서도 그대로 필요하다
            fit = LD.ransac_fit(gx, gy, rng)
            if fit is None:
                continue
            coef, inl = fit
            out.append({"cls": c, "coef": coef, "x": gx, "y": gy, "inlier": inl,
                        "x_range": (float(gx.min()), float(gx.max()))})
    return out


def lane_width_report(lanes):
    """자차 좌우 차선의 **수직** 간격을 거리별로 잰다. 캘리브레이션 성적표다.

    같은 x 에서 y 차이를 그냥 빼면 안 된다 - 커브에서는 두 곡선이 비스듬히
    잘려 폭이 과대평가된다. 접선 기울기로 cos 를 곱해 수직 거리로 고친다.

    읽는 법: 거리에 따라 **벌어지거나 좁아지면 피치**가 틀린 것이고, 어디서나
    일정하게 3.3m 를 벗어나면 **카메라 높이**가 틀린 것이다. 실측(lap4_full):
    차량 피치 +0.79도 프레임에서 보정 없이 6m 3.29m -> 25m 3.84m 로 벌어졌고,
    피치를 넣으면 3.16~3.35m 로 평평해졌다.
    """
    rows = []
    for xq in (6.0, 10.0, 15.0, 20.0, 25.0):
        cand = [(float(np.polyval(l["coef"], xq)), l) for l in lanes
                if l["x_range"][0] - 1.0 <= xq <= l["x_range"][1] + 1.0]
        left = [t for t in cand if t[0] > 0]
        right = [t for t in cand if t[0] < 0]
        if not left or not right:
            continue
        yl, ll = min(left, key=lambda t: t[0])       # 자차에 가장 가까운 좌측
        yr, _ = max(right, key=lambda t: t[0])       # 자차에 가장 가까운 우측
        slope = float(np.polyval(np.polyder(ll["coef"]), xq))
        rows.append((xq, (yl - yr) * math.cos(math.atan(slope))))
    return rows


# ==========================================================================
# 그림
# ==========================================================================

def label_bar(width, text, sub="", h=30):
    bar = np.zeros((h, width, 3), np.uint8)
    cv2.putText(bar, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, sub, (width - tw - 10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (150, 200, 150), 1, cv2.LINE_AA)
    return bar


def panel(img, width, title, sub=""):
    s = width / img.shape[1]
    im = cv2.resize(img, (width, int(round(img.shape[0] * s))),
                    interpolation=cv2.INTER_AREA)
    return np.vstack([label_bar(width, title, sub), im])


def draw_points(frame, pts):
    """3단계 - 뽑은 점을 원본 위에 찍는다."""
    vis = (frame * 0.35).astype(np.uint8)
    for c, (u, v) in pts.items():
        col = CLASS_COLORS[c]
        for uu, vv in zip(u, v):
            cv2.circle(vis, (int(round(uu)), int(round(vv))), 1, col, -1)
    return vis


class GroundPlot:
    """자차 좌표 평면. **조감도 이미지가 아니라 미터 좌표 산점도다.**"""

    PPM = 18            # px per meter

    def __init__(self):
        self.w = int((2 * GROUND_Y_ABS) * self.PPM)
        self.h = int((GROUND_X_MAX - 0.0) * self.PPM)
        self.img = np.zeros((self.h, self.w, 3), np.uint8)
        for xm in range(0, int(GROUND_X_MAX) + 1, 5):
            r = self._r(xm)
            cv2.line(self.img, (0, r), (self.w, r), (48, 48, 48), 1)
            cv2.putText(self.img, f"{xm}m", (4, r - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (110, 110, 110), 1, cv2.LINE_AA)
        for ym in range(-int(GROUND_Y_ABS), int(GROUND_Y_ABS) + 1, 2):
            c = self._c(ym)
            cv2.line(self.img, (c, 0), (c, self.h), (40, 40, 40), 1)
        # 자차 차로 폭 기준선 (+-1.65m) - 폭이 맞는지 눈으로 재는 자
        for ym in (-LD.LANE_WIDTH_M / 2, LD.LANE_WIDTH_M / 2):
            c = self._c(ym)
            cv2.line(self.img, (c, 0), (c, self.h), (70, 70, 0), 1)
        cv2.circle(self.img, (self._c(0), self._r(0)), 5, (0, 255, 255), -1)

    def _r(self, x):
        return int(round(self.h - x * self.PPM))

    def _c(self, y):
        return int(round(self.w / 2 - y * self.PPM))

    def scatter(self, x, y, color, radius=1):
        for xx, yy in zip(np.atleast_1d(x), np.atleast_1d(y)):
            r, c = self._r(xx), self._c(yy)
            if 0 <= r < self.h and 0 <= c < self.w:
                cv2.circle(self.img, (c, r), radius, color, -1)

    def curve(self, coef, x_range, color, thickness=2):
        xs = np.linspace(x_range[0], x_range[1], 60)
        ys = np.polyval(coef, xs)
        pts = np.array([[self._c(y), self._r(x)] for x, y in zip(xs, ys)], np.int32)
        cv2.polylines(self.img, [pts], False, color, thickness, cv2.LINE_AA)


def text_block(width, lines, h):
    img = np.zeros((h, width, 3), np.uint8)
    for i, t in enumerate(lines[:(h - 8) // 18]):
        cv2.putText(img, t, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (200, 200, 200), 1, cv2.LINE_AA)
    return img


# ==========================================================================

def process(det, frame, rng, attitude=None):
    """한 프레임 -> (합성 사진, 요약 dict)"""
    # 1. Segmentation - 보닛 제거 전의 모델 출력 그대로.
    #    `infer_mask` 가 보닛을 안에서 지우므로 잠깐 떼었다가 되돌린다.
    #    복사본을 만들지 않는 이유는 추론 경로가 갈라지면 안 되기 때문이다.
    saved = det.bonnet
    det.bonnet = None
    raw = det.infer_mask(frame)
    det.bonnet = saved
    crop = frame[det.crop_top:]

    n_raw = np.bincount(raw.ravel(), minlength=6)

    # 2. Morphology
    clean = morphology(raw, det.bonnet)
    n_clean = np.bincount(clean.ravel(), minlength=6)

    # 3. Lane Pixel Extraction
    pts = extract_pixels(clean)

    # 4. Calibration -> vehicle/ground coordinate
    ground, gstat = to_ground(det.cam, pts, attitude)

    # 5. RANSAC / Polynomial fitting
    lanes = fit_lanes(ground, rng)

    # --- 그림 -------------------------------------------------------------
    W = 760
    p1 = panel(overlay(crop, raw), W, "1. Segmentation (raw model output)",
               "  ".join(f"{CLASS_SHORT[c]} {n_raw[c]}" for c in CLASS_COLORS))
    p2 = panel(overlay(crop, clean), W, "2. Morphology (bonnet + close + blob)",
               "  ".join(f"{CLASS_SHORT[c]} {n_clean[c]}" for c in CLASS_COLORS))
    p3 = panel(draw_points(crop, pts), W, "3. Lane Pixel Extraction (row-run centers)",
               "  ".join(f"{CLASS_SHORT[c]} {len(pts[c][0])}pt"
                         for c in EXTRACT_CLASSES if c in pts) or "no points")
    left = np.vstack([p1, p2, p3])

    g4 = GroundPlot()
    for c, (x, y) in ground.items():
        g4.scatter(x, y, CLASS_COLORS[c])
    g5 = GroundPlot()
    for c, (x, y) in ground.items():
        g5.scatter(x, y, tuple(int(v * 0.32) for v in CLASS_COLORS[c]))
    for l in lanes:
        col = CLASS_COLORS[l["cls"]]
        g5.scatter(l["x"][l["inlier"]], l["y"][l["inlier"]], col)
        g5.curve(l["coef"], l["x_range"], (255, 255, 255), 2)

    widths = lane_width_report(lanes)
    wtxt = "  ".join(f"{x:.0f}m:{w:.2f}" for x, w in widths) or "n/a"
    att = "level" if attitude is None else         f"pitch {attitude[0]:+.2f} roll {attitude[1]:+.2f}"
    q4 = panel(g4.img, 430, "4. Calibration -> ground (m)",
               f"{att}  kept {sum(v['kept'] for v in gstat.values())}pt")
    q5 = panel(g5.img, 430, "5. RANSAC / Poly fit",
               f"{len(lanes)} lanes")
    right = np.hstack([q4, q5])

    h = max(left.shape[0], right.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
    body = np.hstack([pad(left), pad(right)])

    info = []
    for l in lanes:
        c = l["coef"]
        info.append(f"{CLASS_SHORT[l['cls']]:6s} y={c[0]:+.5f}x^2{c[1]:+.4f}x{c[2]:+.3f}"
                    f"  x {l['x_range'][0]:.1f}~{l['x_range'][1]:.1f}m"
                    f"  pts {l['x'].size} inlier {int(l['inlier'].sum())}"
                    f" ({l['inlier'].mean()*100:.0f}%)")
    # **읽는 법: 절대값이 아니라 거리에 따른 변화(drift)를 본다.** 넓은 구간이나
    # 옆 차로 짝을 잡으면 절대값은 3.3m 가 아닐 수 있지만, 기하가 맞으면 거리와
    # 무관하게 평평하다. 자세가 틀리면 거리에 비례해 벌어지거나 좁아진다.
    drift = ""
    if len(widths) >= 2:
        d = (widths[-1][1] - widths[0][1]) / max(widths[0][1], 1e-6) * 100
        drift = (f"   drift {d:+.0f}% over {widths[0][0]:.0f}-{widths[-1][0]:.0f}m"
                 f"  ({'flat = geometry OK' if abs(d) < 8 else 'NOT flat'})")
    info.append(f"lane width @ x  {wtxt}{drift}")
    info.append(f"vehicle attitude: {att}   (mount pitch/focal are fixed "
                f"cam_set values, never tuned here)")
    ah = sum(v["above_horizon"] for v in gstat.values())
    oo = sum(v["out_of_range"] for v in gstat.values())
    info.append(f"dropped: above-horizon {ah}pt, out-of-range {oo}pt   {LEGEND}")
    foot = text_block(body.shape[1], info, 18 * len(info) + 14)

    summary = {
        "attitude_deg": None if attitude is None else
                        [float(attitude[0]), float(attitude[1])],
        "lanes": [{"cls": CLASS_SHORT[l["cls"]],
                   "coef": [float(v) for v in l["coef"]],
                   "x_range": l["x_range"], "n": int(l["x"].size),
                   "inlier_ratio": float(l["inlier"].mean())} for l in lanes],
        "lane_width": [[float(a), float(b)] for a, b in widths],
        "px": {CLASS_SHORT[c]: int(n_clean[c]) for c in CLASS_COLORS},
        "ground_stat": {CLASS_SHORT[c]: v for c, v in gstat.items()},
    }
    return np.vstack([body, foot]), summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="후처리 단계를 사진으로 떨어뜨린다")
    ap.add_argument("--recording", default=os.path.join(_HERE, "..", "recordings",
                                                        "last_test"))
    ap.add_argument("--frames", default=None,
                    help="프레임 번호 (쉼표 구분). 안 주면 폴더의 사진 전부")
    ap.add_argument("--until", default=None, help="이 번호까지만")
    ap.add_argument("--meta", default=None,
                    help="차량 피치를 읽을 meta.jsonl. 안 주면 녹화 폴더 안의 것")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--attitude", default=ATTITUDE_SOURCE_DEFAULT,
                    choices=["meta", "none"],
                    help="차체 자세(roll/pitch) 보정. meta 면 meta.jsonl 에서 "
                         "프레임마다 읽는다")
    args = ap.parse_args(argv)

    # png 가 frames/ 안에 있는 녹화(lap4_full)와 폴더에 바로 있는 것(last_test)
    # 둘 다 받는다.
    paths = sorted(glob.glob(os.path.join(args.recording, "frames", "*.png")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.recording, "*.png")))
    if not paths:
        raise SystemExit(f"png 를 못 찾았습니다: {args.recording}")
    if args.frames:
        want = {t.strip() for t in args.frames.split(",") if t.strip()}
        paths = [p for p in paths if os.path.basename(p)[:-4] in want]
        if not paths:
            raise SystemExit(f"그 프레임을 못 찾았습니다: {sorted(want)}")
    if args.until:
        paths = [p for p in paths if os.path.basename(p)[:-4] <= args.until]

    det = LD.LaneDetector(args.checkpoint, device=args.device, track=False)
    print(f"[stage] epoch {det.ckpt_info['epoch']} ({det.ckpt_info['backbone']}, "
          f"{det.ckpt_info['num_classes']}클래스) device={det.device}")
    print(f"[stage] 보닛 {det.bonnet_source}  카메라 {det.cam.width}x{det.cam.height} "
          f"fx={det.cam.fx:.0f} cy={det.cam.cy:.0f} 지면 z={LD.ROAD_Z_EGO}m")
    print(f"[stage] {len(paths)}장")

    # 차체 자세. meta.jsonl 의 idx 는 파일명 숫자와 같다.
    att = {}
    if args.attitude == "meta":
        mp = args.meta or os.path.join(args.recording, "meta.jsonl")
        if os.path.isfile(mp):
            with open(mp, encoding="utf-8") as fp:
                for line in fp:
                    d = json.loads(line)
                    att[int(d["idx"])] = (float(d.get("pitch", 0.0)),
                                          float(d.get("roll", 0.0)))
            print(f"[stage] 차체 자세(roll/pitch)를 읽는다: {mp} ({len(att)}프레임)")
        else:
            print(f"[stage] meta.jsonl 이 없어 자세 보정을 끈다: {mp}")
            print("[stage]   -> 먼 쪽 차로폭이 최대 26% 틀어질 수 있다. "
                  "--meta 로 지정하면 보정된다.")

    out_dir = args.out or os.path.join(args.recording, "stage_shots")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    allsum = {}
    for p in paths:
        idx = os.path.basename(p)[:-4]
        frame = cv2.imread(p)
        if frame is None:
            continue
        a = att.get(int(idx)) if args.attitude == "meta" else None
        img, s = process(det, frame, rng, attitude=a)
        cv2.imwrite(os.path.join(out_dir, f"stage_{idx}.png"), img)
        allsum[idx] = s
        w = "  ".join(f"{x:.0f}m {v:.2f}m" for x, v in s["lane_width"]) or "-"
        atxt = "수평" if a is None else f"p{a[0]:+.2f} r{a[1]:+.2f}"
        print(f"[{idx}] {atxt}  차선 {len(s['lanes'])}개  차로폭 {w}")
        for l in s["lanes"]:
            print(f"        {l['cls']:6s} x {l['x_range'][0]:.1f}~{l['x_range'][1]:.1f}m"
                  f"  점 {l['n']}  인라이어 {l['inlier_ratio']*100:.0f}%")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fp:
        json.dump(allsum, fp, ensure_ascii=False, indent=1)
    print(f"\n{len(allsum)}장 저장: {out_dir}")


if __name__ == "__main__":
    main()
