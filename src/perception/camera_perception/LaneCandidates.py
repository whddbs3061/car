"""차선 후보 추출까지만 하는 베이스 (후처리는 여기서부터 새로 붙인다).

LaneEgoSelect.py 에서 검증된 앞단만 떼어냈다:

    프레임 → combine_masks → limit_region(ROI) → lane_candidates

ego lane 선택·경로 생성·정지선 판정·결과 그리기는 **들어 있지 않다.**
그 뒷단을 새로 설계하려고 만든 파일이므로, 후보를 어떻게 쓸지는
`lane_candidates()` 가 돌려주는 리스트를 받아서 직접 작성한다.

화면은 3패널이다: 원본 | ROI 마스크 | 후보

--------------------------------------------------------------------------
lane_candidates() 가 돌려주는 후보 dict
--------------------------------------------------------------------------
    xs, ys      이 차선에 속한 픽셀 좌표 (np.ndarray, 같은 길이)
    fit         직선 피팅 {slope(dx/dy), x0, y0, angle}. _x_at(fit, y) 로 평가한다.
    x_bottom    화면 하단(y=h-1)까지 **직선으로** 외삽한 x.
                차선 간 좌우 순서를 가장 잘 구분하는 값이다.
                다만 조각이 짧으면 크게 튄다 — 연관·판정에 쓰지 말 것 (아래 7번).
    angle       기울기 (deg). dx/dy 의 arctan 이라 수직선이 0 에 가깝다.
    dashed      점선이면 True. 트랙 이력이 쌓이면 시간축 판정, 아니면 fill_ratio
                (아래 8번)
    dash_source "track" 이면 시간축 판정, "fill" 이면 한 프레임 fill_ratio 폴백
    occupancy   앵커 행 점유율 (트랙 판정일 때만, 아니면 None). 실선은 ~1.0
    track_id    프레임 간 같은 차선에 붙는 ID. 추적이 끊기면 새 번호가 나간다
    coasted     True 면 **이 프레임에 검출된 게 아니라** 트랙 예측으로 채운 것.
                xs/ys 는 직선에서 생성한 좌표라 실제 픽셀이 아니다 (아래 9번)
    y_span      세로로 뻗은 길이 (px)
    n_pixels    픽셀 수 (coasted 는 0)
리스트는 x_bottom 오름차순으로 정렬되어 있다.

--------------------------------------------------------------------------
이 앞단에서 이미 해결해 둔 함정들 (되돌리지 말 것)
--------------------------------------------------------------------------
1. ONNX 출력이 두 개다. net.forward() 는 첫 번째("da")만 준다.
   차선("ll")을 얻으려면 이름을 지정해 받아야 한다.
2. 출력은 softmax 이전 logit 이다. `> 0.5` 는 의미가 없고 채널 argmax 로 이진화한다.
3. 차선은 폭이 1~3px 이라 5x5 MORPH_OPEN 을 걸면 통째로 지워진다. 3x3 CLOSE 만 쓴다.
4. ROI 사다리꼴을 좁히지 말 것. 예전 값(0.85/0.07/0.49)은 CNN 마스크의 75%를 잘랐다.
5. MIN_LANE_Y_SPAN 은 반드시 **병합 후**에 건다. 병합 전 개별 점선 조각은
   30px/5줄까지 작아서, 조각 단위로 걸면 진짜 점선이 먼저 죽는다.
6. 수평 성분(abs(vy) < 0.2)은 차선 후보에서 빼되 버리지 않는다 —
   정지선이 바로 그 성분이다. last_horizontals 에 따로 담아둔다.
7. **조각 병합을 x_bottom 으로 하지 말 것.** 조각의 세로 길이는 중앙값 58px 인데
   그걸 y=479 까지 외삽하면 오차가 폭발한다 (x_bottom 의 74% 가 화면 밖으로
   나갔다). 그 값끼리 비교하니 같은 점선의 조각이 서로 다른 차선으로 갈라져
   조각 1개짜리 후보가 81% 였다. 지금은 조각이 실제로 픽셀을 가진 y 에서
   비교하고, 허용치를 세로 간격에 비례해 늘린다.
8. **fill_ratio 로 점선을 판정하지 말 것.** 연결 성분은 정의상 y 가 끊기지
   않으므로 조각 1개짜리 후보의 fill_ratio 는 **항상 정확히 1.000** 이다
   (측정값: 262개 전부 1.000, 점선 판정 0개). 임계값을 올려도 소용없다.
   실선/점선의 실제 차이는 시간축이다 — 고정된 근거리 행에서 실선은 늘
   픽셀이 있고 점선은 대시가 지나갈 때만 있다. fill_ratio 는 트랙 이력이
   모자랄 때의 폴백으로만 남겨 뒀다.
9. coasted 후보의 xs/ys 는 **가짜 픽셀**이다. 급커브에서 조각이 전부 최소
   크기 필터에 걸려 "차선 없음"이 되는 걸 메우려고 트랙 예측으로 채운다.
   픽셀 수를 세거나 마스크로 되돌리는 후처리는 반드시 coasted 를 걸러낼 것.
"""

import argparse
import collections
import ctypes
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
lib_dir = os.path.join(parent_dir, 'lib')

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

import shutil
import threading
import time
import traceback
import cv2
import numpy as np

from lib.define.Camera import Camera
from lib.network.UDP import Receiver


# 모델 입력 크기는 ONNX에 고정되어 있다 (input "images": [1, 3, 360, 640]).
# 출력은 두 개 — "da"(drivable area)와 "ll"(lane line). 둘 다 (1, 2, 360, 640) logit이다.
SEG_INPUT_WIDTH = 640
SEG_INPUT_HEIGHT = 360

# --- 후보 추출 파라미터 ---
MIN_COMPONENT_AREA = 30       # 이보다 작은 성분은 노이즈로 버린다
# 조각 병합 판정: 두 조각을 **한 직선으로 같이 맞춰** 평균 편차를 본다.
# 같은 차선의 대시들은 한 직선 위에 있으므로 편차가 선 굵기(1~3px) 수준이고,
# 다른 차선이면 합동 피팅이 둘 사이를 지나가므로 편차가 간격의 절반쯤 된다.
# 조각의 기울기를 외삽해 비교하는 방식은 쓰지 않는다 (파일 상단 7번) —
# 조각이 짧으면 기울기 자체를 못 믿으므로 어디서 평가하든 실패한다.
# 합동 피팅은 반대로 baseline 이 길어져 기울기 오차가 줄어든다.
MERGE_MAX_RESIDUAL = 8.0      # 합동 피팅 평균 절대 편차 허용치 (px)
MERGE_ANGLE_TOLERANCE = 25.0  # 합동 피팅 전에 거르는 값싼 사전 필터 (deg)
MIN_LANE_PIXELS = 40          # 차선 하나로 인정할 최소 픽셀 수
# 병합이 끝난 뒤 세로 구간이 이보다 짧으면 버린다.
# CNN 이 도로 연석·갓길 경계를 차선으로 잘못 잡은 얼룩(예: 71px/5줄)이 여기서 걸린다.
MIN_LANE_Y_SPAN = 20
# 수평에 가까운 성분은 차선이 아니다 (전방 뷰의 차선은 항상 세로 성분을 갖는다).
# 이 값보다 |vy| 가 작으면 차선 후보에서 빼고 last_horizontals 로 보낸다.
HORIZONTAL_VY = 0.2

# 점선 판별 폴백. 트랙 관측이 모자랄 때만 쓴다 (파일 상단 8번).
# 조각 1개짜리 후보에서는 정의상 1.000 이 나오므로 이걸로는 점선을 못 잡는다.
DASH_FILL_RATIO_THRESHOLD = 0.85

# --- 프레임 간 추적 ---
# 점유를 확인할 근거리 행. ROI 는 y=168~479 이고, 이 행은 원근 왜곡이 작아
# 대시와 간격이 충분히 길게 보이는 구간이다.
ANCHOR_Y = 430
ANCHOR_BAND = 5               # 앵커 행 ±몇 px 를 훑을지
ANCHOR_X_TOLERANCE = 18       # 예측 x 에서 이 범위 안의 마스크를 본다
# 트랙을 새로 만들려면 후보가 이만큼은 근거리로 내려와 있어야 한다. 소실점
# 근처 조각으로 트랙을 만들면 다음 프레임에 다시 못 붙어 1프레임짜리 쓰레기
# 트랙만 쌓인다.
TRACK_BIRTH_NEAR_MARGIN = 60
# 점유율이 이보다 낮으면 점선이 아니라 **차선을 놓친 트랙**이다. 진짜 점선은
# K-City 규격상 0.35~0.6 근처로 나온다. 0에 가까운 트랙은 빈 공간을 찌르고
# 있는 것이므로 점선으로 부르지 말고 버린다.
DASH_MIN_OCCUPANCY = 0.12
TRACK_MAX_DEVIATION = 25.0    # 트랙 직선에서 픽셀들이 이만큼 안쪽이면 같은 차선
TRACK_ANGLE_TOLERANCE = 25.0  # 프레임 간 기울기 변화 허용치 (deg)
# 트랙 직선을 매 프레임 새로 맞추면 소실점 쪽 조각 하나에도 크게 흔들린다.
# 앵커 행에서의 x 와 기울기를 지수평활해서 들고 간다 (1.0 이면 평활 없음).
TRACK_FIT_SMOOTHING = 0.5
TRACK_MAX_MISSES = 8          # 연속 미검출이 이만큼이면 트랙을 버린다
TRACK_COAST_FRAMES = 5        # 미검출 중에도 예측으로 후보를 채워 줄 프레임 수
TRACK_MIN_HITS = 4            # 이만큼 잡힌 트랙만 coast 시킨다 (허깨비 방지)
TRACK_HISTORY = 40            # 점유 이력 길이 (프레임)
TRACK_MIN_OBSERVATIONS = 10   # 확정 관측이 이만큼 쌓여야 점선/실선을 판정한다
# 앵커 행 점유율이 이보다 낮으면 점선 후보. 실선은 ~1.0, K-City 규격(대시 3m +
# 간격 5m)이면 점선은 0.4~0.6 근처로 나온다.
DASH_OCCUPANCY_THRESHOLD = 0.85
# 다만 점유율만으로는 부족하다. 차선이 한 번 사라졌다 마는 경우도 점유율이
# 떨어지기 때문이다 (측정 예: 점유율 0.67 인데 이력이 `####...####.....` 로
# 점멸 1회 — 이건 점선이 아니라 검출이 끊긴 것이다).
# 점선의 특징은 **반복되는 점멸**이다. 상태가 이만큼은 뒤집혀야 점선으로 본다.
DASH_MIN_FLIPS = 2

# --- 표시 색 (BGR). 이 두 가지만 쓴다 ---
DASHED_COLOR = (255, 0, 0)    # 점선 = 파랑
SOLID_COLOR = (0, 0, 255)     # 실선 = 빨강


def _fit_line(xs, ys):
    """픽셀 집합에 직선을 맞춰 {slope(dx/dy), x0, y0, angle} 을 돌려준다.

    2차 곡선을 쓰지 않는 이유는 외삽 안정성이다 — 화면 밖까지 밀어내면 발산한다.
    """
    vx, vy, x0, y0 = cv2.fitLine(
        np.column_stack([xs, ys]).astype(np.float32),
        cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()
    slope = float(vx / vy)
    return {"slope": slope, "x0": float(x0), "y0": float(y0),
            "angle": float(np.degrees(np.arctan(slope)))}


def _x_at(fit, y):
    """직선을 y 에서 평가한다. y 는 스칼라도 배열도 된다."""
    return fit["x0"] + fit["slope"] * (y - fit["y0"])


def _line_deviation(fit, xs, ys, sample=400):
    """픽셀들이 직선에서 x 방향으로 벗어난 평균 거리.

    **픽셀이 실제로 있는 자리에서만** 잰다. 직선을 픽셀이 없는 곳까지 밀어내
    비교하는 순간 파일 상단 7번의 실수를 반복하게 된다.
    """
    if xs.size > sample:
        step = xs.size // sample
        xs, ys = xs[::step], ys[::step]
    return float(np.abs(_x_at(fit, ys.astype(np.float64)) - xs).mean())


def _joint_residual(xs, ys):
    """픽셀 전체를 한 직선으로 맞췄을 때의 평균 편차와 그 직선.

    같은 차선의 대시들이면 편차가 선 굵기 수준(1~3px)으로 작고, 서로 다른
    차선이면 합동 피팅이 둘 사이를 지나가므로 편차가 간격의 절반쯤으로 커진다.
    """
    fit = _fit_line(xs, ys)
    return _line_deviation(fit, xs, ys), fit


class LaneCandidateDetector:
    """마스크에서 차선 후보를 뽑는 데까지만 담당한다."""

    def __init__(self, img_width, img_height, seg_model_path=None, seg_interval=1):
        self.img_width = img_width
        self.img_height = img_height
        self.img_center = img_width / 2.0

        # ROI 사다리꼴. 마스크에 하늘·본네트가 없으므로 지평선 가드 역할만 한다.
        self.trap_bottom_width = 1.00
        self.trap_top_width = 0.80
        self.trap_height = 0.65

        self.seg_net = self.load_segmentation_model(seg_model_path)
        self.seg_output_names = (
            self.seg_net.getUnconnectedOutLayersNames() if self.seg_net is not None else ()
        )

        # CPU 추론이 ~180ms라 매 프레임 돌리면 5~6 FPS가 한계다.
        # seg_interval > 1 이면 N 프레임마다 한 번만 추론하고 직전 마스크를 재사용한다.
        self.seg_interval = max(1, int(seg_interval))
        self._seg_counter = 0
        self._cached_lane_mask = None
        self._cached_drivable_mask = None

        # 프레임 간 트랙. 연속된 영상에서만 의미가 있으므로 서로 무관한 이미지를
        # 넘길 때는 reset_tracks() 를 불러야 한다.
        self._tracks = []
        self._next_track_id = 0

        # 마지막 프레임 결과 (디버그·후처리용)
        self.last_candidates = []
        self.last_horizontals = []      # 정지선 등 수평 성분 (판정하지 않고 담아만 둔다)

    def reset_tracks(self):
        """트랙 이력을 버린다. 연속되지 않은 프레임을 넣기 전에 부른다."""
        self._tracks = []

    # ------------------------------------------------------------------
    # 세그멘테이션
    # ------------------------------------------------------------------
    def load_segmentation_model(self, model_path):
        if model_path is None:
            print("Segmentation model 경로가 지정되지 않았습니다.")
            return None

        model_path = (os.path.join(current_dir, model_path)
                      if not os.path.isabs(model_path) else model_path)
        if not os.path.isfile(model_path):
            print(f"Segmentation model 파일을 찾을 수 없습니다: {model_path}")
            return None

        try:
            net = cv2.dnn.readNet(model_path)
            print(f"Segmentation model loaded: {model_path}")
            return net
        except Exception as e:
            print(f"Segmentation model 로드 실패: {e}")
            return None

    def _run_segmentation(self, img_frame):
        """ONNX를 한 번 돌려 (lane_line, drivable_area) 마스크를 얻는다.

        두 출력 모두 softmax 이전의 logit이므로 채널 argmax로 이진화한다.
        """
        blob = cv2.dnn.blobFromImage(
            img_frame,
            scalefactor=1.0 / 255.0,
            size=(SEG_INPUT_WIDTH, SEG_INPUT_HEIGHT),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.seg_net.setInput(blob)
        outputs = self.seg_net.forward(self.seg_output_names)
        named = dict(zip(self.seg_output_names, outputs))

        def to_mask(logits):
            mask = (logits[0, 1] > logits[0, 0]).astype(np.uint8) * 255
            return cv2.resize(mask, (self.img_width, self.img_height),
                              interpolation=cv2.INTER_NEAREST)

        return to_mask(named["ll"]), to_mask(named["da"])

    def segment_lanes(self, img_frame):
        """차선(lane line) 마스크. seg_interval 프레임마다만 실제로 추론한다."""
        if self.seg_net is None:
            return np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        if self._cached_lane_mask is None or self._seg_counter % self.seg_interval == 0:
            self._cached_lane_mask, self._cached_drivable_mask = self._run_segmentation(img_frame)
        self._seg_counter += 1

        # 차선은 폭이 1~3px로 얇다. OPEN을 걸면 지워지므로 CLOSE로 끊긴 점선만 잇는다.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(self._cached_lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def drivable_mask(self, img_frame):
        """주행 가능 영역 마스크. 컬러 마스크의 ROI로 쓴다."""
        if self.seg_net is None:
            return np.full((self.img_height, self.img_width), 255, dtype=np.uint8)

        self.segment_lanes(img_frame)      # 캐시 갱신 (같은 추론 결과를 공유)
        return self._cached_drivable_mask

    def color_mask(self, img_frame):
        hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv_clahe = cv2.merge((h, s, clahe.apply(v)))

        white = cv2.inRange(hsv_clahe, np.array([0, 0, 140]), np.array([179, 50, 255]))
        yellow = cv2.inRange(hsv_clahe, np.array([10, 80, 80]), np.array([40, 255, 255]))
        mask = cv2.bitwise_or(white, yellow)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def combine_masks(self, img_frame):
        """차선 마스크를 우선 쓰고, 비어 있을 때만 컬러 마스크로 보완한다.

        컬러 마스크 단독은 하늘·구름·본네트를 그대로 통과시키므로
        반드시 주행 가능 영역으로 잘라낸 뒤에 합친다.
        """
        lane_mask = self.segment_lanes(img_frame)
        if cv2.countNonZero(lane_mask) >= 300:
            return lane_mask

        road = cv2.dilate(
            self.drivable_mask(img_frame),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        fallback = cv2.bitwise_and(self.color_mask(img_frame), road)
        return cv2.bitwise_or(lane_mask, fallback)

    def limit_region(self, img_mask):
        """사다리꼴 ROI. 지평선 가드 역할만 하므로 좁히지 말 것."""
        h, w = img_mask.shape[:2]
        mask = np.zeros_like(img_mask)

        points = np.array([[
            int((w * (1 - self.trap_bottom_width)) / 2), h,
            int((w * (1 - self.trap_top_width)) / 2), int(h - h * self.trap_height),
            int(w - (w * (1 - self.trap_top_width)) / 2), int(h - h * self.trap_height),
            int(w - (w * (1 - self.trap_bottom_width)) / 2), h
        ]], dtype=np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(mask, [points], 255)
        return cv2.bitwise_and(img_mask, mask)

    # ------------------------------------------------------------------
    # 후보 추출
    # ------------------------------------------------------------------
    def _segment_lines(self, mask):
        """연결 성분을 선분 후보로 만든다.

        각 성분에 직선을 맞춰 방향과 하단(y=h-1) x절편을 구한다.
        선형 피팅을 쓰는 이유는 외삽이 안정적이기 때문이다. 2차 곡선은
        화면 밖까지 밀어내면 발산해서 절편이 엉뚱하게 나온다.
        """
        h = mask.shape[0]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        segments = []
        self.last_horizontals = []
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
                continue

            ys, xs = np.nonzero(labels == i)
            vx, vy, x0, y0 = cv2.fitLine(
                np.column_stack([xs, ys]).astype(np.float32),
                cv2.DIST_L2, 0, 0.01, 0.01
            ).ravel()

            # 수평 성분은 차선이 아니다. 다만 정지선이 바로 이것이므로 버리지 않고
            # 따로 담아둔다 (판정은 하지 않는다 — 후처리에서 결정할 몫).
            if abs(vy) < HORIZONTAL_VY:
                bx, by, bw, bh, area = stats[i]
                self.last_horizontals.append({
                    "xs": xs, "ys": ys,
                    "box": (int(bx), int(by), int(bw), int(bh)),
                    "area": int(area),
                    "fill": float(area) / float(bw * bh) if bw and bh else 0.0,
                })
                continue

            slope = float(vx / vy)                # dx/dy
            fit = {"slope": slope, "x0": float(x0), "y0": float(y0),
                   "angle": float(np.degrees(np.arctan(slope)))}
            # 병합은 x_bottom 이 아니라 조각이 실제로 픽셀을 가진 위치에서 한다
            # (파일 상단 7번). 그래서 y 구간과 무게중심을 같이 들고 다닌다.
            segments.append({
                "xs": xs,
                "ys": ys,
                "fit": fit,
                "angle": fit["angle"],
                "n": int(xs.size),
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
                "x_center": float(xs.mean()),
                "y_center": float(ys.mean()),
                "x_bottom": float(_x_at(fit, h - 1)),
            })

        return segments

    def _seed_group(self, seg):
        return {
            "parts": [seg],
            "xs": seg["xs"], "ys": seg["ys"],
            "fit": seg["fit"],
            "n": seg["n"],
        }

    def _merge_segments(self, segments):
        """같은 차선의 조각끼리 묶는다 (점선 조각들 → 하나의 차선).

        예전에는 화면 하단까지 외삽한 x절편으로 비교했는데, 조각의 세로 길이가
        중앙값 58px 라 외삽 오차가 폭발해서 x절편의 74% 가 화면 밖으로 나갔다.
        그 값끼리 비교하니 같은 점선의 조각이 갈라져 조각 1개짜리 후보가 81%
        였고, 그게 점선을 실선으로 오판하는 원인이었다 (파일 상단 7·8번).

        지금은 **근거가 많은 쪽의 직선을 상대 조각의 y 중심에 평가**해서 x 차이를
        본다. 즉 조각이 실제로 픽셀을 가진 곳에서 비교한다. 허용치는 두 조각
        사이의 세로 간격에 비례해 늘리되 상한을 둔다.
        """
        merged = []
        # 픽셀이 많은 조각부터 그룹의 씨앗이 되게 한다 — 방향이 가장 믿을 만하고,
        # 소실점 근처의 짧은 조각이 엉뚱한 그룹을 먼저 만드는 걸 막는다.
        for seg in sorted(segments, key=lambda s: -s["n"]):
            best, best_res, best_fit = None, None, None
            for group in merged:
                if abs(seg["angle"] - group["fit"]["angle"]) >= MERGE_ANGLE_TOLERANCE:
                    continue

                xs = np.concatenate([group["xs"], seg["xs"]])
                ys = np.concatenate([group["ys"], seg["ys"]])
                residual, fit = _joint_residual(xs, ys)

                # 먼저 걸리는 그룹이 아니라 **가장 잘 맞는** 그룹에 붙인다.
                # 소실점 근처는 차선들이 모여 있어 여러 그룹이 동시에 걸린다.
                if residual < MERGE_MAX_RESIDUAL and (best_res is None or residual < best_res):
                    best, best_res, best_fit = group, residual, fit

            if best is None:
                merged.append(self._seed_group(seg))
                continue

            best["parts"].append(seg)
            best["xs"] = np.concatenate([best["xs"], seg["xs"]])
            best["ys"] = np.concatenate([best["ys"], seg["ys"]])
            best["fit"] = best_fit          # 합동 피팅 결과를 그대로 쓴다
            best["n"] = int(best["xs"].size)

        h = self.img_height
        candidates = []
        for group in merged:
            xs, ys = group["xs"], group["ys"]
            if xs.size < MIN_LANE_PIXELS:
                continue

            y_span = int(ys.max() - ys.min()) + 1
            if y_span < MIN_LANE_Y_SPAN:       # 반드시 병합 후에 건다 (파일 상단 5번)
                continue

            fill_ratio = np.unique(ys).size / y_span
            n_parts = len(group["parts"])
            candidates.append({
                "xs": xs,
                "ys": ys,
                "fit": group["fit"],
                "x_bottom": float(_x_at(group["fit"], h - 1)),
                "angle": group["fit"]["angle"],
                # 폴백값. _update_tracks 가 트랙 판정으로 덮어쓴다.
                "dashed": fill_ratio < DASH_FILL_RATIO_THRESHOLD,
                "dash_source": "fill",
                "occupancy": None,
                "fill_ratio": float(fill_ratio),
                "n_parts": n_parts,
                "y_span": y_span,
                "n_pixels": int(xs.size),
                "coasted": False,
                # 소실점 쪽에만 있는 조각은 트랙을 못 받는다 (None 으로 남는다).
                "track_id": None,
            })

        return sorted(candidates, key=lambda c: c["x_bottom"])

    # ------------------------------------------------------------------
    # 프레임 간 추적
    # ------------------------------------------------------------------
    @staticmethod
    def _anchor_fit(fit):
        """같은 직선을 앵커 행 기준으로 다시 쓴다 (x0 = 앵커 행에서의 x).

        트랙의 직선은 앵커 행에 고정해 둔다. 그래야 프레임 간 평활이 '근거리
        차선 위치'를 평활하는 의미가 되고, 조각의 y0 가 프레임마다 달라져서
        같은 직선이 다른 숫자로 보이는 일이 없다.
        """
        return {"slope": fit["slope"], "x0": float(_x_at(fit, ANCHOR_Y)),
                "y0": float(ANCHOR_Y), "angle": fit["angle"]}

    @staticmethod
    def _smooth_fit(old, new):
        """앵커 행 x 와 기울기를 지수평활한다."""
        new = LaneCandidateDetector._anchor_fit(new)
        a = TRACK_FIT_SMOOTHING
        slope = (1 - a) * old["slope"] + a * new["slope"]
        return {
            "slope": slope,
            "x0": (1 - a) * old["x0"] + a * new["x0"],
            "y0": float(ANCHOR_Y),
            "angle": float(np.degrees(np.arctan(slope))),
        }

    def _probe_anchor(self, mask, track):
        """앵커 행이 이 프레임에 점유됐는지 **마스크에서 직접** 본다. 못 보면 None.

        후보 안에서 간격을 찾으면 안 된다. 병합이 실패하면 후보 하나가 곧 대시
        하나라서, 대시는 늘 '자기 자신으로 꽉 차' 있고 간격은 아예 후보로
        존재하지 않는다. 그래서 어떤 지표를 만들어도 실선으로 나온다.
        트랙은 차선이 **어디 있어야 하는지** 알고 있으니, 그 자리의 마스크를
        찔러 보면 병합·분할과 무관하게 대시 간격이 잡힌다.

        관측 가능 조건은 '이번 프레임의 후보가 앵커 행까지 닿는가'가 **아니다.**
        그렇게 걸면 순환이 된다 — 닿는 후보는 정의상 그 행에 픽셀이 있으므로
        늘 '점유'가 나오고, 정작 대시 간격인 프레임은 관측에서 빠진다
        (측정: 상위 트랙 점유율이 전부 1.00 으로 눌렸다).
        조건은 '이 트랙이 앵커 행을 가로지르는 차선인가' 여야 하고, 그건 한 번
        가로지르는 걸 본 시점에 확정된다 (anchor_seen). 그 뒤로는 이번 프레임에
        조각이 어디까지 왔든 상관없이 매번 찌른다.
        """
        if not track["anchor_seen"]:
            return None                        # 앵커 행을 지나는 차선인지 아직 모름
        x = _x_at(track["fit"], ANCHOR_Y)
        if not (0 <= x < self.img_width):
            return None                        # 화면 밖으로 나간 차선
        x0 = int(max(0, x - ANCHOR_X_TOLERANCE))
        x1 = int(min(self.img_width, x + ANCHOR_X_TOLERANCE + 1))
        y0 = max(0, ANCHOR_Y - ANCHOR_BAND)
        y1 = min(self.img_height, ANCHOR_Y + ANCHOR_BAND + 1)
        return bool(np.any(mask[y0:y1, x0:x1]))

    def _best_track(self, cand):
        """후보를 어느 트랙에 붙일지. 없으면 None.

        트랙 직선에서 **후보 픽셀이 실제로 있는 자리의** 평균 편차로 고른다.
        직선을 앵커 행이나 화면 하단까지 밀어서 x 를 비교하면, 짧은 조각의
        기울기 오차가 그대로 증폭돼 같은 차선이 매 프레임 새 ID 를 받는다.
        """
        best, best_dev = None, None
        for ti, track in enumerate(self._tracks):
            if abs(cand["angle"] - track["fit"]["angle"]) >= TRACK_ANGLE_TOLERANCE:
                continue
            dev = _line_deviation(track["fit"], cand["xs"], cand["ys"])
            if dev < TRACK_MAX_DEVIATION and (best_dev is None or dev < best_dev):
                best, best_dev = ti, dev
        return best

    def _update_tracks(self, candidates, mask):
        """후보에 트랙 ID 를 붙이고, 앵커 행 점유 이력으로 점선/실선을 판정한다.

        한 프레임 안의 fill_ratio 는 조각 1개짜리 후보에서 **정의상 1.0** 이라
        점선을 영영 못 잡는다 (파일 상단 8번). 실선과 점선의 실제 차이는
        시간축에 있다: 고정된 근거리 행에서 실선은 늘 픽셀이 있고, 점선은
        대시가 지나갈 때만 있다. 자차가 움직이므로 K-City 규격(대시 3m +
        간격 5m)이면 30~60km/h 에서 10~20 프레임 주기로 점멸한다.

        연관은 **다대일**이다. 한 차선이 여러 조각으로 나뉘어 들어와도 같은
        트랙에 붙는다. 1:1 로 묶으면 남는 조각이 매 프레임 새 트랙을 만들어
        (측정: 트랙 수명 중앙값 2프레임) 이력이 쌓이지 않는다. 다대일이면
        프레임 병합이 실패해도 트랙 단위로는 한 차선으로 모인다.

        미검출된 트랙은 잠깐 예측으로 살려 둔다. 급커브에서 조각이 전부 최소
        크기 필터에 걸려 차선이 통째로 사라지는 구간을 메우기 위해서다.
        """
        assigned = [self._best_track(c) for c in candidates]

        # 어느 트랙에도 안 붙은 후보는 새 트랙이 된다. 단 **근거리까지 내려온
        # 후보만** 트랙을 만든다. 소실점 근처 조각으로 트랙을 만들면 다음
        # 프레임에 다시 못 붙어서 1프레임짜리 쓰레기 트랙이 쌓이고(측정: 전체
        # 트랙의 69%), 앵커 행 점유도 판정할 수 없어 이력이 영영 안 찬다.
        # 먼 조각은 기존 트랙에 붙어 피팅을 도울 수는 있다.
        for ci, ti in enumerate(assigned):
            if ti is not None:
                continue
            cand = candidates[ci]
            if int(cand["ys"].max()) < ANCHOR_Y - TRACK_BIRTH_NEAR_MARGIN:
                continue
            self._tracks.append({
                "id": self._next_track_id,
                "fit": self._anchor_fit(cand["fit"]),
                "hits": 0,
                "misses": 0,
                "history": collections.deque(maxlen=TRACK_HISTORY),
                "y_min": int(cand["ys"].min()), "y_max": int(cand["ys"].max()),
                "anchor_seen": False,
            })
            self._next_track_id += 1
            assigned[ci] = len(self._tracks) - 1

        members = collections.defaultdict(list)
        for ci, ti in enumerate(assigned):
            if ti is not None:
                members[ti].append(ci)

        alive, coasted = [], []
        for ti, track in enumerate(self._tracks):
            cis = members.get(ti)

            if cis:
                xs = np.concatenate([candidates[c]["xs"] for c in cis])
                ys = np.concatenate([candidates[c]["ys"] for c in cis])
                track["fit"] = self._smooth_fit(track["fit"], _fit_line(xs, ys))
                track["y_min"], track["y_max"] = int(ys.min()), int(ys.max())
                track["hits"] += 1
                track["misses"] = 0
                # 앵커 행을 실제로 가로지르는 걸 한 번 보면 그 뒤로는 계속 찌른다.
                if track["y_min"] <= ANCHOR_Y <= track["y_max"]:
                    track["anchor_seen"] = True
                track["history"].append(self._probe_anchor(mask, track))
            else:
                track["misses"] += 1
                if track["misses"] > TRACK_MAX_MISSES:
                    continue                   # 폐기 — 오래된 트랙은 현재 장면과 무관하다
                # 조각이 하나도 안 붙은 프레임도 마스크는 찌른다. 대시 간격이
                # 넓으면 차선 전체가 잠깐 사라지는데 그것도 '비었다'는 관측이다.
                # 트랙이 엉뚱한 곳을 보고 있으면 여기서 점유율이 0 으로 떨어져
                # DASH_MIN_OCCUPANCY 에 걸려 스스로 정리된다.
                track["history"].append(self._probe_anchor(mask, track))

            dashed, occupancy = self._track_verdict(track)
            if occupancy is not None and occupancy < DASH_MIN_OCCUPANCY:
                continue                       # 빈 공간을 찌르는 트랙 — 폐기

            alive.append(track)
            for ci in cis or ():
                cand = candidates[ci]
                cand["track_id"] = track["id"]
                if occupancy is not None:
                    cand["occupancy"] = occupancy
                    cand["dashed"] = dashed
                    cand["dash_source"] = "track"

            if (not cis and track["misses"] <= TRACK_COAST_FRAMES
                    and track["hits"] >= TRACK_MIN_HITS):
                coasted.append(self._coast_candidate(track, dashed, occupancy))

        self._tracks = alive
        return sorted(candidates + coasted, key=lambda c: c["x_bottom"])

    @staticmethod
    def _track_verdict(track):
        """트랙의 점유 이력 → (점선인가, 점유율). 관측이 모자라면 (False, None).

        점유율이 낮은 것만으로는 점선이라고 못 한다 — 차선이 한동안 안 잡힌
        구간도 점유율을 떨어뜨린다. 점선은 **되풀이해서 점멸**하므로 상태가
        여러 번 뒤집혔는지까지 본다. 실선은 점멸이 0 회로 아주 깨끗하게 갈린다.
        """
        observed = [o for o in track["history"] if o is not None]
        if len(observed) < TRACK_MIN_OBSERVATIONS:
            return False, None
        occupancy = sum(observed) / len(observed)
        flips = sum(1 for a, b in zip(observed, observed[1:]) if a != b)
        dashed = occupancy < DASH_OCCUPANCY_THRESHOLD and flips >= DASH_MIN_FLIPS
        return dashed, float(occupancy)

    def _coast_candidate(self, track, dashed, occupancy):
        """트랙 예측으로 만든 후보. xs/ys 는 가짜 픽셀이다 (파일 상단 9번)."""
        ys = np.arange(int(track["y_min"]), int(track["y_max"]) + 1, dtype=np.int32)
        xs = np.clip(np.round(_x_at(track["fit"], ys)), 0,
                     self.img_width - 1).astype(np.int32)

        return {
            "xs": xs,
            "ys": ys,
            "fit": track["fit"],
            "x_bottom": float(_x_at(track["fit"], self.img_height - 1)),
            "angle": track["fit"]["angle"],
            "dashed": dashed,
            "dash_source": "track" if occupancy is not None else "fill",
            "occupancy": occupancy,
            "fill_ratio": None,
            "n_parts": 0,
            "y_span": int(ys.size),
            "n_pixels": 0,
            "coasted": True,
            "track_id": track["id"],
        }

    def lane_candidates(self, binary_mask):
        """마스크 → 차선 후보 리스트 (x_bottom 오름차순). 후처리의 입력이다."""
        if binary_mask is None or binary_mask.size == 0:
            self.last_candidates = []
            return []
        candidates = self._merge_segments(self._segment_lines(binary_mask))
        self.last_candidates = self._update_tracks(candidates, binary_mask)
        return self.last_candidates

    def process(self, img_frame):
        """한 프레임 → (ROI 마스크, 후보 리스트). 후처리는 이 반환값에서 시작한다."""
        roi_mask = self.limit_region(self.combine_masks(img_frame))
        return roi_mask, self.lane_candidates(roi_mask)

    # ------------------------------------------------------------------
    # 시각화 (후보 확인용)
    # ------------------------------------------------------------------
    def draw_candidates(self, img_input):
        """후보를 점선=파랑 / 실선=빨강 두 가지 색으로만 그린다 (BGR).

        번호는 후보의 순번이 아니라 **트랙 ID** 다. 프레임을 넘겨 가며 같은
        번호가 유지되는지 보는 게 추적이 제대로 붙었는지 보는 가장 빠른 방법이다.
        coasted(예측으로 채운) 후보는 실제 픽셀이 아니므로 점을 띄엄띄엄 찍어
        구분한다. 색은 늘리지 않는다.

        수평 성분은 그리지 않는다 — 차선이 아니고, 개수는 좌상단 표시로 본다.
        """
        out = img_input.copy()

        # 앵커 행. 점선/실선 판정이 이 행의 점유만 보므로 같이 그려 둔다.
        cv2.line(out, (0, ANCHOR_Y), (self.img_width - 1, ANCHOR_Y), (60, 60, 60), 1)

        for cand in self.last_candidates:
            color = DASHED_COLOR if cand["dashed"] else SOLID_COLOR
            if cand["coasted"]:
                out[cand["ys"][::4], cand["xs"][::4]] = color
            else:
                out[cand["ys"], cand["xs"]] = color

            x_bottom = int(np.clip(cand["x_bottom"], 2, self.img_width - 3))
            label = str(cand.get("track_id", "?"))
            if cand["coasted"]:
                cv2.circle(out, (x_bottom, self.img_height - 6), 6, color, 1)
                label += "~"
            else:
                cv2.circle(out, (x_bottom, self.img_height - 6), 6, color, -1)
            cv2.putText(out, label, (x_bottom - 4, self.img_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        by_track = sum(1 for c in self.last_candidates if c["dash_source"] == "track")
        cv2.putText(out,
                    f"cand={len(self.last_candidates)}  hz={len(self.last_horizontals)}"
                    f"  trk={by_track}/{len(self.last_candidates)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, "dashed", (10, out.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, DASHED_COLOR, 2)
        cv2.putText(out, "solid", (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, SOLID_COLOR, 2)
        return out


# ======================================================================
# 화면 구성 / 실행
# ======================================================================
WINDOW_TITLE = "LaneCandidates"
PANEL_LABELS = ("1 source", "2 ROI mask", "3 candidates")


def _screen_size():
    """화면 해상도. 못 얻으면 1080p를 가정한다."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        w, h = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


_SCREEN_W, _SCREEN_H = _screen_size()
MAX_PANEL_WIDTH = _SCREEN_W - 80
MAX_PANEL_HEIGHT = _SCREEN_H - 160


def _label_panel(img, text):
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = out.shape[1] - tw - 10
    cv2.rectangle(out, (x - 6, 4), (out.shape[1] - 2, th + 14), (0, 0, 0), -1)
    cv2.putText(out, text, (x, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def compose_panel(views, fit_screen=True):
    """패널 3장을 가로로 붙이고 화면에 들어가도록 축소한다.

    fit_screen=False 면 축소하지 않고 원본 크기로 돌려준다 (파일 저장용).
    """
    grid = np.hstack([_label_panel(v, t) for v, t in zip(views, PANEL_LABELS)])
    if not fit_screen:
        return grid

    scale = min(MAX_PANEL_WIDTH / grid.shape[1], MAX_PANEL_HEIGHT / grid.shape[0], 1.0)
    if scale < 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grid


def open_window():
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)


def process_frame(detector, image, fit_screen=True):
    """한 프레임을 파이프라인에 통과시키고 (3패널 화면, 요약문자열)을 돌려준다."""
    img_frame = cv2.resize(image, (640, 480))
    roi_mask, candidates = detector.process(img_frame)

    panel = compose_panel([
        img_frame,
        cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR),
        detector.draw_candidates(img_frame),
    ], fit_screen=fit_screen)

    dashed = sum(1 for c in candidates if c["dashed"])
    by_track = sum(1 for c in candidates if c["dash_source"] == "track")
    coasted = sum(1 for c in candidates if c["coasted"])
    ids = " ".join(f"{c['track_id']}{'~' if c['coasted'] else ''}" for c in candidates)
    info = (f"후보={len(candidates)} (점선 {dashed}, 트랙판정 {by_track}, 예측 {coasted})  "
            f"수평={len(detector.last_horizontals)}  id=[{ids}]")
    return panel, info


def _new_detector():
    # 오프라인은 실시간 제약이 없으므로 매 프레임 추론한다.
    return LaneCandidateDetector(640, 480, seg_model_path="lane_segmentation.onnx",
                                 seg_interval=1)


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")


def resolve_target(target):
    """상대 경로면 CWD → 스크립트 폴더 → 상위(저장소 루트) 순으로 찾는다."""
    if os.path.isabs(target):
        return target
    for base in (os.getcwd(), current_dir, parent_dir):
        candidate = os.path.join(base, target)
        if os.path.exists(candidate):
            return candidate
    return target


def collect_images(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return [os.path.join(path, name) for name in sorted(os.listdir(path))
                if name.lower().endswith(IMG_EXTS)]
    return []


def _finish_save(write_path, save_path, ask_before_save, frames, fps):
    """재생이 끝난 뒤 저장 여부를 확정한다.

    창을 띄웠으면 먼저 보여주고 물어본 뒤에 저장한다. 다시 처리하면 전체 추론을
    또 돌려야 하므로 임시 파일에 이미 써 두었고, "아니오"면 그걸 지운다.
    """
    if not ask_before_save:
        print(f"\n💾 저장 완료: {save_path} ({frames}프레임, {fps:.1f}fps)")
        return

    size_mb = os.path.getsize(write_path) / (1024 * 1024) if os.path.exists(write_path) else 0
    print(f"\n재생이 끝났습니다. ({frames}프레임, {fps:.1f}fps, {size_mb:.0f}MB)")
    try:
        answer = input(f"이 영상을 저장할까요? → {save_path}  [y/N]: ").strip().lower()
    except EOFError:
        # 입력을 받을 수 없는 환경(백그라운드 실행 등)에서 인코딩 결과를 그냥
        # 버리는 게 더 나쁘다. 남겨두고 그렇게 했다고 알린다.
        answer = "y"
        print("  (입력을 받을 수 없어 일단 저장합니다)")

    if answer in ("y", "yes"):
        shutil.move(write_path, save_path)
        print(f"💾 저장 완료: {save_path}")
    else:
        os.remove(write_path)
        print("  저장하지 않았습니다.")


def run_video(path, save_path=None, preview=True):
    """녹화된 주행 영상으로 후보 추출을 확인한다.

    space 일시정지/재개, n 한 프레임 진행, s 현재 화면 저장, ESC/q 종료.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[에러] 영상을 열 수 없습니다: {path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = _new_detector()
    if preview:
        open_window()

    writer = None
    # 임시 파일도 확장자는 .mp4 로 둬야 한다 — VideoWriter 는 확장자로 컨테이너를
    # 고르기 때문에 .tmp 로 두면 파일 자체를 열지 못한다.
    ask_before_save = bool(save_path) and preview
    write_path = (os.path.splitext(save_path)[0] + ".part.mp4") if ask_before_save else save_path

    print("=" * 60)
    print(f"🎬 영상 모드 — {os.path.basename(path)} ({total}프레임)")
    if save_path:
        if ask_before_save:
            print(f"💾 재생이 끝나면 저장할지 물어봅니다 → {save_path}")
        else:
            print(f"💾 저장 대상: {save_path}")
    if preview:
        print("space: 일시정지  |  n: 한 프레임  |  s: 화면 저장  |  ESC/q: 종료")
    print("=" * 60)

    idx = 0
    paused = False

    def handle(image):
        nonlocal writer
        if save_path:
            panel_full, info = process_frame(detector, image, fit_screen=False)
            if writer is None:
                h, w = panel_full.shape[:2]
                writer = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"영상 파일을 열 수 없습니다: {write_path}")
            writer.write(panel_full)
            if not preview:
                return panel_full, info
            scale = min(MAX_PANEL_WIDTH / panel_full.shape[1],
                        MAX_PANEL_HEIGHT / panel_full.shape[0], 1.0)
            shown = (cv2.resize(panel_full, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA) if scale < 1.0 else panel_full)
            return shown, info
        return process_frame(detector, image)

    try:
        while True:
            if not paused:
                ok, image = cap.read()
                if not ok:
                    print("\n영상 끝.")
                    break
                panel, info = handle(image)
                idx += 1
                if idx % 10 == 0 or idx == 1:
                    print(f"[{idx}/{total}] {info}")
                if preview:
                    cv2.imshow(WINDOW_TITLE, panel)

            if not preview:
                continue

            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                paused = not paused
                print("  일시정지" if paused else "  재개")
            elif key == ord("n") and paused:
                ok, image = cap.read()
                if not ok:
                    print("\n영상 끝.")
                    break
                panel, info = handle(image)
                idx += 1
                print(f"[{idx}/{total}] {info}")
                cv2.imshow(WINDOW_TITLE, panel)
            elif key == ord("s"):
                out_path = f"{os.path.splitext(path)[0]}_{idx:06d}.png"
                cv2.imwrite(out_path, panel)
                print(f"  저장: {out_path}")

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if writer is not None and idx > 0 and os.path.exists(write_path):
            _finish_save(write_path, save_path, ask_before_save, idx, fps)


def run_images(paths):
    """이미지 파일/폴더로 후보 추출을 확인한다. n/p 이동, s 저장, ESC/q 종료.

    이미지끼리는 연속된 프레임이 아니므로 매번 트랙을 리셋한다. 그래서 이
    모드에서는 점선/실선이 fill_ratio 폴백으로만 나온다 (신뢰도 낮음).
    """
    detector = _new_detector()
    open_window()

    print("=" * 60)
    print(f"🖼  이미지 모드 — {len(paths)}장")
    print("n/p: 다음·이전  |  s: 결과 저장  |  ESC/q: 종료")
    print("=" * 60)

    index = 0
    try:
        while True:
            path = paths[index]
            image = cv2.imread(path)
            if image is None:
                print(f"[경고] 읽을 수 없는 파일 건너뜀: {path}")
                paths.pop(index)
                if not paths:
                    print("[에러] 읽을 수 있는 이미지가 없습니다.")
                    return
                index %= len(paths)
                continue

            detector.reset_tracks()
            panel, info = process_frame(detector, image)
            print(f"[{index + 1}/{len(paths)}] {os.path.basename(path)}  {info}")
            cv2.imshow(WINDOW_TITLE, panel)

            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("n"), 83):
                index = (index + 1) % len(paths)
            elif key in (ord("p"), 81):
                index = (index - 1) % len(paths)
            elif key == ord("s"):
                out_path = f"{os.path.splitext(path)[0]}_cand.png"
                cv2.imwrite(out_path, panel)
                print(f"  저장: {out_path}")

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cv2.destroyAllWindows()


def run_offline(source, save_path=None, preview=True):
    target = resolve_target(source)

    if os.path.isfile(target) and target.lower().endswith(VIDEO_EXTS):
        run_video(target, save_path=save_path, preview=preview)
        return

    if save_path:
        print("[경고] --save 는 영상 입력에서만 동작합니다. 무시합니다.")

    paths = collect_images(target)
    if paths:
        run_images(paths)
        return

    print(f"[에러] 영상이나 이미지를 찾지 못했습니다: {target}")


# PC/가상머신마다 시뮬레이터 주소가 다르므로 환경변수와 CLI 인자로 바꿀 수 있게 둔다.
IP = os.environ.get("MORAI_CAM_IP", "192.168.0.200")
PORT = int(os.environ.get("MORAI_CAM_PORT", "1101"))
latest_frame = None
frame_lock = threading.Lock()


def camera_thread_worker(ip, port):
    global latest_frame
    cam_receiver = Receiver(ip, port, Camera())

    while True:
        try:
            data = cam_receiver.get_data()
            if data is None:
                time.sleep(0.01)
                continue
            if not hasattr(data, "image") or not data.image.data:
                continue

            image_np = np.frombuffer(data.image.data, dtype=np.uint8)
            if image_np.size == 0:
                continue
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                continue

            with frame_lock:
                latest_frame = cv2.resize(image, (640, 480))

        except (AttributeError, ValueError, OSError, cv2.error) as ex:
            print(f"[camera_thread_worker] recoverable error: {ex}")
            traceback.print_exc()
            time.sleep(0.1)
            continue
        except Exception as ex:
            print(f"[camera_thread_worker] unexpected error: {ex}")
            traceback.print_exc()
            raise


def main(cam_ip=IP, cam_port=PORT):
    t = threading.Thread(target=camera_thread_worker, args=(cam_ip, cam_port), daemon=True)
    t.start()
    time.sleep(1)

    # 2프레임마다 추론하고 사이 프레임은 직전 마스크를 재사용해 표시 프레임률을 확보한다.
    detector = LaneCandidateDetector(640, 480, seg_model_path="lane_segmentation.onnx",
                                     seg_interval=2)
    open_window()

    print("=" * 60)
    print("🚀 차선 후보 추출 (실시간)")
    print(f"   카메라 {cam_ip}:{cam_port}")
    print("종료하려면 결과 창을 선택하고 ESC 키를 누르세요.")
    print("=" * 60)

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                img_frame = latest_frame.copy()

            panel, _info = process_frame(detector, img_frame)
            cv2.imshow(WINDOW_TITLE, panel)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="차선 후보 추출까지만 (기본: 시뮬레이터 UDP 실시간)")
    parser.add_argument("--source", nargs="?", const="samples", default=None,
                        help="오프라인 모드. 영상/이미지 파일 또는 폴더 경로")
    parser.add_argument("--cam-ip", default=IP,
                        help=f"시뮬레이터 카메라 IP (기본 {IP}, 환경변수 MORAI_CAM_IP)")
    parser.add_argument("--cam-port", type=int, default=PORT,
                        help=f"카메라 포트 (기본 {PORT}, 환경변수 MORAI_CAM_PORT)")
    parser.add_argument("--save", metavar="OUT.mp4", default=None,
                        help="3패널 결과를 영상 파일로 저장한다 (--source 가 영상일 때)")
    parser.add_argument("--no-preview", action="store_true",
                        help="창을 띄우지 않는다 (저장만 할 때 더 빠르다)")
    args = parser.parse_args()

    if args.source:
        run_offline(args.source, save_path=args.save, preview=not args.no_preview)
    else:
        main(args.cam_ip, args.cam_port)
