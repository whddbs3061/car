"""딥러닝 차선 분할 + 컬러 마스크 폴백 하이브리드 차선 인식.

lane_segmentation.onnx 는 출력이 두 개다:
    da — drivable area (주행 가능 영역)
    ll — lane line     (차선)
cv2.dnn.forward() 는 첫 번째("da")만 돌려주므로 반드시 이름을 지정해 받는다.
두 출력 모두 softmax 이전 logit이라 채널 argmax로 이진화한다.

파이프라인:
    ll 마스크를 주력으로 쓰고, 비어 있을 때만
    (컬러 마스크 ∩ 팽창시킨 da 마스크)로 보완한다.
    컬러 마스크 단독은 하늘·구름·본네트를 그대로 통과시키므로
    반드시 주행 가능 영역으로 잘라낸 뒤에 합친다.

ego-lane 선택:
    히스토그램 최대점 방식은 차선이 4개 이상 보이면 옆 차로의 바깥 선을 잡고,
    슬라이딩 윈도우가 위로 올라가며 다른 선으로 갈아타 여러 차선을 가로지르는
    곡선을 만든다. 그래서 마스크를 개별 차선으로 분리한 뒤 고르는 방식으로 바꿨다:

        1. connectedComponents 로 성분 분리
        2. 기울기·절편이 비슷한 성분끼리 병합 (점선 처리)
        3. 각 차선을 화면 하단(y=h-1)까지 외삽해 x절편 계산
        4. 자차 중심 기준 좌/우에서 가장 가까운 선을 ego lane 경계로 선택
        5. 선택된 두 차선의 픽셀만으로 다항식 피팅

LaneSegHybrid.py 의 ego-lane 선택 버전.
"""

import argparse
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

import collections
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

# --- ego-lane 선택 파라미터 ---
MIN_COMPONENT_AREA = 30       # 이보다 작은 성분은 노이즈로 버린다
MERGE_X_TOLERANCE = 40        # 하단 x절편이 이만큼 가까우면 같은 차선으로 본다 (px)
MERGE_ANGLE_TOLERANCE = 12.0  # 기울기 차이 허용치 (deg)
MIN_LANE_PIXELS = 40          # 차선 하나로 인정할 최소 픽셀 수
# 병합이 끝난 뒤 세로 구간이 이보다 짧으면 버린다.
# CNN 이 도로 연석·갓길 경계를 차선으로 잘못 잡은 얼룩(예: 71px/5줄)이 여기서 걸린다.
# 반드시 "병합 후"에 걸어야 한다 — 병합 전 개별 점선 조각은 30px/5줄까지 작아서
# 조각 단위로 이 기준을 걸면 진짜 점선이 먼저 잘려나간다. 점선은 조각들이 합쳐지며
# 세로로 길어지고, 노이즈는 병합할 상대가 없어 짧게 남는 것이 둘을 가르는 신호다.
MIN_LANE_Y_SPAN = 20
# 각 fit 은 자기가 피팅된 y 구간 밖으로 이만큼까지만 외삽해서 그린다.
# 검출을 놓친 프레임은 직전 fit 을 재사용하는데, 그리는 구간(fit_y_range)은 매
# 프레임 새로 계산되므로 재사용 fit 이 원래 데이터 범위를 한참 벗어난 곳까지
# 끌려가 휘어버린다(2차 곡선은 범위 밖에서 발산한다). 그 결과 좌우 선이 교차해
# 주행 영역이 도로 밖까지 넓어진다. 없는 구간은 아예 그리지 않는 편이 낫다.
MAX_FIT_EXTRAPOLATION = 30
# 검출을 놓친 프레임은 직전 fit 을 재사용하는데, 이만큼 연속으로 놓치면 버린다.
# 커브를 지나 차선이 실제로 화면에서 사라진 경우 오래된 fit 은 현재 장면과
# 무관해져서 도로를 가로지르는 선이 된다. 없는 차선은 그리지 않는 편이 낫다.
# drive6 측정: 정상적인 순간 미검출은 중앙값 3~4, p90 이 5~7 프레임이고
# 차선이 진짜 사라진 구간만 19~31 프레임이다. 10 은 그 사이를 가른다.
MAX_FIT_REUSE_FRAMES = 10

# --- 쌍(pair) 평가 파라미터 ---
# 좌·우를 따로 고르면 (중심보다 왼쪽 중 가장 오른쪽 / 오른쪽 중 가장 왼쪽) 두 선이
# 함께 말이 되는지 아무도 검사하지 않는다. drive6 에서 폭이 167~1155px 까지 벌어지고
# 좌우가 교차해 폭이 음수가 되는 프레임이 35개 나온 원인이다.
# 그래서 자차 중심을 사이에 둔 모든 쌍을 열거해 비용이 가장 낮은 쌍을 고른다.
PAIR_COST_W_WIDTH = 1.0       # 기대 폭 대비 상대 오차
PAIR_COST_W_PARALLEL = 0.6    # 좌우 기울기 차 (교차하는 조합을 막는다)
PAIR_COST_W_CONTINUITY = 0.8  # 직전 프레임 선택과의 거리
PAIR_COST_W_EVIDENCE = 0.4    # 근거(픽셀 수·y_span)가 빈약할수록 벌점
MAX_PAIR_COST = 1.5           # 이보다 나쁘면 차라리 "검출 없음"으로 둔다

# --- 차로 폭 모델 ---
# 원근 때문에 차로 폭은 화면 아래로 갈수록 넓어진다. 스칼라 중앙값 하나로 일정
# offset 을 주면 먼 쪽에서 경로가 눈에 띄게 어긋난다. 그래서 width(y) = m*y + b 로
# 둔다 (원근에서 폭은 y 에 대해 거의 선형이다).
# 상수로 박지 않고 온라인으로 학습하는 이유: drive6 는 하단 폭 ~600px, CLAUDE.md 의
# samples 캡처는 ~454px 로 카메라 화각에 따라 다르다.
WIDTH_HISTORY_LEN = 90        # 최근 몇 프레임의 (y, width) 표본을 유지할지
MIN_WIDTH_SAMPLES = 10        # 이보다 적으면 폭 기반 기능을 쓰지 않는다

# --- 정지선 파라미터 ---
# 정지선은 차선과 달리 진행 방향에 수직이라 화면에서 거의 수평으로 보인다.
# _segment_lines 가 수평 성분을 버릴 때 따로 모아뒀다가 여기서 판별한다.
STOP_MAX_HEIGHT = 25          # 이보다 두꺼우면 정지선이 아니다 (px)
STOP_MIN_WIDTH = 60           # 이보다 짧으면 노면 파편으로 본다 (px)
STOP_MIN_ASPECT = 4.0         # 가로/세로 비. 정지선은 가로로 길다
STOP_MIN_FILL = 0.35          # 외곽 사각형 대비 채움비. 정지선은 속이 찬 막대다
# 도로 갓길 경계도 수평이라 모양만으로는 구분되지 않는다(drive6 프레임 2708 확인).
# 결정적 차이는 위치 — 정지선은 자차 차로 안에 있고 갓길 경계는 바깥이다.
STOP_MAX_LANE_WIDTH_RATIO = 1.8   # 차로 폭의 이 배를 넘게 길면 갓길/교차 도로로 본다
MAX_CENTER_JUMP = 90          # 직전 프레임 대비 이만큼 넘게 튀면 기각 (px)
REJECT_RESET_FRAMES = 5       # 연속 기각이 이만큼 이어지면 잠금을 풀고 새 값을 받는다

# ll 마스크는 짧은 끊김을 거의 메워서 나오므로(밀도만으론 점선/실선 구분 불가),
# 후보의 y구간 안에서 실제로 픽셀이 있는 행의 비율(fill ratio)로 구분한다.
# 이 값보다 낮으면(=y구간 안에 빈 구간이 있으면) 점선으로 본다.
DASH_FILL_RATIO_THRESHOLD = 0.85


class LaneEgoSelectDetector:
    def __init__(self, img_width, img_height, seg_model_path=None, seg_interval=1):
        self.img_width = img_width
        self.img_height = img_height
        self.img_center = img_width / 2.0

        self.left_fit = None
        self.right_fit = None

        # ego-lane 선택 상태 (시간적 연속성 + 디버그 시각화)
        self.prev_left_x = None
        self.prev_right_x = None
        self._reject_streak = 0
        self.last_candidates = []
        self.last_left = None
        self.last_right = None
        # 색상 결정용. 한쪽을 놓친 프레임은 직전 값을 유지한다 (fit 재사용과 동일 규칙)
        self.left_dashed = False
        self.right_dashed = False
        # 각 fit 이 실제로 피팅된 y 구간. fit 을 재사용할 때 함께 유지해서
        # 원래 데이터가 없던 곳까지 외삽해 그리는 것을 막는다.
        self.left_fit_range = None
        self.right_fit_range = None
        # fit 을 몇 프레임 연속으로 재사용 중인지 (MAX_FIT_REUSE_FRAMES 만료용)
        self._left_reuse = 0
        self._right_reuse = 0
        # 차선과 별개로 인식하는 정지선 (수평 성분에서 고른다)
        self._horizontals = []
        self.last_stop_lines = []

        # 차로 폭 모델 width(y) = m*y + b 의 표본. 좌우가 모두 잡힌 프레임에서만 쌓는다.
        self._width_samples = collections.deque(maxlen=WIDTH_HISTORY_LEN)
        self._width_model = None          # (m, b) 또는 None

        # Pure Pursuit 이 따라갈 경로들 (이미지 좌표계의 x = f(y) 다항식)
        self.ego_path = None              # 현재 차로 중심선
        self.ego_path_range = None        # 그려도 되는 y 구간
        self.neighbor_paths = []          # [(fit, (y_top, y_bottom), 'left'|'right')]
        self.ego_path_source = None       # 'both' | 'left_only' | 'right_only'
        # 차선 픽셀이 관측된 y 구간 — 이 밖으로는 그리지 않는다
        self.fit_y_range = (int(img_height * 0.5), img_height - 1)
        self.left_detect = False
        self.right_detect = False

        # Lane polynomial order settings
        self.left_right_fit_order = 2
        self.center_fit_order = 3

        # 노이즈 많은 컬러 마스크 기준으로 좁게 잡았던 값(0.85/0.07/0.49)은
        # CNN 차선 마스크의 75%를 잘라냈다(광각 뷰인 cam4는 97%).
        # 이제 마스크에 하늘·본네트가 없으므로 ROI는 지평선 가드 역할만 한다.
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

    def load_segmentation_model(self, model_path):
        if model_path is None:
            print("Segmentation model 경로가 지정되지 않았습니다.")
            return None

        model_path = os.path.join(current_dir, model_path) if not os.path.isabs(model_path) else model_path
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

    def color_mask(self, img_frame):
        hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge((h, s, v_clahe))

        lower_white = np.array([0, 0, 140])
        upper_white = np.array([179, 50, 255])
        lower_yellow = np.array([10, 80, 80])
        upper_yellow = np.array([40, 255, 255])

        white_mask = cv2.inRange(hsv_clahe, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv_clahe, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _run_segmentation(self, img_frame):
        """ONNX를 한 번 돌려 (lane_line, drivable_area) 마스크를 얻는다.

        두 출력 모두 softmax 이전의 logit이므로 채널 argmax로 이진화한다.
        (예전 코드처럼 logit에 `> 0.5`를 걸면 임계값이 의미를 갖지 못한다.)
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

        self.segment_lanes(img_frame)  # 캐시 갱신 (같은 추론 결과를 공유)
        return self._cached_drivable_mask

    def combine_masks(self, img_frame):
        """차선 마스크를 우선 쓰고, 비어 있을 때만 컬러 마스크로 보완한다.

        컬러 마스크 단독은 하늘·구름·본네트를 그대로 통과시키므로
        반드시 주행 가능 영역으로 잘라낸 뒤에 합친다.
        """
        lane_mask = self.segment_lanes(img_frame)

        if cv2.countNonZero(lane_mask) >= 300:
            return lane_mask

        # 폴백: 컬러 마스크 ∩ (팽창시킨) 주행 가능 영역
        road = cv2.dilate(
            self.drivable_mask(img_frame),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=1,
        )
        fallback = cv2.bitwise_and(self.color_mask(img_frame), road)
        return cv2.bitwise_or(lane_mask, fallback)

    # extract_center_points / fit_centerline 은 제거했다.
    # 마스크 각 행의 픽셀 평균으로 중심선을 만들던 것인데, 그건 ego 차로 중심이
    # 아니라 화면에 보이는 모든 차선의 무게중심이라 옆 차로가 보이기만 해도
    # 한쪽으로 끌려간다. 경로는 build_paths 가 좌우 경계에서 직접 만든다.

    def compute_heading_curvature(self, fit, y_eval):
        if fit is None:
            return None, None

        if fit.size == 3:
            a, b, _ = fit
            dx_dy = 2 * a * y_eval + b
            d2x_dy2 = 2 * a
        elif fit.size == 4:
            a, b, c, _ = fit
            dx_dy = 3 * a * y_eval ** 2 + 2 * b * y_eval + c
            d2x_dy2 = 6 * a * y_eval + 2 * b
        else:
            return None, None

        heading = np.arctan(dx_dy)
        curvature = abs(d2x_dy2) / ((1 + dx_dy ** 2) ** 1.5 + 1e-8)
        return heading, curvature

    def get_lane_center_x(self, y_eval):
        """주어진 y 에서 자차가 따라가야 할 x. Pure Pursuit 의 입력이 될 값이다.

        예전에는 fit_centerline(마스크 전체 픽셀의 행 평균)을 우선 썼는데, 그건
        ego 차로 중심이 아니라 화면에 보이는 모든 차선의 무게중심이라 옆 차로가
        보이기만 해도 한쪽으로 끌려간다. 경로로 쓸 수 없어 쓰지 않는다.
        """
        if self.ego_path is not None:
            return float(np.polyval(self.ego_path, y_eval))

        if self.left_fit is not None and self.right_fit is not None:
            left_x = np.polyval(self.left_fit, y_eval)
            right_x = np.polyval(self.right_fit, y_eval)
            return float((left_x + right_x) / 2.0)

        return None

    def limit_region(self, img_mask):
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

    def _segment_lines(self, mask):
        """연결 성분을 선분 후보로 만든다.

        각 성분에 직선을 맞춰 방향과 하단(y=h-1) x절편을 구한다.
        선형 피팅을 쓰는 이유는 외삽이 안정적이기 때문이다. 2차 곡선은
        화면 밖까지 밀어내면 발산해서 절편이 엉뚱하게 나온다.
        """
        h = mask.shape[0]
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        segments = []
        self._horizontals = []
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
                continue

            ys, xs = np.nonzero(labels == i)
            vx, vy, x0, y0 = cv2.fitLine(
                np.column_stack([xs, ys]).astype(np.float32),
                cv2.DIST_L2, 0, 0.01, 0.01
            ).ravel()

            # 수평에 가까운 성분은 하단까지 외삽해도 의미가 없다.
            # 전방 뷰의 차선은 항상 세로 성분을 가지므로 차선 후보에서는 뺀다.
            # 다만 정지선이 바로 이 성분이므로 버리지 말고 따로 모아둔다.
            if abs(vy) < 0.2:
                bx, by, bw, bh, area = stats[i]
                self._horizontals.append({
                    "box": (int(bx), int(by), int(bw), int(bh)),
                    "area": int(area),
                    "fill": float(area) / float(bw * bh) if bw and bh else 0.0,
                })
                continue

            slope = vx / vy                       # dx/dy
            x_bottom = x0 + slope * ((h - 1) - y0)
            segments.append({
                "xs": xs,
                "ys": ys,
                "slope": slope,
                "x_bottom": float(x_bottom),
                "angle": float(np.degrees(np.arctan(slope))),
            })

        return segments

    def _merge_segments(self, segments):
        """기울기와 하단 x절편이 비슷한 선분끼리 묶는다 (점선 → 하나의 차선)."""
        merged = []
        for seg in sorted(segments, key=lambda s: s["x_bottom"]):
            for group in merged:
                if (abs(seg["x_bottom"] - group["x_bottom"]) < MERGE_X_TOLERANCE
                        and abs(seg["angle"] - group["angle"]) < MERGE_ANGLE_TOLERANCE):
                    group["parts"].append(seg)
                    # 픽셀 수로 가중평균해 큰 조각의 방향을 더 믿는다
                    total = sum(p["xs"].size for p in group["parts"])
                    group["x_bottom"] = sum(
                        p["x_bottom"] * p["xs"].size for p in group["parts"]) / total
                    group["angle"] = sum(
                        p["angle"] * p["xs"].size for p in group["parts"]) / total
                    break
            else:
                merged.append({
                    "parts": [seg],
                    "x_bottom": seg["x_bottom"],
                    "angle": seg["angle"],
                })

        candidates = []
        for group in merged:
            xs = np.concatenate([p["xs"] for p in group["parts"]])
            ys = np.concatenate([p["ys"] for p in group["parts"]])
            if xs.size < MIN_LANE_PIXELS:
                continue

            y_span = int(ys.max() - ys.min()) + 1
            if y_span < MIN_LANE_Y_SPAN:
                continue

            fill_ratio = np.unique(ys).size / y_span
            candidates.append({
                "xs": xs,
                "ys": ys,
                "x_bottom": group["x_bottom"],
                "angle": group["angle"],
                "dashed": fill_ratio < DASH_FILL_RATIO_THRESHOLD,
            })

        return sorted(candidates, key=lambda c: c["x_bottom"])

    def lane_candidates(self, binary_mask):
        """마스크 → 차선 후보 리스트 (하단 x절편 오름차순)."""
        if binary_mask is None or binary_mask.size == 0:
            return []
        return self._merge_segments(self._segment_lines(binary_mask))

    def expected_width(self, y):
        """폭 모델 width(y). 표본이 부족하면 None."""
        if self._width_model is None:
            return None
        m, b = self._width_model
        return max(1.0, m * y + b)

    def _update_width_model(self, left, right):
        """좌우가 모두 잡힌 프레임에서만 (y, width) 표본을 쌓고 모델을 갱신한다.

        원근에서 폭은 y 에 대해 거의 선형이므로 width(y) = m*y + b 로 둔다.
        이상치(오선택 프레임)에 끌려가지 않도록 최소자승 대신 중앙값으로 맞춘다.
        """
        if left is None or right is None:
            return

        # 두 차선이 함께 관측된 구간에서만 잰다 (외삽한 폭은 믿을 수 없다)
        y_top = max(left["ys"].min(), right["ys"].min())
        y_bottom = min(left["ys"].max(), right["ys"].max())
        if y_bottom - y_top < 20:
            return

        try:
            lf = np.polyfit(left["ys"], left["xs"], 1)
            rf = np.polyfit(right["ys"], right["xs"], 1)
        except Exception:
            return

        for y in np.linspace(y_top, y_bottom, num=5):
            width = float(np.polyval(rf, y) - np.polyval(lf, y))
            if width > 0:
                self._width_samples.append((float(y), width))

        if len(self._width_samples) < MIN_WIDTH_SAMPLES:
            return

        ys = np.array([s[0] for s in self._width_samples])
        ws = np.array([s[1] for s in self._width_samples])

        # 중앙값 기반 기울기: y 를 두 덩어리로 갈라 각각의 중앙값을 잇는다.
        # 표본 절반이 오염돼도 버틴다.
        mid = np.median(ys)
        lo, hi = ys <= mid, ys > mid
        if lo.sum() >= 2 and hi.sum() >= 2:
            y_lo, w_lo = np.median(ys[lo]), np.median(ws[lo])
            y_hi, w_hi = np.median(ys[hi]), np.median(ws[hi])
            if y_hi - y_lo > 1e-6:
                m = (w_hi - w_lo) / (y_hi - y_lo)
                self._width_model = (m, w_lo - m * y_lo)
                return

        # y 가 한 곳에 몰려 기울기를 못 구하면 상수 폭으로 둔다
        self._width_model = (0.0, float(np.median(ws)))

    def _pair_cost(self, left, right):
        """(left, right) 쌍이 하나의 차로로서 얼마나 그럴듯한지. 낮을수록 좋다."""
        width = right["x_bottom"] - left["x_bottom"]
        if width <= 0:
            return None                              # 좌우가 뒤집힌 쌍

        cost = 0.0

        # 폭: 기대 폭 대비 상대 오차. 모델이 아직 없으면 이 항목은 건너뛴다.
        exp_w = self.expected_width(self.img_height - 1)
        if exp_w is not None:
            cost += PAIR_COST_W_WIDTH * abs(width - exp_w) / exp_w

        # 평행성: 좌우가 서로 벌어지거나 수렴하면 차로가 아니다.
        angle_diff = abs(left["angle"] - right["angle"])
        cost += PAIR_COST_W_PARALLEL * (angle_diff / 45.0)

        # 연속성: 직전 프레임 선택에서 멀수록 벌점.
        for cand, prev in ((left, self.prev_left_x), (right, self.prev_right_x)):
            if prev is not None:
                cost += PAIR_COST_W_CONTINUITY * (
                    abs(cand["x_bottom"] - prev) / (2.0 * MAX_CENTER_JUMP))

        # 근거량: 짧은 파편보다 길고 굵은 선을 믿는다.
        for cand in (left, right):
            span = float(cand["ys"].max() - cand["ys"].min()) + 1
            cost += PAIR_COST_W_EVIDENCE * (1.0 - min(1.0, span / (self.img_height * 0.4))) / 2

        return cost

    def select_ego_lane(self, candidates):
        """자차 중심을 사이에 둔 모든 쌍을 평가해 가장 그럴듯한 차로를 고른다.

        좌·우를 독립적으로 고르면(중심보다 왼쪽 중 가장 오른쪽 / 오른쪽 중 가장 왼쪽)
        두 선이 함께 말이 되는지 아무도 검사하지 않는다. 폭이 167px 이든 1155px 이든,
        두 선이 교차하든 통과해버린다. 그래서 쌍 단위로 비용을 매겨 고른다.

        쓸 만한 쌍이 없으면 (None, None) 을 돌려준다. 놓친 프레임을 직전 fit 으로
        메우는 처리는 fit_polynomial 쪽에 이미 있다.
        """
        best = None
        best_cost = None
        for left in candidates:
            if left["x_bottom"] >= self.img_center:
                continue
            for right in candidates:
                if right["x_bottom"] <= self.img_center:
                    continue
                cost = self._pair_cost(left, right)
                if cost is None:
                    continue
                if best_cost is None or cost < best_cost:
                    best, best_cost = (left, right), cost

        if best is None or best_cost > MAX_PAIR_COST:
            # 쓸 만한 쌍이 없다. 직전 값이 계속 안 맞는다는 뜻이므로
            # 연속되면 잠금을 풀어 다음 프레임에 새 차로를 받는다 (차로 변경 대응).
            self._reject_streak += 1
            if self._reject_streak >= REJECT_RESET_FRAMES:
                self.prev_left_x = None
                self.prev_right_x = None
                self._reject_streak = 0
            # 한쪽만 보이는 상황(커브에서 반대쪽 차선이 화면 밖으로 나감, 유도선 등)은
            # 정상이다. 쌍이 없다고 버리지 말고 한 줄이라도 살린다 —
            # 폭 모델로 반대쪽을 추정해 경로를 만들 수 있다.
            return self._select_single(candidates)

        self._reject_streak = 0
        left, right = best
        self.prev_left_x = left["x_bottom"]
        self.prev_right_x = right["x_bottom"]
        self._update_width_model(left, right)
        return left, right

    def _select_single(self, candidates):
        """쌍이 없을 때 한쪽 경계만이라도 고른다.

        직전 선택에 가깝고 근거가 많은 후보를 하나 고른 뒤, 자차 중심 기준으로
        왼쪽인지 오른쪽인지 판정해서 그 자리에 넣는다.
        """
        best = None
        best_cost = None
        for cand in candidates:
            is_left = cand["x_bottom"] < self.img_center
            prev = self.prev_left_x if is_left else self.prev_right_x

            cost = 0.0
            if prev is not None:
                cost += abs(cand["x_bottom"] - prev) / (2.0 * MAX_CENTER_JUMP)
            else:
                # 직전 값이 없으면 자차에 가까운 쪽을 선호한다
                cost += abs(cand["x_bottom"] - self.img_center) / self.img_width
            span = float(cand["ys"].max() - cand["ys"].min()) + 1
            cost += 1.0 - min(1.0, span / (self.img_height * 0.4))

            if best_cost is None or cost < best_cost:
                best, best_cost = cand, cost

        if best is None:
            return None, None

        if best["x_bottom"] < self.img_center:
            self.prev_left_x = best["x_bottom"]
            return best, None
        self.prev_right_x = best["x_bottom"]
        return None, best

    def fit_polynomial(self, binary_mask):
        """ego lane 두 경계를 골라 그 픽셀만으로 다항식을 피팅한다."""
        candidates = self.lane_candidates(binary_mask)
        left, right = self.select_ego_lane(candidates)

        # 디버그 시각화용으로 남겨둔다
        self.last_candidates = candidates
        self.last_left = left
        self.last_right = right

        def fit(cand):
            if cand is None or cand["ys"].size < MIN_LANE_PIXELS:
                return None
            # y 방향으로 충분히 퍼져 있어야 2차 피팅이 의미를 갖는다
            order = self.left_right_fit_order if np.ptp(cand["ys"]) > 20 else 1
            return np.polyfit(cand["ys"], cand["xs"], order)

        left_fit = fit(left)
        right_fit = fit(right)

        self.left_detect = left_fit is not None
        self.right_detect = right_fit is not None

        # 점선/실선 색상 결정용. 이번 프레임에 후보가 잡혔을 때만 갱신하고,
        # 놓친 프레임은 직전 값을 유지한다 (fit 재사용과 동일 규칙)
        if left is not None:
            self.left_dashed = left["dashed"]
        if right is not None:
            self.right_dashed = right["dashed"]

        # 새로 피팅했을 때만 그 fit 이 커버하는 y 구간을 갱신한다.
        # 재사용되는 fit 은 예전 구간을 그대로 들고 있어야 외삽 제한이 걸린다.
        if left_fit is not None:
            self.left_fit_range = (int(left["ys"].min()), int(left["ys"].max()))
        if right_fit is not None:
            self.right_fit_range = (int(right["ys"].min()), int(right["ys"].max()))

        # 한쪽을 놓친 프레임은 직전 값을 유지하되, 계속 놓치면 버린다.
        if left_fit is None:
            self._left_reuse += 1
            if self._left_reuse <= MAX_FIT_REUSE_FRAMES:
                left_fit = self.left_fit
            else:
                self.left_fit_range = None
        else:
            self._left_reuse = 0

        if right_fit is None:
            self._right_reuse += 1
            if self._right_reuse <= MAX_FIT_REUSE_FRAMES:
                right_fit = self.right_fit
            else:
                self.right_fit_range = None
        else:
            self._right_reuse = 0

        self.left_fit = left_fit
        self.right_fit = right_fit
        self.fit_y_range = self._observed_y_range(left, right)
        self.last_stop_lines = self.detect_stop_lines(left_fit, right_fit)
        self.build_paths(candidates, left_fit, right_fit)
        return left_fit, right_fit

    @staticmethod
    def _shift_fit(fit, dx):
        """다항식 x = f(y) 를 x 방향으로 dx 만큼 평행이동한다 (상수항만 바뀐다)."""
        shifted = np.array(fit, dtype=float).copy()
        shifted[-1] += dx
        return shifted

    def _offset_fit_by_halfwidth(self, fit, y_range, sign):
        """경계선을 폭의 절반만큼 옆으로 밀어 중심선을 만든다.

        원근 때문에 폭이 y 마다 다르므로 일정 offset 을 줄 수 없다. 구간 안에서
        width(y)/2 를 여러 y 에 대해 계산한 뒤 그 점들을 다시 피팅한다.
        sign=+1 이면 오른쪽으로, -1 이면 왼쪽으로 민다.
        """
        if fit is None or y_range is None or self._width_model is None:
            return None

        ys = np.linspace(y_range[0], y_range[1], num=30)
        xs = []
        for y in ys:
            half = self.expected_width(y) / 2.0
            xs.append(float(np.polyval(fit, y)) + sign * half)
        try:
            return np.polyfit(ys, np.array(xs), min(self.left_right_fit_order, 2))
        except Exception:
            return None

    def build_paths(self, candidates, left_fit, right_fit):
        """Pure Pursuit 이 따라갈 경로를 만든다.

        - 좌우가 모두 있으면 중심선 = (left + right) / 2
        - 한쪽만 있으면 폭 모델로 반대쪽을 추정해 그 절반만큼 안쪽으로 민다
          (유도선처럼 선이 하나만 보이는 구간)
        - ego 차로 경계가 점선이면 그쪽 옆 차로는 넘어갈 수 있으므로 경로를 하나 더 만든다.
          실선 쪽은 넘어갈 수 없으니 만들지 않는다 (중앙선 침범 방지).
        """
        self.ego_path = None
        self.ego_path_range = None
        self.ego_path_source = None
        self.neighbor_paths = []

        left_range = self._drawable_range(left_fit, self.left_fit_range)
        right_range = self._drawable_range(right_fit, self.right_fit_range)

        if left_fit is not None and right_fit is not None and left_range and right_range:
            y_top = max(left_range[0], right_range[0])
            y_bottom = min(left_range[1], right_range[1])
            if y_bottom - y_top >= 10:
                ys = np.linspace(y_top, y_bottom, num=30)
                xs = (np.polyval(left_fit, ys) + np.polyval(right_fit, ys)) / 2.0
                try:
                    self.ego_path = np.polyfit(ys, xs, self.left_right_fit_order)
                    self.ego_path_range = (y_top, y_bottom)
                    self.ego_path_source = "both"
                except Exception:
                    pass
        elif left_fit is not None and left_range:
            path = self._offset_fit_by_halfwidth(left_fit, left_range, +1)
            if path is not None:
                self.ego_path, self.ego_path_range = path, left_range
                self.ego_path_source = "left_only"
        elif right_fit is not None and right_range:
            path = self._offset_fit_by_halfwidth(right_fit, right_range, -1)
            if path is not None:
                self.ego_path, self.ego_path_range = path, right_range
                self.ego_path_source = "right_only"

        # 넘어갈 수 있는 옆 차로 (경계가 점선인 쪽만)
        for fit, rng, dashed, side, sign in (
                (left_fit, left_range, self.left_dashed, "left", -1),
                (right_fit, right_range, self.right_dashed, "right", +1)):
            if fit is None or rng is None or not dashed:
                continue

            outer = self._outer_candidate(candidates, fit, rng, side)
            if outer is not None:
                ys = np.linspace(rng[0], rng[1], num=30)
                xs = (np.polyval(fit, ys) + np.polyval(outer, ys)) / 2.0
                try:
                    self.neighbor_paths.append((np.polyfit(ys, xs, 2), rng, side))
                    continue
                except Exception:
                    pass

            # 바깥 경계가 안 보이면 폭 모델로 밀어서 만든다
            path = self._offset_fit_by_halfwidth(fit, rng, sign)
            if path is not None:
                self.neighbor_paths.append((path, rng, side))

    def _outer_candidate(self, candidates, boundary_fit, y_range, side):
        """ego 경계 바깥에 있는 다음 차선을 찾는다 (옆 차로의 반대편 경계).

        폭 모델로 추정하는 것보다 실제로 보이는 선을 쓰는 편이 정확하다.
        """
        y_ref = y_range[1]
        boundary_x = float(np.polyval(boundary_fit, y_ref))
        exp_w = self.expected_width(y_ref)
        if exp_w is None:
            return None

        best = None
        best_gap = None
        for cand in candidates:
            try:
                cand_fit = np.polyfit(cand["ys"], cand["xs"],
                                      1 if np.ptp(cand["ys"]) <= 20 else 2)
            except Exception:
                continue
            gap = float(np.polyval(cand_fit, y_ref)) - boundary_x
            if side == "left":
                gap = -gap                     # 왼쪽은 바깥이 더 작은 x
            # 옆 차로 하나만큼 떨어져 있어야 한다 (폭의 0.5~1.5배)
            if not (0.5 * exp_w <= gap <= 1.5 * exp_w):
                continue
            if best_gap is None or gap < best_gap:
                best, best_gap = cand_fit, gap
        return best

    def detect_stop_lines(self, left_fit, right_fit):
        """수평 성분 중에서 정지선을 골라낸다.

        모양(얇고 가로로 길고 속이 참)만으로는 도로 갓길 경계와 구분되지 않는다.
        갓길 경계도 전방에서는 수평으로 보이기 때문이다. 결정적인 차이는 위치 —
        정지선은 자차 차로 좌우 경계 사이에 놓이고, 갓길 경계는 그 바깥이다.
        그래서 ego lane 을 먼저 구한 뒤에 호출해야 한다.
        """
        if left_fit is None or right_fit is None:
            return []                      # 차로를 모르면 위치 검증을 못 한다

        found = []
        for cand in getattr(self, "_horizontals", []):
            bx, by, bw, bh = cand["box"]
            if (bh > STOP_MAX_HEIGHT or bw < STOP_MIN_WIDTH
                    or bw < bh * STOP_MIN_ASPECT or cand["fill"] < STOP_MIN_FILL):
                continue

            cy = by + bh / 2.0
            lane_left = float(np.polyval(left_fit, cy))
            lane_right = float(np.polyval(right_fit, cy))
            if lane_right < lane_left:
                lane_left, lane_right = lane_right, lane_left
            lane_width = lane_right - lane_left
            if lane_width <= 0:
                continue

            # 중심이 차로 안에 있고, 차로 폭에서 크게 벗어나지 않아야 한다.
            center_x = bx + bw / 2.0
            if not (lane_left <= center_x <= lane_right):
                continue
            if bw > lane_width * STOP_MAX_LANE_WIDTH_RATIO:
                continue

            found.append(cand["box"])

        return found

    def _observed_y_range(self, left, right):
        """실제로 차선 픽셀이 관측된 y 구간.

        차선은 소실점으로 수렴하므로 데이터 위쪽으로 외삽하면 소실점을 지나
        좌우가 뒤집힌다. 그래서 두 차선이 모두 관측된 구간에서만 그린다.
        """
        ranges = [(c["ys"].min(), c["ys"].max()) for c in (left, right) if c is not None]
        if not ranges:
            return self.fit_y_range          # 이번 프레임 관측 없음 → 직전 범위 유지

        # 양쪽 모두 있으면 겹치는 구간만 (외삽 없이 보간만 하도록)
        y_top = max(r[0] for r in ranges)
        y_bottom = min(r[1] for r in ranges)
        if y_bottom - y_top < 20:            # 겹침이 거의 없으면 합집합으로 완화
            y_top = min(r[0] for r in ranges)
            y_bottom = max(r[1] for r in ranges)

        return int(y_top), int(min(y_bottom, self.img_height - 1))

    def draw_candidates(self, img_input):
        """차선 후보를 색깔별로, 선택된 ego lane 경계를 굵게 표시한다.

        선택이 틀렸을 때 후보 자체가 없는 건지, 있는데 잘못 고른 건지
        바로 구분하려고 만든 디버그 뷰다.
        """
        out = img_input.copy()
        palette = [(0, 165, 255), (255, 0, 255), (0, 255, 255),
                   (255, 255, 0), (128, 0, 255), (0, 128, 255)]

        for idx, cand in enumerate(self.last_candidates):
            out[cand["ys"], cand["xs"]] = palette[idx % len(palette)]
            cv2.circle(out, (int(cand["x_bottom"]), self.img_height - 5), 5,
                       palette[idx % len(palette)], -1)

        for cand, color, label in ((self.last_left, (0, 255, 0), "L"),
                                   (self.last_right, (0, 0, 255), "R")):
            if cand is None:
                continue
            out[cand["ys"], cand["xs"]] = color
            x_bottom = int(cand["x_bottom"])
            cv2.circle(out, (x_bottom, self.img_height - 5), 9, color, 2)
            cv2.putText(out, label, (x_bottom - 6, self.img_height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 자차 중심 기준선
        cx = int(self.img_center)
        cv2.line(out, (cx, self.img_height - 30), (cx, self.img_height - 1), (255, 255, 255), 1)
        cv2.putText(out, f"cand={len(self.last_candidates)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def _drawable_range(self, fit, fit_range):
        """이 fit 을 그려도 되는 y 구간. 없으면 None.

        관측 구간(fit_y_range)과 이 fit 이 실제로 피팅된 구간을 교집합으로 잡는다.
        재사용 중인 fit 은 예전 구간을 들고 있으므로 여기서 자동으로 짧게 잘린다.
        """
        if fit is None:
            return None

        y_top, y_bottom = self.fit_y_range
        if fit_range is not None:
            y_top = max(y_top, fit_range[0] - MAX_FIT_EXTRAPOLATION)
            y_bottom = min(y_bottom, fit_range[1] + MAX_FIT_EXTRAPOLATION)

        if y_bottom - y_top < 10:        # 남는 구간이 거의 없으면 그리지 않는다
            return None
        return y_top, y_bottom

    @staticmethod
    def _curve_points(fit, y_range):
        if fit is None or y_range is None:
            return []
        return [(int(np.polyval(fit, y)), int(y))
                for y in np.linspace(y_range[0], y_range[1], num=100)]

    @staticmethod
    def _draw_polyline(img, points, color, thickness, dashed=False):
        """점선 여부를 색이 아니라 선 스타일로 표현한다 (색 가짓수를 줄이려고)."""
        if not points:
            return
        pts = np.array(points, dtype=np.int32)
        if not dashed:
            cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)
            return
        step = 8
        for i in range(0, len(pts) - step, step * 2):
            cv2.polylines(img, [pts[i:i + step + 1]], isClosed=False,
                          color=color, thickness=thickness)

    def draw_lane(self, img_input, left_fit, right_fit):
        """결과 패널. Pure Pursuit 이 따라갈 경로가 주인공이다.

        주행 가능 영역 채움은 그리지 않는다 — Pure Pursuit 은 면적이 아니라 선을
        따라가므로 필요 없고, 화면만 어지럽힌다.
        색은 4개로 제한한다: 경계=흰색, 주행 경로=초록, 넘어갈 수 있는 옆 차로=노랑,
        정지선=하늘색. 점선/실선은 색 대신 선 스타일로 구분한다.
        """
        out_img = img_input.copy()

        BOUNDARY_COLOR = (255, 255, 255)
        EGO_PATH_COLOR = (0, 255, 0)
        NEIGHBOR_PATH_COLOR = (0, 255, 255)
        STOP_COLOR = (255, 255, 0)

        # 관측된 구간에서만, 그리고 각 fit 이 실제 데이터를 가진 구간에서만 그린다.
        # 위쪽으로 외삽하면 소실점을 지나 좌우가 뒤집히고, 재사용 fit 을 원래
        # 범위 밖까지 늘이면 휘어서 반대편 차선을 침범한다.
        left_range = self._drawable_range(left_fit, self.left_fit_range)
        right_range = self._drawable_range(right_fit, self.right_fit_range)
        self._draw_polyline(out_img, self._curve_points(left_fit, left_range),
                            BOUNDARY_COLOR, 2, dashed=self.left_dashed)
        self._draw_polyline(out_img, self._curve_points(right_fit, right_range),
                            BOUNDARY_COLOR, 2, dashed=self.right_dashed)

        # 넘어갈 수 있는 옆 차로 경로 (경계가 점선인 쪽에만 생긴다)
        for path, rng, _side in self.neighbor_paths:
            self._draw_polyline(out_img, self._curve_points(path, rng),
                                NEIGHBOR_PATH_COLOR, 3)

        # 현재 차로 경로 — 가장 굵게. 마지막에 그려 위로 올린다.
        self._draw_polyline(out_img, self._curve_points(self.ego_path, self.ego_path_range),
                            EGO_PATH_COLOR, 6)

        for bx, by, bw, bh in self.last_stop_lines:
            cv2.rectangle(out_img, (bx, by), (bx + bw, by + bh), STOP_COLOR, 3)
            cv2.putText(out_img, "STOP", (bx, max(14, by - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, STOP_COLOR, 2)

        # 한쪽 경계만으로 추정한 경로인지 표시해준다 (신뢰도가 다르다)
        if self.ego_path_source in ("left_only", "right_only"):
            cv2.putText(out_img, f"path: {self.ego_path_source}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, EGO_PATH_COLOR, 2)

        h = out_img.shape[0]
        cv2.putText(out_img, "ego path", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, EGO_PATH_COLOR, 2)
        cv2.putText(out_img, "lane change", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEIGHBOR_PATH_COLOR, 2)
        return out_img


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

            display_img = cv2.resize(image, (640, 480))
            with frame_lock:
                latest_frame = display_img

        except (AttributeError, ValueError, OSError, cv2.error) as ex:
            print(f"[camera_thread_worker] recoverable error: {ex}")
            traceback.print_exc()
            time.sleep(0.1)
            continue
        except Exception as ex:
            print(f"[camera_thread_worker] unexpected error: {ex}")
            traceback.print_exc()
            raise


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
    """파일 경로면 그 한 장, 폴더면 안의 이미지 전부를 정렬해서 반환."""
    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if name.lower().endswith(IMG_EXTS)
        ]

    return []


# cv2 창 제목은 Windows에서 UTF-8을 못 받아 한글이 깨진다. ASCII로 둔다.
WINDOW_TITLE = "LaneEgoSelect Offline"
# 패널 순서 = 원본 / ROI 마스크 / 후보 / 결과
PANEL_LABELS = ("1 source", "2 ROI mask", "3 candidates", "4 result")


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


# 창 테두리·제목표시줄·작업표시줄을 뺀 실제로 쓸 수 있는 영역
_SCREEN_W, _SCREEN_H = _screen_size()
MAX_PANEL_WIDTH = _SCREEN_W - 80
MAX_PANEL_HEIGHT = _SCREEN_H - 160


def _label_panel(img, text):
    """패널 우상단에 이름표. 좌상단은 draw_candidates 의 cand= 표시가 쓴다."""
    out = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = out.shape[1] - tw - 10
    cv2.rectangle(out, (x - 6, 4), (out.shape[1] - 2, th + 14), (0, 0, 0), -1)
    cv2.putText(out, text, (x, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def compose_panel(views, fit_screen=True):
    """패널 4장을 2x2로 묶는다.

    가로 4장(640*4 = 2560px)은 1920 모니터에서 네 번째 패널이 통째로 잘린다.
    2x2 는 같은 면적을 절반 폭에 담으므로 축소율이 훨씬 덜하다.

    fit_screen=False 면 축소하지 않고 원본 크기(1280x960)로 돌려준다.
    파일로 저장할 때는 모니터 해상도에 맞출 이유가 없다.
    """
    labeled = [_label_panel(v, t) for v, t in zip(views, PANEL_LABELS)]
    grid = np.vstack([np.hstack(labeled[:2]), np.hstack(labeled[2:])])

    if not fit_screen:
        return grid

    scale = min(MAX_PANEL_WIDTH / grid.shape[1], MAX_PANEL_HEIGHT / grid.shape[0], 1.0)
    if scale < 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grid


def open_window():
    """사용자가 크기를 조절할 수 있는 창 (기본 AUTOSIZE 는 고정이라 잘려도 못 줄인다)."""
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)


def process_frame(detector, image, fit_screen=True):
    """한 프레임을 파이프라인에 통과시키고 (4패널 화면, 요약문자열)을 돌려준다."""
    img_frame = cv2.resize(image, (640, 480))
    combined_mask = detector.combine_masks(img_frame)
    roi_mask = detector.limit_region(combined_mask)
    left_fit, right_fit = detector.fit_polynomial(roi_mask)
    img_result = detector.draw_lane(img_frame, left_fit, right_fit)

    panel = compose_panel([
        img_frame,
        cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR),
        detector.draw_candidates(img_frame),
        img_result,
    ], fit_screen=fit_screen)

    left = detector.last_left
    right = detector.last_right
    left_x = f"{left['x_bottom']:.0f}" if left else "None"
    right_x = f"{right['x_bottom']:.0f}" if right else "None"
    width = (f"{right['x_bottom'] - left['x_bottom']:.0f}"
             if (left and right) else "-")
    path = detector.ego_path_source or "없음"
    info = (f"후보={len(detector.last_candidates)}  "
            f"L={left_x}  R={right_x}  폭={width}  경로={path}")
    if detector.neighbor_paths:
        info += f"  차선변경={len(detector.neighbor_paths)}"
    if detector.last_stop_lines:
        info += f"  정지선={len(detector.last_stop_lines)}"
    return panel, info


def _new_detector():
    # 오프라인은 실시간 제약이 없으므로 매 프레임 추론한다.
    return LaneEgoSelectDetector(640, 480, seg_model_path="lane_segmentation.onnx",
                                 seg_interval=1)


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
        # 입력을 받을 수 없는 환경(백그라운드 실행 등)에서 10분치 인코딩을
        # 그냥 버리는 게 더 나쁘다. 남겨두고 그렇게 했다고 알린다.
        answer = "y"
        print("  (입력을 받을 수 없어 일단 저장합니다)")

    if answer in ("y", "yes", "ㅛ"):
        shutil.move(write_path, save_path)
        print(f"💾 저장 완료: {save_path}")
    else:
        os.remove(write_path)
        print("  저장하지 않았습니다.")


def run_video(path, save_path=None, preview=True):
    """녹화된 주행 영상으로 파이프라인을 확인한다.

    space 일시정지/재개, n 한 프레임 진행, s 현재 화면 저장, ESC/q 종료.
    save_path 를 주면 4패널 화면을 그대로 영상 파일로 저장한다.
    preview=False 면 창을 띄우지 않는다 (저장만 할 때 더 빠르다).
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

    # 저장할 때는 모니터에 맞춰 줄이지 않고 원본 2x2 크기로 남긴다.
    writer = None

    # 창을 띄우는 경우에는 먼저 보여주고 끝난 뒤에 저장할지 물어본다.
    # 다시 처리하면 전체 추론을 또 돌려야 하므로(2772프레임에 ~9분), 일단
    # 임시 파일에 써 두고 "아니오"면 지운다.
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
        """한 프레임 처리 → (표시용 패널, 로그). 저장이 켜져 있으면 파일에도 쓴다."""
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
        # 한 프레임도 못 쓴 경우(에러로 중단 등)는 물어볼 것도 없다
        if writer is not None and idx > 0 and os.path.exists(write_path):
            _finish_save(write_path, save_path, ask_before_save, idx, fps)


def run_images(paths):
    """이미지 파일/폴더로 파이프라인을 확인한다.

    n/→ 다음, p/← 이전, s 저장, ESC/q 종료.
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

            panel, info = process_frame(detector, image)
            print(f"[{index + 1}/{len(paths)}] {os.path.basename(path)}  {info}")
            cv2.imshow(WINDOW_TITLE, panel)

            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q")):
                break
            elif key in (ord("n"), 83):  # 83 = →
                index = (index + 1) % len(paths)
            elif key in (ord("p"), 81):  # 81 = ←
                index = (index - 1) % len(paths)
            elif key == ord("s"):
                out_path = f"{os.path.splitext(path)[0]}_hybrid.png"
                cv2.imwrite(out_path, panel)
                print(f"  저장: {out_path}")

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cv2.destroyAllWindows()


def run_offline(source, save_path=None, preview=True):
    """시뮬레이터 UDP 없이 영상 또는 이미지로 파이프라인을 확인한다."""
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


def main(cam_ip=IP, cam_port=PORT):
    t = threading.Thread(target=camera_thread_worker, args=(cam_ip, cam_port), daemon=True)
    t.start()
    time.sleep(1)

    # CPU 추론이 ~180ms라 매 프레임 돌리면 5~6 FPS가 한계다.
    # 2프레임마다 추론하고 사이 프레임은 직전 마스크를 재사용해 표시 프레임률을 확보한다.
    detector = LaneEgoSelectDetector(640, 480, seg_model_path="lane_segmentation.onnx",
                                   seg_interval=2)

    print("=" * 50)
    print("🚀 실시간 하이브리드(딥러닝 차선 + 컬러 폴백) 차선 인식이 실행되었습니다.")
    print(f"   카메라 {cam_ip}:{cam_port}")
    print("종료하려면 결과 창을 선택하고 ESC 키를 누르세요.")
    print("=" * 50)

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                img_frame = latest_frame.copy()

            combined_mask = detector.combine_masks(img_frame)
            roi_mask = detector.limit_region(combined_mask)
            left_fit, right_fit = detector.fit_polynomial(roi_mask)
            img_result = detector.draw_lane(img_frame, left_fit, right_fit)

            cv2.imshow("LaneSegHybrid Result", img_result)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="하이브리드 차선 인식 (기본: 시뮬레이터 UDP 실시간)")
    parser.add_argument(
        "--source", nargs="?", const="samples", default=None,
        help="오프라인 모드. 이미지 파일 또는 폴더 경로 (값 없이 주면 samples 폴더)")
    parser.add_argument("--cam-ip", default=IP,
                        help=f"시뮬레이터 카메라 IP (기본 {IP}, 환경변수 MORAI_CAM_IP)")
    parser.add_argument("--cam-port", type=int, default=PORT,
                        help=f"카메라 포트 (기본 {PORT}, 환경변수 MORAI_CAM_PORT)")
    parser.add_argument("--save", metavar="OUT.mp4", default=None,
                        help="4패널 결과를 영상 파일로 저장한다 (--source 가 영상일 때)")
    parser.add_argument("--no-preview", action="store_true",
                        help="창을 띄우지 않는다 (저장만 할 때 더 빠르다)")
    args = parser.parse_args()

    if args.source:
        run_offline(args.source, save_path=args.save, preview=not args.no_preview)
    else:
        main(args.cam_ip, args.cam_port)
