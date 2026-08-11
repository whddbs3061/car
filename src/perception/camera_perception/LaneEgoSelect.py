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
MAX_CENTER_JUMP = 90          # 직전 프레임 대비 이만큼 넘게 튀면 기각 (px)
REJECT_RESET_FRAMES = 5       # 연속 기각이 이만큼 이어지면 잠금을 풀고 새 값을 받는다


class LaneEgoSelectDetector:
    def __init__(self, img_width, img_height, seg_model_path=None, seg_interval=1):
        self.img_width = img_width
        self.img_height = img_height
        self.img_center = img_width / 2.0

        self.left_fit = None
        self.right_fit = None
        self.center_fit = None

        # ego-lane 선택 상태 (시간적 연속성 + 디버그 시각화)
        self.prev_left_x = None
        self.prev_right_x = None
        self._reject_streak = 0
        self.last_candidates = []
        self.last_left = None
        self.last_right = None
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

    def extract_center_points(self, binary_mask, step=5):
        h, w = binary_mask.shape[:2]
        top = int(h * (1.0 - self.trap_height))
        center_x = []
        center_y = []

        for y in range(h - 1, top - 1, -step):
            row = binary_mask[y]
            xs = np.where(row > 0)[0]
            if xs.size > 0:
                center_x.append(int(np.mean(xs)))
                center_y.append(y)

        if len(center_x) < 10:
            return np.array([]), np.array([])
        return np.array(center_x), np.array(center_y)

    def fit_centerline(self, binary_mask, order=None):
        center_x, center_y = self.extract_center_points(binary_mask)
        if center_x.size < 10 or center_y.size < 10:
            return None
        if order is None:
            order = self.center_fit_order
        try:
            return np.polyfit(center_y, center_x, order)
        except Exception as ex:
            print(f"centerline fit 실패: {ex}")
            traceback.print_exc()
            return None

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
        if self.center_fit is not None:
            return float(np.polyval(self.center_fit, y_eval))

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
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < MIN_COMPONENT_AREA:
                continue

            ys, xs = np.nonzero(labels == i)
            vx, vy, x0, y0 = cv2.fitLine(
                np.column_stack([xs, ys]).astype(np.float32),
                cv2.DIST_L2, 0, 0.01, 0.01
            ).ravel()

            # 수평에 가까운 성분은 하단까지 외삽해도 의미가 없다.
            # 전방 뷰의 차선은 항상 세로 성분을 가지므로 안전하게 버린다.
            if abs(vy) < 0.2:
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
            candidates.append({
                "xs": xs,
                "ys": ys,
                "x_bottom": group["x_bottom"],
                "angle": group["angle"],
            })

        return sorted(candidates, key=lambda c: c["x_bottom"])

    def lane_candidates(self, binary_mask):
        """마스크 → 차선 후보 리스트 (하단 x절편 오름차순)."""
        if binary_mask is None or binary_mask.size == 0:
            return []
        return self._merge_segments(self._segment_lines(binary_mask))

    def select_ego_lane(self, candidates):
        """자차 중심 기준으로 좌·우에서 가장 가까운 차선을 고른다.

        차선들이 위로 수렴하므로 하단 x절편이 차선 간 순서를 가장 잘 구분한다.
        직전 프레임 대비 크게 튀는 선택은 기각하고 이전 값을 유지한다
        (차로 변경은 따라가야 하므로 임계값을 두고 히스테리시스로 처리).
        """
        left = None
        right = None
        for cand in candidates:                      # x_bottom 오름차순
            if cand["x_bottom"] < self.img_center:
                left = cand                          # 중심 왼쪽 중 가장 오른쪽
            elif right is None:
                right = cand                         # 중심 오른쪽 중 가장 왼쪽
                break

        rejected = False
        if left is not None and self.prev_left_x is not None:
            if abs(left["x_bottom"] - self.prev_left_x) > MAX_CENTER_JUMP:
                left = None
                rejected = True
        if right is not None and self.prev_right_x is not None:
            if abs(right["x_bottom"] - self.prev_right_x) > MAX_CENTER_JUMP:
                right = None
                rejected = True

        # 기각이 계속 이어지면 직전 값이 더 이상 유효하지 않다는 뜻이다.
        # (실제 차로 변경, 카메라 전환 등) 잠금을 풀어 다음 프레임부터 새로 잡는다.
        if rejected:
            self._reject_streak += 1
            if self._reject_streak >= REJECT_RESET_FRAMES:
                self.prev_left_x = None
                self.prev_right_x = None
                self._reject_streak = 0
        else:
            self._reject_streak = 0

        if left is not None:
            self.prev_left_x = left["x_bottom"]
        if right is not None:
            self.prev_right_x = right["x_bottom"]

        return left, right

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

        # 한쪽을 놓친 프레임은 직전 값을 유지한다
        if left_fit is None and self.left_fit is not None:
            left_fit = self.left_fit
        if right_fit is None and self.right_fit is not None:
            right_fit = self.right_fit

        self.left_fit = left_fit
        self.right_fit = right_fit
        self.fit_y_range = self._observed_y_range(left, right)
        self.center_fit = self.fit_centerline(binary_mask, order=3)
        return left_fit, right_fit

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

    def draw_lane(self, img_input, left_fit, right_fit):
        overlay = img_input.copy()
        out_img = img_input.copy()

        # 관측된 구간에서만 그린다. 위쪽으로 외삽하면 소실점을 지나 좌우가 뒤집힌다.
        y_top, y_bottom = self.fit_y_range
        ploty = np.linspace(y_top, y_bottom, num=100)
        left_points = []
        right_points = []

        if left_fit is not None:
            left_points = [(int(np.polyval(left_fit, y)), int(y)) for y in ploty]
        if right_fit is not None:
            right_points = [(int(np.polyval(right_fit, y)), int(y)) for y in ploty]

        if left_points and right_points:
            pts = np.array(left_points + right_points[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)

        if left_points:
            cv2.polylines(out_img, [np.array(left_points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=5)
        if right_points:
            cv2.polylines(out_img, [np.array(right_points, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=5)

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


WINDOW_TITLE = "LaneEgoSelect Offline (원본 | ROI | 후보 | 결과)"


def process_frame(detector, image):
    """한 프레임을 파이프라인에 통과시키고 (4패널 화면, 요약문자열)을 돌려준다."""
    img_frame = cv2.resize(image, (640, 480))
    combined_mask = detector.combine_masks(img_frame)
    roi_mask = detector.limit_region(combined_mask)
    left_fit, right_fit = detector.fit_polynomial(roi_mask)
    img_result = detector.draw_lane(img_frame, left_fit, right_fit)

    panel = np.hstack([
        img_frame,
        cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR),
        detector.draw_candidates(img_frame),
        img_result,
    ])

    left = detector.last_left
    right = detector.last_right
    left_x = f"{left['x_bottom']:.0f}" if left else "None"
    right_x = f"{right['x_bottom']:.0f}" if right else "None"
    width = (f"{right['x_bottom'] - left['x_bottom']:.0f}"
             if (left and right) else "-")
    info = (f"후보={len(detector.last_candidates)}  "
            f"L={left_x}  R={right_x}  폭={width}")
    return panel, info


def _new_detector():
    # 오프라인은 실시간 제약이 없으므로 매 프레임 추론한다.
    return LaneEgoSelectDetector(640, 480, seg_model_path="lane_segmentation.onnx",
                                 seg_interval=1)


def run_video(path):
    """녹화된 주행 영상으로 파이프라인을 확인한다.

    space 일시정지/재개, n 한 프레임 진행, s 현재 화면 저장, ESC/q 종료.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[에러] 영상을 열 수 없습니다: {path}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = _new_detector()

    print("=" * 60)
    print(f"🎬 영상 모드 — {os.path.basename(path)} ({total}프레임)")
    print("space: 일시정지  |  n: 한 프레임  |  s: 화면 저장  |  ESC/q: 종료")
    print("=" * 60)

    idx = 0
    paused = False
    try:
        while True:
            if not paused:
                ok, image = cap.read()
                if not ok:
                    print("\n영상 끝.")
                    break
                panel, info = process_frame(detector, image)
                idx += 1
                if idx % 10 == 0 or idx == 1:
                    print(f"[{idx}/{total}] {info}")
                cv2.imshow(WINDOW_TITLE, panel)

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
                panel, info = process_frame(detector, image)
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
        cv2.destroyAllWindows()


def run_images(paths):
    """이미지 파일/폴더로 파이프라인을 확인한다.

    n/→ 다음, p/← 이전, s 저장, ESC/q 종료.
    """
    detector = _new_detector()

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


def run_offline(source):
    """시뮬레이터 UDP 없이 영상 또는 이미지로 파이프라인을 확인한다."""
    target = resolve_target(source)

    if os.path.isfile(target) and target.lower().endswith(VIDEO_EXTS):
        run_video(target)
        return

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
    args = parser.parse_args()

    if args.source:
        run_offline(args.source)
    else:
        main(args.cam_ip, args.cam_port)
