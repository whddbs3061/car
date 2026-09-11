"""차선 검출 - 클래스별 BEV + 히스토그램 시드 + 슬라이딩 윈도우 + RANSAC + 추적.

**이 파일은 검출과 후처리만 한다.** 그림 그리기(lane_viz.py)와 파일 저장은
들어 있지 않다 - 실주행 경로(live_output.py)가 시각화 코드를 끌고 들어가지
않게 하려는 것이다.

    lane_detection.py   검출 + 후처리 + 제어용 파생값   <- 이 파일
    lane_viz.py         그림
    offline_test.py     녹화 -> png/mp4
    live_overlay.py     실시간 -> 화면
    live_output.py      실시간 -> 값만 (viz 를 import 하지 않는다)

--------------------------------------------------------------------------
파이프라인
--------------------------------------------------------------------------
    원본 1280x720
      -> crop_top(260)         하늘 제거              1280x460
      -> 640x256, ImageNet 정규화                     (학습과 동일해야 함)
      -> 세그멘테이션 5클래스
      -> 보닛 마스크로 노이즈 제거
      -> BEV 변환                                     자차 좌표계 미터
      -- 여기부터 클래스별로 따로 --
      -> 하단 히스토그램에서 봉우리 여러 개 탐색       차선 시작 위치
      -> 봉우리마다 독립 슬라이딩 윈도우               차선별 포인트 수집
      -> RANSAC 곡선 적합                              이상치 배제
      -> 곡선 기준으로 픽셀 재할당 후 재적합           창이 놓친 픽셀 회수
      -> 병합 / 중복 검사
      -> 이전 프레임과 매칭해 track 유지               ID 안정화

유도선(교차로 안내선)도 같은 단계를 타지만 **결과를 차선과 섞지 않고**
`DetectionResult.guides` 로 따로 낸다. 자차 좌우 도색이 하나도 없을 때만
그 중 하나를 골라 기준선으로 삼고, 거기서 일정 거리 떨어진 선을 주행선으로
준다 - 교차로에서 좌회전/직진할 때 쓰라고 넣은 것이다.

--------------------------------------------------------------------------
왜 BEV 인가
--------------------------------------------------------------------------
이미지 좌표에서 곧장 RANSAC 을 돌려 봤더니 두 가지가 계속 걸렸다.

1. **원근 때문에 스케일이 안 맞는다.** 가까운 차선은 폭 20px 에 픽셀이
   빽빽하고 먼 차선은 1~2px 에 듬성듬성하다. 임계값 하나로 양쪽을 다룰 수 없다.
2. **u = f(v) 가 소실점 근처에서 조건수가 나쁘다.** v 가 거의 안 변하는데 u 는
   크게 변해서, 지평선 근처 픽셀 몇 개가 2차 계수를 흔들고 그게 아래로 증폭돼
   화면을 쓸어버리는 곡선이 나왔다.

BEV 에서는 스케일이 균일하고(0.05m/px) 차선이 거의 평행한 세로줄이 되며,
결과가 곧바로 차량 좌표계 미터라 제어에 그대로 넘길 수 있다.

--------------------------------------------------------------------------
클래스별로 따로 도는 이유
--------------------------------------------------------------------------
황색 중앙선과 백색 실선은 붙어 있어도 다른 차선이다. 클래스를 섞으면 멀리서
두 선이 만나는 지점에서 하나로 합쳐진다. **한 차선의 픽셀은 전부 같은 클래스다.**
"""

import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

# 이 폴더는 camera_perception/post_processing/ 이다. 모델 정의(seg_model)와
# 전처리 규격(seg_dataset)은 **학습 쪽 것을 그대로 import 한다** - 여기로
# 복사해 두면 학습에서 바꾼 입력 크기나 정규화가 추론에 반영되지 않아
# 조용히 어긋난다.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAM = os.path.dirname(_HERE)                       # camera_perception/
for _p in (_HERE, _CAM, os.path.join(_CAM, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _find_upward(relpath, start=None, levels=6):
    """위로 올라가며 `relpath` 를 찾는다. 폴더를 옮겨도 따라오게 하려는 것이다."""
    d = start or _HERE
    for _ in range(levels):
        cand = os.path.join(d, relpath)
        if os.path.exists(cand):
            return os.path.normpath(cand)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


# 가중치 파일 이름 후보. 저장소마다 다르게 놓여 있어서 여러 개를 본다.
#   car 쪽:  camera_perception/best.pt
#   ROI 쪽:  Sensor/lane_seg_best.pt
CHECKPOINT_NAMES = ("best.pt", "lane_seg_best.pt")


def default_checkpoint():
    """가중치를 이 폴더 -> 상위 폴더 순으로, 이름 후보를 돌며 찾는다."""
    for d in (_HERE, _CAM):
        for name in CHECKPOINT_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return os.path.join(_HERE, CHECKPOINT_NAMES[0])     # 없으면 에러 메시지용


def default_cam_set():
    """cam_set.json 을 찾는다.

    저장소마다 위치가 다르다.
        car 쪽:  car/data/sensors/cam_set.json   (위로 올라가며 찾음)
        ROI 쪽:  Sensor/cam_set.json             (스크립트 옆)
    """
    for p in (os.path.join(_HERE, "cam_set.json"), os.path.join(_CAM, "cam_set.json")):
        if os.path.isfile(p):
            return p
    return _find_upward(os.path.join("data", "sensors", "cam_set.json"))


from GenerateLabels import load_camera                      # noqa: E402
from seg_dataset import (IMAGENET_MEAN, IMAGENET_STD, INPUT_H,  # noqa: E402
                         INPUT_W, NUM_CLASSES)
from seg_model import LaneSegNet                            # noqa: E402

(CLASS_BG, CLASS_WHITE_SOLID, CLASS_WHITE_DASHED,
 CLASS_YELLOW, CLASS_STOPLINE, CLASS_GUIDE) = range(6)

# **유도선(5)은 LANE_CLASSES 에 넣지 않는다.** 교차로 안에서 차로를 이어주는
# 안내선이라 "넘어가면 안 되는 경계"가 아니고, 자차 좌우 슬롯(+-1)에 끼우면
# 차선 폭 판단과 EGO_MAX_Y_M 검사가 통째로 망가진다.
#
# 대신 **같은 검출 파이프라인을 한 번 더 태워서** `DetectionResult.guides` 로
# 따로 뺀다. 좌우 도색이 아예 없는 교차로에서만 유도선을 자차 차로의 기준선으로
# 삼는다 (아래 "유도선" 파라미터 절). 도색이 보이면 도색이 언제나 이긴다.
LANE_CLASSES = (CLASS_WHITE_SOLID, CLASS_WHITE_DASHED, CLASS_YELLOW)
CLASS_NAMES = ["background", "white_solid", "white_dashed",
               "yellow", "stopline", "guide"]

CROP_TOP = 260              # GenerateLabels.CROP_TOP 과 같아야 한다
DEFAULT_SENSOR_ID = 1       # 전방 카메라

# **자차 원점은 노면이 아니라 후륜축 중심(차축 높이)이고, 노면보다 0.35m 위다.**
# (GenerateLabels 상단 주석의 실측: pos_z 가 링크 노면보다 일정하게 +0.35m)
# 그래서 자차 좌표계에서 노면 평면은 z = -0.35 다. 이걸 0 으로 두면 조감도
# 거리가 통째로 틀어진다.
ROAD_Z_EGO = -0.35

# 조감도 범위. 해상도 0.05m/px 는 차선 폭(0.15~0.35m)이 3~7px 로 남는 값이다.
BEV_X_MIN, BEV_X_MAX = 0.0, 40.0        # 전방 (m)
BEV_Y_MIN, BEV_Y_MAX = -10.0, 10.0      # 좌측이 +
BEV_RES = 0.05                          # m / px
BEV_W = int(round((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES))    # 400
BEV_H = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))    # 800

LANE_HALF_WIDTH = 2.2       # 자차 좌우 차선을 찾을 때 볼 횡방향 범위 (m)

# --------------------------------------------------------------------------
# 보닛 마스크 - **png 파일에 의존하지 않도록 윤곽을 코드에 박아 둔다**
# --------------------------------------------------------------------------
# 학습 라벨에서 보닛은 255(ignore)다. 손실이 안 걸리니 모델은 거기에 뭘 출력
# 해야 하는지 배운 적이 없고, 실제로 알록달록한 노이즈를 뱉는다(실측 확인).
# 12랩 마스크끼리 0.11~0.19 퍼센트 밖에 다르지 않아 하나로 고정해도 안전하고,
# learning/ 은 3.8GB 라 실주행 머신에 없을 수 있어 파일 의존을 없앴다.
BONNET_POLY = (
    (0,458), (1215,459), (1215,444), (1183,437), (1183,428), (1152,422),
    (1151,416), (1136,412), (1131,408), (1119,405), (1119,402), (1103,397),
    (1103,394), (1087,388), (1079,384), (1071,380), (1053,374), (1035,368),
    (1021,362), (1007,356), (989,352), (975,348), (957,344), (927,340),
    (901,336), (871,332), (845,328), (793,324), (703,320), (568,320),
    (487,326), (431,330), (367,338), (319,346), (287,354), (271,360),
    (251,366), (235,372), (215,380), (199,386), (191,392), (175,398),
    (159,406), (137,410), (128,415), (117,422), (99,428), (80,433),
    (71,444), (45,454), (31,458)
)
BONNET_DILATE_PX = 6

# ==========================================================================
# 후처리 파라미터
# ==========================================================================

# --- 3. 히스토그램 시드 ----------------------------------------------------
# **구간을 짧게, 그리고 고정 거리가 아니라 데이터 기준으로 잡는다.**
# 짧아야 하는 이유: 급커브에서는 차선 하나가 긴 구간 동안 y 를 크게 훑어서
#   히스토그램이 번지고 **차선이 없는 자리에 봉우리가 생긴다** (실측: 차선
#   2개짜리 커브에서 시드 6개. 3.5m 로 줄이면 정확히 2개).
# 고정 거리로 두면 안 되는 이유: 보닛이 가리는 정도와 점선 위상에 따라 가장
#   가까운 차선 픽셀이 6m 일 때도 10m 일 때도 있다.
SEED_SPAN_M = 3.5           # 가장 가까운 픽셀에서 이만큼 위까지 히스토그램을 만든다
SEED_MIN_GAP_M = 1.5        # 서로 다른 차선으로 볼 최소 간격 (차로 폭 3.3m)
SEED_MIN_RATIO = 0.25       # 최대 봉우리 대비 이 비율 미만이면 시드로 안 본다
SEED_MAX_COUNT = 6          # 클래스 하나에서 시드 최대 개수

# --- 4. 슬라이딩 윈도우 ----------------------------------------------------
# 창을 세로로 두껍게 잡고 그 안의 픽셀을 통째로 모은다. 행 하나씩 이으면
# BEV 로 펴면서 생긴 톱니 구멍마다 끊겨 한 차선이 10개 넘게 쪼개진다
# (실측: 직선 프레임에서 조각 12개).
WIN_COUNT = 24              # 창 개수
WIN_MARGIN_M = 1.0          # 창 반폭. 급커브에서 차선이 창 하나 사이 0.9m 이동한다
WIN_MIN_PIXELS = 20         # 창 중심을 갱신할 최소 픽셀 수
WIN_MAX_MISS = 6            # 연속으로 이만큼 비면 그 차선은 끝
WIN_DRIFT_GAIN = 0.5        # 창 중심 이동량을 얼마나 이어받을지 (커브 추종)

# --- 6. RANSAC -------------------------------------------------------------
FIT_DEGREE = 2              # BEV 에서는 2차면 충분하다
RANSAC_ITERS = 80
RANSAC_THRESH_M = 0.20      # 이 안이면 인라이어 (미터). 차선 폭의 절반쯤
RANSAC_MIN_RATIO = 0.4      # 인라이어가 이 비율 미만이면 실패로 본다

# **적합 구간에 상한을 둔다.**
# 2차식 하나로 32m 급커브를 덮을 수 없다. 실측(lap4_full 창 결과 152개):
#     전 구간(최대 33m)에 맞췄을 때 인라이어 비율  중앙값 0.87, 5%tile 0.45
#     가까운 20m 만 쓰면                          중앙값 0.96, 5%tile 0.48
#     급커브 프레임(000114): 전구간 0.41 -> 근20m **0.98**
# 임계값을 낮추는 것보다 구간을 자르는 게 맞다. 먼 쪽은 급커브에서 어차피
# 못 믿고, 제어가 쓰는 것도 가까운 쪽이다.
FIT_MAX_SPAN_M = 22.0

# --- 7. 픽셀 재할당 --------------------------------------------------------
# 창은 폭이 정해져 있어 커브에서 가장자리 픽셀을 놓친다. 곡선이 정해진 뒤에는
# **그 곡선 주변 픽셀을 다시 전부 긁어모아** 재적합한다. 창의 한계를 곡선이
# 보완하는 단계다.
REASSIGN_BAND_M = 0.35
# **재할당은 창이 실제로 따라간 구간 안에서만 한다.**
# 이 제한이 없으면 2차식을 40m 까지 연장했을 때 **먼 곳에서 다른 차선 옆을
# 지나가며 그 픽셀을 통째로 빨아들인다.** 곡선 거리만 보고 x 범위를 안 보면
# 같은 클래스의 다른 차선과 병합된다.
REASSIGN_X_MARGIN_M = 4.0

# --- 8. 병합 / 유효성 ------------------------------------------------------
MERGE_DIST_M = 0.6          # 겹치는 구간 평균 간격이 이보다 작으면 같은 차선
MIN_LANE_PIXELS = 100       # 이보다 적으면 차선으로 인정하지 않는다
MIN_LANE_SPAN_M = 3.0       # 세로로 이만큼은 이어져야 한다
MAX_GAP_M = 6.0             # 픽셀이 이만큼 비면 다른 물체다 (실선/점선 공통)

# --- 9. 추적 ---------------------------------------------------------------
# 프레임마다 독립으로 검출하면 ID 가 튄다. 이전 프레임 곡선과 매칭해서 같은
# 차선이면 같은 ID 를 유지하고, 잠깐 안 보이면 몇 프레임은 들고 있는다.
TRACK_MATCH_M = 0.8         # 이 안이면 같은 차선으로 본다
TRACK_MAX_MISS = 5          # 이만큼 연속으로 못 찾으면 버린다
TRACK_MIN_HITS = 2          # 이만큼 연속으로 봐야 출력에 내보낸다

# 좌/우 순번을 매길 때 **모든 차선을 이 전방거리에서 비교한다.**
# 차선마다 자기 시작점에서 재면 점선 대시 위상 때문에 기준이 제각각이 된다.
ORDER_X_M = 7.0

# **자차 좌우(+-1)로 인정할 최대 횡거리.**
# 차로 폭이 3.17~3.57m(실측)라 자차 경계는 1.6~1.8m 언저리다. 차가 한쪽으로
# 치우쳐도 3m 를 넘지 않는다. 이 검사가 없으면 자차 경계가 그 프레임에서 안
# 잡혔을 때 **6~8m 밖 차선에 +-1 을 줘 버린다** (실측: ego_right 가 -1.8m 와
# -6.35m 사이를 오갔다). 제어에 6m 밖 차선을 자차 경계라고 주는 것보다
# "없음"을 주는 게 낫다.
EGO_MAX_Y_M = 3.0

# 차로 폭. 한쪽 경계만 보일 때 반대쪽을 추정하는 데 쓴다.
# (GenerateLabels 주석의 실측 폭 3.38 / 3.17 m 의 중간값)
LANE_WIDTH_M = 3.3

# 횡오차/방위오차를 재는 전방거리. Pure Pursuit 전방주시거리 언저리이면서
# 보닛에 가리지 않는 값으로 잡는다 (차선은 보통 5~10m 부터 관측된다).
EVAL_X_NEAR, EVAL_X_FAR = 7.0, 14.0

# --- 정지선 --------------------------------------------------------------
# 정지선은 진행방향과 직각이라 y = f(x) 로 표현할 수 없다. 차선과 같은
# 파이프라인을 태우지 않고 BEV 픽셀의 중앙값으로 거리만 뽑는다.
# **자차 진로를 가로지르는 것만 인정한다** - 옆 도로 정지선을 잡으면 엉뚱한
# 곳에서 선다.
STOPLINE_MIN_PIXELS = 60
STOPLINE_HALF_WIDTH_M = 2.0

# --- 유도선 (교차로) -------------------------------------------------------
# **좌우 도색이 없는 교차로에서 유도선이 자차 차선 노릇을 한다.** 좌회전이나
# 직진에서 유도선으로부터 일정 거리를 두고 따라가는 것이 목적이다. 교차로 안에
# 차선 도색이 없다는 것은 설계 전제였고, 유도선은 그 구간을 메우는 유일한 도색
# 단서다.
#
# 유도선은 차선과 성격이 달라 같은 파라미터를 쓸 수 없다.
#   - 좌회전 유도선은 곡률이 크다. 2차식 구간을 22m 로 두면 뒤쪽이 안 맞는다.
#   - 촘촘한 점선이라 조각이 짧고 픽셀이 적다 (모델 IoU 도 0.39 로 가장 낮다).
#   - 정지선 너머에서 시작하므로 자차 바로 앞(7m)에는 없을 수 있다.
#
# **아래 값들은 실제 교차로 프레임으로 재보고 맞춰야 하는 것들이다.** 지금 값은
# 차선 쪽 실측값에서 성격 차이만큼 밀어 둔 출발점이지 검증된 값이 아니다.
GUIDE_FIT_MAX_SPAN_M = 12.0     # 급한 좌회전을 2차식 하나로 덮는 한계
GUIDE_MIN_PIXELS = 60           # 차선(100)보다 낮게 - 점선이 촘촘하고 짧다
GUIDE_MIN_SPAN_M = 2.0
GUIDE_MAX_GAP_M = 4.0           # 이만큼 벌어지면 건너편/다른 방향 유도선이다
GUIDE_MAX_Y_M = 5.0             # 이보다 옆이면 다른 진행방향의 유도선이다
GUIDE_EXTRAP_M = 5.0            # 유도선은 멀리서 시작한다. 차선(3m)보다 넉넉히

# **유도선을 자차의 어느 쪽에 두고 얼마나 떨어져 달릴 것인가.**
# 유도선은 차로 경계가 아니라 진로 안내선이라 도색과 실제 주행선의 관계는
# 교차로마다 확인해야 한다. 계획/제어 쪽에서 매 프레임 바꿀 수 있도록
# LaneDetector(guide_keep_side=...) 와 guide_center(keep_side=, offset=) 로도
# 열어 둔다.
GUIDE_KEEP_SIDE = "left"                # 유도선을 자차 왼쪽에 두고 달린다
GUIDE_OFFSET_M = LANE_WIDTH_M / 2.0     # 유도선에서 이만큼 떨어져 달린다

# 유도선이 시작~끝 사이에 y 로 이만큼 못 움직였으면 직진 유도선으로 본다.
# 좌회전/직진 유도선이 같이 보이는 교차로에서 어느 쪽을 따라갈지 고르는 데 쓴다.
GUIDE_STRAIGHT_MAX_BEND_M = 1.5

# 클래스마다 다르게 쓰는 값만 모아 둔다. **detect_in_class 는 한 벌이고 클래스가
# 값을 고른다** - 유도선 때문에 검출 코드가 두 벌로 갈라지면 한쪽만 고치게 된다.
DEFAULT_TUNE = {"fit_max_span_m": FIT_MAX_SPAN_M, "min_pixels": MIN_LANE_PIXELS,
                "min_span_m": MIN_LANE_SPAN_M, "max_gap_m": MAX_GAP_M}
CLASS_TUNE = {
    CLASS_GUIDE: {"fit_max_span_m": GUIDE_FIT_MAX_SPAN_M,
                  "min_pixels": GUIDE_MIN_PIXELS,
                  "min_span_m": GUIDE_MIN_SPAN_M,
                  "max_gap_m": GUIDE_MAX_GAP_M},
}


def tune_for(cls):
    return {**DEFAULT_TUNE, **CLASS_TUNE.get(cls, {})}


def bev_to_ego(rows, cols):
    """조감도 픽셀 -> 자차 좌표 (x 전방, y 좌측). 행 0 이 가장 먼 쪽이다."""
    x = BEV_X_MAX - (np.asarray(rows, dtype=np.float64) + 0.5) * BEV_RES
    y = BEV_Y_MAX - (np.asarray(cols, dtype=np.float64) + 0.5) * BEV_RES
    return x, y


def ego_to_bev(x, y):
    r = (BEV_X_MAX - np.asarray(x, dtype=np.float64)) / BEV_RES - 0.5
    c = (BEV_Y_MAX - np.asarray(y, dtype=np.float64)) / BEV_RES - 0.5
    return r, c


@dataclass
class Lane:
    """차선 하나. 좌표는 **자차 좌표계 미터** (x 전방, y 좌측). y = f(x)."""
    cls: int
    coef: np.ndarray                    # y = polyval(coef, x)
    x_range: tuple
    points: np.ndarray                  # [N,2] (x, y) 이 차선에 할당된 픽셀
    lane_id: int = None                 # -1,-2 = 좌측 / +1,+2 = 우측
    track_id: int = None                # 프레임을 넘어 유지되는 고유 번호
    age: int = 0                        # 몇 프레임 연속으로 보였는지

    @property
    def name(self):
        return CLASS_NAMES[self.cls]

    @property
    def is_dashed(self):
        return self.cls == CLASS_WHITE_DASHED

    @property
    def n_points(self):
        return len(self.points)

    def y_at(self, x, extrap=3.0):
        lo, hi = self.x_range
        if x < lo - extrap or x > hi + extrap:
            return None
        return float(np.polyval(self.coef, x))

    def sample(self, step=0.5):
        xs = np.arange(self.x_range[0], self.x_range[1] + 1e-6, step)
        return np.stack([xs, np.polyval(self.coef, xs)], axis=1)


@dataclass
class DetectionResult:
    lanes: list = field(default_factory=list)
    # **유도선은 lanes 에 섞지 않는다.** lane_id 를 받지 않으므로 자차 좌우 슬롯
    # 배정과 차선 폭 판단에 영향을 주지 않는다.
    guides: list = field(default_factory=list)
    maneuver: str = None                # "left" / "straight" / "right" / None
    guide_keep_side: str = None         # None 이면 GUIDE_KEEP_SIDE
    stopline_dist: float = None         # 자차 앞 정지선까지 (m), 없으면 None
    stopline_y: float = None
    mask: np.ndarray = None
    bev: np.ndarray = None
    infer_ms: float = 0.0
    post_ms: float = 0.0

    def by_id(self, lane_id):
        for l in self.lanes:
            if l.lane_id == lane_id:
                return l
        return None

    @property
    def ego_left(self):
        return self.by_id(-1)

    @property
    def ego_right(self):
        return self.by_id(1)

    @property
    def guide(self):
        """따라갈 유도선 하나. 없으면 None (고르는 규칙은 select_guide)."""
        return select_guide(self.guides, self.maneuver)

    # ---- 제어가 바로 쓰는 파생값 ---------------------------------------
    def center_at(self, x):
        """전방 x m 에서 (주행선 횡위치, 그 값이 어디서 나왔는지).

        **값과 출처를 한 함수에서 같이 낸다.** 따로 두면 "왼쪽 차선이 있다"고
        보고하면서 정작 그 x 에서는 값이 없는 상태가 생긴다.

            "both"   자차 좌우 도색이 둘 다 보임
            "left"   왼쪽만 - 차로 폭(LANE_WIDTH_M)으로 반대쪽을 추정
            "right"  오른쪽만
            "guide"  도색이 없어 **유도선을 기준선으로 썼다** (교차로)
            None     아무것도 없음

        12랩 실측에서 자차 좌우가 둘 다 충분히 보이는 프레임은 절반 남짓이다.
        한쪽만으로도 중심을 낼 수 있어야 제어가 끊기지 않는다.
        """
        l, r = self.ego_left, self.ego_right
        ly = l.y_at(x) if l else None
        ry = r.y_at(x) if r else None
        if ly is not None and ry is not None:
            return (ly + ry) / 2.0, "both"
        if ly is not None:
            return ly - LANE_WIDTH_M / 2.0, "left"
        if ry is not None:
            return ry + LANE_WIDTH_M / 2.0, "right"
        # **도색이 하나도 없을 때만 유도선으로 내려간다.** 교차로 밖에서 유도선이
        # 잘못 잡혀도 도색이 있는 한 제어에 흘러들지 않는다.
        gc = self.guide_center(x)
        return (gc, "guide") if gc is not None else (None, None)

    def lane_center(self, x):
        """전방 x m 에서 주행선의 횡방향 위치. 없으면 None."""
        return self.center_at(x)[0]

    def center_source(self, x=EVAL_X_NEAR):
        """lane_center 가 무엇에서 나왔는지. 제어가 신뢰도를 가를 때 쓴다."""
        return self.center_at(x)[1]

    def guide_center(self, x, keep_side=None, offset=None):
        """유도선에서 일정 거리 떨어진 주행선의 횡위치. 없으면 None.

        **자차 좌우 도색이 없는 교차로용이다.** 좌회전/직진에서 유도선을 한쪽에
        끼고 `offset` 만큼 떨어져 지나가는 것이 목적이라, 차선처럼 두 경계의
        중점을 내는 것이 아니라 **한 선에서 밀어낸 값**이다.
        """
        g = self.guide
        if g is None:
            return None
        # 유도선은 정지선 너머에서 시작해 자차 앞이 비어 있을 수 있다. 차선보다
        # 외삽을 넉넉히 허용하되(GUIDE_EXTRAP_M) 한도를 넘으면 None 이다.
        gy = g.y_at(x, extrap=GUIDE_EXTRAP_M)
        if gy is None:
            return None
        side = keep_side or self.guide_keep_side or GUIDE_KEEP_SIDE
        off = GUIDE_OFFSET_M if offset is None else offset
        return gy - off if side == "left" else gy + off

    def lateral_error(self, x=EVAL_X_NEAR):
        """차로 중심 기준 자차의 횡오차 (m). 음수 = 왼쪽으로 치우침."""
        c = self.lane_center(x)
        return None if c is None else -c

    def heading_error(self, x0=EVAL_X_NEAR, x1=EVAL_X_FAR):
        """차로 방향 대비 자차 방위 오차 (rad). 양수 = 차로가 왼쪽으로 감."""
        c0, c1 = self.lane_center(x0), self.lane_center(x1)
        if c0 is None or c1 is None:
            return None
        return math.atan2(c1 - c0, x1 - x0)

    def as_dict(self, decimals=3, points=False):
        """UDP/JSON 으로 내보낼 형태.

        **출력 규격은 아직 제어팀과 확정 전이다.** 지금은 넉넉히 담아 두고,
        정해지면 여기만 줄이면 세 스크립트에 다 반영된다.

        경계 두 줄을 그대로 주는 이유는 **장애물 회피 때문**이다 - 중심선만
        주면 "어디까지 비켜도 되는지"를 표현할 수 없다.
        """
        def pts(l):
            return None if (l is None or not points) else                 np.round(l.sample(), decimals).tolist()
        l, r = self.ego_left, self.ego_right
        g = self.guide
        le, he = self.lateral_error(), self.heading_error()
        rnd = lambda v, d=decimals: None if v is None else round(v, d)
        return {
            "left": pts(l), "right": pts(r),
            "left_type": l.name if l else None,
            "right_type": r.name if r else None,
            "left_dashed": l.is_dashed if l else None,
            "right_dashed": r.is_dashed if r else None,
            "lateral_error": rnd(le),
            "heading_error": rnd(he, decimals + 1),
            "stopline_dist": rnd(self.stopline_dist),
            "n_lanes": len(self.lanes),
            # **유도선을 left/right 와 같은 자리에 놓지 않는다.** 차로 경계가
            # 아니라서 "여기까지 비켜도 된다"는 뜻이 없다. center_source 가
            # "guide" 일 때만 lateral/heading_error 가 유도선에서 나온 값이다.
            "guide": pts(g),
            "guide_keep_side": (self.guide_keep_side or GUIDE_KEEP_SIDE)
                               if g is not None else None,
            "guide_offset": GUIDE_OFFSET_M if g is not None else None,
            "n_guides": len(self.guides),
            "center_source": self.center_source(),
            "infer_ms": round(self.infer_ms, 1),
            "post_ms": round(self.post_ms, 1),
        }


# ==========================================================================
# 3. 히스토그램 시드
# ==========================================================================
def find_seeds(binary, r_start):
    """가장 가까운 픽셀 근처의 열 히스토그램에서 차선 시작 위치들을 찾는다.

    가장 큰 봉우리를 집고 그 주변 `SEED_MIN_GAP_M` 를 지운 뒤 반복한다.
    차로 폭이 3.3m 라 1.5m 안에 두 차선이 같이 있을 일은 없다.
    """
    r_near = int(min(r_start, BEV_H - 1))
    r_far = max(r_near - int(SEED_SPAN_M / BEV_RES), 0)
    if r_near <= r_far:
        return []
    hist = binary[r_far:r_near].sum(axis=0).astype(np.float32)
    if hist.max() <= 0:
        return []
    hist = cv2.GaussianBlur(hist.reshape(1, -1), (0, 0), 3).ravel()
    floor = hist.max() * SEED_MIN_RATIO
    gap = int(SEED_MIN_GAP_M / BEV_RES)
    seeds, work = [], hist.copy()
    for _ in range(SEED_MAX_COUNT):
        c = int(work.argmax())
        if work[c] < floor:
            break
        seeds.append(c)
        work[max(0, c - gap):c + gap] = 0
    return sorted(seeds)


# ==========================================================================
# 4~5. 봉우리마다 독립 슬라이딩 윈도우로 포인트 수집
# ==========================================================================
def sliding_window(binary, base, r_start):
    """시드 하나에서 위로 올라가며 픽셀을 모은다. (rows, cols) 또는 None.

    **다음 창 위치를 기울기로 예측한다.** 직전 창에서 옮겨간 만큼 이번에도
    옮겨갈 것으로 보고 중심을 미리 민다. 이게 없으면 급커브에서 차선이 창
    밖으로 나가 추적이 옆 차선으로 샌다 (실측: 창 하나 사이 0.9m 이동).
    """
    win_h = max(int(r_start) // WIN_COUNT, 1)
    margin = int(WIN_MARGIN_M / BEV_RES)
    cur, drift, miss = float(base), 0.0, 0
    rr, cc = [], []
    for w in range(WIN_COUNT):
        hi = int(r_start) - w * win_h
        lo = max(hi - win_h, 0)
        if hi <= 0:
            break
        pred = int(round(cur + drift))
        c0 = max(pred - margin, 0)
        win = binary[lo:hi, c0:pred + margin + 1]
        ys, xs = np.nonzero(win)
        if ys.size >= WIN_MIN_PIXELS:
            rr.append(ys + lo)
            cc.append(xs + c0)
            new = float(xs.mean()) + c0
            drift = (1 - WIN_DRIFT_GAIN) * drift + WIN_DRIFT_GAIN * (new - cur)
            cur, miss = new, 0
        else:
            miss += 1
            cur += drift                # 빈 창에서도 추세는 이어 간다
            if miss > WIN_MAX_MISS:
                break
    if not rr:
        return None
    return np.concatenate(rr), np.concatenate(cc)


# ==========================================================================
# 6. RANSAC 곡선 적합
# ==========================================================================
def ransac_fit(x, y, rng):
    """y = f(x) 에 RANSAC 으로 2차식을 맞춘다. (coef, inlier_mask) 또는 None.

    **표본을 x 구간으로 나눠서 뽑는다.** 무작위로 뽑으면 픽셀이 많은 근거리에서
    3점이 다 나와, 짧은 구간에 맞춘 곡선이 먼 쪽으로 발산한다.
    """
    n = len(x)
    need = FIT_DEGREE + 1
    if n < need * 4:
        return None
    edges = np.linspace(x.min(), x.max(), need + 1)
    bands = [np.flatnonzero((x >= edges[i]) & (x <= edges[i + 1])) for i in range(need)]
    bands = [b for b in bands if b.size]
    if not bands:
        return None

    best = None
    for _ in range(RANSAC_ITERS):
        pick = np.array([int(rng.choice(bands[i % len(bands)])) for i in range(need)])
        if len(np.unique(x[pick])) < need:
            continue
        try:
            coef = np.polyfit(x[pick], y[pick], FIT_DEGREE)
        except (np.linalg.LinAlgError, ValueError):
            continue
        inl = np.abs(np.polyval(coef, x) - y) < RANSAC_THRESH_M
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < max(need, n * RANSAC_MIN_RATIO):
        return None
    # 인라이어 전체로 다시 맞춘다 (표본 3점보다 훨씬 안정적이다)
    coef = np.polyfit(x[best], y[best], FIT_DEGREE)
    inl = np.abs(np.polyval(coef, x) - y) < RANSAC_THRESH_M
    if inl.sum() < need:
        return None
    return np.polyfit(x[inl], y[inl], FIT_DEGREE), inl


# ==========================================================================
# 7. 곡선 기준 픽셀 재할당
# ==========================================================================
def reassign(px, py, coef, x_lo, x_hi):
    """곡선 주변 픽셀을 다시 긁어모아 재적합한다.

    창은 폭이 정해져 있어 커브 바깥쪽 픽셀을 놓친다. 곡선이 정해진 뒤에
    **그 곡선 기준으로 다시 모으면** 창이 놓친 것을 회수할 수 있다.

    다만 **창이 실제로 따라간 구간(x_lo~x_hi) 밖으로는 나가지 않는다.**
    곡선 거리만 보고 전 구간에서 긁으면, 연장된 2차식이 먼 곳에서 다른 차선
    옆을 지나며 그 픽셀까지 흡수해 두 차선이 하나로 합쳐진다.
    """
    within = (px >= x_lo - REASSIGN_X_MARGIN_M) & (px <= x_hi + REASSIGN_X_MARGIN_M)
    sel = within & (np.abs(np.polyval(coef, px) - py) < REASSIGN_BAND_M)
    if sel.sum() < FIT_DEGREE + 1:
        return coef, sel
    return np.polyfit(px[sel], py[sel], FIT_DEGREE), sel


def largest_run(x, max_gap=MAX_GAP_M):
    """x 를 정렬해 `max_gap` 넘게 벌어진 곳에서 쪼갠 뒤 가장 큰 조각의 인덱스.

    RANSAC 은 "떨어져 있음"을 벌하지 않는다. 인라이어 판정이 점마다 독립이라
    사이가 비어 있어도 상관하지 않아서, 교차로에서 가까운 중앙선과 건너편
    조각이 한 차선으로 이어진다 (실측: 빈 구간 164행 = 8.2m).
    """
    order = np.argsort(x)
    sx = x[order]
    cuts = np.flatnonzero(np.diff(sx) > max_gap)
    if cuts.size == 0:
        return order
    bounds = np.concatenate([[0], cuts + 1, [len(sx)]])
    b = int(np.argmax(np.diff(bounds)))
    return order[bounds[b]:bounds[b + 1]]


# ==========================================================================
# 8. 병합 / 중복 검사
# ==========================================================================
def merge_lanes(lanes):
    """겹치는 구간 내내 붙어 있는 같은 클래스 곡선을 하나로 합친다.

    시드가 한 차선의 좌우 가장자리에 두 개 서면 거의 같은 곡선이 두 번 나온다.
    포인트가 많은 쪽을 남긴다.
    """
    out = []
    for l in sorted(lanes, key=lambda x: -x.n_points):
        dup = False
        for k in out:
            if k.cls != l.cls:
                continue
            lo = max(l.x_range[0], k.x_range[0])
            hi = min(l.x_range[1], k.x_range[1])
            if hi - lo < MIN_LANE_SPAN_M * 0.5:
                continue
            xs = np.linspace(lo, hi, 20)
            if np.mean(np.abs(np.polyval(l.coef, xs)
                              - np.polyval(k.coef, xs))) < MERGE_DIST_M:
                dup = True
                break
        if not dup:
            out.append(l)
    return out


# ==========================================================================
# 9. 이전 프레임과 track 유지
# ==========================================================================
class LaneTracker:
    """프레임을 넘어 차선 정체성을 유지한다.

    프레임마다 독립으로 검출하면 같은 차선인데 순번이 튀고, 한 프레임 놓치면
    출력이 깜빡인다. 이전 곡선과 겹치는 구간에서 평균 거리로 매칭해서 같은
    차선이면 track_id 를 물려주고, 잠깐 안 보여도 몇 프레임은 들고 있는다.
    """

    def __init__(self):
        self.tracks = []        # {"lane": Lane, "miss": int, "hits": int, "id": int}
        self._next_id = 1

    def update(self, lanes):
        for t in self.tracks:
            t["matched"] = False
        for l in lanes:
            best, bestd = None, TRACK_MATCH_M
            for t in self.tracks:
                if t["matched"] or t["lane"].cls != l.cls:
                    continue
                d = self._dist(t["lane"], l)
                if d is not None and d < bestd:
                    best, bestd = t, d
            if best is None:
                self.tracks.append({"lane": l, "miss": 0, "hits": 1,
                                    "id": self._next_id, "matched": True})
                self._next_id += 1
            else:
                best.update(lane=l, miss=0, matched=True)
                best["hits"] += 1
            (best or self.tracks[-1])["lane"].track_id = (best or self.tracks[-1])["id"]

        for t in self.tracks:
            if not t["matched"]:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= TRACK_MAX_MISS]

        out = []
        for t in self.tracks:
            if t["matched"] and t["hits"] >= TRACK_MIN_HITS:
                t["lane"].track_id = t["id"]
                t["lane"].age = t["hits"]
                out.append(t["lane"])
        # 아직 hits 가 모자란 새 차선도 내보낸다 (첫 등장에서 아무것도 안 나오면
        # 곤란하다). 다만 age 로 신뢰도를 알 수 있게 해 둔다.
        for t in self.tracks:
            if t["matched"] and t["hits"] < TRACK_MIN_HITS:
                t["lane"].track_id = t["id"]
                t["lane"].age = t["hits"]
                out.append(t["lane"])
        return out

    @staticmethod
    def _dist(a, b):
        lo = max(a.x_range[0], b.x_range[0])
        hi = min(a.x_range[1], b.x_range[1])
        if hi - lo < 1.0:
            return None
        xs = np.linspace(lo, hi, 12)
        return float(np.mean(np.abs(np.polyval(a.coef, xs) - np.polyval(b.coef, xs))))


def detect_stopline(bev):
    """정지선까지의 거리. 차선과 달리 곡선을 맞추지 않는다.

    정지선은 진행방향과 직각이라 y = f(x) 로 표현할 수 없다. BEV 픽셀의
    중앙값으로 거리만 뽑고, **자차 진로를 가로지르는 것만** 인정한다 -
    옆 도로 정지선을 잡으면 엉뚱한 곳에서 선다.
    """
    rr, cc = np.nonzero(bev == CLASS_STOPLINE)
    if rr.size < STOPLINE_MIN_PIXELS:
        return None, None
    x, y = bev_to_ego(rr, cc)
    if not (y.min() <= 0.0 <= y.max()):
        return None, None
    near = np.abs(y) <= STOPLINE_HALF_WIDTH_M
    if near.sum() < STOPLINE_MIN_PIXELS // 2:
        return None, None
    return float(np.median(x[near])), float(np.median(y[near]))


def assign_lane_ids(lanes):
    """자차 기준 좌/우 순번.

    **모든 차선을 같은 전방거리에서 비교해야 한다.** 차선마다 자기
    x_range[0] 에서 재면, 점선은 대시 위상에 따라 시작점이 5m 일 때도 15m 일
    때도 있어서 **커브에서 순서가 뒤집힌다** - 주행 중 ego_right 가 옆 차선으로
    넘어가는 원인이다. 2차식이라 몇 미터 외삽은 안정적이다.
    """
    scored = []
    for l in lanes:
        y = float(np.polyval(l.coef, ORDER_X_M))
        if abs(y) > BEV_Y_MAX:          # 외삽이 화면 밖으로 날아가면 신뢰 못 한다
            y = float(np.polyval(l.coef, l.x_range[0]))
        scored.append((y, l))
    left = sorted([s for s in scored if s[0] >= 0], key=lambda s: s[0])
    right = sorted([s for s in scored if s[0] < 0], key=lambda s: -s[0])
    # 가장 안쪽 차선이 자차 차로 폭 밖이면 **자차 경계는 못 찾은 것**이다.
    # 그 경우 +-1 을 비워 두고 +-2 부터 매긴다.
    for side, group in ((-1, left), (1, right)):
        start = 1 if (group and abs(group[0][0]) <= EGO_MAX_Y_M) else 2
        for i, (_, l) in enumerate(group):
            l.lane_id = side * (start + i)
    return [l for _, l in left] + [l for _, l in right]


def select_guide(guides, maneuver=None):
    """따라갈 유도선 하나를 고른다. 없으면 None.

    교차로에는 **좌회전 유도선과 직진 유도선이 같이 보인다.** 아무거나 잡으면
    직진 신호에 좌회전 곡선을 따라간다. 그래서 두 단계로 거른다.

    1. `ORDER_X_M` 에서 자차로부터 `GUIDE_MAX_Y_M` 밖에 있는 것은 버린다.
       옆 차로에서 출발하는 유도선이다.
    2. `maneuver` 를 주면 **곡선이 휘어간 방향**으로 고른다. 시작점과 끝점의
       y 차이(bend)를 보고 좌회전은 +, 우회전은 -, 직진은 0 근처여야 한다.
       조건에 맞는 것이 없으면 **None 을 준다** - 방향이 다른 유도선을 대신
       따라가느니 "없음"이 낫다. 자차 좌우 도색이 없을 때 쓰는 값이라 잘못
       고르면 그대로 교차로를 가로지른다.

    `maneuver` 가 None 이면 자차 진로에 가장 가까운 것을 준다.
    """
    cands = []
    for g in guides:
        y = g.y_at(ORDER_X_M, extrap=GUIDE_EXTRAP_M)
        if y is None:                   # 아직 멀리 있으면 시작점으로 잰다
            y = float(np.polyval(g.coef, g.x_range[0]))
        if abs(y) > GUIDE_MAX_Y_M:
            continue
        bend = float(np.polyval(g.coef, g.x_range[1])
                     - np.polyval(g.coef, g.x_range[0]))
        cands.append((abs(y), bend, g))
    if maneuver == "left":
        cands = [c for c in cands if c[1] >= GUIDE_STRAIGHT_MAX_BEND_M]
    elif maneuver == "right":
        cands = [c for c in cands if c[1] <= -GUIDE_STRAIGHT_MAX_BEND_M]
    elif maneuver == "straight":
        cands = [c for c in cands if abs(c[1]) < GUIDE_STRAIGHT_MAX_BEND_M]
    if not cands:
        return None
    return min(cands, key=lambda c: c[0])[2]


class LaneDetector:
    def __init__(self, checkpoint=None, cam_set=None, bonnet_mask=None,
                 device=None, sensor_id=DEFAULT_SENSOR_ID, crop_top=CROP_TOP,
                 seed=0, track=True, maneuver=None, guide_keep_side=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = checkpoint or default_checkpoint()
        ck = torch.load(checkpoint, map_location=self.device)
        backbone = ck.get("args", {}).get("backbone", "resnet34")

        # **클래스 수는 체크포인트가 정한다.** seg_dataset.NUM_CLASSES 를 그대로
        # 쓰면 유도선이 추가된 lane6 가중치를 5클래스 머리로 읽으려다 터진다.
        # scheme 을 안 들고 있는 옛 체크포인트는 lane5(=NUM_CLASSES) 다.
        n_cls = int(ck.get("num_classes", NUM_CLASSES))
        self.model = LaneSegNet(backbone, pretrained=False,
                                num_classes=n_cls).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ckpt_info = {"epoch": ck.get("epoch"), "backbone": backbone,
                          "num_classes": n_cls,
                          "scheme": ck.get("scheme", "lane5")}
        self.crop_top = crop_top
        self.rng = np.random.default_rng(seed)
        self.tracker = LaneTracker() if track else None
        # 교차로에서 어느 유도선을 따라갈지는 **계획 쪽이 안다.** 매 프레임
        # det.maneuver = "left" 처럼 갈아 끼울 수 있게 속성으로 둔다.
        self.maneuver = maneuver
        self.guide_keep_side = guide_keep_side

        cam_set = cam_set or default_cam_set()
        if not cam_set or not os.path.isfile(cam_set):
            raise SystemExit("cam_set.json 을 못 찾았습니다. --cam-set 으로 지정하세요")
        self.cam = load_camera(cam_set, sensor_id, "horizontal").cropped(crop_top)
        self.src_w, self.src_h = self.cam.width, self.cam.height
        self._H = self._build_homography()

        if bonnet_mask is False:
            self.bonnet, self.bonnet_source = None, "없음(끔)"
        elif bonnet_mask:
            m = cv2.imread(bonnet_mask, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise SystemExit(f"보닛 마스크를 못 읽었습니다: {bonnet_mask}")
            self.bonnet = (m[crop_top:] if m.shape[0] > self.src_h else m) > 0
            self.bonnet_source = bonnet_mask
        else:
            self.bonnet, self.bonnet_source = self._builtin_bonnet(), "내장 폴리곤"

    def _builtin_bonnet(self):
        m = np.zeros((self.src_h, self.src_w), np.uint8)
        cv2.fillPoly(m, [np.array(BONNET_POLY, np.int32)], 1)
        if BONNET_DILATE_PX:
            m = cv2.dilate(m, np.ones((2 * BONNET_DILATE_PX + 1,) * 2, np.uint8))
        return m > 0

    def _build_homography(self):
        """이미지 -> 조감도. **내부 파라미터를 다시 유도하지 않고** 이미 검증된
        정투영 `cam.project()` 로 대응점을 만들어 호모그래피를 푼다."""
        ego = np.array([[8.0, -3.0, ROAD_Z_EGO], [8.0, 3.0, ROAD_Z_EGO],
                        [30.0, -3.0, ROAD_Z_EGO], [30.0, 3.0, ROAD_Z_EGO]])
        uv, valid = self.cam.project(ego)
        if not valid.all():
            raise SystemExit("조감도 기준점이 카메라 뒤에 있습니다. 장착값을 확인하세요.")
        r, c = ego_to_bev(ego[:, 0], ego[:, 1])
        return cv2.getPerspectiveTransform(
            uv.astype(np.float32), np.stack([c, r], axis=1).astype(np.float32))

    @torch.no_grad()
    def infer_mask(self, frame_bgr):
        img = frame_bgr[self.crop_top:] if frame_bgr.shape[0] > self.src_h else frame_bgr
        img = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        small = self.model(x).argmax(1)[0].to(torch.uint8).cpu().numpy()
        mask = cv2.resize(small, (self.src_w, self.src_h),
                          interpolation=cv2.INTER_NEAREST)
        if self.bonnet is not None:
            mask[self.bonnet] = CLASS_BG
        return mask

    def to_bev(self, mask):
        return cv2.warpPerspective(mask, self._H, (BEV_W, BEV_H),
                                   flags=cv2.INTER_NEAREST, borderValue=CLASS_BG)

    def detect_in_class(self, bev, cls):
        """한 클래스에서 선들을 찾는다 (3~8 단계). 유도선도 이 함수를 탄다.

        곡률·픽셀 수·점선 간격이 클래스마다 달라서 임계값만 `tune_for(cls)` 로
        갈아 끼운다. **검출 코드 자체는 한 벌이다.**
        """
        t = tune_for(cls)
        binary = bev == cls
        if not binary.any():
            return []
        # BEV 로 펴면서 생긴 톱니 구멍을 메운다 (세로로 긴 커널 = 차선 방향)
        binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((9, 3), np.uint8)) > 0
        rows_any = np.flatnonzero(binary.any(axis=1))
        if rows_any.size == 0:
            return []
        r_start = int(rows_any.max())

        # 재할당 단계에서 쓸 전체 픽셀
        ar, ac = np.nonzero(binary)
        ax, ay = bev_to_ego(ar, ac)

        out = []
        for base in find_seeds(binary, r_start):
            got = sliding_window(binary, base, r_start)
            if got is None:
                continue
            rr, cc = got
            x, y = bev_to_ego(rr, cc)
            # 2차식이 감당할 수 있는 구간만 남긴다 (급커브에서 먼 쪽이 안 맞는다)
            near = x <= x.min() + t["fit_max_span_m"]
            x, y = x[near], y[near]
            fit = ransac_fit(x, y, self.rng)
            if fit is None:
                continue
            coef, _ = fit
            # 7. 픽셀 재할당 - 창이 따라간 구간 안에서만
            coef, sel = reassign(ax, ay, coef, x.min(), x.max())
            px, py = ax[sel], ay[sel]
            if px.size < t["min_pixels"]:
                continue
            keep = largest_run(px, t["max_gap_m"])      # 떨어진 조각은 자른다
            px, py = px[keep], py[keep]
            span = px.max() - px.min()
            if px.size < t["min_pixels"] or span < t["min_span_m"]:
                continue
            coef = np.polyfit(px, py, FIT_DEGREE) if len(np.unique(px)) > FIT_DEGREE \
                else coef
            out.append(Lane(cls=cls, coef=coef,
                            x_range=(float(px.min()), float(px.max())),
                            points=np.stack([px, py], axis=1)))
        return out

    def run(self, frame_bgr):
        t0 = cv2.getTickCount()
        mask = self.infer_mask(frame_bgr)
        t1 = cv2.getTickCount()

        bev = self.to_bev(mask)
        lanes = []
        for cls in LANE_CLASSES:            # 클래스별로 따로 - 섞이지 않게
            lanes += self.detect_in_class(bev, cls)
        # **유도선도 같은 파이프라인을 태우되 차선과 한 통에 담지 않는다.**
        # 병합·추적까지는 같이 돌린다 - 둘 다 클래스가 같을 때만 비교하므로
        # 섞일 일이 없고, 유도선도 추적을 받아야 점선 위상 때문에 깜빡이지 않는다.
        guides = self.detect_in_class(bev, CLASS_GUIDE)
        found = merge_lanes(lanes + guides)
        if self.tracker is not None:
            found = self.tracker.update(found)
        guides = [l for l in found if l.cls == CLASS_GUIDE]
        # **lane_id 는 차선에만 매긴다.** 유도선이 +-1 을 받으면 차로 폭 판단과
        # EGO_MAX_Y_M 검사가 무너진다.
        lanes = assign_lane_ids([l for l in found if l.cls != CLASS_GUIDE])
        sd, sy = detect_stopline(bev)

        t2 = cv2.getTickCount()
        f = cv2.getTickFrequency()
        return DetectionResult(lanes=lanes, guides=guides,
                               maneuver=self.maneuver,
                               guide_keep_side=self.guide_keep_side,
                               stopline_dist=sd, stopline_y=sy,
                               mask=mask, bev=bev,
                               infer_ms=(t1 - t0) / f * 1000.0,
                               post_ms=(t2 - t1) / f * 1000.0)

