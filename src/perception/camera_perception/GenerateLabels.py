"""HD맵을 카메라에 투영해 차선 종류 학습 라벨을 자동 생성한다.

    meta.jsonl(자차 pose) + lane_boundary_set.json(지도)
        → 지도 ENU → 자차 좌표 → 카메라 좌표 → 이미지 (핀홀)
        → 차선폭만큼 리본을 만들어 클래스별로 래스터화

수작업 라벨링이 필요 없다. 지도에 실선/점선(`lane_shape`), 백색/황색(`lane_color`),
대시 간격(`dash_interval`), 차선폭(`lane_width`)이 전부 들어 있기 때문이다.

--------------------------------------------------------------------------
지금 이 파일의 목적: 초점거리 확정
--------------------------------------------------------------------------
cam_set.json 의 `cameraFOV` 가 **수평인지 수직인지 적혀 있지 않다.** 90도가
수평이면 f=640, 수직이면 f=360 으로 1.78배 차이가 나고, 틀리면 라벨이 전부
어긋난 채로 학습이 "정상적으로" 돌아가 원인을 찾기 어려운 실패가 된다.

그래서 `--fov-axis both` 로 두 가설을 나란히 렌더링해 **눈으로 고른다.**
맞는 쪽은 투영선이 영상 속 차선 위에 얹히고, 틀린 쪽은 도로 밖으로 나가거나
터무니없이 좁게 모인다. 고른 뒤 `--fov-axis` 를 그 값으로 고정해서 쓴다.

--------------------------------------------------------------------------
데이터에서 확정한 좌표 관례 (추측하지 말 것)
--------------------------------------------------------------------------
1. `yaw` 는 ENU East 기준 **반시계**(표준 수학 관례)다. drive6 에서 실제 이동방향
   atan2(dy,dx) 와 yaw 의 차이가 -1.3~-1.7도로 일관되게 작은 것으로 확인했다.
2. 자차 좌표계는 **x 전방 / y 좌측 / z 상방**이다. cam_set.json 의 좌측 카메라가
   `y = +0.65` 인 것이 근거다.
3. **노면 높이를 상수로 두지 말 것.** 맵 전체 표고차가 10.29m (27.14~37.43m) 라
   평지 가정은 경사 구간에서 라벨을 통째로 어긋나게 한다. `meta.jsonl` 의
   `link_id` 로 링크를 찾아 자차에 가장 가까운 점의 z 를 노면으로 쓴다
   (drive6 2772프레임 전부 조회 성공, 자차와 0.2~2.3m).
4. 폴리라인 점 간격이 중앙값 1.97m, 최대 36.1m 다. **재보간 없이는 대시(3m)를
   표현할 수 없다.**

`link_id` 는 HD맵 정보라 인지 파이프라인 입력으로는 쓰지 않는다 (실차에 없다).
여기서 쓰는 건 **라벨 생성** 용도이므로 문제가 없다.

--------------------------------------------------------------------------
정합에서 실제로 문제가 됐던 것 (실측으로 확정)
--------------------------------------------------------------------------
정지 프레임에서 잰 편차 (도색 - 라벨), 황색 중앙선 / 흰 실선 / 좌우 폭:

    yaw 만 + 노면기준 높이   +0.42 m / -0.12 m   폭 2.57 vs 3.12 m
    pos_z 를 원점으로        -0.14 m / +0.06 m   폭 3.38 vs 3.17 m
    + roll/pitch 까지        -0.02 m / +0.02 m   폭 3.38 vs 3.36 m   <- 현재

1. **`pos_z` 는 노면이 아니라 차체 원점**이다. 링크 노면보다 일정하게 +0.35m
   높다 (still +0.363, curve1 +0.340). 승용차 바퀴 반지름과 맞고, 차량 제원
   (전장 4.635 = 휠베이스 3.0 + 앞오버행 0.845 + 뒤오버행 0.79)과 함께 보면
   원점은 후륜축 중심·차축 높이다. 그래서 카메라 노면 높이는 1.20 이 아니라
   pos_z + 1.20 - 노면z = 약 **1.56m** 다. 이걸 1.20 으로 두면 거리 추정이
   전부 어긋나 차로폭이 2.57m 로 나온다 (실제 3.38m).
2. **roll/pitch 를 써야 한다.** 도로 경사·뱅크 때문에 정지 중에도 0 이 아니다
   (정지 프레임 pitch +0.68도). pitch 오차는 가로가 아니라 세로로 터진다 —
   d ~ h/theta 라 delta_d ~ -d^2*delta/h, 10m 에서 2도면 2.9m 다. 직선에서는
   차선이 진행방향과 나란해 티가 안 나고 **커브에서만** 크게 어긋난다.
3. **yaw 바이어스 -1.361도** 는 남는다. 이동방향 atan2(dy,dx) 와 yaw 의 차이를
   직선 주행에서 재도 -1.361도 로 같다. `--yaw-offset -1.361` 로 넣는다.
4. 가로 평행이동(`--lat-offset`)은 **0 이 맞다.** 한때 -0.45 를 넣었는데,
   그건 위의 1·2 를 안 고친 상태에서 좌우 편차의 평균을 맞추려던 것이라
   한쪽을 맞추면 반대쪽이 틀어졌다. 좌우가 서로 반대로 벌어지는 오차는
   평행이동으로 고칠 수 없다 — 그게 높이/자세 문제라는 신호였다.
"""

import argparse
import collections
import json
import math
import os
import re

import cv2
import numpy as np
from scipy.spatial import cKDTree


# --- 클래스 (배경 포함 8채널) ---
CLASS_BG = 0
CLASS_WHITE_SOLID = 1
CLASS_WHITE_DASHED = 2
CLASS_YELLOW = 3
CLASS_STOPLINE = 4          # surface_marking_set.json 필요 — 아직 확보 못 함
CLASS_GUIDE = 5             # 유도선 (촘촘한 점선)
CLASS_ZONE = 6              # 안전지대 등 굵은 노면 도색
CLASS_CROSSWALK = 7         # 횡단보도
CLASS_IGNORE = 255          # 가려짐 등 손실에서 제외할 픽셀

CLASS_NAMES = {
    CLASS_BG: "background", CLASS_WHITE_SOLID: "white_solid",
    CLASS_WHITE_DASHED: "white_dashed", CLASS_YELLOW: "yellow",
    CLASS_STOPLINE: "stopline", CLASS_GUIDE: "guide",
    CLASS_ZONE: "zone", CLASS_CROSSWALK: "crosswalk",
}

# --- 학습용 5클래스 리맵 ---
# 내부 표현은 8클래스 그대로 두고(정보 손실 없는 원본 유지), 저장/출력할 때만
# 합친다. 1차 학습은 {배경, 백색실선, 백색점선, 황색, 정지선} 5종만 쓴다.
# 유도선·안전지대·횡단보도는 1차 대상이 아니지만 **배경으로 두면 안 된다** —
# 크고 선명한 도색을 "아무것도 아니다"라고 가르치게 되므로 ignore 로 뺀다.
TRAIN_CLASS_MAPS = {
    "lane5": {
        CLASS_BG: 0,
        CLASS_WHITE_SOLID: 1,
        CLASS_WHITE_DASHED: 2,
        CLASS_YELLOW: 3,
        CLASS_STOPLINE: 4,
        CLASS_GUIDE: CLASS_IGNORE,
        CLASS_ZONE: CLASS_IGNORE,
        CLASS_CROSSWALK: CLASS_IGNORE,
        CLASS_IGNORE: CLASS_IGNORE,
    },
}
TRAIN_CLASS_NAMES = {
    "lane5": {0: "background", 1: "white_solid", 2: "white_dashed",
              3: "yellow", 4: "stopline"},
}


def remap_classes(label, scheme):
    """8클래스 라벨을 학습용 스킴으로 합친다. scheme 이 None 이면 그대로."""
    if not scheme:
        return label
    table = TRAIN_CLASS_MAPS[scheme]
    out = np.full_like(label, CLASS_IGNORE)
    for src, dst in table.items():
        out[label == src] = dst
    return out

# NGII lane_type → 클래스. 지도 1245개를 실측해 정한 값이다.
#   505(485개) solid/white  폭 0.15  길이중앙 18.8m — 길가장자리
#   501(248개) solid/yellow 폭 0.15  길이중앙 18.4m — 중앙선
#   503(124개) broken/white 폭 0.15  길이중앙 28.2m, dash (3,5)
#              ← 이 3m/5m 이 후처리에서 관측한 점유율 이론값 0.375 와 일치해
#                매핑의 교차검증이 된다
#   530( 97개) solid/white  폭 0.60  길이중앙  3.6m, dash 없음  ← **정지선**
#              링크와 각도 중앙 86도(직각), 합류점 15m 이내 86%, 길이가 차로폭 정도.
#              InspectMapTypes 로 화면을 보면 교차로 진입부를 가로지르는 굵은 흰
#              가로선이다.
#              ⚠ 예전에 "링크와 95% 평행이라 정지선이 아니다"라고 판단했던 것은
#              측정이 틀린 것이었다. 20m 반경의 모든 링크와 비교해 min(각도차)를
#              취했는데, 교차로에는 양방향 링크가 다 있어서 **진행 도로에 직각인
#              정지선이 교차 도로와는 평행**으로 잡힌다. 가장 가까운 세그먼트만
#              봐야 그 선이 실제로 속한 링크와 비교된다.
#   525( 85개) broken/white 폭 0.15  길이중앙 16.1m, dash (0.75,0.75) ← 촘촘한 점선
#   502( 12개) broken/white 폭 0.35  길이중앙  6.0m, dash (0.5,0.5)
#
# 525·502 는 shape/color 만으로는 일반 차선과 구분되지 않아 lane_type 이 필요하다.
# CLASS_ZONE 은 530 을 담으려고 만들었으나 530 이 정지선으로 밝혀져 지금은 비어 있다.
# 재 export 후 굵은 노면 도색이 나오면 그때 쓴다.
LANE_TYPE_TO_CLASS = {
    505: CLASS_WHITE_SOLID, 506: CLASS_WHITE_SOLID, 535: CLASS_WHITE_SOLID,
    501: CLASS_YELLOW, 531: CLASS_YELLOW, 515: CLASS_YELLOW,
    503: CLASS_WHITE_DASHED,
    525: CLASS_GUIDE, 502: CLASS_GUIDE,
    530: CLASS_STOPLINE,
}

# 오버레이 표시색 (BGR)
CLASS_COLORS = {
    CLASS_WHITE_SOLID: (0, 0, 255),      # 빨강
    CLASS_WHITE_DASHED: (255, 0, 0),     # 파랑
    CLASS_YELLOW: (0, 255, 255),         # 노랑
    CLASS_STOPLINE: (255, 255, 0),       # 하늘색
    CLASS_GUIDE: (255, 0, 255),          # 자홍
    CLASS_ZONE: (0, 165, 255),           # 주황
    CLASS_CROSSWALK: (0, 255, 0),        # 초록
    CLASS_IGNORE: (128, 128, 128),       # 회색 — 손실에서 제외되는 영역
}

# 겹선(중앙선 등)을 이 거리 안에서 찾는다. 지도는 겹선을 **같은 자리의 레코드
# 2개**로 표현한다 — 501 중앙선 248개 중 154개(62%)가 다른 501 과 0.02m 미만으로
# 겹쳐 있고, 대조군인 505(길가장자리)는 485개 중 348개가 3m 넘게 단독이다.
# 이걸 모르고 각 레코드를 같은 자리에 폭 0.15m 로 그리면 결과가 그대로 0.15m 라,
# 실제 겹선 폭 0.15+0.1+0.15 = 0.4m 의 **2.7배 좁게** 라벨링된다.
DOUBLE_PAIR_RADIUS = 0.6

RESAMPLE_STEP = 0.25        # 폴리라인 재보간 간격 (m). 대시 3m 를 12조각으로 나눈다
MAX_RANGE = 40.0             # 이보다 먼 차선은 버린다.
# 40m 로 정한 근거 (2026-08-27, 1_2stage_test1 실측):
# 1. **해상도** — 차선폭 0.15m 가 화면에서 차지하는 폭은 fx*0.15/거리 다.
#    30m 3.2px / 40m 2.4px / 50m 1.9px / 80m 1.2px. 50m만 가도 2px 미만이라
#    모델이 분해할 수 없고, 실제 렌더링된 도색은 안티앨리어싱된 흐릿한 자국인데
#    라벨은 딱딱한 1px 선이라 학습에 노이즈만 된다.
# 2. **분포가 40m 에서 꺾인다** — 화면에 찍히는 라벨 픽셀을 ego 전방거리로
#    나눠 보면 0-10m 68.0% / 10-20m 14.8% / 20-30m 4.1% / 30-40m 1.7% 로
#    줄다가 40-50m 2.6% / 50-60m 2.9% / 60-80m 5.9% 로 **다시 늘어난다**.
#    정상 원근이면 계속 줄어야 하는데 늘어난다는 건, 그 거리대에 멀리서 옆으로
#    보이는 다른 도로가 화면 가로를 채우고 있다는 뜻이다 (실측: 60m 이상 라벨의
#    프레임당 가로 폭 중앙값 1244px, 세로 폭 52px — 지평선의 가로 띠).
#    40m 컷은 이 왜곡 구간을 통째로 잘라낸다 (전체 라벨의 11.4%).
# 값이 30 → 80 → 40 으로 바뀐 이력이 있으니 다시 건드리기 전에 읽을 것:
#  - 처음 30m 로 줄였던 건 curve2 idx=416 의 급커브 왜곡 때문이었다(경계선을 거의
#    접선 방향으로 보는 시야에서는 1도 미만 오차도 원근상 몇 m 로 증폭). 하지만
#    그건 원인을 고치지 않고 증상을 가리는 처방이었다.
#  - 그래서 80m 로 되돌렸고, 그 뒤 진짜 원인(카메라 파이프라인 지연 0.090s)을
#    찾아 고쳤다 — CAMERA_PIPELINE_LATENCY 참고. 급커브 왜곡은 그걸로 해결됐다.
#  - 지금의 40m 는 그 왜곡 회피가 아니라 위의 해상도·분포 근거로 정한 값이다.
NEAR_PLANE = 0.5            # 카메라 앞 이 거리보다 가까우면 투영하지 않는다

# pose_dt 는 "보간에 쓴 두 샘플 사이 간격"이 아니라 "아래쪽 샘플로부터의 거리"라서
# 정상 프레임도 [0, 상태 샘플 간격] 안에서 골고루 흩어진다 (상태 스트림 실측 ~12Hz,
# 간격 0.085s). 예전 값 0.02 는 카메라 지연 보정 전 test2 에서 t_cam 이 우연히 상태
# 샘플 바로 뒤에 떨어져 pose_dt 가 작게 나오던 것을 보고 정한 값이라, 보정 후
# 위상이 바뀌자 멀쩡한 프레임의 66% 를 버렸다. 샘플 간격을 넘는 값만 실제로
# 상태 패킷이 유실된 구간이므로 그 언저리로 둔다.
POSE_DT_MAX = 0.09          # 이보다 큰 pose_dt 프레임은 라벨 생성에서 제외한다
BRAKE_SKIP_THRESHOLD = 0.05 # 이보다 큰 브레이크 값이면 감속 중으로 보고 제외한다

# 카메라 파이프라인 지연 (초). RecordDrive.CAMERA_PIPELINE_LATENCY 와 같은 값이다.
# 카메라 타임스탬프는 장면이 렌더된 시각이 아니라 패킷을 내보낸 시각에 가까워서,
# 이미지 내용이 자기 타임스탬프보다 이만큼 과거다. meta.jsonl 에 `cam_latency` 가
# 없는 (= 보정 전에 녹화된) 녹화본은 여기서 되돌려 준다.
#
# 실측(test2, 회전 프레임 15장): 최적 오프셋 중앙값 -0.090s, 표준편차 0.022s.
# 좌회전과 우회전이 **똑같이** 음수 오프셋을 필요로 한다 — yaw 바이어스라면
# 좌/우 부호가 반대여야 하므로 각도 오차가 아니라 고정 지연이다.
CAMERA_PIPELINE_LATENCY = 0.090

# 차선 ID 를 매길 범위. 교차로에서는 다른 방향 도로의 경계가 화면에 잔뜩
# 들어오므로(실측: 한 프레임 70개, 횡방향 60m 밖까지) 자차 주변을 실제로
# 지나가는 것만 번호를 준다. 3.4m 차로 기준 ±12m 면 좌우 3차로 정도가 들어온다.
LANE_ID_MAX_LAT = 12.0      # 자차 기준 횡방향 (m)
LANE_ID_MAX_DIST = 40.0     # 최근접점까지의 거리 (m)

# 파생물에서 잘라낼 위쪽(하늘) 픽셀 수. 원본 프레임은 자르지 않는다
# — CameraModel.cropped 참고.
#
# 1_2stage_test1 507장 전수 조사에서 라벨이 나타나는 최상단이 원본 y=322
# (1퍼센타일 346, 중앙값 360) 이라 260 이면 62px 여유가 남고 잘리는 라벨은 없다.
# 평지 지평선은 y≈382 인데 오르막 구간에서 라벨이 그보다 위로 올라오는 것까지
# 감안한 값이다.
#
# 참고로 ROI/Sensor/LaneCandidates.py 의 사다리꼴 ROI 윗변을 1280x720 으로
# 환산하면 y=252 다. 전혀 다른 근거(후처리 지평선 가드)로 정해진 값인데 비슷한
# 자리라 세로 컷 위치의 교차검증이 된다. 다만 그 사다리꼴의 좌우 빗변은 쓰면
# 안 된다 — 화면 가장자리의 **근거리** 차선을 7.5% 잘라낸다.
CROP_TOP = 260

# 자차 기준 이 횡거리를 넘는 경계는 라벨에서 뺀다. 반대편 차도를 자동으로
# 걸러내려고 한때 12.0 을 썼지만(실측: "라벨인데 실제 도색이 아닌" 픽셀이
# 26.7% → 19.7%), 자동 판정이 부정확해 멀쩡한 경계까지 잘려나갔다. 지금은
# 끄고(전부 표시) 어느 경계를 뺄지는 `--exclude-file` 로 사람이 고른다.
LABEL_MAX_LAT = float("inf")

# 사람이 직접 제외한 지도 경계 레코드 idx (예: 반대편 차도). EditLabels.py 로
# 화면에서 선을 클릭해 고르고 label_exclude.json 에 저장한다. **지도 레코드
# 단위라 한 번 빼면 그 레코드가 나오는 모든 프레임에서 빠진다** — 반대편 차도는
# 어느 프레임에서나 같은 레코드이므로 프레임마다 지울 필요가 없다.
EXCLUDED_BOUNDARIES = set()
# 사람이 통째로 뺀 프레임 번호. 라벨이 못 쓸 만큼 어긋났거나 장면 자체가
# 학습에 부적합할 때 EditLabels.py 에서 프레임 단위로 뺀다.
EXCLUDED_FRAMES = set()
EXCLUDE_FILENAME = "label_exclude.json"
# **경계 제외는 녹화본이 아니라 지도에 딸린 정보다.** 반대편 차도 차선은 어느
# 바퀴를 돌아도 같은 지도 레코드라, 한 번 빼면 앞으로 찍는 모든 녹화본에서
# 빠져야 한다. 그래서 이 파일만 스크립트 폴더에 두고 전 녹화본이 공유한다.
# (프레임 제외는 그 녹화본에만 해당하므로 녹화 폴더의 EXCLUDE_FILENAME 에 남는다.)
GLOBAL_EXCLUDE_FILENAME = "lane_exclude_global.json"


def global_exclude_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        GLOBAL_EXCLUDE_FILENAME)


def _read_exclude(path):
    if not path or not os.path.isfile(path):
        return set(), set()
    with open(path, encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):          # 예전 형식(리스트만)
        return set(data), set()
    return set(data.get("excluded", [])), set(int(i) for i in data.get("excluded_frames", []))


def load_exclusions(path):
    """제외 목록을 읽는다. (경계 idx 집합, 프레임 번호 집합).

    경계는 전역 파일 + 녹화 폴더 파일을 합친다 — 예전에 녹화 폴더에만
    저장해 둔 것도 계속 살아 있게 하려는 것이다.
    """
    g_b, _ = _read_exclude(global_exclude_path())
    l_b, l_f = _read_exclude(path)
    return g_b | l_b, l_f


# ======================================================================
# 카메라 모델
# ======================================================================
class CameraModel:
    """MORAI 카메라의 핀홀 모델. lensDistortion 이 [0,0,0] 이라 왜곡 항이 없다.

    내부 파라미터는 `cameraFOV` 하나만 믿는다. cam_set.json 의 다른 광학값들은
    서로 모순된다 — sensorSize 36x24mm / focalLengthmm 16 이면 수평 FOV 96.4도,
    focalLengthpixel 320 이면 126.9도 인데 cameraFOV 는 90 이다. 오버레이로
    확정한 값은 FOV 90 **수평**, 즉 1280x720 에서 f = 640 이다.

    주점(cx, cy)은 이미지 중심으로 둔다. cam_set 의 sensorShift 가 (0,0) 이라
    이 경우엔 맞지만, 측정한 값이 아니라 가정이라는 점은 알고 있어야 한다.

    **원본 해상도와 저장 해상도를 분리해서 들고 있다.** 지금 RecordDrive 는
    원본 1280x720 그대로 저장하므로 fx = fy = 640 으로 등방이다. 예전처럼
    640x480 (4:3) 으로 줄여 저장하면 비등방 축소라 fx != fy 가 되고, 두 축을
    같은 값으로 두면 세로가 33% 틀린 채 투영된다.
    """

    def __init__(self, width, height, fov_deg, fov_axis, mount_pos, mount_rot,
                 native_width=None, native_height=None):
        self.width = int(width)
        self.height = int(height)
        self.native_width = int(native_width or width)
        self.native_height = int(native_height or height)
        self.fov_deg = float(fov_deg)
        self.fov_axis = fov_axis

        # 원본 해상도에서의 초점거리 (정사각 픽셀이라 한 값)
        half = math.radians(self.fov_deg / 2.0)
        span = self.native_width if fov_axis == "horizontal" else self.native_height
        f_native = (span / 2.0) / math.tan(half)

        # 축마다 따로 배율을 먹인다
        self.fx = f_native * self.width / self.native_width
        self.fy = f_native * self.height / self.native_height
        self.f = self.fx                    # 로그 표시용
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.mount_pos = np.asarray(mount_pos, dtype=np.float64)   # 자차 기준 (x,y,z)
        self.roll, self.pitch, self.yaw = (math.radians(a) for a in mount_rot)
        self.crop_top = 0

    def cropped(self, top):
        """위쪽 `top` 행을 잘라낸 카메라. **주점 cy 가 같이 따라와야 한다.**

        하늘 영역은 라벨이 절대 나오지 않는다(실측: 56프레임에서 라벨 최상단
        v=288, 5퍼센타일 338). 잘라내면 픽셀 수가 줄어 학습·추론이 가벼워지고
        하늘 클러터도 사라진다. 다만 자르면 이미지 원점이 바뀌므로 cy 를 그만큼
        올려주지 않으면 투영이 통째로 세로로 어긋난다 — 라벨 생성·학습·추론
        어느 한 곳이라도 이 값이 안 맞으면 바로 틀어진다.

        **원본 프레임은 자르지 않는다.** 녹화본은 팀 공용이라 그대로 두고,
        여기서 만드는 파생물(마스크·구조화 라벨·오버레이)에만 적용한다.
        학습 쪽이 같은 값으로 자를 수 있도록 구조화 라벨에 `crop_top` 을 남긴다.
        """
        if not top:
            return self
        out = self.scaled(self.width, self.height)
        out.height = self.height - int(top)
        out.cy = self.cy - int(top)
        out.crop_top = int(top)
        return out

    def scaled(self, width, height):
        """같은 센서를 다른 저장 해상도로. 비등방이어도 정확하다."""
        return CameraModel(width, height, self.fov_deg, self.fov_axis,
                           self.mount_pos,
                           (math.degrees(self.roll), math.degrees(self.pitch),
                            math.degrees(self.yaw)),
                           self.native_width, self.native_height)

    def to_camera(self, pts_ego):
        """자차 좌표(x전방, y좌측, z상방) → 카메라 광학 좌표 (x우측, y하방, z전방).

        장착 회전은 roll/pitch/yaw 를 모두 쓴다. 예전에는 yaw 를 읽어만 두고
        쓰지 않았다 — 전방 카메라는 yaw=0 이라 티가 안 났지만 좌/우 카메라
        (yaw 70도, 290도)에 쓰면 통째로 틀린다.
        """
        p = np.asarray(pts_ego, dtype=np.float64) - self.mount_pos

        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        # 차량축 기준 장착 회전의 역변환 (R_m^T p)
        Rm = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr],
        ])
        q = p @ Rm                      # 카메라 몸체축 (x전방, y좌측, z상방)

        # 몸체축 → 광학축 (x우측, y하방, z전방)
        return np.stack([-q[:, 1], -q[:, 2], q[:, 0]], axis=1)

    def project_camera(self, cam_pts):
        """카메라 좌표 → 이미지 좌표. (uv, valid)."""
        Zr = cam_pts[:, 2]
        valid = Zr > NEAR_PLANE
        Zs = np.where(valid, Zr, 1.0)               # 0 나눗셈 방지
        u = self.fx * cam_pts[:, 0] / Zs + self.cx
        v = self.fy * cam_pts[:, 1] / Zs + self.cy
        return np.stack([u, v], axis=1), valid

    def project(self, pts_ego):
        """자차 좌표 → 이미지 좌표. (uv, valid) 를 돌려준다."""
        return self.project_camera(self.to_camera(pts_ego))


def clip_near(cam_pts):
    """폴리곤을 근평면(z > NEAR_PLANE)으로 자른다.

    꼭짓점 하나라도 카메라 뒤면 폴리곤을 통째로 버리는 방식은 쓰면 안 된다.
    자차가 올라선 횡단보도처럼 **가장 중요한 근거리 대상**이 정확히 그 경우라
    통째로 라벨에서 빠진다.
    """
    n = len(cam_pts)
    if n < 3:
        return None
    out = []
    for i in range(n):
        a, b = cam_pts[i], cam_pts[(i + 1) % n]
        ain, bin_ = a[2] > NEAR_PLANE, b[2] > NEAR_PLANE
        if ain:
            out.append(a)
        if ain != bin_:
            t = (NEAR_PLANE - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.asarray(out) if len(out) >= 3 else None


def load_camera(cam_set_path, sensor_id, fov_axis):
    with open(cam_set_path, encoding="utf-8") as fp:
        cfg = json.load(fp)
    for cam in cfg["cameraList"]:
        if int(cam["m_SensorUniqueID"]) != int(sensor_id):
            continue
        cc, pos, rot = cam["cc"], cam["pos"], cam["rot"]
        return CameraModel(
            int(cc["cameraResWidth"]), int(cc["cameraResHeight"]),
            float(cc["cameraFOV"]), fov_axis,
            (float(pos["x"]), float(pos["y"]), float(pos["z"])),
            (float(rot["roll"]), float(rot["pitch"]), float(rot["yaw"])),
        )
    raise SystemExit(f"cam_set 에 SensorUniqueID={sensor_id} 가 없습니다: {cam_set_path}")


# ======================================================================
# 지도
# ======================================================================
def _resample(points, step):
    """폴리라인을 일정 간격으로 다시 뽑고 각 점의 누적 거리를 함께 돌려준다.

    원본 점 간격이 중앙값 1.97m, 최대 36.1m 라 이 과정 없이는 대시(3m)를
    표현할 수 없다.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts, np.zeros(len(pts))

    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return pts, s

    n = max(2, int(total / step) + 1)
    su = np.linspace(0.0, total, n)
    out = np.stack([np.interp(su, s, pts[:, i]) for i in range(3)], axis=1)
    return out, su


def _point_to_polyline(pt, poly):
    """점에서 폴리라인까지의 최단거리. 끝점만 보면 이어붙은 구간과 겹선을 못 가른다."""
    a, b = poly[:-1, :2], poly[1:, :2]
    ab = b - a
    t = np.clip(((pt - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((pt - proj) ** 2).sum(1)).min())




def _line_overlap_distance(a_pts, b_pts):
    """두 폴리라인이 처음부터 끝까지 같은 자리에 겹치는지 보려고, 시작·중간·끝
    3점에서 상대 폴리라인까지의 최단거리 중 최댓값을 돌려준다. 중간점 하나만
    보면 두 선이 그 지점에서만 스치듯 가까운 경우도 겹선으로 오판할 수 있다.
    """
    n = len(a_pts)
    idxs = (0, n // 2, n - 1)
    return max(_point_to_polyline(a_pts[k, :2], b_pts) for k in idxs)


def _pair_doubles(items):
    """겹선 쌍을 찾아 서로 반대 방향의 가로 오프셋을 준다.

    지도는 겹선을 같은 자리의 레코드 2개로 표현하므로, 그대로 그리면 두 번
    덧칠될 뿐 폭이 넓어지지 않는다. 각각을 (폭+간격)/2 만큼 좌우로 밀어야
    실제 겹선의 모양(가운데 빈 틈 포함)이 나온다.

    좌표가 완전히 일치하는 쌍도 **벌려서 두 줄로 그린다.** 한때 그런 쌍을
    "지도가 같은 선을 두 번 기록한 것"으로 보고 한쪽을 버렸는데(계기:
    B2256W001772/001786 을 벌렸더니 실제로 없는 두 번째 선이 생김) 그 판정이
    너무 거칠었다 — K-City 맵에서 75개가 지워졌고 그중 다수가 **진짜 황색
    중앙 겹선**이었다. 15foggy0 idx=63 에서 실제 도로엔 황색 2줄인데 라벨이
    1줄만 나오는 것으로 확인했다. 겹선을 살리는 쪽이 맞고, 유령선이 생기는
    개별 레코드는 EditLabels.py 로 빼는 게 낫다.
    """
    taken = set()
    pairs = 0
    for i, a in enumerate(items):
        if i in taken or a["lat_offset"] != 0.0:
            continue
        best, best_d = None, DOUBLE_PAIR_RADIUS
        for j, b in enumerate(items):
            if j == i or j in taken or b["lane_type"] != a["lane_type"]:
                continue
            d = _line_overlap_distance(a["points"], b["points"])
            if d < best_d:
                best, best_d = j, d
        if best is None:
            continue
        b = items[best]
        h = (a["width"] + a["interval"]) / 2.0
        a["lat_offset"], b["lat_offset"] = -h, +h
        taken.update((i, best))
        pairs += 1
    return pairs


def load_boundaries(mgeo_dir):
    """lane_boundary_set.json 을 읽어 재보간·클래스 매핑·겹선 처리까지 마친다."""
    path = os.path.join(mgeo_dir, "lane_boundary_set.json")
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)

    out, skipped = [], collections.Counter()
    split_shapes = 0
    for rec in raw:
        lane_type = rec["lane_type"][0] if isinstance(rec["lane_type"], list) else rec["lane_type"]
        cls = LANE_TYPE_TO_CLASS.get(lane_type)
        if cls is None:
            skipped[lane_type] += 1
            continue

        pts, s = _resample(rec["points"], RESAMPLE_STEP)
        if len(pts) < 2:
            continue

        shape = " ".join(rec["lane_shape"]) if isinstance(rec["lane_shape"], list) else str(rec["lane_shape"])
        dash_on = float(rec.get("dash_interval_L1") or 0.0)
        dash_off = float(rec.get("dash_interval_L2") or 0.0)
        width = max(float(rec.get("lane_width") or 0.15), 0.10)
        interval = float(rec.get("double_line_interval") or 0.0)

        def make(sh, offset):
            broken = "broken" in sh and dash_on > 0 and dash_off > 0
            # **클래스는 lane_type 이 아니라 이 쪽의 shape 을 따라야 한다.**
            # 'solid broken' 복합선은 한 레코드가 두 줄을 나타내는데, lane_type 으로
            # 정한 클래스를 두 쪽에 그대로 물리면 점선 쪽까지 실선으로 찍힌다.
            # 그러면 "한쪽에서만 넘어올 수 있다"는 정보가 라벨에서 사라진다 —
            # 고속도로 진입 구간에서 차선변경 가부를 판단할 근거가 바로 이것이다.
            side_cls = cls
            if cls in (CLASS_WHITE_SOLID, CLASS_WHITE_DASHED):
                side_cls = CLASS_WHITE_DASHED if broken else CLASS_WHITE_SOLID
            return {
                "idx": rec["idx"], "cls": side_cls, "lane_type": lane_type,
                "points": pts, "s": s, "width": width, "interval": interval,
                "broken": broken, "dash_on": dash_on, "dash_off": dash_off,
                "lat_offset": offset,
                "bbox": (pts[:, 0].min(), pts[:, 0].max(),
                         pts[:, 1].min(), pts[:, 1].max()),
            }

        words = shape.split()
        if len(words) == 2:
            # 'solid broken' 처럼 한 레코드가 겹선 두 줄을 나타낸다 (type 506, 9개).
            # 한쪽만 실선인 복합선이라 통째로 실선 처리하면 절반이 틀린 라벨이 된다.
            #
            # **lane_shape 의 순서는 폴리라인 진행방향 기준 좌→우다.**
            # off 는 좌측 법선 방향이므로 words[0] 이 +h(좌), words[1] 이 -h(우).
            # 예전에는 반대로 넣어 실선과 점선의 좌우가 통째로 뒤바뀌어 있었다
            # (1_2stage_test1 idx=3125 의 B2256W000038 'solid broken' 에서 확인:
            #  solid 라벨이 붙은 쪽의 실제 도색 연속률이 45.9%, broken 라벨이
            #  붙은 쪽이 100% 로 정확히 뒤집혀 있었다).
            h = (width + interval) / 2.0
            out.append(make(words[0], +h))
            out.append(make(words[1], -h))
            split_shapes += 1
        else:
            out.append(make(shape, 0.0))

    pairs = _pair_doubles(out)
    if skipped:
        print(f"[경고] 매핑되지 않은 lane_type 을 건너뜀: {dict(skipped)}")
    print(f"겹선 처리: 같은자리 레코드 쌍 {pairs}쌍, "
          f"복합선(한 레코드 두 줄) {split_shapes}개")
    return out


# singlecrosswalk_set 의 sign_type 을 crosswalk_set(횡단보도 묶음)으로 판별했다.
# 묶음은 신호등을 참조하므로 실제 보행자 횡단 지점이다.
#   묶음에 속함 : 5321x44, 533x12, 534x7   ← 세 종류가 같이 참여한다
#   묶음에 없음 : 5321x16, 544x13, 533x5, 534x3
#   폴리곤 면적 중앙값: 5321 28.3, 533 28.7, 534 35.5, 544 8.6 m2
# 5321/533/534 는 면적이 비슷하고 묶음에도 함께 들어가므로 전부 횡단보도로 본다.
#
# 544 는 면적이 1/3 이고 묶음에 하나도 없다. 영상에 얹어 보니(프레임 440·464·499)
# **대부분 맨 아스팔트 위**에 놓이고 일부만 흰 선에 걸친다 — 도색이 아니라 논리적
# 영역으로 보인다. 그래서 ignore 로 뺀다.
#
# 여기서 unknown 과 ignore 를 구분해야 한다:
#   - unknown : 도색인 건 확실한데 **종류**를 모를 때. 손실에 포함시켜
#               "여기 도색이 있다"까지는 배우게 한다.
#   - ignore  : 가려졌거나 **도색인지조차 불확실**할 때. 손실에서 뺀다.
# 544 를 unknown 으로 두면 없는 도색을 있다고 가르치고, 배경으로 두면 걸쳐 있는
# 흰 선을 무시하라고 가르친다. 13개뿐이라 빼는 손실이 가장 작다.
CROSSWALK_SIGN_TYPES = {"5321", "533", "534"}


def load_crosswalks(mgeo_dir):
    """횡단보도 폴리곤. 라벨에 없으면 크고 선명한 흰 도색이 '배경'으로 학습된다."""
    path = os.path.join(mgeo_dir, "singlecrosswalk_set.json")
    if not os.path.isfile(path):
        print("[경고] singlecrosswalk_set.json 이 없어 횡단보도를 건너뜁니다.")
        return []
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    recs = raw if isinstance(raw, list) else list(raw.values())[0]

    out = collections.Counter()
    result = []
    for r in recs:
        pts = np.asarray(r["points"], dtype=np.float64)
        if len(pts) < 3:
            continue
        sign = str(r.get("sign_type", ""))
        cls = CLASS_CROSSWALK if sign in CROSSWALK_SIGN_TYPES else CLASS_IGNORE
        out[f"{sign}→{CLASS_NAMES.get(cls, 'ignore')}"] += 1
        result.append({
            "idx": r["idx"], "cls": cls, "points": pts,
            "bbox": (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()),
        })
    print(f"횡단보도 sign_type 처리: {dict(out)}")
    return result


def load_link_elevation(mgeo_dir):
    """link_id → 그 링크의 점들. 자차 위치의 노면 높이를 구하는 데 쓴다."""
    path = os.path.join(mgeo_dir, "link_set.json")
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    recs = raw if isinstance(raw, list) else list(raw.values())[0]
    return {r["idx"]: np.asarray(r["points"], dtype=np.float64) for r in recs}


ROAD_PLANE_K = 3   # 로컬 평면을 세울 때 쓰는 최근접점 개수


def _fit_local_plane(pts):
    """점들을 최소자승으로 평면 z = a*x + b*y + c 에 맞춘다.

    점 3개면 그 평면을 정확히 통과한다. 점들이 링크를 따라 거의 일직선으로
    늘어서 있어 폭 방향(횡방향) 정보가 없을 때도(행렬이 rank-deficient),
    lstsq 는 SVD로 최소노름 해를 주기 때문에 계수가 터지지 않고 뱅크
    (횡경사)를 0으로 두는 안전한 근사가 된다 — 3점을 직접 풀 때처럼 값이
    발산하지 않는다.
    """
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    coeffs, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    return coeffs


def build_link_kdtrees(links):
    """링크별로 (x,y) KD-tree 를 만들어 둔다.

    지도 전체에 평면 하나를 세우면 커브・경사가 섞인 도로에서 틀어진다.
    그래서 road_height 가 프레임마다 자차 위치 주변에서 그때그때 로컬
    평면을 새로 세울 수 있도록 준비해 둔다 — "차량 주변의 로컬 평면을
    동적으로 여러 개" 만드는 셈이다. 링크 단위로 트리를 나누는 건 다리
    (overpass) 등 고도가 다른 엉뚱한 링크의 점이 최근접으로 섞여 들어오는
    걸 막기 위해서다 (원래도 "자차가 올라간 링크"로 한정해서 찾았다).
    """
    trees = {}
    for link_id, pts in links.items():
        if len(pts) == 0:
            continue
        tree = cKDTree(pts[:, :2]) if len(pts) >= 2 else None
        trees[link_id] = (tree, pts)
    return trees


def road_height(link_trees, link_id, pos_x, pos_y, fallback, k=ROAD_PLANE_K):
    """자차 주변 k개 최근접점으로 로컬 평면을 세우고 그 위의 z 를 노면
    높이로 쓴다.

    점 하나만 쓰던 예전 방식은 점과 점 사이 경계에서 값이 계단식으로
    뛰었다. 로컬 평면은 그 사이를 매끄럽게 보간하고, 점이 충분히 퍼져
    있으면(교차로・커브 부근) 도로 뱅크(횡경사)까지 반영한다.
    """
    entry = link_trees.get(link_id)
    if entry is None:
        return fallback
    tree, pts = entry
    if tree is None:
        return float(pts[0, 2]) if len(pts) else fallback

    kk = min(k, len(pts))
    _, idx = tree.query([pos_x, pos_y], k=kk)
    idx = np.atleast_1d(idx)
    if kk < 3:
        return float(pts[idx[0], 2])

    a, b, c = _fit_local_plane(pts[idx])
    return float(a * pos_x + b * pos_y + c)


# ======================================================================
# 보닛 (정적 가림막)
# ======================================================================
def detect_bonnet(source, model_path, samples=30):
    """자차 보닛이 가리는 화면 영역을 찾는다.

    지면 폴리곤을 투영하면 **보닛에 가려진 부분까지 칠해진다.** 그대로 두면
    "보닛 위에 차선이 있다"고 학습되므로 ignore 로 빼야 한다.

    입력은 영상이든 PNG 폴더든 상관없다 (find_frame_source 가 돌려준 (kind, path)).

    시간축 분산으로는 못 찾는다 — 보닛이 반사가 있어 하늘·풍경이 비쳐 계속
    변하기 때문이다(실측: y=470 에서도 std 17.3). 대신 기존 모델의 주행가능
    영역(`da`)을 여러 프레임 평균 낸다. drive6 에서 중앙부 drivable 이
    y=380 에서 95%, y=390 에서 36%, y=400 에서 0% 로 뚝 떨어져 경계가 뚜렷하다.
    """
    net = cv2.dnn.readNet(model_path)
    names = net.getUnconnectedOutLayersNames()
    kind, path = source
    if kind == "images":
        step = max(1, len(path) // samples)
        supply = (cv2.imread(q) for q in path[::step])
        stride = 1
    else:
        cap = cv2.VideoCapture(path)
        supply = iter(lambda: cap.read()[1], None)
        stride = 80

    acc, got, n = None, 0, 0
    for img in supply:
        if img is None or got >= samples:
            break
        if n % stride == 0:
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (640, 360), (0, 0, 0),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            out = dict(zip(names, net.forward(names)))
            da = (out["da"][0, 1] > out["da"][0, 0]).astype(np.float32)
            da = cv2.resize(da, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            acc = da if acc is None else acc + da
            got += 1
        n += 1
    if kind == "video":
        cap.release()
    if acc is None:
        return None

    drivable = acc / got
    h, w = drivable.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # 열마다 아래에서 올라가며 주행가능이 살아나는 첫 행까지를 보닛으로 본다.
    # 경계가 곡선이라 행 하나로 자르면 한쪽이 과하게 잘린다.
    for x in range(w):
        y = h - 1
        while y > h // 2 and drivable[y, x] < 0.5:
            y -= 1
        mask[y + 1:, x] = 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))


# ======================================================================
# 라벨 렌더링
# ======================================================================
# 녹화본별 정합 보정. 지도와 시뮬레이터 도색 사이에 남는 오차를 여기서 흡수한다.
# 원인이 확정되지 않았으므로 값을 하드코딩하지 않고 측정해서 넣는다 (--lat-offset,
# --yaw-offset). lap1_full 실측: 편차(m) = +0.278 - 0.0158 x 거리 로, 상수항은
# 가로 위치 오차, 기울기는 yaw 오차 -0.91도 에 해당했다.
LAT_OFFSET_M = 0.0      # +면 라벨을 좌측(+y)으로 민다
YAW_OFFSET_DEG = 0.0    # 자차 yaw 에 더한다


# 차체 자세(roll/pitch)를 쓸지. 도로 경사·뱅크와 서스펜션 때문에 정지 중에도
# 0 이 아니다 (실측: 정지 프레임에서 pitch +0.68도). pitch 오차는 가로가 아니라
# 세로(거리)로 터진다 — d ~ h/theta 이므로 delta_d ~ -d^2*delta/h 라 10m 에서
# 2도면 2.9m 다. 직선에서는 차선이 진행방향과 나란해 티가 안 나지만 커브에서는
# 그 세로 오차가 그대로 가로 오차가 된다.
USE_EGO_ATTITUDE = True
# MORAI 의 pitch/roll 부호 관례는 문서로 확정되지 않아 오버레이로 고른다.
EGO_PITCH_SIGN = 1.0
EGO_ROLL_SIGN = 1.0


def rot_vehicle_to_world(yaw_deg, pitch_deg, roll_deg):
    """차량 좌표(x전방 y좌측 z상방) → 월드(ENU) 회전 행렬. R = Rz Ry Rx."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def to_ego(points_map, row, road_z=None):
    """지도 ENU → 자차 좌표 (x 전방, y 좌측, z 상방).

    **자차의 xyz 와 rpy 를 모두 쓴다.** 예전에는 yaw 만 2D 로 돌리고 높이는
    링크 노면 z 를 기준으로 삼았는데, 그러면 (a) 도로 경사·뱅크가 통째로
    빠지고 (b) 카메라 높이가 실제와 어긋난다.

    `pos_z` 는 노면이 아니라 **차체 원점(후륜축 중심, 차축 높이)** 이다 —
    링크 노면보다 일정하게 +0.35m 높은 것으로 실측했다 (still +0.363,
    curve1 +0.340). 그래서 원점을 pos_z 로 두면 카메라 높이가 저절로
    pos_z + 1.20 - 노면z = 약 1.55m 로 맞는다. 예전 코드는 1.20m 였다.
    """
    ego_xyz = np.array([row["pos_x"], row["pos_y"],
                        row["pos_z"] if "pos_z" in row else (road_z or 0.0)])
    if USE_EGO_ATTITUDE and "pitch" in row and "roll" in row:
        pitch = EGO_PITCH_SIGN * _wrap180(row["pitch"])
        roll = EGO_ROLL_SIGN * _wrap180(row["roll"])
    else:
        pitch = roll = 0.0
    R = rot_vehicle_to_world(row["yaw"] + YAW_OFFSET_DEG, pitch, roll)

    # 월드 → 차량 = R^T (P - C).  (P-C) @ R 로 계산하면 같다.
    out = (points_map - ego_xyz) @ R
    out[:, 1] += LAT_OFFSET_M
    return out


def _wrap180(a):
    """MORAI 는 자세각을 0~360 으로 주기도 한다. -180~180 으로 편다."""
    return ((a + 180.0) % 360.0) - 180.0


def _dash_keep(s, dash_on, dash_off):
    """대시 구간이면 True. 점선을 통으로 그리면 모델이 실선처럼 배운다.

    지도의 dash_interval 을 그대로 믿을 때만 쓴다 (프레임 이미지가 없을 때의
    대비책). 실측으로 확인했듯 이 값이 실제 렌더링된 대시 리듬과 다른
    레코드가 있다 — 같은 lane_type=503, 같은 선언값(3.0/5.0)인데 한 레코드는
    실측 주기 8.0m(선언과 일치), 다른 레코드는 9.4m(17~21% 김)로 서로 다르다.
    전역 배율 하나로 보정할 수 없는 레코드별 편차라, 프레임 이미지가 있으면
    `_paint_mask` 로 실제 도색 여부를 직접 읽는 쪽이 훨씬 정확하다.
    """
    period = dash_on + dash_off
    if period <= 1e-6:
        return np.ones(len(s), dtype=bool)
    return np.mod(s, period) < dash_on


def _paint_mask(frame, cls):
    """프레임에서 이 클래스 색상의 도색으로 보이는 픽셀 마스크.

    점선 on/off 위상은 지도 값(dash_interval)보다 실제 화면의 색이 훨씬
    믿을 만하다 — 지도값은 레코드마다 최대 21% 어긋나는 걸 실측으로 확인했다
    (위 _dash_keep 참고). 그래서 점선은 지도 위상으로 자르지 않고 리본을
    통으로 그린 다음, 실제로 흰색/황색이 아닌 픽셀만 지운다.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if cls == CLASS_YELLOW:
        return cv2.inRange(hsv, (15, 90, 90), (38, 255, 255)) > 0
    return cv2.inRange(hsv, (0, 0, 170), (180, 50, 255)) > 0


def assign_lane_ids(vectors):
    """자차 기준 횡방향 오프셋으로 정렬해 상대 차선 ID 를 붙인다.

    ego 좌우 2개로 한정하지 않는다 — 화면에 보이는 모든 경계에 순번을 매겨야
    인접 차로 경계까지 식별되고, "이 선들이 서로 다른 선이다"라는 정보 자체가
    라벨에 들어간다. 자차 좌표계는 y 가 좌측(+)이므로:

        ego_left   가장 가까운 좌측 경계 (y > 0 중 최소)
        left_2     그 바깥쪽 … left_3 …
        ego_right  가장 가까운 우측 경계 (y < 0 중 |y| 최소)
        right_2    그 바깥쪽 …

    정지선은 진행방향에 직각이라 좌/우 개념이 없다 — 별도로 `stopline_N`.

    **자차 주변을 실제로 지나가는 경계만 번호를 받는다.** 교차로에서는 다른
    방향 도로의 경계가 화면에 잔뜩 들어오는데(실측: 한 프레임에 70개, 횡방향
    60m 밖까지) 그것까지 순번을 매기면 `left_32` 같은 무의미한 ID 가 생기고
    진짜 인접 차로 번호가 밀린다. 범위 밖 경계는 `lane_id=None` 으로 두되
    구조화 라벨에는 그대로 남긴다 — 나중에 다른 용도로 쓸 수 있으므로.
    """
    def near(v):
        return (abs(v["ego_lat_m"]) <= LANE_ID_MAX_LAT
                and v.get("ego_dist_m", 0.0) <= LANE_ID_MAX_DIST)

    for v in vectors:
        v["lane_id"] = None
        v["lane_index"] = None

    left = sorted((v for v in vectors
                   if near(v) and v["ego_lat_m"] > 0 and v["cls"] != CLASS_STOPLINE),
                  key=lambda v: v["ego_lat_m"])
    right = sorted((v for v in vectors
                    if near(v) and v["ego_lat_m"] <= 0 and v["cls"] != CLASS_STOPLINE),
                   key=lambda v: -v["ego_lat_m"])
    stop = sorted((v for v in vectors
                   if near(v) and v["cls"] == CLASS_STOPLINE),
                  key=lambda v: v["ego_fwd_m"])

    for k, v in enumerate(left):
        v["lane_id"] = "ego_left" if k == 0 else f"left_{k + 1}"
        v["lane_index"] = -(k + 1)
    for k, v in enumerate(right):
        v["lane_id"] = "ego_right" if k == 0 else f"right_{k + 1}"
        v["lane_index"] = k + 1
    for k, v in enumerate(stop):
        v["lane_id"] = f"stopline_{k + 1}"
        v["lane_index"] = None
    return vectors


def render_frame_labels(cam, boundaries, link_trees, row, fallback_z, crosswalks=(),
                        bonnet=None, frame=None, vectors=None):
    """한 프레임의 클래스 맵 (h, w) uint8 을 만든다.

    `frame` (원본 카메라 이미지)을 주면 점선 on/off 를 지도의 dash_interval
    대신 **실제 화면 색상**으로 정한다 — 리본을 실선처럼 통으로 그린 뒤
    도색처럼 안 보이는 픽셀만 지운다. `frame` 이 없으면(예: 이미지 없이
    기하만 확인할 때) 예전처럼 지도의 dash_interval 로 자른다.

    `vectors` 에 리스트를 주면 래스터에 굽기 전의 폴리라인(구조화 라벨)을
    거기에 채워 준다. 차선 ID 는 프레임 전체를 모은 뒤 `assign_lane_ids` 로
    붙인다.
    """
    label = np.zeros((cam.height, cam.width), dtype=np.uint8)
    ex, ey = row["pos_x"], row["pos_y"]
    z0 = road_height(link_trees, row.get("link_id", ""), ex, ey, fallback_z)

    # 횡단보도를 먼저 깔고 그 위에 차선을 그린다. 겹치면 선이 이긴다.
    for c in crosswalks:
        xmin, xmax, ymin, ymax = c["bbox"]
        if (xmin - ex > MAX_RANGE or ex - xmax > MAX_RANGE
                or ymin - ey > MAX_RANGE or ey - ymax > MAX_RANGE):
            continue
        ego = to_ego(c["points"], row, z0)
        if (ego[:, 0] > MAX_RANGE).all():
            continue
        clipped = clip_near(cam.to_camera(ego))
        if clipped is None:
            continue
        uv, _ = cam.project_camera(clipped)
        cv2.fillPoly(label, [uv.astype(np.int32)], int(c["cls"]))

    for b in boundaries:
        xmin, xmax, ymin, ymax = b["bbox"]
        if (xmin - ex > MAX_RANGE or ex - xmax > MAX_RANGE
                or ymin - ey > MAX_RANGE or ey - ymax > MAX_RANGE):
            continue

        ego = to_ego(b["points"], row, z0)
        near = (ego[:, 0] > -5.0) & (np.abs(ego[:, 1]) < MAX_RANGE) & (ego[:, 0] < MAX_RANGE)
        if not near.any():
            continue

        # 차선폭만큼 좌우로 벌린 리본을 만들어 원근이 자동으로 반영되게 한다.
        d = np.diff(ego[:, :2], axis=0)
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.maximum(norm, 1e-9)
        nx, ny = -d[:, 1], d[:, 0]                  # 진행방향의 좌측 법선
        hw = b["width"] / 2.0
        off = b["lat_offset"]                       # 겹선이면 좌우로 밀어 그린다

        # frame 이 있으면 지도의 dash_interval 을 아예 믿지 않는다 — 레코드마다
        # 최대 21% 어긋나는 걸 실측으로 확인했다 (_dash_keep 참고). 그래서 리본을
        # 통으로(keep=all True) 그린 뒤 _paint_mask 로 실제 도색 색이 아닌 픽셀만
        # 지운다. 여기서 지도 위상으로 먼저 잘라버리면 그 위에 페인트 마스크가
        # 또 깎아내는 이중 필터링이 되어, 유도선처럼 dash 간격이 촘촘한
        # (0.75/0.75, 0.5/0.5) 클래스는 거의 다 사라진다 — frame 없이 기하만 볼
        # 때만 지도 위상으로 대략의 on/off 구조를 준다.
        use_paint_trim = b["broken"] and frame is not None
        if use_paint_trim:
            keep = np.ones(len(ego), dtype=bool)
        elif b["broken"]:
            keep = _dash_keep(b["s"], b["dash_on"], b["dash_off"])
        else:
            keep = np.ones(len(ego), dtype=bool)

        left = ego[:, :3].copy()
        right = ego[:, :3].copy()
        left[:-1, 0] += nx * (off + hw)
        left[:-1, 1] += ny * (off + hw)
        right[:-1, 0] += nx * (off - hw)
        right[:-1, 1] += ny * (off - hw)
        left[-1] = left[-2]
        right[-1] = right[-2]

        uvl, vl = cam.project(left)
        uvr, vr = cam.project(right)

        quads = []
        centers = []            # 구조화 라벨용: 리본 중심선의 이미지 좌표
        for i in range(len(ego) - 1):
            if not (keep[i] and keep[i + 1] and near[i] and near[i + 1]):
                continue
            if not (vl[i] and vl[i + 1] and vr[i] and vr[i + 1]):
                continue
            quads.append(np.array([uvl[i], uvl[i + 1], uvr[i + 1], uvr[i]],
                                  dtype=np.int32))
            centers.append(((uvl[i] + uvr[i]) / 2.0, i))
        if not quads:
            continue

        # 리본의 실제 중심선. ego 원본이 아니라 lat_offset 을 반영해야
        # 겹선·복합선의 두 줄이 서로 다른 횡위치로 구분된다.
        rows_ = np.array([i for _, i in centers])
        cen = ego[:, :2].copy()
        cen[:-1, 0] += nx * off
        cen[:-1, 1] += ny * off
        cen[-1] = cen[-2]
        sub = cen[rows_]
        # **기준점은 "최소 전방거리"가 아니라 "자차와의 최근접점"이어야 한다.**
        # 전방거리만 보면 70m 앞에서 시작하는 먼 도로의 경계가 그 지점의
        # 횡오프셋으로 번호를 받아버려, 차로 순번이 뒤죽박죽이 된다.
        fwd_ok = sub[:, 0] > 0.0
        pool = np.where(fwd_ok)[0] if fwd_ok.any() else np.arange(len(sub))
        ref = int(pool[np.argmin((sub[pool] ** 2).sum(axis=1))])

        # 자차 도로 밖(반대편 차도 등)은 라벨에 넣지 않는다.
        if abs(float(sub[ref, 1])) > LABEL_MAX_LAT:
            continue
        # 사람이 직접 뺀 경계 (EditLabels.py 로 고른다). 지도 레코드 단위라
        # 한 번 빼면 그 레코드가 보이는 **모든 프레임**에서 빠진다.
        if b["idx"] in EXCLUDED_BOUNDARIES:
            continue

        # 픽셀 래스터만으로는 나중에 차선 ID 를 복원할 수 없다 — 같은 클래스의
        # 여러 경계가 한 캔버스에 섞여 그려지기 때문이다. 그래서 굽기 전에
        # 폴리라인을 따로 모아 둔다. 차선 ID(상대 순번)는 프레임 전체를 모은 뒤
        # 자차 기준 횡방향 오프셋으로 정렬해서 붙인다.
        if vectors is not None and centers:
            vectors.append({
                "map_idx": b["idx"],
                "cls": int(b["cls"]),
                "cls_name": CLASS_NAMES.get(b["cls"], "unknown"),
                "broken": bool(b["broken"]),
                "lane_type": int(b["lane_type"]),
                "width_m": round(float(b["width"]), 3),
                "ego_lat_m": round(float(sub[ref, 1]), 3),
                "ego_fwd_m": round(float(sub[ref, 0]), 3),
                "ego_dist_m": round(float(np.hypot(*sub[ref])), 3),
                "points_uv": [[round(float(u), 1), round(float(v), 1)]
                              for (u, v), _ in centers],
            })

        if use_paint_trim:
            # 지도 dash_interval 로 만든 칸을 캔버스에 그린 다음, 실제
            # 도색처럼 보이는 픽셀만 남긴다.
            ribbon = np.zeros(label.shape, dtype=np.uint8)
            cv2.fillPoly(ribbon, quads, 255)
            painted = _paint_mask(frame, b["cls"]) & (ribbon > 0)
            label[painted] = b["cls"]
        else:
            cv2.fillPoly(label, quads, int(b["cls"]))

    # 보닛에 가려진 부분은 마지막에 통째로 ignore 로 덮는다.
    if bonnet is not None:
        label[bonnet > 0] = CLASS_IGNORE
    return label


def overlay(frame, label, alpha=0.55):
    """라벨을 원본 위에 반투명으로 얹는다. 정합을 눈으로 보는 게 목적이다."""
    out = frame.copy()
    tint = np.zeros_like(frame)
    hit = np.zeros(label.shape, dtype=bool)
    for cls, color in CLASS_COLORS.items():
        m = label == cls
        if m.any():
            tint[m] = color
            hit |= m
    out[hit] = (out[hit] * (1 - alpha) + tint[hit] * alpha).astype(np.uint8)
    return out


# ======================================================================
# 실행
# ======================================================================
def _label_bar(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return out


def load_meta(path):
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


# ======================================================================
# 카메라 지연 보정 (예전 녹화본용)
# ======================================================================
def _unwrap_deg(vals):
    """각도 배열의 ±180 점프를 펴서 보간이 가능하게 만든다."""
    out = [vals[0]]
    for v in vals[1:]:
        out.append(out[-1] + ((v - out[-1] + 180.0) % 360.0) - 180.0)
    return np.asarray(out)


def build_pose_timeline(meta):
    """meta.jsonl 의 (카메라 시각, pose) 를 보간 가능한 배열로 모은다."""
    ok = all(k in meta[0] for k in ("cam_sec", "pos_x", "yaw"))
    if not ok:
        return None
    return {
        "t": np.array([r["cam_sec"] + r["cam_nsec"] * 1e-9 for r in meta]),
        "pos_x": np.array([r["pos_x"] for r in meta]),
        "pos_y": np.array([r["pos_y"] for r in meta]),
        "pos_z": np.array([r.get("pos_z", 0.0) for r in meta]),
        "yaw": _unwrap_deg([_wrap180(r["yaw"]) for r in meta]),
        "pitch": np.array([_wrap180(r.get("pitch", 0.0)) for r in meta]),
        "roll": np.array([_wrap180(r.get("roll", 0.0)) for r in meta]),
        "vel": np.array([r.get("signed_vel") or 0.0 for r in meta]),
    }


def retimed_row(tl, meta, idx, shift):
    """idx 프레임의 pose 를 `shift` 초만큼 옮긴 시점으로 다시 보간한다.

    위치는 양 끝점의 속도벡터를 접선으로 쓰는 3차 Hermite 로 (커브에서 현을
    긋지 않도록), 각도는 선형으로 보간한다. 원래 행의 나머지 필드(link_id,
    brake 등)는 그대로 물려받는다.
    """
    row = dict(meta[idx])
    if tl is None or abs(shift) < 1e-9:
        return row

    t = tl["t"]
    target = t[idx] + shift
    i = int(np.clip(np.searchsorted(t, target) - 1, 0, len(t) - 2))
    t0, t1 = t[i], t[i + 1]
    dt = t1 - t0
    u = 0.0 if dt <= 1e-9 else (target - t0) / dt

    def lin(key):
        return float(tl[key][i] + (tl[key][i + 1] - tl[key][i]) * u)

    y0, y1 = tl["yaw"][i], tl["yaw"][i + 1]
    v0 = (tl["vel"][i] * math.cos(math.radians(y0)),
          tl["vel"][i] * math.sin(math.radians(y0)))
    v1 = (tl["vel"][i + 1] * math.cos(math.radians(y1)),
          tl["vel"][i + 1] * math.sin(math.radians(y1)))
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    row["pos_x"] = float(h00 * tl["pos_x"][i] + h10 * dt * v0[0]
                         + h01 * tl["pos_x"][i + 1] + h11 * dt * v1[0])
    row["pos_y"] = float(h00 * tl["pos_y"][i] + h10 * dt * v0[1]
                         + h01 * tl["pos_y"][i + 1] + h11 * dt * v1[1])
    row["pos_z"] = lin("pos_z")
    row["yaw"] = lin("yaw")
    row["pitch"] = lin("pitch")
    row["roll"] = lin("roll")
    return row


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def find_frame_source(recording):
    """녹화 폴더에서 프레임 출처를 찾는다. (kind, path) 를 돌려준다.

    학습 데이터는 PNG 로 저장한다 — mp4 는 H.264 손실 압축이라 1~3px 얇은 차선을
    뭉갠다. 그래서 두 형식을 다 읽을 수 있어야 하고, **PNG 폴더를 우선**한다
    (둘 다 있으면 mp4 는 육안 확인용으로 남겨둔 것이다).
    """
    frames_dir = os.path.join(recording, "frames")
    if os.path.isdir(frames_dir):
        names = sorted(n for n in os.listdir(frames_dir)
                       if n.lower().endswith(IMAGE_EXTS))
        if names:
            return "images", [os.path.join(frames_dir, n) for n in names]

    names = sorted(n for n in os.listdir(recording)
                   if n.lower().endswith(IMAGE_EXTS) and n != "bonnet_mask.png")
    if names:
        return "images", [os.path.join(recording, n) for n in names]

    video = os.path.join(recording, "drive.mp4")
    if os.path.isfile(video):
        return "video", video
    raise SystemExit(f"프레임을 찾을 수 없습니다 (frames/*.png 또는 drive.mp4): {recording}")


def frame_index_of(path):
    """파일명에서 프레임 번호를 읽는다 (frame_000123.png → 123). 없으면 None."""
    m = re.findall(r"\d+", os.path.basename(path))
    return int(m[-1]) if m else None


def read_frames(source, indices):
    """지정한 인덱스의 프레임만 뽑는다. idx 는 meta.jsonl 의 idx 다.

    **이미지 폴더에서는 리스트 위치가 아니라 파일명의 번호를 쓴다.** 학습용
    프레임은 N장마다 하나씩만 저장하므로(frame_000000, frame_000008, ...)
    위치로 찾으면 8번 프레임 그림에 1번 프레임의 자차 위치로 만든 라벨이
    붙는다. 연속 저장일 때만 우연히 맞는다.
    """
    kind, path = source
    want = sorted(set(indices))

    if kind == "images":
        table = {}
        for q in path:
            i = frame_index_of(q)
            if i is not None:
                table[i] = q
        if not table:
            # 번호가 없는 파일명이면 정렬 순서로 대체한다 (권장하지 않음)
            print("[경고] 파일명에 프레임 번호가 없어 정렬 순서를 씁니다. "
                  "N장마다 저장했다면 라벨이 어긋납니다.")
            table = dict(enumerate(path))

        got = {}
        for i in want:
            q = table.get(i)
            if q is None:
                continue
            img = cv2.imread(q)
            if img is not None:
                got[i] = img
        return got

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없습니다: {path}")
    got, cursor = {}, 0
    for target in want:
        while cursor <= target:
            ok, img = cap.read()
            if not ok:
                cap.release()
                return got
            cursor += 1
        got[target] = img
    cap.release()
    return got


def main():
    global LAT_OFFSET_M, YAW_OFFSET_DEG, LABEL_MAX_LAT
    global EXCLUDED_BOUNDARIES, EXCLUDED_FRAMES
    ap = argparse.ArgumentParser(
        description="HD맵을 카메라에 투영해 차선 종류 라벨을 만든다 "
                    "(지금은 초점거리 확정을 위한 오버레이 검증 단계)")
    ap.add_argument("--recording", required=True,
                    help="meta.jsonl 과 프레임(frames/*.png 또는 drive.mp4)이 있는 폴더")
    ap.add_argument("--mgeo", required=True, help="MGeo 폴더 (lane_boundary_set.json 등)")
    ap.add_argument("--cam-set", required=True, help="cam_set.json 경로")
    ap.add_argument("--sensor-id", type=int, default=1, help="카메라 SensorUniqueID (기본 1=전방)")
    ap.add_argument("--fov-axis", choices=["horizontal", "vertical", "both"], default="both",
                    help="cameraFOV 를 수평/수직 중 무엇으로 볼지. both 면 나란히 비교")
    ap.add_argument("--frames", default=None,
                    help="확인할 프레임 인덱스 (쉼표 구분). 생략하면 녹화 폴더에 "
                         "저장된 프레임 전부")
    ap.add_argument("--out", default="label_check", help="결과를 저장할 폴더")
    ap.add_argument("--detect-bonnet", action="store_true",
                    help="보닛 가림 영역을 찾아 녹화 폴더에 bonnet_mask.png 로 저장한다")
    ap.add_argument("--no-bonnet", action="store_true",
                    help="bonnet_mask.png 가 있어도 무시한다")
    ap.add_argument("--lat-offset", type=float, default=0.0,
                    help="라벨을 좌측(+y)으로 미는 보정 (m). 지도-도색 가로 편차를 "
                         "흡수한다. 녹화본마다 측정해서 넣는다")
    ap.add_argument("--yaw-offset", type=float, default=0.0,
                    help="자차 yaw 에 더하는 보정 (도). 편차가 거리에 비례해 변할 때 쓴다")
    ap.add_argument("--label-max-lat", type=float, default=LABEL_MAX_LAT,
                    help="자차 기준 이 횡거리(m)를 넘는 경계는 라벨에 넣지 않는다 "
                         "(기본: 제한 없음). 자동 판정이라 부정확할 수 있어 기본은 꺼둔다")
    ap.add_argument("--exclude-file", default=None,
                    help=f"사람이 고른 제외 경계 목록 JSON (기본: 녹화 폴더의 "
                         f"{EXCLUDE_FILENAME}). EditLabels.py 로 만든다")
    ap.add_argument("--crop-top", type=int, default=CROP_TOP,
                    help=f"파생물(마스크·구조화 라벨·오버레이)에서 잘라낼 위쪽 픽셀 수 "
                         f"(기본 {CROP_TOP}). 원본 프레임은 건드리지 않는다. "
                         f"0 이면 자르지 않음")
    ap.add_argument("--train-classes", choices=sorted(TRAIN_CLASS_MAPS), default=None,
                    help="학습용 클래스 리맵 스킴. lane5 = 배경/백색실선/백색점선/"
                         "황색/정지선 (유도선·안전지대·횡단보도는 ignore)")
    ap.add_argument("--save-masks", action="store_true",
                    help="오버레이 말고 학습용 라벨 마스크 PNG 를 masks/ 에 저장한다")
    ap.add_argument("--save-vectors", action="store_true",
                    help="차선 ID 가 붙은 구조화 라벨을 vectors/ 에 JSON 으로 저장한다")
    ap.add_argument("--time-offset", type=float, default=None,
                    help="pose 를 이 초만큼 옮겨서 다시 보간한다 (음수 = 과거). "
                         "생략하면 meta.jsonl 에 cam_latency 가 없는 예전 녹화본에만 "
                         f"-{CAMERA_PIPELINE_LATENCY}s 를 자동 적용한다")
    args = ap.parse_args()

    LAT_OFFSET_M = args.lat_offset
    YAW_OFFSET_DEG = args.yaw_offset
    LABEL_MAX_LAT = args.label_max_lat
    exclude_path = args.exclude_file or os.path.join(args.recording, EXCLUDE_FILENAME)
    EXCLUDED_BOUNDARIES, EXCLUDED_FRAMES = load_exclusions(exclude_path)
    if not math.isinf(LABEL_MAX_LAT):
        print(f"라벨 범위: |ego_lat| <= {LABEL_MAX_LAT}m")
    if EXCLUDED_BOUNDARIES:
        print(f"사람이 제외한 경계 {len(EXCLUDED_BOUNDARIES)}개 적용 "
              f"(전역: {global_exclude_path()})")
    if args.lat_offset or args.yaw_offset:
        print(f"정합 보정: 가로 {args.lat_offset:+.3f} m, yaw {args.yaw_offset:+.3f} 도")

    meta = load_meta(os.path.join(args.recording, "meta.jsonl"))

    # 카메라 지연 보정. RecordDrive 가 이미 반영했으면(cam_latency 필드가 있으면)
    # 두 번 넣지 않는다. 예전 녹화본은 여기서 프레임 타임라인을 다시 보간해 되돌린다.
    already = meta[0].get("cam_latency") if meta else None
    if args.time_offset is not None:
        time_shift = args.time_offset
    elif already is None:
        time_shift = -CAMERA_PIPELINE_LATENCY
    else:
        time_shift = 0.0
    pose_tl = build_pose_timeline(meta) if time_shift else None
    if time_shift and pose_tl is None:
        print("[경고] meta.jsonl 에 시각·pose 가 없어 카메라 지연 보정을 건너뜁니다.")
        time_shift = 0.0
    if already is not None:
        print(f"녹화 시 카메라 지연 {already:+.3f}s 가 이미 반영된 녹화본입니다.")
    if time_shift:
        print(f"카메라 지연 보정: pose 를 {time_shift:+.3f}s 시점으로 다시 보간합니다.")

    source = find_frame_source(args.recording)
    print(f"프레임 출처: {source[0]} "
          f"({len(source[1]) if source[0] == 'images' else os.path.basename(source[1])})")
    bonnet_path = os.path.join(args.recording, "bonnet_mask.png")
    if args.detect_bonnet:
        model = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "lane_segmentation.onnx")
        mask = detect_bonnet(source, model)
        if mask is None:
            raise SystemExit("보닛 검출 실패 (프레임을 못 읽었습니다)")
        cv2.imwrite(bonnet_path, mask)
        print(f"보닛 마스크 저장: {bonnet_path}  (가려진 픽셀 {(mask>0).mean()*100:.1f}%)")

    bonnet = None
    if not args.no_bonnet and os.path.isfile(bonnet_path):
        bonnet = cv2.imread(bonnet_path, cv2.IMREAD_GRAYSCALE)
        print(f"보닛 마스크 사용: {bonnet_path} (가려진 픽셀 {(bonnet>0).mean()*100:.1f}%)")

    boundaries = load_boundaries(args.mgeo)
    crosswalks = load_crosswalks(args.mgeo)
    links = load_link_elevation(args.mgeo)
    link_trees = build_link_kdtrees(links)
    fallback_z = float(np.median([b["points"][:, 2].mean() for b in boundaries]))

    print("=" * 62)
    print(f"차선 경계 {len(boundaries)}개, 횡단보도 {len(crosswalks)}개, 링크 {len(links)}개")
    counts = collections.Counter(CLASS_NAMES[b["cls"]] for b in boundaries)
    counts["crosswalk"] = len(crosswalks)
    print(f"클래스 분포: {dict(counts)}")
    print(f"점선 처리 대상: {sum(1 for b in boundaries if b['broken'])}개")

    if args.frames:
        indices = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        # 지정이 없으면 녹화 폴더에 실제로 저장된 프레임 전부.
        indices = sorted(i for i in (frame_index_of(p) for p in source[1])
                         if i is not None) if source[0] == "images" else list(range(len(meta)))
        print(f"프레임 지정이 없어 저장된 {len(indices)}장을 전부 처리합니다.")
    indices = [i for i in indices if i < len(meta)]

    # 사람이 통째로 뺀 프레임 (EditLabels.py 에서 고른다). 이미 만들어 둔
    # 마스크·구조화 라벨이 있으면 같이 지운다 — 뺐는데 파일이 남아 있으면
    # 학습 쪽이 그걸 그대로 읽어 버린다.
    if EXCLUDED_FRAMES:
        dropped = [i for i in indices if i in EXCLUDED_FRAMES]
        indices = [i for i in indices if i not in EXCLUDED_FRAMES]
        print(f"사람이 제외한 프레임 {len(dropped)}장 건너뜀")
        for i in dropped:
            for d, ext in ((os.path.join(args.recording, "masks"), "png"),
                           (os.path.join(args.recording, "vectors"), "json"),
                           (args.out, None)):
                stale = (os.path.join(d, f"frame_{i:06d}.png") if ext is None
                         else os.path.join(d, f"{i:06d}.{ext}"))
                if os.path.isfile(stale):
                    os.remove(stale)

    # pose_extrapolated 가 붙은 프레임은 카메라 촬영 시각을 상태 이력으로 보간하지
    # 못해 "가장 가까운 값"을 대신 쓴 것이다 — 실제 촬영 시점의 자차 위치/자세와
    # 다를 수 있어 라벨이 크게 어긋난다(실측: lap4_full idx=188, 교차로에서
    # 크게 틀어짐). 라벨 생성에는 못 믿을 프레임이라 걸러낸다.
    extrapolated = [i for i in indices if meta[i].get("pose_extrapolated") is not None]
    if extrapolated:
        indices = [i for i in indices if i not in extrapolated]
        print(f"[경고] pose_extrapolated 프레임 {len(extrapolated)}개 건너뜀: {extrapolated}")

    # pose_dt(보간에 쓴 두 상태 샘플 사이 간격)가 크면 그 구간 안에서 등속·등각속도
    # 가정이 깨지기 쉽다 — 회전 중 실측으로 pose_dt 가 커지는 걸 확인했다(11sunny1:
    # 회전 구간 중앙값이 직선의 거의 2배). 임계값을 넘는 프레임은 못 믿고 버린다.
    high_dt = [i for i in indices
               if (meta[i].get("pose_dt") or 0.0) > POSE_DT_MAX]
    if high_dt:
        indices = [i for i in indices if i not in high_dt]
        print(f"[경고] pose_dt > {POSE_DT_MAX}s 프레임 {len(high_dt)}개 건너뜀: {high_dt}")

    # 브레이크를 밟는 중이면(감속 중) 보간 구간 안에서 속도가 빠르게 바뀌어
    # 위치 보간(등속 가정)이 틀어진다 — 정지선/차선이 실제보다 자차 쪽으로
    # 당겨져 보이는 증상이 이것으로 확인됐다(lap4_full idx=184). 브레이크가
    # 걸려 있는 프레임은 걸러낸다.
    braking = [i for i in indices
               if (meta[i].get("brake") or 0.0) > BRAKE_SKIP_THRESHOLD]
    if braking:
        indices = [i for i in indices if i not in braking]
        print(f"[경고] 브레이크 작동 중 프레임 {len(braking)}개 건너뜀: {braking}")

    frames = read_frames(source, indices)
    if not frames:
        raise SystemExit("프레임을 읽지 못했습니다.")

    axes = ["horizontal", "vertical"] if args.fov_axis == "both" else [args.fov_axis]
    cams = {}
    for ax in axes:
        base = load_camera(args.cam_set, args.sensor_id, ax)
        sample = next(iter(frames.values()))
        h, w = sample.shape[:2]
        cam = base if (w, h) == (base.width, base.height) else base.scaled(w, h)
        cam = cam.cropped(args.crop_top)
        cams[ax] = cam
        print(f"  [{ax:>10}] FOV {cam.fov_deg}도  원본 {cam.native_width}x{cam.native_height}"
              f" → 저장 {cam.width}x{cam.height}   fx={cam.fx:.1f} fy={cam.fy:.1f}"
              f"  (cx={cam.cx:.0f} cy={cam.cy:.0f}, pitch={math.degrees(cam.pitch):.1f}도)")
    print("=" * 62)

    os.makedirs(args.out, exist_ok=True)
    mask_dir = os.path.join(args.recording, "masks")
    vec_dir = os.path.join(args.recording, "vectors")
    if args.save_masks:
        os.makedirs(mask_dir, exist_ok=True)
    if args.save_vectors:
        os.makedirs(vec_dir, exist_ok=True)
    if args.train_classes:
        names = TRAIN_CLASS_NAMES[args.train_classes]
        print(f"학습용 클래스 리맵: {args.train_classes} = {names} "
              f"(그 외 도색은 {CLASS_IGNORE}=ignore)")
    if args.crop_top:
        c0 = cams[axes[0]]
        print(f"상단 크롭: {args.crop_top}px 제거 → 파생물 {c0.width}x{c0.height}, "
              f"cy {c0.cy + args.crop_top:.0f} → {c0.cy:.0f} "
              f"(원본 프레임은 그대로. 학습 때 같은 값으로 잘라야 함)")
        if bonnet is not None:
            bonnet = bonnet[args.crop_top:]

    for idx in indices:
        if idx not in frames:
            continue
        frame = frames[idx][args.crop_top:] if args.crop_top else frames[idx]
        row = retimed_row(pose_tl, meta, idx, time_shift)
        panels = []
        for ax in axes:
            vectors = [] if (args.save_vectors and ax == axes[0]) else None
            label = render_frame_labels(cams[ax], boundaries, link_trees, row, fallback_z,
                                        crosswalks, bonnet, frame, vectors)
            if vectors is not None:
                assign_lane_ids(vectors)
                rec = {
                    "idx": idx,
                    # 원본 프레임 경로. 이 파일은 자르지 않았으므로 학습 쪽에서
                    # crop_top 만큼 잘라내야 좌표가 맞는다.
                    "image": f"frames/{idx:06d}.png",
                    "crop_top": int(args.crop_top),
                    "width": int(cams[ax].width), "height": int(cams[ax].height),
                    "train_classes": args.train_classes,
                    "boundaries": vectors,
                }
                with open(os.path.join(vec_dir, f"{idx:06d}.json"), "w",
                          encoding="utf-8") as fp:
                    json.dump(rec, fp, ensure_ascii=False)
            if args.save_masks and ax == axes[0]:
                cv2.imwrite(os.path.join(mask_dir, f"{idx:06d}.png"),
                            remap_classes(label, args.train_classes))
            marked = (label > 0) & (label != CLASS_IGNORE)
            painted = int(marked.sum())
            ratio = painted / label.size * 100
            ign = float((label == CLASS_IGNORE).mean() * 100)
            panels.append(_label_bar(
                overlay(frame, label),
                f"FOV={ax}  fx={cams[ax].fx:.0f} fy={cams[ax].fy:.0f}"
                f"  marking {ratio:.2f}%  ignore {ign:.0f}%"))
            print(f"[{idx:5d}] {ax:>10}  표시 픽셀 {painted:6d} ({ratio:5.2f}%)"
                  f"  ignore {ign:4.1f}%")

        grid = np.hstack([_label_bar(frame, "source")] + panels)
        path = os.path.join(args.out, f"frame_{idx:06d}.png")
        cv2.imwrite(path, grid)

    print(f"\n저장 완료: {args.out}/frame_*.png")
    if args.save_masks:
        print(f"           {mask_dir}/*.png  (학습용 라벨 마스크)")
    if args.save_vectors:
        print(f"           {vec_dir}/*.json  (차선 ID 포함 구조화 라벨)")
    print("→ 투영선이 영상 속 차선 위에 얹히는 쪽이 맞는 가설입니다.")


if __name__ == "__main__":
    main()
