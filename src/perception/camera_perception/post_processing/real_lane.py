#!/usr/bin/env python3
"""차선 인식 후처리 **합본** - 카메라 프레임 하나 -> 차선 / 정지선.

    Segmentation -> Morphology -> Lane pixels -> Calibration(지면 역투영)
      -> Boundary grouping -> RANSAC 곡선 -> Hungarian+Kalman 추적 -> Lane ID
    (정지선은 Calibration 만 공유하는 별도 가지)

**조감도(BEV) 래스터를 만들지 않는다.** 픽셀을 바로 지면으로 역투영해 자차
좌표(미터)에서 모든 후처리를 한다. 워프 보간이 얇은 차선을 끊어 먹지 않고,
0.05m 격자에 갇히지 않으며, 쓰지도 않는 배경 32만 픽셀을 채우지 않는다.

===========================================================================
이 파일이 정본이다
===========================================================================
`pipeline/` 의 단계별 모듈들은 이 파일을 만든 출처이고 **참고용으로 남겨 둔
것**이다. 앞으로 고칠 곳은 여기다. 두 곳을 같이 고치면 갈라지고, 갈라지면
어느 쪽이 맞는지 아무도 모르게 된다.

합치면서 이름 충돌만 풀었다 - 모듈마다 `apply` / `format_stats` 가 있었다.

    morphology          lane_pixels        to_ground
    group_boundaries    fit_curves         assign_lane_ids
    detect_stopline     Tracker            GuideLink

===========================================================================
외부 의존은 torch / cv2 / numpy 뿐이다
===========================================================================
학습 쪽(`seg_model` / `seg_dataset`)을 import 하지 않는다. 그 대신 **규격을
체크포인트에서 읽는다** - `best.pt` 가 input_size / class_names / num_classes /
backbone 을 들고 있으므로 상수로 박을 이유가 없고, 박지 않으면 어긋날 수도
없다. 복사본이 되는 것은 모델 정의 하나뿐인데 구조가 바뀌면
`load_state_dict` 가 키 불일치로 **즉시 터진다** - 조용히 틀리지 않는다.

**`best.pt` 는 저장소에 없다** (98MB, gitignore). 따로 받아서 이 파일 옆이나
상위 폴더에 두면 `default_checkpoint()` 가 찾는다.

===========================================================================
쓰는 법
===========================================================================
    seg = Segmenter()                       # 체크포인트/카메라를 한 번만 올린다
    tracker = Tracker()                     # 프레임을 넘는 상태를 갖는다
    link = GuideLink()                      # 좌측 경계에 이어지는 유도선 기억

    mask, crop = seg.apply(frame_bgr)
    clean, _   = morphology(mask, seg.bonnet)
    pts, _     = lane_pixels(clean, occluded=seg.bonnet)
    gnd, _     = to_ground(pts, seg.cam)                    # 여기부터 자차 좌표 (m)
    bounds, _  = group_boundaries(gnd)
    curves, _  = fit_curves(bounds, rng=rng)
    tracked, _ = tracker.update(curves, dt=dt, context={"s02": stats})
    lanes, _   = assign_lane_ids(tracked, guide_link=link)
    stop, _    = detect_stopline(clean, seg.cam)

좌표계는 **자차 기준 x 전방 / y 좌측 / 미터**다. lane_id 는 왼쪽이 +1, +2,
오른쪽이 -1, -2 이고 **0 은 지금 밟고 있는 선**이다 (차선 변경 중에만 나온다).
"""

import json
import math
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50

# ==========================================================================
# 공통 - 상수 / 카메라 모델 / 자료구조
# ==========================================================================
# 파이프라인 공통 - 상수 / 카메라 모델 / 체크포인트 / 결과 자료구조.
#
# **이 폴더는 `lane_detection.py` 를 대체하는 것이 목적이다.** 그래서 그 파일을
# import 하지 않고 필요한 것을 가져왔다. 대체하려는 대상을 import 하고 있으면
# 영원히 떼어낼 수 없다.
#
# ---------------------------------------------------------------------------
# 가져와도 안전한 이유 - 체크포인트가 자기 규격을 들고 있다
# ---------------------------------------------------------------------------
# 원래 `lane_detection.py` 가 학습 쪽(`seg_model` / `seg_dataset`)을 import 한
# 이유는 "학습에서 바꾼 입력 크기나 정규화가 추론에 조용히 반영되지 않는 것"을
# 막으려는 것이었다. 그 걱정은 타당하지만, `best.pt` 가 이미 다음을 들고 있다.
#
#     input_size     [640, 256]
#     class_names    background / white_solid / white_dashed / yellow / stopline / guide
#     num_classes    6
#     scheme         lane6
#     args.backbone  resnet34
#
# 즉 **규격을 상수로 박지 않고 체크포인트에서 읽으면** 어긋날 수가 없다. 복사본이
# 되는 것은 모델 정의 하나뿐인데, 구조가 바뀌면 `load_state_dict` 가 키 불일치로
# **즉시 터진다** - 조용히 틀리는 것이 아니라 시끄럽게 실패한다. 그래서 복사가
# 안전하다. 남는 위험은 ImageNet 정규화 상수뿐이고 그건 범용 상수다.
#
# ---------------------------------------------------------------------------
# 최종 형태
# ---------------------------------------------------------------------------
# 단계를 다 붙이면 이 파일이 합본의 **머리**가 되고 `s01`~`s12` 본문이 뒤에 붙어
# 파일 하나가 된다. 그래서 여기에는 단계 로직을 두지 않는다 - 상수와 자료구조만.
#
# 카메라 모델과 차체 자세 규약은 `GenerateLabels.py` 에서 **소스를 그대로 추출해**
# 넣었다 (손으로 옮기면 보닛 폴리곤 50쌍 같은 데서 오타가 난다). 학습 라벨이 이
# 규약으로 만들어졌으므로 여기서 바꾸면 추론이 라벨과 어긋난다.

# ==========================================================================
# 클래스 - **기본값일 뿐이다.** 실제로는 체크포인트의 class_names 를 쓴다.
# ==========================================================================
(CLASS_BG, CLASS_WHITE_SOLID, CLASS_WHITE_DASHED,
 CLASS_YELLOW, CLASS_STOPLINE, CLASS_GUIDE) = range(6)

CLASS_NAMES = ["background", "white_solid", "white_dashed",
               "yellow", "stopline", "guide"]

# 차로 경계가 되는 클래스. 유도선(5)은 경계가 아니라 진로 안내선이고,
# 정지선(4)은 진행방향과 직각이라 y=f(x) 로 표현할 수 없어 둘 다 뺀다.
LANE_CLASSES = (CLASS_WHITE_SOLID, CLASS_WHITE_DASHED, CLASS_YELLOW)

# 전경 전부 (2단계에서 같이 정리하고, 사진에서 무엇이 지워졌는지 보려고 남긴다)
FOREGROUND = (CLASS_WHITE_SOLID, CLASS_WHITE_DASHED, CLASS_YELLOW,
              CLASS_STOPLINE, CLASS_GUIDE)

# ImageNet 정규화 - 학습(seg_dataset)과 같아야 한다. 범용 상수라 박아 둔다.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==========================================================================
# 기하 상수
# ==========================================================================
CROP_TOP = 260              # 하늘 제거. 라벨 생성과 같은 값이어야 한다
DEFAULT_SENSOR_ID = 1       # 전방 카메라

# **자차 원점은 노면이 아니라 후륜축 중심이고 노면보다 0.35m 위다.**
# 0 으로 두면 지면 교점 거리가 통째로 틀어진다.
ROAD_Z_EGO = -0.35

# **3.3 이 아니라 3.5 다.** 지도(mgeo K-city)에서 직접 쟀다 - 링크 중심선의
# 수직 단면에서 좌우 최근접 경계까지의 거리, 표본 19894개:
#
#     p10 3.23   p25 3.44   중앙 3.505   p75 3.97
#     0.1m 단위 최빈값: 3.5 (6931개), 3.4 (1592), 3.3 (1275)
#
# `lane_detection.py` 는 3.3 을 쓰는데, 그 값으로 검증하면 캘리브레이션이
# 전 거리에서 +5% 틀린 것처럼 보인다. 실제로는 기준이 틀린 것이었다
# (3.5 기준으로는 8~30m 에서 -2.3~-0.6%).
LANE_WIDTH_M = 3.5

NEAR_PLANE = 0.5      # 카메라 앞 이 거리보다 가까우면 투영하지 않는다

# ==========================================================================
# 보닛 - 학습 라벨에서 255(ignore)라 모델이 여기에 노이즈를 뱉는다.
# 실측(last_test 39장): 클래스 픽셀의 68~92% 가 이 안에 있다.
#   white_solid 88.7%   stopline 92.3%   yellow 81.9%   guide 68.5%
# png 파일에 의존하지 않도록 윤곽을 코드에 박아 둔다 (learning/ 은 3.8GB 라
# 실주행 머신에 없을 수 있다).
# ==========================================================================
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
# 카메라 - GenerateLabels.CameraModel 을 소스째로 가져왔다.
# 학습 라벨이 이 모델로 만들어졌으므로 규약을 바꾸면 추론이 라벨과 어긋난다.
# ==========================================================================
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


# ==========================================================================
# 차체 자세 - 도로 경사/뱅크/서스펜션 때문에 **정지 중에도** 0 이 아니다
# (실측 pitch +0.68도). pitch 오차는 가로가 아니라 거리로 터진다:
# d ~ h/theta 이므로 delta_d ~ -d^2*delta/h, 10m 에서 2도면 2.9m 다.
# 부호 관례는 MORAI 문서에 없어 GenerateLabels 가 오버레이로 확정한 값이다.
# ==========================================================================
USE_EGO_ATTITUDE = True
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


# ==========================================================================
# 결과 - **단계마다의 산출물을 전부 들고 간다.**
#
# 최종 답(lane_id)만 돌려주면 뒤에서 이상한 값이 나왔을 때 어느 단계가 범인인지
# 알 수 없다. 그리고 제어/계획 쪽이 실제로 쓰는 것도 하나가 아니다 - 회피는
# 경계 점열이 필요하고, 추종은 곡선이 필요하다.
# ==========================================================================


@dataclass
class Boundary:
    """한 차선 경계로 묶인 점 그룹. 아직 곡선이 아니다 (6단계 출력)."""
    cls: int
    x: np.ndarray                       # 자차 전방 (m)
    y: np.ndarray                       # 자차 좌측 (m)
    seed_y: float = 0.0                 # 어느 시드에서 자랐는지

    @property
    def x_range(self):
        return (float(self.x.min()), float(self.x.max())) if self.x.size else (0.0, 0.0)


@dataclass
class Curve:
    """RANSAC + 2차식 적합 결과 (9단계 출력)."""
    cls: int
    coef: np.ndarray                    # y = coef[0]x^2 + coef[1]x + coef[2]
    x_range: tuple
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    inlier: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    lane_id: int = 0                    # 12단계에서 채운다. 0 = 미할당
    track_id: int = 0                   # 10~11단계에서 채운다
    age: int = 0                        # 연속으로 몇 프레임 이어졌는지 (10~11단계)
    coasted: bool = False               # 이 프레임에 관측이 없어 예측만으로 낸 것
    # **유도선이 자차 경계 슬롯을 대신 채운 것**이라는 표시 (12단계).
    # lane_id 는 채워지지만 이것은 차로 경계가 아니다 - 넘으면 안 되는 선이
    # 아니라 지나갈 길 힌트다. 회피 계획이 이걸 벽으로 오해하면 안 되고,
    # 차로 폭 계산에도 넣으면 안 된다.
    from_guide: bool = False
    # **이 트랙을 지금 얼마나 믿는가** (0~1, 10~11단계가 채운다).
    # 적합 품질에서 시작해 관측이 빠질 때마다 감쇠한다. miss 횟수 같은
    # 정수 카운터 하나로는 "짧은 점선 gap" 과 "차선이 실제로 없어짐" 을
    # 구분할 수 없다 - 같은 3프레임이라도 직전 관측이 좋았는지 나빴는지에
    # 따라 다르게 취급해야 한다.
    confidence: float = 0.0

    def y_at(self, x):
        return float(np.polyval(self.coef, x))

    @property
    def inlier_ratio(self):
        return float(self.inlier.mean()) if self.inlier.size else 0.0


@dataclass
class StopLine:
    """정지선 하나. **차선과 다른 물건이라 자료구조도 따로 둔다.**

    차선은 진행방향을 따라 누워 `y = f(x)` 인데, 정지선은 진행방향을 **가로질러**
    서 있어서 같은 파라미터화를 쓰면 기울기가 발산한다. 그래서 축을 바꿔
    `x = a*y + b` 로 둔다. 그러면 `b` 가 곧 자차 정면(y=0)까지의 거리다.
    """
    dist: float                         # 자차 정면까지 (m). coef 의 b 와 같다
    coef: np.ndarray                    # x = coef[0]*y + coef[1]
    y_range: tuple                      # 실제로 관측된 가로 구간 (m)
    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    inlier: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    # **관측이 자차 정면을 실제로 덮었는가.** False 면 `dist` 는 옆에서 본
    # 부분을 y=0 까지 외삽한 값이다. 실측으로 210프레임 중 62프레임(30%)만
    # 정면을 덮었으므로, 이 구분을 숨기면 제어가 외삽값을 관측값으로 오해한다.
    covers_front: bool = False
    extrap_m: float = 0.0               # 정면까지 외삽한 거리 (덮었으면 0)
    n_blobs: int = 1                    # x 방향 덩어리 수. 2 이상이면 횡단보도 의심

    @property
    def inlier_ratio(self):
        return float(self.inlier.mean()) if self.inlier.size else 0.0


@dataclass
class LaneResult:
    """한 프레임의 결과. **요청한 다섯 출력이 여기 다 있다.**

        pixels      3. Lane pixel extraction   {cls: (u, v)}  이미지 좌표
        ground      5. Ground / Vehicle coord  {cls: (x, y)}  자차 좌표 (m)
        boundaries  6. Lane boundary           [Boundary]     묶인 점 그룹
        curves      9. Curve fitting           [Curve]        곡선 계수
        lanes      12. Lane ID                 [Curve]        lane_id 가 붙은 것

    `lanes` 는 `curves` 의 부분집합을 가리킨다 (복사가 아니라 같은 객체다).
    lane_id 가 붙지 않은 곡선도 `curves` 에는 남아 있어야 한다 - 왜 ID 를 못
    받았는지 보려면 버려진 것도 보여야 하기 때문이다.
    """
    pixels: dict = field(default_factory=dict)
    ground: dict = field(default_factory=dict)
    boundaries: list = field(default_factory=list)
    curves: list = field(default_factory=list)
    lanes: list = field(default_factory=list)

    # --- 중간 산출물 (사진과 디버깅용) ---
    mask: np.ndarray = None             # 1. 세그멘테이션 원본
    clean: np.ndarray = None            # 2. Morphology 이후
    crop: np.ndarray = None             # crop_top 이후 원본 프레임
    attitude: tuple = None              # (pitch, roll) 도. None 이면 수평 가정
    widths: list = field(default_factory=list)   # 7~8. [(x, width_m), ...]
    stopline: object = None             # StopLine | None. 12단계 번호 밖의 별도 가지
    stats: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)

    def by_lane_id(self, lane_id):
        for c in self.lanes:
            if c.lane_id == lane_id:
                return c
        return None

    @property
    def ego_left(self):
        return self.by_lane_id(1)

    @property
    def ego_right(self):
        return self.by_lane_id(-1)


# ==========================================================================
# 경로 탐색 - 폴더를 옮겨도 따라오게 한다
# ==========================================================================
# 이 파일이 놓인 곳이 곧 post_processing/ 이다 (모듈판은 pipeline/
# 안이라 한 단계 더 들어가 있었다).
_POST = os.path.dirname(os.path.abspath(__file__))
_HERE = _POST
_CAM = os.path.dirname(_POST)                       # camera_perception/

CHECKPOINT_NAMES = ("best.pt", "lane_seg_best.pt")


def _find_upward(relpath, start=None, levels=6):
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


def default_checkpoint():
    for d in (_HERE, _POST, _CAM):
        for name in CHECKPOINT_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return os.path.join(_CAM, CHECKPOINT_NAMES[0])      # 없으면 에러 메시지용


def default_cam_set():
    for d in (_HERE, _POST, _CAM):
        p = os.path.join(d, "cam_set.json")
        if os.path.isfile(p):
            return p
    return _find_upward(os.path.join("data", "sensors", "cam_set.json"))

# ==========================================================================
# 1단계  Segmentation
# ==========================================================================
# 1단계. Segmentation — best.pt 로 클래스 맵을 얻는다.
#
# **`lane_detection.py` 를 import 하지 않는다.** 이 폴더가 그것을 대체하는 것이
# 목적이라, 대체 대상을 붙들고 있으면 영원히 떼어낼 수 없다.
#
# ---------------------------------------------------------------------------
# 규격을 상수로 박지 않고 체크포인트에서 읽는다
# ---------------------------------------------------------------------------
# 학습 쪽(`seg_model` / `seg_dataset`)을 import 하던 이유는 입력 크기·클래스 수가
# 조용히 어긋나는 것을 막기 위해서였다. 그 목적은 **체크포인트를 읽는 것으로 더
# 확실하게** 달성된다.
#
#     input_size     -> 전처리 크기            (박아 두지 않는다)
#     class_names    -> 클래스 이름/개수        (박아 두지 않는다)
#     num_classes    -> 출력 채널 수
#     args.backbone  -> 어떤 ResNet 인지
#
# 복사본이 되는 것은 **모델 정의 하나**뿐이다. 그리고 구조가 바뀌면
# `load_state_dict` 가 키 불일치로 **즉시 터진다** - 조용히 틀리는 게 아니라
# 시끄럽게 실패하므로, 이 복사는 안전하다. (`strict=True` 가 기본값이고,
# 여기서 끄지 않는 것이 핵심이다.)
#
# ---------------------------------------------------------------------------
# 보닛을 여기서 지우지 않는다
# ---------------------------------------------------------------------------
# 돌려주는 마스크는 **모델이 뱉은 그대로**다. 보닛 제거는 2단계의 일이고, 여기가
# 미리 해 버리면 1단계 사진에서 모델이 실제로 무엇을 내는지 볼 수 없다. 그리고
# 볼 것이 많다 - 실측(last_test 39장) 클래스 픽셀의 68~92% 가 보닛 위 노이즈다.

# 모델 정의의 기본 클래스 수. 실제로는 체크포인트 값을 명시적으로 넘긴다.
NUM_CLASSES = len(CLASS_NAMES)

# ==========================================================================
# 모델 정의 - training/seg_model.py 에서 소스째 가져왔다.
#
# 손실(SegLoss)과 혼동행렬은 학습 전용이라 가져오지 않는다. 추론에 필요한 것은
# 그래프뿐이고, 그래프가 어긋나면 load_state_dict 가 즉시 터진다.
#
# ResNet 인코더 + U-Net 디코더다. 차선은 폭 3~7px 로 얇아서, 1/32 특징만 확대
# 하면 선이 뭉개진다. 스킵 연결로 1/4, 1/8 의 고해상 특징을 되살려야 위치가
# 픽셀 단위로 남는다.
# ==========================================================================
_BACKBONES = {"resnet18": resnet18, "resnet34": resnet34,
              "resnet50": resnet50}
# layer1..layer4 의 출력 채널 수
_CHANNELS = {'resnet18': (64, 128, 256, 512), 'resnet34': (64, 128, 256, 512), 'resnet50': (256, 512, 1024, 2048)}


class _Up(nn.Module):
    """2배 확대 후 스킵과 이어붙이고 conv 두 번."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # 입력 크기가 32 의 배수가 아니면 1px 어긋날 수 있다
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class LaneSegNet(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"모르는 백본: {backbone} (가능: {sorted(_BACKBONES)})")
        net = _BACKBONES[backbone](weights="DEFAULT" if pretrained else None)
        c1, c2, c3, c4 = _CHANNELS[backbone]

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)   # 1/2, 64ch
        self.pool = net.maxpool                                    # 1/4
        self.layer1, self.layer2 = net.layer1, net.layer2          # 1/4, 1/8
        self.layer3, self.layer4 = net.layer3, net.layer4          # 1/16, 1/32

        self.up4 = _Up(c4, c3, 256)     # 1/16
        self.up3 = _Up(256, c2, 128)    # 1/8
        self.up2 = _Up(128, c1, 64)     # 1/4
        self.up1 = _Up(64, 64, 32)      # 1/2  (stem 과 이어붙임)
        self.up0 = _Up(32, 0, 16)       # 1/1
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        s = self.stem(x)                # 1/2
        f1 = self.layer1(self.pool(s))  # 1/4
        f2 = self.layer2(f1)            # 1/8
        f3 = self.layer3(f2)            # 1/16
        f4 = self.layer4(f3)            # 1/32
        d = self.up4(f4, f3)
        d = self.up3(d, f2)
        d = self.up2(d, f1)
        d = self.up1(d, s)
        d = self.up0(d)
        return self.head(d)             # [B, C, H, W] 로짓



class Segmenter:
    """best.pt 를 한 번 올려 두고 프레임마다 클래스 맵을 낸다."""

    def __init__(self, checkpoint=None, cam_set=None, device=None,
                 sensor_id=DEFAULT_SENSOR_ID, crop_top=CROP_TOP):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))

        path = checkpoint or default_checkpoint()
        if not os.path.isfile(path):
            raise SystemExit(f"체크포인트를 못 찾았습니다: {path}")
        ck = torch.load(path, map_location=self.device, weights_only=False)

        # --- 규격은 전부 체크포인트에서 ---
        self.input_w, self.input_h = (int(v) for v in ck["input_size"])
        self.class_names = list(ck.get("class_names", CLASS_NAMES))
        n_cls = int(ck.get("num_classes", len(self.class_names)))
        backbone = ck.get("args", {}).get("backbone", "resnet34")

        self.model = LaneSegNet(backbone, pretrained=False, num_classes=n_cls)
        # strict=True (기본값) - 구조가 어긋나면 여기서 터져야 한다
        self.model.load_state_dict(ck["model"])
        self.model.to(self.device).eval()

        self.info = {"epoch": ck.get("epoch"), "backbone": backbone,
                     "num_classes": n_cls, "scheme": ck.get("scheme"),
                     "input_size": (self.input_w, self.input_h),
                     "iou": ck.get("iou"), "path": path}

        # --- 카메라 (4단계 Calibration 이 쓴다) ---
        cs = cam_set or default_cam_set()
        if not cs or not os.path.isfile(cs):
            raise SystemExit("cam_set.json 을 못 찾았습니다")
        self.crop_top = int(crop_top)
        self.cam = load_camera(cs, sensor_id, "horizontal").cropped(self.crop_top)
        self.src_w, self.src_h = self.cam.width, self.cam.height

        # --- 보닛 (2단계 Morphology 가 쓴다) ---
        self.bonnet = self._build_bonnet()

    def _build_bonnet(self):
        m = np.zeros((self.src_h, self.src_w), np.uint8)
        cv2.fillPoly(m, [np.array(BONNET_POLY, np.int32)], 1)
        if BONNET_DILATE_PX:
            k = np.ones((2 * BONNET_DILATE_PX + 1,) * 2, np.uint8)
            m = cv2.dilate(m, k)
        return m > 0

    @torch.no_grad()
    def apply(self, frame_bgr):
        """프레임 -> (mask, crop). mask 는 보닛 제거 **전** 원본 출력이다."""
        img = (frame_bgr[self.crop_top:]
               if frame_bgr.shape[0] > self.src_h else frame_bgr)
        crop = img
        x = cv2.resize(img, (self.input_w, self.input_h),
                       interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)

        small = self.model(x).argmax(1)[0].to(torch.uint8).cpu().numpy()
        # **NEAREST 여야 한다.** 클래스 번호를 보간하면 1 과 3 사이에 없던 2 가
        # 생긴다 (white_solid 와 yellow 사이에 white_dashed 가 끼는 식).
        mask = cv2.resize(small, (self.src_w, self.src_h),
                          interpolation=cv2.INTER_NEAREST)
        return mask, crop

# ==========================================================================
# 2단계  Morphology
# ==========================================================================
# 2단계. Morphology — 마스크에서 **확실한 쓰레기만** 걷어낸다.
#
# ===========================================================================
# 이 단계가 하는 일은 두 개뿐이고, 그게 측정 결과다
# ===========================================================================
#     1) 보닛 마스킹        클래스 픽셀의 68~92% 가 여기 있다
#     2) 아주 작은 성분 제거  8px 미만 - 단일 픽셀 튐만
#
# CLOSE 는 **기본으로 끈다.** 아래 "왜 CLOSE 를 끄는가" 참고.
#
# ---------------------------------------------------------------------------
# 왜 여기서 많이 거르지 않는가
# ---------------------------------------------------------------------------
# 이 단계는 **Calibration 앞**이다. 즉 아직 "이 덩어리가 몇 m 짜리인지" 모른다.
# 그런데 픽셀 면적은 원근에 완전히 오염된 지표다 (last_test 39장 실측, 보닛 제외):
#
#     연결성분 면적 중앙값     0~10m   15~20m   25~30m   30~40m
#     white_solid              3134      222       24       18
#     white_dashed              662       36       15        9
#     guide                     284       15       10       10
#
# 같은 차선이 거리에 따라 3134px -> 18px 이다. 여기서 면적으로 자르면 자르는
# 것이 "노이즈"가 아니라 "먼 곳"이 된다. 실제로 흔히 쓰는 40px 임계는 25m 너머
# white_dashed 성분의 **100%**, 15m 너머 guide 의 **94%** 를 지운다. 면적 기준
# 으로는 4% 밖에 안 버리는 것처럼 보여서 눈치채기 어렵다.
#
# **그래서 노이즈 제거 책임은 뒤로 넘긴다.** Ground 좌표(5단계) 이후로 가면
# 미터로 판단할 수 있고, Width consistency(8단계)는 애초에 그러라고 있는
# 단계다. 여기서는 어느 거리에서도 차선일 수 없는 것 - 단일 픽셀 튐 - 만 뺀다.
#
# ---------------------------------------------------------------------------
# 왜 CLOSE 를 끄는가
# ---------------------------------------------------------------------------
# CLOSE 는 "같은 클래스 안의 구멍"을 메우는 연산이다. 그런데 메울 구멍이 없다.
# last_test 39장, 보닛 제외 도로 영역의 성분 내부 구멍 개수:
#
#     white_solid 8,  white_dashed 1,  yellow 1,  stopline 0,  guide 28
#
# 39장을 통틀어 그렇다. 실제로 CLOSE(3x3) 를 걸면 픽셀이 0.2% 늘고 성분이 25개
# 줄 뿐인데, **그 "성분 25개 감소"가 이득이 아니라 손해다** - 점선 대시가
# 서로 붙는다는 뜻이고, 점선/실선 구분은 대시가 끊겨 있다는 사실에 의존한다.
#
# `lane_detection.py` 의 9x3 세로 커널을 그대로 가져오면 안 된다. 그건 **BEV
# 전용 값**이다. BEV 에서는 차선이 세로로 서 있어 세로 커널이 방향과 맞지만,
# 이미지 공간에서는 차선이 소실점으로 수렴해 방향이 위치마다 다르다. 여기에
# 걸면 픽셀 1.5%, 성분 -74개다.
#
# 그리고 실제 관측된 마스크 오류는 CLOSE 로 고쳐지는 종류가 아니었다
# (000282 정지선, 확대 확인):
#
#     "정지선을 다 못 채움"  -> 왼쪽 끝을 yellow 가 가져감. 구멍이 아니라 클래스 혼동
#     "정지선이 튀어나옴"    -> 6468px 중 23.3% 가 아스팔트 위. 부족이 아니라 과검출
#     "오른쪽 선이 침범"     -> 노면표시가 만나는 지점의 클래스 혼동
#
# 정지선 성분은 364개 열 중 **빈 열이 0개**였다. CLOSE 는 첫째를 못 고치고,
# 둘째를 **더 키우며**(팽창->침식이라 넘친 걸 더 넘치게 한다), 셋째도 못 고친다.
#
# **그래도 인자로는 남겨 둔다.** 다른 녹화에서 실제로 구멍이 나오면 켜서 바로
# 비교할 수 있어야 한다. 끄는 것이 결론이지 금지가 아니다.

# 전경 클래스 전부(_common.FOREGROUND). 정지선도 여기서 같이 정리한다 - 차선
# 적합에는 안 쓰지만(3단계에서 빠진다) 정지선 자체를 쓸 데가 있고, 무엇보다
# 1->2 단계 사진에서 무엇이 지워졌는지 보려면 남아 있어야 한다.

# 어느 거리에서도 차선일 수 없는 크기. 30~40m 의 white_dashed 성분 중앙값이
# 9px 이므로 그보다 확실히 아래로 둔다.
MIN_BLOB_PX = 8

# 기본은 끔 (위 "왜 CLOSE 를 끄는가"). 켜려면 (3, 3) 처럼 튜플을 준다.
CLOSE_KERNEL = None


def morphology(mask, bonnet=None, *, min_blob_px=MIN_BLOB_PX,
          close_kernel=CLOSE_KERNEL, classes=FOREGROUND):
    """클래스 맵을 정리한다. -> (정리된 맵, 통계 dict)

    통계는 클래스마다 {before, bonnet, blob, close, after} 픽셀 수다.
    단계별로 무엇이 얼마나 빠졌는지 사진 아래에 그대로 찍으려고 낸다.

    **클래스마다 따로 돈다.** 황색 중앙선과 백색 실선은 붙어 있어도 다른
    차선이라, 한 통에 넣고 연산하면 둘이 이어진다.
    """
    out = mask.copy()
    stats = {}

    for c in classes:
        stats[c] = {"before": int((out == c).sum()), "bonnet": 0,
                    "blob": 0, "close": 0, "after": 0}

    # --- 1) 보닛 마스킹 ---------------------------------------------------
    # 형태학이 아니라 마스킹이다. 다만 "모델 출력에서 확실한 쓰레기를 뺀다"는
    # 이 단계의 일이라 여기 둔다.
    if bonnet is not None:
        for c in classes:
            stats[c]["bonnet"] = int(((out == c) & bonnet).sum())
        out[bonnet] = CLASS_BG

    ker = np.ones(close_kernel, np.uint8) if close_kernel else None

    for c in classes:
        m = (out == c).astype(np.uint8)
        if not m.any():
            continue

        # --- 2) CLOSE (기본 꺼짐) -----------------------------------------
        if ker is not None:
            closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
            # **배경이던 자리만 채운다.** 다른 클래스가 이미 차지한 픽셀을
            # 빼앗으면 클래스끼리 처리 순서에 따라 결과가 달라진다.
            grow = (closed > 0) & (m == 0) & (out == CLASS_BG)
            stats[c]["close"] = int(grow.sum())
            m = ((m > 0) | grow).astype(np.uint8)
            out[grow] = c

        # --- 3) 아주 작은 성분 제거 ---------------------------------------
        if min_blob_px and min_blob_px > 1:
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n > 1:
                small = np.zeros(n, bool)
                small[1:] = st[1:, cv2.CC_STAT_AREA] < min_blob_px
                drop = small[lab]
                stats[c]["blob"] = int(drop.sum())
                out[drop] = CLASS_BG

        stats[c]["after"] = int((out == c).sum())

    return out, stats


def format_morphology_stats(stats, names=None):
    """통계를 사람이 읽는 줄들로. 사진 아래와 터미널에 같은 것을 쓴다."""
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'before':>9s} {'-bonnet':>9s} {'-blob':>7s} "
             f"{'+close':>7s} {'after':>8s} {'kept':>6s}"]
    for c, s in stats.items():
        keep = s["after"] / s["before"] * 100 if s["before"] else 0.0
        lines.append(f"{names[c]:12s} {s['before']:>9d} {s['bonnet']:>9d} "
                     f"{s['blob']:>7d} {s['close']:>7d} {s['after']:>8d} "
                     f"{keep:>5.1f}%")
    return lines

# ==========================================================================
# 3단계  Lane pixel extraction
# ==========================================================================
# 3단계. Lane pixel extraction — 마스크 -> 중심선 점열 (이미지 좌표).
#
# ===========================================================================
# 이 단계가 하는 일은 하나다: **중심을 정확히 찾는다.**
# ===========================================================================
# 거르지 않고, 줄이지 않고, 합치지 않는다. 그 셋은 전부 뒤 단계의 일이다.
#
# 행마다 가로로 이어진 픽셀 덩어리(런)를 찾아 그 **중점** 하나를 남긴다.
# 점 하나는 `(u, v, w)` - 가로 위치(서브픽셀), 행, 그리고 그 런의 폭이다.
#
# ---------------------------------------------------------------------------
# 왜 행별 런 중점이 정당한가 - 선이 기울어져 있어도 맞는다
# ---------------------------------------------------------------------------
# 처음에는 "차선이 이미지에서 수평에 가까우면 행이 선을 가로지르는 게 아니라
# 선을 따라 잘라서 중점이 엉뚱해진다"고 걱정했다. 실측해 보니 실제로 차선
# 픽셀의 대부분이 수직이 아니었다 (성분 방향 |vy|, 면적 가중):
#
#     클래스        수직 >0.7   기울 0.3~0.7   수평 <0.3
#     white_solid      11.6%        54.0%        34.4%
#     yellow            5.6%        68.7%        25.7%
#     guide            36.5%        24.7%        38.8%
#
# **그런데 걱정이 틀렸다.** 일정한 두께의 곧은 띠를 수평선으로 자르면, 그
# 현(chord)의 중점은 **각도와 무관하게 띠의 중심선 위에 있다.** 기울수록 현이
# 길어질 뿐 중점은 제자리다. 그래서 방향 적응(주축 PCA)이나 세선화로 갈 이유가
# 없다. (덧붙여 `cv2.ximgproc` 가 이 환경에 없어 thinning 은 쓸 수도 없다.)
#
# 중점이 어긋나는 것은 띠가 그 행 안에서 휠 때와 **서로 다른 두 선이 한 런으로
# 붙을 때**다. 후자는 여기서 판별할 수 없다 - 미터를 모르기 때문이다. 그래서
# 폭 `w` 를 들려 보내고 판단은 뒤로 넘긴다.
#
# ---------------------------------------------------------------------------
# 왜 두꺼운 런을 버리지 않는가
# ---------------------------------------------------------------------------
# "런이 90px 넘으면 차선이 아니라 화살표/노면표시" 라는 가드를 흔히 두는데,
# 실측하면 그 가드가 자르는 것이 **원거리**다. 39장에서 90px 를 넘은 런 101개의
# 거리 분포:
#
#     10m 이내 23%,  10~20m 29%,  **20m 너머 49%**
#     (yellow 는 중앙값 26.9m, white_solid 는 16.6m)
#
# 먼 쪽 차선은 이미지에서 수평에 가까워지므로 런이 길어진다. 즉 픽셀 폭으로
# 자르면 "노면표시"가 아니라 "먼 곳"을 자르게 된다. 2단계에서 `MIN_BLOB` 을
# 키우지 않은 것과 **같은 이유이고 같은 함정**이다.
#
# 화살표나 합쳐진 두 선은 ground 좌표로 가면 명백하다 - 차로 폭 3.3m 와 맞지
# 않는다. 그래서 Width consistency 단계에서 미터로 거른다.
#
# ---------------------------------------------------------------------------
# 거리 가중은 여기서 해결되지 않는다
# ---------------------------------------------------------------------------
# 행별 런 중점으로 바꾸면 근거리 편중이 줄기는 한다. 다만 기대만큼은 아니다
# (white_solid, 3~10m 가 차지하는 비중):
#
#     픽셀 그대로 55.3%  ->  행별 런 중점 46.9%
#
# 이미지의 행 간격은 원근 때문에 거리에 비례하지 않으므로 당연하다. **자차
# 좌표로 간 뒤 x 축을 일정 간격으로 잘라 재샘플링**해야 고르게 된다. 그건
# Lane boundary 추출 단계에서 한다.
#
# ---------------------------------------------------------------------------
# 정지선은 뽑지 않는다
# ---------------------------------------------------------------------------
# 정지선은 진행방향과 직각이라 `y = f(x)` 로 표현할 수 없다. 같은 통에 넣으면
# 차선 적합이 망가진다. 유도선(guide)은 차로 경계가 아니지만 **모양은 차선과
# 같아서** 같은 추출을 태우고, 차선과 섞지 않는 것은 뒤 단계에서 한다.

# 차선 모양을 가진 것 전부. 정지선(4)은 위 주석대로 뺀다.
EXTRACT_CLASSES = tuple(LANE_CLASSES) + (CLASS_GUIDE,)

# None = 가드 없음. 숫자를 주면 그보다 두꺼운 런을 버린다 (비교해 보고 싶을 때만.
# 기본을 None 으로 두는 근거는 위 "왜 두꺼운 런을 버리지 않는가").
MAX_RUN_PX = None


def _runs(binary):
    """행마다의 런을 한 번에 찾는다. -> (row, start, end) 배열 셋.

    행마다 파이썬 루프를 도는 대신 양 끝을 0 으로 패딩하고 가로 차분을 본다.
    +1 이 런의 시작, -1 이 끝(배타적)이다. 행별로 패딩했으므로 시작과 끝이
    같은 순서로 짝지어진다.

    런의 중점은 `(start + end - 1) / 2` 다 - 연속한 정수 인덱스의 평균이라
    행별 루프로 `run.mean()` 을 구한 것과 **정확히 같은 값**이다.
    """
    h, w = binary.shape
    pad = np.zeros((h, 1), np.int8)
    d = np.diff(np.hstack([pad, binary.astype(np.int8), pad]), axis=1)
    sr, sc = np.nonzero(d == 1)
    er, ec = np.nonzero(d == -1)
    return sr, sc, ec                       # sr == er (같은 행, 같은 순서)


def lane_pixels(mask, occluded=None, classes=EXTRACT_CLASSES, max_run_px=MAX_RUN_PX,
          drop_clipped=True):
    """클래스 맵 -> ({cls: (u, v, w)}, 통계)

        u  런의 중점 (float, 서브픽셀)
        v  행 (float)
        w  런의 폭 (int, px) - 판단 재료로 뒤에 넘긴다

    `occluded` 에 보닛 마스크를 주면 **거기에 잘린 런을 버린다** (아래 참고).
    """
    out, stats = {}, {}
    w_img = mask.shape[1]
    for c in classes:
        b = mask == c
        n_px = int(b.sum())
        stats[c] = {"px": n_px, "pts": 0, "clipped": 0, "dropped": 0,
                    "w_p50": 0.0, "w_p90": 0.0, "w_max": 0}
        if n_px == 0:
            continue

        rows, start, end = _runs(b)
        u = (start + end - 1) / 2.0
        v = rows.astype(np.float64)
        w = (end - start).astype(np.int32)

        # --- 잘린 런 버리기 -------------------------------------------------
        # 런의 중점이 중심선 위에 있다는 보장은 **런의 양 끝이 도색의 끝일
        # 때**만 성립한다. 보닛이나 화면 가장자리가 도색을 잘라 버리면 남은
        # 조각의 중점은 잘린 반대쪽으로 밀린다.
        #
        # 실측(39장): 보닛에 잘린 런이 점의 2.0~3.6%, 중점이 밀린 양은
        # 중앙값 7px, 최대 20px. 양이 적어 보이지만 **전부 근거리**라 적합에서
        # 가중이 가장 크고, 밀리는 방향이 늘 보닛 바깥쪽이라 무작위가 아니라
        # **계통 오차**다. 그래서 곡선이 보닛 윤곽을 따라 꺾인다.
        #
        # 뒤 단계에서는 못 고친다 - 점이 이미 틀린 자리에 있고 RANSAC 에게는
        # 그것들이 일관된 인라이어로 보인다.
        if drop_clipped:
            bad = (start == 0) | (end == w_img)         # 화면 좌우 끝
            if occluded is not None:
                left = np.clip(start - 1, 0, w_img - 1)
                right = np.clip(end, 0, w_img - 1)
                bad |= occluded[rows, left] | occluded[rows, right]
            stats[c]["clipped"] = int(bad.sum())
            u, v, w = u[~bad], v[~bad], w[~bad]

        if max_run_px:
            keep = w <= max_run_px
            stats[c]["dropped"] = int(w[~keep].sum())
            u, v, w = u[keep], v[keep], w[keep]

        if u.size:
            stats[c].update(pts=int(u.size),
                            w_p50=float(np.percentile(w, 50)),
                            w_p90=float(np.percentile(w, 90)),
                            w_max=int(w.max()))
            out[c] = (u, v, w)
    return out, stats


def format_lane_pixel_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'pixels':>9s} {'points':>8s} {'px/pt':>6s} "
             f"{'clipped':>8s} {'run w p50':>9s} {'p90':>5s} {'max':>5s}"]
    for c, s in stats.items():
        ratio = s["px"] / s["pts"] if s["pts"] else 0.0
        lines.append(f"{names[c]:12s} {s['px']:>9d} {s['pts']:>8d} "
                     f"{ratio:>6.1f} {s['clipped']:>8d} {s['w_p50']:>9.0f} "
                     f"{s['w_p90']:>5.0f} {s['w_max']:>5d}")
    return lines

# ==========================================================================
# 4~5단계  Calibration -> Ground / Vehicle coordinates
# ==========================================================================
# 4단계. Calibration -> Ground / Vehicle coordinates.
#
# 이미지 점 `(u, v)` 를 광선으로 쏘아 **지면과 만나는 자차 좌표 `(x, y)`** 를 얻는다.
# BEV 래스터를 거치지 않는다 - 워프 보간에서 얇은 차선이 깨지지도, 0.05m/px 격자에
# 갇히지도, 쓰지 않는 배경까지 32만 픽셀을 채우지도 않는다.
#
# ===========================================================================
# 이 단계에 들어가는 세 가지
# ===========================================================================
#     1) 카메라 내부/장착   cam_set.json 고정값. 여기서 튜닝하지 않는다.
#     2) 차체 자세          프레임마다 meta.jsonl 의 pitch/roll
#     3) 지면 모델          오늘은 평면. **교체 가능하게 객체로 둔다.**
#
# ---------------------------------------------------------------------------
# 왜 차체 자세가 지면 경사보다 중요한가 (실측)
# ---------------------------------------------------------------------------
# 지면 교점 거리는 d ~ h/theta 라 각도 오차가 거리로 증폭된다: delta_d/d = d*delta/h.
# 카메라는 노면 위 h=1.55m 다.
#
#   **자세 1도 오차** -> 40m 에서 거리 45% 오차
#   **노면 경사**     -> learning 12랩 실측, 전방 40m 고도차 p90 0.10m = 거리 6.5%
#
#     전방      |dz| p50   p90    최대
#      10m        0.01   0.04    0.19 m
#      40m        0.03   0.10    0.26 m
#
# 즉 K-city 에서는 **자세가 노면 경사보다 7배 큰 오차원**이다. 그래서 오늘은
# 평면으로 두고 자세를 정확히 넣는다. 다른 코스라면 이 결론이 바뀔 수 있다.
#
# `ROAD_Z_EGO = -0.35` 를 0 으로 두면 안 된다. 자차 원점이 노면이 아니라 후륜축
# 중심이고, 실측으로 노면보다 일정하게 0.35m 위다.
#
# ---------------------------------------------------------------------------
# 지면을 객체로 두는 이유
# ---------------------------------------------------------------------------
# 계획은 나중에 LiDAR 로 노면을 직접 재는 것이다. 그때 이 파일의 나머지가 바뀌면
# 안 된다. 그래서 지면은 `GroundPlane` 하나로 격리해 둔다.
#
#     오늘      차체 자세로 기울인 평면            GroundPlane.from_attitude()
#     다음      차로 폭 자기보정으로 잔여 pitch 보정  (같은 클래스, normal 만 갱신)
#     나중      LiDAR 로 맞춘 평면/곡면            (같은 인터페이스)
#
# **차로 폭 자기보정**이 LiDAR 없이 할 수 있는 다음 수순이다 - 차로가 3.3m 라는
# 것을 알면, 측정된 폭이 거리에 따라 변하는 정도가 곧 잔여 pitch 다. 지도도
# 라이다도 필요 없고 이미지만으로 관측된다.
#
# ---------------------------------------------------------------------------
# 왜 픽셀이 아니라 거리로 자르는가
# ---------------------------------------------------------------------------
# 지평선 근처는 1px 이 수십 m 다 (장착 pitch 2도 기준):
#
#     거리   40m   60m   100m   무한
#     행 v   104    95     88     78        <- 40m 부터 지평선까지 26px 뿐
#
# 그래서 "v 이상만 쓴다" 같은 픽셀 컷을 걸고 싶어지는데, **자세 보정을 켜면
# 지평선 행이 프레임마다 움직인다.** 고정 픽셀 컷은 프레임마다 다른 거리를 뜻하게
# 된다. 여기서는 이미 거리를 계산했으므로 `x <= X_MAX` 로 거는 것이 같은 일을
# 정확하게 하는 방법이다.

# 자차 좌표에서 쓸 범위. **거리 기준이다** (픽셀 아님).
X_MIN = 3.0             # 보닛에 가려 이보다 가까운 노면은 안 보인다
X_MAX = 40.0            # 학습 라벨의 MAX_RANGE 와 같다. 그 너머는 배운 적이 없다
GROUND_Y_ABS_MAX = 10.0        # |y| 가 이보다 크면 도로 밖이다


class GroundPlane:
    """평면 `n . p = d0` (자차 좌표계). 광선과의 교점을 닫힌 식으로 푼다.

    법선 `n` 은 **월드 수직을 자차 좌표계에서 본 것**이다. 차가 기울면 이
    벡터가 기울고, 그게 곧 "지면이 차 기준으로 기울어 보인다"는 뜻이다.
    """

    def __init__(self, normal=(0.0, 0.0, 1.0), offset=ROAD_Z_EGO):
        self.normal = np.asarray(normal, dtype=np.float64)
        self.offset = float(offset)

    @classmethod
    def from_attitude(cls, pitch_deg=None, roll_deg=None, z0=ROAD_Z_EGO):
        """차체 자세로 기울인 지면. 자세를 안 주면 수평이라고 본다.

        조건은 "자차 좌표 점 p 의 **월드 높이**가 원점보다 z0 만큼 아래" 다.
        (R @ p)_z = z0 이고 R 의 셋째 행이 월드 수직 성분이므로 n = R[2] 다.
        **yaw 는 영향이 없다** - 월드 수직을 축으로 도는 회전이라 당연하다.
        """
        if pitch_deg is None or roll_deg is None or not USE_EGO_ATTITUDE:
            return cls((0.0, 0.0, 1.0), z0)
        R = rot_vehicle_to_world(0.0, EGO_PITCH_SIGN * float(pitch_deg),
                                 EGO_ROLL_SIGN * float(roll_deg))
        return cls(R[2], z0)

    def intersect(self, origin, dirs):
        """광선 origin + t*dirs 와 평면의 교점. -> (t, valid)

        `valid` 가 False 인 것은 **지평선 위이거나 뒤로 가는 광선**이다.
        버리지 않고 표시로 돌려주는 이유는, 지평선 위 픽셀이 얼마나 섞였는지가
        그 자체로 검출 품질의 신호이기 때문이다.
        """
        n = self.normal
        nd = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.offset - float(origin @ n)) / nd
        valid = (nd < -1e-9) & np.isfinite(t) & (t > 0)
        return np.where(valid, t, 0.0), valid


def pixel_rays(cam, uv):
    """픽셀 -> 자차 좌표계 시선 방향 (정규화하지 않음).

    `CameraModel.to_camera` 를 그대로 뒤집은 것이다. 거기서는 자차점 p 를
    q = p @ Rm 로 몸체축에 넣고 광학축을 (-q_y, -q_z, q_x) 로 만든다. 따라서

        몸체축 = (d_z, -d_x, -d_y)      (광학축 d 에서)
        자차축 = 몸체축 @ Rm.T
    """
    import math
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    d_opt = np.stack([(uv[:, 0] - cam.cx) / cam.fx,
                      (uv[:, 1] - cam.cy) / cam.fy,
                      np.ones(len(uv))], axis=1)
    d_body = np.stack([d_opt[:, 2], -d_opt[:, 0], -d_opt[:, 1]], axis=1)

    cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    cr, sr = math.cos(cam.roll), math.sin(cam.roll)
    Rm = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    return d_body @ Rm.T


def unproject(cam, uv, ground):
    """픽셀 -> 지면 위 자차 좌표. -> (xy, valid)"""
    d = pixel_rays(cam, uv)
    t, valid = ground.intersect(cam.mount_pos, d)
    return cam.mount_pos[:2] + t[:, None] * d[:, :2], valid


def to_ground(pts, cam, attitude=None, ground=None,
          x_min=X_MIN, x_max=X_MAX, y_abs=GROUND_Y_ABS_MAX):
    """{cls: (u, v, w)} -> ({cls: (x, y, w_m)}, 통계)

        x     자차 전방 (m)
        y     자차 좌측 (m)
        w_m   그 런을 지면에 내렸을 때의 길이 (m)

    `w_m` 은 도색 폭이 아니라 **행이 자른 현(chord)의 길이**다. 선이 기울수록
    길어진다. 그래도 들고 가는 이유는 두 선이 한 런으로 붙었는지를 미터로
    판단할 재료이기 때문이다 (차로 폭 3.3m 와 비교할 수 있다).
    """
    ground = ground or GroundPlane.from_attitude(*(attitude or (None, None)))
    out, stats = {}, {}

    for c, (u, v, w) in pts.items():
        n_in = int(u.size)
        stats[c] = {"in": n_in, "horizon": 0, "near": 0, "far": 0,
                    "side": 0, "kept": 0}
        if n_in == 0:
            continue

        # 런의 양 끝도 같이 내려 보낸다 - 현의 길이를 미터로 얻으려고
        half = (w - 1) / 2.0
        uv = np.concatenate([np.stack([u, v], 1),
                             np.stack([u - half, v], 1),
                             np.stack([u + half, v], 1)])
        xy, ok = unproject(cam, uv, ground)
        xy_c, xy_l, xy_r = np.split(xy, 3)
        ok = np.split(ok, 3)[0]

        x, y = xy_c[:, 0], xy_c[:, 1]
        stats[c]["horizon"] = int((~ok).sum())
        near = ok & (x < x_min)
        far = ok & (x > x_max)
        side = ok & ~near & ~far & (np.abs(y) > y_abs)
        keep = ok & ~near & ~far & ~side
        stats[c].update(near=int(near.sum()), far=int(far.sum()),
                        side=int(side.sum()), kept=int(keep.sum()))

        if keep.any():
            w_m = np.linalg.norm(xy_r[keep] - xy_l[keep], axis=1)
            out[c] = (x[keep], y[keep], w_m)
    return out, stats


def load_attitude(meta_path):
    """meta.jsonl -> {idx: (pitch_deg, roll_deg)}

    **빌려오지 않는다.** 자세는 그 녹화 자신의 것이어야 한다 - 다른 주행의
    pitch 를 먹이면 없는 기울기를 보정하는 셈이 된다.
    """
    att = {}
    if not meta_path or not os.path.isfile(meta_path):
        return att
    with open(meta_path, encoding="utf-8") as fp:
        for line in fp:
            r = json.loads(line)
            att[int(r["idx"])] = (float(r.get("pitch", 0.0)),
                                  float(r.get("roll", 0.0)))
    return att


def format_ground_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'in':>7s} {'horizon':>8s} {'<Xmin':>6s} "
             f"{'>Xmax':>6s} {'|y|':>5s} {'kept':>7s} {'rate':>6s}"]
    for c, s in stats.items():
        r = s["kept"] / s["in"] * 100 if s["in"] else 0.0
        lines.append(f"{names[c]:12s} {s['in']:>7d} {s['horizon']:>8d} "
                     f"{s['near']:>6d} {s['far']:>6d} {s['side']:>5d} "
                     f"{s['kept']:>7d} {r:>5.1f}%")
    return lines

# ==========================================================================
# 6단계  Lane boundary
# ==========================================================================
# 6단계. Lane boundary — 지면점을 **차선 경계별로** 묶는다.
#
#     입력   {cls: (x, y, w_m)}    4~5단계 출력. 자차 좌표 (m)
#     출력   [Boundary]            한 경계에 속한 점들. 아직 곡선이 아니다
#
# ===========================================================================
# 왜 BEV 래스터 없이도 같은 일을 할 수 있는가
# ===========================================================================
# 옛 구현은 조감도 이미지를 만들어 **하단 히스토그램 + 슬라이딩 윈도우**로 묶었다.
# 그런데 그 조감도의 행은 처음부터 x 를, 열은 y 를 0.05m/px 로 잘라 놓은 것일
# 뿐이다. 즉 히스토그램도 윈도우도 **원래 미터 단위 연산**이었고, 래스터는 그것을
# 정수 격자에 억지로 끼운 중간 단계였다.
#
# 점을 바로 쓰면 격자가 사라진다. 얻는 것:
#
#     - 0.05m 격자에 갇히지 않는다. s03 이 서브픽셀로 구한 중점이 그대로 산다
#     - 워프 보간이 얇은 차선을 끊어 먹던 문제가 없다
#     - 배경까지 800x400 을 채우지 않는다 (프레임당 점 300~500개가 전부다)
#
# ---------------------------------------------------------------------------
# 묶는 방법: 가까운 곳에서 씨를 찾고 멀리까지 따라간다
# ---------------------------------------------------------------------------
#     1) 씨앗    가장 가까운 SEED_X_SPAN_M 구간에서 y 히스토그램 봉우리
#     2) 행진    씨앗마다 x 를 STEP_M 씩 전진하며 창 안의 점을 거둔다
#                (공백을 건넌 뒤에는 **방향이 이어지는지**까지 확인한다)
#     3) 반복    아직 아무 경계에도 안 들어간 점으로 1~2 를 다시 한다
#     4) 정리    같은 점을 많이 공유하는 경계는 하나로 본다
#
# ---------------------------------------------------------------------------
# 3) 반복이 없으면 약하거나 먼 선이 통째로 사라진다
# ---------------------------------------------------------------------------
# 씨앗을 한 번만 찾으면 두 가지로 진다. 300프레임 주행 실측에서 자차 우측 차선이
# **57프레임(19%)** 동안 통째로 없었고, 원인이 이 둘이었다.
#
#     f095  오른쪽 선의 점이 전부 11m 밖   -> 근거리 씨앗 구간에 하나도 없다
#           (구간 안 85점이 전부 y>=0, 구간 밖에 y<0 점이 69개)
#
#     f105  구간 안에 오른쪽 점이 30개 있는데도 씨가 안 생긴다
#           SEED_MIN_RATIO 가 **가장 강한 봉우리 대비** 비율이라, 왼쪽
#           84점짜리 봉우리에 눌려 탈락한다
#
# 둘 다 "한 번의 히스토그램이 전체를 대표한다" 는 가정이 깨진 경우다. 한 번
# 훑고 나서 **남은 점만으로 다시 훑으면** 두 경우가 같이 풀린다 - 강한 선이
# 빠진 뒤에는 약한 선이 그 패스의 최대 봉우리가 되고, 씨앗 구간도 남은 점
# 기준으로 다시 잡히므로 먼 선도 자기 구간을 갖는다.
#
# 비용은 거의 없다. 패스마다 점이 줄어들고 SEED_PASSES 로 상한을 둔다.
#
# **클래스마다 따로 돈다.** 황색 중앙선과 백색 실선은 붙어 있어도 다른 경계다.
# 섞으면 멀리서 두 선이 만나는 지점에서 하나로 합쳐진다 (s02 가 클래스별로 도는
# 이유와 같다).
#
# ---------------------------------------------------------------------------
# 창을 왜 중앙값으로 옮기는가
# ---------------------------------------------------------------------------
# 창 안 점들의 **중앙값**으로 창 중심을 옮긴다. 평균이 아니다. 옆 차선 점이나
# 노이즈가 창 가장자리에 몇 개 들어오면 평균은 그쪽으로 끌려가고, 한 번 끌려간
# 창은 다음 창을 더 끌고 가서 **경계가 옆 차선으로 넘어간다**. 중앙값은 그
# 소수점에 흔들리지 않는다.
#
# ---------------------------------------------------------------------------
# 거리에 따라 점이 성기다는 것을 창 크기에 반영한다
# ---------------------------------------------------------------------------
# 원근 때문에 같은 1m 라도 근거리는 점이 촘촘하고 원거리는 성기다. 40프레임
# 실측(백색실선, 1m 당 점):
#
#     x  3~10m   17.4        x 20~30m    1.1
#     x 10~20m   11.7        x 30~40m    1.0
#
# 20m 에서 **10배**가 꺾인다. 그래서 창을 채우는 최소 점수를 크게 잡으면 먼 쪽이
# 통째로 끊기고, 작게 잡으면 가까운 쪽에서 노이즈가 창을 끌고 간다.
#
# 그래서 창의 성패를 **세 가지로 나눈다.**
#
#     점 >= MIN_PTS   중심을 중앙값으로 옮긴다
#     점 1개 이상     거두기는 하되 중심은 직전 기울기로만 민다
#     점 0개          비었다. MISS_MAX_M 넘게 이어지면 거기서 끝
#
# 가운데 칸이 핵심이다. 이것이 없으면 1점/m 인 원거리에서 창마다 "실패"가 쌓여
# **점을 찾고 있는데도** 6m 만에 끊긴다. 반대로 점 1개로 중심을 옮기게 두면 원거리
# 노이즈 하나가 경계를 통째로 끌고 간다. 거두되 끌려가지는 않는다.
#
# 점선의 대시 간격(약 3m)은 마지막 칸이 처리한다 - 대시 사이는 점이 0개다.

# --- 1) 씨앗 --------------------------------------------------------------
# 가장 가까운 구간에서만 씨를 찾는다. 멀리서 찾으면 두 차선이 소실점 쪽에서
# 붙어 보여 봉우리가 하나로 뭉친다.
#
# **구간의 시작을 x 최솟값에 묶으면 안 된다.** 외톨이 점 하나가 구간을
# 통째로 끌고 간다. 실측: 백색점선 프레임에서 x=6.0m 에 점 1개, 6~10m 는
# 0개, 정작 23개는 10~15m 에 몰려 있었다. 최솟값 기준 4m 구간(6~10m)은
# 그 외톨이만 잡아서 씨앗이 엉뚱한 y 에 섰고 경계가 40프레임 전부 실패했다.
# 그래서 하위 백분위에서 시작하고, 점이 모자라면 구간을 넓힌다.
SEED_X_LO_PCT = 10          # 구간 시작. x 하위 이 백분위 (최솟값이 아니다)
SEED_X_SPAN_M = 4.0         # 기본 구간 길이
SEED_X_SPAN_MAX_M = 12.0    # 점이 모자라면 여기까지 넓힌다
SEED_MIN_BAND_PTS = 12      # 구간이 이만큼은 담아야 히스토그램이 의미 있다
SEED_BIN_M = 0.10           # 히스토그램 칸. 도색 폭(0.15~0.35m)보다 작게
SEED_SMOOTH_BINS = 3        # 칸 3개(0.3m) 이동평균. 한 칸짜리 뾰족함은 노이즈다
SEED_MIN_GAP_M = 1.5        # 봉우리 사이 최소 간격. 차로 폭 3.5m 의 절반 아래
SEED_MIN_RATIO = 0.25       # 최대 봉우리 대비 이 비율 미만이면 씨로 안 본다
SEED_MAX_COUNT = 6          # 클래스 하나에서 씨 최대 개수

# --- 2) 행진 --------------------------------------------------------------
STEP_M = 1.0                # 창 깊이 (전방)
MARGIN_M = 0.8              # 창 반폭. 급커브에서 1m 전진에 y 가 0.6m 움직인다
BOUND_MIN_PTS = 2                 # 창 중심을 갱신할 최소 점수

# **빈 구간을 건너뛰는 것은 점선을 잇자고 넣은 장치다. 실선에는 쓰면 안 된다.**
# 실선은 원래 안 끊긴다. 끊겼다면 그것은 다음 대시가 아니라 **다른 선**이거나
# 가림이다. 그런데 클래스 구분 없이 6m 를 건너뛰게 두면, 교차로에서 교차 도로의
# 도색이나 횡단보도가 그대로 한 경계로 붙는다.
#
# 실측(경계 내부에서 2m 넘게 건너뛴 자리 396건): 이음매 기울기 차가 p90 0.717,
# 최대 3.73 이고 **4건 중 1건이 0.3 이상 꺾였다.** 같은 선이라면 이어져야 하는
# 값이다.
#
# 공백 **길이**로는 못 거른다 - 대시 간격대(2~5m)와 그보다 긴 공백의 기울기차
# 분포가 거의 같다(p50 0.083 vs 0.060). 그래서 **클래스로** 나눈다.
# 기본값(실선·황색선)은 **대시 간격 3m 보다 작게** 둔다. 그래야 이 클래스에서는
# 다음 대시를 집는 일이 구조적으로 불가능하다.
#
# 가림(차량 등)으로 생긴 구멍은 **일부러 안 잇는다.** 가려진 구간의 차선을
# 이어 붙이는 것은 보지 못한 것을 지어내는 일이고, 그렇게 만든 값을 제어가
# 실제 관측과 같은 신뢰도로 받으면 안 된다. 가림 복원은 낮은 우선순위다.
MISS_MAX_M = 2.3            # 기본(실선·황색선)
MISS_MAX_BY_CLASS = {
    CLASS_WHITE_DASHED: 6.0,    # 대시 3m + 빈 구간 3m
    CLASS_GUIDE: 4.0,           # 유도선도 점선이지만 촘촘하고 짧다
}


def miss_max_for(cls):
    """그 클래스에서 건너뛸 수 있는 빈 구간 길이 (m)."""
    return MISS_MAX_BY_CLASS.get(cls, MISS_MAX_M)

# **빈 구간에서는 기울기를 감쇠시키지 않는다.** 곡선은 빈 구간에서도 계속
# 휘므로, 직전 기울기를 그대로 잇는 것이 가장 좋은 예측이다. 감쇠시키면
# 예측이 안쪽으로 처져서 다음 대시를 놓친다.
#
# 실측(f000, 백색점선): 대시가 11~13m 와 18~22m 에 있고 그 사이 5m 가 비었다.
# 기울기는 -0.64 m/m 인데 0.6 배로 감쇠시키면 5m 뒤 예측이 **1.7m 모자라서**
# 다음 대시가 창(+-0.8m) 밖으로 벗어난다. 그래서 대시 하나(2.3m)만 남고
# BOUND_MIN_SPAN_M 에 걸려 버려졌다 - 40프레임 전부 점선 검출 0 이었던 원인이다.
DRIFT_GAIN = 1.0            # 빈 구간에서 직전 기울기를 잇는 비율
DRIFT_MAX_M = 0.8           # 한 창에서 허용할 최대 이동. 옆 차선으로 튀는 것을 막는다

# 오래 비었을수록 예측이 부정확하다. 그만큼 창을 넓혀서 받아 준다. 넓히지
# 않으면 기울기 예측 오차가 조금만 쌓여도 다음 대시를 통째로 놓친다.
# (같은 생각을 10~11단계 추적이 공분산으로 한다 - 여기서는 그 축소판이다.)
MARGIN_GROW = 0.25          # 빈 구간 1m 당 창 반폭을 이만큼 넓힌다
MARGIN_MAX_M = 2.0          # 넓혀도 여기까지. 차로 폭(3.5m)의 절반을 넘기면
                            # 옆 차선을 빨아들인다

# **빈 구간을 건넌 뒤에는 방향까지 맞아야 같은 선으로 본다.**
# 위 장치(기울기 유지 + 창 확장)는 점선을 잇자고 넣은 것인데, 위치만 보고
# 방향은 보지 않아서 **근처에 있는 다른 선도 같이 이어 버린다.** 교차로에서
# 교차 도로의 도색이나 횡단보도가 그렇게 붙는다.
#
# 실측(경계 내부에서 2m 넘게 건너뛴 자리 396건): 이음매에서의 기울기 차가
# p50 0.071 인데 p90 이 0.717, 최대 3.73 이다. **4건 중 1건(25.3%)이 0.3 이상
# 꺾인다.** 같은 선의 다음 대시라면 기울기가 이어져야 하므로, 꺾이는 것은
# 다른 선을 붙인 것이다.
#
# 공백 **길이**로는 못 거른다 - 대시 간격대(2~5m)와 그보다 긴 공백의 기울기차
# 분포가 거의 같다 (p50 0.083 vs 0.060). 거리가 아니라 방향이 판단 기준이다.
# 이으라고 허용한 클래스(점선·유도선) 안에서도, 옆 대시를 잘못 집는 것은
# 막아야 한다. 방향까지 맞아야 같은 선으로 본다.
JOIN_MAX_SLOPE_DIFF = 0.30  # 이음매에서 허용할 기울기 차
JOIN_CHECK_PTS = 3          # 이 정도 점이 모여야 국소 기울기를 말할 수 있다
JOIN_CHECK_SPAN_M = 4.0     # 이음매 양쪽에서 기울기를 잴 구간

# --- 3) 반복 --------------------------------------------------------------
SEED_PASSES = 3             # 남은 점으로 씨앗을 다시 찾는 최대 횟수
SEED_FREE_MIN = 12          # 남은 점이 이보다 적으면 더 안 돈다

# --- 4) 정리 --------------------------------------------------------------
DUP_SHARE = 0.5             # 점의 이 비율 이상을 공유하면 같은 경계로 본다
MIN_PTS_KEEP = 6            # 이보다 점이 적은 경계는 버린다
BOUND_MIN_SPAN_M = 3.0            # 세로로 이만큼은 이어져야 경계로 인정한다


def seed_band(x, lo_pct=SEED_X_LO_PCT, span_m=SEED_X_SPAN_M,
              span_max_m=SEED_X_SPAN_MAX_M, min_pts=SEED_MIN_BAND_PTS):
    """씨앗을 찾을 x 구간 `(lo, hi)`. 점이 모자라면 hi 를 넓힌다.

    시작을 백분위로 잡는 이유는 위 주석에 있다 - 외톨이 점 하나에 구간이
    끌려가면 안 된다. 넓히는 데 상한을 두는 이유는, 구간이 길어질수록 커브에서
    같은 경계의 y 가 크게 변해 봉우리가 뭉개지기 때문이다.
    """
    lo = float(np.percentile(x, lo_pct))
    hi = lo + span_m
    while hi < lo + span_max_m and int(((x >= lo) & (x <= hi)).sum()) < min_pts:
        hi += span_m
    return lo, min(hi, lo + span_max_m)


def find_seeds(x, y, bin_m=SEED_BIN_M, min_gap_m=SEED_MIN_GAP_M,
               min_ratio=SEED_MIN_RATIO, max_count=SEED_MAX_COUNT, band=None):
    """가까운 구간의 y 히스토그램에서 봉우리 위치를 찾는다. -> [y, ...]

    반환은 **강한 순**이 아니라 y 오름차순이다. 순서를 세기로 쓰지 않는다 -
    좌우 순번은 12단계가 곡선을 보고 매긴다.
    """
    if x.size == 0:
        return []
    lo, hi = band or seed_band(x)
    ys = y[(x >= lo) & (x <= hi)]
    if ys.size == 0:
        return []

    lo, hi = ys.min(), ys.max()
    n_bin = max(int(np.ceil((hi - lo) / bin_m)), 1)
    hist, edges = np.histogram(ys, bins=n_bin, range=(lo, lo + n_bin * bin_m))
    if SEED_SMOOTH_BINS > 1 and hist.size >= SEED_SMOOTH_BINS:
        k = np.ones(SEED_SMOOTH_BINS) / SEED_SMOOTH_BINS
        hist = np.convolve(hist.astype(np.float64), k, mode="same")

    peak = hist.max()
    if peak <= 0:
        return []
    centers = (edges[:-1] + edges[1:]) / 2.0

    # 강한 순으로 집어 가되, 이미 잡은 봉우리와 min_gap_m 안이면 건너뛴다.
    # **강한 것부터 자리를 잡아야** 약한 이웃이 강한 봉우리를 밀어내지 않는다.
    picked = []
    for i in np.argsort(hist)[::-1]:
        if hist[i] < peak * min_ratio:
            break
        c = float(centers[i])
        if all(abs(c - p) >= min_gap_m for p in picked):
            picked.append(c)
        if len(picked) >= max_count:
            break
    return sorted(picked)


def _local_slope(x, y, idx, lo, hi):
    """주어진 인덱스들 중 x 가 [lo, hi] 인 점으로 1차 기울기. 못 재면 None."""
    if idx.size < JOIN_CHECK_PTS:
        return None
    xs, ys = x[idx], y[idx]
    m = (xs >= lo) & (xs <= hi)
    if int(m.sum()) < JOIN_CHECK_PTS or len(np.unique(xs[m])) < 2:
        return None
    return float(np.polyfit(xs[m], ys[m], 1)[0])


def march(x, y, seed_y, x_start=None, step_m=STEP_M, margin_m=MARGIN_M,
          min_pts=BOUND_MIN_PTS, miss_max_m=MISS_MAX_M, drift_gain=DRIFT_GAIN,
          drift_max_m=DRIFT_MAX_M, margin_grow=MARGIN_GROW,
          margin_max_m=MARGIN_MAX_M,
          join_max_slope=JOIN_MAX_SLOPE_DIFF):
    """씨앗에서 출발해 x 를 전진하며 창 안의 점을 거둔다. -> 점 인덱스 배열

    `x` 는 **오름차순으로 정렬돼 있어야 한다** (호출부에서 한 번만 정렬한다).
    `x_start` 를 안 주면 씨앗 구간의 시작에서 출발한다 - `x.min()` 에서
    출발하면 안 되는 이유는 씨앗 쪽 주석과 같다.
    """
    if x.size == 0:
        return np.empty(0, np.int64)

    taken = []
    yc = float(seed_y)
    dy = 0.0                                    # 직전 창에서의 이동량
    missed = 0.0
    x0 = float(seed_band(x)[0] if x_start is None else x_start)
    x_end = float(x.max())

    while x0 < x_end:
        x1 = x0 + step_m
        # 정렬돼 있으므로 구간을 이분탐색으로 자른다
        i0, i1 = np.searchsorted(x, (x0, x1))
        # 빈 구간이 길수록 창을 넓힌다 (위 MARGIN_GROW 주석)
        m_now = min(margin_m + missed * margin_grow, margin_max_m)
        if i1 > i0:
            seg_y = y[i0:i1]
            inside = np.abs(seg_y - yc) <= m_now
            n = int(inside.sum())
        else:
            inside, n = None, 0

        if n and missed > 0 and join_max_slope is not None and taken:
            # **공백을 건너뛴 직후다.** 방향이 이어지는지 본다. 같은 선이면
            # 국소 기울기가 직전 구간과 비슷해야 한다 (위 주석의 실측).
            prev_idx = np.concatenate(taken)
            s_prev = _local_slope(x, y, prev_idx,
                                  x[prev_idx].max() - JOIN_CHECK_SPAN_M,
                                  x[prev_idx].max())
            new_idx = np.arange(i0, i1)[inside]
            s_new = _local_slope(x, y, new_idx, x0, x0 + JOIN_CHECK_SPAN_M)
            if s_prev is not None and s_new is not None \
                    and abs(s_new - s_prev) > join_max_slope:
                # 다른 선이다. 여기서 이 경계를 끝낸다 - 억지로 잇지 않는다.
                break

        if n:
            taken.append(np.arange(i0, i1)[inside])

        if n >= min_pts:
            # 평균이 아니라 중앙값 - 가장자리에 들어온 옆 차선 점에 안 끌린다
            y_new = float(np.median(seg_y[inside]))
            dy = np.clip(y_new - yc, -drift_max_m, drift_max_m)
            yc += dy
            missed = 0.0
        elif n:
            # 점이 있으면 **선 위에 있는 것이다.** 거두고 빈 구간 셈도 지운다.
            # 다만 점 한두 개로 중심을 옮기지는 않는다 - 원거리 노이즈 하나가
            # 경계를 옆 차선으로 끌고 가는 경로가 바로 이것이다.
            yc += dy * drift_gain
            missed = 0.0
        else:
            # 진짜로 빈 구간. 직전 기울기로 밀며 기다린다 (대시 간격 약 3m).
            yc += dy * drift_gain
            missed += step_m
            if missed >= miss_max_m:
                break
        x0 = x1

    return np.concatenate(taken) if taken else np.empty(0, np.int64)


def _dedupe(bounds, idx_sets, share=DUP_SHARE):
    """점을 많이 공유하는 경계를 하나로 줄인다. 점이 많은 쪽을 남긴다.

    씨앗이 SEED_MIN_GAP_M 만큼 떨어져 있어도, 커브에서 두 창이 같은 선으로
    수렴하는 일이 있다. 그대로 두면 같은 도색이 경계 둘로 나가고 12단계의
    좌우 순번이 하나씩 밀린다.
    """
    order = sorted(range(len(bounds)), key=lambda i: -idx_sets[i].size)
    keep = []
    for i in order:
        si = set(idx_sets[i].tolist())
        dup = False
        for j in keep:
            sj = set(idx_sets[j].tolist())
            inter = len(si & sj)
            if inter >= share * min(len(si), len(sj)):
                dup = True
                break
        if not dup:
            keep.append(i)
    return [bounds[i] for i in sorted(keep)]


def group_boundaries(ground, classes=None, min_pts_keep=MIN_PTS_KEEP,
          min_span_m=BOUND_MIN_SPAN_M, **march_kw):
    """{cls: (x, y, w_m)} -> ([Boundary], 통계)

    `w_m` 은 여기서 쓰지 않는다 - 7~8단계(폭)와 s03 의 판단 재료다. 받아서
    그냥 흘려보내되, 시그니처를 4~5단계 출력에 그대로 맞춰 둔다.
    """
    out, stats = [], {}
    for c in (classes if classes is not None else sorted(ground)):
        if c not in ground:
            continue
        x, y = ground[c][0], ground[c][1]
        stats[c] = {"pts": int(x.size), "seeds": 0, "grown": 0,
                    "short": 0, "dup": 0, "kept": 0, "assigned": 0}
        if x.size == 0:
            continue

        # 행진이 이분탐색을 쓰므로 **여기서 한 번만** 정렬한다
        srt = np.argsort(x)
        xs, ys = x[srt], y[srt]

        # **아직 아무 경계에도 안 들어간 점**으로 씨앗 찾기를 반복한다.
        # 한 번만 찾으면 강한 선에 눌려 약한 선이 통째로 빠진다 (위 주석).
        free = np.ones(xs.size, bool)
        grown, idx_sets = [], []
        for _ in range(SEED_PASSES):
            if int(free.sum()) < SEED_FREE_MIN:
                break
            where = np.flatnonzero(free)
            fx, fy = xs[where], ys[where]
            band = seed_band(fx)
            seeds = find_seeds(fx, fy, band=band)
            stats[c]["seeds"] += len(seeds)
            got_any = False
            for sy in seeds:
                sub = march(fx, fy, sy, x_start=band[0],
                            miss_max_m=miss_max_for(c), **march_kw)
                if sub.size == 0:
                    continue
                idx = where[sub]                # 원래 배열 인덱스로 되돌린다
                bx, by = xs[idx], ys[idx]
                if idx.size < min_pts_keep or (bx.max() - bx.min()) < min_span_m:
                    stats[c]["short"] += 1
                    continue
                grown.append(Boundary(cls=c, x=bx, y=by, seed_y=float(sy)))
                idx_sets.append(idx)
                free[idx] = False
                got_any = True
            if not got_any:
                break
        stats[c]["grown"] = len(grown)

        kept = _dedupe(grown, idx_sets)
        stats[c]["dup"] = len(grown) - len(kept)
        stats[c]["kept"] = len(kept)
        stats[c]["assigned"] = int(sum(b.x.size for b in kept))
        out += kept

    return out, stats


def format_boundary_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'pts':>6s} {'seeds':>6s} {'grown':>6s} "
             f"{'short':>6s} {'dup':>4s} {'kept':>5s} {'assigned':>9s} {'rate':>6s}"]
    for c, s in stats.items():
        r = s["assigned"] / s["pts"] * 100 if s["pts"] else 0.0
        lines.append(f"{names[c]:12s} {s['pts']:>6d} {s['seeds']:>6d} "
                     f"{s['grown']:>6d} {s['short']:>6d} {s['dup']:>4d} "
                     f"{s['kept']:>5d} {s['assigned']:>9d} {r:>5.1f}%")
    return lines

# ==========================================================================
# 9단계  RANSAC / Poly fitting
# ==========================================================================
# 9단계. RANSAC / Poly fitting — 경계 점열 -> 곡선 계수.
#
#     입력   [Boundary]     6단계 출력. 자차 좌표 (m)
#     출력   [Curve]        y = c0 x^2 + c1 x + c2, 인라이어 표시까지
#
# 파일 번호가 s06 다음 s09 인 것은 오타가 아니다. `_common.LaneResult` 가 정한
# 단계 번호를 따른다 - 7~8 단계(차로 폭 / 자기보정)는 아직 없다. 폭은 제어 출력에
# 필요하지 않아 뒤로 미뤘고, s04 주석이 말하는 "차로 폭 자기보정"이 거기 들어간다.
#
# ===========================================================================
# 왜 RANSAC 인가 - 최소제곱은 한 점에 끌려간다
# ===========================================================================
# 6단계가 묶어 준 점이라도 섞인 것이 남는다. 옆 차선 도색 몇 점, 가드레일
# 그림자, 대시 사이를 건너뛰며 창이 넓어졌을 때 빨려 들어온 것들이다.
#
# 최소제곱은 이상치 하나에 곡선 전체가 기운다. 특히 **원거리 이상치가 치명적**인데,
# 2차식의 곡률 항이 먼 쪽 몇 점으로 결정되기 때문이다. 근거리에서 0.1m 틀리는
# 것과 40m 에서 3m 틀리는 것이 같은 잔차 합을 만든다.
#
# RANSAC 은 "가장 많은 점이 동의하는 곡선"을 고르므로 소수 이상치가 결과를
# 바꾸지 못한다.
#
# ---------------------------------------------------------------------------
# 표본을 x 구간으로 나눠 뽑는다
# ---------------------------------------------------------------------------
# 무작위로 3점을 뽑으면 **점이 많은 근거리에서 셋 다 나온다.** 실측(40프레임):
# 백색실선 점의 밀도가 3~10m 에서 17.4점/m, 30~40m 에서 1.0점/m 이라 무작위
# 3점이 모두 20m 안에서 나올 확률이 높다. 짧은 구간에 맞춘 2차식은 먼 쪽으로
# 발산한다 - 곡률이 그 구간 밖에서는 근거가 없는 값이다.
#
# 그래서 x 를 (차수+1) 칸으로 나누고 **칸마다 하나씩** 뽑는다. 뽑힌 3점이 항상
# 전 구간에 걸치므로 곡률이 관측된 범위 전체의 지지를 받는다.
#
# ---------------------------------------------------------------------------
# 적합 구간을 잘라야 하는 이유
# ---------------------------------------------------------------------------
# 2차식 하나로 급커브 전 구간을 덮을 수 없다. 원 구현의 실측이 그대로 유효하다 -
# 급커브 프레임에서 전 구간 인라이어 비율 0.41, 근 20m 만 쓰면 0.98.
#
# 여기서는 **자르되 버리지 않는다.** 적합은 가까운 FIT_MAX_SPAN_M 구간으로 하고,
# `x_range` 는 그 구간으로 둔다. 먼 점은 `Curve.x/y` 에 남아 있어서 왜 잘렸는지
# 볼 수 있다. 제어가 쓰는 것은 어차피 30m 안쪽이다.

FIT_DEGREE = 2              # 자차 좌표에서는 2차면 충분하다 (BEV 때와 같은 결론)
FIT_MAX_SPAN_M = 22.0       # 적합에 쓸 전방 구간. 위 주석 참고

RANSAC_ITERS = 80           # 실측 기준. 100 이상에서 인라이어 수가 안 늘었다
FIT_RANSAC_THRESH_M = 0.20      # 이 안이면 인라이어. 차선 도색 폭(0.15~0.35m)의 절반쯤
FIT_RANSAC_MIN_RATIO = 0.40     # 인라이어가 이 비율 미만이면 적합 실패로 본다
FIT_MIN_PTS = (FIT_DEGREE + 1) * 4      # 이보다 적으면 RANSAC 이 의미 없다
FIT_MIN_SPAN_M = 3.0            # 세로로 이만큼은 걸쳐야 곡률을 말할 수 있다

# **물리적으로 불가능한 곡률은 적합 실패로 본다.**
# 교차로에서 서로 다른 선(교차 도로 도색, 횡단보도)의 점이 한 경계로 묶이면
# 2차식이 그 둘을 억지로 잇느라 되꺾이는 모양이 된다. 화면에서 갈고리처럼
# 보이는 곡선이 그것이다.
#
# 실측(자차 +-1 을 받은 곡선 522개): |a| 의 p50 0.0027, p90 0.0046 인데
# 최대가 0.192 다. a = 1/(2R) 이므로 0.192 는 **곡률반경 2.6m** - 도로 경계일
# 수 없다. 0.05(반경 10m)로 자르면 정상 곡선의 1.3% 만 걸린다.
#
# 유도선은 좌회전 안내선이 실제로 급하게 휘므로 더 느슨하게 준다.
MAX_CURV_LANE = 0.05        # 차선 경계. 곡률반경 10m
MAX_CURV_GUIDE = 0.15       # 유도선. 곡률반경 3.3m


def ransac_fit(x, y, rng, degree=FIT_DEGREE, iters=RANSAC_ITERS,
               thresh_m=FIT_RANSAC_THRESH_M, min_ratio=FIT_RANSAC_MIN_RATIO):
    """y = f(x) 에 RANSAC 으로 다항식을 맞춘다. -> (coef, inlier) 또는 None."""
    n = len(x)
    need = degree + 1
    if n < need * 4:
        return None

    # x 를 need 칸으로 나눠 칸마다 하나씩 뽑는다 (위 주석)
    edges = np.linspace(x.min(), x.max(), need + 1)
    bands = [np.flatnonzero((x >= edges[i]) & (x <= edges[i + 1])) for i in range(need)]
    bands = [b for b in bands if b.size]
    if not bands:
        return None

    best = None
    for _ in range(iters):
        pick = np.array([int(rng.choice(bands[i % len(bands)])) for i in range(need)])
        if len(np.unique(x[pick])) < need:
            continue
        try:
            coef = np.polyfit(x[pick], y[pick], degree)
        except (np.linalg.LinAlgError, ValueError):
            continue
        inl = np.abs(np.polyval(coef, x) - y) < thresh_m
        if best is None or inl.sum() > best.sum():
            best = inl

    if best is None or best.sum() < max(need, n * min_ratio):
        return None

    # **표본 3점이 아니라 인라이어 전체로 다시 맞춘다.** 3점 적합은 그 3점의
    # 측정 오차를 그대로 물려받는다. 다시 맞추면 인라이어 수십 점으로 평균된다.
    coef = np.polyfit(x[best], y[best], degree)
    inl = np.abs(np.polyval(coef, x) - y) < thresh_m
    if inl.sum() < need:
        return None
    return np.polyfit(x[inl], y[inl], degree), inl


def fit_curves(boundaries, rng=None, max_span_m=FIT_MAX_SPAN_M, min_pts=FIT_MIN_PTS,
          min_span_m=FIT_MIN_SPAN_M, **fit_kw):
    """[Boundary] -> ([Curve], 통계)

    `rng` 를 주면 결과가 재현된다. 안 주면 고정 시드로 만든다 - 프레임마다
    다른 난수를 쓰면 **같은 입력이 다른 곡선을 낸다.** 오프라인에서 코드를
    고쳤을 때 그 차이가 내 수정 때문인지 난수 때문인지 구분할 수 없게 된다.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    out, stats = [], {}

    for b in boundaries:
        s = stats.setdefault(b.cls, {"boundaries": 0, "few": 0, "short": 0,
                                     "failed": 0, "curv": 0, "fitted": 0,
                                     "inl_sum": 0.0})
        s["boundaries"] += 1

        # 적합 구간을 가까운 쪽으로 자른다 (버리지는 않는다)
        near = b.x <= b.x.min() + max_span_m
        fx, fy = b.x[near], b.y[near]

        if fx.size < min_pts:
            s["few"] += 1
            continue
        if fx.max() - fx.min() < min_span_m:
            s["short"] += 1
            continue

        fit = ransac_fit(fx, fy, rng, **fit_kw)
        if fit is None:
            s["failed"] += 1
            continue
        coef, inl = fit

        # 곡률 상한 (위 주석). 적합은 됐지만 도로 경계일 수 없는 모양이다.
        max_curv = MAX_CURV_GUIDE if b.cls == CLASS_GUIDE else MAX_CURV_LANE
        if abs(coef[0]) > max_curv:
            s["curv"] += 1
            continue

        out.append(Curve(cls=b.cls, coef=coef,
                         x_range=(float(fx.min()), float(fx.max())),
                         x=fx, y=fy, inlier=inl))
        s["fitted"] += 1
        s["inl_sum"] += float(inl.mean())

    return out, stats


def format_fit_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'bounds':>7s} {'few':>5s} {'short':>6s} "
             f"{'failed':>7s} {'curv':>5s} {'fitted':>7s} {'inlier':>7s}"]
    for c, s in stats.items():
        inl = s["inl_sum"] / s["fitted"] * 100 if s["fitted"] else 0.0
        lines.append(f"{names[c]:12s} {s['boundaries']:>7d} {s['few']:>5d} "
                     f"{s['short']:>6d} {s['failed']:>7d} {s['curv']:>5d} "
                     f"{s['fitted']:>7d} "
                     f"{inl:>6.1f}%")
    return lines

# ==========================================================================
# 10~11단계  Tracking (Hungarian + Kalman)
# ==========================================================================
# 10~11단계. Tracking — 프레임을 넘어 같은 차선을 같은 것으로 유지한다.
#
#     입력   [Curve]     9단계 출력 (그 프레임만의 관측)
#     출력   [Curve]     track_id / age / coasted 가 채워진 것
#
#     10단계  Data association    어느 관측이 어느 트랙인가        헝가리안
#     11단계  State estimation    그 트랙의 진짜 곡선은 무엇인가   칼만 필터
#
# ===========================================================================
# **이 파일은 파이프라인에서 처음으로 상태를 갖는다.**
# ===========================================================================
# s01~s09, s12 는 전부 "프레임 하나 -> 결과 하나" 인 순수 함수다. 추적은 본질상
# 그럴 수 없다 - 이전 프레임을 알아야 "같은 차선"을 말할 수 있다. 그래서 여기만
# `Tracker` 객체를 쓰고, 호출부가 그것을 들고 있어야 한다.
#
# 상태를 갖는 대가로 **재현성이 프레임 순서에 묶인다.** 오프라인에서 어떤 프레임
# 하나만 다시 돌려도 그 앞을 똑같이 돌리지 않으면 결과가 다르다. 디버깅할 때
# 이걸 잊으면 "왜 아까랑 다르지" 로 한참 헤맨다.
#
# ===========================================================================
# 왜 헝가리안인가 - 탐욕 매칭은 순서에 좌우된다
# ===========================================================================
# 옛 구현은 관측을 하나씩 돌며 "아직 안 쓰인 트랙 중 가장 가까운 것"을 집었다.
# 그러면 **검출 리스트의 순서가 결과를 바꾼다.**
#
# 깨지는 장면이 구체적으로 있다. 점선 두 줄이 게이트(0.8m) 근처로 나란히 있을 때,
# 먼저 스캔된 관측이 남의 트랙을 먼저 채가고 뒤엣것은 짝이 없어 새 ID 를
# 발급받는다. 차선은 그대로인데 track_id 가 튄다.
#
# 헝가리안은 **전체 비용 합이 최소가 되는 짝짓기**를 고르므로 순서와 무관하고,
# 그 장면에서 둘 다 제 트랙을 찾는다.
#
# ---------------------------------------------------------------------------
# scipy 를 쓰지 않고 직접 구현한 이유
# ---------------------------------------------------------------------------
# `_common.py` 머리말이 정한 제약이 "외부 의존은 torch / cv2 / numpy 뿐" 이다.
# 차선 수는 클래스당 많아야 대여섯이라 행렬이 6x6 을 넘지 않는다. 그 크기에서는
# O(n^3) 헝가리안이 수십 마이크로초라 라이브러리를 끌어올 이유가 없다.
# `scipy.optimize.linear_sum_assignment` 와 무작위 행렬로 대조 검증했다.
#
# ===========================================================================
# 칼만 - 상태를 다항식 계수로 두지 않는다
# ===========================================================================
# `y = a x^2 + b x + c` 의 `[a, b, c]` 를 그대로 필터링하고 싶어지는데 하면 안 된다.
# 셋의 크기가 `a ~ 1e-3`, `c ~ 1.7` 로 **세 자릿수 차이**나고 서로 강하게 상관돼
# 있어서, Q 와 R 을 어떤 값으로 줘도 한 성분이 나머지를 지배한다.
#
# 대신 **고정 전방거리에서의 y** 를 상태로 둔다.
#
#     상태  x = [ y(7m), y(15m), y(25m) ]          전부 미터, 같은 스케일
#     관측  z = 적합된 곡선을 같은 세 지점에서 평가     H = I (단위행렬)
#
# 세 점이면 2차식이 유일하게 결정되므로 정보 손실이 없다. H 가 단위행렬이라
# 칼만 식이 단순해지고, 무엇보다 **Q 와 R 을 미터로 생각할 수 있다** - "차선이
# 한 프레임에 몇 cm 움직일 수 있나", "이 적합은 몇 cm 쯤 틀렸나" 로.
#
# ---------------------------------------------------------------------------
# R 을 적합 품질에서 만든다 - 이게 없으면 칼만은 지연만 더한다
# ---------------------------------------------------------------------------
# 모든 관측을 같은 정확도로 취급하면 필터는 그냥 이동평균이고, 얻는 것은 부드러움
# 뿐이고 잃는 것은 반응 속도다. 관측마다 얼마나 믿을지를 달리 줘야 이득이 난다.
#
# 여기서는 세 가지를 본다.
#
#     1. 외삽 거리   knot 이 관측 구간 밖이면 그만큼 σ 를 키운다
#     2. 인라이어    RANSAC 인라이어 비율이 낮으면 σ 를 키운다
#     3. 점 수       점이 적을수록 σ 를 키운다
#
# 1번이 특히 중요하다. 점선은 대시 위상 때문에 관측 구간이 프레임마다 11~22m
# 였다 5~16m 였다 한다. 25m knot 이 3m 외삽인 프레임과 9m 외삽인 프레임을 같은
# 믿음으로 섞으면, **관측이 나빠진 프레임이 좋은 추정을 끌어내린다.**
#
# ---------------------------------------------------------------------------
# 자차 운동이 없으면 예측 모델이 반쪽이다 (알고 쓰는 한계)
# ---------------------------------------------------------------------------
# 차선 다항식은 **자차 좌표계**, 즉 차와 같이 움직이는 좌표계에 있다. 차가 1m
# 전진하고 조금 돌면 같은 차선이라도 y(7m) 값이 달라진다. 제대로 하려면 예측
# 단계에서 그 강체 변환을 태워야 한다.
#
# 지금 파이프라인에는 자차 속도/요레이트가 들어오지 않는다. 그래서 기본 동작은
# **랜덤워크**(F = I, Q 만 키움)이고, 그 대가로 Q 를 크게 잡을 수밖에 없다.
# Q 가 크면 필터가 관측을 많이 믿어 부드럽게 하는 힘이 약해진다.
#
#     Q_RATE_M_PER_S 의 근거: 옛 BEV 경로 실주행 로그(298프레임, 약 13fps)에서
#     연속 프레임 횡오차 변화가 중앙값 0.134m 였다. 0.134 / 0.077s = 1.74 m/s.
#     여유를 둬 3.0 으로 잡았다. **이 값은 차가 움직이는 양이지 센서 잡음이
#     아니다** - 그래서 자차 운동을 넣으면 이 항이 대부분 사라지고 Q 를 10배쯤
#     줄일 수 있다. 이 파이프라인에서 다음으로 이득이 큰 작업이 그것이다.
#
# `predict(dt, ego=(dx, dy, dpsi))` 로 넘기면 그 변환을 태운다. 인터페이스는
# 지금 열어 두되, 값이 없으면 랜덤워크로 돈다.

# --- 상태 정의 ------------------------------------------------------------
# 7m 는 12단계가 순번을 매기는 거리, 25m 는 제어가 전방주시에 쓰는 거리쯤.
# 15m 를 가운데 둬서 세 점이 곡률을 결정하게 한다.
X_KNOTS = np.array([7.0, 15.0, 25.0])
_VANDER = np.vander(X_KNOTS, 3)                 # [[x^2, x, 1], ...]
_VANDER_INV = np.linalg.inv(_VANDER)

# --- 10단계 연관 ----------------------------------------------------------
MATCH_MAX_M = 0.8           # knot 평균 |dy| 가 이보다 크면 같은 차선으로 안 본다
BIG = 1e6                   # 금지된 짝에 넣을 유한한 큰 값 (inf 를 넣으면 못 푼다)

# --- 11단계 상태추정 ------------------------------------------------------
Q_RATE_M_PER_S = 3.0        # 위 주석 참고. 자차 운동을 넣으면 크게 줄일 수 있다
P0_M = 1.0                  # 새 트랙의 초기 불확실성
R_BASE_M = 0.08             # 관측 구간 안에서의 기본 sigma. RANSAC 임계 0.2 의 절반 아래
R_EXTRAP_PER_M = 0.06       # 외삽 1m 당 sigma 증가
R_MIN_INLIER = 0.3          # 인라이어 비율이 이보다 낮으면 이 값으로 바닥을 친다
R_PTS_REF = 30              # 점이 이보다 적으면 sqrt 비례로 sigma 를 키운다
R_MAX_M = 5.0               # sigma 상한. 사실상 "이 knot 은 안 봤다" 는 뜻

# --- 수명: 정수 카운터가 아니라 신뢰도로 ----------------------------------
# **`hits >= MIN_HITS` 같은 하드 카운터를 출력 조건으로 쓰면 안 된다.** 실측으로
# 확인했다 - `MIN_HITS=2` 를 출력 조건에 걸었더니 자차 우측 존재율이
# 88.7% -> 82.3%, 최대 공백이 16 -> 30프레임으로 **나빠졌다.** 우측은 재획득이
# 잦은데 매번 2프레임을 기다리느라, 2프레임을 못 채우는 짧지만 멀쩡한 검출이
# 통째로 버려졌다.
#
# 카운터는 "짧은 점선 gap" 과 "차선이 실제로 없어짐" 을 구분하지 못한다. 같은
# 3프레임 공백이라도 직전 관측이 인라이어 100% 짜리였는지 60% 짜리였는지에 따라
# 얼마나 더 믿어도 되는지가 다르다.
#
# 그래서 신뢰도 하나로 합친다.
#
#     관측 있음  ->  적합 품질로 신뢰도를 끌어올린다
#     관측 없음  ->  MISS_DECAY 배로 깎고 공분산을 키운다
#     CONF_OUT  미만 -> 출력하지 않는다 (트랙은 살려 둔다)
#     CONF_DROP 미만 -> 트랙을 버린다
#
# 품질 좋은 새 트랙은 **첫 프레임부터 바로 나간다** (MIN_HITS 가 막던 것이
# 이것이다). 품질 나쁜 검출은 처음부터 임계 아래라 안 나간다.
CONF_OUT = 0.30             # 이 미만이면 출력하지 않는다
CONF_DROP = 0.15            # 이 미만이면 트랙을 버린다
MISS_DECAY = 0.6            # 관측이 빠진 프레임마다 신뢰도에 곱한다
CONF_BLEND = 0.4            # 새 관측 품질을 신뢰도에 섞는 비율
MAX_MISS = 5                # 신뢰도와 별개로 두는 하드 상한 (무한 관성 방지)

# 적합 품질 -> 0~1. R 과 같은 재료를 쓰되 이쪽은 "믿을 만한가" 한 값으로 줄인다.
QUAL_SPAN_REF_M = 12.0      # 이만큼 걸치면 span 점수 만점
QUAL_PTS_REF = 30           # 이만큼 점이 있으면 점수 만점


# ==========================================================================
# 10단계. 헝가리안 (Jonker-Volgenant 형태의 O(n^3) 최단증가경로)
# ==========================================================================
def hungarian(cost):
    """비용 최소 짝짓기. -> (rows, cols)

    행이 열보다 많으면 전치해서 푼다 (알고리즘이 n <= m 을 전제한다).
    모든 행이 어떤 열에든 배정되므로, **게이트 검사는 푼 뒤에 한다.**
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return np.empty(0, int), np.empty(0, int)
    flip = cost.shape[0] > cost.shape[1]
    c = cost.T if flip else cost
    n, m = c.shape

    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)          # p[j] = 열 j 에 배정된 행(1-based)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    r = np.array(rows, int)
    cl = np.array(cols, int)
    return (cl, r) if flip else (r, cl)


# ==========================================================================
# 11단계. 트랙 하나 = 칼만 필터 하나
# ==========================================================================
def coef_to_knots(coef):
    return np.polyval(coef, X_KNOTS)


def knots_to_coef(ys):
    """세 knot 의 y 로 2차식을 되돌린다. 세 점이면 유일하게 결정된다."""
    return _VANDER_INV @ ys


def measurement_sigma(curve):
    """관측 잡음 sigma 를 knot 마다 만든다. -> (3,)

    위 'R 을 적합 품질에서 만든다' 주석의 세 항목을 그대로 구현한다.
    """
    lo, hi = curve.x_range
    # knot 이 관측 구간 밖으로 나간 거리 (안이면 0)
    out = np.maximum(np.maximum(lo - X_KNOTS, X_KNOTS - hi), 0.0)
    sig = R_BASE_M + R_EXTRAP_PER_M * out

    inl = max(curve.inlier_ratio, R_MIN_INLIER) if curve.inlier.size else 1.0
    sig = sig / inl

    n = curve.x.size
    if 0 < n < R_PTS_REF:
        sig = sig * np.sqrt(R_PTS_REF / n)

    return np.minimum(sig, R_MAX_M)


def measurement_quality(curve):
    """적합 품질을 0~1 한 값으로. -> float

    인라이어 비율을 뼈대로 두고 span 과 점 수로 깎는다. 인라이어가 높아도
    3m 짜리 조각이면 곡률을 말할 수 없고, 점이 몇 개 없으면 그 인라이어
    비율 자체가 우연일 수 있다.
    """
    inl = float(curve.inlier_ratio) if curve.inlier.size else 0.5
    span = curve.x_range[1] - curve.x_range[0]
    q_span = min(span / QUAL_SPAN_REF_M, 1.0)
    q_pts = min(curve.x.size / QUAL_PTS_REF, 1.0) if curve.x.size else 0.0
    # span/점수는 인라이어를 **깎기만** 한다 (0.5~1.0 배). 재료가 부실하다고
    # 좋은 적합을 0 으로 만들면 새 트랙이 영영 못 선다.
    return float(inl * (0.5 + 0.5 * q_span) * (0.5 + 0.5 * q_pts))


class Track:
    """차선 경계 하나의 상태. `x` 는 knot 에서의 y 값 3개다."""

    __slots__ = ("id", "cls", "x", "P", "hits", "misses", "age", "last_curve",
                 "kalman", "conf")

    def __init__(self, track_id, curve, kalman=True):
        self.id = track_id
        self.cls = curve.cls
        self.kalman = kalman
        self.x = coef_to_knots(curve.coef)
        self.P = np.eye(3) * P0_M ** 2
        self.hits = 1
        self.misses = 0
        self.age = 1
        self.conf = measurement_quality(curve)
        self.last_curve = curve

    # --- 예측 -------------------------------------------------------------
    def predict(self, dt, ego=None):
        """dt 초 뒤를 예측한다. `ego=(dx, dy, dpsi)` 를 주면 강체변환을 태운다.

        `kalman=False` 면 아무것도 하지 않는다 - 상태가 곧 마지막 관측이라
        예측할 것이 없다. 연관(10단계)만 켜고 효과를 재는 모드다.
        """
        if not self.kalman:
            return
        if ego is not None:
            self.x, J = _shift_knots(self.x, ego)
            self.P = J @ self.P @ J.T
        q = (Q_RATE_M_PER_S * max(dt, 1e-3)) ** 2
        self.P = self.P + np.eye(3) * q

    # --- 갱신 -------------------------------------------------------------
    def update(self, curve):
        z = coef_to_knots(curve.coef)
        if self.kalman:
            R = np.diag(measurement_sigma(curve) ** 2)
            S = self.P + R                   # H = I 라 이렇게 단순해진다
            K = self.P @ np.linalg.inv(S)
            self.x = self.x + K @ (z - self.x)
            self.P = (np.eye(3) - K) @ self.P
        else:
            self.x = z                       # 관측을 그대로 (필터 없음)
        self.hits += 1
        self.age += 1
        self.misses = 0
        q = measurement_quality(curve)
        # 한 프레임의 품질에 통째로 끌려가지 않게 섞는다. 좋은 관측이 이어지면
        # 신뢰도가 서서히 올라가고, 한 번 나쁜 관측에 곧바로 무너지지 않는다.
        self.conf = (1 - CONF_BLEND) * self.conf + CONF_BLEND * q
        self.last_curve = curve

    def coast(self):
        self.misses += 1
        self.age += 1
        self.conf *= MISS_DECAY

    @property
    def coef(self):
        return knots_to_coef(self.x)

    def distance(self, curve):
        """관측과의 거리 = knot 평균 |dy| (m). 클래스가 다르면 None."""
        if curve.cls != self.cls:
            return None
        return float(np.mean(np.abs(coef_to_knots(curve.coef) - self.x)))


def _shift_knots(ys, ego):
    """자차가 (dx, dy, dpsi) 만큼 움직였을 때의 새 knot y. -> (y', 야코비안)

    옛 자차좌표 점 p 를 새 자차좌표로 옮기면  p' = R(-dpsi) (p - t) 다. 옮긴
    세 점의 x 는 더 이상 knot 이 아니므로 2차식을 다시 맞춰 knot 에서 평가한다.

    야코비안은 **수치 미분**으로 구한다. 3x3 이라 비용이 없고, 손으로 유도하다
    부호 하나 틀리면 조용히 발산하는 종류의 식이라 그게 낫다.
    """
    dx, dy, dpsi = ego

    def f(y3):
        p = np.stack([X_KNOTS, np.asarray(y3)], axis=1)
        p = p - np.array([dx, dy])
        c, s = np.cos(-dpsi), np.sin(-dpsi)
        p = p @ np.array([[c, s], [-s, c]]).T
        # 옮긴 세 점으로 2차식을 다시 맞춰 knot 에서 평가
        return np.polyval(np.polyfit(p[:, 0], p[:, 1], 2), X_KNOTS)

    base = f(ys)
    J = np.empty((3, 3))
    h = 1e-4
    for k in range(3):
        d = np.zeros(3)
        d[k] = h
        J[:, k] = (f(ys + d) - base) / h
    return base, J


# ==========================================================================
# 도로 형상 상태 - **맵으로 갈아끼울 자리**
# ==========================================================================
class GeometryState:
    """지금 도로가 얼마나 "평범한가" 를 0~1 로 알려준다 (0 정상, 1 교차로).

    ---------------------------------------------------------------------
    왜 이산 모드(Normal / Junction)가 아니라 연속값인가
    ---------------------------------------------------------------------
    모드를 매 프레임 판정하면 **경계에서 모드가 깜빡인다.** 그러면 칼만 게인이
    프레임마다 출렁여서 오히려 없느니만 못한 구간이 생기고, 결국 모드에도
    hysteresis 를 달아야 한다 - hysteresis 위에 hysteresis 다.

    연속값이면 게이트와 Q 를 **부드럽게** 키울 수 있어서 그 문제가 없다.

    ---------------------------------------------------------------------
    맵 기반으로 교체하는 방법
    ---------------------------------------------------------------------
    이 클래스를 상속해 `score()` 만 구현하고 `Tracker(geometry=...)` 로 넣으면
    된다. 파이프라인 다른 곳은 손대지 않는다.

        class MapGeometry(GeometryState):
            def __init__(self, mgeo): ...
            def score(self, curves, prev_curves, matched, total):
                # 자차 위치가 교차로 폴리곤 안이면 1.0
                return 1.0 if self._in_junction() else 0.0

        tracker = Tracker(geometry=MapGeometry(mgeo))

    맵이 있으면 맵이 이긴다 - 검출 기반 추정은 결국 증상을 보는 것이고,
    맵은 원인을 안다.
    """

    def score(self, curves, prev_curves, matched, total, context=None):
        """`context` 는 호출부가 넣어 주는 부가 정보다 (예: s02 픽셀 통계).

        구현체마다 필요한 재료가 달라서 열어 둔다 - 도색 기반은 마스크 통계가,
        맵 기반은 자차 위치가 필요하다. 없으면 없는 대로 동작해야 한다.
        """
        return 0.0


class DetectionGeometry(GeometryState):
    """맵이 없을 때의 대체품. **검출 결과의 변화만으로** 추정한다.

    실측(300프레임): 한쪽 경계 결측이 시작된 4곳 중 3곳에서 직전 2프레임 사이
    경계 개수가 변했다. 신호는 있지만 표본이 작고, 개수 급변 프레임 자체가
    9개뿐이라 **이것만으로 교차로를 판정하기에는 약하다.** 그래서 점수를 크게
    주지 않고, 맵이 들어오면 교체되는 것을 전제로 둔다.
    """

    def __init__(self, count_gain=0.25, unmatched_gain=0.5, decay=0.7):
        self.count_gain = count_gain
        self.unmatched_gain = unmatched_gain
        self.decay = decay
        self._score = 0.0

    def score(self, curves, prev_curves, matched, total, context=None):
        s = 0.0
        d = abs(len(curves) - len(prev_curves))
        s += min(d * self.count_gain, 1.0)
        if total:
            s += self.unmatched_gain * (1.0 - matched / total)
        # 한 프레임 튀는 값에 반응하지 않게 감쇠시켜 이어 간다
        self._score = max(self.decay * self._score, min(s, 1.0))
        return self._score


class PaintGeometry(GeometryState):
    """**유도선과 정지선이 함께 보이면 교차로다.** 도색만 보고 판정한다.

    ---------------------------------------------------------------------
    왜 이것이 개수 급변보다 나은가 (실측 300프레임)
    ---------------------------------------------------------------------
    유도선은 교차로 안내선이고 정지선은 교차로 진입부에만 있다. 둘이 같이
    보이는 곳은 교차로뿐이다. 측정하면 그 관계가 그대로 나온다.

        둘 다 >50px 인 구간 **밖**의 자차 차선 결측률    0.0%
        둘 다 >50px 인 구간 **안**의 결측률             45.6%
        결측 67프레임 중 67개가 전부 "둘 다" 구간       <- 놓친 것 0

    즉 **재현율이 100%** 다. 정밀도는 낮지만(플래그 147프레임 중 실제 결측 67)
    그것은 문제가 아니다 - 이산 모드가 아니라 연속 점수로 게이트를 조이는
    구조라, 교차로 근처에서 조금 보수적으로 도는 것은 손해가 아니다.

    `DetectionGeometry`(경계 개수 급변)는 같은 데이터에서 결측 시작 4곳 중
    3곳만 잡았다. 증상이 아니라 원인을 보는 쪽이 이긴다.

    ---------------------------------------------------------------------
    재료는 공짜다
    ---------------------------------------------------------------------
    s02 가 이미 클래스별 픽셀 수를 통계로 내고 있다. 다시 세지 않는다.
    호출부가 `context={"s02": stats}` 로 넘겨 주면 된다.
    """

    def __init__(self, px_ref=200, px_floor=30, decay=0.75):
        self.px_ref = px_ref
        self.px_floor = px_floor        # 이보다 적으면 없는 것으로 본다
        self.decay = decay
        self._score = 0.0

    def _px(self, context, cls):
        st = (context or {}).get("s02") or {}
        v = st.get(cls)
        return float(v.get("after", 0)) if isinstance(v, dict) else 0.0

    def score(self, curves, prev_curves, matched, total, context=None):
        g = self._px(context, CLASS_GUIDE)
        t = self._px(context, CLASS_STOPLINE)
        if g < self.px_floor or t < self.px_floor:
            raw = 0.0
        else:
            # **둘 다** 있어야 한다. 약한 쪽이 점수를 정한다.
            raw = min(min(g / self.px_ref, 1.0), min(t / self.px_ref, 1.0))
        # 한 프레임 깜빡임에 반응하지 않게 감쇠시켜 이어 간다 (실측에서 1프레임
        # 짜리 구간이 여럿 있었다)
        self._score = max(self.decay * self._score, raw)
        return self._score


# ==========================================================================
# 트래커 - 프레임을 넘어 상태를 들고 있는 유일한 객체
# ==========================================================================
class Tracker:
    """`update(curves, dt)` 를 프레임마다 부른다."""

    def __init__(self, match_max_m=MATCH_MAX_M, max_miss=MAX_MISS,
                 conf_out=CONF_OUT, conf_drop=CONF_DROP, kalman=True,
                 assoc="hungarian", geometry=None):
        self.tracks = []
        self._next_id = 1
        # **탐욕도 남겨 둔다.** 헝가리안이 실제로 얼마나 이득인지는 같은 녹화본에
        # 나머지 조건을 고정하고 연관만 바꿔 봐야 안다. 옛 구현과 같은 규칙을
        # 그대로 구현해 두어야 그 비교가 성립한다 - 지워 버리면 "좋아졌다"를
        # 주장할 근거가 사라진다.
        if assoc not in ("hungarian", "greedy"):
            raise ValueError(f"assoc 는 hungarian / greedy 중 하나여야 합니다: {assoc}")
        self.assoc = assoc
        # **연관(10)과 상태추정(11)을 따로 켤 수 있게 둔다.** 헝가리안만 켠
        # 상태에서 track_id 가 얼마나 안정되는지를 먼저 재고, 그 위에 칼만을
        # 얹어야 어느 쪽이 이득을 냈는지 가릴 수 있다.
        self.kalman = kalman
        self.match_max_m = match_max_m
        self.max_miss = max_miss
        self.conf_out = conf_out
        self.conf_drop = conf_drop
        # 맵이 있으면 MapGeometry 를 넣는다 (GeometryState 주석 참고)
        # 기본은 도색 기반 (PaintGeometry 주석의 실측 참고). 맵이 있으면
        # MapGeometry 를 넣어 교체한다.
        self.geometry = geometry if geometry is not None else PaintGeometry()
        self._prev_curves = []
        self.junction_score = 0.0

    def update(self, curves, dt=0.1, ego=None, context=None):
        """[Curve] -> ([Curve], 통계). 반환 곡선의 coef 는 **필터된 값**이다."""
        stats = {"tracks_in": len(self.tracks), "curves": len(curves),
                 "matched": 0, "gated": 0, "new": 0, "coasted": 0,
                 "dropped": 0, "unconfirmed": 0, "out": 0, "junction": 0.0}

        for t in self.tracks:
            t.predict(dt, ego)

        # --- 10단계: 연관 ---------------------------------------------------
        pairs = {}
        # 교차로일수록 게이트를 좁힌다 (최대 절반까지). 애매하면 잇지 않고
        # unmatched 로 두는 편이, 잘못 이어 붙이는 것보다 낫다.
        gate = self.match_max_m * (1.0 - 0.5 * self.junction_score)
        if self.tracks and curves:
            n, m = len(self.tracks), len(curves)
            cost = np.full((n, m), BIG)
            for i, t in enumerate(self.tracks):
                for j, c in enumerate(curves):
                    d = t.distance(c)
                    if d is not None and d <= gate:
                        cost[i, j] = d

            if self.assoc == "hungarian":
                ri, ci = hungarian(cost)
                for i, j in zip(ri, ci):
                    # **게이트는 푼 뒤에 건다.** 헝가리안은 모든 행을 배정하므로
                    # 금지된 짝(BIG)도 결과에 들어온다. 여기서 걸러야 한다.
                    if cost[i, j] < BIG:
                        pairs[i] = j
                    else:
                        stats["gated"] += 1
            else:
                # 옛 구현과 같은 규칙: **관측 순서대로** 아직 안 쓰인 트랙 중
                # 가장 가까운 것을 집는다. 뒤에 오는 관측이 더 그 트랙을
                # 필요로 해도 이미 늦었다 - 이것이 순서 의존성의 정체다.
                used = set()
                for j in range(m):
                    best, bestd = None, gate
                    for i in range(n):
                        if i in used:
                            continue
                        if cost[i, j] < BIG and cost[i, j] < bestd:
                            best, bestd = i, cost[i, j]
                    if best is None:
                        stats["gated"] += 1
                    else:
                        used.add(best)
                        pairs[best] = j

        matched_curves = set(pairs.values())

        # --- 11단계: 갱신 / 관성 ------------------------------------------
        for i, t in enumerate(self.tracks):
            if i in pairs:
                t.update(curves[pairs[i]])
                stats["matched"] += 1
            else:
                t.coast()
                stats["coasted"] += 1

        for j, c in enumerate(curves):
            if j not in matched_curves:
                self.tracks.append(Track(self._next_id, c, self.kalman))
                self._next_id += 1
                stats["new"] += 1

        # --- 도로 형상 점수 갱신 -------------------------------------------
        # 교차로에서는 게이트를 **좁힌다.** 기존 트랙을 억지로 다른 차선에
        # 이어 붙이는 것이 여기서 가장 위험한 실패이기 때문이다. 넓히면
        # 반대로 엉뚱한 경계를 빨아들인다.
        self.junction_score = self.geometry.score(
            curves, self._prev_curves, stats["matched"], len(self.tracks),
            context)
        self._prev_curves = list(curves)
        stats["junction"] = round(self.junction_score, 3)

        before = len(self.tracks)
        # 신뢰도가 바닥나면 버린다. max_miss 는 하드 상한으로만 남긴다.
        self.tracks = [t for t in self.tracks
                       if t.conf >= self.conf_drop and t.misses <= self.max_miss]
        stats["dropped"] = before - len(self.tracks)

        # --- 출력 ----------------------------------------------------------
        # **관측이 없는 프레임에도 내보낸다** (옛 구현은 안 내보내서, 트랙을
        # 들고 있으면서도 출력은 그대로 깜빡였다). 다만 coast 가 길어지면
        # 예측 오차가 커지므로 COAST_MAX 까지만 내보낸다.
        #
        # kalman=False 면 여기서 나가는 것은 "예측"이 아니라 **마지막 관측을
        # 그대로 다시 낸 것**이다. 차가 움직이면 그만큼 뒤처진 값이니,
        # 연관만 켠 모드에서는 coast 출력을 믿을 것이 못 된다.
        out = []
        for t in self.tracks:
            # **신뢰도로 거른다.** 품질 좋은 새 트랙은 첫 프레임부터 나가고,
            # 관측이 끊긴 트랙은 감쇠하다 알아서 임계 아래로 내려간다.
            # 정수 카운터와 달리 "짧지만 좋은 검출" 을 버리지 않는다.
            if t.conf < self.conf_out:
                stats["unconfirmed"] += 1
                continue
            src = t.last_curve
            coasted = t.misses > 0
            out.append(Curve(cls=t.cls, coef=t.coef,
                             x_range=src.x_range,
                             x=np.empty(0) if coasted else src.x,
                             y=np.empty(0) if coasted else src.y,
                             inlier=np.empty(0, bool) if coasted else src.inlier,
                             track_id=t.id, age=t.age, coasted=coasted,
                             confidence=round(float(t.conf), 3)))
        stats["out"] = len(out)
        return out, stats


def format_track_stats(stats, names=None):
    return [f"트랙 {stats['tracks_in']:2d} + 관측 {stats['curves']:2d}  ->  "
            f"매칭 {stats['matched']:2d}  신규 {stats['new']:2d}  "
            f"관성 {stats['coasted']:2d}  저신뢰 {stats['unconfirmed']:2d}  "
            f"junction {stats['junction']:.2f}  "
            f"버림 {stats['dropped']:2d}  출력 {stats['out']:2d}"]

# ==========================================================================
# 12단계  Lane ID
# ==========================================================================
# 12단계. Lane ID — 곡선에 자차 기준 좌우 순번을 매긴다.
#
#     입력   [Curve]      9단계 출력
#     출력   lane_id 가 채워진 [Curve] (같은 객체를 고친다)
#
# 부호는 **y 부호를 따른다.** 자차 좌표계가 y 좌측이 + 이므로 왼쪽이 +1, +2 이고
# 오른쪽이 -1, -2 다. `_common.LaneResult.ego_left` 가 `by_lane_id(1)` 인 것과
# 같은 규약이다. (옛 `lane_detection.py` 는 왼쪽이 음수였다. 반대이니 옮겨 붙일 때
# 주의한다.)
#
# ---------------------------------------------------------------------------
# **0 = 지금 밟고 있는 선**
# ---------------------------------------------------------------------------
# 차선 변경 중에는 경계 하나가 차 밑을 지나간다. 그 선의 y 는 +에서 -로 넘어가는데,
# 0 을 두지 않으면 그 순간 `+1` 과 `-1` 이 깜빡인다 - 같은 도색이 프레임마다
# "내 왼쪽 경계" 였다가 "내 오른쪽 경계" 가 된다. 제어가 그 값을 그대로 먹으면
# 조향이 좌우로 떨린다.
#
# |y| 가 차폭 절반 안이면 **그 선은 좌우 어느 쪽 경계도 아니다. 내가 올라타 있는
# 선이다.** 그래서 0 을 준다.
#
#     차선 변경 중        0  = 밟고 있는 선
#                        +1 = 그 왼쪽 선      (왼쪽 차로의 바깥 경계)
#                        -1 = 그 오른쪽 선    (오른쪽 차로의 바깥 경계)
#
# 제어 입장에서 "지금 0번 선 위에 있고, 갈 수 있는 곳은 0~+1 사이 아니면
# 0~-1 사이" 로 읽힌다. 평상시에는 경계가 +-1.6m 쯤에 있으므로 0 이 나오지
# 않는다 - **차선 변경 중에만 나타나는 값**이다.
#
# 유도선(`CLASS_GUIDE`)은 **순번 매기기에는 들어가지 않는다.** 차로 경계가 아니라
# 진로 안내선이라 +-1 슬롯을 그냥 차지하면 차로 폭 판단이 무너진다.
#
# 다만 **도색이 없을 때만** 자차 경계 자리를 대신 채운다 (아래 참고).
#
# ---------------------------------------------------------------------------
# 도색이 없을 때 유도선으로 자차 좌측을 대신한다
# ---------------------------------------------------------------------------
# 교차로에는 차로 도색이 없고, 점선은 대시가 끊기며, 차선 변경 중에는 경계가
# 사라진다. 300프레임 주행 실측:
#
#     자차 좌 없음    27프레임  ->  유도선 27/27 있고 전부 자차 왼쪽
#     자차 우 없음    51프레임  ->  유도선 50/51 있고 전부 자차 왼쪽
#     둘 다 없음      11프레임  ->  유도선 11/11 있고 전부 자차 왼쪽
#     왼쪽 유도선 y(7m) 중앙값 +1.34m
#
# 즉 도색이 없는 구간에서도 **왼쪽 유도선은 거의 항상 있다.** 그것을 쓰지 않을
# 이유가 없다.
#
# **그런데 그냥 넣으면 안 된다.** 두 가지를 지킨다.
#
#   1) `from_guide=True` 를 반드시 표시한다. 출력의 `left`/`right` 는 제어에게
#      "여기까지 비켜도 된다" 는 뜻인데, 유도선은 넘으면 안 되는 선이 아니라
#      지나갈 길 힌트다. 표시가 없으면 회피 계획이 이것을 벽으로 오해한다.
#      차로 폭 계산에서도 빼야 한다.
#
#   2) **자차 좌측 경계의 연장선인 유도선만 쓴다.** 교차로에는 좌회전/직진 등
#      여러 방향의 안내선이 같이 보인다. 측정된 왼쪽 유도선 y 범위가
#      +0.23 ~ +5.29m 로 벌어져 있고, 그중 내 차로의 연장인 것은 하나뿐이다.
#
# 도색이 하나라도 있으면 그쪽이 이긴다. 유도선은 마지막 수단이다.
#
# ---------------------------------------------------------------------------
# "왼쪽에 있는 유도선" 이 아니라 "좌측 경계에 이어지는 유도선" 이어야 한다
# ---------------------------------------------------------------------------
# 처음에는 "자차 왼쪽에서 가장 가까운 유도선" 을 골랐다. 그런데 **교차로에서
# 장애물을 피하느라 차가 옆으로 밀리면 원래 좌측이던 선이 좌측이 아니게 된다.**
# 그 순간 규칙이 엉뚱한 안내선을 집는다.
#
# 방향이 아니라 **연속성**으로 판단해야 한다. 실측(유도선 179개, 좌측 도색과의
# 이음매에서의 가로 어긋남):
#
#     p10  0.06m      중앙  1.15m      p90  15.59m
#
# 한쪽 무리는 좌측 실선과 거의 같은 선 위에 있고(0.5m 이내가 43%), 나머지는
# 15m 씩 벌어진다. 뚜렷하게 갈린다.
#
# **통과하는 것이 없으면 유도선을 아예 쓰지 않는다.** 회피 중이라 내 차로의
# 연장이 보이지 않는 상황이라면, 제어는 어차피 차선 중심이 아니라 회피 경로를
# 따라가는 중이므로 그 기준이 필요하지도 않다. 엉뚱한 안내선을 따라가느니
# "없음" 이 낫다.
#
# ---------------------------------------------------------------------------
# 연속성은 **기준선이 보이는 동안** 판정해 둔다 (GuideLink)
# ---------------------------------------------------------------------------
# 연속성을 "도색이 사라진 그 프레임" 에 판정하려고 하면 순환에 빠진다. 유도선
# 폴백은 자차 좌측 도색이 **없을 때** 도는데, 비교할 기준선이 바로 그 없어진
# 도색이기 때문이다. 실제로 그렇게 짰더니 300프레임 전부 "기준 좌측 경계 없음"
# 으로 탈락했다.
#
# 순서를 뒤집으면 풀린다.
#
#     좌측 도색이 보인다  ->  이어지는 유도선을 찾아 **track_id 를 기억**해 둔다
#                             (이때는 도색을 쓰므로 유도선은 출력하지 않는다)
#     좌측 도색이 사라졌다 ->  기억해 둔 track_id 의 유도선을 자차 좌측으로 쓴다
#     그 트랙이 죽었다     ->  없음. 다른 유도선으로 대체하지 않는다
#
# 마지막 줄이 중요하다. 회피로 차가 옆으로 밀려 내 차로의 연장이 아니게 되면
# 링크가 끊기고, 그러면 **유도선을 안 쓴다.** 옆에 다른 안내선이 보여도 집지
# 않는다 - 그것은 내 차로의 연장이 아니다.
#
# 링크를 얼마나 오래 유지할지는 따로 정하지 않는다. 유도선도 10~11단계가
# 추적하므로, 그 트랙의 신뢰도가 바닥나 사라지면 링크도 자연히 끊긴다.
#
# ===========================================================================
# 왜 "같은 전방거리"에서 비교해야 하는가
# ===========================================================================
# 곡선마다 자기 시작점에서 y 를 재면 안 된다. 점선은 대시 위상 때문에 시작점이
# 프레임마다 5m 였다 15m 였다 하고, **커브에서는 그 차이가 곧 y 차이**라 순서가
# 뒤집힌다. 주행 중 ego_right 가 옆 차선으로 넘어가는 전형적인 원인이다.
#
# 그래서 한 전방거리를 정해 전부 거기서 잰다.
#
# ---------------------------------------------------------------------------
# 그 전방거리는 **가까워야 한다.** 외삽을 피하려다 순번이 뒤집힌다
# ---------------------------------------------------------------------------
# 외삽이 싫어서 "모든 곡선이 실제로 관측된 구간" 에서 고르는 방법을 먼저 넣었다
# (`ORDER_X = clip(7, max(x_lo), min(x_hi))`). 외삽은 0 이 되지만 **틀린다.**
#
# 실측(40프레임): 그 방식은 `order_x` 를 13.6m 로 골랐고 자차 좌측을 **0/40
# 프레임**에서 찾았다. 급커브 구간이라 13.6m 앞에서는 차선이 이미 옆으로 쓸려가,
# 자차 왼쪽에 있던 황색선이 거기서는 y=-0.40 (정면)이 되어 우측으로 분류됐다.
# 같은 프레임을 7m 에서 재면
#
#     +6.78   +3.00   -0.75   -4.60      간격 3.78 / 3.75 / 3.85m
#
# 로 차로 폭과 맞아떨어지고 자차 좌=황색, 우=백색점선이 정상으로 잡힌다.
#
# lane_id 가 답하는 질문은 "**지금** 내 차로를 무엇이 끼고 있나" 다. 본질적으로
# 근거리 질문이라 먼 데서 재면 아무리 정확해도 다른 질문의 답이 된다. 2차식
# 외삽 오차는 근거리 몇 미터에서는 작고, 그 정도는 감수하는 것이 맞다.
#
# 발산만 막는다 - 외삽한 y 가 도로 밖(`Y_SANITY_M`)으로 날아가면 그 곡선만
# 관측 구간 끝으로 당겨서 잰다.
#
# ---------------------------------------------------------------------------
# +-1 은 "가까워야" 받는다
# ---------------------------------------------------------------------------
# 가장 안쪽 곡선이라도 자차에서 한 차로 폭 넘게 떨어져 있으면 그것은 자차 차로
# 경계가 아니다. 반대쪽 차로의 경계이거나 검출이 하나 빠진 것이다. 그 경우
# +-1 을 **비워 두고** +-2 부터 매긴다.
#
# 제어에 6m 밖 차선을 "자차 경계" 라고 주는 것보다 "없음" 이 낫다. 원 구현 실측:
# 이 검사를 넣기 전 ego 차선이 프레임 사이 4.5~4.9% 확률로 옆 차선으로 튀었고,
# 넣은 뒤 0% 가 됐다.
#
# 순번과 같은 거리(`ORDER_X_M`)에서 잰다. 위에서 그 거리를 근거리로 고정했기
# 때문에 따로 둘 이유가 없다 - 멀리서 재면 커브에서 멀쩡한 자차 경계가 한 차로
# 폭 밖으로 보여 전부 탈락한다.

ORDER_X_M = 7.0             # 전부 이 전방거리에서 재서 순번을 매긴다
GUIDE_FALLBACK = True       # 도색이 없을 때 유도선으로 자차 좌측을 대신할지
GUIDE_MAX_Y_M = 6.0         # 이보다 먼 유도선은 애초에 보지 않는다 (안전장치)
# 좌측 경계와의 연속성 게이트. 실측 분포(가로 어긋남 p10 0.06 / 중앙 1.15 /
# p90 15.59m)에서 첫 무리를 담는 값으로 잡았다.
GUIDE_JOIN_MAX_Y_M = 0.8        # 이음매에서의 가로 어긋남 한도
GUIDE_JOIN_MAX_SLOPE = 0.15     # 이음매에서의 기울기 차 한도 (rad/m)

# --- 유도선이 두 갈래로 갈라질 때 ------------------------------------------
# 좌측 경계에서 직진 유도선과 좌회전 유도선이 같이 나가는 교차로가 있다.
# 둘 다 이음매에서는 똑같이 이어지므로 **가까운 쪽으로는 구분이 안 된다.**
# 갈라지는 것은 먼 쪽이다.
#
# 판별은 **기준선(좌측 경계)을 연장한 것과 비교**해서 한다. "시작 접선 대비
# 휨" 으로 재면 도로 자체의 곡률이 섞여서, 도로가 오른쪽으로 굽으면 직진
# 유도선도 우측으로 휜 것처럼 나온다 (실측 130개 중 우측휨 35개가 그 경우다).
# 기준선 대비로 재면 도로 곡률이 상쇄된다.
#
#     직진 유도선   기준선 연장에서 거의 안 벗어난다
#     좌회전 유도선 좌측(+y)으로 크게 벗어난다
#     우회전/반대편 우측(-y)으로 벗어난다 -> 이 맵에서는 필요 없다
GUIDE_BRANCH_X_M = 20.0         # 갈라짐을 재는 전방거리
GUIDE_BRANCH_M = 1.5            # 이보다 벗어나면 직진이 아니다
# **이 임계는 근거가 약하다.** 녹화본 300프레임에 유도선이 2개 이상인 프레임이
# 10개뿐이라 갈라지는 장면이 거의 없었다. 분기 구간을 따로 녹화해 다시 잡아야
# 한다.
Y_SANITY_M = 12.0           # 외삽한 y 가 이보다 밖이면 발산으로 본다 (도로 밖)
EGO_MAX_Y_M = LANE_WIDTH_M * 0.9    # 3.15m. 자차 경계로 인정할 최대 |y|

# **자차 차로 경계는 자차와 거의 나란하다.** 7m 앞에서 크게 꺾여 나가는 선은
# 교차 도로의 도색이지 내 차로의 경계가 아니다.
#
# 실측(자차 +-1 을 받은 곡선 522개): |기울기(7m)| 의 p50 0.03, p90 0.09.
# 0.5(약 27도)를 넘는 것은 3개(0.6%)뿐이고, 그 3개가 교차로에서 엉뚱한 선을
# 자차 좌측으로 집던 경우다.
#
# 곡선을 버리지는 않는다 - 교차 도로 경계도 그 자체로는 맞는 검출이다.
# 다만 **자차 슬롯(+-1, +-2)을 받을 자격이 없을** 뿐이다.
EGO_MAX_SLOPE = 0.5         # 7m 에서의 |dy/dx| 상한 (약 27도)

# 이 안이면 "밟고 있는 선"으로 보고 lane_id 0 을 준다 (위 주석).
#
# **차폭 절반(0.9m)으로 잡으면 안 된다.** 실측(300프레임 주행): 평상 주행에서
# 가장 가까운 차선의 |y(7m)| 이 p05 0.91m, 최솟값 0.84m 다. 0.9 로 두면 차선을
# 밟은 적이 없는데도 14프레임에서 발동해, 정상 경계가 0 이 되고 그 바깥이
# +-1 로 밀려 dy 가 7m 씩 튀었다.
#
# 실제로 선을 밟으면 y 는 0 을 지나가므로 임계가 좁아도 놓치지 않는다.
# 관측된 최솟값(0.84m)보다 확실히 아래로 둔다.
STRADDLE_Y_M = 0.5


def order_y(c, x=ORDER_X_M, y_sanity=Y_SANITY_M):
    """순번용 y. 외삽이 발산하면 관측 구간 끝으로 당겨서 잰다."""
    y = c.y_at(x)
    if abs(y) <= y_sanity:
        return y
    lo, hi = c.x_range
    return c.y_at(float(np.clip(x, lo, hi)))


def _slope(c, x):
    return 2.0 * c.coef[0] * x + c.coef[1]


def _join_x(a, b):
    """두 곡선을 비교할 x. 겹치면 겹친 구간의 가운데, 아니면 가까운 두 끝의 중점."""
    lo = max(a.x_range[0], b.x_range[0])
    hi = min(a.x_range[1], b.x_range[1])
    if lo <= hi:
        return 0.5 * (lo + hi)
    return 0.5 * (min(a.x_range[1], b.x_range[1]) + max(a.x_range[0], b.x_range[0]))


def left_reference(curves, x=ORDER_X_M, ego_max_y=EGO_MAX_Y_M):
    """연속성 비교의 기준이 될 **자차 좌측 도색 경계**. 없으면 None.

    추적이 켜져 있으면 관성(coast)으로 들고 있는 경계가 여기 섞여 들어온다.
    그것이 곧 "도색이 사라진 직후에도 잠깐은 기준이 남는다" 는 뜻이고,
    트랙 신뢰도가 바닥나면 자연히 사라진다.
    """
    cand = [(c.y_at(x), c) for c in curves
            if c.cls != CLASS_GUIDE and not c.from_guide]
    cand = [(y, c) for y, c in cand if 0.0 < y <= ego_max_y]
    return min(cand, key=lambda t: t[0])[1] if cand else None


def apply_guide_link(curves, link, x=ORDER_X_M, ego_max_y=EGO_MAX_Y_M,
                     max_y=GUIDE_MAX_Y_M):
    """GuideLink 를 한 프레임 돌린다. -> (쓸 유도선 | None, 사유)"""
    return link.update(curves, left_reference(curves, x, ego_max_y), x, max_y)


class GuideLink:
    """좌측 도색에 이어지는 유도선을 **기억**한다 (프레임을 넘는 상태).

    호출부가 들고 있다가 `s12.assign_lane_ids(..., guide_link=link)` 로 넘긴다.
    `Tracker` 와 같은 이유로 상태를 갖는다 - 이전 프레임을 알아야 한다.

    **track_id 가 필요하므로 추적이 켜져 있어야 제구실을 한다.** 추적이 꺼져
    있으면 링크를 걸 수 없어 `None` 만 돌려준다 (그 프레임만 보고 판단하는
    폴백은 위 주석의 순환 문제 때문에 쓰지 않는다).
    """

    __slots__ = ("track_id", "since", "last_dy")

    def __init__(self):
        self.track_id = None
        self.since = 0          # 링크가 유지된 프레임 수
        self.last_dy = None     # 마지막으로 잰 이음매 어긋남 (진단용)

    def update(self, curves, left_ref, x=ORDER_X_M, max_y=GUIDE_MAX_Y_M,
               join_y=GUIDE_JOIN_MAX_Y_M, join_slope=GUIDE_JOIN_MAX_SLOPE,
               maneuver=None):
        """-> (쓸 유도선 Curve | None, 사유 문자열 | None)"""
        guides = [c for c in curves if c.cls == CLASS_GUIDE]

        if left_ref is not None:
            # 기준선이 보인다. 링크를 다시 건다 (도색을 쓰므로 출력은 없음)
            g, why = pick_guide(curves, left_ref, x, max_y, join_y, join_slope,
                                maneuver)
            new_id = g.track_id if (g is not None and g.track_id) else None
            if new_id and new_id == self.track_id:
                self.since += 1
            else:
                self.since = 1 if new_id else 0
            self.track_id = new_id
            self.last_dy = why.get("dy")
            return None, "도색 사용중"

        # 기준선이 사라졌다. 기억해 둔 트랙만 쓴다.
        if self.track_id is None:
            return None, "링크 없음"
        g = next((c for c in guides if c.track_id == self.track_id), None)
        if g is None:
            self.track_id, self.since = None, 0
            return None, "링크 트랙 소멸"
        self.since += 1
        return g, None


def guide_branch(g, left_ref, x_far=GUIDE_BRANCH_X_M):
    """기준선 연장 대비 유도선이 먼 쪽에서 얼마나 벗어나는가. -> (이탈 m, 종류)

    +면 좌측으로 갈라진 것(좌회전), 0 근처면 직진, -면 우측이다.
    """
    xf = min(x_far, g.x_range[1])
    if xf <= g.x_range[0]:
        return 0.0, "straight"
    d = g.y_at(xf) - left_ref.y_at(xf)       # 기준선은 2차식이라 외삽이 자연스럽다
    d0 = g.y_at(_join_x(left_ref, g)) - left_ref.y_at(_join_x(left_ref, g))
    dev = d - d0                              # 이음매에서의 차이를 뺀 순수 갈라짐
    if dev > GUIDE_BRANCH_M:
        return dev, "left"
    if dev < -GUIDE_BRANCH_M:
        return dev, "right"
    return dev, "straight"


def pick_guide(curves, left_ref=None, x=ORDER_X_M, max_y=GUIDE_MAX_Y_M,
               join_y=GUIDE_JOIN_MAX_Y_M, join_slope=GUIDE_JOIN_MAX_SLOPE,
               maneuver=None):
    """**자차 좌측 경계의 연장선인** 유도선. 없으면 None. -> (Curve|None, 탈락사유들)

    "왼쪽에 있는 것" 이 아니라 "이어지는 것" 을 고른다 (위 주석). 기준선이
    없으면 판단할 근거가 없으므로 쓰지 않는다.
    """
    guides = [c for c in curves if c.cls == CLASS_GUIDE]
    if left_ref is None:
        return None, {"reason": "기준 좌측 경계 없음", "n": len(guides)}

    # 이 맵에서 필요한 유도선은 **직진과 좌회전 둘뿐**이다. 우측으로 갈라진
    # 것은 쓰지 않는다.
    want = {"straight", "left"} if maneuver is None else {maneuver}
    best, bestd, why, kinds = None, None, [], {}
    for g in guides:
        y7 = g.y_at(x)
        if abs(y7) > max_y:
            why.append((g, "너무 멂"))
            continue
        xj = _join_x(left_ref, g)
        dy = abs(left_ref.y_at(xj) - g.y_at(xj))
        dth = abs(_slope(left_ref, xj) - _slope(g, xj))
        if dy > join_y:
            why.append((g, f"어긋남 {dy:.2f}m"))
            continue
        if dth > join_slope:
            why.append((g, f"기울기차 {dth:.2f}"))
            continue
        dev, kind = guide_branch(g, left_ref)
        kinds[id(g)] = (kind, dev)
        if kind not in want:
            why.append((g, f"{kind} 갈래 ({dev:+.1f}m)"))
            continue
        if bestd is None or dy < bestd:
            best, bestd = g, dy
    return best, {"reason": None if best else "연속성/갈래 통과 없음",
                  "n": len(guides), "dy": bestd, "rejected": why,
                  "kind": kinds.get(id(best), (None, None))[0]}


def assign_lane_ids(curves, x=ORDER_X_M, ego_max_y=EGO_MAX_Y_M, y_sanity=Y_SANITY_M,
          guide_fallback=GUIDE_FALLBACK, guide_max_y=GUIDE_MAX_Y_M,
          guide_link=None):
    """[Curve] -> (lane_id 가 붙은 [Curve], 통계)

    반환 리스트는 **왼쪽부터 오른쪽 순**이다. 입력 객체를 그대로 고쳐서
    돌려주므로 `LaneResult.curves` 와 `lanes` 가 같은 객체를 가리킨다.
    """
    lanes = [c for c in curves if c.cls != CLASS_GUIDE]
    stats = {"curves": len(curves), "lanes": len(lanes), "order_x": float(x),
             "diverged": 0, "left": 0, "right": 0,
             "ego_left": False, "ego_right": False, "left_from_guide": False,
             "guide_reject": None, "guide_link": None, "off_axis": 0,
             "straddling": False}
    for c in curves:
        c.lane_id = 0
        c.from_guide = False
    if not lanes:
        return _guide_only(curves, stats, x, guide_fallback, guide_max_y,
                           guide_link)

    scored = []
    for c in lanes:
        y = order_y(c, x, y_sanity)
        if abs(c.y_at(x)) > y_sanity:
            stats["diverged"] += 1
        # 자차와 크게 어긋난 방향의 선은 순번에서 뺀다 (위 주석)
        if abs(_slope(c, x)) > EGO_MAX_SLOPE:
            stats["off_axis"] += 1
            continue
        scored.append((y, c))

    # **밟고 있는 선을 먼저 빼낸다.** 좌우 어느 쪽으로도 세지 않는다.
    straddle = [s for s in scored if abs(s[0]) <= STRADDLE_Y_M]
    if straddle:
        # 여럿이면 가장 가운데 것 하나만 0 으로 본다
        _, c0 = min(straddle, key=lambda s: abs(s[0]))
        c0.lane_id = 0
        stats["straddling"] = True
        scored = [s for s in scored if s[1] is not c0]

    left = sorted([s for s in scored if s[0] >= 0], key=lambda s: s[0])
    right = sorted([s for s in scored if s[0] < 0], key=lambda s: -s[0])

    for side, group, key in ((+1, left, "left"), (-1, right, "right")):
        if not group:
            continue
        # 가장 안쪽이 한 차로 폭 넘게 떨어져 있으면 +-1 을 비우고 +-2 부터.
        # 단 **밟고 있는 선이 있으면 그 바로 바깥이 +-1 이다** - 이때는
        # 거리로 재는 검사가 의미 없다 (차로 하나를 건너뛴 것이 아니다).
        if stats["straddling"]:
            start = 1
        else:
            start = 1 if abs(group[0][0]) <= ego_max_y else 2
        for i, (_, c) in enumerate(group):
            c.lane_id = side * (start + i)
        stats[key] = len(group)
        stats["ego_" + key] = (start == 1)

    out = [c for _, c in left][::-1] + [c for _, c in right]
    if stats["straddling"]:
        # 0 번은 좌우 사이에 놓는다 (왼쪽부터 오른쪽 순서 유지)
        c0 = next(c for c in lanes if c.lane_id == 0)
        n_left = len(left)
        out = out[:n_left] + [c0] + out[n_left:]

    # **도색이 하나도 자차 좌측을 못 채웠을 때만** 유도선을 올린다
    if guide_fallback and guide_link is not None:
        # 링크는 **매 프레임** 돌려야 한다. 도색이 보이는 동안 갱신해 두는
        # 것이 이 구조의 핵심이라, ego_left 가 있을 때도 건너뛰면 안 된다.
        g, why = apply_guide_link(curves, guide_link, x, ego_max_y, guide_max_y)
        stats["guide_reject"] = why
        stats["guide_link"] = guide_link.track_id
        if g is not None and not stats["ego_left"]:
            # 도색이 +1 을 못 받았으므로 +1 은 비어 있다. 다만 +2 이상이
            # 이미 있으면 유도선이 그보다 안쪽이어야 말이 된다.
            inner = min((c.y_at(x) for _, c in left), default=None)
            if inner is None or g.y_at(x) < inner:
                g.lane_id = 1
                g.from_guide = True
                stats["ego_left"] = True
                stats["left_from_guide"] = True
                out = [g] + out
    return out, stats


def _guide_only(curves, stats, x, guide_fallback, guide_max_y, guide_link=None):
    """차선 도색이 하나도 없는 프레임. 유도선만이라도 건진다."""
    if guide_fallback and guide_link is not None:
        g, why = apply_guide_link(curves, guide_link, x, max_y=guide_max_y)
        stats["guide_reject"] = why
        if g is not None:
            g.lane_id = 1
            g.from_guide = True
            stats["ego_left"] = True
            stats["left_from_guide"] = True
            stats["left"] = 1
            return [g], stats
    return [], stats


def format_lane_id_stats(stats, names=None):
    names = names or CLASS_NAMES
    return [f"order_x {stats['order_x']:4.1f}m  곡선 {stats['curves']:2d}  "
            f"차선 {stats['lanes']:2d}  발산 {stats['diverged']:2d}  "
            f"{'[0]' if stats.get('straddling') else '   '}  "
            f"좌 {stats['left']:2d}"
            f"{'G' if stats.get('left_from_guide') else ('*' if stats['ego_left'] else ' ')}  "
            f"우 {stats['right']:2d}{'*' if stats['ego_right'] else ' '}"
            f"   (* = +-1 자차 경계 있음)"]

# ==========================================================================
# 정지선 (12단계 번호 밖의 별도 가지)
# ==========================================================================
# 정지선 — **12단계 번호 밖의 별도 가지**다.
#
#     입력   정리된 클래스 맵 (s02 출력) + 카메라
#     출력   StopLine | None
#
#     s04(Calibration) 만 차선 경로와 공유하고, s03/s06/s09/s10/s12 는 타지 않는다.
#
# ===========================================================================
# 왜 차선 경로를 그대로 쓰면 안 되는가
# ===========================================================================
# `s03` 은 **행(row)마다 가로 런의 중점**을 뽑는다. 차선은 이미지에서 세로로
# 서 있으므로 행을 자르면 폭 방향이 잘리고, 그 중점이 곧 중심선 위의 점이 된다.
#
# 정지선은 **가로로 누워 있다.** 행을 자르면 런 하나가 정지선 전체 길이를 덮고,
# 그 중점은 "정지선의 좌우 중앙" 한 점이 된다. 중심선이 아니라 엉뚱한 점이다.
# 같은 이유로 행 방향으로 폭을 재는 것도 틀린다.
#
# 그래서 여기서는 **런 중점이라는 중간 단계를 아예 두지 않는다.** 픽셀을 전부
# 지면으로 내리고, 그 점구름에 직선을 맞춘다. 정지선 두께(실제 0.3~0.45m)는
# 적합 잔차로 흡수되고, 우리가 필요한 것은 거리 하나다.
#
# ---------------------------------------------------------------------------
# 축을 바꿔서 x = a*y + b 로 맞춘다
# ---------------------------------------------------------------------------
# 차선은 `y = f(x)` 다. 정지선에 같은 것을 쓰면 진행방향을 가로지르는 선이라
# 기울기가 무한대로 발산한다. 축만 바꾸면 `s09.ransac_fit` 을 그대로 쓸 수 있고,
# `b` 가 곧 **자차 정면(y=0)까지의 거리**가 된다.
#
# ---------------------------------------------------------------------------
# RANSAC 직선이 덩어리 중앙점보다 낫다 (실측으로 확인)
# ---------------------------------------------------------------------------
# "덩어리의 중앙점을 쓰면 되지 않나" 를 같은 녹화본으로 비교했다. 검증 기준은
# **거리가 프레임을 넘어 매끄럽게 줄어드는가** 다 - 차가 다가가는 중이므로
# 정답을 몰라도 검증이 된다.
#
#                         |d거리| p50   p90    max      줄어든 비율
#     RANSAC 직선 x(y=0)      0.60   1.18    2.61 m       86.3%
#     덩어리 중앙값            0.54   1.26   24.64 m       77.3%
#
# **갈리는 것은 중앙값이 아니라 최대 튐이다.** 중앙점은 덩어리 안 모든 점에
# 끌려가므로, 정지선이 부분적으로만 보이거나 옆 것이 섞이면 통째로 옮겨간다
# (최대 24m). RANSAC 은 가장 많은 점이 동의하는 직선을 고르고 그 직선의 y=0
# 값을 읽으므로, **보이지 않는 정면 부분을 옆에서 본 부분으로 외삽**한다.
# 그것이 정확히 필요한 동작이다.
#
# ---------------------------------------------------------------------------
# 정면을 덮지 않는 검출은 **내보내지 않는다**
# ---------------------------------------------------------------------------
# 실측: 정지선 후보가 잡힌 210프레임 중 관측이 **자차 정면(y=0)을 실제로 덮은
# 것은 62프레임(30%)** 뿐이다. 나머지는 옆에서 본 조각이다.
#
# 처음에는 둘 다 내보내되 `covers_front` 로 구분하는 쪽으로 넣었다 (유도선에
# `from_guide` 를 붙인 것과 같은 생각). 검출 164/300 중 111개가 외삽이었다.
# 그런데 **화면으로 확인해 보니 외삽된 것은 쓸 수 없었다** - 옆에서 본 조각을
# 정면까지 늘린 선이 실제 정지선 위치와 맞지 않는 경우가 많았다.
#
# 그래서 지금은 정면을 덮은 것만 내보낸다 (`REQUIRE_FRONT`). 검출률은 떨어지지만
# **제어에 틀린 정지선 거리를 주는 것보다 없다고 하는 편이 낫다** - 정지선은
# "거기서 멈춘다" 는 결정에 직접 쓰이는 값이라 틀리면 대가가 크다.
#
# `covers_front` / `extrap_m` 자체는 남겨 둔다. 판정 근거를 버리면 나중에
# 왜 안 나왔는지 볼 수 없다.
#
# ---------------------------------------------------------------------------
# 횡단보도는 아직 다루지 않는다
# ---------------------------------------------------------------------------
# 횡단보도 줄무늬는 정지선과 기하가 같다 - 진행방향에 수직인 흰 띠. 모델에
# 횡단보도 클래스가 없어서(6클래스) 정지선으로 분류될 수 있다.
#
# 구분 단서는 **개수**다. 정지선은 한 줄, 횡단보도는 여러 줄이 나란히 있으므로
# 역투영한 점의 x 분포에 덩어리가 여럿 생긴다. 실측: 210프레임 중 덩어리가
# 1개인 것 135, 2개 43, 3개 이상 32 (36%가 여러 덩어리).
#
# 지금은 **가장 가까운 덩어리만 쓰고 개수를 `n_blobs` 에 남기는 것까지**만 한다.
# 분류는 나중에 붙인다.

MIN_PX = 30                 # 이보다 적으면 노이즈로 본다
BLOB_GAP_M = 1.0            # x 방향으로 이만큼 벌어지면 다른 덩어리
X_MIN, X_MAX = 0.5, 40.0    # 학습 범위. 그 너머는 배운 적이 없다
STOP_Y_ABS_MAX = 10.0            # 도로 밖

# **진행방향에 수직이어야 정지선이다.** 실측 |a|(=dx/dy) 의 p50 0.061,
# p90 0.274. 45도로 누운 선은 정지선이 아니라 다른 도색이다.
STOP_MAX_SLOPE = 0.30

STOP_RANSAC_THRESH_M = 0.25      # 정지선 두께(0.3~0.45m)의 절반쯤
STOP_RANSAC_MIN_RATIO = 0.40

# 관측 y 구간이 이 안을 덮으면 "정면을 봤다" 로 친다. 정확히 y=0 하나만
# 보지 않는 이유는, 차폭 안이면 사실상 정면을 본 것이기 때문이다.
FRONT_HALF_M = 0.5

# 정면을 덮지 않은 검출을 내보낼지. 기본은 **안 내보낸다** (위 주석 참고).
# 외삽값도 보고 싶으면 False 로 두면 되고, 그때 `covers_front` 로 구분된다.
REQUIRE_FRONT = True


def _blobs(x, gap=BLOB_GAP_M):
    """x 를 정렬해 gap 이상 벌어지는 곳에서 끊는다. -> [(lo, hi, n), ...]"""
    if x.size == 0:
        return []
    xs = np.sort(x)
    cuts = list(np.flatnonzero(np.diff(xs) > gap)) + [xs.size - 1]
    out, s0 = [], 0
    for c in cuts:
        seg = xs[s0:c + 1]
        if seg.size:
            out.append((float(seg.min()), float(seg.max()), int(seg.size)))
        s0 = c + 1
    return out


def detect_stopline(mask, cam, attitude=None, ground=None, rng=None,
          min_px=MIN_PX, max_slope=STOP_MAX_SLOPE, cls=CLASS_STOPLINE,
          require_front=REQUIRE_FRONT):
    """정리된 클래스 맵 -> (StopLine | None, 통계)

    `mask` 는 s02 출력을 준다 (보닛과 작은 덩어리가 이미 빠진 것).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    ground = ground or GroundPlane.from_attitude(*(attitude or (None, None)))
    stats = {"px": 0, "ground": 0, "blobs": 0, "sel": 0,
             "reason": None, "slope": None, "inlier": None,
             "covers_front": None, "extrap_m": None}

    vv, uu = np.nonzero(mask == cls)
    stats["px"] = int(vv.size)
    if vv.size < min_px:
        stats["reason"] = "픽셀 부족"
        return None, stats

    # **런 중점을 거치지 않고 픽셀을 그대로 내린다** (머리말 참고)
    uv = np.stack([uu.astype(np.float64), vv.astype(np.float64)], axis=1)
    xy, ok = unproject(cam, uv, ground)
    xy = xy[ok]
    keep = (xy[:, 0] > X_MIN) & (xy[:, 0] < X_MAX) & (np.abs(xy[:, 1]) < STOP_Y_ABS_MAX)
    xy = xy[keep]
    stats["ground"] = int(xy.shape[0])
    if xy.shape[0] < min_px:
        stats["reason"] = "지면점 부족"
        return None, stats

    X, Y = xy[:, 0], xy[:, 1]
    bl = _blobs(X)
    stats["blobs"] = len(bl)
    # 가장 가까운 덩어리. 여러 개면 횡단보도일 수 있으나 지금은 개수만 남긴다.
    lo, hi, _ = min(bl, key=lambda b: b[0])
    sel = (X >= lo) & (X <= hi)
    Xs, Ys = X[sel], Y[sel]
    stats["sel"] = int(Xs.size)
    if Xs.size < min_px:
        stats["reason"] = "덩어리 점 부족"
        return None, stats

    # **축을 바꿔서 맞춘다**: x = a*y + b  (머리말 참고)
    fit = ransac_fit(Ys, Xs, rng, degree=1, thresh_m=STOP_RANSAC_THRESH_M,
                         min_ratio=STOP_RANSAC_MIN_RATIO)
    if fit is None:
        stats["reason"] = "직선 적합 실패"
        return None, stats
    coef, inl = fit
    a, b = float(coef[0]), float(coef[1])
    stats["slope"], stats["inlier"] = a, float(inl.mean())

    if abs(a) > max_slope:
        stats["reason"] = f"수직 아님 |a|={abs(a):.2f}"
        return None, stats
    if not (X_MIN < b < X_MAX):
        stats["reason"] = f"거리 범위 밖 {b:.1f}m"
        return None, stats

    y_lo, y_hi = float(Ys.min()), float(Ys.max())
    covers = (y_lo <= FRONT_HALF_M) and (y_hi >= -FRONT_HALF_M)
    # 정면까지 얼마나 외삽했는가 (덮었으면 0)
    extrap = 0.0 if covers else float(min(abs(y_lo), abs(y_hi)))
    stats["covers_front"] = covers
    stats["extrap_m"] = extrap

    if require_front and not covers:
        # 옆에서 본 조각을 정면까지 늘린 값은 실제 위치와 안 맞는 경우가 많다
        stats["reason"] = f"정면 미포함 (외삽 {extrap:.1f}m)"
        return None, stats

    return StopLine(dist=b, coef=np.array([a, b]), y_range=(y_lo, y_hi),
                    x=Xs, y=Ys, inlier=inl, covers_front=covers,
                    extrap_m=extrap, n_blobs=len(bl)), stats


def format_stopline_stats(stats):
    if stats.get("reason"):
        return [f"정지선 없음 ({stats['reason']})  px {stats['px']}"]
    return [f"정지선  px {stats['px']}  지면 {stats['ground']}  "
            f"덩어리 {stats['blobs']}  선택 {stats['sel']}  "
            f"|a| {abs(stats['slope'] or 0):.3f}  인라이어 {stats['inlier'] or 0:.0%}"]
