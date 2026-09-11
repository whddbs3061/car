"""파이프라인 공통 - 상수 / 카메라 모델 / 체크포인트 / 결과 자료구조.

**이 폴더는 `lane_detection.py` 를 대체하는 것이 목적이다.** 그래서 그 파일을
import 하지 않고 필요한 것을 가져왔다. 대체하려는 대상을 import 하고 있으면
영원히 떼어낼 수 없다.

---------------------------------------------------------------------------
가져와도 안전한 이유 - 체크포인트가 자기 규격을 들고 있다
---------------------------------------------------------------------------
원래 `lane_detection.py` 가 학습 쪽(`seg_model` / `seg_dataset`)을 import 한
이유는 "학습에서 바꾼 입력 크기나 정규화가 추론에 조용히 반영되지 않는 것"을
막으려는 것이었다. 그 걱정은 타당하지만, `best.pt` 가 이미 다음을 들고 있다.

    input_size     [640, 256]
    class_names    background / white_solid / white_dashed / yellow / stopline / guide
    num_classes    6
    scheme         lane6
    args.backbone  resnet34

즉 **규격을 상수로 박지 않고 체크포인트에서 읽으면** 어긋날 수가 없다. 복사본이
되는 것은 모델 정의 하나뿐인데, 구조가 바뀌면 `load_state_dict` 가 키 불일치로
**즉시 터진다** - 조용히 틀리는 것이 아니라 시끄럽게 실패한다. 그래서 복사가
안전하다. 남는 위험은 ImageNet 정규화 상수뿐이고 그건 범용 상수다.

---------------------------------------------------------------------------
최종 형태
---------------------------------------------------------------------------
단계를 다 붙이면 이 파일이 합본의 **머리**가 되고 `s01`~`s12` 본문이 뒤에 붙어
파일 하나가 된다. 그래서 여기에는 단계 로직을 두지 않는다 - 상수와 자료구조만.

카메라 모델과 차체 자세 규약은 `GenerateLabels.py` 에서 **소스를 그대로 추출해**
넣었다 (손으로 옮기면 보닛 폴리곤 50쌍 같은 데서 오타가 난다). 학습 라벨이 이
규약으로 만들어졌으므로 여기서 바꾸면 추론이 라벨과 어긋난다.
"""

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

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

    def y_at(self, x):
        return float(np.polyval(self.coef, x))

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
_HERE = os.path.dirname(os.path.abspath(__file__))
_POST = os.path.dirname(_HERE)                      # post_processing/
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
