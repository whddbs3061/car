"""이미지 픽셀 -> 지면 평면 교점 -> 자차 좌표 (m).

`GenerateLabels.CameraModel` 은 자차 좌표 -> 픽셀(`project`)만 있다. 차선 검출은
그 반대가 필요하다 - 마스크에서 뽑은 픽셀이 지면 어디인지 알아야 한다.

**BEV 래스터를 거치지 않는다.** 마스크 전체를 워프해서 조감도 이미지를 만들면
(1) 워프 보간에서 얇은 차선이 깨지고 (2) 0.05m/px 격자에 갇히고 (3) 쓰지도 않는
배경 픽셀까지 800x400 을 채운다. 픽셀을 바로 지면으로 쏘면 그 셋 다 없다.

기하는 `CameraModel` 의 것을 그대로 뒤집는다 - fx/fy/cx/cy 와 장착 회전을
그 객체에서 읽어 쓰므로 `cam_set.json` 이나 `cropped()` 가 바뀌면 같이 따라간다.
여기서 초점거리나 주점을 다시 계산하지 않는 이유가 그것이다.

    project():    자차 -> 몸체축 -> 광학축 -> 픽셀
    이 파일:      픽셀 -> 광학축 광선 -> 몸체축 -> 자차축 -> 지면 평면과 교차

지면은 자차 좌표에서 z = ROAD_Z_EGO (-0.35m) 다. 자차 원점이 노면이 아니라
후륜축 중심이라 0 이 아니다 - 0 으로 두면 거리가 통째로 틀어진다.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAM = os.path.dirname(_HERE)                       # camera_perception/
for _p in (_HERE, _CAM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# **차체 자세 규약은 GenerateLabels 것을 그대로 쓴다.** 부호(EGO_PITCH_SIGN /
# EGO_ROLL_SIGN)는 MORAI 문서에 없어서 거기서 오버레이로 확정한 값이고,
# 학습 라벨이 그 규약으로 만들어졌다. 여기서 따로 정하면 추론이 라벨과
# 어긋난다.
from GenerateLabels import (EGO_PITCH_SIGN, EGO_ROLL_SIGN,  # noqa: E402
                            USE_EGO_ATTITUDE, rot_vehicle_to_world)


def _mount_rotation(cam):
    """장착 회전 Rm. `CameraModel.to_camera` 가 쓰는 것과 같은 행렬이다."""
    cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
    cp, sp = math.cos(cam.pitch), math.sin(cam.pitch)
    cr, sr = math.cos(cam.roll), math.sin(cam.roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def pixel_rays_ego(cam, uv):
    """픽셀 -> 자차 좌표계에서의 시선 방향 벡터 (정규화하지 않음).

    `to_camera` 는 자차점 p 를 q = p @ Rm 로 몸체축에 넣고
    광학축을 (-q_y, -q_z, q_x) 로 만든다. 그래서 역방향은
        몸체축  = (d_z, -d_x, -d_y)          (광학축 d 에서)
        자차축  = 몸체축 @ Rm.T              (p @ Rm 의 역)
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    d_opt = np.stack([(uv[:, 0] - cam.cx) / cam.fx,
                      (uv[:, 1] - cam.cy) / cam.fy,
                      np.ones(len(uv))], axis=1)
    d_body = np.stack([d_opt[:, 2], -d_opt[:, 0], -d_opt[:, 1]], axis=1)
    return d_body @ _mount_rotation(cam).T


def unproject_to_ground(cam, uv, z_ground, attitude=None):
    """픽셀 -> 지면 평면 위의 자차 좌표. (xy, valid) 를 돌려준다.

    `attitude` 에 `(pitch_deg, roll_deg)` 를 주면 **차체가 기운 만큼 지면을
    기울여** 푼다. 안 주면 차가 수평이라고 본다.

    왜 자세가 필요한가 - `GenerateLabels` 가 실측으로 정리해 둔 그대로다.
    지면 교점 거리는 d ~ h/theta 라 delta_d ~ -d^2*delta/h 이고, 10m 에서
    2도면 2.9m 가 틀어진다. 도로 경사·뱅크·서스펜션 때문에 **정지 중에도**
    0 이 아니다 (실측 pitch +0.68도). lap4_full 은 pitch -0.86~+1.62도,
    roll -1.9도까지 나온다.

    푸는 식: 지면은 월드에서 수평이므로 조건은 "차체 좌표 점 p 의 월드 높이가
    자차 원점보다 z_ground 만큼 아래" 다.

        (R @ p)_z = z_ground,   p(t) = mount_pos + t * d
        t = (z_ground - (R @ mount_pos)_z) / (R @ d)_z

    R 은 차량->월드 회전이고, 세 번째 행이 [-sp, cp*sr, cp*cr] 라 **yaw 는
    영향이 없다** (월드 수직을 중심으로 도는 회전이라 당연하다). 그래서
    yaw=0 으로 넣는다.

    `valid` 가 False 인 것은 **지평선 위이거나 카메라 뒤로 가는 광선**이다
    (아래를 향하지 않으면 지면과 만나지 않는다). 버리지 않고 표시로 돌려주는
    이유는, 마스크에 지평선 위 픽셀이 얼마나 섞였는지가 그 자체로 검출 품질의
    신호이기 때문이다.
    """
    d = pixel_rays_ego(cam, uv)

    if attitude is None or not USE_EGO_ATTITUDE:
        down_d = d[:, 2]
        down_c = cam.mount_pos[2]
    else:
        pitch, roll = attitude
        R = rot_vehicle_to_world(0.0, EGO_PITCH_SIGN * float(pitch),
                                 EGO_ROLL_SIGN * float(roll))
        up = R[2]                       # 월드 수직 성분을 뽑는 행
        down_d = d @ up
        down_c = float(cam.mount_pos @ up)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_ground - down_c) / down_d
    valid = (down_d < -1e-9) & np.isfinite(t) & (t > 0)
    t_safe = np.where(valid, t, 0.0)
    xy = cam.mount_pos[:2] + t_safe[:, None] * d[:, :2]
    return xy, valid
