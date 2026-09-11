"""4단계. Calibration -> Ground / Vehicle coordinates.

이미지 점 `(u, v)` 를 광선으로 쏘아 **지면과 만나는 자차 좌표 `(x, y)`** 를 얻는다.
BEV 래스터를 거치지 않는다 - 워프 보간에서 얇은 차선이 깨지지도, 0.05m/px 격자에
갇히지도, 쓰지 않는 배경까지 32만 픽셀을 채우지도 않는다.

===========================================================================
이 단계에 들어가는 세 가지
===========================================================================
    1) 카메라 내부/장착   cam_set.json 고정값. 여기서 튜닝하지 않는다.
    2) 차체 자세          프레임마다 meta.jsonl 의 pitch/roll
    3) 지면 모델          오늘은 평면. **교체 가능하게 객체로 둔다.**

---------------------------------------------------------------------------
왜 차체 자세가 지면 경사보다 중요한가 (실측)
---------------------------------------------------------------------------
지면 교점 거리는 d ~ h/theta 라 각도 오차가 거리로 증폭된다: delta_d/d = d*delta/h.
카메라는 노면 위 h=1.55m 다.

  **자세 1도 오차** -> 40m 에서 거리 45% 오차
  **노면 경사**     -> learning 12랩 실측, 전방 40m 고도차 p90 0.10m = 거리 6.5%

    전방      |dz| p50   p90    최대
     10m        0.01   0.04    0.19 m
     40m        0.03   0.10    0.26 m

즉 K-city 에서는 **자세가 노면 경사보다 7배 큰 오차원**이다. 그래서 오늘은
평면으로 두고 자세를 정확히 넣는다. 다른 코스라면 이 결론이 바뀔 수 있다.

`ROAD_Z_EGO = -0.35` 를 0 으로 두면 안 된다. 자차 원점이 노면이 아니라 후륜축
중심이고, 실측으로 노면보다 일정하게 0.35m 위다.

---------------------------------------------------------------------------
지면을 객체로 두는 이유
---------------------------------------------------------------------------
계획은 나중에 LiDAR 로 노면을 직접 재는 것이다. 그때 이 파일의 나머지가 바뀌면
안 된다. 그래서 지면은 `GroundPlane` 하나로 격리해 둔다.

    오늘      차체 자세로 기울인 평면            GroundPlane.from_attitude()
    다음      차로 폭 자기보정으로 잔여 pitch 보정  (같은 클래스, normal 만 갱신)
    나중      LiDAR 로 맞춘 평면/곡면            (같은 인터페이스)

**차로 폭 자기보정**이 LiDAR 없이 할 수 있는 다음 수순이다 - 차로가 3.3m 라는
것을 알면, 측정된 폭이 거리에 따라 변하는 정도가 곧 잔여 pitch 다. 지도도
라이다도 필요 없고 이미지만으로 관측된다.

---------------------------------------------------------------------------
왜 픽셀이 아니라 거리로 자르는가
---------------------------------------------------------------------------
지평선 근처는 1px 이 수십 m 다 (장착 pitch 2도 기준):

    거리   40m   60m   100m   무한
    행 v   104    95     88     78        <- 40m 부터 지평선까지 26px 뿐

그래서 "v 이상만 쓴다" 같은 픽셀 컷을 걸고 싶어지는데, **자세 보정을 켜면
지평선 행이 프레임마다 움직인다.** 고정 픽셀 컷은 프레임마다 다른 거리를 뜻하게
된다. 여기서는 이미 거리를 계산했으므로 `x <= X_MAX` 로 거는 것이 같은 일을
정확하게 하는 방법이다.
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import (CLASS_NAMES, EGO_PITCH_SIGN, EGO_ROLL_SIGN,  # noqa: E402
                     ROAD_Z_EGO, USE_EGO_ATTITUDE, rot_vehicle_to_world)

# 자차 좌표에서 쓸 범위. **거리 기준이다** (픽셀 아님).
X_MIN = 3.0             # 보닛에 가려 이보다 가까운 노면은 안 보인다
X_MAX = 40.0            # 학습 라벨의 MAX_RANGE 와 같다. 그 너머는 배운 적이 없다
Y_ABS_MAX = 10.0        # |y| 가 이보다 크면 도로 밖이다


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


def apply(pts, cam, attitude=None, ground=None,
          x_min=X_MIN, x_max=X_MAX, y_abs=Y_ABS_MAX):
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


def format_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'in':>7s} {'horizon':>8s} {'<Xmin':>6s} "
             f"{'>Xmax':>6s} {'|y|':>5s} {'kept':>7s} {'rate':>6s}"]
    for c, s in stats.items():
        r = s["kept"] / s["in"] * 100 if s["in"] else 0.0
        lines.append(f"{names[c]:12s} {s['in']:>7d} {s['horizon']:>8d} "
                     f"{s['near']:>6d} {s['far']:>6d} {s['side']:>5d} "
                     f"{s['kept']:>7d} {r:>5.1f}%")
    return lines
