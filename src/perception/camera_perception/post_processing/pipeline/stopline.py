"""정지선 — **12단계 번호 밖의 별도 가지**다.

    입력   정리된 클래스 맵 (s02 출력) + 카메라
    출력   StopLine | None

    s04(Calibration) 만 차선 경로와 공유하고, s03/s06/s09/s10/s12 는 타지 않는다.

===========================================================================
왜 차선 경로를 그대로 쓰면 안 되는가
===========================================================================
`s03` 은 **행(row)마다 가로 런의 중점**을 뽑는다. 차선은 이미지에서 세로로
서 있으므로 행을 자르면 폭 방향이 잘리고, 그 중점이 곧 중심선 위의 점이 된다.

정지선은 **가로로 누워 있다.** 행을 자르면 런 하나가 정지선 전체 길이를 덮고,
그 중점은 "정지선의 좌우 중앙" 한 점이 된다. 중심선이 아니라 엉뚱한 점이다.
같은 이유로 행 방향으로 폭을 재는 것도 틀린다.

그래서 여기서는 **런 중점이라는 중간 단계를 아예 두지 않는다.** 픽셀을 전부
지면으로 내리고, 그 점구름에 직선을 맞춘다. 정지선 두께(실제 0.3~0.45m)는
적합 잔차로 흡수되고, 우리가 필요한 것은 거리 하나다.

---------------------------------------------------------------------------
축을 바꿔서 x = a*y + b 로 맞춘다
---------------------------------------------------------------------------
차선은 `y = f(x)` 다. 정지선에 같은 것을 쓰면 진행방향을 가로지르는 선이라
기울기가 무한대로 발산한다. 축만 바꾸면 `s09.ransac_fit` 을 그대로 쓸 수 있고,
`b` 가 곧 **자차 정면(y=0)까지의 거리**가 된다.

---------------------------------------------------------------------------
RANSAC 직선이 덩어리 중앙점보다 낫다 (실측으로 확인)
---------------------------------------------------------------------------
"덩어리의 중앙점을 쓰면 되지 않나" 를 같은 녹화본으로 비교했다. 검증 기준은
**거리가 프레임을 넘어 매끄럽게 줄어드는가** 다 - 차가 다가가는 중이므로
정답을 몰라도 검증이 된다.

                        |d거리| p50   p90    max      줄어든 비율
    RANSAC 직선 x(y=0)      0.60   1.18    2.61 m       86.3%
    덩어리 중앙값            0.54   1.26   24.64 m       77.3%

**갈리는 것은 중앙값이 아니라 최대 튐이다.** 중앙점은 덩어리 안 모든 점에
끌려가므로, 정지선이 부분적으로만 보이거나 옆 것이 섞이면 통째로 옮겨간다
(최대 24m). RANSAC 은 가장 많은 점이 동의하는 직선을 고르고 그 직선의 y=0
값을 읽으므로, **보이지 않는 정면 부분을 옆에서 본 부분으로 외삽**한다.
그것이 정확히 필요한 동작이다.

---------------------------------------------------------------------------
정면을 덮지 않는 검출은 **내보내지 않는다**
---------------------------------------------------------------------------
실측: 정지선 후보가 잡힌 210프레임 중 관측이 **자차 정면(y=0)을 실제로 덮은
것은 62프레임(30%)** 뿐이다. 나머지는 옆에서 본 조각이다.

처음에는 둘 다 내보내되 `covers_front` 로 구분하는 쪽으로 넣었다 (유도선에
`from_guide` 를 붙인 것과 같은 생각). 검출 164/300 중 111개가 외삽이었다.
그런데 **화면으로 확인해 보니 외삽된 것은 쓸 수 없었다** - 옆에서 본 조각을
정면까지 늘린 선이 실제 정지선 위치와 맞지 않는 경우가 많았다.

그래서 지금은 정면을 덮은 것만 내보낸다 (`REQUIRE_FRONT`). 검출률은 떨어지지만
**제어에 틀린 정지선 거리를 주는 것보다 없다고 하는 편이 낫다** - 정지선은
"거기서 멈춘다" 는 결정에 직접 쓰이는 값이라 틀리면 대가가 크다.

`covers_front` / `extrap_m` 자체는 남겨 둔다. 판정 근거를 버리면 나중에
왜 안 나왔는지 볼 수 없다.

---------------------------------------------------------------------------
횡단보도는 아직 다루지 않는다
---------------------------------------------------------------------------
횡단보도 줄무늬는 정지선과 기하가 같다 - 진행방향에 수직인 흰 띠. 모델에
횡단보도 클래스가 없어서(6클래스) 정지선으로 분류될 수 있다.

구분 단서는 **개수**다. 정지선은 한 줄, 횡단보도는 여러 줄이 나란히 있으므로
역투영한 점의 x 분포에 덩어리가 여럿 생긴다. 실측: 210프레임 중 덩어리가
1개인 것 135, 2개 43, 3개 이상 32 (36%가 여러 덩어리).

지금은 **가장 가까운 덩어리만 쓰고 개수를 `n_blobs` 에 남기는 것까지**만 한다.
분류는 나중에 붙인다.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import CLASS_STOPLINE, StopLine       # noqa: E402
import s04_calibration as s04                      # noqa: E402
import s09_curve_fit as s09                        # noqa: E402

MIN_PX = 30                 # 이보다 적으면 노이즈로 본다
BLOB_GAP_M = 1.0            # x 방향으로 이만큼 벌어지면 다른 덩어리
X_MIN, X_MAX = 0.5, 40.0    # 학습 범위. 그 너머는 배운 적이 없다
Y_ABS_MAX = 10.0            # 도로 밖

# **진행방향에 수직이어야 정지선이다.** 실측 |a|(=dx/dy) 의 p50 0.061,
# p90 0.274. 45도로 누운 선은 정지선이 아니라 다른 도색이다.
MAX_SLOPE = 0.30

RANSAC_THRESH_M = 0.25      # 정지선 두께(0.3~0.45m)의 절반쯤
RANSAC_MIN_RATIO = 0.40

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


def apply(mask, cam, attitude=None, ground=None, rng=None,
          min_px=MIN_PX, max_slope=MAX_SLOPE, cls=CLASS_STOPLINE,
          require_front=REQUIRE_FRONT):
    """정리된 클래스 맵 -> (StopLine | None, 통계)

    `mask` 는 s02 출력을 준다 (보닛과 작은 덩어리가 이미 빠진 것).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    ground = ground or s04.GroundPlane.from_attitude(*(attitude or (None, None)))
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
    xy, ok = s04.unproject(cam, uv, ground)
    xy = xy[ok]
    keep = (xy[:, 0] > X_MIN) & (xy[:, 0] < X_MAX) & (np.abs(xy[:, 1]) < Y_ABS_MAX)
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
    fit = s09.ransac_fit(Ys, Xs, rng, degree=1, thresh_m=RANSAC_THRESH_M,
                         min_ratio=RANSAC_MIN_RATIO)
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


def format_stats(stats):
    if stats.get("reason"):
        return [f"정지선 없음 ({stats['reason']})  px {stats['px']}"]
    return [f"정지선  px {stats['px']}  지면 {stats['ground']}  "
            f"덩어리 {stats['blobs']}  선택 {stats['sel']}  "
            f"|a| {abs(stats['slope'] or 0):.3f}  인라이어 {stats['inlier'] or 0:.0%}"]
