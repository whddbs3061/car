"""9단계. RANSAC / Poly fitting — 경계 점열 -> 곡선 계수.

    입력   [Boundary]     6단계 출력. 자차 좌표 (m)
    출력   [Curve]        y = c0 x^2 + c1 x + c2, 인라이어 표시까지

파일 번호가 s06 다음 s09 인 것은 오타가 아니다. `_common.LaneResult` 가 정한
단계 번호를 따른다 - 7~8 단계(차로 폭 / 자기보정)는 아직 없다. 폭은 제어 출력에
필요하지 않아 뒤로 미뤘고, s04 주석이 말하는 "차로 폭 자기보정"이 거기 들어간다.

===========================================================================
왜 RANSAC 인가 - 최소제곱은 한 점에 끌려간다
===========================================================================
6단계가 묶어 준 점이라도 섞인 것이 남는다. 옆 차선 도색 몇 점, 가드레일
그림자, 대시 사이를 건너뛰며 창이 넓어졌을 때 빨려 들어온 것들이다.

최소제곱은 이상치 하나에 곡선 전체가 기운다. 특히 **원거리 이상치가 치명적**인데,
2차식의 곡률 항이 먼 쪽 몇 점으로 결정되기 때문이다. 근거리에서 0.1m 틀리는
것과 40m 에서 3m 틀리는 것이 같은 잔차 합을 만든다.

RANSAC 은 "가장 많은 점이 동의하는 곡선"을 고르므로 소수 이상치가 결과를
바꾸지 못한다.

---------------------------------------------------------------------------
표본을 x 구간으로 나눠 뽑는다
---------------------------------------------------------------------------
무작위로 3점을 뽑으면 **점이 많은 근거리에서 셋 다 나온다.** 실측(40프레임):
백색실선 점의 밀도가 3~10m 에서 17.4점/m, 30~40m 에서 1.0점/m 이라 무작위
3점이 모두 20m 안에서 나올 확률이 높다. 짧은 구간에 맞춘 2차식은 먼 쪽으로
발산한다 - 곡률이 그 구간 밖에서는 근거가 없는 값이다.

그래서 x 를 (차수+1) 칸으로 나누고 **칸마다 하나씩** 뽑는다. 뽑힌 3점이 항상
전 구간에 걸치므로 곡률이 관측된 범위 전체의 지지를 받는다.

---------------------------------------------------------------------------
적합 구간을 잘라야 하는 이유
---------------------------------------------------------------------------
2차식 하나로 급커브 전 구간을 덮을 수 없다. 원 구현의 실측이 그대로 유효하다 -
급커브 프레임에서 전 구간 인라이어 비율 0.41, 근 20m 만 쓰면 0.98.

여기서는 **자르되 버리지 않는다.** 적합은 가까운 FIT_MAX_SPAN_M 구간으로 하고,
`x_range` 는 그 구간으로 둔다. 먼 점은 `Curve.x/y` 에 남아 있어서 왜 잘렸는지
볼 수 있다. 제어가 쓰는 것은 어차피 30m 안쪽이다.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import CLASS_GUIDE, CLASS_NAMES, Curve      # noqa: E402

FIT_DEGREE = 2              # 자차 좌표에서는 2차면 충분하다 (BEV 때와 같은 결론)
FIT_MAX_SPAN_M = 22.0       # 적합에 쓸 전방 구간. 위 주석 참고

RANSAC_ITERS = 80           # 실측 기준. 100 이상에서 인라이어 수가 안 늘었다
RANSAC_THRESH_M = 0.20      # 이 안이면 인라이어. 차선 도색 폭(0.15~0.35m)의 절반쯤
RANSAC_MIN_RATIO = 0.40     # 인라이어가 이 비율 미만이면 적합 실패로 본다
MIN_PTS = (FIT_DEGREE + 1) * 4      # 이보다 적으면 RANSAC 이 의미 없다
MIN_SPAN_M = 3.0            # 세로로 이만큼은 걸쳐야 곡률을 말할 수 있다

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
               thresh_m=RANSAC_THRESH_M, min_ratio=RANSAC_MIN_RATIO):
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


def apply(boundaries, rng=None, max_span_m=FIT_MAX_SPAN_M, min_pts=MIN_PTS,
          min_span_m=MIN_SPAN_M, **fit_kw):
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


def format_stats(stats, names=None):
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
