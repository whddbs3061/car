"""6단계. Lane boundary — 지면점을 **차선 경계별로** 묶는다.

    입력   {cls: (x, y, w_m)}    4~5단계 출력. 자차 좌표 (m)
    출력   [Boundary]            한 경계에 속한 점들. 아직 곡선이 아니다

===========================================================================
왜 BEV 래스터 없이도 같은 일을 할 수 있는가
===========================================================================
옛 구현은 조감도 이미지를 만들어 **하단 히스토그램 + 슬라이딩 윈도우**로 묶었다.
그런데 그 조감도의 행은 처음부터 x 를, 열은 y 를 0.05m/px 로 잘라 놓은 것일
뿐이다. 즉 히스토그램도 윈도우도 **원래 미터 단위 연산**이었고, 래스터는 그것을
정수 격자에 억지로 끼운 중간 단계였다.

점을 바로 쓰면 격자가 사라진다. 얻는 것:

    - 0.05m 격자에 갇히지 않는다. s03 이 서브픽셀로 구한 중점이 그대로 산다
    - 워프 보간이 얇은 차선을 끊어 먹던 문제가 없다
    - 배경까지 800x400 을 채우지 않는다 (프레임당 점 300~500개가 전부다)

---------------------------------------------------------------------------
묶는 방법: 가까운 곳에서 씨를 찾고 멀리까지 따라간다
---------------------------------------------------------------------------
    1) 씨앗    가장 가까운 SEED_X_SPAN_M 구간에서 y 히스토그램 봉우리
    2) 행진    씨앗마다 x 를 STEP_M 씩 전진하며 창 안의 점을 거둔다
               (공백을 건넌 뒤에는 **방향이 이어지는지**까지 확인한다)
    3) 반복    아직 아무 경계에도 안 들어간 점으로 1~2 를 다시 한다
    4) 정리    같은 점을 많이 공유하는 경계는 하나로 본다

---------------------------------------------------------------------------
3) 반복이 없으면 약하거나 먼 선이 통째로 사라진다
---------------------------------------------------------------------------
씨앗을 한 번만 찾으면 두 가지로 진다. 300프레임 주행 실측에서 자차 우측 차선이
**57프레임(19%)** 동안 통째로 없었고, 원인이 이 둘이었다.

    f095  오른쪽 선의 점이 전부 11m 밖   -> 근거리 씨앗 구간에 하나도 없다
          (구간 안 85점이 전부 y>=0, 구간 밖에 y<0 점이 69개)

    f105  구간 안에 오른쪽 점이 30개 있는데도 씨가 안 생긴다
          SEED_MIN_RATIO 가 **가장 강한 봉우리 대비** 비율이라, 왼쪽
          84점짜리 봉우리에 눌려 탈락한다

둘 다 "한 번의 히스토그램이 전체를 대표한다" 는 가정이 깨진 경우다. 한 번
훑고 나서 **남은 점만으로 다시 훑으면** 두 경우가 같이 풀린다 - 강한 선이
빠진 뒤에는 약한 선이 그 패스의 최대 봉우리가 되고, 씨앗 구간도 남은 점
기준으로 다시 잡히므로 먼 선도 자기 구간을 갖는다.

비용은 거의 없다. 패스마다 점이 줄어들고 SEED_PASSES 로 상한을 둔다.

**클래스마다 따로 돈다.** 황색 중앙선과 백색 실선은 붙어 있어도 다른 경계다.
섞으면 멀리서 두 선이 만나는 지점에서 하나로 합쳐진다 (s02 가 클래스별로 도는
이유와 같다).

---------------------------------------------------------------------------
창을 왜 중앙값으로 옮기는가
---------------------------------------------------------------------------
창 안 점들의 **중앙값**으로 창 중심을 옮긴다. 평균이 아니다. 옆 차선 점이나
노이즈가 창 가장자리에 몇 개 들어오면 평균은 그쪽으로 끌려가고, 한 번 끌려간
창은 다음 창을 더 끌고 가서 **경계가 옆 차선으로 넘어간다**. 중앙값은 그
소수점에 흔들리지 않는다.

---------------------------------------------------------------------------
거리에 따라 점이 성기다는 것을 창 크기에 반영한다
---------------------------------------------------------------------------
원근 때문에 같은 1m 라도 근거리는 점이 촘촘하고 원거리는 성기다. 40프레임
실측(백색실선, 1m 당 점):

    x  3~10m   17.4        x 20~30m    1.1
    x 10~20m   11.7        x 30~40m    1.0

20m 에서 **10배**가 꺾인다. 그래서 창을 채우는 최소 점수를 크게 잡으면 먼 쪽이
통째로 끊기고, 작게 잡으면 가까운 쪽에서 노이즈가 창을 끌고 간다.

그래서 창의 성패를 **세 가지로 나눈다.**

    점 >= MIN_PTS   중심을 중앙값으로 옮긴다
    점 1개 이상     거두기는 하되 중심은 직전 기울기로만 민다
    점 0개          비었다. MISS_MAX_M 넘게 이어지면 거기서 끝

가운데 칸이 핵심이다. 이것이 없으면 1점/m 인 원거리에서 창마다 "실패"가 쌓여
**점을 찾고 있는데도** 6m 만에 끊긴다. 반대로 점 1개로 중심을 옮기게 두면 원거리
노이즈 하나가 경계를 통째로 끌고 간다. 거두되 끌려가지는 않는다.

점선의 대시 간격(약 3m)은 마지막 칸이 처리한다 - 대시 사이는 점이 0개다.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import (CLASS_GUIDE, CLASS_NAMES, CLASS_WHITE_DASHED,  # noqa: E402
                     Boundary)

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
MIN_PTS = 2                 # 창 중심을 갱신할 최소 점수

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
# MIN_SPAN_M 에 걸려 버려졌다 - 40프레임 전부 점선 검출 0 이었던 원인이다.
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
MIN_SPAN_M = 3.0            # 세로로 이만큼은 이어져야 경계로 인정한다


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
          min_pts=MIN_PTS, miss_max_m=MISS_MAX_M, drift_gain=DRIFT_GAIN,
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


def apply(ground, classes=None, min_pts_keep=MIN_PTS_KEEP,
          min_span_m=MIN_SPAN_M, **march_kw):
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


def format_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'pts':>6s} {'seeds':>6s} {'grown':>6s} "
             f"{'short':>6s} {'dup':>4s} {'kept':>5s} {'assigned':>9s} {'rate':>6s}"]
    for c, s in stats.items():
        r = s["assigned"] / s["pts"] * 100 if s["pts"] else 0.0
        lines.append(f"{names[c]:12s} {s['pts']:>6d} {s['seeds']:>6d} "
                     f"{s['grown']:>6d} {s['short']:>6d} {s['dup']:>4d} "
                     f"{s['kept']:>5d} {s['assigned']:>9d} {r:>5.1f}%")
    return lines
