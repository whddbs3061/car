"""3단계. Lane pixel extraction — 마스크 -> 중심선 점열 (이미지 좌표).

===========================================================================
이 단계가 하는 일은 하나다: **중심을 정확히 찾는다.**
===========================================================================
거르지 않고, 줄이지 않고, 합치지 않는다. 그 셋은 전부 뒤 단계의 일이다.

행마다 가로로 이어진 픽셀 덩어리(런)를 찾아 그 **중점** 하나를 남긴다.
점 하나는 `(u, v, w)` - 가로 위치(서브픽셀), 행, 그리고 그 런의 폭이다.

---------------------------------------------------------------------------
왜 행별 런 중점이 정당한가 - 선이 기울어져 있어도 맞는다
---------------------------------------------------------------------------
처음에는 "차선이 이미지에서 수평에 가까우면 행이 선을 가로지르는 게 아니라
선을 따라 잘라서 중점이 엉뚱해진다"고 걱정했다. 실측해 보니 실제로 차선
픽셀의 대부분이 수직이 아니었다 (성분 방향 |vy|, 면적 가중):

    클래스        수직 >0.7   기울 0.3~0.7   수평 <0.3
    white_solid      11.6%        54.0%        34.4%
    yellow            5.6%        68.7%        25.7%
    guide            36.5%        24.7%        38.8%

**그런데 걱정이 틀렸다.** 일정한 두께의 곧은 띠를 수평선으로 자르면, 그
현(chord)의 중점은 **각도와 무관하게 띠의 중심선 위에 있다.** 기울수록 현이
길어질 뿐 중점은 제자리다. 그래서 방향 적응(주축 PCA)이나 세선화로 갈 이유가
없다. (덧붙여 `cv2.ximgproc` 가 이 환경에 없어 thinning 은 쓸 수도 없다.)

중점이 어긋나는 것은 띠가 그 행 안에서 휠 때와 **서로 다른 두 선이 한 런으로
붙을 때**다. 후자는 여기서 판별할 수 없다 - 미터를 모르기 때문이다. 그래서
폭 `w` 를 들려 보내고 판단은 뒤로 넘긴다.

---------------------------------------------------------------------------
왜 두꺼운 런을 버리지 않는가
---------------------------------------------------------------------------
"런이 90px 넘으면 차선이 아니라 화살표/노면표시" 라는 가드를 흔히 두는데,
실측하면 그 가드가 자르는 것이 **원거리**다. 39장에서 90px 를 넘은 런 101개의
거리 분포:

    10m 이내 23%,  10~20m 29%,  **20m 너머 49%**
    (yellow 는 중앙값 26.9m, white_solid 는 16.6m)

먼 쪽 차선은 이미지에서 수평에 가까워지므로 런이 길어진다. 즉 픽셀 폭으로
자르면 "노면표시"가 아니라 "먼 곳"을 자르게 된다. 2단계에서 `MIN_BLOB` 을
키우지 않은 것과 **같은 이유이고 같은 함정**이다.

화살표나 합쳐진 두 선은 ground 좌표로 가면 명백하다 - 차로 폭 3.3m 와 맞지
않는다. 그래서 Width consistency 단계에서 미터로 거른다.

---------------------------------------------------------------------------
거리 가중은 여기서 해결되지 않는다
---------------------------------------------------------------------------
행별 런 중점으로 바꾸면 근거리 편중이 줄기는 한다. 다만 기대만큼은 아니다
(white_solid, 3~10m 가 차지하는 비중):

    픽셀 그대로 55.3%  ->  행별 런 중점 46.9%

이미지의 행 간격은 원근 때문에 거리에 비례하지 않으므로 당연하다. **자차
좌표로 간 뒤 x 축을 일정 간격으로 잘라 재샘플링**해야 고르게 된다. 그건
Lane boundary 추출 단계에서 한다.

---------------------------------------------------------------------------
정지선은 뽑지 않는다
---------------------------------------------------------------------------
정지선은 진행방향과 직각이라 `y = f(x)` 로 표현할 수 없다. 같은 통에 넣으면
차선 적합이 망가진다. 유도선(guide)은 차로 경계가 아니지만 **모양은 차선과
같아서** 같은 추출을 태우고, 차선과 섞지 않는 것은 뒤 단계에서 한다.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import CLASS_GUIDE, CLASS_NAMES, LANE_CLASSES   # noqa: E402

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


def apply(mask, occluded=None, classes=EXTRACT_CLASSES, max_run_px=MAX_RUN_PX,
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


def format_stats(stats, names=None):
    names = names or CLASS_NAMES
    lines = [f"{'class':12s} {'pixels':>9s} {'points':>8s} {'px/pt':>6s} "
             f"{'clipped':>8s} {'run w p50':>9s} {'p90':>5s} {'max':>5s}"]
    for c, s in stats.items():
        ratio = s["px"] / s["pts"] if s["pts"] else 0.0
        lines.append(f"{names[c]:12s} {s['px']:>9d} {s['pts']:>8d} "
                     f"{ratio:>6.1f} {s['clipped']:>8d} {s['w_p50']:>9.0f} "
                     f"{s['w_p90']:>5.0f} {s['w_max']:>5d}")
    return lines
