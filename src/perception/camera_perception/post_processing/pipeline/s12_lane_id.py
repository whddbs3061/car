"""12단계. Lane ID — 곡선에 자차 기준 좌우 순번을 매긴다.

    입력   [Curve]      9단계 출력
    출력   lane_id 가 채워진 [Curve] (같은 객체를 고친다)

부호는 **y 부호를 따른다.** 자차 좌표계가 y 좌측이 + 이므로 왼쪽이 +1, +2 이고
오른쪽이 -1, -2 다. `_common.LaneResult.ego_left` 가 `by_lane_id(1)` 인 것과
같은 규약이다. (옛 `lane_detection.py` 는 왼쪽이 음수였다. 반대이니 옮겨 붙일 때
주의한다.)

---------------------------------------------------------------------------
**0 = 지금 밟고 있는 선**
---------------------------------------------------------------------------
차선 변경 중에는 경계 하나가 차 밑을 지나간다. 그 선의 y 는 +에서 -로 넘어가는데,
0 을 두지 않으면 그 순간 `+1` 과 `-1` 이 깜빡인다 - 같은 도색이 프레임마다
"내 왼쪽 경계" 였다가 "내 오른쪽 경계" 가 된다. 제어가 그 값을 그대로 먹으면
조향이 좌우로 떨린다.

|y| 가 차폭 절반 안이면 **그 선은 좌우 어느 쪽 경계도 아니다. 내가 올라타 있는
선이다.** 그래서 0 을 준다.

    차선 변경 중        0  = 밟고 있는 선
                       +1 = 그 왼쪽 선      (왼쪽 차로의 바깥 경계)
                       -1 = 그 오른쪽 선    (오른쪽 차로의 바깥 경계)

제어 입장에서 "지금 0번 선 위에 있고, 갈 수 있는 곳은 0~+1 사이 아니면
0~-1 사이" 로 읽힌다. 평상시에는 경계가 +-1.6m 쯤에 있으므로 0 이 나오지
않는다 - **차선 변경 중에만 나타나는 값**이다.

유도선(`CLASS_GUIDE`)은 **순번 매기기에는 들어가지 않는다.** 차로 경계가 아니라
진로 안내선이라 +-1 슬롯을 그냥 차지하면 차로 폭 판단이 무너진다.

다만 **도색이 없을 때만** 자차 경계 자리를 대신 채운다 (아래 참고).

---------------------------------------------------------------------------
도색이 없을 때 유도선으로 자차 좌측을 대신한다
---------------------------------------------------------------------------
교차로에는 차로 도색이 없고, 점선은 대시가 끊기며, 차선 변경 중에는 경계가
사라진다. 300프레임 주행 실측:

    자차 좌 없음    27프레임  ->  유도선 27/27 있고 전부 자차 왼쪽
    자차 우 없음    51프레임  ->  유도선 50/51 있고 전부 자차 왼쪽
    둘 다 없음      11프레임  ->  유도선 11/11 있고 전부 자차 왼쪽
    왼쪽 유도선 y(7m) 중앙값 +1.34m

즉 도색이 없는 구간에서도 **왼쪽 유도선은 거의 항상 있다.** 그것을 쓰지 않을
이유가 없다.

**그런데 그냥 넣으면 안 된다.** 두 가지를 지킨다.

  1) `from_guide=True` 를 반드시 표시한다. 출력의 `left`/`right` 는 제어에게
     "여기까지 비켜도 된다" 는 뜻인데, 유도선은 넘으면 안 되는 선이 아니라
     지나갈 길 힌트다. 표시가 없으면 회피 계획이 이것을 벽으로 오해한다.
     차로 폭 계산에서도 빼야 한다.

  2) **자차 좌측 경계의 연장선인 유도선만 쓴다.** 교차로에는 좌회전/직진 등
     여러 방향의 안내선이 같이 보인다. 측정된 왼쪽 유도선 y 범위가
     +0.23 ~ +5.29m 로 벌어져 있고, 그중 내 차로의 연장인 것은 하나뿐이다.

도색이 하나라도 있으면 그쪽이 이긴다. 유도선은 마지막 수단이다.

---------------------------------------------------------------------------
"왼쪽에 있는 유도선" 이 아니라 "좌측 경계에 이어지는 유도선" 이어야 한다
---------------------------------------------------------------------------
처음에는 "자차 왼쪽에서 가장 가까운 유도선" 을 골랐다. 그런데 **교차로에서
장애물을 피하느라 차가 옆으로 밀리면 원래 좌측이던 선이 좌측이 아니게 된다.**
그 순간 규칙이 엉뚱한 안내선을 집는다.

방향이 아니라 **연속성**으로 판단해야 한다. 실측(유도선 179개, 좌측 도색과의
이음매에서의 가로 어긋남):

    p10  0.06m      중앙  1.15m      p90  15.59m

한쪽 무리는 좌측 실선과 거의 같은 선 위에 있고(0.5m 이내가 43%), 나머지는
15m 씩 벌어진다. 뚜렷하게 갈린다.

**통과하는 것이 없으면 유도선을 아예 쓰지 않는다.** 회피 중이라 내 차로의
연장이 보이지 않는 상황이라면, 제어는 어차피 차선 중심이 아니라 회피 경로를
따라가는 중이므로 그 기준이 필요하지도 않다. 엉뚱한 안내선을 따라가느니
"없음" 이 낫다.

---------------------------------------------------------------------------
연속성은 **기준선이 보이는 동안** 판정해 둔다 (GuideLink)
---------------------------------------------------------------------------
연속성을 "도색이 사라진 그 프레임" 에 판정하려고 하면 순환에 빠진다. 유도선
폴백은 자차 좌측 도색이 **없을 때** 도는데, 비교할 기준선이 바로 그 없어진
도색이기 때문이다. 실제로 그렇게 짰더니 300프레임 전부 "기준 좌측 경계 없음"
으로 탈락했다.

순서를 뒤집으면 풀린다.

    좌측 도색이 보인다  ->  이어지는 유도선을 찾아 **track_id 를 기억**해 둔다
                            (이때는 도색을 쓰므로 유도선은 출력하지 않는다)
    좌측 도색이 사라졌다 ->  기억해 둔 track_id 의 유도선을 자차 좌측으로 쓴다
    그 트랙이 죽었다     ->  없음. 다른 유도선으로 대체하지 않는다

마지막 줄이 중요하다. 회피로 차가 옆으로 밀려 내 차로의 연장이 아니게 되면
링크가 끊기고, 그러면 **유도선을 안 쓴다.** 옆에 다른 안내선이 보여도 집지
않는다 - 그것은 내 차로의 연장이 아니다.

링크를 얼마나 오래 유지할지는 따로 정하지 않는다. 유도선도 10~11단계가
추적하므로, 그 트랙의 신뢰도가 바닥나 사라지면 링크도 자연히 끊긴다.

===========================================================================
왜 "같은 전방거리"에서 비교해야 하는가
===========================================================================
곡선마다 자기 시작점에서 y 를 재면 안 된다. 점선은 대시 위상 때문에 시작점이
프레임마다 5m 였다 15m 였다 하고, **커브에서는 그 차이가 곧 y 차이**라 순서가
뒤집힌다. 주행 중 ego_right 가 옆 차선으로 넘어가는 전형적인 원인이다.

그래서 한 전방거리를 정해 전부 거기서 잰다.

---------------------------------------------------------------------------
그 전방거리는 **가까워야 한다.** 외삽을 피하려다 순번이 뒤집힌다
---------------------------------------------------------------------------
외삽이 싫어서 "모든 곡선이 실제로 관측된 구간" 에서 고르는 방법을 먼저 넣었다
(`ORDER_X = clip(7, max(x_lo), min(x_hi))`). 외삽은 0 이 되지만 **틀린다.**

실측(40프레임): 그 방식은 `order_x` 를 13.6m 로 골랐고 자차 좌측을 **0/40
프레임**에서 찾았다. 급커브 구간이라 13.6m 앞에서는 차선이 이미 옆으로 쓸려가,
자차 왼쪽에 있던 황색선이 거기서는 y=-0.40 (정면)이 되어 우측으로 분류됐다.
같은 프레임을 7m 에서 재면

    +6.78   +3.00   -0.75   -4.60      간격 3.78 / 3.75 / 3.85m

로 차로 폭과 맞아떨어지고 자차 좌=황색, 우=백색점선이 정상으로 잡힌다.

lane_id 가 답하는 질문은 "**지금** 내 차로를 무엇이 끼고 있나" 다. 본질적으로
근거리 질문이라 먼 데서 재면 아무리 정확해도 다른 질문의 답이 된다. 2차식
외삽 오차는 근거리 몇 미터에서는 작고, 그 정도는 감수하는 것이 맞다.

발산만 막는다 - 외삽한 y 가 도로 밖(`Y_SANITY_M`)으로 날아가면 그 곡선만
관측 구간 끝으로 당겨서 잰다.

---------------------------------------------------------------------------
+-1 은 "가까워야" 받는다
---------------------------------------------------------------------------
가장 안쪽 곡선이라도 자차에서 한 차로 폭 넘게 떨어져 있으면 그것은 자차 차로
경계가 아니다. 반대쪽 차로의 경계이거나 검출이 하나 빠진 것이다. 그 경우
+-1 을 **비워 두고** +-2 부터 매긴다.

제어에 6m 밖 차선을 "자차 경계" 라고 주는 것보다 "없음" 이 낫다. 원 구현 실측:
이 검사를 넣기 전 ego 차선이 프레임 사이 4.5~4.9% 확률로 옆 차선으로 튀었고,
넣은 뒤 0% 가 됐다.

순번과 같은 거리(`ORDER_X_M`)에서 잰다. 위에서 그 거리를 근거리로 고정했기
때문에 따로 둘 이유가 없다 - 멀리서 재면 커브에서 멀쩡한 자차 경계가 한 차로
폭 밖으로 보여 전부 탈락한다.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import CLASS_GUIDE, CLASS_NAMES, LANE_WIDTH_M     # noqa: E402

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

    호출부가 들고 있다가 `s12.apply(..., guide_link=link)` 로 넘긴다.
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


def apply(curves, x=ORDER_X_M, ego_max_y=EGO_MAX_Y_M, y_sanity=Y_SANITY_M,
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


def format_stats(stats, names=None):
    names = names or CLASS_NAMES
    return [f"order_x {stats['order_x']:4.1f}m  곡선 {stats['curves']:2d}  "
            f"차선 {stats['lanes']:2d}  발산 {stats['diverged']:2d}  "
            f"{'[0]' if stats.get('straddling') else '   '}  "
            f"좌 {stats['left']:2d}"
            f"{'G' if stats.get('left_from_guide') else ('*' if stats['ego_left'] else ' ')}  "
            f"우 {stats['right']:2d}{'*' if stats['ego_right'] else ' '}"
            f"   (* = +-1 자차 경계 있음)"]
