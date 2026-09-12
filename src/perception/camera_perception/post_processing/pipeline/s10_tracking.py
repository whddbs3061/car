"""10~11단계. Tracking — 프레임을 넘어 같은 차선을 같은 것으로 유지한다.

    입력   [Curve]     9단계 출력 (그 프레임만의 관측)
    출력   [Curve]     track_id / age / coasted 가 채워진 것

    10단계  Data association    어느 관측이 어느 트랙인가        헝가리안
    11단계  State estimation    그 트랙의 진짜 곡선은 무엇인가   칼만 필터

===========================================================================
**이 파일은 파이프라인에서 처음으로 상태를 갖는다.**
===========================================================================
s01~s09, s12 는 전부 "프레임 하나 -> 결과 하나" 인 순수 함수다. 추적은 본질상
그럴 수 없다 - 이전 프레임을 알아야 "같은 차선"을 말할 수 있다. 그래서 여기만
`Tracker` 객체를 쓰고, 호출부가 그것을 들고 있어야 한다.

상태를 갖는 대가로 **재현성이 프레임 순서에 묶인다.** 오프라인에서 어떤 프레임
하나만 다시 돌려도 그 앞을 똑같이 돌리지 않으면 결과가 다르다. 디버깅할 때
이걸 잊으면 "왜 아까랑 다르지" 로 한참 헤맨다.

===========================================================================
왜 헝가리안인가 - 탐욕 매칭은 순서에 좌우된다
===========================================================================
옛 구현은 관측을 하나씩 돌며 "아직 안 쓰인 트랙 중 가장 가까운 것"을 집었다.
그러면 **검출 리스트의 순서가 결과를 바꾼다.**

깨지는 장면이 구체적으로 있다. 점선 두 줄이 게이트(0.8m) 근처로 나란히 있을 때,
먼저 스캔된 관측이 남의 트랙을 먼저 채가고 뒤엣것은 짝이 없어 새 ID 를
발급받는다. 차선은 그대로인데 track_id 가 튄다.

헝가리안은 **전체 비용 합이 최소가 되는 짝짓기**를 고르므로 순서와 무관하고,
그 장면에서 둘 다 제 트랙을 찾는다.

---------------------------------------------------------------------------
scipy 를 쓰지 않고 직접 구현한 이유
---------------------------------------------------------------------------
`_common.py` 머리말이 정한 제약이 "외부 의존은 torch / cv2 / numpy 뿐" 이다.
차선 수는 클래스당 많아야 대여섯이라 행렬이 6x6 을 넘지 않는다. 그 크기에서는
O(n^3) 헝가리안이 수십 마이크로초라 라이브러리를 끌어올 이유가 없다.
`scipy.optimize.linear_sum_assignment` 와 무작위 행렬로 대조 검증했다.

===========================================================================
칼만 - 상태를 다항식 계수로 두지 않는다
===========================================================================
`y = a x^2 + b x + c` 의 `[a, b, c]` 를 그대로 필터링하고 싶어지는데 하면 안 된다.
셋의 크기가 `a ~ 1e-3`, `c ~ 1.7` 로 **세 자릿수 차이**나고 서로 강하게 상관돼
있어서, Q 와 R 을 어떤 값으로 줘도 한 성분이 나머지를 지배한다.

대신 **고정 전방거리에서의 y** 를 상태로 둔다.

    상태  x = [ y(7m), y(15m), y(25m) ]          전부 미터, 같은 스케일
    관측  z = 적합된 곡선을 같은 세 지점에서 평가     H = I (단위행렬)

세 점이면 2차식이 유일하게 결정되므로 정보 손실이 없다. H 가 단위행렬이라
칼만 식이 단순해지고, 무엇보다 **Q 와 R 을 미터로 생각할 수 있다** - "차선이
한 프레임에 몇 cm 움직일 수 있나", "이 적합은 몇 cm 쯤 틀렸나" 로.

---------------------------------------------------------------------------
R 을 적합 품질에서 만든다 - 이게 없으면 칼만은 지연만 더한다
---------------------------------------------------------------------------
모든 관측을 같은 정확도로 취급하면 필터는 그냥 이동평균이고, 얻는 것은 부드러움
뿐이고 잃는 것은 반응 속도다. 관측마다 얼마나 믿을지를 달리 줘야 이득이 난다.

여기서는 세 가지를 본다.

    1. 외삽 거리   knot 이 관측 구간 밖이면 그만큼 σ 를 키운다
    2. 인라이어    RANSAC 인라이어 비율이 낮으면 σ 를 키운다
    3. 점 수       점이 적을수록 σ 를 키운다

1번이 특히 중요하다. 점선은 대시 위상 때문에 관측 구간이 프레임마다 11~22m
였다 5~16m 였다 한다. 25m knot 이 3m 외삽인 프레임과 9m 외삽인 프레임을 같은
믿음으로 섞으면, **관측이 나빠진 프레임이 좋은 추정을 끌어내린다.**

---------------------------------------------------------------------------
자차 운동이 없으면 예측 모델이 반쪽이다 (알고 쓰는 한계)
---------------------------------------------------------------------------
차선 다항식은 **자차 좌표계**, 즉 차와 같이 움직이는 좌표계에 있다. 차가 1m
전진하고 조금 돌면 같은 차선이라도 y(7m) 값이 달라진다. 제대로 하려면 예측
단계에서 그 강체 변환을 태워야 한다.

지금 파이프라인에는 자차 속도/요레이트가 들어오지 않는다. 그래서 기본 동작은
**랜덤워크**(F = I, Q 만 키움)이고, 그 대가로 Q 를 크게 잡을 수밖에 없다.
Q 가 크면 필터가 관측을 많이 믿어 부드럽게 하는 힘이 약해진다.

    Q_RATE_M_PER_S 의 근거: 옛 BEV 경로 실주행 로그(298프레임, 약 13fps)에서
    연속 프레임 횡오차 변화가 중앙값 0.134m 였다. 0.134 / 0.077s = 1.74 m/s.
    여유를 둬 3.0 으로 잡았다. **이 값은 차가 움직이는 양이지 센서 잡음이
    아니다** - 그래서 자차 운동을 넣으면 이 항이 대부분 사라지고 Q 를 10배쯤
    줄일 수 있다. 이 파이프라인에서 다음으로 이득이 큰 작업이 그것이다.

`predict(dt, ego=(dx, dy, dpsi))` 로 넘기면 그 변환을 태운다. 인터페이스는
지금 열어 두되, 값이 없으면 랜덤워크로 돈다.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _common import (CLASS_GUIDE, CLASS_NAMES, CLASS_STOPLINE,  # noqa: E402
                     Curve)

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


def format_stats(stats, names=None):
    return [f"트랙 {stats['tracks_in']:2d} + 관측 {stats['curves']:2d}  ->  "
            f"매칭 {stats['matched']:2d}  신규 {stats['new']:2d}  "
            f"관성 {stats['coasted']:2d}  저신뢰 {stats['unconfirmed']:2d}  "
            f"junction {stats['junction']:.2f}  "
            f"버림 {stats['dropped']:2d}  출력 {stats['out']:2d}"]
