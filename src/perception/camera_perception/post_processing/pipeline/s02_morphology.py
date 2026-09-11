"""2단계. Morphology — 마스크에서 **확실한 쓰레기만** 걷어낸다.

===========================================================================
이 단계가 하는 일은 두 개뿐이고, 그게 측정 결과다
===========================================================================
    1) 보닛 마스킹        클래스 픽셀의 68~92% 가 여기 있다
    2) 아주 작은 성분 제거  8px 미만 - 단일 픽셀 튐만

CLOSE 는 **기본으로 끈다.** 아래 "왜 CLOSE 를 끄는가" 참고.

---------------------------------------------------------------------------
왜 여기서 많이 거르지 않는가
---------------------------------------------------------------------------
이 단계는 **Calibration 앞**이다. 즉 아직 "이 덩어리가 몇 m 짜리인지" 모른다.
그런데 픽셀 면적은 원근에 완전히 오염된 지표다 (last_test 39장 실측, 보닛 제외):

    연결성분 면적 중앙값     0~10m   15~20m   25~30m   30~40m
    white_solid              3134      222       24       18
    white_dashed              662       36       15        9
    guide                     284       15       10       10

같은 차선이 거리에 따라 3134px -> 18px 이다. 여기서 면적으로 자르면 자르는
것이 "노이즈"가 아니라 "먼 곳"이 된다. 실제로 흔히 쓰는 40px 임계는 25m 너머
white_dashed 성분의 **100%**, 15m 너머 guide 의 **94%** 를 지운다. 면적 기준
으로는 4% 밖에 안 버리는 것처럼 보여서 눈치채기 어렵다.

**그래서 노이즈 제거 책임은 뒤로 넘긴다.** Ground 좌표(5단계) 이후로 가면
미터로 판단할 수 있고, Width consistency(8단계)는 애초에 그러라고 있는
단계다. 여기서는 어느 거리에서도 차선일 수 없는 것 - 단일 픽셀 튐 - 만 뺀다.

---------------------------------------------------------------------------
왜 CLOSE 를 끄는가
---------------------------------------------------------------------------
CLOSE 는 "같은 클래스 안의 구멍"을 메우는 연산이다. 그런데 메울 구멍이 없다.
last_test 39장, 보닛 제외 도로 영역의 성분 내부 구멍 개수:

    white_solid 8,  white_dashed 1,  yellow 1,  stopline 0,  guide 28

39장을 통틀어 그렇다. 실제로 CLOSE(3x3) 를 걸면 픽셀이 0.2% 늘고 성분이 25개
줄 뿐인데, **그 "성분 25개 감소"가 이득이 아니라 손해다** - 점선 대시가
서로 붙는다는 뜻이고, 점선/실선 구분은 대시가 끊겨 있다는 사실에 의존한다.

`lane_detection.py` 의 9x3 세로 커널을 그대로 가져오면 안 된다. 그건 **BEV
전용 값**이다. BEV 에서는 차선이 세로로 서 있어 세로 커널이 방향과 맞지만,
이미지 공간에서는 차선이 소실점으로 수렴해 방향이 위치마다 다르다. 여기에
걸면 픽셀 1.5%, 성분 -74개다.

그리고 실제 관측된 마스크 오류는 CLOSE 로 고쳐지는 종류가 아니었다
(000282 정지선, 확대 확인):

    "정지선을 다 못 채움"  -> 왼쪽 끝을 yellow 가 가져감. 구멍이 아니라 클래스 혼동
    "정지선이 튀어나옴"    -> 6468px 중 23.3% 가 아스팔트 위. 부족이 아니라 과검출
    "오른쪽 선이 침범"     -> 노면표시가 만나는 지점의 클래스 혼동

정지선 성분은 364개 열 중 **빈 열이 0개**였다. CLOSE 는 첫째를 못 고치고,
둘째를 **더 키우며**(팽창->침식이라 넘친 걸 더 넘치게 한다), 셋째도 못 고친다.

**그래도 인자로는 남겨 둔다.** 다른 녹화에서 실제로 구멍이 나오면 켜서 바로
비교할 수 있어야 한다. 끄는 것이 결론이지 금지가 아니다.
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 전경 클래스 전부(_common.FOREGROUND). 정지선도 여기서 같이 정리한다 - 차선
# 적합에는 안 쓰지만(3단계에서 빠진다) 정지선 자체를 쓸 데가 있고, 무엇보다
# 1->2 단계 사진에서 무엇이 지워졌는지 보려면 남아 있어야 한다.
from _common import CLASS_BG, CLASS_NAMES, FOREGROUND   # noqa: E402

# 어느 거리에서도 차선일 수 없는 크기. 30~40m 의 white_dashed 성분 중앙값이
# 9px 이므로 그보다 확실히 아래로 둔다.
MIN_BLOB_PX = 8

# 기본은 끔 (위 "왜 CLOSE 를 끄는가"). 켜려면 (3, 3) 처럼 튜플을 준다.
CLOSE_KERNEL = None


def apply(mask, bonnet=None, *, min_blob_px=MIN_BLOB_PX,
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


def format_stats(stats, names=None):
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
