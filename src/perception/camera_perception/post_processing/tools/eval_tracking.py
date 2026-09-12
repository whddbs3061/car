#!/usr/bin/env python3
"""[개발 도구] 추적을 껐을 때 / 헝가리안만 / 헝가리안+칼만 을 **같은 녹화본**으로 비교한다.

    python3 tools/eval_tracking.py --rec rec/drive01

===========================================================================
왜 세 가지를 따로 재는가
===========================================================================
10단계(연관)와 11단계(상태추정)는 다른 일을 한다. 둘을 한꺼번에 켜고 좋아졌다고
하면 **어느 쪽이 값을 했는지 모른다.** 나중에 한쪽이 문제를 일으켜도 어디를
봐야 할지 모르게 된다.

    off         프레임마다 독립. track_id 없음
    greedy      연관만, 옛 방식(관측 순서대로 가장 가까운 트랙)
    hungarian   연관만, 전역 최소비용. **greedy 와 이것만 다르다**
    kalman      헝가리안 + 상태추정

`off` 의 id전환은 **0 으로 나오지만 0 이 아니다** - track_id 자체가 없어서
셀 수가 없는 것이다. 연관의 효과는 greedy 와 hungarian 을 비교해야 보인다.

---------------------------------------------------------------------------
무엇을 재는가
---------------------------------------------------------------------------
화면으로는 0.1m 차이를 못 가린다. 제어가 실제로 겪는 것을 숫자로 잡는다.

    1) 자차 차선 존재율    +1 / -1 이 있는 프레임 비율, 가장 긴 공백
       -> 제어 입장에서 "값이 없는" 시간이 얼마나 되는지

    2) 프레임 간 y(7m) 변화  중앙값 / p90 / 최대
       -> 이것이 곧 조향 떨림이다. 옛 BEV 경로 실측이 중앙값 0.134m, 최대 3.25m

    3) track_id 전환       자차 차선이 끊기지 않고 이어지는 동안 id 가 바뀐 횟수
       -> 연관(10단계)이 하는 일의 직접 지표. off 모드에는 없다

    4) 자차 차로 폭        중앙값 / 표준편차
       -> 좌우가 같은 차로를 잡고 있는지. 폭이 출렁이면 한쪽이 옆 차선이다

**2) 와 4) 는 관측이 있는 프레임끼리만 비교한다.** coast 로 채운 프레임을 섞으면
"부드러워졌다" 가 당연해져서 비교가 무의미해진다.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LANE = os.path.dirname(_HERE)
for _p in (_LANE, os.path.join(_LANE, "pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s01_segmentation as s01          # noqa: E402
import s02_morphology as s02            # noqa: E402
import s03_lane_pixels as s03           # noqa: E402
import s04_calibration as s04           # noqa: E402
import s06_boundary as s06              # noqa: E402
import s09_curve_fit as s09             # noqa: E402
import s10_tracking as s10              # noqa: E402
import s12_lane_id as s12               # noqa: E402

EVAL_X = 7.0            # 지표를 재는 전방거리 (12단계 순번 거리와 같게 둔다)


def load_meta(rec):
    rows = []
    with open(os.path.join(rec, "meta.jsonl")) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def stage9_pass(rec, meta, seg, seed=0):
    """무거운 앞단(s01~s09)을 **한 번만** 돌려서 곡선을 캐시한다."""
    rng = np.random.default_rng(seed)
    out = []
    for i, m in enumerate(meta):
        frame = cv2.imread(os.path.join(rec, m["file"]))
        if frame is None:
            raise SystemExit(f"프레임을 못 읽었습니다: {m['file']}")
        mask, _ = seg.apply(frame)
        clean, st2 = s02.apply(mask, seg.bonnet)
        pts, _ = s03.apply(clean, occluded=seg.bonnet)
        gnd, _ = s04.apply(pts, seg.cam)
        bounds, _ = s06.apply(gnd)
        curves, _ = s09.apply(bounds, rng=rng)
        out.append((curves, {"s02": st2}))
        if (i + 1) % 50 == 0:
            print(f"  s01~s09  {i+1}/{len(meta)}")
    return out


def replay(curves_per_frame, meta, mode):
    """캐시된 곡선에 추적 계층만 갈아 끼워 재생한다. -> 프레임별 요약"""
    # **연관만 바꾼 짝을 만든다.** greedy 와 hungarian 은 kalman=False 로 같고
    # 다른 것은 연관 알고리즘뿐이다. 그래야 10단계의 기여만 분리된다.
    cfg = {"greedy":    dict(assoc="greedy",    kalman=False),
           "hungarian": dict(assoc="hungarian", kalman=False),
           "kalman":    dict(assoc="hungarian", kalman=True)}
    tracker = None
    if mode != "off":
        if mode not in cfg:
            raise SystemExit(f"모르는 모드: {mode}")
        tracker = s10.Tracker(**cfg[mode])

    rows = []
    prev_cap = None
    for (curves, ctx), m in zip(curves_per_frame, meta):
        cap = m.get("capture") or m.get("recv")
        dt = (cap - prev_cap) if prev_cap is not None else 0.1
        prev_cap = cap
        dt = float(np.clip(dt, 1e-3, 1.0))

        if tracker is None:
            used, coasted_ids = curves, set()
        else:
            used, _ = tracker.update(curves, dt=dt, context=ctx)
            coasted_ids = {c.track_id for c in used if c.coasted}

        lanes, _ = s12.apply(used)
        rec = {"dt": dt}
        for slot in (1, -1):
            c = next((x for x in lanes if x.lane_id == slot), None)
            rec[slot] = None if c is None else {
                "y": c.y_at(EVAL_X), "tid": c.track_id,
                "coasted": bool(c.track_id in coasted_ids)}
        rows.append(rec)
    return rows


def metrics(rows, mode):
    n = len(rows)
    out = {"mode": mode, "n": n}

    for slot, name in ((1, "left"), (-1, "right")):
        have = [r[slot] is not None for r in rows]
        out[f"{name}_rate"] = sum(have) / n
        # 가장 긴 공백
        gap = best = 0
        for h in have:
            gap = 0 if h else gap + 1
            best = max(best, gap)
        out[f"{name}_maxgap"] = best

        # 프레임 간 변화 - **관측된 프레임끼리만**
        d = []
        tid_sw = 0
        prev = None
        for r in rows:
            c = r[slot]
            if c is None or c.get("coasted"):
                prev = None
                continue
            if prev is not None:
                d.append(abs(c["y"] - prev["y"]))
                if c["tid"] and prev["tid"] and c["tid"] != prev["tid"]:
                    tid_sw += 1
            prev = c
        out[f"{name}_dy_p50"] = float(np.median(d)) if d else float("nan")
        out[f"{name}_dy_p90"] = float(np.percentile(d, 90)) if d else float("nan")
        out[f"{name}_dy_max"] = float(np.max(d)) if d else float("nan")
        out[f"{name}_switch"] = tid_sw

    both = [r for r in rows if r[1] is not None and r[-1] is not None]
    out["both_rate"] = len(both) / n
    w = [abs(r[1]["y"] - r[-1]["y"]) for r in both]
    out["width_p50"] = float(np.median(w)) if w else float("nan")
    out["width_std"] = float(np.std(w)) if w else float("nan")
    return out


def print_table(results):
    keys = [("left_rate", "좌+1 존재율", "{:.1%}"), ("right_rate", "우-1 존재율", "{:.1%}"),
            ("both_rate", "둘 다", "{:.1%}"),
            ("left_maxgap", "좌 최대공백(f)", "{:.0f}"),
            ("right_maxgap", "우 최대공백(f)", "{:.0f}"),
            ("left_dy_p50", "좌 dy p50(m)", "{:.3f}"),
            ("left_dy_p90", "좌 dy p90(m)", "{:.3f}"),
            ("left_dy_max", "좌 dy max(m)", "{:.3f}"),
            ("right_dy_p50", "우 dy p50(m)", "{:.3f}"),
            ("right_dy_p90", "우 dy p90(m)", "{:.3f}"),
            ("right_dy_max", "우 dy max(m)", "{:.3f}"),
            ("left_switch", "좌 id전환", "{:.0f}"),
            ("right_switch", "우 id전환", "{:.0f}"),
            ("width_p50", "차로폭 중앙(m)", "{:.3f}"),
            ("width_std", "차로폭 표준편차", "{:.3f}")]
    hdr = f"{'지표':<18}" + "".join(f"{r['mode']:>14}" for r in results)
    print(hdr)
    print("-" * len(hdr))
    for k, label, fmt in keys:
        line = f"{label:<18}"
        for r in results:
            v = r.get(k)
            line += f"{(fmt.format(v) if v == v else '-'):>14}"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="[개발용] 추적 3-way 비교")
    ap.add_argument("--rec", required=True, help="record_drive.py 로 만든 폴더")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--modes", default="off,greedy,hungarian,kalman")
    args = ap.parse_args(argv)

    meta = load_meta(args.rec)
    print(f"[eval] {args.rec}  {len(meta)}프레임")
    seg = s01.Segmenter(checkpoint=args.checkpoint, device=args.device)

    print("[eval] s01~s09 (한 번만)")
    curves = stage9_pass(args.rec, meta, seg)

    results = []
    for mode in args.modes.split(","):
        rows = replay(curves, meta, mode.strip())
        results.append(metrics(rows, mode.strip()))
    print()
    print_table(results)


if __name__ == "__main__":
    main()
