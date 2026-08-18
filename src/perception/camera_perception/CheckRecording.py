"""녹화본 한 바퀴를 받아 학습에 쓸 만한지 한 번에 점검한다.

    python CheckRecording.py --recording <녹화폴더>

시험 주행 직후에 돌려서, 본 녹화를 계속해도 되는지 판단하는 용도다.
라벨 정합은 여기서 보지 않는다 — 그건 `GenerateLabels.py` 의 오버레이로 본다.

--------------------------------------------------------------------------
무엇을 왜 보는가
--------------------------------------------------------------------------
* **실효 프레임률** — mp4 의 명목 fps 와 다르다. drive6 는 명목 20fps 로 썼지만
  실제로는 9.0 fps 라 재생이 2배 빠르다. 실제 타이밍은 meta.jsonl 의 `t` 뿐이다.

* **프레임 간격의 꼬리** — 장애물 시나리오는 렉을 만든다. 중앙값이 멀쩡해도
  최댓값이 몇 초씩 벌어지면 그 구간은 통째로 비어 있다는 뜻이다
  (drive6 는 장애물 시나리오가 아닌데도 중앙 95ms / 최대 3408ms 이고,
  0.5초 넘게 끊긴 구간이 7회 합계 11.4초 = 주행 시간의 3.7% 다).

* **자차 위치 정체** — 카메라 프레임과 자차 상태를 **신선도 확인 없이** 짝짓기
  때문에, 상태가 늦으면 새 그림에 옛 위치로 만든 라벨이 붙는다. 위치가 직전과
  완전히 같으면 정체로 본다. drive6 도 17.3% 다.

* **거리 기준 샘플링 장수** — PNG 는 프레임 수가 아니라 **이동 거리**로 솎아야
  한다. 렉이 오면 fps 가 흔들려 프레임 기준 간격이 무너지지만, 거리 기준은
  그대로다. 정차 중에 같은 그림이 쌓이는 것도 자동으로 막힌다.

* **ObjectInfo 수신** — 없으면 가려짐을 처리할 수 없어 그 녹화본으로는 라벨을
  만들 수 없다. **녹화 때만 얻을 수 있으므로 여기서 반드시 확인한다.**
"""

import argparse
import json
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def load_meta(path):
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def sec(v):
    return f"{int(v // 60)}분 {v % 60:.0f}초" if v >= 60 else f"{v:.1f}초"


def main():
    ap = argparse.ArgumentParser(description="녹화본 한 바퀴 점검")
    ap.add_argument("--recording", required=True, help="meta.jsonl 이 있는 폴더")
    ap.add_argument("--sample-every", type=float, default=4.0,
                    help="거리 기준 샘플링 간격 (m). 기본 4")
    args = ap.parse_args()

    meta_path = os.path.join(args.recording, "meta.jsonl")
    if not os.path.isfile(meta_path):
        raise SystemExit(f"meta.jsonl 이 없습니다: {meta_path}")
    rows = load_meta(meta_path)
    n = len(rows)
    if n < 10:
        raise SystemExit(f"프레임이 너무 적습니다 ({n}줄).")

    t = np.array([r.get("t", 0.0) for r in rows])
    pos = np.array([[r.get("pos_x", 0.0), r.get("pos_y", 0.0)] for r in rows])
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    total = float(step.sum())
    dur = float(t[-1] - t[0])
    fps = n / dur if dur > 0 else 0.0

    print("=" * 62)
    print(f"{os.path.basename(os.path.abspath(args.recording))}  —  {n}프레임")
    print("=" * 62)
    print(f"  주행 시간   {sec(dur)}")
    print(f"  주행 거리   {total:.0f} m")
    print(f"  평균 속도   {total / dur * 3.6:.0f} km/h" if dur > 0 else "")
    print(f"  실효 fps    {fps:.1f}   (mp4 의 명목 fps 와 다를 수 있음)")

    # --- 렉 ---
    gap = np.diff(t)
    big = gap[gap > 0.5]
    print(f"\n[프레임 간격]  중앙 {np.median(gap) * 1000:.0f}ms  "
          f"95% {np.percentile(gap, 95) * 1000:.0f}ms  최대 {gap.max() * 1000:.0f}ms")
    if len(big):
        print(f"  0.5초 넘게 끊긴 구간 {len(big)}회, 합계 {big.sum():.1f}초 "
              f"({big.sum() / dur * 100:.1f}%)")
    verdict = "양호" if gap.max() < 1.0 else ("주의" if gap.max() < 3.0 else "심함")
    print(f"  → 렉 판정: {verdict}   (drive6 기준: 중앙 95ms / 최대 3408ms)")

    # --- 자차 위치 정체 ---
    same = np.array([bool(rows[i - 1].get("pos_x") == rows[i].get("pos_x")
                          and rows[i - 1].get("pos_y") == rows[i].get("pos_y")
                          and rows[i - 1].get("yaw") == rows[i].get("yaw"))
                     for i in range(1, n)])
    stale = int(same.sum())
    vel = np.array([abs(r.get("signed_vel", 0.0)) for r in rows[1:]])
    err = np.where(same & (vel > 1), vel / 3.6 * gap, 0.0)
    print(f"\n[자차 위치 정체]  {stale}/{n - 1} 프레임 ({stale / (n - 1) * 100:.1f}%)"
          f"   (drive6: 17.3%)")
    if err.max() > 0:
        e = err[err > 0]
        print(f"  그때 예상 위치 오차: 중앙 {np.median(e) * 100:.0f}cm  "
              f"최대 {e.max() * 100:.0f}cm")
    print("  → 이 프레임들은 학습에서 빼는 게 안전하다 (그림과 위치가 다른 시점)")

    # --- ObjectInfo ---
    has_obj = sum(1 for r in rows if r.get("objects"))
    print(f"\n[ObjectInfo]  {has_obj}/{n} 프레임에 objects 필드 있음")
    if has_obj == 0:
        print("  ⚠ 없음 — 가려짐을 처리할 수 없어 **이 녹화본으로는 라벨을 만들 수 없다.**")
        print("    RecordDrive 에 ObjectInfo 수신을 넣고 다시 녹화할 것.")
    else:
        cnt = [len(r["objects"]) for r in rows if r.get("objects")]
        print(f"  물체 수: 중앙 {int(np.median(cnt))}  최대 {max(cnt)}")

    # --- 거리 기준 샘플링 ---
    print(f"\n[거리 기준 샘플링]  프레임 간 이동: 중앙 {np.median(step) * 100:.0f}cm  "
          f"최대 {step.max() * 100:.0f}cm")
    for m in (3.0, args.sample_every, 5.0):
        k = int(total // m)
        print(f"  {m:.0f}m 마다 → {k:5d}장 / 1바퀴,  8바퀴면 {k * 8:6d}장")

    # --- 프레임 저장 상태 ---
    fdir = os.path.join(args.recording, "frames")
    imgs = sorted(f for f in os.listdir(fdir)) if os.path.isdir(fdir) else []
    mp4 = os.path.join(args.recording, "drive.mp4")
    print(f"\n[저장된 프레임]  frames/ {len(imgs)}장,  drive.mp4 "
          f"{'있음' if os.path.isfile(mp4) else '없음'}")
    if imgs and cv2 is not None:
        s = cv2.imread(os.path.join(fdir, imgs[0]))
        if s is not None:
            print(f"  해상도 {s.shape[1]}x{s.shape[0]}"
                  f"   (대회 카메라 원본은 1280x720)")
    elif os.path.isfile(mp4) and cv2 is not None:
        cap = cv2.VideoCapture(mp4)
        print(f"  mp4 해상도 {int(cap.get(3))}x{int(cap.get(4))}"
              f"   (축소 저장이면 1280x720 이 아니다)")
        cap.release()

    print("\n" + "=" * 62)
    print("다음: GenerateLabels.py 로 오버레이를 뽑아 라벨이 도색 위에 얹히는지 본다")
    print("=" * 62)


if __name__ == "__main__":
    main()
