#!/usr/bin/env python3
"""단계별로 **사진을 뽑는** 검증 도구. 창도 재생도 없다.

===========================================================================
실행
===========================================================================
cd C:/MSC/AutoMobility/car/src/perception/camera_perception/post_processing/pipeline
C:/Users/user/anaconda3/envs/vision_env/python.exe -u run_stages.py

  일부만        --frames 000282,000315
  팽창 없이     --no-dilate      (마스크 두께를 있는 그대로 본다)
  비교          --close 3x3 --suffix _close3x3
  다른 녹화     --recording ../../recordings/lap4_full
===========================================================================

출력 - **단계마다 자기 폴더**에, 원본 해상도(1280) 그대로

    <recording>/s00_original/000282.png
    <recording>/s01_segmentation/000282.png
    <recording>/s02_morphology/000282.png

한 폴더 안에서 화살표로 넘기면 **같은 단계를 39장 연속으로** 볼 수 있다.
합성 패널로 묶으면 한 단계가 화면의 1/3 로 줄어 먼 쪽 차선이 안 보인다.

**이 파일은 실주행 경로가 아니다.** 실주행은 나중에 `lane_pipeline.py` 가
같은 단계 함수들을 시각화 없이 순서대로만 호출한다. 실시간 루프에 그리기가
끼면 프레임을 놓치고, 그러면 화면에서 본 지연이 실제 지연과 달라진다
(`PIPELINE.md` 의 live_output 규칙과 같은 이유).

단계를 새로 붙일 때 손대는 곳은 `stage_images()` 하나뿐이다. 폴더는 이름으로
자동 생성된다.

---------------------------------------------------------------------------
마스크 그림의 팽창(dilate)에 대하여
---------------------------------------------------------------------------
먼 쪽 차선은 두께가 1~2px 라 그대로 그리면 화면에서 안 보인다. 그래서 기본은
1px 부풀려 그린다. **다만 그러면 "마스크가 도색보다 두껍다"를 판단할 수 없다** -
실제로 000282 정지선에서 그 혼동이 있었다. 두께를 따질 때는 `--no-dilate` 로
보고, 제목에도 어느 쪽인지 항상 적는다.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_POST = os.path.dirname(_HERE)                      # 기본 녹화 경로용일 뿐이다
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)                       # sys.path 에는 이 폴더만

import s02_morphology as s02                         # noqa: E402
import s03_lane_pixels as s03                        # noqa: E402
import s04_calibration as s04                        # noqa: E402
from _common import (CLASS_GUIDE, CLASS_NAMES, CLASS_STOPLINE,  # noqa: E402
                     CLASS_WHITE_DASHED, CLASS_WHITE_SOLID, CLASS_YELLOW,
                     LANE_WIDTH_M)
from s01_segmentation import Segmenter               # noqa: E402

CLASS_COLORS = {
    CLASS_WHITE_SOLID: (255, 0, 255),      # magenta
    CLASS_WHITE_DASHED: (255, 255, 0),     # cyan
    CLASS_YELLOW: (0, 255, 0),             # green
    CLASS_STOPLINE: (0, 0, 255),           # red
    CLASS_GUIDE: (0, 165, 255),            # orange
}
LEGEND = "solid=magenta  dashed=cyan  yellow=green  stop=red  guide=orange"


def overlay(frame, mask, alpha=0.75, dilate=1):
    vis = frame.copy()
    color = np.zeros_like(vis)
    hit = np.zeros(vis.shape[:2], bool)
    k = np.ones((2 * dilate + 1,) * 2, np.uint8) if dilate else None
    for c, bgr in CLASS_COLORS.items():
        m = mask == c
        if not m.any():
            continue
        if k is not None:
            m = cv2.dilate(m.astype(np.uint8), k) > 0
        color[m] = bgr
        hit |= m
    vis[hit] = (vis[hit] * (1 - alpha) + color[hit] * alpha).astype(np.uint8)
    return vis


def framed(img, title, sub="", footer=()):
    """제목줄 + 그림 + 아래 설명줄. 그림은 **원본 해상도 그대로** 둔다."""
    w = img.shape[1]
    bar = np.zeros((32, w, 3), np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.66,
                (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        (tw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(bar, sub, (w - tw - 10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (150, 210, 150), 1, cv2.LINE_AA)
    out = [bar, img]
    if footer:
        fb = np.zeros((20 * len(footer) + 14, w, 3), np.uint8)
        for i, t in enumerate(footer):
            cv2.putText(fb, t, (10, 21 + i * 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (200, 200, 200), 1, cv2.LINE_AA)
        out.append(fb)
    return np.vstack(out)


def _counts(mask):
    n = np.bincount(mask.ravel(), minlength=len(CLASS_NAMES))
    return "  ".join(f"{CLASS_NAMES[c][:6]} {n[c]}" for c in CLASS_COLORS)


def draw_points(frame, pts, radius=1):
    """3단계 - 뽑힌 중심점을 어둡게 깐 원본 위에 찍는다.

    마스크를 같이 칠하지 않는다. 점만 남겨야 "선의 한가운데에 찍혔는가"를
    눈으로 볼 수 있다 - 마스크를 깔면 점이 그 안에 묻힌다.
    """
    vis = (frame * 0.30).astype(np.uint8)
    for c, (u, v, _w) in pts.items():
        col = CLASS_COLORS[c]
        for uu, vv in zip(u, v):
            cv2.circle(vis, (int(round(uu)), int(round(vv))), radius, col, -1)
    return vis


class GroundPlot:
    """자차 좌표 평면. **조감도 이미지가 아니라 미터 좌표 산점도다.**

    BEV 래스터는 마스크를 통째로 워프해 만든 그림이고, 이건 점을 미터 좌표에
    찍은 그래프다. 겉모습이 비슷해 헷갈리기 쉬우니 축 눈금을 항상 적어 둔다.
    """

    PPM = 18                        # px per meter

    def __init__(self, x_max=s04.X_MAX, y_abs=s04.Y_ABS_MAX):
        self.x_max, self.y_abs = x_max, y_abs
        self.w = int(2 * y_abs * self.PPM)
        self.h = int(x_max * self.PPM)
        self.img = np.zeros((self.h, self.w, 3), np.uint8)
        for xm in range(0, int(x_max) + 1, 5):
            r = self._r(xm)
            cv2.line(self.img, (0, r), (self.w, r), (46, 46, 46), 1)
            cv2.putText(self.img, f"{xm}m", (4, r - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.36, (110, 110, 110), 1, cv2.LINE_AA)
        for ym in range(-int(y_abs), int(y_abs) + 1, 2):
            cv2.line(self.img, (self._c(ym), 0), (self._c(ym), self.h),
                     (38, 38, 38), 1)
        # 자차 차로 폭 기준선 - 폭이 맞는지 눈으로 재는 자
        for ym in (-LANE_WIDTH_M / 2, LANE_WIDTH_M / 2):
            cv2.line(self.img, (self._c(ym), 0), (self._c(ym), self.h),
                     (72, 72, 0), 1)
        cv2.circle(self.img, (self._c(0), self._r(0)), 5, (0, 255, 255), -1)

    def _r(self, x):
        return int(round(self.h - x * self.PPM))

    def _c(self, y):
        return int(round(self.w / 2 - y * self.PPM))

    def scatter(self, x, y, color, radius=1):
        for xx, yy in zip(np.atleast_1d(x), np.atleast_1d(y)):
            r, c = self._r(xx), self._c(yy)
            if 0 <= r < self.h and 0 <= c < self.w:
                cv2.circle(self.img, (c, r), radius, color, -1)


def stage_images(seg, frame, dilate, close_kernel, min_blob, attitude=None):
    """구현된 단계를 전부 돌리고 ({폴더이름: 사진}, 통계) 를 낸다.

    **단계를 추가할 때 손대는 곳은 여기뿐이다.**
    """
    mask, crop = seg.apply(frame)                            # 1단계
    clean, m_stats = s02.apply(mask, seg.bonnet,             # 2단계
                               min_blob_px=min_blob,
                               close_kernel=close_kernel)
    # 보닛을 넘겨 거기에 잘린 런을 버린다 (s03 의 '잘린 런 버리기')
    pts, p_stats = s03.apply(clean, occluded=seg.bonnet)      # 3단계
    ground = s04.GroundPlane.from_attitude(*(attitude or (None, None)))
    gnd, g_stats = s04.apply(pts, seg.cam, ground=ground)     # 4단계

    tag = f"dilate={dilate}" if dilate else "NO dilation (true thickness)"
    n1 = int(np.bincount(mask.ravel(), minlength=6)[1:].sum())
    n2 = int(np.bincount(clean.ravel(), minlength=6)[1:].sum())
    n3 = sum(s["pts"] for s in p_stats.values())
    n4 = sum(s["kept"] for s in g_stats.values())
    att_txt = ("level (no attitude)" if attitude is None else
               f"pitch {attitude[0]:+.2f} roll {attitude[1]:+.2f} deg")

    gp = GroundPlot()
    for c, (x, y, _w) in gnd.items():
        gp.scatter(x, y, CLASS_COLORS[c])

    imgs = {
        "s00_original": framed(crop, "0. original"),
        "s01_segmentation": framed(
            overlay(crop, mask, dilate=dilate),
            "1. Segmentation (raw model output, bonnet NOT removed)",
            f"{n1}px  {tag}",
            [_counts(mask), LEGEND]),
        "s02_morphology": framed(
            overlay(crop, clean, dilate=dilate),
            "2. Morphology (bonnet mask + tiny blob)",
            f"{n2}px  {tag}",
            [f"close={close_kernel or 'OFF'}   min_blob={min_blob}px   "
             f"kept {n2}/{n1} = {n2 / max(n1, 1) * 100:.1f}%"]
            + s02.format_stats(m_stats)),
        "s03_lane_pixels": framed(
            draw_points(crop, pts),
            "3. Lane pixel extraction (row-run centers, bonnet-clipped runs dropped)",
            f"{n3} points   stopline excluded",
            [f"max_run={s03.MAX_RUN_PX or 'OFF'}   "
             f"{n2}px -> {n3}pt   {LEGEND}"]
            + s03.format_stats(p_stats)),
        "s04_ground": framed(
            gp.img, "4. Calibration -> ground (meters, NOT a BEV image)",
            f"{n4} pts   {att_txt}",
            [f"x {s04.X_MIN}~{s04.X_MAX}m   |y|<{s04.Y_ABS_MAX}m   "
             f"ground: plane z={ground.offset}m   {n3}pt -> {n4}pt"]
            + s04.format_stats(g_stats)),
    }
    return imgs, {"s02": m_stats, "s03": p_stats, "s04": g_stats}


def main(argv=None):
    ap = argparse.ArgumentParser(description="파이프라인 단계를 사진으로 뽑는다")
    ap.add_argument("--recording",
                    default=os.path.join(_POST, "..", "recordings", "last_test"))
    ap.add_argument("--frames", default=None, help="프레임 번호 (쉼표 구분)")
    ap.add_argument("--out", default=None,
                    help="단계 폴더를 만들 위치 (기본: 녹화 폴더 안)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-dilate", action="store_true",
                    help="마스크를 부풀리지 않고 그린다 (두께를 따질 때)")
    ap.add_argument("--close", default=None, metavar="HxW",
                    help="CLOSE 커널 (예: 3x3). 기본은 끔 - s02 주석 참고")
    ap.add_argument("--min-blob", type=int, default=s02.MIN_BLOB_PX)
    ap.add_argument("--meta", default=None,
                    help="차체 자세를 읽을 meta.jsonl (기본: 녹화 폴더 안)")
    ap.add_argument("--attitude", default="meta", choices=["meta", "none"],
                    help="pitch/roll 보정. none 이면 차가 수평이라고 본다")
    ap.add_argument("--suffix", default="",
                    help="폴더 이름 뒤에 붙인다. 설정을 바꿔 뽑은 것을 "
                         "따로 두고 비교할 때 (예: --suffix _close3x3)")
    args = ap.parse_args(argv)

    close_kernel = None
    if args.close:
        h, w = (int(v) for v in args.close.lower().split("x"))
        close_kernel = (h, w)

    paths = sorted(glob.glob(os.path.join(args.recording, "frames", "*.png"))) \
        or sorted(glob.glob(os.path.join(args.recording, "*.png")))
    if not paths:
        raise SystemExit(f"png 를 못 찾았습니다: {args.recording}")
    if args.frames:
        want = {t.strip() for t in args.frames.split(",") if t.strip()}
        paths = [p for p in paths if os.path.basename(p)[:-4] in want]
        if not paths:
            raise SystemExit(f"그 프레임을 못 찾았습니다: {sorted(want)}")

    seg = Segmenter(args.checkpoint, device=args.device)
    print(f"[pipe] epoch {seg.info['epoch']} ({seg.info['backbone']}, "
          f"{seg.info['num_classes']}클래스, 입력 {seg.info['input_size']})")
    print(f"[pipe] 카메라 {seg.cam.width}x{seg.cam.height} fx={seg.cam.fx:.0f} "
          f"cy={seg.cam.cy:.0f}  보닛 {int(seg.bonnet.sum())}px")
    print(f"[pipe] close={close_kernel or 'OFF'}  min_blob={args.min_blob}px"
          + (f"  suffix={args.suffix}" if args.suffix else ""))
    print(f"[pipe] {len(paths)}장")

    # **자세는 그 녹화 자신의 것이어야 한다.** 다른 주행의 pitch 를 먹이면
    # 없는 기울기를 보정하게 된다.
    att = {}
    if args.attitude == "meta":
        mp = args.meta or os.path.join(args.recording, "meta.jsonl")
        att = s04.load_attitude(mp)
        if att:
            print(f"[pipe] 자세 {len(att)}프레임: {mp}")
        else:
            print(f"[pipe] meta.jsonl 이 없어 수평 가정으로 돌린다: {mp}")
            print("[pipe]   -> 먼 쪽 거리가 자세 1도당 40m 에서 45% 틀어진다")

    root = args.out or args.recording
    total = {"s02": {}, "s03": {}, "s04": {}}
    made = set()
    failed = []
    skipped = 0
    for p in paths:
        idx = os.path.basename(p)[:-4]
        frame = cv2.imread(p)
        if frame is None:
            continue
        # **카메라 해상도와 다른 png 는 건너뛴다.** 녹화 폴더에 스크린샷이나
        # 진단 이미지가 섞이면 크롭 크기가 어긋나 엉뚱한 곳에서 터진다
        # (실제로 겪었다 - 3276x510 진단 이미지가 입력으로 잡혔다).
        if frame.shape[1] != seg.cam.width or frame.shape[0] < seg.src_h:
            print(f"[{idx}] 건너뜀 - 크기 {frame.shape[1]}x{frame.shape[0]}, "
                  f"{seg.cam.width} 폭이어야 한다")
            skipped += 1
            continue
        imgs, stats = stage_images(seg, frame, 0 if args.no_dilate else 1,
                                   close_kernel, args.min_blob,
                                   attitude=att.get(int(idx)))
        for name, img in imgs.items():
            d = os.path.join(root, name + args.suffix)
            if d not in made:
                os.makedirs(d, exist_ok=True)
                made.add(d)
            # **반환값을 본다.** Windows 에서 그 파일을 뷰어나 IDE 가 열고 있으면
            # 쓰기가 막히는데 imwrite 는 예외 없이 False 만 돌려준다. 실측으로
            # 39장 중 3장이 이렇게 조용히 빠졌고, 다 본 줄 알고 36장만 봤다.
            if not cv2.imwrite(os.path.join(d, f"{idx}.png"), img):
                failed.append(os.path.join(name + args.suffix, f"{idx}.png"))

        # 합계는 단계마다 따로 쌓는다. 폭 백분위수는 더할 수 없는 값이라
        # 합계에서는 최대값만 의미가 있다.
        for stage, st in stats.items():
            for c, s in st.items():
                t = total[stage].setdefault(c, dict.fromkeys(s, 0))
                for k, v in s.items():
                    t[k] = max(t[k], v) if k.startswith("w_") else t[k] + v

        before = sum(s["before"] for s in stats["s02"].values())
        after = sum(s["after"] for s in stats["s02"].values())
        npt = sum(s["pts"] for s in stats["s03"].values())
        print(f"[{idx}] {before:>7d} -> {after:>6d}px "
              f"({after / max(before, 1) * 100:4.1f}%) -> {npt:>5d}pt")

    print(f"\n{len(paths)}장 합계 — 2단계 Morphology")
    for line in s02.format_stats(total["s02"]):
        print("  " + line)
    print(f"\n{len(paths)}장 합계 — 3단계 Lane pixel extraction  (run w 는 최대값)")
    for line in s03.format_stats(total["s03"]):
        print("  " + line)
    print("\n저장:")
    want = len(paths) - skipped
    for d in sorted(made):
        n = len(glob.glob(os.path.join(d, "*.png")))
        # n < want 는 쓰기 실패, n > want 는 이전 실행이 남긴 것이다. 둘 다
        # "다 봤다"고 착각하게 만들 수 있어 구분해서 알린다.
        if n < want:
            flag = f"   <- {want}장이어야 한다. 쓰기 실패!"
        elif n > want:
            flag = f"   (이번 실행은 {want}장, 나머지는 이전 실행 잔여)"
        else:
            flag = ""
        print(f"  {d}  ({n}장){flag}")

    # **조용히 빠지는 것을 막는다.** Windows 에서 그 png 를 뷰어나 IDE 가 열고
    # 있으면 imwrite 가 예외 없이 False 만 돌려준다. 실제로 39장 중 일부가
    # 이렇게 빠졌고, 다 본 줄 알고 덜 본 채로 판단할 뻔했다.
    if failed:
        print(f"\n[경고] {len(failed)}장을 쓰지 못했습니다 "
              f"(그 파일을 뷰어나 IDE 가 열고 있지 않은지 확인하세요):")
        for f in failed[:10]:
            print(f"  {f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
