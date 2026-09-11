#!/usr/bin/env python3
"""[1/3] 녹화 영상으로 돌려서 **결과를 저장**한다 (집에서 눈으로 확인용).

===========================================================================
실행 (그대로 복사해서 붙여넣기)
===========================================================================
cd C:/MSC/AutoMobility/car/src/perception/camera_perception/post_processing
C:/Users/user/anaconda3/envs/vision_env/python.exe offline_test.py --recording ../recordings/lap4_full --until 000471 --bev --video

  일부만 빠르게      --count 40 --stride 5
  모델 출력만 보기   --mask-only     (모델 탓/후처리 탓 구분용)
  특정 프레임만      --frames 000114,000315
  추적 끄기          --no-track      (프레임별 독립 검출과 비교할 때)
  교차로 방향 지정    --maneuver left (좌회전 유도선만 따라간다)
  GPU 로            --device cuda

**lap4_full 은 000472 부터 도로가 아니다.** --until 000471 로 자른다.
**lap1_full 은 쓰지 않는다** - JPEG 이 잘려 저장된 프레임이 32/358장(8.9%)
섞여 있어 화면 아래쪽이 평평한 회색으로 채워진다. 입력 자체가 없는 것이라
후처리로 고칠 수 없다.
===========================================================================

시뮬레이터가 필요 없다. `frames/*.png` 만 있으면 된다.

검출·후처리는 `lane_detection.LaneDetector.run()` 하나만 쓴다 - live_overlay.py,
live_output.py 와 **완전히 같은 코드**다. 여기서 본 결과가 실주행 결과와
다르지 않아야 하므로 후처리를 복사해 두지 않는다.

출력
    <recording>/detect_check/detect_XXXXXX.png    프레임별 그림
    <recording>/detect_check/result.mp4           --video 를 주면 영상으로도
    <recording>/detect_check/output.jsonl         프레임별 출력값
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lane_detection import CLASS_NAMES, LaneDetector
from lane_viz import draw, draw_bev, draw_mask_only


def build_arg_parser():
    ap = argparse.ArgumentParser(description="녹화 영상으로 차선 검출 확인")
    ap.add_argument("--recording", required=True, help="frames/ 가 있는 폴더")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cam-set", default=None)
    ap.add_argument("--bonnet", default=None,
                    help="보닛 마스크 png. 기본은 코드에 박힌 폴리곤이라 보통 "
                         "줄 필요가 없다")
    ap.add_argument("--no-bonnet", action="store_true", help="보닛 제거를 끈다")
    ap.add_argument("--frames", default=None, help="프레임 번호들 (쉼표 구분)")
    ap.add_argument("--count", type=int, default=9999)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--until", default=None, metavar="IDX",
                    help="이 프레임 번호까지만 처리한다 (뒤쪽이 도로가 아닐 때)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--bev", action="store_true", help="조감도를 오른쪽에 붙인다")
    ap.add_argument("--video", action="store_true", help="mp4 로도 저장한다")
    ap.add_argument("--fps", type=float, default=10.0, help="--video 재생 속도")
    ap.add_argument("--no-png", action="store_true", help="개별 png 는 저장하지 않는다")
    ap.add_argument("--no-track", action="store_true", help="프레임 간 추적을 끈다")
    ap.add_argument("--mask-only", action="store_true",
                    help="후처리 없이 모델 출력만 본다")
    ap.add_argument("--maneuver", default=None,
                    choices=["left", "straight", "right"],
                    help="교차로에서 따라갈 유도선 방향. 안 주면 자차 진로에 "
                         "가장 가까운 유도선을 쓴다")
    ap.add_argument("--guide-side", default=None, choices=["left", "right"],
                    help="유도선을 자차의 어느 쪽에 두고 달릴지 (기본 left)")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    paths = sorted(glob.glob(os.path.join(args.recording, "frames", "*.png")))
    if not paths:
        raise SystemExit(f"frames/*.png 가 없습니다: {args.recording}")
    if args.frames:
        want = {x.strip() for x in args.frames.split(",") if x.strip()}
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in want]
        if not paths:
            raise SystemExit(f"그 프레임을 못 찾았습니다: {sorted(want)}")
    else:
        if args.until:
            paths = [p for p in paths
                     if os.path.splitext(os.path.basename(p))[0] <= args.until]
        paths = paths[::max(args.stride, 1)][:args.count]

    det = LaneDetector(args.checkpoint, cam_set=args.cam_set,
                       bonnet_mask=False if args.no_bonnet else args.bonnet,
                       device=args.device, track=not args.no_track,
                       maneuver=args.maneuver, guide_keep_side=args.guide_side)
    print(f"[offline] epoch {det.ckpt_info['epoch']} ({det.ckpt_info['backbone']}) "
          f"device={det.device} 보닛 {det.bonnet_source} "
          f"추적 {'끔' if args.no_track else '켬'}")
    print(f"[offline] 프레임 {len(paths)}장")

    sub = "mask_check" if args.mask_only else "detect_check"
    out_dir = args.out or os.path.join(args.recording, sub)
    os.makedirs(out_dir, exist_ok=True)
    writer = None
    n_lane = n_left = n_right = n_both = n_stop = 0
    n_guide = n_guide_used = 0
    tot_ms = 0.0

    fp = None if args.mask_only else open(
        os.path.join(out_dir, "output.jsonl"), "w", encoding="utf-8")
    try:
        for p in paths:
            idx = os.path.splitext(os.path.basename(p))[0]
            frame = cv2.imread(p)
            if frame is None:
                continue
            r = det.run(frame)
            tot_ms += r.infer_ms + r.post_ms

            if args.mask_only:
                c = np.bincount(r.mask.ravel(), minlength=len(CLASS_NAMES))
                print(f"[{idx}] " + "  ".join(f"{CLASS_NAMES[k][:9]} {c[k]}"
                                              for k in range(1, len(CLASS_NAMES))))
                img = draw_mask_only(r, frame, det.crop_top)
            else:
                n_lane += len(r.lanes)
                n_left += r.ego_left is not None
                n_right += r.ego_right is not None
                n_both += (r.ego_left is not None and r.ego_right is not None)
                n_stop += r.stopline_dist is not None
                n_guide += bool(r.guides)
                n_guide_used += r.center_source() == "guide"
                rec = r.as_dict(points=True)
                rec["idx"] = idx
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                line = "  ".join(
                    f"{l.lane_id:+d} {l.name}({l.n_points}px "
                    f"x{l.x_range[0]:.1f}~{l.x_range[1]:.1f}m)" for l in r.lanes)
                if r.guides:
                    g = r.guide
                    line += f"  |guide x{len(r.guides)}" + (
                        f" -> x{g.x_range[0]:.1f}~{g.x_range[1]:.1f}m" if g else "")
                print(f"[{idx}] " + (line or "(검출 없음)")
                      + (f"  center={r.center_source()}"
                         if r.center_source() else ""))
                img = draw(r, frame, det)
                if args.bev:
                    b = draw_bev(r)
                    h = max(img.shape[0], b.shape[0])
                    pad = lambda im: cv2.copyMakeBorder(
                        im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                    img = np.hstack([pad(img), pad(b)])

            if not args.no_png:
                cv2.imwrite(os.path.join(out_dir, f"{sub.split('_')[0]}_{idx}.png"), img)
            if args.video:
                if writer is None:
                    writer = cv2.VideoWriter(
                        os.path.join(out_dir, "result.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                        (img.shape[1], img.shape[0]))
                writer.write(img)
    finally:
        if fp is not None:
            fp.close()
        if writer is not None:
            writer.release()

    n = max(len(paths), 1)
    print(f"\n{len(paths)}장 저장: {out_dir}")
    if args.video:
        print(f"영상: {os.path.join(out_dir, 'result.mp4')}")
    if not args.mask_only:
        print(f"  자차 좌 {n_left}/{n} ({n_left/n*100:.1f}%)  "
              f"우 {n_right}/{n} ({n_right/n*100:.1f}%)  "
              f"둘 다 {n_both}/{n} ({n_both/n*100:.1f}%)")
        print(f"  정지선 {n_stop}/{n} ({n_stop/n*100:.1f}%)  "
              f"프레임당 평균 차선 {n_lane/n:.2f}개")
        # **유도선이 보인 프레임과 실제로 쓴 프레임을 따로 센다.** 둘이 크게
        # 벌어지면 검출이 아니라 고르는 규칙(select_guide)이 걸러낸 것이다.
        print(f"  유도선 보임 {n_guide}/{n} ({n_guide/n*100:.1f}%)  "
              f"주행선으로 씀 {n_guide_used}/{n} ({n_guide_used/n*100:.1f}%)")
    print(f"  평균 {tot_ms/n:.0f}ms ({n/max(tot_ms,1)*1000:.1f} FPS)")


if __name__ == "__main__":
    main()
