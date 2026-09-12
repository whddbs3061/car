#!/usr/bin/env python3
"""[개발 도구] 주행 구간을 녹화한다. 후처리 비교의 **고정 기준**을 만들려는 것이다.

    python3 tools/record_drive.py --out rec/drive01 --frames 300

===========================================================================
이 폴더(`tools/`)는 최종 코드에 들어가지 않는다
===========================================================================
주행 경로는 `pipeline/` 과 `live_pipeline.py` 뿐이다. 녹화는 개발 중에만 쓰는
도구이고, 실시간 루프에 디스크 쓰기가 끼면 프레임을 놓쳐서 "화면에서 본 지연"이
실제 지연과 달라진다. 그래서 폴더를 갈라 둔다 - `pipeline/` 의 어떤 파일도
여기를 import 하지 않는다.

---------------------------------------------------------------------------
왜 녹화가 필요한가 - 실시간으로는 비교를 못 한다
---------------------------------------------------------------------------
추적을 켜고 끈 차이를 재려면 **같은 입력**에 두 설정을 돌려야 한다. 시뮬레이터
앞에서 껐다 켜면 그 사이에 차가 움직여서 다른 장면이 되고, 그러면 달라진 것이
설정 때문인지 장면 때문인지 영원히 못 가린다.

게다가 추적은 상태를 갖는다 - 프레임 순서가 결과의 일부다. 같은 순서로 다시
돌릴 수 있어야 한다.

---------------------------------------------------------------------------
촬영 시각을 같이 저장한다
---------------------------------------------------------------------------
시뮬레이터가 패킷에 넣어 주는 **촬영 시각**을 `meta.jsonl` 에 남긴다. 받은
시각도 같이 남긴다. 둘이 다르기 때문이다 - 실측 중앙값 128ms.

지금은 안 쓰더라도 나중에 IMU 를 붙일 때 이 값이 없으면 정합을 검증할 수 없다.
그때 다시 녹화하려면 같은 구간을 같은 속도로 또 달려야 하는데, 그게 안 된다.
"""

import argparse
import json
import os
import sys
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_LANE = os.path.dirname(_HERE)
if _LANE not in sys.path:
    sys.path.insert(0, _LANE)

from morai_camera import DEFAULT_IP, DEFAULT_PORT, CameraStream    # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="[개발용] 주행 녹화")
    ap.add_argument("--out", required=True, help="저장 폴더")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--jpeg", type=int, default=0,
                    help="0 이면 png(무손실). 1~100 이면 그 품질의 jpg")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    cam = CameraStream(args.ip, args.port).start()
    print(f"[rec] {args.ip}:{args.port} 대기 중...")
    if not cam.wait_first(timeout=15.0):
        raise SystemExit("카메라 프레임이 안 옵니다. 시뮬레이터를 확인하세요.")

    ext = "png" if args.jpeg <= 0 else "jpg"
    enc = [] if args.jpeg <= 0 else [cv2.IMWRITE_JPEG_QUALITY, args.jpeg]
    meta_path = os.path.join(args.out, "meta.jsonl")

    print(f"[rec] {args.frames}프레임 녹화. **지금 차를 주행시키세요.**")
    n, last_seq, t0 = 0, -1, time.time()
    lat_sum = 0.0
    with open(meta_path, "w") as mf:
        while n < args.frames and time.time() - t0 < args.timeout:
            frame, seq, stamp = cam.latest(with_stamp=True)
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq
            recv = time.time()
            name = f"{n:04d}.{ext}"
            cv2.imwrite(os.path.join(args.out, name), frame, enc)
            mf.write(json.dumps({"i": n, "file": name, "seq": seq,
                                 "capture": round(stamp, 4),
                                 "recv": round(recv, 4),
                                 "latency_ms": round((recv - stamp) * 1e3, 1)},
                                ensure_ascii=False) + "\n")
            lat_sum += (recv - stamp) * 1e3
            n += 1
            if n % 25 == 0:
                el = time.time() - t0
                print(f"[rec] {n}/{args.frames}  {n/el:4.1f} fps  "
                      f"지연 평균 {lat_sum/n:5.1f} ms")
    cam.stop()

    el = time.time() - t0
    print(f"[rec] 완료 {n}프레임 / {el:.1f}s ({n/max(el,1e-3):.1f} fps) -> {args.out}")
    if n < args.frames:
        print(f"[rec] **{args.frames}프레임을 못 채웠습니다.** 시뮬레이터가 "
              f"프레임을 계속 보내는지 확인하세요.")


if __name__ == "__main__":
    main()
