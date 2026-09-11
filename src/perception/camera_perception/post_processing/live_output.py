#!/usr/bin/env python3
"""[3/3] 시뮬레이터 카메라를 실시간으로 받아 **출력값만** 낸다 (그림 없음).

===========================================================================
실행 (그대로 복사해서 붙여넣기)
===========================================================================
cd C:/MSC/AutoMobility/car/src/perception/camera_perception/post_processing
C:/Users/user/anaconda3/envs/vision_env/python.exe live_output.py --udp 127.0.0.1:7600

  화면으로만 보기    --udp 를 지운다 (표준출력에 JSON 한 줄씩)
  전송만 하기        --quiet 추가
  점열까지 보내기    --points 추가
  미리보기 창        --preview        (제어값 출력은 그대로, 그림만 곁들임)
  GPU 로            --device cuda
===========================================================================

**저장은 하지 않는다.** 실주행에서 제어에 넘길 값을 뽑는 경로다.
저장이 필요하면 offline_test.py 를 쓴다.

후처리는 lane_detection.LaneDetector.run() 하나만 쓴다. 세 스크립트가 같은
코드를 부르므로 **집에서 본 결과와 실주행 결과가 갈릴 일이 없다.**

--------------------------------------------------------------------------
미리보기는 언제든 떼어낼 수 있게 격리해 두었다
--------------------------------------------------------------------------
`--preview` 는 기본 꺼짐이고, 켜도 **값 전송이 먼저 끝난 뒤에** 그린다.
그리기 코드는 아래 `PREVIEW 블록`과 `# [PREVIEW]` 가 붙은 네 줄에만 있다 —
그 블록과 그 네 줄만 지우면 lane_viz 를 import 조차 하지 않는 순수 제어
경로로 정확히 되돌아간다 (지운 뒤 남는 코드는 그대로 동작한다).

--------------------------------------------------------------------------
출력 규격은 아직 미정이다
--------------------------------------------------------------------------
지금은 DetectionResult.as_dict() 를 통째로 보낸다. 제어팀과 정하고 나면
lane_detection.DetectionResult.as_dict() 한 곳만 줄이면 세 스크립트에 다 반영된다.

들어 있는 것:
    left / right        자차 좌우 차선 경계, 자차 좌표계 점열 [[x, y], ...]
                        (x 전방, y 좌측, 미터). **경계 두 줄을 그대로 주는 것은
                        장애물 회피 때문이다** - 중심선만 주면 "어디까지 비켜도
                        되는지"를 표현할 수 없다.
    left_type/right_type  white_solid / white_dashed / yellow (차선변경 가부 판단)
    lateral_error       차로 중심 기준 횡오차 (m). 음수 = 왼쪽으로 치우침
    heading_error       차로 방향 대비 방위 오차 (rad)
    stopline_dist       자차 앞 정지선까지 (m), 없으면 null
"""

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# **여기서는 lane_viz 를 import 하지 않는다.** 제어 경로에 시각화 코드가
# 상시로 들어오지 않게 하려는 것이다. 미리보기는 아래 PREVIEW 블록 안에서만
# 지연 import 한다 (--preview 를 안 주면 모듈이 로드조차 되지 않는다).
from lane_detection import LaneDetector, default_checkpoint
from morai_camera import DEFAULT_IP, DEFAULT_PORT, CameraStream


# ==========================================================================
# PREVIEW 블록 시작 - 미리보기 (없어도 되는 기능)
# --------------------------------------------------------------------------
# **떼어내는 방법**: 이 블록 전체와, 아래에서 `# [PREVIEW]` 가 붙은 네 줄을
# 지우면 끝이다. 나머지 코드는 preview 를 전혀 참조하지 않으므로 그대로 돈다.
#
# 설계 원칙 두 가지:
#   1. **값이 먼저다.** 메인 루프에서 UDP 전송·표준출력이 끝난 뒤에 그린다.
#      그리기가 늦어도 제어로 가는 값의 지연은 늘어나지 않는다.
#   2. **솎아서 그린다.** CPU 에서는 그리기가 프레임을 잡아먹으므로 기본
#      2프레임에 1번만 그린다 (--preview-every).
# 창에서 q/ESC 를 누르면 그리기만 멈추고 값 출력은 계속된다.
# ==========================================================================
class _Preview:
    def __init__(self, detector, every, scale, udp_label=None):
        import cv2                       # 지연 import - 블록을 지우면 같이 사라진다
        from lane_viz import draw
        self._cv2, self._draw = cv2, draw
        self.det, self.every, self.scale = detector, max(every, 1), scale
        self.udp_label = udp_label
        self.i = self.n = 0
        self.ms = 0.0
        self.on = True

    def _value_strip(self, img, res):
        """**제어로 나가는 값을 화면 아래에 띄운다.**

        lane_viz.draw 의 상단 HUD 는 차선 ID·클래스만 보여준다. 이 스크립트는
        제어값을 내보내는 게 본업이므로, 실제로 전송되는 숫자를 같이 봐야
        "화면은 멀쩡한데 값이 이상한" 경우를 잡을 수 있다.
        lane_viz 를 고치지 않고 여기서 덧그리는 이유는, 그 파일이 세 스크립트
        공용이고 이 블록만 지우면 원래대로 돌아가야 하기 때문이다.
        """
        cv2 = self._cv2
        h, w = img.shape[:2]
        le, he = res.lateral_error(), res.heading_error()
        items = [
            ("lateral", f"{le:+.2f} m" if le is not None else "-", le is not None),
            ("heading", f"{he:+.3f} rad" if he is not None else "-", he is not None),
            ("stopline", f"{res.stopline_dist:.1f} m"
             if res.stopline_dist is not None else "-", res.stopline_dist is not None),
            ("send", self.udp_label or "stdout", True),
        ]
        bar = 34
        cv2.rectangle(img, (0, h - bar), (w, h), (0, 0, 0), -1)
        x = 10
        for name, val, ok in items:
            cv2.putText(img, name, (x, h - bar + 13), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (150, 150, 150), 1)
            cv2.putText(img, val, (x, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (120, 255, 140) if ok else (110, 110, 200), 1)
            x += max(len(val) * 11, len(name) * 8) + 34
        return img

    def show(self, frame, res):
        if not self.on:
            return
        self.i += 1
        if self.i % self.every:
            return
        t0 = time.time()
        img = self._draw(res, frame, self.det)
        img = self._value_strip(img, res)
        if self.scale != 1.0:
            img = self._cv2.resize(img, None, fx=self.scale, fy=self.scale)
        self._cv2.imshow("lane preview (q=닫기, 값 출력은 계속)", img)
        if (self._cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            self.close()
            print("[preview] 창을 닫았습니다. 값 출력은 계속됩니다.", file=sys.stderr)
            return
        self.ms += (time.time() - t0) * 1000.0
        self.n += 1

    def close(self):
        if not self.on:
            return
        self.on = False
        self._cv2.destroyAllWindows()
        if self.n:
            print(f"[preview] 그리기 평균 {self.ms / self.n:.0f}ms x {self.n}회",
                  file=sys.stderr)


def _preview_add_args(ap):
    ap.add_argument("--preview", action="store_true",
                    help="미리보기 창을 띄운다 (값 출력은 그대로. 창의 q 로 닫힘)")
    ap.add_argument("--preview-every", type=int, default=2,
                    help="N 프레임마다 한 번만 그린다 (기본 2)")
    ap.add_argument("--preview-scale", type=float, default=1.0, help="창 크기 배율")


def _preview_make(args, detector):
    if not getattr(args, "preview", False):
        return None
    print(f"[preview] 켜짐 - {args.preview_every}프레임마다 그림", file=sys.stderr)
    return _Preview(detector, args.preview_every, args.preview_scale,
                    udp_label=args.udp)
# ==========================================================================
# PREVIEW 블록 끝
# ==========================================================================


def build_arg_parser():
    ap = argparse.ArgumentParser(description="시뮬레이터 실시간 차선 출력값")
    ap.add_argument("--checkpoint", default=default_checkpoint())
    ap.add_argument("--cam-set", default=None)
    ap.add_argument("--bonnet", default=None,
                    help="보닛 마스크 png. 기본은 코드에 박힌 폴리곤")
    ap.add_argument("--no-bonnet", action="store_true", help="보닛 제거를 끈다")
    ap.add_argument("--no-track", action="store_true", help="프레임 간 추적을 끈다")
    ap.add_argument("--ip", default=DEFAULT_IP, help="카메라 수신 IP")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="카메라 수신 포트")
    ap.add_argument("--device", default=None, help="cuda / cpu")
    ap.add_argument("--udp", default=None, metavar="HOST:PORT",
                    help="이 주소로 JSON 을 UDP 전송한다 (예: 127.0.0.1:7600)")
    ap.add_argument("--quiet", action="store_true",
                    help="표준출력으로 JSON 을 찍지 않는다 (UDP 전송만)")
    ap.add_argument("--points", action="store_true",
                    help="left/right 점열까지 보낸다 (기본은 빼고 요약값만 - "
                         "점열이 프레임당 수십 개라 UDP 패킷이 커진다)")
    ap.add_argument("--stats-sec", type=float, default=2.0,
                    help="통계를 몇 초마다 stderr 로 찍을지 (0=안 찍음)")
    _preview_add_args(ap)                                          # [PREVIEW]
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    sender = None
    if args.udp:
        host, _, port = args.udp.partition(":")
        if not port:
            raise SystemExit("--udp 는 HOST:PORT 형식입니다 (예: 127.0.0.1:7600)")
        sender = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))

    pipe = LaneDetector(args.checkpoint, cam_set=args.cam_set,
                        bonnet_mask=False if args.no_bonnet else args.bonnet,
                        device=args.device, track=not args.no_track)
    print(f"[out] epoch {pipe.ckpt_info['epoch']} ({pipe.ckpt_info['backbone']}) "
          f"device={pipe.device} 보닛 {pipe.bonnet_source}", file=sys.stderr)

    preview = _preview_make(args, pipe)                            # [PREVIEW]

    cam = CameraStream(args.ip, args.port).start()
    print(f"[out] {args.ip}:{args.port} 대기 중...", file=sys.stderr)
    if not cam.wait_first(timeout=15.0):
        raise SystemExit("카메라 프레임이 안 옵니다. 시뮬레이터와 IP/포트를 확인하세요.")
    print("[out] 수신 시작. Ctrl+C 로 종료합니다.", file=sys.stderr)

    last_seq = -1
    n = n_left = n_right = n_stop = 0
    t_stat = time.time()
    ms_sum = 0.0

    try:
        while True:
            frame, seq = cam.latest()
            if frame is None or seq == last_seq:
                time.sleep(0.002)
                continue
            last_seq = seq

            res = pipe.run(frame)
            payload = res.as_dict(points=args.points)
            payload["seq"] = seq
            payload["t"] = round(time.time(), 3)

            line = json.dumps(payload, ensure_ascii=False)
            if sender is not None:
                sender[0].sendto(line.encode("utf-8"), sender[1])
            if not args.quiet:
                print(line, flush=True)

            if preview:                                            # [PREVIEW]
                preview.show(frame, res)                           # [PREVIEW]

            n += 1
            n_left += res.ego_left is not None
            n_right += res.ego_right is not None
            n_stop += res.stopline_dist is not None
            ms_sum += res.infer_ms + res.post_ms

            if args.stats_sec and time.time() - t_stat >= args.stats_sec:
                dt = time.time() - t_stat
                print(f"[out] {n/dt:5.1f} FPS  평균 {ms_sum/max(n,1):5.0f}ms  "
                      f"좌 {n_left}/{n} 우 {n_right}/{n} 정지선 {n_stop}/{n}",
                      file=sys.stderr)
                t_stat = time.time()
                n = n_left = n_right = n_stop = 0
                ms_sum = 0.0
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if preview:                                                # [PREVIEW]
            preview.close()                                        # [PREVIEW]
        if sender is not None:
            sender[0].close()
        print("[out] 종료", file=sys.stderr)


if __name__ == "__main__":
    main()
