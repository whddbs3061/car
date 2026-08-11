"""주행 시퀀스 녹화 도구.

시뮬레이터에서 카메라 프레임과 차량 상태(EgoVehicleStatus)를 함께 받아
디스크에 저장한다. 차선 인식 파라미터를 재현 가능하게 튜닝하려면
같은 시퀀스에 반복해서 돌릴 수 있어야 하기 때문이다.

저장 구조:
    recordings/<이름>/
        drive.mp4               주행 영상 (LaneEgoSelect --source 로 바로 재생 가능)
        meta.jsonl              프레임별 차량 상태 한 줄씩

영상으로 저장하는 이유는 옮기기 쉬워서다. 프레임 9000장이 600MB를 넘는 데 비해
mp4 는 수십 MB 수준이다. --format frames 로 프레임 폴더를 쓸 수도 있다.

mp4 의 fps 는 --fps 로 지정하는 명목값이다. UDP 프레임 간격은 일정하지 않으므로
실제 타이밍은 meta.jsonl 의 t 값을 봐야 한다. 프레임 순서와 내용은 그대로 보존된다.

meta.jsonl 의 각 줄:
    idx, t          프레임 번호와 녹화 시작 기준 경과 시간(초)
    pos_x/y, yaw    위치와 방위 — 구간을 나중에 잘라내는 기준
    signed_vel      속도 (m/s)
    ang_vel_z       yaw rate — 곡선 구간을 찾는 기준
    steer, accel, brake, gear
    link_id         자차가 올라가 있는 HD 맵 링크 ID.
                    차로 단위 정답이라 차로 변경 시점이 자동으로 라벨링된다.
                    인지 파이프라인 입력으로 쓰면 안 되고, 평가용으로만 쓴다.

사용법:
    python RecordDrive.py --name full --no-preview      # 긴 주행은 미리보기 끄면 가볍다
    python RecordDrive.py --name curve --max-frames 600
    python RecordDrive.py --name straight --format frames

    ESC 또는 Ctrl+C 로 종료. 종료 시 요약을 출력한다.

이 스크립트는 시뮬레이터가 있는 PC에서만 돌린다. 만들어진 drive.mp4 와
meta.jsonl 을 옮겨서 LaneEgoSelect.py 로 오프라인 튜닝을 한다.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
lib_dir = os.path.join(parent_dir, 'lib')

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

import cv2
import numpy as np

from lib.define.Camera import Camera
from lib.define.EgoVehicleStatus import EgoVehicleStatus
from lib.network.UDP import Receiver

# Sensor/ 의 다른 카메라 스크립트와 같은 기본값.
# PC/가상머신마다 주소가 다르므로 환경변수로도 바꿀 수 있다.
DEFAULT_CAM_IP = os.environ.get("MORAI_CAM_IP", "192.168.0.200")
DEFAULT_CAM_PORT = int(os.environ.get("MORAI_CAM_PORT", "1101"))
# 시뮬레이터 Network Settings 의 sim→user 목적지 포트. 실측으로 확인했다:
# 192.168.0.151:1910 → 이쪽 1911 로 229바이트(#MoraiInfo$ ... \r\n) 가 들어온다.
# 예전 기본값이던 909 는 특권 포트라 일반 사용자로는 bind 조차 안 된다.
DEFAULT_STATUS_PORT = int(os.environ.get("MORAI_STATUS_PORT", "1911"))

FRAME_SIZE = (640, 480)

_frame_lock = threading.Lock()
_latest_frame = None

_status_lock = threading.Lock()
_latest_status = None


def camera_worker(ip, port):
    global _latest_frame
    receiver = Receiver(ip, port, Camera())

    while True:
        try:
            data = receiver.get_data()
            if data is None:
                time.sleep(0.01)
                continue
            if not hasattr(data, "image") or not data.image.data:
                continue

            jpeg = data.image.data
            # 잘린 JPEG 은 imdecode 가 성공시키고 못 받은 아래쪽을 회색(125)으로
            # 채워 돌려준다. 화면이 회색으로 깜빡이는 게 이것이므로 EOI 마커가
            # 없으면 아예 버린다. Camera.parsing 에서도 거르지만 여기서 한 번 더 본다.
            if jpeg[:2] != b'\xff\xd8' or jpeg[-2:] != b'\xff\xd9':
                continue

            buf = np.frombuffer(jpeg, dtype=np.uint8)
            if buf.size == 0:
                continue

            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                continue

            with _frame_lock:
                _latest_frame = cv2.resize(image, FRAME_SIZE)

        except (AttributeError, ValueError, OSError, cv2.error) as ex:
            print(f"[camera] recoverable error: {ex}")
            time.sleep(0.1)
        except Exception as ex:
            print(f"[camera] unexpected error: {ex}")
            traceback.print_exc()
            raise


def status_worker(ip, port):
    global _latest_status
    receiver = Receiver(ip, port, EgoVehicleStatus())

    while True:
        try:
            data = receiver.get_data()
            if data is None:
                time.sleep(0.01)
                continue

            link_id = getattr(data, "link_id", b"")
            if isinstance(link_id, bytes):
                link_id = link_id.split(b"\x00")[0].decode("utf-8", errors="replace")

            with _status_lock:
                _latest_status = {
                    "pos_x": float(data.pos_x),
                    "pos_y": float(data.pos_y),
                    "yaw": float(data.yaw),
                    "signed_vel": float(data.signed_vel),
                    "ang_vel_z": float(data.ang_vel_z),
                    "steer": float(data.steer),
                    "accel": float(data.accel),
                    "brake": float(data.brake),
                    "gear": int(data.gear),
                    "link_id": link_id,
                }

        except (AttributeError, ValueError, OSError) as ex:
            print(f"[status] recoverable error: {ex}")
            time.sleep(0.1)
        except Exception as ex:
            print(f"[status] unexpected error: {ex}")
            traceback.print_exc()
            raise


def record(args):
    out_dir = os.path.join(args.out, args.name)
    frames_dir = os.path.join(out_dir, "frames")
    video_path = os.path.join(out_dir, "drive.mp4")

    save_video = args.format in ("video", "both")
    save_frames = args.format in ("frames", "both")

    if save_video and os.path.exists(video_path):
        print(f"[에러] 이미 영상이 있습니다: {video_path}")
        print("       --name 을 바꾸거나 파일을 지우고 다시 실행하세요.")
        return
    if save_frames and os.path.exists(frames_dir) and os.listdir(frames_dir):
        print(f"[에러] 이미 프레임이 들어 있는 폴더입니다: {frames_dir}")
        print("       --name 을 바꾸거나 폴더를 비우고 다시 실행하세요.")
        return

    # 포트를 미리 잡아본다. Receiver 는 데몬 스레드 안에서 bind 하기 때문에
    # 실패해도 스레드만 조용히 죽고 녹화는 그대로 진행된다. 그러면 5분을 다 찍고
    # 나서야 차량 상태가 없다는 걸 알게 된다.
    for label, ip, port in (("카메라", args.cam_ip, args.cam_port),
                            ("차량 상태", args.status_ip, args.status_port)):
        if label == "차량 상태" and args.no_status:
            continue
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((ip, port))
        except PermissionError:
            print(f"[에러] {label} 포트 {port} 를 열 수 없습니다 (권한 없음).")
            print(f"       1024 미만은 특권 포트입니다. 한 번만 풀어주면 됩니다:")
            print(f"         sudo sysctl -w net.ipv4.ip_unprivileged_port_start={port}")
            print(f"       재부팅 후에도 유지하려면:")
            print(f"         echo 'net.ipv4.ip_unprivileged_port_start={port}' | "
                  f"sudo tee /etc/sysctl.d/99-morai.conf")
            return
        except OSError as ex:
            print(f"[에러] {label} 포트 {ip}:{port} bind 실패 — {ex}")
            print("       이미 다른 프로세스가 쓰고 있거나 IP 가 이 PC 의 것이 아닙니다.")
            return
        finally:
            probe.close()

    os.makedirs(out_dir, exist_ok=True)
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    writer = None
    if save_video:
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, FRAME_SIZE)
        if not writer.isOpened():
            print(f"[에러] 영상 파일을 열 수 없습니다: {video_path}")
            return

    threading.Thread(target=camera_worker,
                     args=(args.cam_ip, args.cam_port), daemon=True).start()
    if not args.no_status:
        threading.Thread(target=status_worker,
                         args=(args.status_ip, args.status_port), daemon=True).start()

    print("=" * 60)
    print(f"📹 녹화 준비 — {out_dir}")
    print(f"   카메라 {args.cam_ip}:{args.cam_port}", end="")
    print("  (차량 상태 없음)" if args.no_status
          else f"  차량 상태 {args.status_ip}:{args.status_port}")
    print("   ESC 또는 Ctrl+C 로 종료")
    print("=" * 60)
    print("첫 프레임 대기 중...")

    meta_path = os.path.join(out_dir, "meta.jsonl")
    idx = 0
    started = None
    last_saved = None
    status_seen = False
    link_ids = set()

    try:
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            while True:
                with _frame_lock:
                    frame = None if _latest_frame is None else _latest_frame.copy()

                if frame is None:
                    time.sleep(0.01)
                    continue

                # 같은 프레임을 두 번 저장하지 않는다
                if last_saved is not None and np.array_equal(frame, last_saved):
                    time.sleep(0.005)
                    continue
                last_saved = frame

                now = time.time()
                if started is None:
                    started = now
                    print("녹화 시작.")

                with _status_lock:
                    status = dict(_latest_status) if _latest_status else {}
                if status:
                    status_seen = True
                    if status.get("link_id"):
                        link_ids.add(status["link_id"])

                if writer is not None:
                    writer.write(frame)
                if save_frames:
                    cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
                meta_file.write(json.dumps(
                    {"idx": idx, "t": round(now - started, 3), **status},
                    ensure_ascii=False) + "\n")
                meta_file.flush()
                idx += 1

                if idx % 30 == 0:
                    elapsed = now - started
                    fps = idx / elapsed if elapsed > 0 else 0
                    vel = status.get("signed_vel")
                    vel_txt = f"{vel:5.1f}m/s" if vel is not None else "  --  "
                    print(f"  {idx:5d}프레임  {elapsed:6.1f}s  {fps:4.1f}FPS  "
                          f"{vel_txt}  link={status.get('link_id', '-')}")

                if args.max_frames and idx >= args.max_frames:
                    print(f"\n--max-frames({args.max_frames}) 도달, 종료합니다.")
                    break

                if not args.no_preview:
                    cv2.imshow("RecordDrive (ESC 종료)", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

    except KeyboardInterrupt:
        print("\n중단됨.")
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if idx == 0:
        print("\n[경고] 저장된 프레임이 없습니다. 시뮬레이터가 켜져 있는지,")
        print(f"       카메라가 {args.cam_ip}:{args.cam_port} 로 송신 중인지 확인하세요.")
        return

    elapsed = time.time() - started
    print("\n" + "=" * 60)
    print(f"✅ {idx}프레임 / {elapsed:.1f}초 / 평균 {idx / elapsed:.1f}FPS")
    if save_video:
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"   영상  : {video_path}  ({size_mb:.1f} MB, 명목 {args.fps}fps)")
    if save_frames:
        print(f"   프레임: {frames_dir}")
    print(f"   메타  : {meta_path}")
    if args.no_status:
        print("   차량 상태: 기록 안 함 (--no-status)")
    elif status_seen:
        print(f"   link_id {len(link_ids)}종 관측 — 차로 변경 라벨로 쓸 수 있습니다.")
    else:
        print(f"   [경고] 차량 상태가 한 번도 안 들어왔습니다 "
              f"({args.status_ip}:{args.status_port}).")
        print("          --status-ip 127.0.0.1 로 다시 시도해 보세요.")
    print("=" * 60)
    print("\n재생:")
    print(f"  python LaneEgoSelect.py --source {video_path if save_video else frames_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="주행 시퀀스 녹화 (카메라 + 차량 상태)")
    parser.add_argument("--name", required=True, help="녹화 이름 (폴더명이 된다)")
    parser.add_argument("--out", default=os.path.join(parent_dir, "recordings"),
                        help="저장 위치 (기본: 저장소 루트의 recordings/)")
    parser.add_argument("--cam-ip", default=DEFAULT_CAM_IP)
    parser.add_argument("--cam-port", type=int, default=DEFAULT_CAM_PORT)
    parser.add_argument("--status-ip", default=None,
                        help="차량 상태 IP (기본: 카메라와 동일). 안 들어오면 127.0.0.1 시도")
    parser.add_argument("--status-port", type=int, default=DEFAULT_STATUS_PORT)
    parser.add_argument("--format", choices=("video", "frames", "both"), default="video",
                        help="저장 형식 (기본 video: drive.mp4)")
    parser.add_argument("--fps", type=int, default=20,
                        help="mp4 의 명목 fps (기본 20). 실제 타이밍은 meta.jsonl 의 t 참고")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="이 프레임 수에 도달하면 자동 종료 (0이면 무제한)")
    parser.add_argument("--no-status", action="store_true",
                        help="차량 상태 없이 프레임만 녹화")
    parser.add_argument("--no-preview", action="store_true",
                        help="미리보기 창을 띄우지 않는다 (긴 주행에 권장)")

    args = parser.parse_args()
    if args.status_ip is None:
        args.status_ip = args.cam_ip
    return args


if __name__ == "__main__":
    record(parse_args())
