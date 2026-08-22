"""주행 시퀀스 녹화 도구.

시뮬레이터에서 카메라 프레임과 차량 상태(EgoVehicleStatus)를 함께 받아
디스크에 저장한다. 차선 인식 파라미터를 재현 가능하게 튜닝하려면
같은 시퀀스에 반복해서 돌릴 수 있어야 하기 때문이다.

저장 구조:
    recordings/<이름>/
        drive.mp4               주행 영상 (미리보기·육안 확인용. --format both 로 병행 가능)
        frames/000000.png ...   학습용 프레임. 기본 PNG 무손실 (--frame-format jpg 로 변경 가능)
        meta.jsonl              프레임별 차량 상태 + ObjectInfo 한 줄씩

카메라 프레임은 축소하지 않고 시뮬레이터가 보낸 원본 해상도 그대로 저장한다
(예전에는 640x480 으로 리사이즈했으나, 라벨 생성기가 가정하는 원본 해상도와
어긋나는 미지수를 없애기 위해 제거했다).

mp4 의 fps 는 --fps 로 지정하는 명목값이다. UDP 프레임 간격은 일정하지 않으므로
실제 타이밍은 meta.jsonl 의 t 값을 봐야 한다. 프레임 순서와 내용은 그대로 보존된다.

meta.jsonl 의 각 줄:
    idx, t          프레임 번호와 녹화 시작 기준 경과 시간(초)
    cam_sec/nsec    카메라 프레임의 시뮬레이터 시각
    status_sec/nsec 자차 상태의 시뮬레이터 시각. cam_* 과 비교하면 두 스트림의
                    짝이 맞는지 확인할 수 있다 (급커브에서는 수십 ms 도 크다)
    pos_x/y/z, yaw  위치와 방위 — 구간을 나중에 잘라내는 기준
    roll, pitch     차체 자세 (-180~180 으로 펴서 저장). 시뮬레이터는 0~360 으로 준다
    pose_dt         이 줄의 pose 를 만든 보간 기준점과 카메라 시각의 차 (초).
                    0 에 가까울수록 좋다
    pose_extrapolated  상태 이력 밖이라 보간하지 못한 프레임에만 붙는다

**자차 상태는 프레임의 시뮬레이터 시각으로 보간해서 넣는다.** 카메라 프레임은
조립+디코드 때문에 상태보다 100~200ms 늦게 도착하므로(실측 +145ms), 받은 순간의
최신 상태를 그대로 붙이면 급커브에서 yaw 가 5도 이상 어긋나 라벨이 차선 밖으로
나간다. 직선에서는 티가 안 나서 놓치기 쉽다.
    signed_vel      속도 (m/s)
    ang_vel_z       yaw rate — 곡선 구간을 찾는 기준
    steer, accel, brake, gear
    link_id         자차가 올라가 있는 HD 맵 링크 ID.
                    차로 단위 정답이라 차로 변경 시점이 자동으로 라벨링된다.
                    인지 파이프라인 입력으로 쓰면 안 되고, 평가용으로만 쓴다.
    objects         [{pose_x, pose_y, pose_z, heading, size_x, size_y, size_z,
                     objType}, ...] — ObjectInfo 로 받은 그 프레임의 물체 목록.
                    라벨 생성기가 가려짐(ignore) 마스크를 만드는 근거다.
                    영상만 보고는 복원할 수 없어 녹화 시점에만 얻을 수 있다.

사용법:
    python RecordDrive.py --name full --no-preview --format both
    python RecordDrive.py --name curve --max-frames 600
    python RecordDrive.py --name straight --format frames --frame-format jpg

    ESC 또는 Ctrl+C 로 종료. 종료 시 요약을 출력한다.

이 스크립트는 시뮬레이터가 있는 PC에서만 돌린다. lib.define.Camera /
EgoVehicleStatus / ObjectInfo / lib.network.UDP 는 morai_ws/src/ROI/lib 를
그대로 재사용한다 (읽기 전용, ROI 는 건드리지 않는다).
"""

import argparse
import collections
import ctypes
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = current_dir
# lib.define.Camera / EgoVehicleStatus / ObjectInfo / UDP 는 morai_ws/src/ROI/lib
# 에 있다 (ROI/src/perception/lib 는 과거 구조의 잔재라 .py 가 없다 — pycache 만
# 남아 있다). ROI 를 건드리지 않고 읽기 전용으로 그 lib 를 재사용한다.
roi_root = os.environ.get(
    "MORAI_ROI_ROOT",
    os.path.join(os.path.dirname(current_dir), "ROI"))
lib_dir = os.path.join(roi_root, 'lib')

if roi_root not in sys.path:
    sys.path.insert(0, roi_root)
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)

import cv2
import numpy as np

from lib.define.Camera import Camera
from lib.define.EgoVehicleStatus import EgoVehicleStatus
from lib.define.ObjectInfo import ObjectInfo
from lib.network.UDP import Receiver

# Sensor/ 의 다른 카메라 스크립트와 같은 기본값.
# PC/가상머신마다 주소가 다르므로 환경변수로도 바꿀 수 있다.
DEFAULT_CAM_IP = os.environ.get("MORAI_CAM_IP", "192.168.0.200")
DEFAULT_CAM_PORT = int(os.environ.get("MORAI_CAM_PORT", "1101"))
# 시뮬레이터 Network Settings 의 sim→user 목적지 포트. 실측으로 확인했다:
# 192.168.0.151:1910 → 이쪽 1911 로 229바이트(#MoraiInfo$ ... \r\n) 가 들어온다.
# 예전 기본값이던 909 는 특권 포트라 일반 사용자로는 bind 조차 안 된다.
DEFAULT_STATUS_PORT = int(os.environ.get("MORAI_STATUS_PORT", "1911"))
# ObjectInfo 는 카메라·차량상태와 같은 인터페이스로 들어온다. EgoNetwork 쪽
# 예제는 127.0.0.1 을 쓰지만 그건 시뮬레이터와 같은 PC 에서 돌릴 때 얘기다.
# 여기서는 지정이 없으면 카메라 IP 를 따라간다 (parse_args 참고).
DEFAULT_OBJECTINFO_IP = os.environ.get("MORAI_OBJECTINFO_IP")
DEFAULT_OBJECTINFO_PORT = int(os.environ.get("MORAI_OBJECTINFO_PORT", "7505"))

_frame_lock = threading.Lock()
_latest_frame = None
# 프레임 식별자 (image.sec, image.nsec). Receiver.get_data() 는 새 프레임이
# 없어도 직전 객체를 그대로 돌려주므로, 이 키가 바뀌었을 때만 새 프레임이다.
_latest_frame_key = None

_status_lock = threading.Lock()
_latest_status = None
# 자차 상태 이력 (시뮬레이터 시각, 상태). 카메라 프레임은 조립+디코드 때문에
# 상태보다 100~200ms 늦게 손에 들어온다. 그 순간의 최신 상태를 그냥 붙이면
# 급커브에서 프레임과 pose 가 어긋난다 — 실측 +145ms, yaw rate 37도/s 에서
# 5.4도 오차라 라벨이 차선 밖으로 나간다. 그래서 이력을 들고 있다가 프레임의
# 시각으로 보간해서 쓴다.
_status_hist = collections.deque(maxlen=400)


def _wrap180(a):
    """MORAI 는 자세각을 0~360 으로 준다. -180~180 으로 펴야 보간이 된다."""
    return ((a + 180.0) % 360.0) - 180.0


# 자세(yaw/pitch/roll) 보간은 오일러 각을 축마다 따로 선형보간하면 안 된다.
# 두 축이 동시에 많이 바뀌는 순간(급커브 + 뱅크가 겹치는 구간)에는 축별
# 선형보간이 실제 3D 회전 경로에서 벗어난다 — 회전은 교환법칙이 없기 때문이다.
# 쿼터니언으로 바꿔 SLERP(구면 선형보간)하면 항상 최단 회전 경로를 지난다.
# 각도-쿼터니언 변환은 GL.rot_vehicle_to_world 의 R = Rz(yaw)Ry(pitch)Rx(roll)
# 관례와 맞춰야 한다 (표준 ZYX/항공 관례와 동일).
def _euler_to_quat(yaw_deg, pitch_deg, roll_deg):
    y, p, r = (math.radians(a) / 2.0 for a in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = cy * sp * cr + sy * cp * sr
    qz = sy * cp * cr - cy * sp * sr
    return (qw, qx, qy, qz)


def _quat_to_euler(q):
    qw, qx, qy, qz = q
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    s = max(-1.0, min(1.0, 2 * (qw * qy - qz * qx)))
    pitch = math.asin(s)
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _quat_slerp(q0, q1, t):
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:                       # 더 짧은 쪽 호를 taken다
        q1 = tuple(-x for x in q1)
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:                    # 거의 같은 자세면 선형보간으로 충분
        out = tuple(a + (b - a) * t for a, b in zip(q0, q1))
    else:
        theta0 = math.acos(dot)
        theta = theta0 * t
        q2 = tuple(b - a * dot for a, b in zip(q0, q1))
        n = math.sqrt(sum(x * x for x in q2)) or 1e-12
        q2 = tuple(x / n for x in q2)
        out = tuple(a * math.cos(theta) + b * math.sin(theta) for a, b in zip(q0, q2))
    n = math.sqrt(sum(x * x for x in out)) or 1e-12
    return tuple(x / n for x in out)


def interp_status(t_cam):
    """카메라 프레임 시각에 해당하는 자차 상태를 선형보간한다."""
    with _status_lock:
        hist = list(_status_hist)
    if not hist:
        return None, None
    if t_cam <= hist[0][0]:
        return dict(hist[0][1]), t_cam - hist[0][0]
    if t_cam >= hist[-1][0]:
        return dict(hist[-1][1]), t_cam - hist[-1][0]

    lo, hi = 0, len(hist) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if hist[mid][0] <= t_cam:
            lo = mid
        else:
            hi = mid
    t0, a = hist[lo]
    t1, b = hist[hi]
    if t1 - t0 < 1e-9:
        return dict(a), 0.0
    r = (t_cam - t0) / (t1 - t0)

    out = dict(a)
    for k in ("pos_x", "pos_y", "pos_z", "signed_vel", "ang_vel_z",
              "steer", "accel", "brake"):
        if k in a and k in b:
            out[k] = a[k] + (b[k] - a[k]) * r
    if "yaw" in a and "yaw" in b and "pitch" in a and "roll" in a:
        q0 = _euler_to_quat(a["yaw"], a["pitch"], a["roll"])
        q1 = _euler_to_quat(b["yaw"], b["pitch"], b["roll"])
        yaw_i, pitch_i, roll_i = _quat_to_euler(_quat_slerp(q0, q1, r))
        out["yaw"], out["pitch"], out["roll"] = yaw_i, pitch_i, roll_i
    # link_id, gear 는 이산값이라 가까운 쪽을 쓴다
    near = a if r < 0.5 else b
    for k in ("link_id", "gear"):
        if k in near:
            out[k] = near[k]
    return out, 0.0

_objects_lock = threading.Lock()
_latest_objects = None


def camera_worker(ip, port):
    global _latest_frame, _latest_frame_key
    receiver = Receiver(ip, port, Camera())
    last_key = None

    while True:
        try:
            data = receiver.get_data()
            if data is None:
                time.sleep(0.01)
                continue
            if not hasattr(data, "image") or not data.image.data:
                time.sleep(0.002)
                continue

            # get_data() 는 폴링마다 같은 객체를 돌려준다 (초당 150회 이상).
            # 키를 안 보고 매번 디코드하면 같은 1280x720 JPEG 을 계속 다시
            # 푸느라 CPU 를 다 쓰고 녹화가 0.6 FPS 까지 떨어진다.
            key = (data.image.sec, data.image.nsec)
            if key == last_key:
                time.sleep(0.002)
                continue
            last_key = key

            jpeg = data.image.data

            # ROI/lib 의 Camera.parsing() 은 패킷당 64979B 짜리 청크를 버퍼에
            # 계속 이어붙이다가 tail=='EI' 인 패킷을 만나야 finalize 하고 버퍼를
            # 비운다. 정상 프레임은 3패킷 = 194937B 다. 그런데 한 프레임의 tail
            # 패킷을 네트워크가 흘리면 버퍼가 리셋되지 않은 채 **다음 프레임의
            # 패킷들과 이어붙는다** — 실측으로 5패킷(324895B) 이 붙은 경우를
            # 봤다. 이 상태는 JPEG 헤더와 앞부분 스캔라인은 멀쩡해서 imdecode 가
            # 예외 없이 성공하지만, 프레임 경계가 어긋난 지점부터는 디코더가
            # 남은 매크로블록을 회색(128 근방)으로 채운다 — 화면 하단이 통째로
            # 평탄한 회색이 되는 버그가 바로 이것이다. ROI 는 건드리지 않으므로
            # 여기서 길이로 걸러낸다: 정상 프레임은 항상 64979 의 정수배이고
            # 이 카메라 해상도/화질에서는 최대 3패킷이다. 그걸 벗어나면 프레임
            # 경계가 섞인 것이므로 통째로 버린다.
            PACKET_SIZE = 64979
            if len(jpeg) % PACKET_SIZE != 0 or len(jpeg) > 3 * PACKET_SIZE:
                continue

            if jpeg[:2] != b'\xff\xd8':
                continue
            eoi = jpeg.rfind(b'\xff\xd9')
            if eoi < 0:
                continue
            jpeg = jpeg[:eoi + 2]

            buf = np.frombuffer(jpeg, dtype=np.uint8)
            if buf.size == 0:
                continue

            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                continue

            with _frame_lock:
                _latest_frame = image
                _latest_frame_key = key

        except (AttributeError, ValueError, OSError, cv2.error) as ex:
            print(f"[camera] recoverable error: {ex}")
            time.sleep(0.1)
        except Exception as ex:
            print(f"[camera] unexpected error: {ex}")
            traceback.print_exc()
            raise


def status_worker(ip, port):
    """자차 상태는 소켓을 직접 읽는다 — 한 패킷도 버리지 않기 위해서다.

    `Receiver` 는 수신 프로세스 → 큐 → `parsed_data` 한 칸 구조라 **최신값
    하나만** 남는다. 폴링 사이에 도착한 샘플은 덮어써져 사라진다. 실측으로
    시뮬레이터는 23.7Hz 로 보내는데 Receiver 경유로는 11.7Hz 밖에 못 건졌다 —
    절반 손실이다. 카메라 프레임은 ~10Hz 라 상태가 프레임보다 촘촘해야
    프레임 시각으로 보간할 수 있는데, 절반을 잃으면 보간 구간이 벌어져
    그만큼 라벨이 흔들린다 (급커브에서 특히).
    """
    global _latest_status
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 ** 20)
    sock.bind((ip, port))
    data = EgoVehicleStatus()
    size = ctypes.sizeof(data)

    while True:
        try:
            raw, _ = sock.recvfrom(size)
            if len(raw) < size:
                continue
            ctypes.memmove(ctypes.addressof(data), raw, size)

            link_id = getattr(data, "link_id", b"")
            if isinstance(link_id, bytes):
                link_id = link_id.split(b"\x00")[0].decode("utf-8", errors="replace")

            snapshot = {
                "pos_x": float(data.pos_x),
                "pos_y": float(data.pos_y),
                "pos_z": float(data.pos_z),
                "yaw": _wrap180(float(data.yaw)),
                # 급커브에서 차체가 기울면 카메라도 같이 기운다.
                "roll": _wrap180(float(data.roll)),
                "pitch": _wrap180(float(data.pitch)),
                # 시뮬레이터 시각. 프레임과 짝을 맞추는 기준이다.
                "status_sec": int(data.sec),
                "status_nsec": int(data.nsec),
                "signed_vel": float(data.signed_vel),
                "ang_vel_z": float(data.ang_vel_z),
                "steer": float(data.steer),
                "accel": float(data.accel),
                "brake": float(data.brake),
                "gear": int(data.gear),
                "link_id": link_id,
            }
            t_sim = snapshot["status_sec"] + snapshot["status_nsec"] * 1e-9

            with _status_lock:
                _latest_status = snapshot
                if not _status_hist or t_sim > _status_hist[-1][0]:
                    _status_hist.append((t_sim, snapshot))

        except (AttributeError, ValueError, OSError) as ex:
            print(f"[status] recoverable error: {ex}")
            time.sleep(0.1)
        except Exception as ex:
            print(f"[status] unexpected error: {ex}")
            traceback.print_exc()
            raise


def objectinfo_worker(ip, port):
    # HD맵 투영 라벨은 "가려짐"을 모른다. 여기서 objects 를 안 받아두면
    # GenerateLabels.py 가 그 프레임의 ignore 마스크를 만들 수 없다.
    global _latest_objects
    receiver = Receiver(ip, port, ObjectInfo())

    while True:
        try:
            data = receiver.get_data()
            if data is None:
                time.sleep(0.01)
                continue

            objects = []
            for obj in data.data:
                if obj.obj_id == 0:
                    break
                objects.append({
                    "pose_x": float(obj.pose_x),
                    "pose_y": float(obj.pose_y),
                    "pose_z": float(obj.pose_z),
                    "heading": float(obj.heading),
                    "size_x": float(obj.size_x),
                    "size_y": float(obj.size_y),
                    "size_z": float(obj.size_z),
                    "objType": int(obj.objType),
                })

            with _objects_lock:
                _latest_objects = objects

        except (AttributeError, ValueError, OSError) as ex:
            print(f"[objectinfo] recoverable error: {ex}")
            time.sleep(0.1)
        except Exception as ex:
            print(f"[objectinfo] unexpected error: {ex}")
            traceback.print_exc()
            raise


def writer_worker(q, video_path, frames_dir, frame_ext, fps, save_video, state):
    """mp4/PNG 쓰기를 수신 경로에서 떼어낸다.

    같은 스레드에서 쓰면 인코딩+디스크(1280x720 PNG 약 115ms)가 도는 동안
    Receiver 의 큐가 밀려 프레임 수신 자체가 느려진다(실측 5.7 -> 1.3 FPS).
    VideoWriter 는 스레드 하나에서만 써야 하므로 생성도 여기서 한다.
    """
    writer = None
    try:
        while True:
            job = q.get()
            if job is None:
                break
            frame, idx, do_frame = job
            if save_video:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        state["error"] = f"영상 파일을 열 수 없습니다: {video_path}"
                        break
                writer.write(frame)
            if do_frame:
                path = os.path.join(frames_dir, f"{idx:06d}{frame_ext}")
                if frame_ext == ".jpg":
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(path, frame)
                state["saved"] += 1
    except Exception as ex:
        state["error"] = f"{ex}"
        traceback.print_exc()
    finally:
        if writer is not None:
            writer.release()
        state["done"] = True


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
                            ("차량 상태", args.status_ip, args.status_port),
                            ("ObjectInfo", args.objectinfo_ip, args.objectinfo_port)):
        if label == "차량 상태" and args.no_status:
            continue
        if label == "ObjectInfo" and args.no_objectinfo:
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

    frame_ext = ".png" if args.frame_format == "png" else ".jpg"
    # 큐가 무한히 자라면 메모리를 다 먹는다(프레임 하나가 2.7MB). 상한에 닿으면
    # 그때는 수신 쪽이 기다린다 — 데이터를 조용히 버리는 것보다 낫다.
    write_q = queue.Queue(maxsize=48)
    write_state = {"saved": 0, "error": None, "done": False}
    writer_thread = threading.Thread(
        target=writer_worker,
        args=(write_q, video_path, frames_dir, frame_ext, args.fps,
              save_video, write_state),
        daemon=True)
    writer_thread.start()

    threading.Thread(target=camera_worker,
                     args=(args.cam_ip, args.cam_port), daemon=True).start()
    if not args.no_status:
        threading.Thread(target=status_worker,
                         args=(args.status_ip, args.status_port), daemon=True).start()
    if not args.no_objectinfo:
        threading.Thread(target=objectinfo_worker,
                         args=(args.objectinfo_ip, args.objectinfo_port), daemon=True).start()

    print("=" * 60)
    print(f"📹 녹화 준비 — {out_dir}")
    print(f"   카메라 {args.cam_ip}:{args.cam_port}", end="")
    print("  (차량 상태 없음)" if args.no_status
          else f"  차량 상태 {args.status_ip}:{args.status_port}", end="")
    print("  (ObjectInfo 없음)" if args.no_objectinfo
          else f"  ObjectInfo {args.objectinfo_ip}:{args.objectinfo_port}")
    print("   ESC 또는 Ctrl+C 로 종료")
    print("=" * 60)
    print("첫 프레임 대기 중...")

    meta_path = os.path.join(out_dir, "meta.jsonl")
    idx = 0
    started = None
    last_key = None
    status_seen = False
    objectinfo_seen = False
    link_ids = set()
    last_png_pos = None
    no_pos_warned = False
    extrapolated = 0

    try:
        with open(meta_path, "w", encoding="utf-8") as meta_file:
            while True:
                # 같은 프레임을 두 번 저장하지 않는다. 픽셀 비교(1280x720x3)
                # 대신 카메라가 붙여준 키로 판정한다 — 프레임마다 2.7MB 를
                # 비교하면 그것만으로 CPU 를 잡아먹는다.
                with _frame_lock:
                    if _latest_frame is None or _latest_frame_key == last_key:
                        frame = None
                    else:
                        frame = _latest_frame.copy()
                        last_key = _latest_frame_key
                        cam_key = _latest_frame_key

                if frame is None:
                    time.sleep(0.005)
                    continue

                now = time.time()
                if started is None:
                    started = now
                    print("녹화 시작.")

                # 프레임의 시뮬레이터 시각으로 자차 상태를 보간해서 붙인다.
                # "지금 최신 상태"를 붙이면 급커브에서 라벨이 차선 밖으로 나간다.
                t_cam = cam_key[0] + cam_key[1] * 1e-9
                status, extrap = interp_status(t_cam)
                status = dict(status) if status else {}
                if status:
                    status_seen = True
                    if status.get("link_id"):
                        link_ids.add(status["link_id"])
                    raw_dt = t_cam - (status["status_sec"]
                                      + status["status_nsec"] * 1e-9)
                    status["pose_dt"] = round(raw_dt, 4)
                    if extrap:
                        # 보간 구간 밖 = 이력에 없는 시각. 값을 신뢰하기 어렵다.
                        status["pose_extrapolated"] = round(extrap, 4)
                        extrapolated += 1

                if not args.no_objectinfo:
                    with _objects_lock:
                        objects = list(_latest_objects) if _latest_objects else []
                    if objects:
                        objectinfo_seen = True
                    status["objects"] = objects

                # 학습용 프레임은 이동 거리 기준으로 고른다. 매 프레임 저장하면
                # 1280x720 PNG 인코딩+디스크(약 115ms)가 수신 스레드를 밀어내
                # 프레임 수신 자체가 느려진다(실측 5.7 -> 1.3 FPS). 거리 기준은
                # 렉과 무관하게 구간이 균등해지고, 정차 중 같은 그림이 쌓이는
                # 것도 막는다.
                save_this = False
                if save_frames:
                    pos = (status.get("pos_x"), status.get("pos_y"))
                    if args.png_interval_m <= 0:
                        save_this = True
                    elif pos[0] is None:
                        save_this = True          # 위치를 모르면 일단 남긴다
                        if not no_pos_warned:
                            print("  [주의] 차량 위치가 없어 거리 기준 샘플링이 "
                                  "안 됩니다. 모든 프레임을 저장합니다.")
                            no_pos_warned = True
                    elif last_png_pos is None:
                        save_this = True
                    else:
                        dx = pos[0] - last_png_pos[0]
                        dy = pos[1] - last_png_pos[1]
                        save_this = (dx * dx + dy * dy) >= args.png_interval_m ** 2
                    if save_this and pos[0] is not None:
                        last_png_pos = pos

                # 파일명은 idx 를 그대로 쓴다 — meta.jsonl 의 idx 와 짝을 맞추는
                # 근거가 이 번호다 (일부 프레임만 저장하므로 연번이 아니라 idx).
                if save_this:
                    status["png"] = True
                if write_state["error"]:
                    print(f"[에러] 저장 실패 — {write_state['error']}")
                    break
                write_q.put((frame, idx, save_this))

                # 카메라 프레임의 시뮬레이터 시각. 자차 상태의 status_sec/nsec 과
                # 짝이 맞는지 나중에 검증할 수 있어야 한다 — 급커브에서는 수십 ms
                # 차이도 라벨을 눈에 띄게 밀어낸다.
                rec = {"idx": idx, "t": round(now - started, 3),
                       "cam_sec": int(cam_key[0]), "cam_nsec": int(cam_key[1]),
                       **status}
                meta_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
        # 큐에 남은 프레임을 마저 쓴다. 여기서 안 기다리면 마지막 몇 장이
        # 파일에 없는데 meta.jsonl 에는 있는 상태가 된다.
        pending = write_q.qsize()
        if pending:
            print(f"저장 대기 중인 {pending}장을 마저 씁니다...")
        write_q.put(None)
        writer_thread.join(timeout=120)
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
        print(f"   프레임: {frames_dir}  ({write_state['saved']}장 저장"
              f"{'' if args.png_interval_m <= 0 else f', {args.png_interval_m}m 간격'})")
    print(f"   메타  : {meta_path}")
    if args.no_status:
        print("   차량 상태: 기록 안 함 (--no-status)")
    elif status_seen:
        print(f"   link_id {len(link_ids)}종 관측 — 차로 변경 라벨로 쓸 수 있습니다.")
    else:
        print(f"   [경고] 차량 상태가 한 번도 안 들어왔습니다 "
              f"({args.status_ip}:{args.status_port}).")
        print("          --status-ip 127.0.0.1 로 다시 시도해 보세요.")
    if extrapolated:
        print(f"   [주의] {extrapolated}/{idx} 프레임은 상태 이력 밖이라 보간하지 못하고 "
              f"가장 가까운 값을 썼습니다 (pose_extrapolated 참고).")
    if args.no_objectinfo:
        print("   ObjectInfo: 기록 안 함 (--no-objectinfo) — 가려짐 라벨을 만들 수 없습니다.")
    elif objectinfo_seen:
        print("   ObjectInfo 수신 확인 — 가려짐(ignore) 라벨 생성에 쓸 수 있습니다.")
    else:
        print(f"   [경고] ObjectInfo 가 한 번도 안 들어왔습니다 "
              f"({args.objectinfo_ip}:{args.objectinfo_port}). "
              "이 녹화본은 가려짐 라벨을 만들 수 없습니다.")
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
    parser.add_argument("--objectinfo-ip", default=None,
                        help="ObjectInfo IP (기본: 카메라와 동일)")
    parser.add_argument("--objectinfo-port", type=int, default=DEFAULT_OBJECTINFO_PORT)
    parser.add_argument("--no-objectinfo", action="store_true",
                        help="ObjectInfo 없이 녹화 (가려짐 라벨을 못 만들게 된다)")
    parser.add_argument("--format", choices=("video", "frames", "both"), default="video",
                        help="저장 형식 (기본 video: drive.mp4)")
    parser.add_argument("--frame-format", choices=("png", "jpg"), default="png",
                        help="--format frames/both 일 때 프레임 파일 형식 "
                             "(기본 png: 무손실, 얇은 차선 보존. jpg는 q95)")
    parser.add_argument("--png-interval-m", type=float, default=4.0,
                        help="학습용 프레임을 몇 m 이동할 때마다 저장할지 (기본 4.0). "
                             "0 이면 매 프레임. mp4 와 meta.jsonl 은 항상 전 프레임 기록")
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
    if args.objectinfo_ip is None:
        args.objectinfo_ip = DEFAULT_OBJECTINFO_IP or args.cam_ip
    return args


if __name__ == "__main__":
    record(parse_args())
