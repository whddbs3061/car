"""MORAI 시뮬레이터 카메라 UDP 수신 — live_overlay.py / live_output.py 공용.

`RecordDrive.camera_worker` 와 같은 방식이다. 별도 스레드가 최신 프레임 하나만
들고 있고, 본 루프는 필요할 때 그걸 집어간다 (프레임 큐를 쌓지 않는다 —
실시간 처리에서는 늦은 프레임을 미루는 것보다 버리는 게 맞다).

**같은 프레임을 다시 디코드하지 않는다.** `Receiver.get_data()` 는 폴링마다
같은 객체를 돌려주는데(초당 150회 이상), 키를 안 보고 매번 imdecode 하면
같은 1280x720 JPEG 을 계속 다시 푸느라 CPU 를 다 쓴다 (RecordDrive 에서
녹화가 0.6 FPS 까지 떨어졌던 원인).
"""

import os
import sys
import threading
import time

import cv2
import numpy as np

# ROI 저장소의 lib 를 읽기 전용으로 재사용한다 (RecordDrive 와 같은 방식).
# **상대 경로를 몇 단계 올라갈지 고정하지 않는다** - 이 파일이 폴더를 옮기면
# 그 숫자가 조용히 틀어져서 import 가 깨진다. ROI/lib 를 위로 올라가며 찾는다.
_here = os.path.dirname(os.path.abspath(__file__))


def _find_roi_root():
    env = os.environ.get("MORAI_ROI_ROOT")
    if env:
        return env
    d = _here
    for _ in range(8):
        cand = os.path.join(d, "ROI")
        if os.path.isdir(os.path.join(cand, "lib")):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


_roi_root = _find_roi_root()
if _roi_root is None:
    raise SystemExit(
        "ROI/lib 를 못 찾았습니다. MORAI_ROI_ROOT 환경변수로 ROI 저장소 "
        "경로를 지정하세요 (카메라 UDP 수신에 ROI/lib 의 Receiver 를 씁니다).")
for _p in (_roi_root, os.path.join(_roi_root, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.define.Camera import Camera          # noqa: E402
from lib.network.UDP import Receiver          # noqa: E402

# **이 IP 는 시뮬레이터 주소가 아니라 이쪽에서 bind 하는 로컬 주소다.**
# Receiver 가 socket.bind((ip, port)) 를 하기 때문에 시뮬레이터 PC 의 IP 를
# 넣으면 "Cannot assign requested address" 로 죽는다. 어느 인터페이스로
# 들어오든 받도록 0.0.0.0 으로 둔다.
DEFAULT_IP = os.environ.get("MORAI_CAM_IP", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("MORAI_CAM_PORT", "1101"))


class CameraStream:
    """최신 프레임 하나만 유지하는 UDP 카메라 수신기.

        cam = CameraStream().start()
        frame, seq = cam.latest()          # 아직 없으면 (None, -1)
    """

    def __init__(self, ip=DEFAULT_IP, port=DEFAULT_PORT):
        self.ip, self.port = ip, port
        self._frame = None
        self._key = None
        self._seq = -1
        self._stamp = 0.0               # 패킷에 실린 촬영 시각 (초)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.decode_errors = 0

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def latest(self, with_stamp=False):
        """최신 프레임. `with_stamp=True` 면 (frame, seq, stamp) 를 준다.

        `stamp` 는 시뮬레이터가 패킷에 넣은 **촬영 시각**(초, `time.time()` 과
        같은 epoch)이다. 받은 시각이 아니다.

        이 둘을 구분해야 하는 이유는 실측 때문이다 - 촬영에서 수신까지
        중앙값 128ms, p90 145ms 가 걸린다. IMU 같은 다른 센서를 이 프레임에
        맞출 때 "지금 시각" 을 쓰면 128ms 만큼 미래의 값을 쓰게 된다.
        요레이트 5도/s 인 완만한 커브에서도 0.64도 차이이고, 지면 투영에서
        자세 1도는 40m 에서 거리 45% 오차다 (s04 주석 실측).

        기본값을 False 로 둔 것은 기존 호출부(`frame, seq = cam.latest()`)를
        깨지 않기 위해서다.
        """
        with self._lock:
            if self._frame is None:
                return (None, -1, 0.0) if with_stamp else (None, -1)
            if with_stamp:
                return self._frame.copy(), self._seq, self._stamp
            return self._frame.copy(), self._seq

    def wait_first(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest()[0] is not None:
                return True
            time.sleep(0.05)
        return False

    def _worker(self):
        receiver = Receiver(self.ip, self.port, Camera())
        while not self._stop.is_set():
            try:
                data = receiver.get_data()
                if data is None or not hasattr(data, "image") or not data.image.data:
                    time.sleep(0.005)
                    continue

                jpeg = bytes(data.image.data)
                stamp = float(data.image.sec) + float(data.image.nsec) * 1e-9
                key = (len(jpeg), jpeg[:16], jpeg[-16:])
                if key == self._key:            # 같은 프레임 - 다시 풀지 않는다
                    time.sleep(0.002)
                    continue

                if jpeg[:2] != b"\xff\xd8":     # SOI 가 아니면 버린다
                    continue
                eoi = jpeg.rfind(b"\xff\xd9")
                if eoi < 0:
                    continue
                buf = np.frombuffer(jpeg[:eoi + 2], dtype=np.uint8)
                if buf.size == 0:
                    continue
                image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    self.decode_errors += 1
                    continue

                with self._lock:
                    self._stamp = stamp
                    self._frame = image
                    self._key = key
                    self._seq += 1
            except (AttributeError, ValueError, OSError, cv2.error) as ex:
                print(f"[camera] 복구 가능한 오류: {ex}")
                time.sleep(0.1)
