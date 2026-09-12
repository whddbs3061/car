#!/usr/bin/env python3
"""시뮬레이터 IMU UDP 수신 — **촬영 시각에 맞춰** 자세를 꺼내 쓴다.

`morai_camera.py` 와 같은 구조다. 다만 IMU 는 "지금 값" 이 아니라 **그 프레임이
찍힌 순간의 값**을 써야 하므로 링버퍼에 시각과 함께 쌓아 둔다.

===========================================================================
왜 "최신값" 을 쓰면 안 되는가
===========================================================================
실측: 촬영에서 우리가 받기까지 카메라는 중앙값 128ms, IMU 는 50~60ms 다.
**IMU 가 70ms 쯤 먼저 도착한다.** 프레임을 처리할 때 "지금 IMU" 를 쓰면 그
프레임보다 70~130ms 미래의 자세를 적용하게 된다.

요레이트 0.3 rad/s 인 보통 커브에서 100ms 는 0.03 rad 이고, 25m 앞에서
**0.75m** 차이다. 보정하려다 더 틀리는 양이다.

카메라와 IMU 가 **같은 시계**(`sec`/`nsec`)를 쓰므로, 프레임의 촬영 시각으로
버퍼를 찾아 쓰면 이 문제가 통째로 사라진다.

===========================================================================
자세를 그대로 쓰면 안 된다 (실측)
===========================================================================
IMU 가 주는 것은 **중력 기준** 자세다. `s04` 의 지면 모델이 필요한 것은
**노면 기준** 자세다. 차가 경사로에 있으면 둘이 노면 경사만큼 다르다.

    카메라-노면 pitch  =  차체-중력 pitch  -  노면 경사
                          (IMU 가 줌)         (모름. 맵이나 LiDAR 가 필요)

실측으로 확인했다. 정차 중인 차의 IMU pitch 가 **-3.52도** 였는데(평지에 선
차라면 0 근처여야 한다), 그 값을 그대로 지면 모델에 넣으면 차로 폭이 4.11m 로
나온다. 지도 실측값은 3.5m 다. 100프레임 스윕 결과:

    pitch     차로폭 중앙   표준편차
     0.00       3.205       0.277      <- 자세를 무시했을 때
    -1.00       3.457       0.215      <- 지도값에 가장 가깝다
    -2.00       3.708       0.177      <- 산포가 가장 작다
    -3.52       4.110       0.416      <- IMU 값 그대로. 과보정

그래서 **절대 자세가 아니라 기준선 대비 변화분**을 쓴다. 급제동 다이브나 가속
스쿼트 같은 빠른 변화는 IMU 가 정확히 알고, 노면 경사 같은 느린 성분은 기준선에
흡수된다. 느린 트림은 나중에 7~8단계(차로 폭 자기보정)가 영상만으로 잡는다.

`baseline_tau_s` 로 기준선이 따라가는 속도를 정한다. 짧게 잡으면 실제 pitch
변화까지 기준선이 먹어 버리고, 길게 잡으면 경사 구간에 들어갔을 때 한동안
과보정한다.
"""

import math
import os
import sys
import threading
import time
from collections import deque

# ROI 저장소의 lib 를 읽기 전용으로 재사용한다 (morai_camera.py 와 같은 방식).
# 상대 경로 단계 수를 고정하지 않는다 - 폴더를 옮기면 조용히 깨진다.
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
        "경로를 지정하세요 (IMU UDP 수신에 ROI/lib 의 Receiver 를 씁니다).")
for _p in (_roi_root, os.path.join(_roi_root, "lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lib.define.IMU import IMU              # noqa: E402
from lib.network.UDP import Receiver        # noqa: E402

DEFAULT_IP = os.environ.get("MORAI_IMU_IP", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("MORAI_IMU_PORT", "4001"))

BUFFER_SEC = 2.0            # 이만큼의 과거를 들고 있는다 (카메라 지연의 10배 넘게)
BASELINE_TAU_S = 20.0       # 기준선이 따라가는 시정수 (위 주석 참고)


def _rpy(w, x, y, z):
    """쿼터니언 -> (roll, pitch, yaw) 라디안."""
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _wrap(a):
    """각도를 (-pi, pi] 로."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class ImuStream:
    """시각이 붙은 IMU 링버퍼.

        imu = ImuStream().start()
        att = imu.at(frame_stamp)          # 그 프레임 시각의 (roll, pitch, yaw)
        dpsi = imu.delta_yaw(t_prev, t_now)
    """

    def __init__(self, ip=DEFAULT_IP, port=DEFAULT_PORT,
                 buffer_sec=BUFFER_SEC, baseline_tau_s=BASELINE_TAU_S):
        self.ip, self.port = ip, port
        self.buffer_sec = buffer_sec
        self.baseline_tau_s = baseline_tau_s
        self._buf = deque()                 # (stamp, roll, pitch, yaw)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._base = None                   # (roll, pitch) 기준선
        self.n_packets = 0

    def start(self):
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def wait_first(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._buf:
                    return True
            time.sleep(0.02)
        return False

    # --- 조회 -------------------------------------------------------------
    def at(self, stamp):
        """그 시각의 (roll, pitch, yaw) 라디안. 버퍼가 비면 None.

        앞뒤 샘플을 선형 보간한다. 버퍼 범위를 벗어나면 가장 가까운 끝을 준다 -
        외삽하지 않는다. IMU 는 100Hz 쯤 오므로 보간 오차가 작다.
        """
        with self._lock:
            if not self._buf:
                return None
            buf = list(self._buf)
        if stamp <= buf[0][0]:
            return buf[0][1:]
        if stamp >= buf[-1][0]:
            return buf[-1][1:]
        lo, hi = 0, len(buf) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if buf[mid][0] <= stamp:
                lo = mid
            else:
                hi = mid
        t0, r0, p0, y0 = buf[lo]
        t1, r1, p1, y1 = buf[hi]
        a = 0.0 if t1 <= t0 else (stamp - t0) / (t1 - t0)
        return (r0 + a * _wrap(r1 - r0),
                p0 + a * _wrap(p1 - p0),
                y0 + a * _wrap(y1 - y0))

    def delta_yaw(self, t0, t1):
        """두 시각 사이의 **요 변화**(rad). 못 구하면 None.

        각속도를 적분하지 않고 **자세 차이**를 쓴다. 적분은 드리프트가 쌓이는데
        쿼터니언 차이는 그렇지 않다.
        """
        a, b = self.at(t0), self.at(t1)
        if a is None or b is None:
            return None
        return _wrap(b[2] - a[2])

    def attitude_deg(self, stamp, relative=True):
        """`s04` 에 넣을 (pitch, roll) **도** 단위. 없으면 None.

        `relative=True` 면 기준선 대비 변화분이다 (기본값). 절대 자세를 그대로
        쓰면 노면 경사까지 차체 기울기로 오해해 과보정한다 - 머리말의 실측 참고.
        """
        a = self.at(stamp)
        if a is None:
            return None
        roll, pitch = a[0], a[1]
        if relative:
            if self._base is None:
                return 0.0, 0.0
            roll -= self._base[0]
            pitch -= self._base[1]
        return math.degrees(pitch), math.degrees(roll)

    # --- 수신 -------------------------------------------------------------
    def _worker(self):
        receiver = Receiver(self.ip, self.port, IMU())
        last_key = None
        while not self._stop.is_set():
            try:
                d = receiver.get_data()
                if d is None or not hasattr(d, "ori_w"):
                    time.sleep(0.002)
                    continue
                key = (d.sec, d.nsec)
                if key == last_key or (d.sec == 0 and d.nsec == 0):
                    time.sleep(0.001)
                    continue
                last_key = key
                stamp = float(d.sec) + float(d.nsec) * 1e-9
                roll, pitch, yaw = _rpy(d.ori_w, d.ori_x, d.ori_y, d.ori_z)

                with self._lock:
                    self._buf.append((stamp, roll, pitch, yaw))
                    cut = stamp - self.buffer_sec
                    while self._buf and self._buf[0][0] < cut:
                        self._buf.popleft()
                    self.n_packets += 1
                    # 기준선을 천천히 따라가게 한다 (느린 성분 = 노면 경사)
                    if self._base is None:
                        self._base = (roll, pitch)
                    else:
                        a = min(1.0, 0.01 / max(self.baseline_tau_s, 1e-3))
                        self._base = (self._base[0] + a * _wrap(roll - self._base[0]),
                                      self._base[1] + a * _wrap(pitch - self._base[1]))
            except Exception:
                time.sleep(0.01)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IMU 수신 확인")
    ap.add_argument("--ip", default=DEFAULT_IP)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--seconds", type=float, default=10.0)
    a = ap.parse_args()
    imu = ImuStream(a.ip, a.port).start()
    print(f"[imu] {a.ip}:{a.port} 대기 중...")
    if not imu.wait_first(timeout=10.0):
        raise SystemExit("IMU 패킷이 안 옵니다. 포트를 확인하세요.")
    t0 = time.time()
    prev = None
    while time.time() - t0 < a.seconds:
        now = time.time()
        att = imu.at(now - 0.13)          # 카메라 지연만큼 과거를 본다
        if att:
            d = imu.delta_yaw(prev, now - 0.13) if prev else 0.0
            prev = now - 0.13
            rel = imu.attitude_deg(now - 0.13)
            print(f"roll {math.degrees(att[0]):+6.2f}  pitch {math.degrees(att[1]):+6.2f}  "
                  f"yaw {math.degrees(att[2]):+7.2f}  |  기준선대비 pitch {rel[0]:+5.2f} "
                  f"roll {rel[1]:+5.2f}  |  dyaw {math.degrees(d or 0):+6.3f}deg  "
                  f"({imu.n_packets} pkt)")
        time.sleep(0.3)
    imu.stop()
