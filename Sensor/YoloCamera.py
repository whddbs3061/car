import ctypes

# X11 멀티스레드 충돌 방지 설정
try:
  X11 = ctypes.CDLL("libX11.so.6")
  X11.XInitThreads()
except Exception:
  pass

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO  # YOLO 모델 사용을 위한 라이브러리
from lib.define.Camera import Camera
from lib.network.UDP import Receiver

# 공통 IP 설정
IP = "192.168.0.200"

# 카메라별 포트 및 설정 정의
CAMERA_CONFIGS = [
    {"name": "Cam 1", "port": 1101},
    {"name": "Cam 2", "port": 1111},
    {"name": "Cam 3", "port": 1121},
    {"name": "Cam 4", "port": 1131},
]

# 4개 카메라의 최신 프레임을 공유하기 위한 딕셔너리와 락(Lock)
latest_frames = {}
frame_lock = threading.Lock()

# YOLO 모델 로드 (가장 가볍고 빠른 yolov8n.pt 기준, 필요시 다른 가중치로 변경 가능)
# 스레드별 병렬 추론을 위해 전역으로 로드하거나 각 스레드에서 로드할 수 있습니다.
model = YOLO("yolo11n.pt")


def camera_thread_worker(cam_name, port):
  """각 카메라별 독립적인 UDP 수신 및 YOLO 객체 검출 스레드 함수"""
  cam_data = Receiver(IP, port, Camera())

  # 초기 대기 화면 설정
  default_image = np.zeros((480, 640, 3), dtype=np.uint8)
  cv2.putText(
      default_image,
      f"{cam_name} Connecting...",
      (140, 240),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.8,
      (255, 255, 255),
      2,
  )

  with frame_lock:
    latest_frames[cam_name] = default_image

  while True:
    try:
      data = cam_data.get_data()
      if data is None:
        time.sleep(0.01)
        continue

      # 데이터 유효성 검사
      if not hasattr(data, "image") or not data.image.data:
        continue

      image_np = np.frombuffer(data.image.data, dtype=np.uint8)
      if image_np.size == 0:
        continue

      image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
      if image is None or image.size == 0:
        continue

      # 해상도 고정 
      image = cv2.resize(image, (640, 480))

      # YOLO 객체 검출 수행
      # verbose=False로 불필요한 로그 출력 제거, conf=0.5는 50% 이상 확신하는 객체만 검출, classes=[]는 COCO 데이터셋에서 각각 사람 자동ck 신호등 정지표시판 개에 해당하는 클래스 번호
      results = model(image, verbose=False, conf=0.5, classes=[0,2,9,11,16])

      # YOLO 추론 결과 시각화 (박스 및 라벨 자동 드로잉)
      annotated_frame = results[0].plot()

      # 카메라 이름 텍스트 오버레이 추가
      cv2.putText(
          annotated_frame,
          cam_name,
          (30, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.0,
          (0, 255, 0),
          2,
      )

      # 공유 딕셔너리에 최신 프레임 업데이트
      with frame_lock:
        latest_frames[cam_name] = annotated_frame

    except Exception:
      continue


def main():
  # 멀티 스레딩 시작
  for config in CAMERA_CONFIGS:
    t = threading.Thread(
        target=camera_thread_worker,
        args=(config["name"], config["port"]),
        daemon=True,
    )
    t.start()

  # 초기 프레임 수집 대기
  time.sleep(1.5)

  try:
    while True:
      with frame_lock:
        img1 = latest_frames.get(
            "Cam 1", np.zeros((480, 640, 3), dtype=np.uint8)
        )
        img2 = latest_frames.get(
            "Cam 2", np.zeros((480, 640, 3), dtype=np.uint8)
        )
        img3 = latest_frames.get(
            "Cam 3", np.zeros((480, 640, 3), dtype=np.uint8)
        )
        img4 = latest_frames.get(
            "Cam 4", np.zeros((480, 640, 3), dtype=np.uint8)
        )

      # 2x2 그리드로 이미지 병합
      top_row = np.hstack((img1, img2))
      bottom_row = np.hstack((img3, img4))
      grid_img = np.vstack((top_row, bottom_row))

      # 화면 크기 조절 (필요에 따라 해상도 조정)
      grid_img = cv2.resize(grid_img, (1280, 960))

      # 4분할 YOLO 검출 화면 출력
      cv2.imshow("MORAI YOLOv8 Multi-Camera Detection", grid_img)

      if cv2.waitKey(1) & 0xFF == ord("q"):
        break

  except KeyboardInterrupt:
    print("프로그램을 종료합니다.")
  finally:
    cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
