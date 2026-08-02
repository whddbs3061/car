import ctypes

# X11 멀티스레드 충돌 방지 설정
try:
    X11 = ctypes.CDLL("libX11.so.6")
    X11.XInitThreads()
except Exception:
    pass

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
lib_dir = os.path.join(parent_dir, 'lib')

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)
import threading
import time
import cv2
import numpy as np
from lib.define.Camera import Camera
from lib.network.UDP import Receiver

# 공통 IP 설정
IP = "192.168.0.200"

CAMERA_CONFIGS = [
    {"name": "Cam 1", "port": 1101},
    {"name": "Cam 2", "port": 1111},
    {"name": "Cam 3", "port": 1121},
    {"name": "Cam 4", "port": 1131},
]

latest_frames = {}
frame_lock = threading.Lock()

# 트랙바 조절용 더미 함수
def nothing(x):
    pass

def camera_thread_worker(cam_name, port):
    cam_data = Receiver(IP, port, Camera())
    
    while True:
        try:
            data = cam_data.get_data()
            if data is None:
                time.sleep(0.01)
                continue

            if not hasattr(data, "image") or not data.image.data:
                continue

            image_np = np.frombuffer(data.image.data, dtype=np.uint8)
            if image_np.size == 0:
                continue

            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                continue

            # 기본 해상도 리사이즈 (640x480)
            display_img = cv2.resize(image, (640, 480))

            # 공유 딕셔너리에 최신 프레임 업데이트
            with frame_lock:
                latest_frames[cam_name] = display_img

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

    time.sleep(1)  # 초기 수집 대기

    print("="*50)
    print("🛠️ 실시간 전처리 파라미터 튜닝 모드가 시작되었습니다.")
    print("튜닝할 창(Trackbar Window)에서 값을 조절하며 결과를 확인하세요.")
    print("종료하려면 터미널에서 Ctrl+C 또는 창을 누르고 ESC를 누르세요.")
    print("="*50)

    # 튜닝용 윈도우 생성 및 트랙바 부착
    window_name = "Lane Detection Tuning (Cam 1)"
    cv2.namedWindow(window_name)

    # HSV 및 Canny 임계값 트랙바 생성
    # 색상(H): 0~179, 채도(S): 0~255, 명도(V): 0~255
    cv2.createTrackbar("H_min", window_name, 0, 179, nothing)
    cv2.createTrackbar("H_max", window_name, 179, 179, nothing)
    cv2.createTrackbar("S_min", window_name, 0, 255, nothing)
    cv2.createTrackbar("S_max", window_name, 255, 255, nothing)
    cv2.createTrackbar("V_min", window_name, 200, 255, nothing)  # 기본 흰색/밝은선 기준 조절용
    cv2.createTrackbar("V_max", window_name, 255, 255, nothing)
    cv2.createTrackbar("Canny_T1", window_name, 50, 255, nothing)
    cv2.createTrackbar("Canny_T2", window_name, 150, 255, nothing)
    
    # ROI 하단 자르기 비율 조절 (0 ~ 100%)
    cv2.createTrackbar("ROI_Height_%", window_name, 50, 100, nothing)

    try:
        while True:
            with frame_lock:
                # 튜닝할 대상 카메라 프레임 가져오기 (기본 Cam 1)
                img = latest_frames.get("Cam 1")

            if img is not None:
                h, w, _ = img.shape

                # 1. 트랙바에서 현재 값 읽어오기
                h_min = cv2.getTrackbarPos("H_min", window_name)
                h_max = cv2.getTrackbarPos("H_max", window_name)
                s_min = cv2.getTrackbarPos("S_min", window_name)
                s_max = cv2.getTrackbarPos("S_max", window_name)
                v_min = cv2.getTrackbarPos("V_min", window_name)
                v_max = cv2.getTrackbarPos("V_max", window_name)
                t1 = cv2.getTrackbarPos("Canny_T1", window_name)
                t2 = cv2.getTrackbarPos("Canny_T2", window_name)
                roi_ratio = cv2.getTrackbarPos("ROI_Height_%", window_name)

                # 2. ROI 설정 (화면 하단부만 자르기)
                roi_start_y = int(h * (100 - roi_ratio) / 100)
                roi_img = img[roi_start_y:h, 0:w]

                # 3. 색상 필터링 (HSV 변환 및 마스킹)
                hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
                lower_bound = np.array([h_min, s_min, v_min])
                upper_bound = np.array([h_max, s_max, v_max])
                mask = cv2.inRange(hsv, lower_bound, upper_bound)

                # 4. Canny Edge 검출
                edges = cv2.Canny(mask, t1, t2)

                # 5. 시각화를 위해 결과 이미지를 원본 크기(ROI 영역)에 맞춰 합치기
                # 디버깅 편의를 위해 마스크 화면과 엣지 화면을 컬러로 변환해서 가로/세로로 띄움
                edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                # 원본 이미지에 ROI 영역 표시 선 그리기
                debug_img = img.copy()
                cv2.line(debug_img, (0, roi_start_y), (w, roi_start_y), (0, 0, 255), 2)

                # 창에 띄우기 (원본+ROI선 / 마스크 / 엣지)
                cv2.imshow("1. Original with ROI", debug_img)
                cv2.imshow("2. Color Mask", mask)
                cv2.imshow(window_name, edges_colored)

            # ESC 키 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
