import ctypes

import sys
import os

# 현재 파일(LineDetect.py)이 있는 Sensor 폴더의 부모(jongyun) 및 lib 경로를 sys.path에 추가
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

class RoadLaneDetector:
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height
        self.img_center = img_width / 2.0
        
        self.left_m = 0.0
        self.right_m = 0.0
        self.left_b = (0, 0)
        self.right_b = (0, 0)
        self.left_detect = False
        self.right_detect = False

        # 관심 영역(ROI) 파라미터 (하단 51% 활용)
        self.trap_bottom_width = 0.85
        self.trap_top_width = 0.07
        self.trap_height = 0.49

    def filter_colors(self, img_frame):

        img_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        

        h, s, v = cv2.split(img_hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        

        img_hsv_clahe = cv2.merge((h, s, v_clahe))


        lower_white = np.array([0, 0, 140])
        upper_white = np.array([179, 40, 255])
        lower_yellow = np.array([10, 80, 80])
        upper_yellow = np.array([40, 255, 255])

        white_mask = cv2.inRange(img_hsv_clahe, lower_white, upper_white)
        yellow_mask = cv2.inRange(img_hsv_clahe, lower_yellow, upper_yellow)
        
        white_image = cv2.bitwise_and(img_frame, img_frame, mask=white_mask)
        yellow_image = cv2.bitwise_and(img_frame, img_frame, mask=yellow_mask)

        output = cv2.addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0)
        return output

    def limit_region(self, img_edges):
        h, w = img_edges.shape[:2]
        mask = np.zeros_like(img_edges)

        points = np.array([[
            int((w * (1 - self.trap_bottom_width)) / 2), h,
            int((w * (1 - self.trap_top_width)) / 2), int(h - h * self.trap_height),
            int(w - (w * (1 - self.trap_top_width)) / 2), int(h - h * self.trap_height),
            int(w - (w * (1 - self.trap_bottom_width)) / 2), h
        ]], dtype=np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(mask, [points], 255)
        output = cv2.bitwise_and(img_edges, mask)
        return output

    def hough_lines(self, img_mask):
        lines = cv2.HoughLinesP(img_mask, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=20)
        return lines if lines is not None else []

    def separate_line(self, img_edges, lines):
        left_lines = []
        right_lines = []
        selected_lines = []
        slopes = []
        slope_thresh = 0.3

        self.left_detect = False
        self.right_detect = False

        if lines is None or len(lines) == 0:
            return [right_lines, left_lines]

        for line in lines:

            if len(line) == 0:
                continue
            x1, y1, x2, y2 = line[0] if isinstance(line[0], (np.ndarray, list, tuple)) else line
            
            slope = (float(y2) - float(y1)) / (float(x2) - float(x1) + 0.00001)

            if abs(slope) > slope_thresh:
                slopes.append(slope)
                selected_lines.append([x1, y1, x2, y2])

        for i, line in enumerate(selected_lines):
            x1, y1, x2, y2 = line
            if slopes[i] > 0 and x2 > self.img_center and x1 > self.img_center:
                right_lines.append(line)
                self.right_detect = True
            elif slopes[i] < 0 and x2 < self.img_center and x1 < self.img_center:
                left_lines.append(line)
                self.left_detect = True

        return [right_lines, left_lines]
    def regression(self, separated_lines, img_input):
        h, w = img_input.shape[:2]
        right_pts = []
        left_pts = []

        if self.right_detect:
            for line in separated_lines[0]:
                right_pts.append((line[0], line[1]))
                right_pts.append((line[2], line[3]))
            if right_pts:
                vx, vy, x0, y0 = cv2.fitLine(np.array(right_pts, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                self.right_m = vy / (vx + 0.00001)
                self.right_b = (int(x0), int(y0))

        if self.left_detect:
            for line in separated_lines[1]:
                left_pts.append((line[0], line[1]))
                left_pts.append((line[2], line[3]))
            if left_pts:
                vx, vy, x0, y0 = cv2.fitLine(np.array(left_pts, dtype=np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
                self.left_m = vy / (vx + 0.00001)
                self.left_b = (int(x0), int(y0))

        ini_y = h
        fin_y = int(h * (1.0 - self.trap_height))

        right_ini_x = int(((ini_y - self.right_b[1]) / (self.right_m + 0.00001)) + self.right_b[0])
        right_fin_x = int(((fin_y - self.right_b[1]) / (self.right_m + 0.00001)) + self.right_b[0])

        left_ini_x = int(((ini_y - self.left_b[1]) / (self.left_m + 0.00001)) + self.left_b[0])
        left_fin_x = int(((fin_y - self.left_b[1]) / (self.left_m + 0.00001)) + self.left_b[0])

        return [
            (right_ini_x, ini_y),
            (right_fin_x, fin_y),
            (left_ini_x, ini_y),
            (left_fin_x, fin_y)
        ]

    def predict_dir(self):
        thres_vp = 10
        vx = (self.right_m * self.right_b[0] - self.left_m * self.left_b[0] - self.right_b[1] + self.left_b[1]) / (self.right_m - self.left_m + 0.00001)

        if vx < self.img_center - thres_vp:
            return "Left Turn"
        elif vx > self.img_center + thres_vp:
            return "Right Turn"
        else:
            return "Straight"

    def draw_line(self, img_input, lane, dir_str):
        output = img_input.copy()
        poly_points = np.array([lane[2], lane[0], lane[1], lane[3]], dtype=np.int32)

        cv2.fillPoly(output, [poly_points], (0, 230, 30))
        cv2.addWeighted(output, 0.3, img_input, 0.7, 0, img_input)

        cv2.line(img_input, lane[0], lane[1], (0, 255, 255), 5, cv2.LINE_AA)
        cv2.line(img_input, lane[2], lane[3], (255, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img_input, dir_str, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        return img_input


# 공통 IP 설정 및 카메라 포트 설정
IP = "192.168.0.200"
PORT = 1101  # Cam 1 포트

latest_frame = None
frame_lock = threading.Lock()

def camera_thread_worker(ip, port):
    global latest_frame
    cam_receiver = Receiver(ip, port, Camera())
    
    while True:
        try:
            data = cam_receiver.get_data()
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

            display_img = cv2.resize(image, (640, 480))

            with frame_lock:
                latest_frame = display_img

        except Exception:
            continue

def main():
    # UDP 수신기 멀티스레드 시작
    t = threading.Thread(target=camera_thread_worker, args=(IP, PORT), daemon=True)
    t.start()
    time.sleep(1)

    # 디텍터 인스턴스 생성 (640x480 기준 중앙점 설정)
    detector = RoadLaneDetector(640, 480)

    print("="*50)
    print("🚀 실시간 MORAI 차선 인식 프로그램이 실행되었습니다.")
    print("종료하려면 결과 창을 선택하고 ESC 키를 누르세요.")
    print("="*50)

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                img_frame = latest_frame.copy()

            # 1. 색상 필터링 (흰색/노란색)
            img_filter = detector.filter_colors(img_frame)

            # 2. Grayscale 변환
            img_gray = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)

            # 3. Canny Edge Detection
            img_edges = cv2.Canny(img_gray, 50, 150)

            # 4. 관심 영역(ROI) 지정
            img_mask = detector.limit_region(img_edges)

            # 5. Hough 변환으로 직선 성분 추출
            lines = detector.hough_lines(img_mask)

            img_result = img_frame.copy()

            if len(lines) > 0:
                # 6. 좌우 차선 분리 및 선형 회귀
                separated_lines = detector.separate_line(img_mask, lines)
                lane = detector.regression(separated_lines, img_frame)

                # 7. 진행 방향 예측
                dir_str = detector.predict_dir()

                # 8. 최종 결과 시각화
                img_result = detector.draw_line(img_frame, lane, dir_str)

            # 9. 결과 영상 출력 (imshow 딱 1개만 실행)
            cv2.imshow("Lane Detection Result", img_result)

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
