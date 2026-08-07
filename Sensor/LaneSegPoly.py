import ctypes
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
import traceback
import cv2
import numpy as np

from lib.define.Camera import Camera
from lib.network.UDP import Receiver


class LaneSegPolyDetector:
    def __init__(self, img_width, img_height, seg_model_path=None):
        self.img_width = img_width
        self.img_height = img_height
        self.img_center = img_width / 2.0

        self.left_fit = None
        self.right_fit = None
        self.center_fit = None
        self.left_detect = False
        self.right_detect = False

        # Lane polynomial order settings
        self.left_right_fit_order = 2
        self.center_fit_order = 3

        self.trap_bottom_width = 0.85
        self.trap_top_width = 0.07
        self.trap_height = 0.49

        self.seg_net = self.load_segmentation_model(seg_model_path)

    def load_segmentation_model(self, model_path):
        if model_path is None:
            print("Segmentation model 경로가 지정되지 않았습니다.")
            return None

        model_path = os.path.join(current_dir, model_path) if not os.path.isabs(model_path) else model_path
        if not os.path.isfile(model_path):
            print(f"Segmentation model 파일을 찾을 수 없습니다: {model_path}")
            return None

        try:
            net = cv2.dnn.readNet(model_path)
            print(f"Segmentation model loaded: {model_path}")
            return net
        except Exception as e:
            print(f"Segmentation model 로드 실패: {e}")
            return None

    def color_mask(self, img_frame):
        hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_clahe = clahe.apply(v)
        hsv_clahe = cv2.merge((h, s, v_clahe))

        lower_white = np.array([0, 0, 140])
        upper_white = np.array([179, 50, 255])
        lower_yellow = np.array([10, 80, 80])
        upper_yellow = np.array([40, 255, 255])

        white_mask = cv2.inRange(hsv_clahe, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv_clahe, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(white_mask, yellow_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def segment_lanes(self, img_frame):
        if self.seg_net is None:
            return np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        blob = cv2.dnn.blobFromImage(
            img_frame,
            scalefactor=1.0 / 255.0,
            size=(self.img_width, self.img_height),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        self.seg_net.setInput(blob)
        output = self.seg_net.forward()

        if output.ndim == 4:
            if output.shape[1] == 1:
                lane_prob = output[0, 0, :, :]
            else:
                lane_prob = output[0, 1, :, :] if output.shape[1] > 1 else output[0, 0, :, :]
        elif output.ndim == 3:
            lane_prob = output[0, :, :]
        else:
            lane_prob = output.squeeze()

        lane_mask = (lane_prob > 0.5).astype(np.uint8) * 255
        lane_mask = cv2.resize(lane_mask, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return lane_mask

    def combine_masks(self, img_frame):
        seg_mask = self.segment_lanes(img_frame)
        color_mask = self.color_mask(img_frame)
        combined = cv2.bitwise_and(seg_mask, color_mask)

        if cv2.countNonZero(combined) < 1000:
            combined = cv2.bitwise_or(seg_mask, color_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        return combined

    def extract_center_points(self, binary_mask, step=5):
        h, w = binary_mask.shape[:2]
        top = int(h * (1.0 - self.trap_height))
        center_x = []
        center_y = []

        for y in range(h - 1, top - 1, -step):
            row = binary_mask[y]
            xs = np.where(row > 0)[0]
            if xs.size > 0:
                center_x.append(int(np.mean(xs)))
                center_y.append(y)

        if len(center_x) < 10:
            return np.array([]), np.array([])
        return np.array(center_x), np.array(center_y)

    def fit_centerline(self, binary_mask, order=None):
        center_x, center_y = self.extract_center_points(binary_mask)
        if center_x.size < 10 or center_y.size < 10:
            return None
        if order is None:
            order = self.center_fit_order
        try:
            return np.polyfit(center_y, center_x, order)
        except Exception as ex:
            print(f"centerline fit 실패: {ex}")
            traceback.print_exc()
            return None

    def compute_heading_curvature(self, fit, y_eval):
        if fit is None:
            return None, None

        if fit.size == 3:
            a, b, _ = fit
            dx_dy = 2 * a * y_eval + b
            d2x_dy2 = 2 * a
        elif fit.size == 4:
            a, b, c, _ = fit
            dx_dy = 3 * a * y_eval ** 2 + 2 * b * y_eval + c
            d2x_dy2 = 6 * a * y_eval + 2 * b
        else:
            return None, None

        heading = np.arctan(dx_dy)
        curvature = abs(d2x_dy2) / ((1 + dx_dy ** 2) ** 1.5 + 1e-8)
        return heading, curvature

    def get_lane_center_x(self, y_eval):
        if self.center_fit is not None:
            return float(np.polyval(self.center_fit, y_eval))

        if self.left_fit is not None and self.right_fit is not None:
            left_x = np.polyval(self.left_fit, y_eval)
            right_x = np.polyval(self.right_fit, y_eval)
            return float((left_x + right_x) / 2.0)

        return None

    def limit_region(self, img_mask):
        h, w = img_mask.shape[:2]
        mask = np.zeros_like(img_mask)

        points = np.array([[
            int((w * (1 - self.trap_bottom_width)) / 2), h,
            int((w * (1 - self.trap_top_width)) / 2), int(h - h * self.trap_height),
            int(w - (w * (1 - self.trap_top_width)) / 2), int(h - h * self.trap_height),
            int(w - (w * (1 - self.trap_bottom_width)) / 2), h
        ]], dtype=np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(mask, [points], 255)
        return cv2.bitwise_and(img_mask, mask)

    def find_lane_pixels(self, binary_warped):
        if binary_warped is None or binary_warped.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        margin = 100
        minpix = 50
        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            if good_left_inds.size > 0:
                left_lane_inds.append(good_left_inds)
            if good_right_inds.size > 0:
                right_lane_inds.append(good_right_inds)

            if good_left_inds.size > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if good_right_inds.size > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        if len(left_lane_inds) == 0 or len(right_lane_inds) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty

    def fit_polynomial(self, binary_warped):
        leftx, lefty, rightx, righty = self.find_lane_pixels(binary_warped)
        left_fit = None
        right_fit = None

        if leftx.size > 50 and lefty.size > 50:
            left_fit = np.polyfit(lefty, leftx, self.left_right_fit_order)
            self.left_detect = True
        else:
            self.left_detect = False

        if rightx.size > 50 and righty.size > 50:
            right_fit = np.polyfit(righty, rightx, self.left_right_fit_order)
            self.right_detect = True
        else:
            self.right_detect = False

        if left_fit is None and self.left_fit is not None:
            left_fit = self.left_fit
        if right_fit is None and self.right_fit is not None:
            right_fit = self.right_fit

        self.left_fit = left_fit
        self.right_fit = right_fit
        self.center_fit = self.fit_centerline(binary_warped, order=3)
        return left_fit, right_fit

    def draw_lane(self, img_input, left_fit, right_fit):
        h, w = img_input.shape[:2]
        overlay = img_input.copy()
        out_img = img_input.copy()

        ploty = np.linspace(int(h * (1.0 - self.trap_height)), h - 1, num=100)
        left_points = []
        right_points = []

        if left_fit is not None:
            left_points = [(int(np.polyval(left_fit, y)), int(y)) for y in ploty]
        if right_fit is not None:
            right_points = [(int(np.polyval(right_fit, y)), int(y)) for y in ploty]

        if left_points and right_points:
            pts = np.array(left_points + right_points[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, out_img, 0.7, 0, out_img)

        if left_points:
            cv2.polylines(out_img, [np.array(left_points, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=5)
        if right_points:
            cv2.polylines(out_img, [np.array(right_points, dtype=np.int32)], isClosed=False, color=(255, 0, 0), thickness=5)

        return out_img


IP = "192.168.0.200"
PORT = 1101
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

        except (AttributeError, ValueError, OSError, cv2.error) as ex:
            print(f"[camera_thread_worker] recoverable error: {ex}")
            traceback.print_exc()
            time.sleep(0.1)
            continue
        except Exception as ex:
            print(f"[camera_thread_worker] unexpected error: {ex}")
            traceback.print_exc()
            raise


def main():
    t = threading.Thread(target=camera_thread_worker, args=(IP, PORT), daemon=True)
    t.start()
    time.sleep(1)

    detector = LaneSegPolyDetector(640, 480, seg_model_path="lane_segmentation.onnx")

    print("=" * 50)
    print("🚀 실시간 딥러닝+컬러 폴리노미얼 차선 인식이 실행되었습니다.")
    print("종료하려면 결과 창을 선택하고 ESC 키를 누르세요.")
    print("=" * 50)

    try:
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                img_frame = latest_frame.copy()

            combined_mask = detector.combine_masks(img_frame)
            roi_mask = detector.limit_region(combined_mask)
            left_fit, right_fit = detector.fit_polynomial(roi_mask)
            img_result = detector.draw_lane(img_frame, left_fit, right_fit)

            cv2.imshow("LaneSegPoly Result", img_result)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n프로그램을 종료합니다.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
