import argparse
import os
import sys

import cv2
import numpy as np


def load_segmentation_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Segmentation model not found: {model_path}")

    net = cv2.dnn.readNet(model_path)
    return net


def preprocess_image(image: np.ndarray, width: int, height: int):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blob = cv2.dnn.blobFromImage(
        image_rgb,
        scalefactor=1.0 / 255.0,
        size=(width, height),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False,
    )
    return blob


def segment_lane(net: cv2.dnn_Net, image: np.ndarray, width: int, height: int):
    blob = preprocess_image(image, width, height)
    net.setInput(blob)
    output = net.forward()

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
    lane_mask = cv2.resize(lane_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return lane_mask


def smooth_binary_mask(mask: np.ndarray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def fit_lane_polynomials(binary_warped: np.ndarray):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0] / nwindows)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 50

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
        return None, None

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if leftx.size > 50 else None
    right_fit = np.polyfit(righty, rightx, 2) if rightx.size > 50 else None
    return left_fit, right_fit


def draw_lane_overlay(image: np.ndarray, mask: np.ndarray, left_fit, right_fit):
    overlay = image.copy()
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(overlay, 0.8, mask_color, 0.2, 0)

    if left_fit is not None or right_fit is not None:
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        if left_fit is not None:
            left_points = np.array([(int(np.polyval(left_fit, y)), int(y)) for y in ploty], dtype=np.int32)
            cv2.polylines(overlay, [left_points], isClosed=False, color=(0, 255, 255), thickness=4)
        if right_fit is not None:
            right_points = np.array([(int(np.polyval(right_fit, y)), int(y)) for y in ploty], dtype=np.int32)
            cv2.polylines(overlay, [right_points], isClosed=False, color=(255, 0, 0), thickness=4)

    return overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Deep learning only lane segmentation demo")
    parser.add_argument("--model", required=True, help="ONNX lane segmentation model path")
    parser.add_argument("--source", required=True,
                        help="Input source: image path, video path, or webcam index (0, 1, ...)")
    parser.add_argument("--width", type=int, default=640, help="Segmentation model input width")
    parser.add_argument("--height", type=int, default=360, help="Segmentation model input height")
    return parser.parse_args()


def open_video_source(source: str):
    if source.isdigit():
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
    if os.path.isfile(source):
        return cv2.VideoCapture(source)
    raise ValueError(f"Invalid source: {source}")


def run_inference(model_path: str, source: str, width: int, height: int):
    net = load_segmentation_model(model_path)

    if os.path.isfile(source) and source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image = cv2.imread(source)
        if image is None:
            raise RuntimeError(f"Unable to read image: {source}")
        mask = segment_lane(net, image, width, height)
        mask = smooth_binary_mask(mask)
        left_fit, right_fit = fit_lane_polynomials(mask)
        output = draw_lane_overlay(image, mask, left_fit, right_fit)
        cv2.imshow("DeepLaneSeg Result", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = open_video_source(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = segment_lane(net, frame, width, height)
        mask = smooth_binary_mask(mask)
        left_fit, right_fit = fit_lane_polynomials(mask)
        output = draw_lane_overlay(frame, mask, left_fit, right_fit)

        cv2.imshow("DeepLaneSeg Result", output)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.source, args.width, args.height)
