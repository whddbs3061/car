import ctypes
import argparse
import os
import threading
import math
from collections import Counter

# X11 멀티스레드 충돌 방지 설정
try:
    X11 = ctypes.CDLL("libX11.so.6")
    X11.XInitThreads()
except Exception:
    pass

import sys
from pathlib import Path

PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

import time
import cv2
import numpy as np
from camera_perception.camera_udp import LatestCameraReceiver
from camera_perception.traffic_signal import (
    TrackedSignalVotes,
    register_cbam_model_layers,
)
from camera_perception.signal_detection import (
    validate_signal_roi, signal_crop_rect, signal_box_allowed,
    merge_signal_detections,
)
from camera_perception.highway_vehicle import (
    HIGHWAY_VEHICLE_CLASSES,
    highway_vehicle_detected,
)

# 공통 IP 설정 (기존 기본값은 유지하고 환경변수/CLI로 덮어쓸 수 있다.)
IP = os.environ.get("MORAI_YOLO_CAM_IP", "0.0.0.0")

# Cam 4 전용 설정 (Port: 1131)
CAM_NAME = "Cam 4"
PORT = int(os.environ.get("MORAI_YOLO_CAM_PORT", "1131"))

# 💡 1. 투트랙 모델 로드
# (1) 기본 사물 탐지 모델 (사람, 차량, 버스, 정지표지판, 동물 등)
BASE_MODEL_PATH = os.environ.get("MORAI_YOLO_BASE_MODEL", "yolov8s.pt")

# (2) 커스텀 모델 (신호등 R/G/Y, 모라이 장애물 등)
CUSTOM_MODEL_PATH = os.environ.get("MORAI_YOLO_CUSTOM_MODEL", "best0917.pt")
CAR_DETECTED_TOPIC = os.environ.get(
    "MORAI_YOLO_CAR_TOPIC", "/perception/camera/car_detected"
)
PERSON_DETECTED_TOPIC = os.environ.get(
    "MORAI_YOLO_PERSON_TOPIC", "/perception/camera/person_detected"
)
INFERENCE_SIZE = int(os.environ.get("MORAI_YOLO_INFERENCE_SIZE", "416"))
# 0 means that the MORAI source rate controls the display.  Adding a 33 ms GUI
# wait to a receiver that already waits for a 30 Hz frame would halve the rate.
DISPLAY_FPS = float(os.environ.get("MORAI_YOLO_DISPLAY_FPS", "0.0"))
CPU_THREADS = int(os.environ.get("MORAI_YOLO_CPU_THREADS", "0"))

# COCO person, car, bus, truck and stop sign. Bus/truck detections reach the
# obstacle topic, while only the unified "car" class activates situation gates.
BASE_TARGET_CLASSES = [0, 2, 5, 7, 11]
TRAFFIC_KEYWORDS = (
    "red", "green", "yellow", "left", "right", "arrow", "amber", "traffic"
)


def _parse_traffic_signal(label):
    normalized = label.lower()
    if normalized in ("unknown", "traffic", "traffic light"):
        return "Unknown", "UNKNOWN", (128, 128, 128)
    # A yellow directional/mixed label is still a stop indication. Preserve it
    # before the left/right branches can discard its colour.
    if "yellow" in normalized or "amber" in normalized:
        return "Yellow", "YELLOW", (0, 255, 255)
    if "red" in normalized and "green" in normalized:
        return "Unknown", "CONFLICT", (128, 128, 128)
    # Preserve the direction of compatible combined classes in one head.
    if "green" in normalized and "left" in normalized:
        return "Green_Left", "GREEN + LEFT", (0, 255, 128)
    if "green" in normalized and "right" in normalized:
        return "Green_Right", "GREEN + RIGHT", (0, 255, 128)
    if "green" in normalized and "arrow" in normalized:
        return "Green_Arrow", "GREEN + ARROW", (0, 255, 128)
    if "green" in normalized:
        return "Green", "GREEN", (0, 255, 0)
    if "red" in normalized and "left" in normalized:
        return "Red_Left", "RED + LEFT", (0, 165, 255)
    if "red" in normalized and "right" in normalized:
        return "Red_Right", "RED + RIGHT", (0, 165, 255)
    if "red" in normalized and "arrow" in normalized:
        return "Red_Arrow", "RED + ARROW", (0, 165, 255)
    if "red" in normalized and "yellow" in normalized:
        return "Red_Yellow", "RED + YELLOW", (0, 128, 255)
    if "left" in normalized:
        return "Left", "LEFT", (255, 255, 0)
    if "right" in normalized:
        return "Right", "RIGHT", (255, 255, 0)
    if "arrow" in normalized:
        return "Arrow", "ARROW", (255, 255, 0)
    if "red" in normalized:
        return "Red", "RED", (0, 0, 255)
    if "yellow" in normalized or "amber" in normalized:
        return "Yellow", "YELLOW", (0, 255, 255)
    return None, None, None

def _resolve_model_path(model_path):
    """Resolve bundled feature-camera weights before trying Ultralytics cache."""
    if os.path.isabs(model_path):
        return model_path
    package_path = Path(__file__).resolve().parents[1]
    bundled_path = package_path / "models" / model_path
    return str(bundled_path) if bundled_path.exists() else model_path


def _tracking_available():
    # Docker intentionally disables Ultralytics auto-install. Missing LAP is
    # a supported inference-only mode, not a reason to lose all signal frames.
    try:
        import lap
        return callable(getattr(lap, "lapjv", None))
    except Exception:
        return False


def main(ip=IP, port=PORT, base_model_path=BASE_MODEL_PATH,
         custom_model_path=CUSTOM_MODEL_PATH, confidence=0.6,
         car_detected_topic=CAR_DETECTED_TOPIC,
         person_detected_topic=PERSON_DETECTED_TOPIC,
         traffic_light_topic="/detection/traffic_light",
         obstacle_topic="/detection/obstacle",
         inference_size=INFERENCE_SIZE, display_fps=DISPLAY_FPS,
         cpu_threads=CPU_THREADS, show_raw_preview=False,
         signal_roi=(0.3, 0.05, 0.6, 0.6), signal_full_frame=True,
         window_size=5):
    """Cam 4 UDP receive, asynchronous YOLO inference, and live display.

    Camera receive/display must not wait for model inference.  The inference
    worker always replaces its pending input with the newest frame, so a slow
    CPU lowers detection FPS without building seconds of stale video.
    """
    # argparse/help와 ROS launch 구조 검증은 모델 설정 파일 접근 없이 가능하게 한다.
    import rospy
    from std_msgs.msg import Bool, Header
    from common.msg import ObjectInfo, ObjectInfoArray
    from ultralytics import YOLO

    show_raw_preview = bool(show_raw_preview)
    signal_roi = validate_signal_roi(signal_roi)
    signal_full_frame = bool(signal_full_frame)
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least one")

    # PyTorch otherwise tends to occupy every vCPU in a small VirtualBox VM,
    # starving the UDP/decode/GUI thread as soon as the first inference starts.
    import torch
    register_cbam_model_layers(torch)
    selected_cpu_threads = None
    if not torch.cuda.is_available():
        available = max(1, os.cpu_count() or 1)
        selected_cpu_threads = (
            max(1, int(cpu_threads))
            if int(cpu_threads) > 0
            else max(1, min(2, available - 1))
        )
        torch.set_num_threads(selected_cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # It can only be set before inter-op work begins. Inference still
            # respects set_num_threads when another library initialized it.
            pass
        print(
            f"[{CAM_NAME}] CPU inference threads={selected_cpu_threads} "
            f"(available={available})"
        )

    rospy.init_node("yolo_camera", anonymous=False)
    preview = None
    if os.environ.get('MORAI_CAMERA_DEBUG', '').lower() in ('1', 'true'):
        from camera_perception.debug_images import DebugImagePublisher, render_objects
        preview = DebugImagePublisher('cam4')
    car_detected_publisher = rospy.Publisher(
        car_detected_topic, Bool, queue_size=1
    )
    person_detected_publisher = rospy.Publisher(
        person_detected_topic, Bool, queue_size=1
    )
    traffic_light_publisher = rospy.Publisher(
        traffic_light_topic, ObjectInfoArray, queue_size=1
    )
    obstacle_publisher = rospy.Publisher(
        obstacle_topic, ObjectInfoArray, queue_size=1
    )
    detection_state = {"car": False, "person": False}
    detection_state_lock = threading.Lock()

    def publish_detection_state(_event=None):
        with detection_state_lock:
            car = detection_state["car"]
            person = detection_state["person"]
        car_detected_publisher.publish(Bool(data=car))
        person_detected_publisher.publish(Bool(data=person))

    detection_heartbeat_timer = rospy.Timer(
        rospy.Duration(0.1),
        publish_detection_state,
    )

    def object_message(box, model, class_name=None):
        cls_id = int(box.cls[0])
        xc, yc, width, height = box.xywh[0].detach().cpu().tolist()
        message = ObjectInfo()
        message.class_name = class_name or str(model.names[cls_id]).capitalize()
        message.conf = float(box.conf[0])
        message.x_center = float(xc)
        message.y_center = float(yc)
        message.width = float(width)
        message.height = float(height)
        return message

    def object_array(sequence, objects, received_stamp):
        message = ObjectInfoArray()
        message.header = Header(
            seq=int(sequence),
            stamp=received_stamp if received_stamp is not None else rospy.Time(),
            frame_id="camera_link",
        )
        message.objects = list(objects)
        return message

    print(f"[{CAM_NAME}] YOLOv8 모델 로딩 중...")
    base_model = YOLO(_resolve_model_path(base_model_path))

    resolved_custom_path = _resolve_model_path(custom_model_path)
    custom_model = None
    roi_model = None
    tracking_enabled = signal_full_frame and _tracking_available()
    if os.path.isfile(resolved_custom_path):
        custom_model = YOLO(resolved_custom_path)
        # Tracking callbacks and coordinates must never be shared between the
        # full-image tracker and the crop predictor.
        roi_model = YOLO(resolved_custom_path) if signal_full_frame else custom_model
        print(f"[{CAM_NAME}] 커스텀 모델 로드 완료: {resolved_custom_path}")
        rospy.loginfo("CAM4 best0917 pipeline: roi=%s full_frame=%s tracking=%s",
                      signal_roi, signal_full_frame, tracking_enabled)
    else:
        print(f"[{CAM_NAME}] 경고: 커스텀 모델을 찾지 못해 기본 YOLO만 실행합니다: "
              f"{resolved_custom_path}")

    cam_data = LatestCameraReceiver(ip, port, stamp_clock=rospy.Time.now)
    last_frame_sequence = 0

    pending_condition = threading.Condition()
    pending_frame = {
        "sequence": 0, "image": None, "received_at": 0.0,
        "received_stamp": None,
    }
    result_lock = threading.Lock()
    latest_result = {
        "revision": 0,
        "sequence": 0,
        "source_image": None,
        "detections": (),
        "stage": "WAITING",
        "inference_ms": 0.0,
        "latency_ms": 0.0,
        "completed_at": 0.0,
        "fps": 0.0,
    }
    stop_worker = threading.Event()
    signal_votes = TrackedSignalVotes(window=window_size, green_votes=min(3, window_size))

    def collect_custom_detections(result, model, image_width, image_height,
                                  source="full", crop_rect=None):
        """Collect raw candidates in original pixels; vote only after merging."""
        candidates, obstacle_objects = [], []
        boxes = result.boxes if result.boxes is not None else ()
        for box in boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                continue
            label = str(model.names[cls_id])
            normalized = label.lower()
            x1, y1, x2, y2 = box.xyxy[0].detach().cpu().tolist()
            if any(keyword in normalized for keyword in TRAFFIC_KEYWORDS):
                if crop_rect is not None:
                    cx0, cy0, cx1, cy1 = crop_rect
                    # A cropped half-housing can resemble a different signal.
                    # Let the full-image pass handle boxes touching crop edges.
                    if (x1 <= 1 or y1 <= 1 or x2 >= cx1-cx0-1 or y2 >= cy1-cy0-1):
                        continue
                    x1, x2, y1, y2 = x1+cx0, x2+cx0, y1+cy0, y2+cy0
                if not signal_box_allowed(x1, y1, x2, y2, image_width, image_height):
                    continue
                class_name, _, _ = _parse_traffic_signal(label)
                if class_name is None:
                    class_name = "Unknown"
                track_id_tensor = getattr(box, "id", None)
                track_id = (int(track_id_tensor[0])
                            if source == "full" and track_id_tensor is not None else None)
                candidates.append(dict(xyxy=(x1, y1, x2, y2), label=class_name,
                                       score=score, track_id=track_id, source=source))
                continue
            # ROI is for signals only; preserve the existing full-image
            # obstacle contract without double-counting cropped obstacles.
            if source != "full":
                continue
            ignored = ("cone", "drum", "barrier", "bike", "bicycle", "truck",
                       "bus", "car", "motorcycle", "vehicle")
            if any(keyword in normalized for keyword in ignored):
                continue
            width, height = x2-x1, y2-y1
            if height <= 0 or (y1+y2)*0.5/image_height > 0.80 or width/height > 1.5:
                continue
            obstacle_objects.append(object_message(box, model))
        return candidates, obstacle_objects

    def finalize_signals(candidates):
        merged = merge_signal_detections(candidates)
        counts = Counter(record["track_id"] for record in merged
                         if record["track_id"] is not None)
        visible_ids = set(counts)
        # Conflicting current evidence must reach the existing housing/route
        # association. Never let a majority erase RED or a turn arrow.
        for track_id, count in counts.items():
            if count > 1:
                signal_votes.observe(track_id, "Unknown")
        traffic_objects, signal_detections = [], []
        for record in merged:
            track_id = record["track_id"]
            label = (signal_votes.observe(track_id, record["label"])
                     if track_id is not None and counts[track_id] == 1
                     else record["label"])
            class_name, display_text, color = _parse_traffic_signal(label)
            if class_name is None:
                continue
            x1, y1, x2, y2 = record["xyxy"]
            message = ObjectInfo()
            message.class_name, message.conf = class_name, record["score"]
            message.x_center, message.y_center = (x1+x2)*0.5, (y1+y2)*0.5
            message.width, message.height = x2-x1, y2-y1
            traffic_objects.append(message)
            signal_detections.append((x1, y1, x2, y2, display_text, record["score"], color))
        signal_votes.retain(visible_ids)
        return traffic_objects, signal_detections

    def submit_preview(*args):
        if preview is not None:
            try:
                preview.submit(*args)
            except Exception as preview_error:
                rospy.logwarn_throttle(2.0, "CAM4 preview error: %s", preview_error)

    def inference_worker():
        nonlocal custom_model, tracking_enabled
        last_inferred_sequence = 0
        smoothed_fps = 0.0
        last_base_completed_at = 0.0
        while not stop_worker.is_set() and not rospy.is_shutdown():
            with pending_condition:
                pending_condition.wait_for(
                    lambda: stop_worker.is_set()
                    or pending_frame["sequence"] > last_inferred_sequence,
                    timeout=0.1,
                )
                if stop_worker.is_set():
                    return
                sequence = pending_frame["sequence"]
                image = pending_frame["image"]
                received_at = pending_frame["received_at"]
                received_stamp = pending_frame["received_stamp"]

            if image is None or sequence <= last_inferred_sequence:
                continue
            last_inferred_sequence = sequence
            started_at = time.monotonic()
            signal_published = False

            try:
                # Repeated sensor timestamps cannot accumulate permission votes.
                source_seconds = received_stamp.to_sec() if received_stamp is not None else None
                if source_seconds == 0:
                    source_seconds = None
                if not signal_votes.begin_frame(sequence, source_seconds, received_at):
                    continue
                try:
                    base_results = base_model.predict(
                        source=image, classes=BASE_TARGET_CLASSES,
                        imgsz=inference_size, conf=confidence, verbose=False)
                    base_boxes = base_results[0].boxes if base_results[0].boxes is not None else ()
                except Exception as base_error:
                    # The signal checkpoint has its own inference path.
                    base_boxes = ()
                    rospy.logerr_throttle(1.0, "CAM4 base detector failed: %s", base_error)
                detected_labels = {
                    str(base_model.names[int(box.cls[0])]).strip().lower()
                    for box in base_boxes
                }
                base_objects = []
                for box in base_boxes:
                    label = str(base_model.names[int(box.cls[0])]).lower()
                    class_name = "Car" if label == "car" else label.capitalize()
                    base_objects.append(object_message(box, base_model, class_name))

                # Publish the COCO road-vehicle/person result immediately.
                # CAM4 displays only boxes also published to the signal topic.
                base_completed_at = time.monotonic()
                base_elapsed = max(base_completed_at - started_at, 1e-6)
                base_interval = (
                    base_completed_at - last_base_completed_at
                    if last_base_completed_at > 0.0
                    else base_elapsed
                )
                instant_fps = 1.0 / max(base_interval, 1e-6)
                smoothed_fps = (
                    instant_fps
                    if smoothed_fps <= 0.0
                    else 0.8 * smoothed_fps + 0.2 * instant_fps
                )
                last_base_completed_at = base_completed_at

                with result_lock:
                    latest_result.update(
                        revision=latest_result["revision"] + 1,
                        sequence=sequence,
                        # The receiver and GUI never mutate this decoded image.
                        # Avoid one full-frame copy on the latency-critical path.
                        source_image=image,
                        detections=(),
                        stage="SIGNAL_PENDING" if custom_model is not None else "NO_SIGNAL_MODEL",
                        inference_ms=base_elapsed * 1000.0,
                        latency_ms=max(
                            base_completed_at - received_at, 0.0
                        ) * 1000.0,
                        completed_at=base_completed_at,
                        fps=smoothed_fps,
                    )

                # One shared unified-car state feeds both the highway and
                # intersection situation gates.
                car_detected = highway_vehicle_detected(detected_labels)
                person_detected = "person" in detected_labels
                with detection_state_lock:
                    detection_state["car"] = car_detected
                    detection_state["person"] = person_detected
                publish_detection_state()
                obstacle_publisher.publish(
                    object_array(sequence, base_objects, received_stamp)
                )

                if car_detected:
                    rospy.loginfo_throttle(
                        1.0,
                        "YOLO unified Car detected (%s); camera condition is true",
                        ",".join(
                            sorted(detected_labels.intersection(HIGHWAY_VEHICLE_CLASSES))
                        ),
                    )
                if person_detected:
                    rospy.logwarn_throttle(
                        1.0,
                        "YOLO person detected; pedestrian fusion camera condition is true",
                    )

                signal_detections = []
                pass_errors = []
                if custom_model is not None:
                    candidates, custom_obstacle_objects = [], []
                    successful_passes = 0
                    if signal_full_frame:
                        try:
                            if tracking_enabled:
                                try:
                                    custom_results = custom_model.track(
                                        source=image, imgsz=inference_size, conf=confidence,
                                        persist=True, tracker="bytetrack.yaml", verbose=False)
                                except Exception as tracker_error:
                                    rospy.logwarn("CAM4 tracker failed; switching permanently to "
                                                  "fresh frame predictor: %s", tracker_error)
                                    tracking_enabled = False
                                    signal_votes.reset()
                                    signal_votes.begin_frame(sequence, source_seconds, received_at)
                                    # A failed track() can leave callbacks that still
                                    # expect trackers during predict(). Discard that instance.
                                    custom_model = YOLO(resolved_custom_path)
                                    custom_results = custom_model.predict(
                                        source=image, imgsz=inference_size,
                                        conf=confidence, verbose=False)
                            else:
                                custom_results = custom_model.predict(
                                    source=image, imgsz=inference_size,
                                    conf=confidence, verbose=False)
                            candidates, custom_obstacle_objects = collect_custom_detections(
                                custom_results[0], custom_model, image.shape[1], image.shape[0])
                            successful_passes += 1
                        except Exception as full_error:
                            pass_errors.append("full: " + str(full_error))
                            signal_votes.reset()
                            signal_votes.begin_frame(sequence, source_seconds, received_at)
                    try:
                        crop_rect = signal_crop_rect(signal_roi, image.shape[1], image.shape[0])
                        x0, y0, x1, y1 = crop_rect
                        crop_results = roi_model.predict(
                            source=image[y0:y1, x0:x1], imgsz=inference_size,
                            conf=confidence, verbose=False)
                        crop_candidates, _ = collect_custom_detections(
                            crop_results[0], roi_model, image.shape[1], image.shape[0],
                            source="roi", crop_rect=crop_rect)
                        candidates.extend(crop_candidates)
                        successful_passes += 1
                    except Exception as crop_error:
                        pass_errors.append("roi: " + str(crop_error))
                    if not successful_passes:
                        raise RuntimeError("; ".join(pass_errors))
                    if pass_errors:
                        rospy.logwarn_throttle(2.0, "CAM4 partial inference: %s",
                                               "; ".join(pass_errors))
                    traffic_objects, signal_detections = finalize_signals(candidates)
                    if signal_detections:
                        labels = ", ".join(sorted({d[4] for d in signal_detections}))
                        rospy.loginfo_throttle(1.0, "%s published signals: %s", CAM_NAME, labels)

                    traffic_light_publisher.publish(
                        object_array(sequence, traffic_objects, received_stamp)
                    )
                    signal_published = True
                    obstacle_publisher.publish(
                        object_array(
                            sequence, base_objects + custom_obstacle_objects,
                            received_stamp,
                        )
                    )

                    # The CAM4 overlay matches only objects sent to the
                    # traffic-light topic for this exact frame.
                    custom_completed_at = time.monotonic()
                    with result_lock:
                        latest_result.update(
                            revision=latest_result["revision"] + 1,
                            sequence=sequence,
                            source_image=image,
                            detections=tuple(signal_detections),
                            stage="SIGNAL",
                            inference_ms=max(
                                custom_completed_at - started_at, 0.0
                            ) * 1000.0,
                            latency_ms=max(
                                custom_completed_at - received_at, 0.0
                            ) * 1000.0,
                            completed_at=custom_completed_at,
                            fps=smoothed_fps,
                        )
                else:
                    traffic_light_publisher.publish(
                        object_array(sequence, (), received_stamp)
                    )
                    signal_published = True

                if preview is not None:
                    drawn = tuple(signal_detections)
                    submit_preview(image, lambda f=image, d=drawn: render_objects(f, d),
                        received_stamp if received_stamp is not None else rospy.Time(), sequence,
                        dict(base_model=_resolve_model_path(base_model_path),
                             custom_model=resolved_custom_path, custom_loaded=custom_model is not None,
                             pipeline="best0917_roi_full_v2", signal_roi=list(signal_roi),
                             full_frame=signal_full_frame, tracking=tracking_enabled,
                             error="; ".join(pass_errors),
                             inference_ms=(time.monotonic()-started_at)*1000.,
                             detections=[dict(label=d[4], confidence=d[5]) for d in drawn]))

            except Exception as error:
                if signal_published:
                    rospy.logerr_throttle(1.0, "CAM4 post-publication error: %s", error)
                    continue
                signal_votes.reset()
                with detection_state_lock:
                    detection_state.update(car=False, person=False)
                # Clear the current observation on inference failure. A recent
                # timestamp must never make a previous green appear fresh.
                if not signal_published:
                    traffic_light_publisher.publish(object_array(sequence, (), received_stamp))
                with result_lock:
                    latest_result.update(revision=latest_result["revision"]+1,
                                         sequence=sequence, source_image=image,
                                         detections=(), stage="ERROR")
                if preview is not None:
                    submit_preview(image, lambda f=image: render_objects(f, ()),
                                   received_stamp if received_stamp is not None else rospy.Time(),
                                   sequence, dict(custom_model=resolved_custom_path,
                                                  custom_loaded=custom_model is not None,
                                                  pipeline="best0917_roi_full_v2",
                                                  error=str(error), detections=[]))
                rospy.logerr_throttle(1.0, "YOLO inference error: %s", error)

    worker = threading.Thread(
        target=inference_worker,
        name="morai-yolo-inference",
        daemon=True,
    )
    worker.start()
    last_display_at = 0.0
    smoothed_live_fps = 0.0
    last_live_image = None
    last_live_frame_at = None
    last_detection_display_revision = 0
    live_window = f"MORAI {CAM_NAME} Live Preview"
    detection_window = f"MORAI {CAM_NAME} YOLO Detection (Frame Matched)"
    
    print(f"[{CAM_NAME}] MORAI UDP 카메라 연결 시도 중... ({ip}:{port})")

    while not rospy.is_shutdown():
        try:
            frame = cam_data.wait_for_latest(last_frame_sequence, timeout=0.1)
            if frame is None:
                # Even with no UDP frame, pump GUI events so the window does
                # not become frozen/unresponsive. Show an explicit watchdog
                # warning instead of silently leaving the last image onscreen.
                now = time.monotonic()
                stale_for = (
                    now - last_live_frame_at
                    if last_live_frame_at is not None
                    else float("inf")
                )
                if stale_for > 0.5:
                    health = cam_data.health_snapshot()
                    rospy.logwarn_throttle(
                        1.0,
                        "No complete camera frame for %.2fs; receiver=%s",
                        stale_for,
                        health,
                    )
                    if show_raw_preview:
                        waiting = (
                            last_live_image.copy()
                            if last_live_image is not None
                            else np.zeros((480, 640, 3), dtype=np.uint8)
                        )
                        cv2.rectangle(
                            waiting,
                            (0, 0),
                            (waiting.shape[1], 38),
                            (0, 0, 180),
                            -1,
                        )
                        cv2.putText(
                            waiting,
                            "NO NEW CAMERA FRAME - check MORAI UDP",
                            (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (255, 255, 255),
                            2,
                        )
                        cv2.imshow(live_window, waiting)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            last_frame_sequence = frame.sequence

            image_np = np.frombuffer(frame.jpeg_data, dtype=np.uint8)
            if image_np.size == 0:
                continue

            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                continue
            if show_raw_preview:
                last_live_image = image
            last_live_frame_at = time.monotonic()

            # Replace the pending inference job instead of queueing this frame.
            with pending_condition:
                pending_frame["sequence"] = frame.sequence
                pending_frame["image"] = image
                pending_frame["received_at"] = frame.received_at
                pending_frame["received_stamp"] = frame.received_stamp
                pending_condition.notify()

            now = time.monotonic()
            if (
                display_fps > 0.0
                and last_display_at > 0.0
                and now - last_display_at < 1.0 / display_fps
            ):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            if last_display_at > 0.0:
                instant_live_fps = 1.0 / max(now - last_display_at, 1e-6)
                smoothed_live_fps = (
                    instant_live_fps
                    if smoothed_live_fps <= 0.0
                    else 0.9 * smoothed_live_fps + 0.1 * instant_live_fps
                )
            last_display_at = now

            # The live preview intentionally has no boxes. A box is only valid
            # for the exact source frame used by its YOLO inference.
            with result_lock:
                shown_result = dict(latest_result)
            result_age_ms = (
                (time.monotonic() - shown_result["completed_at"]) * 1000.0
                if shown_result["completed_at"] > 0.0
                else 0.0
            )
            if show_raw_preview:
                display_frame = image.copy()
                status = (
                    f"LIVE {smoothed_live_fps:.1f} FPS | "
                    f"YOLO {shown_result['fps']:.1f} FPS | "
                    f"infer {shown_result['inference_ms']:.0f} ms | "
                    f"latency {shown_result['latency_ms']:.0f} ms | "
                    f"age {result_age_ms:.0f} ms"
                )
                cv2.rectangle(
                    display_frame,
                    (0, 0),
                    (display_frame.shape[1], 30),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    display_frame,
                    status,
                    (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow(live_window, display_frame)

            # The initial revision has no boxes; the signal revision shows
            # only objects published to the traffic-light topic.
            result_revision = int(shown_result["revision"])
            result_sequence = int(shown_result["sequence"])
            matched_source = shown_result["source_image"]
            if (
                preview is None
                and matched_source is not None
                and result_revision > last_detection_display_revision
            ):
                matched_frame = matched_source.copy()
                for x1, y1, x2, y2, label, score, color in shown_result["detections"]:
                    p1 = (max(0, int(x1)), max(0, int(y1)))
                    p2 = (
                        min(matched_frame.shape[1] - 1, int(x2)),
                        min(matched_frame.shape[0] - 1, int(y2)),
                    )
                    cv2.rectangle(matched_frame, p1, p2, color, 2)
                    cv2.putText(
                        matched_frame,
                        f"{label} {score:.2f}",
                        (p1[0], max(18, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                detection_status = (
                    f"{shown_result['stage']} FRAME {result_sequence} | "
                    f"YOLO {shown_result['fps']:.1f} FPS | "
                    f"infer {shown_result['inference_ms']:.0f} ms | "
                    f"latency {shown_result['latency_ms']:.0f} ms"
                )
                cv2.rectangle(
                    matched_frame,
                    (0, 0),
                    (matched_frame.shape[1], 30),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    matched_frame,
                    detection_status,
                    (8, 21),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
                cv2.imshow(detection_window, matched_frame)
                last_detection_display_revision = result_revision

            # 'q' 키를 누르면 모니터링 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"[{CAM_NAME}] Error: {e}")
            time.sleep(0.01)

    stop_worker.set()
    with pending_condition:
        pending_condition.notify_all()
    worker.join(timeout=1.0)
    if preview is not None:
        preview.close()
    try:
        with detection_state_lock:
            detection_state["car"] = False
            detection_state["person"] = False
        car_detected_publisher.publish(Bool(data=False))
        person_detected_publisher.publish(Bool(data=False))
    except rospy.ROSException:
        pass
    detection_heartbeat_timer.shutdown()
    cam_data.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MORAI UDP YOLO 객체 탐지")
    parser.add_argument("--cam-ip", default=IP)
    parser.add_argument("--cam-port", type=int, default=PORT)
    parser.add_argument("--base-model", default=BASE_MODEL_PATH)
    parser.add_argument("--custom-model", default=CUSTOM_MODEL_PATH)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--car-detected-topic", default=CAR_DETECTED_TOPIC)
    parser.add_argument("--person-detected-topic", default=PERSON_DETECTED_TOPIC)
    parser.add_argument(
        "--traffic-light-topic", default="/detection/traffic_light"
    )
    parser.add_argument("--obstacle-topic", default="/detection/obstacle")
    parser.add_argument("--inference-size", type=int, default=INFERENCE_SIZE)
    parser.add_argument(
        "--display-fps",
        type=float,
        default=DISPLAY_FPS,
        help="maximum live display FPS; 0 follows the MORAI source rate",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=CPU_THREADS,
        help="PyTorch CPU threads; 0 reserves at least one vCPU for camera/GUI",
    )
    parser.add_argument(
        "--show-raw-preview",
        type=int,
        choices=(0, 1),
        default=0,
        help="show the unprocessed camera window (0 disables it)",
    )
    parser.add_argument("--signal-roi", nargs=4, type=float,
                        default=(0.3, 0.05, 0.6, 0.6))
    parser.add_argument("--signal-full-frame", type=int, choices=(0, 1), default=1)
    parser.add_argument("--window-size", type=int, default=5)
    args = parser.parse_args()
    main(args.cam_ip, args.cam_port, args.base_model, args.custom_model,
         args.confidence, args.car_detected_topic, args.person_detected_topic,
         args.traffic_light_topic, args.obstacle_topic,
         args.inference_size, args.display_fps, args.cpu_threads,
         bool(args.show_raw_preview), args.signal_roi,
         bool(args.signal_full_frame), args.window_size)
