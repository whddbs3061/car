"""YOLO 신호등 클래스명을 주행/정지 조건으로 변환한다."""

import math
from collections import Counter, deque
from types import SimpleNamespace


# CAM4's original signal ROI ended at 65% of the image height.  Keep vertical
# heads possible; cap area to reject gantry-sized false signal detections.
SIGNAL_ROI_MAX_CENTER_Y = 0.65
SIGNAL_MIN_ASPECT_RATIO = 2.0
SIGNAL_MAX_ASPECT_RATIO = 8.0
SIGNAL_MAX_AREA_FRACTION = 0.01


def traffic_bbox_plausible(x, y, width, height, image_height):
    """Only reject invalid boxes; oblique/vertical heads need not be wide.

    Shape or screen centre is not evidence that a light belongs to our lane.
    Route-linked control associates boxes with mapped heads; the sensor-only
    launch uses an additional image-space guard below.
    """
    return (all(isinstance(v, (int, float)) and math.isfinite(v)
                for v in (x, y, width, height, image_height))
            and image_height > 0 and width >= 2 and height >= 2
            and x >= 0 and 0 <= y <= image_height)


def traffic_bbox_in_signal_roi(x, y, width, height, image_width, image_height):
    """Reject implausible CAM4 signal boxes before publishing light evidence.

    This is an image-space guard, not lane-to-signal association.  Keep narrow
    vertical and oblique heads possible, but reject boxes on the road, poles,
    and large gantries.  A rejected box must never authorize GREEN.
    """
    if not traffic_bbox_plausible(x, y, width, height, image_height):
        return False
    if not isinstance(image_width, (int, float)) or not math.isfinite(image_width):
        return False
    if image_width <= 0 or x > image_width:
        return False
    aspect_ratio = width / height
    area_fraction = width * height / (image_width * image_height)
    return (y / image_height <= SIGNAL_ROI_MAX_CENTER_Y
            and SIGNAL_MIN_ASPECT_RATIO <= aspect_ratio <= SIGNAL_MAX_ASPECT_RATIO
            and area_fraction <= SIGNAL_MAX_AREA_FRACTION)


class TrackedSignalVotes:
    """Smooth one tracked lamp without delaying current stop evidence.

    Call begin_frame once after merging the full-frame and ROI detections,
    observe at most once per track, and retain once with the merged visible IDs.
    A duplicate frame must not be published or counted again. Missing IDs use
    the raw class; downstream route-associated control still confirms permission
    over distinct, fresh source images before releasing a stop.

    Legacy observe(track_id, class_name) callers remain supported, but cannot
    detect silence or duplicate images without calling begin_frame. RED_LEFT
    remains current evidence immediately; its red lamp does not erase the lit
    arrow's directional meaning. Stop/unknown observations erase older votes.
    """

    def __init__(self, window=5, green_votes=3):
        self.window = int(window)
        self.green_votes = int(green_votes)
        self.reset()

    def reset(self):
        """Discard votes and frame clocks after source/tracker changes."""
        self.history = {}
        self._last_sequence = None
        self._last_source_stamp = None
        self._last_received = None

    def begin_frame(self, sequence, source_stamp, received):
        """Accept a new frame and clear votes after a gap or clock restart.

        source_stamp is the original ROS image time, or None when unavailable;
        received uses one monotonic clock. With no source time, sequence and
        receipt time still protect the vote history; they do not make that
        image fresh enough for downstream control. False means skip voting and
        publication. Duplicates do not refresh either accepted-frame watermark.
        """
        valid_sequence = (isinstance(sequence, int) and not isinstance(sequence, bool)
                          and sequence >= 0)
        valid_received = (isinstance(received, (int, float))
                          and not isinstance(received, bool)
                          and math.isfinite(received) and received >= 0)
        valid_stamp = (source_stamp is None or
                       (isinstance(source_stamp, (int, float))
                        and not isinstance(source_stamp, bool)
                        and math.isfinite(source_stamp) and source_stamp > 0))
        if not (valid_sequence and valid_received and valid_stamp):
            self.reset()
            return False
        clock_reset = (
            (self._last_received is not None and received < self._last_received)
            or (source_stamp is not None and self._last_source_stamp is not None
                and source_stamp < self._last_source_stamp)
            or (self._last_sequence is not None and sequence < self._last_sequence)
        )
        if clock_reset:
            self.reset()
        if (sequence == self._last_sequence
                or (source_stamp is not None and source_stamp == self._last_source_stamp)):
            return False
        if ((self._last_received is not None and received - self._last_received > 0.5)
                or (source_stamp is not None and self._last_source_stamp is not None
                    and source_stamp - self._last_source_stamp > 0.5)):
            self.history.clear()
        self._last_sequence = sequence
        if source_stamp is not None:
            self._last_source_stamp = source_stamp
        self._last_received = received
        return True

    def observe(self, track_id, class_name):
        if track_id is None:
            return class_name
        key = int(track_id)
        name = str(class_name).lower()
        if any(word in name for word in ("unknown", "conflict")):
            self.history.pop(key, None)
            return class_name
        if any(word in name for word in ("red", "yellow", "amber", "green", "left", "right", "arrow")):
            samples = self.history.setdefault(key, deque(maxlen=self.window))
            samples.append(class_name)
            return class_name if Counter(samples)[class_name] >= self.green_votes else "Unknown"
        self.history.pop(key, None)
        return class_name

    def retain(self, visible_ids):
        visible = set(visible_ids)
        for key in tuple(self.history):
            if key not in visible:
                del self.history[key]


def register_cbam_model_layers(torch_module=None):
    """Register custom layers referenced by the team's YOLO checkpoints.

    The signal checkpoint was serialized with C2f_EMA and EMA under
    ``ultralytics.nn.modules.block``.  Registering only on ``tasks`` is enough
    for YAML parsing but not enough for torch's pickle loader, which resolves
    the original module path stored in the checkpoint.
    """
    if torch_module is None:
        import torch as torch_module
    # The camera loop's dependency-free tests replace torch with a small stub.
    if not hasattr(torch_module, "nn"):
        return
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.block as block
    from ultralytics.nn.modules.block import C2f

    nn = torch_module.nn

    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            hidden = max(1, channels // reduction)
            self.fc = nn.Sequential(
                nn.Linear(channels, hidden, bias=False), nn.ReLU(inplace=True),
                nn.Linear(hidden, channels, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            batch, channels, _, _ = x.size()
            average = self.fc(x.mean((2, 3)).view(batch, channels)).view(batch, channels, 1, 1)
            maximum = self.fc(x.amax((2, 3)).view(batch, channels)).view(batch, channels, 1, 1)
            return x * self.sigmoid(average + maximum)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            average = torch_module.mean(x, dim=1, keepdim=True)
            maximum, _ = torch_module.max(x, dim=1, keepdim=True)
            return x * self.sigmoid(self.conv(torch_module.cat([average, maximum], dim=1)))

    class CBAM(nn.Module):
        def __init__(self, c1, kernel_size=7):
            super().__init__()
            self.ca = ChannelAttention(c1)
            self.sa = SpatialAttention(kernel_size)

        def forward(self, x):
            return self.sa(self.ca(x))

    class EMA(nn.Module):
        """Efficient Multi-scale Attention used during signal-model training."""

        def __init__(self, channels, factor=8):
            super().__init__()
            self.groups = factor
            assert channels // self.groups > 0
            self.softmax = nn.Softmax(-1)
            self.agp = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
            group_channels = channels // self.groups
            self.gn = nn.GroupNorm(group_channels, group_channels)
            self.conv1x1 = nn.Conv2d(
                group_channels, group_channels, kernel_size=1, stride=1, padding=0
            )
            self.conv3x3 = nn.Conv2d(
                group_channels, group_channels, kernel_size=3, stride=1, padding=1
            )

        def forward(self, x):
            batch, channels, height, width = x.size()
            group_x = x.reshape(batch * self.groups, -1, height, width)
            x_h = self.pool_h(group_x)
            x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
            hw = self.conv1x1(torch_module.cat([x_h, x_w], dim=2))
            x_h, x_w = torch_module.split(hw, [height, width], dim=2)

            x1 = self.gn(
                group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
            )
            x2 = self.conv3x3(group_x)
            x11 = self.softmax(
                self.agp(x1).reshape(batch * self.groups, -1, 1).permute(0, 2, 1)
            )
            x12 = x2.reshape(batch * self.groups, channels // self.groups, -1)
            x21 = self.softmax(
                self.agp(x2).reshape(batch * self.groups, -1, 1).permute(0, 2, 1)
            )
            x22 = x1.reshape(batch * self.groups, channels // self.groups, -1)
            weights = (
                torch_module.matmul(x11, x12) + torch_module.matmul(x21, x22)
            ).reshape(batch * self.groups, 1, height, width)
            return (group_x * weights.sigmoid()).reshape(batch, channels, height, width)

    class C2f_EMA(C2f):
        """C2f block whose inner bottlenecks use the training EMA module."""

        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__(c1, c2, n, shortcut, g, e)
            self.m = nn.ModuleList(EMA(self.c) for _ in range(n))

    tasks.CBAM = CBAM
    tasks.EMA = EMA
    tasks.C2f_EMA = C2f_EMA
    modules.EMA = EMA
    modules.C2f_EMA = C2f_EMA
    block.EMA = EMA
    block.C2f_EMA = C2f_EMA


def _same_signal_housing(first, second):
    """Require adjacent lamp boxes or a lamp box inside its combined head."""
    try:
        x1, y1, w1, h1 = (float(getattr(first, key)) for key in
                          ("x_center", "y_center", "width", "height"))
        x2, y2, w2, h2 = (float(getattr(second, key)) for key in
                          ("x_center", "y_center", "width", "height"))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    if not all(math.isfinite(v) and v > 0 for v in (w1, h1, w2, h2)):
        return False
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return False
    similarly_sized_neighbors = (
        min(w1, w2) / max(w1, w2) >= 0.4
        and min(h1, h2) / max(h1, h2) >= 0.4
        and abs(x1 - x2) <= 2.0 * max(w1, w2)
        and abs(y1 - y2) <= 0.75 * max(h1, h2)
    )
    if similarly_sized_neighbors:
        return True
    # The checkpoint can detect the whole RED_LEFT housing as well as its red
    # bulb. Those boxes differ greatly in width but one nearly contains the
    # other. Containment is stronger evidence than a loose distance threshold.
    intersection = (max(0.0, min(x1 + w1 / 2, x2 + w2 / 2)
                        - max(x1 - w1 / 2, x2 - w2 / 2))
                    * max(0.0, min(y1 + h1 / 2, y2 + h2 / 2)
                          - max(y1 - h1 / 2, y2 - h2 / 2)))
    return intersection / min(w1 * h1, w2 * h2) >= 0.8


def directional_observation(objects, min_confidence=0.5):
    """Preserve one directional class; conflicting signal heads are UNKNOWN.

    Combined classes (e.g. Red_Left) must describe one detected signal head.
    Never merge a red head with an unrelated green/arrow head into permission.
    Bare Left/Right mean the model's illuminated arrow classes; generic Arrow
    has no direction and cannot authorize a maneuver.
    """
    supported = {"RED", "YELLOW", "GREEN", "LEFT", "RIGHT", "GREEN_LEFT",
                 "GREEN_RIGHT", "RED_LEFT", "RED_RIGHT"}
    evidence = {}
    candidates = []
    for item in objects:
        try:
            score = float(item.conf)
        except (AttributeError, ValueError, TypeError, OverflowError):
            continue
        if not math.isfinite(score) or not min_confidence <= score <= 1.0:
            continue
        name = str(getattr(item, "class_name", "UNKNOWN")).strip().upper().replace(" ", "_").replace("-", "_")
        if "YELLOW" in name or "AMBER" in name:
            name = "YELLOW"
        if name not in supported:
            name = "UNKNOWN"
        evidence[name] = max(evidence.get(name, 0.0), score)
        candidates.append((name, score, item))
    for arrow in ("LEFT", "RIGHT"):
        # The detector may call an illuminated green arrow GREEN_LEFT while
        # detecting the red straight lamp as a separate box.  When both boxes
        # belong to one housing, red applies to straight and the arrow permits
        # only the corresponding turn.
        compatible = {"RED", arrow, "GREEN_" + arrow, "RED_" + arrow}
        if (len(candidates) >= 2
                and {name for name, _, _ in candidates} <= compatible
                and any(name in (arrow, "GREEN_" + arrow, "RED_" + arrow)
                        for name, _, _ in candidates)
                and any(name in ("RED", "RED_" + arrow)
                        for name, _, _ in candidates)):
            # A single detector may report the red lamp, the lit arrow, and
            # an overlapping combined-class box. Merge only one local housing;
            # a remote red from another head must not authorize this turn.
            anchor = next(item for name, _, item in candidates
                          if name in (arrow, "GREEN_" + arrow, "RED_" + arrow))
            if all(item is anchor or _same_signal_housing(anchor, item)
                   for _, _, item in candidates):
                return "RED_" + arrow, min(score for _, score, _ in candidates)
        # A green circle and its lit arrow can be separate detections too.
        # Preserve both permissions only for one spatially coherent housing.
        # A distant arrow, red, yellow, or unsupported label remains ambiguous.
        compatible = {"GREEN", arrow, "GREEN_" + arrow}
        if (len(candidates) >= 2
                and {name for name, _, _ in candidates} <= compatible
                and any(name in ("GREEN", "GREEN_" + arrow)
                        for name, _, _ in candidates)
                and any(name in (arrow, "GREEN_" + arrow)
                        for name, _, _ in candidates)):
            anchor = next(item for name, _, item in candidates
                          if name in (arrow, "GREEN_" + arrow))
            if all(item is anchor or _same_signal_housing(anchor, item)
                   for _, _, item in candidates):
                return "GREEN_" + arrow, min(score for _, score, _ in candidates)
    if len(evidence) != 1 or "UNKNOWN" in evidence:
        return "UNKNOWN", 0.0
    return next(iter(evidence.items()))


def count_signal_housings(objects, min_confidence=0.5):
    """Count plausible heads without treating duplicate bulb boxes as heads.

    Only one coherent combined indication, or strongly overlapping copies of
    one class, collapse to one head. Separate green heads still require map
    association; unanimous colour alone does not identify our traffic light.
    """
    candidates = []
    for item in objects:
        try:
            score = float(item.conf)
        except (AttributeError, ValueError, TypeError, OverflowError):
            continue
        if math.isfinite(score) and min_confidence <= score <= 1.0:
            candidates.append(item)
    if len(candidates) <= 1:
        return len(candidates)
    state, _ = directional_observation(candidates, min_confidence)
    if state in ("RED_LEFT", "RED_RIGHT", "GREEN_LEFT", "GREEN_RIGHT"):
        if all(_same_signal_housing(a, b) for i, a in enumerate(candidates)
               for b in candidates[i + 1:]):
            return 1
    if state == "UNKNOWN":
        return len(candidates)
    # Do not use adjacent-lamp tolerances for duplicate whole-head detections.
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            try:
                boxes = [tuple(float(getattr(obj, key)) for key in
                               ("x_center", "y_center", "width", "height"))
                         for obj in (first, second)]
                if any(not all(math.isfinite(v) for v in box)
                       or box[2] <= 0 or box[3] <= 0 for box in boxes):
                    return len(candidates)
                (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes
                overlap = (max(0.0, min(x1 + w1/2, x2 + w2/2)
                               - max(x1 - w1/2, x2 - w2/2))
                           * max(0.0, min(y1 + h1/2, y2 + h2/2)
                                 - max(y1 - h1/2, y2 - h2/2)))
                union = w1*h1 + w2*h2 - overlap
                if union <= 0 or overlap / union < 0.6:
                    return len(candidates)
            except (AttributeError, TypeError, ValueError, OverflowError):
                return len(candidates)
    return 1


def straight_observation(objects, min_confidence=0.5):
    """Conservative straight-ahead state for consumers without route intent.

    An illuminated turn arrow does not authorize straight travel. Preserve
    directional classes separately for the route-associated maneuver node.
    Conflicting heads never become green, even if one has higher confidence.
    """
    state, confidence = directional_observation(objects, min_confidence)
    if state in ("GREEN", "GREEN_LEFT", "GREEN_RIGHT"):
        return "GREEN", confidence
    if state in ("RED", "RED_LEFT", "RED_RIGHT", "LEFT", "RIGHT"):
        return "RED", confidence
    if state == "YELLOW":
        return state, confidence
    return "UNKNOWN", 0.0


def _class_observation(class_names):
    return straight_observation([SimpleNamespace(class_name=name, conf=1.0)
                                 for name in class_names])[0]


def traffic_signal_has_green(class_names):
    """Only compatible, explicitly recognized green evidence permits straight travel."""
    return _class_observation(class_names) == "GREEN"


def traffic_signal_requires_stop(class_names):
    """Visible conflicting/unsupported heads request a stop; emptiness is unknown."""
    names = list(class_names)
    return bool(names) and not traffic_signal_has_green(names)


class TrafficSignalStopLatch:
    """Latch stops until distinct, continuous green source frames confirm release.

    Empty frames cannot release a stop. Callers enforce source age; repeated
    source stamps never refresh evidence, and callback backlog cannot satisfy
    the confirmation interval unless receipt time also spans that interval.
    """

    def __init__(self, clear_confirmation_s=0.5, max_gap_s=0.8):
        self.clear_confirmation_s = float(clear_confirmation_s)
        self.max_gap_s = float(max_gap_s)
        if (not math.isfinite(self.clear_confirmation_s) or self.clear_confirmation_s < 0
                or not math.isfinite(self.max_gap_s) or self.max_gap_s <= 0):
            raise ValueError("Confirmation must be finite/nonnegative and gap finite/positive")
        self.stop_required = False
        self.last_stamp = self.last_received = self.last_state = None
        self.clear_since = None
        self.clear_received = None
        self.green_samples = 0

    def revoke(self, reset_clock=False):
        self.stop_required = True
        self.clear_since = self.clear_received = None
        self.green_samples = 0
        if reset_clock:
            self.last_stamp = self.last_received = self.last_state = None
        return self.stop_required

    def update(self, class_names, timestamp_sec, received_sec=None):
        names = list(class_names)
        return self.observe(_class_observation(names), timestamp_sec, received_sec,
                            detected=bool(names))

    def observe(self, state, timestamp_sec, received_sec=None, detected=True):
        stamp = float(timestamp_sec)
        received = stamp if received_sec is None else float(received_sec)
        if not math.isfinite(stamp) or stamp <= 0 or not math.isfinite(received):
            return self.revoke()
        evidence = (state, bool(detected))
        if self.last_stamp is not None and stamp <= self.last_stamp:
            if stamp < self.last_stamp or evidence != self.last_state:
                self.revoke()
            return self.stop_required
        if (self.last_stamp is not None and
                (stamp - self.last_stamp > self.max_gap_s
                 or not 0 <= received - self.last_received <= self.max_gap_s)):
            self.revoke()
        self.last_stamp, self.last_received, self.last_state = stamp, received, evidence
        if state != "GREEN":
            self.clear_since = self.clear_received = None
            self.green_samples = 0
            if state in ("RED", "YELLOW") or detected:
                self.stop_required = True
            return self.stop_required
        if self.clear_since is None:
            self.clear_since, self.clear_received = stamp, received
            self.green_samples = 0
            self.stop_required = True
        self.green_samples += 1
        if (self.green_samples >= 2
                and stamp - self.clear_since + 1e-9 >= self.clear_confirmation_s
                and received - self.clear_received + 1e-9 >= self.clear_confirmation_s):
            self.stop_required = False
        return self.stop_required
