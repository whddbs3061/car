#!/usr/bin/env python3
"""후처리를 **눈으로 보면서 고치는** 도구 (drive6 같은 녹화 영상용).

===========================================================================
실행 (그대로 복사해서 붙여넣기)
===========================================================================
cd C:/MSC/AutoMobility/car/src/perception/camera_perception

  1) 한 번만: 모델 출력 캐시 만들기 (drive6 2772프레임 ≈ 21분)
     C:/Users/user/anaconda3/envs/vision_env/python.exe -u ReviewLaneEgo.py --build-cache

  2) 그 뒤로는 바로 리뷰
     C:/Users/user/anaconda3/envs/vision_env/python.exe ReviewLaneEgo.py
===========================================================================

**왜 따로 만들었나.** `LaneEgoSelect.py --source drive.mp4` 는 재생만 되고
되감기·프레임 점프가 없다. 상수 하나 고칠 때마다 처음부터 다시 돌려야 하는데
이 PC 실측이 추론 421ms / 후처리 49ms 다. 즉 상수를 고쳐 보는 시간의 90% 가
**바뀌지도 않은 모델 추론을 다시 하는 데** 쓰인다. GPU 는 못 쓴다 (torch.cuda
False, cv2.dnn 은 이 빌드에서 백엔드 지정 자체가 무시되고, onnxruntime CPU 는
651ms 로 더 느리다).

그래서 모델 출력(ll/da 마스크)을 프레임마다 한 번만 계산해 파일에 packbits 로
넣어 두고, 리뷰할 때는 그걸 먹인다. 후처리만 도니까 재생이 ~20fps 로 돌고
되감기·점프·코드 재적용이 즉시 된다.

**검출·후처리 코드는 여기에 한 줄도 복사하지 않는다.** `LaneEgoSelect` 를
import 해서 그 `LaneEgoSelectDetector` / `process_frame` 을 그대로 쓴다.
복사본을 두면 "여기서 본 결과"와 "실주행 결과"가 갈린다.

캐시하는 것은 `_run_segmentation()` 의 **날것 출력**이다. CLOSE 모폴로지,
컬러 폴백, ROI, 성분 병합, 쌍 선택, 정지선, 폭 모델은 전부 캐시 뒤에 있으므로
평소대로 튜닝된다. 반대로 **추론 단계 자체**(blob 스케일, 이진화 기준)를 고칠
때는 캐시가 의미를 잃으므로 `--no-cache` 로 돌린다.

---------------------------------------------------------------------------
조작
---------------------------------------------------------------------------
    space   재생 / 일시정지
    n, →    다음 프레임          p, ←    이전 프레임
    .       +10                  ,       -10
    ]       +100                 [       -100
    g       프레임 번호로 점프 (터미널에 입력)
    r       고친 후처리 코드 다시 읽어 현재 프레임에 적용
    s       현재 화면 png 저장
    q, ESC  종료
    아래 슬라이더로도 이동한다.

---------------------------------------------------------------------------
되감기가 정확한 이유 (와 정확하지 않을 때)
---------------------------------------------------------------------------
후처리는 상태를 들고 간다 - 직전 fit 재사용, 재사용 카운터, 폭 모델 표본 90개,
정지선 잠금. 그래서 "프레임 k 를 본다"는 것은 k 하나를 계산하는 게 아니라
**k 까지 온 경로**를 재현하는 것이다. 이 도구는 재생하며 SNAP_EVERY 프레임마다
detector 상태를 통째로 스냅샷해 두고, 점프하면 그 이전 스냅샷에서 이어 돌린다
(HUD 에 `state exact`).

스냅샷이 없는 곳으로 처음 점프하면 0번부터 다 돌리는 대신 `--warmup` 프레임
앞에서 시작한다 (HUD 에 `state warm`). 폭 모델 표본이 90프레임, fit 재사용
한도가 10프레임이라 그보다 넉넉히 앞에서 출발하면 상태가 사실상 같아지기
때문이다. 그래도 **완전히 같다고 보장하지는 않으므로** HUD 에 표시한다.
`r` 로 코드를 다시 읽으면 옛 코드로 만든 스냅샷은 전부 버린다.
"""

import argparse
import contextlib
import copy
import importlib
import io
import json
import os
import sys
import time

import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import LaneEgoSelect as LES


DEFAULT_VIDEO = r"C:\MSC\AutoMobility\samples\drive6\drive.mp4"
WINDOW = "ReviewLaneEgo - 후처리 리뷰"

# 재생하며 몇 프레임마다 detector 상태를 스냅샷할지. 20 이면 점프 한 번에
# 최대 19프레임(≈1초)만 다시 돌리면 되고, 2772프레임에 스냅샷 139개다.
SNAP_EVERY = 20

# HUD 에 현재 값을 띄울 후처리 상수들. `r` 로 코드를 다시 읽었을 때 내가 고친
# 값이 실제로 반영됐는지 화면에서 바로 확인하려고 둔다.
WATCH_CONSTS = [
    "MIN_COMPONENT_AREA", "MERGE_X_TOLERANCE", "MERGE_ANGLE_TOLERANCE",
    "MIN_LANE_PIXELS", "MIN_LANE_Y_SPAN", "MAX_FIT_REUSE_FRAMES",
    "MAX_PAIR_COST", "DASH_FILL_RATIO_THRESHOLD",
]

# detector 상태 스냅샷에서 뺄 것들. ONNX 그물은 deepcopy 가 안 되고 상태도 아니다.
NO_COPY = ("seg_net", "seg_output_names")


# ===========================================================================
# 모델 출력 캐시
# ===========================================================================

def cache_paths(video):
    base = os.path.splitext(video)[0] + "_segcache"
    return base + ".npy", base + ".json"


def build_cache(video, resume=True):
    """영상 전체에 ONNX 를 한 번 돌려 ll/da 마스크를 파일에 넣는다.

    마스크는 0/255 이진이라 packbits 로 8배 줄여 넣는다 (프레임당 76KB →
    drive6 2772프레임에 213MB). npy 로 두면 열 때 memmap 이라 통째로 메모리에
    올리지 않고 필요한 프레임만 읽는다.

    중간에 끊겨도 meta 의 done 부터 이어서 만든다 - 21분짜리를 처음부터
    다시 돌리는 것이 제일 나쁘다.
    """
    npy_path, meta_path = cache_paths(video)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"[에러] 영상을 열 수 없습니다: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    det = LES._new_detector()
    if det.seg_net is None:
        raise SystemExit("[에러] 세그멘테이션 모델을 못 읽었습니다.")

    h, w = det.img_height, det.img_width
    packed_w = (w + 7) // 8
    shape = (total, 2, h, packed_w)

    meta = {}
    done = 0
    if resume and os.path.exists(npy_path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fp:
            meta = json.load(fp)
        if meta.get("shape") == list(shape) and meta.get("video_size") == os.path.getsize(video):
            done = int(meta.get("done", 0))
            print(f"[캐시] 이어서 만듭니다 — {done}/{total} 프레임 완료됨")
        else:
            print("[캐시] 기존 캐시가 이 영상과 맞지 않아 새로 만듭니다")
            done = 0

    mode = "r+" if (done > 0 and os.path.exists(npy_path)) else "w+"
    arr = np.lib.format.open_memmap(npy_path, mode=mode, dtype=np.uint8, shape=shape)

    meta.update({
        "video": os.path.abspath(video),
        "video_size": os.path.getsize(video),
        "shape": list(shape),
        "width": w, "height": h, "total": total,
        "model": os.path.abspath(os.path.join(current_dir, "lane_segmentation.onnx")),
        "done": done,
    })

    if done:
        cap.set(cv2.CAP_PROP_POS_FRAMES, done)

    print(f"[캐시] {video}")
    print(f"[캐시] {total}프레임, {os.path.basename(npy_path)} ({arr.nbytes/1e6:.0f}MB)")
    t0 = time.time()
    i = done
    try:
        while i < total:
            ok, image = cap.read()
            if not ok:
                print(f"[캐시] 영상이 {i}프레임에서 끝났습니다 (헤더는 {total})")
                break
            frame = cv2.resize(image, (w, h))
            ll, da = det._run_segmentation(frame)
            arr[i, 0] = np.packbits(ll > 0, axis=-1)
            arr[i, 1] = np.packbits(da > 0, axis=-1)
            i += 1
            if i % 20 == 0 or i == total:
                el = time.time() - t0
                per = el / max(i - done, 1)
                eta = per * (total - i)
                print(f"  {i}/{total}  {per*1000:.0f}ms/프레임  남은 시간 {eta/60:.1f}분",
                      flush=True)
                meta["done"] = i
                with open(meta_path, "w", encoding="utf-8") as fp:
                    json.dump(meta, fp, ensure_ascii=False, indent=2)
    except KeyboardInterrupt:
        print("\n[캐시] 중단 — 다음에 --build-cache 를 다시 주면 여기서 이어집니다")
    finally:
        cap.release()
        arr.flush()
        meta["done"] = i
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)

    print(f"[캐시] {i}/{total} 프레임, {(time.time()-t0)/60:.1f}분")
    return i >= total


def load_cache(video):
    """(memmap 배열, meta) 또는 (None, 사유)."""
    npy_path, meta_path = cache_paths(video)
    if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
        return None, "캐시 파일이 없습니다"
    with open(meta_path, encoding="utf-8") as fp:
        meta = json.load(fp)
    if meta.get("video_size") != os.path.getsize(video):
        return None, "캐시가 다른 영상으로 만들어졌습니다"
    arr = np.load(npy_path, mmap_mode="r")
    if list(arr.shape) != meta.get("shape"):
        return None, "캐시 파일이 깨졌습니다"
    return arr, meta


# ===========================================================================
# 프레임 공급
# ===========================================================================

class FrameSource:
    """영상에서 임의 프레임을 꺼낸다.

    jpg 로 미리 다 디코드해 메모리에 들고 있는 방법도 있지만 재인코딩이
    컬러 폴백(`color_mask`)의 입력을 미세하게 바꾼다. 후처리를 검증하는
    도구가 입력을 건드리면 안 되므로 원본 디코드 결과를 그대로 쓰고,
    대신 최근 프레임만 캐시해서 한두 프레임 되감기를 빠르게 한다.
    """

    def __init__(self, path, cache_size=120):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"[에러] 영상을 열 수 없습니다: {path}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.pos = -1                 # 다음에 read() 하면 나올 프레임 번호 - 1
        self.cache = {}
        self.order = []
        self.cache_size = cache_size

    def get(self, i):
        if i in self.cache:
            return self.cache[i]
        if i != self.pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i - 1
        ok, image = self.cap.read()
        if not ok:
            return None
        self.pos = i
        self.cache[i] = image
        self.order.append(i)
        while len(self.order) > self.cache_size:
            self.cache.pop(self.order.pop(0), None)
        return image

    def release(self):
        self.cap.release()


# ===========================================================================
# 재생기
# ===========================================================================

class Replayer:
    def __init__(self, source, cache, total, warmup=120, live=False):
        self.src = source
        self.cache = cache          # memmap (N,2,H,packed_W) 또는 None
        self.total = total
        self.warmup = warmup
        self.live = live            # True 면 캐시 없이 매 프레임 추론
        self.net = None
        self.det = None
        self.idx = -1
        self.exact = True           # 이 프레임이 0번부터 이어져 온 상태인가
        self.snaps = {}
        self.panel = None
        self.info = ""
        self.mtime = os.path.getmtime(LES.__file__)
        self._new_detector()

    # --- detector ---------------------------------------------------------

    def _new_detector(self):
        """현재 LES 모듈로 detector 를 새로 만든다.

        ONNX 그물은 한 번만 읽어 재사용한다. `r` 로 코드를 다시 읽을 때마다
        모델까지 다시 읽을 이유가 없고, 캐시 모드에서는 그물을 쓰지도 않는다
        (다만 `seg_net is None` 이면 segment_lanes 가 빈 마스크를 돌려주므로
        객체 자체는 있어야 한다).
        """
        # seg_model_path=None 으로 만들고 그물은 아래에서 직접 꽂는다.
        # 생성자가 그때 찍는 안내문("경로가 지정되지 않았습니다")은 여기서는
        # 사실이 아니라 헷갈리기만 하므로 삼킨다.
        with contextlib.redirect_stdout(io.StringIO()):
            det = LES.LaneEgoSelectDetector(640, 480, seg_model_path=None,
                                            seg_interval=1)
        if self.net is None:
            self.net = cv2.dnn.readNet(
                os.path.join(current_dir, "lane_segmentation.onnx"))
        det.seg_net = self.net
        det.seg_output_names = self.net.getUnconnectedOutLayersNames()
        self.det = det

    def _feed(self, i):
        """i 번 프레임의 모델 출력을 detector 에 물린다 (캐시 모드일 때)."""
        if self.live or self.cache is None:
            return
        p = self.cache[i]
        ll = np.unpackbits(p[0], axis=-1)[:, :self.det.img_width] * 255
        da = np.unpackbits(p[1], axis=-1)[:, :self.det.img_width] * 255
        ll = np.ascontiguousarray(ll, dtype=np.uint8)
        da = np.ascontiguousarray(da, dtype=np.uint8)
        # 인스턴스 속성이 클래스 메서드를 가린다. segment_lanes 이후의 CLOSE·
        # 폴백·ROI 는 그대로 도므로 후처리는 실주행과 완전히 같은 코드를 탄다.
        self.det._run_segmentation = lambda img_frame: (ll, da)

    # --- 한 프레임 ---------------------------------------------------------

    def _run(self, i, draw):
        image = self.src.get(i)
        if image is None:
            return False
        # 스냅샷은 **프레임 i 를 처리하기 직전** 상태다. 그래야 "스냅샷 s 를
        # 되돌리고 s 부터 k 까지 돌린다"가 어긋남 없이 성립한다 (처리 후
        # 상태로 저장하면 s == k 인 점프에서 같은 프레임을 두 번 먹인다).
        if i % SNAP_EVERY == 0 and i not in self.snaps:
            self.snaps[i] = (
                copy.deepcopy({k: v for k, v in self.det.__dict__.items()
                               if k not in NO_COPY and k != "_run_segmentation"}),
                self.exact,
            )
        self._feed(i)
        if draw:
            self.panel, self.info = LES.process_frame(self.det, image)
        else:
            # 상태만 이어가면 되는 구간은 그리기를 건너뛴다 (13ms/프레임 절약)
            frame = cv2.resize(image, (640, 480))
            mask = self.det.combine_masks(frame)
            roi = self.det.limit_region(mask)
            self.det.fit_polynomial(roi)
        self.idx = i
        return True

    def goto(self, k, force_cold=False):
        """프레임 k 로 간다. 상태를 어디서부터 이어 돌릴지는 스냅샷이 정한다."""
        k = max(0, min(int(k), self.total - 1))

        if not force_cold and k == self.idx + 1:
            self._run(k, draw=True)
            return

        start = None
        if not force_cold:
            cands = [s for s in self.snaps if s <= k]
            if cands:
                start = max(cands)

        t0 = time.time()
        if start is not None:
            self._restore(start)
            begin = start
        else:
            # 스냅샷이 없다 - 0부터 다 돌리면 후반부는 2분씩 걸린다. warmup
            # 만큼만 앞에서 새 detector 로 출발하고 근사임을 표시한다.
            self._new_detector()
            begin = max(0, k - self.warmup)
            self.exact = (begin == 0)

        for i in range(begin, k):
            if not self._run(i, draw=False):
                break
        self._run(k, draw=True)
        el = time.time() - t0
        if el > 0.5:
            print(f"  ({begin}→{k} {el:.1f}초 재계산)")

    def _restore(self, i):
        state, exact = self.snaps[i]
        self.det.__dict__.update(copy.deepcopy(state))
        self.exact = exact

    # --- 코드 재적용 -------------------------------------------------------

    def reload_code(self):
        """고쳐 놓은 LaneEgoSelect.py 를 다시 읽어 현재 프레임에 적용한다."""
        global LES
        try:
            LES = importlib.reload(LES)
        except Exception as exc:
            print(f"[재적용 실패] {type(exc).__name__}: {exc}")
            return False
        self.mtime = os.path.getmtime(LES.__file__)
        # 옛 코드로 만든 상태는 못 믿는다. 전부 버리고 현재 프레임을 다시 만든다.
        self.snaps.clear()
        self._new_detector()
        here = self.idx
        self.idx = -1
        self.goto(here, force_cold=True)
        return True

    def code_changed(self):
        try:
            return os.path.getmtime(LES.__file__) != self.mtime
        except OSError:
            return False

    def consts(self):
        return [(n, getattr(LES, n)) for n in WATCH_CONSTS if hasattr(LES, n)]


# ===========================================================================
# 화면
# ===========================================================================

def with_hud(panel, rep, playing, dirty):
    """패널 위에 현재 프레임·상태·상수값 띠를 붙인다."""
    bar_h = 58
    out = cv2.copyMakeBorder(panel, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    w = out.shape[1]

    state = "exact" if rep.exact else f"warm({rep.warmup})"
    head = (f"{rep.idx:>5d}/{rep.total-1}   "
            f"{'PLAY' if playing else 'PAUSE'}   state {state}   "
            f"{'LIVE(추론)' if rep.live else 'CACHE'}")
    cv2.putText(out, head, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if rep.exact else (0, 210, 255), 1, cv2.LINE_AA)

    if dirty:
        msg = "코드가 바뀌었습니다 - r"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, msg, (w - tw - 12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 220, 255), 2, cv2.LINE_AA)

    cv2.putText(out, rep.info, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (180, 255, 180), 1, cv2.LINE_AA)

    txt = "  ".join(f"{n.split('_', 1)[-1][:9]}={v:g}" for n, v in rep.consts())
    (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(out, txt, (max(10, w - tw - 12), 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (170, 170, 170), 1, cv2.LINE_AA)
    return out


HELP = """조작
  space 재생/정지   n,→ 다음   p,← 이전   . ±10 ,   ] ±100 [
  g 프레임 점프     r 후처리 코드 다시 읽기   s 화면 저장   q/ESC 종료"""


def review(video, warmup, live, start, frame_cache):
    cache, meta = (None, None) if live else load_cache(video)
    if not live and cache is None:
        print(f"[캐시] {meta} — 먼저 --build-cache 로 만들면 20배 빨라집니다.")
        print("[캐시] 일단 매 프레임 추론합니다 (한 프레임 ~0.45초).")
        live = True

    src = FrameSource(video, cache_size=frame_cache)
    total = src.total
    if cache is not None:
        done = int(meta.get("done", 0))
        total = min(total, cache.shape[0], done)
        if done < src.total:
            print(f"[캐시] {done}/{src.total} 프레임까지만 만들어져 있어 "
                  f"거기까지만 봅니다")

    rep = Replayer(src, cache, total, warmup=warmup, live=live)

    print("=" * 66)
    print(f"🎬 {os.path.basename(video)}  {total}프레임")
    print(HELP)
    print("=" * 66)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    seeking = {"lock": False}

    def on_track(v):
        if seeking["lock"]:
            return
        if v != rep.idx:
            rep.goto(v)

    cv2.createTrackbar("frame", WINDOW, 0, max(total - 1, 1), on_track)

    rep.goto(start)
    if rep.panel is None:
        raise SystemExit(f"[에러] {start} 번 프레임을 읽지 못했습니다")
    playing = False
    out_dir = os.path.join(os.path.dirname(os.path.abspath(video)), "review")

    try:
        while True:
            dirty = rep.code_changed()
            cv2.imshow(WINDOW, with_hud(rep.panel, rep, playing, dirty))
            seeking["lock"] = True
            cv2.setTrackbarPos("frame", WINDOW, rep.idx)
            seeking["lock"] = False

            key = cv2.waitKey(1 if playing else 20) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord(" "):
                playing = not playing
            elif key in (ord("n"), ord("d"), 83):
                playing = False
                rep.goto(rep.idx + 1)
            elif key in (ord("p"), ord("a"), 81):
                playing = False
                rep.goto(rep.idx - 1)
            elif key == ord("."):
                playing = False
                rep.goto(rep.idx + 10)
            elif key == ord(","):
                playing = False
                rep.goto(rep.idx - 10)
            elif key == ord("]"):
                playing = False
                rep.goto(rep.idx + 100)
            elif key == ord("["):
                playing = False
                rep.goto(rep.idx - 100)
            elif key == ord("g"):
                playing = False
                try:
                    v = input(f"프레임 번호 (0~{total-1}): ").strip()
                    if v:
                        rep.goto(int(v))
                except (ValueError, EOFError):
                    print("  숫자가 아닙니다")
            elif key == ord("r"):
                playing = False
                print(f"[재적용] {os.path.basename(LES.__file__)} 다시 읽는 중...")
                if rep.reload_code():
                    print("[재적용] 완료  " +
                          "  ".join(f"{n}={v:g}" for n, v in rep.consts()))
            elif key == ord("s"):
                os.makedirs(out_dir, exist_ok=True)
                out = os.path.join(out_dir, f"review_{rep.idx:06d}.png")
                cv2.imwrite(out, with_hud(rep.panel, rep, playing, dirty))
                print(f"  저장: {out}")
            elif playing:
                if rep.idx + 1 >= total:
                    playing = False
                    print("  영상 끝")
                else:
                    rep.goto(rep.idx + 1)
                    if rep.idx % 20 == 0:
                        print(f"[{rep.idx}/{total-1}] {rep.info}")

    except KeyboardInterrupt:
        pass
    finally:
        src.release()
        cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="녹화 영상으로 차선 후처리를 눈으로 보며 고친다")
    ap.add_argument("--source", default=DEFAULT_VIDEO,
                    help=f"영상 경로 (기본 drive6: {DEFAULT_VIDEO})")
    ap.add_argument("--build-cache", action="store_true",
                    help="모델 출력 캐시를 만든다 (한 번만, drive6 ≈ 21분)")
    ap.add_argument("--no-cache", action="store_true",
                    help="캐시를 쓰지 않고 매 프레임 추론한다 "
                         "(추론 단계 자체를 고칠 때)")
    ap.add_argument("--start", type=int, default=0, help="이 프레임부터 본다")
    ap.add_argument("--warmup", type=int, default=120,
                    help="스냅샷 없는 곳으로 점프할 때 상태를 데우는 프레임 수")
    ap.add_argument("--frame-cache", type=int, default=120,
                    help="원본 프레임을 메모리에 몇 장 들고 있을지")
    args = ap.parse_args(argv)

    video = args.source
    if not os.path.isfile(video):
        raise SystemExit(f"[에러] 영상이 없습니다: {video}")

    if args.build_cache:
        build_cache(video)
        return

    review(video, args.warmup, args.no_cache, args.start, args.frame_cache)


if __name__ == "__main__":
    main()
