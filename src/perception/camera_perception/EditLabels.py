"""라벨에서 뺄 차선 경계를 화면에서 클릭해 고른다.

`GenerateLabels.py --save-vectors` 가 만든 `vectors/*.json` 을 읽어 각 경계를
원본 위에 그려 주고, 클릭하면 그 경계를 제외 목록에 넣는다. 결과는 녹화 폴더의
`label_exclude.json` 에 저장되고, 다음번 `GenerateLabels.py` 실행 때 자동으로
적용된다.

--------------------------------------------------------------------------
왜 프레임이 아니라 "지도 레코드" 단위인가
--------------------------------------------------------------------------
반대편 차도의 차선은 어느 프레임에서 보든 **같은 지도 레코드**(예:
`B2256W000395`)다. 그래서 한 프레임에서 한 번만 지우면 그 레코드가 보이는
모든 프레임에서 같이 빠진다 — 수천 장을 일일이 지울 필요가 없다.

반대로, 같은 선이라도 프레임마다 다르게 처리하고 싶다면(예: 이 프레임에서만
차에 가려짐) 이 도구로는 안 되고 픽셀 단위 편집이 필요하다.

--------------------------------------------------------------------------
사용법
--------------------------------------------------------------------------
    python EditLabels.py --recording recordings/1_2stage_test

조작:
    마우스 좌클릭   가장 가까운 경계를 제외/복구 토글
    x               **이 프레임을 통째로 제외/복구 토글**
    n / p           다음 / 이전 프레임
    j               제외되지 않은 다음 프레임으로 건너뛰기
    u               마지막 토글 취소
    a               현재 프레임에서 제외된 것 전부 복구
    s               저장 (종료할 때도 자동 저장 여부를 묻는다)
    q 또는 ESC      종료

경계 제외는 **지도 레코드 단위**라 스크립트 폴더의 `lane_exclude_global.json`
에 저장되고 **앞으로 찍는 모든 녹화본에 계속 적용된다** — 반대편 차도는 어느
바퀴를 돌아도 같은 레코드이므로 바퀴마다 다시 지울 필요가 없다.
프레임 제외는 그 녹화본에만 해당하므로 녹화 폴더의 `label_exclude.json` 에
남는다. 제외한 프레임은 다음 GenerateLabels.py 실행 때 건너뛰고, 이미
만들어져 있던 masks/vectors 파일도 같이 지워진다.

화면에는 크롭선(학습에 쓰지 않고 잘려나가는 위쪽 영역)도 같이 표시된다.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

EXCLUDE_FILENAME = "label_exclude.json"
# 경계 제외는 지도에 딸린 정보라 전 녹화본이 공유한다 (GenerateLabels 와 같은 파일).
GLOBAL_EXCLUDE_FILENAME = "lane_exclude_global.json"
GLOBAL_EXCLUDE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   GLOBAL_EXCLUDE_FILENAME)
CLICK_RADIUS = 25          # 이 픽셀 안에서 가장 가까운 선을 고른다

CLASS_COLORS = {
    "white_solid": (0, 0, 255), "white_dashed": (255, 0, 0),
    "yellow": (0, 255, 255), "stopline": (255, 255, 0),
    "guide": (255, 0, 255), "zone": (0, 165, 255), "crosswalk": (0, 255, 0),
}
EXCLUDED_COLOR = (110, 110, 110)


def _dist_to_polyline(pt, pts):
    """점에서 폴리라인까지의 최단거리 (픽셀)."""
    if len(pts) < 2:
        return float(np.linalg.norm(np.asarray(pt) - pts[0]))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    denom = np.maximum((ab * ab).sum(1), 1e-9)
    t = np.clip(((pt - a) * ab).sum(1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.sqrt(((pt - proj) ** 2).sum(1)).min())


class Editor:
    def __init__(self, recording):
        self.recording = recording
        self.vec_dir = os.path.join(recording, "vectors")
        if not os.path.isdir(self.vec_dir):
            raise SystemExit(
                f"vectors/ 가 없습니다: {self.vec_dir}\n"
                "  먼저 GenerateLabels.py 를 --save-vectors 로 실행하세요.")
        self.files = sorted(f for f in os.listdir(self.vec_dir) if f.endswith(".json"))
        if not self.files:
            raise SystemExit("vectors/ 에 JSON 이 없습니다.")
        self.path = os.path.join(recording, EXCLUDE_FILENAME)
        self.excluded = set()           # 지도 경계 idx — 전역, 모든 녹화본에 적용
        self.excluded_frames = set()    # 프레임 번호 — 이 녹화본에만
        g_b, _ = self._read(GLOBAL_EXCLUDE_PATH)
        l_b, l_f = self._read(self.path)
        self.excluded = g_b | l_b       # 예전에 녹화 폴더에 남긴 것도 흡수
        self.excluded_frames = l_f
        print(f"기존 제외: 경계 {len(self.excluded)}개(전역 {len(g_b)}), "
              f"프레임 {len(self.excluded_frames)}장")
        self.pos = 0
        self.history = []
        self.dirty = False
        self.load()

    @staticmethod
    def _read(path):
        if not os.path.isfile(path):
            return set(), set()
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            return set(data), set()
        return set(data.get("excluded", [])), set(int(i) for i in data.get("excluded_frames", []))

    def load(self):
        with open(os.path.join(self.vec_dir, self.files[self.pos]), encoding="utf-8") as fp:
            self.data = json.load(fp)
        img = cv2.imread(os.path.join(self.recording, self.data["image"]))
        if img is None:
            raise SystemExit(f"프레임을 못 읽었습니다: {self.data['image']}")
        # **원본 전체를 보여 준다.** 크롭으로 잘려나가는 위쪽까지 봐야 그 컷이
        # 뭘 버리고 있는지 판단할 수 있다. 구조화 라벨의 좌표는 크롭 기준이므로
        # crop_top 만큼 내려서 원본 좌표계에 맞춘다 (클릭 좌표와도 같아진다).
        self.crop_top = int(self.data.get("crop_top", 0))
        self.img = img
        self.polys = [(b, np.asarray(b["points_uv"], dtype=np.float64)
                       + np.array([0.0, self.crop_top]))
                      for b in self.data["boundaries"]]

    def frame_dropped(self):
        return self.data["idx"] in self.excluded_frames

    def draw(self):
        vis = self.img.copy()
        if self.crop_top:
            # 잘려나가는 영역은 어둡게 덮고 경계선을 긋는다.
            top = vis[:self.crop_top]
            vis[:self.crop_top] = (0.35 * top).astype(np.uint8)
            cv2.line(vis, (0, self.crop_top), (vis.shape[1], self.crop_top),
                     (0, 200, 255), 2)
            cv2.putText(vis, f"CROP LINE  y={self.crop_top}  (above = discarded)",
                        (10, self.crop_top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 4)
            cv2.putText(vis, f"CROP LINE  y={self.crop_top}  (above = discarded)",
                        (10, self.crop_top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 200, 255), 1)
        for b, pts in self.polys:
            gone = b["map_idx"] in self.excluded
            col = EXCLUDED_COLOR if gone else CLASS_COLORS.get(b["cls_name"], (200, 200, 200))
            cv2.polylines(vis, [pts.astype(np.int32)], False, col, 1 if gone else 2)

        if self.frame_dropped():
            # 프레임 통째 제외는 한눈에 알아보게 붉게 덮고 테두리를 두른다.
            tint = np.zeros_like(vis); tint[:] = (0, 0, 160)
            vis = (0.6 * vis + 0.4 * tint).astype(np.uint8)
            cv2.rectangle(vis, (2, 2), (vis.shape[1] - 3, vis.shape[0] - 3), (0, 0, 255), 4)
            cv2.putText(vis, "FRAME EXCLUDED  (x to restore)",
                        (vis.shape[1] // 2 - 210, vis.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(vis, "FRAME EXCLUDED  (x to restore)",
                        (vis.shape[1] // 2 - 210, vis.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        n_gone = sum(1 for b, _ in self.polys if b["map_idx"] in self.excluded)
        bar = [
            f"[{self.pos + 1}/{len(self.files)}] idx={self.data['idx']}",
            f"boundaries {len(self.polys)} (excluded {n_gone})",
            f"total: {len(self.excluded)} boundaries, {len(self.excluded_frames)} frames"
            + ("  *unsaved*" if self.dirty else ""),
        ]
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(vis, "   |   ".join(bar), (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, "click=line  x=drop frame  n/p=frame  j=next kept  u=undo  s=save  q=quit",
                    (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        return vis

    def toggle_frame(self):
        i = self.data["idx"]
        if i in self.excluded_frames:
            self.excluded_frames.discard(i)
            print(f"  프레임 복구: idx={i}")
        else:
            self.excluded_frames.add(i)
            print(f"  프레임 제외: idx={i}")
        self.history.append(("frame", i))
        self.dirty = True

    def next_kept(self):
        for k in range(self.pos + 1, len(self.files)):
            idx = int(self.files[k].split(".")[0])
            if idx not in self.excluded_frames:
                self.pos = k
                self.load()
                return
        print("  뒤쪽에 제외되지 않은 프레임이 없습니다.")

    def click(self, x, y):
        best, best_d = None, CLICK_RADIUS
        for b, pts in self.polys:
            d = _dist_to_polyline(np.array([x, y], dtype=np.float64), pts)
            if d < best_d:
                best, best_d = b, d
        if best is None:
            return
        mid = best["map_idx"]
        if mid in self.excluded:
            self.excluded.discard(mid)
            act = "복구"
        else:
            self.excluded.add(mid)
            act = "제외"
        self.history.append(("boundary", mid))
        self.dirty = True
        print(f"  {act}: {mid}  ({best['cls_name']}, lane_id={best.get('lane_id')})")

    def undo(self):
        if not self.history:
            return
        kind, key = self.history.pop()
        target = self.excluded if kind == "boundary" else self.excluded_frames
        if key in target:
            target.discard(key)
        else:
            target.add(key)
        self.dirty = True
        print(f"  되돌림: {kind} {key}")

    def restore_frame(self):
        n = 0
        for b, _ in self.polys:
            if b["map_idx"] in self.excluded:
                self.excluded.discard(b["map_idx"])
                n += 1
        if n:
            self.dirty = True
            print(f"  이 프레임의 제외 {n}개를 복구했습니다.")

    def save(self):
        # 경계는 전역 파일로 — 지도 레코드라 다음 바퀴에도 계속 적용된다.
        with open(GLOBAL_EXCLUDE_PATH, "w", encoding="utf-8") as fp:
            json.dump({"excluded": sorted(self.excluded)}, fp,
                      ensure_ascii=False, indent=1)
        # 프레임은 이 녹화본에만.
        with open(self.path, "w", encoding="utf-8") as fp:
            json.dump({"excluded_frames": sorted(self.excluded_frames)}, fp,
                      ensure_ascii=False, indent=1)
        self.dirty = False
        print(f"저장: 경계 {len(self.excluded)}개 -> {GLOBAL_EXCLUDE_PATH} (전역)")
        print(f"      프레임 {len(self.excluded_frames)}장 -> {self.path}")

    def run(self):
        win = "EditLabels"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(
            win, lambda e, x, y, f, p: self.click(x, y)
            if e == cv2.EVENT_LBUTTONDOWN else None)
        while True:
            cv2.imshow(win, self.draw())
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("n") and self.pos < len(self.files) - 1:
                self.pos += 1; self.load()
            elif k == ord("p") and self.pos > 0:
                self.pos -= 1; self.load()
            elif k == ord("x"):
                self.toggle_frame()
            elif k == ord("j"):
                self.next_kept()
            elif k == ord("u"):
                self.undo()
            elif k == ord("a"):
                self.restore_frame()
            elif k == ord("s"):
                self.save()
        cv2.destroyAllWindows()
        if self.dirty:
            print("저장하지 않은 변경이 있습니다. 저장할까요? [Y/n] ", end="", flush=True)
            if (sys.stdin.readline().strip().lower() or "y").startswith("y"):
                self.save()


def main():
    ap = argparse.ArgumentParser(
        description="라벨에서 뺄 차선 경계를 클릭으로 고른다 (지도 레코드 단위)")
    ap.add_argument("--recording", required=True,
                    help="vectors/ 가 있는 녹화 폴더")
    args = ap.parse_args()
    Editor(args.recording).run()


if __name__ == "__main__":
    main()
