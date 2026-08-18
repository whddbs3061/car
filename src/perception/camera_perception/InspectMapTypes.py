"""MGeo 의 타입 코드가 실제로 무엇인지 **눈으로 보고 정하기 위한** 도구.

타입마다 측정값(개수·폭·길이·대시·링크와의 각도·합류점 거리)과, 그 타입만 밝게
칠한 실제 프레임 여러 장을 한 장의 대조표로 묶어 낸다.

--------------------------------------------------------------------------
왜 필요한가
--------------------------------------------------------------------------
NGII 코드표가 저장소 어디에도 없다. `lane_type_def: "ngii_model2"` 라는 문자열만
있고 대응표가 없어서 코드가 무슨 선인지는 추측할 수밖에 없다. 그리고 추측은
실제로 두 번 틀렸다:

  - `530` 을 정지선으로 의심했으나 링크와 95% 평행이라 아니었다
  - `sign_type 534` 를 횡단보도로 넣었더니 맨 아스팔트 위에 얹혔다

숫자만으로는 판단이 안 서고, 화면만 봐서는 어느 게 그 타입인지 모른다.
**둘을 나란히 놓아야** 정해진다.

--------------------------------------------------------------------------
쓰는 법
--------------------------------------------------------------------------
    python InspectMapTypes.py --recording <녹화폴더> --mgeo <MGeo폴더> \
        --cam-set <cam_set.json> --set lane_boundary_set

`--set` 에 MGeo 파일 이름을 확장자 없이 준다. **아직 없는 파일에도 그대로 쓰인다**:

    --set surface_marking_set     (정지선·노면표시)
    --set parking_space_set       (주차구역)
    --set singlecrosswalk_set     (횡단보도)

타입 필드(`lane_type` / `sign_type` / `type` ...)와 선/면 여부는 자동으로 찾는다.
"""

import argparse
import collections
import json
import math
import os

import cv2
import numpy as np

import GenerateLabels as GL


# 타입 코드가 들어 있을 만한 필드 이름 (앞에 있는 것부터 찾는다)
TYPE_FIELDS = ("lane_type", "sign_type", "type", "sub_type", "type_code", "lane_code")

HILITE = (0, 255, 0)        # 지금 보는 타입 = 초록
OTHERS = (110, 110, 110)    # 나머지 = 회색

_frames = {}


def _first(v):
    return v[0] if isinstance(v, list) and v else v


def load_set(mgeo_dir, name):
    """MGeo 파일 하나를 읽어 (레코드, 타입필드명, 선/면) 을 돌려준다."""
    path = os.path.join(mgeo_dir, name + ".json")
    if not os.path.isfile(path):
        have = [f[:-5] for f in sorted(os.listdir(mgeo_dir)) if f.endswith(".json")]
        raise SystemExit(f"파일이 없습니다: {path}\n  이 폴더에 있는 것: {have}")
    with open(path, encoding="utf-8") as fp:
        raw = json.load(fp)
    recs = raw if isinstance(raw, list) else list(raw.values())[0]
    recs = [r for r in recs if isinstance(r, dict) and r.get("points")]
    if not recs:
        raise SystemExit(f"{name}.json 에 points 를 가진 레코드가 없습니다 (빈 파일).")

    field = next((f for f in TYPE_FIELDS if f in recs[0]), None)
    # lane_width 가 있으면 선(폴리라인), 없으면 면(폴리곤)으로 본다.
    kind = "line" if "lane_width" in recs[0] else "polygon"
    return recs, field, kind


def stats_of(recs, kind):
    """측정값. 추측은 넣지 않고 파일에 실제로 든 값만 집계한다."""
    out = {"개수": len(recs)}
    lens, areas = [], []
    for r in recs:
        p = np.asarray(r["points"], dtype=np.float64)[:, :2]
        lens.append(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()))
        areas.append(float((p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min())))
    if kind == "line":
        out["길이중앙(m)"] = round(float(np.median(lens)), 1)
    else:
        out["bbox면적중앙(m2)"] = round(float(np.median(areas)), 1)

    for f, label in (("lane_width", "폭(m)"), ("lane_color", "색"),
                     ("lane_shape", "형태"), ("double_line_interval", "겹선간격")):
        if f in recs[0]:
            c = collections.Counter(str(_first(r.get(f))) for r in recs)
            out[label] = c.most_common(1)[0][0]
    if "dash_interval_L1" in recs[0]:
        c = collections.Counter((r.get("dash_interval_L1"), r.get("dash_interval_L2"))
                                for r in recs)
        out["대시(칠,빈)"] = str(c.most_common(1)[0][0])
    out["점개수중앙"] = int(np.median([len(r["points"]) for r in recs]))
    return out


def link_geometry(mgeo_dir):
    """합류 노드 위치와 링크 방향. 코드 의미를 가릴 때 가장 큰 단서다."""
    with open(os.path.join(mgeo_dir, "link_set.json"), encoding="utf-8") as fp:
        raw = json.load(fp)
    recs = raw if isinstance(raw, list) else list(raw.values())[0]
    inn = collections.defaultdict(list)
    seg = []
    for r in recs:
        if r.get("to_node_idx"):
            inn[r["to_node_idx"]].append(r)
        pts = r.get("points") or []
        for i in range(len(pts) - 1):
            seg.append((pts[i][0], pts[i][1],
                        math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])))
    merge = np.array([[v[0]["points"][-1][0], v[0]["points"][-1][1]]
                      for v in inn.values() if len(v) >= 2])
    return merge, np.array(seg)


def relation_to_links(recs, merge, seg):
    """합류점까지 거리와 링크와의 각도.

    "정지선은 링크와 직각" 처럼 의미를 가르는 판단이 이 두 값에서 나온다.
    """
    d_merge, ang = [], []
    for r in recs:
        p = np.asarray(r["points"], dtype=np.float64)[:, :2]
        c = p.mean(0)
        if len(merge):
            d_merge.append(float(np.sqrt(((merge - c) ** 2).sum(1)).min()))
        if len(p) >= 2 and len(seg):
            h = math.atan2(p[-1][1] - p[0][1], p[-1][0] - p[0][0])
            near = seg[np.argsort(((seg[:, :2] - c) ** 2).sum(1))[:12]]
            diffs = []
            for _, _, hn in near:
                d = abs(math.degrees(math.atan2(math.sin(h - hn), math.cos(h - hn))))
                diffs.append(d if d <= 90 else 180 - d)
            ang.append(min(diffs))

    out = {}
    if d_merge:
        out["합류점거리중앙(m)"] = round(float(np.median(d_merge)), 1)
        out["합류15m이내"] = f"{(np.array(d_merge) < 15).mean() * 100:.0f}%"
    if ang:
        out["링크와각도중앙"] = f"{np.median(ang):.0f}도 (0=평행, 90=직각)"
    return out


def best_frames(recs, meta, count):
    """이 타입이 자차에 가장 가까웠던 프레임들. 서로 겹치지 않게 고른다."""
    ego = np.array([[m["pos_x"], m["pos_y"]] for m in meta])
    cand = []
    for r in recs:
        p = np.asarray(r["points"], dtype=np.float64)[:, :2]
        d = np.sqrt(((ego[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
        i = int(d.argmin())
        cand.append((float(d[i]), i))
    cand.sort()

    picked = []
    for dist, i in cand:
        # 지나쳐 버린 뒤보다 조금 앞에서 봐야 화면에 들어온다
        i = max(0, i - 12)
        if all(abs(i - j) > 40 for _, j in picked):
            picked.append((dist, i))
        if len(picked) >= count:
            break
    return picked


def draw_layer(cam, recs, color, thick, canvas, links, row, fallback_z, kind):
    ex, ey = row["pos_x"], row["pos_y"]
    z0 = GL.road_height(links, row.get("link_id", ""), ex, ey, fallback_z)
    for r in recs:
        p = np.asarray(r["points"], dtype=np.float64)
        ego = GL.to_ego(p, ex, ey, row["yaw"], z0)
        if (ego[:, 0] > GL.MAX_RANGE).all() or (np.abs(ego[:, 1]) > GL.MAX_RANGE).all():
            continue
        if kind == "polygon":
            clipped = GL.clip_near(cam.to_camera(ego))
            if clipped is None:
                continue
            uv, _ = cam.project_camera(clipped)
            cv2.fillPoly(canvas, [uv.astype(np.int32)], color)
        else:
            uv, ok = cam.project(ego)
            pts = uv[ok]
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts.astype(np.int32)], False, color, thick)


def sheet(picks, target, others, cam, links, meta, fallback_z, kind, bonnet, title, info):
    """대조표 한 장: 위에 측정값, 아래에 그 타입을 칠한 프레임들."""
    tiles = []
    for dist, idx in picks:
        img = _frames.get(idx)
        if img is None:
            continue
        canvas = np.zeros_like(img)
        draw_layer(cam, others, OTHERS, 2, canvas, links, meta[idx], fallback_z, kind)
        draw_layer(cam, target, HILITE, 4, canvas, links, meta[idx], fallback_z, kind)
        if bonnet is not None:
            canvas[bonnet > 0] = 0

        hit = canvas.any(axis=2)
        out = img.copy()
        out[hit] = (out[hit] * 0.35 + canvas[hit] * 0.65).astype(np.uint8)
        cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(out, f"frame {idx}   nearest {dist:.1f}m", (6, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(out)

    if not tiles:
        return None
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])

    head = np.zeros((26 * (len(info) + 2), grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(head, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, HILITE, 2)
    for i, (k, v) in enumerate(info.items()):
        cv2.putText(head, f"{k} : {v}", (10, 26 * (i + 2) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return np.vstack([head, grid])


def main():
    ap = argparse.ArgumentParser(
        description="MGeo 타입 코드를 측정값 + 실제 화면으로 함께 보여준다 "
                    "(코드 의미를 직접 판단하기 위한 도구)")
    ap.add_argument("--recording", required=True, help="meta.jsonl 과 프레임이 있는 폴더")
    ap.add_argument("--mgeo", required=True, help="MGeo 폴더")
    ap.add_argument("--cam-set", required=True, help="cam_set.json")
    ap.add_argument("--set", dest="setname", default="lane_boundary_set",
                    help="볼 MGeo 파일 이름 (확장자 없이). 예: surface_marking_set")
    ap.add_argument("--sensor-id", type=int, default=1)
    ap.add_argument("--fov-axis", default="horizontal", choices=["horizontal", "vertical"])
    ap.add_argument("--per-type", type=int, default=4, help="타입당 보여줄 프레임 수")
    ap.add_argument("--out", default="inspect", help="결과 폴더")
    args = ap.parse_args()

    recs, field, kind = load_set(args.mgeo, args.setname)
    meta = GL.load_meta(os.path.join(args.recording, "meta.jsonl"))
    links = GL.load_link_elevation(args.mgeo)
    merge, seg = link_geometry(args.mgeo)
    fallback_z = float(np.median([np.asarray(r["points"])[:, 2].mean() for r in recs]))

    by = collections.defaultdict(list)
    for r in recs:
        by[str(_first(r.get(field))) if field else "(타입필드없음)"].append(r)

    print("=" * 68)
    print(f"{args.setname}.json — 레코드 {len(recs)}개, 형태 {kind}, 타입 필드 '{field}'")
    print(f"타입 분포: { {k: len(v) for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))} }")
    print("=" * 68)

    source = GL.find_frame_source(args.recording)
    bonnet_path = os.path.join(args.recording, "bonnet_mask.png")
    bonnet = cv2.imread(bonnet_path, cv2.IMREAD_GRAYSCALE) if os.path.isfile(bonnet_path) else None

    picks = {t: best_frames(rs, meta, args.per_type) for t, rs in by.items()}
    need = sorted({i for v in picks.values() for _, i in v})
    _frames.update(GL.read_frames(source, need))
    if not _frames:
        raise SystemExit("프레임을 읽지 못했습니다.")
    h, w = next(iter(_frames.values())).shape[:2]
    cam = GL.load_camera(args.cam_set, args.sensor_id, args.fov_axis).scaled(w, h)

    os.makedirs(args.out, exist_ok=True)
    for t, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        info = stats_of(rs, kind)
        info.update(relation_to_links(rs, merge, seg))
        ids = {id(r) for r in rs}
        others = [r for r in recs if id(r) not in ids]
        img = sheet(picks[t], rs, others, cam, links, meta, fallback_z, kind, bonnet,
                    f"{args.setname}   {field} = {t}", info)
        if img is None:
            print(f"  [{t:>6}] {len(rs):4d}개 — 이 녹화본에서는 안 보임 (건너뜀)")
            continue
        path = os.path.join(args.out, f"{args.setname}_{t}.png")
        cv2.imwrite(path, img)
        print(f"  [{t:>6}] {len(rs):4d}개 → {os.path.basename(path)}")

    print(f"\n저장 완료: {args.out}/")
    print("→ 초록이 그 타입, 회색이 나머지입니다. 위쪽 측정값과 아래 화면을 같이 보고 정하세요.")


if __name__ == "__main__":
    main()
