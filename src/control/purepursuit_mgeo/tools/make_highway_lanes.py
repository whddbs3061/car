#!/usr/bin/env python3
"""Write config/highway_lanes.json: MGeo lane centrelines of the highway zone.

highway_overtake_escape2 checks its lane count against the map: the lane whose
centreline is nearest the ego.  Lanes are numbered like the planner's
lane_index (0 = entry lane, 1..3 = L1..L3, 4 = L4) and chained from MGeo
links (R_KR_PR_K-city_2025 link_set.json) from the ramp to the lane funnel
at the zone end; each lane ends on its own merge link there.  The exit lane
and the toll links are left out.

    python tools/make_highway_lanes.py            # write the file
    python tools/make_highway_lanes.py --check    # exit 1 if it is stale
"""
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[2]
LINKS = REPO / "data/mgeo/R_KR_PR_K-city_2025/link_set.json"
OUTPUT = PACKAGE / "config/highway_lanes.json"
LANES = (
    (0, "entry", ("A2256W000846", "A2256W000411")),
    (1, "L1", ("A2256W000406", "A2256W000409", "A2256W000420", "A2256W000408", "A2256W000445")),
    (2, "L2", ("A2256W000175", "A2256W000410", "A2256W000430", "A2256W000434", "A2256W000422")),
    (3, "L3", ("A2256W000029", "A2256W000418", "A2256W000431", "A2256W000435", "A2256W000423")),
    (4, "L4", ("A2256W000010", "A2256W000429", "A2256W000433", "A2256W000421", "A2256W000425")),
)


def build() -> str:
    links = {link["idx"]: link for link in json.loads(LINKS.read_text(encoding="utf-8"))}
    lanes = []
    for index, name, chain in LANES:
        points = []
        for idx in chain:
            for p in links[idx]["points"]:
                xy = [round(float(p[0]), 3), round(float(p[1]), 3)]
                if not points or points[-1] != xy:
                    points.append(xy)
        lanes.append({"index": index, "name": name, "links": list(chain), "points": points})
    return json.dumps({"source": "MGeo R_KR_PR_K-city_2025 link_set.json, map frame",
                       "lanes": lanes}, separators=(",", ":")) + "\n"


def main(argv) -> int:
    text = build()
    if "--check" in argv:
        current = OUTPUT.is_file() and OUTPUT.read_text(encoding="utf-8") == text
        print("%s: %s" % ("CURRENT" if current else "STALE (run make_highway_lanes.py)", OUTPUT))
        return 0 if current else 1
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print("wrote %s" % OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
