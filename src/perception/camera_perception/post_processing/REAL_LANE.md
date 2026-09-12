# real_lane 사용법

카메라 한 대로 **차선**과 **정지선**을 찾아 `/perception/camera/lane_info` 토픽에
JSON 으로 내보냅니다. 좌표는 전부 **차량 기준 (x 앞, y 왼쪽, 미터)** 입니다.

---

## 1. 실행

```bash
# ROS 노드로
rosrun purepursuit_mgeo real_lane_node.py

# ROS 없이 확인만 (표준출력으로 JSON)
python3 real_lane_node.py --no-ros

# 화면으로 보기
python3 live_pipeline.py
```

**필요한 것**

| | |
|---|---|
| `best.pt` | 6클래스 모델. **저장소에 없습니다** (98MB). 받아서 `camera_perception/` 나 `post_processing/` 에 두면 자동으로 찾습니다 |
| `cam_set.json` | `data/sensors/cam_set.json` 을 자동으로 찾습니다 |
| UDP 1101 | 카메라 |
| UDP 4001 | IMU (없어도 동작합니다) |

---

## 2. 기존 노드와 바꿔 끼우기

`lane_info_runner.py` 에서 **한 줄만** 바꾸면 됩니다.

```python
target = pkg_root / "lane" / "live_lane_info_publisher_v2.py"   # 기존
target = pkg_root / "lane" / "real_lane_node.py"                # 이걸로
```

- 토픽 이름, JSON 키, 포트 전부 같습니다
- `lane_info_semantic_adapter.py`, 런처, 회피 로직은 **고칠 필요 없습니다**

---

## 3. 토픽

### `/perception/camera/lane_info` — 항상 나갑니다

`std_msgs/String` 안에 JSON 이 들어 있습니다. 약 3.8 KB / 프레임, 12Hz.

```python
import json, rospy
from std_msgs.msg import String

def cb(msg):
    info = json.loads(msg.data)
    if info["left_lane"]["detected"]:
        print(info["left_lane"]["type"])        # "yellow" / "white_solid" / ...

rospy.Subscriber("/perception/camera/lane_info", String, cb)
```

**주요 키**

| 키 | 내용 |
|---|---|
| `left_lane` / `right_lane` | 내 왼쪽 / 오른쪽 차선. 아래 표 참고 |
| `left_boundary_points` | 왼쪽 차선을 0.5m 간격 점으로. `[[x, y], ...]` |
| `right_boundary_points` | 오른쪽 차선 점열 |
| `centerline_points` | 두 차선의 중앙선 점열 |
| `lane_width_m` | 차로 폭 |
| `lateral_error_m` | 차로 중앙에서 얼마나 벗어났나 (+면 왼쪽) |
| `heading_error_rad` | 차로 방향과 차량 방향의 차이 |
| `stopline_detected` | 정지선이 보이나 (true/false) |
| `stopline_distance_m` | 정지선까지 거리 |
| `lane_state` | `"both"` / `"left"` / `"right"` / `null` |
| `lane_valid` | 좌우 중 하나라도 있나 |
| `reasons` | 왜 없는지 (`["NO_RIGHT"]` 등) |
| `n_lanes`, `infer_ms`, `post_ms` | 참고용 |

**`left_lane` / `right_lane` 안에 들어 있는 것**

| 키 | 내용 |
|---|---|
| `detected` | 찾았나 |
| `type` | `white_solid` / `white_dashed` / `yellow` / `guide` |
| `dashed` | 점선인가 |
| `coef` | 곡선 계수 `[a, b, c]`. `y = a·x² + b·x + c` |
| `x_range_m` | 이 곡선이 유효한 거리 범위 `[가까운쪽, 먼쪽]` |
| `track_id` | 같은 차선이면 계속 같은 번호 |
| `age` | 몇 프레임째 보이나 |
| `confidence` | 0~1. 얼마나 믿을 만한가 |
| `inlier_ratio` | 0~1. 곡선이 점들과 얼마나 잘 맞나 |
| **`from_guide`** | **true 면 차선 도색이 아니라 유도선입니다** ↓ |
| **`coasted`** | **true 면 이번에 안 보여서 예측한 값입니다** ↓ |

> **`from_guide` 와 `coasted` 는 꼭 확인하세요.**
>
> `from_guide: true` — 교차로처럼 차선 도색이 없을 때 유도선으로 대신 채운 값입니다.
> 유도선은 "넘으면 안 되는 선"이 아니라 "이쪽으로 가라"는 안내선입니다.
> 장애물 회피에서 벽으로 쓰면 안 됩니다.
>
> `coasted: true` — 이번 프레임에 안 보여서 예측만으로 낸 값입니다.
> `confidence` 가 같이 떨어집니다.

**`straddling_lane`** — 차선 변경 중 **밟고 있는 선**입니다. 평소엔 `null` 입니다.
이게 있으면 `left_lane` 은 그 선의 왼쪽, `right_lane` 은 오른쪽이 됩니다.

**`stopline`** — 정지선 상세

| 키 | 내용 |
|---|---|
| `distance_m` | 거리 |
| `covers_front` | 차 정면을 실제로 관측했나 (항상 true. 아니면 아예 안 내보냅니다) |
| `n_blobs` | 2 이상이면 횡단보도일 수 있습니다 |
| `inlier_ratio` | 직선이 얼마나 잘 맞나 |

### `/perception/camera/lane_info_diag` — 문제 생겼을 때

단계별로 뭐가 몇 개 나왔고, 안 나왔으면 왜 안 나왔는지 들어 있습니다.
5번 항목 참고.

---

## 4. 단계별로 뭘 얻을 수 있나

```
카메라
  ↓
1. Segmentation      차선/정지선 픽셀 분류
  ↓
2. Morphology        보닛·노이즈 제거
  ↓
3. Lane pixels       차선 픽셀을 점으로
  ↓
4. Calibration       점을 지면에 투영 → 여기부터 차량 좌표 (m)
  ↓                        └→ 정지선 (따로 처리)
6. Boundary          점들을 차선별로 묶기
  ↓
9. Curve fitting     차선마다 곡선 맞추기
  ↓
10~11. Tracking      프레임 넘어 같은 차선 추적
  ↓
12. Lane ID          왼쪽 +1 / 오른쪽 -1 / 밟고 있는 선 0
```

| 단계 | 무엇을 주나 | 좌표 | 받는 법 |
|---|---|---|---|
| 1~2 | 클래스 마스크 | 이미지 | 토픽 없음 (너무 큼. 코드에서 직접) |
| 3~4 | 차선 픽셀 | **차량** | `~publish_lane_pixels:=true` → `lane_points` |
| 6 | 차선별 점 묶음 | **차량** | `~publish_boundaries:=true` → `boundaries` |
| 9 | 곡선 계수 전부 | **차량** | 기본 ON → `curves` |
| 10~11 | `track_id`, `confidence` | — | `left_lane.track_id` 등 |
| 12 | 최종 차선 + 제어값 | **차량** | 기본 ON → `left_lane` / `*_points` / `lateral_error_m` |
| 정지선 | 거리 | **차량** | 기본 ON → `stopline_distance_m` |

**켜는 법**

```bash
rosrun ... real_lane_node.py _publish_boundaries:=true _publish_lane_pixels:=true
python3 real_lane_node.py --no-ros --publish-boundaries
```

| 옵션 | 기본 | 추가 크기 |
|---|---|---|
| `~publish_curves` | ON | (기본 포함) |
| `~publish_boundaries` | OFF | +4.4 KB/프레임 |
| `~publish_lane_pixels` | OFF | +4.8 KB/프레임 |
| `~publish_diag` | ON | 1.4 KB/프레임 |

> 1~2단계 마스크는 점으로 안 내보냅니다. 프레임당 13,000점이라 2.2 MB/s 입니다.
> 마스크가 필요하면 코드에서 직접 쓰세요.

**차선을 장애물처럼 쓰려면** `left_boundary_points` / `right_boundary_points` 를
쓰세요. 0.5m 간격 점열이라 그대로 벽으로 넣으면 됩니다. 마스크보다 훨씬 가볍습니다.

---

## 5. 안 될 때

### 진단 토픽 보기

```bash
rostopic echo /perception/camera/lane_info_diag
```

| 증상 | 볼 곳 |
|---|---|
| 차선이 아예 없음 | `ground_kept` 가 0 → 모델/카메라 문제. 아니면 `boundary` |
| 경계는 있는데 차선이 없음 | `fit.failed`, `fit.curv` |
| 곡선은 있는데 좌우 번호를 못 받음 | `lane_id.off_axis`, `ego_left`, `ego_right` |
| 정지선이 없음 | `stopline.reason` |
| 유도선이 안 잡힘 | `lane_id.guide_reject`, `guide_link` |
| 교차로에서 이상함 | `track.junction` (0=평범, 1=교차로) |
| `track_id` 가 자꾸 바뀜 | `tracks[].conf` |

### 화면으로 보기

```bash
python3 live_pipeline.py
```

| 키 | 동작 |
|---|---|
| `1` `2` `3` `4` `6` `9` `0` | 그 단계까지만 (0 = 마지막) |
| `k` | 추적 모드 바꾸기 |
| `t` | 위에서 본 화면 켜고 끄기 |
| `p` | 일시정지 |
| `s` | 화면 저장 |
| `q` | 종료 |

화면 라벨: `+1 yellow 0.87 #14` = 왼쪽 차선 / 황색 / 신뢰도 0.87 / track_id 14.
`G` 가 붙으면 유도선, **가는 선**은 예측값입니다.

### 단계를 끊어서 보기

```bash
python3 real_lane_node.py --no-ros --stage 6     # 6단계까지만
```

---

## 6. 코드

| 파일 | |
|---|---|
| **`real_lane.py`** | **차선 인식 전부. 고칠 곳은 여기입니다** |
| `real_lane_node.py` | 위를 돌려서 토픽으로 내보내는 노드 |
| `morai_camera.py` | 카메라 UDP 수신 |
| `morai_imu.py` | IMU UDP 수신 |
| `live_pipeline.py` | 화면으로 보는 뷰어 |
| `pipeline/` | `real_lane.py` 를 만든 원본. **참고용이고 고치지 않습니다** |
| `tools/` | 녹화·평가 도구 (주행에는 안 씁니다) |

`real_lane.py` 안의 주요 함수 — 단계 순서대로입니다.

```python
seg = Segmenter()          # 모델·카메라 설정을 한 번만 올림
tracker = Tracker()        # 프레임 넘어 상태를 들고 있음
link = GuideLink()         # 유도선 연결 기억

mask, crop = seg.apply(frame)              # 1단계
clean, _   = morphology(mask, seg.bonnet)  # 2단계
pts, _     = lane_pixels(clean, occluded=seg.bonnet)   # 3단계
gnd, _     = to_ground(pts, seg.cam)       # 4단계 → 차량 좌표
bounds, _  = group_boundaries(gnd)         # 6단계
curves, _  = fit_curves(bounds, rng=rng)   # 9단계
tracked, _ = tracker.update(curves, dt=dt) # 10~11단계
lanes, _   = assign_lane_ids(tracked, guide_link=link)  # 12단계
stop, _    = detect_stopline(clean, seg.cam)            # 정지선
```

모든 함수가 `(결과, 통계)` 를 돌려줍니다. 통계가 진단 토픽에 실리는 내용입니다.

자주 만지게 되는 값들은 `real_lane.py` 안에 상수로 모여 있고, 각각 왜 그 값인지
주석에 적혀 있습니다.

| 상수 | 기본 | 뜻 |
|---|---|---|
| `MISS_MAX_M` | 2.3 | 실선이 끊겨도 이어붙일 최대 거리 |
| `MISS_MAX_BY_CLASS` | 점선 6.0 / 유도선 4.0 | 점선은 더 멀리 이어붙임 |
| `MAX_CURV_LANE` | 0.05 | 이보다 많이 휘면 차선이 아님 |
| `CONF_OUT` / `CONF_DROP` | 0.30 / 0.15 | 내보낼 / 버릴 신뢰도 |
| `EGO_MAX_SLOPE` | 0.5 | 차와 너무 어긋난 선은 내 차선이 아님 |
| `STRADDLE_Y_M` | 0.5 | 이보다 가까우면 "밟고 있는 선" |
| `REQUIRE_FRONT` | True | 정면을 안 본 정지선은 안 내보냄 |

---

## 7. 아직 안 되는 것

- **교차로에서 좌우 번호가 흔들릴 수 있습니다.** 
- **가림(앞차 등) 뒤 차선은 복원하지 않습니다.** 
- **차로 폭이 일정하지 않습니다.**
---

## 8. 그 외 파일
 real_lane.py        3,070줄. 후처리 전부
    real_lane_node.py   퍼블리셔
    morai_imu.py        IMU 수신
    morai_camera.py     +15/-2 (with_stamp 만 추가)   ← 기존 파일 중 유일한 수정
    REAL_LANE.md        사용법
    live_pipeline.py    뷰어
    tools/              녹화·평가
