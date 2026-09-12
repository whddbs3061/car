# real_lane — 차선 인식 후처리

카메라 프레임 하나 → 차선 / 정지선. 조감도(BEV) 래스터를 만들지 않고 픽셀을 바로
지면으로 역투영해 **자차 좌표(미터)** 에서 모든 후처리를 한다.

```
real_lane.py        후처리 전부 (합본, 3,069줄). 여기가 정본이다
real_lane_node.py   이걸 돌려서 ROS 토픽/UDP/표준출력으로 내보내는 노드
pipeline/           합본을 만든 출처. 참고용으로 남겨 둔 것이고 고치지 않는다
live_pipeline.py    시뮬레이터로 단계별 눈으로 확인하는 뷰어
tools/              녹화·평가 도구. 주행 경로에 들어가지 않는다
```

---

## 빨리 써보기

```bash
# ROS 없이, 시뮬레이터만 켜고
python3 real_lane_node.py --no-ros

# ROS 노드로
rosrun purepursuit_mgeo real_lane_node.py

# 화면으로 단계별 확인
python3 live_pipeline.py
```

`best.pt` 가 필요하다. **저장소에 없다** (98MB, gitignore). 받아서
`camera_perception/best.pt` 나 `post_processing/best.pt` 에 두면 찾는다.
6클래스(`scheme: lane6`, epoch 24) 모델이어야 유도선이 나온다.

---

## 기존 노드를 그대로 대체한다

통합 런처는 이렇게 물려 있다.

```
morai_avoidance_highway_roundabout_final.launch
  └─ lane_info_runner.py ──exec──> live_lane_info_publisher_v2.py   (옛 BEV 경로)
                                     │
                                     ├─ UDP 1101 수신
                                     └─ /perception/camera/lane_info  (String, JSON)
                                              │
                                     lane_info_semantic_adapter.py
                                              │
                                     dashed_lane_detected / left_solid_lane_detected /
                                     left_yellow_solid_lane_detected /
                                     right_solid_lane_detected /
                                     stopline_detected / stopline_distance_m
```

**어댑터가 실제로 읽는 키는 5개뿐이다.**

```
left_lane.detected   left_lane.type   left_lane.dashed     (right_lane 도 같은 셋)
stopline_detected    stopline_distance_m
```

`real_lane_node.py` 는 그 키를 포함해 v2 가 내보내던 구조를 **그대로** 낸다
(300프레임으로 키 누락 0 확인). 그래서 **런처도 어댑터도 회피 로직도 고칠 필요가
없다** — `lane_info_runner.py` 가 가리키는 스크립트만 바꾸면 된다.

```python
# lane_info_runner.py
target = pkg_root / "lane" / "live_lane_info_publisher_v2.py"   # 이 줄만
target = pkg_root / "lane" / "real_lane_node.py"                # 이렇게
```

포트도 같은 1101 슬롯이라 새로운 충돌은 없다.

### 좌우 부호가 뒤집혔다

옛 `lane_detection.py` 는 **왼쪽이 음수**(-1)였고, 새 파이프라인은 자차 좌표계
y 부호를 따라 **왼쪽이 +1** 이다. JSON 의 `left_lane` / `right_lane` 은 둘 다
"내 왼쪽/오른쪽 경계" 라는 뜻이므로 노드가 맞춰서 담는다. **JSON 을 읽는 쪽은
바뀔 것이 없다.** `lane_id` 를 직접 쓰는 코드만 주의하면 된다.

---

## 출력

### 항상 나가는 것

| 키 | 뜻 |
|---|---|
| `left_lane` / `right_lane` | `detected`, `type`, `dashed`, `track_id`, `age`, `coef`, `x_range_m`, `n_points`, `confidence`, `inlier_ratio`, `from_guide`, `coasted` |
| `straddling_lane` | **지금 밟고 있는 선** (`lane_id == 0`). 차선 변경 중에만 나온다. 계약에 없던 값이라 기존 소비자는 무시한다 |
| `left_boundary_points` / `right_boundary_points` / `centerline_points` | 자차 좌표 `[[x, y], ...]`, 0.5m 간격 |
| `lane_width_m` | 7m 앞에서 잰 차로 폭 |
| `lateral_error_m` / `heading_error_rad` | 중심선 기준 제어 오차 |
| `stopline_detected` / `stopline_distance_m` | 정지선 |
| `stopline` | 거리·계수·`inlier_ratio`·`covers_front`·`n_blobs` |
| `lane_valid` / `output_status` / `lane_state` / `reasons` | 상태 |
| `n_lanes` / `infer_ms` / `post_ms` | 진단 |

좌표계는 **자차 기준 x 전방 / y 좌측 / 미터** (`coordinate_convention` 에도 적힌다).

크기는 프레임당 중앙 3.8 KB, 12Hz 에서 **44 KB/s** 다.

### 꼭 봐야 할 두 플래그

**`from_guide`** — 도색이 아니라 **유도선**이 그 자리를 채웠다는 뜻이다.
출력의 `left`/`right` 는 제어에 "여기까지 비켜도 된다" 는 의미인데, 유도선은
넘으면 안 되는 선이 아니라 **지나갈 길 힌트**다. 이 플래그를 무시하면 회피
계획이 유도선을 벽으로 오해한다.

**`coasted`** — 이 프레임에 관측이 없어 **예측만으로 낸 값**이다. 점선 구간을
메우라고 있는 것이고, 신뢰도(`confidence`)가 같이 떨어진다.

### 필요하면 켜는 것

```bash
rosrun ... real_lane_node.py _publish_boundaries:=true
python3 real_lane_node.py --no-ros --publish-lane-pixels
```

| 파라미터 | 기본 | 추가되는 키 | 크기 |
|---|---|---|---|
| `~publish_curves` | **ON** | `curves` — 곡선 전부(자차 슬롯 못 받은 것 포함) | 포함됨 |
| `~publish_boundaries` | OFF | `boundaries` — 6단계 경계 점열 | +4.4 KB/프레임 |
| `~publish_lane_pixels` | OFF | `lane_points` — 3단계 픽셀을 **자차 좌표로** 내린 것 | +4.8 KB/프레임 |
| `~publish_diag` | **ON** | `<topic>_diag` 토픽 | 1.4 KB/프레임 |

**2단계 마스크는 점으로 내보내지 않는다.** 실측 프레임당 13,359 픽셀이라 JSON
으로 183 KB, 12Hz 에서 **2.24 MB/s** 다. 마스크가 필요하면 이미지로 받고, 자차
좌표가 필요하면 받는 쪽에서 `real_lane.unproject()` 를 부른다 (같은 파일에 있다).

### 차선을 장애물로 쓰려면

`left_boundary_points` / `right_boundary_points` 를 쓴다. 0.5m 간격으로 샘플링한
폴리라인이라 계획기가 그대로 벽으로 쓸 수 있고, 경계당 40~80점이라 가볍다.
마스크 픽셀을 쓸 이유가 없다 — 계획기가 필요한 건 "넘으면 안 되는 경계선"이지
노이즈 픽셀이 아니다.

---

## 스스로 디버깅하기

### 1. `lane_diag` 를 본다

```bash
rostopic echo /perception/camera/lane_info_diag
```

```json
{
  "ms":     {"1": 5.3, "2": 13.1, "3": 19.9, "4": 19.9, "6": 22.9, "9": 40.9, "12": 0.4},
  "counts": {"boundaries": 5, "curves": 5, "lanes": 4},
  "ground_kept": {"white_solid": 260, "white_dashed": 39, "yellow": 56, "guide": 187},
  "boundary":   {"white_solid": {"seeds": 3, "grown": 3, "short": 0, "kept": 2}},
  "fit":        {"white_solid": {"failed": 0, "curv": 0, "fitted": 2}},
  "track":      {"matched": 4, "new": 1, "coasted": 0, "junction": 0.35, ...},
  "lane_id":    {"order_x": 7.0, "off_axis": 1, "straddling": false,
                 "guide_reject": "도색 사용중", "guide_link": 14},
  "stopline":   {"px": 608, "blobs": 1, "reason": null, "inlier": 0.67},
  "tracks":     [{"id": 14, "cls": "guide", "conf": 0.87, "miss": 0, "age": 23}]
}
```

**증상별로 어디를 볼지:**

| 증상 | 볼 곳 |
|---|---|
| 차선이 아예 안 나옴 | `ground_kept` 가 0 이면 세그멘테이션/보닛, 아니면 `boundary` |
| 경계는 있는데 곡선이 없음 | `fit.failed`(RANSAC 실패) / `fit.curv`(곡률 상한에 걸림) |
| 곡선은 있는데 `lane_id` 를 못 받음 | `lane_id.off_axis`(자차와 어긋난 방향), `ego_left`/`ego_right` |
| 정지선이 안 나옴 | `stopline.reason` — `픽셀 부족` / `직선 적합 실패` / `정면 미포함` |
| 유도선이 안 잡힘 | `lane_id.guide_reject`, `guide_link` |
| 교차로에서 이상함 | `track.junction` 이 올라가는지 (0 정상 ~ 1 교차로) |
| `track_id` 가 자꾸 바뀜 | `tracks[].conf` 가 임계(0.30) 근처를 오르내리는지 |

### 2. 단계를 끊어서 본다

```bash
python3 real_lane_node.py --no-ros --stage 6      # 6단계까지만
python3 live_pipeline.py --stage 6                # 화면으로
```

`live_pipeline.py` 는 실행 중에 키로 바꾼다: `1 2 3 4 6 9 0`(=12단계),
`k` 추적 모드(off/greedy/hungarian/kalman), `t` 조감 패널, `p` 정지, `s` 저장.

### 3. 녹화해서 반복 재생한다

추적은 **상태를 갖는다.** 같은 순서로 다시 돌려야 같은 결과가 나오므로, 시뮬레이터
앞에서 설정을 바꿔가며 비교할 수 없다(그 사이 차가 움직인다).

```bash
python3 tools/record_drive.py --out rec/drive01 --frames 300   # 주행 중에
python3 tools/eval_tracking.py --rec rec/drive01               # 4-way 비교
```

```
지표                  off     greedy   hungarian   kalman
좌+1 존재율          91.0%    ...
우 dy max(m)         1.369    ...
차로폭 표준편차        0.333    ...
```

> `off` 의 `id전환` 이 0 으로 보이는 것은 `track_id` 자체가 없어서 셀 수 없는
> 것이지 0 이 아니다. 연관의 효과는 **greedy 와 hungarian 을 비교**해야 보인다.

---

## 파이프라인

```
Segmentation      6클래스 (배경/백색실선/백색점선/황색/정지선/유도선)
  ↓
Morphology        보닛 제거 + 작은 덩어리 제거
  ↓
Lane pixels       행별 런 중점 → 점열 (이미지 좌표)
  ↓
Calibration       픽셀 → 광선 → 지면 교점 = 자차 좌표 (m)      ← BEV 대체 지점
  ↓                                    └─→ 정지선 (별도 가지)
Boundary          씨앗 + 행진 → 경계별 점 묶음
  ↓
Curve fitting     RANSAC + 2차식 → 계수 + 인라이어
  ↓
Tracking          Kalman predict → Hungarian → track_id → Kalman update → coast
  ↓
Lane ID           7m 에서 좌우 순번. +1 왼쪽 / -1 오른쪽 / 0 밟고 있는 선
```

각 단계가 왜 그렇게 되어 있는지는 `real_lane.py` 의 단계별 주석에 **측정값과
함께** 적혀 있다. 임계를 바꾸기 전에 그 주석을 먼저 읽는 것을 권한다 — 대부분은
실측으로 정한 값이고, 시도했다가 되돌린 것들도 이유와 함께 남겨 놓았다.

### 자주 만지게 되는 임계

| 파일 위치 | 상수 | 기본 | 뜻 |
|---|---|---|---|
| 6단계 | `MISS_MAX_M` | 2.3 | 실선·황색선이 건너뛸 수 있는 빈 구간 |
| 6단계 | `MISS_MAX_BY_CLASS` | 점선 6.0 / 유도선 4.0 | 클래스별 예외 |
| 6단계 | `JOIN_MAX_SLOPE_DIFF` | 0.30 | 이음매에서 허용할 기울기 차 |
| 9단계 | `MAX_CURV_LANE` | 0.05 | 곡률 상한 (반경 10m) |
| 10~11단계 | `CONF_OUT` / `CONF_DROP` | 0.30 / 0.15 | 출력·폐기 신뢰도 |
| 10~11단계 | `Q_RATE_M_PER_S` | 3.0 | 예측 잡음 (자차 운동 없어서 큼) |
| 12단계 | `EGO_MAX_SLOPE` | 0.5 | 자차 슬롯을 받을 최대 기울기 |
| 12단계 | `STRADDLE_Y_M` | 0.5 | 이 안이면 "밟고 있는 선"(0) |
| 정지선 | `REQUIRE_FRONT` | True | 정면을 안 덮으면 안 내보냄 |

---

## 알려진 한계

- **자차 운동이 안 들어온다.** 칼만 예측이 랜덤워크라 급커브에서 coast 가 뒤처진다.
  `predict(dt, ego=(dx, dy, dpsi))` 인터페이스는 열려 있고, IMU 요레이트를 넣으면
  `Q_RATE_M_PER_S` 를 10배쯤 줄일 수 있다.
- **차로 폭이 지도값(3.5m)보다 좁게 나온다** (측정 중앙 3.21m, 표준편차 0.28).
  차체 자세(pitch) 때문일 가능성이 크다. 7~8단계(차로 폭 자기보정)가 이걸 영상만
  으로 역산하는 단계인데 아직 없다.
- **횡단보도를 구분하지 않는다.** 정지선과 기하가 같아서(진행방향에 수직인 흰 띠)
  정지선으로 잡힐 수 있다. 단서는 개수이고(`stopline.n_blobs` 가 2 이상),
  지금은 가장 가까운 덩어리만 쓰고 개수만 남긴다.
- **`lane_id` 에 hysteresis 가 없다.** 매 프레임 새로 매기므로 교차로처럼 곡선
  개수가 출렁이는 곳에서 순번이 흔들릴 수 있다. `track_id` 는 안정적이다.
- **가림 뒤 차선을 복원하지 않는다.** 일부러 그렇게 뒀다 — 보지 못한 구간을
  지어내는 일이고, 그 값을 제어가 실제 관측과 같은 신뢰도로 받으면 안 된다.
