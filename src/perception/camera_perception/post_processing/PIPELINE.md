# 차선 인식 후처리

`camera_perception/post_processing/` 안에 있다.

```
lane_detection.py   검출 + 후처리 + 제어용 파생값   (그림/저장 없음)
lane_viz.py         그림 그리는 함수 전부
morai_camera.py     시뮬레이터 카메라 UDP 수신

offline_test.py     [1] 녹화 -> png / mp4 / jsonl      detection + viz
live_overlay.py     [2] 실시간 -> 화면 오버레이         detection + viz
live_output.py      [3] 실시간 -> 값만                  detection 만
```

**세 스크립트는 각각 독립 실행되지만 검출·후처리는 `LaneDetector.run()` 하나만
쓴다.** 복사본을 두면 하나만 고치고 나머지를 잊게 되고, 그러면 "집에서 본
결과"와 "실주행 결과"가 갈린다.

**`live_output.py` 는 `lane_viz` 를 import 하지 않는다.** 실주행에서 제어로 값만
보내는 경로에 시각화가 들어올 이유가 없다. 실시간 루프에 그리기/저장이 끼면
프레임을 놓치고, 그러면 화면에서 본 지연이 실제 지연과 달라진다.

모델 정의(`seg_model.py`)와 전처리 규격(`seg_dataset.py`)은 학습 쪽
(`../training/`) 것을 그대로 import 한다. 여기로 복사해 두면 학습에서 바꾼
입력 크기나 정규화가 추론에 반영되지 않아 조용히 어긋난다.

경로는 전부 자동으로 찾는다 - `best.pt` 는 이 폴더와 `../` 를, `cam_set.json`
과 `ROI/lib` 은 위로 올라가며 찾는다. 폴더를 옮겨도 따라온다.

---

## 파이프라인

```
원본 1280x720
  -> crop_top(260)   하늘 제거              1280x460
  -> 640x256, ImageNet 정규화               (학습과 동일해야 함)
  -> 세그멘테이션 5클래스
  -> 보닛 마스크로 노이즈 제거
  -> BEV 변환                               자차 좌표계 미터
  -- 여기부터 클래스별로 따로 --
  -> 하단 히스토그램에서 봉우리 탐색         차선 시작 위치
  -> 봉우리마다 독립 슬라이딩 윈도우         차선별 포인트 수집
  -> RANSAC 곡선 적합                        이상치 배제
  -> 곡선 기준 픽셀 재할당 후 재적합         창이 놓친 픽셀 회수
  -> 병합 / 중복 검사
  -> 이전 프레임과 매칭해 track 유지         ID 안정화
```

**클래스별로 따로 도는 이유**: 황색 중앙선과 백색 실선은 붙어 있어도 다른
차선이다. 섞으면 멀리서 두 선이 만나는 지점에서 하나로 합쳐진다.

**유도선(교차로 안내선)도 같은 단계를 탄다.** 다만 결과는 `lanes` 가 아니라
`guides` 로 따로 나오고 `lane_id` 를 받지 않는다 - 아래 "유도선" 절을 본다.

**보닛 제거가 내장돼 있다** (`BONNET_POLY`). 학습 라벨에서 보닛은 255(ignore)라
모델이 거기에 뭘 출력해야 하는지 배운 적이 없고 실제로 노이즈를 뱉는다. 12랩
마스크가 서로 0.11~0.19% 밖에 안 달라서 윤곽을 코드에 박아 두었다 -
`learning/` 은 3.8GB 라 실주행 머신에 없을 수 있다.

---

## [1] offline_test.py

```bash
cd C:/MSC/AutoMobility/car/src/perception/camera_perception/post_processing
C:/Users/user/anaconda3/envs/vision_env/python.exe offline_test.py --recording ../recordings/lap4_full --until 000471 --bev --video
```

| 옵션 | 뜻 |
|---|---|
| `--until 000471` | 이 번호까지만. **lap4_full 은 000472 부터 도로가 아니다** |
| `--bev` | 조감도를 오른쪽에 붙인다 |
| `--video` | mp4 로도 저장 |
| `--mask-only` | 후처리 없이 **모델 출력만** (모델 탓/후처리 탓 구분용) |
| `--no-track` | 프레임 간 추적을 끈다 |
| `--count / --stride` | 일부만 빠르게 |
| `--maneuver left` | 교차로에서 따라갈 유도선 방향 (left/straight/right) |
| `--guide-side` | 유도선을 자차 어느 쪽에 두고 달릴지 (기본 left) |

출력은 `<recording>/detect_check/` 에 저장된다 (`detect_*.png`, `result.mp4`,
`output.jsonl`).

**`lap1_full` 은 쓰지 않는다.** JPEG 이 잘려 저장된 프레임이 32/358장(8.9%)
섞여 있어 화면 아래쪽이 평평한 회색으로 채워진다. 입력 자체가 없는 것이라
후처리로 고칠 수 없다. lap4_full 은 0장이다.

## [2] live_overlay.py

```bash
C:/Users/user/anaconda3/envs/vision_env/python.exe live_overlay.py --bev
```

저장 기능은 일부러 없다. 저장이 필요하면 [1] 을 쓴다.

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 |
| `m` | 마스크 표시 토글 |
| `l` | 차선 표시 토글 |
| `b` | 조감도 창 토글 |
| `p` | 일시정지 |

CPU 에서는 `--every 2` 로 화면을 부드럽게 할 수 있다.

## [3] live_output.py

```bash
C:/Users/user/anaconda3/envs/vision_env/python.exe live_output.py --udp 127.0.0.1:7600
```

오버레이도 저장도 하지 않는다. `--quiet` 로 표준출력을 끄고 UDP 만, `--points`
로 경계 점열까지 보낼 수 있다.

---

## 출력 규격 (제어팀과 확정 전)

```json
{
  "left":  [[7.0, 1.72], [7.5, 1.74], ...],
  "right": [[7.0, -1.60], ...],
  "left_type": "yellow", "right_type": "white_solid",
  "left_dashed": false, "right_dashed": false,
  "lateral_error": -0.04,
  "heading_error": -0.010,
  "stopline_dist": 17.4,
  "n_lanes": 3,

  "guide": [[12.0, 0.51], ...],
  "guide_keep_side": "left", "guide_offset": 1.65,
  "n_guides": 2,
  "center_source": "guide"
}
```

`center_source` 는 `lateral_error` / `heading_error` 가 **무엇에서 나온 값인지**를
말한다 - `both` / `left` / `right` / `guide` / `null`. `guide` 면 도색이 아니라
유도선에서 나온 값이니 제어가 신뢰도를 낮춰 잡을 수 있다.

좌표계는 **자차 기준 x 전방 / y 좌측 / 미터**다 (`GenerateLabels` 와 동일).

**경계 두 줄을 그대로 주는 이유는 장애물 회피 때문이다.** 중심선 하나만 주면
"어디까지 비켜도 되는지"를 표현할 수 없다. 평상시에는 두 경계의 중점이 곧
주행 경로이고, 회피가 필요하면 계획기가 이 통로 안에서 경로를 옆으로 민다.
`lateral_error` / `heading_error` 는 그 중심선에서 파생한 값이다.

정해지면 `lane_detection.LaneResult.as_dict()` 한 곳만 줄이면 세 스크립트에
다 반영된다.

---

## lap4_full 도로 구간 131장 성적

```
자차 좌   119/131 (90.8%)
자차 우   125/131 (95.4%)
둘 다     115/131 (87.8%)
정지선     13/131 ( 9.9%)
프레임당 평균 차선 2.97개

횡오차    117/131장  중앙값 -0.04m
방위오차  116/131장  중앙값 -0.010rad
```

CPU 에서 약 560ms/프레임. 실주행은 RTX 4090 이라 문제되지 않는다.

---

## 유도선 (교차로)

교차로 안에는 차로 도색이 없어서 `ego_left` / `ego_right` 가 둘 다 비는데,
**유도선은 그 구간에 남아 있는 유일한 도색 단서다.** 좌회전이나 직진에서
유도선을 한쪽에 끼고 일정 거리 떨어져 지나가라고 넣었다.

```
lanes   백색실선 / 백색점선 / 황색      lane_id +-1, +-2 ...  차로 경계
guides  유도선                          lane_id 없음          진로 안내선
```

**유도선을 `lanes` 에 섞지 않는 이유**는 그것이 차로 경계가 아니기 때문이다.
`+-1` 슬롯에 끼우면 차로 폭 판단과 `EGO_MAX_Y_M` 검사가 통째로 망가지고,
출력의 `left` / `right` 는 "여기까지 비켜도 된다"는 뜻이라 회피 계획이 유도선을
벽으로 오해한다.

**자차 좌우 도색이 하나도 없을 때만** 유도선으로 내려간다 (`center_at`).
도색이 한쪽이라도 보이면 차로 폭으로 반대쪽을 추정하는 기존 경로가 이긴다.

**어느 유도선을 따라갈지**는 `select_guide` 가 고른다. 교차로에는 좌회전
유도선과 직진 유도선이 같이 보이므로

1. `ORDER_X_M`(7m) 에서 `GUIDE_MAX_Y_M`(5m) 밖이면 다른 진행방향으로 보고 버린다
2. `maneuver` 를 주면 곡선이 휘어간 방향(시작~끝 y 차이)으로 고른다
3. 방향이 맞는 것이 없으면 **`None`** 을 준다 - 엉뚱한 유도선을 따라가느니 낫다

`maneuver` 는 계획 쪽이 안다. `LaneDetector(maneuver="left")` 또는 매 프레임
`det.maneuver = "left"` 로 갈아 끼운다.

| 값 | 뜻 |
|---|---|
| `GUIDE_KEEP_SIDE` `"left"` | 유도선을 자차 **왼쪽**에 두고 달린다 |
| `GUIDE_OFFSET_M` 1.65m | 유도선에서 이만큼 떨어져 달린다 (차로 폭의 절반) |
| `GUIDE_FIT_MAX_SPAN_M` 12m | 좌회전은 곡률이 커서 차선(22m)보다 짧게 끊는다 |
| `GUIDE_EXTRAP_M` 5m | 유도선은 정지선 너머에서 시작한다. 차선(3m)보다 넉넉히 |
| `GUIDE_STRAIGHT_MAX_BEND_M` 1.5m | 이 안이면 직진 유도선으로 본다 |

**이 값들은 아직 실측으로 확인한 값이 아니다.** 차선 쪽 실측값에서 성격
차이만큼 밀어 둔 출발점이다. 특히 `GUIDE_KEEP_SIDE` 와 `GUIDE_OFFSET_M` 은
도색 규칙과 실제 주행선의 관계라서 **교차로 프레임을 실제로 보고 정해야 한다.**
`offline_test.py --maneuver left --bev --video` 로 조감도에서 유도선(주황)과
주행선(흰색)의 간격을 눈으로 맞추는 것이 가장 빠르다.

유도선은 모델 IoU 도 0.393 으로 여섯 클래스 중 가장 낮다 (`best.pt`, epoch 24).
후처리를 아무리 조여도 이 위로는 못 간다.

---

## 알려진 한계

**가까운 6m 는 안 보인다.** 보닛이 가려서 차선은 보통 5~10m 부터 관측된다.
그보다 가까운 지점은 2차식으로 외삽하며 한도를 넘으면 `None` 을 돌려준다.

**자차 좌우가 둘 다 보이는 프레임은 88%.** 한쪽만 보이면 차로 폭(3.3m)으로
반대쪽을 추정한다.

**자차 경계가 3m 밖이면 `±1` 을 비워 둔다** (`EGO_MAX_Y_M`). 제어에 6m 밖
차선을 자차 경계라고 주는 것보다 "없음"이 낫다. 이 검사를 넣기 전에는 ego 차선이
프레임 사이 4.5~4.9% 확률로 옆 차선으로 튀었고, 넣은 뒤 0% 가 됐다.

**적합 구간은 22m 로 자른다** (`FIT_MAX_SPAN_M`). 2차식 하나로 32m 급커브를
덮을 수 없다. 실측: 급커브 프레임에서 전 구간 인라이어 0.41 -> 근 20m 0.98.

**교차로 안에는 차선 도색이 없다.** 정지선에서 위치를 잡고 추측항법으로
통과하는 것이 기본이고, 유도선이 보이는 교차로에서는 아래처럼 유도선을 쓴다.

**지도에 없는 가림물 뒤에도 차선이 있다고 학습돼 있다.** 시뮬레이터가
`objects` 로 보고하지 않는 드럼통·가드레일·건물이 그렇다. 차선 픽셀의 약 1%
(색으로 판별 가능한 것만 센 하한선)가 도색이 아닌 것 위에 있다.
