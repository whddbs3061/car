# 고속도로 추월 플래너 — 통합코드 별도 실행본

대회용 통합코드는 **그대로 두고**, 통합코드 사본(`morai_ws_overtake`, 브랜치 `overtake`)에
**새 파일만 추가**해 고속도로 구간만 추월 플래너로 바꿔 돌리는 실행본이다.
**별도 컨테이너 `morai-overtake` / 별도 이미지 `morai-final:overtake`** 에서 실행한다.

## 0. 요약

- **원본 통합코드는 바꾸지 않았다.** 이 사본에서도 기존 파일은 그대로이고 새 파일만 있다.
- **대회 컨테이너는 읽기만 한다.** `docker commit`으로 스냅샷만 뜬다. 스크립트가 대회 컨테이너·이미지 이름으로는 아무것도 하지 않는다.
- **고속도로 구간 밖에서는 원본과 출력이 같다.** 원본 `highway_lane_strategy`와 경로·정지·속도·seq가 메시지 단위로 같다. 정지·입력 끊김·잘못된 경로가 섞인 240주기 입력으로 확인했다.
- **이름은 통합 규약 그대로다.** 노드 `highway_lane_strategy`, 토픽 `/highway_lane_strategy/*`, 상태 `OFF / WAIT_GAP / LANE_CHANGE / INNER_HOLD / DONE`.
- **`~trajectory`를 발행한다.** 경로·정지·속도가 한 메시지에 들어 있고, 제어기는 이것만 읽는다.
- **구간과 허가는 통합 저장소 방식이다.** `route_mission_gate`의 `/planning/highway_route_active`가 구간을, `/planning/highway_lane_change_request`가 허가를 준다.

## 1. 원본과 다른 점 (전부)

| 구분 | 원본 (대회용) | 별도 실행본 |
|---|---|---|
| launch | `final_ws_highway_bringup.launch` | `final_ws_highway_overtake_bringup.launch` (원본에서 자동 생성) |
| 고속도로 노드 | `highway_lane_strategy_node.py` | `highway_overtake_node.py` (노드 이름은 똑같이 `highway_lane_strategy`) |
| 제어기 `max_speed_kph` | `MAX_SPEED_KPH` | `max(MAX_SPEED_KPH, HIGHWAY_MAX_KPH)` |
| 구간 파일 | `route_mission_regions.json` (인계 없음) | `route_mission_regions_overtake.json` (인계 s=1590.249 두 줄만 다름) |
| 그 밖의 노드·파라미터 | | 원본과 같음 (생성기가 그대로 복사, 테스트로 확인) |

launch와 구간 파일은 손으로 만들지 않는다. `docker/final_ws/make_overtake_launch.py`가
**컨테이너 안에 있는 원본 파일에서 만든다.** 대회 PC의 원본이 이 저장소와 조금 달라도
고속도로 노드와 속도 상한 말고는 대회 설정과 같아진다.

새로 추가된 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `src/control/purepursuit_mgeo/src/purepursuit_mgeo/highway_overtake.py` | 판단 로직 (ROS 없음) |
| `src/control/purepursuit_mgeo/scripts/highway_overtake_node.py` | ROS 입출력 (`highway_lane_strategy` 대체) |
| `src/control/purepursuit_mgeo/config/highway_overtake.yaml` | 구간 좌표·플래너 파라미터 |
| `src/control/purepursuit_mgeo/config/route_mission_regions_overtake.json` | 생성됨: 원본 + 인계 지점 |
| `src/bringup/morai_bringup/launch/final_ws_highway_overtake_bringup.launch` | 생성됨: 원본 launch + 위 3가지 차이 |
| `src/control/purepursuit_mgeo/tools/highway_overtake_sim.py` | 오프라인 시뮬레이터 (통합 제어기 흉내) |
| `src/control/purepursuit_mgeo/test/test_highway_overtake*.py`, `test_overtake_launch.py` | 테스트 31개 |
| `docker/final_ws/run_overtake.sh`, `overtake.env.example`, `make_overtake_launch.py` | 별도 컨테이너 실행·생성 |

### 알아둘 원본 동작: 고속도로 끝에서 정지

원본 `route_mission_regions.json`은 `handoff_enabled=false`다. 그래서 고속도로 구간 끝(s=1741.7)에
도달하면 `/route_mission_gate/ready=false`가 되고 제어기가 정지한다. 원본 문서
`HIGHWAY_LANE_CHANGE_INTEGRATION_20261005_KO.md` 56행에 적힌, 인계 지점을 측정하기 전의 의도된 동작이다.

별도 실행본은 플래너가 전역 경로로 넘겨주는 지점(s=1590.249)을 인계 지점으로 넣었다.
그래서 고속도로 뒤 구간도 계속 달린다.

## 2. 통합 규약 대응

### 토픽 (노드 이름 `highway_lane_strategy`)

| 토픽 | 형식 | 읽는 곳 |
|---|---|---|
| `~trajectory` | String JSON (plan_transport schema 1, seq 1씩 증가) | adaptive_curvature_purepursuit (유일한 제어 입력) |
| `~active_path` `~stop_required` `~target_speed_mps` | Path / Bool / Float64 | RViz·진단 |
| `~state` | String JSON | route_mission_gate (`state`=="LANE_CHANGE"), stop_cause_monitor |
| `~active` `~fast_change_active` `~lead_brake_required` `~lead_emergency_brake` | Bool | 진단 |
| `~zone_active` `~markers` | Bool / MarkerArray | 추가 진단 |

입력 토픽은 다음과 같다.

- `/avoidance_path_manager/trajectory` (기본 경로, `read_trajectory`)
- `/localization/odometry`
- `/perception/lidar/tracked_obstacles_map` (map 좌표가 아니면 무시)
- `/perception/camera/lane_info` (`frame_id`가 base_link가 아니면 무시)
- `/planning/highway_route_active`, `/planning/highway_lane_change_request`
- `/planning/highway_handoff_due` (상태 표시용)

### 상태 이름

| 통합 이름 (`state`) | 플래너 단계 (`phase`) | 뜻 |
|---|---|---|
| OFF | OFF | 구간 밖. 원본과 같은 통과 |
| WAIT_GAP | WAIT_LANE, CHASE | 카메라 차선 대기 / 간격을 판단하며 차로 유지 |
| LANE_CHANGE | CHANGE | 왼쪽으로 한 차로 이동 중 |
| INNER_HOLD | REACQUIRE, FOLLOW | 새 차로 측정 대기 / 마지막 차로에서 앞차 추종 |
| DONE | DONE | 구간 끝에서 전역 경로로 넘김 |

### 구간과 허가 (통합 저장소 방식)

1. `route_mission_gate`가 경로 거리 s=1118.7부터 인계 지점까지, 카메라 고속도로 판정과 함께 `/planning/highway_route_active`를 낸다.
2. 플래너는 다음 두 조건을 모두 만족할 때만 켜진다.
   - `/planning/highway_route_active`가 0.6 s 안에 true로 들어왔다.
   - 차량이 플래너 자체 구간(s=1160.2~1590.2) 안에 있다.
3. 차로 변경은 `/planning/highway_lane_change_request`도 true일 때만 시작한다. 구간이나 허가가 끊기면 새 변경은 시작하지 않고 차로 유지만 한다. 이미 시작한 변경은 끝낸다.
4. s=1590.2에서 전역 경로로 넘긴다(DONE). 이 지점이 인계 지점이라 route_mission_gate도 같은 곳에서 고속도로 판정을 끝낸다.

### 구간 밖 통과 (OFF·DONE)

원본 노드와 같은 식으로 출력한다.

- **경로:** 기본 경로를 그대로 낸다.
- **정지:** 기본 경로가 0.8 s 넘게 끊기거나 기본 경로가 정지를 요구하면 정지한다.
- **속도:** `MAX_SPEED_KPH/3.6`이다. 올릴 때 0.8 m/s², 내릴 때 1.8 m/s²로 바뀐다.

구간에 들어갈 때는 그때 내던 속도에서 시작한다. 나올 때는 구간 속도에서 이 규칙으로 돌아간다.

## 3. 속도 상한 (전체 주행)

| 설정 | 파일 (`$HOME/morai_ws_overtake/docker/final_ws/`) | 적용 범위 |
|---|---|---|
| `MAX_SPEED_KPH` | `highway-test.env` | 고속도로 구간 밖 (원본과 같은 뜻) |
| `HIGHWAY_CRUISE_KPH` (기본 54) | `overtake.env` | 구간 안 순항 |
| `HIGHWAY_MAX_KPH` (기본 79.2) | `overtake.env` | 구간 안 추월 최고속도 |

구간 밖에서 노드는 원본처럼 `MAX_SPEED_KPH`를 낸다. 제어기는 "곡률 속도 프로파일"과 그 값 중
작은 쪽을 따른다. 제어기 상한만 `max(MAX_SPEED_KPH, HIGHWAY_MAX_KPH)`로 올렸으므로,
구간 밖 명령 속도는 원본과 같다. 무작위 경로 40개로 수치를 확인했다.

### 제어기 한계에 맞춘 플래너 값

플래너는 제어기가 실제로 따라갈 수 있는 경로와 속도만 계획한다.
`highway-test.env`의 아래 값이 플래너보다 낮으면 플래너 값을 낮춘다. 시작 로그에 `(controller limit)`으로 표시된다.

| highway-test.env | 플래너에서 낮아지는 값 |
|---|---|
| `LATERAL_ACCEL_LIMIT_MPS2` | 차로 변경 횡가속도 = min(1.5, 0.9 × 값) |
| `MAX_ACCEL_MPS2` | 간격 판단의 가속 가정, 목표 속도 상승률 |
| `MAX_DECEL_MPS2` | 간격 판단의 감속 가정, 구간 끝 감속 = min(1.5, 0.8 × 값) |

맞추지 않으면 문제가 생긴다. 시뮬레이터 예제 설정(1.0 / 1.0 / 1.5)에서 정렬을 끄면 제어기가
차로 변경 도중 감속한다. 그러면 간격 판단이 어긋나 카메라 노이즈 seed 7에서 **충돌**했다.

### 권장 순서

1. `bash run_overtake.sh monitor`: 제어 송신 없이 상태만 본다.
   `rostopic echo /highway_lane_strategy/state`에서 구간 진입 시 `WAIT_GAP` → `LANE_CHANGE`로 바뀌는지 확인한다.
2. `bash run_overtake.sh highway 54 54` 후 `drive`: 추월 가속 없이 차로 변경만 확인한다.
3. `bash run_overtake.sh highway 54 79.2` 후 `drive`.
4. 구간 밖은 대회와 같은 `MAX_SPEED_KPH`를 쓴다. `init`이 대회 폴더 값을 복사한다. 바꾸려면 `bash run_overtake.sh speed 40`.

고속도로에는 `bash run_test.sh turn mission`(`LATERAL_ACCEL_LIMIT_MPS2=3.8`, `MAX_ACCEL_MPS2=1.4`)이 맞다.
예제 기본값(1.0 / 1.0)이면 차로 변경이 길고 늦어져, 구간 끝까지 L3에 못 가는 판이 생긴다(5장 결과).
`turn mission` 값은 구간 밖 커브 주행에도 적용된다. 원본과 같은 파일, 같은 뜻이다.
이 사본 폴더에서 실행하면 이 폴더의 `highway-test.env`만 바뀐다.

## 4. Docker로 실행하기

모든 명령은 Ubuntu에서 `Ctrl+Alt+T`로 연 일반 터미널(호스트 터미널)에 붙인다.
코드 블록 하나씩 붙이고 끝나면 다음으로 넘어간다.

**개념 세 줄**

- **이미지**는 설치된 환경과 코드를 묶은 것이고, **컨테이너**는 그 이미지를 실행하는 공간이다.
- `docker commit`은 대회 컨테이너의 현재 상태를 새 이미지로 복사한다. 대회 컨테이너는 그대로다.
- 대회 컨테이너와 별도 컨테이너는 같은 ROS 마스터와 MORAI 포트를 쓴다. **동시에 켜지 않는다.**
  스크립트가 대회 컨테이너가 켜져 있으면 `setup`과 `drive`를 거부한다.

### 4-1. 코드 옮기기 (한 번)

USB에 `morai_ws_overtake.bundle`을 복사해 Ubuntu에 꽂는다. `<USB이름>`은 파일 탐색기에 보이는 이름으로 바꾼다.

```bash
git clone -b overtake "/media/$USER/<USB이름>/morai_ws_overtake.bundle" "$HOME/morai_ws_overtake"
```

대회 코드 `$HOME/AutoVehicle`은 건드리지 않는다. 새 폴더 `$HOME/morai_ws_overtake`만 생긴다.

### 4-2. 설정 만들기 (한 번)

```bash
cd "$HOME/morai_ws_overtake/docker/final_ws" && bash run_overtake.sh init
```

`init`은 대회 폴더(`$HOME/AutoVehicle/morai_ws/docker/final_ws`)를 읽기만 한다.

- `highway.env`: IP·GPU 설정을 복사하고, 컨테이너·이미지 이름만 `morai-overtake` / `morai-final:overtake`로 바꾼다.
- `highway-test.env`: 속도·조향·센서 설정을 그대로 복사한다.
- `overtake.env`: 스냅샷할 대회 컨테이너 이름과 고속도로 속도를 적는다.

대회 폴더가 다른 곳이면 `bash run_overtake.sh init /경로/final_ws`로 실행한다.

### 4-3. 대회 컨테이너를 끄고 별도 컨테이너 만들기 (한 번)

```bash
cd "$HOME/AutoVehicle/morai_ws/docker/final_ws" && bash run_highway.sh stop
```

```bash
cd "$HOME/morai_ws_overtake/docker/final_ws" && bash run_overtake.sh setup
```

`setup`은 다음을 차례로 한다. 완료 기준은 마지막 줄의 `OVERTAKE_CHECK_OK`다.

1. 대회 컨테이너 스냅샷 → 이미지 `morai-final:overtake`
2. 컨테이너 `morai-overtake` 생성 (GPU·IP 설정은 대회와 같음, 로그 볼륨은 별도)
3. 새 파일 복사 → launch·구간 파일 생성 → 컨테이너 안에서 테스트 31개 실행 → launch 해석 확인

### 4-4. 주행

먼저 제어 송신 없이 확인한다.

```bash
cd "$HOME/morai_ws_overtake/docker/final_ws" && bash run_overtake.sh monitor
```

문제없으면 `Ctrl+C`로 끄고 주행한다.

```bash
cd "$HOME/morai_ws_overtake/docker/final_ws" && bash run_overtake.sh drive
```

`drive`는 `bash run_test.sh drive 5`(full)와 같은 인자를 쓴다. 인자는 `run_test.sh show 5`에서 그대로 가져온다.
여기에 고속도로 속도 두 개만 더해 실행한다. 확인만 하려면 `bash run_overtake.sh show`.
카메라 화면은 새 터미널에서 이 폴더의 `bash run_test.sh rviz`로 연다(별도 컨테이너에 연결).

### 4-5. 끝내고 대회 설정으로 돌아가기

```bash
cd "$HOME/morai_ws_overtake/docker/final_ws" && bash run_overtake.sh stop
```

```bash
cd "$HOME/AutoVehicle/morai_ws/docker/final_ws" && bash run_highway.sh start
```

### 4-6. 다음에 켤 때

대회 컨테이너를 끈 상태에서 아래를 실행한다. 그다음 4-4로 간다.

```bash
cd "$HOME/morai_ws_overtake/docker/final_ws" && bash run_overtake.sh start
```

### 4-7. 코드를 고친 뒤

새 bundle을 받은 경우의 예다. 그다음 `bash run_overtake.sh update`를 실행한다.

```bash
cd "$HOME/morai_ws_overtake" && git pull "/media/$USER/<USB이름>/morai_ws_overtake.bundle" overtake
```

`update`는 파일을 다시 넣고 테스트까지 돌린다.

### 4-8. 상태 확인과 삭제

- `bash run_overtake.sh status`: 대회·별도 컨테이너 상태를 본다.
- `bash run_overtake.sh shell`: 별도 컨테이너 안으로 들어간다.
- `bash run_overtake.sh remove --yes`: 별도 컨테이너와 이미지를 지운다. 대회 쪽은 지우지 않는다.

## 5. 오프라인 검증 (2026-10-28, MORAI 실주행 전)

**테스트 31개 통과**

- 플래너 18개: 간격 판단, 카메라 입력, 구간·허가 게이트, 상태 이름, 제어기 한계 정렬, 시뮬레이터 3판
- 노드 8개: 원본과의 구간 밖 동일성 240주기, trajectory 형식과 seq, 게이트 없을 때 통과, 허가 없을 때 차로 유지, base_link 장애물 무시, 플래너 예외 때 정지, 제어기 한계 정렬
- launch 5개: 생성 파일이 최신인지, 원본과 다른 곳이 의도한 3곳뿐인지, 배선 유지, 구간 파일, 제어기 상한 중립성

**시뮬레이터** (`tools/highway_overtake_sim.py`)

통합 제어기(`adaptive_curvature_purepursuit`)의 조향·곡률 속도·가감속 제한과 route_mission_gate를 흉내 낸다.

| 설정 (`LATERAL / ACCEL / DECEL`) | 시나리오 | 결과 |
|---|---|---|
| turn mission (3.8 / 1.4 / 1.5) | morai·카메라 노이즈·빈 도로 × 10 | **30/30 통과**. 3회 변경 30/30, 구간 통과 평균 26.0 s (21.6~34.4) |
| 예제 기본값 (1.0 / 1.0 / 1.5) | 같음 | 28/30. 2판은 2회만 바꾸고 구간 끝에 도달, 인계 때 가로질러 충돌 |
| turn mission | 스트레스 4종 × 5 (임의로 만든 극단 교통) | 12/20. 원래 브랜치 시뮬레이터와 같은 판이 같은 이유로 실패 (6장 1번) |

- **경로 이탈:** 전역 경로에서 최대 7.2 m다. route_mission_gate 정지 기준 11 m보다 작다. turn mission 설정의 기본 시나리오 30판에서 측정했다.
- **구간 끝 속도:** 54.4~55.3 km/h. 60 km/h 미만이다.
- **최대 횡가속:** 2.2 m/s²다. 차로 변경 시작부에서 제어기가 경로를 따라가며 생기는 과도이고, 경로 설계값은 1.5 이하다.

다시 돌리는 방법은 다음과 같다. `src/control/purepursuit_mgeo` 폴더에서 실행한다.

```bash
python tools/highway_overtake_sim.py --scenario all                       # turn mission 설정
python tools/highway_overtake_sim.py --scenario all --controller example  # 예제 기본값
python tools/highway_overtake_sim.py --scenario morai --seed 3 --view     # 한 판 재생
PYTHONPATH=src python -m unittest discover -s test -p "test_*overtake*.py"
```

## 6. 남은 위험과 결정할 것

1. **구간 끝까지 L3에 못 가면 인계 때 전역 경로로 차로를 가로지른다.** 전역 경로는 s=1590에서 L3에 있다.
   - 이 이동은 간격 판단 없이 제어기가 한다. 횡가속 6~11 m/s²가 나왔고 일부 충돌했다.
   - 스트레스 시나리오와 예제 기본값 일부 판에서 나왔다.
   - 기존 설계("구간 끝에서 전역 경로로 인계")에서 생기는 현상이고, 원래 브랜치 시뮬레이터에서도 같은 판이 같은 이유로 실패했다.
   - 대안은 정하지 않았다. 예: 구간 끝 뒤에도 같은 차로를 유지하다 간격을 보고 합류.
2. **MORAI 실주행은 아직 하지 않았다.** 시뮬레이터는 제어기 식을 그대로 쓰지만 차량 동역학과 센서는 흉내다.
3. 이번에 고친 것은 다음 두 가지이고, 설계값(5°, 1.5 m/s², 간격 1 m/3 m)은 그대로다.
   - **제어기 한계 정렬:** 위 3장.
   - **차로 변경 직후 경로 연속성:** 변경을 70%에서 "완료"로 세는 규칙은 그대로다. 대신 남은 S자 경로를 끝까지 따르고, 새 차로 중앙으로는 현재 위치·방향에서 이어 붙인다.
     - 이전에는 끝부분이 잘려 통합 제어기에서 순간 횡가속 4.4 m/s²가 났다(카메라 노이즈 seed 6).
     - 고친 뒤 같은 판은 1.48 m/s²다.
