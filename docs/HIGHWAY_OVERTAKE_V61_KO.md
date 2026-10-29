# v61 통합본 + overtake 고속도로 (별도 실행본)

팀원의 v61 장애물 통합본(`integrated_obstacle_v61.launch`)에서 **고속도로만 overtake 플래너로 바꾼** 실행본입니다.

- 기준: 팀원 컨테이너의 `/root/morai_integration_test/AutoVehicle/morai_ws`를 2026-10-29 16:55에 찍은 스냅샷입니다. 동적 장애물 5초 대기, 터널, EKF 패치가 이미 적용된 상태입니다.
- 팀원 파일은 한 줄도 바꾸지 않았습니다. 새 파일만 추가합니다.
- 팀원 컨테이너는 복사만 하고, 별도 컨테이너 `morai-overtake-v61`에서 돌립니다.

## 1. 무엇이 다른가

`integrated_obstacle_v61_overtake.launch`는 팀원 launch를 자동으로 복사해서, 아래 세 노드만 바꾼 파일입니다. 노드 이름은 그대로입니다.

| 노드 | 팀원 launch | 새 launch |
|---|---|---|
| `highway_lane_strategy` | `v61_highway_with_avoidance_node.py` (팀원 고속도로) | `highway_overtake_node.py` (overtake). 같은 입출력 토픽 |
| `curvature_speed_purepursuit` | 제어기 원본 | `curvature_pp_overtake_node.py`: 원본을 그대로 불러오고, 고속도로 구간에서만 속도 상한과 횡가속 한계를 바꿈 |
| `curvature_signal_controller` | fusion 원본 | `maneuver_fusion_overtake_node.py`: 원본을 그대로 불러오고, 고속도로 구간에서만 속도 상한(`speed_governor`)을 바꿈 |

- **고속도로 구간(s=1160~1590) 안:** overtake 플래너가 경로와 속도를 정합니다.
- **구간 밖:** overtake 노드는 회피 노드의 출력을 팀원 고속도로 노드와 똑같이 통과시킵니다(240틱 비교 테스트). 제어기와 fusion도 팀원 값 그대로 돕니다.
- 구간 여부는 `/highway_lane_strategy/zone_active`로 알립니다. 이 신호가 0.5초 넘게 끊기면 제어기와 fusion은 팀원 값으로 돌아갑니다.

## 2. 처음 한 번 (대회 PC 호스트 터미널)

```bash
git clone -b highway-overtake-v61 https://github.com/whddbs3061/car.git $HOME/morai_ws_v61
cd $HOME/morai_ws_v61/docker/final_ws
bash run_overtake_v61.sh init          # 팀원 컨테이너 이름을 찾아 설정 파일 생성
```
`overtake_v61.env`의 `ROS_IP`(기본 192.168.10.21, 팀원 README와 같은 값)를 확인합니다.

```bash
sudo docker stop morai-highway-gpu     # 팀원 컨테이너 끄기 (이름은 init 출력 참고)
bash run_overtake_v61.sh setup         # 스냅샷 → 별도 컨테이너 → 설치 → 점검
```
마지막에 `TEAM_FILES_OK 513 ...`과 `OVERTAKE_V61_CHECK_OK`가 나오면 성공입니다. `TEAM_FILES_OK`는 복사본 안의 팀원 파일 513개가 이 브랜치의 스냅샷과 똑같다는 뜻입니다. 다르면 아무것도 실행하지 않고 멈춥니다. 그때는 팀원이 그 뒤에 코드를 바꾼 것이니 새 스냅샷이 필요합니다.

## 3. 주행

```bash
bash run_overtake_v61.sh monitor       # 제어 신호 없이 확인
bash run_overtake_v61.sh drive         # 실제 주행 (overtake 고속도로)
bash run_overtake_v61.sh team-drive    # 같은 컨테이너에서 팀원 launch 그대로 (비교용)
```
속도 설정은 이 폴더의 `overtake_v61.env`에만 저장됩니다.
```bash
bash run_overtake_v61.sh speed 43.2          # 구간 밖 max_speed_kph (팀원 기본 43.2)
bash run_overtake_v61.sh highway 54 79.2     # 고속도로 순항 / 추월 상한
```

## 4. 되돌리기

| 방법 | 명령 |
|---|---|
| 팀원 코드로 주행 | `bash run_overtake_v61.sh team-drive` |
| 팀원 컨테이너로 복귀 | `bash run_overtake_v61.sh stop` 후 `sudo docker start morai-highway-gpu` |
| 별도 실행본 완전 삭제 | `bash run_overtake_v61.sh remove --yes` 후 `$HOME/morai_ws_v61` 폴더 삭제 |

팀원 컨테이너는 처음부터 바뀌지 않습니다. 두 컨테이너는 같은 ROS 마스터와 MORAI 포트를 쓰므로 동시에 켜지 않습니다(`drive`가 거부합니다).

## 5. 확인한 것 (오프라인)

- overtake 테스트 41개 통과(1개 건너뜀: 팀원 `route_mission_regions.json`의 경로 SHA가 이 작업공간의 경로 파일과 맞지 않음. v61 launch는 이 파일을 쓰지 않음)
- 구간 밖 통과 동작이 팀원 v61 고속도로 노드와 240틱 동안 경로·정지·속도 모두 같음
- 새 launch는 세 노드와 새 인자 4개 말고는 팀원 launch와 같음. 두 상속 노드는 팀원 설정을 하나도 빠뜨리지 않고 그대로 가짐
- 상속 노드는 구간 신호가 신선하고 참일 때만 값을 바꾸고, 끝나면 원래 값으로 되돌림

MORAI 실주행 검증은 아직 하지 않았습니다.

## 6. 고속도로 노드 두 가지 (2026-10-30 추가)

이전 노드는 그대로 두고, 앞차 회피 판을 별도 노드로 추가했습니다.

| 실행 | launch | 고속도로 노드 | 플래너 |
|---|---|---|---|
| `drive` / `monitor` | `integrated_obstacle_v61_overtake.launch` | `highway_overtake_node.py` (이전 그대로) | `highway_overtake.py` (이전 그대로) |
| `drive-escape` / `monitor-escape` | `integrated_obstacle_v61_overtake_escape.launch` | `highway_overtake_escape_node.py` | `highway_overtake_escape.py` |

- **이전 파일:** `highway_overtake.py`, `highway_overtake_node.py`, `highway_overtake.yaml`, `integrated_obstacle_v61_overtake.launch`는 커밋 `347d0c9`와 바이트 단위로 같습니다.
- **escape 플래너:** `highway_overtake.py`의 복사본에 앞차 회피만 더한 파일입니다. 두 파일을 `diff`하면 바뀐 부분(약 40줄)만 보입니다.
  - 내 차로의 느린 앞차는 옆으로 빠지는 순간까지 범퍼 간격 3 m만 유지되면 변경을 시작합니다.
  - 변경하는 동안 그 앞차 때문에 감속하지 않습니다(1.5 m 이내면 비상정지).
- **escape 노드:** 이전 노드 파일을 그대로 불러오고 플래너만 escape 판으로 바꿔 끼웁니다. 토픽, 설정 파일, 상태 메시지는 같습니다.
- **병합할 때:** launch의 고속도로 노드 `type`을 `highway_overtake_node.py` 또는 `highway_overtake_escape_node.py`로 고르면 됩니다.
- **오프라인 결과(대회 트래픽 30판, `mission` 값):** 구간 통과 28.1 s → 24.7 s, 구간 최저 속도 42.7 → 53.2 km/h, 실패 0판.

## 7. 대회 네트워크 포트 (2026-10-30)

새 launch 두 개(`integrated_obstacle_v61_overtake*.launch`)는 차량 상태를 **Competition Vehicle Status**로 받습니다.

| 통신 | 포트 |
|---|---|
| Competition Vehicle Status (Sim → User) | 시뮬레이터 9088 → 우리 **9099** (`ego_status_port` 기본값) |
| Ego Ctrl Cmd (User → Sim) | 우리 9094 → 시뮬레이터 9093 (그대로) |
| Collision Data (Sim → User) | 9091 → 9092 (`mission_evaluator`, 그대로) |
| GPS·IMU·카메라·LiDAR | 팀원 launch 그대로 |

- **Ego Vehicle Status(1911)는 쓰지 않습니다(실격 사유).** 새 launch에는 1911이 없습니다.
- **기존 브리지로 읽습니다.** Competition Vehicle Status 패킷(`#MoraiInfo$`, 181바이트 또는 229바이트)은 앞 181바이트 배열이 기존 `morai_udp_drive_bridge` 파서와 같습니다(`car/src/path_planning/morai_udp_competition_status.py`로 확인). 그래서 브리지 코드는 그대로 두고 받는 포트만 바꿨습니다.
- **팀원 launch는 아직 1911:** `team-drive`로 띄우는 팀원 원본 `integrated_obstacle_v61.launch`는 여전히 1911입니다. 대회에서 팀원 launch를 쓴다면 팀원도 바꿔야 합니다.

## 8. escape2 (2026-10-30): 진입 겹선 + 지도로 차로 번호 확인

`drive-escape2` / `monitor-escape2` → `integrated_obstacle_v61_overtake_escape2.launch` → `highway_overtake_escape2_node.py` → `highway_overtake_escape2.py`. 기존 노드와 앞차 회피 노드는 그대로입니다.

앞차 회피 판(`highway_overtake_escape.py`)의 복사본에 두 가지를 더했습니다.

- **진입 차로 경계(겹선):** 진입 차로의 왼쪽 선은 합류 경계(MGeo `solid broken`)이고, 진입 쪽이 점선이라 넘을 수 있습니다.
  - 그래서 첫 변경은 카메라가 그 선을 실선으로 읽어도 시작합니다.
  - 진입 차로에서는 왼쪽 실선을 보고 "최종 차로"로 판단하지 않습니다.
- **지도로 차로 번호 확인:** `config/highway_lanes.json`(MGeo 링크로 만든 진입 차로·L1~L4 중심선, `tools/make_highway_lanes.py`)에서 차에 가장 가까운 차로를 찾습니다.
  - 가장 가까운 차로와 1.2 m 안에 있고, 다음 차로와 1.5 m 넘게 차이 나면 확실한 것으로 봅니다.
  - 확실한 결과가 플래너가 세는 차로 번호와 1초 넘게 다르면 지도 쪽으로 고칩니다.
  - 차로 변경 중이거나 차로가 합쳐지는 곳(진입 테이퍼, 구간 끝 깔때기)에서는 고치지 않습니다.
  - 최종 차로로 판단했는데 지도상 아래 차로면 차로 변경을 다시 시작합니다.
  - 상태 토픽의 `map_lane`(index, offset, margin, clear)과 이벤트 `lane count corrected by the map`로 확인할 수 있습니다.

오프라인 결과(`--planner escape2`):

| 트래픽 | 결과 |
|---|---|
| 대회 트래픽 30판 × 제어기 두 가지 | 앞차 회피 판과 같음(문제 0) |
| 진입 겹선을 실선으로 읽는 카메라(`entry_solid`) | 앞차 회피 판 20판 모두 실패 → escape2 20판 모두 통과 |

일부러 차로 번호를 하나 틀리게 만든 시험에서는 13~20 m 안에 바로잡았습니다.

## 9. escape3 (2026-10-30): 카메라로 넘은 선 세기 + 더 비스듬한 변경

`drive-escape3` / `monitor-escape3` → `integrated_obstacle_v61_overtake_escape3.launch` → `highway_overtake_escape3_node.py` → `highway_overtake_escape3.py`. stable, escape, escape2는 그대로입니다.

escape2(`highway_overtake_escape2.py`)의 복사본을 다음처럼 바꿨습니다.

- **넘은 선 세기:** 카메라(`real_lane`)는 차가 밟은 선을 `0`(`straddling_lane`), 내 차로의 왼쪽 선을 `+1`, 오른쪽 선을 `-1`로 내보냅니다.
  - 왼쪽 선(`+1`)이 `0`이 되면 왼쪽으로 한 번 넘은 것으로 셉니다. 그래서 첫 번째는 선 종류와 상관없이 합류 분리선입니다.
  - 오른쪽 선(`-1`)이 `0`이 되는 것(진입 차로 오른쪽 선을 밟음)은 세지 않습니다.
  - 밟았다가 원래 쪽으로 돌아오면 취소합니다.
  - 겹선의 두 줄은 한 번으로 셉니다(15 m 안의 두 번째 선).
  - 구간 시작 때 이미 밟고 있던 선은 세지 않습니다.
- **세 번 넘으면 더 넘지 않음:** 넘은 선(`crossings`)과 끝낸 변경 수 중 큰 값이 3이면 차로 변경을 더 시작하지 않습니다. 마지막 차로의 왼쪽 선을 밟으면 그 선 오른쪽 차로로 되돌아옵니다(최대 2.5 m/s²).
- **넘은 선이 오른쪽에 보이면 카메라 차로로 바로 전환:** 변경 중에는 경로를 처음 프레임에 고정하는데, 그 방향이 조금만 틀려도 경로가 계속 옆으로 벌어집니다. 넘은 선이 오른쪽 선(`-1`)으로 보이는 순간 카메라가 보는 새 차로로 바꾸고 그 중심으로 붙습니다(`crossing_reacquire`).
- **더 비스듬한 변경:** `config/highway_overtake_escape3.yaml`이 `highway_overtake.yaml` 위에 `max_entry_angle_deg 3.0`, `max_lateral_accel_mps2 1.0`을 덮어씁니다. 한 차로 변경 길이는 54/72/79 km/h에서 85/98/105 m입니다(escape2는 60/75/82 m).
- **지도 확인은 뺐습니다:** 차로 수는 카메라가 넘은 선과 끝낸 변경으로만 셉니다.

오프라인 결과(`--planner escape3`, `--controller mission`, 시나리오마다 5판):

| 트래픽 | escape2 | escape3 |
|---|---|---|
| 대회 트래픽, 카메라 노이즈, 진입 겹선 실선 읽기, `real_lane`식 카메라 | 모두 통과 | 모두 통과 |
| 카메라 각도 +1.0° 오차 (3판) | 4.1 m 넘어감, 2판 L4 진입 | 0.7 m, 모두 통과 |
| 카메라 각도 +1.5° 오차 | 4.5 m 넘어감, L4 진입·충돌 | L4 진입 없음, 다만 L3에서 1.2 m 밀려 실선 접촉 |
| 차체 최대 각도 / 최대 횡가속도 | 5.0° / 1.9 m/s² | 3.3° / 1.5 m/s² |

`real_camera`는 `real_lane`처럼 선을 판정하는(7 m 앞 ±0.5 m, `track_id`) 시뮬레이터 카메라이고, 진입 겹선을 두 선으로 봅니다. `camera_yaw_bias`는 거기에 카메라 각도 오차를 더한 것입니다. 기존 시나리오 결과는 바뀌지 않았습니다.

모든 차로가 우리보다 빠른 가상 시나리오(`fast_rear`, 대회에는 없을 것으로 봄)에서는 변경이 길어진 만큼 빈틈을 덜 찾아 escape2보다 나쁩니다(5판 중 통과 4 → 2).

## 10. escape4 (2026-10-30): 빠를수록 더 비스듬히

`drive-escape4` / `monitor-escape4` → `integrated_obstacle_v61_overtake_escape4.launch` → `highway_overtake_escape4_node.py` → `highway_overtake_escape4.py`. escape3까지는 그대로입니다.

escape3의 복사본에 **옆속도 상한**을 더했습니다. 옆속도는 차로를 바꿀 때 차가 왼쪽으로 1초에 움직이는 거리이며, 속도 × tan(각도)입니다. escape3는 각도가 3°로 고정이라 빠를수록 옆속도가 커졌습니다(54 km/h 0.79 m/s → 79 km/h 1.15 m/s). escape4는 옆속도를 `max_lateral_speed_mps` 0.8 m/s 이하로 묶습니다. 그래서 한 차로를 넘는 데 언제나 4.4초 이상 걸리고, 빠를수록 각도가 작아집니다.

| 속도 | escape3 | escape4 |
|---|---|---|
| 54 km/h | 3.0°, 85 m | 3.0°, 85 m |
| 63 km/h | 3.0°, 91 m | 2.6°, 98 m |
| 72 km/h | 3.0°, 98 m | 2.3°, 112 m |
| 79 km/h | 3.0°, 105 m | 2.1°, 123 m |

(변경 각도, 한 차로 변경 길이.) 값은 `config/highway_overtake_escape4.yaml`에 있습니다.

**변경 경로는 지금 속도 기준:** escape3까지는 변경 중 가속할 수 있게 변경 경로를 항상 최고 속도(79 km/h) 기준으로 만들었습니다. 옆속도를 묶은 채로 그렇게 하면 54 km/h로 달릴 때도 변경이 123 m로 길어집니다. 그래서 escape4는 지금 속도 + `plan_speed_margin_mps` 2 m/s(약 7 km/h) 기준으로 만듭니다. 변경 중에도 그 속도까지만 가속하므로 실제 옆속도도 0.8 m/s 근처로 유지됩니다.

**추월이 오래 걸리면 뒤로 붙기:** 변경이 길어지면 옆 차로 차 옆에 머무는 시간도 길어집니다. 그래서 escape4는 두 가지를 더 합니다.

- 변경을 막는 왼쪽 차로 차 앞으로 들어가는 데(필요한 간격 ÷ 상대속도) `pass_time_max_s` 3초보다 오래 걸리면 앞지르려 하지 않습니다. 그 차보다 느리게 감속해 먼저 보내고 그 뒤로 붙습니다. 그 차가 우리보다 빠르면 시간은 무한대라 언제나 먼저 보냅니다.
- 변경 중, 옆 차로 차의 중심이 우리 차 중심보다 앞에 있으면 아직 옆에 나란히 있어도 앞차로 보고 따라갑니다. escape3까지는 그 차가 완전히 앞으로 나간 뒤에만 앞차로 봤습니다. 그래서 그 사이에 가속해 속도를 맞춰 버리고 옆에 붙은 채로 넘어갈 수 있었습니다.

오프라인 결과(`--controller mission`, 시나리오마다 5판):

| 트래픽 | escape3 | escape4 |
|---|---|---|
| 대회 트래픽, `real_camera`, 진입 겹선 실선 읽기, 카메라 노이즈 | 모두 통과, 비상정지 0 | 모두 통과, 비상정지 0 |
| 최대 옆속도 | 1.21~1.25 m/s | 0.89~0.95 m/s |
| 고속도로 구간 시간 | 24.4~24.5 s | 25.5~26.4 s |

경로를 최고 속도 기준으로 만든 첫 escape4에서는 변경이 길어져, 떠나는 차로의 느린 앞차나 옆 차로 차에 옆으로 붙는 비상정지가 샘플 교통 20판 중 4번 있었습니다. 지금 속도 기준으로 바꾼 뒤에는 0번입니다. `slow_leads`(가상, 차가 브레이크를 안 밟음)에서 남은 충돌은 뒤차 추돌뿐입니다.

## 11. 고속도로 끝 28 km/h (escape3, escape4)

고속도로는 A2256W000411 시작부터 A2256W000153 끝(70.21, -365.75; 경로 s 1741.7, 팀원 `highway_route_end_s_m`)까지이고, 그 뒤 톨게이트까지는 30 km/h 제한입니다. escape3와 escape4는 우리 구간이 끝난 뒤(s 1590.2)에도 노드가 내보내는 속도를 낮춥니다. 1.0 m/s²로 감속해서 고속도로 끝 10 m 앞부터 28 km/h가 되게 합니다. 고속도로 끝을 지나면 다시 팀원 로직 속도(`max_speed_kph`)로 돌아갑니다.

값은 각 yaml의 `highway_end_s`, `highway_end_speed_mps`, `highway_end_decel_mps2`, `highway_end_hold_m`입니다. 공용 노드(`highway_overtake_node.py`)는 그대로 두고, escape3/escape4 노드 래퍼에서만 구간 밖 속도에 이 상한을 씌웁니다. 시뮬레이터에서는 고속도로 끝 속도가 28.4 km/h였습니다. stable, escape, escape2는 바뀌지 않았습니다.
