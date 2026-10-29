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
