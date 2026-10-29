# 정상 주행 통합 소스용 회피 구간 조건 패치

대상: /root/morai_integration_test/AutoVehicle/morai_ws
실행 런치: morai_bringup integrated_obstacle_v61.launch
기준: 사용자가 첨부한 morai_integrated_actual_source.tar.gz

이 패치는 현재 정상 주행하는 통합 워크스페이스에만 적용합니다.
이전 morai_docker_v2_overlay.zip 및 v3가 아닙니다.

## 변경 범위

실행 관련 파일은 4개입니다.

| 파일 | 변경 |
| --- | --- |
| src/bringup/morai_bringup/launch/integrated_obstacle_v61.launch | 구간 설정 파일 인자 1개와 설정 로드 1개만 추가 |
| src/control/purepursuit_mgeo/scripts/v61_obstacle_avoidance_node.py | 구간별 새 회피 진입 조건과 상태 출력 추가 |
| src/control/purepursuit_mgeo/src/purepursuit_mgeo/route_avoidance_gate.py | 구간/진행거리 판정 모듈 추가 |
| src/control/purepursuit_mgeo/config/avoidance_regions.yaml | 구간 좌표/경로 fingerprint 설정 추가 |

CHANGES.patch에서 수정된 두 기존 파일의 diff를 확인할 수 있습니다.
CMake, 부모 회피 알고리즘, 최종 제어, 카메라, 신호등, 고속도로, 회전교차로 코드는 교체하지 않습니다.
기존 launch 인자·포트·목표속도·제어 파라미터도 변경하지 않습니다.
기존 입력 정지 및 미션 소유권 우선순위와 회피 계획 실패 시 동작은 그대로 둡니다.
기존 입력 정지/다른 미션 활성화로 회피가 중단되는 경우의 처리를 새로 바꾼 패치는 아닙니다.

## 추가한 동작

- 새 회피 시작 허용: 지정 장애물 구간 안 **또는 GPS 음영 상태**.
- NORMAL 상태로 구간 밖/GPS 정상일 때는 회피를 새로 시작하지 않고 기존 기본 경로를 전달합니다.
- 이미 AVOIDANCE/RETURN이면 구간 끝을 지나거나 GPS가 복구되어도 원래 FSM의 복귀 완료까지 이어갑니다.
- GPS 음영 예외에도 기존 EKF 위치/입력 유효성 검사와 고속도로·회전교차로 소유권 판정을 적용합니다.
- 구간 밖이라는 이유만으로 새 정지 명령을 만들지는 않습니다. 기존 상위 정지 요구는 유지합니다.
- GPS 음영 판정은 기존 GPS freshness 로직을 사용합니다. GPS 통신이 끊기거나 수신하지 못하는 경우도 음영으로 취급됩니다.

### 첨부한 실제 전역경로에서 검증한 구간

0 기반 인덱스이며 시작 포함, 끝 제외입니다.

| 구간 | 인덱스 | 경로 누적거리 s |
| --- | --- | --- |
| static_obstacle | [553,782) | [271.923,384.276)m |
| dynamic_obstacle | [828,998) | [407.276,488.319)m |

전역경로 4,430점 fingerprint가 설정과 같은지 시작할 때 확인합니다.
구간 이름은 위치 구분이며 새 정적/동적 장애물 분류기는 추가하지 않습니다.
좌표와 경로는 검증했지만 실제 MORAI 장애물 배치가 이 구간 안에 있는지까지 시뮬레이터에서 확인한 것은 아닙니다.
한 번의 순방향 주행 기준입니다. 장면 리셋/차량 순간이동/새 바퀴 시작 시 노드를 재시작하세요.

## 적용

현재 주행 launch를 Ctrl+C로 종료합니다.

### 우분투 PC

```bash
sudo docker cp "$HOME/Downloads/morai_integrated_region_patch.zip" morai-highway-gpu:/tmp/morai_integrated_region_patch.zip
sudo docker exec -it morai-highway-gpu bash
```

### 도커 안: 정상 통합 소스 백업

```bash
tar -czf /tmp/morai_integrated_before_region_patch.tar.gz \
  -C /root/morai_integration_test/AutoVehicle/morai_ws \
  src/bringup/morai_bringup/launch/integrated_obstacle_v61.launch \
  src/control/purepursuit_mgeo
```

백업이 성공한 경우에만 적용합니다.

```bash
python3 -m zipfile -e /tmp/morai_integrated_region_patch.zip /root/morai_integration_test/AutoVehicle/morai_ws
chmod +x /root/morai_integration_test/AutoVehicle/morai_ws/src/control/purepursuit_mgeo/scripts/v61_obstacle_avoidance_node.py
cd /root/morai_integration_test/AutoVehicle/morai_ws
source /opt/ros/noetic/setup.bash
source /root/morai_integration_test/AutoVehicle/morai_ws/devel/setup.bash
catkin_make
```

빌드 성공 후 실행합니다. 이전에 정상 주행한 실행 명령을 그대로 사용합니다.
ROS 주소는 실제 IP 192.168.10.21로 명시합니다. 이전 192.168.0.185를 사용하면 시작이 대기할 수 있습니다.

```bash
source /opt/ros/noetic/setup.bash
source /root/morai_integration_test/AutoVehicle/morai_ws/devel/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.10.21
export ROS_MASTER_URI=http://192.168.10.21:11311

roslaunch morai_bringup integrated_obstacle_v61.launch \
  workspace_path:=/root/morai_integration_test/AutoVehicle/morai_ws \
  enable_control:=true \
  max_speed_kph:=43.2
```

이 ZIP은 /opt/AutoVehicle/morai_ws에 풀지 마세요.
직접 덮어쓰는 방식이라 첨부 이후 해당 두 파일에 추가 편집했다면 그 변경은 자동 병합되지 않습니다.

## 상태 확인

새 도커 터미널에서도 같은 setup과 ROS 주소를 설정합니다.

```bash
source /opt/ros/noetic/setup.bash
source /root/morai_integration_test/AutoVehicle/morai_ws/devel/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.10.21
export ROS_MASTER_URI=http://192.168.10.21:11311
rostopic echo /integration/avoidance/status
```

추가된 항목:
- avoidance_enabled: 현재 새 회피 또는 진행 중 회피 계획이 허용되는지
- avoidance_gate.entry_allowed: 구간/GPS 및 기존 입력·미션 조건에 따른 새 회피 진입 허용
- avoidance_gate.regions: static_obstacle / dynamic_obstacle 활성 구간
- avoidance_gate.gps_blackout, gps_override: GPS 음영 예외
- avoidance_gate.continuing: 구간 밖에서도 진행 중 회피·복귀를 이어가는 상태
- avoidance_gate.progress_waypoint_index, progress_s_m: 진행 위치

active는 기존과 같이 실제 회피 경로 사용 상태입니다. avoidance_enabled=true여도 장애물이 없으면 active=false일 수 있습니다.
기존 고속도로·회전교차로·입력 정지에 의해 회피가 비활성화될 수 있습니다.

## 원복

ROS 종료 후 도커 안에서 백업을 복원합니다.

```bash
tar -xzf /tmp/morai_integrated_before_region_patch.tar.gz -C /root/morai_integration_test/AutoVehicle/morai_ws
```

이번에 추가된 두 실행 파일을 제거합니다. 적용 전 첨부본에는 두 파일이 없었습니다.

```bash
rm -f \
 /root/morai_integration_test/AutoVehicle/morai_ws/src/control/purepursuit_mgeo/config/avoidance_regions.yaml \
 /root/morai_integration_test/AutoVehicle/morai_ws/src/control/purepursuit_mgeo/src/purepursuit_mgeo/route_avoidance_gate.py
```

source 후 catkin_make하고, 성공하면 다시 source devel/setup.bash 뒤 위의 원래 launch 명령으로 실행합니다.
README/테스트/패치 정보 파일은 주행 코드에서 참조하지 않습니다.
백업에는 src만 포함되며 build/devel 스냅샷은 아닙니다.

## 검증 기록

- 새 구간/어댑터 테스트 20개 통과.
- 첨부본 기존 Frenet/종방향/경로 테스트 7개 통과.
- 실제 첨부 경로 4,430점을 순방향으로 검사해 구간 경계 확인.
- 생성자 설정 로드, GPS 음영 예외, GPS 복구/구간 이탈 후 복귀 유지 확인.
- 고속도로/회전교차로/미션 상태 미수신/입력 정지의 기존 우선순위 확인.
- launch의 두 추가 줄을 제거하면 첨부 원본과 바이트 단위로 일치함을 확인.
- Python 3.8 문법과 launch XML 검사 통과.
- ZIP 압축 해제 후 payload 체크섬 확인.
- ROS catkin 빌드·실제 MORAI 주행은 미수행. 위 테스트는 ROS 경계를 모킹한 Python 검증입니다.

원한다면 사용자 환경에서 테스트를 다시 실행할 수 있습니다.

```bash
cd /root/morai_integration_test/AutoVehicle/morai_ws
PYTHONPATH=src/control/purepursuit_mgeo/src python3 -m unittest discover -s REGION_PATCH_TESTS -v
```
