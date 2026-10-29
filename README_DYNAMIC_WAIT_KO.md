# 동적 장애물 구간: 20km/h → 정지 대기 → 경로 이탈 확인 후 재출발

대상 워크스페이스: /root/morai_integration_test/AutoVehicle/morai_ws
실행 런치: morai_bringup integrated_obstacle_v61.launch
기준: 사용자가 첨부한 실제 통합 소스 + 직전 morai_integrated_region_patch.zip

이 ZIP은 직전 회피 구간 패치의 기능을 포함하는 누적본입니다.
기존 실행 명령, 정상 구간 최대속도 43.2km/h(12m/s), 카메라·신호등·고속도로·회전교차로 연결은 유지합니다.
고속도로에서 보고된 bad_heading/lane_recovery_timeout 문제를 수정한 패치는 아닙니다.

## 동작

| 상황 | 동작 |
| --- | --- |
| 동적 구간 40m 전부터 | 진입 전 감속을 위해 20km/h 상한 적용 |
| 동적 구간 안, 전방 경로가 비어 있음 | 최대 20km/h로 기존 경로 추종 |
| 전방 경로를 막거나 2초 이내 가로지를 것으로 예측되는 장애물 | 가속 0, 제동 1로 정지 요청 |
| 정지 원인인 장애물이 경로 안에서 멈춤 | 계속 정지 |
| 정지 원인 ID가 추적에서 사라짐/다른 ID로 바뀜 | 확인되지 않은 통과로 간주하여 정지 유지 |
| 같은 ID가 경로 밖에서 0.6초 이상 연속 확인됨 | 기존 경로 주행 재개, 구간 내 20km/h 상한 유지 |
| 구간 통과, 대기 원인 없음 | 기존 정상 구간 속도 제한으로 복귀 |

동적 구간은 실제 전역경로의 0 기반 [828,998), s=[407.276,488.319)m 입니다.
20km/h는 주행 상한입니다. 기존 곡률·신호등·안전 제한이 더 낮은 속도/정지를 요구하면 그 제한을 유지합니다.
물리적으로 즉시 20km/h가 되거나 장애물 검출 순간 즉시 정차한다는 뜻은 아닙니다. 40m 접근 감속 여유는 설정으로 조정할 수 있습니다.

이 구간에서는 새로운 독립 Frenet 우회를 시작하지 않고 기본 경로에서 정지 대기합니다.
구간 진입 전부터 진행 중이던 회피/복귀는 기존 완료 로직을 유지하고, 새 정지 보호가 최종 명령에서 우선 적용됩니다.
정적 장애물 구간의 기존 회피 기능과 GPS 음영 회피 허용은 직전 구간 패치와 같습니다.

## 장애물 인지와 경로 밖 판정

기존 /integration/avoidance/obstacle_states를 사용합니다. 새 LiDAR 클러스터/추적기는 추가하지 않습니다.
사용하는 정보는 기존 tracker의 ID, map 중심 좌표, OBB 크기·방향과 지도 기준 속도입니다.

- 기준 경로와 최신 실제 추종 경로가 있으면 두 경로의 전방 구간을 함께 검사합니다.
- 전방 약 30m와 차량 앞쪽 오프셋 3.845m, 뒤쪽 1m의 경로를 검사합니다.
- 경로를 차량 폭 1.892m의 절반 + 여유 0.3m로 확장하고 장애물 박스와의 겹침을 검사합니다.
- 일정 속도 가정으로 2초 동안 이동하는 OBB의 연속 swept footprint도 검사합니다. 몇 시점만 검사해서 빠른 횡단을 놓치는 방식을 사용하지 않습니다.
- 재출발에는 경로 여유를 0.3m 더 확장합니다. 같은 ID가 이 범위와 예측 범위를 모두 벗어나야 합니다.
- 최소 3개 새 관측과 0.6초 확인이 필요합니다. 반복된 소스 타임스탬프는 확인 횟수를 늘리지 않습니다.
- 차량 속도 0.2m/s 이하, 장애물의 정지 원인 위치에서 0.2m 이상 이동을 확인합니다.
- 여러 장애물로 정지했으면 모든 대기 ID가 관측으로 해제되어야 합니다.

분류가 STOPPED나 STATIC으로 바뀌어도 경로 안에 남아 있는 장애물 때문에 정지한 상태를 자동 해제하지 않습니다.
이 구간에서는 STATIC/UNKNOWN으로 보고된 물체라도 실제 경로를 막으면 충돌 방지를 위해 정지합니다.
따라서 물체가 물리적으로 동적인지 분류하는 정확도만으로 정지 여부를 결정하지 않습니다.

입력이 0.6초 이상 오래됐거나 잘못된 경우 구간 안 또는 대기 중에는 정지를 유지합니다.
구간 밖 정상 주행에 이 장애물 입력 미수신만으로 새 정지 요구를 만들지는 않습니다.
대기 ID가 끊기면 사람이 통과했다고 추정하지 않습니다. 5초 이상 관측이 끊기거나 위치가 비현실적으로 점프한 ID는 재식별 불확실 상태를 유지할 수 있으며, 원인 확인 후 런치 재시작이 필요할 수 있습니다.
차량/장면 리셋이나 순간이동, 새 바퀴 시작 시 런치를 재시작해야 합니다.

## 제어 연결

새 /ctrl_cmd 발행자는 없습니다.
기존 curvature_signal_controller의 최종 출력에 감속/정지 제한을 적용합니다.
기존 신호등·미션·경로·안전 정지를 해제하거나 가속/조향을 강제로 덮어쓰지 않습니다.
경로가 비면 새 보호 로직의 제한만 해제하며, 기존 제어기가 경로 주행을 이어갑니다.

실행 관련 파일 6개를 포함합니다.
- integrated_obstacle_v61.launch: 동일 실행 파일, 기존 인자 유지. 최종 노드에도 동일 구간 설정 로드.
- avoidance_regions.yaml: 기존 구간 + 새 20km/h/대기 설정.
- v61_obstacle_avoidance_node.py: 동적 구간의 새 우회 시작을 대기 모드로 전환.
- maneuver_fusion_node.py: 장애물 구간 보호와 최종 명령 연결.
- route_avoidance_gate.py: 기존 구간 판정 모듈.
- dynamic_zone_guard.py: 새 속도·경로 겹침·ID 대기/해제 로직.
CMake나 ROS 메시지 정의 변경은 없습니다.

## 적용

먼저 주행 launch를 Ctrl+C로 종료하세요.

### 우분투 PC

```bash
sudo docker cp "$HOME/Downloads/morai_integrated_dynamic_wait.zip" morai-highway-gpu:/tmp/morai_integrated_dynamic_wait.zip
sudo docker exec -it morai-highway-gpu bash
```

### 도커 안: 현재 정상 통합 소스 백업

```bash
tar -czf /tmp/morai_before_dynamic_wait.tar.gz \
  -C /root/morai_integration_test/AutoVehicle/morai_ws \
  src/bringup/morai_bringup/launch/integrated_obstacle_v61.launch \
  src/control/purepursuit_mgeo \
  src/control/turn_signal_controller
```

백업이 오류 없이 끝난 경우에만 덮어쓰기:

```bash
python3 -m zipfile -e /tmp/morai_integrated_dynamic_wait.zip /root/morai_integration_test/AutoVehicle/morai_ws
chmod +x /root/morai_integration_test/AutoVehicle/morai_ws/src/control/purepursuit_mgeo/scripts/v61_obstacle_avoidance_node.py
chmod +x /root/morai_integration_test/AutoVehicle/morai_ws/src/control/turn_signal_controller/scripts/maneuver_fusion_node.py
cd /root/morai_integration_test/AutoVehicle/morai_ws
source /opt/ros/noetic/setup.bash
source /root/morai_integration_test/AutoVehicle/morai_ws/devel/setup.bash
catkin_make
```

빌드 성공 후, 원래 명령 그대로 실행:

```bash
source /root/morai_integration_test/AutoVehicle/morai_ws/devel/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.10.21
export ROS_MASTER_URI=http://192.168.10.21:11311

roslaunch morai_bringup integrated_obstacle_v61.launch \
  workspace_path:=/root/morai_integration_test/AutoVehicle/morai_ws \
  enable_control:=true \
  max_speed_kph:=43.2
```

/opt/AutoVehicle/morai_ws에 적용하지 마세요. 현재 정상 주행하는 /root 경로가 대상입니다.
직접 덮어쓰므로 첨부 이후 같은 파일을 별도로 편집했다면 자동 병합되지 않습니다. CHANGES_FROM_REGION_PATCH.patch로 변경을 검토할 수 있습니다.

## 상태 확인

주행 런치를 유지하고 조회용 도커 터미널에서 ROS/워크스페이스 setup과 위 ROS 주소를 설정한 뒤:

```bash
rostopic echo /control/maneuver_status
rostopic echo /ctrl_cmd
```

maneuver_status의 dynamic_zone_guard:
- dynamic_approach_20kph: 접근 감속 상한
- dynamic_path_clear: 구간 경로가 비어 주행 허용
- dynamic_wait_for_path_clear: 장애물 통과 확인 대기
- dynamic_obstacle_input_unavailable: 장애물 입력 미수신/오래됨/형식 이상
- dynamic_route_unmatched / dynamic_pose_unavailable: 위치 확인 불가
- outside_dynamic_zone: 이 보호 로직의 추가 속도/정지 제한 없음
- waiting_ids: 정지 원인 추적 ID
- unobserved_ids: 현재 관측에서 빠진 대기 ID
- identity_uncertain_ids: 재식별이 불확실한 대기 ID
- clearance_frames: ID별 경로 이탈 확인 프레임 수
- input_error: 입력 형식/신선도 오류
- cap_mps: 구간 주행 때 5.555...m/s, 정지 때 0, 구간 밖 제한 없을 때 null
최상위 effective_speed_target_kph는 다른 기존 제약과 합쳐진 목표값입니다.

정지 요구이면 mode=SAFE_STOP, reason에 dynamic_wait_for_path_clear 등이 포함되고 최종 CtrlCmd accel=0/brake=1이 됩니다.
기존 speed_governor_target_kph는 기존 경로 속도 제어기의 값입니다. 새 제한 적용 결과는 effective_speed_target_kph와 dynamic_zone_guard를 확인하세요.
다른 정지 원인이 남아 있으면 대기 ID가 해제돼도 차량은 출발하지 않습니다.

## 설정 위치

src/control/purepursuit_mgeo/config/avoidance_regions.yaml
- dynamic_zone_speed_kph: 20.0
- dynamic_zone_approach_m: 40.0
- dynamic_zone_lookahead_m: 30.0
- dynamic_zone_prediction_s: 2.0
- dynamic_zone_clear_confirm_s: 0.6
- dynamic_zone_margin_m / clearance_extra_m: 각각 0.3

## 원복

ROS 종료 후:

```bash
tar -xzf /tmp/morai_before_dynamic_wait.tar.gz -C /root/morai_integration_test/AutoVehicle/morai_ws
```

직전 구간 패치에서 이번 버전으로 올린 경우 새로 추가된 실행 모듈은 dynamic_zone_guard.py 하나입니다.
복원 후 이 파일을 제거할 수 있습니다.

```bash
rm -f /root/morai_integration_test/AutoVehicle/morai_ws/src/control/purepursuit_mgeo/src/purepursuit_mgeo/dynamic_zone_guard.py
```

복원된 소스로 catkin_make, source devel/setup.bash 후 원래 통합 런치를 실행하세요.
백업은 소스용이며 build/devel 전체 스냅샷은 아닙니다.

## 검증 및 한계

VALIDATION_DYNAMIC_KO.md 참고. ROS 빌드와 실제 MORAI 주행은 아직 수행하지 못했습니다.
이 패치는 주어진 경로/센서 좌표와 일정 속도 예측에 의존합니다. 실제 제동 거리, LiDAR ID 유지 및 박스 오차는 MORAI에서 확인해야 합니다.
