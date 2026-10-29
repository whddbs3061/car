# new_final_ws 회전교차로 V6.1 통합

`test/roundabout` 신호 제어 후속 수정: [교차로 진입 후 주행 유지·신뢰도 0.6 초과](20261030_SIGNAL_ENTRY_RELEASE_KO.md).

## 통합 기준

- 기준 코드: `shinejihun1227/AutoVehicle`의 `new_final_ws`
- 판단 기능 출처: `ohdang23/yujin_morai`의 V6.1 (`6c7d805`)
- 경로, 위치 추정, Competition Status, 조향, 속도 PI, 가감속/페달 변화율,
  카메라·신호 제어와 최종 `/ctrl_cmd` 발행은 모두 `new_final_ws`를 유지한다.
- V6.1은 LiDAR 차량 관측으로 회전교차로 진입 가능 여부와 미션 속도 상한만 계산한다.
  원본의 별도 `roundabout_test_driver`와 PID는 사용하지 않는다.
- 메인 곡률 속도 상한과 V6.1 미션 상한 중 작은 값을 메인 PI에 넣는다.
  V6.1의 정지/비상 요청은 메인 PI 계산 전에 합성한다.

이식된 판단은 ID 비의존 관측 이력, 가림 구간 차량 흐름 예측, 3×3 경로 footprint
예측, COMMIT 유지, 정적 장애물 보호와 V6.1 상대속도 선행차 추종을 포함한다.
순환 경로는 메인 MGeo의 `A2256W000134 → A2256W000135 → A2256W000132 →
A2256W000133` 링크로 만들었고, 메인 전역 경로의 기존 V6.1 인덱스
1720~1900과 conflict 42~52를 사용한다.

## 실행 방법

기존 `new_final_ws`와 명령이 같다. MORAI LiDAR 목적지는 Ubuntu의 UDP 2001로
설정하고, GPS 3001, IMU 4001, Competition Status 9099, CAM1 1101, CAM4 1131은
기존 설정을 유지한다.

Docker 갱신:

```bash
cd ~/AutoVehicle_new_final_ws/morai_ws/docker/final_ws
bash run_highway.sh stop
bash install_curvature_signal.sh --config
bash run_highway.sh start
bash run_test.sh show 2
bash run_test.sh monitor 2
```

`monitor 2`에서 위치·LiDAR·판단 상태를 확인한 뒤 종료하고 주행한다.

```bash
bash run_test.sh drive 2
```

Native ROS Noetic 빌드와 실행도 기존 방식 그대로다.

```bash
cd ~/AutoVehicle_new_final_ws/morai_ws
source /opt/ros/noetic/setup.bash
catkin_make -j2 -l2 -DPYTHON_EXECUTABLE="$(command -v python3)"
cd ..
xvfb-run -a bash RUN_NEW_FINAL.sh monitor
# 확인 후
xvfb-run -a bash RUN_NEW_FINAL.sh drive
```

일시적으로 회전교차로 판단을 끄려면 기존 명령 끝에
`enable_roundabout_mission:=false`를 추가한다. 기본값은 `true`다.

## 확인 토픽

```bash
rostopic hz /perception/lidar/tracked_obstacles_map
rostopic echo /roundabout_merge_gate/status
rostopic echo /experimental/roundabout_mission_stop
rostopic echo /experimental/roundabout_mission_reason
```

Docker에서는 기존 진단 명령이 회전교차로 상태와 LiDAR 수신도 함께 출력한다.

```bash
bash run_test.sh diagnose 2
```

정상 상태 흐름은 `CRUISE → APPROACH/READY/WAIT → COMMIT → FOLLOW → EXIT`다.
판단 메시지가 끊기거나 odometry/LiDAR가 미션 구간에서 오래되면 메인 제어기는
fail-safe 정지를 요청한다. `allowed=false`만으로 전역 정지하지는 않는다.

## 검증 한계

오프라인 단위 테스트, 설정/경로 일치, launch XML과 메인 제어 합성은 검사한다.
Windows 작업 환경에서는 ROS Noetic catkin 빌드, 실제 MORAI LiDAR 패킷,
가림막 뒤 차량 인지와 무충돌 통과를 검증할 수 없다. 첫 실주행은 `monitor 2`와
저속·무차량 조건부터 시작하고, `decision_compute_ms`, 좌표계와 LiDAR 장착값을
확인해야 한다.
