# final_ws 기본 주행 + 고속도로 + 회전교차로 6.1

기준: ohdang23/morai_merge 8e08a0c. 지훈님 final_ws의 c97840c 소스와 비교했으며,
같은 제어 소스를 중복 복사하지 않고 final_ws_bringup의 실행 구성을 연결했다.
기존 integrated_no_obstacle_v61.launch는 유지한다. 두 launch를 동시에 실행하지 않는다.

## 제어 흐름

일반 구간 curvature_speed_purepursuit_node.py → /integration/nominal_cmd
고속도로 adaptive_curvature_purepursuit_node.py → /integration/highway_cmd
회전교차로 roundabout_test_driver 6.1 → 상태 기반 명령
위 입력 → no_obstacle_integration → /control/ctrl_cmd
→ 정지선 wrapper → 차선 fallback → maneuver_fusion → control_mux → /ctrl_cmd → UDP

회전교차로 active가 우선이며, 고속도로 구간은 기존 누적거리 1118.742~1741.721m와
경로 거리 11m 이내 조건을 사용한다. 해당 구간에서만 고속도로 제어 명령을 선택한다.
선택한 명령이나 회전교차로 상태가 오래되면 다른 제어기로 전환하지 않고 정지한다.
지도/경로 파일을 바꾸면 이 구간 경계도 다시 확인해야 한다.
고속도로 경로 인정 및 시작점 0.5m 보완, 기존 PP 수정은 보존한다.
고속도로 센서 판단 노드는 실행되지만 최종 제어 권한은 구간에 따라 선택된다.

ROI 카메라 lane/YOLO와 LiDAR는 final_ws가 한 번만 실행한다.
고속도로도 동일한 lane_info를 사용한다. 이전 별도 highway_best 추론 노드를 추가로
띄우지 않으므로 기존 highway_best 전용 추론과 카메라 모델 구성이 다르다.
기본 lane_seg_best.pt, best0917.pt, yolov8s.pt가 필요하다.
ROI 안전정지 adapter 및 control_mux를 활성화한다. 장애물 우회 경로 생성은 추가하지
않지만 전방 장애물, 보행자, 교차로 안전정지 동작은 final_ws 기본 구성대로 적용된다.

## 적용

ROS launch를 종료하고 install.py에 catkin workspace 경로를 전달한다.
installer는 수정 대상 원본 해시를 검사하여 로컬 변경을 덮어쓰지 않는다.
다르면 중단하고 그 파일을 비교해야 한다. 기존 파일은 workspace 바깥 백업 폴더에 보존한다.

```bash
python3 install.py /root/morai_integration_test/AutoVehicle/morai_ws
source /opt/ros/noetic/setup.bash
cd /root/morai_integration_test/AutoVehicle/morai_ws
catkin_make
source devel/setup.bash
export ROS_MASTER_URI=http://192.168.10.21:11311
export ROS_IP=192.168.10.21
roslaunch morai_bringup final_ws_missions_v61.launch \
  workspace_path:="$PWD" enable_control:=false
```

기본 IP는 Windows 192.168.10.14, Ego 수신 포트는 1911이다.
제어 false에서도 내부 nominal 명령을 발행하여 흐름을 검사하지만 UDP 제어 송신과
램프 송신은 꺼져 있다. 기존 단독 final_ws의 기본 동작은 바꾸지 않았다.

## 최초 확인

```bash
rostopic info /ctrl_cmd
rostopic echo -n 1 /control/mux_status
rostopic echo -n 1 /control/maneuver_status
rostopic echo -n 1 /no_obstacle_integration/status
rostopic echo -n 1 /detection/fused_safety_stop
timeout 3 rostopic echo -n 1 /Ego_topic
```

/ctrl_cmd의 유일한 발행자는 control_mux여야 한다.
최초 상태 검사 후에만 동일 실행의 enable_control을 true로 변경한다.

## 신호등·방향지시등 설정

원본 final_ws 설정 turn_signal_maneuvers.yaml을 기본 사용한다.
그 파일의 signal_camera.calibrated는 false이므로 보정 검증 전 교차로 통행은
허용되지 않을 수 있다. 단순히 true로 바꾸어 정지를 우회하지 않는다.
실제 검증된 설정은 turn_signal_maneuvers_file 인자로 지정할 수 있다.
기존 curvature_signal.yaml은 제약 조건이 다른 설정이므로 그대로 지정하면
원본 final_ws의 신호 판단 동작과 달라질 수 있다.
방향지시등은 원본의 5초 선행 조건을 유지한다. 시뮬레이터 램프 UDP 9097 설정을
확인한다. 램프 없는 시뮬레이터 시험은 test_without_turn_signals:=true를 명시한다.

## 검증과 한계

오프라인 테스트 12개 통과: 미션 명령 우선순위, stale/future 명령 정지, 구간 종료 후
일반 주행 복귀, include 인자 연결, 중복 센서 노드 및 송신 gate 검사.
Python 문법 검사와 XML 파싱 통과. 이 환경에는 ROS가 없어 catkin 빌드,
roslaunch 실제 확장, UDP 연결, MORAI 실주행은 검증하지 못했다.
이전에 관찰된 /Ego_topic 미수신 및 YOLO 오검출 문제를 해결하는 패치는 아니다.
신호/안전정지 상태가 유효하지 않으면 이번 구성도 정지할 수 있다.
