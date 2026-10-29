현재 카메라·신호·우회전·차선 보정 업데이트: [20261029_CAMERA_SIGNAL_KO.md](20261029_CAMERA_SIGNAL_KO.md).
회전교차로 V6.1 통합과 동일 실행 방법: [ROUNDABOUT_V61_INTEGRATION_KO.md](ROUNDABOUT_V61_INTEGRATION_KO.md).
기존 로컬 env는 `bash run_test.sh lane on`으로 새 보정 프리셋을 적용한다.

# new_final_ws: final_ws의 Competition 상태 입력 전환

## 기준과 변경 범위

`final_ws`의 `c97840c987880736b0b4b10ef346f90e904d9d28`에서 분기했다.
기존 곡률 주행·순환 경로·CAM1 차선·CAM4 신호등·정지선 기능을 유지한다.
`morai_ws_full`에서 나중에 통합한 고속도로·장애물 코드는 가져오지 않았다.
기존 `final_ws`, `jihun_ws`, `merge_ws`, `morai_ws_full`은 변경하지 않는다.
기준·파일 출처·검증 결과는 루트 `NEW_FINAL_PROVENANCE.json`에 기록했다.

catkin 작업 공간은 저장소 루트가 아닌 **`morai_ws/`**다.
이 브랜치의 주행 런치는 **`final_ws_curvature_signal.launch`**다.
`morai_ws_full`의 `RUN_INTEGRATED.sh`를 실행하는 구성과 다르다.

## 기존 위치 추정과 달라진 입력

기존 final_ws의 **위치는 GPS+IMU**, **주행 속도 피드백은 일반 Ego Vehicle Status**였다.
전체 localization이 Competition Vehicle Status에서 나오던 구조가 아니다.
기존 EKF의 `/localization/gps`, `/Imu` 구독과 추정 구현은 변경하지 않았다.

| 정보 | new_final_ws 입력과 처리 |
|---|---|
| map 위치 x/y/z | GPS UDP 3001 → GPS-MGeo 변환 → GPS/IMU EKF |
| 자세·각속도·관성 가속도 | IMU UDP 4001 → 기존 EKF |
| 실제 전진 속도 | Competition UDP 9099의 velocity.x, km/h → m/s 1회 변환 |
| 경로·신호등 투영용 pose | EKF `/localization/odometry` |
| 일반 Ego Vehicle Status | 수신 parser와 모드 전환 기능 제거 |

Competition에는 위치 xyz, 속도 yz, 가속도 xyz와 일부 타이어 정보가 없다.
해당 예약 슬롯을 해석하지 않으며 ROS 내부 벡터도 **NaN(미제공)**으로 표시한다.
Competition을 구독하는 속도 제어·신호 제어·평가기는 `abs(velocity.x)`를 사용한다.
EKF Odometry를 구독하는 기존 노드의 속도 계산은 유지한다.
IMU 센서에서 직접 받은 가속도는 별도 입력이므로 기존처럼 사용한다.
GPS 없이 Competition만 받아서는 초기 map 위치를 알 수 없다.
초기화 이후 GPS blackout 동안에는 기존 EKF의 IMU 예측이 동작한다.

GPS-MGeo 정렬은 실측 확인 전까지 `transform_validated: false`다.
예전 문서의 Ego 위치와 비교하는 절차는 이 브랜치에서 적용하지 않는다.

## 패킷과 내부 토픽 이름

첨부 MORAI 안내의 “EgoVehicleStatus.py 예제 활용”은 **UDP 형식 재사용**을 뜻한다.
팀 Competition parser와 보관된 NetworkModule 구조체를 대조하여 181/229바이트,
`#MoraiInfo$`, data length 152/200, 원본 시각, CRLF를 검증한다.
확장 패킷을 181바이트로 잘라 읽지 않는다. 중복·역순 원본 시각은 폐기하므로
장면 초기화로 시각이 되돌아가면 주행 런치를 다시 시작한다.

ROS 이름 **`/Ego_topic`**과 타입 **`morai_msgs/EgoVehicleStatus`**는
기존 제어 코드와 연결하기 위한 내부 호환 이름이다. 외부 수신은 Competition 전용이다.
`/morai_udp_drive_bridge/status_metadata`에서 source, listen_port, packet_length와
미제공 필드 여부를 확인할 수 있다. `ego`, `ego_legacy`, `auto` 모드는 허용하지 않는다.

**같은 예제 형식을 공유하므로 헤더만으로 일반 Ego와 Competition 송신을 구별할 수 없다.**
MORAI 송신 항목을 반드시 **Competition Vehicle Status**로 선택하고 일반 Ego 송신과
동일 토픽을 발행하는 별도 ROS publisher를 꺼야 한다.
제어 출력인 `EgoCtrlCmd`는 명령 프로토콜 이름이므로 유지한다.

## 개인 테스트 네트워크

| 항목 | 설정 |
|---|---|
| Windows MORAI / Ubuntu | 192.168.10.14 / 192.168.10.21 |
| Competition 상태 목적지 | Ubuntu 192.168.10.21 : **9099 / UDP** |
| GPS / IMU 목적지 | Ubuntu : 3001 / 4001 |
| CAM1 / CAM4 목적지 | Ubuntu : 1101 / 1131 |
| 차량 명령 | Ubuntu 송신 9094 → Windows MORAI 수신 9093 |

9099은 사용자가 지정한 개인 테스트 수신 포트다. 대회 Windows loopback 전달기의
포트와 같은 의미로 간주하지 않는다. 실제 전달기의 목적지 포트는 별도 확인한다.
YAML은 참고 설정이며 실제값은 launch 인자와 로컬 env가 결정한다.

Docker 설정은 `highway.env`의 IP·컨테이너 이름과
`highway-test.env`의 **`COMPETITION_STATUS_PORT=9099`**이다.
예전 저장값 `EGO_STATUS_PORT=1911`은 무시한다.
launch 인자 `ego_status_port`는 호환 이름이며 기본값은 9099이다.

## 기존 Ubuntu Docker 환경에서 실행

먼저 이전 ROS 주행 런치를 Ctrl+C로 종료한다. 별도 폴더를 만들어 기존 환경을 보존한다.

```bash
git clone --single-branch --branch new_final_ws https://github.com/shinejihun1227/AutoVehicle.git ~/AutoVehicle_new_final_ws
cd ~/AutoVehicle_new_final_ws/morai_ws/docker/final_ws
cp highway.env.example highway.env
cp highway-test.env.example highway-test.env
```

새 `highway.env`의 `IMAGE_NAME`, `TORCH_FLAVOR`는 현재 설치된 ROS 이미지와 맞춘다.
현재 예제는 Ubuntu .21, MORAI .14, **`CONTAINER_NAME=morai-new-final-net10`**이다.
기존 컨테이너와 다른 이름을 사용한다. 기존의 `highway-test.env`를 새 폴더로 복사해도 된다.
그 파일에 `COMPETITION_STATUS_PORT=9099`을 지정한다.
새 예제 상한 속도는 5km/h이며, 기존 설정을 복사하면 그 값이 우선한다.
기존처럼 35km/h 상한으로 시작하려면 `bash run_test.sh speed 35`를 실행한다.

기존 ROS 이미지가 있으면 다음 순서로 새 컨테이너에 설치한다.
이미지가 없다면 먼저 `bash run_highway.sh build`가 필요하다.

```bash
python3 fetch_highway_model.py
bash run_highway.sh start
bash run_highway.sh stop
bash install_curvature_signal.sh --config
bash run_highway.sh start
bash run_test.sh show 2
bash run_test.sh monitor 2
```

`show 2`에서 `VehicleStatus=Competition_Vehicle_Status port=9099`, IP와 컨테이너를 확인한다.
`monitor 2`는 제어 UDP를 보내지 않는다. 설치 검사에서 **`RUNTIME_MATCH`**가 나와야 한다.
소스만 pull하고 예전 컨테이너를 실행하면 새 수신부가 적용되지 않는다.
MORAI의 Competition·GPS·IMU 송신 확인 후 monitor를 Ctrl+C로 종료하고 주행한다.

```bash
bash run_test.sh drive 2
```

별도 터미널의 `bash run_test.sh diagnose 2`로 속도·pose·최종 제어를 확인할 수 있다.

## 이미 Noetic 환경을 직접 사용하는 경우

ROS Noetic·morai_msgs·기존 카메라 의존성이 준비된 환경에서 새 `morai_ws/`를 빌드한다.
Docker 안에서는 기존 morai-entrypoint 셸 환경을 사용한다.
호스트에 roslaunch만 추가 설치하는 것으로 대체할 수 없다.

```bash
cd ~/AutoVehicle_new_final_ws/morai_ws
source /opt/ros/noetic/setup.bash
# morai_msgs가 별도 작업 공간에 있으면 그 devel/setup.bash도 source한다.
python3 docker/final_ws/fetch_highway_model.py
catkin_make -j2 -l2 -DPYTHON_EXECUTABLE="$(command -v python3)"
cd ..
xvfb-run -a bash RUN_NEW_FINAL.sh monitor
```

monitor 종료 후 `xvfb-run -a bash RUN_NEW_FINAL.sh drive`로 실행한다.
이 스크립트는 새 폴더의 morai_bringup 선택을 검사하고 Windows .14와 상태 9099을 전달한다.
직접 launch 기본값은 상한 35km/h, 앞범퍼 정지선 여유 0.5m다.
Docker의 저장 env와는 별도이며 이번 전환에서 제어값을 자동 재튜닝하지 않았다.
이전 통합 브랜치에서 요청한 7m 정차를 시험하려면 직접 실행에
`stopline_hold_distance_m:=7.0`을 추가하거나 Docker에서 `bash run_test.sh clearance 7`을 설정한다.

## 검증 범위

루트에서 `python3 morai_ws/tools/validate_new_final.py`를 실행한다. NumPy·PyYAML·Bash가 필요하다.
패킷→내부 메시지→기존 속도 제어, 미제공 필드 NaN, 구형 포트 무시, 실제 GPS·IMU EKF,
신호·정지선·경로 제어·평가기 및 Python 3.8 문법과 launch XML을 검사한다.
결과는 `morai_ws/test-results/new_final/`에 저장한다.

실제 Competition UDP 캡처, Ubuntu catkin 빌드, AV 전환과 MORAI 주행은
이 Windows 작업 환경에서 수행하지 못했다. 합성 패킷 검증을 실제 연동 완료로 간주하지 않는다.
공식 기준은 작업 폴더에서 확인한 v1.2 / 최종 14개 체크포인트이며,
이 변경만으로 전체 대회 규정 충족을 보장하지 않는다.
