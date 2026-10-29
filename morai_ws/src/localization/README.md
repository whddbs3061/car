# 위치 추정과 Competition 차량 상태

`new_final_ws`의 실행 기준은 [Competition 상태 전환 안내](../../docs/NEW_FINAL_COMPETITION_KO.md)다.
기존 `final_ws`의 GPS 변환기와 EKF 구현을 그대로 유지한다.

| 정보 | 입력 | 사용처 |
|---|---|---|
| 위경도·고도·GPS 속도 | GPS UDP 3001 → `/gps` → `/localization/gps` | MGeo 좌표 변환, EKF 초기 위치와 GPS 보정 |
| 자세·각속도·가속도 | IMU UDP 4001 → `/Imu` | EKF 예측 |
| 실제 전진 속도 | Competition 상태 UDP 9099 → `/Ego_topic.velocity.x` | 속도 제어, 신호 제어의 속도 판단, 평가 |
| 차량 map pose | `/localization/odometry` | 경로 추종, 정지선·신호등 투영 |

일반 Ego Vehicle Status 수신 경로는 제거했다. `/Ego_topic`과
`morai_msgs/EgoVehicleStatus`는 기존 코드와 연결하기 위한 **내부 호환 이름**이다.
이 토픽의 위치·가속도 xyz, 속도 yz는 NaN(미제공)이며 위치 검증 기준으로 쓰지 않는다.
Competition 상태는 heading과 전진 속도 등을 제공하지만 map 위치를 제공하지 않는다.
따라서 Competition 상태만으로 절대 위치를 초기화할 수 없다.

GPS-MGeo 정렬은 Competition의 예약 위치 슬롯으로 보정하지 않는다.
`localization_alignment.yaml`의 `transform_validated: false`를 유지하며 실제 GPS·IMU와
지도의 일치 여부를 별도로 확인해야 한다. GPS가 없으면 새 초기 위치를 만들어내지 않는다.
이미 초기화된 EKF의 GPS blackout 동안에는 기존 IMU 예측이 동작한다.

기존 좌표 설계 문서는 이력을 위해 보존한다. 그 안의 Ego 위치와 비교하는 절차는
이 브랜치의 허용 입력이나 실행 절차가 아니다.
