# merged 브랜치 실행 방법

이 문서는 `whddbs3061/car` 저장소의 `merged` 브랜치를 처음 받아 실행하는 명령만
정리한다. catkin 작업공간은 저장소 루트가 아니라 `morai_ws/`이다.

## 1. 코드 받기

```bash
cd ~
git clone --branch merged --single-branch \
  https://github.com/whddbs3061/car.git car_merged
cd ~/car_merged/morai_ws/docker/final_ws
```

`car_merged`는 예시 폴더명이다. 기존 저장소와 다른 폴더명을 사용하면 파일이
겹치지 않는다.

이미 clone한 경우에는 다음만 실행한다.

```bash
cd ~/car_merged
git switch merged
git pull --ff-only origin merged
cd morai_ws/docker/final_ws
```

## 2. 최초 설정

```bash
cp highway.env.example highway.env
cp highway-test.env.example highway-test.env
```

`highway.env`에서 Ubuntu IP, MORAI Windows IP, Docker 이미지와 컨테이너 이름을
현재 PC에 맞춘다. `highway-test.env`의 기본 기능 설정은 다음과 같다.

```text
HIGHWAY_ENABLED=true
AVOIDANCE_ENABLED=true
COMPETITION_STATUS_SOURCE_PORT=9088
COMPETITION_STATUS_PORT=9099
CONTROL_SOURCE_PORT=9094
CONTROL_REMOTE_PORT=9093
```

MORAI 네트워크 설정은 다음과 맞춘다.

| 데이터 | Host/송신 포트 | Destination/수신 포트 |
|---|---:|---:|
| Competition Vehicle Status | MORAI 9088 | Ubuntu 9099 |
| Collision Data | MORAI 9091 | Ubuntu 9092 |
| Ego Ctrl Cmd | Ubuntu 9094 | MORAI 9093 |

## 3. 모델 준비와 Docker 이미지 생성

새 PC 또는 이미지가 없을 때 한 번 실행한다.

```bash
python3 fetch_highway_model.py
bash run_highway.sh build
```

빌드가 성공해야 다음 단계로 진행한다. 실패한 상태에서 `start`,
`install_curvature_signal.sh`을 연속 실행하지 않는다.

## 4. 컨테이너 시작과 설정 확인

```bash
bash run_highway.sh start
bash run_test.sh show 2
```

`show 2`에서 다음을 확인한다.

- 실행 launch: `final_ws_curvature_signal.launch`
- Competition 상태 포트: `9099`, 송신 포트: `9088`
- 제어 포트: Ubuntu `9094` → MORAI `9093`
- `HIGHWAY_ENABLED`, `AVOIDANCE_ENABLED` 값
- LiDAR, CAM1, CAM4 포트와 모델 경로

## 5. 제어를 끈 상태로 확인

```bash
bash run_test.sh monitor 2
```

차량 제어 UDP는 송신하지 않는다. GPS/IMU 위치, Competition 속도, 카메라,
LiDAR, 회전교차로 상태, 회피 상태와 고속도로 상태가 정상인지 확인한다.
확인 후 `Ctrl+C`로 종료한다.

## 6. 실제 주행 시작

```bash
bash run_test.sh drive 2
```

이 명령만 제어 UDP를 활성화한다. 이상이 있으면 `Ctrl+C`로 즉시 종료한다.

## 7. 상황별 기능 분리

ZIP 회피와 Escape4 고속도로 로직은 서로 다른 상황이다. 두 기능은 경로와 제어기가
분리되어 있고 최종 mux만 공유한다.

### 일반 장애물 회피만 시험

`highway-test.env`를 다음과 같이 설정한다.

```text
HIGHWAY_ENABLED=false
AVOIDANCE_ENABLED=true
```

```bash
bash run_test.sh monitor 2
# 확인 후 Ctrl+C
bash run_test.sh drive 2
```

### 고속도로 Escape4만 시험

```text
HIGHWAY_ENABLED=true
AVOIDANCE_ENABLED=false
```

```bash
bash run_test.sh monitor 2
# 확인 후 Ctrl+C
bash run_test.sh drive 2
```

### 전체 코스

```text
HIGHWAY_ENABLED=true
AVOIDANCE_ENABLED=true
```

고속도로 구간에서는 Escape4 제어기, 그 밖의 장애물 회피 상황에서는 ZIP 회피
제어기, 평상시에는 기존 메인 주행·회전교차로 제어기가 명령권을 가진다.

## 8. 코드 업데이트 후 기존 컨테이너에 설치

이미지를 다시 만들지 않고 최신 코드를 기존 컨테이너에 넣을 때만 실행한다.
실행 중인 주행 launch를 먼저 `Ctrl+C`로 종료한다.

```bash
cd ~/car_merged/morai_ws/docker/final_ws
bash run_highway.sh stop
bash install_curvature_signal.sh --config
bash run_highway.sh start
bash run_test.sh show 2
bash run_test.sh monitor 2
```

## 9. 종료

주행 launch에서 `Ctrl+C`를 누른 다음 컨테이너를 종료한다.

```bash
bash run_highway.sh stop
```

## 10. 오프라인 검증

```bash
cd ~/car_merged
python3 morai_ws/tools/validate_new_final.py
```

오프라인 테스트 통과는 실제 MORAI 주행 성공을 의미하지 않는다. 첫 주행은 반드시
`monitor 2`로 센서·경로·상태를 확인한 뒤 낮은 속도에서 진행한다.
