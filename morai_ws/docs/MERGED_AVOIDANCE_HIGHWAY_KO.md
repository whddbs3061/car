# merged 주행·회전교차로·고속도로·장애물 회피 통합

## 기준과 병합 원칙

- 기준 저장소/브랜치: `whddbs3061/car`의 `test/roundabout-highway`
- 기준 커밋: `4259fc184149bc9b3c53f6b929e8a21754febeee`
- 결과 브랜치: `merged`
- 추가 기능 원본: `morai_v7_avoidance_sources.zip`
- ZIP SHA-256: `EB47FB5A857723AFA3279322216BA25547B45D7735E973B2F9401E70A2FB8AD9`

ZIP 209개 파일을 LF 기준으로 비교했다. 131개는 메인과 동일했고, 24개는
메인 쪽에 더 최신 통합·안전 수정이 있었으며, 54개는 과거 단계별 launch·백업 또는
병렬 제어기였다. 따라서 ZIP을 통째로 덮어쓰지 않고, 메인에 이미 존재하는 최신
Frenet planner, bypass guard, path manager를 실제 최종 launch에 연결했다.

ZIP의 별도 roundabout test driver와 구형 command selector는 넣지 않았다. 기존
곡률 Pure Pursuit가 일반 주행과 회전교차로 제어변수를 계속 소유한다.

## 최종 제어권

1. 일반 구간: 기존 `curvature_speed_purepursuit`
2. 장애물 회피 상태: `avoidance_path_manager` 경로를 메인 제어 파라미터와 맞춘
   전용 회피 곡률 제어기가 추종
3. 고속도로 구간: Escape4가 기존 고속도로 base 경로에서 차선 변경·추월·복귀를
   수행하고 고속도로 전용 제어기가 추종
4. `highway_command_mux`가 상황에 맞는 한 명령만 `/control/ctrl_cmd`로 전달
5. 기존 신호/정지선 fusion이 최종 `/ctrl_cmd`를 생성

ZIP 회피와 Escape4는 서로 다른 상황이다. 회피 경로를 Escape4 base 경로로 넣지
않는다. 고속도로 phase가 활성화된 구간에는 고속도로 제어기가 우선하고, 그 밖에서
회피 상태가 활성화되면 회피 제어기가 명령권을 가진다. 둘 다 비활성이면 기존 메인
명령으로 돌아간다. 필요한 상태나 선택된 명령이 오래되면 전제동한다.

## UDP 포트

| 데이터 | MORAI Host/송신 | Ubuntu Destination/수신 |
|---|---:|---:|
| Competition Vehicle Status | 9088 | 9099 |
| Collision Data | 9091 | 9092 |
| Ego Ctrl Cmd | Ubuntu 9094 | MORAI 9093 |

상태·충돌 수신기는 기본적으로 송신 포트도 검사한다. 중계 프로그램이 임의 송신
포트를 쓰는 환경에서는 각각 `ego_status_source_port:=0`,
`collision_source_port:=0`으로 검사를 명시적으로 끌 수 있다.

## 실행

전체 실행 명령은 [merged 브랜치 실행 방법](MERGED_RUN_KO.md)을 따른다.

```bash
cd ~/car/morai_ws/docker/final_ws
cp highway.env.example highway.env
cp highway-test.env.example highway-test.env
python3 fetch_highway_model.py
bash run_highway.sh build
bash run_highway.sh start
bash run_test.sh monitor 2
```

모니터에서 센서와 경로가 정상임을 확인한 뒤에만 제어를 켠다.

```bash
bash run_test.sh drive 2
```

프로필 2는 기본적으로 다음을 함께 켠다.

```text
HIGHWAY_ENABLED=true
AVOIDANCE_ENABLED=true
COMPETITION_STATUS_PORT=9099
CONTROL_REMOTE_PORT=9093
CONTROL_SOURCE_PORT=9094
```

## 검증 범위

```bash
cd ~/car
python3 morai_ws/tools/validate_new_final.py
```

오프라인 테스트는 배선, 경로 계획, 제어권 전환, stale 입력 전제동과 UDP 프로토콜을
검사한다. 실제 MORAI 차량 주행, 실차 통신 지연, 주최 측 전달기 호환성은 현장에서
별도로 확인해야 한다.
