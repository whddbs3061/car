# MORAI 자율주행 통합 저장소

**`new_final_ws` 브랜치 실행 기준:** [Competition 전용 상태 수신과 실행 안내](morai_ws/docs/NEW_FINAL_COMPETITION_KO.md).
기존 `final_ws`의 GPS·IMU 위치 추정과 곡률·신호등 주행을 유지하고,
차량 상태 입력을 Competition Vehicle Status / Ubuntu UDP **9099**로 교체했습니다.
현재 개인 테스트 IP는 Ubuntu `192.168.10.21`, Windows MORAI `192.168.10.14`입니다.
catkin 작업 공간은 저장소 루트가 아닌 **`morai_ws/`**입니다.
`morai_ws_full` 브랜치의 통합 런치를 이 브랜치에서 실행하지 않습니다.

아래 기존 문서는 개발 이력을 위해 보존합니다. 예전 Ego 위치 검증·포트·IP 안내보다
위의 `new_final_ws` 안내를 우선합니다. 기준 커밋과 검증 결과는
[NEW_FINAL_PROVENANCE.json](NEW_FINAL_PROVENANCE.json)에 기록했습니다.

현장 실행은 [주행 테스트 5가지 — 간단 실행 안내](morai_ws/docs/FIELD_TEST_CHECKLIST_20260921_KO.md)를 사용합니다.
곡률 → 신호 정지 → 장애물 회피 → 끼어들기 → 전체 조합을 번호로 선택하고, 최고속도도 한 줄로 변경합니다.

Ubuntu 22.04 + RTX 4090 제어 PC와 Windows MORAI를 사용하는 경우:
**[Docker 처음 설치 → 재부팅 후 실행 → 기능별 테스트 안내](morai_ws/docs/TWO_PC_DOCKER_HIGHWAY_KO.md)**를 따릅니다.
IP는 Ubuntu `192.168.0.185`, MORAI `192.168.0.147` 기준입니다.

새로운 개발 기준은 `morai_ws/`입니다. 기존에 여러 팀과 실험에서 사용하던
실행 코드는 정리했으며, 대회 경로·MGeo·센서·팀 간 인터페이스 정보는
`morai_ws/docs`와 `morai_ws/data`에서 관리합니다.

고속도로 차선 변경·차로 유지·경로 복귀와 회전교차로 위치 설정은
[통합 기능 안내](morai_ws/docs/HIGHWAY_LANE_CHANGE_INTEGRATION_20261005_KO.md)를 참고하세요.
`final_ws_curvature_signal.launch`에 병합한 고속도로 escape4(`enable_highway`)는
[escape4 병합 안내](morai_ws/docs/HIGHWAY_ESCAPE4_KO.md)를 참고하세요.
`merged` 브랜치의 메인 주행·회전교차로·Escape4·장애물 회피 결합과 UDP 포트는
[merged 통합 안내](morai_ws/docs/MERGED_AVOIDANCE_HIGHWAY_KO.md)를 참고하세요.
실행 명령만 순서대로 보려면
[merged 브랜치 실행 방법](morai_ws/docs/MERGED_RUN_KO.md)을 사용하세요.

카메라 포트는 다음으로 통일합니다.

```text
MORAI 1100 → Ubuntu 1101   전방
MORAI 1110 → Ubuntu 1111   좌측
MORAI 1120 → Ubuntu 1121   우측
MORAI 1130 → Ubuntu 1131   네 번째 카메라
```

시작할 때는 [morai_ws/README.md](morai_ws/README.md), [MGeo와 대회 경로 분석](morai_ws/docs/MAP_AND_PATH_ANALYSIS.md),
[이관 계획](morai_ws/docs/MIGRATION_PLAN.md), [세부 규정집 요약](morai_ws/docs/세부규정집_요약.md)을
먼저 확인합니다.

현재 코드의 폴더별 역할과 실제 실행 순서는
[현재 구성 및 실행 보고서](morai_ws/docs/현재_구성_및_실행_보고서.md)를 기준으로 확인합니다.

새 Docker를 만들고 `morai_msgs` 설치부터 곡률 기반 주행·GPS blackout까지 처음부터
검증하려면 [새 Docker 통합 테스트 가이드](TEST_FROM_SCRATCH_KO.md)를 먼저 실행합니다.
