# CLAUDE.md

MORAI 시뮬레이터(24.R2) 기반 자율주행 차선 인식 프로젝트.

## 환경

Python 3.10. conda는 필수가 아니다.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

검증된 조합: opencv-python 5.0.0 / numpy 2.2.5 / matplotlib 3.10.9

**onnxruntime은 설치하지 않는다.** 아래 "성능" 참고.

## 실행

```
# 오프라인 (녹화 영상 또는 이미지)
python Sensor/LaneEgoSelect.py --source recordings/full/drive.mp4
python Sensor/LaneEgoSelect.py --source samples

# 실시간 (시뮬레이터 필요)
python Sensor/LaneEgoSelect.py

# 주행 녹화 (시뮬레이터 있는 PC에서만)
python Sensor/RecordDrive.py --name full --no-preview
```

시뮬레이터 주소는 환경변수 `MORAI_CAM_IP` / `MORAI_CAM_PORT` 또는 `--cam-ip` /
`--cam-port`로 바꾼다. 기본값 `192.168.0.200:1101`. PC/가상머신마다 다르므로
프레임이 안 들어오면 여기부터 확인한다.

## lane_segmentation.onnx — 반드시 알아야 할 것

**출력이 두 개다.** YOLOP 계열 모델이다.

```
input  "images": [1, 3, 360, 640]   ← 크기 고정, 리사이즈 불가
output "da":     [1, 2, 360, 640]   drivable area (주행 가능 영역)
output "ll":     [1, 2, 360, 640]   lane line (차선)  ← 이게 차선 검출기
```

함정 세 가지:

1. **`net.forward()`는 첫 번째 출력(`da`)만 돌려준다.** 차선을 얻으려면
   `net.forward(net.getUnconnectedOutLayersNames())`로 이름을 지정해 받아야 한다.
   이걸 놓치면 "차선 모델이 아니라 도로 영역 모델"이라는 잘못된 결론에 도달한다.
2. **출력은 softmax 이전 logit이다** (범위 약 −6.5 ~ 3.3). `> 0.5` 임계값은
   의미가 없다. 채널 argmax(`logits[0,1] > logits[0,0]`)로 이진화한다.
3. **차선은 폭이 1~3px로 얇다.** 5x5 `MORPH_OPEN`을 걸면 통째로 지워진다.
   3x3 `MORPH_CLOSE`로 끊긴 점선만 잇는다.

## 성능 — onnxruntime을 쓰지 않는 이유

같은 모델, 같은 머신(CPU 8스레드) 벤치마크:

| 방식 | 속도 |
|---|---|
| **cv2.dnn** | **177 ms · 5.6 FPS** |
| onnxruntime intra=8 | 337 ms · 3.0 FPS |
| onnxruntime intra=4 | 438 ms · 2.3 FPS |

cv2.dnn이 약 2배 빠르다. "ONNX니까 onnxruntime"은 이 경우 틀렸다.
바꾸려면 먼저 다시 측정할 것.

추론이 느리므로 `seg_interval`로 N프레임마다만 추론하고 사이 프레임은 직전
마스크를 재사용한다. 도로·차선은 프레임 간 변화가 느려서 유효하다.

| seg_interval | 처리율 |
|---|---|
| 1 | 5.2 FPS |
| 2 (실시간 기본값) | 10.7 FPS |
| 3 | 13.9 FPS |

## 파이프라인

```
프레임 → combine_masks → limit_region(ROI) → fit_polynomial → draw_lane
```

**combine_masks**: `ll` 마스크가 주력. 비어 있을 때만 (HSV 컬러 마스크 ∩ 팽창시킨
`da` 마스크)로 보완한다. HSV 단독은 하늘·구름·본네트를 그대로 통과시키므로
(맑은 날 24%, 안개 씬은 65%가 마스크로 잡힘) 반드시 주행 가능 영역으로 잘라낸다.

**limit_region**: 사다리꼴 ROI. 값은 `1.00 / 0.80 / 0.65`
(bottom/top/height). 예전 값 `0.85 / 0.07 / 0.49`는 노이즈 많은 HSV 마스크
기준이라 CNN 차선 마스크의 75%를 잘라냈다(광각인 cam4는 97%). 지금 마스크에는
하늘·본네트가 없으므로 ROI는 지평선 가드 역할만 한다. **다시 좁히지 말 것.**

**fit_polynomial — ego-lane 선택**: 히스토그램 최대점 + 슬라이딩 윈도우 방식은
차선이 4개 이상 보이면 옆 차로의 바깥 선을 잡고, 윈도우가 위로 올라가며 다른
선으로 갈아타 여러 차선을 가로지르는 곡선을 만든다. 그래서 다음으로 바꿨다:

1. `connectedComponentsWithStats`로 성분 분리
2. 기울기·하단 x절편이 비슷한 성분끼리 병합 (점선이 여러 조각으로 쪼개지므로)
3. 각 차선을 화면 하단(y=h−1)까지 **직선**으로 외삽해 x절편 계산
   (2차 곡선은 화면 밖까지 밀면 발산해서 절편이 엉뚱해진다)
4. 자차 중심 기준 좌/우에서 가장 가까운 선을 ego lane 경계로 선택
5. 선택된 두 차선의 픽셀만으로 다항식 피팅

**draw_lane**: 차선 픽셀이 관측된 y 구간(`fit_y_range`) 안에서만 그린다.
차선은 소실점으로 수렴하므로 데이터 위쪽으로 외삽하면 소실점을 지나 **좌우가
역전되어 곡선이 교차한다.** 고정 y 범위로 되돌리지 말 것.

**시간 필터**: 직전 프레임 대비 `MAX_CENTER_JUMP`(90px)를 넘게 튀면 기각한다.
단, 연속 기각이 `REJECT_RESET_FRAMES`(5) 이어지면 잠금을 푼다 — 실제 차로
변경에서 영구 고착되는 것을 막기 위해서다.

튜닝 상수는 `Sensor/LaneEgoSelect.py` 상단에 모여 있다.

## 파일

| 파일 | 역할 |
|---|---|
| `Sensor/LaneEgoSelect.py` | **주 작업 파일.** 하이브리드 + ego-lane 선택 |
| `Sensor/RecordDrive.py` | 주행 녹화 (mp4 + meta.jsonl) |
| `Sensor/LaneSegHybrid.py` | ego-lane 선택 이전 스냅샷 |
| `Sensor/LaneSegPoly.py` | 하이브리드 이전 스냅샷 |
| `Sensor/DeepLaneSeg.py` | 단독 세그멘테이션 데모 |
| `Sensor/Line1.py` | 실시간 HSV 컬러 마스크만 |
| `Sensor/Line1 copy.py` | HSV 트랙바 오프라인 튜닝 |
| `lane_segmentation.onnx` | Sensor/ 에 있음 (1.7MB) |

## 검증

`samples/` 에 시뮬레이터 캡처 12장이 있다. GUI 없이 헤드리스로 돌리려면
`cv2.imshow` / `cv2.waitKey`를 스텁으로 갈아끼운다:

```python
m.cv2.imshow = lambda name, img: shown.append(img.shape)
m.cv2.waitKey = lambda *a: next(keys)
m.cv2.destroyAllWindows = lambda: None
```

현재 결과: cam1 6장(맑음 3 + 안개 3)과 cam4 4장은 선택 정확. 로그의 **차로 폭**이
안정적인지(453~456) 보는 게 가장 빠른 판정법이다.

## 알려진 한계

- **sample1(교차로)**: 차선이 거의 수평이라 `abs(vy) < 0.2` 필터에 걸려 후보가
  1개만 나온다. 교차로는 자기 차로 개념 자체가 모호해 우선순위를 낮게 뒀다.
- **cam3**: 측면 카메라(사이드미러가 보임). 전방 뷰 학습 모델이라 실패하는 게
  정상이다. 차선 검출 입력에서 제외한다.
- **시간 필터 상수가 미검증**: `MAX_CENTER_JUMP`, `REJECT_RESET_FRAMES`는
  정지 이미지로 정한 추정값이다. 실제 주행 영상으로 재조정해야 한다.

## link_id

`EgoVehicleStatus`(포트 909)의 38바이트 필드. 자차가 올라가 있는 HD 맵 링크로,
**차로 단위 정답**이다. 차로를 옮기면 값이 바뀐다.

**인지 파이프라인 입력으로 쓰지 않는다** — 실차에 없는 정보이고, 쓰면 카메라
차선 인식을 만드는 의미가 없어진다. 평가·라벨링 전용이다. 차로 변경 시점이
자동으로 라벨링되므로 시간 필터 상수를 정하는 근거로 쓴다.

## 다음 단계

1. 전역 경로 전체를 한 번 녹화 (구간별로 쪼개지 말 것 — 나중에 `meta.jsonl`의
   `pos_x/y`, `ang_vel_z`, `link_id`로 잘라낼 수 있고, 직선→곡선 진입 같은
   전이 구간이 오히려 중요하다)
2. 그 영상으로 오프라인 반복하며 시간 필터 상수 튜닝
3. 지표 로깅: 차로 폭 일관성, 프레임 간 x절편 변화량, 기각률, FPS
