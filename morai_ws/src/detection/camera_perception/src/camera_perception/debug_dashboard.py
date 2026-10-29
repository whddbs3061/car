"""Bounded read-only telemetry store and HTTP endpoints (no ROS dependency)."""
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs


EXPLANATIONS = {
    'camera_observation_stream_stale': '카메라 관측이 오래되거나 들어오지 않습니다. 일반 구간은 곡률 경로를 따르고, 신호 교차로는 확인된 신호 허가 전까지 정지 위치에서 대기합니다.',
    'signal_observation_stale': 'CAM4 신호등 검출 결과가 들어오지 않거나 지연됐습니다.',
    'signal_camera_uncalibrated': 'Cam4 보정 확인이 완료되지 않아 경로별 신호 허가를 낼 수 없습니다.',
    'signal_pose_unsynchronized': 'CAM4 프레임 시각과 차량 위치 시각을 맞추지 못했습니다.',
    'stopline_pose_unsynchronized': '정지선 프레임 시각과 차량 위치 시각을 맞추지 못했습니다.',
    'route_context_unavailable': '현재 위치에서 경로 신호등 연결을 찾지 못했습니다. 신호등 이벤트가 없는 구간에서는 곡률 경로 주행을 계속합니다.',
    'reference_path_not_received': '곡률 제어 노드가 발행하는 기준 경로를 받지 못했습니다.',
    'signal_localization_unreliable': '차량 위치를 기준 경로에 유효하게 연결하지 못했습니다.',
    'odometry_or_route_unavailable': '유효한 위치·속도 또는 기준 경로가 없습니다.',
    'vehicle_speed_stale_or_unavailable': 'Competition 차량 속도가 없거나 0.5초 이상 지연되어 정지합니다. Competition 상태 UDP 9099 송신과 수신 설정을 확인하세요.',
    'signal_pose_pair_pending': 'CAM4 영상 시각에 맞는 차량 위치를 기다리는 중입니다.',
    'no_projected_targets': '현재 CAM4 화면에 투영되는 경로 신호등이 없습니다. 카메라 설정·차량 위치·정차 여유를 확인하세요.',
    'no_match': 'MGeo 신호등 투영 위치에 맞는 CAM4 검출이 없습니다.',
    'verified_green_carry': '현재 교차로에서 앞서 확인한 초록 신호를 짧게 유지하며 통과 중입니다.',
    'ambiguous_other_head': '검출 상자에 다른 경로의 신호등도 겹쳐 신호 허가를 보류합니다.',
    'conflicting_target_heads': '같은 경로에 연결된 신호등들의 허용 방향이 서로 다릅니다.',
    'matched_route_group': '현재 경로에 연결된 MGeo 신호등과 CAM4 검출이 일치합니다.',
    'matched_correlated_pair': '같은 교차로의 두 신호가 일치하여 신뢰도가 높은 검출을 선택했습니다.',
    'nominal_stale_or_not_type1': '곡률 제어 명령이 없거나 오래됐거나 명령 형식이 다릅니다.',
    'unassociated_visible_signal': '신호등은 검출됐지만 진행 경로의 신호등으로 연결되지 않았습니다.',
    'unmapped_signal_or_stopline': '지도 경로와 연결되지 않은 신호등·정지선이 감지됐습니다.',
    'entry_without_current_permission': '유효한 신호 허가 없이 교차로 진입 경계를 넘었습니다.',
    'next_junction_speed_envelope': '현재 교차로를 통과하면서 다음 정지선에 대비해 속도를 낮춥니다. 다음 신호의 정지·통과 판단은 현재 회전이 끝난 뒤 적용합니다.',
    'next_junction_spacing_unavailable': '두 교차로 사이에 충분한 제동 거리가 없어 안전 정지합니다.',
}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


class Telemetry:
    def __init__(self, clock=time.monotonic):
        self.clock, self.lock = clock, threading.Lock()
        self.values, self.images = {}, {}

    def update(self, key, value, stamp=None):
        with self.lock:
            self.values[key] = (clean(value), self.clock(), stamp)

    def put_image(self, camera, raw, data, stamp, sequence):
        if camera not in ('cam1', 'cam4') or not data or len(data) > 4*1024*1024:
            return
        with self.lock:
            self.images[(camera, bool(raw))] = (bytes(data), self.clock(), float(stamp), int(sequence))

    def image(self, camera, raw):
        with self.lock:
            return self.images.get((camera, bool(raw)), (None,))[0]

    def snapshot(self, ros_now):
        now = self.clock()
        with self.lock:
            values = {k: dict(value=v[0], age_s=round(now-v[1], 3),
                             source_age_s=None if v[2] is None else round(ros_now-v[2], 3))
                      for k, v in self.values.items()}
            images = {c: {('raw' if raw else 'overlay'): dict(
                          sequence=v[3], stamp=v[2], age_s=round(now-v[1], 3),
                          source_age_s=round(ros_now-v[2], 3))
                          for (name, raw), v in self.images.items() if name == c}
                      for c in ('cam1', 'cam4')}
        return dict(values=values, images=images, diagnosis=diagnose(values))


def diagnose(values):
    def fresh(name, limit=1.):
        item = values.get(name)
        if not item or not 0 <= item['age_s'] <= limit:
            return None
        source = item.get('source_age_s')
        return item['value'] if source is None or -.05 <= source <= limit else None
    config, status, command = fresh('config', 3.), fresh('maneuver'), fresh('final', .6)
    codes = []
    if status:
        codes += [s for s in str(status.get('reason', '')).split(',') if s]
        selection = status.get('signal_selection_reason')
        if (status.get('signal_release_latched')
                and selection in ('no_match', 'no_projected_targets')):
            selection = 'verified_green_carry'
        if selection and selection not in codes:
            codes.append(selection)
    notes = [dict(code=c, text=EXPLANATIONS.get(c, c)) for c in codes]
    route_camera = (config is not None
                    and config.get('require_route_signal_context') is False
                    and bool(config.get('signalized_context_ids')))
    sensor_only = (config is not None and not route_camera
                   and config.get('require_route_signal_context') is False
                   and config.get('stopline_requires_detected_signal') is True)
    if route_camera and status and status.get('controller_profile') != 'route_camera':
        notes.append(dict(code='controller_profile_mismatch',
                          text='실행 중인 신호 제어기가 지정 교차로 설정과 다릅니다. 설치한 launch와 ID 목록을 확인하세요.'))
    if sensor_only and status and status.get('controller_profile') != 'sensor_only':
        notes.append(dict(code='controller_profile_mismatch',
                          text='실행 중인 신호 제어기가 센서 전용 설정과 다릅니다. 이전 launch를 종료하고 센서 전용 파일을 설치한 뒤 다시 실행하세요.'))
    camera = fresh('cam4', 3.)
    if camera and camera.get('custom_loaded') is False:
        notes.append(dict(code='signal_model_missing', text='CAM4 신호등 모델 파일이 없어 기본 사물 모델만 실행 중입니다. 모델 경로를 확인하세요.'))
    lane_camera = fresh('cam1', 3.)
    if lane_camera and lane_camera.get('source_size') != lane_camera.get('camera_size'):
        notes.append(dict(code='cam1_size_mismatch', text='CAM1 실제 영상 크기와 차선 카메라 보정 해상도가 다릅니다. MORAI CAM1 설정과 cam_set.json을 대조하세요.'))
    if config is None:
        title, state = '실행 설정 수신 대기', 'waiting'
    elif config.get('control_output_enabled') is False:
        title, state = 'MONITOR — 차량 제어 송신 꺼짐', 'monitor'
    elif config.get('control_output_enabled') is not True:
        title, state = '차량 제어 송신 설정을 확인할 수 없음', 'waiting'
    elif status is None:
        title, state = '신호 제어 상태 미수신 또는 지연 — 노드 실행 확인', 'waiting'
    elif command is None:
        title, state = '최종 차량 명령 미수신 또는 지연', 'waiting'
    elif status.get('mode') == 'SAFE_STOP' or command.get('brake', 0) > 0:
        title, state = '정지·제동 명령 출력 중 — 아래 사유 확인', 'stop'
    elif command.get('accel', 0) > 0:
        title, state = '가속 명령 출력 중', 'command'
        notes.append(dict(code='vehicle_response', text='차량이 계속 정지해 있으면 MORAI 외부 제어 모드·수신 IP/포트·기어를 확인하세요.'))
    else:
        title, state = '가속 명령 없음 — 곡률 목표속도·도착 상태 확인', 'waiting'
    return dict(title=title, state=state, notes=notes)


def create_server(store, html, ros_clock, port=8765, host='127.0.0.1'):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlsplit(self.path)
            if request.path == '/':
                data, mime, code = html, 'text/html; charset=utf-8', 200
            elif request.path == '/api/status':
                data = json.dumps(store.snapshot(ros_clock()), ensure_ascii=False,
                                  allow_nan=False).encode('utf-8')
                mime, code = 'application/json; charset=utf-8', 200
            elif request.path in ('/image/cam1', '/image/cam4'):
                data = store.image(request.path.rsplit('/', 1)[1],
                                   parse_qs(request.query).get('raw') == ['1'])
                mime, code = 'image/jpeg', 200 if data else 503
                data = data or b'Waiting for inference image'
            else:
                data, mime, code = b'Not found', 'text/plain', 404
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass
    return ThreadingHTTPServer((host, port), Handler)
