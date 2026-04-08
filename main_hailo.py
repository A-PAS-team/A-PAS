"""
A-PAS Main System - Raspberry Pi 5 + Hailo-8 AI HAT+
=====================================================
YOLO: Hailo-8 NPU (HEF)
LSTM: RPi5 CPU (ONNX Runtime)
Tracking: Simple centroid tracker (ByteTrack 대체, 경량)
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque, defaultdict

# ==========================================
# ⚙️ [설정]
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH = "/home/pi/A-PAS/test_video.mp4"
# VIDEO_PATH = 0  # USB 카메라
# VIDEO_PATH = "/dev/video0"  # HDMI-to-CSI 캡처보드

# --- 모델 경로 ---
HEF_MODEL_PATH  = "/home/pi/A-PAS/models/yolov8n.hef"
ONNX_MODEL_PATH = "/home/pi/A-PAS/models/best_model_10fps.onnx"

# --- 해상도 ---
IMG_W, IMG_H   = 1920, 1080
YOLO_INPUT_SIZE = 640  # YOLOv8n HEF 입력 크기

# --- 시퀀스 설정 (10FPS) ---
SEQ_LENGTH   = 20
PRED_LENGTH  = 10
INPUT_SIZE   = 17
DT           = 0.1

# --- 샘플링 ---
VIDEO_FPS       = 30
TARGET_FPS      = 10
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # 3

# --- 탐지 설정 ---
CONF_THRESHOLD = 0.3
IOU_THRESHOLD  = 0.45
TARGET_CLASSES = [0, 2, 3, 5, 7]  # Person, Car, Motorcycle, Bus, Truck
CLASS_MAP      = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}

# --- 속도 스무딩 ---
SMOOTH_WINDOW = 5

# --- 충돌 판정 ---
COLLISION_DIST = 80
COLLISION_TTC  = 3

# --- 디스플레이 ---
DISPLAY_WIDTH  = 960   # 포터블 모니터 성능 고려하여 축소 표시
DISPLAY_HEIGHT = 540


# ==========================================
# 🔧 [Hailo 추론 엔진]
# ==========================================
class HailoYOLODetector:
    """Hailo-8 NPU 기반 YOLOv8 추론기"""

    def __init__(self, hef_path, conf_thresh=0.3, iou_thresh=0.45):
        from hailo_platform import (
            HEF, VDevice, ConfigureParams,
            InputVStreamParams, OutputVStreamParams,
            FormatType, HailoStreamInterface
        )
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

        print("📦 Hailo-8 NPU 초기화 중...")

        # HEF 로드
        self.hef = HEF(hef_path)
        self.target = VDevice()

        # 네트워크 구성
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        # 입출력 스트림 정보 추출
        input_vstream_info  = self.hef.get_input_vstream_infos()
        output_vstream_info = self.hef.get_output_vstream_infos()

        self.input_name  = input_vstream_info[0].name
        self.input_shape = input_vstream_info[0].shape  # (H, W, C) 보통 (640, 640, 3)
        self.output_names = [o.name for o in output_vstream_info]

        # VStream 파라미터
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group
        )

        print(f"  ✅ HEF 로드 완료: {hef_path}")
        print(f"  📐 입력: {self.input_shape}")
        print(f"  📤 출력 레이어: {len(self.output_names)}개")

    def preprocess(self, frame):
        """프레임 전처리: 리사이즈 + 패딩 (letterbox)"""
        h, w = frame.shape[:2]
        input_h, input_w = self.input_shape[0], self.input_shape[1]

        # letterbox 리사이즈
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # 패딩
        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        self._scale = scale
        self._pad_w = pad_w
        self._pad_h = pad_h

        return padded

    def postprocess(self, raw_outputs, orig_w, orig_h):
        """
        YOLOv8 HEF 후처리 - 출력 형식에 따라 디코딩
        반환: list of [x_center, y_center, w, h, conf, class_id]
        """
        # HEF 출력은 모델 버전에 따라 다름
        # 일반적인 YOLOv8 HEF: NMS 내장 → (batch, num_detections, 6)
        # 또는 raw anchor 출력 → 직접 디코딩 필요

        detections = []

        # Case 1: NMS 포함 HEF (hailo model zoo 기본)
        # 출력이 여러 레이어로 나뉘는 경우가 많음
        for name, output in raw_outputs.items():
            out = np.array(output)
            if out.ndim == 3:
                # (batch, num_dets, 5+) 또는 (batch, num_dets, 6)
                out = out[0]  # batch 차원 제거
                for det in out:
                    if len(det) >= 6:
                        # [y1, x1, y2, x2, conf, class_id] (Hailo 기본 포맷)
                        y1, x1, y2, x2, conf, cls = det[:6]
                    elif len(det) >= 5:
                        y1, x1, y2, x2, conf = det[:5]
                        cls = 0
                    else:
                        continue

                    if conf < self.conf_thresh:
                        continue

                    # Hailo 출력은 0~1 정규화된 좌표인 경우가 많음
                    if y2 <= 1.0 and x2 <= 1.0:
                        x1 *= orig_w
                        y1 *= orig_h
                        x2 *= orig_w
                        y2 *= orig_h

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w  = x2 - x1
                    h  = y2 - y1

                    detections.append([cx, cy, w, h, float(conf), int(cls)])

            elif out.ndim == 2:
                # (num_dets, 6) 직접
                for det in out:
                    if len(det) >= 6:
                        y1, x1, y2, x2, conf, cls = det[:6]
                    elif len(det) >= 5:
                        y1, x1, y2, x2, conf = det[:5]
                        cls = 0
                    else:
                        continue

                    if conf < self.conf_thresh:
                        continue

                    if y2 <= 1.0 and x2 <= 1.0:
                        x1 *= orig_w
                        y1 *= orig_h
                        x2 *= orig_w
                        y2 *= orig_h

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w  = x2 - x1
                    h  = y2 - y1

                    detections.append([cx, cy, w, h, float(conf), int(cls)])

        return detections

    def detect(self, frame):
        """프레임 입력 → 탐지 결과 반환"""
        from hailo_platform import InferVStreams

        orig_h, orig_w = frame.shape[:2]
        input_data = self.preprocess(frame)

        # 배치 차원 추가
        input_data = np.expand_dims(input_data, axis=0)

        # NPU 추론
        with InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params
        ) as pipeline:
            raw_outputs = pipeline.infer({self.input_name: input_data})

        # 후처리
        detections = self.postprocess(raw_outputs, orig_w, orig_h)

        # 타겟 클래스 필터링
        filtered = [d for d in detections if int(d[5]) in TARGET_CLASSES]

        return filtered


# ==========================================
# 🔧 [간이 트래커] (ByteTrack 대체, 경량)
# ==========================================
class SimpleCentroidTracker:
    """IoU 기반 간이 트래커 - RPi5 CPU 부담 최소화"""

    def __init__(self, max_disappeared=15):
        self.next_id = 0
        self.objects = {}       # {id: [cx, cy, w, h, class_id]}
        self.disappeared = {}   # {id: count}
        self.max_disappeared = max_disappeared

    def update(self, detections):
        """
        detections: list of [cx, cy, w, h, conf, class_id]
        반환: list of (track_id, [cx, cy, w, h], class_id)
        """
        if len(detections) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]
            return []

        if len(self.objects) == 0:
            results = []
            for det in detections:
                cx, cy, w, h, conf, cls = det
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = [cx, cy, w, h, int(cls)]
                self.disappeared[tid] = 0
                results.append((tid, [cx, cy, w, h], int(cls)))
            return results

        # IoU 매칭
        obj_ids = list(self.objects.keys())
        obj_boxes = np.array([[o[0]-o[2]/2, o[1]-o[3]/2,
                               o[0]+o[2]/2, o[1]+o[3]/2]
                              for o in self.objects.values()])
        det_boxes = np.array([[d[0]-d[2]/2, d[1]-d[3]/2,
                               d[0]+d[2]/2, d[1]+d[3]/2]
                              for d in detections])

        iou_matrix = self._compute_iou(obj_boxes, det_boxes)

        matched_obj = set()
        matched_det = set()
        results = []

        # 탐욕적 매칭 (IoU 높은 순)
        while True:
            if iou_matrix.size == 0:
                break
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[max_idx] < 0.2:
                break

            oi, di = max_idx
            tid = obj_ids[oi]
            det = detections[di]
            cx, cy, w, h, conf, cls = det

            self.objects[tid] = [cx, cy, w, h, int(cls)]
            self.disappeared[tid] = 0
            results.append((tid, [cx, cy, w, h], int(cls)))

            matched_obj.add(oi)
            matched_det.add(di)
            iou_matrix[oi, :] = -1
            iou_matrix[:, di] = -1

        # 매칭 안 된 기존 객체 → disappeared++
        for oi, tid in enumerate(obj_ids):
            if oi not in matched_obj:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.objects[tid]
                    del self.disappeared[tid]

        # 매칭 안 된 새 탐지 → 새 ID 부여
        for di, det in enumerate(detections):
            if di not in matched_det:
                cx, cy, w, h, conf, cls = det
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = [cx, cy, w, h, int(cls)]
                self.disappeared[tid] = 0
                results.append((tid, [cx, cy, w, h], int(cls)))

        return results

    @staticmethod
    def _compute_iou(boxes_a, boxes_b):
        """IoU 행렬 계산"""
        iou = np.zeros((len(boxes_a), len(boxes_b)))
        for i, a in enumerate(boxes_a):
            for j, b in enumerate(boxes_b):
                x1 = max(a[0], b[0])
                y1 = max(a[1], b[1])
                x2 = min(a[2], b[2])
                y2 = min(a[3], b[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = (a[2]-a[0]) * (a[3]-a[1])
                area_b = (b[2]-b[0]) * (b[3]-b[1])
                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0
        return iou


# ==========================================
# 🔧 [피처 계산 + LSTM 추론]
# ==========================================
track_history    = {}
prev_boxes       = {}
predictions      = {}
velocity_history = {}
prev_velocity    = {}

DIAG = np.sqrt(IMG_W**2 + IMG_H**2)


def compute_features(box, prev_box, track_id, dt=DT):
    """17개 피처 중 물리 12개 계산"""
    x, y, w, h = box

    if prev_box is not None:
        raw_vx = (x - prev_box[0]) / dt
        raw_vy = (y - prev_box[1]) / dt
    else:
        raw_vx, raw_vy = 0.0, 0.0

    if track_id not in velocity_history:
        velocity_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
    velocity_history[track_id].append([raw_vx, raw_vy])

    vx = float(np.mean([v[0] for v in velocity_history[track_id]]))
    vy = float(np.mean([v[1] for v in velocity_history[track_id]]))

    if track_id in prev_velocity:
        ax = (vx - prev_velocity[track_id][0]) / dt
        ay = (vy - prev_velocity[track_id][1]) / dt
    else:
        ax, ay = 0.0, 0.0

    prev_velocity[track_id] = [vx, vy]

    speed   = np.sqrt(vx**2 + vy**2)
    heading = np.arctan2(vy, vx)

    return np.array([
        x / IMG_W,   y / IMG_H,
        vx / IMG_W,  vy / IMG_H,
        ax / IMG_W,  ay / IMG_H,
        speed / DIAG,
        np.sin(heading), np.cos(heading),
        w / IMG_W,   h / IMG_H,
        (w * h) / (IMG_W * IMG_H)
    ], dtype=np.float32)


def add_to_history(track_id, box, class_id, prev_box):
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=SEQ_LENGTH)

    physics = compute_features(box, prev_box, track_id)
    one_hot = np.zeros(5, dtype=np.float32)
    if class_id in CLASS_MAP:
        one_hot[CLASS_MAP[class_id]] = 1.0

    track_history[track_id].append(np.concatenate([physics, one_hot]))


def run_lstm(track_id, lstm_session):
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None

    seq = np.array(track_history[track_id], dtype=np.float32)
    seq = seq[np.newaxis, ...]
    output = lstm_session.run(None, {"input": seq})[0]
    pred = output[0].copy()

    pred[:, 0] *= IMG_W
    pred[:, 1] *= IMG_H

    return pred


def calc_ttc(pred_a, pred_b):
    min_dist = float('inf')
    collision_frame = PRED_LENGTH

    for t in range(PRED_LENGTH):
        dist = np.sqrt(
            (pred_a[t, 0] - pred_b[t, 0])**2 +
            (pred_a[t, 1] - pred_b[t, 1])**2
        )
        if dist < min_dist:
            min_dist = dist
        if dist < COLLISION_DIST:
            collision_frame = t
            break

    return min_dist, collision_frame


# ==========================================
# 🎨 [시각화]
# ==========================================
def draw_predictions(frame, pred, color=(0, 255, 255)):
    for t in range(len(pred) - 1):
        pt1 = (int(pred[t, 0]),   int(pred[t, 1]))
        pt2 = (int(pred[t+1, 0]), int(pred[t+1, 1]))
        cv2.line(frame, pt1, pt2, color, 2)
    cv2.circle(frame, (int(pred[-1, 0]), int(pred[-1, 1])), 6, color, -1)


def draw_alert(frame, ttc_sec, min_dist):
    cv2.rectangle(frame, (0, 0), (420, 80), (0, 0, 200), -1)
    cv2.putText(frame, "WARNING  COLLISION RISK",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"TTC: {ttc_sec:.1f}s | Dist: {min_dist:.0f}px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def trigger_alert():
    """GPIO LED + Buzzer 트리거 (플레이스홀더)"""
    # TODO: rpi_ws281x LED strip + GPIO buzzer 연동
    pass


# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    # --- 모델 로드 ---
    print("=" * 50)
    print("🚀 A-PAS System - RPi5 + Hailo-8 AI HAT+")
    print("=" * 50)

    # YOLO (Hailo NPU)
    try:
        detector = HailoYOLODetector(
            HEF_MODEL_PATH,
            conf_thresh=CONF_THRESHOLD,
            iou_thresh=IOU_THRESHOLD
        )
    except ImportError:
        print("⚠️  hailo_platform 미설치! CPU fallback 모드로 전환...")
        print("   pip install hailo_platform 또는 sudo apt install hailo-all")
        print("   지금은 ultralytics CPU 모드로 실행합니다.\n")
        detector = None
    except Exception as e:
        print(f"⚠️  Hailo 초기화 실패: {e}")
        print("   CPU fallback 모드로 전환...\n")
        detector = None

    # CPU fallback (Hailo 안 될 때)
    use_hailo = detector is not None
    if not use_hailo:
        from ultralytics import YOLO
        yolo_cpu = YOLO(YOLO_MODEL_PATH)
        print("  ✅ YOLO (CPU) 로드 완료")

    # LSTM (CPU ONNX)
    print("📦 LSTM 모델 로드 중...")
    lstm_session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    print(f"  ✅ LSTM (ONNX) 로드 완료")

    # 트래커
    tracker = SimpleCentroidTracker(max_disappeared=15)

    # --- 영상 열기 ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상 열 수 없음: {VIDEO_PATH}")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    print(f"\n▶ 입력: {VIDEO_PATH}")
    print(f"  해상도: {int(cap.get(3))}x{int(cap.get(4))} @ {actual_fps:.0f}FPS")
    print(f"  모드: {'Hailo-8 NPU' if use_hailo else 'CPU fallback'}")
    print(f"  LSTM 샘플링: 매 {SAMPLE_INTERVAL}프레임")
    print(f"\n종료: 'q' 키\n")

    # --- 성능 기록 ---
    latency_yolo = []
    latency_lstm = []
    latency_total = []
    frame_count = 0

    try:
        while cap.isOpened():
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            is_sample = (frame_count % SAMPLE_INTERVAL == 0)

            # ① YOLO 탐지
            t_yolo = time.perf_counter()

            if use_hailo:
                detections = detector.detect(frame)
            else:
                results = yolo_cpu.track(
                    frame, persist=True, imgsz=640,
                    classes=TARGET_CLASSES, verbose=False,
                    conf=CONF_THRESHOLD
                )
                detections = []
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    confs = results[0].boxes.conf.cpu().numpy()
                    clses = results[0].boxes.cls.int().cpu().numpy()
                    tids  = results[0].boxes.id.int().cpu().numpy()
                    for box, c, cls, tid in zip(boxes, confs, clses, tids):
                        detections.append([box[0], box[1], box[2], box[3],
                                           float(c), int(cls)])

            yolo_ms = (time.perf_counter() - t_yolo) * 1000
            latency_yolo.append(yolo_ms)

            # ② 트래킹
            if use_hailo:
                tracked = tracker.update(detections)
            else:
                # ultralytics CPU는 자체 트래킹 사용
                tracked = []
                if results[0].boxes.id is not None:
                    for box, tid, cls in zip(boxes, tids, clses):
                        tracked.append((int(tid), list(box), int(cls)))

            # ③ 피처 축적 + LSTM (샘플 프레임만)
            collision_risk = False
            min_dist_global = float('inf')
            ttc_global = PRED_LENGTH

            t_lstm = time.perf_counter()

            for track_id, box, class_id in tracked:
                if is_sample:
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, class_id, prev_box)
                    prev_boxes[track_id] = box

                    pred = run_lstm(track_id, lstm_session)
                    if pred is not None:
                        predictions[track_id] = pred

                # 시각화
                if track_id in predictions:
                    draw_predictions(frame, predictions[track_id])

                cx, cy, w, h = box
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            lstm_ms = (time.perf_counter() - t_lstm) * 1000
            if is_sample:
                latency_lstm.append(lstm_ms)

            # ④ TTC
            if is_sample and len(predictions) >= 2:
                id_list = list(predictions.keys())
                for i in range(len(id_list)):
                    for j in range(i + 1, len(id_list)):
                        min_dist, col_frame = calc_ttc(
                            predictions[id_list[i]],
                            predictions[id_list[j]]
                        )
                        if min_dist < min_dist_global:
                            min_dist_global = min_dist
                            ttc_global = col_frame

                if (min_dist_global < COLLISION_DIST and
                        ttc_global < COLLISION_TTC / DT):
                    collision_risk = True
                    trigger_alert()

            # ⑤ 경고
            if collision_risk:
                draw_alert(frame, ttc_global * DT, min_dist_global)

            # ⑥ 성능 오버레이
            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000
            latency_total.append(total_ms)
            fps = 1000 / total_ms if total_ms > 0 else 0

            mode_tag = "NPU" if use_hailo else "CPU"
            sample_tag = "[S]" if is_sample else "   "
            info = (f"{sample_tag} {mode_tag} | YOLO: {yolo_ms:.0f}ms | "
                    f"Total: {total_ms:.0f}ms | FPS: {fps:.1f}")
            cv2.putText(frame, info, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 포터블 모니터용 리사이즈
            display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow("A-PAS Monitor", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⛔ 사용자 중단")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # --- 최종 통계 ---
        if latency_total:
            yolo_arr  = np.array(latency_yolo)
            total_arr = np.array(latency_total)

            print("\n" + "=" * 50)
            print("📊 A-PAS 성능 통계")
            print("=" * 50)
            print(f"  모드              : {'Hailo-8 NPU' if use_hailo else 'CPU fallback'}")
            print(f"  처리 프레임       : {frame_count}")
            print(f"  LSTM 실행 횟수    : {frame_count // SAMPLE_INTERVAL}")
            print(f"  --- YOLO ---")
            print(f"  평균              : {yolo_arr.mean():.1f}ms")
            print(f"  최소/최대         : {yolo_arr.min():.1f} / {yolo_arr.max():.1f}ms")
            print(f"  YOLO FPS          : {1000/yolo_arr.mean():.1f}")
            if latency_lstm:
                lstm_arr = np.array(latency_lstm)
                print(f"  --- LSTM ---")
                print(f"  평균              : {lstm_arr.mean():.1f}ms")
            print(f"  --- 전체 ---")
            print(f"  평균 Latency      : {total_arr.mean():.1f}ms")
            print(f"  평균 FPS          : {1000/total_arr.mean():.1f}")
            print("=" * 50)


if __name__ == "__main__":
    main()
