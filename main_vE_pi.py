"""
A-PAS Main System - Version E Minimal bbox branch
================================================
main_vE_minimal.py copy for bbox-ratio normalized front danger zone.

Only the vehicle front danger rectangle sizing is changed in this branch.
The base file is left untouched.
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import math
import supervision as sv
from collections import deque, defaultdict, Counter

# --- mc_inference import ---
sys.path.insert(0, "ai_model/trajectory/mcdropout")
from mc_inference import run_lstm_with_uncertainty


# ==========================================
# ⚙️ [설정]
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH      = "test_video1.mp4"
HEF_MODEL_PATH  = "best_v7.hef"
ONNX_MODEL_PATH = "ai_model/trajectory/mcdropout/residual_mcd_10fps_pred1p5s.onnx"  # changed for 10fps obs1s pred1.5s

# --- 해상도 ---
IMG_W, IMG_H = 1920, 1080
DIAG         = float(np.sqrt(IMG_W**2 + IMG_H**2))

# --- 모델 설정 ---
SEQ_LENGTH   = 10  # changed for 10fps obs1s pred1.5s
PRED_LENGTH  = 15  # changed for 10fps obs1s pred1.5s
INPUT_SIZE   = 17
DT           = 0.1

# --- 영상 → 모델 샘플링 ---
VIDEO_FPS       = 30
TARGET_FPS      = 10
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # = 3

# --- 속도 스무딩 ---
SMOOTH_WINDOW = 3

# --- 물리적 클리핑 ---
VEL_CLIP = 2.0
ACC_CLIP = 1.0

# --- 충돌 판정 ---
COLLISION_DIST = 45.0  # v8: tighten collision dist
COLLISION_TTC  = 1.5  # changed for 10fps obs1s pred1.5s
TRAJECTORY_RISK_TTC = 0.8  # minimal_v1: 먼 미래 예측은 위험/주의 모두 표시하지 않음
SIGMA_SCALE    = 3.0  # v5: use actual position for clamp, reduce sigma inflation
# --- 동적 임계값 비율 (차량 bbox 대각선 기준) ---
BBOX_DANGER_RATIO  = 0.20   # danger: 차량 대각선의 20%
BBOX_MIN_DANGER    = 20.0   # 최소 danger 거리 (너무 작은 객체 보호)
MOTO_MIN_DIAG = 130.0  # 오토바이(class 2)는 최소 이 대각선으로 취급 (차량급 위험 범위)
FRONT_DANGER_LENGTH_MULT = 4.5  # 전방 danger만 옆폭은 유지하고 길이만 4.5배 연장
FRONT_DANGER_REF_DIAG = 170.0  # 일반적인 차량 bbox 대각선 기준값
_NORMAL_SIDE = max(BBOX_MIN_DANGER, FRONT_DANGER_REF_DIAG * BBOX_DANGER_RATIO)
FRONT_DANGER_SLOW_SCALE = 0.50  # 베이스라인 살짝 상향 (was 0.43)
FRONT_DANGER_SLOW_SPEED = 60.0  # 이전 80과 40의 중간값
TURN_LATERAL_BOOST = 1.4  # 회전 감지 시 front_side 40% 확장
FRONT_DANGER_NORMAL_SPEED = 160.0
FRONT_DANGER_SIZE_SCALE_MIN = 0.75  # bbox 비율 정규화: 작은 차량/원거리 과축소 방지
FRONT_DANGER_SIZE_SCALE_MAX = 1.6   # close camera 압축 완화 (was 1.25)
FRONT_DANGER_SIDE_MIN = 20.0
FRONT_DANGER_SIDE_MAX = 78.0  # close camera 여유 (was 65)
FRONT_DANGER_LENGTH_MIN = 42.0  # 이전 60과 25의 중간값
FRONT_DANGER_LENGTH_MAX = 240.0  # close camera 여유 (was 180)
FRONT_DANGER_CLASS_SCALE = {1: 1.0, 2: 0.9, 3: 1.15}
FRONT_CAUTION_SIDE_MULT = 1.4
FRONT_CAUTION_LENGTH_MULT = 1.4
FRONT_CAUTION_SIDE_MAX = 110.0   # danger 78 x 1.4 (was 75)
FRONT_CAUTION_LENGTH_MAX = 280.0  # danger 240 x ~1.17 비례 (was 200)
HEADING_LONG_WINDOW = 5      # 직선 주행 시 사용할 velocity 평균 윈도우 (sample)
HEADING_COHERENCE_THRESH = 0.92  # vector coherence 임계값 (1에 가까울수록 직선)

# --- 횡단보도 ROI ---
# 사용 예시 (1920x1080 해상도 기준, 좌상단 원점):
# CROSSWALK_ROIS = [
#     [(19, 681), (525, 343), (565, 358), (123, 684)],
#     [(633, 339), (1077, 482), (1078, 442), (670, 320)],
# ]
# 빈 리스트면 ROI 필터 비활성화 → 모든 보행자 평가
CROSSWALK_ROIS_BY_VIDEO = {
    1: [
        [(222, 598), (608, 460), (634, 490), (222, 664)],
        [(688, 432), (914, 334), (952, 364), (726, 468)],
        [(1138, 370), (1194, 348), (1710, 592), (1656, 634)],
    ],
    2: [
        [(6, 670), (168, 508), (212, 522), (62, 684)],
        [(234, 438), (374, 302), (422, 310), (288, 454)],
        [(542, 278), (564, 252), (876, 314), (866, 342)],
    ],
    3: [
        [(2, 486), (2, 542), (252, 378), (212, 360)],
        [(288, 312), (422, 222), (464, 238), (338, 332)],
        [(586, 214), (616, 192), (928, 250), (928, 274)],
    ],
    4: [
        [(110, 644), (292, 424), (352, 436), (164, 672)],
        [(346, 356), (434, 252), (490, 260), (392, 370)],
        [(582, 220), (602, 192), (886, 226), (882, 264)],
    ],
    5: [
        [(26, 680), (302, 500), (338, 520), (110, 686)],
        [(378, 450), (516, 354), (548, 368), (412, 472)],
        [(670, 360), (700, 336), (1024, 430), (1024, 462)],
    ],
}


def select_crosswalk_rois(video_path):
    """VIDEO_PATH의 test_video 번호에 맞는 ROI를 선택한다."""
    video_name = str(video_path).lower()
    for video_idx, rois in CROSSWALK_ROIS_BY_VIDEO.items():
        if f"test_video{video_idx}" in video_name:
            return rois
    return []


CROSSWALK_ROIS = select_crosswalk_rois(VIDEO_PATH)
ROI_MARGIN = 20

# --- YOLO 설정 ---
YOLO_IMGSZ   = 1280
USE_YOLO_EVERY_FRAME = False

# --- MC Dropout 설정 ---
MC_N_SAMPLES         = "adaptive"
MC_BASE_SAMPLES      = 4
MC_ADAPTIVE_SAMPLES  = 4
MC_USE_BATCH         = True
ADAPTIVE_REFINE_EVERY = 4
MAX_ADAPTIVE_PAIRS   = 1

# --- 예측 시각화 안정화 ---
EMA_ALPHA       = 0.85  # v12: faster EMA + behind caution suppress
EMA_ALPHA_STD   = 0.85  # v12: faster EMA + behind caution suppress
WARMUP_HIDE     = 5
# --- 런타임 캡 ---
MAX_LSTM_TRACKS = 10
MAX_TTC_PAIRS   = 12  # ROI 통과 pair 모두 커버 (성능 영향 ~0.3ms/sample, 무시 수준)
TTC_PAIR_CURRENT_DIST = 500.0  # v3: stopped vehicle EMA bypass
STOPPED_VEL_THRESH = 40.0  # v6: fix stopped detection threshold

# --- 시각화 설정 ---
DISPLAY_SCALE    = 0.75
PRED_DRAW_STRIDE = 3
MAX_VIS_TRACKS   = 24

# --- 통계 ---
STATS_WARMUP_FRAMES = 3
DANGER_VISUAL_HOLD = 10  # samples (= 1초 at 10 FPS sample rate). 빨강 한 번 뜨면 유지.

# --- Pi 배포용 플래그 ---
ENABLE_TRACK_MERGE = False  # True면 가까운 트랙 ID 머지 (보행자 누락 위험)
DEBUG_DRAW_DANGER_ZONE = False  # True면 danger/caution zone 시각화

# --- 클래스 매핑 ---
VEHICLE_CLASSES = {1, 2, 3}
CONF_THRESHOLD  = {0: 0.5, 1: 0.6, 2: 0.45, 3: 0.35}  # 차량/오토바이 false positive 감소 (나뭇잎/그림자 거름)
CLASS_MAP       = {0: 0, 1: 1, 2: 2, 3: 3}
CLASS_NAMES     = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}
CLASS_COLORS    = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}
COCO_TO_APAS_CLASS = {
    0: 0,  # person
    2: 1,  # car
    3: 2,  # motorcycle
    5: 3,  # bus -> large vehicle
    7: 3,  # truck -> large vehicle
}

COLORS = {
    "safe":    (0, 220, 0),
    "caution": (0, 140, 255),   # BGR: clear orange / RGB(255,140,0)
    "danger":  (0, 0, 255),
    "text":    (230, 230, 230),
}


# ==========================================
# 📊 [Latency Profiler]
class HailoYOLODetector:
    """Hailo HEF YOLO detector. 출력은 기존 파이프라인의 xywh 형식으로 맞춘다."""

    def __init__(self, hef_path, conf_thresh=0.25):
        from hailo_platform import (
            ConfigureParams,
            FormatType,
            HEF,
            HailoStreamInterface,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )

        self.conf_thresh = conf_thresh
        self.network_group_handle = None
        self.pipeline = None

        print("Hailo-8 NPU 초기화 중...")
        self.hef = HEF(hef_path)
        self.target = VDevice()
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        self.input_vstream_params = InputVStreamParams.make(
            self.network_group, format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(self.network_group)
        print(f"  HEF 및 NPU 준비 완료: {hef_path}")
        print(f"  입력 shape: {self.input_shape}")

    def __enter__(self):
        from hailo_platform import InferVStreams

        print("  Hailo network group 활성화")
        self.network_group_handle = self.network_group.activate(self.network_group_params)
        self.network_group_handle.__enter__()
        self.pipeline = InferVStreams(
            self.network_group,
            self.input_vstream_params,
            self.output_vstream_params,
        )
        self.pipeline.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pipeline is not None:
            self.pipeline.__exit__(exc_type, exc, tb)
            self.pipeline = None
        if self.network_group_handle is not None:
            self.network_group_handle.__exit__(exc_type, exc, tb)
            self.network_group_handle = None

    def preprocess(self, frame):
        """원본 비율을 유지해 HEF 입력 크기로 letterbox 전처리."""
        h, w = frame.shape[:2]
        input_h, input_w = self.input_shape[0], self.input_shape[1]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_w = (input_w - new_w) // 2
        pad_h = (input_h - new_h) // 2

        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        self._scale = scale
        self._pad_w = pad_w
        self._pad_h = pad_h
        return padded

    def _map_class_id(self, raw_cls, class_count=None):
        raw_cls = int(raw_cls)
        if class_count == 80:
            return COCO_TO_APAS_CLASS.get(raw_cls)
        if class_count == 4:
            return raw_cls if raw_cls in CLASS_MAP else None
        if raw_cls in CLASS_MAP:
            return raw_cls
        return COCO_TO_APAS_CLASS.get(raw_cls)

    def _append_detection(self, detections, y1, x1, y2, x2, conf, raw_cls, orig_w, orig_h, class_count=None):
        class_id = self._map_class_id(raw_cls, class_count=class_count)
        if class_id is None:
            return

        conf = float(conf)
        if conf < self.conf_thresh:
            return

        y1, x1, y2, x2 = float(y1), float(x1), float(y2), float(x2)
        if max(abs(y1), abs(x1), abs(y2), abs(x2)) <= 1.5:
            input_h, input_w = self.input_shape[0], self.input_shape[1]
            x1 = (x1 * input_w - self._pad_w) / self._scale
            x2 = (x2 * input_w - self._pad_w) / self._scale
            y1 = (y1 * input_h - self._pad_h) / self._scale
            y2 = (y2 * input_h - self._pad_h) / self._scale

        x1 = float(np.clip(x1, 0, orig_w - 1))
        x2 = float(np.clip(x2, 0, orig_w - 1))
        y1 = float(np.clip(y1, 0, orig_h - 1))
        y2 = float(np.clip(y2, 0, orig_h - 1))

        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            return
        detections.append([(x1 + x2) / 2.0, (y1 + y2) / 2.0, w, h, conf, class_id])

    def _parse_class_major_nms(self, output, orig_w, orig_h):
        """Hailo model-zoo NMS: output[batch][class] = [y1,x1,y2,x2,score]."""
        detections = []
        batches = output
        if isinstance(batches, np.ndarray) and batches.dtype != object:
            return None
        if isinstance(batches, np.ndarray):
            batches = batches.tolist()
        if not isinstance(batches, (list, tuple)) or not batches:
            return None

        first_batch = batches[0]
        if isinstance(first_batch, np.ndarray):
            first_batch = first_batch.tolist()
        if not isinstance(first_batch, (list, tuple)) or len(first_batch) not in (4, 80):
            return None

        class_count = len(first_batch)
        for raw_cls, class_dets in enumerate(first_batch):
            if class_dets is None:
                continue
            class_arr = np.asarray(class_dets, dtype=np.float32)
            if class_arr.size == 0:
                continue
            class_arr = class_arr.reshape(-1, class_arr.shape[-1])
            for det in class_arr:
                if len(det) >= 5:
                    self._append_detection(
                        detections, det[0], det[1], det[2], det[3], det[4],
                        raw_cls, orig_w, orig_h, class_count=class_count
                    )
        return detections

    def _parse_flat_nms(self, output, orig_w, orig_h):
        """Flat NMS: rows of [y1,x1,y2,x2,score,class]."""
        detections = []
        try:
            out = np.asarray(output, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if out.size == 0:
            return detections
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2:
            return None

        for det in out:
            if len(det) >= 6:
                self._append_detection(
                    detections, det[0], det[1], det[2], det[3], det[4],
                    det[5], orig_w, orig_h
                )
        return detections

    def postprocess(self, raw_outputs, orig_w, orig_h):
        detections = []
        for output in raw_outputs.values():
            parsed = self._parse_class_major_nms(output, orig_w, orig_h)
            if parsed is None:
                parsed = self._parse_flat_nms(output, orig_w, orig_h)
            if parsed:
                detections.extend(parsed)
        return detections

    def detect(self, frame):
        if self.pipeline is None:
            raise RuntimeError("Hailo detector는 활성화된 context 안에서만 사용할 수 있습니다.")
        orig_h, orig_w = frame.shape[:2]
        input_data = np.expand_dims(self.preprocess(frame), axis=0)
        raw_outputs = self.pipeline.infer({self.input_name: input_data})
        return self.postprocess(raw_outputs, orig_w, orig_h)


# ==========================================
# 📊 [Latency Profiler]

# ==========================================
STAGES = [
    "yolo_inference",
    "detection_filter",
    "tracker",
    "feature_lstm_mc",
    "ttc",
    "visualization",
    "imshow_waitkey",
]


class StageTimer:
    def __init__(self):
        self._last = time.perf_counter()
        self.times = {name: 0.0 for name in STAGES}

    def mark(self, stage):
        now = time.perf_counter()
        self.times[stage] += (now - self._last) * 1000.0
        self._last = now


class LatencyProfiler:
    def __init__(self):
        self.records = []

    def add(self, record):
        self.records.append(record)

    def print_report(self, skip_first=STATS_WARMUP_FRAMES):
        recs = [r for r in self.records if r["frame"] > skip_first]
        if not recs:
            recs = self.records
        if not recs:
            return

        print(f"\n{'='*72}")
        print("Latency breakdown by frame type (ms)")
        print(f"{'='*72}")

        for frame_type in ("sample", "reuse"):
            subset = [r for r in recs if r["type"] == frame_type]
            if not subset:
                continue
            print(f"\n[{frame_type.upper()}] frames={len(subset)}")
            for stage in ["total", *STAGES]:
                values = [r["total_ms"] if stage == "total"
                          else r["stages"].get(stage, 0.0) for r in subset]
                arr = np.array(values, dtype=np.float32)
                print(f"  {stage:<22} mean={arr.mean():7.2f}  "
                      f"p50={np.percentile(arr,50):7.2f}  "
                      f"p90={np.percentile(arr,90):7.2f}  "
                      f"max={arr.max():7.2f}")

        slow = sorted(recs, key=lambda r: r["total_ms"], reverse=True)[:5]
        print("\nTop slow frames:")
        for r in slow:
            parts = ", ".join(f"{k}={v:.1f}" for k, v in r["stages"].items() if v > 0.05)
            print(f"  frame={r['frame']:>5} type={r['type']:<6} "
                  f"total={r['total_ms']:.2f}ms "
                  f"tracks={r['tracks']} preds={r['preds']} "
                  f"mc={r['mc_samples']} | {parts}")
        print(f"{'='*72}")


# ==========================================
# 📊 [트래킹 상태]
# ==========================================
track_history        = {}
prev_boxes           = {}
predictions          = {}
predictions_std      = {}
pred_frame_count     = {}
velocity_history     = {}
heading_vel_history  = {}  # heading 추정 전용 (maxlen=HEADING_LONG_WINDOW). LSTM 입력에 영향 없음.
prev_velocity        = {}
track_classes        = {}
class_votes          = defaultdict(lambda: deque(maxlen=10))
prev_frames          = {}
last_adaptive_refine = {}
danger_visual_cache  = {}  # (ped_id, veh_id) -> samples remaining


# ==========================================
# 🔧 [피처 추출]
# ==========================================
def compute_features(box, prev_box, track_id):
    x, y, w, h = box
    if track_id not in velocity_history:
        velocity_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
    if track_id not in heading_vel_history:
        heading_vel_history[track_id] = deque(maxlen=HEADING_LONG_WINDOW)

    if prev_box is not None:
        raw_vx = (x - prev_box[0]) / DT
        raw_vy = (y - prev_box[1]) / DT
        velocity_history[track_id].append([raw_vx, raw_vy])
        # heading 전용 history에도 동시 append (LSTM 영역 영향 없음)
        heading_vel_history[track_id].append([raw_vx, raw_vy])
    else:
        # 첫 프레임은 prev_box가 없어 실제 velocity를 알 수 없음.
        # (0,0)을 history에 넣으면 SMOOTH_WINDOW 동안 평균이 오염되어
        # heading 추정이 지연되므로, history에는 추가하지 않음.
        # LSTM feature로는 (0,0)을 그대로 사용.
        raw_vx, raw_vy = 0.0, 0.0

    hist = velocity_history[track_id]
    n = len(hist)
    if n > 0:
        vx = sum(v[0] for v in hist) / n
        vy = sum(v[1] for v in hist) / n
    else:
        vx, vy = 0.0, 0.0

    if track_id in prev_velocity:
        ax = (vx - prev_velocity[track_id][0]) / DT
        ay = (vy - prev_velocity[track_id][1]) / DT
    else:
        ax, ay = 0.0, 0.0
    prev_velocity[track_id] = [vx, vy]

    speed = math.sqrt(vx * vx + vy * vy)
    heading = math.atan2(vy, vx)

    vx_clip = max(-VEL_CLIP, min(VEL_CLIP, vx / IMG_W))
    vy_clip = max(-VEL_CLIP, min(VEL_CLIP, vy / IMG_H))
    ax_clip = max(-ACC_CLIP, min(ACC_CLIP, ax / IMG_W))
    ay_clip = max(-ACC_CLIP, min(ACC_CLIP, ay / IMG_H))

    return np.array([
        x / IMG_W, y / IMG_H,
        vx_clip, vy_clip,
        ax_clip, ay_clip,
        speed / DIAG,
        math.sin(heading), math.cos(heading),
        w / IMG_W, h / IMG_H,
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
    track_classes[track_id] = class_id


# ==========================================
# 🧠 [LSTM 추론]
# ==========================================
def mc_samples_for_base():
    if MC_N_SAMPLES == "adaptive":
        return MC_BASE_SAMPLES
    return int(MC_N_SAMPLES)


def run_lstm_mc(session, track_id, n_samples=None):
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None, None, 0

    if n_samples is None:
        n_samples = mc_samples_for_base()

    seq = np.array(track_history[track_id], dtype=np.float32)
    seq = seq[np.newaxis, ...]

    mean, std = run_lstm_with_uncertainty(
        session, seq, n_samples=n_samples, use_batch=MC_USE_BATCH
    )

    mean_px = mean.copy()
    std_px  = std.copy()
    mean_px[..., 0] *= IMG_W
    mean_px[..., 1] *= IMG_H
    std_px[..., 0]  *= IMG_W
    std_px[..., 1]  *= IMG_H

    vel_hist = velocity_history.get(track_id)  # v4: clamp stopped vehicle, depart caution suppress, ghost cleanup
    if track_classes.get(track_id) in VEHICLE_CLASSES and vel_hist and len(vel_hist) > 0:
        recent_vel = np.array(vel_hist, dtype=np.float32).mean(axis=0)  # v6: fix stopped detection threshold
        if math.hypot(float(recent_vel[0]), float(recent_vel[1])) < STOPPED_VEL_THRESH:
            current_box = prev_boxes.get(track_id)  # v5: use actual position for clamp, reduce sigma inflation
            if current_box is not None:
                current_pos = np.array([[current_box[0], current_box[1]]], dtype=np.float32)
                mean_px[:] = current_pos
            std_px[:] = np.minimum(std_px, 1.0)

    return mean_px, std_px, n_samples


def update_prediction(track_id, mean_new, std_new):
    if track_id not in predictions:
        predictions[track_id]     = mean_new.copy()
        predictions_std[track_id] = std_new.copy()
        pred_frame_count[track_id] = 1
        return

    vel_hist = velocity_history.get(track_id)  # v3: stopped vehicle EMA bypass
    if track_classes.get(track_id) in VEHICLE_CLASSES and vel_hist and len(vel_hist) > 0:
        recent_vel = np.array(vel_hist, dtype=np.float32).mean(axis=0)  # v6: fix stopped detection threshold
        if math.hypot(float(recent_vel[0]), float(recent_vel[1])) < STOPPED_VEL_THRESH:
            predictions[track_id] = mean_new.copy()
            predictions_std[track_id] = std_new.copy()
            pred_frame_count[track_id] += 1
            return

    predictions[track_id] = EMA_ALPHA * mean_new + (1 - EMA_ALPHA) * predictions[track_id]  # v13: remove shift_blend, forward 140deg danger cone
    predictions_std[track_id] = EMA_ALPHA_STD * std_new + (1 - EMA_ALPHA_STD) * predictions_std[track_id]  # v13: remove shift_blend, forward 140deg danger cone
    pred_frame_count[track_id] += 1


def select_tracks_for_prediction(boxes, track_ids, class_ids, confs):
    """ByteTrack 결과에서 safety-relevant 트랙만 선택"""
    items = list(zip(boxes, track_ids, class_ids, confs))
    persons = [(b, t, c, cf) for b, t, c, cf in items if c == 0]
    vehicles = [(b, t, c, cf) for b, t, c, cf in items if c in VEHICLE_CLASSES]

    selected = {}
    for b, t, c, cf in sorted(persons, key=lambda x: x[3], reverse=True):
        selected[t] = (b, t, c, cf)
        if len(selected) >= MAX_LSTM_TRACKS:
            return list(selected.values())

    if persons and vehicles:
        def dist_to_nearest_ped(veh):
            return min((veh[0][0]-p[0][0])**2 + (veh[0][1]-p[0][1])**2 for p in persons)
        ranked_vehicles = sorted(vehicles, key=dist_to_nearest_ped)
    else:
        ranked_vehicles = sorted(vehicles, key=lambda x: x[3], reverse=True)

    for b, t, c, cf in ranked_vehicles:
        selected[t] = (b, t, c, cf)
        if len(selected) >= MAX_LSTM_TRACKS:
            break

    return list(selected.values())


_roi_mask = None
def _build_roi_mask():
    global _roi_mask
    if not CROSSWALK_ROIS:
        _roi_mask = None
        return
    _roi_mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for polygon in CROSSWALK_ROIS:
        if len(polygon) < 3:
            continue
        pts = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(_roi_mask, [pts], 255)
    if ROI_MARGIN > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_MARGIN*2+1, ROI_MARGIN*2+1))
        _roi_mask = cv2.dilate(_roi_mask, kernel)


def is_point_in_roi_fast(x, y):
    if _roi_mask is None:
        return True
    ix, iy = int(round(x)), int(round(y))
    if 0 <= ix < IMG_W and 0 <= iy < IMG_H:
        return _roi_mask[iy, ix] > 0
    return False


def _veh_diag(box):
    """차량 bbox의 대각선 길이."""
    return math.hypot(float(box[2]), float(box[3]))


def _front_danger_shape(v_speed, veh_diag, class_id):
    """전방 danger 사각형 크기: 차량 bbox 대각선 비율로 정규화."""
    if class_id == 2:
        veh_diag = max(veh_diag, MOTO_MIN_DIAG)

    class_scale = FRONT_DANGER_CLASS_SCALE.get(class_id, 1.0)
    size_scale = (veh_diag / max(FRONT_DANGER_REF_DIAG, 1e-6)) * class_scale
    size_scale = max(FRONT_DANGER_SIZE_SCALE_MIN, min(FRONT_DANGER_SIZE_SCALE_MAX, size_scale))

    side = _NORMAL_SIDE * size_scale
    side = max(FRONT_DANGER_SIDE_MIN, min(FRONT_DANGER_SIDE_MAX, side))

    if v_speed <= FRONT_DANGER_SLOW_SPEED:
        speed_scale = FRONT_DANGER_SLOW_SCALE
    elif v_speed >= FRONT_DANGER_NORMAL_SPEED:
        speed_scale = 1.0
    else:
        ratio = (v_speed - FRONT_DANGER_SLOW_SPEED) / max(1e-6, FRONT_DANGER_NORMAL_SPEED - FRONT_DANGER_SLOW_SPEED)
        speed_scale = FRONT_DANGER_SLOW_SCALE + ratio * (1.0 - FRONT_DANGER_SLOW_SCALE)

    front_length = side * FRONT_DANGER_LENGTH_MULT * speed_scale
    front_length = max(FRONT_DANGER_LENGTH_MIN, min(FRONT_DANGER_LENGTH_MAX, front_length))
    return side, front_length


def _front_danger_hit(v_box, p_box, hx, hy, side_radius, front_length, ped_radius):
    """전방 사각형 안 + ped_radius 마진 판정.

    Args:
        v_box, p_box: 차량/보행자 bbox (cx, cy, w, h)
        hx, hy: 차량 heading 단위 벡터
        side_radius: 사각형 옆폭
        front_length: 사각형 전방 길이
        ped_radius: 보행자 bbox 반경 (가장자리 마진용)

    Returns: True if 보행자 bbox가 사각형에 닿음 (가장자리 포함).
    """
    if side_radius <= 0.0 or front_length <= 0.0:
        return False

    to_ped_x = float(p_box[0] - v_box[0])
    to_ped_y = float(p_box[1] - v_box[1])
    forward = to_ped_x * hx + to_ped_y * hy

    # forward 범위 + ped_radius 마진
    if forward < -ped_radius or forward > front_length + ped_radius:
        return False

    # lateral 거리 + ped_radius 마진
    lateral_x = to_ped_x - forward * hx
    lateral_y = to_ped_y - forward * hy
    lateral = math.hypot(lateral_x, lateral_y)
    return lateral <= side_radius + ped_radius


def get_trajectory(track_id, current_box):
    """track_id의 미래 PRED_LENGTH 스텝 궤적을 반환.
    LSTM 예측 사용 가능하면 사용, 아니면 velocity_history로 직선 외삽.
    Returns: (mean_array, std_array) — shape (PRED_LENGTH, 2) and (PRED_LENGTH,)
             각 step의 std는 px 단위 단일 스칼라 (x/y 평균)
    """
    # 1. LSTM 예측이 준비되었으면 그대로 사용
    if track_id in predictions and pred_frame_count.get(track_id, 0) >= 1:
        mean = predictions[track_id]
        std_xy = predictions_std[track_id]
        # std는 (PRED_LENGTH, 2) → 스칼라 (x/y 평균)로 변환
        std = np.mean(std_xy, axis=1)
        return mean, std

    # 2. Kinematic fallback: velocity_history로 직선 외삽
    if current_box is None:
        return None, None

    vel_hist = velocity_history.get(track_id)
    if not vel_hist or len(vel_hist) == 0:
        # velocity 없음 → 정지 가정
        vx, vy = 0.0, 0.0
    else:
        n = len(vel_hist)
        vx = sum(v[0] for v in vel_hist) / n
        vy = sum(v[1] for v in vel_hist) / n

    mean = np.zeros((PRED_LENGTH, 2), dtype=np.float32)
    std = np.zeros(PRED_LENGTH, dtype=np.float32)
    for t in range(PRED_LENGTH):
        dt = (t + 1) * DT
        mean[t, 0] = current_box[0] + vx * dt
        mean[t, 1] = current_box[1] + vy * dt
        # 시간 비례 불확실성을 80px로 cap -> late horizon 폭주 방지
        # 50px보다 보수적으로 시작해서 빠른 객체 조기 감지를 덜 죽임
        std[t] = min(15.0 + t * 4.0, 70.0)

    return mean, std


def assess_risk(p_id, p_box, v_id, v_box):
    """?? (???, ??) pair ?? ?? ? ?? ?? ??.

    ?? ??: ??? ???? ?? ?? ?? ????,
              ?? ?? ??? ????? ??.

    ?? ??:
      1. ROI filter (???? ???? ?/??)
      2. ?? ?? (?? ??? ?? ??)
      3. forward/lateral ?? (???? ?? ?? ?? ?? + ?? ???)
      4. get_trajectory()? ?? ?? + uncertainty
      5. ?? ?? ?? < safety_margin ? danger / caution

    Returns:
        dict: {ped_id, veh_id, min_dist, ttc_sec, danger, caution,
               point, std_radius, source} or None
    """
    # === 1. ROI filter ===
    if not is_point_in_roi_fast(p_box[0], p_box[1]):
        return None

    # === 2. velocity + heading ===
    veh_vel_hist = velocity_history.get(v_id)
    if not veh_vel_hist or len(veh_vel_hist) == 0:
        return None
    n = len(veh_vel_hist)
    vx = sum(v[0] for v in veh_vel_hist) / n
    vy = sum(v[1] for v in veh_vel_hist) / n
    v_speed = math.hypot(vx, vy)
    if v_speed < STOPPED_VEL_THRESH:
        return None

    # === 2.5. 회전 감지 (heading_vel_history maxlen=5 활용) ===
    is_turning = False
    turn_hx, turn_hy = 0.0, 0.0
    heading_hist = heading_vel_history.get(v_id)
    if heading_hist and len(heading_hist) >= 4:
        hist_list = list(heading_hist)
        rec = hist_list[-2:]
        old = hist_list[:-2]
        rx = (rec[0][0] + rec[1][0]) * 0.5
        ry = (rec[0][1] + rec[1][1]) * 0.5
        ox = sum(v[0] for v in old) / len(old)
        oy = sum(v[1] for v in old) / len(old)
        rs = math.hypot(rx, ry)
        os_ = math.hypot(ox, oy)
        if rs > STOPPED_VEL_THRESH and os_ > STOPPED_VEL_THRESH:
            cos_change = (rx * ox + ry * oy) / (rs * os_)
            if cos_change < HEADING_COHERENCE_THRESH:
                is_turning = True
                turn_hx = rx / rs
                turn_hy = ry / rs

    # === 3. forward 계산 (회전 시 recent heading 사용) ===
    if is_turning:
        hx, hy = turn_hx, turn_hy
    else:
        hx, hy = vx / v_speed, vy / v_speed
    dx = p_box[0] - v_box[0]
    dy = p_box[1] - v_box[1]
    forward = dx * hx + dy * hy

    if forward < 0.0:
        return None

    veh_d = _veh_diag(v_box)

    # === 4. Zone check (?? ?? ??? + ped_radius ??) ===
    front_side, front_length = _front_danger_shape(v_speed, veh_d, track_classes.get(v_id))
    if is_turning:
        front_side = min(front_side * TURN_LATERAL_BOOST, FRONT_DANGER_SIDE_MAX)

    ped_radius = max(float(p_box[2]), float(p_box[3])) * 0.5
    if _front_danger_hit(v_box, p_box, hx, hy, front_side, front_length, ped_radius):
        ttc_est = forward / max(v_speed, 1.0)
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(max(0.0, forward - front_length)) if forward > front_length else 0.0,
            "ttc_sec": float(min(ttc_est, COLLISION_TTC)),
            "danger": True,
            "caution": True,
            "point": np.array(
                [(p_box[0] + v_box[0]) * 0.5, (p_box[1] + v_box[1]) * 0.5],
                dtype=np.float32
            ),
            "std_radius": 10.0,
            "source": "zone",
        }

    caution_side = min(front_side * FRONT_CAUTION_SIDE_MULT, FRONT_CAUTION_SIDE_MAX)
    caution_length = min(front_length * FRONT_CAUTION_LENGTH_MULT, FRONT_CAUTION_LENGTH_MAX)
    if _front_danger_hit(v_box, p_box, hx, hy, caution_side, caution_length, ped_radius):
        ttc_est = forward / max(v_speed, 1.0)
        is_imminent = forward <= ped_radius + 30.0
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(max(0.0, forward - caution_length)) if forward > caution_length else 0.0,
            "ttc_sec": float(min(ttc_est, COLLISION_TTC)),
            "danger": is_imminent,
            "caution": True,
            "point": np.array(
                [(p_box[0] + v_box[0]) * 0.5, (p_box[1] + v_box[1]) * 0.5],
                dtype=np.float32
            ),
            "std_radius": 10.0,
            "source": "zone_imminent" if is_imminent else "zone_caution",
        }

    # === 5. ?? ?? ===
    p_traj, p_std = get_trajectory(p_id, p_box)
    v_traj, v_std = get_trajectory(v_id, v_box)
    if p_traj is None or v_traj is None:
        return None

    # === 5. ?? ?? ?? + uncertainty ===
    min_dist = float('inf')
    min_t = 0
    danger_t = -1
    danger_threshold_step = int(COLLISION_TTC / DT)

    for t in range(PRED_LENGTH):
        dist = math.hypot(
            float(p_traj[t][0] - v_traj[t][0]),
            float(p_traj[t][1] - v_traj[t][1])
        )
        margin = COLLISION_DIST + float(p_std[t]) + float(v_std[t])

        if dist < margin and danger_t < 0 and t < danger_threshold_step:
            danger_t = t

        if dist < min_dist:
            min_dist = dist
            min_t = t

    # === 6. ?? ?? ===
    if danger_t >= 0 and danger_t * DT <= TRAJECTORY_RISK_TTC:
        point = np.array(
            ((p_traj[danger_t][0] + v_traj[danger_t][0]) * 0.5,
             (p_traj[danger_t][1] + v_traj[danger_t][1]) * 0.5),
            dtype=np.float32
        )
        std_r = (float(p_std[danger_t]) + float(v_std[danger_t])) * 0.5
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(min_dist),
            "ttc_sec": float(danger_t * DT),
            "danger": True,
            "caution": True,
            "point": point,
            "std_radius": float(std_r),
            "source": "minimal_v1",
        }

    if min_dist < COLLISION_DIST * 1.5 and min_t * DT <= TRAJECTORY_RISK_TTC:
        point = np.array(
            ((p_traj[min_t][0] + v_traj[min_t][0]) * 0.5,
             (p_traj[min_t][1] + v_traj[min_t][1]) * 0.5),
            dtype=np.float32
        )
        std_r = (float(p_std[min_t]) + float(v_std[min_t])) * 0.5
        return {
            "ped_id": p_id,
            "veh_id": v_id,
            "min_dist": float(min_dist),
            "ttc_sec": float(min_t * DT),
            "danger": False,
            "caution": True,
            "point": point,
            "std_radius": float(std_r),
            "source": "minimal_v1",
        }

    return None


def needs_adaptive_refine(pair_result):
    if MC_N_SAMPLES != "adaptive":
        return False
    return (pair_result["danger"]
            or pair_result["caution"]
            or pair_result["min_dist"] <= COLLISION_DIST * 1.8)


def refine_adaptive_pairs(session, risk_results, sample_index):
    if MC_N_SAMPLES != "adaptive":
        return 0

    total_extra = 0
    refined = 0
    for result in risk_results:
        if refined >= MAX_ADAPTIVE_PAIRS:
            break
        if not needs_adaptive_refine(result):
            continue

        did_refine = False
        for tid in [result["ped_id"], result["veh_id"]]:
            last_refine = last_adaptive_refine.get(tid, -9999)
            if sample_index - last_refine < ADAPTIVE_REFINE_EVERY:
                continue
            if tid not in track_history:
                continue

            mean, std, n = run_lstm_mc(session, tid, n_samples=MC_ADAPTIVE_SAMPLES)
            if mean is not None:
                update_prediction(tid, mean, std)
                total_extra += n
                last_adaptive_refine[tid] = sample_index
                did_refine = True

        if did_refine:
            refined += 1

    return total_extra


# ==========================================
# 🎨 [시각화]
# ==========================================
def scaled_pt(pt):
    return int(pt[0] * DISPLAY_SCALE), int(pt[1] * DISPLAY_SCALE)


def draw_bbox(frame, box, track_id, class_id, conf, level="safe"):
    x, y, w, h = box
    x1, y1 = scaled_pt((x - w/2, y - h/2))
    x2, y2 = scaled_pt((x + w/2, y + h/2))
    color = COLORS[level]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} [{CLASS_NAMES.get(class_id, '?')}] {conf:.2f}"
    cv2.putText(frame, label, (x1, max(14, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_prediction(frame, track_id, level="safe"):
    if level == "safe":
        return
    if track_id not in predictions:
        return
    n_pred = pred_frame_count.get(track_id, 0)
    if n_pred < WARMUP_HIDE:
        return

    mean = predictions[track_id]
    color = COLORS[level]
    limit = min(len(mean), PRED_LENGTH)

    for i in range(0, limit - PRED_DRAW_STRIDE, PRED_DRAW_STRIDE):
        p1 = scaled_pt(mean[i])
        p2 = scaled_pt(mean[i + PRED_DRAW_STRIDE])
        thickness = 2 if level == "danger" else 1
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_risk_marker(frame, result):
    color = COLORS["danger"] if result["danger"] else COLORS["caution"]
    center = scaled_pt(result["point"])
    size = 7 if result["danger"] else 5

    cv2.line(frame, (center[0]-size, center[1]), (center[0]+size, center[1]), color, 2, cv2.LINE_AA)
    cv2.line(frame, (center[0], center[1]-size), (center[0], center[1]+size), color, 2, cv2.LINE_AA)

    radius = int(np.clip(result.get("std_radius", 10) * DISPLAY_SCALE, 8, 65))
    cv2.circle(frame, center, radius, color, 1, cv2.LINE_AA)

    src = result.get("source", "-")
    cv2.putText(frame, f"TTC {result['ttc_sec']:.1f}s {src}",
                (center[0]+8, center[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_alert(frame, ttc_sec, min_dist):
    w_scaled = int(500 * DISPLAY_SCALE)
    h_scaled = int(90 * DISPLAY_SCALE)
    cv2.rectangle(frame, (0, 0), (w_scaled, h_scaled), COLORS["danger"], -1)
    cv2.putText(frame, "!! COLLISION WARNING !!",
                (int(10*DISPLAY_SCALE), int(35*DISPLAY_SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85*DISPLAY_SCALE, (255,255,255), 2)
    cv2.putText(frame, f"TTC: {ttc_sec:.1f}s | Dist: {min_dist:.0f}px",
                (int(10*DISPLAY_SCALE), int(70*DISPLAY_SCALE)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55*DISPLAY_SCALE, (255,255,255), 1)


def draw_status(frame, record, n_tracks, n_preds):
    fps = 1000.0 / record["total_ms"] if record["total_ms"] > 0 else 0
    text = (f"{record['type']} FPS:{fps:.1f} Lat:{record['total_ms']:.1f}ms "
            f"MC:{MC_N_SAMPLES} Tracks:{n_tracks} Preds:{n_preds}")
    y = frame.shape[0] - 12
    cv2.rectangle(frame, (4, y-20), (int(700*DISPLAY_SCALE), y+6), (0,0,0), -1)
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5*DISPLAY_SCALE, COLORS["text"], 1, cv2.LINE_AA)


def draw_minimal_danger_zone(frame, v_box, v_id):
    """assess_risk와 같은 zone 계산식으로 danger/caution 영역을 표시."""
    if not DEBUG_DRAW_DANGER_ZONE:
        return

    veh_vel_hist = velocity_history.get(v_id)
    if not veh_vel_hist:
        return
    n = len(veh_vel_hist)
    vx = sum(v[0] for v in veh_vel_hist) / n
    vy = sum(v[1] for v in veh_vel_hist) / n
    v_speed = math.hypot(vx, vy)
    if v_speed < STOPPED_VEL_THRESH:
        return

    hx, hy = vx / v_speed, vy / v_speed
    veh_d = _veh_diag(v_box)
    front_side, front_length = _front_danger_shape(v_speed, veh_d, track_classes.get(v_id))
    if front_side <= 0.0 or front_length <= 0.0:
        return

    caution_side = min(front_side * FRONT_CAUTION_SIDE_MULT, FRONT_CAUTION_SIDE_MAX)
    caution_length = min(front_length * FRONT_CAUTION_LENGTH_MULT, FRONT_CAUTION_LENGTH_MAX)

    cx, cy = scaled_pt(v_box[:2])
    s = DISPLAY_SCALE
    px, py = -hy, hx

    def zone_poly(side, length):
        return np.array([
            (int(cx + px * side * s), int(cy + py * side * s)),
            (int(cx - px * side * s), int(cy - py * side * s)),
            (int(cx + hx * length * s - px * side * s), int(cy + hy * length * s - py * side * s)),
            (int(cx + hx * length * s + px * side * s), int(cy + hy * length * s + py * side * s)),
        ], dtype=np.int32)

    cv2.polylines(frame, [zone_poly(caution_side, caution_length)], True, COLORS["caution"], 1, cv2.LINE_AA)
    cv2.polylines(frame, [zone_poly(front_side, front_length)], True, COLORS["danger"], 1, cv2.LINE_AA)


def draw_roi(frame):
    """횡단보도 ROI가 설정되어 있으면 화면에 다각형 외곽선으로 표시."""
    if not CROSSWALK_ROIS:
        return
    for polygon in CROSSWALK_ROIS:
        if len(polygon) < 3:
            continue
        pts = np.array(
            [(int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE)) for x, y in polygon],
            dtype=np.int32
        )
        cv2.polylines(frame, [pts], True, (80, 255, 120), 2, cv2.LINE_AA)


def make_visual_frame(frame):
    if DISPLAY_SCALE == 1.0:
        return frame
    w = int(frame.shape[1] * DISPLAY_SCALE)
    h = int(frame.shape[0] * DISPLAY_SCALE)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


# ==========================================
# 🧹 [트랙 정리]
# ==========================================
def cleanup_stale_tracks(active_ids, frame_count, max_lost=60):
    stale = [
        tid for tid in track_history
        if tid not in active_ids
        and (frame_count - prev_frames.get(tid, 0)) > max_lost
    ]
    for tid in stale:
        for d in [track_history, prev_boxes, predictions, predictions_std,
                  pred_frame_count,
                  velocity_history, heading_vel_history, prev_velocity, track_classes, prev_frames,
                  class_votes, last_adaptive_refine]:
            d.pop(tid, None)


def merge_duplicate_tracks(merge_dist=100):
    ids = list(predictions.keys())
    to_remove = set()
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if ids[i] in prev_boxes and ids[j] in prev_boxes:
                dist = math.hypot(
                    float(prev_boxes[ids[i]][0] - prev_boxes[ids[j]][0]),
                    float(prev_boxes[ids[i]][1] - prev_boxes[ids[j]][1])
                )
                if dist < merge_dist:
                    to_remove.add(max(ids[i], ids[j]))
    for tid in to_remove:
        for d in [track_history, prev_boxes, predictions, predictions_std,
                  pred_frame_count,
                  velocity_history, heading_vel_history, prev_velocity, track_classes, prev_frames,
                  class_votes, last_adaptive_refine]:
            d.pop(tid, None)


# ==========================================
# 🎬 [메인]
# ==========================================
def main():
    ort_providers = ["CPUExecutionProvider"]

    print(f"\n{'='*60}")
    print("A-PAS Version C ? Sampled ByteTrack Hybrid")
    print(f"{'='*60}")
    print("  YOLO backend    : Hailo HEF")
    print("  YOLO device     : hailo8")
    print(f"  YOLO imgsz      : {YOLO_IMGSZ}")
    print(f"  HEF path        : {HEF_MODEL_PATH}")
    print(f"  ONNX path       : {ONNX_MODEL_PATH}")
    print(f"  ORT providers   : {ort_providers}")
    mode_text = "every frame" if USE_YOLO_EVERY_FRAME else f"sample frames only (every {SAMPLE_INTERVAL} frames)"
    print(f"  Tracking        : supervision.ByteTrack, YOLO {mode_text}")
    print(f"  LSTM sampling   : every {SAMPLE_INTERVAL} frames")
    print(f"  MC Dropout      : {MC_N_SAMPLES} (base={MC_BASE_SAMPLES}, adaptive={MC_ADAPTIVE_SAMPLES})")
    print(f"  Display scale   : {DISPLAY_SCALE}")
    print(f"  Display rate    : sample frames only (~{TARGET_FPS} FPS)")
    print(f"  Runtime caps    : lstm_tracks={MAX_LSTM_TRACKS}, ttc_pairs={MAX_TTC_PAIRS}")
    print(f"  Warmup skip     : {STATS_WARMUP_FRAMES} frames")
    print(f"{'='*60}\n")

    hailo_detector = HailoYOLODetector(HEF_MODEL_PATH, conf_thresh=0.25)
    bytetracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=15,
        minimum_matching_threshold=0.8,
        frame_rate=10,
    )
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    lstm_session = ort.InferenceSession(
        ONNX_MODEL_PATH,
        sess_options=sess_options,
        providers=ort_providers,
    )
    _build_roi_mask()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    profiler = LatencyProfiler()
    frame_count = 0
    yolo_runs = 0
    sample_index = 0

    # 비샘플 프레임에서 시각화/경고를 재사용할 캐시
    last_tracked = []  # [(box, track_id, voted_class, conf), ...]
    last_draw_data = last_tracked
    risk_results = []
    last_ttc_sec = PRED_LENGTH * DT
    last_min_dist = float('inf')
    hailo_active = False

    try:
        hailo_detector.__enter__()
        hailo_active = True
        while cap.isOpened():
            frame_start = time.perf_counter()
            timer = StageTimer()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            is_sample = (frame_count % SAMPLE_INTERVAL == 0)
            mc_samples_used = 0
            active_ids = set()

            collision_risk = False
            min_dist_global = last_min_dist
            ttc_sec_global = last_ttc_sec

            # ① YOLO + ByteTrack (Version C: 샘플 프레임만, 필요 시 fallback)
            run_yolo_this_frame = USE_YOLO_EVERY_FRAME or is_sample
            boxes_arr = []
            track_ids_arr = []
            class_ids_arr = []
            confs_arr = []
            if run_yolo_this_frame:
                yolo_runs += 1
                raw_dets = hailo_detector.detect(frame)
                if raw_dets:
                    raw_arr = np.array(raw_dets, dtype=np.float32)
                    cx, cy, w, h = raw_arr[:, 0], raw_arr[:, 1], raw_arr[:, 2], raw_arr[:, 3]
                    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
                    sv_dets = sv.Detections(
                        xyxy=xyxy,
                        confidence=raw_arr[:, 4],
                        class_id=raw_arr[:, 5].astype(int),
                    )
                else:
                    sv_dets = sv.Detections.empty()

                tracked = bytetracker.update_with_detections(sv_dets)
                if len(tracked) > 0 and tracked.tracker_id is not None:
                    for i in range(len(tracked)):
                        x1, y1, x2, y2 = tracked.xyxy[i]
                        cx_, cy_ = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        w_, h_ = x2 - x1, y2 - y1
                        boxes_arr.append(np.array([cx_, cy_, w_, h_], dtype=np.float32))
                        track_ids_arr.append(int(tracked.tracker_id[i]))
                        class_ids_arr.append(int(tracked.class_id[i]))
                        confs_arr.append(float(tracked.confidence[i]))
            timer.mark("yolo_inference")

            # 필터링
            filtered_items = []
            for box, track_id, class_id, conf in zip(boxes_arr, track_ids_arr, class_ids_arr, confs_arr):
                active_ids.add(track_id)
                x, y, w, h = box
                if class_id == 0 and w * h < 400:
                    continue
                if conf < CONF_THRESHOLD.get(int(class_id), 0.25):
                    continue
                class_votes[track_id].append(int(class_id))
                voted_class = Counter(class_votes[track_id]).most_common(1)[0][0]
                filtered_items.append((box.astype(np.float32), track_id, voted_class, float(conf)))

            if run_yolo_this_frame and filtered_items:
                last_tracked = filtered_items
                last_draw_data = last_tracked
            elif not run_yolo_this_frame:
                active_ids.update(tid for _, tid, _, _ in last_tracked)

            timer.mark("detection_filter")
            timer.mark("tracker")  # ByteTrack은 YOLO 내부에서 처리됨

            # ② LSTM (샘플 프레임만)
            if is_sample:
                sample_index += 1

                prediction_tracks = select_tracks_for_prediction(
                    [it[0] for it in filtered_items],
                    [it[1] for it in filtered_items],
                    [it[2] for it in filtered_items],
                    [it[3] for it in filtered_items],
                )

                # ── 1단계: 모든 트랙에 대해 history/velocity 갱신 (heading 추정용) ──
                # add_to_history와 compute_features는 가볍고, heading/사각형 표시에 필수.
                # LSTM 추론(run_lstm_mc)만 prediction_tracks로 제한.
                for box, track_id, class_id, conf in filtered_items:
                    # 갭 체크 — sample 기반(SAMPLE_INTERVAL=3)이라 frame_diff>15 (≈5 sample, 0.5초)에서 cleanup.
                    if track_id in prev_frames:
                        frame_diff = frame_count - prev_frames[track_id]
                        if frame_diff > 15:
                            if track_id in velocity_history:
                                velocity_history[track_id].clear()
                            if track_id in heading_vel_history:
                                heading_vel_history[track_id].clear()
                            if track_id in prev_velocity:
                                prev_velocity[track_id] = [0.0, 0.0]
                            if track_id in track_history:
                                track_history[track_id].clear()

                    prev_frames[track_id] = frame_count
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, class_id, prev_box)
                    prev_boxes[track_id] = box

                # ── 2단계: 선택된 트랙에 대해서만 LSTM 추론 (성능 보호) ──
                for box, track_id, class_id, conf in prediction_tracks:
                    mean, std, n = run_lstm_mc(lstm_session, track_id)
                    if mean is not None:
                        update_prediction(track_id, mean, std)
                        mc_samples_used += n

                if ENABLE_TRACK_MERGE:
                    merge_duplicate_tracks()

            timer.mark("feature_lstm_mc")


            # ? TTC (?? ????)
            if is_sample:
                risk_results = []

                # === 1. (???, ??) pair ?? ?? + ?? ?? ===
                ped_items = [(b, t) for b, t, c, _ in filtered_items if c == 0]
                veh_items = [(b, t) for b, t, c, _ in filtered_items if c in VEHICLE_CLASSES]

                pair_candidates = []
                for p_box, p_id in ped_items:
                    if not is_point_in_roi_fast(float(p_box[0]), float(p_box[1])):
                        continue
                    for v_box, v_id in veh_items:
                        dist_now = math.hypot(
                            float(p_box[0] - v_box[0]),
                            float(p_box[1] - v_box[1])
                        )
                        if dist_now <= TTC_PAIR_CURRENT_DIST:
                            pair_candidates.append((dist_now, p_id, p_box, v_id, v_box))

                pair_candidates.sort(key=lambda x: x[0])
                pair_candidates = pair_candidates[:MAX_TTC_PAIRS]

                # === 2. assess_risk? ?? ?? ===
                for _, p_id, p_box, v_id, v_box in pair_candidates:
                    result = assess_risk(p_id, p_box, v_id, v_box)
                    if result is not None:
                        risk_results.append(result)

                # Adaptive refine
                extra = refine_adaptive_pairs(lstm_session, risk_results, sample_index)
                mc_samples_used += extra

                # === 3. collision_risk ?? (draw_alert?) ===
                for r in risk_results:
                    if r["danger"]:
                        collision_risk = True
                        if r["ttc_sec"] < ttc_sec_global:
                            ttc_sec_global = r["ttc_sec"]
                        if r["min_dist"] < min_dist_global:
                            min_dist_global = r["min_dist"]

                risk_results.sort(key=lambda r: (not r["danger"], r["ttc_sec"]))
                last_ttc_sec = ttc_sec_global
                last_min_dist = min_dist_global

            timer.mark("ttc")

            # ④ 시각화 (sample frame에서만 화면 갱신 — reuse는 redundant 회피)
            key = 0
            display_this_frame = is_sample or USE_YOLO_EVERY_FRAME
            if display_this_frame:
                vis = make_visual_frame(frame)
                draw_roi(vis)

                risk_tids = set()
                danger_tids = set()
                for r in risk_results:
                    if r["danger"]:
                        danger_tids.add(r["ped_id"])
                        danger_tids.add(r["veh_id"])
                        risk_tids.add(r["ped_id"])
                        risk_tids.add(r["veh_id"])
                    elif r.get("caution", False):
                        risk_tids.add(r["ped_id"])
                        risk_tids.add(r["veh_id"])

                # 1초 hold: danger/caution 모두 적용
                for r in risk_results:
                    if r["danger"] or r.get("caution", False):
                        pair_key = (r["ped_id"], r["veh_id"])
                        level = "danger" if r["danger"] else "caution"
                        danger_visual_cache[pair_key] = (DANGER_VISUAL_HOLD, level, r)

                cached_markers = []
                expired_pairs = []
                for pair_key in list(danger_visual_cache.keys()):
                    frames, level, snapshot = danger_visual_cache[pair_key]
                    ped_id, veh_id = pair_key

                    if level == "danger":
                        danger_tids.add(ped_id)
                        danger_tids.add(veh_id)
                        risk_tids.add(ped_id)
                        risk_tids.add(veh_id)
                        cached_markers.append(snapshot)
                    else:
                        risk_tids.add(ped_id)
                        risk_tids.add(veh_id)
                        # caution zone 진입 시 bbox + marker 동시 발화
                        cached_markers.append(snapshot)

                    new_frames = frames - 1
                    if new_frames <= 0:
                        expired_pairs.append(pair_key)
                    else:
                        danger_visual_cache[pair_key] = (new_frames, level, snapshot)

                for pair_key in expired_pairs:
                    del danger_visual_cache[pair_key]

                # 현재 프레임의 detection 사용 (ByteTrack이라 매 프레임 있음)
                draw_data = filtered_items if filtered_items else last_draw_data
                if filtered_items:
                    last_draw_data = filtered_items

                for box, track_id, class_id, conf in draw_data[:MAX_VIS_TRACKS]:
                    if track_id in danger_tids:
                        level = "danger"
                    elif track_id in risk_tids:
                        level = "caution"
                    else:
                        level = "safe"

                    draw_bbox(vis, box, track_id, class_id, conf, level)
                    if class_id in VEHICLE_CLASSES:
                        draw_minimal_danger_zone(vis, box, track_id)
                    draw_prediction(vis, track_id, level)

                for r in cached_markers[:3]:
                    draw_risk_marker(vis, r)

                if collision_risk:
                    draw_alert(vis, ttc_sec_global, min_dist_global)

            timer.mark("visualization")

            # 기록
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            record = {
                "frame": frame_count,
                "type": "sample" if is_sample else "reuse",
                "total_ms": total_ms,
                "stages": timer.times.copy(),
                "tracks": len(active_ids),
                "preds": len(predictions),
                "mc_samples": mc_samples_used,
            }
            if display_this_frame:
                draw_status(vis, record, len(active_ids), len(predictions))
                cv2.imshow("A-PAS Monitor", vis)
                key = cv2.waitKey(1) & 0xFF
            timer.mark("imshow_waitkey")
            record["total_ms"] = (time.perf_counter() - frame_start) * 1000.0
            record["stages"] = timer.times.copy()
            profiler.add(record)

            if key == ord('q'):
                break

            if frame_count % 30 == 0:
                cleanup_stale_tracks(active_ids, frame_count, max_lost=60)

    finally:
        if hailo_active:
            exc_type, exc, tb = sys.exc_info()
            hailo_detector.__exit__(exc_type, exc, tb)
        cap.release()
        cv2.destroyAllWindows()

        recs = profiler.records
        if recs:
            report_recs = [r for r in recs if r["frame"] > STATS_WARMUP_FRAMES]
            if not report_recs:
                report_recs = recs
            total = np.array([r["total_ms"] for r in report_recs], dtype=np.float32)
            sample_recs = [r for r in report_recs if r["type"] == "sample"]
            reuse_recs  = [r for r in report_recs if r["type"] == "reuse"]

            print(f"\n{'='*60}")
            print("📊 A-PAS Version C — 최종 성능 통계")
            print(f"{'='*60}")
            print(f"  처리 프레임        : {frame_count:,}장 (stats skip first {STATS_WARMUP_FRAMES})")
            print(f"  YOLO 실행 횟수     : {yolo_runs:,}회 ({100*yolo_runs/max(1,frame_count):.1f}%)")
            print(f"  전체 평균 Latency  : {total.mean():.2f}ms")
            print(f"  전체 평균 FPS      : {1000/total.mean():.1f}")
            print(f"  Min / Max Latency  : {total.min():.2f}ms / {total.max():.2f}ms")
            if sample_recs:
                arr = np.array([r["total_ms"] for r in sample_recs], dtype=np.float32)
                print(f"  [샘플]  평균 Latency : {arr.mean():.2f}ms")
            if reuse_recs:
                arr = np.array([r["total_ms"] for r in reuse_recs], dtype=np.float32)
                print(f"  [재사용] 평균 Latency: {arr.mean():.2f}ms")
            print(f"  MC mode            : {MC_N_SAMPLES} (base={MC_BASE_SAMPLES})")
            print(f"  YOLO runs          : {yolo_runs:,} / {frame_count:,} frames")
            print(f"  Config             : imgsz={YOLO_IMGSZ}, scale={DISPLAY_SCALE}, yolo_every_frame={USE_YOLO_EVERY_FRAME}")
            print(f"{'='*60}")

            profiler.print_report()


if __name__ == "__main__":
    main()
