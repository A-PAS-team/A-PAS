"""
A-PAS Main System — Version C (Sampled ByteTrack Hybrid)
==========================================================
변경 요약:
  1. Version A의 ByteTrack 기반 안정성을 유지
  2. Version B의 샘플 프레임 처리 구조를 결합
  3. 샘플 프레임에서만 YOLO + ByteTrack + LSTM + TTC 수행
  4. 비샘플 프레임은 last_tracked/risk_results 캐시로 시각화 재사용
  5. risk_results, last_ttc_sec, last_min_dist 캐시로 경고 UI 기본값 버그 방지
  6. cleanup/merge에서 상태 dict 누락 없이 정리

최적화 요약:
  1. ByteTrack 유지 (기본: YOLO 샘플 프레임만, fallback 가능)
  2. YOLO GPU auto-detect + FP16
  3. Adaptive MC Dropout (기본 6, 위험 시 12)
  4. MAX_LSTM_TRACKS / MAX_TTC_PAIRS 런타임 캡
  5. DISPLAY_SCALE=0.75
  6. 경량 시각화 (polyline + cross marker, frame.copy() 0회)
  7. Stage별 latency profiler + warmup 제외

핵심 파이프라인:
  CARLA/Camera → YOLOv8 (샘플 프레임) → ByteTrack
  → 17-Feature 추출 (SAMPLE_INTERVAL마다) → Residual LSTM MC Dropout (ONNX)
  → TTC 충돌 판정 → 경고 출력

배포 환경: Raspberry Pi 5 + Hailo-8 AI HAT+
모델: Residual LSTM (ADE 2.08px, 10FPS, PRED=1초)

TODO: persist=True 상태에서 YOLO 프레임을 건너뛰는 방식은 ByteTrack 내부 상태와
      충돌할 가능성이 있다. 문제가 생기면 USE_YOLO_EVERY_FRAME=True로 fallback.
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
import math
from collections import deque, defaultdict, Counter
from ultralytics import YOLO

# --- mc_inference import ---
try:
    sys.path.insert(0, "ai_model/trajectory/mcdropout")
    from mc_inference import run_lstm_with_uncertainty
except ImportError:
    sys.path.insert(0, r"C:/Users/USER/Desktop/UIJIN_Workfolder/workplace/A-PAS/ai_model/trajectory/mcdropout")
    try:
        from mc_inference import run_lstm_with_uncertainty
    except ImportError as exc:
        print("❌ mc_inference import 실패.")
        raise exc


# ==========================================
# ⚙️ [설정]
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH      = "test_video5.mp4"
YOLO_MODEL_PATH = "best_v7.pt"
ONNX_MODEL_PATH = "ai_model/trajectory/mcdropout/residual_mcd_10fps.onnx"  # changed for 10fps obs1s pred1.5s

# --- 해상도 ---
IMG_W, IMG_H = 1920, 1080
DIAG         = float(np.sqrt(IMG_W**2 + IMG_H**2))

# --- 모델 설정 ---
SEQ_LENGTH   = 20  # changed for 10fps obs1s pred1.5s
PRED_LENGTH  = 10  # changed for 10fps obs1s pred1.5s
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
SIGMA_SCALE    = 3.0  # v5: use actual position for clamp, reduce sigma inflation
BBOX_DANGER_DIST = 45.0   # v2: approach/departure check
BBOX_CAUTION_DIST = 120.0 # v2: approach/departure check
MAX_BBOX_RISKS = 3
MAX_BBOX_RISKS_PER_PED = 2
DANGER_VISUAL_HOLD = 30    # v11-hold: 1s danger visual persistence

# --- 횡단보도 ROI ---
# 사용 예시 (1920x1080 해상도 기준, 좌상단 원점):
# CROSSWALK_ROIS = [
#     [(19, 681), (525, 343), (565, 358), (123, 684)],
#     [(633, 339), (1077, 482), (1078, 442), (670, 320)],
# ]
# 빈 리스트면 ROI 필터 비활성화 → 모든 보행자 평가
CROSSWALK_ROIS = [
    [(19, 681), (525, 343), (565, 358), (123, 684)],
    [(633, 339), (1077, 482), (1078, 442), (670, 320)],
]
ROI_MARGIN = 20

# --- YOLO 설정 ---
YOLO_IMGSZ   = 1280
YOLO_DEVICE  = "auto"
YOLO_HALF    = True
USE_YOLO_EVERY_FRAME = False

# --- ORT 설정 ---
ORT_USE_CUDA = False

# --- MC Dropout 설정 ---
MC_N_SAMPLES         = "adaptive"
MC_BASE_SAMPLES      = 4
MC_ADAPTIVE_SAMPLES  = 8
MC_USE_BATCH         = True
ADAPTIVE_REFINE_EVERY = 4
MAX_ADAPTIVE_PAIRS   = 1

# --- 예측 시각화 안정화 ---
EMA_ALPHA       = 0.85  # v12: faster EMA + behind caution suppress
EMA_ALPHA_STD   = 0.85  # v12: faster EMA + behind caution suppress
WARMUP_HIDE     = 5
WARMUP_FADE     = 15
WARMUP_TTC_GATE = 2

# --- 충돌 hysteresis ---
COLLISION_HOLD = 2

# --- 런타임 캡 ---
MAX_LSTM_TRACKS = 10
MAX_TTC_PAIRS   = 6
TTC_PAIR_CURRENT_DIST = 500.0  # v3: stopped vehicle EMA bypass
STOPPED_VEL_THRESH = 40.0  # v6: fix stopped detection threshold

# --- 시각화 설정 ---
DISPLAY_SCALE    = 0.75
PRED_DRAW_STRIDE = 3
MAX_VIS_TRACKS   = 24

# --- 통계 ---
STATS_WARMUP_FRAMES = 3

# --- 클래스 매핑 ---
VEHICLE_CLASSES = {1, 2, 3}
CONF_THRESHOLD  = {0: 0.35, 1: 0.25, 2: 0.30, 3: 0.25}
CLASS_MAP       = {0: 0, 1: 1, 2: 2, 3: 3}
CLASS_NAMES     = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}
CLASS_COLORS    = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}

COLORS = {
    "safe":    (0, 220, 0),
    "caution": (0, 220, 255),
    "danger":  (0, 0, 255),
    "text":    (230, 230, 230),
}
_SIGMA_DIV = SIGMA_SCALE / math.sqrt(2)  # calc_ttc용 사전 계산 상수


# ==========================================
# 🔧 [GPU / ORT 자동 감지]
# ==========================================
def resolve_yolo_device():
    if YOLO_DEVICE != "auto":
        return YOLO_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def cuda_status_text():
    try:
        import torch
        if torch.cuda.is_available():
            return f"available=True, name={torch.cuda.get_device_name(0)}"
        return "available=False"
    except Exception as exc:
        return f"torch import failed: {exc}"


def resolve_ort_providers():
    available = ort.get_available_providers()
    if ORT_USE_CUDA and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def yolo_uses_half(device):
    return YOLO_HALF and str(device).lower() != "cpu"


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
predictions_raw_mean = {}
predictions_raw_std  = {}
predictions          = {}
predictions_std      = {}
pred_frame_count     = {}
velocity_history     = {}
prev_velocity        = {}
last_valid_heading   = {}
track_classes        = {}
class_votes          = defaultdict(lambda: deque(maxlen=10))
prev_frames          = {}
last_adaptive_refine = {}
danger_visual_cache  = {}  # v11-hold: 1s danger visual persistence
behind_cooldown      = {}  # v11-hold-fix: behind 판정 후 0.5초간 재발 방지


# ==========================================
# 🔧 [피처 추출]
# ==========================================
def compute_features(box, prev_box, track_id):
    x, y, w, h = box
    if prev_box is not None:
        raw_vx = (x - prev_box[0]) / DT
        raw_vy = (y - prev_box[1]) / DT
    else:
        raw_vx, raw_vy = 0.0, 0.0

    if track_id not in velocity_history:
        velocity_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
    velocity_history[track_id].append([raw_vx, raw_vy])

    hist = velocity_history[track_id]
    n = len(hist)
    vx = sum(v[0] for v in hist) / n
    vy = sum(v[1] for v in hist) / n

    if track_id in prev_velocity:
        ax = (vx - prev_velocity[track_id][0]) / DT
        ay = (vy - prev_velocity[track_id][1]) / DT
    else:
        ax, ay = 0.0, 0.0
    prev_velocity[track_id] = [vx, vy]

    speed = math.sqrt(vx * vx + vy * vy)
    if speed >= STOPPED_VEL_THRESH:
        last_valid_heading[track_id] = np.array([vx, vy], dtype=np.float32)
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
    predictions_raw_mean[track_id] = mean_new
    predictions_raw_std[track_id]  = std_new

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


# ==========================================
# ⚠️ [TTC 계산]
# ==========================================
def calc_bbox_edge_dist(box_a, box_b):
    """xywh bbox 두 개의 edge 간 최소 거리를 계산한다. 겹치면 -1.0."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    a_left, a_top = ax - aw / 2.0, ay - ah / 2.0
    a_right, a_bottom = ax + aw / 2.0, ay + ah / 2.0
    b_left, b_top = bx - bw / 2.0, by - bh / 2.0
    b_right, b_bottom = bx + bw / 2.0, by + bh / 2.0

    dx = max(0.0, max(a_left - b_right, b_left - a_right))
    dy = max(0.0, max(a_top - b_bottom, b_top - a_bottom))
    if dx == 0.0 and dy == 0.0:
        return -1.0
    return math.sqrt(dx * dx + dy * dy)


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """점과 선분 사이의 최소 거리."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def is_point_in_roi(x, y, polygon, margin=0):
    """점이 ROI 다각형 내부 또는 margin 거리 안에 있는지 판정."""
    if not polygon:
        return True
    if len(polygon) < 3:
        return True

    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-6) + xi
        )
        if intersects:
            inside = not inside
        j = i

    if inside:
        return True
    if margin <= 0:
        return False

    min_dist = float('inf')
    for i in range(len(polygon)):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % len(polygon)]
        min_dist = min(min_dist, _point_to_segment_dist(x, y, ax, ay, bx, by))
    return min_dist <= margin


def is_point_in_any_roi(x, y, polygons, margin=0):
    """여러 ROI 중 하나라도 포함하면 True. ROI 목록이 비어 있으면 필터 비활성화."""
    if not polygons:
        return True
    return any(is_point_in_roi(x, y, polygon, margin) for polygon in polygons)


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


def is_pedestrian_behind_vehicle(ped_id, veh_id):
    """차량 진행 방향 기준으로 보행자가 뒤쪽에 있으면 True.
    한 번 behind 판정 후 15프레임(0.5초) 쿨다운으로 bbox 노이즈 재발 방지."""
    pair_key = (ped_id, veh_id)
    # v11-hold-fix: 쿨다운 활성 → 즉시 True
    if pair_key in behind_cooldown and behind_cooldown[pair_key] > 0:
        behind_cooldown[pair_key] -= 1
        if behind_cooldown[pair_key] <= 0:
            del behind_cooldown[pair_key]
        return True

    p_box = prev_boxes.get(ped_id)
    v_box = prev_boxes.get(veh_id)
    if p_box is None or v_box is None:
        return False

    heading_vec = None
    veh_vel_hist = velocity_history.get(veh_id)
    if veh_vel_hist and len(veh_vel_hist) > 0:
        veh_vel = np.array(veh_vel_hist, dtype=np.float32).mean(axis=0)
        if math.hypot(float(veh_vel[0]), float(veh_vel[1])) >= STOPPED_VEL_THRESH:
            heading_vec = veh_vel

    if heading_vec is None:
        cached_heading = last_valid_heading.get(veh_id)
        if cached_heading is not None:
            heading_vec = np.array(cached_heading, dtype=np.float32)

    if heading_vec is None:
        veh_hist = track_history.get(veh_id)
        if veh_hist and len(veh_hist) > 0:
            last_feat = np.array(veh_hist[-1], dtype=np.float32)
            if len(last_feat) > 8:
                sin_h = float(last_feat[7])
                cos_h = float(last_feat[8])
                heading_vec = np.array([cos_h, sin_h], dtype=np.float32)  # v10: behind-vehicle suppression

    if heading_vec is None:
        return False

    heading_norm = math.hypot(float(heading_vec[0]), float(heading_vec[1]))
    if heading_norm < 1e-6:
        return False

    to_ped = np.array([p_box[0] - v_box[0], p_box[1] - v_box[1]], dtype=np.float32)
    to_ped_norm = math.hypot(float(to_ped[0]), float(to_ped[1]))
    if to_ped_norm < 1e-6:
        return False

    is_behind = float(np.dot(heading_vec / heading_norm, to_ped / to_ped_norm)) <= 0.15  # v11-hold-fix: wider behind/side suppression
    if is_behind:
        behind_cooldown[pair_key] = 15  # v11-hold-fix: 0.5초 쿨다운
    return is_behind


def find_bbox_proximity_risks(filtered_items):
    """Layer 1: 현재 bbox edge 근접 위험을 예측/warmup과 무관하게 즉시 판정."""
    persons = [
        (box, tid, conf) for box, tid, cls, conf in filtered_items
        if cls == 0 and is_point_in_roi_fast(box[0], box[1])
    ]
    vehicles = [(box, tid, conf) for box, tid, cls, conf in filtered_items if cls in VEHICLE_CLASSES]

    risks_by_ped = defaultdict(list)
    for p_box, p_id, _ in persons:
        for v_box, v_id, _ in vehicles:
            edge_dist = calc_bbox_edge_dist(p_box, v_box)
            if edge_dist > BBOX_CAUTION_DIST:
                continue

            danger = edge_dist <= BBOX_DANGER_DIST
            p_center = np.array(p_box[:2], dtype=np.float32)
            v_center = np.array(v_box[:2], dtype=np.float32)
            point = (p_center + v_center) * 0.5

            veh_vel_hist = velocity_history.get(v_id)  # v2: approach/departure check
            # v11-hold-fix: 새 트랙(ID 변경 포함)은 속도 이력 부족 → BBOX danger 억제
            if not veh_vel_hist or len(veh_vel_hist) < 2:
                danger = False
            elif len(veh_vel_hist) > 0:
                veh_vel = np.array(veh_vel_hist, dtype=np.float32).mean(axis=0)
                speed = math.hypot(float(veh_vel[0]), float(veh_vel[1]))
                near_overlap = edge_dist <= (BBOX_DANGER_DIST / 2.0)
                approach_score = None
                to_ped = p_center - v_center
                to_ped_norm = math.hypot(float(to_ped[0]), float(to_ped[1]))
                if to_ped_norm > 1e-6:
                    approach_score = float(np.dot(veh_vel, to_ped / to_ped_norm))
                    if speed >= STOPPED_VEL_THRESH and approach_score > speed * 0.3:
                        danger = edge_dist <= BBOX_DANGER_DIST * 2.5

                if not near_overlap:
                    if speed < STOPPED_VEL_THRESH:
                        danger = False
                    else:
                        if approach_score is not None and approach_score < speed * 0.3:
                            danger = False
                else:
                    if speed >= STOPPED_VEL_THRESH and approach_score is not None and approach_score < speed * 0.3:
                        danger = False

            if is_pedestrian_behind_vehicle(p_id, v_id):
                continue  # v10: behind-vehicle suppression

            if danger:
                print(f"[BBOX-DANGER] ped={p_id} veh={v_id} edge_dist={edge_dist:.1f}")  # v7: debug log + actual distance safety gate

            result = {
                "ped_id": p_id,
                "veh_id": v_id,
                "min_dist": float(edge_dist),
                "ttc_sec": 0.0 if danger else 0.5,
                "danger": bool(danger),
                "caution": True,
                "point": point.astype(np.float32),
                "std_radius": 10.0,
                "source": "bbox",
            }

            risks_by_ped[p_id].append(result)

    risks = []
    for ped_risks in risks_by_ped.values():
        ped_risks.sort(key=lambda r: (not r["danger"], r["min_dist"]))
        risks.extend(ped_risks[:MAX_BBOX_RISKS_PER_PED])

    risks.sort(key=lambda r: (not r["danger"], r["min_dist"]))
    return risks[:MAX_BBOX_RISKS]


def bbox_size_for_track(track_id):
    """prev_boxes에서 bbox 반경성 크기(max(w,h)/2)를 가져온다."""
    box = prev_boxes.get(track_id)
    if box is None:
        return 0.0
    return float(max(box[2], box[3]) / 2.0)


def dedupe_risk_results(results):
    """같은 보행자-차량 pair가 여러 layer에서 나오면 더 위험한 결과만 남긴다."""
    merged = {}
    for result in results:
        key = (result["ped_id"], result["veh_id"])
        old = merged.get(key)
        if old is None:
            merged[key] = result
            continue

        replace = False
        if result["danger"] and not old["danger"]:
            replace = True
        elif result["danger"] == old["danger"]:
            if result["ttc_sec"] < old["ttc_sec"]:
                replace = True
            elif result["ttc_sec"] == old["ttc_sec"] and result["min_dist"] < old["min_dist"]:
                replace = True

        if replace:
            merged[key] = result
    return list(merged.values())


def calc_ttc_with_uncertainty(mean_a, std_a, mean_b, std_b, bbox_size_a=0.0, bbox_size_b=0.0):
    min_dist        = float('inf')
    collision_frame = PRED_LENGTH
    current_dist = math.hypot(float(mean_a[0][0] - mean_b[0][0]), float(mean_a[0][1] - mean_b[0][1]))  # v2: approach/departure check
    end_dist = math.hypot(float(mean_a[-1][0] - mean_b[-1][0]), float(mean_a[-1][1] - mean_b[-1][1]))    # v2: approach/departure check
    is_departing = end_dist > current_dist and current_dist > COLLISION_DIST  # v4: clamp stopped vehicle, depart caution suppress, ghost cleanup
    if current_dist > TTC_PAIR_CURRENT_DIST * 0.6:
        is_departing = True  # v6: fix stopped detection threshold

    for t in range(PRED_LENGTH):
        dist = math.hypot(float(mean_a[t][0] - mean_b[t][0]), float(mean_a[t][1] - mean_b[t][1]))
        sigma_a = _SIGMA_DIV * math.hypot(float(std_a[t][0]), float(std_a[t][1]))
        sigma_b = _SIGMA_DIV * math.hypot(float(std_b[t][0]), float(std_b[t][1]))
        bbox_margin = (bbox_size_a + bbox_size_b) / 2.0
        safety_margin = max(COLLISION_DIST, bbox_margin) + sigma_a + sigma_b

        if dist < min_dist:
            min_dist = dist
        if not is_departing and dist < safety_margin:
            collision_frame = t
            break

    ttc_seconds = collision_frame * DT
    return min_dist, collision_frame, ttc_seconds, is_departing


def find_risk_pairs():
    valid_ids = {
        tid for tid in predictions_raw_mean
        if pred_frame_count.get(tid, 0) >= WARMUP_TTC_GATE
    }
    pids = [
        tid for tid in valid_ids
        if track_classes.get(tid) == 0
        and tid in prev_boxes
        and is_point_in_roi_fast(prev_boxes[tid][0], prev_boxes[tid][1])
    ]
    vids = [tid for tid in valid_ids if track_classes.get(tid) in VEHICLE_CLASSES]

    candidates = []
    for p in pids:
        for v in vids:
            if p in prev_boxes and v in prev_boxes:
                dist_now = math.hypot(
                    float(prev_boxes[p][0] - prev_boxes[v][0]),
                    float(prev_boxes[p][1] - prev_boxes[v][1])
                )
                if dist_now <= TTC_PAIR_CURRENT_DIST:
                    candidates.append((dist_now, p, v))

    candidates.sort(key=lambda x: x[0])
    return [(p, v) for _, p, v in candidates[:MAX_TTC_PAIRS]]


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
                predictions_raw_mean[tid] = mean
                predictions_raw_std[tid] = std
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

    cv2.putText(frame, f"TTC {result['ttc_sec']:.1f}s",
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
                  predictions_raw_mean, predictions_raw_std, pred_frame_count,
                  velocity_history, prev_velocity, last_valid_heading, track_classes, prev_frames,
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
                  predictions_raw_mean, predictions_raw_std, pred_frame_count,
                  velocity_history, prev_velocity, last_valid_heading, track_classes, prev_frames,
                  class_votes, last_adaptive_refine]:
            d.pop(tid, None)


# ==========================================
# 🎬 [메인]
# ==========================================
def main():
    yolo_device = resolve_yolo_device()
    yolo_half = yolo_uses_half(yolo_device)
    ort_providers = resolve_ort_providers()

    print(f"\n{'='*60}")
    print("A-PAS Version C — Sampled ByteTrack Hybrid")
    print(f"{'='*60}")
    print(f"  CUDA            : {cuda_status_text()}")
    print(f"  YOLO device     : {yolo_device}, half={yolo_half}")
    print(f"  YOLO imgsz      : {YOLO_IMGSZ}")
    print(f"  ORT providers   : {ort_providers}")
    mode_text = "every frame" if USE_YOLO_EVERY_FRAME else f"sample frames only (every {SAMPLE_INTERVAL} frames)"
    print(f"  Tracking        : ByteTrack (persist=True, YOLO {mode_text})")
    print(f"  LSTM sampling   : every {SAMPLE_INTERVAL} frames")
    print(f"  MC Dropout      : {MC_N_SAMPLES} (base={MC_BASE_SAMPLES}, adaptive={MC_ADAPTIVE_SAMPLES})")
    print(f"  Display scale   : {DISPLAY_SCALE}")
    print(f"  Runtime caps    : lstm_tracks={MAX_LSTM_TRACKS}, ttc_pairs={MAX_TTC_PAIRS}")
    print(f"  Warmup skip     : {STATS_WARMUP_FRAMES} frames")
    print(f"{'='*60}\n")

    yolo_model = YOLO(YOLO_MODEL_PATH)
    try:
        yolo_model.fuse()
    except Exception:
        pass

    lstm_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=ort_providers)
    _build_roi_mask()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    profiler = LatencyProfiler()
    frame_count = 0
    yolo_runs = 0
    collision_streak = 0
    sample_index = 0

    # 비샘플 프레임에서 시각화/경고를 재사용할 캐시
    last_tracked = []  # [(box, track_id, voted_class, conf), ...]
    last_draw_data = last_tracked
    risk_results = []
    last_ttc_sec = PRED_LENGTH * DT
    last_min_dist = float('inf')

    try:
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
            ttc_global = PRED_LENGTH
            ttc_sec_global = last_ttc_sec

            # ① YOLO + ByteTrack (Version C: 샘플 프레임만, 필요 시 fallback)
            run_yolo_this_frame = USE_YOLO_EVERY_FRAME or is_sample
            results = []
            if run_yolo_this_frame:
                yolo_runs += 1
                results = yolo_model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=YOLO_IMGSZ,
                    classes=[0, 1, 2, 3],
                    conf=0.1,
                    iou=0.3,
                    agnostic_nms=False,
                    verbose=False,
                    device=yolo_device,
                    half=yolo_half,
                )
            timer.mark("yolo_inference")

            # Detection parsing
            has_detections = (results and results[0].boxes
                              and results[0].boxes.id is not None)
            boxes_arr = []
            track_ids_arr = []
            class_ids_arr = []
            confs_arr = []

            if has_detections:
                boxes_arr = results[0].boxes.xywh.cpu().numpy()
                track_ids_arr = results[0].boxes.id.int().cpu().tolist()
                class_ids_arr = results[0].boxes.cls.int().cpu().tolist()
                confs_arr = results[0].boxes.conf.cpu().numpy()

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
                prediction_tids = {t for _, t, _, _ in prediction_tracks}

                for box, track_id, class_id, conf in prediction_tracks:
                    # 갭 체크
                    if track_id in prev_frames:
                        frame_diff = frame_count - prev_frames[track_id]
                        if frame_diff > 5:
                            if track_id in velocity_history:
                                velocity_history[track_id].clear()
                            if track_id in prev_velocity:
                                prev_velocity[track_id] = [0.0, 0.0]
                            last_valid_heading.pop(track_id, None)
                            if track_id in track_history:
                                track_history[track_id].clear()

                    prev_frames[track_id] = frame_count
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, class_id, prev_box)
                    prev_boxes[track_id] = box

                    mean, std, n = run_lstm_mc(lstm_session, track_id)
                    if mean is not None:
                        update_prediction(track_id, mean, std)
                        mc_samples_used += n

                # 비예측 트랙도 상태 갱신
                for box, track_id, class_id, conf in filtered_items:
                    if track_id not in prediction_tids:
                        prev_frames[track_id] = frame_count
                        prev_boxes[track_id] = box

                merge_duplicate_tracks()

            timer.mark("feature_lstm_mc")

            # ③ TTC (샘플 프레임만)
            if is_sample:
                risk_results = []
                risk_results.extend(find_bbox_proximity_risks(filtered_items))
                pairs = find_risk_pairs()
                for p_id, v_id in pairs:
                    bbox_size_p = bbox_size_for_track(p_id)
                    bbox_size_v = bbox_size_for_track(v_id)
                    min_dist, col_frame, ttc_sec, is_departing = calc_ttc_with_uncertainty(
                        predictions_raw_mean[p_id], predictions_raw_std[p_id],
                        predictions_raw_mean[v_id], predictions_raw_std[v_id],
                        bbox_size_p, bbox_size_v,
                    )
                    danger = col_frame < PRED_LENGTH and ttc_sec < COLLISION_TTC
                    behind_vehicle = is_pedestrian_behind_vehicle(p_id, v_id)  # v12: faster EMA + behind caution suppress
                    caution = False if (is_departing or behind_vehicle) else (danger or min_dist <= COLLISION_DIST * 1.5)
                    actual_dist = math.hypot(
                        float(prev_boxes[p_id][0] - prev_boxes[v_id][0]),
                        float(prev_boxes[p_id][1] - prev_boxes[v_id][1])
                    )  # v7: debug log + actual distance safety gate
                    if danger and actual_dist > 180.0:  # v8: tighten collision dist
                        danger = False
                        caution = False
                    if behind_vehicle:
                        danger = False
                        caution = False
                        continue  # v12: faster EMA + behind caution suppress

                    event_frame = col_frame if col_frame < PRED_LENGTH else min(
                        range(PRED_LENGTH),
                        key=lambda t: math.hypot(
                            float(predictions_raw_mean[p_id][t][0] - predictions_raw_mean[v_id][t][0]),
                            float(predictions_raw_mean[p_id][t][1] - predictions_raw_mean[v_id][t][1])
                        )
                    )
                    point = (predictions_raw_mean[p_id][event_frame] +
                             predictions_raw_mean[v_id][event_frame]) * 0.5
                    std_r = SIGMA_SCALE * (
                        math.hypot(float(predictions_raw_std[p_id][event_frame][0]), float(predictions_raw_std[p_id][event_frame][1])) +
                        math.hypot(float(predictions_raw_std[v_id][event_frame][0]), float(predictions_raw_std[v_id][event_frame][1]))
                    ) / 2.0

                    result = {
                        "ped_id": p_id, "veh_id": v_id,
                        "min_dist": min_dist, "ttc_sec": ttc_sec,
                        "danger": danger, "caution": caution,
                        "point": point.astype(np.float32),
                        "std_radius": std_r,
                        "source": "trajectory",
                        "is_departing": bool(is_departing),  # v4: clamp stopped vehicle, depart caution suppress, ghost cleanup
                        "actual_dist": actual_dist,  # v7: debug log + actual distance safety gate
                    }
                    risk_results.append(result)
                    if danger:
                        print(f"[DANGER] ped={p_id} veh={v_id} ttc={ttc_sec:.2f}s "
                              f"min_dist={min_dist:.1f} actual_dist={actual_dist:.1f} "
                              f"departing={is_departing}")  # v7: debug log + actual distance safety gate

                    if min_dist < min_dist_global:
                        min_dist_global = min_dist
                        ttc_global = col_frame
                        ttc_sec_global = ttc_sec

                risk_results = dedupe_risk_results(risk_results)

                # v11-hold: 1s danger visual persistence (dedupe 이후에 적용)
                active_danger_pairs = set()  # v11-hold-fix: post-dedupe danger upgrade
                for r in risk_results:
                    pair_key = (r["ped_id"], r["veh_id"])  # v11-hold-fix: post-dedupe danger upgrade
                    if r["danger"]:
                        danger_visual_cache[pair_key] = {  # v11-hold-fix: preserve red marker through detection gaps
                            "expire": frame_count + DANGER_VISUAL_HOLD,
                            "point": r["point"].copy(),
                            "std_radius": float(r["std_radius"]),
                            "ttc_sec": float(r["ttc_sec"]),
                            "min_dist": float(r.get("min_dist", 0)),
                            "ped_id": r["ped_id"],
                            "veh_id": r["veh_id"],
                        }
                        active_danger_pairs.add(pair_key)

                for r in risk_results:
                    pair_key = (r["ped_id"], r["veh_id"])  # v11-hold-fix: post-dedupe danger upgrade
                    if pair_key in active_danger_pairs:
                        continue
                    if (pair_key in danger_visual_cache
                            and pair_key[0] in prev_boxes
                            and pair_key[1] in prev_boxes
                            and is_pedestrian_behind_vehicle(pair_key[0], pair_key[1])):
                        del danger_visual_cache[pair_key]  # v11-hold-fix: behind vehicle clears held danger
                        continue
                    if (not r["danger"]
                            and pair_key in danger_visual_cache
                            and frame_count <= danger_visual_cache[pair_key]["expire"]):
                        cache = danger_visual_cache[pair_key]  # v11-hold-fix: preserve red marker through detection gaps
                        r["danger"] = True  # v11-hold-fix: preserve red marker through detection gaps
                        r["caution"] = True  # v11-hold-fix: preserve red marker through detection gaps
                        r["point"] = cache["point"]
                        r["std_radius"] = cache["std_radius"]
                        r["ttc_sec"] = cache["ttc_sec"]
                        r["min_dist"] = cache["min_dist"]
                        active_danger_pairs.add(pair_key)

                cached_pairs_in_results = {
                    (r["ped_id"], r["veh_id"]) for r in risk_results
                }  # v11-hold-fix: post-dedupe danger upgrade
                for pair_key, cache in list(danger_visual_cache.items()):
                    if pair_key in cached_pairs_in_results or pair_key in active_danger_pairs:
                        continue
                    if (pair_key[0] in prev_boxes
                            and pair_key[1] in prev_boxes
                            and is_pedestrian_behind_vehicle(pair_key[0], pair_key[1])):
                        del danger_visual_cache[pair_key]  # v11-hold-fix: behind vehicle clears held danger
                        continue
                    if frame_count <= cache["expire"]:
                        risk_results.append({  # v11-hold-fix: post-dedupe danger upgrade
                            "ped_id": cache["ped_id"], "veh_id": cache["veh_id"],
                            "min_dist": cache["min_dist"], "ttc_sec": cache["ttc_sec"],
                            "danger": True, "caution": True,
                            "point": cache["point"],
                            "std_radius": cache["std_radius"],
                            "source": "hold",
                            "is_departing": False,
                        })

                for pair_key in list(danger_visual_cache.keys()):
                    if frame_count > danger_visual_cache[pair_key]["expire"]:
                        del danger_visual_cache[pair_key]  # v11-hold-fix: post-dedupe danger upgrade

                # Adaptive refine
                extra = refine_adaptive_pairs(lstm_session, risk_results, sample_index)
                mc_samples_used += extra
                if extra > 0:
                    for r in risk_results:
                        if r.get("source") in ("bbox", "hold"):  # v11-hold-fix: post-dedupe danger upgrade
                            continue
                        p_id, v_id = r["ped_id"], r["veh_id"]
                        if p_id in predictions_raw_mean and v_id in predictions_raw_mean:
                            bbox_size_p = bbox_size_for_track(p_id)
                            bbox_size_v = bbox_size_for_track(v_id)
                            min_dist, col_frame, ttc_sec, is_departing = calc_ttc_with_uncertainty(
                                predictions_raw_mean[p_id], predictions_raw_std[p_id],
                                predictions_raw_mean[v_id], predictions_raw_std[v_id],
                                bbox_size_p, bbox_size_v,
                            )
                            r["min_dist"] = min_dist
                            r["ttc_sec"] = ttc_sec
                            r["danger"] = col_frame < PRED_LENGTH and ttc_sec < COLLISION_TTC
                            behind_vehicle = is_pedestrian_behind_vehicle(p_id, v_id)  # v12: faster EMA + behind caution suppress
                            r["caution"] = False if (is_departing or behind_vehicle) else (r["danger"] or min_dist <= COLLISION_DIST * 1.5)
                            r["is_departing"] = bool(is_departing)
                            if behind_vehicle:
                                r["danger"] = False
                                r["caution"] = False
                            if p_id in prev_boxes and v_id in prev_boxes:
                                actual_dist = math.hypot(
                                    float(prev_boxes[p_id][0] - prev_boxes[v_id][0]),
                                    float(prev_boxes[p_id][1] - prev_boxes[v_id][1])
                                )  # v7: debug log + actual distance safety gate
                                r["actual_dist"] = actual_dist
                                if r["danger"] and actual_dist > 180.0:  # v8: tighten collision dist
                                    r["danger"] = False
                                    r["caution"] = False
                            if min_dist < min_dist_global:
                                min_dist_global = min_dist
                                ttc_sec_global = ttc_sec

                for r in risk_results:
                    pair_key = (r["ped_id"], r["veh_id"])  # v11-hold-fix: behind final danger gate
                    if (pair_key[0] in prev_boxes
                            and pair_key[1] in prev_boxes
                            and is_pedestrian_behind_vehicle(pair_key[0], pair_key[1])):
                        r["danger"] = False  # v11-hold-fix: behind final danger gate
                        r["caution"] = False  # v11-hold-fix: behind final danger gate
                        danger_visual_cache.pop(pair_key, None)  # v11-hold-fix: behind final danger gate

                if any(r["danger"] for r in risk_results):
                    collision_streak += 1
                else:
                    collision_streak = 0
                collision_risk = collision_streak >= COLLISION_HOLD
                risk_results.sort(key=lambda r: (not r["danger"], r["ttc_sec"]))
                if risk_results:
                    last_ttc_sec = float(risk_results[0]["ttc_sec"])
                    last_min_dist = float(risk_results[0]["min_dist"])
                    ttc_sec_global = last_ttc_sec
                    min_dist_global = last_min_dist
                else:
                    last_ttc_sec = PRED_LENGTH * DT
                    last_min_dist = float('inf')
                    ttc_sec_global = last_ttc_sec
                    min_dist_global = last_min_dist
            else:
                # 비샘플: 이전 risk 결과와 UI 값을 유지
                if collision_streak >= COLLISION_HOLD and risk_results:
                    collision_risk = True
                    ttc_sec_global = last_ttc_sec
                    min_dist_global = last_min_dist

            timer.mark("ttc")

            # ④ 시각화
            vis = make_visual_frame(frame)
            draw_roi(vis)

            if not is_sample:
                # v11-hold-fix: 비샘플에서도 behind-vehicle 체크로 지나간 차량 즉시 해제
                reuse_filtered = []
                for r in risk_results:
                    pid, vid = r.get("ped_id"), r.get("veh_id")
                    pair_key = (pid, vid)
                    if (pid in prev_boxes and vid in prev_boxes
                            and is_pedestrian_behind_vehicle(pid, vid)):
                        danger_visual_cache.pop(pair_key, None)
                        continue
                    if (pair_key in danger_visual_cache
                            and frame_count <= danger_visual_cache[pair_key]["expire"]) \
                       or (pid in prev_boxes and vid in prev_boxes):
                        reuse_filtered.append(r)
                risk_results = reuse_filtered
                if not risk_results:
                    collision_risk = False
                    ttc_sec_global = PRED_LENGTH * DT
                    min_dist_global = float('inf')

            risk_tids = set()
            danger_tids = set()
            for r in risk_results:
                risk_tids.add(r["ped_id"])
                risk_tids.add(r["veh_id"])
                if r["danger"]:
                    danger_tids.add(r["ped_id"])
                    danger_tids.add(r["veh_id"])

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
                draw_prediction(vis, track_id, level)

            for r in risk_results[:3]:
                if r["danger"] or (r["caution"] and r["ttc_sec"] <= 1.0):
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
