"""
A-PAS Main System — 30FPS Residual LSTM (Optimized)
=====================================================
최적화: YOLO 3프레임 샘플링 + IoU 기반 자체 트래킹
  → ByteTrack 제거, persist=False
  → 비샘플 프레임: 이전 detection 재사용 (그리기만)
  → YOLO 연산량 약 1/3로 감소

핵심 파이프라인:
  CARLA/Camera → YOLOv8 (3프레임마다) → IoU Matching
  → 17-Feature 추출 → Residual LSTM (ONNX, CPU)
  → TTC 충돌 판정 → 경고 출력

배포 환경: Raspberry Pi 5 + Hailo-8 AI HAT+
모델: Residual LSTM (ADE 2.08px, 10FPS, PRED=1초)
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import sys
from collections import deque, defaultdict, Counter
from ultralytics import YOLO

try:
    sys.path.insert(0, "ai_model/trajectory/mcdropout")
    from mc_inference import run_lstm_with_uncertainty
except ImportError:
    sys.path.insert(0, r"C:/Users/USER/Desktop/UIJIN_Workfolder/workplace/A-PAS/ai_model/trajectory/mcdropout")
    try:
        from mc_inference import run_lstm_with_uncertainty
    except ImportError as exc:
        print("❌ mc_inference import 실패. PYTHONPATH 또는 models/mcdropout 경로를 확인하세요.")
        raise exc

# ==========================================
# ⚙️ [설정] 10FPS 모델 기준
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH      = "test_video5.mp4"
YOLO_MODEL_PATH = "best_v6.pt"
ONNX_MODEL_PATH = "ai_model/trajectory/mcdropout/residual_mcd_10fps.onnx"

# --- 해상도 ---
IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

# --- 모델 설정 ---
SEQ_LENGTH   = 20
PRED_LENGTH  = 10
INPUT_SIZE   = 17
DT           = 0.1

# --- 영상 → 모델 샘플링 ---
VIDEO_FPS       = 30
TARGET_FPS      = 10
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # = 3

# --- 속도 스무딩 ---
SMOOTH_WINDOW = 3

# --- 물리적 클리핑 (정규화 단위) ---
VEL_CLIP = 2.0
ACC_CLIP = 1.0

# --- 충돌 판정 ---
COLLISION_DIST = 80
COLLISION_TTC  = 3

# MC Dropout 설정
MC_N_SAMPLES = 20
MC_USE_BATCH = True
SIGMA_SCALE  = 8.0

# 예측 시각화 안정화
EMA_ALPHA       = 0.4
EMA_ALPHA_STD   = 0.6
WARMUP_HIDE     = 5
WARMUP_FADE     = 15
WARMUP_TTC_GATE = 5

# 충돌 hysteresis
COLLISION_HOLD = 2

# TTC 페어링용
VEHICLE_CLASSES = {1, 2, 3}

# 클래스별 confidence threshold
CONF_THRESHOLD = {
    0: 0.35, 1: 0.25, 2: 0.30, 3: 0.25
}

# --- 클래스 매핑 ---
CLASS_MAP   = {0: 0, 1: 1, 2: 2, 3: 3}
CLASS_NAMES = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}

# ==========================================
# 🆕 [IoU 기반 자체 트래커]
# ==========================================
class SimpleIoUTracker:
    """
    ByteTrack 대신 사용하는 경량 IoU 기반 트래커.
    - 이전 프레임 bbox와 현재 bbox 간 IoU로 매칭
    - Greedy 매칭 (IoU 높은 순) + 중심거리 fallback
    - 매칭 안 되면 새 ID 할당
    """
    def __init__(self, iou_threshold=0.25, max_lost=15):
        self.next_id = 1
        self.tracks = {}       # {track_id: {"box": [x,y,w,h], "class_id": int, "lost": int}}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def _iou(self, box_a, box_b):
        """xywh 포맷 IoU 계산"""
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax1, ay1, ax2, ay2 = ax - aw/2, ay - ah/2, ax + aw/2, ay + ah/2
        bx1, by1, bx2, by2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    def _center_dist(self, box_a, box_b):
        """중심점 거리 (IoU 실패 시 fallback)"""
        return np.sqrt((box_a[0] - box_b[0])**2 + (box_a[1] - box_b[1])**2)

    def update(self, detections):
        """
        detections: list of (box, class_id, conf)
            box = [x, y, w, h] (xywh 중심좌표)

        Returns: list of (box, track_id, class_id, conf)
        """
        if not detections:
            to_remove = []
            for tid in self.tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return []

        # --- Greedy IoU 매칭 ---
        det_boxes = [d[0] for d in detections]
        track_ids_list = list(self.tracks.keys())

        iou_matrix = np.zeros((len(track_ids_list), len(det_boxes)))
        for i, tid in enumerate(track_ids_list):
            for j, dbox in enumerate(det_boxes):
                iou_matrix[i, j] = self._iou(self.tracks[tid]["box"], dbox)

        matched_tracks = set()
        matched_dets = set()
        results = [None] * len(detections)

        while True:
            if iou_matrix.size == 0:
                break
            max_val = iou_matrix.max()
            if max_val < self.iou_threshold:
                break

            idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            t_idx, d_idx = idx[0], idx[1]

            tid = track_ids_list[t_idx]
            box, class_id, conf = detections[d_idx]

            self.tracks[tid]["box"] = box
            self.tracks[tid]["class_id"] = class_id
            self.tracks[tid]["lost"] = 0
            results[d_idx] = (box, tid, class_id, conf)

            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        # --- 매칭 안 된 detection: 중심 거리 fallback ---
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_tracks = [i for i in range(len(track_ids_list)) if i not in matched_tracks]

        if unmatched_dets and unmatched_tracks:
            for d_idx in list(unmatched_dets):
                best_dist = 150
                best_t_idx = None
                dbox = det_boxes[d_idx]

                for t_idx in unmatched_tracks:
                    tid = track_ids_list[t_idx]
                    dist = self._center_dist(self.tracks[tid]["box"], dbox)
                    if dist < best_dist:
                        best_dist = dist
                        best_t_idx = t_idx

                if best_t_idx is not None:
                    tid = track_ids_list[best_t_idx]
                    box, class_id, conf = detections[d_idx]
                    self.tracks[tid]["box"] = box
                    self.tracks[tid]["class_id"] = class_id
                    self.tracks[tid]["lost"] = 0
                    results[d_idx] = (box, tid, class_id, conf)

                    unmatched_tracks.remove(best_t_idx)
                    unmatched_dets.remove(d_idx)

        # --- 새 트랙 생성 ---
        for d_idx in unmatched_dets:
            box, class_id, conf = detections[d_idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "box": box,
                "class_id": class_id,
                "lost": 0
            }
            results[d_idx] = (box, tid, class_id, conf)

        # --- lost 증가 + stale 제거 ---
        for t_idx in unmatched_tracks:
            tid = track_ids_list[t_idx]
            self.tracks[tid]["lost"] += 1

        to_remove = [tid for tid, t in self.tracks.items() if t["lost"] > self.max_lost]
        for tid in to_remove:
            del self.tracks[tid]

        return [r for r in results if r is not None]

    def get_active_ids(self):
        return set(self.tracks.keys())


# ==========================================
# 🧠 [모델 로드]
# ==========================================
print("📦 모델 로드 중...")
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]
)
print(f"  ✅ YOLO: {YOLO_MODEL_PATH}")
print(f"  ✅ LSTM: {ONNX_MODEL_PATH}")
print(f"  🖥️  Providers: {lstm_session.get_providers()}")

# ==========================================
# 📊 [트래킹 상태]
# ==========================================
tracker = SimpleIoUTracker(iou_threshold=0.25, max_lost=15)

track_history        = {}
prev_boxes           = {}
predictions_raw_mean = {}
predictions_raw_std  = {}
predictions          = {}
predictions_std      = {}
pred_frame_count     = {}
collision_streak     = 0
velocity_history     = {}
prev_velocity        = {}
track_classes        = {}
class_votes          = defaultdict(lambda: deque(maxlen=10))
prev_frames          = {}

# 비샘플 프레임용 캐시
last_tracked = []  # [(box, track_id, class_id, conf), ...]

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

    vx = float(np.mean([v[0] for v in velocity_history[track_id]]))
    vy = float(np.mean([v[1] for v in velocity_history[track_id]]))

    if track_id in prev_velocity:
        ax = (vx - prev_velocity[track_id][0]) / DT
        ay = (vy - prev_velocity[track_id][1]) / DT
    else:
        ax, ay = 0.0, 0.0
    prev_velocity[track_id] = [vx, vy]

    speed   = np.sqrt(vx**2 + vy**2)
    heading = np.arctan2(vy, vx)

    physics = np.array([
        x / IMG_W,   y / IMG_H,
        np.clip(vx / IMG_W, -VEL_CLIP, VEL_CLIP),
        np.clip(vy / IMG_H, -VEL_CLIP, VEL_CLIP),
        np.clip(ax / IMG_W, -ACC_CLIP, ACC_CLIP),
        np.clip(ay / IMG_H, -ACC_CLIP, ACC_CLIP),
        speed / DIAG,
        np.sin(heading), np.cos(heading),
        w / IMG_W,   h / IMG_H,
        (w * h) / (IMG_W * IMG_H)
    ], dtype=np.float32)

    return physics


def add_to_history(track_id, box, class_id, prev_box):
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=SEQ_LENGTH)

    physics = compute_features(box, prev_box, track_id)
    one_hot = np.zeros(5, dtype=np.float32)
    if class_id in CLASS_MAP:
        one_hot[CLASS_MAP[class_id]] = 1.0

    track_history[track_id].append(
        np.concatenate([physics, one_hot])
    )
    track_classes[track_id] = class_id


def run_lstm_mc(track_id):
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None, None

    seq = np.array(track_history[track_id], dtype=np.float32)
    seq = seq[np.newaxis, ...]

    mean, std = run_lstm_with_uncertainty(
        lstm_session, seq,
        n_samples=MC_N_SAMPLES,
        use_batch=MC_USE_BATCH,
    )

    mean_px = mean.copy()
    std_px  = std.copy()
    mean_px[..., 0] *= IMG_W
    mean_px[..., 1] *= IMG_H
    std_px[..., 0]  *= IMG_W
    std_px[..., 1]  *= IMG_H

    return mean_px, std_px


def update_prediction(track_id, mean_new, std_new):
    predictions_raw_mean[track_id] = mean_new
    predictions_raw_std[track_id]  = std_new

    if track_id not in predictions:
        predictions[track_id]     = mean_new.copy()
        predictions_std[track_id] = std_new.copy()
        pred_frame_count[track_id] = 1
        return

    def shift_blend(prev, curr, alpha):
        shifted = np.empty_like(prev)
        shifted[:-1] = prev[1:]
        shifted[-1]  = prev[-1] + (prev[-1] - prev[-2])
        return alpha * curr + (1 - alpha) * shifted

    predictions[track_id] = shift_blend(
        predictions[track_id], mean_new, EMA_ALPHA
    )
    predictions_std[track_id] = shift_blend(
        predictions_std[track_id], std_new, EMA_ALPHA_STD
    )
    pred_frame_count[track_id] += 1


# ==========================================
# ⚠️ [TTC 계산]
# ==========================================
def calc_ttc_with_uncertainty(mean_a, std_a, mean_b, std_b):
    min_dist        = float('inf')
    collision_frame = PRED_LENGTH

    for t in range(PRED_LENGTH):
        dist = np.linalg.norm(mean_a[t] - mean_b[t])
        sigma_a = SIGMA_SCALE * float(np.linalg.norm(std_a[t])) / np.sqrt(2)
        sigma_b = SIGMA_SCALE * float(np.linalg.norm(std_b[t])) / np.sqrt(2)
        safety_margin = COLLISION_DIST + sigma_a + sigma_b

        if dist < min_dist:
            min_dist = dist
        if dist < safety_margin:
            collision_frame = t
            break

    ttc_seconds = collision_frame * DT
    return min_dist, collision_frame, ttc_seconds


def find_pedestrian_vehicle_pairs(pred_dict, track_classes):
    pairs = []
    pids = [tid for tid in pred_dict if track_classes.get(tid) == 0]
    vids = [tid for tid in pred_dict if track_classes.get(tid) in VEHICLE_CLASSES]
    for p in pids:
        for v in vids:
            pairs.append((p, v))
    return pairs


# ==========================================
# 🎨 [시각화]
# ==========================================
def draw_predictions(frame, mean, std, n_pred, color=(0, 255, 255)):
    """
    경량 시각화: frame.copy()/addWeighted 제거 → cv2.line/circle만 사용.
    - 중앙선: 실선 (두께 점진 감소)
    - 불확실성 경계: 점선 (상/하 sigma 경계)
    - 끝점: 원 + 불확실성 원
    메모리 복사 0회 → 비샘플 프레임에서도 부담 없음.
    """
    if n_pred < WARMUP_HIDE:
        return

    n = len(mean)
    if n < 2:
        return

    # --- 중앙선 (실선, 두께 점진 감소) ---
    for t in range(n - 1):
        pt1 = tuple(mean[t].astype(int))
        pt2 = tuple(mean[t + 1].astype(int))
        # 가까운 미래 = 두꺼움, 먼 미래 = 얇음
        thickness = max(1, 3 - (t * 2 // n))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # --- 불확실성 경계 (점선) ---
    for t in range(n):
        # 방향 벡터 계산
        if t == 0:
            direction = mean[1] - mean[0]
        elif t == n - 1:
            direction = mean[-1] - mean[-2]
        else:
            direction = mean[t + 1] - mean[t - 1]

        norm_val = np.linalg.norm(direction)
        if norm_val < 1e-6:
            perpendicular = np.array([0.0, 1.0])
        else:
            perpendicular = np.array([-direction[1], direction[0]]) / norm_val

        sigma = SIGMA_SCALE * float(np.linalg.norm(std[t]))
        sigma = np.clip(sigma, 5, 200)

        upper = (mean[t] + sigma * perpendicular).astype(int)
        lower = (mean[t] - sigma * perpendicular).astype(int)

        # 점선 효과: 짝수 인덱스만 그리기
        if t % 2 == 0:
            # 경계 틱 마크 (짧은 선분)
            center_pt = tuple(mean[t].astype(int))
            cv2.line(frame, center_pt, tuple(upper), color, 1, cv2.LINE_AA)
            cv2.line(frame, center_pt, tuple(lower), color, 1, cv2.LINE_AA)

        # 경계선 연결 (상/하 각각)
        if t < n - 1:
            direction_next = mean[t + 1] - mean[t] if t < n - 1 else direction
            norm_next = np.linalg.norm(direction_next)
            if norm_next < 1e-6:
                perp_next = perpendicular
            else:
                perp_next = np.array([-direction_next[1], direction_next[0]]) / norm_next

            sigma_next = SIGMA_SCALE * float(np.linalg.norm(std[t + 1]))
            sigma_next = np.clip(sigma_next, 5, 200)

            upper_next = (mean[t + 1] + sigma_next * perp_next).astype(int)
            lower_next = (mean[t + 1] - sigma_next * perp_next).astype(int)

            # 점선: 2스텝마다 그리기
            if t % 2 == 0:
                cv2.line(frame, tuple(upper), tuple(upper_next), color, 1, cv2.LINE_AA)
                cv2.line(frame, tuple(lower), tuple(lower_next), color, 1, cv2.LINE_AA)

    # --- 끝점 표시 ---
    # 중앙점 (실점)
    cv2.circle(frame, tuple(mean[-1].astype(int)), 5, color, -1)

    # 불확실성 원 (빈 원)
    end_sigma = SIGMA_SCALE * float(np.linalg.norm(std[-1]))
    end_sigma = int(np.clip(end_sigma, 8, 150))
    cv2.circle(frame, tuple(mean[-1].astype(int)), end_sigma, color, 1, cv2.LINE_AA)


CLASS_COLORS = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}

def draw_bbox(frame, box, track_id, class_id, conf=None):
    x, y, w, h = box
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cls_name = CLASS_NAMES.get(class_id, "?")
    color = CLASS_COLORS.get(class_id, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} [{cls_name}]"
    if conf is not None:
        label += f" {conf:.2f}"
    cv2.putText(frame, label,
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_alert(frame, ttc_sec, min_dist):
    cv2.rectangle(frame, (0, 0), (500, 90), (0, 0, 200), -1)
    cv2.putText(frame, "!! COLLISION WARNING !!",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"TTC: {ttc_sec:.1f}s | Dist: {min_dist:.0f}px",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_status(frame, fps, latency_ms, is_sample, n_tracks, n_preds):
    tag = "[YOLO+LSTM]" if is_sample else "[reuse]    "
    info = (f"{tag} FPS:{fps:.0f} | Lat:{latency_ms:.0f}ms | "
            f"Tracks:{n_tracks} | Preds:{n_preds}")
    cv2.putText(frame, info, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


# ==========================================
# 🔌 [경고 출력]
# ==========================================
def trigger_alert(collision_risk, ttc_sec):
    if collision_risk:
        print(f"  🚨 ALERT! TTC={ttc_sec:.1f}s")


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
        track_history.pop(tid, None)
        prev_boxes.pop(tid, None)
        predictions.pop(tid, None)
        predictions_std.pop(tid, None)
        predictions_raw_mean.pop(tid, None)
        predictions_raw_std.pop(tid, None)
        pred_frame_count.pop(tid, None)
        velocity_history.pop(tid, None)
        prev_velocity.pop(tid, None)
        track_classes.pop(tid, None)
        prev_frames.pop(tid, None)
        class_votes.pop(tid, None)


def merge_duplicate_tracks(predictions, prev_boxes, merge_dist=100):
    ids = list(predictions.keys())
    to_remove = set()
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if ids[i] in prev_boxes and ids[j] in prev_boxes:
                box_i = prev_boxes[ids[i]]
                box_j = prev_boxes[ids[j]]
                dist = np.sqrt((box_i[0]-box_j[0])**2 + (box_i[1]-box_j[1])**2)
                if dist < merge_dist:
                    to_remove.add(max(ids[i], ids[j]))
    for tid in to_remove:
        predictions.pop(tid, None)
        predictions_std.pop(tid, None)
        predictions_raw_mean.pop(tid, None)
        predictions_raw_std.pop(tid, None)
        pred_frame_count.pop(tid, None)


# ==========================================
# 🎬 [메인 루프]
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
    exit()

print(f"\n🚀 A-PAS 시스템 시작! (Optimized — YOLO 샘플링)")
print(f"   모델: Residual LSTM (10FPS, SEQ={SEQ_LENGTH}, PRED={PRED_LENGTH})")
print(f"   ⚡ YOLO: {SAMPLE_INTERVAL}프레임마다 실행 (persist=False, IoU 트래킹)")
print(f"   📐 비샘플 프레임: 이전 detection 재사용 (그리기만)")
print(f"   관찰: {SEQ_LENGTH * DT:.1f}초 → 예측: {PRED_LENGTH * DT:.1f}초")
print(f"   속도 스무딩: {SMOOTH_WINDOW}프레임 이동평균")
print(f"   충돌 기준: 거리 {COLLISION_DIST}px, TTC {COLLISION_TTC}초")
print(f"   종료: 'q' 키\n")
print(f"🔬 Phase 3 활성화:")
print(f"   MC Sampling: n={MC_N_SAMPLES}, batch={MC_USE_BATCH}")
print(f"   부채꼴: SIGMA_SCALE={SIGMA_SCALE}")
print(f"   Warmup: {WARMUP_HIDE}~{WARMUP_FADE} 프레임")
print(f"   Hysteresis: {COLLISION_HOLD} 프레임 연속 위험 시 경보")

latency_list     = []
frame_count      = 0
yolo_call_count  = 0
cleanup_interval = 30

try:
    while cap.isOpened():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count    += 1
        is_sample_frame = (frame_count % SAMPLE_INTERVAL == 0)

        collision_risk  = False
        min_dist_global = float('inf')
        ttc_global      = PRED_LENGTH
        ttc_sec_global  = PRED_LENGTH * DT
        active_ids      = set()

        # ======================================
        # ✅ 샘플 프레임: YOLO + IoU 매칭 + LSTM
        # ======================================
        if is_sample_frame:
            # YOLO detection만 (persist=False, ByteTrack 없음)
            results = yolo_model(
                frame,
                imgsz=1280,
                classes=[0, 1, 2, 3],
                conf=0.1,
                iou=0.3,
                verbose=False
            )
            yolo_call_count += 1

            # detection 결과 수집
            detections = []
            if results and results[0].boxes and len(results[0].boxes) > 0:
                boxes_raw = results[0].boxes.xywh.cpu().numpy()
                cls_raw   = results[0].boxes.cls.int().cpu().tolist()
                conf_raw  = results[0].boxes.conf.cpu().numpy()

                for box, cls_id, conf in zip(boxes_raw, cls_raw, conf_raw):
                    x, y, w, h = box
                    if cls_id == 0 and w * h < 400:
                        continue
                    if conf < CONF_THRESHOLD.get(int(cls_id), 0.25):
                        continue
                    detections.append((box, int(cls_id), float(conf)))

            # IoU 기반 트래킹
            tracked = tracker.update(detections)
            last_tracked = tracked
            active_ids = tracker.get_active_ids()

            for box, track_id, class_id, conf in tracked:
                class_votes[track_id].append(int(class_id))
                voted_class_id = Counter(class_votes[track_id]).most_common(1)[0][0]

                # 갭 체크
                if track_id in prev_frames:
                    frame_diff = frame_count - prev_frames[track_id]
                    if frame_diff > 5 * SAMPLE_INTERVAL:
                        if track_id in velocity_history:
                            velocity_history[track_id].clear()
                        if track_id in prev_velocity:
                            prev_velocity[track_id] = [0.0, 0.0]
                        if track_id in track_history:
                            track_history[track_id].clear()
                prev_frames[track_id] = frame_count

                # 피처 추출 + LSTM
                prev_box = prev_boxes.get(track_id, None)
                add_to_history(track_id, box, voted_class_id, prev_box)
                prev_boxes[track_id] = box

                mean, std = run_lstm_mc(track_id)
                if mean is not None:
                    update_prediction(track_id, mean, std)

                # 시각화
                draw_bbox(frame, box, track_id, voted_class_id, conf)
                if track_id in predictions:
                    n_pred = pred_frame_count.get(track_id, 0)
                    draw_predictions(
                        frame,
                        predictions[track_id],
                        predictions_std[track_id],
                        n_pred
                    )

            merge_duplicate_tracks(predictions, prev_boxes)

            # TTC 계산
            if len(predictions_raw_mean) >= 2:
                valid_ids = {
                    tid for tid in predictions_raw_mean
                    if pred_frame_count.get(tid, 0) >= WARMUP_TTC_GATE
                }
                valid_preds = {tid: predictions_raw_mean[tid] for tid in valid_ids}
                pairs = find_pedestrian_vehicle_pairs(valid_preds, track_classes)
                pair_risk = False

                for p_id, v_id in pairs:
                    min_dist, col_frame, ttc_sec = calc_ttc_with_uncertainty(
                        predictions_raw_mean[p_id], predictions_raw_std[p_id],
                        predictions_raw_mean[v_id], predictions_raw_std[v_id],
                    )
                    if min_dist < min_dist_global:
                        min_dist_global = min_dist
                        ttc_global      = col_frame
                        ttc_sec_global  = ttc_sec
                    if col_frame < PRED_LENGTH and ttc_sec < COLLISION_TTC:
                        pair_risk = True

                if pair_risk:
                    collision_streak += 1
                else:
                    collision_streak = 0

                if collision_streak >= COLLISION_HOLD:
                    collision_risk = True
            else:
                collision_streak = 0

        # ======================================
        # ✅ 비샘플 프레임: 이전 결과 재사용
        # ======================================
        else:
            active_ids = tracker.get_active_ids()

            for item in last_tracked:
                box, track_id, class_id, conf = item
                voted_class_id = class_id
                if track_id in class_votes and len(class_votes[track_id]) > 0:
                    voted_class_id = Counter(class_votes[track_id]).most_common(1)[0][0]

                draw_bbox(frame, box, track_id, voted_class_id, conf)
                if track_id in predictions:
                    n_pred = pred_frame_count.get(track_id, 0)
                    draw_predictions(
                        frame,
                        predictions[track_id],
                        predictions_std[track_id],
                        n_pred
                    )

            # TTC 경고 유지
            if len(predictions_raw_mean) >= 2:
                valid_ids = {
                    tid for tid in predictions_raw_mean
                    if pred_frame_count.get(tid, 0) >= WARMUP_TTC_GATE
                }
                valid_preds = {tid: predictions_raw_mean[tid] for tid in valid_ids}
                pairs = find_pedestrian_vehicle_pairs(valid_preds, track_classes)

                for p_id, v_id in pairs:
                    min_dist, col_frame, ttc_sec = calc_ttc_with_uncertainty(
                        predictions_raw_mean[p_id], predictions_raw_std[p_id],
                        predictions_raw_mean[v_id], predictions_raw_std[v_id],
                    )
                    if min_dist < min_dist_global:
                        min_dist_global = min_dist
                        ttc_global      = col_frame
                        ttc_sec_global  = ttc_sec

                if collision_streak >= COLLISION_HOLD:
                    collision_risk = True

        # ⑤ 경고 출력
        if collision_risk:
            draw_alert(frame, ttc_sec_global, min_dist_global)
        trigger_alert(collision_risk, ttc_sec_global)

        # ⑥ 트랙 정리
        if frame_count % cleanup_interval == 0:
            cleanup_stale_tracks(active_ids, frame_count, max_lost=60)

        # 성능 표시
        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        latency_list.append(latency_ms)
        fps = 1000 / latency_ms if latency_ms > 0 else 0

        draw_status(frame, fps, latency_ms, is_sample_frame,
                    len(track_history), len(predictions))

        cv2.imshow("A-PAS Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # ==========================================
    # 📊 [최종 성능 통계]
    # ==========================================
    if latency_list:
        arr = np.array(latency_list)

        sample_indices = np.arange(SAMPLE_INTERVAL - 1, len(arr), SAMPLE_INTERVAL)
        reuse_indices  = np.setdiff1d(np.arange(len(arr)), sample_indices)

        sample_latencies = arr[sample_indices] if len(sample_indices) > 0 else np.array([])
        reuse_latencies  = arr[reuse_indices]  if len(reuse_indices) > 0  else np.array([])

        print(f"\n{'='*60}")
        print(f"📊 A-PAS 최종 성능 통계 (YOLO 샘플링 + IoU 트래킹)")
        print(f"{'='*60}")
        print(f"  모델              : Residual LSTM (10FPS, MC Dropout)")
        print(f"  관찰/예측         : {SEQ_LENGTH * DT:.1f}초 → {PRED_LENGTH * DT:.1f}초")
        print(f"  처리 프레임        : {frame_count:,}장")
        print(f"  YOLO 실행 횟수     : {yolo_call_count:,}회 "
              f"(전체의 {yolo_call_count/max(1,frame_count)*100:.1f}%)")
        print(f"  ──────────────────────────────────────")
        print(f"  [전체]  평균 Latency : {arr.mean():.2f}ms")
        print(f"  [전체]  평균 FPS     : {1000/arr.mean():.1f}")
        print(f"  ──────────────────────────────────────")
        if len(sample_latencies) > 0:
            print(f"  [샘플]  평균 Latency : {sample_latencies.mean():.2f}ms (YOLO+LSTM)")
        if len(reuse_latencies) > 0:
            print(f"  [재사용] 평균 Latency: {reuse_latencies.mean():.2f}ms (bbox 재사용)")
        print(f"  ──────────────────────────────────────")
        print(f"  최소 Latency       : {arr.min():.2f}ms")
        print(f"  최대 Latency       : {arr.max():.2f}ms")
        print(f"{'='*60}")