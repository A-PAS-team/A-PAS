"""
A-PAS Main System — 30FPS Residual LSTM
========================================
핵심 파이프라인:
  CARLA/Camera → YOLOv8 (Hailo NPU) → ByteTrack
  → 17-Feature 추출 → Residual LSTM (ONNX, CPU)
  → TTC 충돌 판정 → 경고 출력

배포 환경: Raspberry Pi 5 + Hailo-8 AI HAT+
모델: Residual LSTM (ADE 2.08px, 30FPS, PRED=1초)
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import sys  # ✨ NEW
from collections import deque
from ultralytics import YOLO
from collections import defaultdict, deque, Counter

try:  # ✨ NEW
    sys.path.insert(0, "models/mcdropout")  # ✨ NEW
    from mc_inference import run_lstm_with_uncertainty  # ✨ NEW
except ImportError:  # ✨ NEW
    sys.path.insert(0, r"C:\Users\dkdld\A-PAS\models\mcdropout")  # ✨ NEW
    try:  # ✨ NEW
        from mc_inference import run_lstm_with_uncertainty  # ✨ NEW
    except ImportError as exc:  # ✨ NEW
        print("❌ mc_inference import 실패. PYTHONPATH 또는 models/mcdropout 경로를 확인하세요.")  # ✨ NEW
        raise exc  # ✨ NEW

# ==========================================
# ⚙️ [설정] 30FPS Residual LSTM
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH      = "test_video5.mp4"       # 또는 CARLA HDMI 캡처
YOLO_MODEL_PATH = "best_v6.pt"         # Hailo 사용 시 .hef로 교체
ONNX_MODEL_PATH = "models/mcdropout/residual_mcd.onnx"  # ✨ MODIFIED

# --- 해상도 ---
IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

# --- 모델 설정 (30FPS) ---
SEQ_LENGTH   = 60          # ✨ MODIFIED: 2초 관찰 (MC Dropout 모델 기준)
PRED_LENGTH  = 30          # ✨ MODIFIED: 1초 예측
INPUT_SIZE   = 17
DT           = 1 / 30      # 30FPS

# --- 영상 → 모델 샘플링 ---
VIDEO_FPS       = 30
TARGET_FPS      = 30       # 30FPS 모델이므로 매 프레임 처리
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # = 1

# --- 속도 스무딩 ---
SMOOTH_WINDOW = 3          # 30FPS에서는 3프레임 = 0.1초

# --- 물리적 클리핑 (정규화 단위) ---
VEL_CLIP = 2.0             # vx, vy 클리핑
ACC_CLIP = 1.0             # ax, ay 클리핑

# --- 충돌 판정 ---
COLLISION_DIST = 80        # 픽셀 거리 임계값
COLLISION_TTC  = 3         # 초

# ✨ NEW: MC Dropout 설정
MC_N_SAMPLES = 5
MC_USE_BATCH = True
SIGMA_SCALE = 2.0

# ✨ NEW: 예측 시각화 안정화
EMA_ALPHA = 0.4
EMA_ALPHA_STD = 0.6
WARMUP_HIDE = 5
WARMUP_FADE = 15
WARMUP_TTC_GATE = 5

# ✨ NEW: 충돌 hysteresis
COLLISION_HOLD = 2

# ✨ NEW: TTC 페어링용
VEHICLE_CLASSES = {1, 2, 3}

# ✨ NEW: 클래스별 confidence threshold
CONF_THRESHOLD = {
    0: 0.35, 1: 0.25, 2: 0.30, 3: 0.25
}

# --- 클래스 매핑 ---
CLASS_MAP    = {0: 0, 1: 1, 2: 2, 3: 3}
CLASS_NAMES  = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}

class_votes = defaultdict(lambda: deque(maxlen=10))
prev_frames = {}  # 갭 체크를 위해 track_id별 마지막 관측 프레임을 저장

# ==========================================
# 🧠 [모델 로드]
# ==========================================
print("📦 모델 로드 중...")
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CPUExecutionProvider"]  # RPi5: CPU 전용
)
print(f"  ✅ YOLO: {YOLO_MODEL_PATH}")
print(f"  ✅ LSTM: {ONNX_MODEL_PATH}")
print(f"  🖥️  Providers: {lstm_session.get_providers()}")

# ==========================================
# 📊 [트래킹 상태]
# ==========================================
track_history    = {}   # {track_id: deque([17-dim feature], maxlen=SEQ)}
prev_boxes       = {}   # {track_id: [x, y, w, h]}
predictions_raw_mean = {}   # ✨ NEW: 판정용 MC mean (EMA 없음)
predictions_raw_std  = {}   # ✨ NEW: 판정용 MC std (EMA 없음)
predictions          = {}   # ✨ MODIFIED: 시각화용 mean (EMA)
predictions_std      = {}   # ✨ NEW: 시각화용 std (EMA)
pred_frame_count     = {}   # ✨ NEW: warmup 카운터
collision_streak     = 0    # ✨ NEW: hysteresis 카운터
velocity_history = {}   # {track_id: deque([[vx, vy]])}
prev_velocity    = {}   # {track_id: [vx, vy]}
track_classes    = {}   # {track_id: class_id}

# ==========================================
# 🔧 [피처 추출] 학습 파이프라인과 일치
# ==========================================
def compute_features(box, prev_box, track_id):
    """
    현재/이전 박스에서 12개 물리 피처 생성.
    학습 시 csv_to_npy_30fps.py의 get_features()와 동일한 정규화 적용.
    """
    x, y, w, h = box

    # --- 속도 계산 ---
    if prev_box is not None:
        raw_vx = (x - prev_box[0]) / DT
        raw_vy = (y - prev_box[1]) / DT
    else:
        raw_vx, raw_vy = 0.0, 0.0

    # --- 속도 이동평균 스무딩 ---
    if track_id not in velocity_history:
        velocity_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
    velocity_history[track_id].append([raw_vx, raw_vy])

    vx = float(np.mean([v[0] for v in velocity_history[track_id]]))
    vy = float(np.mean([v[1] for v in velocity_history[track_id]]))

    # --- 가속도 계산 ---
    if track_id in prev_velocity:
        ax = (vx - prev_velocity[track_id][0]) / DT
        ay = (vy - prev_velocity[track_id][1]) / DT
    else:
        ax, ay = 0.0, 0.0
    prev_velocity[track_id] = [vx, vy]

    # --- 파생 물리량 ---
    speed   = np.sqrt(vx**2 + vy**2)
    heading = np.arctan2(vy, vx)

    # --- 정규화 + 클리핑 (학습 파이프라인과 동일) ---
    physics = np.array([
        x / IMG_W,                                    # 0: pos_x
        y / IMG_H,                                    # 1: pos_y
        np.clip(vx / IMG_W, -VEL_CLIP, VEL_CLIP),    # 2: vel_x
        np.clip(vy / IMG_H, -VEL_CLIP, VEL_CLIP),    # 3: vel_y
        np.clip(ax / IMG_W, -ACC_CLIP, ACC_CLIP),     # 4: acc_x
        np.clip(ay / IMG_H, -ACC_CLIP, ACC_CLIP),     # 5: acc_y
        speed / DIAG,                                  # 6: speed
        np.sin(heading),                               # 7: sin_h
        np.cos(heading),                               # 8: cos_h
        w / IMG_W,                                     # 9: width
        h / IMG_H,                                     # 10: height
        (w * h) / (IMG_W * IMG_H)                      # 11: area
    ], dtype=np.float32)

    return physics  # (12,)


def add_to_history(track_id, box, class_id, prev_box):
    """트래킹 히스토리에 17개 피처(12 물리 + 5 원-핫) 추가"""
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=SEQ_LENGTH)

    physics = compute_features(box, prev_box, track_id)

    one_hot = np.zeros(5, dtype=np.float32)
    if class_id in CLASS_MAP:
        one_hot[CLASS_MAP[class_id]] = 1.0

    track_history[track_id].append(
        np.concatenate([physics, one_hot])  # (17,)
    )
    track_classes[track_id] = class_id


# ✨ MODIFIED: 기존 단일 추론 함수는 롤백 대비 주석으로 유지
# def run_lstm(track_id):
#     """LSTM 추론 → 미래 경로 반환 (픽셀 단위)"""
#     if len(track_history[track_id]) < SEQ_LENGTH:
#         return None
#
#     seq = np.array(track_history[track_id], dtype=np.float32)
#     seq = seq[np.newaxis, ...]
#
#     output = lstm_session.run(None, {"input": seq})[0]
#     pred = output[0].copy()
#
#     pred[:, 0] *= IMG_W
#     pred[:, 1] *= IMG_H
#
#     return pred


def run_lstm_mc(track_id):
    """✨ MODIFIED: MC Dropout으로 mean + std 추론."""
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None, None

    seq = np.array(track_history[track_id], dtype=np.float32)
    seq = seq[np.newaxis, ...]  # ✨ MODIFIED: (1, 60, 17)

    mean, std = run_lstm_with_uncertainty(  # ✨ NEW
        lstm_session, seq,
        n_samples=MC_N_SAMPLES,
        use_batch=MC_USE_BATCH,
    )

    mean_px = mean.copy()  # ✨ NEW
    std_px = std.copy()  # ✨ NEW
    mean_px[..., 0] *= IMG_W  # ✨ NEW
    mean_px[..., 1] *= IMG_H  # ✨ NEW
    std_px[..., 0] *= IMG_W  # ✨ NEW
    std_px[..., 1] *= IMG_H  # ✨ NEW

    return mean_px, std_px


def update_prediction(track_id, mean_new, std_new):
    """✨ NEW: raw / smooth 분리 + 시간정렬 EMA."""
    predictions_raw_mean[track_id] = mean_new
    predictions_raw_std[track_id] = std_new

    if track_id not in predictions:
        predictions[track_id] = mean_new.copy()
        predictions_std[track_id] = std_new.copy()
        pred_frame_count[track_id] = 1
        return

    def shift_blend(prev, curr, alpha):
        """✨ NEW: 예측 시점이 한 프레임 앞으로 흐르도록 정렬 후 EMA."""
        shifted = np.empty_like(prev)
        shifted[:-1] = prev[1:]
        shifted[-1] = prev[-1] + (prev[-1] - prev[-2])
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
# ✨ MODIFIED: 기존 고정 마진 TTC는 롤백 대비 주석으로 유지
# def calc_ttc(pred_a, pred_b):
#     """두 객체의 예측 궤적 간 TTC 계산"""
#     min_dist = float('inf')
#     collision_frame = PRED_LENGTH
#
#     for t in range(PRED_LENGTH):
#         dist = np.sqrt(
#             (pred_a[t, 0] - pred_b[t, 0])**2 +
#             (pred_a[t, 1] - pred_b[t, 1])**2
#         )
#         if dist < min_dist:
#             min_dist = dist
#         if dist < COLLISION_DIST:
#             collision_frame = t
#             break
#
#     ttc_seconds = collision_frame * DT
#     return min_dist, collision_frame, ttc_seconds


def calc_ttc_with_uncertainty(mean_a, std_a, mean_b, std_b):
    """✨ MODIFIED: 부채꼴 영역 기반 충돌 판정."""
    min_dist = float('inf')
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
    """✨ NEW: 보행자-차량 쌍만 반환. Car-Car TTC는 우선순위 낮음."""
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
    """✨ MODIFIED: 반투명 채워진 부채꼴 + 중앙선 + warmup fade."""
    if n_pred < WARMUP_HIDE:
        return
    if n_pred < WARMUP_FADE:
        fade = (n_pred - WARMUP_HIDE) / max(1, WARMUP_FADE - WARMUP_HIDE)
        fade = max(0.1, fade)
    else:
        fade = 1.0

    n = len(mean)
    tiers = [
        (0,           n // 3,      0.30, 3),
        (n // 3,      2 * n // 3,  0.20, 2),
        (2 * n // 3,  n,           0.10, 1),
    ]

    for start, end, fill_alpha, line_thickness in tiers:
        upper_pts, lower_pts = [], []
        for t in range(start, min(end, n)):
            if t == 0:
                direction = mean[1] - mean[0] if n > 1 else np.array([1.0, 0.0])
            elif t == n - 1:
                direction = mean[-1] - mean[-2]
            else:
                direction = mean[t+1] - mean[t-1]

            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                perpendicular = np.array([0.0, 1.0])
            else:
                perpendicular = np.array([-direction[1], direction[0]]) / norm

            sigma = SIGMA_SCALE * float(np.mean(std[t]))
            sigma = np.clip(sigma, 2, 150)

            upper_pts.append(mean[t] + sigma * perpendicular)
            lower_pts.append(mean[t] - sigma * perpendicular)

        if not upper_pts:
            continue

        polygon = np.array(upper_pts + lower_pts[::-1], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(
            overlay, fill_alpha * fade,
            frame, 1 - fill_alpha * fade,
            0, frame
        )

        for t in range(start, min(end, n) - 1):
            pt1 = tuple(mean[t].astype(int))
            pt2 = tuple(mean[t+1].astype(int))
            cv2.line(frame, pt1, pt2, color, line_thickness, cv2.LINE_AA)

    end_alpha = 0.95 * fade
    if end_alpha > 0.05:
        overlay = frame.copy()
        cv2.circle(overlay, tuple(mean[-1].astype(int)), 5, color, -1)
        cv2.addWeighted(overlay, end_alpha, frame, 1 - end_alpha, 0, frame)


CLASS_COLORS = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}

def draw_bbox(frame, box, track_id, class_id, conf=None):
    """✨ MODIFIED: conf 파라미터 추가."""
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
    """위험 경고 UI"""
    cv2.rectangle(frame, (0, 0), (500, 90), (0, 0, 200), -1)
    cv2.putText(frame, "!! COLLISION WARNING !!",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, f"TTC: {ttc_sec:.1f}s | Dist: {min_dist:.0f}px",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_status(frame, fps, latency_ms, is_sample, n_tracks, n_preds):
    """상태 정보 표시"""
    tag = "[LSTM]" if is_sample else "      "
    info = (f"{tag} FPS:{fps:.0f} | Lat:{latency_ms:.0f}ms | "
            f"Tracks:{n_tracks} | Preds:{n_preds}")
    cv2.putText(frame, info, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ==========================================
# 🔌 [경고 출력] — GPIO 연동 시 여기만 수정
# ==========================================
def trigger_alert(collision_risk, ttc_sec):
    """
    경고 트리거 (LED/부저).
    현재는 콘솔 출력만. GPIO 연동 시 여기에 추가.
    
    TODO (Phase 2):
      import rpi_lgpio
      - LED strip: 위험 시 빨간색, 안전 시 초록색
      - 부저: 위험 시 ON
    """
    if collision_risk:
        print(f"  🚨 ALERT! TTC={ttc_sec:.1f}s")
    # else:
    #     LED → 초록, 부저 OFF

# ==========================================
# 🧹 [트랙 정리]
# ==========================================
def cleanup_stale_tracks(active_ids, frame_count, max_lost=60):
    """
    active_ids에 없더라도 최근 60프레임 이내면 유지
    → ByteTrack의 track_buffer와 동일한 기준
    """
    stale = [
        tid for tid in track_history
        if tid not in active_ids
        and (frame_count - prev_frames.get(tid, 0)) > max_lost
    ]
    for tid in stale:
        track_history.pop(tid, None)
        prev_boxes.pop(tid, None)
        predictions.pop(tid, None)
        predictions_std.pop(tid, None)         # ✨ NEW
        predictions_raw_mean.pop(tid, None)    # ✨ NEW
        predictions_raw_std.pop(tid, None)     # ✨ NEW
        pred_frame_count.pop(tid, None)        # ✨ NEW
        velocity_history.pop(tid, None)
        prev_velocity.pop(tid, None)
        track_classes.pop(tid, None)
        prev_frames.pop(tid, None)
        class_votes.pop(tid, None)
    

def merge_duplicate_tracks(predictions, prev_boxes, merge_dist=100):
    """같은 객체에 두 개 track_id 붙은 경우 하나만 남김"""
    ids = list(predictions.keys())
    to_remove = set()
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            if ids[i] in prev_boxes and ids[j] in prev_boxes:
                box_i = prev_boxes[ids[i]]
                box_j = prev_boxes[ids[j]]
                dist = np.sqrt((box_i[0]-box_j[0])**2 + (box_i[1]-box_j[1])**2)
                if dist < merge_dist:
                    to_remove.add(max(ids[i], ids[j]))  # 새 track 제거
    for tid in to_remove:
        predictions.pop(tid, None)
        predictions_std.pop(tid, None)         # ✨ NEW
        predictions_raw_mean.pop(tid, None)    # ✨ NEW
        predictions_raw_std.pop(tid, None)     # ✨ NEW
        pred_frame_count.pop(tid, None)        # ✨ NEW

# ==========================================
# 🎬 [메인 루프]
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
    exit()

print(f"\n🚀 A-PAS 시스템 시작!")
print(f"   모델: Residual LSTM (30FPS, SEQ={SEQ_LENGTH}, PRED={PRED_LENGTH})")
print(f"   관찰: {SEQ_LENGTH/30:.1f}초 → 예측: {PRED_LENGTH/30:.1f}초")
print(f"   속도 스무딩: {SMOOTH_WINDOW}프레임 이동평균")
print(f"   충돌 기준: 거리 {COLLISION_DIST}px, TTC {COLLISION_TTC}초")
print(f"   종료: 'q' 키\n")
print(f"\n🔬 Phase 3 활성화:")  # ✨ NEW
print(f"   MC Sampling: n={MC_N_SAMPLES}, batch={MC_USE_BATCH}")  # ✨ NEW
print(f"   부채꼴: SIGMA_SCALE={SIGMA_SCALE} (2σ 신뢰구간)")  # ✨ NEW
print(f"   Warmup: {WARMUP_HIDE}~{WARMUP_FADE} 프레임")  # ✨ NEW
print(f"   첫 추론 대기: {SEQ_LENGTH/VIDEO_FPS:.1f}초 (SEQ_LENGTH={SEQ_LENGTH})")  # ✨ NEW
print(f"   동적 안전 마진: COLLISION_DIST + 2σ_pred")  # ✨ NEW
print(f"   Hysteresis: {COLLISION_HOLD} 프레임 연속 위험 시 경보")  # ✨ NEW

latency_list = []
frame_count  = 0
cleanup_interval = 30  # 30프레임마다 stale 트랙 정리

try:
    while cap.isOpened():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        is_sample_frame = (frame_count % SAMPLE_INTERVAL == 0)

        # ① YOLO 추론 (매 프레임)
        # ① YOLO 추론 (매 프레임)
        results = yolo_model.track(
            frame, 
            persist=True,
            tracker="bytetrack.yaml",  # [핵심] RPi5 엣지 배포용 초경량 트래커 지정
            imgsz=1280,                # 파인튜닝 시 1280을 썼다면 동일하게 유지
            classes=[0, 1, 2, 3],
            conf=0.1,                  # [핵심] ByteTrack의 2차 매칭(저신뢰도 박스 활용)을 위해 낮게 설정
            iou=0.3,
            agnostic_nms=False,
            verbose=False
        )   

        collision_risk  = False
        min_dist_global = float('inf')
        ttc_global      = PRED_LENGTH
        ttc_sec_global  = PRED_LENGTH * DT
        active_ids      = set()

        # 추적된 객체가 하나라도 있는지 확인
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs     = results[0].boxes.conf.cpu().numpy() # 신뢰도 점수 확인용 (선택)

            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                active_ids.add(track_id)
                x, y, w, h = box
                
                # Person(0) 클래스 필터링: 30x30px 미만의 너무 작은 박스는 노이즈일 확률이 큼
                if class_id == 0 and w * h < 400:   
                    continue
                
                # --- [추가 1] 다수결 투표 (Majority Voting) ---
                class_votes[track_id].append(int(class_id))
                # 최근 10프레임 중 최빈값 클래스 추출
                voted_class_id = Counter(class_votes[track_id]).most_common(1)[0][0]

                if conf < CONF_THRESHOLD.get(int(class_id), 0.25):  # ✨ NEW
                    continue  # ✨ NEW: conf 낮은 detection 스킵

                # --- [추가 2] 갭(Gap) 체크 및 Zeroing ---
                if track_id in prev_frames:
                    frame_diff = frame_count - prev_frames[track_id]
                    
                    if frame_diff > 5:
                        # 5프레임 초과 갭 발생 시: 기존 속도 기록 초기화
                        # (기존 main.py의 변수명에 맞게 clear 또는 초기값 할당)
                        if track_id in velocity_history:
                            velocity_history[track_id].clear()
                        if track_id in prev_velocity:
                            prev_velocity[track_id] = [0.0, 0.0]  # 혹은 del prev_velocity[track_id]
                            
                # 현재 프레임 번호 업데이트
                prev_frames[track_id] = frame_count
                

                # (옵션) 1-Euro 필터 등 공간적 평활화를 적용한다면 이 위치에서 box 좌표를 스무딩합니다.
                # 예: x, y, w, h = one_euro_filter(track_id, x, y, w, h)

                # ② 피처 추출 + LSTM (샘플 프레임)
                if is_sample_frame:
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, voted_class_id, prev_box)
                    prev_boxes[track_id] = box

                    mean, std = run_lstm_mc(track_id)  # ✨ MODIFIED
                    if mean is not None:  # ✨ MODIFIED
                        update_prediction(track_id, mean, std)  # ✨ NEW

                # ③ 시각화 (디버깅용: ID와 신뢰도 점수를 같이 띄워 추적 안정성 확인)
                # draw_bbox 함수 내부를 수정하여 conf 값도 같이 표시되게 하면 분석하기 좋습니다.
                draw_bbox(frame, box, track_id, voted_class_id, conf)  # ✨ MODIFIED
                if track_id in predictions:
                    n_pred = pred_frame_count.get(track_id, 0)  # ✨ NEW
                    draw_predictions(  # ✨ MODIFIED
                        frame,
                        predictions[track_id],
                        predictions_std[track_id],
                        n_pred
                    )

            if is_sample_frame:
                merge_duplicate_tracks(predictions, prev_boxes)

            # ④ TTC 계산 (보행자-차량 쌍만 + uncertainty margin + hysteresis)  # ✨ MODIFIED
            if is_sample_frame and len(predictions_raw_mean) >= 2:
                valid_ids = {  # ✨ NEW
                    tid for tid in predictions_raw_mean
                    if pred_frame_count.get(tid, 0) >= WARMUP_TTC_GATE
                }
                valid_preds = {tid: predictions_raw_mean[tid] for tid in valid_ids}  # ✨ NEW
                pairs = find_pedestrian_vehicle_pairs(valid_preds, track_classes)  # ✨ NEW
                pair_risk = False  # ✨ NEW

                for p_id, v_id in pairs:
                    min_dist, col_frame, ttc_sec = calc_ttc_with_uncertainty(  # ✨ MODIFIED
                        predictions_raw_mean[p_id], predictions_raw_std[p_id],
                        predictions_raw_mean[v_id], predictions_raw_std[v_id],
                    )
                    if min_dist < min_dist_global:
                        min_dist_global = min_dist
                        ttc_global      = col_frame
                        ttc_sec_global  = ttc_sec
                    if col_frame < PRED_LENGTH and ttc_sec < COLLISION_TTC:  # ✨ NEW
                        pair_risk = True

                if pair_risk:  # ✨ NEW
                    collision_streak += 1
                else:
                    collision_streak = 0

                if collision_streak >= COLLISION_HOLD:  # ✨ NEW
                    collision_risk = True
            elif is_sample_frame:  # ✨ NEW
                collision_streak = 0  # ✨ NEW

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
        
        # ⏱️ [수정된 부분] 30FPS 재생 속도 동기화 (Frame Rate Lock)
        # 1프레임당 목표 소요 시간 (30FPS = 약 33.3ms)에서 실제 처리 시간(latency_ms)을 뺌
        #delay = int(33.3 - latency_ms)
        
        # 처리 시간이 33.3ms를 넘더라도 최소 1ms는 키보드 입력을 받기 위해 대기
        #wait_time = max(1, delay) 
        
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
        print(f"\n{'='*55}")
        print(f"📊 A-PAS 최종 성능 통계")
        print(f"{'='*55}")
        print(f"  모델           : Residual LSTM (30FPS)")
        print(f"  관찰/예측      : {SEQ_LENGTH/30:.1f}초 → {PRED_LENGTH/30:.1f}초")
        print(f"  처리 프레임     : {frame_count:,}장")
        print(f"  LSTM 실행 횟수  : {frame_count // SAMPLE_INTERVAL:,}회")
        print(f"  히스토리 딜레이 : {SEQ_LENGTH / VIDEO_FPS:.1f}초")
        print(f"  평균 Latency   : {arr.mean():.2f}ms")
        print(f"  최소 Latency   : {arr.min():.2f}ms")
        print(f"  최대 Latency   : {arr.max():.2f}ms")
        print(f"  평균 FPS       : {1000/arr.mean():.1f}")
        print(f"  모델 ADE       : 2.26px (Residual, 1s→2s)")
        print(f"{'='*55}")
