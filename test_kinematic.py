"""
A-PAS Kinematic LSTM 빠른 테스트
==================================
ONNX 변환 없이 PyTorch 직접 추론으로 예측선 시각화.
MC Dropout 없이 단일 추론 (deterministic).

사용법:
  python test_kinematic_quick.py
"""

import cv2
import numpy as np
import torch
import time
import math
from collections import deque, defaultdict, Counter
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 모델 import ---
import sys
sys.path.insert(0, "ai_model/trajectory/lstm_kinematic")
from residual_lstm_kinematic import ResidualLSTMKinematic, ModelConfig

# ==========================================
# ⚙️ [설정]
# ==========================================
VIDEO_PATH      = "test_video/test_video5.mp4"
YOLO_MODEL_PATH = "best_v8.pt"
CHECKPOINT_PATH = "models/Kinematic/best_kinematic_10fps_v3kin_cctv_carla.pth"

IMG_W, IMG_H = 1920, 1080
DIAG         = math.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH   = 10   # 1초 @ 10FPS
PRED_LENGTH  = 20   # 2초 @ 10FPS
INPUT_SIZE   = 17
DT           = 0.1

VIDEO_FPS       = 30
TARGET_FPS      = 10
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # 3

SMOOTH_WINDOW   = 3
HEADING_SMOOTH  = 3
VEL_CLIP        = 2.0
ACC_CLIP        = 1.0
YAW_RATE_MAX    = math.pi
LOW_SPEED_THRESH = 30.0

DISPLAY_SCALE = 0.75

CLASS_MAP    = {0: 0, 1: 1, 2: 2, 3: 3}
CLASS_NAMES  = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}
CLASS_COLORS = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}
VEHICLE_CLASSES = {1, 2, 3}
CONF_THRESHOLD = {0: 0.4, 1: 0.4, 2: 0.35, 3: 0.3}

# ==========================================
# 🧠 [모델 로드]
# ==========================================
print("📦 모델 로드 중...")
yolo_model = YOLO(YOLO_MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
model_config = ModelConfig.from_dict(ckpt["model_config"])
lstm_model = ResidualLSTMKinematic(model_config).to(device)
lstm_model.load_state_dict(ckpt["model_state_dict"])
lstm_model.eval()

print(f"  ✅ YOLO: {YOLO_MODEL_PATH}")
print(f"  ✅ Kinematic LSTM: {CHECKPOINT_PATH}")
print(f"  🖥️  Device: {device}")
print(f"  📐 ADE: {ckpt.get('ade', '?')}px, FDE: {ckpt.get('fde', '?')}px")

# ==========================================
# 📊 [트래킹 상태]
# ==========================================
track_history    = {}   # {tid: deque([17-dim], maxlen=SEQ)}
prev_boxes       = {}
predictions      = {}   # {tid: (pred_length, 2) 픽셀 좌표}
velocity_history = {}
prev_velocity    = {}
track_classes    = {}
class_votes      = defaultdict(lambda: deque(maxlen=10))
prev_frames      = {}

# heading smoothing 용
vx_smooth_hist   = {}
vy_smooth_hist   = {}

# ==========================================
# 🔧 [피처 추출] — csv_to_npy와 동일한 로직
# ==========================================
def compute_features(box, prev_box, track_id):
    x, y, w, h = box

    if track_id not in velocity_history:
        velocity_history[track_id] = deque(maxlen=SMOOTH_WINDOW)
    if track_id not in vx_smooth_hist:
        vx_smooth_hist[track_id] = deque(maxlen=HEADING_SMOOTH)
        vy_smooth_hist[track_id] = deque(maxlen=HEADING_SMOOTH)

    if prev_box is not None:
        raw_vx = (x - prev_box[0]) / DT
        raw_vy = (y - prev_box[1]) / DT
    else:
        raw_vx, raw_vy = 0.0, 0.0

    velocity_history[track_id].append([raw_vx, raw_vy])

    hist = velocity_history[track_id]
    n = len(hist)
    vx = sum(v[0] for v in hist) / n
    vy = sum(v[1] for v in hist) / n

    # 가속도
    if track_id in prev_velocity:
        ax = (vx - prev_velocity[track_id][0]) / DT
        ay = (vy - prev_velocity[track_id][1]) / DT
    else:
        ax, ay = 0.0, 0.0
    prev_velocity[track_id] = [vx, vy]

    speed = math.hypot(vx, vy)
    heading = math.atan2(vy, vx)

    # yaw_rate: smoothed velocity 기반
    vx_smooth_hist[track_id].append(vx)
    vy_smooth_hist[track_id].append(vy)

    vx_sm = sum(vx_smooth_hist[track_id]) / len(vx_smooth_hist[track_id])
    vy_sm = sum(vy_smooth_hist[track_id]) / len(vy_smooth_hist[track_id])
    heading_smooth = math.atan2(vy_sm, vx_sm)

    # yaw_rate 계산 (이전 heading과의 차이)
    prev_heading = getattr(compute_features, '_prev_heading', {})
    if track_id in prev_heading:
        h_diff = heading_smooth - prev_heading[track_id]
        # wrap-around
        while h_diff > math.pi:
            h_diff -= 2 * math.pi
        while h_diff < -math.pi:
            h_diff += 2 * math.pi
        yaw_rate = h_diff / DT
    else:
        yaw_rate = 0.0
    prev_heading[track_id] = heading_smooth
    compute_features._prev_heading = prev_heading

    # 저속 마스킹
    if speed < LOW_SPEED_THRESH:
        yaw_rate = 0.0

    # 물리적 클램프
    yaw_rate = max(-YAW_RATE_MAX, min(YAW_RATE_MAX, yaw_rate))

    # 정규화
    features = np.array([
        x / IMG_W,
        y / IMG_H,
        max(-VEL_CLIP, min(VEL_CLIP, vx / IMG_W)),
        max(-VEL_CLIP, min(VEL_CLIP, vy / IMG_H)),
        max(-ACC_CLIP, min(ACC_CLIP, ax / IMG_W)),
        max(-ACC_CLIP, min(ACC_CLIP, ay / IMG_H)),
        speed / DIAG,
        math.sin(heading),
        math.cos(heading),
        w / IMG_W,
        h / IMG_H,
        (w * h) / (IMG_W * IMG_H),
        yaw_rate / YAW_RATE_MAX,  # 정규화
    ], dtype=np.float32)

    return features  # (13,)


def add_to_history(track_id, box, class_id, prev_box):
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=SEQ_LENGTH)

    physics = compute_features(box, prev_box, track_id)

    one_hot = np.zeros(4, dtype=np.float32)
    if class_id in CLASS_MAP:
        one_hot[CLASS_MAP[class_id]] = 1.0

    track_history[track_id].append(
        np.concatenate([physics, one_hot])  # (17,)
    )
    track_classes[track_id] = class_id


# ==========================================
# 🧠 [LSTM 추론]
# ==========================================
def run_lstm(track_id):
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None

    seq = np.array(track_history[track_id], dtype=np.float32)
    seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)  # (1, 10, 17)

    with torch.no_grad():
        pred = lstm_model(seq_tensor)  # (1, 20, 2) 정규화 좌표

    pred_np = pred.cpu().numpy()[0]  # (20, 2)

    # 정규화 → 픽셀
    pred_px = pred_np.copy()
    pred_px[:, 0] *= IMG_W
    pred_px[:, 1] *= IMG_H

    return pred_px


# ==========================================
# 🎨 [시각화]
# ==========================================
def scaled_pt(x, y):
    return int(x * DISPLAY_SCALE), int(y * DISPLAY_SCALE)


def draw_prediction(frame, pred, class_id):
    if class_id in VEHICLE_CLASSES:
        color = (0, 200, 255)   # 차량: 노란색
    else:
        color = (255, 100, 100) # 보행자: 파란색

    for t in range(len(pred) - 1):
        pt1 = scaled_pt(pred[t, 0], pred[t, 1])
        pt2 = scaled_pt(pred[t+1, 0], pred[t+1, 1])
        thickness = max(1, 3 - (t * 2 // len(pred)))
        cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # 끝점
    end = scaled_pt(pred[-1, 0], pred[-1, 1])
    cv2.circle(frame, end, 5, color, -1)

    # 1초 지점 표시 (10번째 스텝)
    if len(pred) > 10:
        mid = scaled_pt(pred[9, 0], pred[9, 1])
        cv2.circle(frame, mid, 3, (255, 255, 255), -1)


def draw_bbox(frame, box, track_id, class_id, conf):
    x, y, w, h = box
    x1, y1 = scaled_pt(x - w/2, y - h/2)
    x2, y2 = scaled_pt(x + w/2, y + h/2)
    color = CLASS_COLORS.get(class_id, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID:{track_id} [{CLASS_NAMES.get(class_id, '?')}] {conf:.2f}"
    cv2.putText(frame, label, (x1, max(14, y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


# ==========================================
# 🎬 [메인 루프]
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 영상 열기 실패: {VIDEO_PATH}")
    exit()
IMG_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
IMG_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DIAG  = math.sqrt(IMG_W**2 + IMG_H**2)
print(f"\n🚀 Kinematic LSTM 테스트 시작!")
print(f"   영상: {VIDEO_PATH}")
print(f"   SEQ={SEQ_LENGTH} (1초) → PRED={PRED_LENGTH} (2초)")
print(f"   디스플레이: {DISPLAY_SCALE}x")
print(f"   종료: 'q' 키\n")

frame_count = 0
latency_list = []

try:
    while cap.isOpened():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        is_sample = (frame_count % SAMPLE_INTERVAL == 0)

        # ① YOLO
        results = yolo_model.track(
            frame, persist=True,
            tracker="bytetrack.yaml",
            imgsz=1280,
            classes=[0, 1, 2, 3],
            conf=0.1,
            iou=0.3,
            verbose=False
        )

        active_ids = set()

        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes     = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            confs     = results[0].boxes.conf.cpu().numpy()

            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                active_ids.add(track_id)

                if class_id == 0 and box[2] * box[3] < 400:
                    continue
                if conf < CONF_THRESHOLD.get(int(class_id), 0.25):
                    continue

                class_votes[track_id].append(int(class_id))
                voted_class = Counter(class_votes[track_id]).most_common(1)[0][0]

                # 갭 체크
                if track_id in prev_frames:
                    if frame_count - prev_frames[track_id] > 15:
                        velocity_history.pop(track_id, None)
                        prev_velocity.pop(track_id, None)
                        track_history.pop(track_id, None)
                        vx_smooth_hist.pop(track_id, None)
                        vy_smooth_hist.pop(track_id, None)
                        if hasattr(compute_features, '_prev_heading'):
                            compute_features._prev_heading.pop(track_id, None)
                prev_frames[track_id] = frame_count

                # ② 피처 + LSTM (샘플 프레임만)
                if is_sample:
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, voted_class, prev_box)
                    prev_boxes[track_id] = box

                    pred = run_lstm(track_id)
                    if pred is not None:
                        predictions[track_id] = pred

        # ③ 시각화 (샘플 프레임만)
        if is_sample:
            vis = cv2.resize(frame,
                             (int(IMG_W * DISPLAY_SCALE), int(IMG_H * DISPLAY_SCALE)))

            # bbox + 예측선
            if results and results[0].boxes and results[0].boxes.id is not None:
                for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                    if conf < CONF_THRESHOLD.get(int(class_id), 0.25):
                        continue
                    voted_class = Counter(class_votes.get(track_id, [class_id])).most_common(1)[0][0]
                    draw_bbox(vis, box, track_id, voted_class, conf)

                    if track_id in predictions:
                        draw_prediction(vis, predictions[track_id], voted_class)

            # 성능 표시
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            latency_list.append(latency_ms)
            fps = 1000 / latency_ms if latency_ms > 0 else 0

            info = f"FPS:{fps:.1f} Lat:{latency_ms:.0f}ms Tracks:{len(active_ids)} Preds:{len(predictions)}"
            cv2.putText(vis, info, (10, vis.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # 모델 정보
            cv2.putText(vis, "Kinematic LSTM (1s obs -> 2s pred)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("A-PAS Kinematic LSTM Test", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # stale 트랙 정리
        if frame_count % 90 == 0:
            stale = [tid for tid in track_history
                     if tid not in active_ids
                     and (frame_count - prev_frames.get(tid, 0)) > 60]
            for tid in stale:
                for d in [track_history, prev_boxes, predictions,
                          velocity_history, prev_velocity, track_classes,
                          prev_frames, class_votes,
                          vx_smooth_hist, vy_smooth_hist]:
                    d.pop(tid, None)

finally:
    cap.release()
    cv2.destroyAllWindows()

    if latency_list:
        arr = np.array(latency_list)
        print(f"\n{'='*55}")
        print(f"📊 Kinematic LSTM 테스트 결과")
        print(f"{'='*55}")
        print(f"  처리 프레임    : {frame_count:,}")
        print(f"  평균 Latency  : {arr.mean():.2f}ms")
        print(f"  평균 FPS      : {1000/arr.mean():.1f}")
        print(f"  Min / Max     : {arr.min():.2f} / {arr.max():.2f}ms")
        print(f"  모델 ADE      : {ckpt.get('ade', '?')}px")
        print(f"  모델 FDE      : {ckpt.get('fde', '?')}px")
        print(f"{'='*55}")