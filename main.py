import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 10FPS 모델 기준 + Fine-tuned YOLO
# ==========================================
VIDEO_PATH      = "my_video.mp4"

# ✅ Fine-tuned 모델 경로 (A-PAS CARLA 전용)
YOLO_MODEL_PATH = "ai_model/yolo_train/runs/detect/apas_carla/weights/best.pt"
ONNX_MODEL_PATH = "models/Residual/best_model_30fps.onnx"

IMG_W, IMG_H    = 1920, 1080
SEQ_LENGTH      = 20
PRED_LENGTH     = 10
INPUT_SIZE      = 17
DT              = 0.1

# 30FPS 영상 → 10FPS 샘플링
VIDEO_FPS       = 30
TARGET_FPS      = 10
SAMPLE_INTERVAL = VIDEO_FPS // TARGET_FPS  # = 3

# 속도 스무딩
SMOOTH_WINDOW   = 5

# 충돌 판정 기준
COLLISION_DIST  = 80
COLLISION_TTC   = 3

# ✅ A-PAS Fine-tuned 모델 클래스 번호
#    0=person, 1=car, 2=motorcycle, 3=bus, 4=truck
APAS_CLASSES = [0, 1, 2, 3, 4]

# fine-tuned 모델은 이미 A-PAS 번호로 나오므로 매핑 불필요
# (예전 COCO 번호 매핑: {0:0, 2:1, 5:2, 7:3, 3:4} 은 제거)
# 그냥 클래스 ID 그대로 사용하면 됨

# ==========================================
# 🧠 [모델 로드]
# ==========================================
print("📦 모델 로드 중...")
yolo_model   = YOLO(YOLO_MODEL_PATH)
lstm_session = ort.InferenceSession(
    ONNX_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
print(f"  ✅ YOLO 로드 완료: {YOLO_MODEL_PATH}")
print(f"  ✅ LSTM (ONNX) 로드 완료")
print(f"  🖥️  LSTM 실행 디바이스: {lstm_session.get_providers()}")

# ==========================================
# 📊 [트래킹 히스토리]
# ==========================================
track_history    = {}
prev_boxes       = {}
predictions      = {}
velocity_history = {}
prev_velocity    = {}

# ==========================================
# 🔧 [유틸 함수]
# ==========================================
def compute_features(box, prev_box, track_id, dt=DT):
    """현재/이전 박스에서 12개 물리 피처 벡터 생성"""
    x, y, w, h = box
    diag = np.sqrt(IMG_W**2 + IMG_H**2)

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

    physics = np.array([
        x / IMG_W,   y / IMG_H,
        vx / IMG_W,  vy / IMG_H,
        ax / IMG_W,  ay / IMG_H,
        speed / diag,
        np.sin(heading), np.cos(heading),
        w / IMG_W,   h / IMG_H,
        (w * h) / (IMG_W * IMG_H)
    ], dtype=np.float32)

    return physics  # 12개


def add_to_history(track_id, box, class_id, prev_box):
    """트래킹 히스토리에 17개 피처 추가"""
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=SEQ_LENGTH)

    physics = compute_features(box, prev_box, track_id)

    # ✅ fine-tuned 모델은 A-PAS 번호 (0~4)로 직접 나오므로 매핑 불필요
    one_hot = np.zeros(5, dtype=np.float32)
    if 0 <= class_id < 5:
        one_hot[class_id] = 1.0

    track_history[track_id].append(
        np.concatenate([physics, one_hot])  # 17개
    )


def run_lstm(track_id):
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None

    seq    = np.array(track_history[track_id], dtype=np.float32)
    seq    = seq[np.newaxis, ...]
    output = lstm_session.run(None, {"input": seq})[0]
    pred   = output[0].copy()

    pred[:, 0] *= IMG_W
    pred[:, 1] *= IMG_H

    return pred


def calc_ttc(pred_a, pred_b):
    min_dist        = float('inf')
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


def draw_predictions(frame, pred, color=(0, 255, 255)):
    for t in range(len(pred) - 1):
        pt1 = (int(pred[t, 0]),     int(pred[t, 1]))
        pt2 = (int(pred[t+1, 0]),   int(pred[t+1, 1]))
        cv2.line(frame, pt1, pt2, color, 2)
    cv2.circle(frame,
               (int(pred[-1, 0]), int(pred[-1, 1])), 6, color, -1)


def draw_alert(frame, ttc_sec, min_dist):
    cv2.rectangle(frame, (0, 0), (420, 80), (0, 0, 200), -1)
    cv2.putText(frame, "WARNING  COLLISION RISK",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"TTC: {ttc_sec:.1f}s | Dist: {min_dist:.0f}px",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


# ==========================================
# 🎬 [메인 루프]
# ==========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
    exit()

print(f"\n🚀 A-PAS 모니터링 시작! (영상: {VIDEO_PATH})")
print(f"   샘플링: {VIDEO_FPS}FPS → {TARGET_FPS}FPS")
print(f"   클래스: person, car, motorcycle, bus, truck (A-PAS 5종)")
print("종료: 'q' 키\n")

latency_list = []
frame_count  = 0

try:
    while cap.isOpened():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count     += 1
        is_sample_frame  = (frame_count % SAMPLE_INTERVAL == 0)

        # ✅ Fine-tuned 모델용 필터 (A-PAS 번호)
        results = yolo_model.track(
            frame, persist=True,
            imgsz=640,
            classes=APAS_CLASSES,    # [0, 1, 2, 3, 4]
            conf=0.3,
            verbose=False
        )

        collision_risk  = False
        min_dist_global = float('inf')
        ttc_global      = PRED_LENGTH

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):

                if is_sample_frame:
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, class_id, prev_box)
                    prev_boxes[track_id] = box

                    pred = run_lstm(track_id)
                    if pred is not None:
                        predictions[track_id] = pred

                if track_id in predictions:
                    draw_predictions(frame, predictions[track_id])

                # 바운딩 박스 (클래스별 색상)
                class_colors = {
                    0: (0, 0, 255),      # person: 빨강
                    1: (0, 255, 0),      # car: 초록
                    2: (255, 0, 255),    # motorcycle: 핑크
                    3: (255, 165, 0),    # bus: 주황
                    4: (255, 255, 0),    # truck: 노랑
                }
                class_names = {0: "person", 1: "car", 2: "moto", 3: "bus", 4: "truck"}

                x, y, w, h = box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                color = class_colors.get(class_id, (0, 255, 0))
                name  = class_names.get(class_id, f"cls{class_id}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ID:{track_id}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # TTC 계산
            if is_sample_frame and len(predictions) >= 2:
                id_list = list(predictions.keys())
                for i in range(len(id_list)):
                    for j in range(i+1, len(id_list)):
                        min_dist, col_frame = calc_ttc(
                            predictions[id_list[i]],
                            predictions[id_list[j]]
                        )
                        if min_dist < min_dist_global:
                            min_dist_global = min_dist
                            ttc_global      = col_frame

                if min_dist_global < COLLISION_DIST and \
                   ttc_global < COLLISION_TTC / DT:
                    collision_risk = True

        if collision_risk:
            draw_alert(frame, ttc_global * DT, min_dist_global)

        t_end      = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        latency_list.append(latency_ms)

        fps        = 1000 / latency_ms if latency_ms > 0 else 0
        sample_tag = "[S]" if is_sample_frame else "   "
        cv2.putText(frame,
                    f"{sample_tag} FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("A-PAS Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    if latency_list:
        arr = np.array(latency_list)
        print("\n" + "="*50)
        print("📊 A-PAS 최종 성능 통계")
        print("="*50)
        print(f"  처리 프레임 수    : {frame_count}장")
        print(f"  LSTM 실행 횟수    : {frame_count // SAMPLE_INTERVAL}회")
        print(f"  평균 Latency      : {arr.mean():.2f}ms")
        print(f"  평균 FPS          : {1000/arr.mean():.1f}")
        print("="*50)