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
from collections import deque
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 30FPS Residual LSTM
# ==========================================
# --- 입력 소스 ---
VIDEO_PATH      = "test_video5.mp4"       # 또는 CARLA HDMI 캡처
YOLO_MODEL_PATH = "best_v6.pt"         # Hailo 사용 시 .hef로 교체
ONNX_MODEL_PATH = "models/Residual/best_model_30fps.onnx"

# --- 해상도 ---
IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

# --- 모델 설정 (30FPS) ---
SEQ_LENGTH   = 60          # 2.0초 관찰
PRED_LENGTH  = 30          # 1.0초 예측
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

# --- 클래스 매핑 ---
CLASS_MAP    = {0: 0, 1: 1, 2: 2, 3: 3}
CLASS_NAMES  = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}

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
predictions      = {}   # {track_id: (PRED, 2) 픽셀 좌표}
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


def run_lstm(track_id):
    """LSTM 추론 → 미래 경로 반환 (픽셀 단위)"""
    if len(track_history[track_id]) < SEQ_LENGTH:
        return None

    seq = np.array(track_history[track_id], dtype=np.float32)
    seq = seq[np.newaxis, ...]              # (1, 60, 17)

    output = lstm_session.run(None, {"input": seq})[0]  # (1, 30, 2)
    pred = output[0].copy()                 # (30, 2) 정규화 좌표

    # 역정규화
    pred[:, 0] *= IMG_W
    pred[:, 1] *= IMG_H

    return pred

# ==========================================
# ⚠️ [TTC 계산]
# ==========================================
def calc_ttc(pred_a, pred_b):
    """두 객체의 예측 궤적 간 TTC 계산"""
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

    ttc_seconds = collision_frame * DT
    return min_dist, collision_frame, ttc_seconds

# ==========================================
# 🎨 [시각화]
# ==========================================
def draw_predictions(frame, pred, color=(0, 255, 255)):
    """예측 궤적 시각화"""
    for t in range(len(pred) - 1):
        pt1 = (int(pred[t, 0]),   int(pred[t, 1]))
        pt2 = (int(pred[t+1, 0]), int(pred[t+1, 1]))
        cv2.line(frame, pt1, pt2, color, 2)
    cv2.circle(frame, (int(pred[-1, 0]), int(pred[-1, 1])), 6, color, -1)


CLASS_COLORS = {0: (0,0,255), 1: (0,255,0), 2: (255,0,255), 3: (255,165,0)}

def draw_bbox(frame, box, track_id, class_id):
    x, y, w, h = box
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    cls_name = CLASS_NAMES.get(class_id, "?")
    color = CLASS_COLORS.get(class_id, (0, 255, 0))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{track_id} [{cls_name}]",
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
def cleanup_stale_tracks(active_ids):
    """현재 프레임에 없는 트랙 제거 (메모리 관리)"""
    stale = [tid for tid in track_history if tid not in active_ids]
    for tid in stale:
        track_history.pop(tid, None)
        prev_boxes.pop(tid, None)
        predictions.pop(tid, None)
        velocity_history.pop(tid, None)
        prev_velocity.pop(tid, None)
        track_classes.pop(tid, None)

def merge_duplicate_tracks(predictions, prev_boxes, merge_dist=50):
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
            agnostic_nms=True,
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
                if class_id == 0 and w * h < 900:   
                    continue
                
                # (옵션) 1-Euro 필터 등 공간적 평활화를 적용한다면 이 위치에서 box 좌표를 스무딩합니다.
                # 예: x, y, w, h = one_euro_filter(track_id, x, y, w, h)

                # ② 피처 추출 + LSTM (샘플 프레임)
                if is_sample_frame:
                    prev_box = prev_boxes.get(track_id, None)
                    add_to_history(track_id, box, class_id, prev_box)
                    prev_boxes[track_id] = box

                    pred = run_lstm(track_id)
                    if pred is not None:
                        predictions[track_id] = pred

                # ③ 시각화 (디버깅용: ID와 신뢰도 점수를 같이 띄워 추적 안정성 확인)
                # draw_bbox 함수 내부를 수정하여 conf 값도 같이 표시되게 하면 분석하기 좋습니다.
                draw_bbox(frame, box, track_id, class_id)
                if track_id in predictions:
                    draw_predictions(frame, predictions[track_id])

            if is_sample_frame:
                merge_duplicate_tracks(predictions, prev_boxes)

            # ④ TTC 계산 (예측이 2개 이상일 때)
            if is_sample_frame and len(predictions) >= 2:
                id_list = list(predictions.keys())
                for i in range(len(id_list)):
                    for j in range(i + 1, len(id_list)):
                        min_dist, col_frame, ttc_sec = calc_ttc(
                            predictions[id_list[i]],
                            predictions[id_list[j]]
                        )
                        if min_dist < min_dist_global:
                            min_dist_global = min_dist
                            ttc_global      = col_frame
                            ttc_sec_global  = ttc_sec

                if (min_dist_global < COLLISION_DIST and
                    ttc_sec_global < COLLISION_TTC):
                    collision_risk = True

        # ⑤ 경고 출력
        if collision_risk:
            draw_alert(frame, ttc_sec_global, min_dist_global)
        trigger_alert(collision_risk, ttc_sec_global)

        # ⑥ 트랙 정리
        if frame_count % cleanup_interval == 0:
            cleanup_stale_tracks(active_ids)

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
        print(f"  모델 ADE       : 2.08px (Residual)")
        print(f"{'='*55}")