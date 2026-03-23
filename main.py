import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
from gpiozero import LED  # ✅ LED 제어 라이브러리 추가

# ==========================================
# ⚙️ [설정] 하드웨어 및 모델 설정
# ==========================================
IMG_W, IMG_H = 1920, 1080 
SEQ_LENGTH = 10 
PRED_LENGTH = 10 
HIDDEN_SIZE = 128
NUM_LAYERS = 2

YOLO_MODEL_PATH = "yolov8s.pt"
LSTM_MODEL_PATH = "models/best_model_pi.pt"
VIDEO_PATH = "my_video.mp4" 

# 하드웨어 설정
alert_led = LED(18)  # ✅ GPIO 18번 핀에 LED 연결
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 [모델 정의]
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, PRED_LENGTH * 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, PRED_LENGTH, 2)

# 모델 로드
print("📦 모델 로드 중...")
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = LSTMModel(2, HIDDEN_SIZE, NUM_LAYERS).to(device)
lstm_model = torch.jit.load(LSTM_MODEL_PATH, map_location=device)
lstm_model.eval()

track_history = {}
last_predictions = {}
cap = cv2.VideoCapture(VIDEO_PATH)

# ==========================================
# 🎬 [실행] 메인 루프
# ==========================================
print("🚀 A-PAS 모니터링 시작!")

# ⚙️ 업그레이드 설정
INPUT_SIZE = 6  # [x, y, vx, vy, w, h] 6개로 확장 예정
ADE_LIST = []   # 성능 지표 저장을 위한 리스트
FDE_LIST = []

# ... (모델 로드는 torch.jit.load 사용) ...

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        results = yolo_model.track(frame, persist=True, imgsz=320, classes=[0], verbose=False)
        collision_risk = False
        all_current_preds = {}

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy() # [x, y, w, h]
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                # 1. 속도 계산 및 피처 생성
                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=SEQ_LENGTH)
                    vx, vy = 0, 0 # 처음 등장 시 속도 0
                else:
                    prev_x, prev_y = track_history[track_id][-1][0], track_history[track_id][-1][1]
                    vx, vy = x - prev_x, y - prev_y # 단순 차이로 속도 계산
                
                # 피처 벡터 [x, y, vx, vy, w, h]
                curr_feature = np.array([float(x), float(y), float(vx), float(vy), float(w), float(h)])
                track_history[track_id].append(curr_feature)

                # 2. LSTM 추론 (입력이 6개인 모델인 경우)
                if len(track_history[track_id]) == SEQ_LENGTH:
                    # 데이터 정규화 (IMG_W, IMG_H 등으로 나누기) 및 Tensor 변환 로직...
                    # (중략)
                    
                    # 3. TTC 계산 및 시각화 (제안하신 코드 적용)
                    # ... (모든 객체 간 min_dist, collision_time 계산) ...
                    if min_dist < 80 and collision_time < 5: 
                        collision_risk = True
                        # TTC 표시 (t=5면 약 0.5초 뒤 충돌 예상)
                        cv2.putText(frame, f"TTC: {collision_time*0.1:.1f}s", 
                                    (int(x), int(y)-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 4. 성능 지표(ADE/FDE) 누적 기록
                if track_id in last_predictions:
                    error = np.linalg.norm(np.array([x, y]) - last_predictions[track_id])
                    ADE_LIST.append(error)
                    # 실제 1초(10프레임) 뒤 시점에 도달했을 때 비교하면 FDE가 됨

        # (LED 제어 로직 동일)
        # ...
        
finally:
    # 종료 시 통계 출력 (논문용 데이터)
    if ADE_LIST:
        print(f"📊 최종 성능 지표 | ADE: {np.mean(ADE_LIST):.2f} px")
    alert_led.off()
    cap.release()