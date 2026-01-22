import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================
# ⚙️ [설정] 학습 환경과 100% 일치 필수
# ==========================================
IMG_W, IMG_H = 1920, 1080 
SEQ_LENGTH = 5     
PRED_LENGTH = 3    
HIDDEN_SIZE = 128
NUM_LAYERS = 2

YOLO_MODEL_PATH = "yolov8n.pt"
LSTM_MODEL_PATH = "models/best_trajectory_model.pth"
VIDEO_PATH = "test_video.mp4" 

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
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = LSTMModel(2, HIDDEN_SIZE, NUM_LAYERS).to(device)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
lstm_model.eval()

track_history = {}
future_predictions = {} # 충돌 판정을 위해 모든 객체의 미래 위치 저장
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.resize(frame, (IMG_W, IMG_H))
    
    results = yolo_model.track(frame, persist=True, verbose=False)
    collision_detected = False
    future_predictions.clear()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # 1단계: 모든 객체의 미래 경로 예측
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=SEQ_LENGTH)
            track_history[track_id].append((float(x), float(y)))

            if len(track_history[track_id]) == SEQ_LENGTH:
                input_seq = np.array(list(track_history[track_id]))
                input_seq[:, 0] /= IMG_W
                input_seq[:, 1] /= IMG_H
                input_seq_torch = torch.FloatTensor([input_seq]).to(device)
                
                with torch.no_grad():
                    pred = lstm_model(input_seq_torch).cpu().numpy()[0]
                    # 다시 픽셀 좌표로 복원해서 저장
                    pred[:, 0] *= IMG_W
                    pred[:, 1] *= IMG_H
                    future_predictions[track_id] = pred

        # 2단계: 충돌 감지 및 시각화
        for track_id, pred in future_predictions.items():
            # 과거 궤적 그리기 (파란색)
            pts = np.array(list(track_history[track_id]), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

            # 미래 예측 점 그리기 (빨간색)
            for p_x, p_y in pred:
                cv2.circle(frame, (int(p_x), int(p_y)), 5, (0, 0, 255), -1)

            # 충돌 체크: 다른 객체들의 미래 위치와 비교
            for other_id, other_pred in future_predictions.items():
                if track_id == other_id: continue
                
                # 내 마지막 예측 지점과 상대방의 마지막 예측 지점 사이 거리 계산
                dist = np.linalg.norm(pred[-1] - other_pred[-1])
                
                if dist < 80: # 충돌 임계값 (80픽셀 이내면 위험)
                    collision_detected = True
                    # 위험 객체들 강조 (노란색 박스)
                    cv2.putText(frame, "COLLISION RISK", (int(pred[-1,0]), int(pred[-1,1])-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # 3단계: 전체 화면 경고 레이아웃
    if collision_detected:
        # 화면 테두리에 빨간색 사각형
        cv2.rectangle(frame, (0, 0), (IMG_W, IMG_H), (0, 0, 255), 30)
        cv2.putText(frame, "!!! EMERGENCY WARNING !!!", (IMG_W//2 - 400, 100), 
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("A-PAS Final: Collision Prevention", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()