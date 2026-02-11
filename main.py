import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================
# ⚙️ [설정] 학습(tr_trajectory.py)과 반드시 일치해야 함
# ==========================================
IMG_W, IMG_H = 1920, 1080 
SEQ_LENGTH = 10     # 5에서 10으로 수정 (10-10 법칙 적용)
PRED_LENGTH = 10    # 3에서 10으로 수정 (1초 뒤 예측)
HIDDEN_SIZE = 128
NUM_LAYERS = 2

YOLO_MODEL_PATH = "yolov8n.pt"
LSTM_MODEL_PATH = "models/best_trajectory_model.pth"
VIDEO_PATH = "my_video.mp4" 

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
last_predictions = {} # 이상 행동(Loss) 계산을 위한 저장소
cap = cv2.VideoCapture(VIDEO_PATH)

# ==========================================
# 🎬 [실행] 메인 루프
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.resize(frame, (IMG_W, IMG_H))
    
    results = yolo_model.track(frame, persist=True, verbose=False)
    collision_risk = False
    all_current_preds = {} # 이번 프레임의 모든 미래 좌표 저장

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            curr_pos = np.array([float(x), float(y)])
            
            # 1. 이상 행동(Anomaly Loss) 계산
            if track_id in last_predictions:
                # 이전 프레임에서 예측했던 '현재 시점'의 위치와 실제 위치 비교
                pred_pos = last_predictions[track_id]
                anomaly_loss = np.linalg.norm(curr_pos - pred_pos)
                
                # 오차가 크면 (예: 50px 이상) 경고 텍스트 표시
                if anomaly_loss > 50:
                    cv2.putText(frame, f"UNSTABLE: {int(anomaly_loss)}", (int(x), int(y)-40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=SEQ_LENGTH)
            track_history[track_id].append(curr_pos)

            # 2. 미래 경로 예측
            if len(track_history[track_id]) == SEQ_LENGTH:
                input_seq = np.array(list(track_history[track_id]))
                input_seq[:, 0] /= IMG_W
                input_seq[:, 1] /= IMG_H
                input_seq_torch = torch.FloatTensor([input_seq]).to(device)
                
                with torch.no_grad():
                    pred = lstm_model(input_seq_torch).cpu().numpy()[0]
                    pred[:, 0] *= IMG_W
                    pred[:, 1] *= IMG_H
                    all_current_preds[track_id] = pred
                    # 다음 프레임 오차 계산을 위해 첫 번째 예측 지점 저장
                    last_predictions[track_id] = pred[0] 

                    # 예측 경로 시각화 (빨간 선)
                    cv2.polylines(frame, [pred.astype(np.int32)], False, (0, 0, 255), 2)

        # 3. 충돌 감지 로직 (객체 간 거리 비교)
        for id1, pred1 in all_current_preds.items():
            for id2, pred2 in all_current_preds.items():
                if id1 >= id2: continue # 중복 검사 방지
                
                # 미래 1초 뒤(마지막 점)의 거리 계산
                dist = np.linalg.norm(pred1[-1] - pred2[-1])
                if dist < 100: # 100픽셀 이내 접근 시 위험
                    collision_risk = True
                    cv2.line(frame, tuple(pred1[-1].astype(int)), tuple(pred2[-1].astype(int)), (0, 255, 255), 3)

    # 4. 전체 시스템 경고 출력
    if collision_risk:
        cv2.rectangle(frame, (0, 0), (IMG_W, IMG_H), (0, 0, 255), 20)
        cv2.putText(frame, "!!! COLLISION WARNING !!!", (IMG_W//2-350, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("A-PAS Final Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()