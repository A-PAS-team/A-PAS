import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==========================================
# ⚙️ [설정] 학습 환경과 100% 일치해야 함
# ==========================================
# 1. 영상 해상도 (학습 때와 동일하게 설정)
IMG_W, IMG_H = 1920, 1080 

# 2. 하이퍼파라미터 (학습 때 수정한 값과 동일하게 설정)
SEQ_LENGTH = 5     
PRED_LENGTH = 3    
HIDDEN_SIZE = 128
NUM_LAYERS = 2

# 3. 모델 경로
YOLO_MODEL_PATH = "yolov8n.pt"
LSTM_MODEL_PATH = "models/best_trajectory_model.pth"
VIDEO_PATH = "my_video.mp4" # 실제 영상 파일명으로 수정

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 [모델 정의] 학습 때와 동일한 구조
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True 가 핵심입니다.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, PRED_LENGTH * 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(-1, PRED_LENGTH, 2)

# ==========================================
# 🚀 [준비] 모델 로드
# ==========================================
print(f"🚀 시스템 시작 (장치: {device})")
yolo_model = YOLO(YOLO_MODEL_PATH)
lstm_model = LSTMModel(2, HIDDEN_SIZE, NUM_LAYERS).to(device)
lstm_model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
lstm_model.eval()

track_history = {}
cap = cv2.VideoCapture(VIDEO_PATH)

# ==========================================
# 🎬 [실행] 메인 루프
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 영상 크기가 설정과 다를 경우를 대비해 리사이즈 (선택 사항)
    frame = cv2.resize(frame, (IMG_W, IMG_H))

    # YOLO 추적
    results = yolo_model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center = (float(x), float(y))
            
            # 1. 궤적 업데이트
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=SEQ_LENGTH)
            track_history[track_id].append(center)

            # 2. 과거 궤적 그리기 (파란색 실선)
            if len(track_history[track_id]) > 1:
                points = np.array(list(track_history[track_id]), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 0), thickness=2)

            # 3. 미래 예측 (데이터가 충분히 쌓였을 때 실행)
            if len(track_history[track_id]) == SEQ_LENGTH:
                # [정규화] 입력 좌표를 0~1 사이로 변환
                input_seq = np.array(list(track_history[track_id]))
                input_seq[:, 0] /= IMG_W
                input_seq[:, 1] /= IMG_H
                input_seq_torch = torch.FloatTensor([input_seq]).to(device)
                
                with torch.no_grad():
                    prediction = lstm_model(input_seq_torch).cpu().numpy()[0]
                
                # [복원] 결과 좌표를 다시 영상 크기로 변환하여 그리기
                for i in range(len(prediction)):
                    # 예측된 점은 빨간색으로 표시
                    pred_x = int(prediction[i, 0] * IMG_W)
                    pred_y = int(prediction[i, 1] * IMG_H)
                    
                    # 화면 범위를 벗어나지 않을 때만 그리기
                    if 0 <= pred_x < IMG_W and 0 <= pred_y < IMG_H:
                        cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)

    # 결과 화면 출력
    cv2.imshow("A-PAS: Intelligent Trajectory Prediction", frame)
    
    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 시스템이 안전하게 종료되었습니다.")