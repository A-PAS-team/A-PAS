import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# ⚙️ [설정] 프로젝트 규격에 맞게 고정
# ==========================================
TRAIN_DATA_DIR = "data/Training"
VAL_DATA_DIR = "data/Validation"
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 해상도 설정 (정규화의 핵심)
IMG_W, IMG_H = 1920, 1080 

# 하이퍼파라미터 (10프레임 데이터셋 최적화)
SEQ_LENGTH = 10     # 과거 관찰 기간 (0.5초)
PRED_LENGTH = 10    # 미래 예측 기간 (0.3초)
INPUT_SIZE = 2     # x, y 좌표
HIDDEN_SIZE = 128  
NUM_LAYERS = 2     
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 충돌 예측 모델 학습 시작 (Device: {device})")

# ==========================================
# 📊 [데이터셋] 정규화 및 슬라이딩 윈도우
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.sequences = []
        self.labels = []
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        print(f"📂 {data_dir}에서 데이터를 로드 중입니다...")

        for file in csv_files:
            df = pd.read_csv(file)
            # 객체(track_id)별로 분리해서 학습
            for track_id, group in df.groupby('track_id'):
                # 1. 좌표 정규화 (0~1 사이 값으로 변환)
                coords = group[['x_center', 'y_center']].values
                coords[:, 0] /= IMG_W
                coords[:, 1] /= IMG_H
                
                # 최소 길이(SEQ+PRED) 미달인 짧은 데이터는 버림
                if len(coords) < (SEQ_LENGTH + PRED_LENGTH):
                    continue
                
                # 슬라이딩 윈도우: 1프레임씩 밀어가며 데이터 생성
                for i in range(len(coords) - SEQ_LENGTH - PRED_LENGTH + 1):
                    self.sequences.append(coords[i : i + SEQ_LENGTH])
                    self.labels.append(coords[i + SEQ_LENGTH : i + SEQ_LENGTH + PRED_LENGTH])

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.FloatTensor(np.array(self.labels))
        print(f"✔️ 총 {len(self.sequences)}개의 학습 시퀀스 생성 완료!")

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

# ==========================================
# 🧠 [모델] LSTM (Long Short-Term Memory)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True: (Batch, Seq, Feature) 구조 사용
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, PRED_LENGTH * 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        # 마지막 시점의 특징만 사용하여 미래 좌표 예측
        out = self.fc(out[:, -1, :]) 
        return out.view(-1, PRED_LENGTH, 2)

# ==========================================
# 🚀 [실행] 학습 루프
# ==========================================
train_dataset = TrajectoryDataset(TRAIN_DATA_DIR)
val_dataset = TrajectoryDataset(VAL_DATA_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for seqs, targets in train_loader:
        seqs, targets = seqs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for v_seqs, v_targets in val_loader:
            v_seqs, v_targets = v_seqs.to(device), v_targets.to(device)
            v_outputs = model(v_seqs)
            val_loss += criterion(v_outputs, v_targets).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_trajectory_model.pth"))
        print(f"⭐ 최고 성능 모델 업데이트! (Val Loss: {best_val_loss:.6f})")

print("\n🎉 학습이 완료되었습니다. 이제 main.py에서 충돌을 감지할 수 있습니다!")