import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# ⚙️ [설정] 하이퍼파라미터 및 경로
# ==========================================
TRAIN_DATA_DIR = "data/Training"
VAL_DATA_DIR = "data/Validation"
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 필수 설정 (본인 영상 해상도에 맞게 수정)
IMG_W, IMG_H = 1920, 1080 
SEQ_LENGTH = 10     
PRED_LENGTH = 10    
INPUT_SIZE = 2      
HIDDEN_SIZE = 128   
NUM_LAYERS = 2      
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 📊 [데이터셋] 정규화 로직 포함
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.sequences = []
        self.labels = []
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        for file in csv_files:
            df = pd.read_csv(file)
            for track_id, group in df.groupby('track_id'):
                # 좌표를 0~1 사이로 정규화
                coords = group[['x_center', 'y_center']].values
                coords[:, 0] /= IMG_W
                coords[:, 1] /= IMG_H
                
                if len(coords) < (SEQ_LENGTH + PRED_LENGTH):
                    continue
                
                for i in range(len(coords) - SEQ_LENGTH - PRED_LENGTH + 1):
                    self.sequences.append(coords[i : i + SEQ_LENGTH])
                    self.labels.append(coords[i + SEQ_LENGTH : i + SEQ_LENGTH + PRED_LENGTH])

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.FloatTensor(np.array(self.labels))

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

# ==========================================
# 🧠 [모델] LSTM 구조 (batch_first 수정 완료)
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

# 학습 실행 로직 (이전과 동일하지만 정규화된 데이터로 수행)
# ... (생략: 이전 tr_trajectory.py의 학습 루프 부분과 동일)
# ==========================================
# 🚀 [실행] 학습 루프
# ==========================================
# 1. 데이터 로더 준비
train_dataset = TrajectoryDataset(TRAIN_DATA_DIR)
val_dataset = TrajectoryDataset(VAL_DATA_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. 모델, 손실함수, 최적화기 설정
model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
criterion = nn.MSELoss() # 평균 제곱 오차 (좌표 오차 계산)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Epoch 반복
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
    
    # 검증(Validation)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for v_seqs, v_targets in val_loader:
            v_seqs, v_targets = v_seqs.to(device), v_targets.to(device)
            v_outputs = model(v_seqs)
            v_loss = criterion(v_outputs, v_targets)
            val_loss += v_loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # 성능이 가장 좋은 모델 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_trajectory_model.pth"))
        print(f"⭐ 모델 저장 완료! (Val Loss: {best_val_loss:.4f})")

print("\n🎉 모든 학습이 완료되었습니다!")