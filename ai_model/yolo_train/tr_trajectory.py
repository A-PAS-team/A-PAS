import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import glob
import os

# ==========================================
# ⚙️ [설정] 피처 확장 규격 (6개 피처)
# ==========================================
TRAIN_DATA_DIR = "data/Training"
VAL_DATA_DIR = "data/Validation"
MODEL_SAVE_DIR = "models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

IMG_W, IMG_H = 1920, 1080 

# 하이퍼파라미터 업그레이드
SEQ_LENGTH = 10 
PRED_LENGTH = 10 
INPUT_SIZE = 6      # ✅ [x, y, vx, vy, w, h]
HIDDEN_SIZE = 128  
NUM_LAYERS = 2     
BATCH_SIZE = 64
EPOCHS = 100        # 피처가 늘어났으므로 학습 횟수 상향
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 📊 [데이터셋] 6개 피처 자동 계산 및 정규화
# ==========================================
class TrajectoryDataset(Dataset):
    def __init__(self, data_dir):
        self.sequences = []
        self.labels = []
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        print(f"📂 {data_dir} 데이터 로드 및 피처 엔지니어링 중...")

        for file in csv_files:
            df = pd.read_csv(file)
            for track_id, group in df.groupby('track_id'):
                # 1. 기본 피처 추출
                data = group[['x_center', 'y_center', 'width', 'height']].values
                
                if len(data) < (SEQ_LENGTH + PRED_LENGTH): continue
                
                # 2. 속도(vx, vy) 계산 (차분 활용)
                # t시점의 속도 = t시점 위치 - (t-1)시점 위치
                velocity = np.diff(data[:, :2], axis=0, prepend=data[:1, :2])
                
                # 3. 6개 피처 결합 [x, y, vx, vy, w, h]
                features = np.hstack([data[:, :2], velocity, data[:, 2:]])
                
                # 4. 정규화 (전체 규격 대비 0~1 사이로)
                features[:, [0, 2, 4]] /= IMG_W  # x, vx, w
                features[:, [1, 3, 5]] /= IMG_H  # y, vy, h
                
                for i in range(len(features) - SEQ_LENGTH - PRED_LENGTH + 1):
                    self.sequences.append(features[i : i + SEQ_LENGTH])
                    # 라벨은 미래의 '좌표(x, y)'만 예측 (충돌 판정용)
                    self.labels.append(features[i + SEQ_LENGTH : i + SEQ_LENGTH + PRED_LENGTH, :2])

        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.labels = torch.FloatTensor(np.array(self.labels))
        print(f"✔️ 시퀀스 생성 완료: {len(self.sequences)}개")

    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

# ==========================================
# 🧠 [모델] LSTM (입력 피처 6개 대응)
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

# ==========================================
# 🚀 [학습] ADE/FDE 지표 측정 포함
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
    total_ade = 0
    total_fde = 0
    sample_count = 0
    
    with torch.no_grad():
        for v_seqs, v_targets in val_loader:
            v_seqs, v_targets = v_seqs.to(device), v_targets.to(device)
            v_outputs = model(v_seqs)
            val_loss += criterion(v_outputs, v_targets).item()
            
            # --- ADE/FDE 계산 (Pixel 단위 역정규화) ---
            # 실제 좌표와 예측 좌표의 차이 계산
            diff = (v_outputs - v_targets)
            diff[..., 0] *= IMG_W
            diff[..., 1] *= IMG_H
            dist = torch.sqrt(torch.sum(diff**2, dim=-1)) # (Batch, PRED_LENGTH)
            
            total_ade += torch.mean(dist).item() * v_seqs.size(0)
            total_fde += torch.mean(dist[:, -1]).item() * v_seqs.size(0)
            sample_count += v_seqs.size(0)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    ade = total_ade / sample_count
    fde = total_fde / sample_count
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | ADE: {ade:.2f}px | FDE: {fde:.2f}px")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_trajectory_model.pth"))
        print(f"⭐ 최고 성능 모델 업데이트! (ADE: {ade:.2f}px)")

print("\n🎉 고성능 피처 확장 모델 학습 완료!")