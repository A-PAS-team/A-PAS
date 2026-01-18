import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob  # 👈 [추가] 파일 목록 찾는 라이브러리

# ==========================================
# ⚙️ [설정] CSV 파일들이 모여있는 폴더 패턴
# ==========================================
# 예: data 폴더 안에 있는 모든 csv 파일
# 또는 "normal_data_*.csv" 라고 쓰면 번호 달린 파일 다 가져옴
CSV_PATTERN = "data/normal_data_*.csv" 

MODEL_SAVE_PATH = "trajectory_model.pth"
PAST_FRAMES = 10
FUTURE_FRAMES = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
# ==========================================

class TrajectoryDataset(Dataset):
    def __init__(self, csv_pattern, past_len, future_len):
        self.past_len = past_len
        self.future_len = future_len
        self.samples = []
        
        # 1. 패턴에 맞는 모든 파일 찾기
        file_list = glob.glob(csv_pattern)
        
        if not file_list:
            print(f"❌ 오류: '{csv_pattern}'에 해당하는 파일이 하나도 없습니다!")
            return

        print(f"📂 총 {len(file_list)}개의 CSV 파일을 발견했습니다. 데이터 로딩 중...")

        # 2. 파일 하나씩 열어서 데이터 싹 긁어모으기
        for file_path in file_list:
            print(f"   Reading {file_path}...")
            try:
                df = pd.read_csv(file_path)
                
                # 좌표 컬럼 선택 (map_x 우선, 없으면 x_center)
                if 'map_x' in df.columns:
                    coords_col = ['map_x', 'map_y']
                else:
                    coords_col = ['x_center', 'y_center']

                # ID별로 그룹화
                # (중요: 파일 A의 1번 사람과 파일 B의 1번 사람은 다른 사람이므로
                #  파일 단위로 루프 안에서 처리해야 섞이지 않음! 👍)
                for track_id, group in df.groupby('track_id'):
                    group = group.sort_values('frame')
                    track_data = group[coords_col].values.astype(np.float32)
                    
                    if len(track_data) < past_len + future_len:
                        continue
                        
                    # 슬라이딩 윈도우
                    for i in range(len(track_data) - past_len - future_len + 1):
                        x_seq = track_data[i : i + past_len]
                        y_seq = track_data[i + past_len : i + past_len + future_len]
                        self.samples.append((x_seq, y_seq))
                        
            except Exception as e:
                print(f"   ⚠️ {file_path} 읽기 실패: {e}")

        print(f"✅ 모든 파일 로딩 완료! 총 샘플 수: {len(self.samples)}개")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# --- (아래 모델 정의와 학습 코드는 기존과 동일) ---

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, past_frames=10, future_frames=5):
        super(TrajectoryLSTM, self).__init__()
        self.future_frames = future_frames
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, future_frames * 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        predicted = self.fc(last_hidden)
        predicted = predicted.view(-1, self.future_frames, 2)
        return predicted

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 학습 시작! Device: {device}")
    
    # 여기서 csv_pattern을 넘겨줍니다.
    dataset = TrajectoryDataset(CSV_PATTERN, PAST_FRAMES, FUTURE_FRAMES)
    
    if len(dataset) == 0:
        print("데이터가 없어서 종료합니다.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = TrajectoryLSTM(past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 학습 완료! {MODEL_SAVE_PATH}")
    
    # 시각화 (데이터셋이 비어있지 않다면)
    if len(dataset) > 0:
        visualize_result(model, dataset, device)

def visualize_result(model, dataset, device):
    model.eval()
    idx = np.random.randint(0, len(dataset))
    x, y_true = dataset[idx]
    with torch.no_grad():
        x_input = x.unsqueeze(0).to(device)
        y_pred = model(x_input).cpu().squeeze(0)
        
    plt.figure(figsize=(8, 8))
    plt.plot(x[:, 0], x[:, 1], 'bo-', label='Past')
    plt.plot(y_true[:, 0], y_true[:, 1], 'go-', label='True Future')
    plt.plot(y_pred[:, 0], y_pred[:, 1], 'rx--', label='Predicted')
    plt.legend()
    plt.title("Trajectory Test")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()