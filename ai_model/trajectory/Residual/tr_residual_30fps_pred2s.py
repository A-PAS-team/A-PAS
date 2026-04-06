"""
tr_residual_30fps_pred2s.py
===========================
Residual LSTM: 1초 관찰 → 2초 예측 @ 30FPS
SEQ=30, PRED=60
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ [설정]
# ==========================================
IMG_W, IMG_H        = 1920, 1080
INPUT_SIZE          = 17
HIDDEN_SIZE         = 256
NUM_LAYERS          = 2
SEQ_LENGTH          = 30        # 1.0초 관찰
PRED_LENGTH         = 60        # 2.0초 예측
BATCH_SIZE          = 64
EPOCHS              = 200
LR                  = 0.001
EARLY_STOP_PATIENCE = 15
MODEL_DIR           = "models/Residual"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "Training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
print(f"📐 SEQ={SEQ_LENGTH}({SEQ_LENGTH/30:.1f}s) → PRED={PRED_LENGTH}({PRED_LENGTH/30:.1f}s)")

# ==========================================
# 📊 [데이터 로드]
# ==========================================
def load_data():
    print("📂 데이터 로드 중... [30FPS, 1s→2s, pred2s]")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_30fps_pred2s.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train_30fps_pred2s.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val_30fps_pred2s.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val_30fps_pred2s.npy"))
    print(f"  ✅ Train: {X_train.shape} / Val: {X_val.shape}")
    print(f"  ✅ y shape: {y_train.shape}  (N, {PRED_LENGTH}, 2)")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    return train_loader, val_loader

# ==========================================
# 🧠 [모델] Residual LSTM
# ==========================================
class ResidualLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        offset   = self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)
        last_pos = x[:, -1, :2].unsqueeze(1)
        return last_pos + offset

# ==========================================
# 📈 [그래프]
# ==========================================
def save_plots(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"A-PAS Residual LSTM — 1s Observe → 2s Predict (30FPS)",
                 fontsize=14, fontweight='bold')
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="Train", color="royalblue")
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="tomato")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ep, history["ade"], color="mediumseagreen")
    axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label="Target 10px")
    axes[1].set_title("ADE (px)"); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(ep, history["fde"], color="mediumpurple")
    axes[2].axhline(y=15, color='red', linestyle='--', alpha=0.5, label="Target 15px")
    axes[2].set_title("FDE (px)"); axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_report_residual_30fps_pred2s.png"), dpi=150)
    plt.close()

# ==========================================
# 🚀 [학습]
# ==========================================
def train():
    train_loader, val_loader = load_data()
    model     = ResidualLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📐 파라미터: {total_params:,}개")

    best_val = float('inf'); patience_cnt = 0
    history = {"train_loss":[], "val_loss":[], "ade":[], "fde":[]}

    for epoch in range(EPOCHS):
        model.train(); t_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b); loss.backward(); optimizer.step()
            t_loss += loss.item()

        model.eval(); v_loss=0; t_ade=0; t_fde=0; n=0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                v_loss += criterion(pred, y_b).item()
                p=pred.clone(); t=y_b.clone()
                p[...,0]*=IMG_W; p[...,1]*=IMG_H; t[...,0]*=IMG_W; t[...,1]*=IMG_H
                d = torch.sqrt(((p-t)**2).sum(-1))
                t_ade += d.mean(1).sum().item(); t_fde += d[:,-1].sum().item()
                n += X_b.size(0)

        at=t_loss/len(train_loader); av=v_loss/len(val_loader)
        ade=t_ade/n; fde=t_fde/n
        history["train_loss"].append(at); history["val_loss"].append(av)
        history["ade"].append(ade); history["fde"].append(fde)

        print(f"[{epoch+1:3d}/{EPOCHS}] T:{at:.5f} V:{av:.5f} "
              f"ADE:{ade:.2f} FDE:{fde:.2f} LR:{optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(av)

        if av < best_val:
            best_val=av; patience_cnt=0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_residual_30fps_pred2s.pth"))
            print(f"  ⭐ Best! ADE:{ade:.2f} FDE:{fde:.2f}")
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early Stop @ {epoch+1}"); break

    save_plots(history)
    print(f"\n{'='*55}")
    print(f"  Residual LSTM — 1s → 2s Prediction")
    print(f"  Best Val: {best_val:.5f}")
    print(f"  최종 ADE: {history['ade'][-1]:.2f}px")
    print(f"  최종 FDE: {history['fde'][-1]:.2f}px")
    print(f"  파라미터: {total_params:,}개")
    print(f"{'='*55}")

if __name__ == "__main__":
    train()
