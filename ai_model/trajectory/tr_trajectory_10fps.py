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
# ⚙️ [설정] 10FPS
# ==========================================
IMG_W, IMG_H        = 1920, 1080
INPUT_SIZE          = 17
HIDDEN_SIZE         = 128
NUM_LAYERS          = 2
SEQ_LENGTH          = 20
PRED_LENGTH         = 10
BATCH_SIZE          = 64
EPOCHS              = 100
LR                  = 0.001
EARLY_STOP_PATIENCE = 15
MODEL_DIR           = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "Training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ==========================================
# 📊 [데이터 로드]
# ==========================================
def load_data():
    print("📂 데이터 로드 중... [10FPS]")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_10fps.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train_10fps.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val_10fps.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val_10fps.npy"))
    print(f"  ✅ Train: {X_train.shape} / Val: {X_val.shape}")

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
# 🧠 [모델]
# ==========================================
class TrajectoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)

# ==========================================
# 📈 [그래프 저장]
# ==========================================
def save_plots(history):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("A-PAS LSTM - Training Report (10FPS)", fontsize=14)
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
    axes[0].set_title("Train vs Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["ade"], label="ADE (px)", color="mediumseagreen")
    axes[1].set_title("ADE (Average Displacement Error)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Pixel Error")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, history["fde"], label="FDE (px)", color="mediumpurple")
    axes[2].set_title("FDE (Final Displacement Error)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Pixel Error")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_report_10fps.png"), dpi=150)
    plt.close()
    print("  📊 그래프 저장됨: models/training_report_10fps.png")

# ==========================================
# 🚀 [학습]
# ==========================================
def train():
    train_loader, val_loader = load_data()

    model     = TrajectoryLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss    = float('inf')
    early_stop_count = 0
    history = {"train_loss": [], "val_loss": [], "ade": [], "fde": []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0; total_ade = 0; total_fde = 0; n_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()

                p = pred.detach().clone()
                t = y_batch.detach().clone()
                p[..., 0] *= IMG_W; p[..., 1] *= IMG_H
                t[..., 0] *= IMG_W; t[..., 1] *= IMG_H

                dist = torch.sqrt(torch.sum((p - t) ** 2, dim=-1))
                total_ade += dist.mean(dim=1).sum().item()
                total_fde += dist[:, -1].sum().item()
                n_samples += X_batch.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        ade       = total_ade  / n_samples
        fde       = total_fde  / n_samples

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["ade"].append(ade)
        history["fde"].append(fde)

        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
              f"Train: {avg_train:.5f} | Val: {avg_val:.5f} | "
              f"ADE: {ade:.2f}px | FDE: {fde:.2f}px")

        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  📉 현재 LR: {current_lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            early_stop_count = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_model_10fps.pth"))
            print(f"  ⭐ Best model saved! (Val: {avg_val:.5f} | ADE: {ade:.2f}px)")
        else:
            early_stop_count += 1
            print(f"  ⏳ Early stop count: {early_stop_count}/{EARLY_STOP_PATIENCE}")

        if early_stop_count >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early Stopping! {epoch+1}에포크에서 종료.")
            break

    save_plots(history)
    print(f"\n🎉 학습 완료!")
    print(f"   Best Val Loss : {best_val_loss:.5f}")
    print(f"   최종 ADE      : {history['ade'][-1]:.2f}px")
    print(f"   최종 FDE      : {history['fde'][-1]:.2f}px")

if __name__ == "__main__":
    train()