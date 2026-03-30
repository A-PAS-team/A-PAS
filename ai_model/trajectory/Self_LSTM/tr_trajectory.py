import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS Trajectory LSTM Training")
parser.add_argument('--fps',     type=int,   default=30,           choices=[10, 30],
                    help="FPS 설정 (10 or 30)")
parser.add_argument('--version', type=str,   default="v1_cctv",    choices=["v1_cctv", "v2_cctv_carla"],
                    help="데이터 버전 (v1_cctv or v2_cctv_carla)")
parser.add_argument('--hidden',  type=int,   default=None,
                    help="Hidden size (기본값: 10FPS=256, 30FPS=256)")
parser.add_argument('--epochs',  type=int,   default=150,
                    help="최대 에포크 수")
parser.add_argument('--patience',type=int,   default=15,
                    help="Early stopping patience")
args = parser.parse_args()

# ==========================================
# ⚙️ [설정] - FPS에 따라 자동 결정
# ==========================================
IMG_W, IMG_H = 1920, 1080
INPUT_SIZE   = 17
BATCH_SIZE   = 64
LR           = 0.001

if args.fps == 10:
    SEQ_LENGTH  = 20
    PRED_LENGTH = 10
    HIDDEN_SIZE = args.hidden if args.hidden else 256
else:  # 30
    SEQ_LENGTH  = 60
    PRED_LENGTH = 30
    HIDDEN_SIZE = args.hidden if args.hidden else 256

NUM_LAYERS          = 2
EPOCHS              = args.epochs
EARLY_STOP_PATIENCE = args.patience
FPS                 = args.fps
VERSION             = args.version

TAG      = f"{FPS}fps_{VERSION}"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "Training")

# Self LSTM 폴더에 저장
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "..", "models", "Self_LSTM")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(f"  A-PAS Trajectory LSTM Training")
print("=" * 60)
print(f"  FPS        : {FPS}")
print(f"  Version    : {VERSION}")
print(f"  SEQ_LENGTH : {SEQ_LENGTH}")
print(f"  PRED_LENGTH: {PRED_LENGTH}")
print(f"  HIDDEN_SIZE: {HIDDEN_SIZE}")
print(f"  EPOCHS     : {EPOCHS}")
print(f"  PATIENCE   : {EARLY_STOP_PATIENCE}")
print(f"  Device     : {device}")
print("=" * 60)

# ==========================================
# 📊 [데이터 로드]
# ==========================================
def load_data():
    print(f"\n📂 데이터 로드 중... [{TAG}]")
    X_train = np.load(os.path.join(DATA_DIR, f"X_train_final_{FPS}fps_{VERSION}.npy"))
    y_train = np.load(os.path.join(DATA_DIR, f"y_train_final_{FPS}fps_{VERSION}.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, f"X_val_final_{FPS}fps_{VERSION}.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, f"y_val_final_{FPS}fps_{VERSION}.npy"))
    print(f"  ✅ Train: {X_train.shape} / Val: {X_val.shape}")

    # 클래스 분포 출력
    class_names = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
    class_idx   = np.argmax(X_train[:, 0, 12:17], axis=1)
    print(f"\n  📊 클래스 분포 (Train):")
    for i, name in enumerate(class_names):
        count = (class_idx == i).sum()
        print(f"     {name:<12}: {count:>6}개 ({count/len(X_train)*100:.1f}%)")

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
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
    fig.suptitle(f"A-PAS LSTM - Training Report ({TAG})", fontsize=14)
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="tomato")
    axes[0].set_title("Train vs Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history["ade"], label="ADE (px)", color="mediumseagreen")
    axes[1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Target (10px)')
    axes[1].set_title("ADE (Average Displacement Error)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Pixel Error")
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs, history["fde"], label="FDE (px)", color="mediumpurple")
    axes[2].axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Target (15px)')
    axes[2].set_title("FDE (Final Displacement Error)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Pixel Error")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"training_report_{TAG}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 그래프 저장됨: {path}")

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

    print(f"\n🚀 학습 시작!")
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
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
        print(f"  📉 LR: {current_lr:.6f}")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            early_stop_count = 0
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"best_model_{TAG}.pth"))
            print(f"  ⭐ Best model saved! (Val: {avg_val:.5f} | ADE: {ade:.2f}px)")
        else:
            early_stop_count += 1
            print(f"  ⏳ Early stop: {early_stop_count}/{EARLY_STOP_PATIENCE}")

        if early_stop_count >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early Stopping! {epoch+1}에포크에서 종료.")
            break

    save_plots(history)

    # ==========================================
    # 📊 [최종 결과 미리보기]
    # ==========================================
    print(f"\n{'='*60}")
    print(f"  🎉 학습 완료! [{TAG}]")
    print(f"{'='*60}")
    print(f"  Best Val Loss : {best_val_loss:.5f}")
    print(f"  최종 ADE      : {history['ade'][-1]:.2f}px  {'✅' if history['ade'][-1] < 10 else '🟡' if history['ade'][-1] < 15 else '🔴'}")
    print(f"  최종 FDE      : {history['fde'][-1]:.2f}px  {'✅' if history['fde'][-1] < 15 else '🟡' if history['fde'][-1] < 20 else '🔴'}")
    print(f"  Best ADE      : {min(history['ade']):.2f}px")
    print(f"  Best FDE      : {min(history['fde']):.2f}px")
    print(f"  총 에포크     : {len(history['ade'])}")
    print(f"{'='*60}")
    print(f"\n📋 저장된 파일:")
    print(f"  모델  : models/Self_LSTM/best_model_{TAG}.pth")
    print(f"  그래프: models/Self_LSTM/training_report_{TAG}.png")
    print(f"\n📋 다음 단계:")
    print(f"  python comprehensive_eval.py --fps {FPS} --version {VERSION}")

if __name__ == "__main__":
    train()