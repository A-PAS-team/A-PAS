import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ [설정] 30FPS Attention LSTM (sqrt(d_k) 적용)
# ==========================================
IMG_W, IMG_H        = 1920, 1080
INPUT_SIZE          = 17
HIDDEN_SIZE         = 256
NUM_LAYERS          = 2
SEQ_LENGTH          = 60       # 2.0초 @ 30FPS
PRED_LENGTH         = 30       # 1.0초 @ 30FPS
BATCH_SIZE          = 64
EPOCHS              = 200
LR                  = 0.001
EARLY_STOP_PATIENCE = 15
MODEL_DIR           = "models/Attention"
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "..", "data", "Training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ==========================================
# 📊 [데이터 로드]
# ==========================================
def load_data():
    print("📂 데이터 로드 중... [30FPS Attention]")
    X_train = np.load(os.path.join(DATA_DIR, "X_train_final_30fps_v2_cctv_carla.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train_final_30fps_v2_cctv_carla.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val_final_30fps_v2_cctv_carla.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val_final_30fps_v2_cctv_carla.npy"))
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
# 🧠 [모델] Attention LSTM
#   - Scaled Dot-Product Attention (sqrt(d_k))
#   - h_n[-1] + context concat → FC
#   - forward() returns (pred, attn_weights)
# ==========================================
class AttentionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.W_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.scale = math.sqrt(HIDDEN_SIZE)  # sqrt(256) = 16.0

        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
        )

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)

        query  = self.W_q(h_n[-1]).unsqueeze(1)
        keys   = self.W_k(lstm_out)
        values = self.W_v(lstm_out)

        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values).squeeze(1)

        combined = torch.cat([h_n[-1], context], dim=-1)
        pred = self.fc(combined).view(-1, PRED_LENGTH, 2)
        return pred, attn_weights.squeeze(1)

# ==========================================
# 📈 [그래프 + 히트맵]
# ==========================================
def save_plots(history):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("A-PAS Attention LSTM (sqrt(d_k))", fontsize=14, fontweight='bold')
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0,0].plot(ep, history["train_loss"], label="Train", color="royalblue")
    axes[0,0].plot(ep, history["val_loss"],   label="Val",   color="tomato")
    axes[0,0].set_title("Loss"); axes[0,0].legend(); axes[0,0].grid(True)

    axes[0,1].plot(ep, history["ade"], color="mediumseagreen")
    axes[0,1].axhline(y=10, color='red', linestyle='--', alpha=0.5)
    axes[0,1].set_title("ADE (px)"); axes[0,1].grid(True)

    axes[1,0].plot(ep, history["fde"], color="mediumpurple")
    axes[1,0].axhline(y=15, color='red', linestyle='--', alpha=0.5)
    axes[1,0].set_title("FDE (px)"); axes[1,0].grid(True)

    axes[1,1].plot(ep, history["attn_std"], color="darkorange")
    axes[1,1].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label="Healthy: 0.05")
    axes[1,1].set_title("Attention std (↑=Sharper)"); axes[1,1].legend(); axes[1,1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_report_attn_30fps.png"), dpi=150)
    plt.close()

def save_attention_heatmap(model, val_loader, epoch):
    model.eval()
    with torch.no_grad():
        X_b, _ = next(iter(val_loader))
        _, attn = model(X_b[:8].to(device))
    fig, axes = plt.subplots(2, 4, figsize=(20, 6))
    fig.suptitle(f"Attention Weights @ Epoch {epoch}")
    for i, ax in enumerate(axes.flat):
        w = attn[i].cpu().numpy()
        ax.bar(range(SEQ_LENGTH), w, color='steelblue', width=1.0)
        ax.set_ylim(0, max(0.05, w.max()*1.2))
        ax.set_title(f"#{i+1} std={w.std():.4f}")
        ax.axhline(y=1/SEQ_LENGTH, color='red', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"attn_heatmap_ep{epoch:03d}.png"), dpi=100)
    plt.close()

# ==========================================
# 🚀 [학습]
# ==========================================
def train():
    train_loader, val_loader = load_data()
    model     = AttentionLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    print(f"  📐 파라미터: {sum(p.numel() for p in model.parameters()):,}개")
    print(f"  📐 scale = sqrt({HIDDEN_SIZE}) = {math.sqrt(HIDDEN_SIZE):.1f}")

    best_val = float('inf'); patience_cnt = 0
    history = {"train_loss":[], "val_loss":[], "ade":[], "fde":[], "attn_std":[]}

    for epoch in range(EPOCHS):
        model.train(); t_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred, _ = model(X_b)
            loss = criterion(pred, y_b); loss.backward(); optimizer.step()
            t_loss += loss.item()

        model.eval(); v_loss=0; t_ade=0; t_fde=0; n=0; a_stds=[]
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred, attn = model(X_b)
                v_loss += criterion(pred, y_b).item()
                p=pred.clone(); t=y_b.clone()
                p[...,0]*=IMG_W; p[...,1]*=IMG_H; t[...,0]*=IMG_W; t[...,1]*=IMG_H
                d = torch.sqrt(((p-t)**2).sum(-1))
                t_ade += d.mean(1).sum().item(); t_fde += d[:,-1].sum().item()
                n += X_b.size(0); a_stds.append(attn.std(-1).mean().item())

        at=t_loss/len(train_loader); av=v_loss/len(val_loader)
        ade=t_ade/n; fde=t_fde/n; astd=np.mean(a_stds)
        history["train_loss"].append(at); history["val_loss"].append(av)
        history["ade"].append(ade); history["fde"].append(fde); history["attn_std"].append(astd)

        print(f"[{epoch+1:3d}/{EPOCHS}] T:{at:.5f} V:{av:.5f} "
              f"ADE:{ade:.2f} FDE:{fde:.2f} Attn:{astd:.4f} LR:{optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step(av)

        if (epoch+1) % 10 == 0: save_attention_heatmap(model, val_loader, epoch+1)

        if av < best_val:
            best_val=av; patience_cnt=0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_attn_30fps.pth"))
            print(f"  ⭐ Best! ADE:{ade:.2f} FDE:{fde:.2f} Attn:{astd:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOP_PATIENCE:
                print(f"\n🛑 Early Stop @ {epoch+1}"); break

    save_plots(history)
    fs = history['attn_std'][-1]
    print(f"\n{'='*55}")
    print(f"  Best Val: {best_val:.5f} | ADE: {history['ade'][-1]:.2f} | FDE: {history['fde'][-1]:.2f}")
    print(f"  Attn std: {history['attn_std'][0]:.4f} → {fs:.4f}")
    print(f"{'='*55}")
    if   fs > 0.05: print("  ✅ Attention 활성 → 최종 후보!")
    elif fs > 0.02: print("  ⚠️ Attention 약함 → ADE 보고 판단")
    else:           print("  ℹ️ Attention ≈ uniform → Residual 전환 권장")

if __name__ == "__main__":
    train()