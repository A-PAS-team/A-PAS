import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS Trajectory Visualization")
parser.add_argument('--fps',     type=int, default=30, choices=[10, 30])
parser.add_argument('--version', type=str, default="v2_cctv_carla",
                    choices=["v1_cctv", "v2_cctv_carla"])
parser.add_argument('--model',   type=str, default="residual",
                    choices=["self", "attention", "residual", "attention_residual"])
parser.add_argument('--samples', type=int, default=3,
                    help="클래스별 샘플 수 (기본값: 3)")
args = parser.parse_args()

# ==========================================
# ⚙️ [설정]
# ==========================================
IMG_W, IMG_H = 1920, 1080
INPUT_SIZE   = 17

if args.fps == 10:
    SEQ_LENGTH  = 20
    PRED_LENGTH = 10
    HIDDEN_SIZE = 256
else:
    SEQ_LENGTH  = 60
    PRED_LENGTH = 30
    HIDDEN_SIZE = 256

FPS              = args.fps
VERSION          = args.version
MODEL            = args.model
TAG              = f"{FPS}fps_{VERSION}"
SAMPLES_PER_CLASS = args.samples

MODEL_DIR_MAP = {
    "self"               : "Self_LSTM",
    "attention"          : "Attention",
    "residual"           : "Residual",
    "attention_residual" : "Attention_Residual",
}
MODEL_LABEL = {
    "self"               : "Self LSTM",
    "attention"          : "Attention LSTM",
    "residual"           : "Residual LSTM",
    "attention_residual" : "Attention+Residual LSTM",
}

CLASS_NAMES  = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "Training")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models", MODEL_DIR_MAP[MODEL])
OUTPUT_DIR = os.path.join(MODEL_DIR, "..", MODEL_DIR_MAP[MODEL])
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Model: {MODEL_LABEL[MODEL]} | {TAG}")

# ==========================================
# 🧠 [모델 정의]
# ==========================================
class SelfLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)

class AttentionLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.attn_fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc      = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        query        = h_n[-1]
        query_proj   = self.attn_fc(query).unsqueeze(2)
        attn_scores  = torch.bmm(lstm_out, query_proj).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context      = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.fc(context).view(-1, PRED_LENGTH, 2)

class ResidualLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        offset   = self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)
        last_pos = x[:, -1, :2].unsqueeze(1)
        return last_pos + offset

class AttentionResidualLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm    = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.attn_fc = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc      = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        query        = h_n[-1]
        query_proj   = self.attn_fc(query).unsqueeze(2)
        attn_scores  = torch.bmm(lstm_out, query_proj).squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context      = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        offset       = self.fc(context).view(-1, PRED_LENGTH, 2)
        last_pos     = x[:, -1, :2].unsqueeze(1)
        return last_pos + offset

MODEL_CLASS = {
    "self"               : SelfLSTM,
    "attention"          : AttentionLSTM,
    "residual"           : ResidualLSTM,
    "attention_residual" : AttentionResidualLSTM,
}

# ==========================================
# 📂 [데이터 & 모델 로드]
# ==========================================
print("\nLoading data & model...")
X_val = np.load(os.path.join(DATA_DIR, f"X_val_final_{FPS}fps_{VERSION}.npy"))
y_val = np.load(os.path.join(DATA_DIR, f"y_val_final_{FPS}fps_{VERSION}.npy"))

model_path = os.path.join(MODEL_DIR, f"best_model_{TAG}.pth")
model = MODEL_CLASS[MODEL]().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"  Val: {X_val.shape}")

# ==========================================
# 🎯 [클래스별 샘플 선택 및 추론]
# ==========================================
class_indices = np.argmax(X_val[:, 0, 12:17], axis=1)

selected = {}
for i, name in enumerate(CLASS_NAMES):
    mask = np.where(class_indices == i)[0]
    if len(mask) == 0:
        print(f"  ⚠️  {name}: 샘플 없음, 스킵")
        continue
    n = min(SAMPLES_PER_CLASS, len(mask))
    chosen = np.random.choice(mask, n, replace=False)
    selected[name] = chosen
    print(f"  ✅ {name}: {n}개 선택 (전체 {len(mask)}개 중)")

all_indices = np.concatenate(list(selected.values()))
X_sel = torch.FloatTensor(X_val[all_indices]).to(device)

with torch.no_grad():
    preds = model(X_sel).cpu().numpy()

# 역정규화
obs_px  = X_val[all_indices, :, :2].copy()
gt_px   = y_val[all_indices].copy()
pred_px = preds.copy()

obs_px[:, :, 0]  *= IMG_W;  obs_px[:, :, 1]  *= IMG_H
gt_px[:, :, 0]   *= IMG_W;  gt_px[:, :, 1]   *= IMG_H
pred_px[:, :, 0] *= IMG_W;  pred_px[:, :, 1] *= IMG_H

# ==========================================
# 🎨 [시각화]
# ==========================================
n_classes = len(selected)
fig, axes = plt.subplots(n_classes, SAMPLES_PER_CLASS,
                         figsize=(SAMPLES_PER_CLASS * 5, n_classes * 4))
fig.suptitle(
    f"A-PAS {MODEL_LABEL[MODEL]} - Trajectory Prediction vs Ground Truth ({TAG})",
    fontsize=13, fontweight='bold'
)

if n_classes == 1:
    axes = [axes]

sample_idx = 0
for row, (name, indices) in enumerate(selected.items()):
    color = CLASS_COLORS[CLASS_NAMES.index(name)]
    for col in range(SAMPLES_PER_CLASS):
        ax = axes[row][col] if SAMPLES_PER_CLASS > 1 else axes[row]

        if col >= len(indices):
            ax.axis('off')
            continue

        obs  = obs_px[sample_idx]
        gt   = gt_px[sample_idx]
        pred = pred_px[sample_idx]

        dist = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
        ade  = dist.mean()
        fde  = dist[-1]

        # 관찰 경로
        ax.plot(obs[:, 0], obs[:, 1], 'o-',
                color='gray', linewidth=1.5, markersize=3,
                label='Observed', alpha=0.7)

        # Ground Truth 연결
        ax.plot([obs[-1, 0], gt[0, 0]], [obs[-1, 1], gt[0, 1]],
                '--', color='green', linewidth=1, alpha=0.5)
        ax.plot(gt[:, 0], gt[:, 1], 's-',
                color='green', linewidth=2, markersize=4,
                label='Ground Truth')

        # 예측 경로 연결
        ax.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]],
                '--', color=color, linewidth=1, alpha=0.5)
        ax.plot(pred[:, 0], pred[:, 1], '^--',
                color=color, linewidth=2, markersize=4,
                label='Predicted')

        # 시작/끝 강조
        ax.scatter(obs[0, 0],   obs[0, 1],   s=60, color='gray',  zorder=5)
        ax.scatter(obs[-1, 0],  obs[-1, 1],  s=80, color='black', zorder=5, marker='*')
        ax.scatter(gt[-1, 0],   gt[-1, 1],   s=80, color='green', zorder=5, marker='*')
        ax.scatter(pred[-1, 0], pred[-1, 1], s=80, color=color,   zorder=5, marker='*')

        ax.set_title(f"{name} #{col+1}\nADE: {ade:.1f}px | FDE: {fde:.1f}px",
                     fontsize=9)
        ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
        ax.set_xlabel("X (px)", fontsize=7); ax.set_ylabel("Y (px)", fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)

        if col == 0:
            ax.legend(fontsize=7, loc='best')

        sample_idx += 1

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"trajectory_{MODEL}_{TAG}.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  Saved: {save_path}")
print("Visualization complete!")