import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS Trajectory Visualization")
parser.add_argument('--fps',     type=int, default=30, choices=[10, 30])
parser.add_argument('--version', type=str, default="v2_cctv_carla",
                    choices=["v1_cctv", "v2_cctv_carla"])
parser.add_argument('--model',   type=str, default="attention_residual",
                    choices=["self", "attention", "residual", "attention_residual"])
parser.add_argument('--num',     type=int, default=16, help="시각화할 샘플 수")
parser.add_argument('--mode',    type=str, default="all",
                    choices=["all", "best", "worst", "random"],
                    help="샘플 선택 모드")
args = parser.parse_args()

# ==========================================
# ⚙️ [설정]
# ==========================================
IMG_W, IMG_H = 1920, 1080
INPUT_SIZE   = 17

if args.fps == 10:
    SEQ_LENGTH  = 20; PRED_LENGTH = 10; HIDDEN_SIZE = 256; DT = 0.1
else:
    SEQ_LENGTH  = 60; PRED_LENGTH = 30; HIDDEN_SIZE = 256; DT = 1/30

FPS     = args.fps
VERSION = args.version
MODEL   = args.model
TAG     = f"{FPS}fps_{VERSION}"

MODEL_DIR_MAP = {
    "self": "Self_LSTM", "attention": "Attention",
    "residual": "Residual", "attention_residual": "Attention_Residual",
}
MODEL_FILE_MAP = {
    "self": f"best_model_{TAG}.pth", "attention": f"best_attn_{TAG}.pth",
    "residual": f"best_model_{TAG}.pth", "attention_residual": f"best_attn_res_{FPS}fps.pth",
}
CLASS_NAMES = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
CLASS_COLORS = {
    "Person": "#E74C3C", "Car": "#3498DB", "Bus": "#2ECC71",
    "Truck": "#F39C12", "Motorcycle": "#9B59B6"
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "Training")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models", MODEL_DIR_MAP[MODEL])
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 🧠 [모델 정의] — 학습 코드와 동일
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
        self.lstm  = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.W_q   = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_k   = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_v   = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.scale = math.sqrt(HIDDEN_SIZE)
        self.fc    = nn.Linear(HIDDEN_SIZE * 2, PRED_LENGTH * 2)
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        q = self.W_q(h_n[-1]).unsqueeze(1)
        k = self.W_k(lstm_out); v = self.W_v(lstm_out)
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) / self.scale, dim=-1)
        ctx  = torch.bmm(attn, v).squeeze(1)
        return self.fc(torch.cat([h_n[-1], ctx], -1)).view(-1, PRED_LENGTH, 2)

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
        self.input_proj = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.lstm       = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.W_q = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_k = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_v = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.scale = math.sqrt(HIDDEN_SIZE)
        self.fc    = nn.Linear(HIDDEN_SIZE * 2, PRED_LENGTH * 2)
    def forward(self, x):
        projected = self.input_proj(x)
        lstm_out, (h_n, _) = self.lstm(projected)
        res = self.layer_norm(lstm_out + projected)
        q = self.W_q(h_n[-1]).unsqueeze(1)
        k = self.W_k(res); v = self.W_v(res)
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) / self.scale, dim=-1)
        ctx  = torch.bmm(attn, v).squeeze(1)
        return self.fc(torch.cat([h_n[-1], ctx], -1)).view(-1, PRED_LENGTH, 2)

MODEL_CLASS = {
    "self": SelfLSTM, "attention": AttentionLSTM,
    "residual": ResidualLSTM, "attention_residual": AttentionResidualLSTM,
}

# ==========================================
# 📂 [로드 + 추론]
# ==========================================
print(f"🔍 A-PAS Trajectory Visualization")
print(f"   Model: {MODEL} | {TAG} | Mode: {args.mode}")

X_val = np.load(os.path.join(DATA_DIR, f"X_val_final_{FPS}fps_{VERSION}.npy"))
y_val = np.load(os.path.join(DATA_DIR, f"y_val_final_{FPS}fps_{VERSION}.npy"))

model_path = os.path.join(MODEL_DIR, MODEL_FILE_MAP[MODEL])
model = MODEL_CLASS[MODEL]().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.no_grad():
    preds_list = []
    for i in range(0, len(X_val), 512):
        batch = torch.FloatTensor(X_val[i:i+512]).to(device)
        out = model(batch)
        if isinstance(out, tuple): out = out[0]
        preds_list.append(out.cpu().numpy())
preds = np.concatenate(preds_list, axis=0)

# Denormalize
obs_px  = X_val[:, :, :2].copy()
gt_px   = y_val.copy()
pred_px = preds.copy()
obs_px[..., 0]  *= IMG_W;  obs_px[..., 1]  *= IMG_H
gt_px[..., 0]   *= IMG_W;  gt_px[..., 1]   *= IMG_H
pred_px[..., 0] *= IMG_W;  pred_px[..., 1] *= IMG_H

dist           = np.sqrt(np.sum((pred_px - gt_px)**2, axis=-1))
ade_per_sample = dist.mean(axis=1)
fde_per_sample = dist[:, -1]
class_indices  = np.argmax(X_val[:, 0, 12:17], axis=1)

print(f"   Samples: {len(X_val):,} | Mean ADE: {ade_per_sample.mean():.2f}px\n")

# ==========================================
# 🎯 [샘플 선택]
# ==========================================
N = min(args.num, len(X_val))

if args.mode == "best":
    indices = np.argsort(ade_per_sample)[:N]
    title_suffix = "Best Cases (Lowest ADE)"
elif args.mode == "worst":
    indices = np.argsort(ade_per_sample)[-N:][::-1]
    title_suffix = "Worst Cases (Highest ADE)"
elif args.mode == "random":
    indices = np.random.choice(len(X_val), N, replace=False)
    title_suffix = "Random Samples"
else:  # "all" — best + worst + random 혼합
    n_each = N // 3
    best_idx   = np.argsort(ade_per_sample)[:n_each]
    worst_idx  = np.argsort(ade_per_sample)[-n_each:][::-1]
    remaining  = set(range(len(X_val))) - set(best_idx) - set(worst_idx)
    random_idx = np.random.choice(list(remaining), N - 2*n_each, replace=False)
    indices    = np.concatenate([best_idx, random_idx, worst_idx])
    title_suffix = "Best + Random + Worst"

# ==========================================
# 🎨 [1] 개별 궤적 시각화 (Grid)
# ==========================================
cols = 4
rows = math.ceil(N / cols)
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
fig.suptitle(f"A-PAS {MODEL.upper()} — Trajectory Visualization ({TAG})\n{title_suffix}",
             fontsize=14, fontweight='bold')

if rows == 1: axes = axes[np.newaxis, :]
if cols == 1: axes = axes[:, np.newaxis]

for idx_in_grid, sample_idx in enumerate(indices):
    r, c = idx_in_grid // cols, idx_in_grid % cols
    ax = axes[r][c]

    obs  = obs_px[sample_idx]       # (SEQ, 2)
    gt   = gt_px[sample_idx]        # (PRED, 2)
    pred = pred_px[sample_idx]      # (PRED, 2)
    cls  = CLASS_NAMES[class_indices[sample_idx]]
    ade  = ade_per_sample[sample_idx]
    fde  = fde_per_sample[sample_idx]

    # 관찰 궤적 (회색 점선)
    ax.plot(obs[:, 0], obs[:, 1], 'o-', color='gray', markersize=1.5,
            linewidth=1, alpha=0.6, label='Observed')

    # 정답 궤적 (초록 실선)
    full_gt = np.vstack([obs[-1:], gt])
    ax.plot(full_gt[:, 0], full_gt[:, 1], 's-', color='#2ECC71', markersize=3,
            linewidth=2, label='Ground Truth')

    # 예측 궤적 (빨간 점선)
    full_pred = np.vstack([obs[-1:], pred])
    ax.plot(full_pred[:, 0], full_pred[:, 1], '^--', color='#E74C3C', markersize=3,
            linewidth=2, label='Predicted')

    # 관찰→예측 연결점 강조
    ax.plot(obs[-1, 0], obs[-1, 1], 'ko', markersize=6, zorder=5)

    # 최종 지점 오차 시각화
    ax.plot([gt[-1, 0], pred[-1, 0]], [gt[-1, 1], pred[-1, 1]],
            'r-', linewidth=1.5, alpha=0.5)

    ax.set_title(f"#{sample_idx} [{cls}]\nADE:{ade:.1f} FDE:{fde:.1f}",
                 fontsize=9, color='darkred' if ade > 5 else 'darkgreen')
    ax.set_aspect('equal')
    ax.invert_yaxis()  # 이미지 좌표계 (y축 반전)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)

    if idx_in_grid == 0:
        ax.legend(fontsize=6, loc='upper left')

# 빈 subplot 처리
for idx_in_grid in range(len(indices), rows * cols):
    r, c = idx_in_grid // cols, idx_in_grid % cols
    axes[r][c].axis('off')

plt.tight_layout()
save_path = os.path.join(MODEL_DIR,
    f"trajectory_viz_{MODEL}_{TAG}_{args.mode}.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Grid 저장: {save_path}")

# ==========================================
# 🎨 [2] 전체 궤적 오버레이 (1920x1080 캔버스)
# ==========================================
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)  # y축 반전
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')
ax.set_title(f"A-PAS {MODEL.upper()} — All Trajectories Overlay ({TAG})\n"
             f"ADE: {ade_per_sample.mean():.2f}px | FDE: {fde_per_sample.mean():.2f}px",
             fontsize=13, fontweight='bold', color='white')

# 랜덤 200개 샘플 오버레이
overlay_n = min(200, len(X_val))
overlay_idx = np.random.choice(len(X_val), overlay_n, replace=False)

for si in overlay_idx:
    obs  = obs_px[si]
    gt   = gt_px[si]
    pred = pred_px[si]
    cls  = CLASS_NAMES[class_indices[si]]

    # 관찰 (어두운 선)
    ax.plot(obs[:, 0], obs[:, 1], '-', color='gray', linewidth=0.3, alpha=0.3)

    # 정답 (초록)
    full_gt = np.vstack([obs[-1:], gt])
    ax.plot(full_gt[:, 0], full_gt[:, 1], '-', color='#2ECC71',
            linewidth=0.5, alpha=0.4)

    # 예측 (해당 클래스 색상)
    full_pred = np.vstack([obs[-1:], pred])
    ax.plot(full_pred[:, 0], full_pred[:, 1], '-',
            color=CLASS_COLORS.get(cls, 'white'), linewidth=0.5, alpha=0.5)

# 범례
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='gray',    linewidth=1, label='Observed'),
    Line2D([0],[0], color='#2ECC71', linewidth=1, label='Ground Truth'),
]
for cls_name, cls_color in CLASS_COLORS.items():
    legend_elements.append(
        Line2D([0],[0], color=cls_color, linewidth=1, label=f'Pred ({cls_name})')
    )
ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
          facecolor='#2a2a3e', edgecolor='gray', labelcolor='white')

ax.set_xlabel("X (px)", color='white'); ax.set_ylabel("Y (px)", color='white')
ax.tick_params(colors='white')

plt.tight_layout()
save_path2 = os.path.join(MODEL_DIR,
    f"trajectory_overlay_{MODEL}_{TAG}.png")
plt.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  📊 Overlay 저장: {save_path2}")

# ==========================================
# 🎨 [3] 클래스별 Best/Worst 비교
# ==========================================
fig, axes = plt.subplots(2, 5, figsize=(25, 8))
fig.suptitle(f"A-PAS {MODEL.upper()} — Best vs Worst per Class ({TAG})",
             fontsize=14, fontweight='bold')

for ci, cls_name in enumerate(CLASS_NAMES):
    mask = class_indices == ci
    if mask.sum() < 2:
        axes[0][ci].set_title(f"{cls_name}\nN/A"); axes[0][ci].axis('off')
        axes[1][ci].set_title(f"{cls_name}\nN/A"); axes[1][ci].axis('off')
        continue

    cls_ade = ade_per_sample[mask]
    cls_indices = np.where(mask)[0]

    # Best
    best_si = cls_indices[np.argmin(cls_ade)]
    ax = axes[0][ci]
    obs = obs_px[best_si]; gt = gt_px[best_si]; pred = pred_px[best_si]
    ax.plot(obs[:,0], obs[:,1], 'o-', color='gray', markersize=1, linewidth=1, alpha=0.5)
    ax.plot(np.vstack([obs[-1:],gt])[:,0], np.vstack([obs[-1:],gt])[:,1],
            's-', color='#2ECC71', markersize=2, linewidth=2)
    ax.plot(np.vstack([obs[-1:],pred])[:,0], np.vstack([obs[-1:],pred])[:,1],
            '^--', color='#E74C3C', markersize=2, linewidth=2)
    ax.set_title(f"{cls_name} BEST\nADE:{ade_per_sample[best_si]:.1f}",
                 fontsize=9, color='darkgreen')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(True, alpha=0.2)

    # Worst
    worst_si = cls_indices[np.argmax(cls_ade)]
    ax = axes[1][ci]
    obs = obs_px[worst_si]; gt = gt_px[worst_si]; pred = pred_px[worst_si]
    ax.plot(obs[:,0], obs[:,1], 'o-', color='gray', markersize=1, linewidth=1, alpha=0.5)
    ax.plot(np.vstack([obs[-1:],gt])[:,0], np.vstack([obs[-1:],gt])[:,1],
            's-', color='#2ECC71', markersize=2, linewidth=2)
    ax.plot(np.vstack([obs[-1:],pred])[:,0], np.vstack([obs[-1:],pred])[:,1],
            '^--', color='#E74C3C', markersize=2, linewidth=2)
    ax.set_title(f"{cls_name} WORST\nADE:{ade_per_sample[worst_si]:.1f}",
                 fontsize=9, color='darkred')
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.grid(True, alpha=0.2)

plt.tight_layout()
save_path3 = os.path.join(MODEL_DIR,
    f"trajectory_class_compare_{MODEL}_{TAG}.png")
plt.savefig(save_path3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  📊 Class 비교 저장: {save_path3}")

print(f"\n🎉 시각화 완료! ({MODEL_DIR})")