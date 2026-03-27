import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "Training")
MODEL_PATH  = os.path.join(BASE_DIR, "..", "models", "Self_LSTM", "best_model_10fps_jamsil.pth")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "models", "Self_LSTM")

IMG_W, IMG_H    = 1920, 1080
INPUT_SIZE      = 17
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
SEQ_LENGTH      = 20
PRED_LENGTH     = 10

CLASS_NAMES  = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

SAMPLES_PER_CLASS = 3   # 클래스별 샘플 수
RANDOM_SEED       = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ==========================================
# 🧠 [모델 정의]
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
# 📂 [데이터 로드]
# ==========================================
print("📂 데이터 로드 중...")
X_val = np.load(os.path.join(DATA_DIR, "X_val_10fps_jamsil.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val_10fps_jamsil.npy"))
print(f"  ✅ Val: {X_val.shape}")

# ==========================================
# 🧠 [모델 로드]
# ==========================================
print("📦 모델 로드 중...")
model = TrajectoryLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("  ✅ 완료")

# ==========================================
# 🎯 [클래스별 샘플 추출 및 추론]
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

# 추론
all_indices = np.concatenate(list(selected.values()))
X_sel = torch.FloatTensor(X_val[all_indices]).to(device)

with torch.no_grad():
    preds = model(X_sel).cpu().numpy()  # (N, 10, 2)

# 역정규화
obs_px  = X_val[all_indices, :, :2].copy()
gt_px   = y_val[all_indices].copy()
pred_px = preds.copy()

obs_px[:, :, 0]  *= IMG_W;  obs_px[:, :, 1]  *= IMG_H
gt_px[:, :, 0]   *= IMG_W;  gt_px[:, :, 1]   *= IMG_H
pred_px[:, :, 0] *= IMG_W;  pred_px[:, :, 1] *= IMG_H

# ==========================================
# 📊 [시각화]
# ==========================================
n_classes = len(selected)
fig, axes = plt.subplots(n_classes, SAMPLES_PER_CLASS,
                         figsize=(SAMPLES_PER_CLASS * 5, n_classes * 4))
fig.suptitle("A-PAS LSTM - Trajectory Prediction vs Ground Truth (10FPS_jamsil)",
             fontsize=14, fontweight='bold')

# axes가 1D일 경우 대비
if n_classes == 1:
    axes = [axes]

sample_idx = 0
for row, (name, indices) in enumerate(selected.items()):
    color = CLASS_COLORS[CLASS_NAMES.index(name)]
    for col in range(SAMPLES_PER_CLASS):
        ax = axes[row][col] if len(indices) > 1 else axes[row][0]

        if col >= len(indices):
            ax.axis('off')
            continue

        obs  = obs_px[sample_idx]   # (20, 2)
        gt   = gt_px[sample_idx]    # (10, 2)
        pred = pred_px[sample_idx]  # (10, 2)

        # ADE / FDE 계산
        dist = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
        ade  = dist.mean()
        fde  = dist[-1]

        # 관찰 경로 (실선)
        ax.plot(obs[:, 0], obs[:, 1], 'o-',
                color='gray', linewidth=1.5, markersize=3,
                label='Observed', alpha=0.7)

        # 실제 미래 경로 (실선)
        # obs 마지막 점 → gt 첫 점 연결
        connect_x = [obs[-1, 0], gt[0, 0]]
        connect_y = [obs[-1, 1], gt[0, 1]]
        ax.plot(connect_x, connect_y, '--', color='green', linewidth=1, alpha=0.5)
        ax.plot(gt[:, 0], gt[:, 1], 's-',
                color='green', linewidth=2, markersize=4,
                label='Ground Truth')

        # 예측 미래 경로 (점선)
        connect_x2 = [obs[-1, 0], pred[0, 0]]
        connect_y2 = [obs[-1, 1], pred[0, 1]]
        ax.plot(connect_x2, connect_y2, '--', color=color, linewidth=1, alpha=0.5)
        ax.plot(pred[:, 0], pred[:, 1], '^--',
                color=color, linewidth=2, markersize=4,
                label='Predicted')

        # 시작/끝 점 강조
        ax.scatter(obs[0, 0],   obs[0, 1],   s=60, color='gray',  zorder=5)
        ax.scatter(obs[-1, 0],  obs[-1, 1],  s=80, color='black', zorder=5, marker='*')
        ax.scatter(gt[-1, 0],   gt[-1, 1],   s=80, color='green', zorder=5, marker='*')
        ax.scatter(pred[-1, 0], pred[-1, 1], s=80, color=color,   zorder=5, marker='*')

        ax.set_title(f"{name} #{col+1}\nADE: {ade:.1f}px | FDE: {fde:.1f}px",
                     fontsize=9)
        ax.set_xlim(0, IMG_W)
        ax.set_ylim(IMG_H, 0)  # y축 반전 (이미지 좌표계)
        ax.set_xlabel("X (px)", fontsize=7)
        ax.set_ylabel("Y (px)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.legend(fontsize=7, loc='best')

        sample_idx += 1

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "trajectory_visualization_10fps_jamsil.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  📊 저장됨: {save_path}")
print("🎉 시각화 완료!")