import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# ==========================================
# ⚙️ [Settings]
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "Training")
MODEL_PATH  = os.path.join(BASE_DIR, "..", "models", "Self_LSTM", "best_model_10fps_jamsil.pth")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "models", "Self_LSTM", "eval")

IMG_W, IMG_H    = 1920, 1080
INPUT_SIZE      = 17
HIDDEN_SIZE     = 128
NUM_LAYERS      = 2
SEQ_LENGTH      = 20
PRED_LENGTH     = 10
DT              = 0.1

CLASS_NAMES  = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==========================================
# 🧠 [Model]
# ==========================================
class TrajectoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)

# ==========================================
# 📂 [Load]
# ==========================================
print("Loading data & model...")
X_val = np.load(os.path.join(DATA_DIR, "X_val_10fps_jamsil.npy"))
y_val = np.load(os.path.join(DATA_DIR, "y_val_10fps_jamsil.npy"))

model = TrajectoryLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

with torch.no_grad():
    preds_list = []
    for i in range(0, len(X_val), 512):
        batch = torch.FloatTensor(X_val[i:i+512]).to(device)
        preds_list.append(model(batch).cpu().numpy())
preds = np.concatenate(preds_list, axis=0)

# Denormalize
obs_px  = X_val[:, :, :2].copy()
gt_px   = y_val.copy()
pred_px = preds.copy()
obs_px[..., 0]  *= IMG_W;  obs_px[..., 1]  *= IMG_H
gt_px[..., 0]   *= IMG_W;  gt_px[..., 1]   *= IMG_H
pred_px[..., 0] *= IMG_W;  pred_px[..., 1] *= IMG_H

dist           = np.sqrt(np.sum((pred_px - gt_px)**2, axis=-1))  # (N, 10)
ade_per_sample = dist.mean(axis=1)
fde_per_sample = dist[:, -1]
class_indices  = np.argmax(X_val[:, 0, 12:17], axis=1)

print(f"  Done. Total {len(X_val)} samples\n")

# ==========================================
# 📊 [1] Overall Summary
# ==========================================
print("="*55)
print("📊 [1] Overall Performance Summary")
print("="*55)
print(f"  Total Samples  : {len(X_val):,}")
print(f"  Mean ADE       : {ade_per_sample.mean():.2f}px")
print(f"  Mean FDE       : {fde_per_sample.mean():.2f}px")
print(f"  Median ADE     : {np.median(ade_per_sample):.2f}px")
print(f"  Median FDE     : {np.median(fde_per_sample):.2f}px")
print(f"  ADE 90th pct   : {np.percentile(ade_per_sample, 90):.2f}px")
print(f"  FDE 90th pct   : {np.percentile(fde_per_sample, 90):.2f}px")

# ==========================================
# 📊 [2] Class-wise Performance
# ==========================================
print("\n" + "="*55)
print("📊 [2] Class-wise Performance")
print("="*55)
print(f"{'Class':<12} {'Count':>6} {'ADE(px)':>9} {'FDE(px)':>9} {'Grade':>6}")
print("-"*55)
class_results = {}
for i, name in enumerate(CLASS_NAMES):
    mask = class_indices == i
    if mask.sum() == 0:
        print(f"{name:<12} {'N/A':>6}")
        continue
    ade = ade_per_sample[mask].mean()
    fde = fde_per_sample[mask].mean()
    class_results[name] = {"count": int(mask.sum()), "ade": ade, "fde": fde}
    grade = "Good" if ade < 10 else ("Fair" if ade < 15 else "Poor")
    print(f"{name:<12} {mask.sum():>6} {ade:>9.2f} {fde:>9.2f} {grade:>6}")

# ==========================================
# 📊 [3] Speed-wise Performance
# ==========================================
print("\n" + "="*55)
print("📊 [3] Speed-wise Performance")
print("="*55)
speed = X_val[:, :, 6].mean(axis=1) * np.sqrt(IMG_W**2 + IMG_H**2)
slow_mask   = speed < np.percentile(speed, 33)
medium_mask = (speed >= np.percentile(speed, 33)) & (speed < np.percentile(speed, 66))
fast_mask   = speed >= np.percentile(speed, 66)

for label, mask in [("Slow   (bottom 33%)", slow_mask),
                    ("Medium (middle 33%)", medium_mask),
                    ("Fast   (top    33%)", fast_mask)]:
    ade = ade_per_sample[mask].mean()
    fde = fde_per_sample[mask].mean()
    print(f"  {label}: ADE {ade:.2f}px | FDE {fde:.2f}px | n={mask.sum()}")

# ==========================================
# 📊 [4] Error per Prediction Step
# ==========================================
print("\n" + "="*55)
print("📊 [4] Error per Prediction Step")
print("="*55)
step_errors = dist.mean(axis=0)
for t, err in enumerate(step_errors):
    bar = "█" * int(err / 2)
    print(f"  t+{(t+1)*DT:.1f}s : {err:>6.2f}px  {bar}")

# ==========================================
# 📊 [5] Motion Pattern: Straight vs Turning
# ==========================================
print("\n" + "="*55)
print("📊 [5] Motion Pattern Performance")
print("="*55)

# np.unwrap to prevent wrap-around at +/-pi boundary
vel = obs_px[:, 1:, :] - obs_px[:, :-1, :]
heading = np.arctan2(vel[:, :, 1], vel[:, :, 0])
heading = np.unwrap(heading, axis=1)
heading_change = np.abs(np.diff(heading, axis=1)).mean(axis=1)

straight_mask = heading_change < np.percentile(heading_change, 50)
turning_mask  = heading_change >= np.percentile(heading_change, 50)

ade_s = ade_per_sample[straight_mask].mean()
fde_s = fde_per_sample[straight_mask].mean()
ade_t = ade_per_sample[turning_mask].mean()
fde_t = fde_per_sample[turning_mask].mean()

print(f"  Straight : ADE {ade_s:.2f}px | FDE {fde_s:.2f}px | n={straight_mask.sum()}")
print(f"  Turning  : ADE {ade_t:.2f}px | FDE {fde_t:.2f}px | n={turning_mask.sum()}")
print(f"  -> Turning is worse by ADE {ade_t - ade_s:.2f}px")

# ==========================================
# 📊 [6] Error Distribution
# ==========================================
print("\n" + "="*55)
print("📊 [6] Error Distribution")
print("="*55)
for thr in [5, 10, 15, 20, 30]:
    ratio = (ade_per_sample < thr).mean() * 100
    print(f"  ADE < {thr:>2}px : {ratio:.1f}%")

# ==========================================
# 🎨 [Graphs]
# ==========================================
print("\nSaving graphs...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("A-PAS LSTM - Comprehensive Evaluation (10FPS_jamsil)", fontsize=14, fontweight='bold')

w = 0.35

# --- Graph 1: ADE/FDE by Class ---
ax = axes[0][0]
names = list(class_results.keys())
ades  = [class_results[n]["ade"] for n in names]
fdes  = [class_results[n]["fde"] for n in names]
x     = np.arange(len(names))
bars1 = ax.bar(x - w/2, ades, w, label='ADE', color='#4C72B0')
bars2 = ax.bar(x + w/2, fdes, w, label='FDE', color='#DD8452')
ax.axhline(10, color='green', linestyle='--', alpha=0.6, label='ADE Target (10px)')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_title("ADE / FDE by Class"); ax.set_ylabel("Pixel Error")
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=8)

# --- Graph 2: ADE/FDE by Speed ---
ax = axes[0][1]
speed_labels = ["Slow", "Medium", "Fast"]
ades_spd = [ade_per_sample[slow_mask].mean(),
            ade_per_sample[medium_mask].mean(),
            ade_per_sample[fast_mask].mean()]
fdes_spd = [fde_per_sample[slow_mask].mean(),
            fde_per_sample[medium_mask].mean(),
            fde_per_sample[fast_mask].mean()]
x = np.arange(3)
ax.bar(x - w/2, ades_spd, w, label='ADE', color='#55A868')
ax.bar(x + w/2, fdes_spd, w, label='FDE', color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(speed_labels)
ax.set_title("ADE / FDE by Speed"); ax.set_ylabel("Pixel Error")
ax.legend(); ax.grid(axis='y', alpha=0.3)

# --- Graph 3: Error per Prediction Step ---
ax = axes[0][2]
t_steps = [(t+1)*DT for t in range(PRED_LENGTH)]
ax.plot(t_steps, step_errors, 'o-', color='#8172B2', linewidth=2, markersize=6)
ax.fill_between(t_steps, step_errors, alpha=0.2, color='#8172B2')
ax.set_title("Error per Prediction Step")
ax.set_xlabel("Prediction Time (s)"); ax.set_ylabel("Mean Error (px)")
ax.grid(True, alpha=0.3)
for t, err in zip(t_steps, step_errors):
    ax.text(t, err+0.2, f'{err:.1f}', ha='center', fontsize=8)

# --- Graph 4: ADE Distribution ---
ax = axes[1][0]
ax.hist(ade_per_sample, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
ax.axvline(ade_per_sample.mean(),     color='red',    linestyle='--',
           label=f'Mean {ade_per_sample.mean():.1f}px')
ax.axvline(np.median(ade_per_sample), color='orange', linestyle='--',
           label=f'Median {np.median(ade_per_sample):.1f}px')
ax.set_title("ADE Error Distribution")
ax.set_xlabel("ADE (px)"); ax.set_ylabel("Sample Count")
ax.legend(); ax.grid(axis='y', alpha=0.3)

# --- Graph 5: Straight vs Turning ---
ax = axes[1][1]
categories = ["Straight", "Turning"]
ades_m = [ade_s, ade_t]
fdes_m = [fde_s, fde_t]
x = np.arange(2)
ax.bar(x - w/2, ades_m, w, label='ADE', color='#55A868')
ax.bar(x + w/2, fdes_m, w, label='FDE', color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_title("Straight vs Turning Performance"); ax.set_ylabel("Pixel Error")
ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (a, f) in enumerate(zip(ades_m, fdes_m)):
    ax.text(i-w/2, a+0.2, f'{a:.1f}', ha='center', fontsize=9)
    ax.text(i+w/2, f+0.2, f'{f:.1f}', ha='center', fontsize=9)

# --- Graph 6: ADE CDF ---
ax = axes[1][2]
thrs   = np.arange(0, 60, 0.5)
ratios = [(ade_per_sample < t).mean() * 100 for t in thrs]
ax.plot(thrs, ratios, color='#DD8452', linewidth=2)
for thr in [5, 10, 15, 20]:
    r = (ade_per_sample < thr).mean() * 100
    ax.axvline(thr, color='gray', linestyle='--', alpha=0.5)
    ax.text(thr+0.5, r-5, f'{thr}px\n{r:.0f}%', fontsize=8, color='gray')
ax.set_title("ADE Cumulative Distribution (CDF)")
ax.set_xlabel("ADE Threshold (px)"); ax.set_ylabel("Ratio (%)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "comprehensive_eval_10fps_jamsil.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"  Saved: {save_path}")
print("\nEvaluation complete!")