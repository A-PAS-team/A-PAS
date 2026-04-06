import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS Comprehensive Evaluation")
parser.add_argument('--fps',     type=int, default=30, choices=[10, 30])
parser.add_argument('--version', type=str, default="v2_cctv_carla",
                    choices=["v1_cctv", "v2_cctv_carla"])
parser.add_argument('--model',   type=str, default="attention_residual",
                    choices=["self", "attention", "residual", "attention_residual"])
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
    DT          = 0.1
else:
    SEQ_LENGTH  = 60
    PRED_LENGTH = 30
    HIDDEN_SIZE = 256
    DT          = 1/30

FPS     = args.fps
VERSION = args.version
MODEL   = args.model
TAG     = f"{FPS}fps_{VERSION}"

MODEL_DIR_MAP = {
    "self"               : "Self_LSTM",
    "attention"          : "Attention",
    "residual"           : "Residual",
    "attention_residual" : "Attention_Residual",
}

# ✅ 모델별 .pth 파일명 매핑
MODEL_FILE_MAP = {
    "self"               : f"best_model_{TAG}.pth",
    "attention"          : f"best_attn_{TAG}.pth",
    "residual"           : f"best_model_{TAG}.pth",
    "attention_residual" : f"best_attn_res_{FPS}fps.pth",
}

CLASS_NAMES  = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "Training")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models", MODEL_DIR_MAP[MODEL])
OUTPUT_DIR = os.path.join(MODEL_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Model: {MODEL} | {TAG}")

# ==========================================
# 🧠 [모델 정의]
# ==========================================

# --- Self LSTM (기본) ---
class SelfLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)

# --- Attention LSTM (sqrt(d_k) 버전) ---
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
        k = self.W_k(lstm_out)
        v = self.W_v(lstm_out)
        attn = F.softmax(torch.bmm(q, k.transpose(1,2)) / self.scale, dim=-1)
        ctx  = torch.bmm(attn, v).squeeze(1)
        pred = self.fc(torch.cat([h_n[-1], ctx], -1)).view(-1, PRED_LENGTH, 2)
        return pred

# --- Residual LSTM ---
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

# --- ✅ Attention + Residual LSTM (학습 코드와 동일 구조) ---
class AttentionResidualLSTM(nn.Module):
    """
    tr_attention_residual_30fps.py와 완전 동일한 구조.
    - Input Projection (17 → 256)
    - LSTM (256 → 256)
    - Residual Connection + LayerNorm
    - Scaled Dot-Product Attention (sqrt(d_k))
    - Simple Linear FC (512 → 60)
    """
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.lstm       = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, 2,
                                  batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)
        self.W_q        = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_k        = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_v        = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.scale      = math.sqrt(HIDDEN_SIZE)
        self.fc         = nn.Linear(HIDDEN_SIZE * 2, PRED_LENGTH * 2)

    def forward(self, x):
        projected = self.input_proj(x)
        lstm_out, (h_n, _) = self.lstm(projected)
        residual_out = self.layer_norm(lstm_out + projected)

        query  = self.W_q(h_n[-1]).unsqueeze(1)
        keys   = self.W_k(residual_out)
        values = self.W_v(residual_out)

        scores       = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        context      = torch.bmm(attn_weights, values).squeeze(1)

        combined = torch.cat([h_n[-1], context], dim=-1)
        pred     = self.fc(combined).view(-1, PRED_LENGTH, 2)
        return pred  # eval에서는 attn_weights 불필요

MODEL_CLASS = {
    "self"               : SelfLSTM,
    "attention"          : AttentionLSTM,
    "residual"           : ResidualLSTM,
    "attention_residual" : AttentionResidualLSTM,
}

# ==========================================
# 📂 [로드]
# ==========================================
print("\nLoading data & model...")
X_val = np.load(os.path.join(DATA_DIR, f"X_val_final_{FPS}fps_{VERSION}.npy"))
y_val = np.load(os.path.join(DATA_DIR, f"y_val_final_{FPS}fps_{VERSION}.npy"))

model_file = MODEL_FILE_MAP[MODEL]
model_path = os.path.join(MODEL_DIR, model_file)
print(f"  Model file: {model_path}")

model = MODEL_CLASS[MODEL]().to(device)

# ✅ 학습 시 forward()가 tuple 반환하는 모델의 state_dict 호환 처리
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

with torch.no_grad():
    preds_list = []
    for i in range(0, len(X_val), 512):
        batch = torch.FloatTensor(X_val[i:i+512]).to(device)
        out   = model(batch)
        # tuple 반환하는 모델 대응 (학습용 코드에서 attn_weights 같이 반환하는 경우)
        if isinstance(out, tuple):
            out = out[0]
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

print(f"  Done. {len(X_val)} samples\n")

# ==========================================
# 📊 [1] Overall Summary
# ==========================================
print("="*55)
print(f"📊 [1] Overall Performance - {MODEL.upper()} ({TAG})")
print("="*55)
print(f"  Total Samples  : {len(X_val):,}")
print(f"  Mean ADE       : {ade_per_sample.mean():.2f}px")
print(f"  Mean FDE       : {fde_per_sample.mean():.2f}px")
print(f"  Median ADE     : {np.median(ade_per_sample):.2f}px")
print(f"  Median FDE     : {np.median(fde_per_sample):.2f}px")
print(f"  ADE 90th pct   : {np.percentile(ade_per_sample, 90):.2f}px")
print(f"  FDE 90th pct   : {np.percentile(fde_per_sample, 90):.2f}px")
print(f"  ADE 95th pct   : {np.percentile(ade_per_sample, 95):.2f}px")
print(f"  FDE 95th pct   : {np.percentile(fde_per_sample, 95):.2f}px")

# ==========================================
# 📊 [2] Class-wise
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
    grade = "Good" if ade < 5 else ("Fair" if ade < 10 else "Poor")
    print(f"{name:<12} {mask.sum():>6} {ade:>9.2f} {fde:>9.2f} {grade:>6}")

# ==========================================
# 📊 [3] Speed-wise
# ==========================================
print("\n" + "="*55)
print("📊 [3] Speed-wise Performance")
print("="*55)
speed       = X_val[:, :, 6].mean(axis=1) * np.sqrt(IMG_W**2 + IMG_H**2)
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
# 📊 [4] Step-wise
# ==========================================
print("\n" + "="*55)
print("📊 [4] Error per Prediction Step")
print("="*55)
step_errors = dist.mean(axis=0)
for t, err in enumerate(step_errors):
    bar = "█" * int(err / 1)  # 스케일 조정 (ADE가 낮아졌으므로)
    print(f"  t+{(t+1)*DT:.2f}s : {err:>6.2f}px  {bar}")

# ==========================================
# 📊 [5] Straight vs Turning
# ==========================================
print("\n" + "="*55)
print("📊 [5] Motion Pattern Performance")
print("="*55)
vel            = obs_px[:, 1:, :] - obs_px[:, :-1, :]
heading        = np.arctan2(vel[:, :, 1], vel[:, :, 0])
heading        = np.unwrap(heading, axis=1)
heading_change = np.abs(np.diff(heading, axis=1)).mean(axis=1)
straight_mask  = heading_change < np.percentile(heading_change, 50)
turning_mask   = heading_change >= np.percentile(heading_change, 50)
ade_s = ade_per_sample[straight_mask].mean()
fde_s = fde_per_sample[straight_mask].mean()
ade_t = ade_per_sample[turning_mask].mean()
fde_t = fde_per_sample[turning_mask].mean()
print(f"  Straight : ADE {ade_s:.2f}px | FDE {fde_s:.2f}px | n={straight_mask.sum()}")
print(f"  Turning  : ADE {ade_t:.2f}px | FDE {fde_t:.2f}px | n={turning_mask.sum()}")
print(f"  -> Turning worse by ADE {ade_t - ade_s:.2f}px")

# ==========================================
# 📊 [6] Error Distribution + Miss Rate
# ==========================================
print("\n" + "="*55)
print("📊 [6] Error Distribution & Miss Rate")
print("="*55)
for thr in [2, 3, 5, 10, 15, 20]:
    ade_ratio = (ade_per_sample < thr).mean() * 100
    fde_ratio = (fde_per_sample < thr).mean() * 100
    print(f"  < {thr:>2}px : ADE {ade_ratio:.1f}% | FDE {fde_ratio:.1f}%")

# Miss Rate (FDE가 임계값 초과한 비율)
print(f"\n  Miss Rate (FDE > 10px) : {(fde_per_sample > 10).mean()*100:.1f}%")
print(f"  Miss Rate (FDE > 20px) : {(fde_per_sample > 20).mean()*100:.1f}%")
print(f"  Miss Rate (FDE > 30px) : {(fde_per_sample > 30).mean()*100:.1f}%")

# ==========================================
# 🎨 [Graphs]
# ==========================================
print("\nSaving graphs...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"A-PAS {MODEL.upper()} - Comprehensive Evaluation ({TAG})",
             fontsize=14, fontweight='bold')
w = 0.35

# Graph 1: ADE/FDE by Class
ax = axes[0][0]
names = list(class_results.keys())
ades  = [class_results[n]["ade"] for n in names]
fdes  = [class_results[n]["fde"] for n in names]
x     = np.arange(len(names))
bars1 = ax.bar(x - w/2, ades, w, label='ADE', color='#4C72B0')
bars2 = ax.bar(x + w/2, fdes, w, label='FDE', color='#DD8452')
ax.axhline(10, color='green', linestyle='--', alpha=0.6, label='Target (10px)')
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_title("ADE / FDE by Class"); ax.set_ylabel("Pixel Error")
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{bar.get_height():.1f}', ha='center', fontsize=8)

# Graph 2: ADE/FDE by Speed
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

# Graph 3: Error per Step (Step-wise overlay)
ax = axes[0][2]
t_steps = [(t+1)*DT for t in range(PRED_LENGTH)]
ax.plot(t_steps, step_errors, 'o-', color='#8172B2', linewidth=2, markersize=4)
ax.fill_between(t_steps, step_errors, alpha=0.2, color='#8172B2')
ax.set_title("Error per Prediction Step")
ax.set_xlabel("Prediction Time (s)"); ax.set_ylabel("Mean Error (px)")
ax.grid(True, alpha=0.3)

# Graph 4: ADE Distribution (Histogram)
ax = axes[1][0]
ax.hist(ade_per_sample, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
ax.axvline(ade_per_sample.mean(),     color='red',    linestyle='--',
           label=f'Mean {ade_per_sample.mean():.2f}px')
ax.axvline(np.median(ade_per_sample), color='orange', linestyle='--',
           label=f'Median {np.median(ade_per_sample):.2f}px')
ax.set_title("ADE Error Distribution")
ax.set_xlabel("ADE (px)"); ax.set_ylabel("Sample Count")
ax.legend(); ax.grid(axis='y', alpha=0.3)

# Graph 5: Straight vs Turning
ax = axes[1][1]
categories = ["Straight", "Turning"]
ades_m = [ade_s, ade_t]
fdes_m = [fde_s, fde_t]
x = np.arange(2)
ax.bar(x - w/2, ades_m, w, label='ADE', color='#55A868')
ax.bar(x + w/2, fdes_m, w, label='FDE', color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_title("Straight vs Turning"); ax.set_ylabel("Pixel Error")
ax.legend(); ax.grid(axis='y', alpha=0.3)
for i, (a, f) in enumerate(zip(ades_m, fdes_m)):
    ax.text(i-w/2, a+0.1, f'{a:.1f}', ha='center', fontsize=9)
    ax.text(i+w/2, f+0.1, f'{f:.1f}', ha='center', fontsize=9)

# Graph 6: CDF (ADE + FDE 동시)
ax = axes[1][2]
thrs = np.arange(0, 30, 0.5)
ade_cdf = [(ade_per_sample < t).mean() * 100 for t in thrs]
fde_cdf = [(fde_per_sample < t).mean() * 100 for t in thrs]
ax.plot(thrs, ade_cdf, color='#4C72B0', linewidth=2, label='ADE CDF')
ax.plot(thrs, fde_cdf, color='#DD8452', linewidth=2, label='FDE CDF')
for thr in [3, 5, 10]:
    r_ade = (ade_per_sample < thr).mean() * 100
    r_fde = (fde_per_sample < thr).mean() * 100
    ax.axvline(thr, color='gray', linestyle='--', alpha=0.4)
    ax.text(thr+0.3, min(r_ade, r_fde)-8,
            f'{thr}px\nA:{r_ade:.0f}%\nF:{r_fde:.0f}%', fontsize=7, color='gray')
ax.set_title("ADE & FDE Cumulative Distribution (CDF)")
ax.set_xlabel("Error Threshold (px)"); ax.set_ylabel("Ratio (%)")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"comprehensive_eval_{MODEL}_{TAG}.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}")
print("\n🎉 Evaluation complete!")