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

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS Model Comparison")
parser.add_argument('--fps',     type=int, default=30, choices=[10, 30])
parser.add_argument('--version', type=str, default="v2_cctv_carla",
                    choices=["v1_cctv", "v2_cctv_carla"])
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
TAG     = f"{FPS}fps_{VERSION}"

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "Training")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Person", "Car", "Bus", "Truck", "Motorcycle"]

# ✅ (폴더명, 모델키, .pth 파일명) — 모델별로 파일명이 다름
MODELS_CONFIG = {
    "Self LSTM"     : ("Self_LSTM",         "self",               f"best_model_{TAG}.pth"),
    "Attention"     : ("Attention",          "attention",          f"best_attn_{TAG}.pth"),
    "Residual"      : ("Residual",           "residual",           f"best_model_{TAG}.pth"),
    "Attn+Residual" : ("Attention_Residual", "attention_residual", f"best_attn_res_{FPS}fps.pth"),
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {TAG}")

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

# ✅ 학습 코드(tr_attention_residual_30fps.py)와 동일 구조
class AttentionResidualLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.lstm       = nn.LSTM(HIDDEN_SIZE, HIDDEN_SIZE, 2,
                                  batch_first=True, dropout=0.2)
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
    "self"               : SelfLSTM,
    "attention"          : AttentionLSTM,
    "residual"           : ResidualLSTM,
    "attention_residual" : AttentionResidualLSTM,
}

# ==========================================
# 📂 [데이터 로드]
# ==========================================
print("\nLoading data...")
X_val = np.load(os.path.join(DATA_DIR, f"X_val_final_{FPS}fps_{VERSION}.npy"))
y_val = np.load(os.path.join(DATA_DIR, f"y_val_final_{FPS}fps_{VERSION}.npy"))
print(f"  Val: {X_val.shape}")

gt_px = y_val.copy()
gt_px[..., 0] *= IMG_W; gt_px[..., 1] *= IMG_H

class_indices = np.argmax(X_val[:, 0, 12:17], axis=1)
speed         = X_val[:, :, 6].mean(axis=1) * np.sqrt(IMG_W**2 + IMG_H**2)

# ==========================================
# 🔄 [모델별 추론]
# ==========================================
results = {}

for model_name, (model_dir, model_key, model_file) in MODELS_CONFIG.items():
    model_path = os.path.join(BASE_DIR, "..", "models", model_dir, model_file)

    if not os.path.exists(model_path):
        print(f"  ⏭️  {model_name}: 모델 없음, 스킵 ({model_path})")
        continue

    print(f"  Loading {model_name}...")
    model = MODEL_CLASS[model_key]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds_list = []
        for i in range(0, len(X_val), 512):
            batch = torch.FloatTensor(X_val[i:i+512]).to(device)
            out   = model(batch)
            if isinstance(out, tuple): out = out[0]
            preds_list.append(out.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)

    pred_px = preds.copy()
    pred_px[..., 0] *= IMG_W; pred_px[..., 1] *= IMG_H

    dist           = np.sqrt(np.sum((pred_px - gt_px)**2, axis=-1))
    ade_per_sample = dist.mean(axis=1)
    fde_per_sample = dist[:, -1]
    step_errors    = dist.mean(axis=0)

    results[model_name] = {
        "ade"        : ade_per_sample,
        "fde"        : fde_per_sample,
        "step_errors": step_errors,
        "params"     : sum(p.numel() for p in model.parameters()),
    }
    print(f"    ADE: {ade_per_sample.mean():.2f}px | FDE: {fde_per_sample.mean():.2f}px")

if not results:
    print("❌ 로드된 모델 없음!")
    exit()

# ==========================================
# 📊 [콘솔 요약표]
# ==========================================
print(f"\n{'='*75}")
print(f"  Model Comparison Summary [{TAG}]")
print(f"{'='*75}")
print(f"{'Model':<20} {'Params':>10} {'ADE(px)':>9} {'FDE(px)':>9} "
      f"{'Med ADE':>9} {'ADE<5px':>8} {'ADE<10px':>9}")
print("-"*75)
for name, r in results.items():
    ade_mean   = r['ade'].mean()
    fde_mean   = r['fde'].mean()
    ade_median = np.median(r['ade'])
    ade_5      = (r['ade'] < 5).mean() * 100
    ade_10     = (r['ade'] < 10).mean() * 100
    grade      = "✅" if ade_mean < 5 else ("🟡" if ade_mean < 10 else "🔴")
    print(f"{name:<20} {r['params']:>10,} {ade_mean:>9.2f} {fde_mean:>9.2f} "
          f"{ade_median:>9.2f} {ade_5:>7.1f}% {ade_10:>8.1f}% {grade}")
print(f"{'='*75}")

# ==========================================
# 🎨 [비교 그래프 - 3개]
# ==========================================
print("\nSaving comparison graphs...")
model_names = list(results.keys())
colors      = COLORS[:len(model_names)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f"A-PAS Model Comparison ({TAG})", fontsize=14, fontweight='bold')

# --- Graph 1: ADE / FDE by Model ---
ax  = axes[0]
x   = np.arange(len(model_names))
w   = 0.35
ades_mean = [results[n]['ade'].mean() for n in model_names]
fdes_mean = [results[n]['fde'].mean() for n in model_names]
bars1 = ax.bar(x - w/2, ades_mean, w, label='ADE', color=colors, alpha=0.85)
bars2 = ax.bar(x + w/2, fdes_mean, w, label='FDE',
               color=colors, alpha=0.5, hatch='//')
ax.axhline(10, color='green', linestyle='--', alpha=0.7, label='Target (10px)')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
ax.set_title("ADE / FDE by Model"); ax.set_ylabel("Pixel Error (px)")
ax.legend(); ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, ades_mean):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

# --- Graph 2: Error per Prediction Step ---
ax      = axes[1]
t_steps = [(t+1)*DT for t in range(PRED_LENGTH)]
for name, color in zip(model_names, colors):
    ax.plot(t_steps, results[name]['step_errors'], 'o-',
            label=name, color=color, linewidth=2, markersize=4)
ax.set_title("Error per Prediction Step")
ax.set_xlabel("Prediction Time (s)"); ax.set_ylabel("Mean Error (px)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- Graph 3: ADE CDF Overlay ---
ax   = axes[2]
thrs = np.arange(0, 30, 0.5)
for name, color in zip(model_names, colors):
    ratios = [(results[name]['ade'] < t).mean() * 100 for t in thrs]
    ax.plot(thrs, ratios, label=name, color=color, linewidth=2)
for thr in [3, 5, 10]:
    ax.axvline(thr, color='gray', linestyle='--', alpha=0.4)
    ax.text(thr+0.3, 5, f'{thr}px', fontsize=8, color='gray')
ax.set_title("ADE Cumulative Distribution (CDF)")
ax.set_xlabel("ADE Threshold (px)"); ax.set_ylabel("Ratio (%)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, f"model_comparison_{TAG}.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {save_path}")
print("\n🎉 Comparison complete!")