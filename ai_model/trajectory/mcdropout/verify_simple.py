# -*- coding: utf-8 -*-
# verify_simple.py
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path
import sys

sys.path.insert(0, r"C:\Users\dkdld\A-PAS\models\mcdropout")
from residual_lstm_mcd import build_model

DATA_DIR = Path(r"C:\Users\dkdld\A-PAS\ai_model\data\Training")
MODEL_DIR = Path(r"C:\Users\dkdld\A-PAS\models\mcdropout")

# 새 파일명으로 직접 지정
X_val = np.load(DATA_DIR / "X_val_30fps_carla.npy")[:200]
y_val = np.load(DATA_DIR / "y_val_30fps_carla.npy")[:200]
print(f"Loaded: X {X_val.shape}, y {y_val.shape}\n")

# 1. PyTorch eval (deterministic)
ckpt = torch.load(MODEL_DIR / "checkpoints/best_residual_mcd.pt",
                  map_location="cpu", weights_only=False)
model = build_model(ckpt.get("model_config", {}))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

with torch.no_grad():
    pred_torch = model(torch.FloatTensor(X_val)).numpy()

diff = pred_torch - y_val
diff[..., 0] *= 1920; diff[..., 1] *= 1080
ade_torch = np.linalg.norm(diff, axis=-1).mean()
fde_torch = np.linalg.norm(diff[:, -1, :], axis=-1).mean()
print(f"[1] PyTorch eval mode (deterministic)")
print(f"    ADE = {ade_torch:.3f}px")
print(f"    FDE = {fde_torch:.3f}px")
print(f"    학습 로그 ADE 1.07px와 비교\n")

# 2. ONNX MC sampling
session = ort.InferenceSession(str(MODEL_DIR / "residual_mcd.onnx"))
input_name = session.get_inputs()[0].name

N = 30
preds = np.stack([
    session.run(None, {input_name: X_val.astype(np.float32)})[0]
    for _ in range(N)
], axis=0)
pred_mean = preds.mean(axis=0)
pred_std = preds.std(axis=0)

diff = pred_mean - y_val
diff[..., 0] *= 1920; diff[..., 1] *= 1080
ade_onnx = np.linalg.norm(diff, axis=-1).mean()
fde_onnx = np.linalg.norm(diff[:, -1, :], axis=-1).mean()
print(f"[2] ONNX MC sampling (mean over {N} runs)")
print(f"    ADE = {ade_onnx:.3f}px")
print(f"    FDE = {fde_onnx:.3f}px\n")

# 3. Uncertainty over time
std_t = np.linalg.norm(pred_std, axis=-1).mean(axis=0)
print(f"[3] Uncertainty over time (std)")
print(f"    t=0  (33ms 후):   {std_t[0]:.5f}  ({std_t[0]*2202:.1f}px)")
print(f"    t=10 (333ms 후):  {std_t[10]:.5f}  ({std_t[10]*2202:.1f}px)")
print(f"    t=20 (667ms 후):  {std_t[20]:.5f}  ({std_t[20]*2202:.1f}px)")
print(f"    t=29 (1초 후):    {std_t[29]:.5f}  ({std_t[29]*2202:.1f}px)")
ratio = std_t[29] / max(std_t[0], 1e-9)
print(f"    Ratio std[29]/std[0]: {ratio:.2f}x\n")

# 4. Consistency
diff_models = np.linalg.norm(pred_torch - pred_mean, axis=-1).mean()
print(f"[4] PyTorch vs ONNX consistency")
print(f"    Mean diff: {diff_models:.5f} ({diff_models*2202:.2f}px)\n")

# Summary
print("=" * 60)
print("Phase 1 Verification Summary")
print("=" * 60)
print(f"  Training time ADE (logged):  1.07px")
print(f"  PyTorch eval ADE:             {ade_torch:.2f}px")
print(f"  ONNX MC mean ADE:             {ade_onnx:.2f}px")
print(f"  PyTorch vs ONNX diff:         {diff_models*2202:.2f}px")
print(f"  Uncertainty growth (1초):     {ratio:.2f}x")

# Pass/fail
checks = []
checks.append(("PyTorch ADE matches training", abs(ade_torch - 1.07) < 1.0))
checks.append(("PyTorch vs ONNX consistent", abs(ade_torch - ade_onnx) < 1.0))
checks.append(("Uncertainty grows over time", ratio > 1.5))

print("\nChecks:")
all_pass = True
for name, passed in checks:
    mark = "✅" if passed else "❌"
    print(f"  {mark} {name}")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("🎉 Phase 1 COMPLETE - Ready for Phase 2!")
else:
    print("⚠️  Some checks failed - review above")
print("=" * 60)