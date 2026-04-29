# -*- coding: utf-8 -*-
# check_data_detail.py
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\dkdld\A-PAS\ai_model\data\Training")

X = np.load(DATA_DIR / "X_train_final_30fps_pred2s_04_04-c.npy")
y = np.load(DATA_DIR / "y_train_final_30fps_pred2s_04_04-c.npy")

# Compute distance for ALL samples (use vectorized op)
last_obs = X[:, -1, :2]   # (N, 2)
first_pred = y[:, 0, :]   # (N, 2)
distances = np.linalg.norm(last_obs - first_pred, axis=-1)

print(f"Total samples: {len(distances):,}")
print(f"\nDistance distribution (normalized coords):")
print(f"  mean:  {distances.mean():.4f}  ({distances.mean()*2202:.1f}px)")
print(f"  std:   {distances.std():.4f}")
print(f"  median:{np.median(distances):.4f}  ({np.median(distances)*2202:.1f}px)")
print(f"  p50:   {np.percentile(distances, 50):.4f}")
print(f"  p90:   {np.percentile(distances, 90):.4f}  ({np.percentile(distances, 90)*2202:.1f}px)")
print(f"  p99:   {np.percentile(distances, 99):.4f}  ({np.percentile(distances, 99)*2202:.1f}px)")
print(f"  max:   {distances.max():.4f}")

# How many samples are "bad" (distance > 0.05 = ~110px)
threshold = 0.05
bad_ratio = (distances > threshold).mean()
print(f"\nSamples with distance > 0.05 ({threshold*2202:.0f}px): {bad_ratio*100:.1f}%")

threshold = 0.1
bad_ratio = (distances > threshold).mean()
print(f"Samples with distance > 0.10 ({threshold*2202:.0f}px): {bad_ratio*100:.1f}%")

# Look at the worst samples
print(f"\n--- 5 worst samples (largest distance) ---")
worst_idx = np.argsort(distances)[-5:][::-1]
for idx in worst_idx:
    print(f"Sample {idx}:")
    print(f"  last obs:    {X[idx, -1, :2]}")
    print(f"  first pred:  {y[idx, 0]}")
    print(f"  last pred:   {y[idx, -1]}")
    print(f"  distance:    {distances[idx]:.4f} ({distances[idx]*2202:.0f}px)")
    # Check velocity at last obs
    print(f"  vel at last: {X[idx, -1, 2:4]}")
    # Class one-hot
    cls = X[idx, -1, 12:17].argmax()
    print(f"  class:       {cls}")
    print()

# Check if the y trajectory is internally consistent
print("\n--- y trajectory internal consistency ---")
y_diffs = np.linalg.norm(y[:, 1:] - y[:, :-1], axis=-1)  # frame-to-frame distance
print(f"y frame-to-frame distance:")
print(f"  mean: {y_diffs.mean():.5f}")
print(f"  max:  {y_diffs.max():.5f}")
print(f"  (should be small ~0.001-0.01 for normal trajectory at 30fps)")