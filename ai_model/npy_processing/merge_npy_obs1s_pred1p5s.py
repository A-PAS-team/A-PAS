import numpy as np
import os

# ==========================================
# Config
# ==========================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data", "Training")
TASK_TAG   = "obs1s_pred1p5s"
SEED       = 42

# ==========================================
# Load
# ==========================================
print("=" * 55)
print("A-PAS NPY Merge [CCTV + CARLA]")
print(f"  source: {DATA_DIR}")
print("=" * 55)

sources = ["cctv", "carla"]
train_X, train_y, val_X, val_y = [], [], [], []

for tag in sources:
    xtr = os.path.join(DATA_DIR, f"X_train_10fps_{TASK_TAG}_{tag}.npy")
    ytr = os.path.join(DATA_DIR, f"y_train_10fps_{TASK_TAG}_{tag}.npy")
    xvl = os.path.join(DATA_DIR, f"X_val_10fps_{TASK_TAG}_{tag}.npy")
    yvl = os.path.join(DATA_DIR, f"y_val_10fps_{TASK_TAG}_{tag}.npy")

    if not os.path.exists(xtr):
        print(f"  [skip] {tag} files not found")
        continue

    x1, y1 = np.load(xtr), np.load(ytr)
    x2, y2 = np.load(xvl), np.load(yvl)
    print(f"  [{tag}] train: {x1.shape[0]:,} | val: {x2.shape[0]:,}")

    train_X.append(x1); train_y.append(y1)
    val_X.append(x2);   val_y.append(y2)

# ==========================================
# Merge + Shuffle
# ==========================================
X_train = np.concatenate(train_X, axis=0)
y_train = np.concatenate(train_y, axis=0)
X_val   = np.concatenate(val_X, axis=0)
y_val   = np.concatenate(val_y, axis=0)

# Shuffle train
rng = np.random.default_rng(SEED)
train_idx = rng.permutation(len(X_train))
X_train = X_train[train_idx]
y_train = y_train[train_idx]

# Shuffle val
val_idx = rng.permutation(len(X_val))
X_val = X_val[val_idx]
y_val = y_val[val_idx]

print(f"\n  merged train: {X_train.shape}")
print(f"  merged val:   {X_val.shape}")
print(f"  NaN: {np.isnan(X_train).sum()} | inf: {np.isinf(X_train).sum()}")

# ==========================================
# Save
# ==========================================
np.save(os.path.join(DATA_DIR, f"X_train_10fps_{TASK_TAG}_merged.npy"), X_train)
np.save(os.path.join(DATA_DIR, f"y_train_10fps_{TASK_TAG}_merged.npy"), y_train)
np.save(os.path.join(DATA_DIR, f"X_val_10fps_{TASK_TAG}_merged.npy"), X_val)
np.save(os.path.join(DATA_DIR, f"y_val_10fps_{TASK_TAG}_merged.npy"), y_val)

print(f"\n  saved -> {DATA_DIR}")
print(f"    X_train_10fps_{TASK_TAG}_merged.npy  ({X_train.shape[0]:,})")
print(f"    y_train_10fps_{TASK_TAG}_merged.npy  ({y_train.shape[0]:,})")
print(f"    X_val_10fps_{TASK_TAG}_merged.npy    ({X_val.shape[0]:,})")
print(f"    y_val_10fps_{TASK_TAG}_merged.npy    ({y_val.shape[0]:,})")
print("\nDone!")
