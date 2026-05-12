import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# Config: 30FPS CSV -> 10FPS NPY
# Task: observe 1.0s -> predict 1.5s
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "Training")

IMG_W, IMG_H = 1920, 1080
DIAG = np.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH = 10      # 1.0s @ 10FPS
PRED_LENGTH = 15     # 1.5s @ 10FPS
DT = 0.1             # 10FPS timestep
SAMPLE_RATE = 3      # 30FPS source -> 10FPS samples
TASK_TAG = "obs1s_pred1p5s"

CLASS_MAP = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}


# ==========================================
# Feature extraction
# ==========================================
def get_features(group):
    """Convert one track group into 10FPS 17-feature trajectory."""
    df = group.sort_values("frame").reset_index(drop=True)

    # 30FPS -> 10FPS
    df = df.iloc[::SAMPLE_RATE].reset_index(drop=True)

    # Use actual frame gaps after sampling. Normal 10FPS gap is 3 source frames.
    df["frame_diff"] = df["frame"].diff().fillna(SAMPLE_RATE)
    actual_dt = (df["frame_diff"] / SAMPLE_RATE) * DT

    df["vx"] = df["x_center"].diff() / actual_dt
    df["vy"] = df["y_center"].diff() / actual_dt
    df["ax"] = df["vx"].diff() / actual_dt
    df["ay"] = df["vy"].diff() / actual_dt

    # Reset velocity/acceleration around large track gaps.
    mask = df["frame_diff"] > (SAMPLE_RATE * 2)
    next_mask = mask.shift(-1).fillna(False)
    combined = mask | next_mask
    df.loc[combined, "vx"] = 0
    df.loc[combined, "vy"] = 0
    df.loc[combined, "ax"] = 0
    df.loc[combined, "ay"] = 0

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    df["speed"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    df["heading"] = np.arctan2(df["vy"], df["vx"])
    df["sin_h"] = np.sin(df["heading"])
    df["cos_h"] = np.cos(df["heading"])
    df["area"] = df["width"] * df["height"]

    features = np.zeros((len(df), 17), dtype=np.float32)
    features[:, 0] = df["x_center"] / IMG_W
    features[:, 1] = df["y_center"] / IMG_H
    features[:, 2] = df["vx"] / IMG_W
    features[:, 3] = df["vy"] / IMG_H
    features[:, 4] = df["ax"] / IMG_W
    features[:, 5] = df["ay"] / IMG_H
    features[:, 6] = df["speed"] / DIAG
    features[:, 7] = df["sin_h"]
    features[:, 8] = df["cos_h"]
    features[:, 9] = df["width"] / IMG_W
    features[:, 10] = df["height"] / IMG_H
    features[:, 11] = df["area"] / (IMG_W * IMG_H)

    cls_idx = CLASS_MAP.get(int(df["class_id"].iloc[0]), 0)
    features[:, 12 + cls_idx] = 1.0

    return features


def process_csv_folder(csv_folder, tag):
    """Process one CSV folder and return X/y arrays."""
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        print(f"  CSV files not found: {csv_folder}")
        return None, None

    print(f"\n{'=' * 55}")
    print(f"[{tag}] processing {len(csv_files)} CSV files")
    print(f"{'=' * 55}")

    X, y = [], []

    for csv_path in tqdm(csv_files, desc=f"  {tag}"):
        df = pd.read_csv(csv_path)
        filename = os.path.splitext(os.path.basename(csv_path))[0]

        # Make track IDs unique across CSV files.
        df["global_track_id"] = filename + "_" + df["track_id"].astype(str)

        for _, group in df.groupby("global_track_id"):
            if len(group) < (SEQ_LENGTH + PRED_LENGTH) * SAMPLE_RATE:
                continue

            full_features = get_features(group)
            if len(full_features) < SEQ_LENGTH + PRED_LENGTH:
                continue

            for i in range(len(full_features) - SEQ_LENGTH - PRED_LENGTH + 1):
                X.append(full_features[i:i + SEQ_LENGTH])
                y.append(full_features[i + SEQ_LENGTH:i + SEQ_LENGTH + PRED_LENGTH, :2])

    if not X:
        print(f"  [{tag}] no valid sequence extracted")
        return None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    X[:, :, 2] = np.clip(X[:, :, 2], -2.0, 2.0)
    X[:, :, 3] = np.clip(X[:, :, 3], -2.0, 2.0)
    X[:, :, 4] = np.clip(X[:, :, 4], -1.0, 1.0)
    X[:, :, 5] = np.clip(X[:, :, 5], -1.0, 1.0)
    y = np.clip(y, 0, 1)

    print(f"  [{tag}] sequences: {len(X):,}")
    print(f"     X shape: {X.shape} | y shape: {y.shape}")
    print(f"     vx std: {X[:, :, 2].std():.4f} | ax std: {X[:, :, 4].std():.4f}")
    print(f"     NaN: {np.isnan(X).sum()} | inf: {np.isinf(X).sum()}")

    return X, y


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print("A-PAS CSV -> NPY start! [10FPS, obs=1.0s, pred=1.5s]")
    print(f"  SEQ_LENGTH={SEQ_LENGTH}, PRED_LENGTH={PRED_LENGTH}, SAMPLE_RATE={SAMPLE_RATE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sources = {
        "cctv": os.path.join(BASE_DIR, "..", "data", "csv"),
        "carla": os.path.join(BASE_DIR, "..", "data", "csv_carla"),
    }

    for tag, csv_folder in sources.items():
        if not os.path.exists(csv_folder):
            print(f"\n[skip] {tag} folder not found: {csv_folder}")
            continue

        X, y = process_csv_folder(csv_folder, tag)
        if X is None:
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=True
        )

        np.save(os.path.join(OUTPUT_DIR, f"X_train_10fps_{TASK_TAG}_{tag}.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, f"y_train_10fps_{TASK_TAG}_{tag}.npy"), y_train)
        np.save(os.path.join(OUTPUT_DIR, f"X_val_10fps_{TASK_TAG}_{tag}.npy"), X_val)
        np.save(os.path.join(OUTPUT_DIR, f"y_val_10fps_{TASK_TAG}_{tag}.npy"), y_val)

        print(f"  saved: X_train_10fps_{TASK_TAG}_{tag}.npy -> {X_train.shape}")
        print(f"  saved: y_train_10fps_{TASK_TAG}_{tag}.npy -> {y_train.shape}")
        print(f"  saved: X_val_10fps_{TASK_TAG}_{tag}.npy -> {X_val.shape}")
        print(f"  saved: y_val_10fps_{TASK_TAG}_{tag}.npy -> {y_val.shape}")

    print("\nDone.")
