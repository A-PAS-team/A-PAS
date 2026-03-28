import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정] 30FPS 원본
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "data", "Training")

IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH  = 60    # 2.0초 @ 30FPS
PRED_LENGTH = 30    # 1.0초 @ 30FPS
DT          = 1/30

CLASS_MAP = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}

# ==========================================
# 🔍 [피처 추출]
# ==========================================
def get_features(group):
    df = group.sort_values('frame').reset_index(drop=True)

    df['frame_diff'] = df['frame'].diff().fillna(1)
    actual_dt = df['frame_diff'] * DT

    df['vx'] = df['x_center'].diff() / actual_dt
    df['vy'] = df['y_center'].diff() / actual_dt
    df['ax'] = df['vx'].diff() / actual_dt
    df['ay'] = df['vy'].diff() / actual_dt

    # 트래킹 끊김 처리 - gap 행 + 다음 행도 제로화
    mask      = df["frame_diff"] > 10
    next_mask = mask.shift(-1).fillna(False)
    combined  = mask | next_mask
    df.loc[combined, 'vx'] = 0
    df.loc[combined, 'vy'] = 0
    df.loc[combined, 'ax'] = 0
    df.loc[combined, 'ay'] = 0

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    df['speed']   = np.sqrt(df['vx']**2 + df['vy']**2)
    df['heading'] = np.arctan2(df['vy'], df['vx'])
    df['sin_h']   = np.sin(df['heading'])
    df['cos_h']   = np.cos(df['heading'])
    df['area']    = df['width'] * df['height']

    features = np.zeros((len(df), 17), dtype=np.float32)
    features[:, 0]  = df['x_center'] / IMG_W
    features[:, 1]  = df['y_center'] / IMG_H
    features[:, 2]  = df['vx'] / IMG_W
    features[:, 3]  = df['vy'] / IMG_H
    features[:, 4]  = df['ax'] / IMG_W
    features[:, 5]  = df['ay'] / IMG_H
    features[:, 6]  = df['speed'] / DIAG
    features[:, 7]  = df['sin_h']
    features[:, 8]  = df['cos_h']
    features[:, 9]  = df['width'] / IMG_W
    features[:, 10] = df['height'] / IMG_H
    features[:, 11] = df['area'] / (IMG_W * IMG_H)

    cls_idx = CLASS_MAP.get(int(df['class_id'].iloc[0]), 0)
    features[:, 12 + cls_idx] = 1.0

    return features


def process_csv_folder(csv_folder, tag):
    """CSV 폴더 하나를 처리하여 X, y 반환"""
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        print(f"  ❌ CSV 파일 없음: {csv_folder}")
        return None, None

    print(f"\n{'='*55}")
    print(f"📂 [{tag}] {len(csv_files)}개 CSV 처리 중...")
    print(f"{'='*55}")

    X, y = [], []

    for csv_path in tqdm(csv_files, desc=f"  {tag}"):
        df = pd.read_csv(csv_path)
        filename = os.path.splitext(os.path.basename(csv_path))[0]

        # ✅ Global Track ID: 파일명 + track_id 조합으로 고유 ID 생성
        df['global_track_id'] = filename + "_" + df['track_id'].astype(str)

        for gid, group in df.groupby('global_track_id'):
            if len(group) < SEQ_LENGTH + PRED_LENGTH:
                continue

            full_features = get_features(group)
            if len(full_features) < SEQ_LENGTH + PRED_LENGTH:
                continue

            for i in range(len(full_features) - SEQ_LENGTH - PRED_LENGTH + 1):
                X.append(full_features[i: i + SEQ_LENGTH])
                y.append(full_features[i + SEQ_LENGTH: i + SEQ_LENGTH + PRED_LENGTH, :2])

    if not X:
        print(f"  ❌ [{tag}] 추출된 시퀀스 없음")
        return None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ✅ 클리핑 적용
    X[:, :, 2] = np.clip(X[:, :, 2], -2.0, 2.0)   # vx
    X[:, :, 3] = np.clip(X[:, :, 3], -2.0, 2.0)   # vy
    X[:, :, 4] = np.clip(X[:, :, 4], -1.0, 1.0)   # ax
    X[:, :, 5] = np.clip(X[:, :, 5], -1.0, 1.0)   # ay
    y = np.clip(y, 0, 1)

    print(f"  ✅ [{tag}] 시퀀스: {len(X):,}개")
    print(f"     vx 표준편차: {X[:,:,2].std():.4f} | ax 표준편차: {X[:,:,4].std():.4f}")
    print(f"     NaN: {np.isnan(X).sum()} | inf: {np.isinf(X).sum()}")

    return X, y


# ==========================================
# 🚀 [메인]
# ==========================================
if __name__ == "__main__":
    print("🚀 A-PAS CSV → NPY 변환 시작! [30FPS + Global Track ID + Clipping]")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 📂 소스별 처리 - 경로만 여기서 관리
    # ==========================================
    sources = {
        "jamsil"  : os.path.join(BASE_DIR, "..", "data", "jamsil_DDP_csv"),
        "jaehoon" : os.path.join(BASE_DIR, "..", "data", "Jaehoon_csv"),
        "carla"   : os.path.join(BASE_DIR, "..", "data", "CARLA_csv"),
    }

    for tag, csv_folder in sources.items():
        if not os.path.exists(csv_folder):
            print(f"\n⏭️  {tag} 폴더 없음, 스킵: {csv_folder}")
            continue

        X, y = process_csv_folder(csv_folder, tag)
        if X is None:
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=True
        )

        np.save(os.path.join(OUTPUT_DIR, f"X_train_30fps_{tag}.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, f"y_train_30fps_{tag}.npy"), y_train)
        np.save(os.path.join(OUTPUT_DIR, f"X_val_30fps_{tag}.npy"),   X_val)
        np.save(os.path.join(OUTPUT_DIR, f"y_val_30fps_{tag}.npy"),   y_val)

        print(f"  💾 저장: X_train_30fps_{tag}.npy → {X_train.shape}")

    print("\n🎉 전체 변환 완료!")
    print("📋 다음 단계: python merge_npy.py")