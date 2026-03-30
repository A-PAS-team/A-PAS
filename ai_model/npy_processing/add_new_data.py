import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER  = os.path.join(BASE_DIR, "..", "data", "AddCARLA_csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "data", "Training")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TAG          = "carla_person"   # ← 저장 파일명에 사용될 태그
IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)
CLASS_MAP    = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}
CLASS_NAMES  = ["Person", "Car", "Bus", "Truck", "Motorcycle"]

# ==========================================
# 🔍 [피처 추출 - 10FPS]
# ==========================================
def get_features_10fps(group):
    SEQ_LENGTH  = 20
    PRED_LENGTH = 10
    DT          = 0.1
    SAMPLE_RATE = 3

    df = group.sort_values('frame').reset_index(drop=True)
    df = df.iloc[::SAMPLE_RATE].reset_index(drop=True)

    df['frame_diff'] = df['frame'].diff().fillna(3)
    actual_dt = (df['frame_diff'] / 3) * DT

    df['vx'] = df['x_center'].diff() / actual_dt
    df['vy'] = df['y_center'].diff() / actual_dt
    df['ax'] = df['vx'].diff() / actual_dt
    df['ay'] = df['vy'].diff() / actual_dt

    mask      = df['frame_diff'] > 6
    next_mask = mask.shift(-1).fillna(False)
    combined  = mask | next_mask
    df.loc[combined, 'vx'] = 0
    df.loc[combined, 'vy'] = 0
    df.loc[combined, 'ax'] = 0
    df.loc[combined, 'ay'] = 0

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

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

# ==========================================
# 🔍 [피처 추출 - 30FPS]
# ==========================================
def get_features_30fps(group):
    DT = 1/30

    df = group.sort_values('frame').reset_index(drop=True)

    df['frame_diff'] = df['frame'].diff().fillna(1)
    actual_dt = df['frame_diff'] * DT

    df['vx'] = df['x_center'].diff() / actual_dt
    df['vy'] = df['y_center'].diff() / actual_dt
    df['ax'] = df['vx'].diff() / actual_dt
    df['ay'] = df['vy'].diff() / actual_dt

    mask      = df['frame_diff'] > 10
    next_mask = mask.shift(-1).fillna(False)
    combined  = mask | next_mask
    df.loc[combined, 'vx'] = 0
    df.loc[combined, 'vy'] = 0
    df.loc[combined, 'ax'] = 0
    df.loc[combined, 'ay'] = 0

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

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

# ==========================================
# 🔄 [공통 처리 함수]
# ==========================================
def process(fps_mode):
    if fps_mode == 10:
        SEQ_LENGTH  = 20
        PRED_LENGTH = 10
        SAMPLE_RATE = 3
        get_feat    = get_features_10fps
        min_frames  = (SEQ_LENGTH + PRED_LENGTH) * SAMPLE_RATE
    else:
        SEQ_LENGTH  = 60
        PRED_LENGTH = 30
        SAMPLE_RATE = 1
        get_feat    = get_features_30fps
        min_frames  = SEQ_LENGTH + PRED_LENGTH

    print(f"\n{'='*55}")
    print(f"📂 [{fps_mode}FPS] 변환 시작 → tag: {TAG}")
    print(f"{'='*55}")

    csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not csv_files:
        print(f"  ❌ CSV 없음: {CSV_FOLDER}")
        return

    print(f"  ✅ {len(csv_files)}개 CSV 발견")

    X, y = [], []

    for csv_path in tqdm(csv_files, desc=f"  {fps_mode}FPS 변환"):
        df       = pd.read_csv(csv_path)
        filename = os.path.splitext(os.path.basename(csv_path))[0]

        # ✅ Global Track ID
        df['global_track_id'] = filename + "_" + df['track_id'].astype(str)

        for gid, group in df.groupby('global_track_id'):
            if len(group) < min_frames:
                continue

            full_features = get_feat(group)
            if len(full_features) < SEQ_LENGTH + PRED_LENGTH:
                continue

            for i in range(len(full_features) - SEQ_LENGTH - PRED_LENGTH + 1):
                X.append(full_features[i: i + SEQ_LENGTH])
                y.append(full_features[i + SEQ_LENGTH: i + SEQ_LENGTH + PRED_LENGTH, :2])

    if not X:
        print(f"  ❌ 추출된 시퀀스 없음")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # ✅ 클리핑
    X[:, :, 2] = np.clip(X[:, :, 2], -2.0, 2.0)  # vx
    X[:, :, 3] = np.clip(X[:, :, 3], -2.0, 2.0)  # vy
    X[:, :, 4] = np.clip(X[:, :, 4], -1.0, 1.0)  # ax
    X[:, :, 5] = np.clip(X[:, :, 5], -1.0, 1.0)  # ay
    y = np.clip(y, 0, 1)

    # 클래스 분포 출력
    class_idx = np.argmax(X[:, 0, 12:17], axis=1)
    print(f"\n  📊 클래스 분포:")
    for i, name in enumerate(CLASS_NAMES):
        count = (class_idx == i).sum()
        print(f"     {name:<12}: {count:>5}개 ({count/len(X)*100:.1f}%)")

    # 저장
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )

    np.save(os.path.join(OUTPUT_DIR, f"X_train_{fps_mode}fps_{TAG}.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, f"y_train_{fps_mode}fps_{TAG}.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, f"X_val_{fps_mode}fps_{TAG}.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, f"y_val_{fps_mode}fps_{TAG}.npy"),   y_val)

    print(f"\n  💾 저장 완료!")
    print(f"     Train : {X_train.shape}")
    print(f"     Val   : {X_val.shape}")
    print(f"     ax 표준편차: {X_train[:,:,4].std():.4f}")
    print(f"     NaN: {np.isnan(X_train).sum()} | inf: {np.isinf(X_train).sum()}")

# ==========================================
# 🚀 [메인]
# ==========================================
if __name__ == "__main__":
    print("🚀 A-PAS 신규 데이터 변환 시작!")
    print(f"   폴더: {os.path.abspath(CSV_FOLDER)}")

    process(10)
    process(30)

    print(f"\n🎉 완료! 다음 단계:")
    print(f"  1. merge_npy.py의 SOURCES에 아래 경로 추가:")
    print(f"     os.path.join(BASE_DIR, \"..\", \"data\", \"Training\", \"X_train_10fps_{TAG}.npy\")")
    print(f"     os.path.join(BASE_DIR, \"..\", \"data\", \"Training\", \"X_val_10fps_{TAG}.npy\")")
    print(f"  2. python merge_npy.py 재실행")
    print(f"  3. python tr_residual.py --fps 30 --version v2_cctv_carla 재학습")