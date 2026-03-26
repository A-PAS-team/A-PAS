import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정] 30FPS CSV → 10FPS 샘플링
# ==========================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER   = os.path.join(BASE_DIR, "data", "csv")
OUTPUT_DIR   = os.path.join(BASE_DIR, "data", "Training")

IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH   = 20       # 2.0초 @ 10FPS
PRED_LENGTH  = 10       # 1.0초 @ 10FPS
DT           = 0.1      # 10FPS
SAMPLE_RATE  = 3        # 30FPS → 10FPS

CLASS_MAP = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}

# ==========================================
# 🔍 [피처 추출]
# ==========================================
def get_features(group):
    df = group.sort_values('frame').reset_index(drop=True)

    # 10FPS 샘플링
    df = df.iloc[::SAMPLE_RATE].reset_index(drop=True)

    # 원본 프레임 번호 기준 actual_dt
    df['frame_diff'] = df['frame'].diff().fillna(3)
    actual_dt = (df['frame_diff'] / 3) * DT

    # 속도/가속도
    df['vx'] = df['x_center'].diff() / actual_dt
    df['vy'] = df['y_center'].diff() / actual_dt
    df['ax'] = df['vx'].diff() / actual_dt
    df['ay'] = df['vy'].diff() / actual_dt

    # 트래킹 끊김 처리
    df.loc[df['frame_diff'] > 6, ['vx', 'vy', 'ax', 'ay']] = 0
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # 물리량 계산
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
# 🚀 [메인]
# ==========================================
print("📂 CSV 파일 로드 중... [10FPS 샘플링 버전]")
csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
if not csv_files:
    print("❌ CSV 파일이 없습니다!")
    exit()

print(f"  ✅ {len(csv_files)}개 파일 발견")
df_raw = pd.concat(
    [pd.read_csv(f) for f in tqdm(csv_files, desc="파일 읽기")],
    ignore_index=True
)
print(f"  ✅ 총 {len(df_raw):,}행 로드 완료")

X, y = [], []
track_groups = df_raw.groupby('track_id')
print(f"\n⚙️ 시퀀스 생성 중... (총 {len(track_groups)}개 트랙)")

for track_id, group in tqdm(track_groups, desc="시퀀스 추출"):
    if len(group) < (SEQ_LENGTH + PRED_LENGTH) * SAMPLE_RATE:
        continue

    full_features = get_features(group)
    if len(full_features) < SEQ_LENGTH + PRED_LENGTH:
        continue

    for i in range(len(full_features) - SEQ_LENGTH - PRED_LENGTH + 1):
        X.append(full_features[i: i + SEQ_LENGTH])
        y.append(full_features[i + SEQ_LENGTH: i + SEQ_LENGTH + PRED_LENGTH, :2])

if not X:
    print("❌ 추출된 시퀀스가 없습니다!")
    exit()

X = np.array(X, dtype=np.float32)
y = np.clip(np.array(y, dtype=np.float32), 0, 1)

print(f"\n📊 전체 시퀀스: {len(X):,}개")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(os.path.join(OUTPUT_DIR, "X_train_10fps.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train_10fps.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val_10fps.npy"),   X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val_10fps.npy"),   y_val)

print(f"\n💾 저장 완료!")
print(f"  Train: {X_train.shape}")
print(f"  Val  : {X_val.shape}")
print(f"  ax 표준편차: {X_train[:,:,4].std():.4f}")
print(f"  NaN: {np.isnan(X_train).sum()} | inf: {np.isinf(X_train).sum()}")
print(f"\n🎉 완료! {OUTPUT_DIR} 확인하세요.")