import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정]
# ==========================================
CSV_FOLDER   = r"D:\cctv.csv(30fps)\csv"
IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH   = 60   # 2.0초 @ 30FPS
PRED_LENGTH  = 30   # 1.0초 @ 30FPS
DT           = 1/30 # ≈ 0.0333

CLASS_MAP = {0: 0, 2: 1, 5: 2, 7: 3, 3: 4}

# ==========================================
# 🔍 [피처 추출 함수]
# ==========================================
def get_features(group):
    df = group.sort_values('frame').reset_index(drop=True)

    # ✅ 컬럼명 수정
    df['vx'] = df['x_center'].diff() / DT
    df['vy'] = df['y_center'].diff() / DT
    df['ax'] = df['vx'].diff() / DT
    df['ay'] = df['vy'].diff() / DT
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

# ==========================================
# 🚀 [메인]
# ==========================================
print("📂 CSV 파일 로드 중...")
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

# ==========================================
# ⚙️ [시퀀스 생성]
# ==========================================
X, y = [], []
track_groups = df_raw.groupby('track_id')
print(f"\n⚙️ 시퀀스 생성 중... (총 {len(track_groups)}개 트랙)")

for track_id, group in tqdm(track_groups):
    if len(group) < SEQ_LENGTH + PRED_LENGTH:
        continue

    full_features = get_features(group)

    for i in range(len(full_features) - SEQ_LENGTH - PRED_LENGTH + 1):
        X.append(full_features[i: i + SEQ_LENGTH])
        y.append(full_features[i + SEQ_LENGTH: i + SEQ_LENGTH + PRED_LENGTH, :2])

if not X:
    print("❌ 추출된 시퀀스가 없습니다!")
    exit()

# ==========================================
# 💾 [저장]
# ==========================================
X = np.clip(np.array(X, dtype=np.float32), 0, 1)
y = np.clip(np.array(y, dtype=np.float32), 0, 1)

print(f"\n📊 전체 시퀀스: {len(X):,}개")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

os.makedirs("data/Training", exist_ok=True)
np.save("data/Training/X_train_30fps.npy", X_train)
np.save("data/Training/y_train_30fps.npy", y_train)
np.save("data/Training/X_val_30fps.npy",   X_val)
np.save("data/Training/y_val_30fps.npy",   y_val)

print(f"\n💾 저장 완료!")
print(f"  Train: {X_train.shape}")
print(f"  Val  : {X_val.shape}")
print(f"\n🎉 완료! data/Training/ 폴더 확인하세요.")