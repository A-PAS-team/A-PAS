"""
csv_to_npy_30fps_pred2s.py
==========================
SEQ=30 (1초 관찰) → PRED=60 (2초 예측) @ 30FPS

변경점 (기존 csv_to_npy_30fps.py 대비):
  - SEQ_LENGTH: 60 → 30
  - PRED_LENGTH: 30 → 60
  - 최소 트랙 길이: 90프레임(3초)
  - 파일명: *_pred2s.npy로 분리 (기존 1초 데이터 보존)
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정] 1초 관찰 → 2초 예측
# ==========================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER   = os.path.join(BASE_DIR, "..", "data", "csv_carla")
OUTPUT_DIR   = os.path.join(BASE_DIR, "..", "data", "Training")

IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH   = 30        # 1.0초 @ 30FPS  (기존 60)
PRED_LENGTH  = 60        # 2.0초 @ 30FPS  (기존 30)
TOTAL_REQ    = SEQ_LENGTH + PRED_LENGTH  # 90프레임 = 3초
DT           = 1/30

# 물리적 클리핑 (학습 파이프라인 표준)
VEL_CLIP     = 2.0
ACC_CLIP     = 1.0

CLASS_MAP = {0: 0, 1: 1, 2: 2, 3: 3}

# ==========================================
# 🔍 [피처 추출]
# ==========================================
def get_features(group, source_file=""):
    df = group.sort_values('frame').reset_index(drop=True)

    # Global Track ID (파일명 + track_id)
    global_tid = f"{source_file}_{df['track_id'].iloc[0]}"

    # 원본 프레임 번호 기준 actual_dt
    df['frame_diff'] = df['frame'].diff().fillna(1)
    actual_dt = df['frame_diff'] * DT

    # 속도/가속도
    df['vx'] = df['x_center'].diff() / actual_dt
    df['vy'] = df['y_center'].diff() / actual_dt
    df['ax'] = df['vx'].diff() / actual_dt
    df['ay'] = df['vy'].diff() / actual_dt

    # 트래킹 끊김 처리 (갭 프레임 + 다음 프레임 zeroing)
    gap_mask = df['frame_diff'] > 2
    gap_indices = df.index[gap_mask]
    for idx in gap_indices:
        df.loc[idx, ['vx', 'vy', 'ax', 'ay']] = 0
        if idx + 1 < len(df):
            df.loc[idx + 1, ['ax', 'ay']] = 0

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
    features[:, 2]  = np.clip(df['vx'] / IMG_W, -VEL_CLIP, VEL_CLIP)
    features[:, 3]  = np.clip(df['vy'] / IMG_H, -VEL_CLIP, VEL_CLIP)
    features[:, 4]  = np.clip(df['ax'] / IMG_W, -ACC_CLIP, ACC_CLIP)
    features[:, 5]  = np.clip(df['ay'] / IMG_H, -ACC_CLIP, ACC_CLIP)
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
print(f"📂 CSV 파일 로드 중... [30FPS, SEQ={SEQ_LENGTH}, PRED={PRED_LENGTH}]")
print(f"   1초 관찰 → 2초 예측 (최소 트랙: {TOTAL_REQ}프레임 = {TOTAL_REQ/30:.1f}초)")

csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
if not csv_files:
    print("❌ CSV 파일이 없습니다!")
    exit()

print(f"  ✅ {len(csv_files)}개 파일 발견")

X, y = [], []
total_tracks = 0
skipped_short = 0

for csv_file in tqdm(csv_files, desc="파일 처리"):
    df_raw = pd.read_csv(csv_file)
    source_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Global Track ID 생성
    df_raw['global_tid'] = source_name + "_" + df_raw['track_id'].astype(str)

    track_groups = df_raw.groupby('global_tid')

    for tid, group in track_groups:
        total_tracks += 1

        if len(group) < TOTAL_REQ:
            skipped_short += 1
            continue

        full_features = get_features(group, source_name)
        if len(full_features) < TOTAL_REQ:
            skipped_short += 1
            continue

        for i in range(len(full_features) - TOTAL_REQ + 1):
            X.append(full_features[i : i + SEQ_LENGTH])
            y.append(full_features[i + SEQ_LENGTH : i + TOTAL_REQ, :2])

if not X:
    print("❌ 추출된 시퀀스가 없습니다!")
    print(f"   전체 트랙: {total_tracks}개, 길이 부족 스킵: {skipped_short}개")
    print(f"   최소 {TOTAL_REQ}프레임({TOTAL_REQ/30:.1f}초) 이상의 트랙이 필요합니다.")
    exit()

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print(f"\n📊 데이터 통계:")
print(f"  전체 트랙      : {total_tracks:,}개")
print(f"  길이 부족 스킵  : {skipped_short:,}개")
print(f"  추출 시퀀스     : {len(X):,}개")
print(f"  X shape        : {X.shape}")
print(f"  y shape        : {y.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ✅ 파일명에 _pred2s 붙여서 기존 1초 데이터와 분리
np.save(os.path.join(OUTPUT_DIR, "X_train_30fps_pred2s_04-c.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "y_train_30fps_pred2s_04-c.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "X_val_30fps_pred2s_04-c.npy"),   X_val)
np.save(os.path.join(OUTPUT_DIR, "y_val_30fps_pred2s_04-c.npy"),   y_val)

print(f"\n💾 저장 완료!")
print(f"  Train : {X_train.shape} → X: (N, {SEQ_LENGTH}, 17), y: (N, {PRED_LENGTH}, 2)")
print(f"  Val   : {X_val.shape}")
print(f"  파일  : *_30fps_pred2s.npy")
print(f"\n🎉 완료! {OUTPUT_DIR} 확인하세요.")
