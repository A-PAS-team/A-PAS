"""
A-PAS CSV → NPY 변환 (v3 Kinematic)
=====================================
기존 csv_to_npy_10fps.py에서 변경:
  - 입력 피처: 17개 (12물리 + yaw_rate + 4원-핫, crosswalk 제외)
  - 원-핫: index 13~16 (person, car, motorcycle, large_veh)
  - 라벨: (x, y) 좌표 유지 (Kinematic Layer가 학습 중 역전파 처리)
  - SEQ=10, PRED=20 (1초 관측, 2초 예측)

사용법:
  python csv_to_npy_10fps_v3_kinematic.py

결과:
  ai_model/data/Training/X_train_10fps_{tag}_v3kin.npy
  ai_model/data/Training/y_train_10fps_{tag}_v3kin.npy
  ...
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "data", "Training")

IMG_W, IMG_H = 1920, 1080
DIAG         = np.sqrt(IMG_W**2 + IMG_H**2)

SEQ_LENGTH  = 10    # 1.0초 @ 10FPS
PRED_LENGTH = 20    # 2.0초 @ 10FPS
DT          = 0.1   # 10FPS
SAMPLE_RATE = 3     # 30FPS → 10FPS

N_FEATURES  = 17    # 12물리 + yaw_rate + 4원-핫 (crosswalk 제외)

# crosswalk(class 4) 제외: 궤적 예측 대상이 아님
CLASS_MAP = {0: 0, 2: 1, 5: 2, 7: 3}

# yaw_rate 정규화 기준 (rad/s)
# π rad/s = 180°/s, 물리적 상한 (급커브 차량 ~0.5, 보행자 급회전 ~2.0)
YAW_RATE_MAX = np.pi  # ≈ 3.14 rad/s


# ==========================================
# 🔍 [피처 추출] — 18개
# ==========================================
def get_features(group):
    df = group.sort_values('frame').reset_index(drop=True)

    # 10FPS 샘플링
    df = df.iloc[::SAMPLE_RATE].reset_index(drop=True)

    # 원본 프레임 번호 기준 actual_dt
    df['frame_diff'] = df['frame'].diff().fillna(3)
    actual_dt = (df['frame_diff'] / 3) * DT

    # 속도
    df['vx'] = df['x_center'].diff() / actual_dt
    df['vy'] = df['y_center'].diff() / actual_dt

    # 가속도
    df['ax'] = df['vx'].diff() / actual_dt
    df['ay'] = df['vy'].diff() / actual_dt

    # 트래킹 끊김 처리 — gap 행 + 다음 행도 제로화
    mask      = df['frame_diff'] > 6
    next_mask = mask.shift(-1).fillna(False)
    combined  = mask | next_mask
    df.loc[combined, ['vx', 'vy', 'ax', 'ay']] = 0

    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # 물리량
    df['speed']   = np.sqrt(df['vx']**2 + df['vy']**2)
    df['area']    = df['width'] * df['height']

    # --- heading & yaw_rate: smoothed velocity 기반 ---
    # raw vx/vy로 heading을 구하면 프레임 간 노이즈로 yaw_rate가 폭발.
    # → velocity를 이동평균 스무딩한 후 heading 계산.
    HEADING_SMOOTH = 3  # 3프레임 이동평균 (0.3초 @ 10FPS)

    vx_vals = df['vx'].values.astype(np.float64)
    vy_vals = df['vy'].values.astype(np.float64)

    # 이동평균 (causal: 현재 + 과거만 사용)
    vx_smooth = np.convolve(vx_vals, np.ones(HEADING_SMOOTH)/HEADING_SMOOTH, mode='same')
    vy_smooth = np.convolve(vy_vals, np.ones(HEADING_SMOOTH)/HEADING_SMOOTH, mode='same')

    # 경계 보정: 처음 HEADING_SMOOTH 프레임은 평균 개수가 적으므로 원본 사용
    vx_smooth[:HEADING_SMOOTH-1] = vx_vals[:HEADING_SMOOTH-1]
    vy_smooth[:HEADING_SMOOTH-1] = vy_vals[:HEADING_SMOOTH-1]

    # smoothed heading (yaw_rate 계산용)
    heading_smooth = np.arctan2(vy_smooth, vx_smooth)

    # raw heading (sin_h, cos_h 피처용 — 기존과 호환 유지)
    df['heading'] = np.arctan2(df['vy'], df['vx'])
    df['sin_h']   = np.sin(df['heading'])
    df['cos_h']   = np.cos(df['heading'])

    # --- yaw_rate 계산 (smoothed heading 기반) ---
    heading_diff = np.zeros(len(df), dtype=np.float64)
    heading_diff[1:] = heading_smooth[1:] - heading_smooth[:-1]

    # wrap-around: [-π, π] 범위로 정규화
    heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi

    # actual_dt 값 추출
    dt_vals = actual_dt.values.copy()
    dt_vals[dt_vals < 1e-6] = 1.0  # division by zero 방지

    yaw_rate_raw = heading_diff / dt_vals

    # 저속 마스킹: speed < 30 px/s 이면 heading 노이즈 → yaw_rate = 0
    speed_vals = df['speed'].values
    low_speed_mask = speed_vals < 30.0
    yaw_rate_raw[low_speed_mask] = 0.0

    # 물리적 상한 클램프: |yaw_rate| > π rad/s 는 노이즈
    yaw_rate_raw = np.clip(yaw_rate_raw, -np.pi, np.pi)

    # gap 발생 시 yaw_rate도 0
    yaw_rate_raw[combined.values] = 0.0

    # 첫 행은 항상 0
    yaw_rate_raw[0] = 0.0

    # inf/nan 안전 처리
    yaw_rate_raw = np.nan_to_num(yaw_rate_raw, nan=0.0, posinf=0.0, neginf=0.0)

    df['yaw_rate'] = yaw_rate_raw

    # ==========================================
    # 피처 배열 (17개: 12물리 + yaw_rate + 4원-핫)
    # ==========================================
    features = np.zeros((len(df), N_FEATURES), dtype=np.float32)

    features[:, 0]  = df['x_center'] / IMG_W           # pos_x
    features[:, 1]  = df['y_center'] / IMG_H           # pos_y
    features[:, 2]  = df['vx'] / IMG_W                 # vel_x
    features[:, 3]  = df['vy'] / IMG_H                 # vel_y
    features[:, 4]  = df['ax'] / IMG_W                 # acc_x
    features[:, 5]  = df['ay'] / IMG_H                 # acc_y
    features[:, 6]  = df['speed'] / DIAG               # speed
    features[:, 7]  = df['sin_h']                       # sin(heading)
    features[:, 8]  = df['cos_h']                       # cos(heading)
    features[:, 9]  = df['width'] / IMG_W              # width
    features[:, 10] = df['height'] / IMG_H             # height
    features[:, 11] = df['area'] / (IMG_W * IMG_H)     # area
    features[:, 12] = np.clip(                          # yaw_rate (NEW)
        df['yaw_rate'] / YAW_RATE_MAX, -1.0, 1.0
    )

    # 원-핫 클래스 (index 13~16, crosswalk 제외)
    cls_idx = CLASS_MAP.get(int(df['class_id'].iloc[0]), None)
    if cls_idx is not None:
        features[:, 13 + cls_idx] = 1.0

    return features


def process_csv_folder(csv_folder, tag):
    """CSV 폴더 하나를 처리하여 X, y 반환"""
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        print(f"  ❌ CSV 파일 없음: {csv_folder}")
        return None, None

    print(f"\n{'='*55}")
    print(f"📂 [{tag}] {len(csv_files)}개 CSV 처리 중...")
    print(f"   SEQ={SEQ_LENGTH} (1초) → PRED={PRED_LENGTH} (2초) @ 10FPS")
    print(f"   피처: {N_FEATURES}개 (yaw_rate 포함)")
    print(f"{'='*55}")

    X, y = [], []

    for csv_path in tqdm(csv_files, desc=f"  {tag}"):
        df = pd.read_csv(csv_path)
        filename = os.path.splitext(os.path.basename(csv_path))[0]

        # Global Track ID
        df['global_track_id'] = filename + "_" + df['track_id'].astype(str)

        for gid, group in df.groupby('global_track_id'):
            if len(group) < (SEQ_LENGTH + PRED_LENGTH) * SAMPLE_RATE:
                continue

            full_features = get_features(group)
            if len(full_features) < SEQ_LENGTH + PRED_LENGTH:
                continue

            for i in range(len(full_features) - SEQ_LENGTH - PRED_LENGTH + 1):
                X.append(full_features[i: i + SEQ_LENGTH])
                # 라벨: (x, y) 좌표 유지 — Kinematic Layer 역전파로 처리
                y.append(full_features[i + SEQ_LENGTH: i + SEQ_LENGTH + PRED_LENGTH, :2])

    if not X:
        print(f"  ❌ [{tag}] 추출된 시퀀스 없음")
        return None, None

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # 클리핑
    X[:, :, 2] = np.clip(X[:, :, 2], -2.0, 2.0)     # vx
    X[:, :, 3] = np.clip(X[:, :, 3], -2.0, 2.0)     # vy
    X[:, :, 4] = np.clip(X[:, :, 4], -1.0, 1.0)     # ax
    X[:, :, 5] = np.clip(X[:, :, 5], -1.0, 1.0)     # ay
    X[:, :, 12] = np.clip(X[:, :, 12], -1.0, 1.0)   # yaw_rate
    y = np.clip(y, 0, 1)

    print(f"  ✅ [{tag}] 시퀀스: {len(X):,}개")
    print(f"     vx std: {X[:,:,2].std():.4f} | ax std: {X[:,:,4].std():.4f}")
    print(f"     yaw_rate std: {X[:,:,12].std():.4f}")
    print(f"     yaw_rate 비영 비율: {(np.abs(X[:,:,12]) > 0.01).mean():.2%}")
    print(f"     NaN: {np.isnan(X).sum()} | inf: {np.isinf(X).sum()}")

    return X, y


# ==========================================
# 🚀 [메인]
# ==========================================
if __name__ == "__main__":
    print("🚀 A-PAS CSV → NPY [v3 Kinematic: 18피처, 1초관측 2초예측]")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sources = {
        "cctv"  : os.path.join(BASE_DIR, "..", "data", "csv"),
        "carla" : os.path.join(BASE_DIR, "..", "data", "csv_carla"),
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

        suffix = f"10fps_{tag}_v3kin"
        np.save(os.path.join(OUTPUT_DIR, f"X_train_{suffix}.npy"), X_train)
        np.save(os.path.join(OUTPUT_DIR, f"y_train_{suffix}.npy"), y_train)
        np.save(os.path.join(OUTPUT_DIR, f"X_val_{suffix}.npy"),   X_val)
        np.save(os.path.join(OUTPUT_DIR, f"y_val_{suffix}.npy"),   y_val)

        print(f"  💾 저장: X_train_{suffix}.npy → {X_train.shape}")

    print("\n🎉 변환 완료!")
    print("📋 다음 단계: merge_npy 후 tr_residual_kinematic.py 실행")
    # 검증 출력
    for tag in ["cctv", "carla"]:
        path = os.path.join(OUTPUT_DIR, f"X_train_10fps_{tag}_v3kin.npy")
        if os.path.exists(path):
            X = np.load(path)
            yaw = np.abs(X[:, :, 12]).flatten()
            print(f"  [{tag}] yaw_rate: mean={yaw.mean():.4f}, 포화(>0.99)={( yaw>0.99).mean():.1%}")