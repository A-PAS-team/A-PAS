import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "Training")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 [합칠 NPY 파일 목록]
# 새로운 데이터 소스 추가 시 여기에만 추가하면 됩니다.
# ==========================================
SOURCES_10FPS = [
    # CCTV 데이터 (잠실)
    os.path.join(BASE_DIR, "..", "data", "Training", "X_train_10fps_jamsil.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "X_val_10fps_jamsil.npy"),

    # CARLA 데이터
    os.path.join(BASE_DIR, "..", "data", "Training", "X_train_10fps_CARLA.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "X_val_10fps_CARLA.npy"),

    # 추가 CCTV 데이터 받으면 여기에 추가
    # os.path.join(BASE_DIR, "..", "data", "Training", "X_train_10fps_xxx.npy"),
    # os.path.join(BASE_DIR, "..", "data", "Training", "X_val_10fps_xxx.npy"),
]

SOURCES_10FPS_Y = [
    # CCTV 데이터 (잠실)
    os.path.join(BASE_DIR, "..", "data", "Training", "y_train_10fps_jamsil.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "y_val_10fps_jamsil.npy"),

    # CARLA 데이터
    os.path.join(BASE_DIR, "..", "data", "Training", "y_train_10fps_CARLA.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "y_val_10fps_CARLA.npy"),

    # os.path.join(BASE_DIR, "..", "data", "Training", "y_train_10fps_xxx.npy"),
    # os.path.join(BASE_DIR, "..", "data", "Training", "y_val_10fps_xxx.npy"),
]

SOURCES_30FPS = [
    # CCTV 데이터 (잠실)
    os.path.join(BASE_DIR, "..", "data", "Training", "X_train_30fps_jamsil.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "X_val_30fps_jamsil.npy"),

    # CARLA 데이터
    os.path.join(BASE_DIR, "..", "data", "Training", "X_train_30fps_CARLA.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "X_val_30fps_CARLA.npy"),

    # os.path.join(BASE_DIR, "..", "data", "Training", "X_train_30fps_xxx.npy"),
    # os.path.join(BASE_DIR, "..", "data", "Training", "X_val_30fps_xxx.npy"),
]

SOURCES_30FPS_Y = [
    # CCTV 데이터 (잠실)
    os.path.join(BASE_DIR, "..", "data", "Training", "y_train_30fps_jamsil.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "y_val_30fps_jamsil.npy"),

    # CARLA 데이터
    os.path.join(BASE_DIR, "..", "data", "Training", "y_train_30fps_CARLA.npy"),
    os.path.join(BASE_DIR, "..", "data", "Training", "y_val_30fps_CARLA.npy"),

    # os.path.join(BASE_DIR, "..", "data", "Training", "y_train_30fps_xxx.npy"),
    # os.path.join(BASE_DIR, "..", "data", "Training", "y_val_30fps_xxx.npy"),
]

# ==========================================
# 🔧 [유틸 함수]
# ==========================================
def merge_and_split(x_paths, y_paths, tag):
    """여러 NPY 파일을 합치고 9:1로 분할하여 저장"""
    print(f"\n{'='*50}")
    print(f"📦 [{tag}] 병합 시작")
    print(f"{'='*50}")

    X_list, y_list = [], []
    for xp, yp in zip(x_paths, y_paths):
        if os.path.exists(xp) and os.path.exists(yp):
            X_list.append(np.load(xp))
            y_list.append(np.load(yp))
            print(f"  ✅ 로드: {os.path.basename(xp)} → {X_list[-1].shape}")
        else:
            print(f"  ⏭️  스킵 (없음): {os.path.basename(xp)}")

    if not X_list:
        print(f"  ❌ [{tag}] 로드할 파일 없음, 스킵합니다.")
        return

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"\n  📊 병합 결과: {X_all.shape} / {y_all.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, shuffle=True
    )

    np.save(os.path.join(OUTPUT_DIR, f"X_train_final_{tag}_v1.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, f"y_train_final_{tag}_v1.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, f"X_val_final_{tag}_v1.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, f"y_val_final_{tag}_v1.npy"),   y_val)

    print(f"\n  💾 저장 완료!")
    print(f"     Train : {X_train.shape}")
    print(f"     Val   : {X_val.shape}")
    print(f"     NaN   : {np.isnan(X_train).sum()} | inf: {np.isinf(X_train).sum()}")

# ==========================================
# 🚀 [메인]
# ==========================================
if __name__ == "__main__":
    print("🚀 A-PAS NPY 병합 시작!")

    merge_and_split(SOURCES_10FPS, SOURCES_10FPS_Y, "10fps")
    merge_and_split(SOURCES_30FPS, SOURCES_30FPS_Y, "30fps")

    print(f"\n🎉 전체 병합 완료! {OUTPUT_DIR} 확인하세요.")
    print("\n📋 다음 단계:")
    print("  10FPS → python ai_model/trajectory/tr_trajectory_10fps.py")
    print("  30FPS → python ai_model/trajectory/tr_trajectory_30fps.py")