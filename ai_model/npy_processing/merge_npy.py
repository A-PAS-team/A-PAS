import numpy as np
import os
from sklearn.model_selection import train_test_split

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "data", "Training")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "Training")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 📂 [버전별 소스 목록]
# v1: CCTV only (jamsil + jaehoon)
# v2: CCTV + CARLA
# 새 소스 추가 시 여기에만 추가
# ==========================================
VERSIONS = {
    "v1_cctv": {
        "10fps": ["jamsil", "jaehoon"],
        "30fps": ["jamsil", "jaehoon"],
    },
    "v2_cctv_carla": {
        "10fps": ["jamsil", "jaehoon", "carla"],
        "30fps": ["jamsil", "jaehoon", "carla"],
    },
}

# ==========================================
# 🔧 [유틸 함수]
# ==========================================
def merge_and_split(sources, fps, version):
    print(f"\n{'='*55}")
    print(f"📦 [{version}] {fps} 병합 시작")
    print(f"{'='*55}")

    X_list, y_list = [], []

    for src in sources:
        x_train = os.path.join(DATA_DIR, f"X_train_{fps}_{src}.npy")
        y_train = os.path.join(DATA_DIR, f"y_train_{fps}_{src}.npy")
        x_val   = os.path.join(DATA_DIR, f"X_val_{fps}_{src}.npy")
        y_val   = os.path.join(DATA_DIR, f"y_val_{fps}_{src}.npy")

        for xp, yp in [(x_train, y_train), (x_val, y_val)]:
            if os.path.exists(xp) and os.path.exists(yp):
                X_list.append(np.load(xp))
                y_list.append(np.load(yp))
                print(f"  ✅ {os.path.basename(xp)} → {X_list[-1].shape}")
            else:
                print(f"  ⏭️  스킵 (없음): {os.path.basename(xp)}")

    if not X_list:
        print(f"  ❌ 로드할 파일 없음, 스킵")
        return

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"\n  📊 병합 결과: {X_all.shape}")

    # 클래스별 샘플 수 출력
    class_names = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
    class_idx   = np.argmax(X_all[:, 0, 12:17], axis=1)
    print(f"\n  📊 클래스 분포:")
    for i, name in enumerate(class_names):
        count = (class_idx == i).sum()
        ratio = count / len(X_all) * 100
        print(f"     {name:<12}: {count:>6}개 ({ratio:.1f}%)")

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, shuffle=True
    )

    np.save(os.path.join(OUTPUT_DIR, f"X_train_final_{fps}_{version}.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, f"y_train_final_{fps}_{version}.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, f"X_val_final_{fps}_{version}.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, f"y_val_final_{fps}_{version}.npy"),   y_val)

    print(f"\n  💾 저장 완료!")
    print(f"     Train : {X_train.shape}")
    print(f"     Val   : {X_val.shape}")
    print(f"     NaN   : {np.isnan(X_train).sum()} | inf: {np.isinf(X_train).sum()}")

# ==========================================
# 🚀 [메인]
# ==========================================
if __name__ == "__main__":
    print("🚀 A-PAS NPY 병합 시작!")

    for version, fps_dict in VERSIONS.items():
        for fps, sources in fps_dict.items():
            merge_and_split(sources, fps, version)

    print(f"\n🎉 전체 병합 완료!")
    print("\n📋 생성된 파일:")
    print("  X_train_final_10fps_v1_cctv.npy      ← CCTV only")
    print("  X_train_final_10fps_v2_cctv_carla.npy ← CCTV + CARLA")
    print("  X_train_final_30fps_v1_cctv.npy")
    print("  X_train_final_30fps_v2_cctv_carla.npy")
    print("\n📋 다음 단계:")
    print("  10FPS → python ai_model/trajectory/tr_trajectory_10fps.py")
    print("  30FPS → python ai_model/trajectory/tr_trajectory_30fps.py")