"""
merge_npy_pred2s.py
===================
SEQ=30, PRED=60 (pred2s) 기준 NPY 병합 스크립트

소스별로 각각 csv_to_npy_30fps_pred2s.py를 돌린 후
이 스크립트로 합칩니다.

사용법:
  1. 각 소스 폴더의 CSV_FOLDER를 바꿔가며 csv_to_npy_30fps_pred2s.py 실행
     → 저장 파일명을 소스별로 다르게 지정 (아래 SOURCES 참고)
  2. python merge_npy_pred2s.py
"""

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
# 📂 [소스 목록]
# csv_to_npy_30fps_pred2s.py 실행 시
# 저장 파일명을 소스별로 구분해서 저장했다면 여기에 추가
#
# 파일명 규칙: X_train_30fps_pred2s_{tag}.npy
# ==========================================
SOURCES = {
    "04"   : True,   # X_train_30fps_pred2s_04.npy   (CCTV)
    "04-c" : True,   # X_train_30fps_pred2s_04-c.npy (CARLA)
}

# ==========================================
# 🔧 [클래스 분포 출력]
# ==========================================
def print_class_dist(X, label=""):
    class_names = ["Person", "Car", "Motorcycle", "Large_Veh", "Unknown"]
    print(f"\n  📊 클래스 분포 {label}:")
    for i, name in enumerate(class_names):
        count = (X[:, 0, 12 + i] == 1.0).sum()
        ratio = count / len(X) * 100
        print(f"     {name:<12}: {count:>8,}개 ({ratio:.1f}%)")
    print(f"     {'총합':<12}: {len(X):>8,}개")

# ==========================================
# 🚀 [메인]
# ==========================================
if __name__ == "__main__":
    print("🚀 A-PAS NPY 병합 시작! [pred2s: SEQ=30, PRED=60]")

    X_list, y_list = [], []

    for tag, enabled in SOURCES.items():
        if not enabled:
            print(f"\n  ⏭️  [{tag}] 비활성화, 스킵")
            continue

        x_path = os.path.join(DATA_DIR, f"X_train_30fps_pred2s_{tag}.npy")
        y_path = os.path.join(DATA_DIR, f"y_train_30fps_pred2s_{tag}.npy")
        x_val_path = os.path.join(DATA_DIR, f"X_val_30fps_pred2s_{tag}.npy")
        y_val_path = os.path.join(DATA_DIR, f"y_val_30fps_pred2s_{tag}.npy")

        loaded = 0
        for xp, yp in [(x_path, y_path), (x_val_path, y_val_path)]:
            if os.path.exists(xp) and os.path.exists(yp):
                X = np.load(xp)
                y = np.load(yp)
                X_list.append(X)
                y_list.append(y)
                print(f"  ✅ [{tag}] {os.path.basename(xp)}: {X.shape}")
                loaded += 1
            else:
                print(f"  ⚠️  [{tag}] 없음: {os.path.basename(xp)}")

        if loaded == 0:
            print(f"  ❌ [{tag}] 파일을 찾을 수 없습니다. 경로 확인: {x_path}")

    if not X_list:
        print("\n❌ 병합할 파일이 없습니다!")
        print("   csv_to_npy_30fps_pred2s.py를 소스별로 먼저 실행하세요.")
        exit()

    # 병합
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    print(f"\n  📊 병합 전체: {X_all.shape}")
    print_class_dist(X_all, "(병합 후)")

    # NaN/inf 체크
    nan_count = np.isnan(X_all).sum()
    inf_count = np.isinf(X_all).sum()
    print(f"\n  🔍 품질 체크: NaN={nan_count} | inf={inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("  ⚠️  NaN/inf 발견! csv_to_npy 단계를 확인하세요.")

    # Train/Val 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.1, random_state=42, shuffle=True
    )

    # 저장
    version = "_".join([tag for tag, en in SOURCES.items() if en])
    np.save(os.path.join(OUTPUT_DIR, f"X_train_final_30fps_pred2s_{version}.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, f"y_train_final_30fps_pred2s_{version}.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, f"X_val_final_30fps_pred2s_{version}.npy"),   X_val)
    np.save(os.path.join(OUTPUT_DIR, f"y_val_final_30fps_pred2s_{version}.npy"),   y_val)

    print(f"\n💾 저장 완료!")
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  파일  : *_final_30fps_pred2s_{version}.npy")
    print(f"\n🎉 완료! 다음 단계:")
    print(f"  python ai_model/trajectory/tr_residual_30fps_pred2s.py")