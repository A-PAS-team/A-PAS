import numpy as np
import os

check_list = [
    ("CSV 기반 (30->10fps)", "X_train.npy", "y_train.npy"),
    ("JPG 기반 (ByteTrack)", "X_Training_17f.npy", "y_Training_17f.npy"),
    ("CSV 검증용", "X_val.npy", "y_val.npy"),
    ("JPG 검증용", "X_Validation_17f.npy", "y_Validation_17f.npy")
]

def analyze_all_data():
    print(f"{'데이터명':<20} | {'Shape':<18} | {'Feature':<7} | {'Y-Max':<7}")
    print("-" * 75)

    for label, x_file, y_file in check_list:
        if not os.path.exists(x_file):
            print(f"{label:<20} | ❌ 파일 없음")
            continue

        X = np.load(x_file)
        
        # ❗ 에러 방지: 차원이 3차원인지 먼저 확인
        if X.ndim == 3:
            feat_count = X.shape[2]
            y_max = X[:,:,1].max()
            status = "⚠️" if y_max > 1.01 else "✅"
            shape_str = str(X.shape)
        else:
            feat_count = "N/A"
            y_max = 0.0
            status = "❓ (데이터 부족/오류)"
            shape_str = str(X.shape)
        
        print(f"{label:<20} | {shape_str:<18} | {feat_count:<7} | {y_max:<7.4f} {status}")

    print("\n📌 [데이터 규격 상세 비교 (첫 샘플 첫 프레임)]")
    for label, x_file, _ in check_list:
        if os.path.exists(x_file):
            X = np.load(x_file)
            if X.ndim == 3 and X.shape[0] > 0:
                print(f"🔹 {label}: {X[0, 0, :].round(3)}")

if __name__ == "__main__":
    analyze_all_data()