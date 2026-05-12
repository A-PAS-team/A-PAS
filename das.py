# das.py에 임시로
import numpy as np
X = np.load("ai_model/data/Training/X_train_final_10fps_v3kin_cctv_carla.npy")
yaw = np.abs(X[:, :, 12])
max_yaw = np.max(yaw, axis=1)
print(f"[cctv_carla npy] median: {np.median(max_yaw):.4f}, 포화(>0.99): {(max_yaw > 0.99).mean():.1%}, 비영: {(max_yaw > 0.01).mean():.1%}")