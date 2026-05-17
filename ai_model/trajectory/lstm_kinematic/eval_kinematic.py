"""
A-PAS Kinematic LSTM 평가 스크립트
====================================
학습 완료 후 상세 분석:
  1. 전체 ADE/FDE
  2. 커브 vs 직진 분리 평가
  3. 클래스별 (Person/Car/Motorcycle/Large_Veh) 평가
  4. 예측 궤적 시각화 (커브 샘플)
  5. 베이스라인(기존 Residual LSTM) 대비 비교 (선택)

사용법:
  python eval_kinematic.py --checkpoint models/Kinematic/best_kinematic_10fps_v3kin.pth
  python eval_kinematic.py --checkpoint best.pth --baseline models/Residual/best_residual.pth
"""

import torch
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from residual_lstm_kinematic import (
    ResidualLSTMKinematic, ModelConfig, count_parameters,
)

# ==========================================
# ⚙️ [설정]
# ==========================================
IMG_W, IMG_H = 1920, 1080
SEQ_LENGTH   = 10
PRED_LENGTH  = 20
INPUT_SIZE   = 17
DT           = 0.1

# yaw_rate threshold: 이 값 이상이면 "커브" 시퀀스로 분류
CURVE_YAW_RATE_THRESH = 0.05  # 정규화 단위 (0.05 = 0.05 rad/s)

CLASS_NAMES = {0: "Person", 1: "Car", 2: "Motorcycle", 3: "Large_Veh"}

parser = argparse.ArgumentParser(description="A-PAS Kinematic LSTM Evaluation")
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--baseline',   type=str, default=None,
                    help="베이스라인 모델 checkpoint (비교용)")
parser.add_argument('--data-dir',   type=str, default=None)
parser.add_argument('--n-samples',  type=int, default=None,
                    help="평가할 샘플 수 (None이면 전체)")
parser.add_argument('--save-dir',   type=str, default=None)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
if args.data_dir:
    DATA_DIR = args.data_dir
else:
    DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data", "Training")

if args.save_dir:
    SAVE_DIR = args.save_dir
else:
    SAVE_DIR = os.path.join(BASE_DIR, "..", "..", "..", "models", "Kinematic", "eval")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================================
# 📊 [데이터 로드]
# ==========================================
def load_val_data():
    """검증 데이터 로드. merged 또는 개별 소스."""
    # merged 파일 시도
    for pattern in [
        "X_val_final_10fps_v3kin*.npy",
        "X_val_10fps_cctv_v3kin.npy",
    ]:
        import glob
        matches = glob.glob(os.path.join(DATA_DIR, pattern))
        if matches:
            break

    # 개별 소스 합침
    X_parts, y_parts = [], []
    for tag in ["cctv", "carla"]:
        xp = os.path.join(DATA_DIR, f"X_val_10fps_{tag}_v3kin.npy")
        yp = os.path.join(DATA_DIR, f"y_val_10fps_{tag}_v3kin.npy")
        if os.path.exists(xp):
            X_parts.append(np.load(xp))
            y_parts.append(np.load(yp))
            print(f"  ✅ {tag}: {X_parts[-1].shape}")

    if not X_parts:
        raise FileNotFoundError("검증 데이터 없음!")

    X_val = np.concatenate(X_parts, axis=0)
    y_val = np.concatenate(y_parts, axis=0)

    if args.n_samples:
        X_val = X_val[:args.n_samples]
        y_val = y_val[:args.n_samples]

    print(f"  총 검증 샘플: {len(X_val):,}")
    return X_val, y_val


def load_model(checkpoint_path):
    """체크포인트에서 모델 로드."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_config" in ckpt:
        config = ModelConfig.from_dict(ckpt["model_config"])
    else:
        config = ModelConfig()

    model = ResidualLSTMKinematic(config).to(device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    print(f"  모델: {checkpoint_path}")
    print(f"  파라미터: {count_parameters(model):,}")
    if "epoch" in ckpt:
        print(f"  학습 epoch: {ckpt['epoch']}")
    if "ade" in ckpt:
        print(f"  학습 시 ADE: {ckpt['ade']:.2f}px")

    return model


# ==========================================
# 📏 [평가 함수]
# ==========================================
def compute_metrics(model, X_val, y_val):
    """전체/커브/직진/클래스별 ADE/FDE 계산."""
    model.eval()

    all_ade = []
    all_fde = []
    all_is_curve = []
    all_class_idx = []

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(X_val), batch_size):
            end = min(start + batch_size, len(X_val))
            X_batch = torch.FloatTensor(X_val[start:end]).to(device)
            y_batch = torch.FloatTensor(y_val[start:end]).to(device)

            pred = model(X_batch)  # (batch, pred, 2)

            # 픽셀 변환
            p = pred.clone()
            t = y_batch.clone()
            p[..., 0] *= IMG_W; p[..., 1] *= IMG_H
            t[..., 0] *= IMG_W; t[..., 1] *= IMG_H

            dist = torch.sqrt(torch.sum((p - t) ** 2, dim=-1))  # (batch, pred)
            ade = dist.mean(dim=1).cpu().numpy()   # (batch,)
            fde = dist[:, -1].cpu().numpy()         # (batch,)

            all_ade.extend(ade.tolist())
            all_fde.extend(fde.tolist())

            # 커브 판별: 입력의 yaw_rate (index 12) 절댓값 최대
            yaw_rates = X_batch[:, :, 12].cpu().numpy()
            max_yaw = np.max(np.abs(yaw_rates), axis=1)
            all_is_curve.extend((max_yaw > CURVE_YAW_RATE_THRESH).tolist())

            # 클래스 판별: 원-핫 (index 13~16, crosswalk 제외)
            one_hot = X_batch[:, 0, 13:17].cpu().numpy()
            cls_idx = np.argmax(one_hot, axis=1)
            all_class_idx.extend(cls_idx.tolist())

    all_ade = np.array(all_ade)
    all_fde = np.array(all_fde)
    all_is_curve = np.array(all_is_curve)
    all_class_idx = np.array(all_class_idx)

    results = {
        "total_ade": float(all_ade.mean()),
        "total_fde": float(all_fde.mean()),
        "total_ade_std": float(all_ade.std()),
        "total_fde_std": float(all_fde.std()),
        "n_total": len(all_ade),
    }

    # 커브 vs 직진
    curve_mask = all_is_curve
    straight_mask = ~all_is_curve
    if curve_mask.sum() > 0:
        results["curve_ade"] = float(all_ade[curve_mask].mean())
        results["curve_fde"] = float(all_fde[curve_mask].mean())
        results["n_curve"] = int(curve_mask.sum())
    if straight_mask.sum() > 0:
        results["straight_ade"] = float(all_ade[straight_mask].mean())
        results["straight_fde"] = float(all_fde[straight_mask].mean())
        results["n_straight"] = int(straight_mask.sum())

    # 클래스별
    for cls_id in range(4):
        cls_mask = all_class_idx == cls_id
        if cls_mask.sum() > 0:
            name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            results[f"{name}_ade"] = float(all_ade[cls_mask].mean())
            results[f"{name}_fde"] = float(all_fde[cls_mask].mean())
            results[f"n_{name}"] = int(cls_mask.sum())

    return results


def print_results(results, label=""):
    """결과 출력."""
    print(f"\n{'='*55}")
    print(f"📊 평가 결과 {label}")
    print(f"{'='*55}")
    print(f"  전체 ADE: {results['total_ade']:.2f} ± {results['total_ade_std']:.2f} px")
    print(f"  전체 FDE: {results['total_fde']:.2f} ± {results['total_fde_std']:.2f} px")
    print(f"  샘플 수 : {results['n_total']:,}")

    if "curve_ade" in results:
        print(f"\n  [커브]   ADE: {results['curve_ade']:.2f}px  FDE: {results['curve_fde']:.2f}px  (n={results['n_curve']:,})")
    if "straight_ade" in results:
        print(f"  [직진]   ADE: {results['straight_ade']:.2f}px  FDE: {results['straight_fde']:.2f}px  (n={results['n_straight']:,})")

    for cls_id in range(4):
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        key = f"{name}_ade"
        if key in results:
            print(f"  [{name:<12}] ADE: {results[key]:.2f}px  FDE: {results[f'{name}_fde']:.2f}px  (n={results[f'n_{name}']:,})")

    print(f"{'='*55}")


def save_comparison_plot(results_kin, results_base=None):
    """비교 막대 그래프 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    categories = ["Total"]
    kin_ade = [results_kin["total_ade"]]
    kin_fde = [results_kin["total_fde"]]

    if "curve_ade" in results_kin:
        categories.append("Curve")
        kin_ade.append(results_kin["curve_ade"])
        kin_fde.append(results_kin["curve_fde"])
    if "straight_ade" in results_kin:
        categories.append("Straight")
        kin_ade.append(results_kin["straight_ade"])
        kin_fde.append(results_kin["straight_fde"])

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width/2, kin_ade, width, label="Kinematic", color="royalblue")
    axes[1].bar(x - width/2, kin_fde, width, label="Kinematic", color="royalblue")

    if results_base:
        base_ade = [results_base["total_ade"]]
        base_fde = [results_base["total_fde"]]
        if "curve_ade" in results_base:
            base_ade.append(results_base["curve_ade"])
            base_fde.append(results_base["curve_fde"])
        if "straight_ade" in results_base:
            base_ade.append(results_base["straight_ade"])
            base_fde.append(results_base["straight_fde"])

        axes[0].bar(x + width/2, base_ade, width, label="Baseline", color="tomato")
        axes[1].bar(x + width/2, base_fde, width, label="Baseline", color="tomato")

    axes[0].set_title("ADE Comparison (px)")
    axes[1].set_title("FDE Comparison (px)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "eval_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 비교 그래프 저장: {path}")


# ==========================================
# 🚀 [메인]
# ==========================================
# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    print(f"\n🔍 A-PAS Kinematic LSTM 평가")
    print(f"   Device: {device}")

    X_val, y_val = load_val_data()

    # ==========================================
    # 📦 모델 로드
    # ==========================================
    print(f"\n📦 Kinematic 모델 로드...")
    model_kin = load_model(args.checkpoint)

    # ==========================================
    # 📊 ADE / FDE 평가
    # ==========================================
    results_kin = compute_metrics(model_kin, X_val, y_val)
    print_results(results_kin, "[Kinematic LSTM]")

    # ==========================================
    # 🔍 Predicted yaw_rate distribution 분석
    # ==========================================
    print(f"\n{'='*55}")
    print("🔍 Predicted yaw_rate Distribution Analysis")
    print(f"{'='*55}")

    all_pred_yaw = []
    all_pred_speed = []

    model_kin.eval()

    with torch.no_grad():
        batch_size = 256

        for start in range(0, len(X_val), batch_size):
            end = min(start + batch_size, len(X_val))

            X_batch = torch.FloatTensor(X_val[start:end]).to(device)

            # positions + raw outputs
            _, pred_speed, pred_yaw = model_kin.forward_with_raw(X_batch)

            pred_yaw_np = pred_yaw.cpu().numpy().reshape(-1)
            pred_speed_np = pred_speed.cpu().numpy().reshape(-1)

            all_pred_yaw.append(pred_yaw_np)
            all_pred_speed.append(pred_speed_np)

    all_pred_yaw = np.concatenate(all_pred_yaw)
    all_pred_speed = np.concatenate(all_pred_speed)

    # ==========================================
    # 📊 GT yaw_rate distribution
    # ==========================================
    gt_yaw = X_val[:, :, 12].reshape(-1)

    # ==========================================
    # 📊 Pred yaw_rate 통계
    # ==========================================
    print("\n📊 Predicted yaw_rate stats")
    print(f"  mean             : {all_pred_yaw.mean():.6f}")
    print(f"  std              : {all_pred_yaw.std():.6f}")
    print(f"  abs mean         : {np.abs(all_pred_yaw).mean():.6f}")
    print(f"  min / max        : {all_pred_yaw.min():.6f} / {all_pred_yaw.max():.6f}")

    pred_straight_ratio = (np.abs(all_pred_yaw) < 0.01).mean()

    print(f"  |yaw| < 0.01     : {pred_straight_ratio:.2%}")

    # ==========================================
    # 📊 GT yaw_rate 통계
    # ==========================================
    print("\n📊 Ground Truth yaw_rate stats")
    print(f"  mean             : {gt_yaw.mean():.6f}")
    print(f"  std              : {gt_yaw.std():.6f}")
    print(f"  abs mean         : {np.abs(gt_yaw).mean():.6f}")
    print(f"  min / max        : {gt_yaw.min():.6f} / {gt_yaw.max():.6f}")

    gt_straight_ratio = (np.abs(gt_yaw) < 0.01).mean()

    print(f"  |yaw| < 0.01     : {gt_straight_ratio:.2%}")

    # ==========================================
    # 📈 GT vs Pred 비교
    # ==========================================
    print(f"\n{'='*55}")
    print("📈 GT vs Pred yaw_rate 비교")
    print(f"{'='*55}")

    std_ratio = all_pred_yaw.std() / (gt_yaw.std() + 1e-6)
    abs_ratio = np.abs(all_pred_yaw).mean() / (np.abs(gt_yaw).mean() + 1e-6)

    print(f"  std ratio        : {std_ratio:.4f}")
    print(f"  abs mean ratio   : {abs_ratio:.4f}")

    if std_ratio < 0.3:
        print("\n🚨 WARNING: yaw_rate collapse 의심!")
        print("   모델이 회전을 거의 출력하지 않을 가능성이 큼")

    # ==========================================
    # 📊 speed 통계
    # ==========================================
    print(f"\n{'='*55}")
    print("📊 Predicted speed stats")
    print(f"{'='*55}")

    print(f"  mean             : {all_pred_speed.mean():.6f}")
    print(f"  std              : {all_pred_speed.std():.6f}")
    print(f"  abs mean         : {np.abs(all_pred_speed).mean():.6f}")
    print(f"  min / max        : {all_pred_speed.min():.6f} / {all_pred_speed.max():.6f}")

    # ==========================================
    # 📦 베이스라인 비교 (선택)
    # ==========================================
    results_base = None

    if args.baseline:
        print(f"\n📦 베이스라인 모델 로드...")

        try:
            from residual_lstm_mcd import build_model as build_baseline

            ckpt = torch.load(args.baseline, map_location=device)

            baseline = build_baseline(ckpt.get("model_config"))

            baseline.load_state_dict(ckpt["model_state_dict"])

            baseline.eval().to(device)

            # yaw_rate 제거 → 기존 16피처 모델용
            X_val_17 = np.concatenate(
                [X_val[:, :, :12], X_val[:, :, 13:]],
                axis=-1
            )

            results_base = compute_metrics(
                baseline,
                X_val_17,
                y_val
            )

            print_results(results_base, "[Baseline Residual LSTM]")

            print(f"\n📈 개선율:")
            print(
                f"  ADE: {results_base['total_ade']:.2f} → "
                f"{results_kin['total_ade']:.2f} "
                f"({(1 - results_kin['total_ade']/results_base['total_ade'])*100:+.1f}%)"
            )

            print(
                f"  FDE: {results_base['total_fde']:.2f} → "
                f"{results_kin['total_fde']:.2f} "
                f"({(1 - results_kin['total_fde']/results_base['total_fde'])*100:+.1f}%)"
            )

        except Exception as e:
            print(f"  ⚠️ 베이스라인 평가 실패: {e}")

    # ==========================================
    # 📊 그래프 저장
    # ==========================================
    save_comparison_plot(results_kin, results_base)

    print(f"\n🎉 평가 완료!")


if __name__ == "__main__":
    main()