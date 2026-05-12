"""
A-PAS Residual LSTM + Kinematic Layer 학습
============================================
- 모델: ResidualLSTMKinematic (18피처 입력, Kinematic Layer 역전파)
- Loss: MSE + Physics Loss (heading consistency + curvature smoothness)
- 평가: ADE/FDE (Kinematic Layer 통과 후 좌표 기준)

사용법:
  python tr_residual_kinematic.py
  python tr_residual_kinematic.py --epochs 100 --patience 20
  python tr_residual_kinematic.py --finetune path/to/checkpoint.pth

결과:
  models/Kinematic/best_kinematic_{TAG}.pth
  models/Kinematic/training_report_{TAG}.png
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from residual_lstm_kinematic import (
    ResidualLSTMKinematic, ModelConfig, PhysicsLoss,
    CONFIG, count_parameters,
)

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS Residual LSTM + Kinematic Training")
parser.add_argument('--epochs',       type=int,   default=150)
parser.add_argument('--patience',     type=int,   default=15)
parser.add_argument('--batch-size',   type=int,   default=64)
parser.add_argument('--lr',           type=float, default=0.001)
parser.add_argument('--hidden',       type=int,   default=256)
parser.add_argument('--lambda-head',  type=float, default=0.1,
                    help="Heading consistency loss weight")
parser.add_argument('--lambda-curv',  type=float, default=0.05,
                    help="Curvature smoothness loss weight")
parser.add_argument('--finetune',     type=str,   default=None,
                    help="기존 checkpoint에서 fine-tuning (FC 레이어만 재초기화)")
parser.add_argument('--version',      type=str,   default="v3kin_cctv_carla")
args = parser.parse_args()

# ==========================================
# ⚙️ [설정]
# ==========================================
IMG_W, IMG_H = 1920, 1080
SEQ_LENGTH   = 10   # 1초 @ 10FPS
PRED_LENGTH  = 20   # 2초 @ 10FPS
INPUT_SIZE   = 17
TAG          = f"10fps_{args.version}"

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "..", "data", "Training")
MODEL_DIR = os.path.join(BASE_DIR, "..", "..", "..", "models", "Kinematic")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("  A-PAS Residual LSTM + Kinematic Layer Training")
print("=" * 60)
print(f"  SEQ / PRED    : {SEQ_LENGTH} / {PRED_LENGTH}")
print(f"  Input size    : {INPUT_SIZE} (12물리 + yaw_rate + 4원-핫)")
print(f"  Hidden size   : {args.hidden}")
print(f"  Epochs        : {args.epochs}")
print(f"  Batch size    : {args.batch_size}")
print(f"  LR            : {args.lr}")
print(f"  Lambda head   : {args.lambda_head}")
print(f"  Lambda curv   : {args.lambda_curv}")
print(f"  Fine-tune     : {args.finetune or 'None (from scratch)'}")
print(f"  Device        : {device}")
print(f"  Tag           : {TAG}")
print("=" * 60)


# ==========================================
# 📊 [데이터 로드]
# ==========================================
def load_data():
    print(f"\n📂 데이터 로드 중... [{TAG}]")

    # merged 파일이 있으면 사용, 없으면 개별 파일 로드
    merged_prefix = f"final_{TAG}"
    x_train_path = os.path.join(DATA_DIR, f"X_train_{merged_prefix}.npy")

    if os.path.exists(x_train_path):
        X_train = np.load(x_train_path)
        y_train = np.load(os.path.join(DATA_DIR, f"y_train_{merged_prefix}.npy"))
        X_val   = np.load(os.path.join(DATA_DIR, f"X_val_{merged_prefix}.npy"))
        y_val   = np.load(os.path.join(DATA_DIR, f"y_val_{merged_prefix}.npy"))
    else:
        # 개별 소스 파일 직접 로드 + 합치기
        print("  ⚠️  merged 파일 없음, 개별 소스 합침")
        X_parts, y_parts = [], []
        X_val_parts, y_val_parts = [], []

        for tag in ["cctv", "carla"]:
            suffix = f"10fps_{tag}_v3kin"
            xp = os.path.join(DATA_DIR, f"X_train_{suffix}.npy")
            if os.path.exists(xp):
                X_parts.append(np.load(xp))
                y_parts.append(np.load(os.path.join(DATA_DIR, f"y_train_{suffix}.npy")))
                X_val_parts.append(np.load(os.path.join(DATA_DIR, f"X_val_{suffix}.npy")))
                y_val_parts.append(np.load(os.path.join(DATA_DIR, f"y_val_{suffix}.npy")))
                print(f"    ✅ {tag}: {X_parts[-1].shape}")
            else:
                print(f"    ⏭️  {tag} 없음")

        if not X_parts:
            raise FileNotFoundError("학습 데이터 없음! csv_to_npy_10fps_v3_kinematic.py를 먼저 실행하세요.")

        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)
        X_val   = np.concatenate(X_val_parts, axis=0)
        y_val   = np.concatenate(y_val_parts, axis=0)

    print(f"  ✅ Train: {X_train.shape} / Val: {X_val.shape}")
    assert X_train.shape[-1] == INPUT_SIZE, f"피처 수 불일치: {X_train.shape[-1]} != {INPUT_SIZE}"

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


# ==========================================
# 📈 [그래프 저장]
# ==========================================
def save_plots(history):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(f"A-PAS Kinematic LSTM — Training Report ({TAG})", fontsize=14)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Row 1: Loss, ADE, FDE
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", color="royalblue")
    axes[0, 0].plot(epochs, history["val_loss"],   label="Val",   color="tomato")
    axes[0, 0].set_title("Total Loss"); axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["ade"], label="ADE (px)", color="mediumseagreen")
    axes[0, 1].axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Target')
    axes[0, 1].set_title("ADE"); axes[0, 1].set_ylabel("px")
    axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[0, 2].plot(epochs, history["fde"], label="FDE (px)", color="mediumpurple")
    axes[0, 2].axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Target')
    axes[0, 2].set_title("FDE"); axes[0, 2].set_ylabel("px")
    axes[0, 2].legend(); axes[0, 2].grid(True)

    # Row 2: Physics losses, MSE, LR
    axes[1, 0].plot(epochs, history["mse_loss"],     label="MSE",     color="royalblue")
    axes[1, 0].plot(epochs, history["heading_loss"],  label="Heading", color="orange")
    axes[1, 0].plot(epochs, history["curvature_loss"],label="Curvature",color="green")
    axes[1, 0].set_title("Loss Components"); axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, history["lr"], label="LR", color="gray")
    axes[1, 1].set_title("Learning Rate"); axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_yscale('log'); axes[1, 1].grid(True)

    # 커브 vs 직진 ADE (있으면)
    if "ade_curve" in history and history["ade_curve"]:
        axes[1, 2].plot(epochs, history["ade_curve"],    label="Curve ADE",    color="red")
        axes[1, 2].plot(epochs, history["ade_straight"], label="Straight ADE", color="blue")
        axes[1, 2].set_title("Curve vs Straight ADE"); axes[1, 2].set_ylabel("px")
        axes[1, 2].legend(); axes[1, 2].grid(True)
    else:
        axes[1, 2].axis('off')

    for ax in axes.flat:
        ax.set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, f"training_report_{TAG}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 그래프 저장: {path}")


# ==========================================
# 🚀 [학습]
# ==========================================
def train():
    train_loader, val_loader = load_data()

    # --- 모델 생성 ---
    model_config = ModelConfig(
        input_size=INPUT_SIZE,
        hidden_size=args.hidden,
        num_layers=2,
        pred_length=PRED_LENGTH,
        dropout_p=0.2,
        dt=0.1,
    )
    model = ResidualLSTMKinematic(model_config).to(device)

    # Fine-tuning: 기존 체크포인트 로드 (FC만 재초기화)
    if args.finetune:
        print(f"\n🔄 Fine-tuning from: {args.finetune}")
        state = torch.load(args.finetune, map_location=device)
        # LSTM 가중치만 로드 (FC는 output 크기가 다를 수 있음)
        lstm_state = {k: v for k, v in state.items() if k.startswith("lstm.")}
        model.load_state_dict(lstm_state, strict=False)
        print(f"  ✅ LSTM 가중치 로드 완료 ({len(lstm_state)} keys)")
        print(f"  ⚠️  FC 레이어는 새로 초기화됨")

    # --- Loss ---
    mse_criterion = nn.MSELoss()
    physics_loss_fn = PhysicsLoss(
        lambda_heading=args.lambda_head,
        lambda_curvature=args.lambda_curv,
    )

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5,
    )

    print(f"\n  📐 파라미터 수: {count_parameters(model):,}")
    print(f"  📉 Physics Loss: λ_heading={args.lambda_head}, λ_curvature={args.lambda_curv}")

    best_val_loss    = float('inf')
    early_stop_count = 0
    history = {
        "train_loss": [], "val_loss": [],
        "mse_loss": [], "heading_loss": [], "curvature_loss": [],
        "ade": [], "fde": [], "lr": [],
        "ade_curve": [], "ade_straight": [],
    }

    print(f"\n🚀 학습 시작!")
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0
        train_mse_sum = 0
        train_head_sum = 0
        train_curv_sum = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # forward_with_raw → positions + speed + yaw_rate
            pred, speed, yaw_rate = model.forward_with_raw(X_batch)

            # MSE loss (좌표 기준)
            mse_loss = mse_criterion(pred, y_batch)

            # Physics loss
            phys_loss = physics_loss_fn(pred, yaw_rate)

            # Total loss
            loss = mse_loss + phys_loss
            loss.backward()

            # Gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss_sum += loss.item()
            train_mse_sum  += mse_loss.item()
            with torch.no_grad():
                train_head_sum += physics_loss_fn.heading_consistency_loss(pred).item()
                train_curv_sum += physics_loss_fn.curvature_smoothness_loss(yaw_rate).item()

        n_batches = len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0
        total_ade = 0
        total_fde = 0
        n_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                pred, speed, yaw_rate = model.forward_with_raw(X_batch)
                mse_loss = mse_criterion(pred, y_batch)
                phys_loss = physics_loss_fn(pred, yaw_rate)
                val_loss_sum += (mse_loss + phys_loss).item()

                # ADE/FDE (픽셀 단위)
                p = pred.clone()
                t = y_batch.clone()
                p[..., 0] *= IMG_W; p[..., 1] *= IMG_H
                t[..., 0] *= IMG_W; t[..., 1] *= IMG_H

                dist = torch.sqrt(torch.sum((p - t) ** 2, dim=-1))
                total_ade += dist.mean(dim=1).sum().item()
                total_fde += dist[:, -1].sum().item()
                n_samples += X_batch.size(0)

        avg_train = train_loss_sum / n_batches
        avg_val   = val_loss_sum / len(val_loader)
        ade       = total_ade / n_samples
        fde       = total_fde / n_samples
        current_lr = optimizer.param_groups[0]['lr']

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["mse_loss"].append(train_mse_sum / n_batches)
        history["heading_loss"].append(train_head_sum / n_batches)
        history["curvature_loss"].append(train_curv_sum / n_batches)
        history["ade"].append(ade)
        history["fde"].append(fde)
        history["lr"].append(current_lr)

        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"Loss: {avg_train:.5f} (MSE:{train_mse_sum/n_batches:.5f} "
              f"H:{train_head_sum/n_batches:.5f} C:{train_curv_sum/n_batches:.5f}) | "
              f"Val: {avg_val:.5f} | ADE: {ade:.2f}px | FDE: {fde:.2f}px")

        scheduler.step(avg_val)
        if current_lr != optimizer.param_groups[0]['lr']:
            print(f"  📉 LR: {current_lr:.6f} → {optimizer.param_groups[0]['lr']:.6f}")

        # Best model 저장
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            early_stop_count = 0

            save_path = os.path.join(MODEL_DIR, f"best_kinematic_{TAG}.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config.to_dict(),
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "ade": ade,
                "fde": fde,
            }, save_path)
            print(f"  ⭐ Best saved! Val:{avg_val:.5f} ADE:{ade:.2f}px FDE:{fde:.2f}px")
        else:
            early_stop_count += 1
            print(f"  ⏳ Early stop: {early_stop_count}/{args.patience}")

        if early_stop_count >= args.patience:
            print(f"\n🛑 Early Stopping at epoch {epoch+1}")
            break

    # --- 학습 완료 ---
    save_plots(history)

    print(f"\n{'='*60}")
    print(f"  🎉 학습 완료! [Kinematic LSTM — {TAG}]")
    print(f"{'='*60}")
    print(f"  파라미터 수   : {count_parameters(model):,}")
    print(f"  Best Val Loss : {best_val_loss:.5f}")
    print(f"  Best ADE      : {min(history['ade']):.2f}px")
    print(f"  Best FDE      : {min(history['fde']):.2f}px")
    print(f"  총 에포크     : {len(history['ade'])}")
    print(f"  Physics Loss  : λ_H={args.lambda_head}, λ_C={args.lambda_curv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
