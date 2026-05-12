"""
Train Residual LSTM with MC Dropout for A-PAS.

This file is intentionally separate from the existing training scripts.
It reuses the same NPY data interface as the current pred2s Residual LSTM.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from residual_lstm_mcd import (
    CONFIG as MODEL_CONFIG,
    build_model,
    count_parameters,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]

CONFIG: Dict[str, Any] = {
    "img_w": 1920,
    "img_h": 1080,
    "seq_length": 20,
    "pred_length": 10,
    "batch_size": 64,
    "epochs": 300,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "early_stop_patience": 15,
    "num_workers": 0,
    "mc_val_samples": 5,
    "use_tensorboard": True,
    "use_wandb": False,
    "data_dir": PROJECT_ROOT / "ai_model" / "data" / "Training",
    "x_train": "X_train_final_10fps_v2_cctv_carla.npy",
    "y_train": "y_train_final_10fps_v2_cctv_carla.npy",
    "x_val": "X_val_final_10fps_v2_cctv_carla.npy",
    "y_val": "y_val_final_10fps_v2_cctv_carla.npy",
    "checkpoint_dir": BASE_DIR / "checkpoints",
    "checkpoint_name": "best_residual_mcd_10fps.pt",
    "log_dir": BASE_DIR / "runs",
    # TODO: 확인 필요 - 실제 CARLA 데이터셋 버전/파일명이 바뀌면 위 파일명을 맞춰야 함.
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=CONFIG["data_dir"])
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=CONFIG["lr"])
    parser.add_argument("--dropout-p", type=float, default=MODEL_CONFIG["dropout_p"])
    parser.add_argument("--mc-val-samples", type=int, default=CONFIG["mc_val_samples"])
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def load_data(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    print(f"[data] loading from: {data_dir}")
    x_train = np.load(data_dir / CONFIG["x_train"])
    y_train = np.load(data_dir / CONFIG["y_train"])
    x_val = np.load(data_dir / CONFIG["x_val"])
    y_val = np.load(data_dir / CONFIG["y_val"])

    print(f"[data] train X/y: {x_train.shape} / {y_train.shape}")
    print(f"[data] val   X/y: {x_val.shape} / {y_val.shape}")

    expected_x_tail = (CONFIG["seq_length"], MODEL_CONFIG["input_size"])
    expected_y_tail = (CONFIG["pred_length"], 2)
    if x_train.shape[1:] != expected_x_tail or y_train.shape[1:] != expected_y_tail:
        raise ValueError(
            f"Unexpected shape. X tail should be {expected_x_tail}, "
            f"y tail should be {expected_y_tail}."
        )

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def make_loggers(args: argparse.Namespace) -> tuple[Optional[Any], Optional[Any]]:
    tb_writer = None
    wandb_run = None

    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(CONFIG["log_dir"]))
            print(f"[log] tensorboard: {CONFIG['log_dir']}")
        except Exception as exc:
            print(f"[log] tensorboard unavailable, console only: {exc}")

    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project="a-pas-mc-dropout",
                config={**CONFIG, **MODEL_CONFIG, "dropout_p": args.dropout_p},
            )
            print("[log] wandb enabled")
        except Exception as exc:
            print(f"[log] wandb unavailable, console only: {exc}")

    return tb_writer, wandb_run


def ade_fde_px(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, int]:
    pred_px = pred.detach().clone()
    target_px = target.detach().clone()
    pred_px[..., 0] *= CONFIG["img_w"]
    pred_px[..., 1] *= CONFIG["img_h"]
    target_px[..., 0] *= CONFIG["img_w"]
    target_px[..., 1] *= CONFIG["img_h"]
    dist = torch.sqrt(((pred_px - target_px) ** 2).sum(dim=-1))
    ade_sum = dist.mean(dim=1).sum().item()
    fde_sum = dist[:, -1].sum().item()
    return ade_sum, fde_sum, pred.shape[0]


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mc_samples: int,
) -> tuple[float, float, float]:
    """Training-time validation is deterministic, matching the baseline.

    mc_samples is kept for CLI/call-site compatibility; MC sampling is used
    only by inference/export verification code.
    """
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    n = 0

    for x_b, y_b in val_loader:
        x_b = x_b.to(device)
        y_b = y_b.to(device)

        pred = model(x_b)
        total_loss += criterion(pred, y_b).item()
        ade_sum, fde_sum, batch_n = ade_fde_px(pred, y_b)
        total_ade += ade_sum
        total_fde += fde_sum
        n += batch_n

    return total_loss / len(val_loader), total_ade / n, total_fde / n


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    val_loss: float,
    ade: float,
    fde: float,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {**model_config},
            "train_config": {**train_config},
            "epoch": epoch,
            "val_loss": val_loss,
            "ade_px": ade,
            "fde_px": fde,
        },
        path,
    )


def train() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = {**MODEL_CONFIG, "dropout_p": args.dropout_p}
    train_cfg = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seq_length": CONFIG["seq_length"],
        "pred_length": CONFIG["pred_length"],
        "x_train": CONFIG["x_train"],
        "y_train": CONFIG["y_train"],
        "x_val": CONFIG["x_val"],
        "y_val": CONFIG["y_val"],
    }

    print(f"[device] {device}")
    print(f"[model] SEQ={CONFIG['seq_length']} -> PRED={CONFIG['pred_length']}, dropout={args.dropout_p}")
    print("[model] Current dataset mode: 2s observe -> 1s predict at 10 FPS.")
    print("[note] Expected deterministic ADE impact from MC Dropout: +0.3~0.5 px.")

    train_loader, val_loader = load_data(args.data_dir, args.batch_size, CONFIG["num_workers"])
    model = build_model(model_cfg).to(device)
    print(f"[model] trainable parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    tb_writer, wandb_run = make_loggers(args)

    best_val = float("inf")
    patience = 0
    ckpt_path = CONFIG["checkpoint_dir"] / CONFIG["checkpoint_name"]
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, ade, fde = validate(
            model,
            val_loader,
            criterion,
            device,
            mc_samples=args.mc_val_samples,
        )
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"ADE={ade:.2f}px FDE={fde:.2f}px lr={lr_now:.6g}"
        )

        if tb_writer is not None:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("metric/ade_px", ade, epoch)
            tb_writer.add_scalar("metric/fde_px", fde, epoch)
            tb_writer.add_scalar("optim/lr", lr_now, epoch)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "loss/train": train_loss,
                    "loss/val": val_loss,
                    "metric/ade_px": ade,
                    "metric/fde_px": fde,
                    "optim/lr": lr_now,
                    "epoch": epoch,
                }
            )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            save_checkpoint(model, ckpt_path, epoch, val_loss, ade, fde, model_cfg, train_cfg)
            print(f"[ckpt] saved best: {ckpt_path} (ADE={ade:.2f}px, FDE={fde:.2f}px)")
        else:
            patience += 1
            if patience >= CONFIG["early_stop_patience"]:
                print(f"[early-stop] epoch={epoch}")
                break

    elapsed_h = (time.time() - start_time) / 3600
    print(f"[done] best_val={best_val:.6f}, elapsed={elapsed_h:.2f}h")

    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train()
