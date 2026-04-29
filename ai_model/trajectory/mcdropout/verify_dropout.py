"""
Verify whether ONNX Runtime keeps Dropout active for MC sampling.

Procedure:
  1. Run the same input through residual_mcd.onnx 10 times.
  2. Check whether outputs differ.
  3. std == 0 means Dropout is dead/deterministic in inference.

Fallback options if ONNX Dropout is deterministic:
  a) Compare opset 12, 14, 17.
     Pros: simple, no model rewrite.
     Cons: backend-dependent; ORT may still optimize Dropout away.
  b) Replace nn.Dropout with explicit stochastic masking/noise
     such as torch.rand_like(x) > p and x * mask / (1 - p).
     Pros: exports as primitive ops and is easier to verify.
     Cons: random export support can vary; must audit generated ONNX graph.
  c) Custom op for stochastic masking.
     Pros: full control over runtime behavior.
     Cons: worst portability; extra deployment work on Raspberry Pi/Hailo setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    "onnx_path": BASE_DIR / "residual_mcd.onnx",
    "checkpoint_path": BASE_DIR / "checkpoints" / "best_residual_mcd.pt",
    "num_runs": 10,
    "batch_size": 1,
    "seq_length": 60,
    "input_size": 17,
    "std_epsilon": 1e-8,
    "providers": ["CPUExecutionProvider"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=CONFIG["onnx_path"])
    parser.add_argument("--runs", type=int, default=CONFIG["num_runs"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-pytorch-modes", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CONFIG["checkpoint_path"])
    return parser.parse_args()


def verify_onnx_dropout(onnx_path: Path, runs: int, seed: int) -> bool:
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    input_data = rng.normal(
        loc=0.5,
        scale=0.1,
        size=(CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["input_size"]),
    ).astype(np.float32)
    input_data[..., :2] = np.clip(input_data[..., :2], 0.0, 1.0)

    session = ort.InferenceSession(str(onnx_path), providers=CONFIG["providers"])
    input_name = session.get_inputs()[0].name
    outputs = []

    for _ in range(runs):
        out = session.run(None, {input_name: input_data})[0]
        outputs.append(out)

    stacked = np.stack(outputs, axis=0)
    per_element_std = stacked.std(axis=0)
    mean_std = float(per_element_std.mean())
    max_std = float(per_element_std.max())
    max_pairwise_delta = float(np.max(np.abs(stacked[1:] - stacked[:1])))

    print("[onnx] same input repeated")
    print(f"  runs              : {runs}")
    print(f"  output shape      : {stacked.shape}")
    print(f"  mean std          : {mean_std:.10f}")
    print(f"  max std           : {max_std:.10f}")
    print(f"  max pairwise delta: {max_pairwise_delta:.10f}")

    active = max_std > CONFIG["std_epsilon"] and max_pairwise_delta > CONFIG["std_epsilon"]
    if active:
        print("[result] PASS - Dropout appears active in ONNX Runtime.")
    else:
        print("[result] FAIL - Dropout appears deterministic/dead in ONNX Runtime.")
        print("[next] Try opset 12/14/17 or explicit stochastic masking.")
    return active


def verify_pytorch_modes(checkpoint_path: Path) -> None:
    import torch
    import torch.nn as nn

    from residual_lstm_mcd import MCDropout, build_model, set_eval_with_mc_dropout

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model = build_model(checkpoint.get("model_config", {}))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = build_model()
        model.load_state_dict(checkpoint)

    set_eval_with_mc_dropout(model)

    dropout_active = []
    norm_eval = []
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            dropout_active.append(module.training)
        if isinstance(module, MCDropout):
            dropout_active.append(module.force_mc_dropout)
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            norm_eval.append(not module.training)

    print("[pytorch] mode check")
    print(f"  dropout active for MC     : {all(dropout_active)} ({len(dropout_active)} layers)")
    print(f"  BN/LayerNorm eval=True    : {all(norm_eval) if norm_eval else True} ({len(norm_eval)} layers)")
    assert all(dropout_active), "At least one Dropout layer is not active for MC sampling."
    assert all(norm_eval) if norm_eval else True, "A BN/LayerNorm layer is not in eval mode."


def main() -> None:
    args = parse_args()
    if args.check_pytorch_modes:
        verify_pytorch_modes(args.checkpoint)
    verify_onnx_dropout(args.onnx, args.runs, args.seed)


if __name__ == "__main__":
    main()
