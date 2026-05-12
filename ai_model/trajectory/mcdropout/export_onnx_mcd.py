"""
Export MC Dropout Residual LSTM to ONNX.

Important:
  model.eval() is called first so BN/LayerNorm stay in inference mode.
  Then Dropout modules only are switched back to train mode for MC sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

from residual_lstm_mcd import build_model, enable_mc_dropout


BASE_DIR = Path(__file__).resolve().parent

CONFIG: Dict[str, Any] = {
    "checkpoint_path": BASE_DIR / "checkpoints" / "best_residual_mcd_10fps.pt",
    "onnx_path": BASE_DIR / "residual_mcd_10fps.onnx",
    "seq_length": 20,
    "input_size": 17,
    "opset_version": 14,
    "batch_size": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CONFIG["checkpoint_path"])
    parser.add_argument("--onnx", type=Path, default=CONFIG["onnx_path"])
    parser.add_argument("--opset", type=int, default=CONFIG["opset_version"])
    return parser.parse_args()


def load_checkpoint(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint.get("model_config", {})
    return checkpoint, {}


def keep_dropout_active_only(model: nn.Module) -> None:
    model.eval()
    enable_mc_dropout(model)


def export() -> None:
    args = parse_args()
    state_dict, model_config = load_checkpoint(args.checkpoint)
    model = build_model(model_config)
    model.load_state_dict(state_dict)

    keep_dropout_active_only(model)

    dummy_input = torch.randn(
        CONFIG["batch_size"],
        CONFIG["seq_length"],
        CONFIG["input_size"],
        dtype=torch.float32,
    )
    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint: {args.checkpoint}")
    print(f"[export] onnx: {args.onnx}")
    print(f"[export] opset: {args.opset}")
    print("[export] Dropout modules are train=True; other modules remain eval.")

    torch.onnx.export(
        model,
        dummy_input,
        str(args.onnx),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    dummy_np = dummy_input.numpy()
    out1 = session.run(None, {"input": dummy_np})[0]
    out2 = session.run(None, {"input": dummy_np})[0]
    diff = float(np.abs(out1 - out2).max())
    print(f"[dropout check] max diff between runs: {diff:.8f}")
    if diff > 1e-8:
        print("[dropout check] PASS - stochastic active")
    else:
        print("[dropout check] FAIL - deterministic, MC Dropout 비활성")

    try:
        import onnx

        onnx_model = onnx.load(str(args.onnx))
        onnx.checker.check_model(onnx_model)
        dropout_nodes = [node for node in onnx_model.graph.node if node.op_type == "Dropout"]
        random_nodes = [node for node in onnx_model.graph.node if "Random" in node.op_type]
        print(
            "[check] ONNX checker passed. "
            f"Dropout nodes: {len(dropout_nodes)}, Random nodes: {len(random_nodes)}"
        )
    except Exception as exc:
        print(f"[check] ONNX checker skipped/failed: {exc}")

    print("[done] export complete")


if __name__ == "__main__":
    export()
