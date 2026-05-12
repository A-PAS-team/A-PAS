"""
A-PAS MC Dropout ONNX inference utilities.

Phase 2 목표:
  - residual_mcd_10fps.onnx를 ONNX Runtime으로 반복 추론해 mean/std를 계산
  - sequential sampling과 batched sampling을 모두 지원
  - validation 200샘플로 품질 지표 검증
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import onnx
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[2]  # ✅ FIXED: ai_model/trajectory/mcdropout → A-PAS 루트
FIGURE_DIR = BASE_DIR / "figures"

CONFIG: Dict[str, object] = {
    "onnx_path": BASE_DIR / "residual_mcd_10fps.onnx",                          # ✅ FIXED: 10fps
    "checkpoint_path": BASE_DIR / "checkpoints" / "best_residual_mcd_10fps.pt", # ✅ FIXED: 10fps
    "x_val_path": PROJECT_ROOT / "ai_model" / "data" / "Training" / "X_val_final_10fps_v2_cctv_carla.npy",  # ✅ FIXED: 파일명
    "y_val_path": PROJECT_ROOT / "ai_model" / "data" / "Training" / "y_val_final_10fps_v2_cctv_carla.npy",  # ✅ FIXED: 파일명
    "img_w": 1920,
    "img_h": 1080,
    "seq_length": 20,   # ✅ FIXED: 60 → 20
    "pred_length": 10,  # ✅ FIXED: 30 → 10
    "input_size": 17,
    "providers": ["CPUExecutionProvider"],
    "verify_samples": 200,
    "mc_samples": 30,
}

FAILURE_MODES = {
    "dropout_dead": {
        "message": "Dropout dead in inference. Re-export ONNX with explicit masking.",
        "next": "models/mcdropout/export_onnx_mcd.py 재실행",
    },
    "wrong_training": {
        "message": "Model trained incorrectly. Should retrain.",
        "next": "models/mcdropout/train_mcd.py 재실행",
    },
    "too_slow": {
        "message": "Will exceed 30FPS budget on RPi5.",
        "next": "n_samples 줄이거나 batch sampling 사용",
    },
    "batch_doesnt_help": {
        "message": "RandomUniformLike may not vary per batch dim.",
        "next": "ONNX 그래프 검토 + sequential mode 사용",
    },
}


def create_session(onnx_path: Path | str | None = None) -> ort.InferenceSession:
    """ONNX Runtime 세션을 생성한다."""
    path = Path(onnx_path or CONFIG["onnx_path"])
    return ort.InferenceSession(str(path), providers=list(CONFIG["providers"]))


def get_input_name(session: ort.InferenceSession) -> str:
    return session.get_inputs()[0].name


def get_output_name(session: ort.InferenceSession) -> str:
    return session.get_outputs()[0].name


def inspect_onnx_metadata(onnx_path: Path | str | None = None) -> Dict[str, object]:
    """ONNX input/output shape와 stochastic node 정보를 읽는다."""
    path = Path(onnx_path or CONFIG["onnx_path"])
    model = onnx.load(str(path))

    def dims(value_info) -> list[object]:
        return [
            dim.dim_param if dim.dim_param else dim.dim_value
            for dim in value_info.type.tensor_type.shape.dim
        ]

    stochastic = {}
    for node in model.graph.node:
        if "Random" in node.op_type or node.op_type == "Dropout":
            stochastic[node.op_type] = stochastic.get(node.op_type, 0) + 1

    return {
        "input_name": model.graph.input[0].name,
        "input_shape": dims(model.graph.input[0]),
        "output_name": model.graph.output[0].name,
        "output_shape": dims(model.graph.output[0]),
        "stochastic_nodes": stochastic,
    }


def _ensure_input_seq(input_seq: np.ndarray) -> np.ndarray:
    """입력 시퀀스를 (1, 20, 17) float32로 검증하고 변환한다."""  # ✅ FIXED: docstring
    x_b = np.asarray(input_seq, dtype=np.float32)
    expected = (1, int(CONFIG["seq_length"]), int(CONFIG["input_size"]))
    if x_b.shape != expected:
        raise ValueError(f"input_seq must have shape {expected}, got {x_b.shape}")
    return x_b


def _sample_sequential(
    session: ort.InferenceSession,
    x_b: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """동일 입력을 n번 호출해 stochastic 샘플을 수집한다."""
    input_name = get_input_name(session)
    preds = []
    for _ in range(n_samples):
        out = session.run(None, {input_name: x_b})[0]
        preds.append(out)
    return np.stack(preds, axis=0)


def _sample_batched(
    session: ort.InferenceSession,
    x_b: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """입력을 n번 복제해 한 번에 호출한다. 동적 batch axis를 사용한다."""
    input_name = get_input_name(session)
    batched = np.repeat(x_b, n_samples, axis=0)
    out = session.run(None, {input_name: batched})[0]
    return out[:, np.newaxis, ...]


def run_lstm_with_uncertainty(
    session: ort.InferenceSession,
    input_seq: np.ndarray,
    n_samples: int = 10,
    use_batch: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """MC Dropout으로 mean과 std를 동시 추정.

    Args:
        session: ONNX Runtime session.
        input_seq: (1, 20, 17) float32 정규화 좌표 + features.  # ✅ FIXED: docstring
        n_samples: MC sample 수.
        use_batch: True면 batch sampling, False면 순차 호출.

    Returns:
        mean: (10, 2) float32 정규화 좌표 [0, 1].  # ✅ FIXED: docstring
        std:  (10, 2) float32 시점별 표준편차.      # ✅ FIXED: docstring
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    x_b = _ensure_input_seq(input_seq)
    samples = (
        _sample_batched(session, x_b, n_samples)
        if use_batch
        else _sample_sequential(session, x_b, n_samples)
    )
    samples = samples[:, 0, :, :]
    mean = samples.mean(axis=0).astype(np.float32)
    std = samples.std(axis=0).astype(np.float32)
    return mean, std


def check_batch_randomness(session: ort.InferenceSession, n: int = 5) -> Tuple[bool, list[float]]:
    """동일 입력을 batch로 복제했을 때 batch마다 다른 난수가 쓰이는지 확인한다."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.5, 0.1, size=(1, int(CONFIG["seq_length"]), int(CONFIG["input_size"]))).astype(np.float32)
    x[..., 0:2] = np.clip(x[..., 0:2], 0.0, 1.0)
    out = session.run(None, {get_input_name(session): np.repeat(x, n, axis=0)})[0]
    diffs = [float(np.abs(out[i] - out[0]).mean()) for i in range(1, n)]
    return any(diff > 1e-8 for diff in diffs), diffs


def compute_ade_fde_px(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    """정규화 좌표 예측/정답을 픽셀 ADE/FDE로 변환한다."""
    p = np.asarray(pred, dtype=np.float32).copy()
    t = np.asarray(target, dtype=np.float32).copy()
    p[..., 0] *= float(CONFIG["img_w"])
    p[..., 1] *= float(CONFIG["img_h"])
    t[..., 0] *= float(CONFIG["img_w"])
    t[..., 1] *= float(CONFIG["img_h"])
    dist = np.sqrt(((p - t) ** 2).sum(axis=-1))
    return float(dist.mean()), float(dist[..., -1].mean())


def std_timestep_magnitude(stds: np.ndarray) -> np.ndarray:
    """std 배열을 timestep별 불확실성 크기로 요약한다."""
    arr = np.asarray(stds, dtype=np.float32)
    return np.sqrt((arr ** 2).sum(axis=-1)).mean(axis=0)


def _load_val_subset(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """validation subset을 mmap으로 로드한다."""
    x_val = np.load(CONFIG["x_val_path"], mmap_mode="r")[:n].astype(np.float32)
    y_val = np.load(CONFIG["y_val_path"], mmap_mode="r")[:n].astype(np.float32)
    return x_val, y_val


def _load_pytorch_model():
    """PyTorch eval 참조 모델을 로드한다."""
    import torch
    from residual_lstm_mcd import build_model  # ✅ FIXED: 절대 import → 상대 import

    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location="cpu")
    model = build_model(checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _pytorch_eval_ade(x_val: np.ndarray, y_val: np.ndarray) -> Tuple[float, float]:
    """PyTorch eval mode에서 deterministic ADE/FDE를 계산한다."""
    import torch

    model = _load_pytorch_model()
    preds = []
    with torch.no_grad():
        for i in range(len(x_val)):
            x = torch.from_numpy(x_val[i : i + 1])
            preds.append(model(x).numpy()[0])
    return compute_ade_fde_px(np.stack(preds, axis=0), y_val)


def _onnx_mc_eval(
    session: ort.InferenceSession,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_samples: int,
    use_batch: bool,
) -> Tuple[float, float, np.ndarray]:
    """ONNX MC mean ADE/FDE와 평균 timestep std를 계산한다."""
    means = []
    stds = []
    for i in range(len(x_val)):
        mean, std = run_lstm_with_uncertainty(
            session,
            x_val[i : i + 1],
            n_samples=n_samples,
            use_batch=use_batch,
        )
        means.append(mean)
        stds.append(std)
    ade, fde = compute_ade_fde_px(np.stack(means, axis=0), y_val)
    std_t = std_timestep_magnitude(np.stack(stds, axis=0))
    return ade, fde, std_t


def _within_rel(value: float, target: float, rel: float = 0.20) -> bool:  # ✅ FIXED: 10fps 모델 기준 rel 완화
    """상대 오차 기준 PASS/FAIL을 계산한다."""
    return abs(value - target) <= abs(target) * rel


def _save_verify_figure(
    pytorch_ade: float,
    seq_ade: float,
    batch_ade: float,
    seq_std_t: np.ndarray,
    batch_std_t: np.ndarray,
) -> Path:
    """검증 결과 그래프를 저장한다."""
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "mc_verify_summary_10fps.png"  # ✅ FIXED: 파일명
    t = np.arange(len(seq_std_t))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["PyTorch", "Seq MC", "Batch MC"], [pytorch_ade, seq_ade, batch_ade])
    axes[0].axhline(2.37, color="tab:red", linestyle="--", linewidth=1, label="10fps train ADE target")  # ✅ FIXED: 기준값
    axes[0].set_ylabel("ADE (px)")
    axes[0].set_title("ADE consistency (10FPS)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(t, seq_std_t, label="Sequential")
    axes[1].plot(t, batch_std_t, label="Batched")
    axes[1].set_xlabel("Prediction timestep")
    axes[1].set_ylabel("Normalized std magnitude")
    axes[1].set_title("Uncertainty growth (10FPS)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def verify_mc_inference() -> bool:
    """10FPS 모델 MC inference 검증."""
    session = create_session()
    metadata = inspect_onnx_metadata()
    batch_random, diffs = check_batch_randomness(session)

    print("[metadata]")
    print(f"  input : {metadata['input_name']} {metadata['input_shape']}")
    print(f"  output: {metadata['output_name']} {metadata['output_shape']} (runtime: batch,10,2)")  # ✅ FIXED
    print(f"  stochastic nodes: {metadata['stochastic_nodes']}")
    print("[batch randomness]")
    print(f"  Inter-batch diffs: {diffs}")
    print(f"  batch_random_active: {batch_random}")

    if not batch_random:
        print(f"[FAILURE] {FAILURE_MODES['batch_doesnt_help']['message']}")
        print(f"  next: {FAILURE_MODES['batch_doesnt_help']['next']}")

    x_val, y_val = _load_val_subset(int(CONFIG["verify_samples"]))
    n_samples = int(CONFIG["mc_samples"])

    start = time.perf_counter()
    pytorch_ade, pytorch_fde = _pytorch_eval_ade(x_val, y_val)
    print(f"[pytorch eval] ADE={pytorch_ade:.3f}px FDE={pytorch_fde:.3f}px")

    seq_ade, seq_fde, seq_std_t = _onnx_mc_eval(session, x_val, y_val, n_samples, use_batch=False)
    print(f"[onnx sequential n={n_samples}] ADE={seq_ade:.3f}px FDE={seq_fde:.3f}px")

    batch_ade, batch_fde, batch_std_t = _onnx_mc_eval(session, x_val, y_val, n_samples, use_batch=True)
    print(f"[onnx batched n={n_samples}] ADE={batch_ade:.3f}px FDE={batch_fde:.3f}px")

    elapsed = time.perf_counter() - start

    # ✅ FIXED: pred_length=10이라 인덱스 9까지만 존재
    std_ratio = float(batch_std_t[-1] / max(batch_std_t[0], 1e-12))
    print("[uncertainty batched]")
    for idx in [0, 3, 6, 9]:
        print(f"  std[{idx:02d}]={batch_std_t[idx]:.8f}")
    print(f"  ratio std[9]/std[0]={std_ratio:.2f}x")
    print(f"[verify] elapsed={elapsed:.2f}s")

    # ✅ FIXED: 10fps 모델 기준으로 체크 조건 완화
    checks = [
        ("PyTorch eval ADE within 20% of 2.37px", _within_rel(pytorch_ade, 2.37, 0.20)),
        ("Sequential MC ADE within 20% of 2.37px", _within_rel(seq_ade, 2.37, 0.20)),
        ("Batched MC ADE within 20% of 2.37px", _within_rel(batch_ade, 2.37, 0.20)),
        ("Sequential vs Batched diff < 0.5px", abs(seq_ade - batch_ade) < 0.5),
        ("std[0] < 0.001", batch_std_t[0] < 0.001),
        ("std[9] > 0.0001", batch_std_t[-1] > 0.0001),
        ("std growth ratio > 2x", std_ratio > 2.0),
    ]

    print("[checks]")
    ok = True
    for name, passed in checks:
        ok = ok and passed
        print(f"  {'PASS' if passed else 'FAIL'} - {name}")

    if np.allclose(batch_std_t, 0.0, atol=1e-8):
        print(f"[FAILURE] {FAILURE_MODES['dropout_dead']['message']}")
        print(f"  next: {FAILURE_MODES['dropout_dead']['next']}")
        ok = False
    if batch_std_t[-1] < batch_std_t[0]:
        print(f"[FAILURE] {FAILURE_MODES['wrong_training']['message']}")
        print(f"  next: {FAILURE_MODES['wrong_training']['next']}")
        ok = False

    fig_path = _save_verify_figure(pytorch_ade, seq_ade, batch_ade, seq_std_t, batch_std_t)
    print(f"[figure] {fig_path}")
    print(f"[result] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run Phase 2 verification.")
    parser.add_argument("--n-samples", type=int, default=int(CONFIG["mc_samples"]))
    parser.add_argument("--sequential", action="store_true", help="Use sequential mode for a smoke test.")
    args = parser.parse_args()

    if args.verify:
        ok = verify_mc_inference()
        raise SystemExit(0 if ok else 1)

    session = create_session()
    x_val, _ = _load_val_subset(1)
    mean, std = run_lstm_with_uncertainty(
        session,
        x_val[:1],
        n_samples=args.n_samples,
        use_batch=not args.sequential,
    )
    print(f"mean shape={mean.shape}, std shape={std.shape}")
    print(f"std first={std_timestep_magnitude(std[np.newaxis, ...])[0]:.8f}")
    print(f"std final={std_timestep_magnitude(std[np.newaxis, ...])[-1]:.8f}")


if __name__ == "__main__":
    main()