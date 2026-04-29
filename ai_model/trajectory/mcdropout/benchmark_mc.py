"""
Benchmark MC sample count for A-PAS ONNX MC Dropout inference.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from mc_inference import (
    FAILURE_MODES,
    FIGURE_DIR,
    _load_val_subset,
    create_session,
    run_lstm_with_uncertainty,
    std_timestep_magnitude,
)


N_VALUES = [3, 5, 10, 20, 30, 50]
LATENCY_REPEATS = 100
QUALITY_REPEATS = 100


def _measure_latency(session, x: np.ndarray, n_samples: int, use_batch: bool) -> tuple[float, float]:
    """100회 반복 평균/표준편차 latency(ms)를 측정한다."""
    times = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        run_lstm_with_uncertainty(session, x, n_samples=n_samples, use_batch=use_batch)
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.asarray(times, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def _measure_std_stability(session, x: np.ndarray, n_samples: int, use_batch: bool) -> float:
    """같은 입력의 std 추정값 안정성을 coefficient of variation으로 측정한다."""
    values = []
    for _ in range(QUALITY_REPEATS):
        _, std = run_lstm_with_uncertainty(session, x, n_samples=n_samples, use_batch=use_batch)
        std_t = std_timestep_magnitude(std[np.newaxis, ...])
        values.append(float(std_t[-1]))
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.std() / max(arr.mean(), 1e-12))


def _plot(rows: List[Dict[str, float]]) -> Path:
    """벤치마크 결과 PNG를 저장한다."""
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "benchmark_mc.png"
    n = [r["n"] for r in rows]
    seq_lat = [r["seq_ms"] for r in rows]
    bat_lat = [r["bat_ms"] for r in rows]
    seq_cv = [r["seq_cv"] * 100.0 for r in rows]
    bat_cv = [r["bat_cv"] * 100.0 for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(n, seq_lat, marker="o", label="Sequential")
    axes[0].plot(n, bat_lat, marker="o", label="Batched")
    axes[0].axhline(33.3, color="tab:red", linestyle="--", linewidth=1, label="30FPS budget")
    axes[0].set_xlabel("n_samples")
    axes[0].set_ylabel("Windows latency (ms)")
    axes[0].set_title("Latency")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(n, seq_cv, marker="o", label="Sequential")
    axes[1].plot(n, bat_cv, marker="o", label="Batched")
    axes[1].axhline(5.0, color="tab:red", linestyle="--", linewidth=1, label="5% target")
    axes[1].set_xlabel("n_samples")
    axes[1].set_ylabel("Final std CV (%)")
    axes[1].set_title("Uncertainty stability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(n, [v * 5 for v in bat_lat], marker="o", label="RPi5 x5")
    axes[2].plot(n, [v * 8 for v in bat_lat], marker="o", label="RPi5 x8")
    axes[2].axhline(33.3, color="tab:red", linestyle="--", linewidth=1, label="30FPS budget")
    axes[2].set_xlabel("n_samples")
    axes[2].set_ylabel("Estimated RPi5 batched latency (ms)")
    axes[2].set_title("RPi5 estimate")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def benchmark_n_samples() -> None:
    """n_samples 변화에 따른 latency vs uncertainty quality를 측정한다."""
    session = create_session()
    x_val, _ = _load_val_subset(1)
    x = x_val[:1]

    rows: List[Dict[str, float]] = []
    print("[benchmark] repeats latency=100, quality=100")
    for n in N_VALUES:
        seq_ms, seq_ms_std = _measure_latency(session, x, n, use_batch=False)
        bat_ms, bat_ms_std = _measure_latency(session, x, n, use_batch=True)
        seq_cv = _measure_std_stability(session, x, n, use_batch=False)
        bat_cv = _measure_std_stability(session, x, n, use_batch=True)
        speedup = seq_ms / max(bat_ms, 1e-12)
        row = {
            "n": float(n),
            "seq_ms": seq_ms,
            "seq_ms_std": seq_ms_std,
            "bat_ms": bat_ms,
            "bat_ms_std": bat_ms_std,
            "seq_cv": seq_cv,
            "bat_cv": bat_cv,
            "speedup": speedup,
        }
        rows.append(row)
        print(
            f"n={n:2d} | seq={seq_ms:.3f}±{seq_ms_std:.3f}ms "
            f"| batch={bat_ms:.3f}±{bat_ms_std:.3f}ms "
            f"| speedup={speedup:.2f}x | batch_cv={bat_cv*100:.2f}%"
        )

    candidates = [
        r for r in rows
        if r["bat_ms"] * 8.0 < 33.3 and r["bat_cv"] < 0.05
    ]
    if candidates:
        recommended = int(candidates[0]["n"])
        print(f"[recommend] n_samples={recommended} fits conservative RPi5 x8 30FPS estimate.")
    else:
        stable = [r for r in rows if r["bat_cv"] < 0.05]
        recommended = int(stable[0]["n"]) if stable else 10
        print("[recommend] No n satisfies RPi5 x8 <33.3ms and CV<5%.")
        print(f"[recommend] Use n_samples={recommended} for quality, then measure real RPi5 latency in Phase 3.")

    n10 = next((r for r in rows if int(r["n"]) == 10), None)
    if n10 and n10["bat_ms"] > 50.0:
        print(f"[FAILURE] {FAILURE_MODES['too_slow']['message']}")
        print(f"  next: {FAILURE_MODES['too_slow']['next']}")
    if all(r["speedup"] < 1.1 for r in rows):
        print(f"[FAILURE] {FAILURE_MODES['batch_doesnt_help']['message']}")
        print(f"  next: {FAILURE_MODES['batch_doesnt_help']['next']}")

    print("# TODO: 확인 필요 - RPi5 추정은 Windows latency x5~x8 보수 추정이며 Phase 3 실측으로 보정 필요.")
    fig_path = _plot(rows)
    print(f"[figure] {fig_path}")


if __name__ == "__main__":
    benchmark_n_samples()

