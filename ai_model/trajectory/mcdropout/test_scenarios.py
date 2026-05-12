"""
Dummy trajectory scenarios for A-PAS MC Dropout uncertainty inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from mc_inference import FIGURE_DIR, create_session, run_lstm_with_uncertainty, std_timestep_magnitude


DT = 1.0 / 30.0


def _features_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """정규화 x/y 궤적에서 17-feature 시퀀스를 만든다."""
    seq = np.zeros((1, 60, 17), dtype=np.float32)
    vx = np.gradient(x, DT)
    vy = np.gradient(y, DT)
    ax = np.gradient(vx, DT)
    ay = np.gradient(vy, DT)
    speed = np.sqrt(vx**2 + vy**2)
    heading = np.arctan2(vy, vx)

    seq[0, :, 0] = x
    seq[0, :, 1] = y
    seq[0, :, 2] = np.clip(vx, -2.0, 2.0)
    seq[0, :, 3] = np.clip(vy, -2.0, 2.0)
    seq[0, :, 4] = np.clip(ax, -1.0, 1.0)
    seq[0, :, 5] = np.clip(ay, -1.0, 1.0)
    seq[0, :, 6] = np.clip(speed / np.sqrt(2.0), 0.0, 2.0)
    seq[0, :, 7] = np.sin(heading)
    seq[0, :, 8] = np.cos(heading)
    seq[0, :, 9] = 0.035
    seq[0, :, 10] = 0.090
    seq[0, :, 11] = seq[0, :, 9] * seq[0, :, 10]
    seq[0, :, 12] = 1.0
    return seq


def make_scenario_a_straight() -> np.ndarray:
    """A: 직진 보행자. 예상: std 작음."""
    x = np.linspace(0.2, 0.25, 60, dtype=np.float32)
    y = np.full(60, 0.5, dtype=np.float32)
    return _features_from_xy(x, y)


def make_scenario_b_stopped() -> np.ndarray:
    """B: 멈춰선 보행자. 예상: std 중간."""
    x = np.full(60, 0.45, dtype=np.float32)
    y = np.full(60, 0.1, dtype=np.float32)
    return _features_from_xy(x, y)


def make_scenario_c_sudden_turn() -> np.ndarray:
    """C: 갑자기 방향 바꾼 보행자. 예상: std 큼."""
    x1 = np.linspace(0.2, 0.38, 30, dtype=np.float32)
    y1 = np.full(30, 0.5, dtype=np.float32)
    x2 = np.full(30, 0.38, dtype=np.float32)
    y2 = np.linspace(0.5, 0.72, 30, dtype=np.float32)
    return _features_from_xy(np.concatenate([x1, x2]), np.concatenate([y1, y2]))


def _plot_results(results: list[Tuple[str, np.ndarray]]) -> Path:
    """시나리오별 std 그래프를 저장한다."""
    import matplotlib.pyplot as plt

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURE_DIR / "scenarios_uncertainty.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, std_t in results:
        ax.plot(std_t, label=label)
    ax.set_xlabel("Prediction timestep")
    ax.set_ylabel("Normalized std magnitude")
    ax.set_title("Scenario uncertainty profiles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    """세 가지 더미 시나리오의 불확실성 패턴을 검증한다."""
    session = create_session()
    scenarios = [
        ("A straight", make_scenario_a_straight()),
        ("B stopped", make_scenario_b_stopped()),
        ("C sudden_turn", make_scenario_c_sudden_turn()),
    ]

    results = []
    means = []
    for label, seq in scenarios:
        _, std = run_lstm_with_uncertainty(session, seq, n_samples=30, use_batch=True)
        std_t = std_timestep_magnitude(std[np.newaxis, ...])
        results.append((label, std_t))
        means.append(float(std_t.mean()))
        print(f"{label:14s} mean_std={means[-1]:.8f} final_std={std_t[-1]:.8f}")

    ordered = means[0] < means[1] < means[2]
    print(f"[check] A < B < C mean std: {'PASS' if ordered else 'FAIL'}")
    if not ordered:
        print("# TODO: 확인 필요 - 더미 feature 분포가 학습 데이터와 달라 직관 순서가 깨질 수 있음.")

    fig_path = _plot_results(results)
    print(f"[figure] {fig_path}")


if __name__ == "__main__":
    main()
