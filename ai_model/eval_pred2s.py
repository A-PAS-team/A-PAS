"""
eval_pred2s.py — A-PAS 2초 예측 모델 평가 스크립트
====================================================
대상: Residual LSTM, Attention+Residual LSTM
스펙: 30FPS, SEQ=30(1초 관찰), PRED=60(2초 예측)

사용법:
  # Residual 단독 평가
  python eval_pred2s.py --model residual

  # Attention+Residual 단독 평가
  python eval_pred2s.py --model attention_residual

  # 두 모델 비교 평가
  python eval_pred2s.py --model compare

출력:
  models/<모델>/eval_pred2s/  아래에 그래프 + 결과 텍스트 저장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ==========================================
# ⚙️ [인자 파싱]
# ==========================================
parser = argparse.ArgumentParser(description="A-PAS 2초 예측 모델 평가")
parser.add_argument('--model', type=str, default="compare",
                    choices=["residual", "attention_residual", "compare"],
                    help="평가할 모델 (compare=두 모델 비교)")
parser.add_argument('--num_viz', type=int, default=16,
                    help="궤적 시각화 샘플 수")
args = parser.parse_args()

# ==========================================
# ⚙️ [설정]
# ==========================================
IMG_W, IMG_H    = 1920, 1080
INPUT_SIZE      = 17
HIDDEN_SIZE     = 256
NUM_LAYERS      = 2
SEQ_LENGTH      = 30       # 1초 관찰 @ 30FPS
PRED_LENGTH     = 60       # 2초 예측 @ 30FPS
DT              = 1 / 30
BATCH_SIZE      = 64

CLASS_NAMES  = ["Person", "Car", "Bus", "Truck", "Motorcycle"]
CLASS_COLORS = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "Training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")
print(f"📐 SEQ={SEQ_LENGTH}({SEQ_LENGTH/30:.1f}s) → PRED={PRED_LENGTH}({PRED_LENGTH/30:.1f}s)")

# ==========================================
# 🧠 [모델 정의] — 학습 코드와 100% 동일해야 함
# ==========================================

class ResidualLSTM(nn.Module):
    """tr_residual_30fps_pred2s.py와 동일"""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        offset   = self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)
        last_pos = x[:, -1, :2].unsqueeze(1)   # (B, 1, 2)
        return last_pos + offset


class AttentionResidualLSTM(nn.Module):
    """
    tr_attention_residual_30fps_pred2s.py와 동일
    Input(17) → Proj(256) → LSTM → Residual+LN → Attention → Linear
    """
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.lstm = nn.LSTM(
            HIDDEN_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)

        # Scaled Dot-Product Attention
        self.W_q  = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_k  = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.W_v  = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.scale = math.sqrt(HIDDEN_SIZE)

        # 단순 Linear (좌표 회귀 최적)
        self.fc = nn.Linear(HIDDEN_SIZE * 2, PRED_LENGTH * 2)

    def forward(self, x):
        projected = self.input_proj(x)
        lstm_out, (h_n, _) = self.lstm(projected)
        residual_out = self.layer_norm(lstm_out + projected)

        q = self.W_q(h_n[-1]).unsqueeze(1)
        k = self.W_k(residual_out)
        v = self.W_v(residual_out)

        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        ctx  = torch.bmm(attn, v).squeeze(1)

        pred = self.fc(torch.cat([h_n[-1], ctx], dim=-1))
        return pred.view(-1, PRED_LENGTH, 2)


# ==========================================
# 📦 [모델 로드 유틸]
# ==========================================
MODEL_INFO = {
    "residual": {
        "class":    ResidualLSTM,
        "dir":      "Residual",
        "pth":      "best_residual_30fps_pred2s.pth",
        "label":    "Residual LSTM",
        "color":    "#2196F3",
    },
    "attention_residual": {
        "class":    AttentionResidualLSTM,
        "dir":      "Attention_Residual",
        "pth":      "best_attn_res_30fps_pred2s.pth",
        "label":    "Attention+Residual LSTM",
        "color":    "#FF5722",
    },
}


def load_model(model_key):
    """모델 로드 + 파라미터 수 출력"""
    info = MODEL_INFO[model_key]
    model_path = os.path.join(
        BASE_DIR, "..", "models", info["dir"], info["pth"]
    )

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        print(f"   학습 먼저 실행하세요:")
        print(f"   python tr_{model_key}_30fps_pred2s.py")
        return None

    model = info["class"]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✅ {info['label']} 로드 완료")
    print(f"     경로: {model_path}")
    print(f"     파라미터: {total_params:,} (학습: {trainable:,})")

    return model


def load_data():
    """검증 데이터 로드"""
    print("\n📂 데이터 로드 중...")
    X_path = os.path.join(DATA_DIR, "X_val_30fps_pred2s.npy")
    y_path = os.path.join(DATA_DIR, "y_val_30fps_pred2s.npy")

    if not os.path.exists(X_path):
        print(f"❌ 데이터 없음: {X_path}")
        print(f"   csv_to_npy_30fps_pred2s.py 먼저 실행하세요.")
        return None, None

    X_val = np.load(X_path)
    y_val = np.load(y_path)
    print(f"  ✅ X_val: {X_val.shape}")
    print(f"  ✅ y_val: {y_val.shape}")
    return X_val, y_val


# ==========================================
# 📊 [평가 함수들]
# ==========================================
def evaluate_model(model, X_val, y_val):
    """
    모델 평가 — ADE, FDE, 시점별 오차, 클래스별 오차 등 산출
    반환: dict of 모든 평가 결과
    """
    model.eval()
    all_preds = []
    all_gts   = []

    # 배치 추론
    with torch.no_grad():
        for i in range(0, len(X_val), BATCH_SIZE):
            X_batch = torch.FloatTensor(X_val[i:i+BATCH_SIZE]).to(device)
            y_batch = y_val[i:i+BATCH_SIZE]

            out = model(X_batch)
            # tuple 반환 대응 (attention 모델이 attn_weights 같이 반환하는 경우)
            if isinstance(out, tuple):
                out = out[0]

            all_preds.append(out.cpu().numpy())
            all_gts.append(y_batch)

    preds = np.concatenate(all_preds, axis=0)  # (N, 60, 2) 정규화 좌표
    gts   = np.concatenate(all_gts,   axis=0)  # (N, 60, 2) 정규화 좌표

    # 픽셀 단위 변환
    preds_px = preds.copy()
    gts_px   = gts.copy()
    preds_px[..., 0] *= IMG_W;  preds_px[..., 1] *= IMG_H
    gts_px[..., 0]   *= IMG_W;  gts_px[..., 1]   *= IMG_H

    # --- 기본 지표 ---
    dists = np.sqrt(np.sum((preds_px - gts_px) ** 2, axis=-1))  # (N, 60)

    ade_per_sample = dists.mean(axis=1)      # (N,)
    fde_per_sample = dists[:, -1]            # (N,)

    ade = ade_per_sample.mean()
    fde = fde_per_sample.mean()
    ade_std = ade_per_sample.std()
    fde_std = fde_per_sample.std()

    # --- 시점별 오차 (timestep-wise) ---
    timestep_ade = dists.mean(axis=0)  # (60,)

    # --- 구간별 요약 (0.5초, 1.0초, 1.5초, 2.0초) ---
    interval_summary = {}
    for sec, frame_end in [(0.5, 15), (1.0, 30), (1.5, 45), (2.0, 60)]:
        interval_ade = dists[:, :frame_end].mean()
        interval_fde = dists[:, frame_end - 1].mean()
        interval_summary[sec] = {"ade": interval_ade, "fde": interval_fde}

    # --- 백분위수 ---
    p50_ade = np.percentile(ade_per_sample, 50)
    p90_ade = np.percentile(ade_per_sample, 90)
    p95_ade = np.percentile(ade_per_sample, 95)
    p99_ade = np.percentile(ade_per_sample, 99)

    # --- Miss Rate (FDE 기준) ---
    miss_10 = (fde_per_sample > 10).mean() * 100
    miss_20 = (fde_per_sample > 20).mean() * 100
    miss_30 = (fde_per_sample > 30).mean() * 100
    miss_50 = (fde_per_sample > 50).mean() * 100

    # --- 클래스별 분석 ---
    class_results = {}
    for ci, cname in enumerate(CLASS_NAMES):
        mask = X_val[:, 0, 12 + ci] == 1.0
        if mask.sum() == 0:
            continue
        c_ade = ade_per_sample[mask].mean()
        c_fde = fde_per_sample[mask].mean()
        c_count = mask.sum()
        class_results[cname] = {
            "ade": c_ade, "fde": c_fde, "count": int(c_count)
        }

    # --- Residual offset 분석 (Residual 계열 전용) ---
    offsets = preds - gts  # 정규화 좌표 단위 오차
    offset_stats = {
        "mean_dx": offsets[..., 0].mean(),
        "mean_dy": offsets[..., 1].mean(),
        "std_dx":  offsets[..., 0].std(),
        "std_dy":  offsets[..., 1].std(),
    }

    return {
        "ade": ade, "fde": fde,
        "ade_std": ade_std, "fde_std": fde_std,
        "ade_per_sample": ade_per_sample,
        "fde_per_sample": fde_per_sample,
        "timestep_ade": timestep_ade,
        "interval_summary": interval_summary,
        "percentiles": {"p50": p50_ade, "p90": p90_ade, "p95": p95_ade, "p99": p99_ade},
        "miss_rate": {"10px": miss_10, "20px": miss_20, "30px": miss_30, "50px": miss_50},
        "class_results": class_results,
        "offset_stats": offset_stats,
        "preds_px": preds_px,
        "gts_px": gts_px,
        "dists": dists,
        "n_samples": len(X_val),
    }


# ==========================================
# 📈 [시각화]
# ==========================================
def plot_single_model(results, model_key, output_dir):
    """단일 모델 평가 결과 시각화"""
    info  = MODEL_INFO[model_key]
    label = info["label"]
    color = info["color"]

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 시점별 ADE ---
    fig, ax = plt.subplots(figsize=(12, 5))
    timesteps_sec = np.arange(1, PRED_LENGTH + 1) / 30
    ax.plot(timesteps_sec, results["timestep_ade"], color=color, linewidth=2)

    for sec in [0.5, 1.0, 1.5, 2.0]:
        frame = int(sec * 30)
        if frame <= PRED_LENGTH:
            val = results["timestep_ade"][frame - 1]
            ax.axvline(x=sec, color='gray', linestyle='--', alpha=0.3)
            ax.annotate(f'{val:.1f}px', (sec, val),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=9, color=color, fontweight='bold')

    ax.set_xlabel("Prediction Time (sec)", fontsize=12)
    ax.set_ylabel("ADE (px)", fontsize=12)
    ax.set_title(f"{label} — Timestep Prediction Error (2s Pred)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend([label], fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "timestep_ade.png"), dpi=150)
    plt.close()

    # --- 2. ADE/FDE 분포 (히스토그램) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{label} — Error Distribution (2s Prediction)", fontsize=14)

    axes[0].hist(results["ade_per_sample"], bins=50, color=color, alpha=0.7, edgecolor='white')
    axes[0].axvline(results["ade"], color='red', linestyle='--', label=f'Mean: {results["ade"]:.2f}px')
    axes[0].axvline(results["percentiles"]["p95"], color='orange', linestyle='--',
                    label=f'P95: {results["percentiles"]["p95"]:.2f}px')
    axes[0].set_xlabel("ADE (px)"); axes[0].set_ylabel("Count")
    axes[0].set_title("ADE Distribution"); axes[0].legend()

    axes[1].hist(results["fde_per_sample"], bins=50, color=color, alpha=0.7, edgecolor='white')
    axes[1].axvline(results["fde"], color='red', linestyle='--', label=f'Mean: {results["fde"]:.2f}px')
    axes[1].set_xlabel("FDE (px)"); axes[1].set_ylabel("Count")
    axes[1].set_title("FDE Distribution (Final Point at 2s)"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_distribution.png"), dpi=150)
    plt.close()

    # --- 3. Performance by Class ---
    if results["class_results"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{label} — Performance by Class", fontsize=14)

        names, ades, fdes, counts = [], [], [], []
        colors = []
        for ci, cname in enumerate(CLASS_NAMES):
            if cname in results["class_results"]:
                names.append(cname)
                ades.append(results["class_results"][cname]["ade"])
                fdes.append(results["class_results"][cname]["fde"])
                counts.append(results["class_results"][cname]["count"])
                colors.append(CLASS_COLORS[ci])

        x = np.arange(len(names))
        axes[0].bar(x, ades, color=colors, alpha=0.8)
        axes[0].set_xticks(x); axes[0].set_xticklabels(names)
        axes[0].set_ylabel("ADE (px)"); axes[0].set_title("ADE by Class")
        for i, v in enumerate(ades):
            axes[0].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9)

        axes[1].bar(x, fdes, color=colors, alpha=0.8)
        axes[1].set_xticks(x); axes[1].set_xticklabels(names)
        axes[1].set_ylabel("FDE (px)"); axes[1].set_title("FDE by Class (2s)")
        for i, v in enumerate(fdes):
            axes[1].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "class_performance.png"), dpi=150)
        plt.close()

    # --- 4. CDF ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_ade = np.sort(results["ade_per_sample"])
    cdf = np.arange(1, len(sorted_ade) + 1) / len(sorted_ade)
    ax.plot(sorted_ade, cdf, color=color, linewidth=2, label=f'ADE (mean={results["ade"]:.2f}px)')

    sorted_fde = np.sort(results["fde_per_sample"])
    cdf_fde = np.arange(1, len(sorted_fde) + 1) / len(sorted_fde)
    ax.plot(sorted_fde, cdf_fde, color=color, linewidth=2, linestyle='--',
            label=f'FDE (mean={results["fde"]:.2f}px)')

    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5, label='95th percentile')
    ax.set_xlabel("Error (px)", fontsize=12)
    ax.set_ylabel("Cumulative Ratio", fontsize=12)
    ax.set_title(f"{label} — Error CDF (2s Prediction)", fontsize=14)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cdf.png"), dpi=150)
    plt.close()

    print(f"  📊 그래프 저장: {output_dir}")


def plot_comparison(res_residual, res_attn_res, output_dir):
    """두 모델 비교 시각화"""
    os.makedirs(output_dir, exist_ok=True)

    r_info  = MODEL_INFO["residual"]
    ar_info = MODEL_INFO["attention_residual"]

    # --- 1. 시점별 ADE 비교 ---
    fig, ax = plt.subplots(figsize=(12, 6))
    timesteps_sec = np.arange(1, PRED_LENGTH + 1) / 30

    ax.plot(timesteps_sec, res_residual["timestep_ade"],
            color=r_info["color"], linewidth=2.5, label=r_info["label"])
    ax.plot(timesteps_sec, res_attn_res["timestep_ade"],
            color=ar_info["color"], linewidth=2.5, label=ar_info["label"])

    for sec in [0.5, 1.0, 1.5, 2.0]:
        ax.axvline(x=sec, color='gray', linestyle='--', alpha=0.2)

    ax.set_xlabel("Prediction Time (sec)", fontsize=12)
    ax.set_ylabel("ADE (px)", fontsize=12)
    ax.set_title("Residual vs Attention+Residual — Timestep Prediction Error", fontsize=14)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_timestep_ade.png"), dpi=150)
    plt.close()

    # --- 2. 구간별 비교 바 차트 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ADE / FDE by Time Interval", fontsize=14)

    intervals = [0.5, 1.0, 1.5, 2.0]
    r_ades  = [res_residual["interval_summary"][s]["ade"]  for s in intervals]
    ar_ades = [res_attn_res["interval_summary"][s]["ade"] for s in intervals]
    r_fdes  = [res_residual["interval_summary"][s]["fde"]  for s in intervals]
    ar_fdes = [res_attn_res["interval_summary"][s]["fde"] for s in intervals]

    x = np.arange(len(intervals))
    w = 0.35

    axes[0].bar(x - w/2, r_ades,  w, color=r_info["color"],  alpha=0.8, label=r_info["label"])
    axes[0].bar(x + w/2, ar_ades, w, color=ar_info["color"], alpha=0.8, label=ar_info["label"])
    axes[0].set_xticks(x); axes[0].set_xticklabels([f'{s}s' for s in intervals])
    axes[0].set_ylabel("ADE (px)"); axes[0].set_title("ADE by Interval")
    axes[0].legend()
    for i in range(len(intervals)):
        axes[0].text(i - w/2, r_ades[i] + 0.3,  f'{r_ades[i]:.1f}',  ha='center', fontsize=8)
        axes[0].text(i + w/2, ar_ades[i] + 0.3, f'{ar_ades[i]:.1f}', ha='center', fontsize=8)

    axes[1].bar(x - w/2, r_fdes,  w, color=r_info["color"],  alpha=0.8, label=r_info["label"])
    axes[1].bar(x + w/2, ar_fdes, w, color=ar_info["color"], alpha=0.8, label=ar_info["label"])
    axes[1].set_xticks(x); axes[1].set_xticklabels([f'{s}s' for s in intervals])
    axes[1].set_ylabel("FDE (px)"); axes[1].set_title("FDE by Interval")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_interval.png"), dpi=150)
    plt.close()

    # --- 3. CDF 비교 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error CDF Comparison", fontsize=14)

    for res, info in [(res_residual, r_info), (res_attn_res, ar_info)]:
        sorted_ade = np.sort(res["ade_per_sample"])
        cdf = np.arange(1, len(sorted_ade) + 1) / len(sorted_ade)
        axes[0].plot(sorted_ade, cdf, color=info["color"], linewidth=2,
                     label=f'{info["label"]} (mean={res["ade"]:.2f}px)')

    axes[0].axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel("ADE (px)"); axes[0].set_ylabel("Cumulative Ratio")
    axes[0].set_title("ADE CDF"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    for res, info in [(res_residual, r_info), (res_attn_res, ar_info)]:
        sorted_fde = np.sort(res["fde_per_sample"])
        cdf = np.arange(1, len(sorted_fde) + 1) / len(sorted_fde)
        axes[1].plot(sorted_fde, cdf, color=info["color"], linewidth=2,
                     label=f'{info["label"]} (mean={res["fde"]:.2f}px)')

    axes[1].axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_xlabel("FDE (px)"); axes[1].set_ylabel("Cumulative Ratio")
    axes[1].set_title("FDE CDF (at 2s)"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_cdf.png"), dpi=150)
    plt.close()

    # --- 4. 클래스별 비교 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance Comparison by Class", fontsize=14)

    common_classes = [c for c in CLASS_NAMES
                      if c in res_residual["class_results"]
                      and c in res_attn_res["class_results"]]
    if common_classes:
        x = np.arange(len(common_classes))
        w = 0.35

        r_c_ade  = [res_residual["class_results"][c]["ade"]  for c in common_classes]
        ar_c_ade = [res_attn_res["class_results"][c]["ade"] for c in common_classes]
        r_c_fde  = [res_residual["class_results"][c]["fde"]  for c in common_classes]
        ar_c_fde = [res_attn_res["class_results"][c]["fde"] for c in common_classes]

        axes[0].bar(x - w/2, r_c_ade,  w, color=r_info["color"],  alpha=0.8, label=r_info["label"])
        axes[0].bar(x + w/2, ar_c_ade, w, color=ar_info["color"], alpha=0.8, label=ar_info["label"])
        axes[0].set_xticks(x); axes[0].set_xticklabels(common_classes)
        axes[0].set_ylabel("ADE (px)"); axes[0].set_title("ADE by Class"); axes[0].legend(fontsize=9)

        axes[1].bar(x - w/2, r_c_fde,  w, color=r_info["color"],  alpha=0.8, label=r_info["label"])
        axes[1].bar(x + w/2, ar_c_fde, w, color=ar_info["color"], alpha=0.8, label=ar_info["label"])
        axes[1].set_xticks(x); axes[1].set_xticklabels(common_classes)
        axes[1].set_ylabel("FDE (px)"); axes[1].set_title("FDE by Class"); axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "compare_class.png"), dpi=150)
    plt.close()

    print(f"  📊 비교 그래프 저장: {output_dir}")


def plot_trajectories(model, model_key, X_val, y_val, results, output_dir, num_viz=16):
    """궤적 시각화 — 관찰/GT/예측 경로"""
    os.makedirs(output_dir, exist_ok=True)
    info = MODEL_INFO[model_key]

    ade_per_sample = results["ade_per_sample"]

    # Best/Worst/Random 선정
    best_idx  = np.argsort(ade_per_sample)[:num_viz // 4]
    worst_idx = np.argsort(ade_per_sample)[-num_viz // 4:]
    remain    = num_viz - len(best_idx) - len(worst_idx)
    pool      = list(set(range(len(X_val))) - set(best_idx) - set(worst_idx))
    random_idx = np.array(random.sample(pool, min(remain, len(pool))))

    selected = np.concatenate([best_idx, random_idx, worst_idx])
    labels   = (['BEST'] * len(best_idx) +
                ['RANDOM'] * len(random_idx) +
                ['WORST'] * len(worst_idx))

    cols = 4
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(f"{info['label']} — Trajectory Visualization (2s Pred)", fontsize=16, y=1.02)

    if rows == 1:
        axes = axes.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        for plot_i, (idx, lbl) in enumerate(zip(selected, labels)):
            r, c = plot_i // cols, plot_i % cols
            ax = axes[r, c]

            x_seq = X_val[idx]      # (30, 17)
            y_gt  = y_val[idx]      # (60, 2) 정규화

            # 관찰 궤적 (정규화 → 픽셀)
            obs_x = x_seq[:, 0] * IMG_W
            obs_y = x_seq[:, 1] * IMG_H

            # GT 궤적
            gt_x = y_gt[:, 0] * IMG_W
            gt_y = y_gt[:, 1] * IMG_H

            # 예측 궤적
            inp = torch.FloatTensor(x_seq).unsqueeze(0).to(device)
            out = model(inp)
            if isinstance(out, tuple):
                out = out[0]
            pred = out[0].cpu().numpy()
            pred_x = pred[:, 0] * IMG_W
            pred_y = pred[:, 1] * IMG_H

            ade_i = ade_per_sample[idx]

            # 클래스 확인
            cls_vec = x_seq[0, 12:]
            cls_idx = np.argmax(cls_vec)
            cls_name = CLASS_NAMES[cls_idx] if cls_vec[cls_idx] > 0 else "Unknown"

            # 그리기
            ax.plot(obs_x, obs_y, 'o-', color='#888888', markersize=2, linewidth=1,
                    label='Observed (1s)', alpha=0.6)
            ax.plot(gt_x, gt_y, 's-', color='#2ECC71', markersize=2, linewidth=1.5,
                    label='GT (2s)')
            ax.plot(pred_x, pred_y, '^-', color=info["color"], markersize=2, linewidth=1.5,
                    label='Predicted (2s)')

            # 시작/끝 강조
            ax.scatter(obs_x[0], obs_y[0], color='black', s=50, zorder=5, marker='D')
            ax.scatter(gt_x[-1], gt_y[-1], color='#2ECC71', s=60, zorder=5, marker='*')
            ax.scatter(pred_x[-1], pred_y[-1], color=info["color"], s=60, zorder=5, marker='*')

            ax.set_title(f"[{lbl}] {cls_name} | ADE={ade_i:.1f}px", fontsize=10)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.2)

    # 빈 subplot 제거
    for i in range(len(selected), rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectories.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🛤️  궤적 시각화 저장: {output_dir}/trajectories.png")


# ==========================================
# 📝 [결과 출력]
# ==========================================
def print_results(results, model_key):
    """콘솔에 결과 출력"""
    info = MODEL_INFO[model_key]
    r = results

    print(f"\n{'='*60}")
    print(f"  📊 {info['label']} — 2초 예측 평가 결과")
    print(f"{'='*60}")
    print(f"  샘플 수        : {r['n_samples']:,}")
    print(f"  ADE            : {r['ade']:.2f} ± {r['ade_std']:.2f} px")
    print(f"  FDE (2초)      : {r['fde']:.2f} ± {r['fde_std']:.2f} px")

    print(f"\n  --- 구간별 성능 ---")
    for sec in [0.5, 1.0, 1.5, 2.0]:
        s = r["interval_summary"][sec]
        print(f"  {sec:.1f}초 | ADE: {s['ade']:6.2f}px | FDE: {s['fde']:6.2f}px")

    print(f"\n  --- 백분위수 (ADE) ---")
    p = r["percentiles"]
    print(f"  P50: {p['p50']:.2f}px | P90: {p['p90']:.2f}px | "
          f"P95: {p['p95']:.2f}px | P99: {p['p99']:.2f}px")

    print(f"\n  --- Miss Rate (FDE 기준) ---")
    m = r["miss_rate"]
    print(f"  >10px: {m['10px']:.1f}% | >20px: {m['20px']:.1f}% | "
          f">30px: {m['30px']:.1f}% | >50px: {m['50px']:.1f}%")

    if r["class_results"]:
        print(f"\n  --- 클래스별 ---")
        for cname, cr in r["class_results"].items():
            print(f"  {cname:12s} | ADE: {cr['ade']:6.2f}px | "
                  f"FDE: {cr['fde']:6.2f}px | N={cr['count']:,}")

    print(f"\n  --- Offset 분포 ---")
    o = r["offset_stats"]
    print(f"  mean(Δx): {o['mean_dx']:.6f} | std(Δx): {o['std_dx']:.6f}")
    print(f"  mean(Δy): {o['mean_dy']:.6f} | std(Δy): {o['std_dy']:.6f}")
    print(f"{'='*60}")


def save_results_txt(results, model_key, output_dir):
    """결과를 텍스트 파일로 저장"""
    info = MODEL_INFO[model_key]
    r = results
    path = os.path.join(output_dir, "eval_results.txt")

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"A-PAS 2초 예측 모델 평가 결과\n")
        f.write(f"모델: {info['label']}\n")
        f.write(f"SEQ={SEQ_LENGTH} (1s) → PRED={PRED_LENGTH} (2s) @ 30FPS\n")
        f.write(f"{'='*50}\n\n")

        f.write(f"샘플 수 : {r['n_samples']:,}\n")
        f.write(f"ADE     : {r['ade']:.2f} ± {r['ade_std']:.2f} px\n")
        f.write(f"FDE(2s) : {r['fde']:.2f} ± {r['fde_std']:.2f} px\n\n")

        f.write(f"--- 구간별 ---\n")
        for sec in [0.5, 1.0, 1.5, 2.0]:
            s = r["interval_summary"][sec]
            f.write(f"  {sec:.1f}s | ADE: {s['ade']:.2f}px | FDE: {s['fde']:.2f}px\n")

        f.write(f"\n--- 백분위수 (ADE) ---\n")
        p = r["percentiles"]
        f.write(f"  P50: {p['p50']:.2f} | P90: {p['p90']:.2f} | "
                f"P95: {p['p95']:.2f} | P99: {p['p99']:.2f}\n")

        f.write(f"\n--- Miss Rate (FDE) ---\n")
        m = r["miss_rate"]
        f.write(f"  >10px: {m['10px']:.1f}% | >20px: {m['20px']:.1f}% | "
                f">30px: {m['30px']:.1f}% | >50px: {m['50px']:.1f}%\n")

        if r["class_results"]:
            f.write(f"\n--- 클래스별 ---\n")
            for cname, cr in r["class_results"].items():
                f.write(f"  {cname:12s} | ADE: {cr['ade']:.2f} | "
                        f"FDE: {cr['fde']:.2f} | N={cr['count']}\n")

    print(f"  💾 결과 저장: {path}")


# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    # 데이터 로드
    X_val, y_val = load_data()
    if X_val is None:
        return

    models_to_eval = []
    if args.model == "compare":
        models_to_eval = ["residual", "attention_residual"]
    else:
        models_to_eval = [args.model]

    all_results = {}

    for mkey in models_to_eval:
        print(f"\n{'─'*50}")
        print(f"📦 {MODEL_INFO[mkey]['label']} 평가 시작")
        print(f"{'─'*50}")

        model = load_model(mkey)
        if model is None:
            continue

        # 평가
        results = evaluate_model(model, X_val, y_val)
        all_results[mkey] = results

        # 결과 출력
        print_results(results, mkey)

        # 단일 모델 시각화
        out_dir = os.path.join(
            BASE_DIR, "..", "models", MODEL_INFO[mkey]["dir"], "eval_pred2s"
        )
        plot_single_model(results, mkey, out_dir)

        # 궤적 시각화
        plot_trajectories(model, mkey, X_val, y_val, results, out_dir, args.num_viz)

        # 결과 텍스트 저장
        save_results_txt(results, mkey, out_dir)

    # 비교 모드
    if args.model == "compare" and len(all_results) == 2:
        print(f"\n{'─'*50}")
        print(f"📊 모델 비교 분석")
        print(f"{'─'*50}")

        compare_dir = os.path.join(BASE_DIR, "..", "models", "compare_pred2s")
        plot_comparison(
            all_results["residual"],
            all_results["attention_residual"],
            compare_dir
        )

        # 비교 요약
        r  = all_results["residual"]
        ar = all_results["attention_residual"]

        print(f"\n{'='*60}")
        print(f"  📊 비교 요약 (2초 예측)")
        print(f"{'='*60}")
        print(f"  {'지표':12s} | {'Residual':>12s} | {'Attn+Res':>12s} | {'차이':>10s}")
        print(f"  {'-'*52}")

        ade_diff = ar["ade"] - r["ade"]
        fde_diff = ar["fde"] - r["fde"]

        print(f"  {'ADE':12s} | {r['ade']:10.2f}px | {ar['ade']:10.2f}px | "
              f"{ade_diff:+8.2f}px")
        print(f"  {'FDE (2s)':12s} | {r['fde']:10.2f}px | {ar['fde']:10.2f}px | "
              f"{fde_diff:+8.2f}px")

        for sec in [0.5, 1.0, 1.5, 2.0]:
            r_a  = r["interval_summary"][sec]["ade"]
            ar_a = ar["interval_summary"][sec]["ade"]
            diff = ar_a - r_a
            print(f"  {'ADE@'+str(sec)+'s':12s} | {r_a:10.2f}px | {ar_a:10.2f}px | "
                  f"{diff:+8.2f}px")

        winner = "Residual" if r["ade"] < ar["ade"] else "Attention+Residual"
        print(f"\n  🏆 ADE 기준 우승: {winner}")
        print(f"{'='*60}")

        # 비교 결과 텍스트 저장
        with open(os.path.join(compare_dir, "comparison_summary.txt"), 'w',
                  encoding='utf-8') as f:
            f.write("A-PAS 2초 예측 모델 비교\n")
            f.write(f"SEQ={SEQ_LENGTH}(1s) → PRED={PRED_LENGTH}(2s) @ 30FPS\n\n")
            f.write(f"{'지표':12s} | {'Residual':>10s} | {'Attn+Res':>10s} | {'차이':>8s}\n")
            f.write(f"{'-'*48}\n")
            f.write(f"{'ADE':12s} | {r['ade']:8.2f}px | {ar['ade']:8.2f}px | {ade_diff:+6.2f}px\n")
            f.write(f"{'FDE(2s)':12s} | {r['fde']:8.2f}px | {ar['fde']:8.2f}px | {fde_diff:+6.2f}px\n")
            for sec in [0.5, 1.0, 1.5, 2.0]:
                r_a  = r["interval_summary"][sec]["ade"]
                ar_a = ar["interval_summary"][sec]["ade"]
                f.write(f"{'ADE@'+str(sec)+'s':12s} | {r_a:8.2f}px | {ar_a:8.2f}px | {ar_a-r_a:+6.2f}px\n")
            f.write(f"\nWinner (ADE): {winner}\n")

        print(f"\n  💾 비교 결과: {compare_dir}/comparison_summary.txt")

    print(f"\n🎉 평가 완료!")


if __name__ == "__main__":
    main()