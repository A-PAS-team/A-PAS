"""
A-PAS Residual LSTM + Kinematic Layer + MC Dropout
====================================================
기존 ResidualLSTMMCDropout를 베이스로,
LSTM 출력을 (speed, yaw_rate) → Kinematic Layer → (x, y) 변환.

핵심 변경:
  - LSTM FC 출력: (pred_length * 2) → (speed, yaw_rate) per step
  - Kinematic Layer: 미분 가능한 CTRV 물리 변환
  - 역전파가 Kinematic Layer를 통과 → LSTM이 물리적 의미를 학습

Input : (batch, seq_length, 17)  # 12물리 + yaw_rate + 4원-핫 (crosswalk 제외)
Output: (batch, pred_length, 2)  # 정규화 (x, y) 좌표

참고 논문: X-TRACK (Chugh et al., 2025) - Physics-Aware xLSTM
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict

import torch
import torch.nn as nn


# ==========================================
# ⚙️ [기본 설정]
# ==========================================
CONFIG: Dict[str, Any] = {
    "input_size": 17,       # 12 물리 + yaw_rate + 4 원-핫 (crosswalk 제외)
    "hidden_size": 256,
    "num_layers": 2,
    "pred_length": 20,      # 2초 예측 @ 10FPS
    "dropout_p": 0.2,
    "dt": 0.1,              # 10FPS
    "img_w": 1920,
    "img_h": 1080,
}

_DIAG = math.sqrt(1920**2 + 1080**2)


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 17
    hidden_size: int = 256
    num_layers: int = 2
    pred_length: int = 20
    dropout_p: float = 0.2
    dt: float = 0.1
    img_w: int = 1920
    img_h: int = 1080

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ModelConfig":
        known = {k: v for k, v in cfg.items() if k in cls.__annotations__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==========================================
# 🔧 [MC Dropout] — 기존과 동일
# ==========================================
class MCDropout(nn.Module):
    """ONNX 호환 stochastic dropout."""

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("dropout probability must be in [0, 1).")
        self.p = float(p)
        self.force_mc_dropout = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or (not self.training and not self.force_mc_dropout):
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(dtype=x.dtype)
        return x * mask / keep_prob


# ==========================================
# 🔧 [Kinematic Layer] — 미분 가능한 CTRV 물리 변환
# ==========================================
class KinematicLayer(nn.Module):
    """CTRV(Constant Turn Rate & Velocity) 기반 좌표 변환.

    핵심: 정규화 공간에서 직접 삼각함수를 적용하면 x/y 스케일 비등방성으로
    궤적이 일그러짐 (Gemini review #1).
    → 픽셀 공간으로 복원 → CTRV 물리 연산 → 다시 정규화.

    모든 연산이 torch 텐서 → 역전파 가능.
    sin/cos 연산은 ONNX에서도 지원.

    Args:
        dt: 시간 간격 (초). 10FPS면 0.1
        img_w: 이미지 너비 (픽셀)
        img_h: 이미지 높이 (픽셀)
    """

    def __init__(self, dt: float = 0.1, img_w: int = 1920, img_h: int = 1080) -> None:
        super().__init__()
        self.dt = dt
        self.img_w = float(img_w)
        self.img_h = float(img_h)
        self.diag = math.sqrt(img_w**2 + img_h**2)

    def forward(
        self,
        speed: torch.Tensor,
        yaw_rate: torch.Tensor,
        init_pos: torch.Tensor,
        init_heading: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            speed:        (batch, pred_length) — FC 출력 (softplus 통과, 임의 스케일)
            yaw_rate:     (batch, pred_length) — rad/step
            init_pos:     (batch, 2) — 마지막 관찰 위치 (정규화 x/IMG_W, y/IMG_H)
            init_heading: (batch,) — 마지막 관찰 heading (rad)

        Returns:
            positions: (batch, pred_length, 2) — 예측 좌표 (정규화)
        """
        batch_size, pred_length = speed.shape
        device = speed.device
        dtype = speed.dtype

        positions = torch.zeros(batch_size, pred_length, 2, device=device, dtype=dtype)

        # ✅ 정규화 좌표 → 픽셀 공간으로 복원
        x = init_pos[:, 0] * self.img_w     # (batch,) 픽셀
        y = init_pos[:, 1] * self.img_h     # (batch,) 픽셀
        psi = init_heading                    # (batch,) rad

        # speed는 FC→softplus 출력이라 스케일이 임의.
        # DIAG로 곱해서 픽셀/초 단위로 해석.
        # (학습 중 역전파로 FC가 적절한 스케일을 학습)
        speed_px = speed * self.diag          # (batch, pred) 픽셀/초

        for t in range(pred_length):
            v_t = speed_px[:, t]              # (batch,) 픽셀/초
            decay_factors = torch.tensor(
                [0.97 ** t for t in range(pred_length)],
                device=device,
                dtype=dtype
            )                                   # (pred_length,) 감쇠 계수
            w_t = yaw_rate[:, t] * decay_factors[t]          # (batch,) rad/step

            # 직진 vs 회전 분기 (미분 가능 soft blending)
            w_abs = torch.abs(w_t)
            is_straight = (w_abs < 0.01).float()

            # --- 직진 ---
            dx_straight = v_t * torch.cos(psi) * self.dt
            dy_straight = v_t * torch.sin(psi) * self.dt

            # --- 회전 (CTRV) ---
            w_safe = w_t + is_straight * 1e-4
            radius = v_t / w_safe
            psi_new = psi + w_t * self.dt
            dx_turn = radius * (torch.sin(psi_new) - torch.sin(psi))
            dy_turn = -radius * (torch.cos(psi_new) - torch.cos(psi))

            # soft blending
            dx = is_straight * dx_straight + (1.0 - is_straight) * dx_turn
            dy = is_straight * dy_straight + (1.0 - is_straight) * dy_turn

            x = x + dx  # 픽셀 단위 갱신
            y = y + dy  # 픽셀 단위 갱신
            psi = psi + w_t * self.dt

            # ✅ 픽셀 → 정규화 좌표로 변환하여 저장
            positions[:, t, 0] = x / self.img_w
            positions[:, t, 1] = y / self.img_h

        return positions


# ==========================================
# 🧠 [모델] Residual LSTM + Kinematic Layer
# ==========================================
class ResidualLSTMKinematic(nn.Module):
    """Residual LSTM + Kinematic Layer + MC Dropout.

    구조:
        입력 (batch, seq, 17)
        → LSTM encoder → hidden state
        → MC Dropout
        → FC → (speed, yaw_rate) × pred_length
        → Kinematic Layer → (x, y) × pred_length

    Residual Connection:
        기존처럼 last_pos 기반이 아니라,
        Kinematic Layer가 init_pos에서 물리적으로 외삽.
        → 구조적으로 residual과 동일한 효과.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig.from_dict({**CONFIG, **kwargs})
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_p if config.num_layers > 1 else 0.0,
        )
        self.mc_dropout = MCDropout(p=config.dropout_p)
        self.fc = nn.Linear(config.hidden_size, config.pred_length * 2)

        self.kinematic = KinematicLayer(
            dt=config.dt, img_w=config.img_w, img_h=config.img_h
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, 17) — 입력 시퀀스

        Returns:
            positions: (batch, pred_length, 2) — 예측 좌표 (정규화)

        피처 레이아웃 (17개):
            [0] pos_x, [1] pos_y,
            [2] vx, [3] vy,
            [4] ax, [5] ay,
            [6] speed, [7] sin_h, [8] cos_h,
            [9] w, [10] h, [11] area,
            [12] yaw_rate,
            [13..16] one-hot class (person, car, motorcycle, large_veh)
        """
        batch_size = x.size(0)

        # --- LSTM 인코딩 ---
        _, (h_n, _) = self.lstm(x)
        h_last = self.mc_dropout(h_n[-1])  # (batch, hidden)

        # --- FC → (speed, yaw_rate) ---
        raw = self.fc(h_last)  # (batch, pred_length * 2)
        raw = raw.view(batch_size, self.config.pred_length, 2)

        speed = raw[:, :, 0]      # (batch, pred_length)
        yaw_rate = raw[:, :, 1]   # (batch, pred_length)

        # speed는 양수여야 함 (softplus로 양수 보장)
        speed = torch.nn.functional.softplus(speed)

        # --- 초기 상태 추출 (마지막 관찰 프레임) ---
        init_pos = x[:, -1, :2]                    # (batch, 2)
        init_heading = torch.atan2(
            x[:, -1, 7],   # sin_h
            x[:, -1, 8],   # cos_h
        )                                           # (batch,)

        # --- Kinematic Layer ---
        positions = self.kinematic(
            speed, yaw_rate, init_pos, init_heading
        )

        return positions

    def forward_with_raw(self, x: torch.Tensor):
        """학습 시 Physics Loss 계산용. positions + (speed, yaw_rate) 반환."""
        batch_size = x.size(0)

        _, (h_n, _) = self.lstm(x)
        h_last = self.mc_dropout(h_n[-1])

        raw = self.fc(h_last).view(batch_size, self.config.pred_length, 2)
        speed = torch.nn.functional.softplus(raw[:, :, 0])
        yaw_rate = raw[:, :, 1]

        init_pos = x[:, -1, :2]
        init_heading = torch.atan2(x[:, -1, 7], x[:, -1, 8])

        positions = self.kinematic(speed, yaw_rate, init_pos, init_heading)

        return positions, speed, yaw_rate


# ==========================================
# 📉 [Physics-Informed Loss]
# ==========================================
class PhysicsLoss(nn.Module):
    """Heading Consistency + Curvature Smoothness Loss.

    1. Heading Consistency: 예측 궤적의 방향이 급변하지 않도록
    2. Curvature Smoothness: yaw_rate가 급변하지 않도록

    학습 시 MSE와 가중합:
        total_loss = MSE + lambda_heading * heading_loss
                         + lambda_curvature * curvature_loss
    """

    def __init__(
        self,
        lambda_heading: float = 0.1,
        lambda_curvature: float = 0.05,
    ) -> None:
        super().__init__()
        self.lambda_heading = lambda_heading
        self.lambda_curvature = lambda_curvature

    def heading_consistency_loss(self, positions: torch.Tensor) -> torch.Tensor:
        """예측 궤적의 heading이 급변하지 않도록 정규화.

        정지 상태(이동량 ≈ 0)에서는 heading이 노이즈에 의해 랜덤 →
        해당 스텝의 loss를 마스킹하여 gradient 폭주 방지 (Gemini review #2).

        Args:
            positions: (batch, pred_length, 2) — 정규화 좌표

        Returns:
            loss: scalar
        """
        # 각 스텝의 이동 방향 벡터
        delta = positions[:, 1:, :] - positions[:, :-1, :]  # (batch, pred-1, 2)

        # 연속 방향 벡터 간 코사인 유사도
        d1 = delta[:, :-1, :]   # (batch, pred-2, 2)
        d2 = delta[:, 1:, :]    # (batch, pred-2, 2)

        dot = (d1 * d2).sum(dim=-1)                         # (batch, pred-2)
        norm1 = torch.norm(d1, dim=-1)
        norm2 = torch.norm(d2, dim=-1)

        # ✅ 정지 상태 마스킹: 이동량이 매우 작으면 heading이 무의미
        # 정규화 좌표 기준 ~5px/1920 ≈ 0.0026
        min_movement = 2.5e-3
        valid_mask = (norm1 > min_movement) & (norm2 > min_movement)
        valid_mask = valid_mask.float()  # (batch, pred-2)

        norm1 = norm1.clamp(min=1e-6)
        norm2 = norm2.clamp(min=1e-6)
        cos_sim = dot / (norm1 * norm2)

        # 마스킹된 loss: 유효한 스텝만 평균
        masked_loss = (1.0 - cos_sim) * valid_mask
        n_valid = valid_mask.sum().clamp(min=1.0)
        return masked_loss.sum() / n_valid

    def curvature_smoothness_loss(self, yaw_rate: torch.Tensor) -> torch.Tensor:
        """yaw_rate의 급격한 변화를 억제.

        Args:
            yaw_rate: (batch, pred_length)

        Returns:
            loss: scalar
        """
        # yaw_rate의 시간 차분 → 급변 penalty
        d_omega = yaw_rate[:, 1:] - yaw_rate[:, :-1]  # (batch, pred-1)
        return (d_omega ** 2).mean()

    def forward(
        self,
        positions: torch.Tensor,
        yaw_rate: torch.Tensor,
    ) -> torch.Tensor:
        """총 Physics Loss 계산.

        Args:
            positions: (batch, pred_length, 2)
            yaw_rate: (batch, pred_length)

        Returns:
            physics_loss: scalar
        """
        h_loss = self.heading_consistency_loss(positions)
        c_loss = self.curvature_smoothness_loss(yaw_rate)

        return self.lambda_heading * h_loss + self.lambda_curvature * c_loss


# ==========================================
# 🔧 [유틸리티]
# ==========================================
def build_model(config: Dict[str, Any] | ModelConfig | None = None) -> ResidualLSTMKinematic:
    if config is None:
        model_config = ModelConfig.from_dict(CONFIG)
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        model_config = ModelConfig.from_dict({**CONFIG, **config})
    return ResidualLSTMKinematic(model_config)


def enable_mc_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MCDropout):
            module.force_mc_dropout = True


def set_eval_with_mc_dropout(model: nn.Module) -> None:
    model.eval()
    enable_mc_dropout(model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)