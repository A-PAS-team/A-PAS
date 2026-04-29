"""
Residual LSTM with MC Dropout for A-PAS trajectory prediction.

Input : (batch, 60, 17)
Output: (batch, 30, 2), normalized x/y coordinates in [0, 1]

Dropout placement:
  1. Inside the stacked LSTM between recurrent layers.
  2. After the final LSTM hidden state and before the FC head.

Expected trade-off:
  MC Dropout typically increases deterministic ADE by about +0.3~0.5 px,
  but provides predictive uncertainty through repeated stochastic inference.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

import torch
import torch.nn as nn


CONFIG: Dict[str, Any] = {
    "input_size": 17,
    "hidden_size": 256,
    "num_layers": 2,
    "pred_length": 10,
    "dropout_p": 0.2,
}


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 17
    hidden_size: int = 256
    num_layers: int = 2
    pred_length: int = 10
    dropout_p: float = 0.2

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ModelConfig":
        known = {k: v for k, v in cfg.items() if k in cls.__annotations__}
        return cls(**known)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MCDropout(nn.Module):
    """Dropout with an exportable stochastic path for ONNX Runtime.

    nn.Dropout can become deterministic in ONNX Runtime even when exported
    with training mode preserved. This module uses primitive random masking
    when training or MC sampling is forced, so the ONNX graph can contain
    RandomUniformLike/Greater/Cast-style stochastic ops.
    """

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


class ResidualLSTMMCDropout(nn.Module):
    """Baseline-compatible Residual LSTM with MC Dropout support."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h_last = self.mc_dropout(h_n[-1])
        offset = self.fc(h_last).view(-1, self.config.pred_length, 2)
        last_pos = x[:, -1, :2].unsqueeze(1)
        return last_pos + offset


def build_model(config: Dict[str, Any] | ModelConfig | None = None) -> ResidualLSTMMCDropout:
    if config is None:
        model_config = ModelConfig.from_dict(CONFIG)
    elif isinstance(config, ModelConfig):
        model_config = config
    else:
        model_config = ModelConfig.from_dict({**CONFIG, **config})
    return ResidualLSTMMCDropout(model_config)


def enable_mc_dropout(model: nn.Module) -> None:
    """Keep only dropout layers in train mode for MC sampling."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, MCDropout):
            module.force_mc_dropout = True


def set_eval_with_mc_dropout(model: nn.Module) -> None:
    """Set BN/LayerNorm/etc. to eval while keeping Dropout stochastic."""
    model.eval()
    enable_mc_dropout(model)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
