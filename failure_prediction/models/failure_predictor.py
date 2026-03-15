"""Baseline MLP failure predictor.

Predicts P(failure within K steps | x_t) from a single embedding field.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# 512 -> 256 -> 128 -> 1. Input: feat_decoder_mean. Output: logit. BCEWithLogitsLoss.
class FailurePredictorMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        dims = [input_dim] + hidden_dims + [1]
        # Linear -> ReLU -> Dropout for each hidden layer; final Linear -> logit
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits [B]. Use BCEWithLogitsLoss (no sigmoid in forward)."""
        out = self.mlp(x)
        return out.squeeze(-1)
