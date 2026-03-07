"""Random Network Distillation (RND) for OOD detection in embedding space.

Fixed random target network f_target, trainable predictor f_pred.
Score s_RND(x) = ||f_pred(x) - f_target(x)||_2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RNDTarget(nn.Module):
    """Fixed random target network. Weights are frozen."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictor(nn.Module):
    """Trainable predictor network."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule(nn.Module):
    """RND: target (fixed) + predictor (trainable). Score = L2 prediction error."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int | None = None,
        seed: int = 42,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        output_dim = output_dim or hidden_dims[-1]
        self.output_dim = output_dim
        self.target = RNDTarget(input_dim, hidden_dims, output_dim, seed=seed)
        self.predictor = RNDPredictor(input_dim, hidden_dims, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return RND score (L2 error) per sample. Shape [B]."""
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        return torch.norm(pred_out - target_out, dim=1, p=2)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """MSE loss for training predictor."""
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        return ((pred_out - target_out) ** 2).mean()


def compute_rnd_scores(
    model: "RNDModule | RNDPredictor",
    embeddings: "np.ndarray",
    device: str = "cpu",
) -> "np.ndarray":
    """Compute RND scores (L2 prediction error) for embeddings.

    Args:
        model: RNDModule (preferred) or RNDPredictor. RNDModule has target+predictor.
        embeddings: (N, D) numpy array.
        device: Device for inference.

    Returns:
        (N,) RND scores.
    """
    import numpy as np

    x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        if hasattr(model, "target"):
            t = model.target(x)
            p = model.predictor(x)
            scores = torch.norm(p - t, dim=1, p=2).cpu().numpy()
        else:
            p = model(x)
            scores = torch.norm(p, dim=1, p=2).cpu().numpy()
    return scores.astype(np.float32)
