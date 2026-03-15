"""Interfaces for pluggable risk scoring and intervention.

RiskScorer: supervised MLP (predict_step).
FiperScorer: RND + ACE (compute_scores).
InterventionPolicy: when to interrupt (placeholder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class RiskScore:
    """Output of risk model for one timestep. prob = sigmoid(logit)."""
    logit: float
    prob: float
    raw_score: float | None = None


class RiskScorer(ABC):
    """Interface for supervised failure risk prediction."""

    @abstractmethod
    def predict_step(self, features: Any) -> RiskScore:
        """Compute risk for a single timestep. features: embedding or dict."""
        pass


@dataclass
class FiperScores:
    """RND + ACE scores for one timestep."""
    rnd_score: float
    ace_score: float
    alarm: bool = False


class FiperScorer(ABC):
    """Interface for FIPER-style OOD + uncertainty scoring."""

    @abstractmethod
    def compute_scores(self, embedding: Any, action_chunk: Any | None = None) -> FiperScores:
        """Compute RND and ACE scores, optionally trigger alarm."""
        pass


@dataclass
class InterventionDecision:
    """Placeholder for future: whether to interrupt, recovery action, confidence."""
    should_interrupt: bool
    reason: str = ""
    confidence: float = 0.0


class InterventionPolicy(ABC):
    """Interface for deciding when to intervene (placeholder)."""

    @abstractmethod
    def should_interrupt(
        self,
        risk_score: RiskScore | None = None,
        fiper_scores: FiperScores | None = None,
        **kwargs: Any,
    ) -> InterventionDecision:
        """Decide whether to interrupt based on risk/FIPER scores."""
        pass
