"""Similarity scores with genuine/impostor labels, the input of every metric.

A score is accepted when ``score >= threshold``. ``labels`` is 1 for a genuine
attempt (same person, or a known person in open-set) and 0 for an impostor
(different people, or an unknown person).
"""

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class VerificationScores(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    scores: np.ndarray
    labels: np.ndarray

    @field_validator("scores", "labels", mode="before")
    @classmethod
    def _as_1d_array(cls, value: object) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError("must be one-dimensional")
        return array

    @model_validator(mode="after")
    def _check_labels(self) -> Self:
        if len(self.scores) != len(self.labels):
            raise ValueError("scores and labels must have the same length")
        if not np.isin(self.labels, (0, 1)).all():
            raise ValueError("labels must be 0 (impostor) or 1 (genuine)")
        if not (self.labels == 1).any() or not (self.labels == 0).any():
            raise ValueError("both genuine and impostor scores are needed")
        if not np.isfinite(self.scores).all():
            raise ValueError("scores must be finite")
        return self

    @property
    def genuine(self) -> np.ndarray:
        return self.scores[self.labels == 1]

    @property
    def impostor(self) -> np.ndarray:
        return self.scores[self.labels == 0]


def false_accept_rate(scores: VerificationScores, threshold: float) -> float:
    """Fraction of impostor scores accepted at ``threshold``."""
    return float(np.mean(scores.impostor >= threshold))


def false_reject_rate(scores: VerificationScores, threshold: float) -> float:
    """Fraction of genuine scores rejected at ``threshold``."""
    return float(np.mean(scores.genuine < threshold))
