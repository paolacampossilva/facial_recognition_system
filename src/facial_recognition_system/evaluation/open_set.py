"""Open-set identification metrics of decision D7 for the private dataset.

Each attempt has its true identity (or ``UNKNOWN_LABEL`` for someone outside
the gallery), the gallery identity the comparison strategy picked and the
rejection score of that pick. Below the threshold the attempt becomes
"unknown". Labels must be anonymous IDs (``P01``), so no name reaches the
aggregated results or the confusion matrix (LGPD).
"""

import re
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from facial_recognition_system.evaluation.scores import VerificationScores

UNKNOWN_LABEL: str = "unknown"
ANONYMOUS_ID_PATTERN: re.Pattern[str] = re.compile(r"^P\d{2,}$")


class OpenSetAttempts(BaseModel):
    # Hidden input keeps a rejected name out of the error message and the logs.
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, hide_input_in_errors=True)

    true_labels: list[str]
    predicted_labels: list[str]
    scores: np.ndarray

    @field_validator("true_labels", "predicted_labels")
    @classmethod
    def _check_anonymous(cls, labels: list[str]) -> list[str]:
        named = sorted(
            {
                label
                for label in labels
                if label != UNKNOWN_LABEL and not ANONYMOUS_ID_PATTERN.match(label)
            }
        )
        if named:
            raise ValueError(
                f"{len(named)} label(s) are not anonymous IDs like 'P01' or '{UNKNOWN_LABEL}'"
            )
        return labels

    @field_validator("scores", mode="before")
    @classmethod
    def _as_1d_array(cls, value: object) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim != 1:
            raise ValueError("must be one-dimensional")
        return array

    @model_validator(mode="after")
    def _check_lengths(self) -> Self:
        if not len(self.true_labels) == len(self.predicted_labels) == len(self.scores):
            raise ValueError("true_labels, predicted_labels and scores must have the same length")
        if not self.true_labels:
            raise ValueError("at least one attempt is needed")
        if UNKNOWN_LABEL in self.predicted_labels:
            raise ValueError("predicted_labels hold the gallery pick; rejection comes from scores")
        return self

    def rejection_scores(self) -> VerificationScores:
        """Known attempts as genuine and unknown ones as impostors, to choose the threshold."""
        is_known = [int(label != UNKNOWN_LABEL) for label in self.true_labels]
        return VerificationScores(scores=self.scores, labels=np.array(is_known))


class ConfusionMatrix(BaseModel):
    """Rows are true labels and columns are decisions, both in ``labels`` order."""

    labels: list[str]
    matrix: list[list[int]]


class OpenSetMetrics(BaseModel):
    threshold: float
    accuracy: float
    f1_macro: float
    far: float | None
    frr: float | None
    misidentification_rate: float | None
    confusion: ConfusionMatrix


def open_set_metrics(attempts: OpenSetAttempts, threshold: float) -> OpenSetMetrics:
    """Computes the aggregated open-set metrics at ``threshold``.

    Args:
        attempts: Attempts with true labels, gallery picks and rejection scores.
        threshold: Threshold chosen on validation; picks below it become unknown.

    Returns:
        Accuracy and F1 macro with "unknown" as a class, FAR (unknowns accepted),
        FRR (known people rejected), misidentification rate (known people accepted
        as someone else) and the confusion matrix. A rate is None when the
        attempts have no unknown or no known person to compute it on.
    """
    true_labels = np.array(attempts.true_labels)
    accepted = attempts.scores >= threshold
    decisions = np.where(accepted, np.array(attempts.predicted_labels), UNKNOWN_LABEL)

    is_unknown = true_labels == UNKNOWN_LABEL
    is_known = ~is_unknown
    far = _rate(decisions[is_unknown] != UNKNOWN_LABEL)
    frr = _rate(decisions[is_known] == UNKNOWN_LABEL)
    misidentified = (decisions != UNKNOWN_LABEL) & (decisions != true_labels)
    misidentification_rate = _rate(misidentified[is_known])

    labels = sorted(set(true_labels) | set(decisions))
    matrix = confusion_matrix(true_labels, decisions, labels=labels)

    return OpenSetMetrics(
        threshold=threshold,
        accuracy=float(accuracy_score(true_labels, decisions)),
        f1_macro=float(f1_score(true_labels, decisions, labels=labels, average="macro")),
        far=far,
        frr=frr,
        misidentification_rate=misidentification_rate,
        confusion=ConfusionMatrix(labels=labels, matrix=matrix.tolist()),
    )


def _rate(events: np.ndarray) -> float | None:
    return float(events.mean()) if len(events) else None
