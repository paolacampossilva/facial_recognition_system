"""Threshold selection on the validation scores only (decision D8).

Public verification uses the EER threshold. The private dataset and the
application use the smallest threshold with FAR <= 1%, falling back to the EER
threshold when that leaves an FRR above 20%.
"""

from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import roc_curve

from facial_recognition_system.evaluation.scores import (
    VerificationScores,
    false_accept_rate,
    false_reject_rate,
)


class ThresholdCriterion(StrEnum):
    EER = "eer"
    FAR_TARGET = "far_target"


class ThresholdChoice(BaseModel):
    """The chosen threshold and its error rates on the scores it was chosen on."""

    value: float
    criterion: ThresholdCriterion
    chosen_on: Literal["val"] = "val"
    far: float = Field(ge=0, le=1)
    frr: float = Field(ge=0, le=1)


class EqualErrorRate(BaseModel):
    eer: float = Field(ge=0, le=1)
    threshold: float


def equal_error_rate(scores: VerificationScores) -> EqualErrorRate:
    """Finds the threshold where FAR and FRR are closest and averages them.

    Args:
        scores: Genuine and impostor similarity scores.

    Returns:
        The EER and the threshold where it happens.
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        scores.labels, scores.scores, drop_intermediate=False
    )
    false_negative_rate = 1 - true_positive_rate
    # thresholds[0] is +inf (nothing accepted); it never ties better than a real score.
    index = int(np.argmin(np.abs(false_negative_rate - false_positive_rate)))
    eer = (false_positive_rate[index] + false_negative_rate[index]) / 2
    return EqualErrorRate(eer=float(eer), threshold=float(thresholds[index]))


def choose_eer_threshold(scores: VerificationScores) -> ThresholdChoice:
    """Chooses the EER threshold (public verification, D8)."""
    threshold = equal_error_rate(scores).threshold
    return ThresholdChoice(
        value=threshold,
        criterion=ThresholdCriterion.EER,
        far=false_accept_rate(scores, threshold),
        frr=false_reject_rate(scores, threshold),
    )


def choose_far_threshold(
    scores: VerificationScores, max_far: float = 0.01, max_frr: float = 0.20
) -> ThresholdChoice:
    """Chooses the smallest threshold with FAR <= ``max_far`` (private and application, D8).

    Args:
        scores: Validation scores; impostors are the unknown attempts.
        max_far: Highest acceptable false accept rate.
        max_frr: Highest acceptable false reject rate at that threshold. Above
            it, the EER threshold is used instead, as D8 requires.

    Returns:
        The chosen threshold, with the criterion that was actually applied.
    """
    # Candidates: every score, plus one just above the highest, which accepts nothing.
    highest = scores.scores.max()
    candidates = np.unique(np.append(scores.scores, np.nextafter(highest, np.inf)))
    sorted_impostors = np.sort(scores.impostor)
    accepted_impostors = len(sorted_impostors) - np.searchsorted(
        sorted_impostors, candidates, side="left"
    )
    far_per_candidate = accepted_impostors / len(sorted_impostors)
    threshold = float(candidates[far_per_candidate <= max_far].min())

    frr = false_reject_rate(scores, threshold)
    if frr > max_frr:
        return choose_eer_threshold(scores)

    return ThresholdChoice(
        value=threshold,
        criterion=ThresholdCriterion.FAR_TARGET,
        far=false_accept_rate(scores, threshold),
        frr=frr,
    )
