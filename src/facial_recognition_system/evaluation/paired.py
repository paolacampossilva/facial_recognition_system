"""Paired comparison of two models on the same verification pairs (D5).

Both models score the same pairs, so their errors can be compared pair by pair:
which errors of the reference the other model fixes, which new ones it makes,
and how much the EER changes. The EER difference gets a paired bootstrap
interval: every resample draws the same pairs for both models, with genuine and
impostor pairs resampled separately, as in the single-model intervals (D7).
"""

from typing import Self

import numpy as np
from pydantic import BaseModel, model_validator

from facial_recognition_system.evaluation.scores import VerificationScores
from facial_recognition_system.evaluation.threshold import equal_error_rate


class ErrorCounts(BaseModel):
    errors: int
    false_accepts: int
    false_rejects: int


class ErrorComparison(BaseModel):
    """Errors of a model against a reference, at each model's own threshold."""

    reference: ErrorCounts
    other: ErrorCounts
    fixed: int
    introduced: int
    shared: int


class PairedEerDifference(BaseModel):
    """EER of the other model minus the reference, as fractions (0.01 = 1 p.p.)."""

    observed: float
    bootstrap_mean: float
    low: float
    high: float
    share_other_better: float
    confidence: float
    n_resamples: int

    @model_validator(mode="after")
    def _check_interval(self) -> Self:
        if self.low > self.high:
            raise ValueError("low must not be above high")
        return self


def error_mask(scores: VerificationScores, threshold: float) -> np.ndarray:
    """Marks the pairs a model gets wrong when accepting ``score >= threshold``."""
    return (scores.scores >= threshold) != (scores.labels == 1)


def error_counts(scores: VerificationScores, threshold: float) -> ErrorCounts:
    accepted = scores.scores >= threshold
    genuine = scores.labels == 1
    return ErrorCounts(
        errors=int((accepted != genuine).sum()),
        false_accepts=int((accepted & ~genuine).sum()),
        false_rejects=int((~accepted & genuine).sum()),
    )


def _check_same_pairs(reference: VerificationScores, other: VerificationScores) -> None:
    if not np.array_equal(reference.labels, other.labels):
        raise ValueError("both models must score the same pairs, in the same order")


def compare_errors(
    reference: VerificationScores,
    reference_threshold: float,
    other: VerificationScores,
    other_threshold: float,
) -> ErrorComparison:
    """Counts the errors each model makes and how they overlap.

    Args:
        reference: Scores of the reference model (E01).
        reference_threshold: Threshold of the reference, chosen on validation.
        other: Scores of the compared model on the same pairs.
        other_threshold: Threshold of the compared model, chosen on validation.

    Returns:
        The errors of each model, the reference errors the other model fixes,
        the new errors it makes and the errors both make.

    Raises:
        ValueError: The two score sets are not on the same pairs.
    """
    _check_same_pairs(reference, other)
    reference_errors = error_mask(reference, reference_threshold)
    other_errors = error_mask(other, other_threshold)
    return ErrorComparison(
        reference=error_counts(reference, reference_threshold),
        other=error_counts(other, other_threshold),
        fixed=int((reference_errors & ~other_errors).sum()),
        introduced=int((~reference_errors & other_errors).sum()),
        shared=int((reference_errors & other_errors).sum()),
    )


def paired_eer_difference(
    reference: VerificationScores,
    other: VerificationScores,
    rng: np.random.Generator,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> PairedEerDifference:
    """EER difference between two models on the same pairs, with a paired bootstrap interval.

    Args:
        reference: Scores of the reference model (E01).
        other: Scores of the compared model on the same pairs.
        rng: Random generator. Comparing several models with one generator, in a
            fixed order, gives the same intervals on every run.
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the percentile interval.

    Returns:
        The observed difference (other minus reference), the bootstrap mean, the
        interval and the share of resamples where the other model has the lower EER.

    Raises:
        ValueError: The two score sets are not on the same pairs.
    """
    _check_same_pairs(reference, other)
    genuine_indices = np.flatnonzero(reference.labels == 1)
    impostor_indices = np.flatnonzero(reference.labels == 0)

    def difference(indices: np.ndarray | None = None) -> float:
        if indices is None:
            return equal_error_rate(other).eer - equal_error_rate(reference).eer
        resampled = [
            VerificationScores(scores=scores.scores[indices], labels=scores.labels[indices])
            for scores in (reference, other)
        ]
        return equal_error_rate(resampled[1]).eer - equal_error_rate(resampled[0]).eer

    differences = np.array(
        [
            difference(
                np.concatenate(
                    [
                        rng.choice(genuine_indices, size=len(genuine_indices)),
                        rng.choice(impostor_indices, size=len(impostor_indices)),
                    ]
                )
            )
            for _ in range(n_resamples)
        ]
    )
    tail = (1 - confidence) / 2 * 100
    return PairedEerDifference(
        observed=difference(),
        bootstrap_mean=float(differences.mean()),
        low=float(np.percentile(differences, tail)),
        high=float(np.percentile(differences, 100 - tail)),
        share_other_better=float((differences < 0).mean()),
        confidence=confidence,
        n_resamples=n_resamples,
    )
