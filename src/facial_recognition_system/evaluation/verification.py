"""Verification metrics of decision D7, with bootstrap confidence intervals.

The threshold for accuracy and F1 is always the one chosen on validation; the
test metrics reuse it without readjusting.
"""

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

from facial_recognition_system.evaluation.scores import VerificationScores
from facial_recognition_system.evaluation.threshold import equal_error_rate


class VerificationMetrics(BaseModel):
    roc_auc: float
    eer: float
    tar_at_far: float
    accuracy: float
    f1: float


class MetricInterval(BaseModel):
    value: float
    low: float
    high: float


class VerificationReport(BaseModel):
    """Point metrics with their bootstrap intervals, as stored in the results JSON."""

    threshold: float
    target_far: float
    confidence: float
    n_resamples: int
    roc_auc: MetricInterval
    eer: MetricInterval
    tar_at_far: MetricInterval
    accuracy: MetricInterval
    f1: MetricInterval


class SimilarityDistributions(BaseModel):
    """Scores of same-person and different-people pairs, for the histograms."""

    genuine: list[float]
    impostor: list[float]


def tar_at_far(scores: VerificationScores, target_far: float) -> float:
    """Highest true accept rate among thresholds with FAR <= ``target_far``."""
    false_positive_rate, true_positive_rate, _ = roc_curve(
        scores.labels, scores.scores, drop_intermediate=False
    )
    return float(true_positive_rate[false_positive_rate <= target_far].max())


def verification_metrics(
    scores: VerificationScores, threshold: float, target_far: float = 0.01
) -> VerificationMetrics:
    """Computes the point metrics of D7.

    Args:
        scores: Genuine and impostor similarity scores of the pairs.
        threshold: Threshold chosen on validation, used for accuracy and F1.
        target_far: FAR at which the TAR is reported.

    Returns:
        ROC-AUC, EER, TAR@FAR, and accuracy and F1 at ``threshold``.
    """
    predictions = (scores.scores >= threshold).astype(int)
    return VerificationMetrics(
        roc_auc=float(roc_auc_score(scores.labels, scores.scores)),
        eer=equal_error_rate(scores).eer,
        tar_at_far=tar_at_far(scores, target_far),
        accuracy=float(accuracy_score(scores.labels, predictions)),
        f1=float(f1_score(scores.labels, predictions)),
    )


def evaluate_verification(
    scores: VerificationScores,
    threshold: float,
    target_far: float = 0.01,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> VerificationReport:
    """Computes the metrics of D7 with percentile bootstrap intervals.

    Genuine and impostor pairs are resampled separately, so every resample
    keeps the class balance and both classes, and the threshold stays fixed.

    Args:
        scores: Genuine and impostor similarity scores of the pairs.
        threshold: Threshold chosen on validation.
        target_far: FAR at which the TAR is reported.
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the intervals.
        seed: Seed of the resampling.

    Returns:
        Every metric with its value and interval.
    """
    point = verification_metrics(scores, threshold, target_far)
    rng = np.random.default_rng(seed)
    genuine_indices = np.flatnonzero(scores.labels == 1)
    impostor_indices = np.flatnonzero(scores.labels == 0)

    resampled: list[VerificationMetrics] = []
    for _ in range(n_resamples):
        indices = np.concatenate(
            [
                rng.choice(genuine_indices, size=len(genuine_indices)),
                rng.choice(impostor_indices, size=len(impostor_indices)),
            ]
        )
        resample = VerificationScores(scores=scores.scores[indices], labels=scores.labels[indices])
        resampled.append(verification_metrics(resample, threshold, target_far))

    tail = (1 - confidence) / 2 * 100

    def interval(name: str) -> MetricInterval:
        values = [getattr(metrics, name) for metrics in resampled]
        low, high = np.percentile(values, [tail, 100 - tail])
        return MetricInterval(value=getattr(point, name), low=float(low), high=float(high))

    return VerificationReport(
        threshold=threshold,
        target_far=target_far,
        confidence=confidence,
        n_resamples=n_resamples,
        roc_auc=interval("roc_auc"),
        eer=interval("eer"),
        tar_at_far=interval("tar_at_far"),
        accuracy=interval("accuracy"),
        f1=interval("f1"),
    )


def similarity_distributions(scores: VerificationScores) -> SimilarityDistributions:
    return SimilarityDistributions(
        genuine=scores.genuine.tolist(), impostor=scores.impostor.tolist()
    )
