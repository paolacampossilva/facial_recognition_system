"""Evaluation of one public experiment: threshold on validation, metrics on both splits.

This is the ``threshold`` and ``metrics`` part of the results JSON (decisions D7
and D8). The threshold is the EER threshold of the validation pairs and the test
metrics reuse it without readjusting.
"""

from pydantic import BaseModel

from facial_recognition_system.evaluation.scores import VerificationScores
from facial_recognition_system.evaluation.threshold import ThresholdChoice, choose_eer_threshold
from facial_recognition_system.evaluation.verification import (
    VerificationReport,
    evaluate_verification,
)


class SplitMetrics(BaseModel):
    val: VerificationReport
    test: VerificationReport


class ExperimentEvaluation(BaseModel):
    threshold: ThresholdChoice
    metrics: SplitMetrics


def evaluate_experiment(
    val_scores: VerificationScores,
    test_scores: VerificationScores,
    target_far: float = 0.01,
    n_resamples: int = 1000,
    seed: int = 42,
) -> ExperimentEvaluation:
    """Chooses the threshold on validation and reports the metrics of both splits.

    Args:
        val_scores: Similarity scores of the validation pairs.
        test_scores: Similarity scores of the test pairs.
        target_far: FAR at which the TAR is reported.
        n_resamples: Number of bootstrap resamples of each interval.
        seed: Seed of the resampling.

    Returns:
        The EER threshold of validation and the metrics with intervals of each split.
    """
    threshold = choose_eer_threshold(val_scores)
    metrics = SplitMetrics(
        val=evaluate_verification(val_scores, threshold.value, target_far, n_resamples, seed=seed),
        test=evaluate_verification(
            test_scores, threshold.value, target_far, n_resamples, seed=seed
        ),
    )
    return ExperimentEvaluation(threshold=threshold, metrics=metrics)
