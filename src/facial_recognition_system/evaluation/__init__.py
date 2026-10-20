"""Metrics, threshold selection and plots."""

from facial_recognition_system.evaluation.open_set import (
    UNKNOWN_LABEL,
    ConfusionMatrix,
    OpenSetAttempts,
    OpenSetMetrics,
    open_set_metrics,
)
from facial_recognition_system.evaluation.scores import (
    VerificationScores,
    false_accept_rate,
    false_reject_rate,
)
from facial_recognition_system.evaluation.threshold import (
    EqualErrorRate,
    ThresholdChoice,
    ThresholdCriterion,
    choose_eer_threshold,
    choose_far_threshold,
    equal_error_rate,
)
from facial_recognition_system.evaluation.verification import (
    MetricInterval,
    SimilarityDistributions,
    VerificationMetrics,
    VerificationReport,
    evaluate_verification,
    similarity_distributions,
    tar_at_far,
    verification_metrics,
)

__all__ = [
    "UNKNOWN_LABEL",
    "ConfusionMatrix",
    "EqualErrorRate",
    "MetricInterval",
    "OpenSetAttempts",
    "OpenSetMetrics",
    "SimilarityDistributions",
    "ThresholdChoice",
    "ThresholdCriterion",
    "VerificationMetrics",
    "VerificationReport",
    "VerificationScores",
    "choose_eer_threshold",
    "choose_far_threshold",
    "equal_error_rate",
    "evaluate_verification",
    "false_accept_rate",
    "false_reject_rate",
    "open_set_metrics",
    "similarity_distributions",
    "tar_at_far",
    "verification_metrics",
]
