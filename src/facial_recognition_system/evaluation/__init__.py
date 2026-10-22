"""Metrics, threshold selection and plots."""

from facial_recognition_system.evaluation.experiment import (
    ExperimentEvaluation,
    SplitMetrics,
    evaluate_experiment,
)
from facial_recognition_system.evaluation.open_set import (
    UNKNOWN_LABEL,
    ConfusionMatrix,
    OpenSetAttempts,
    OpenSetMetrics,
    open_set_metrics,
)
from facial_recognition_system.evaluation.open_set_plots import (
    plot_confusion_matrix,
    plot_far_frr,
)
from facial_recognition_system.evaluation.plots import plot_roc, plot_similarity_histogram
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
    "ExperimentEvaluation",
    "MetricInterval",
    "OpenSetAttempts",
    "OpenSetMetrics",
    "SimilarityDistributions",
    "SplitMetrics",
    "ThresholdChoice",
    "ThresholdCriterion",
    "VerificationMetrics",
    "VerificationReport",
    "VerificationScores",
    "choose_eer_threshold",
    "choose_far_threshold",
    "equal_error_rate",
    "evaluate_experiment",
    "evaluate_verification",
    "false_accept_rate",
    "false_reject_rate",
    "open_set_metrics",
    "plot_confusion_matrix",
    "plot_far_frr",
    "plot_roc",
    "plot_similarity_histogram",
    "similarity_distributions",
    "tar_at_far",
    "verification_metrics",
]
