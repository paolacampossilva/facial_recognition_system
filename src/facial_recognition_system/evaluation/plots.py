"""ROC curves and similarity histograms of the verification pairs (decision D7).

The figures only show aggregated scores, never a photo or an identity. Labels
are in Portuguese because the figures go to the README and the video.
"""

from collections.abc import Mapping

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve

from facial_recognition_system.evaluation.scores import (
    VerificationScores,
    false_accept_rate,
    false_reject_rate,
)

# The x axis is logarithmic, so FAR = 0 is drawn at this floor.
MIN_PLOTTED_FAR: float = 1e-4
HISTOGRAM_BINS: int = 50


def plot_roc(
    curves: Mapping[str, VerificationScores],
    title: str,
    threshold: float | None = None,
    target_far: float = 0.01,
) -> Figure:
    """Draws one ROC curve per entry of ``curves``, with a logarithmic FAR axis.

    Args:
        curves: Scores of each curve, keyed by its legend label.
        title: Title of the figure.
        threshold: Validation threshold, marked on every curve when given.
        target_far: FAR of the reported TAR, drawn as a vertical line.

    Returns:
        The figure, ready to be saved.
    """
    figure = Figure(figsize=(6, 5), layout="constrained")
    axes = figure.subplots()
    for label, scores in curves.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            scores.labels, scores.scores, drop_intermediate=False
        )
        line = axes.plot(
            np.maximum(false_positive_rate, MIN_PLOTTED_FAR), true_positive_rate, label=label
        )[0]
        if threshold is not None:
            far = max(false_accept_rate(scores, threshold), MIN_PLOTTED_FAR)
            tar = 1 - false_reject_rate(scores, threshold)
            axes.plot(far, tar, "o", color=line.get_color())

    axes.axvline(
        target_far, color="gray", linestyle="--", linewidth=1, label=f"FAR = {target_far:.0%}"
    )
    axes.set_xscale("log")
    axes.set_xlim(MIN_PLOTTED_FAR, 1)
    axes.set_ylim(0, 1.01)
    axes.set_xlabel("FAR (taxa de falsa aceitação)")
    axes.set_ylabel("TAR (taxa de aceitação verdadeira)")
    axes.set_title(title)
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(loc="lower right")
    return figure


def plot_similarity_histogram(
    scores: VerificationScores, title: str, threshold: float | None = None
) -> Figure:
    """Draws the similarity of same-person and different-people pairs, normalized per class.

    Args:
        scores: Genuine and impostor similarity scores.
        title: Title of the figure.
        threshold: Validation threshold, drawn as a vertical line when given.

    Returns:
        The figure, ready to be saved.
    """
    figure = Figure(figsize=(6, 4), layout="constrained")
    axes = figure.subplots()
    bins = np.linspace(scores.scores.min(), scores.scores.max(), HISTOGRAM_BINS + 1)
    axes.hist(scores.genuine, bins=bins, density=True, alpha=0.6, label="mesma pessoa")
    axes.hist(scores.impostor, bins=bins, density=True, alpha=0.6, label="pessoas diferentes")
    if threshold is not None:
        axes.axvline(threshold, color="black", linestyle="--", label=f"threshold = {threshold:.3f}")
    axes.set_xlabel("similaridade de cosseno")
    axes.set_ylabel("densidade")
    axes.set_title(title)
    axes.legend()
    return figure
