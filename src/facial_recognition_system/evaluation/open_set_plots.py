"""Figures of the private open-set evaluation (E08).

They only show aggregated counts and rates, with anonymous IDs. Labels are in
Portuguese because the figures go to the README and the video.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure

from facial_recognition_system.evaluation.open_set import UNKNOWN_LABEL, ConfusionMatrix

UNKNOWN_DISPLAY: str = "desconhecido"


def plot_confusion_matrix(confusion: ConfusionMatrix, title: str) -> Figure:
    """Draws the confusion matrix with the count in each cell.

    Args:
        confusion: Rows are true labels and columns decisions, anonymous IDs only.
        title: Title of the figure.

    Returns:
        The figure, ready to be saved.
    """
    matrix = np.array(confusion.matrix)
    names = [UNKNOWN_DISPLAY if label == UNKNOWN_LABEL else label for label in confusion.labels]
    size = max(4.0, 0.5 * len(names) + 2)

    figure = Figure(figsize=(size, size), layout="constrained")
    axes = figure.subplots()
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes, shrink=0.8)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row, column in np.ndindex(matrix.shape):
        color = "white" if matrix[row, column] > threshold else "black"
        axes.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)

    axes.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axes.set_yticks(range(len(names)), names)
    axes.set_xlabel("decisão do sistema")
    axes.set_ylabel("identidade real")
    axes.set_title(title)
    return figure


def plot_far_frr(
    thresholds: Sequence[float],
    far: Sequence[float],
    frr: Sequence[float],
    chosen_threshold: float,
    title: str,
) -> Figure:
    """Draws FAR and FRR against the threshold, marking the chosen one.

    Args:
        thresholds: Threshold of each point, increasing.
        far: Unknown attempts accepted at each threshold.
        frr: Known attempts rejected at each threshold.
        chosen_threshold: Threshold chosen on validation.
        title: Title of the figure.

    Returns:
        The figure, ready to be saved.
    """
    figure = Figure(figsize=(6, 4), layout="constrained")
    axes = figure.subplots()
    axes.plot(thresholds, far, label="FAR (desconhecidos aceitos)")
    axes.plot(thresholds, frr, label="FRR (conhecidos rejeitados)")
    axes.axvline(
        chosen_threshold,
        color="black",
        linestyle="--",
        label=f"threshold = {chosen_threshold:.3f}",
    )
    axes.set_ylim(0, 1.01)
    axes.set_xlabel("escore de rejeição")
    axes.set_ylabel("taxa")
    axes.set_title(title)
    axes.grid(True, alpha=0.3)
    axes.legend()
    return figure
