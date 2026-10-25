"""Figures of the private open-set evaluation (E08).

They only show aggregated counts and rates, nothing per participant. Labels are in
Portuguese because the figures go to the README and the video.
"""

from collections.abc import Sequence

from matplotlib.figure import Figure

from facial_recognition_system.evaluation.open_set import OpenSetCounts

NOT_APPLICABLE: str = "—"


def plot_open_set_counts(counts: OpenSetCounts, title: str) -> Figure:
    """Draws the table of aggregated totals, with nothing per participant.

    Args:
        counts: Aggregated outcome counts of the known and unknown attempts.
        title: Title of the figure.

    Returns:
        The figure, ready to be saved.
    """
    columns = [
        "Reconhecida\ncorretamente",
        "Atribuída a outra pessoa\n(desconhecidos: aceito\ncomo alguém)",
        "Rejeitada como\ndesconhecida",
    ]
    rows = [
        f"Pessoas cadastradas\n({counts.known_attempts} tentativas)",
        f"Desconhecidos\n({counts.unknown_attempts} tentativas)",
    ]
    cells = [
        [str(counts.known_correct), str(counts.known_misidentified), str(counts.known_rejected)],
        [NOT_APPLICABLE, str(counts.unknown_accepted), str(counts.unknown_rejected)],
    ]

    figure = Figure(figsize=(8, 3), layout="constrained")
    axes = figure.subplots()
    axes.axis("off")
    table = axes.table(
        cellText=cells, rowLabels=rows, colLabels=columns, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
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
