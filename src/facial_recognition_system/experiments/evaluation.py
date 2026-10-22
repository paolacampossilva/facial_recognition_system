"""Evaluation of an extracted experiment, filling ``threshold`` and ``metrics`` of its JSON.

Reads ``<results_dir>/<EXP>.json`` and the val and test embeddings it lists,
scores the pairs of the contract, chooses the threshold on validation (D8),
computes the metrics of both splits (D7) and writes them back to the same JSON,
keeping every other field. Saves ``<EXP>_roc.png`` and ``<EXP>_hist.png`` to
``figures_dir``.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from facial_recognition_system.evaluation import (
    ExperimentEvaluation,
    VerificationScores,
    evaluate_experiment,
    plot_roc,
    plot_similarity_histogram,
)
from facial_recognition_system.experiments.config import ExperimentConfig
from facial_recognition_system.experiments.embeddings import load_embeddings
from facial_recognition_system.experiments.errors import MissingEmbeddingsError
from facial_recognition_system.experiments.pairs import pair_scores, read_pairs

EVALUATED_SPLITS: tuple[str, str] = ("val", "test")


class ExperimentScores(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    val: VerificationScores
    test: VerificationScores


class EvaluatedExperiment(BaseModel):
    """The evaluation written to the JSON, with the scores the figures were drawn from."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    experiment: str
    scores: ExperimentScores
    evaluation: ExperimentEvaluation


def load_experiment_scores(config: ExperimentConfig, record: dict[str, Any]) -> ExperimentScores:
    """Scores the val and test pairs with the embeddings listed in the results record.

    Raises:
        MissingEmbeddingsError: The record has no embeddings file for val or test.
        KeyError: A photo of a pair is not in the embeddings.
    """
    embedding_files = record.get("embeddings") or {}
    scores: dict[str, VerificationScores] = {}
    for split in EVALUATED_SPLITS:
        if split not in embedding_files:
            raise MissingEmbeddingsError(config.experiment, split)
        embedding_set = load_embeddings(Path(embedding_files[split]["file"]))
        pairs = read_pairs(config.dataset.pairs_file(split))
        scores[split] = pair_scores(embedding_set, pairs)
    return ExperimentScores(**scores)


def save_figures(evaluated: EvaluatedExperiment, figures_dir: Path) -> list[Path]:
    """Saves the ROC of val and test and the test histogram of one experiment.

    Returns:
        The saved files.
    """
    experiment = evaluated.experiment
    threshold = evaluated.evaluation.threshold.value
    figures_dir.mkdir(parents=True, exist_ok=True)

    roc = plot_roc(
        {"validação": evaluated.scores.val, "teste": evaluated.scores.test},
        title=f"{experiment}: curva ROC",
        threshold=threshold,
    )
    histogram = plot_similarity_histogram(
        evaluated.scores.test,
        title=f"{experiment}: similaridade nos pares de teste",
        threshold=threshold,
    )

    roc_path = figures_dir / f"{experiment}_roc.png"
    histogram_path = figures_dir / f"{experiment}_hist.png"
    roc.savefig(roc_path, dpi=150)
    histogram.savefig(histogram_path, dpi=150)
    return [roc_path, histogram_path]


def run_evaluation(
    config: ExperimentConfig, results_dir: Path, figures_dir: Path, n_resamples: int = 1000
) -> EvaluatedExperiment:
    """Evaluates one experiment, writes its metrics to the results JSON and saves its figures.

    Args:
        config: Config of the experiment, for its id, seed and pair files.
        results_dir: Folder with ``<EXP>.json``, written by the extraction.
        figures_dir: Folder the figures are saved to.
        n_resamples: Number of bootstrap resamples of each interval.

    Returns:
        The evaluation and the scores it came from.

    Raises:
        FileNotFoundError: The results JSON, an embeddings file or a pairs file is missing.
        MissingEmbeddingsError: The results JSON has no embeddings for val or test.
    """
    results_path = results_dir / f"{config.experiment}.json"
    record = json.loads(results_path.read_text(encoding="utf-8"))

    scores = load_experiment_scores(config, record)
    evaluation = evaluate_experiment(
        scores.val, scores.test, n_resamples=n_resamples, seed=config.seed
    )

    record |= evaluation.model_dump(mode="json")
    results_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", "utf-8")

    evaluated = EvaluatedExperiment(
        experiment=config.experiment, scores=scores, evaluation=evaluation
    )
    save_figures(evaluated, figures_dir)
    return evaluated
