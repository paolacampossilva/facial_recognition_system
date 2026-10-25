"""Evaluates extracted experiments and fills ``threshold`` and ``metrics`` of their JSON.

For each config, scores the val and test pairs with the embeddings listed in
``<results-dir>/<EXP>.json``, chooses the EER threshold on validation, computes
the metrics of both splits with bootstrap intervals and writes them to the same
JSON. Saves ``<EXP>_roc.png`` and ``<EXP>_hist.png`` and, with more than one
config, ``comparison_roc.png`` with the test ROC of every experiment.

Usage:
    python scripts/evaluate.py --config configs/E01.yaml configs/E02.yaml configs/E04.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from facial_recognition_system.evaluation import plot_roc
from facial_recognition_system.experiments import (
    EvaluatedExperiment,
    ExperimentError,
    load_experiment_config,
    run_evaluation,
)

COMPARISON_FIGURE: str = "comparison_roc.png"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("experiments/figures"))
    parser.add_argument("--resamples", type=int, default=1000)
    return parser.parse_args(argv)


def _summary(evaluated: EvaluatedExperiment) -> str:
    threshold = evaluated.evaluation.threshold
    test = evaluated.evaluation.metrics.test
    return (
        f"{evaluated.experiment}: threshold {threshold.value:.4f} (val EER), "
        f"test EER {test.eer.value:.2%} [{test.eer.low:.2%}, {test.eer.high:.2%}], "
        f"ROC-AUC {test.roc_auc.value:.4f}, TAR@FAR=1% {test.tar_at_far.value:.2%}, "
        f"accuracy {test.accuracy.value:.2%}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluates every config given and prints a summary of each one.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.

    Returns:
        0 on success, 1 when a config is invalid or an experiment cannot be evaluated.
    """
    args = _parse_args(argv)
    evaluated_experiments: list[EvaluatedExperiment] = []
    for config_path in args.config:
        try:
            config, _ = load_experiment_config(config_path)
        except (OSError, ValidationError) as error:
            print(f"Invalid experiment config {config_path}: {error}", file=sys.stderr)
            return 1

        try:
            evaluated = run_evaluation(
                config, args.results_dir, args.figures_dir, n_resamples=args.resamples
            )
        except (OSError, ExperimentError, KeyError) as error:
            print(f"Evaluation of {config.experiment} failed: {error}", file=sys.stderr)
            return 1

        print(_summary(evaluated))
        evaluated_experiments.append(evaluated)

    if len(evaluated_experiments) > 1:
        comparison = plot_roc(
            {evaluated.experiment: evaluated.scores.test for evaluated in evaluated_experiments},
            title="Curva ROC nos pares de teste",
        )
        comparison_path = args.figures_dir / COMPARISON_FIGURE
        comparison.savefig(comparison_path, dpi=150)
        print(f"Comparison: {comparison_path}")

    print(f"Results: {args.results_dir}, figures: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
