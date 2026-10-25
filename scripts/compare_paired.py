"""Compares the fine-tuned models with the E01 baseline on the same test pairs (D5).

Prints the numbers of "Impacto do fine-tuning" in the README: errors at each
model's validation threshold, the reference errors each model fixes and the new
ones it makes, the errors shared by all models, the EER difference with a paired
bootstrap interval, the relative change of the trainable weights and the mean
cosine between the embeddings of the same photo. Needs the evaluated JSONs and
the test embeddings; the weight change also needs the checkpoints of
``train.py`` and is skipped without them.

Usage:
    python scripts/compare_paired.py --reference configs/E01.yaml \
        --config configs/E04.yaml configs/E06.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from facial_recognition_system.experiments import load_experiment_config
from facial_recognition_system.experiments.paired_analysis import (
    PairedAnalysis,
    run_paired_analysis,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _points(fraction: float) -> str:
    return f"{fraction * 100:+.2f} p.p."


def _report(analysis: PairedAnalysis) -> str:
    lines = [f"Test pairs: {analysis.test_pairs}, reference {analysis.reference}"]
    first = analysis.results[0].errors.reference if analysis.results else None
    if first is not None:
        lines.append(
            f"{analysis.reference}: {first.errors} errors "
            f"({first.false_accepts} false accepts + {first.false_rejects} false rejects)"
        )
    for result in analysis.results:
        errors, difference = result.errors, result.eer_difference
        lines.append(
            f"{result.experiment}: {errors.other.errors} errors "
            f"({errors.other.false_accepts} + {errors.other.false_rejects}), "
            f"fixes {errors.fixed} / introduces {errors.introduced}, "
            f"repeats {errors.shared} of {errors.reference.errors}"
        )
        lines.append(
            f"  EER difference {_points(difference.observed)} "
            f"(bootstrap mean {_points(difference.bootstrap_mean)}, "
            f"{difference.confidence:.0%} CI {_points(difference.low)} "
            f"to {_points(difference.high)}, "
            f"lower EER than {analysis.reference} in {difference.share_other_better:.0%} "
            f"of {difference.n_resamples} resamples)"
        )
        change = result.weight_change
        weights = (
            "checkpoint not found, weight change skipped"
            if change is None
            else f"trainable weights moved {change.overall:.1%} overall, "
            f"at most {change.largest_layer:.1%} in one layer; "
            f"frozen weights {'unchanged' if change.frozen_unchanged else 'CHANGED'}"
        )
        lines.append(f"  {weights}")
        lines.append(
            f"  mean cosine to the {analysis.reference} embedding of the same photo: "
            f"{result.mean_cosine_to_reference:.3f}"
        )
    lines.append(f"Errors shared by all models: {analysis.errors_shared_by_all}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        reference, _ = load_experiment_config(args.reference)
        others = [load_experiment_config(path)[0] for path in args.config]
    except (OSError, ValidationError) as error:
        print(f"Invalid config: {error}", file=sys.stderr)
        return 1

    try:
        analysis = run_paired_analysis(
            reference, others, args.results_dir, n_resamples=args.resamples, seed=args.seed
        )
    except (OSError, KeyError, ValueError) as error:
        print(f"Paired analysis failed: {error}", file=sys.stderr)
        return 1

    print(_report(analysis))
    return 0


if __name__ == "__main__":
    sys.exit(main())
