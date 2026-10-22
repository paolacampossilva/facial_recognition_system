"""Runs E07 (comparison strategy) and E08 (open-set threshold) on the private dataset.

Run only on the group's machine, with the vault open: it reads the private crops
from ``faces/private/`` and the originals from ``data/private/`` (for the leakage
check between sessions), writes the embeddings to ``embeddings/private/<E07>/``
and, outside the vault, only aggregated results: ``<results-dir>/E07.json``,
``E08.json`` and the figures ``E08_confusion.png`` and ``E08_threshold.png``.

Usage:
    python scripts/preprocess_faces.py --input-dir data/private --output-dir faces/private
    python scripts/run_open_set.py --config configs/E07_E08.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import ValidationError

from facial_recognition_system.data import SplitError
from facial_recognition_system.experiments import load_open_set_config, run_open_set
from facial_recognition_system.models import FaceEmbedder, build_embedder
from facial_recognition_system.preprocessing import InvalidImageError
from facial_recognition_system.training import load_checkpoint


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings/private"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("experiments/figures"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, embedder: FaceEmbedder | None = None) -> int:
    """Runs E07 and E08 and prints an aggregated summary.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
        embedder: Embedder to use (injected in tests). Defaults to the config's
            backbone, with the checkpoint of the source experiment when given.

    Returns:
        0 on success, 1 when the config is invalid or the private data cannot be used.
    """
    args = _parse_args(argv)
    try:
        config, raw_config = load_open_set_config(args.config)
    except (OSError, ValidationError) as error:
        print(f"Invalid open-set config {args.config}: {error}", file=sys.stderr)
        return 1

    if embedder is None:
        embedder = build_embedder(config.model.backbone, config.model.pretrained)
        if config.model.checkpoint is not None:
            load_checkpoint(config.model.checkpoint, embedder)
            embedder.model_id = f"{embedder.model_id}-{config.source_experiment}"

    try:
        run = run_open_set(
            config,
            raw_config,
            embedder,
            torch.device(args.device),
            args.embeddings_dir,
            args.results_dir,
            args.figures_dir,
        )
    except (OSError, SplitError, InvalidImageError) as error:
        print(f"Open-set run failed: {error}", file=sys.stderr)
        return 1

    print(
        f"Gallery: {run.gallery_participants} participants, "
        f"{run.removed_duplicates} near-duplicate(s) removed"
    )
    for part, summary in run.splits.items():
        print(
            f"{part}: {summary.photos} photos ({summary.known_photos} known, "
            f"{summary.unknown_photos} private unknown, {summary.lfw_unknown_photos} LFW unknown)"
        )
    print(f"{config.selection_experiment} (validation, each at its own D8 threshold):")
    for candidate in run.selection.candidates:
        print(
            f"  {candidate.strategy} {candidate.params}: F1 macro {candidate.val.f1_macro:.3f}, "
            f"accuracy {candidate.val.accuracy:.3f}, threshold {candidate.threshold.value:.4f} "
            f"({candidate.threshold.criterion})"
        )
    study = run.study
    print(
        f"{config.threshold_experiment} ({study.strategy}, threshold {study.threshold.value:.4f}, "
        f"{study.threshold.criterion}): test accuracy {study.test.accuracy:.3f}, "
        f"F1 macro {study.test.f1_macro:.3f}, FAR {study.test.far}, FRR {study.test.frr}, "
        f"misidentification {study.test.misidentification_rate}"
    )
    print(f"Results: {args.results_dir}, figures: {args.figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
