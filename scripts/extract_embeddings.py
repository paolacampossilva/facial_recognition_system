"""Embeds the public split lists of an experiment and records how it was run.

For each split in the config, writes ``<embeddings-dir>/<EXP>/<split>.npz`` (outside
Git) and then ``<results-dir>/<EXP>.json`` with the hypothesis, config, seed,
library versions, hardware and extraction times. ``threshold`` and ``metrics``
stay ``null``: the evaluation fills them from the embeddings. Running again
replaces both, so metrics computed from older embeddings are not kept.

Usage:
    python scripts/extract_embeddings.py --config configs/E01.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from pydantic import ValidationError

from facial_recognition_system.experiments import load_experiment_config, run_extraction
from facial_recognition_system.models import FaceEmbedder, build_embedder
from facial_recognition_system.preprocessing import InvalidImageError


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, embedder: FaceEmbedder | None = None) -> int:
    """Runs the extraction of one experiment and prints a summary.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
        embedder: Embedder to use (injected in tests). Defaults to the config's backbone.

    Returns:
        0 on success, 1 when the config is invalid or a split or crop is missing.
    """
    args = _parse_args(argv)
    try:
        config, raw_config = load_experiment_config(args.config)
    except (OSError, ValidationError) as error:
        print(f"Invalid experiment config {args.config}: {error}", file=sys.stderr)
        return 1

    if embedder is None:
        embedder = build_embedder(config.model.backbone, config.model.pretrained)

    try:
        record = run_extraction(
            config,
            raw_config,
            embedder,
            torch.device(args.device),
            args.embeddings_dir,
            args.results_dir,
        )
    except (OSError, InvalidImageError) as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1

    for split, summary in record["embeddings"].items():
        print(
            f"{split}: {summary['photos']} photos, {summary['identities']} identities, "
            f"D = {summary['dimension']}, {summary['seconds']} s -> {summary['file']}"
        )
    print(f"Results: {args.results_dir / f'{config.experiment}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
