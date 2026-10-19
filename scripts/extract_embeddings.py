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
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from facial_recognition_system.experiments import (
    ExperimentConfig,
    embed_crops,
    hardware_info,
    library_versions,
    load_experiment_config,
    read_split,
    save_embeddings,
    seed_everything,
)
from facial_recognition_system.models import FaceEmbedder, build_embedder
from facial_recognition_system.preprocessing import InvalidImageError


def run_extraction(
    config: ExperimentConfig,
    raw_config: dict[str, Any],
    embedder: FaceEmbedder,
    device: torch.device,
    embeddings_dir: Path,
    results_dir: Path,
) -> dict[str, Any]:
    """Embeds every split of ``config`` and writes the results JSON.

    Returns:
        The results record, also written to ``results_dir / <EXP>.json``.
    """
    seed_everything(config.seed)
    embedder = embedder.to(device).eval()

    splits: dict[str, Any] = {}
    for split in config.dataset.splits:
        paths = read_split(config.dataset.split_file(split))
        start = time.perf_counter()
        embedding_set = embed_crops(
            embedder, config.dataset.crops_dir, paths, config.embedding.batch_size, device
        )
        seconds = time.perf_counter() - start

        output_path = embeddings_dir / config.experiment / f"{split}.npz"
        save_embeddings(embedding_set, output_path)
        splits[split] = {
            "file": output_path.as_posix(),
            "photos": len(embedding_set.paths),
            "identities": len(set(embedding_set.labels)),
            "dimension": int(embedding_set.embeddings.shape[1]),
            "seconds": round(seconds, 2),
        }

    record = {
        "experiment": config.experiment,
        "hypothesis": config.hypothesis,
        "model": embedder.model_id,
        "strategy": config.strategy,
        "dataset": config.dataset.name,
        "seed": config.seed,
        "config": raw_config,
        "libraries": library_versions(),
        "hardware": hardware_info(device),
        "embeddings": splits,
        "training": None,
        "threshold": None,
        "metrics": None,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{config.experiment}.json"
    results_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return record


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
