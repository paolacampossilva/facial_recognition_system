"""Embedding extraction of an experiment and its results JSON (contract between the slices).

For each split in the config, writes ``<embeddings_dir>/<EXP>/<split>.npz`` and
then ``<results_dir>/<EXP>.json`` with the hypothesis, config, seed, library
versions, hardware and extraction times. ``threshold`` and ``metrics`` stay
``null``: the evaluation fills them from the embeddings. Running again replaces
both, so metrics computed from older embeddings are not kept.
"""

import json
import time
from pathlib import Path
from typing import Any

import torch

from facial_recognition_system.experiments.config import ExperimentConfig
from facial_recognition_system.experiments.embeddings import (
    embed_crops,
    read_split,
    save_embeddings,
)
from facial_recognition_system.experiments.environment import (
    hardware_info,
    library_versions,
    seed_everything,
)
from facial_recognition_system.models import FaceEmbedder


def run_extraction(
    config: ExperimentConfig,
    raw_config: dict[str, Any],
    embedder: FaceEmbedder,
    device: torch.device,
    embeddings_dir: Path,
    results_dir: Path,
    training: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Embeds every split of ``config`` and writes the results JSON.

    Args:
        training: Summary of the fine-tuning, for experiments that train (E04, E06).

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
        "training": training,
        "threshold": None,
        "metrics": None,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{config.experiment}.json"
    results_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", "utf-8")
    return record
