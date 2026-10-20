"""Fine-tunes the backbone of an experiment and extracts the embeddings of the best epoch.

Partial fine-tuning of decision D5 on the subset of decision D2: only the
``trainable`` children of the network learn, through a classification head
over the training identities, and the epoch with the lowest validation EER
is kept (``checkpoints/<EXP>/best.pt``, outside Git). The best epoch then
embeds the validation and test splits, and ``experiments/results/<EXP>.json``
gets a ``training`` summary, as in scripts/extract_embeddings.py.

``--smoke`` runs 2 epochs of 2 batches, validated on 100 pairs, to catch
errors in seconds before the real run. It saves its checkpoint under
``checkpoints/smoke/`` and writes no embeddings or results.

Usage:
    python scripts/train.py --config configs/E04.yaml --smoke
    python scripts/train.py --config configs/E04.yaml
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader

from facial_recognition_system.data import ImagePair
from facial_recognition_system.evaluation import equal_error_rate
from facial_recognition_system.experiments import (
    ExperimentConfig,
    TrainingConfig,
    embed_crops,
    load_experiment_config,
    pair_paths,
    pair_scores,
    read_pairs,
    read_split,
    run_extraction,
    seed_everything,
)
from facial_recognition_system.models import FaceEmbedder, build_embedder
from facial_recognition_system.preprocessing import InvalidImageError
from facial_recognition_system.training import (
    ArcFaceHead,
    FaceDataset,
    TrainingHistory,
    build_augmentation,
    fit,
    freeze_except,
    label_indices,
    load_checkpoint,
    select_training_subset,
)

SMOKE_EPOCHS: int = 2
SMOKE_STEPS: int = 2
SMOKE_PAIRS_PER_CLASS: int = 50


def smoke_pairs(pairs: Sequence[ImagePair]) -> list[ImagePair]:
    """The first positive and negative pairs, enough for a quick EER."""
    positives = [pair for pair in pairs if pair.same][:SMOKE_PAIRS_PER_CLASS]
    negatives = [pair for pair in pairs if not pair.same][:SMOKE_PAIRS_PER_CLASS]
    return positives + negatives


def train_experiment(
    config: ExperimentConfig,
    training: TrainingConfig,
    embedder: FaceEmbedder,
    device: torch.device,
    checkpoint_path: Path,
    smoke: bool = False,
) -> tuple[TrainingHistory, dict[str, Any]]:
    """Fine-tunes ``embedder`` in place and leaves the best epoch loaded in it.

    Returns:
        The history and the ``training`` summary of the results JSON.
    """
    seed_everything(config.seed)
    dataset_config = config.dataset

    train_paths = select_training_subset(
        read_split(dataset_config.split_file("train")),
        training.subset.min_photos,
        training.subset.max_photos,
        config.seed,
    )
    labels = label_indices(train_paths)
    loader = DataLoader(
        FaceDataset(
            dataset_config.crops_dir,
            train_paths,
            labels,
            embedder.preprocess,
            build_augmentation(training.augmentation),
        ),
        batch_size=training.batch_size,
        shuffle=True,
        # last_bn is a batch norm: a final batch of one photo would fail, and a
        # tiny one would give noisy statistics.
        drop_last=True,
        num_workers=training.num_workers,
        persistent_workers=training.num_workers > 0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )

    val_pairs = read_pairs(dataset_config.pairs_file("val"))
    if smoke:
        val_pairs = smoke_pairs(val_pairs)
    val_paths = pair_paths(val_pairs)

    def validate() -> float:
        embedding_set = embed_crops(
            embedder, dataset_config.crops_dir, val_paths, config.embedding.batch_size, device
        )
        return equal_error_rate(pair_scores(embedding_set, val_pairs)).eer

    embedder = embedder.to(device)
    backbone_parameters = freeze_except(embedder.network, training.trainable)
    head = ArcFaceHead(
        embedder.embedding_dim, len(labels), training.arcface.scale, training.arcface.margin
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": training.backbone_lr},
            {"params": head.parameters(), "lr": training.head_lr},
        ],
        weight_decay=training.weight_decay,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history = fit(
        embedder,
        head,
        loader,
        validate,
        optimizer,
        trainable=training.trainable,
        max_epochs=SMOKE_EPOCHS if smoke else training.max_epochs,
        patience=training.patience,
        checkpoint_path=checkpoint_path,
        device=device,
        max_steps=SMOKE_STEPS if smoke else None,
    )
    load_checkpoint(checkpoint_path, embedder)
    embedder.to(device)

    summary = {
        "loss": training.loss.value,
        "subset": {"identities": len(labels), "photos": len(train_paths)},
        "trainable_parameters": sum(parameter.numel() for parameter in backbone_parameters),
        "total_parameters": sum(parameter.numel() for parameter in embedder.network.parameters()),
        "epochs_run": len(history.epochs) - 1,
        "best_epoch": history.best_epoch,
        "initial_val_eer": history.initial_val_eer,
        "best_val_eer": history.best_val_eer,
        "stopped_early": history.stopped_early,
        "seconds": history.seconds,
        "peak_gpu_memory_mb": (
            round(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else None
        ),
        "checkpoint": checkpoint_path.as_posix(),
        "history": [record.model_dump() for record in history.epochs],
    }
    return history, summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="2 short epochs to catch errors")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("embeddings"))
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, embedder: FaceEmbedder | None = None) -> int:
    """Trains one experiment and prints the EER of every epoch.

    Args:
        argv: Command-line arguments. Defaults to sys.argv.
        embedder: Embedder to fine-tune (injected in tests). Defaults to the config's backbone.

    Returns:
        0 on success, 1 when the config is invalid or has no ``training`` section,
        or a split, pair list or crop is missing.
    """
    args = _parse_args(argv)
    try:
        config, raw_config = load_experiment_config(args.config)
    except (OSError, ValidationError) as error:
        print(f"Invalid experiment config {args.config}: {error}", file=sys.stderr)
        return 1
    if config.training is None:
        print(f"{args.config} has no training section", file=sys.stderr)
        return 1

    if embedder is None:
        embedder = build_embedder(config.model.backbone, config.model.pretrained)
    device = torch.device(args.device)
    checkpoints_dir = args.checkpoints_dir / "smoke" if args.smoke else args.checkpoints_dir
    checkpoint_path = checkpoints_dir / config.experiment / "best.pt"

    try:
        history, summary = train_experiment(
            config, config.training, embedder, device, checkpoint_path, smoke=args.smoke
        )
        for record in history.epochs:
            loss = "-" if record.train_loss is None else f"{record.train_loss:.4f}"
            print(f"epoch {record.epoch}: loss {loss}, val EER {record.val_eer:.4f}")
        print(
            f"best epoch {history.best_epoch} (val EER {history.best_val_eer:.4f}, "
            f"pretrained {history.initial_val_eer:.4f}) -> {checkpoint_path}"
        )
        if args.smoke:
            print("Smoke run finished: no embeddings or results written.")
            return 0

        embedder.model_id = f"{embedder.model_id}-{config.training.loss}-{config.experiment}"
        record = run_extraction(
            config,
            raw_config,
            embedder,
            device,
            args.embeddings_dir,
            args.results_dir,
            training=summary,
        )
    except (OSError, InvalidImageError, KeyError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1

    for split, split_summary in record["embeddings"].items():
        print(f"{split}: {split_summary['photos']} photos -> {split_summary['file']}")
    print(f"Results: {args.results_dir / f'{config.experiment}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
