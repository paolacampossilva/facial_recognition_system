"""Runs E07 and E08 on the private dataset and writes their aggregated results.

Builds the private split (D9), removes session 2 photos that repeat a gallery
photo (two-stage dHash), embeds the private crops plus the LFW extra unknowns,
runs the strategy selection (E07) and the threshold study (E08), and writes
``<EXP>.json`` for both. Embeddings go to ``<embeddings_dir>/<E07>/`` (inside
the vault). The JSONs and the figures only hold aggregated numbers and
anonymous IDs.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

from facial_recognition_system.data import (
    PRIVATE_SPLIT_ORDER,
    LabeledPhoto,
    LeakageConfig,
    PrivateSplit,
    PrivateSplitConfig,
    build_private_split,
    check_leakage,
    exclude_private_photos,
    find_private_crops,
    hash_photos,
)
from facial_recognition_system.evaluation import (
    UNKNOWN_LABEL,
    plot_confusion_matrix,
    plot_far_frr,
)
from facial_recognition_system.experiments.embeddings import (
    EmbeddingSet,
    embed_crops,
    read_split,
    save_embeddings,
)
from facial_recognition_system.experiments.environment import (
    hardware_info,
    library_versions,
    seed_everything,
)
from facial_recognition_system.experiments.open_set import (
    LabeledEmbeddings,
    OpenSetConfig,
    StrategySelection,
    ThresholdStudy,
    build_strategies,
    select_strategy,
    study_threshold,
)
from facial_recognition_system.models import FaceEmbedder

QUERY_SPLITS: tuple[str, str] = ("val", "test")


class SplitSummary(BaseModel):
    """Counts of one part of the private split, with no ID or path."""

    photos: int
    known_photos: int
    unknown_photos: int
    lfw_unknown_photos: int


class OpenSetRun(BaseModel):
    selection: StrategySelection
    study: ThresholdStudy
    gallery_participants: int
    removed_duplicates: int
    splits: dict[str, SplitSummary]


def prepare_private_split(config: OpenSetConfig) -> tuple[PrivateSplit, int]:
    """Builds the private split and drops the later copies of near-duplicate photos.

    Returns:
        The split and how many photos the leakage check removed.
    """
    photos = find_private_crops(config.private.crops_dir)
    split = build_private_split(photos, PrivateSplitConfig(seed=config.seed))
    leakage_config = LeakageConfig()
    hashed = hash_photos(
        split.paths_by_split(),
        config.private.originals_dir,
        config.private.crops_dir,
        config.private.original_suffixes,
        leakage_config,
    )
    report = check_leakage(hashed, PRIVATE_SPLIT_ORDER, leakage_config)
    return exclude_private_photos(split, report.excluded), len(report.excluded)


def _embed(
    embedder: FaceEmbedder,
    crops_dir: Path,
    photos: list[LabeledPhoto],
    config: OpenSetConfig,
    device: torch.device,
) -> EmbeddingSet:
    embedding_set = embed_crops(
        embedder, crops_dir, [photo.path for photo in photos], config.embedding.batch_size, device
    )
    # embed_crops labels by parent folder ("P01/session2"); the split label is the identity.
    return embedding_set.model_copy(update={"labels": [photo.label for photo in photos]})


def embed_private_split(
    split: PrivateSplit,
    config: OpenSetConfig,
    embedder: FaceEmbedder,
    device: torch.device,
    embeddings_dir: Path,
) -> dict[str, EmbeddingSet]:
    """Embeds the gallery and the val and test attempts, LFW extra unknowns included.

    Saves each part to ``<embeddings_dir>/<E07>/<part>.npz``.
    """
    embedder = embedder.to(device).eval()
    parts: dict[str, EmbeddingSet] = {}
    for part in PRIVATE_SPLIT_ORDER:
        private = _embed(embedder, config.private.crops_dir, getattr(split, part), config, device)
        if part in QUERY_SPLITS:
            lfw_paths = read_split(config.dataset.splits_dir / f"lfw_{part}_unknowns.txt")
            lfw_photos = [LabeledPhoto(path=path, label=UNKNOWN_LABEL) for path in lfw_paths]
            lfw = _embed(embedder, config.dataset.crops_dir, lfw_photos, config, device)
            private = EmbeddingSet(
                embeddings=np.concatenate([private.embeddings, lfw.embeddings]),
                paths=private.paths + lfw.paths,
                labels=private.labels + lfw.labels,
                model_id=private.model_id,
            )
        save_embeddings(private, embeddings_dir / config.selection_experiment / f"{part}.npz")
        parts[part] = private
    return parts


def _summary(split: PrivateSplit, part: str, embedding_set: EmbeddingSet) -> SplitSummary:
    private_photos: list[LabeledPhoto] = getattr(split, part)
    unknown_private = sum(photo.label == UNKNOWN_LABEL for photo in private_photos)
    return SplitSummary(
        photos=len(embedding_set.labels),
        known_photos=len(private_photos) - unknown_private,
        unknown_photos=unknown_private,
        lfw_unknown_photos=len(embedding_set.labels) - len(private_photos),
    )


def _record(
    experiment: str,
    hypothesis: str,
    strategy: str,
    config: OpenSetConfig,
    raw_config: dict[str, Any],
    model_id: str,
    device: torch.device,
    run: OpenSetRun,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "hypothesis": hypothesis,
        "model": model_id,
        "source_experiment": config.source_experiment,
        "strategy": strategy,
        "dataset": "private + lfw unknowns",
        "seed": config.seed,
        "config": raw_config,
        "libraries": library_versions(),
        "hardware": hardware_info(device),
        "split": {
            "gallery_participants": run.gallery_participants,
            "removed_duplicates": run.removed_duplicates,
            "parts": {part: summary.model_dump() for part, summary in run.splits.items()},
        },
    }


def write_results(
    run: OpenSetRun,
    config: OpenSetConfig,
    raw_config: dict[str, Any],
    model_id: str,
    device: torch.device,
    results_dir: Path,
) -> None:
    """Writes the E07 and E08 JSONs. E07 keeps no confusion matrix, only the rates."""
    selection, study = run.selection, run.study
    selection_record = _record(
        config.selection_experiment,
        config.selection_hypothesis,
        "comparação de estratégias na galeria privada",
        config,
        raw_config,
        model_id,
        device,
        run,
    ) | {
        "threshold": selection.chosen.threshold.model_dump(mode="json"),
        "metrics": {
            "chosen": {"strategy": selection.chosen.strategy, "params": selection.chosen.params},
            "candidates": [
                {
                    "strategy": candidate.strategy,
                    "params": candidate.params,
                    "threshold": candidate.threshold.model_dump(mode="json"),
                    "val": candidate.val.model_dump(mode="json", exclude={"confusion"}),
                }
                for candidate in selection.candidates
            ],
        },
    }
    study_record = _record(
        config.threshold_experiment,
        config.threshold_hypothesis,
        f"threshold open-set da estratégia {study.strategy}",
        config,
        raw_config,
        model_id,
        device,
        run,
    ) | {
        "threshold": study.threshold.model_dump(mode="json"),
        "metrics": {
            "strategy": study.strategy,
            "params": study.params,
            "eer_threshold": study.eer_threshold.model_dump(mode="json"),
            "val_curve": [point.model_dump() for point in study.val_curve],
            "val": study.val.model_dump(mode="json"),
            "test": study.test.model_dump(mode="json"),
            "test_at_eer": study.test_at_eer.model_dump(mode="json", exclude={"confusion"}),
        },
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    for record in (selection_record, study_record):
        path = results_dir / f"{record['experiment']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", "utf-8")


def save_open_set_figures(study: ThresholdStudy, experiment: str, figures_dir: Path) -> None:
    """Saves ``<EXP>_confusion.png`` (test) and ``<EXP>_threshold.png`` (validation curve)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    confusion = plot_confusion_matrix(
        study.test.confusion, title=f"{experiment}: matriz de confusão no teste"
    )
    curve = plot_far_frr(
        [point.threshold for point in study.val_curve],
        [point.far for point in study.val_curve],
        [point.frr for point in study.val_curve],
        study.threshold.value,
        title=f"{experiment}: FAR e FRR na validação ({study.strategy})",
    )
    confusion.savefig(figures_dir / f"{experiment}_confusion.png", dpi=150)
    curve.savefig(figures_dir / f"{experiment}_threshold.png", dpi=150)


def run_open_set(
    config: OpenSetConfig,
    raw_config: dict[str, Any],
    embedder: FaceEmbedder,
    device: torch.device,
    embeddings_dir: Path,
    results_dir: Path,
    figures_dir: Path,
) -> OpenSetRun:
    """Runs E07 and E08 end to end and writes their results and figures.

    Args:
        config: Validated E07/E08 config.
        raw_config: The YAML as read, stored in the JSONs.
        embedder: The model of ``config.source_experiment``, weights already loaded.
        device: Device the embedder runs on.
        embeddings_dir: Private embeddings folder (inside the vault).
        results_dir: Folder of the results JSONs.
        figures_dir: Folder of the figures.

    Returns:
        The selection, the threshold study and the split counts.

    Raises:
        SplitError: The private dataset cannot be split (too few participants,
            missing session, non-anonymous folder).
        FileNotFoundError: A crop, an original or a split list is missing.
    """
    seed_everything(config.seed)
    split, removed = prepare_private_split(config)
    parts = embed_private_split(split, config, embedder, device, embeddings_dir)
    labeled = {
        part: LabeledEmbeddings(embeddings=embedding_set.embeddings, labels=embedding_set.labels)
        for part, embedding_set in parts.items()
    }

    strategies = build_strategies(config.strategies, config.seed)
    selection = select_strategy(strategies, labeled["gallery"], labeled["val"], config.threshold)
    chosen = strategies[selection.candidates.index(selection.chosen)]
    study = study_threshold(
        chosen, labeled["gallery"], labeled["val"], labeled["test"], config.threshold
    )

    run = OpenSetRun(
        selection=selection,
        study=study,
        gallery_participants=len(split.gallery_participants),
        removed_duplicates=removed,
        splits={part: _summary(split, part, parts[part]) for part in PRIVATE_SPLIT_ORDER},
    )
    write_results(run, config, raw_config, embedder.model_id, device, results_dir)
    save_open_set_figures(study, config.threshold_experiment, figures_dir)
    return run
