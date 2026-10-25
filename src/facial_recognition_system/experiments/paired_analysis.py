"""Paired analysis of the fine-tuned models against the E01 baseline (D5).

Reproduces the numbers of "Impacto do fine-tuning" in the README and of D5:
errors and their overlap on the test pairs, the EER difference with a paired
bootstrap interval, how much the trainable weights moved and how similar the
embedding of the same photo is before and after fine-tuning. Uses only public
LFW data: the evaluated JSONs, the test embeddings and the checkpoints.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel

from facial_recognition_system.evaluation.paired import (
    ErrorComparison,
    PairedEerDifference,
    compare_errors,
    error_mask,
    paired_eer_difference,
)
from facial_recognition_system.experiments.config import ExperimentConfig
from facial_recognition_system.experiments.embeddings import EmbeddingSet, load_embeddings
from facial_recognition_system.experiments.evaluation import load_experiment_scores
from facial_recognition_system.models import build_embedder


class WeightChange(BaseModel):
    """Relative change of the trainable weights, ``||after - before|| / ||before||``."""

    overall: float
    largest_layer: float
    frozen_unchanged: bool


class PairedResult(BaseModel):
    experiment: str
    errors: ErrorComparison
    eer_difference: PairedEerDifference
    weight_change: WeightChange | None
    mean_cosine_to_reference: float


class PairedAnalysis(BaseModel):
    reference: str
    test_pairs: int
    results: list[PairedResult]
    errors_shared_by_all: int


def relative_weight_change(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    trainable: Sequence[str],
) -> WeightChange:
    """Measures how much fine-tuning moved the trainable weights.

    Args:
        before: Parameters of the pretrained network, by name.
        after: Parameters of the fine-tuned network, by name.
        trainable: Top-level modules that were trained (``block8``, ...).

    Returns:
        The relative change of all trainable parameters together, the largest
        relative change of a single weight matrix or kernel, and whether every
        frozen parameter stayed identical.
    """
    changed_squares = original_squares = 0.0
    largest = 0.0
    frozen_unchanged = True
    for name, original in before.items():
        updated = after[name]
        if name.split(".", 1)[0] not in trainable:
            frozen_unchanged = frozen_unchanged and torch.equal(original, updated)
            continue
        delta = (updated - original).double()
        changed_squares += float(delta.pow(2).sum())
        original_squares += float(original.double().pow(2).sum())
        if original.ndim > 1 and float(original.norm()) > 0:
            largest = max(largest, float(delta.norm() / original.double().norm()))
    return WeightChange(
        overall=(changed_squares / original_squares) ** 0.5 if original_squares else 0.0,
        largest_layer=largest,
        frozen_unchanged=frozen_unchanged,
    )


def mean_cosine(reference: EmbeddingSet, other: EmbeddingSet) -> float:
    """Mean cosine between the embeddings of the same photos in two L2-normalized sets."""
    if reference.paths != other.paths:
        raise ValueError("both embedding sets must list the same photos in the same order")
    return float(np.mean(np.sum(reference.embeddings * other.embeddings, axis=1)))


def _record(config: ExperimentConfig, results_dir: Path) -> dict:
    return json.loads((results_dir / f"{config.experiment}.json").read_text(encoding="utf-8"))


def _weight_change(config: ExperimentConfig, record: dict) -> WeightChange | None:
    checkpoint_path = Path((record.get("training") or {}).get("checkpoint", ""))
    if config.training is None or not checkpoint_path.is_file():
        return None
    pretrained = build_embedder(config.model.backbone, config.model.pretrained).network
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    before = dict(pretrained.named_parameters())
    after = {name: checkpoint["network"][name] for name in before}
    return relative_weight_change(
        {name: value.detach() for name, value in before.items()}, after, config.training.trainable
    )


def run_paired_analysis(
    reference: ExperimentConfig,
    others: Sequence[ExperimentConfig],
    results_dir: Path,
    n_resamples: int = 1000,
    seed: int = 42,
) -> PairedAnalysis:
    """Compares each model in ``others`` with ``reference`` on the test pairs.

    One generator seeded with ``seed`` draws the bootstrap resamples of every
    comparison, in the order of ``others``, so the same call always gives the
    same intervals. The weight change is skipped when the checkpoint listed in
    the JSON is missing (it is not versioned and comes from ``train.py``).

    Args:
        reference: Config of the reference experiment (E01).
        others: Configs of the compared experiments (E04, E06).
        results_dir: Folder of the evaluated ``<EXP>.json``.
        n_resamples: Bootstrap resamples per comparison.
        seed: Seed of the bootstrap generator.

    Returns:
        The comparison of each model and the errors shared by all of them.
    """
    rng = np.random.default_rng(seed)
    reference_record = _record(reference, results_dir)
    reference_scores = load_experiment_scores(reference, reference_record).test
    reference_threshold = reference_record["threshold"]["value"]
    reference_embeddings = load_embeddings(Path(reference_record["embeddings"]["test"]["file"]))

    shared = error_mask(reference_scores, reference_threshold)
    results = []
    for config in others:
        record = _record(config, results_dir)
        scores = load_experiment_scores(config, record).test
        threshold = record["threshold"]["value"]
        embeddings = load_embeddings(Path(record["embeddings"]["test"]["file"]))
        shared &= error_mask(scores, threshold)
        results.append(
            PairedResult(
                experiment=config.experiment,
                errors=compare_errors(reference_scores, reference_threshold, scores, threshold),
                eer_difference=paired_eer_difference(
                    reference_scores, scores, rng, n_resamples=n_resamples
                ),
                weight_change=_weight_change(config, record),
                mean_cosine_to_reference=mean_cosine(reference_embeddings, embeddings),
            )
        )
    return PairedAnalysis(
        reference=reference.experiment,
        test_pairs=len(reference_scores.labels),
        results=results,
        errors_shared_by_all=int(shared.sum()),
    )
