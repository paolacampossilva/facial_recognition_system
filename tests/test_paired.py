from pathlib import Path

import numpy as np
import pytest
import torch

from facial_recognition_system.evaluation import VerificationScores
from facial_recognition_system.evaluation.paired import (
    compare_errors,
    error_counts,
    paired_eer_difference,
)
from facial_recognition_system.experiments import EmbeddingSet
from facial_recognition_system.experiments.paired_analysis import (
    mean_cosine,
    relative_weight_change,
)
from scripts.compare_paired import main


def _scores(scores: list[float], labels: list[int]) -> VerificationScores:
    return VerificationScores(scores=np.array(scores), labels=np.array(labels))


LABELS = [1, 1, 1, 0, 0, 0]


@pytest.mark.unit
def test_error_counts_split_false_accepts_and_rejects() -> None:
    counts = error_counts(_scores([0.9, 0.4, 0.8, 0.6, 0.2, 0.1], LABELS), threshold=0.5)

    assert (counts.errors, counts.false_accepts, counts.false_rejects) == (2, 1, 1)


@pytest.mark.unit
def test_compare_errors_counts_fixed_introduced_and_shared() -> None:
    # Reference misses pairs 1 and 3; the other model misses pairs 3 and 5.
    reference = _scores([0.9, 0.4, 0.8, 0.6, 0.2, 0.1], LABELS)
    other = _scores([0.9, 0.7, 0.8, 0.6, 0.2, 0.55], LABELS)

    comparison = compare_errors(reference, 0.5, other, 0.5)

    assert (comparison.fixed, comparison.introduced, comparison.shared) == (1, 1, 1)
    assert comparison.reference.errors == comparison.other.errors == 2


@pytest.mark.unit
def test_compare_errors_rejects_different_pairs() -> None:
    with pytest.raises(ValueError, match="same pairs"):
        compare_errors(_scores([0.9, 0.1], [1, 0]), 0.5, _scores([0.9, 0.1, 0.8], [1, 0, 1]), 0.5)


@pytest.mark.unit
def test_paired_eer_difference_of_identical_models_is_zero() -> None:
    scores = _scores([0.9, 0.4, 0.8, 0.6, 0.2, 0.1], LABELS)

    difference = paired_eer_difference(scores, scores, np.random.default_rng(0), n_resamples=50)

    assert difference.observed == difference.low == difference.high == 0
    assert difference.share_other_better == 0


@pytest.mark.unit
def test_paired_eer_difference_is_negative_when_other_separates_better() -> None:
    reference = _scores([0.9, 0.4, 0.8, 0.6, 0.2, 0.1], LABELS)
    other = _scores([0.9, 0.7, 0.8, 0.3, 0.2, 0.1], LABELS)

    difference = paired_eer_difference(reference, other, np.random.default_rng(0), n_resamples=50)

    assert difference.observed < 0
    assert difference.low <= difference.bootstrap_mean <= difference.high


@pytest.mark.unit
def test_paired_eer_difference_is_reproducible_with_the_same_seed() -> None:
    reference = _scores([0.9, 0.4, 0.8, 0.6, 0.2, 0.1], LABELS)
    other = _scores([0.9, 0.7, 0.8, 0.3, 0.2, 0.1], LABELS)

    first = paired_eer_difference(reference, other, np.random.default_rng(42), n_resamples=30)
    second = paired_eer_difference(reference, other, np.random.default_rng(42), n_resamples=30)

    assert first == second


@pytest.mark.unit
def test_relative_weight_change_measures_only_trainable_modules() -> None:
    before = {"block8.weight": torch.ones(2, 2), "conv.weight": torch.ones(2, 2)}
    after = {"block8.weight": torch.full((2, 2), 1.1), "conv.weight": torch.ones(2, 2)}

    change = relative_weight_change(before, after, trainable=["block8"])

    assert change.overall == pytest.approx(0.1)
    assert change.largest_layer == pytest.approx(0.1)
    assert change.frozen_unchanged


@pytest.mark.unit
def test_relative_weight_change_flags_a_changed_frozen_module() -> None:
    before = {"block8.weight": torch.ones(2, 2), "conv.weight": torch.ones(2, 2)}
    after = {"block8.weight": torch.ones(2, 2), "conv.weight": torch.zeros(2, 2)}

    assert not relative_weight_change(before, after, trainable=["block8"]).frozen_unchanged


def _embeddings(vectors: list[list[float]], paths: list[str]) -> EmbeddingSet:
    return EmbeddingSet(
        embeddings=np.array(vectors, dtype=np.float32),
        paths=paths,
        labels=[path.split("/")[0] for path in paths],
        model_id="test",
    )


@pytest.mark.unit
def test_mean_cosine_of_the_same_photos() -> None:
    reference = _embeddings([[1, 0], [0, 1]], ["a/1.png", "b/1.png"])
    other = _embeddings([[1, 0], [1, 0]], ["a/1.png", "b/1.png"])

    assert mean_cosine(reference, other) == pytest.approx(0.5)


@pytest.mark.unit
def test_mean_cosine_rejects_different_photos() -> None:
    reference = _embeddings([[1, 0]], ["a/1.png"])
    other = _embeddings([[1, 0]], ["b/1.png"])

    with pytest.raises(ValueError, match="same photos"):
        mean_cosine(reference, other)


@pytest.mark.unit
def test_main_fails_cleanly_with_a_missing_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    assert main(["--reference", str(missing), "--config", str(missing)]) == 1
