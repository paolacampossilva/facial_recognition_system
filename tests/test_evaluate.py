import csv
import json
from pathlib import Path

import numpy as np
import pytest

from facial_recognition_system.evaluation import (
    ThresholdCriterion,
    VerificationScores,
    choose_eer_threshold,
    evaluate_experiment,
)
from facial_recognition_system.experiments import EmbeddingSet, save_embeddings
from scripts.evaluate import main

IDENTITIES: int = 12
PHOTOS_PER_IDENTITY: int = 3
DIMENSION: int = 8
RESAMPLES: int = 20


def _scores(genuine: np.ndarray, impostor: np.ndarray) -> VerificationScores:
    return VerificationScores(
        scores=np.concatenate([genuine, impostor]),
        labels=np.concatenate([np.ones(len(genuine), int), np.zeros(len(impostor), int)]),
    )


def _write_split(root: Path, split: str, seed: int) -> None:
    """Embeds each identity around its own direction and writes the npz and the pairs CSV."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(IDENTITIES, DIMENSION))
    paths: list[str] = []
    rows: list[np.ndarray] = []
    for identity in range(IDENTITIES):
        for photo in range(PHOTOS_PER_IDENTITY):
            paths.append(f"{split}_{identity:02d}/{split}_{identity:02d}_{photo:04d}.png")
            rows.append(centers[identity] + rng.normal(scale=0.6, size=DIMENSION))
    embeddings = np.array(rows, dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    save_embeddings(
        EmbeddingSet(
            embeddings=embeddings,
            paths=paths,
            labels=[path.split("/")[0] for path in paths],
            model_id="synthetic",
        ),
        root / "embeddings" / "E99" / f"{split}.npz",
    )

    with (root / "splits" / f"lfw_{split}_pairs.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path_a", "path_b", "same"])
        for identity in range(IDENTITIES):
            first = identity * PHOTOS_PER_IDENTITY
            writer.writerow([paths[first], paths[first + 1], 1])
            other = ((identity + 1) % IDENTITIES) * PHOTOS_PER_IDENTITY
            writer.writerow([paths[first + 2], paths[other], 0])


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """An extracted E99: config, embeddings, pairs and a results JSON without metrics."""
    (tmp_path / "splits").mkdir()
    _write_split(tmp_path, "val", seed=1)
    _write_split(tmp_path, "test", seed=2)

    (tmp_path / "E99.yaml").write_text(
        "experiment: E99\n"
        "hypothesis: direções aleatórias separam as pessoas\n"
        "strategy: sem treino\n"
        "model:\n  backbone: facenet\n"
        f"dataset:\n  splits_dir: {(tmp_path / 'splits').as_posix()}\n",
        encoding="utf-8",
    )
    record = {
        "experiment": "E99",
        "model": "synthetic",
        "embeddings": {
            split: {"file": (tmp_path / "embeddings" / "E99" / f"{split}.npz").as_posix()}
            for split in ("val", "test")
        },
        "threshold": None,
        "metrics": None,
    }
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "E99.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def run(root: Path, *configs: Path) -> int:
    return main(
        [
            "--config",
            *(str(config) for config in configs),
            "--results-dir",
            str(root / "results"),
            "--figures-dir",
            str(root / "figures"),
            "--resamples",
            str(RESAMPLES),
        ]
    )


def read_record(root: Path) -> dict:
    return json.loads((root / "results" / "E99.json").read_text(encoding="utf-8"))


@pytest.mark.unit
def test_evaluate_experiment_applies_the_validation_threshold_to_test() -> None:
    val = _scores(np.linspace(0.3, 1.0, 101), np.linspace(0.0, 0.7, 101))
    # Test is shifted up, so its own EER threshold would be different.
    test = _scores(np.linspace(0.5, 1.2, 101), np.linspace(0.2, 0.9, 101))

    evaluation = evaluate_experiment(val, test, n_resamples=RESAMPLES)

    assert evaluation.threshold == choose_eer_threshold(val)
    assert evaluation.metrics.val.threshold == evaluation.threshold.value
    assert evaluation.metrics.test.threshold == evaluation.threshold.value
    assert evaluation.metrics.test.n_resamples == RESAMPLES


@pytest.mark.unit
def test_main_fills_threshold_and_metrics_and_keeps_the_other_fields(experiment: Path) -> None:
    assert run(experiment, experiment / "E99.yaml") == 0

    record = read_record(experiment)
    assert record["experiment"] == "E99"
    assert record["model"] == "synthetic"
    assert record["embeddings"]["val"]["file"].endswith("val.npz")
    assert record["threshold"]["criterion"] == ThresholdCriterion.EER
    assert record["threshold"]["chosen_on"] == "val"
    for split in ("val", "test"):
        metrics = record["metrics"][split]
        assert metrics["threshold"] == record["threshold"]["value"]
        assert metrics["n_resamples"] == RESAMPLES
        for name in ("roc_auc", "eer", "tar_at_far", "accuracy", "f1"):
            assert metrics[name]["low"] <= metrics[name]["value"] <= metrics[name]["high"]


@pytest.mark.unit
def test_main_saves_the_figures_of_the_contract(experiment: Path) -> None:
    assert run(experiment, experiment / "E99.yaml") == 0

    assert sorted(path.name for path in (experiment / "figures").iterdir()) == [
        "E99_hist.png",
        "E99_roc.png",
    ]


@pytest.mark.unit
def test_main_saves_the_comparison_roc_with_more_than_one_experiment(experiment: Path) -> None:
    config = (experiment / "E99.yaml").read_text(encoding="utf-8")
    (experiment / "E98.yaml").write_text(config.replace("E99", "E98"), encoding="utf-8")
    record = (experiment / "results" / "E99.json").read_text(encoding="utf-8")
    (experiment / "results" / "E98.json").write_text(record, encoding="utf-8")

    assert run(experiment, experiment / "E98.yaml", experiment / "E99.yaml") == 0
    assert (experiment / "figures" / "comparison_roc.png").exists()


@pytest.mark.unit
def test_main_fails_when_the_results_json_has_no_test_embeddings(experiment: Path) -> None:
    record = read_record(experiment)
    del record["embeddings"]["test"]
    (experiment / "results" / "E99.json").write_text(json.dumps(record), encoding="utf-8")

    assert run(experiment, experiment / "E99.yaml") == 1
    assert read_record(experiment)["metrics"] is None


@pytest.mark.unit
def test_main_fails_when_the_experiment_was_not_extracted(experiment: Path) -> None:
    (experiment / "results" / "E99.json").unlink()

    assert run(experiment, experiment / "E99.yaml") == 1


@pytest.mark.unit
def test_main_fails_when_a_pair_uses_a_photo_without_embedding(experiment: Path) -> None:
    pairs_path = experiment / "splits" / "lfw_test_pairs.csv"
    with pairs_path.open("a", newline="") as file:
        csv.writer(file).writerow(["Nobody/Nobody_0001.png", "test_00/test_00_0000.png", 0])

    assert run(experiment, experiment / "E99.yaml") == 1
    assert read_record(experiment)["metrics"] is None


@pytest.mark.unit
def test_main_rejects_invalid_config(experiment: Path) -> None:
    (experiment / "E99.yaml").write_text("experiment: primeiro\n", encoding="utf-8")

    assert run(experiment, experiment / "E99.yaml") == 1
