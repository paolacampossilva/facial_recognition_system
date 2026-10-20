import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from facial_recognition_system.experiments import (
    EmbeddingSet,
    embed_crops,
    load_embeddings,
    load_experiment_config,
    save_embeddings,
)
from facial_recognition_system.models import FaceEmbedder
from facial_recognition_system.preprocessing import Normalization
from scripts.extract_embeddings import main

COLORS: dict[str, tuple[int, int, int]] = {
    "Ada_Lovelace/Ada_Lovelace_0001.png": (200, 30, 30),
    "Ada_Lovelace/Ada_Lovelace_0002.png": (190, 40, 30),
    "Alan_Turing/Alan_Turing_0001.png": (30, 30, 200),
}


class MeanColorEmbedder(FaceEmbedder):
    """Embeds a face as its mean RGB color, so the expected output is known."""

    model_id = "mean-color"
    embedding_dim = 3
    normalization = Normalization.FACENET

    def extract(self, faces: torch.Tensor) -> torch.Tensor:
        return faces.mean(dim=(2, 3))


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    crops_dir = tmp_path / "crops"
    for path, color in COLORS.items():
        (crops_dir / path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 160), color).save(crops_dir / path)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "lfw_val.txt").write_text("".join(f"{path}\n" for path in COLORS))
    (splits_dir / "lfw_test.txt").write_text("Alan_Turing/Alan_Turing_0001.png\n")
    return tmp_path


def write_config(root: Path, **overrides: str) -> Path:
    fields = {
        "experiment": "E99",
        "hypothesis": "a cor média separa as pessoas",
        "strategy": "sem treino",
        "crops_dir": (root / "crops").as_posix(),
        "splits_dir": (root / "splits").as_posix(),
    } | overrides
    config_path = root / "E99.yaml"
    config_path.write_text(
        f"experiment: {fields['experiment']}\n"
        f"hypothesis: {fields['hypothesis']}\n"
        f"strategy: {fields['strategy']}\n"
        "model:\n  backbone: facenet\n  pretrained: false\n"
        "dataset:\n"
        f"  crops_dir: {fields['crops_dir']}\n"
        f"  splits_dir: {fields['splits_dir']}\n"
        "  splits: [val, test]\n"
        "embedding:\n  batch_size: 2\n",
        encoding="utf-8",
    )
    return config_path


def run(root: Path, config_path: Path) -> int:
    return main(
        [
            "--config",
            str(config_path),
            "--device",
            "cpu",
            "--embeddings-dir",
            str(root / "embeddings"),
            "--results-dir",
            str(root / "results"),
        ],
        embedder=MeanColorEmbedder(),
    )


@pytest.mark.unit
def test_main_writes_embeddings_in_contract_format(dataset: Path) -> None:
    assert run(dataset, write_config(dataset)) == 0

    val = load_embeddings(dataset / "embeddings" / "E99" / "val.npz")
    assert val.paths == list(COLORS)
    assert val.labels == ["Ada_Lovelace", "Ada_Lovelace", "Alan_Turing"]
    assert val.model_id == "mean-color"
    assert val.embeddings.dtype == np.float32
    assert val.embeddings.shape == (3, 3)
    assert np.allclose(np.linalg.norm(val.embeddings, axis=1), 1.0, atol=1e-6)
    # Same person (similar reds) is closer than a different person (blue).
    assert val.embeddings[0] @ val.embeddings[1] > val.embeddings[0] @ val.embeddings[2]

    test = load_embeddings(dataset / "embeddings" / "E99" / "test.npz")
    assert test.paths == ["Alan_Turing/Alan_Turing_0001.png"]


@pytest.mark.unit
def test_main_writes_results_json_without_metrics(dataset: Path) -> None:
    assert run(dataset, write_config(dataset)) == 0

    record = json.loads((dataset / "results" / "E99.json").read_text(encoding="utf-8"))
    assert record["experiment"] == "E99"
    assert record["hypothesis"] == "a cor média separa as pessoas"
    assert record["model"] == "mean-color"
    assert record["seed"] == 42
    assert record["config"]["embedding"] == {"batch_size": 2}
    assert {"python", "torch", "torchvision", "facenet-pytorch", "numpy"} <= set(
        record["libraries"]
    )
    assert record["hardware"]["device"] == "cpu"
    assert record["embeddings"]["val"]["photos"] == 3
    assert record["embeddings"]["val"]["identities"] == 2
    assert record["embeddings"]["val"]["dimension"] == 3
    assert record["threshold"] is None
    assert record["metrics"] is None


@pytest.mark.unit
def test_main_fails_when_a_crop_is_missing(dataset: Path) -> None:
    (dataset / "crops" / "Alan_Turing" / "Alan_Turing_0001.png").unlink()

    assert run(dataset, write_config(dataset)) == 1
    assert not (dataset / "results" / "E99.json").exists()


@pytest.mark.unit
def test_main_rejects_invalid_config(dataset: Path) -> None:
    assert run(dataset, write_config(dataset, experiment="primeiro")) == 1


@pytest.mark.unit
def test_embed_crops_of_empty_split_has_backbone_dimension(dataset: Path) -> None:
    result = embed_crops(MeanColorEmbedder(), dataset / "crops", [], 4, torch.device("cpu"))

    assert result.embeddings.shape == (0, 3)


@pytest.mark.unit
def test_embeddings_round_trip(tmp_path: Path) -> None:
    original = EmbeddingSet(
        embeddings=np.eye(2, dtype=np.float32),
        paths=["A/A_0001.png", "B/B_0001.png"],
        labels=["A", "B"],
        model_id="facenet-vggface2",
    )

    save_embeddings(original, tmp_path / "out" / "val.npz")
    loaded = load_embeddings(tmp_path / "out" / "val.npz")

    assert np.array_equal(loaded.embeddings, original.embeddings)
    assert loaded.paths == original.paths
    assert loaded.labels == original.labels
    assert loaded.model_id == original.model_id


@pytest.mark.unit
@pytest.mark.parametrize("config", ["E01.yaml", "E02.yaml", "E04.yaml"])
def test_versioned_configs_are_valid(config: str) -> None:
    experiment, raw = load_experiment_config(Path("configs") / config)

    assert experiment.experiment == Path(config).stem
    assert raw["seed"] == 42
