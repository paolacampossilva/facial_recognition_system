import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from facial_recognition_system.experiments import load_open_set_config
from facial_recognition_system.models import FaceEmbedder
from facial_recognition_system.preprocessing import Normalization
from scripts.run_open_set import main

PARTICIPANTS: dict[str, tuple[int, int, int]] = {
    "P01": (220, 40, 40),
    "P02": (40, 220, 40),
    "P03": (40, 40, 220),
    "P04": (220, 220, 40),
    "P05": (40, 220, 220),
    "P06": (220, 40, 220),
}
# Dark colors point away from every participant (mid-gray would embed as a near-zero vector).
LFW_UNKNOWN_COLORS: list[tuple[int, int, int]] = [(30, 30, 30), (45, 35, 30), (30, 40, 45)]
PHOTOS_PER_SESSION: int = 5


class MeanColorEmbedder(FaceEmbedder):
    """Embeds a face as its mean RGB color, so each participant is a cluster."""

    model_id = "mean-color"
    embedding_dim = 3
    normalization = Normalization.FACENET

    def extract(self, faces: torch.Tensor) -> torch.Tensor:
        return faces.mean(dim=(2, 3))


def _noisy_image(color: tuple[int, int, int], rng: np.random.Generator) -> Image.Image:
    # Noise gives every photo its own dHash, so only real copies count as leakage.
    pixels = np.clip(np.array(color) + rng.normal(scale=25, size=(160, 160, 3)), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A fake vault with originals and crops of six participants, plus LFW unknowns."""
    rng = np.random.default_rng(0)
    for participant, color in PARTICIPANTS.items():
        for session in ("session1", "session2"):
            for photo in range(PHOTOS_PER_SESSION):
                image = _noisy_image(color, rng)
                original = tmp_path / "data/private" / participant / session / f"{photo}.jpg"
                crop = tmp_path / "faces/private" / participant / session / f"{photo}.png"
                original.parent.mkdir(parents=True, exist_ok=True)
                crop.parent.mkdir(parents=True, exist_ok=True)
                image.save(original)
                image.save(crop)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    for split in ("val", "test"):
        paths = []
        for index, color in enumerate(LFW_UNKNOWN_COLORS):
            path = f"Lfw_{split}_{index}/Lfw_{split}_{index}_0001.png"
            (tmp_path / "lfw" / path).parent.mkdir(parents=True, exist_ok=True)
            _noisy_image(color, rng).save(tmp_path / "lfw" / path)
            paths.append(path)
        (splits_dir / f"lfw_{split}_unknowns.txt").write_text("\n".join(paths) + "\n")

    (tmp_path / "E07_E08.yaml").write_text(
        "selection_hypothesis: a estratégia mais simples basta\n"
        "threshold_hypothesis: FAR de 1% com FRR baixo\n"
        "source_experiment: E01\n"
        "model:\n  backbone: facenet\n  pretrained: false\n"
        "private:\n"
        f"  crops_dir: {(tmp_path / 'faces/private').as_posix()}\n"
        f"  originals_dir: {(tmp_path / 'data/private').as_posix()}\n"
        "dataset:\n"
        f"  crops_dir: {(tmp_path / 'lfw').as_posix()}\n"
        f"  splits_dir: {splits_dir.as_posix()}\n"
        "embedding:\n  batch_size: 8\n",
        encoding="utf-8",
    )
    return tmp_path


def run(root: Path) -> int:
    return main(
        [
            "--config",
            str(root / "E07_E08.yaml"),
            "--device",
            "cpu",
            "--embeddings-dir",
            str(root / "embeddings/private"),
            "--results-dir",
            str(root / "results"),
            "--figures-dir",
            str(root / "figures"),
        ],
        embedder=MeanColorEmbedder(),
    )


@pytest.mark.unit
def test_main_writes_aggregated_results_of_e07_and_e08(workspace: Path) -> None:
    assert run(workspace) == 0

    selection = json.loads((workspace / "results/E07.json").read_text(encoding="utf-8"))
    study = json.loads((workspace / "results/E08.json").read_text(encoding="utf-8"))
    assert [c["strategy"] for c in selection["metrics"]["candidates"]] == [
        "nearest_neighbor",
        "prototype",
        "knn",
        "svm",
        "svm",
        "svm",
    ]
    assert study["metrics"]["strategy"] == selection["metrics"]["chosen"]["strategy"]
    assert study["threshold"]["chosen_on"] == "val"
    assert study["split"]["gallery_participants"] == 4
    assert study["split"]["parts"]["val"]["lfw_unknown_photos"] == len(LFW_UNKNOWN_COLORS)
    # Well-separated colors: no known person is taken for someone else.
    assert study["metrics"]["test"]["misidentification_rate"] == 0.0
    assert "counts" in study["metrics"]["val"]
    assert "counts" in study["metrics"]["test"]
    assert "counts" not in study["metrics"]["test_at_eer"]
    assert all("counts" not in c["val"] for c in selection["metrics"]["candidates"])


@pytest.mark.unit
def test_results_hold_no_photo_path_and_no_participant_id(workspace: Path) -> None:
    assert run(workspace) == 0

    for experiment in ("E07", "E08"):
        text = (workspace / f"results/{experiment}.json").read_text(encoding="utf-8")
        assert "session1/" not in text
        assert "session2/" not in text
        assert "Lfw_" not in text
        assert re.search(r"P\d{2,}", text) is None


@pytest.mark.unit
def test_main_saves_embeddings_with_identity_labels_and_figures(workspace: Path) -> None:
    assert run(workspace) == 0

    with np.load(workspace / "embeddings/private/E07/gallery.npz") as gallery:
        assert set(gallery["labels"].tolist()) <= set(PARTICIPANTS)
    assert (workspace / "figures/E08_confusion.png").exists()
    assert (workspace / "figures/E08_threshold.png").exists()


@pytest.mark.unit
def test_main_removes_a_session2_copy_of_a_gallery_photo(workspace: Path) -> None:
    for root, suffix in (("data/private", ".jpg"), ("faces/private", ".png")):
        for participant in PARTICIPANTS:
            source = workspace / root / participant / "session1" / f"0{suffix}"
            shutil.copy(source, workspace / root / participant / "session2" / f"copy{suffix}")

    assert run(workspace) == 0

    study = json.loads((workspace / "results/E08.json").read_text(encoding="utf-8"))
    # Copies of gallery participants are dropped; the unknowns' copies stay in one split.
    assert study["split"]["removed_duplicates"] == 4


@pytest.mark.unit
def test_main_fails_with_too_few_participants(workspace: Path) -> None:
    for participant in ("P04", "P05", "P06"):
        shutil.rmtree(workspace / "faces/private" / participant)

    assert run(workspace) == 1
    assert not (workspace / "results").exists()


@pytest.mark.unit
def test_main_fails_without_naming_a_non_anonymous_folder(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace / "faces/private/P06").rename(workspace / "faces/private/Maria_Silva")

    assert run(workspace) == 1
    assert "Maria" not in capsys.readouterr().err


@pytest.mark.unit
def test_main_rejects_invalid_config(workspace: Path) -> None:
    (workspace / "E07_E08.yaml").write_text("source_experiment: primeiro\n", encoding="utf-8")

    assert run(workspace) == 1


@pytest.mark.unit
def test_versioned_open_set_config_is_valid() -> None:
    config, raw = load_open_set_config(Path("configs/E07_E08.yaml"))

    assert (config.selection_experiment, config.threshold_experiment) == ("E07", "E08")
    assert config.threshold.max_far == 0.01
    assert raw["seed"] == 42
