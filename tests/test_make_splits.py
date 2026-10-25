import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.make_splits import LEAKAGE_REPORT_FILENAME, main

SPLIT_FILES: tuple[str, ...] = (
    "lfw_train.txt",
    "lfw_val.txt",
    "lfw_test.txt",
    "lfw_val_pairs.csv",
    "lfw_test_pairs.csv",
    "lfw_val_unknowns.txt",
    "lfw_test_unknowns.txt",
    LEAKAGE_REPORT_FILENAME,
)


def _noise(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8))


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Original photos and face crops; every single-photo identity is the same photo."""
    originals, crops = tmp_path / "lfw", tmp_path / "lfw_crop160"
    counts = {f"Person_{index:02d}": 2 + index % 6 for index in range(30)}
    counts |= {f"Single_{index:02d}": 1 for index in range(10)}
    seed = 0
    for name, count in counts.items():
        (originals / name).mkdir(parents=True)
        (crops / name).mkdir(parents=True)
        for index in range(1, count + 1):
            seed += 1
            image = _noise(0 if name.startswith("Single_") else seed)
            image.save(originals / name / f"{name}_{index:04d}.jpg", quality=95)
            image.save(crops / name / f"{name}_{index:04d}.png")
    (crops / "preprocessing_report.json").write_text("{}")
    return originals, crops


def _args(dataset: tuple[Path, Path], output_dir: Path, unknowns: int) -> list[str]:
    originals, crops = dataset
    return [
        "--originals-dir",
        str(originals),
        "--crops-dir",
        str(crops),
        "--output-dir",
        str(output_dir),
        "--unknowns-per-split",
        str(unknowns),
    ]


@pytest.mark.unit
def test_main_writes_every_split_file(
    dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "splits"

    exit_code = main(_args(dataset, output_dir, unknowns=3))

    assert exit_code == 0
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(SPLIT_FILES)
    assert "train:" in capsys.readouterr().out

    train = (output_dir / "lfw_train.txt").read_text().splitlines()
    assert train[0].startswith("Person_")
    assert all((dataset[1] / path).is_file() for path in train)

    with (output_dir / "lfw_test_pairs.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == ["path_a", "path_b", "same"]
    assert {row["same"] for row in rows} == {"0", "1"}


@pytest.mark.unit
def test_main_removes_test_copies_of_validation_photos(
    dataset: tuple[Path, Path], tmp_path: Path
) -> None:
    output_dir = tmp_path / "splits"

    exit_code = main(_args(dataset, output_dir, unknowns=2))

    report = json.loads((output_dir / LEAKAGE_REPORT_FILENAME).read_text())
    test = (output_dir / "lfw_test.txt").read_text().splitlines()
    val = (output_dir / "lfw_val.txt").read_text().splitlines()
    assert exit_code == 0
    assert report["excluded"]
    assert all(path.startswith("Single_") for path in report["excluded"])
    assert not set(report["excluded"]) & set(test + val)
    assert any(path.startswith("Single_") for path in val)
    assert not any(path.startswith("Single_") for path in test)


@pytest.mark.unit
def test_main_fails_when_crops_folder_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--crops-dir", str(tmp_path / "missing")])

    assert exit_code == 1
    assert "Folder not found" in capsys.readouterr().err


@pytest.mark.unit
def test_main_fails_when_unknowns_are_not_enough(
    dataset: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(_args(dataset, tmp_path / "splits", unknowns=50))

    assert exit_code == 1
    assert "Cannot build the splits" in capsys.readouterr().err
    assert not (tmp_path / "splits").exists()
