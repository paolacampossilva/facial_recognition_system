import csv
from pathlib import Path

import pytest

from scripts.make_splits import main

SPLIT_FILES: tuple[str, ...] = (
    "lfw_train.txt",
    "lfw_val.txt",
    "lfw_test.txt",
    "lfw_val_pairs.csv",
    "lfw_test_pairs.csv",
    "lfw_val_unknowns.txt",
    "lfw_test_unknowns.txt",
)


@pytest.fixture
def crops_dir(tmp_path: Path) -> Path:
    root = tmp_path / "lfw_crop160"
    counts = {f"Person_{index:02d}": 2 + index % 6 for index in range(30)}
    counts |= {f"Single_{index:02d}": 1 for index in range(10)}
    for name, count in counts.items():
        (root / name).mkdir(parents=True)
        for index in range(1, count + 1):
            (root / name / f"{name}_{index:04d}.png").touch()
    (root / "preprocessing_report.json").write_text("{}")
    return root


@pytest.mark.unit
def test_main_writes_every_split_file(
    crops_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "splits"

    exit_code = main(
        [
            "--crops-dir",
            str(crops_dir),
            "--output-dir",
            str(output_dir),
            "--unknowns-per-split",
            "3",
        ]
    )

    assert exit_code == 0
    assert sorted(path.name for path in output_dir.iterdir()) == sorted(SPLIT_FILES)
    assert "train:" in capsys.readouterr().out

    train = (output_dir / "lfw_train.txt").read_text().splitlines()
    assert train[0].startswith("Person_")
    assert all((crops_dir / path).is_file() for path in train)

    with (output_dir / "lfw_test_pairs.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == ["path_a", "path_b", "same"]
    assert {row["same"] for row in rows} == {"0", "1"}

    unknowns = (output_dir / "lfw_val_unknowns.txt").read_text().splitlines()
    assert len(unknowns) == 3


@pytest.mark.unit
def test_main_fails_when_crops_folder_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--crops-dir", str(tmp_path / "missing")])

    assert exit_code == 1
    assert "Crops folder not found" in capsys.readouterr().err


@pytest.mark.unit
def test_main_fails_when_unknowns_are_not_enough(
    crops_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "--crops-dir",
            str(crops_dir),
            "--output-dir",
            str(tmp_path / "splits"),
            "--unknowns-per-split",
            "50",
        ]
    )

    assert exit_code == 1
    assert "Cannot build the splits" in capsys.readouterr().err
    assert not (tmp_path / "splits").exists()
