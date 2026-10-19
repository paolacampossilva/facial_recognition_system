from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.check_leakage import main


def _write_photo(originals: Path, crops: Path, name: str, seed: int) -> str:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 256, (48, 48, 3), dtype=np.uint8))
    identity = name.split("_")[0]
    (originals / identity).mkdir(parents=True, exist_ok=True)
    (crops / identity).mkdir(parents=True, exist_ok=True)
    image.save(originals / identity / f"{name}.jpg", quality=95)
    image.save(crops / identity / f"{name}.png")
    return f"{identity}/{name}.png"


def _run(tmp_path: Path, seeds: dict[str, int]) -> int:
    originals, crops, splits_dir = tmp_path / "lfw", tmp_path / "crops", tmp_path / "splits"
    splits_dir.mkdir()
    for split, seed in seeds.items():
        path = _write_photo(originals, crops, f"{split.title()}_0001", seed)
        (splits_dir / f"lfw_{split}.txt").write_text(f"{path}\n")
    return main(
        [
            "--originals-dir",
            str(originals),
            "--crops-dir",
            str(crops),
            "--splits-dir",
            str(splits_dir),
        ]
    )


@pytest.mark.unit
def test_main_passes_when_no_photo_crosses_splits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run(tmp_path, {"train": 1, "val": 2, "test": 3})

    assert exit_code == 0
    assert "No near-duplicates" in capsys.readouterr().out


@pytest.mark.unit
def test_main_fails_and_lists_the_photo_shared_by_two_splits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run(tmp_path, {"train": 1, "val": 2, "test": 1})

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "train:Train/Train_0001.png = test:Test/Test_0001.png" in stderr


@pytest.mark.unit
def test_main_fails_when_a_split_list_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--splits-dir", str(tmp_path)])

    assert exit_code == 1
    assert "Failed to read" in capsys.readouterr().err
