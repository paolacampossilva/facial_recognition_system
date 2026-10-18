import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from facial_recognition_system.preprocessing import BoundingBox, FaceDetection
from scripts.preprocess_faces import REPORT_FILENAME, main

FACE = FaceDetection(
    box=BoundingBox(left=10, top=10, right=50, bottom=50),
    probability=0.99,
    landmarks=((20, 25), (40, 25), (30, 32), (22, 42), (38, 42)),
)


class BlackMeansNoFaceDetector:
    """Finds one face in any image, except fully black ones."""

    def detect(self, image: Image.Image) -> list[FaceDetection]:
        if np.asarray(image).max() == 0:
            return []
        return [FACE]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "lfw"
    (root / "Ada_Lovelace").mkdir(parents=True)
    (root / "Alan_Turing").mkdir()
    Image.new("RGB", (60, 60), (200, 150, 120)).save(root / "Ada_Lovelace" / "Ada_0001.jpg")
    Image.new("RGB", (60, 60), (190, 140, 110)).save(root / "Ada_Lovelace" / "Ada_0002.jpg")
    Image.new("RGB", (60, 60), (0, 0, 0)).save(root / "Alan_Turing" / "Alan_0001.jpg")
    (root / "Alan_Turing" / "Alan_0002.jpg").write_text("corrupted")
    return root


@pytest.mark.unit
def test_main_saves_crops_and_reports_failures(dataset: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "crops"

    exit_code = main(
        ["--input-dir", str(dataset), "--output-dir", str(output_dir)],
        detector=BlackMeansNoFaceDetector(),
    )

    assert exit_code == 0
    assert Image.open(output_dir / "Ada_Lovelace" / "Ada_0001.png").size == (160, 160)
    assert (output_dir / "Ada_Lovelace" / "Ada_0002.png").exists()
    assert not (output_dir / "Alan_Turing").exists()

    report = json.loads((output_dir / REPORT_FILENAME).read_text())
    assert report["total"] == 4
    assert report["processed"] == 2
    assert report["config"]["selection"] == "strict"
    reasons = {failure["path"]: failure["reason"] for failure in report["failures"]}
    assert "no face detected" in reasons["Alan_Turing/Alan_0001.jpg"]
    assert "could not open" in reasons["Alan_Turing/Alan_0002.jpg"]


@pytest.mark.unit
def test_second_run_skips_existing_crops(dataset: Path, tmp_path: Path) -> None:
    args = ["--input-dir", str(dataset), "--output-dir", str(tmp_path / "crops")]
    main(args, detector=BlackMeansNoFaceDetector())

    main(args, detector=BlackMeansNoFaceDetector())

    report = json.loads((tmp_path / "crops" / REPORT_FILENAME).read_text())
    assert report["skipped"] == 2
    assert report["processed"] == 0


@pytest.mark.unit
def test_main_fails_when_input_folder_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        ["--input-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")],
        detector=BlackMeansNoFaceDetector(),
    )

    assert exit_code == 1
    assert "Input folder not found" in capsys.readouterr().err
