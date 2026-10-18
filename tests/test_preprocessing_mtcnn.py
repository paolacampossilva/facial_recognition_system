from pathlib import Path

import pytest
from PIL import Image

from facial_recognition_system.preprocessing import (
    AlignmentMode,
    FacePreprocessor,
    FaceSelection,
    MTCNNDetector,
    NoFaceDetectedError,
    PreprocessingConfig,
    load_image,
)

LFW_SAMPLE: Path = Path("data/public/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")


@pytest.fixture(scope="module")
def detector() -> MTCNNDetector:
    return MTCNNDetector(device="cpu")


@pytest.mark.integration
def test_mtcnn_finds_no_face_in_blank_image(detector: MTCNNDetector) -> None:
    preprocessor = FacePreprocessor(detector, PreprocessingConfig())

    with pytest.raises(NoFaceDetectedError):
        preprocessor.process(Image.new("RGB", (250, 250), (128, 128, 128)))


@pytest.mark.integration
@pytest.mark.parametrize("alignment", list(AlignmentMode))
def test_mtcnn_crops_lfw_face(detector: MTCNNDetector, alignment: AlignmentMode) -> None:
    if not LFW_SAMPLE.exists():
        pytest.skip("LFW not downloaded; run scripts/download_datasets.py")
    config = PreprocessingConfig(selection=FaceSelection.CENTER, alignment=alignment)

    face = FacePreprocessor(detector, config).process(load_image(LFW_SAMPLE))

    assert face.size == (160, 160)
