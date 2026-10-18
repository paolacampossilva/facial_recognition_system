from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from facial_recognition_system.preprocessing import (
    AlignmentMode,
    BoundingBox,
    FaceDetection,
    FacePreprocessor,
    FaceSelection,
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
    Normalization,
    PreprocessingConfig,
    load_image,
    to_tensor,
)
from facial_recognition_system.preprocessing.alignment import (
    ARCFACE_TEMPLATE,
    TEMPLATE_SIZE,
    align_face,
    estimate_similarity_transform,
)

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def make_detection(
    left: float, top: float, right: float, bottom: float, probability: float = 0.99
) -> FaceDetection:
    width = right - left
    height = bottom - top
    landmarks = tuple(
        (left + width * x / TEMPLATE_SIZE, top + height * y / TEMPLATE_SIZE)
        for x, y in ARCFACE_TEMPLATE
    )
    return FaceDetection(
        box=BoundingBox(left=left, top=top, right=right, bottom=bottom),
        probability=probability,
        landmarks=landmarks,
    )


class FakeDetector:
    def __init__(self, detections: list[FaceDetection]) -> None:
        self.detections = detections

    def detect(self, image: Image.Image) -> list[FaceDetection]:
        return self.detections


def two_color_image() -> Image.Image:
    """200x100 image: red on the left half, blue on the right half."""
    image = Image.new("RGB", (200, 100), BLUE)
    image.paste(RED, (0, 0, 100, 100))
    return image


def dominant_color(image: Image.Image) -> tuple[int, int, int]:
    pixels = np.asarray(image).reshape(-1, 3)
    red, green, blue = (int(value) for value in np.median(pixels, axis=0))
    return (red, green, blue)


@pytest.mark.unit
def test_single_face_returns_square_rgb_crop() -> None:
    preprocessor = FacePreprocessor(
        FakeDetector([make_detection(10, 10, 90, 90)]), PreprocessingConfig()
    )

    face = preprocessor.process(two_color_image())

    assert face.size == (160, 160)
    assert face.mode == "RGB"


@pytest.mark.unit
def test_crop_keeps_only_the_face_box() -> None:
    preprocessor = FacePreprocessor(
        FakeDetector([make_detection(110, 10, 190, 90)]), PreprocessingConfig()
    )

    face = preprocessor.process(two_color_image())

    assert dominant_color(face) == BLUE


@pytest.mark.unit
def test_grayscale_input_produces_rgb_crop() -> None:
    preprocessor = FacePreprocessor(
        FakeDetector([make_detection(10, 10, 90, 90)]), PreprocessingConfig()
    )

    face = preprocessor.process(Image.new("L", (100, 100), 128))

    assert face.mode == "RGB"


@pytest.mark.unit
def test_no_detection_raises() -> None:
    preprocessor = FacePreprocessor(FakeDetector([]), PreprocessingConfig())

    with pytest.raises(NoFaceDetectedError):
        preprocessor.process(two_color_image())


@pytest.mark.unit
def test_low_probability_faces_are_ignored() -> None:
    detector = FakeDetector([make_detection(10, 10, 90, 90, probability=0.5)])
    preprocessor = FacePreprocessor(detector, PreprocessingConfig(min_probability=0.9))

    with pytest.raises(NoFaceDetectedError):
        preprocessor.process(two_color_image())


@pytest.mark.unit
def test_multiple_faces_with_strict_selection_raises() -> None:
    detector = FakeDetector([make_detection(10, 10, 90, 90), make_detection(110, 10, 190, 90)])
    preprocessor = FacePreprocessor(detector, PreprocessingConfig(selection=FaceSelection.STRICT))

    with pytest.raises(MultipleFacesError) as error:
        preprocessor.process(two_color_image())

    assert error.value.face_count == 2


@pytest.mark.unit
def test_center_selection_keeps_face_closest_to_image_center() -> None:
    off_center = make_detection(0, 0, 60, 60)
    centered = make_detection(105, 25, 155, 75)
    detector = FakeDetector([off_center, centered])
    preprocessor = FacePreprocessor(detector, PreprocessingConfig(selection=FaceSelection.CENTER))

    face = preprocessor.process(two_color_image())

    assert dominant_color(face) == BLUE


@pytest.mark.unit
def test_largest_selection_keeps_biggest_face() -> None:
    small = make_detection(110, 40, 130, 60)
    large = make_detection(5, 5, 95, 95)
    detector = FakeDetector([small, large])
    preprocessor = FacePreprocessor(detector, PreprocessingConfig(selection=FaceSelection.LARGEST))

    face = preprocessor.process(two_color_image())

    assert dominant_color(face) == RED


@pytest.mark.unit
def test_similarity_alignment_returns_requested_size() -> None:
    config = PreprocessingConfig(alignment=AlignmentMode.SIMILARITY, image_size=112)
    preprocessor = FacePreprocessor(FakeDetector([make_detection(10, 10, 90, 90)]), config)

    face = preprocessor.process(two_color_image())

    assert face.size == (112, 112)


@pytest.mark.unit
def test_similarity_transform_recovers_known_transform() -> None:
    angle = np.deg2rad(20)
    scale = 1.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = np.array([12.0, -5.0])
    source = ARCFACE_TEMPLATE
    target = source @ (scale * rotation).T + translation

    matrix = estimate_similarity_transform(source, target)

    np.testing.assert_allclose(matrix[:, :2], scale * rotation, atol=1e-9)
    np.testing.assert_allclose(matrix[:, 2], translation, atol=1e-9)


@pytest.mark.unit
def test_align_face_is_identity_when_landmarks_already_match_template() -> None:
    gradient = np.tile(np.arange(112, dtype=np.uint8) * 2, (112, 1))
    image = Image.fromarray(np.stack([gradient, gradient.T, gradient], axis=-1))
    landmarks = tuple((float(x), float(y)) for x, y in ARCFACE_TEMPLATE)

    aligned = align_face(image, landmarks, image_size=112)

    difference = np.abs(np.asarray(aligned, dtype=int) - np.asarray(image, dtype=int))
    assert difference[2:-2, 2:-2].max() <= 2


@pytest.mark.unit
def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValidationError):
        PreprocessingConfig(image_size=0)


@pytest.mark.unit
def test_load_image_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    Image.new("RGB", (100, 50), RED).save(path, exif=exif)

    image = load_image(path)

    assert image.size == (50, 100)


@pytest.mark.unit
def test_load_image_rejects_non_image_file(tmp_path: Path) -> None:
    path = tmp_path / "not_an_image.jpg"
    path.write_text("plain text")

    with pytest.raises(InvalidImageError):
        load_image(path)


@pytest.mark.unit
def test_load_image_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "missing.jpg")


@pytest.mark.unit
def test_facenet_normalization_maps_pixels_to_unit_range() -> None:
    face = Image.new("RGB", (4, 4), (0, 255, 128))

    tensor = to_tensor(face, Normalization.FACENET)

    assert tensor.shape == (3, 4, 4)
    assert tensor[0, 0, 0].item() == pytest.approx(-127.5 / 128)
    assert tensor[1, 0, 0].item() == pytest.approx(127.5 / 128)
    assert tensor[2, 0, 0].item() == pytest.approx(0.5 / 128)


@pytest.mark.unit
def test_imagenet_normalization_uses_channel_statistics() -> None:
    face = Image.new("RGB", (4, 4), (255, 0, 0))

    tensor = to_tensor(face, Normalization.IMAGENET)

    assert tensor[0, 0, 0].item() == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-5)
    assert tensor[1, 0, 0].item() == pytest.approx(-0.456 / 0.224, rel=1e-5)
