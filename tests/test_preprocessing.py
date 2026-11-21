from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from facial_recognition_system.preprocessing import (
    AlignmentMode,
    BoundingBox,
    DetectedFace,
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


def encoded(image: Image.Image, image_format: str, **options: object) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=image_format, **options)
    return buffer.getvalue()


@pytest.mark.unit
def test_load_image_accepts_the_bytes_of_an_image() -> None:
    image = load_image(encoded(Image.new("RGB", (30, 20), RED), "PNG"))

    assert image.mode == "RGB"
    assert image.size == (30, 20)
    assert dominant_color(image) == RED


@pytest.mark.unit
def test_load_image_converts_bytes_of_other_modes_to_rgb() -> None:
    image = load_image(encoded(Image.new("L", (8, 8), 128), "PNG"))

    assert image.mode == "RGB"


@pytest.mark.unit
def test_load_image_applies_exif_orientation_to_bytes() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed

    image = load_image(encoded(Image.new("RGB", (100, 50), RED), "JPEG", exif=exif))

    assert image.size == (50, 100)


@pytest.mark.unit
@pytest.mark.parametrize("content", [b"", b"plain text", b"\x89PNG\r\n\x1a\n-truncated"])
def test_load_image_rejects_bytes_that_are_not_an_image(content: bytes) -> None:
    with pytest.raises(InvalidImageError) as error:
        load_image(content)

    assert error.value.source == "upload"


@pytest.mark.unit
def test_invalid_image_error_keeps_the_path_it_came_from(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jpg"

    with pytest.raises(InvalidImageError) as error:
        load_image(missing)

    assert error.value.source == missing


@pytest.mark.unit
def test_load_image_rejects_an_image_too_large_to_be_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    content = encoded(Image.new("RGB", (50, 50), RED), "PNG")

    with pytest.raises(InvalidImageError):
        load_image(content)


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


MIN_FACE_PX = 80
MAX_FACES = 20


def group_image() -> Image.Image:
    """600x300 image for the boxes of a group photo; each face is 100 px wide."""
    return Image.new("RGB", (600, 300), BLUE)


def faces_in(
    detections: list[FaceDetection], image: Image.Image | None = None
) -> list[DetectedFace]:
    preprocessor = FacePreprocessor(FakeDetector(detections), PreprocessingConfig())
    return preprocessor.process_all(image or group_image(), MIN_FACE_PX, MAX_FACES)


@pytest.mark.unit
def test_process_all_with_no_face_raises_the_same_error_as_process() -> None:
    with pytest.raises(NoFaceDetectedError):
        faces_in([])


@pytest.mark.unit
def test_process_all_with_one_face_gives_the_crop_of_process() -> None:
    detection = make_detection(100, 10, 200, 100)
    image = two_color_image()
    preprocessor = FacePreprocessor(FakeDetector([detection]), PreprocessingConfig())

    (face,) = preprocessor.process_all(image, MIN_FACE_PX, MAX_FACES)

    assert np.array_equal(np.asarray(face.crop), np.asarray(preprocessor.process(image)))
    assert face.crop.size == (160, 160)
    assert face.probability == detection.probability


@pytest.mark.unit
def test_process_all_gives_every_face_from_left_to_right() -> None:
    right = make_detection(400, 100, 500, 200)
    left = make_detection(50, 100, 150, 200)
    middle = make_detection(220, 100, 320, 200)

    faces = faces_in([right, left, middle])

    assert [face.box.left for face in faces] == [50, 220, 400]
    assert all(face.crop.size == (160, 160) for face in faces)


@pytest.mark.unit
def test_faces_on_the_same_left_edge_go_from_top_to_bottom() -> None:
    lower = make_detection(100, 180, 200, 280)
    upper = make_detection(100, 10, 200, 110)

    faces = faces_in([lower, upper])

    assert [face.box.top for face in faces] == [10, 180]


@pytest.mark.unit
def test_faces_below_the_minimum_probability_are_ignored() -> None:
    sure = make_detection(50, 100, 150, 200, probability=0.95)
    unsure = make_detection(300, 100, 400, 200, probability=0.89)

    assert [face.box.left for face in faces_in([sure, unsure])] == [50]
    with pytest.raises(NoFaceDetectedError):
        faces_in([unsure])


@pytest.mark.unit
def test_a_face_shorter_than_the_minimum_side_is_ignored() -> None:
    big = make_detection(50, 100, 150, 200)
    narrow = make_detection(300, 100, 379, 200)  # 79 px wide
    exactly = make_detection(450, 100, 530, 200)  # 80 px wide

    faces = faces_in([big, narrow, exactly])

    assert [face.box.left for face in faces] == [50, 450]


@pytest.mark.unit
def test_when_every_face_is_too_small_it_is_no_face() -> None:
    with pytest.raises(NoFaceDetectedError):
        faces_in([make_detection(50, 100, 100, 150)])


@pytest.mark.unit
def test_only_the_biggest_faces_are_kept_up_to_the_limit() -> None:
    sizes = [90 + index for index in range(25)]  # 25 faces, the biggest are the last ones
    detections = [
        make_detection(index * 20, 0, index * 20 + size, size) for index, size in enumerate(sizes)
    ]
    image = Image.new("RGB", (1000, 300), BLUE)

    faces = faces_in(detections, image)

    assert len(faces) == MAX_FACES
    assert [face.box.left for face in faces] == [index * 20 for index in range(5, 25)]


@pytest.mark.unit
@pytest.mark.parametrize("max_faces", [0, -1])
def test_process_all_needs_room_for_at_least_one_face(max_faces: int) -> None:
    preprocessor = FacePreprocessor(
        FakeDetector([make_detection(50, 100, 150, 200)]), PreprocessingConfig()
    )

    with pytest.raises(ValueError, match="max_faces"):
        preprocessor.process_all(group_image(), MIN_FACE_PX, max_faces)


@pytest.mark.unit
def test_the_box_is_whole_pixels_inside_the_image() -> None:
    crossing = make_detection(-7.4, 10.6, 603.2, 190.4)  # fractional, past both vertical edges

    (face,) = faces_in([crossing])

    assert (face.box.left, face.box.top, face.box.right, face.box.bottom) == (0, 11, 600, 190)


@pytest.mark.unit
def test_the_minimum_side_and_the_ranking_use_the_limited_box() -> None:
    mostly_outside = make_detection(-50, 100, 30, 200)  # 80 px wide raw, 30 px inside the image
    inside = make_detection(300, 100, 400, 200)

    faces = faces_in([mostly_outside, inside])

    assert [face.box.left for face in faces] == [300]


@pytest.mark.unit
def test_a_box_that_ends_up_outside_the_image_is_ignored() -> None:
    with pytest.raises(NoFaceDetectedError):
        faces_in([make_detection(700, 100, 800, 200)])


@pytest.mark.unit
def test_the_strict_path_still_rejects_what_process_all_accepts() -> None:
    detections = [make_detection(50, 100, 150, 200), make_detection(300, 100, 400, 200)]
    config = PreprocessingConfig(selection=FaceSelection.STRICT)
    preprocessor = FacePreprocessor(FakeDetector(detections), config)

    with pytest.raises(MultipleFacesError):
        preprocessor.process(group_image())
    assert len(preprocessor.process_all(group_image(), MIN_FACE_PX, MAX_FACES)) == 2


@pytest.mark.unit
@pytest.mark.parametrize("selection", list(FaceSelection))
def test_the_selection_does_not_change_process_all(selection: FaceSelection) -> None:
    detections = [make_detection(50, 100, 150, 200), make_detection(300, 100, 400, 200)]
    preprocessor = FacePreprocessor(
        FakeDetector(detections), PreprocessingConfig(selection=selection)
    )

    assert len(preprocessor.process_all(group_image(), MIN_FACE_PX, MAX_FACES)) == 2


@pytest.mark.unit
def test_process_all_works_with_the_similarity_alignment() -> None:
    config = PreprocessingConfig(alignment=AlignmentMode.SIMILARITY)
    preprocessor = FacePreprocessor(FakeDetector([make_detection(100, 10, 200, 100)]), config)

    (face,) = preprocessor.process_all(two_color_image(), MIN_FACE_PX, MAX_FACES)

    assert face.crop.size == (160, 160)


class SizeRecordingDetector:
    """Remembers the size of the image it was given and puts one face inside it."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def detect(self, image: Image.Image) -> list[FaceDetection]:
        self.seen.append(image.size)
        width, height = image.size
        return [make_detection(0, 0, width, height)]


@pytest.mark.unit
def test_the_detector_sees_the_photo_in_its_exif_orientation() -> None:
    """The boxes of process_all are in the orientation the browser shows (spec 001, A15)."""
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    upload = encoded(Image.new("RGB", (400, 200), RED), "JPEG", exif=exif)
    detector = SizeRecordingDetector()
    preprocessor = FacePreprocessor(detector, PreprocessingConfig())

    (face,) = preprocessor.process_all(load_image(upload), MIN_FACE_PX, MAX_FACES)

    assert detector.seen == [(200, 400)]
    assert (face.box.right, face.box.bottom) == (200, 400)
