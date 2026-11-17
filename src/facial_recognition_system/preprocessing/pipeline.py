"""Image to face crop: load, detect, select one face and crop or align it."""

from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from math import dist
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from facial_recognition_system.preprocessing.alignment import (
    AlignmentMode,
    align_face,
    crop_face,
)
from facial_recognition_system.preprocessing.detection import (
    BoundingBox,
    FaceDetection,
    FaceDetector,
)
from facial_recognition_system.preprocessing.errors import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)


class FaceSelection(StrEnum):
    """What to do when an image has more than one face.

    ``strict`` rejects the image (private photos, application), ``center`` keeps
    the face closest to the image center (LFW, whose photos are centered on the
    labeled person) and ``largest`` keeps the biggest box.
    """

    STRICT = "strict"
    CENTER = "center"
    LARGEST = "largest"


class PreprocessingConfig(BaseModel):
    """Parameters of the preprocessing pipeline, saved with every run."""

    model_config = ConfigDict(frozen=True)

    image_size: int = Field(default=160, gt=0)
    margin: int = Field(default=0, ge=0)
    min_probability: float = Field(default=0.9, ge=0.0, le=1.0)
    selection: FaceSelection = FaceSelection.STRICT
    alignment: AlignmentMode = AlignmentMode.CROP


UPLOAD_SOURCE: str = "upload"


@dataclass(frozen=True)
class DetectedFace:
    """One face found by `FacePreprocessor.process_all`: where it is and its crop.

    A dataclass, like `FaceDetection`, because it carries a PIL image that nothing validates.
    """

    # In pixels of the image after the EXIF orientation, whole numbers inside the image.
    box: BoundingBox
    probability: float
    crop: Image.Image


def limit_to_image(box: BoundingBox, width: int, height: int) -> BoundingBox | None:
    """Rounds a detector box to whole pixels and keeps it inside the image.

    The detector can return fractional values and a box that crosses the border (only the
    cropping clamps it), so the box is fixed before it is measured or shown to anyone.

    Returns:
        The box, or None when nothing of it is left inside the image.
    """
    left = max(0, round(box.left))
    top = max(0, round(box.top))
    right = min(width, round(box.right))
    bottom = min(height, round(box.bottom))
    if right <= left or bottom <= top:
        return None
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def load_image(source: Path | bytes) -> Image.Image:
    """Opens an image as RGB, applying the EXIF orientation of phone photos.

    Args:
        source: A file path, or the bytes of an uploaded or captured image.

    Raises:
        InvalidImageError: The file is missing, empty, corrupted, too large to be safe or not an
            image.
    """
    opened = BytesIO(source) if isinstance(source, bytes) else source
    try:
        with Image.open(opened) as image:
            oriented = ImageOps.exif_transpose(image)
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImageError(UPLOAD_SOURCE if isinstance(source, bytes) else source) from error


class FacePreprocessor:
    """Runs detection, face selection and cropping/alignment on one image."""

    def __init__(self, detector: FaceDetector, config: PreprocessingConfig) -> None:
        self._detector = detector
        self._config = config

    def process(self, image: Image.Image) -> Image.Image:
        """Returns the square face crop of the single selected face.

        Raises:
            NoFaceDetectedError: No face reached ``min_probability``.
            MultipleFacesError: More than one face with the ``strict`` selection.
        """
        rgb_image = image.convert("RGB")
        detections = [
            detection
            for detection in self._detector.detect(rgb_image)
            if detection.probability >= self._config.min_probability
        ]
        if not detections:
            raise NoFaceDetectedError(self._config.min_probability)

        return self._crop(rgb_image, self._select(detections, rgb_image.size))

    def process_all(
        self, image: Image.Image, min_face_px: int, max_faces: int
    ) -> list[DetectedFace]:
        """Returns every usable face of the image, with its box and crop (spec 001, R6 to R10).

        Unlike `process`, several faces are not an error and the ``selection`` of the config does
        not apply. The detector, the minimum probability and the crop are the same, so the
        threshold of the experiments still holds.

        Args:
            image: The image, already oriented (see `load_image`).
            min_face_px: The shortest side, in pixels, the box of a face must have once it is
                limited to the image. Smaller faces are ignored.
            max_faces: How many faces to keep. The ones with the biggest area win.

        Returns:
            The faces from left to right (ties from top to bottom).

        Raises:
            ValueError: ``max_faces`` is less than 1.
            NoFaceDetectedError: No face reached ``min_probability`` and ``min_face_px``.
        """
        if max_faces < 1:
            raise ValueError(f"max_faces must be at least 1, got {max_faces}")
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        usable: list[tuple[FaceDetection, BoundingBox]] = []
        for detection in self._detector.detect(rgb_image):
            if detection.probability < self._config.min_probability:
                continue
            box = limit_to_image(detection.box, width, height)
            if box is None or min(box.right - box.left, box.bottom - box.top) < min_face_px:
                continue
            usable.append((detection, box))
        if not usable:
            raise NoFaceDetectedError(self._config.min_probability)

        biggest = sorted(usable, key=lambda found: found[1].area, reverse=True)[:max_faces]
        left_to_right = sorted(biggest, key=lambda found: (found[1].left, found[1].top))
        return [
            DetectedFace(
                box=box, probability=detection.probability, crop=self._crop(rgb_image, detection)
            )
            for detection, box in left_to_right
        ]

    def _crop(self, rgb_image: Image.Image, face: FaceDetection) -> Image.Image:
        if self._config.alignment is AlignmentMode.SIMILARITY:
            return align_face(rgb_image, face.landmarks, self._config.image_size)
        return crop_face(rgb_image, face.box, self._config.image_size, self._config.margin)

    def _select(
        self, detections: list[FaceDetection], image_size: tuple[int, int]
    ) -> FaceDetection:
        if len(detections) == 1:
            return detections[0]

        match self._config.selection:
            case FaceSelection.STRICT:
                raise MultipleFacesError(len(detections))
            case FaceSelection.LARGEST:
                return max(detections, key=lambda detection: detection.box.area)
            case FaceSelection.CENTER:
                image_center = (image_size[0] / 2, image_size[1] / 2)
                return min(
                    detections, key=lambda detection: dist(detection.box.center, image_center)
                )
