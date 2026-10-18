"""Image to face crop: load, detect, select one face and crop or align it."""

from enum import StrEnum
from math import dist
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from facial_recognition_system.preprocessing.alignment import (
    AlignmentMode,
    align_face,
    crop_face,
)
from facial_recognition_system.preprocessing.detection import FaceDetection, FaceDetector
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


def load_image(path: Path) -> Image.Image:
    """Opens an image as RGB, applying the EXIF orientation of phone photos.

    Raises:
        InvalidImageError: The file is missing, corrupted or not an image.
    """
    try:
        with Image.open(path) as image:
            oriented = ImageOps.exif_transpose(image)
            return oriented.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageError(path) from error


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

        face = self._select(detections, rgb_image.size)

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
