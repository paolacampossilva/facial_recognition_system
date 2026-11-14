"""Fakes and helpers shared by the tests of the application services (specs 002, 003 and 006).

The detector and the embedder are scripted, so no test needs a network, a GPU or model weights.
Photos are solid-color PNGs: the color picks the embedding, the width picks how many faces the
detector finds.
"""

from io import BytesIO

import numpy as np
from PIL import Image

from app.core.encoder import FaceEncoder
from app.core.models import EnrollmentRequest
from facial_recognition_system.preprocessing import (
    BoundingBox,
    FaceDetection,
    FacePreprocessor,
    PreprocessingConfig,
)

DIMENSION: int = 512
THRESHOLD: float = 0.766
MODEL_ID: str = "fake-model"
# The scripted detector reads the width of the photo: one face, no face or two faces.
ONE_FACE, NO_FACE, TWO_FACES = 100, 101, 102
ANA_COLOR = (200, 0, 0)
BIA_COLOR = (0, 200, 0)
CARLA_COLOR = (0, 0, 200)

Color = tuple[int, int, int]


def unit(seed: int) -> np.ndarray:
    """A reproducible random unit vector."""
    vector = np.random.default_rng(seed).standard_normal(DIMENSION).astype(np.float32)
    return vector / np.linalg.norm(vector)


def with_cosine(base: np.ndarray, cosine: float, seed: int) -> np.ndarray:
    """A unit vector whose cosine similarity with `base` is `cosine`."""
    other = unit(seed)
    orthogonal = other - (other @ base) * base
    orthogonal /= np.linalg.norm(orthogonal)
    return (cosine * base + np.sqrt(1 - cosine**2) * orthogonal).astype(np.float32)


def detection(width: int) -> FaceDetection:
    landmarks = tuple((float(index), float(index)) for index in range(5))
    box = BoundingBox(left=0, top=0, right=width, bottom=100)
    return FaceDetection(box=box, probability=0.99, landmarks=landmarks)


class ScriptedDetector:
    """Finds the faces the width of the photo asks for."""

    def detect(self, image: Image.Image) -> list[FaceDetection]:
        match image.width:
            case 101:
                return []
            case 102:
                return [detection(51), detection(102)]
            case _:
                return [detection(image.width)]


class ColorEmbedder:
    """Gives each solid-color face the vector it was assigned."""

    def __init__(self, vectors: dict[Color, np.ndarray], model_id: str = MODEL_ID) -> None:
        self.vectors = vectors
        self.model_id = model_id

    def embed(self, face: Image.Image) -> np.ndarray:
        pixel = face.getpixel((0, 0))
        assert isinstance(pixel, tuple)
        return self.vectors[(pixel[0], pixel[1], pixel[2])]


def photo(color: Color = ANA_COLOR, width: int = ONE_FACE) -> bytes:
    """The bytes of a solid-color PNG."""
    buffer = BytesIO()
    Image.new("RGB", (width, 100), color).save(buffer, format="PNG")
    return buffer.getvalue()


ANA = unit(1)
BIA = unit(2)
VECTORS: dict[Color, np.ndarray] = {
    ANA_COLOR: ANA,
    BIA_COLOR: BIA,
    CARLA_COLOR: with_cosine(ANA, 0.9, seed=3),  # a face 0.9 similar to Ana's
}


def build_encoder(
    vectors: dict[Color, np.ndarray] | None = None, model_id: str = MODEL_ID
) -> FaceEncoder:
    """A real FaceEncoder over the scripted detector and embedder."""
    preprocessor = FacePreprocessor(ScriptedDetector(), PreprocessingConfig())
    return FaceEncoder(preprocessor, ColorEmbedder(vectors or VECTORS, model_id))


def request(name: str, *photos: bytes) -> EnrollmentRequest:
    return EnrollmentRequest(name=name, photos=list(photos))
