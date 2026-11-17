"""Fakes and helpers shared by the tests of the application services (specs 002, 003 and 006).

The detector and the embedder are scripted, so no test needs a network, a GPU or model weights.
Photos are solid-color PNGs: the color picks the embedding, the width picks how many faces the
detector finds.
"""

import contextlib
import socket
import threading
import time
from collections.abc import Iterator
from io import BytesIO
from typing import TypeGuard, cast

import numpy as np
import uvicorn
from fastapi import FastAPI
from PIL import Image

from app.core.encoder import FaceEncoder
from app.core.models import EnrollmentRequest, FaceResult, RecognitionResponse
from facial_recognition_system.preprocessing import (
    BoundingBox,
    FaceDetection,
    FacePreprocessor,
    PreprocessingConfig,
)

DIMENSION: int = 512
THRESHOLD: float = 0.766
MODEL_ID: str = "fake-model"
# The scripted detector reads the width of the photo: one face, no face or two faces (the
# strict path of the enrollment). A photo as tall as GROUP_HEIGHT is a group photo instead: every
# run of one color is a face (see `group_photo`).
ONE_FACE, NO_FACE, TWO_FACES = 100, 101, 102
GROUP_HEIGHT = 200
FACE_WIDTH = 100
DIVIDER_WIDTH = 4
DIVIDER_COLOR = (0, 0, 0)  # the gap between two faces, never a face
MIN_FACE_PX = 80  # the application default (spec 001, R7)
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


def detection_at(left: int, top: int, right: int, bottom: int) -> FaceDetection:
    landmarks = tuple((float(index), float(index)) for index in range(5))
    box = BoundingBox(left=left, top=top, right=right, bottom=bottom)
    return FaceDetection(box=box, probability=0.99, landmarks=landmarks)


def detection(width: int) -> FaceDetection:
    return detection_at(0, 0, width, 100)


def faces_by_color(image: Image.Image) -> list[FaceDetection]:
    """One face per run of one color along the top row, as `group_photo` lays them out."""
    row = np.asarray(image.convert("RGB"))[0]
    found: list[FaceDetection] = []
    start = 0
    for column in range(1, image.width + 1):
        ends_run = column == image.width or not np.array_equal(row[column], row[start])
        if not ends_run:
            continue
        if tuple(int(value) for value in row[start]) != DIVIDER_COLOR:
            found.append(detection_at(start, 0, column, image.height))
        start = column
    return found


class ScriptedDetector:
    """Finds the faces the width of the photo asks for."""

    def detect(self, image: Image.Image) -> list[FaceDetection]:
        if image.height == GROUP_HEIGHT:
            return faces_by_color(image)
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


def is_color(face: Color | tuple[Color, int]) -> TypeGuard[Color]:
    return isinstance(face[0], int)


def group_photo(*faces: Color | tuple[Color, int]) -> bytes:
    """The bytes of a PNG with one face per argument, left to right, with a gap between faces.

    A face is its color (the person, as for `photo`) or `(color, width)` to make a small one.
    """
    parts = [
        (face, FACE_WIDTH) if is_color(face) else cast(tuple[Color, int], face) for face in faces
    ]
    total = sum(width for _, width in parts) + DIVIDER_WIDTH * (len(parts) - 1)
    image = Image.new("RGB", (total, GROUP_HEIGHT), DIVIDER_COLOR)
    left = 0
    for color, width in parts:
        image.paste(color, (left, 0, left + width, GROUP_HEIGHT))
        left += width + DIVIDER_WIDTH
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def only_face(response: RecognitionResponse) -> FaceResult:
    """The one face of a response, failing the test when there is more than one."""
    assert len(response.faces) == 1
    return response.faces[0]


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


def free_port() -> int:
    """A port nothing listens on right now, for a test server on the loopback address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


SERVER_START_SECONDS: float = 10


@contextlib.contextmanager
def serving(application: FastAPI, **config: object) -> Iterator[str]:
    """Runs the application on a free loopback port in a thread and yields its URL.

    Args:
        application: The application to serve.
        **config: Extra `uvicorn.Config` options, such as `log_config`.
    """
    port = free_port()
    options: dict[str, object] = {"log_level": "warning", **config}
    server = uvicorn.Server(uvicorn.Config(application, host="127.0.0.1", port=port, **options))  # type: ignore[arg-type]
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + SERVER_START_SECONDS
    while not server.started:
        if time.monotonic() > deadline:
            raise TimeoutError("the test server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=SERVER_START_SECONDS)
