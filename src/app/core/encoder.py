"""From the bytes of a photo to face embeddings, shared by enrollment and recognition."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from facial_recognition_system.preprocessing import BoundingBox, FacePreprocessor, load_image


class Embedder(Protocol):
    """Turns one face crop into an L2-normalized embedding of the configured model.

    The real one wraps a torch model; tests inject a fake, so they need no weights or GPU.
    """

    model_id: str

    def embed(self, face: Image.Image) -> np.ndarray:
        """Returns a one-dimensional float32 vector with unit norm."""
        ...


@dataclass(frozen=True)
class EncodedFace:
    """The embedding of one face and where the face is (whole pixels inside the image).

    A dataclass, not a Pydantic model, because it carries a numpy array that nothing validates.
    """

    box: BoundingBox
    vector: np.ndarray


@dataclass(frozen=True)
class EncodedImage:
    """Every usable face of an image, left to right, with the size of the oriented image."""

    width: int
    height: int
    faces: list[EncodedFace]


class FaceEncoder:
    """Photo bytes in, embeddings out: decode, find the faces, embed them."""

    def __init__(self, preprocessor: FacePreprocessor, embedder: Embedder) -> None:
        self._preprocessor = preprocessor
        self._embedder = embedder

    @property
    def model_id(self) -> str:
        """The model that makes the embeddings: stored with each one and used to filter them."""
        return self._embedder.model_id

    def encode(self, image: bytes) -> np.ndarray:
        """Returns the embedding of the single face in the image.

        Raises:
            InvalidImageError: The bytes are not an image.
            NoFaceDetectedError: No face was found.
            MultipleFacesError: More than one face was found.
        """
        face = self._preprocessor.process(load_image(image))
        return self._embedder.embed(face)

    def encode_all(self, image: bytes, min_face_px: int, max_faces: int) -> EncodedImage:
        """Returns the embedding and the box of every usable face (spec 001, R6 to R10).

        This is the recognition path: several faces are not an error. Enrollment keeps using
        `encode`, which accepts exactly one.

        Args:
            image: The bytes of the photo.
            min_face_px: Faces whose box has a shorter side than this are ignored.
            max_faces: How many faces to keep, the biggest ones.

        Raises:
            InvalidImageError: The bytes are not an image.
            NoFaceDetectedError: No usable face was found.
        """
        oriented = load_image(image)
        found = self._preprocessor.process_all(oriented, min_face_px, max_faces)
        faces = [
            EncodedFace(box=face.box, vector=self._embedder.embed(face.crop)) for face in found
        ]
        return EncodedImage(width=oriented.width, height=oriented.height, faces=faces)
