"""From the bytes of a photo to a face embedding, shared by enrollment and recognition."""

from typing import Protocol

import numpy as np
from PIL import Image

from facial_recognition_system.preprocessing import FacePreprocessor, load_image


class Embedder(Protocol):
    """Turns one face crop into an L2-normalized embedding of the configured model.

    The real one wraps a torch model; tests inject a fake, so they need no weights or GPU.
    """

    model_id: str

    def embed(self, face: Image.Image) -> np.ndarray:
        """Returns a one-dimensional float32 vector with unit norm."""
        ...


class FaceEncoder:
    """Photo bytes in, embedding out: decode, find the single face, embed it."""

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
