"""Errors raised when an image cannot be turned into a face crop."""

from pathlib import Path


class PreprocessingError(Exception):
    """Base error for an image that could not be preprocessed."""


class InvalidImageError(PreprocessingError):
    """The file could not be opened or decoded as an image.

    `source` is the path of the file, or a label such as "upload" when the image came as bytes.
    """

    def __init__(self, source: Path | str = "upload") -> None:
        super().__init__(f"could not open {source} as an image")
        self.source = source


class NoFaceDetectedError(PreprocessingError):
    """No face reached the minimum detection probability."""

    def __init__(self, min_probability: float) -> None:
        super().__init__(f"no face detected with probability >= {min_probability}")
        self.min_probability = min_probability


class MultipleFacesError(PreprocessingError):
    """More than one face was found and the selection policy requires exactly one."""

    def __init__(self, face_count: int) -> None:
        super().__init__(f"{face_count} faces detected, expected exactly one")
        self.face_count = face_count
