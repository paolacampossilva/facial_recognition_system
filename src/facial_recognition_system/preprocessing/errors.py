"""Errors raised when an image cannot be turned into a face crop."""

from pathlib import Path


class PreprocessingError(Exception):
    """Base error for an image that could not be preprocessed."""


class InvalidImageError(PreprocessingError):
    """The file could not be opened or decoded as an image."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"could not open {path} as an image")
        self.path = path


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
