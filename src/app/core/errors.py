"""Errors of the application services (specs 002, 003 and 006).

The image errors of spec 001 live in `facial_recognition_system.preprocessing.errors`.
"""

from typing import Literal


class AppError(Exception):
    """Base error of the application: the edge can catch the whole family."""


class DuplicatePersonError(AppError):
    """The person is already enrolled, by name or by face similarity."""

    def __init__(self, name: str, reason: Literal["name", "similarity"]) -> None:
        super().__init__(f"person {name!r} already enrolled (reason: {reason})")
        self.name = name
        self.reason: Literal["name", "similarity"] = reason


class EmptyGalleryError(AppError):
    """Nobody is enrolled, so there is nothing to compare a face with."""

    def __init__(self) -> None:
        super().__init__("no person enrolled")


class UploadTooLargeError(AppError):
    """An uploaded file is bigger than the configured limit."""

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"file larger than {limit_bytes} bytes")
        self.limit_bytes = limit_bytes


class PersonNotFoundError(AppError):
    """No enrolled person has this id."""

    def __init__(self, person_id: int) -> None:
        super().__init__(f"person {person_id} not found")
        self.person_id = person_id
