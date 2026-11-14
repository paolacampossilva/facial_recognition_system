"""Pydantic models of the application services (specs 002, 003 and 006)."""

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

MAX_NAME_LENGTH: int = 100
MIN_PHOTOS: int = 1
MAX_PHOTOS: int = 10


class EnrollmentRequest(BaseModel):
    """A name and the photos used to enroll one person."""

    name: str = Field(min_length=1, max_length=MAX_NAME_LENGTH)
    photos: list[bytes] = Field(min_length=MIN_PHOTOS, max_length=MAX_PHOTOS)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: object) -> object:
        """Remove the spaces at both ends before the length check."""
        return value.strip() if isinstance(value, str) else value


class EnrollmentResult(BaseModel):
    """What the enrollment of one person produced."""

    person_id: int
    name: str
    embedding_count: int


class PersonSummary(BaseModel):
    """One enrolled person as the person list shows it."""

    id: int
    name: str
    embedding_count: int


class RecognitionStatus(StrEnum):
    """Outcome of comparing a face with the gallery."""

    RECOGNIZED = "recognized"
    UNKNOWN = "unknown"


class RecognitionResult(BaseModel):
    """The status, the name (only when recognized) and the best similarity."""

    status: RecognitionStatus
    person_name: str | None  # None when the status is unknown
    similarity: float


class DeletionResult(BaseModel):
    """What the deletion of one person removed."""

    person_id: int
    deleted_embeddings: int
