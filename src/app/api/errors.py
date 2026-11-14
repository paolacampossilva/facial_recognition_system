"""Translates the errors of the services into the single API error model (spec 005)."""

from http import HTTPStatus
from typing import Literal

from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from app.core.errors import (
    DuplicatePersonError,
    EmptyGalleryError,
    PersonNotFoundError,
    UploadTooLargeError,
)
from facial_recognition_system.preprocessing.errors import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)


class ErrorResponse(BaseModel):
    """Body of every error response: a stable code and an English message.

    `reason` and `person_name` are only set for `duplicate_person`, so the interface can tell
    a repeated name from a repeated face and name the person who is already enrolled.
    """

    code: str
    message: str
    reason: Literal["name", "similarity"] | None = None
    person_name: str | None = None


def to_error_response(error: Exception) -> tuple[int, ErrorResponse]:
    """Maps an error to its HTTP status and error body.

    Unknown errors become `internal_error` without leaking the message of the exception.
    """
    if isinstance(error, NoFaceDetectedError):
        return HTTPStatus.UNPROCESSABLE_ENTITY, ErrorResponse(
            code="no_face_detected", message="No face found in the image"
        )
    if isinstance(error, MultipleFacesError):
        return HTTPStatus.UNPROCESSABLE_ENTITY, ErrorResponse(
            code="multiple_faces", message="More than one face in the image"
        )
    if isinstance(error, InvalidImageError):
        return HTTPStatus.UNPROCESSABLE_ENTITY, ErrorResponse(
            code="invalid_image", message="The file is not a valid image"
        )
    if isinstance(error, DuplicatePersonError):
        return HTTPStatus.CONFLICT, ErrorResponse(
            code="duplicate_person",
            message=f"Person already enrolled (by {error.reason})",
            reason=error.reason,
            person_name=error.name,
        )
    if isinstance(error, EmptyGalleryError):
        return HTTPStatus.CONFLICT, ErrorResponse(
            code="empty_gallery", message="No person is enrolled"
        )
    if isinstance(error, PersonNotFoundError):
        return HTTPStatus.NOT_FOUND, ErrorResponse(
            code="person_not_found", message="Person not found"
        )
    # pydantic's ValidationError is a ValueError; RequestValidationError covers bad route input.
    if isinstance(error, ValueError | RequestValidationError | UploadTooLargeError):
        return HTTPStatus.UNPROCESSABLE_ENTITY, ErrorResponse(
            code="invalid_request", message="Invalid name or photos"
        )
    return HTTPStatus.INTERNAL_SERVER_ERROR, ErrorResponse(
        code="internal_error", message="Unexpected error"
    )
