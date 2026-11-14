"""HTTP routes: thin functions that read the request, call a service and return its result."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile, status

from app.api.ports import Services
from app.core.errors import UploadTooLargeError
from app.core.models import (
    DeletionResult,
    EnrollmentRequest,
    EnrollmentResult,
    PersonSummary,
    RecognitionResult,
)

logger = logging.getLogger(__name__)

# TODO(#48): move to the Settings class of the configuration module (spec 007).
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024


async def read_upload(upload: UploadFile) -> bytes:
    """Reads an uploaded file into memory, refusing files over the size limit."""
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(MAX_UPLOAD_BYTES)
    return content


def build_router() -> APIRouter:
    """Builds the router; the services come from `request.app.state.services`."""
    router = APIRouter(prefix="/api")

    def services_of(request: Request) -> Services:
        return request.app.state.services

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/persons")
    def list_persons(request: Request) -> list[PersonSummary]:
        return services_of(request).persons.list_persons()

    @router.post("/persons", status_code=status.HTTP_201_CREATED)
    async def enroll_person(
        request: Request,
        name: Annotated[str, Form()],
        photos: Annotated[list[UploadFile], File()],
    ) -> EnrollmentResult:
        contents = [await read_upload(photo) for photo in photos]
        enrollment = EnrollmentRequest(name=name, photos=contents)
        result = services_of(request).enrollment.enroll(enrollment)
        logger.info(
            "person enrolled",
            extra={"person_id": result.person_id, "embedding_count": result.embedding_count},
        )
        return result

    @router.post("/recognitions")
    async def recognize(
        request: Request, photo: Annotated[UploadFile, File()]
    ) -> RecognitionResult:
        content = await read_upload(photo)
        result = services_of(request).recognition.recognize(content)
        logger.info("face recognized", extra={"status": result.status.value})
        return result

    @router.delete("/persons/{person_id}")
    def delete_person(request: Request, person_id: int) -> DeletionResult:
        result = services_of(request).deletion.delete(person_id)
        logger.info(
            "person deleted",
            extra={"person_id": result.person_id, "deleted_embeddings": result.deleted_embeddings},
        )
        return result

    return router
