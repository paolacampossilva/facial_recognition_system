"""HTTP routes: thin functions that read the request, call a service and return its result."""

from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.ports import Services
from app.config import Settings
from app.core.errors import TooManyPhotosError, UploadTooLargeError
from app.core.models import (
    MAX_PHOTOS,
    ClientConfig,
    DeletionResult,
    EnrollmentRequest,
    EnrollmentResult,
    PersonSummary,
    RecognitionResult,
)


async def read_upload(upload: UploadFile, limit_bytes: int) -> bytes:
    """Reads an uploaded file into memory, refusing files over the configured size limit."""
    content = await upload.read(limit_bytes + 1)
    if len(content) > limit_bytes:
        raise UploadTooLargeError(limit_bytes)
    return content


def build_router() -> APIRouter:
    """Builds the router; the services come from `request.app.state.services`."""
    router = APIRouter(prefix="/api")

    def services_of(request: Request) -> Services:
        return request.app.state.services

    def settings_of(request: Request) -> Settings:
        return request.app.state.settings

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/config")
    def config(request: Request) -> ClientConfig:
        return ClientConfig(threshold=settings_of(request).threshold)

    @router.get("/persons")
    def list_persons(request: Request) -> list[PersonSummary]:
        return services_of(request).persons.list_persons()

    @router.post("/persons", status_code=status.HTTP_201_CREATED)
    async def enroll_person(
        request: Request,
        name: Annotated[str, Form()],
        photos: Annotated[list[UploadFile], File()],
    ) -> EnrollmentResult:
        if len(photos) > MAX_PHOTOS:
            raise TooManyPhotosError(MAX_PHOTOS)
        limit = settings_of(request).max_upload_bytes
        contents = [await read_upload(photo, limit) for photo in photos]
        enrollment = EnrollmentRequest(name=name, photos=contents)
        # Reading the photos is async, but the model and the database are not: they run on a
        # worker thread so a slow enrollment does not freeze every other request.
        return await run_in_threadpool(services_of(request).enrollment.enroll, enrollment)

    @router.post("/recognitions")
    async def recognize(
        request: Request, photo: Annotated[UploadFile, File()]
    ) -> RecognitionResult:
        content = await read_upload(photo, settings_of(request).max_upload_bytes)
        return await run_in_threadpool(services_of(request).recognition.recognize, content)

    @router.delete("/persons/{person_id}")
    def delete_person(request: Request, person_id: int) -> DeletionResult:
        return services_of(request).deletion.delete(person_id)

    return router
