"""What the API needs from the core services.

The real services (enrollment, recognition, deletion and the person repository) and the fakes of
`app.fakes` both satisfy these protocols, so the API tests inject fakes without a model.
"""

from typing import Protocol

from app.core.models import (
    DeletionResult,
    EnrollmentRequest,
    EnrollmentResult,
    PersonSummary,
    RecognitionResult,
)


class EnrollmentPort(Protocol):
    """Enrolls one person from a name and photos."""

    def enroll(self, request: EnrollmentRequest) -> EnrollmentResult: ...


class RecognitionPort(Protocol):
    """Recognizes the face in one image."""

    def recognize(self, image: bytes) -> RecognitionResult: ...


class DeletionPort(Protocol):
    """Deletes one enrolled person and everything tied to them."""

    def delete(self, person_id: int) -> DeletionResult: ...


class PersonsPort(Protocol):
    """Lists the enrolled people."""

    def list_persons(self) -> list[PersonSummary]: ...


class Services(Protocol):
    """The four services the routes call."""

    @property
    def enrollment(self) -> EnrollmentPort: ...

    @property
    def recognition(self) -> RecognitionPort: ...

    @property
    def deletion(self) -> DeletionPort: ...

    @property
    def persons(self) -> PersonsPort: ...
