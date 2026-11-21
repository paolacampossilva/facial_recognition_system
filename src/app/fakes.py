"""In-memory fake services that exercise every state of the interface without a model.

Scripted behavior, so each screen state can be reached by hand:

- Enrolling the name `sem-rosto`, `varios-rostos` or `invalida` raises the image error of the
  same name; `clone` raises a duplicate by face similarity; a name already used raises a duplicate
  by name.
- Recognizing raises `EmptyGalleryError` while nobody is enrolled. Otherwise a file with an even
  size is one face `recognized` as the first enrolled person and a file with an odd size is one
  `unknown` face. A file that starts with `grupo` has two faces, the first enrolled person and an
  unknown one, and one that starts with `sem-rosto` raises the no-face error.
"""

from pathlib import Path

from app.core.errors import DuplicatePersonError, EmptyGalleryError, PersonNotFoundError
from app.core.models import (
    DeletionResult,
    EnrollmentRequest,
    EnrollmentResult,
    FaceBox,
    FaceResult,
    PersonSummary,
    RecognitionResponse,
    RecognitionStatus,
)
from facial_recognition_system.preprocessing.errors import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)

RECOGNIZED_SIMILARITY: float = 0.91
UNKNOWN_SIMILARITY: float = 0.42
MIN_FACE_PROBABILITY: float = 0.9
GROUP_PREFIX: bytes = b"grupo"
NO_FACE_PREFIX: bytes = b"sem-rosto"
IMAGE_SIZE: tuple[int, int] = (600, 400)
SINGLE_BOX: FaceBox = FaceBox(left=150, top=60, right=450, bottom=360)
GROUP_BOXES: tuple[FaceBox, FaceBox] = (
    FaceBox(left=60, top=80, right=260, bottom=330),
    FaceBox(left=340, top=90, right=540, bottom=340),
)


class FakeServices:
    """All four services over one in-memory list of people."""

    def __init__(self) -> None:
        self._people: dict[int, PersonSummary] = {}
        self._next_id: int = 1

    @property
    def enrollment(self) -> "FakeServices":
        return self

    @property
    def recognition(self) -> "FakeServices":
        return self

    @property
    def deletion(self) -> "FakeServices":
        return self

    @property
    def persons(self) -> "FakeServices":
        return self

    def list_persons(self) -> list[PersonSummary]:
        """Returns the enrolled people in enrollment order."""
        return list(self._people.values())

    def enroll(self, request: EnrollmentRequest) -> EnrollmentResult:
        """Enrolls a person, or raises the error the name asks for."""
        name = request.name
        if name == "sem-rosto":
            raise NoFaceDetectedError(MIN_FACE_PROBABILITY)
        if name == "varios-rostos":
            raise MultipleFacesError(2)
        if name == "invalida":
            raise InvalidImageError(Path("upload"))
        if name == "clone":
            raise DuplicatePersonError(self._first_name() or "Ana", "similarity")
        if any(person.name.casefold() == name.casefold() for person in self._people.values()):
            raise DuplicatePersonError(name, "name")
        person = PersonSummary(id=self._next_id, name=name, embedding_count=len(request.photos))
        self._people[person.id] = person
        self._next_id += 1
        return EnrollmentResult(
            person_id=person.id, name=person.name, embedding_count=person.embedding_count
        )

    def recognize(self, image: bytes) -> RecognitionResponse:
        """Recognizes the faces the file asks for (see the module docstring)."""
        if image.startswith(NO_FACE_PREFIX):
            raise NoFaceDetectedError(MIN_FACE_PROBABILITY)
        first_name = self._first_name()
        if first_name is None:
            raise EmptyGalleryError()
        known = FaceResult(
            box=SINGLE_BOX,
            status=RecognitionStatus.RECOGNIZED,
            person_name=first_name,
            similarity=RECOGNIZED_SIMILARITY,
        )
        stranger = FaceResult(
            box=SINGLE_BOX,
            status=RecognitionStatus.UNKNOWN,
            person_name=None,
            similarity=UNKNOWN_SIMILARITY,
        )
        if image.startswith(GROUP_PREFIX):
            faces = [
                known.model_copy(update={"box": GROUP_BOXES[0]}),
                stranger.model_copy(update={"box": GROUP_BOXES[1]}),
            ]
        else:
            faces = [known if len(image) % 2 == 0 else stranger]
        return RecognitionResponse(
            image_width=IMAGE_SIZE[0], image_height=IMAGE_SIZE[1], faces=faces
        )

    def delete(self, person_id: int) -> DeletionResult:
        """Deletes a person, or raises when the id does not exist."""
        person = self._people.pop(person_id, None)
        if person is None:
            raise PersonNotFoundError(person_id)
        return DeletionResult(person_id=person.id, deleted_embeddings=person.embedding_count)

    def _first_name(self) -> str | None:
        return next((person.name for person in self._people.values()), None)
