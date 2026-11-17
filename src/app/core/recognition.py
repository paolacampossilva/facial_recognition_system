"""Recognition of the faces of an image against the enrolled people (specs 003 and 004)."""

import logging

import numpy as np

from app.core.encoder import EncodedFace, FaceEncoder
from app.core.errors import EmptyGalleryError
from app.core.models import (
    FaceBox,
    FaceResult,
    RecognitionResponse,
    RecognitionResult,
    RecognitionStatus,
)
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.recognition.strategies import NearestNeighbor

logger = logging.getLogger(__name__)

# Four digits are plenty for a log line, and hide float32 noise such as 0.99999994 for 1.0.
LOGGED_SIMILARITY_DIGITS: int = 4
# Spec 001, R8: a bigger group photo would cost an embedding per face for little use.
MAX_FACES: int = 20


def decide(similarity: float, person_name: str, threshold: float) -> RecognitionResult:
    """Turns the best match into a result: recognized, or unknown below the threshold.

    An unknown face never carries the name of the closest person: below the threshold the
    system does not say who it looks like (spec 004, R2).

    Args:
        similarity: Cosine similarity with the most similar stored photo.
        person_name: Owner of that photo.
        threshold: The lowest similarity accepted as the same person (decision D08).
    """
    if similarity >= threshold:
        return RecognitionResult(
            status=RecognitionStatus.RECOGNIZED, person_name=person_name, similarity=similarity
        )
    return RecognitionResult(
        status=RecognitionStatus.UNKNOWN, person_name=None, similarity=similarity
    )


def decide_faces(
    faces: list[EncodedFace], labels: list[str], scores: np.ndarray, threshold: float
) -> list[FaceResult]:
    """Decides each face on its own from its best match, keeping the order and the box.

    Args:
        faces: The faces of the image, left to right.
        labels: The owner of the most similar stored photo of each face.
        scores: The cosine similarity with that photo, for each face.
        threshold: The lowest similarity accepted as the same person (decision D08).
    """
    results: list[FaceResult] = []
    for face, label, score in zip(faces, labels, scores, strict=True):
        decision = decide(float(score), label, threshold)
        box = FaceBox.model_validate(face.box.model_dump())  # whole pixels inside the image
        results.append(FaceResult(box=box, **decision.model_dump()))
    return results


class RecognitionService:
    """Recognizes every face of a photo by its nearest neighbor in the gallery (decision D16)."""

    def __init__(
        self,
        encoder: FaceEncoder,
        repository: SqlitePersonRepository,
        threshold: float,
        min_face_px: int,
    ) -> None:
        self._encoder = encoder
        self._repository = repository
        self._threshold = threshold
        self._min_face_px = min_face_px

    def recognize(self, image: bytes) -> RecognitionResponse:
        """Says, for each face of the image, whether it is an enrolled person.

        The gallery is read once per image, with no cache, so an enrollment or a deletion counts
        at once. Nothing is stored: not the image and not the embeddings. Each face is decided
        alone with the same threshold, and the same person can come out for more than one face.

        Raises:
            InvalidImageError: The bytes are not an image.
            NoFaceDetectedError: The image has no usable face (none, or all too small).
            EmptyGalleryError: Nobody is enrolled for the configured model.
        """
        # The gallery goes first: with nobody enrolled there is nothing to compare, so the
        # detection and the embeddings of up to 20 faces would be wasted.
        gallery = self._repository.gallery(self._encoder.model_id)
        if not gallery.labels:
            raise EmptyGalleryError()
        encoded = self._encoder.encode_all(image, self._min_face_px, MAX_FACES)
        vectors = np.stack([face.vector for face in encoded.faces])
        prediction = NearestNeighbor().fit(gallery.embeddings, gallery.labels).predict(vectors)
        faces = decide_faces(encoded.faces, prediction.labels, prediction.scores, self._threshold)

        logger.info("image recognized", extra={"face_count": len(faces)})
        for face in faces:
            logger.info(
                "face recognized",
                extra={
                    "status": face.status.value,
                    "similarity": round(face.similarity, LOGGED_SIMILARITY_DIGITS),
                },
            )
        return RecognitionResponse(
            image_width=encoded.width, image_height=encoded.height, faces=faces
        )
