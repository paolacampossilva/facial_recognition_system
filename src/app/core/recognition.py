"""Recognition of the faces of an image against the enrolled people (specs 003 and 004)."""

import logging

import numpy as np

from app.core.encoder import FaceEncoder
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
        encoded = self._encoder.encode_all(image, self._min_face_px, MAX_FACES)
        gallery = self._repository.gallery(self._encoder.model_id)
        if not gallery.labels:
            raise EmptyGalleryError()
        vectors = np.stack([face.vector for face in encoded.faces])
        prediction = NearestNeighbor().fit(gallery.embeddings, gallery.labels).predict(vectors)
        logger.info("image recognized", extra={"face_count": len(encoded.faces)})

        faces: list[FaceResult] = []
        for face, label, score in zip(
            encoded.faces, prediction.labels, prediction.scores, strict=True
        ):
            decision = decide(float(score), label, self._threshold)
            logger.info(
                "face recognized",
                extra={
                    "status": decision.status.value,
                    "similarity": round(decision.similarity, LOGGED_SIMILARITY_DIGITS),
                },
            )
            box = FaceBox(
                left=int(face.box.left),
                top=int(face.box.top),
                right=int(face.box.right),
                bottom=int(face.box.bottom),
            )
            faces.append(FaceResult(box=box, **decision.model_dump()))
        return RecognitionResponse(
            image_width=encoded.width, image_height=encoded.height, faces=faces
        )
