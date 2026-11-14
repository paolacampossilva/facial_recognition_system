"""Recognition of a face against the enrolled people (specs 003 and 004)."""

import logging

from app.core.encoder import FaceEncoder
from app.core.errors import EmptyGalleryError
from app.core.models import RecognitionResult, RecognitionStatus
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.recognition.strategies import NearestNeighbor

logger = logging.getLogger(__name__)


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
    """Recognizes the face in a photo by its nearest neighbor in the gallery (decision D16)."""

    def __init__(
        self, encoder: FaceEncoder, repository: SqlitePersonRepository, threshold: float
    ) -> None:
        self._encoder = encoder
        self._repository = repository
        self._threshold = threshold

    def recognize(self, image: bytes) -> RecognitionResult:
        """Says whether the face in the image is an enrolled person.

        The gallery is read on every call, with no cache, so an enrollment or a deletion counts
        at once. Nothing is stored: not the image and not the embedding.

        Raises:
            InvalidImageError: The bytes are not an image.
            NoFaceDetectedError: The image has no face.
            MultipleFacesError: The image has more than one face.
            EmptyGalleryError: Nobody is enrolled for the configured model.
        """
        vector = self._encoder.encode(image)
        gallery = self._repository.gallery(self._encoder.model_id)
        if not gallery.labels:
            raise EmptyGalleryError()
        prediction = (
            NearestNeighbor().fit(gallery.embeddings, gallery.labels).predict(vector[None, :])
        )
        result = decide(float(prediction.scores[0]), prediction.labels[0], self._threshold)
        logger.info(
            "face recognized",
            extra={"status": result.status.value, "similarity": result.similarity},
        )
        return result
