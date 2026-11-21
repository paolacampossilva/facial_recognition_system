"""Enrollment of a person from a name and photos (spec 002)."""

import logging
import threading
from collections.abc import Sequence

import numpy as np

from app.core.encoder import FaceEncoder
from app.core.errors import DuplicatePersonError
from app.core.models import EnrollmentRequest, EnrollmentResult
from app.storage.models import Gallery, NewPerson
from app.storage.repository import SqlitePersonRepository

logger = logging.getLogger(__name__)


def find_duplicate(gallery: Gallery, vectors: Sequence[np.ndarray], threshold: float) -> str | None:
    """Returns the name of the enrolled person a new face is too similar to, if any.

    A face is the same person's when its cosine similarity with a stored photo is at least the
    threshold, the same rule recognition uses (decision D08).

    Args:
        gallery: The stored embeddings, one row per photo.
        vectors: The L2-normalized embeddings of the photos being enrolled.
        threshold: The minimum similarity that counts as the same person.

    Returns:
        The owner of the most similar stored photo when it reaches the threshold, else None.
    """
    if not gallery.labels:
        return None
    similarities = gallery.embeddings @ np.stack(vectors).T
    row, _ = np.unravel_index(similarities.argmax(), similarities.shape)
    if float(similarities.max()) >= threshold:
        return gallery.labels[int(row)]
    return None


class EnrollmentService:
    """Enrolls a person: embeds every photo, refuses duplicates and stores all or nothing."""

    def __init__(
        self, encoder: FaceEncoder, repository: SqlitePersonRepository, threshold: float
    ) -> None:
        self._encoder = encoder
        self._repository = repository
        self._threshold = threshold
        # Checking for a duplicate and storing the person are two repository calls: without this
        # lock, two requests for the same face under different names could both pass the check.
        self._lock = threading.Lock()

    def enroll(self, request: EnrollmentRequest) -> EnrollmentResult:
        """Enrolls the person with the embeddings of all the photos.

        Nothing is stored unless every photo works, and the photos themselves are never stored.

        Raises:
            InvalidImageError: A photo is not an image.
            NoFaceDetectedError: A photo has no face.
            MultipleFacesError: A photo has more than one face.
            DuplicatePersonError: The name is taken, or the face is already enrolled under another
                name (the error carries the name of the person who is already enrolled).
        """
        # The slow part, embedding the photos, stays outside the lock.
        vectors = [self._encoder.encode(photo) for photo in request.photos]
        with self._lock:
            gallery = self._repository.gallery(self._encoder.model_id)
            existing = find_duplicate(gallery, vectors, self._threshold)
            if existing is not None:
                raise DuplicatePersonError(existing, "similarity")
            person = self._repository.add_person(
                NewPerson(name=request.name, model_id=self._encoder.model_id, vectors=vectors)
            )
        logger.info(
            "person enrolled",
            extra={"person_id": person.id, "embedding_count": person.embedding_count},
        )
        return EnrollmentResult(
            person_id=person.id, name=person.name, embedding_count=person.embedding_count
        )
