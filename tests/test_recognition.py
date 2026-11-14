"""Recognition tests (specs 003 and 004): scripted detector and embedder, real database."""

import logging
from pathlib import Path

import numpy as np
import pytest

from app.core.enrollment import EnrollmentService
from app.core.errors import EmptyGalleryError
from app.core.models import RecognitionStatus
from app.core.recognition import RecognitionService, decide
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.preprocessing import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from tests.app_support import (
    ANA,
    ANA_COLOR,
    BIA_COLOR,
    NO_FACE,
    THRESHOLD,
    TWO_FACES,
    VECTORS,
    Color,
    build_encoder,
    photo,
    request,
    with_cosine,
)

pytestmark = pytest.mark.unit

QUERY_COLOR: Color = (10, 10, 10)
QUERY = with_cosine(ANA, 0.8, seed=5)  # a face 0.8 similar to Ana's
WITH_QUERY = {**VECTORS, QUERY_COLOR: QUERY}


def recognizer(
    repository: SqlitePersonRepository,
    threshold: float = THRESHOLD,
    model_id: str = "fake-model",
) -> RecognitionService:
    return RecognitionService(build_encoder(WITH_QUERY, model_id), repository, threshold)


def enroll(
    repository: SqlitePersonRepository, name: str, color: Color, model_id: str = "fake-model"
) -> None:
    encoder = build_encoder(WITH_QUERY, model_id)
    EnrollmentService(encoder, repository, THRESHOLD).enroll(request(name, photo(color)))


def stored_similarity(repository: SqlitePersonRepository) -> float:
    """The similarity of QUERY with the stored Ana, computed the way the strategy does."""
    gallery = repository.gallery("fake-model")
    return float((QUERY[None, :] @ gallery.embeddings.T)[0, 0])


def test_the_photo_of_an_enrolled_person_is_recognized(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = recognizer(repository).recognize(photo(ANA_COLOR))

    assert result.status is RecognitionStatus.RECOGNIZED
    assert result.person_name == "Ana"
    assert result.similarity >= THRESHOLD


def test_the_closest_person_is_the_one_returned(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    enroll(repository, "Bia", BIA_COLOR)

    assert recognizer(repository).recognize(photo(BIA_COLOR)).person_name == "Bia"
    assert recognizer(repository).recognize(photo(ANA_COLOR)).person_name == "Ana"


def test_a_similarity_equal_to_the_threshold_is_recognized(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    similarity = stored_similarity(repository)

    result = recognizer(repository, threshold=similarity).recognize(photo(QUERY_COLOR))

    assert result.status is RecognitionStatus.RECOGNIZED
    assert result.similarity == similarity


def test_a_similarity_just_below_the_threshold_is_unknown(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    threshold = float(np.nextafter(stored_similarity(repository), 1.0))

    result = recognizer(repository, threshold=threshold).recognize(photo(QUERY_COLOR))

    assert result.status is RecognitionStatus.UNKNOWN


@pytest.mark.parametrize(
    ("threshold", "status"),
    [(0.7, RecognitionStatus.RECOGNIZED), (0.9, RecognitionStatus.UNKNOWN)],
)
def test_the_threshold_comes_from_the_configuration_and_changes_the_result(
    repository: SqlitePersonRepository, threshold: float, status: RecognitionStatus
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = recognizer(repository, threshold=threshold).recognize(photo(QUERY_COLOR))  # 0.8

    assert result.status is status


def test_only_embeddings_of_the_configured_model_are_compared(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, model_id="model-a")
    enroll(repository, "Bia", BIA_COLOR, model_id="model-b")

    only_a = recognizer(repository, model_id="model-a").recognize(photo(BIA_COLOR))
    only_b = recognizer(repository, model_id="model-b").recognize(photo(BIA_COLOR))

    assert only_a.status is RecognitionStatus.UNKNOWN  # Bia's own vectors are another model's
    assert only_b.person_name == "Bia"


def test_an_empty_gallery_is_an_error_and_not_an_unknown_person(
    repository: SqlitePersonRepository,
) -> None:
    with pytest.raises(EmptyGalleryError):
        recognizer(repository).recognize(photo(ANA_COLOR))


def test_a_gallery_with_only_another_model_counts_as_empty(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, model_id="model-a")

    with pytest.raises(EmptyGalleryError):
        recognizer(repository, model_id="model-b").recognize(photo(ANA_COLOR))


@pytest.mark.parametrize(
    ("image", "error"),
    [
        (photo(width=NO_FACE), NoFaceDetectedError),
        (photo(width=TWO_FACES), MultipleFacesError),
        (b"not an image", InvalidImageError),
    ],
)
def test_image_errors_of_spec_001_propagate(
    repository: SqlitePersonRepository, image: bytes, error: type[Exception]
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    with pytest.raises(error):
        recognizer(repository).recognize(image)


def test_recognizing_changes_nothing_in_the_database_or_on_disk(tmp_path: Path) -> None:
    folder = tmp_path / "private"
    repository = SqlitePersonRepository(folder / "app.db")
    enroll(repository, "Ana", ANA_COLOR)
    database = folder / "app.db"
    before = (repository.list_persons(), database.read_bytes(), sorted(folder.iterdir()))

    recognizer(repository).recognize(photo(ANA_COLOR))
    recognizer(repository).recognize(photo(BIA_COLOR))

    after = (repository.list_persons(), database.read_bytes(), sorted(folder.iterdir()))
    repository.close()
    assert after == before


def test_an_enrollment_and_a_deletion_count_at_once_without_restarting(
    repository: SqlitePersonRepository,
) -> None:
    service = recognizer(repository)  # the same service the whole time
    with pytest.raises(EmptyGalleryError):
        service.recognize(photo(ANA_COLOR))

    enroll(repository, "Ana", ANA_COLOR)
    assert service.recognize(photo(ANA_COLOR)).person_name == "Ana"

    repository.delete_person(1)
    with pytest.raises(EmptyGalleryError):
        service.recognize(photo(ANA_COLOR))

    enroll(repository, "Bia", BIA_COLOR)
    assert service.recognize(photo(ANA_COLOR)).status is RecognitionStatus.UNKNOWN


def test_an_unknown_person_is_the_other_author_before_enrolling(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = recognizer(repository).recognize(photo(BIA_COLOR))

    assert result.status is RecognitionStatus.UNKNOWN
    assert result.person_name is None


def test_an_unknown_result_does_not_reveal_the_closest_person(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    result = recognizer(repository, threshold=0.99).recognize(photo(QUERY_COLOR))

    assert result.status is RecognitionStatus.UNKNOWN
    assert set(result.model_dump()) == {"status", "person_name", "similarity"}
    assert "Ana" not in str(result.model_dump())
    assert 0 < result.similarity < 0.99


def test_the_log_has_the_status_and_the_similarity_but_not_the_name(
    repository: SqlitePersonRepository, caplog: pytest.LogCaptureFixture
) -> None:
    enroll(repository, "Ana Souza", ANA_COLOR)

    with caplog.at_level(logging.INFO, logger="app.core.recognition"):
        recognizer(repository).recognize(photo(ANA_COLOR))

    (record,) = caplog.records
    assert record.status == "recognized"  # type: ignore[attr-defined]
    assert record.similarity >= THRESHOLD  # type: ignore[attr-defined]
    assert "Ana" not in caplog.text and "Souza" not in caplog.text


def test_decide_accepts_a_similarity_equal_to_the_threshold() -> None:
    result = decide(0.766, "Ana", 0.766)

    assert (result.status, result.person_name) == (RecognitionStatus.RECOGNIZED, "Ana")


def test_decide_rejects_a_similarity_just_below_the_threshold() -> None:
    result = decide(0.765, "Ana", 0.766)

    assert (result.status, result.person_name, result.similarity) == (
        RecognitionStatus.UNKNOWN,
        None,
        0.765,
    )


def test_decide_accepts_a_similarity_above_the_threshold() -> None:
    assert decide(0.99, "Bia", 0.766).person_name == "Bia"
