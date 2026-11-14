"""Enrollment tests (spec 002): fakes for the detector and the model, a real in-memory database."""

import logging
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.enrollment import EnrollmentService, find_duplicate
from app.core.errors import DuplicatePersonError
from app.core.models import EnrollmentRequest
from app.storage.models import Gallery
from app.storage.repository import SqlitePersonRepository
from facial_recognition_system.preprocessing import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from facial_recognition_system.recognition.strategies import NearestNeighbor
from tests.app_support import (
    ANA,
    ANA_COLOR,
    BIA_COLOR,
    CARLA_COLOR,
    DIMENSION,
    MODEL_ID,
    NO_FACE,
    THRESHOLD,
    TWO_FACES,
    build_encoder,
    photo,
    request,
)

pytestmark = pytest.mark.unit


def build_service(
    repository: SqlitePersonRepository, threshold: float = THRESHOLD
) -> EnrollmentService:
    return EnrollmentService(build_encoder(), repository, threshold)


def test_one_photo_stores_one_embedding_of_512_dimensions(
    repository: SqlitePersonRepository,
) -> None:
    result = build_service(repository).enroll(request("Ana", photo(ANA_COLOR)))

    assert (result.name, result.embedding_count) == ("Ana", 1)
    gallery = repository.gallery(MODEL_ID)
    assert gallery.embeddings.shape == (1, DIMENSION)
    assert gallery.labels == ["Ana"]


def test_three_photos_store_three_embeddings(repository: SqlitePersonRepository) -> None:
    result = build_service(repository).enroll(
        request("Ana", photo(ANA_COLOR), photo(ANA_COLOR), photo(ANA_COLOR))
    )

    assert result.embedding_count == 3
    assert [p.embedding_count for p in repository.list_persons()] == [3]


@pytest.mark.parametrize(
    ("bad_photo", "error"),
    [
        (photo(width=NO_FACE), NoFaceDetectedError),
        (photo(width=TWO_FACES), MultipleFacesError),
        (b"this is not an image", InvalidImageError),
        (b"", InvalidImageError),
    ],
)
def test_one_bad_photo_raises_its_error_and_stores_nothing(
    repository: SqlitePersonRepository, bad_photo: bytes, error: type[Exception]
) -> None:
    service = build_service(repository)

    with pytest.raises(error):
        service.enroll(request("Ana", photo(ANA_COLOR), bad_photo, photo(ANA_COLOR)))

    assert repository.list_persons() == []
    assert repository.gallery(MODEL_ID).labels == []


def test_a_repeated_name_is_a_duplicate_by_name_and_leaves_the_enrollment_alone(
    repository: SqlitePersonRepository,
) -> None:
    service = build_service(repository)
    service.enroll(request("Ana", photo(ANA_COLOR)))

    with pytest.raises(DuplicatePersonError) as error:
        service.enroll(request("ana", photo(BIA_COLOR)))

    assert error.value.reason == "name"
    assert [(p.name, p.embedding_count) for p in repository.list_persons()] == [("Ana", 1)]


def test_a_face_too_similar_to_an_enrolled_one_is_a_duplicate_by_similarity(
    repository: SqlitePersonRepository,
) -> None:
    service = build_service(repository)
    service.enroll(request("Ana", photo(ANA_COLOR)))

    with pytest.raises(DuplicatePersonError) as error:
        service.enroll(request("Carla", photo(CARLA_COLOR)))  # cosine 0.9 with Ana

    assert error.value.reason == "similarity"
    assert error.value.name == "Ana"
    assert [p.name for p in repository.list_persons()] == ["Ana"]


def test_a_different_face_is_enrolled_normally(repository: SqlitePersonRepository) -> None:
    service = build_service(repository)
    service.enroll(request("Ana", photo(ANA_COLOR)))

    result = service.enroll(request("Bia", photo(BIA_COLOR)))  # cosine ~0 with Ana

    assert result.name == "Bia"
    assert [p.name for p in repository.list_persons()] == ["Ana", "Bia"]


def test_the_threshold_decides_which_faces_are_duplicates(
    repository: SqlitePersonRepository,
) -> None:
    service = build_service(repository, threshold=0.95)  # above the 0.9 of Carla
    service.enroll(request("Ana", photo(ANA_COLOR)))

    assert service.enroll(request("Carla", photo(CARLA_COLOR))).name == "Carla"


def test_a_name_that_is_not_valid_never_reaches_the_database() -> None:
    with pytest.raises(ValidationError):
        request("   ", photo())
    with pytest.raises(ValidationError):
        request("a" * 101, photo())


@pytest.mark.parametrize("count", [0, 11])
def test_zero_or_more_than_ten_photos_are_refused_before_any_work(count: int) -> None:
    with pytest.raises(ValidationError):
        EnrollmentRequest(name="Ana", photos=[photo()] * count)


def test_no_photo_is_written_to_disk_and_the_tables_hold_only_embeddings(tmp_path: Path) -> None:
    folder = tmp_path / "private"
    repository = SqlitePersonRepository(folder / "app.db")
    build_service(repository).enroll(request("Ana", photo(ANA_COLOR), photo(ANA_COLOR)))
    columns = {
        table: {row[1] for row in repository._connection.execute(f"PRAGMA table_info({table})")}
        for table in ("persons", "embeddings")
    }
    repository.close()

    assert [path.name for path in folder.iterdir()] == ["app.db"]
    assert columns["persons"] == {"id", "name", "name_key", "created_at"}
    assert columns["embeddings"] == {
        "id",
        "person_id",
        "vector",
        "dimension",
        "model_id",
        "created_at",
    }


def test_an_enrolled_person_is_recognized_without_restarting(
    repository: SqlitePersonRepository,
) -> None:
    build_service(repository).enroll(request("Ana", photo(ANA_COLOR)))

    gallery = repository.gallery(MODEL_ID)
    prediction = NearestNeighbor().fit(gallery.embeddings, gallery.labels).predict(ANA[None, :])

    assert prediction.labels == ["Ana"]
    assert prediction.scores[0] == pytest.approx(1.0, abs=1e-6)


def test_the_log_has_the_id_and_the_count_but_not_the_name(
    repository: SqlitePersonRepository, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="app.core.enrollment"):
        build_service(repository).enroll(request("Ana Souza", photo(ANA_COLOR), photo(ANA_COLOR)))

    (record,) = caplog.records
    assert (record.person_id, record.embedding_count) == (1, 2)  # type: ignore[attr-defined]
    assert "Ana" not in caplog.text and "Souza" not in caplog.text


def gallery_of(rows: list[list[float]], labels: list[str]) -> Gallery:
    return Gallery(embeddings=np.array(rows, dtype=np.float32), labels=labels)


def test_find_duplicate_with_an_empty_gallery_finds_nothing() -> None:
    empty = Gallery(embeddings=np.empty((0, 0), dtype=np.float32), labels=[])

    assert find_duplicate(empty, [ANA], THRESHOLD) is None


def test_find_duplicate_reaches_the_threshold_with_a_greater_or_equal_similarity() -> None:
    gallery = gallery_of([[1.0, 0.0]], ["Ana"])
    new = [np.array([0.5, 0.5], dtype=np.float32)]  # similarity exactly 0.5

    assert find_duplicate(gallery, new, 0.5) == "Ana"
    assert find_duplicate(gallery, new, 0.5000001) is None


def test_find_duplicate_names_the_owner_of_the_best_match_among_all_new_photos() -> None:
    gallery = gallery_of([[1.0, 0.0], [0.0, 1.0]], ["Ana", "Bia"])
    new = [np.array([0.6, 0.0], dtype=np.float32), np.array([0.1, 0.9], dtype=np.float32)]

    assert find_duplicate(gallery, new, 0.8) == "Bia"
    assert find_duplicate(gallery, new, 0.95) is None
