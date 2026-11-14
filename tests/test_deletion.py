"""Deletion tests (spec 006): the service over a real in-memory or on-disk database."""

import logging
from pathlib import Path

import pytest

from app.core.deletion import DeletionService
from app.core.enrollment import EnrollmentService
from app.core.errors import EmptyGalleryError, PersonNotFoundError
from app.core.models import DeletionResult, RecognitionStatus
from app.core.recognition import RecognitionService
from app.storage.repository import SqlitePersonRepository
from tests.app_support import (
    ANA,
    ANA_COLOR,
    BIA_COLOR,
    THRESHOLD,
    build_encoder,
    photo,
    request,
)

pytestmark = pytest.mark.unit


def enroll(repository: SqlitePersonRepository, name: str, *colors: tuple[int, int, int]) -> None:
    service = EnrollmentService(build_encoder(), repository, THRESHOLD)
    service.enroll(request(name, *[photo(color) for color in colors]))


def count(repository: SqlitePersonRepository, sql: str) -> int:
    return repository._connection.execute(sql).fetchone()[0]


def test_deleting_one_person_leaves_the_others_untouched(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, ANA_COLOR, ANA_COLOR)
    enroll(repository, "Bia", BIA_COLOR, BIA_COLOR)

    DeletionService(repository).delete(1)

    assert [(p.name, p.embedding_count) for p in repository.list_persons()] == [("Bia", 2)]
    assert count(repository, "SELECT COUNT(*) FROM embeddings WHERE person_id = 1") == 0
    assert count(repository, "SELECT COUNT(*) FROM embeddings WHERE person_id = 2") == 2


def test_the_result_has_the_id_and_how_many_embeddings_were_removed(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, ANA_COLOR, ANA_COLOR)

    result = DeletionService(repository).delete(1)

    assert result == DeletionResult(person_id=1, deleted_embeddings=3)


def test_a_deleted_person_is_no_longer_recognized_and_nobody_left_is_an_empty_gallery(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    enroll(repository, "Bia", BIA_COLOR)
    recognition = RecognitionService(build_encoder(), repository, THRESHOLD)
    assert recognition.recognize(photo(ANA_COLOR)).person_name == "Ana"

    DeletionService(repository).delete(1)
    after_ana = recognition.recognize(photo(ANA_COLOR))
    DeletionService(repository).delete(2)

    assert after_ana.status is RecognitionStatus.UNKNOWN
    with pytest.raises(EmptyGalleryError):
        recognition.recognize(photo(ANA_COLOR))


def test_a_deleted_name_can_be_enrolled_again(repository: SqlitePersonRepository) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    DeletionService(repository).delete(1)

    enroll(repository, "Ana", ANA_COLOR)

    assert [p.name for p in repository.list_persons()] == ["Ana"]


def test_deleting_an_unknown_id_raises_and_changes_nothing(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)

    with pytest.raises(PersonNotFoundError):
        DeletionService(repository).delete(99)

    assert [p.name for p in repository.list_persons()] == ["Ana"]
    assert count(repository, "SELECT COUNT(*) FROM embeddings") == 1


def test_a_failure_in_the_middle_of_the_deletion_removes_nothing(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR, ANA_COLOR)
    repository._connection.execute(
        "CREATE TEMP TRIGGER fail_delete BEFORE DELETE ON persons"
        " BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )

    with pytest.raises(Exception, match="boom"):
        DeletionService(repository).delete(1)

    assert [(p.name, p.embedding_count) for p in repository.list_persons()] == [("Ana", 2)]


def test_an_error_of_the_repository_reaches_the_caller_and_nothing_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenRepository:
        def delete_person(self, person_id: int) -> DeletionResult:
            raise RuntimeError("disk full")

    service = DeletionService(BrokenRepository())  # type: ignore[arg-type]

    with caplog.at_level(logging.INFO, logger="app.core.deletion"), pytest.raises(RuntimeError):
        service.delete(1)

    assert caplog.records == []


def test_foreign_keys_and_secure_delete_stay_on_in_the_connection_the_service_uses(
    repository: SqlitePersonRepository,
) -> None:
    enroll(repository, "Ana", ANA_COLOR)
    DeletionService(repository).delete(1)

    assert repository._connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert repository._connection.execute("PRAGMA secure_delete").fetchone() == (1,)


def test_the_log_has_the_id_and_the_count_but_not_the_name(
    repository: SqlitePersonRepository, caplog: pytest.LogCaptureFixture
) -> None:
    enroll(repository, "Ana Souza", ANA_COLOR, ANA_COLOR)

    with caplog.at_level(logging.INFO, logger="app.core.deletion"):
        DeletionService(repository).delete(1)

    (record,) = caplog.records
    assert (record.person_id, record.deleted_embeddings) == (1, 2)  # type: ignore[attr-defined]
    assert "Ana" not in caplog.text and "Souza" not in caplog.text


def test_after_enrolling_and_deleting_the_vector_is_gone_from_the_database_file(
    tmp_path: Path,
) -> None:
    database = tmp_path / "private" / "app.db"
    repository = SqlitePersonRepository(database)
    enroll(repository, "Ana", ANA_COLOR)
    secret = ANA.tobytes()
    assert secret in database.read_bytes()  # the check below only means something if it was there

    DeletionService(repository).delete(1)
    repository.close()

    assert secret not in database.read_bytes()
