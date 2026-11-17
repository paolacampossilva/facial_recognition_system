"""Tests of the SQLite repository (D12, specs 002, 003 and 006)."""

import os
import sqlite3
import stat
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from app.core.errors import CorruptEmbeddingError, DuplicatePersonError, PersonNotFoundError
from app.storage.models import NewPerson
from app.storage.repository import MEMORY_DATABASE, SqlitePersonRepository, name_key

pytestmark = pytest.mark.unit

DIMENSION: int = 512
MODEL: str = "facenet-e06"


def vector(seed: int, dimension: int = DIMENSION) -> np.ndarray:
    return np.random.default_rng(seed).random(dimension, dtype=np.float32)


def person(name: str, *seeds: int, model_id: str = MODEL) -> NewPerson:
    return NewPerson(name=name, model_id=model_id, vectors=[vector(seed) for seed in seeds])


@pytest.fixture
def repository() -> Iterator[SqlitePersonRepository]:
    repo = SqlitePersonRepository(MEMORY_DATABASE)
    yield repo
    repo.close()


def rows(repo: SqlitePersonRepository, sql: str) -> list[tuple]:
    return repo._connection.execute(sql).fetchall()


def test_a_new_database_is_created_empty_with_both_tables_and_the_pragmas_on(
    repository: SqlitePersonRepository,
) -> None:
    tables = {
        name for (name,) in rows(repository, "SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"persons", "embeddings"} <= tables
    assert repository.list_persons() == []
    assert rows(repository, "PRAGMA foreign_keys") == [(1,)]
    assert rows(repository, "PRAGMA secure_delete") == [(1,)]
    assert rows(repository, "PRAGMA user_version") == [(1,)]


def test_a_vector_comes_back_as_float32_with_its_dimension_and_model(
    repository: SqlitePersonRepository,
) -> None:
    repository.add_person(person("Ana", 1))
    ((blob, dimension, model_id),) = rows(
        repository, "SELECT vector, dimension, model_id FROM embeddings"
    )
    assert dimension == DIMENSION and model_id == MODEL
    assert np.array_equal(np.frombuffer(blob, dtype=np.float32), vector(1))


def test_people_are_listed_in_enrollment_order_with_their_embedding_count(
    repository: SqlitePersonRepository,
) -> None:
    ana = repository.add_person(person("Ana", 1, 2, 3))
    bia = repository.add_person(person("Bia", 4))
    assert (ana.id, ana.embedding_count) == (1, 3)
    assert [(p.name, p.embedding_count) for p in repository.list_persons()] == [
        ("Ana", 3),
        ("Bia", 1),
    ]
    assert bia.id == 2


@pytest.mark.parametrize(
    ("first", "second"), [("Ana", "ana"), ("Ágata", "ágata"), ("JOSÉ", "josé")]
)
def test_a_repeated_name_ignoring_case_and_accents_is_a_duplicate_and_changes_nothing(
    repository: SqlitePersonRepository, first: str, second: str
) -> None:
    repository.add_person(person(first, 1))
    with pytest.raises(DuplicatePersonError) as error:
        repository.add_person(person(second, 2))
    assert error.value.reason == "name"
    assert [p.name for p in repository.list_persons()] == [first]
    assert rows(repository, "SELECT COUNT(*) FROM embeddings") == [(1,)]


def test_the_name_key_folds_case_and_composed_accents() -> None:
    assert name_key("Ágata") == name_key("ágata") == name_key("Ágata")


def test_a_failure_in_the_middle_of_an_enrollment_stores_nothing(
    repository: SqlitePersonRepository,
) -> None:
    repository._connection.execute(
        "CREATE TEMP TRIGGER fail_second AFTER INSERT ON embeddings WHEN NEW.id = 2"
        " BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.add_person(person("Ana", 1, 2, 3))
    assert repository.list_persons() == []
    assert rows(repository, "SELECT COUNT(*) FROM embeddings") == [(0,)]


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [np.zeros((2, 2), dtype=np.float32)],
        [np.array([np.nan, 1.0], dtype=np.float32)],
        [np.ones(4, dtype=np.float32), np.ones(5, dtype=np.float32)],
    ],
)
def test_invalid_vectors_are_rejected_before_the_database_is_touched(vectors: list) -> None:
    with pytest.raises(ValidationError):
        NewPerson(name="Ana", model_id=MODEL, vectors=vectors)


@pytest.mark.parametrize("name", ["", "   ", "a" * 101])
def test_an_invalid_name_is_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        NewPerson(name=name, model_id=MODEL, vectors=[vector(1)])


def test_the_gallery_has_one_row_per_photo_and_the_name_of_each_owner(
    repository: SqlitePersonRepository,
) -> None:
    repository.add_person(person("Ana", 1, 2))
    repository.add_person(person("Bia", 3))
    gallery = repository.gallery(MODEL)
    assert gallery.embeddings.shape == (3, DIMENSION)
    assert gallery.embeddings.dtype == np.float32
    assert gallery.labels == ["Ana", "Ana", "Bia"]
    assert np.array_equal(gallery.embeddings[2], vector(3))


def test_the_gallery_ignores_embeddings_of_other_models(repository: SqlitePersonRepository) -> None:
    repository.add_person(person("Ana", 1, model_id="model-a"))
    repository.add_person(person("Bia", 2, model_id="model-b"))
    assert repository.gallery("model-a").labels == ["Ana"]
    assert repository.gallery("model-b").labels == ["Bia"]


def test_an_empty_gallery_is_an_empty_matrix_and_no_labels(
    repository: SqlitePersonRepository,
) -> None:
    repository.add_person(person("Ana", 1, model_id="model-a"))
    gallery = repository.gallery("other-model")
    assert gallery.embeddings.shape == (0, 0)
    assert gallery.labels == []


def test_a_corrupt_embedding_is_reported_when_the_gallery_is_read(
    repository: SqlitePersonRepository,
) -> None:
    repository.add_person(person("Ana", 1))
    repository._connection.execute("UPDATE embeddings SET dimension = 7")
    with pytest.raises(CorruptEmbeddingError):
        repository.gallery(MODEL)


def test_deleting_a_person_cascades_to_their_embeddings_and_spares_the_others(
    repository: SqlitePersonRepository,
) -> None:
    repository.add_person(person("Ana", 1, 2, 3))
    repository.add_person(person("Bia", 4, 5))
    result = repository.delete_person(1)
    assert (result.person_id, result.deleted_embeddings) == (1, 3)
    assert [(p.name, p.embedding_count) for p in repository.list_persons()] == [("Bia", 2)]
    assert rows(repository, "SELECT COUNT(*) FROM embeddings WHERE person_id = 1") == [(0,)]
    assert rows(repository, "SELECT COUNT(*) FROM embeddings WHERE person_id = 2") == [(2,)]


def test_a_deleted_name_can_be_enrolled_again(repository: SqlitePersonRepository) -> None:
    repository.add_person(person("Ana", 1))
    repository.delete_person(1)
    assert repository.add_person(person("Ana", 2)).name == "Ana"


def test_deleting_an_unknown_id_raises_and_changes_nothing(
    repository: SqlitePersonRepository,
) -> None:
    repository.add_person(person("Ana", 1))
    with pytest.raises(PersonNotFoundError) as error:
        repository.delete_person(99)
    assert error.value.person_id == 99
    assert [p.name for p in repository.list_persons()] == ["Ana"]
    assert rows(repository, "SELECT COUNT(*) FROM embeddings") == [(1,)]


def test_a_failure_during_the_deletion_removes_nothing(repository: SqlitePersonRepository) -> None:
    repository.add_person(person("Ana", 1, 2))
    repository._connection.execute(
        "CREATE TEMP TRIGGER fail_delete BEFORE DELETE ON persons"
        " BEGIN SELECT RAISE(ABORT, 'boom'); END"
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.delete_person(1)
    assert [(p.name, p.embedding_count) for p in repository.list_persons()] == [("Ana", 2)]


def test_a_deleted_vector_is_gone_from_the_database_file(tmp_path: Path) -> None:
    path = tmp_path / "private" / "app.db"
    repo = SqlitePersonRepository(path)
    secret = vector(42).tobytes()
    repo.add_person(person("Ana", 42))
    assert secret in path.read_bytes()  # the check below only means something if it was there
    repo.delete_person(1)
    repo.close()
    assert secret not in path.read_bytes()


def test_the_database_file_and_its_folder_are_created_and_private(tmp_path: Path) -> None:
    path = tmp_path / "data" / "private" / "app.db"
    SqlitePersonRepository(path).close()
    assert path.is_file()
    if sys.platform != "win32":  # Windows has no POSIX permission bits
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX permission bits")
def test_the_database_is_private_from_the_moment_it_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the final chmod turned off, only the creation itself can have made the file private."""
    monkeypatch.setattr(os, "chmod", lambda *args, **kwargs: None)
    previous_umask = os.umask(0o000)  # a permissive umask, as the worst case
    try:
        path = tmp_path / "data" / "private" / "app.db"
        SqlitePersonRepository(path).close()
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX permission bits")
def test_a_database_an_older_version_left_open_to_others_is_made_private(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    sqlite3.connect(path).close()
    path.chmod(0o644)

    SqlitePersonRepository(path).close()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_data_survives_reopening_the_file(tmp_path: Path) -> None:
    path = tmp_path / "app.db"
    first = SqlitePersonRepository(path)
    first.add_person(person("Ana", 1, 2))
    first.close()
    second = SqlitePersonRepository(path)
    assert [(p.name, p.embedding_count) for p in second.list_persons()] == [("Ana", 2)]
    assert second.gallery(MODEL).embeddings.shape == (2, DIMENSION)
    second.close()


def test_the_repository_can_be_used_from_another_thread(
    repository: SqlitePersonRepository,
) -> None:
    with ThreadPoolExecutor(max_workers=1) as pool:
        summary = pool.submit(repository.add_person, person("Ana", 1)).result()
        listed = pool.submit(repository.list_persons).result()
    assert listed == [summary]
