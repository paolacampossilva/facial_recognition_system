"""Tests of the logs of the application: allowed fields only, never personal data (spec 007)."""

import logging
import sys
from collections.abc import Iterator

import pytest

from app.core.deletion import DeletionService
from app.core.enrollment import EnrollmentService
from app.core.recognition import RecognitionService
from app.logs import APP_LOGGER, AllowListFormatter, configure_logging
from app.storage.repository import SqlitePersonRepository
from tests.app_support import ANA_COLOR, THRESHOLD, build_encoder, photo, request

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_app_logger() -> Iterator[None]:
    """`configure_logging` changes the shared `app` logger; put it back for the other tests."""
    logger = logging.getLogger(APP_LOGGER)
    saved = (logger.handlers[:], logger.level, logger.propagate)
    yield
    logger.handlers, logger.level, logger.propagate = saved[0], saved[1], saved[2]


def record_with(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("app.core.test", logging.INFO, __file__, 1, "event", None, None)
    record.__dict__.update(extra)
    return record


def test_a_field_outside_the_allow_list_never_reaches_the_line() -> None:
    line = AllowListFormatter().format(record_with(name_of_person="Ana", person_id=3))

    assert line.endswith("INFO app.core.test: event person_id=3")
    assert "Ana" not in line


def test_the_allowed_fields_come_before_the_traceback() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = record_with(status="unknown")
        record.exc_info = sys.exc_info()

    first_line, *rest = AllowListFormatter().format(record).splitlines()

    assert first_line.endswith("event status=unknown")
    assert rest[0] == "Traceback (most recent call last):"


def test_enrolling_recognizing_and_deleting_logs_no_name_and_no_embedding(
    repository: SqlitePersonRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("INFO")
    encoder = build_encoder()

    EnrollmentService(encoder, repository, THRESHOLD).enroll(
        request("Ana Teste", photo(ANA_COLOR), photo(ANA_COLOR))
    )
    RecognitionService(encoder, repository, THRESHOLD).recognize(photo(ANA_COLOR))
    DeletionService(repository).delete(1)

    lines = capsys.readouterr().err.splitlines()
    assert [line.split(": ", 1)[1] for line in lines] == [
        "person enrolled person_id=1 embedding_count=2",
        "face recognized status=recognized similarity=1.0",
        "person deleted person_id=1 deleted_embeddings=2",
    ]
    # The exact lines leave no room for an embedding; the name is checked on its own anyway.
    text = "\n".join(lines)
    assert "Ana" not in text and "Teste" not in text


def test_above_info_the_info_lines_are_not_written(
    repository: SqlitePersonRepository, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("WARNING")

    EnrollmentService(build_encoder(), repository, THRESHOLD).enroll(
        request("Ana", photo(ANA_COLOR))
    )

    assert capsys.readouterr().err == ""


def test_configuring_twice_does_not_duplicate_the_lines(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    configure_logging("INFO")

    logging.getLogger("app.core.test").info("once")

    assert capsys.readouterr().err.count("once") == 1


PERSONAL_MESSAGE: str = "database /private/data.db password=hunter2 Ana Souza"


def test_a_traceback_names_the_exception_but_never_its_message() -> None:
    try:
        raise RuntimeError(PERSONAL_MESSAGE)
    except RuntimeError:
        record = record_with()
        record.exc_info = sys.exc_info()

    text = AllowListFormatter().format(record)

    assert text.endswith("RuntimeError")
    assert "Traceback (most recent call last):" in text
    assert "Ana Souza" not in text
    assert "hunter2" not in text
