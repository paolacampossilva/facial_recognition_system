"""The wiring of the real services: with a fake encoder and an in-memory database, no model."""

from collections.abc import Iterator

import pytest

from app.config import Settings
from app.core.models import RecognitionStatus
from app.services import build_services
from app.storage.repository import MEMORY_DATABASE, SqlitePersonRepository
from tests.app_support import ANA_COLOR, BIA_COLOR, build_encoder, photo, request


@pytest.fixture
def repository() -> Iterator[SqlitePersonRepository]:
    repo = SqlitePersonRepository(MEMORY_DATABASE)
    yield repo
    repo.close()


@pytest.mark.unit
def test_the_services_share_one_repository(repository: SqlitePersonRepository) -> None:
    services = build_services(Settings(), build_encoder(), repository)

    enrolled = services.enrollment.enroll(request("Ana", photo(ANA_COLOR)))

    assert [person.name for person in services.persons.list_persons()] == ["Ana"]
    assert services.recognition.recognize(photo(ANA_COLOR)).status == RecognitionStatus.RECOGNIZED
    services.deletion.delete(enrolled.person_id)
    assert services.persons.list_persons() == []


@pytest.mark.unit
def test_the_configured_threshold_decides_who_is_unknown(
    repository: SqlitePersonRepository,
) -> None:
    services = build_services(Settings(threshold=1.0), build_encoder(), repository)
    services.enrollment.enroll(request("Ana", photo(ANA_COLOR)))

    result = services.recognition.recognize(photo(BIA_COLOR))

    assert result.status == RecognitionStatus.UNKNOWN
