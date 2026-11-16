"""Fixtures shared by the tests of the application."""

import os
from collections.abc import Iterator

import pytest

from app import config
from app.storage.repository import MEMORY_DATABASE, SqlitePersonRepository


@pytest.fixture(autouse=True)
def default_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the default settings: no .env and no FACE_APP_ variable of whoever
    runs the tests (spec 007). A test that needs one sets it itself."""
    monkeypatch.setattr(config, "ENV_FILE", None)
    for name in list(os.environ):
        if name.startswith(config.ENV_PREFIX):
            monkeypatch.delenv(name)


@pytest.fixture
def repository() -> Iterator[SqlitePersonRepository]:
    """An empty in-memory database, closed after the test."""
    repo = SqlitePersonRepository(MEMORY_DATABASE)
    yield repo
    repo.close()
