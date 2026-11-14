"""Fixtures shared by the tests of the application."""

from collections.abc import Iterator

import pytest

from app.storage.repository import MEMORY_DATABASE, SqlitePersonRepository


@pytest.fixture
def repository() -> Iterator[SqlitePersonRepository]:
    """An empty in-memory database, closed after the test."""
    repo = SqlitePersonRepository(MEMORY_DATABASE)
    yield repo
    repo.close()
