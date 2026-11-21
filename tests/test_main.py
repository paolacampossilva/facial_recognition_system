"""Tests of how the application is built for the server: wiring and the database lifecycle."""

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import build_application
from tests.app_support import ANA_COLOR, THRESHOLD, build_encoder, photo

pytestmark = pytest.mark.unit


def settings_in(folder: Path) -> Settings:
    return Settings(threshold=THRESHOLD, database_path=folder / "data" / "app.db")


def test_the_application_works_over_the_configured_database(tmp_path: Path) -> None:
    application = build_application(settings_in(tmp_path), build_encoder())

    with TestClient(application, base_url="http://127.0.0.1") as client:
        created = client.post(
            "/api/persons",
            data={"name": "Ana"},
            files=[("photos", ("a.png", photo(ANA_COLOR), "image/png"))],
        )
        names = [person["name"] for person in client.get("/api/persons").json()]

    assert created.status_code == 201
    assert names == ["Ana"]
    assert (tmp_path / "data" / "app.db").is_file()


def test_the_database_is_closed_when_the_server_stops(tmp_path: Path) -> None:
    application = build_application(settings_in(tmp_path), build_encoder())
    repository = application.state.services.persons

    with TestClient(application, base_url="http://127.0.0.1"):
        repository.list_persons()  # works while the server runs

    with pytest.raises(sqlite3.ProgrammingError):
        repository.list_persons()
