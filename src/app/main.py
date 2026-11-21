"""Entrypoint of the server: `uvicorn app.main:create --factory --host 127.0.0.1 --port 8000`.

Nothing is loaded when this module is imported. The server calls `create` once it is ready to
start, so a missing model or a wrong setting fails there, where the launcher can report it.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.app import create_app
from app.config import Settings, load_settings
from app.core.encoder import FaceEncoder
from app.logs import configure_logging
from app.services import build_services, load_encoder
from app.storage.repository import SqlitePersonRepository


def build_application(settings: Settings, encoder: FaceEncoder) -> FastAPI:
    """Opens the database and builds the application; the database closes when the server stops."""
    repository = SqlitePersonRepository(settings.database_path)

    @asynccontextmanager
    async def close_database_on_exit(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            repository.close()

    services = build_services(settings, encoder, repository)
    return create_app(services, settings, lifespan=close_database_on_exit)


def create() -> FastAPI:
    """Reads the configuration, loads the model and builds the application.

    Raises:
        ConfigError: A setting is invalid.
        ModelLoadError: The checkpoint is missing or invalid, or the device is unavailable.
    """
    settings = load_settings()
    configure_logging(settings.log_level)
    return build_application(settings, load_encoder(settings))
