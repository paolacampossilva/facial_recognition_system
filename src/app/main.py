"""Entrypoint: `uvicorn app.main:app --host 127.0.0.1 --port 8000`."""

from app.api.app import create_app
from app.config import load_settings
from app.logs import configure_logging
from app.services import build_services, load_encoder
from app.storage.repository import SqlitePersonRepository

settings = load_settings()
configure_logging(settings.log_level)

encoder = load_encoder(settings)
repository = SqlitePersonRepository(settings.database_path)
app = create_app(build_services(settings, encoder, repository), settings)
