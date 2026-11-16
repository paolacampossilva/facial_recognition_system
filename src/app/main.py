"""Entrypoint: `uvicorn app.main:app --host 127.0.0.1 --port 8000`."""

from app.api.app import create_app
from app.config import load_settings
from app.fakes import FakeServices
from app.logs import configure_logging

settings = load_settings()
configure_logging(settings.log_level)

# TODO(#49): replace the fake services with the real ones and the configured model.
app = create_app(FakeServices(), settings)
