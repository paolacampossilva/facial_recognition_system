"""Entrypoint: `uvicorn app.main:app --host 127.0.0.1 --port 8000`."""

from app.api.app import create_app
from app.fakes import FakeServices

# TODO(#48): the threshold moves to the Settings class of the configuration module (spec 007).
THRESHOLD: float = 0.766  # decision D08

# TODO(#49): replace the fake services with the real ones and the configured model.
app = create_app(FakeServices(), THRESHOLD)
