"""Tests that run a real server on the loopback address, in a thread.

They use sockets (never anything outside this computer), so they are integration tests, not unit
tests: what they check needs a running server, such as which logger prints a traceback, whether
the event loop stays free while a request works, and how the browser waits for the server.
"""

import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.api.app import create_app
from app.config import Settings
from app.core.models import RecognitionResult, RecognitionStatus
from app.launcher import wait_for_server
from app.logs import APP_LOGGER, configure_logging, uvicorn_log_config
from tests.app_support import THRESHOLD, free_port, serving

pytestmark = pytest.mark.integration

PERSONAL_MESSAGE: str = "database /private/data.db password=hunter2 Ana Souza"


@pytest.fixture(autouse=True)
def restore_app_logger() -> Iterator[None]:
    """`configure_logging` changes the shared `app` logger; put it back for the other tests."""
    logger = logging.getLogger(APP_LOGGER)
    saved = (logger.handlers[:], logger.level, logger.propagate)
    yield
    logger.handlers, logger.level, logger.propagate = saved


class ExplodingServices:
    """Services whose every call fails with a message that holds personal data."""

    @property
    def persons(self) -> "ExplodingServices":
        return self

    def list_persons(self) -> list[object]:
        raise RuntimeError(PERSONAL_MESSAGE)


def test_an_unexpected_error_never_prints_its_message_through_the_server_logger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    application = create_app(ExplodingServices(), Settings(threshold=THRESHOLD))  # type: ignore[arg-type]

    with (
        serving(application, log_level="info", log_config=uvicorn_log_config()) as url,
        pytest.raises(urllib.error.HTTPError) as response,
    ):
        urllib.request.urlopen(f"{url}/api/persons")  # noqa: S310

    printed = capsys.readouterr().err
    assert response.value.code == 500
    assert "RuntimeError" in printed
    assert "Ana Souza" not in printed
    assert "hunter2" not in printed


SLOW_SECONDS: float = 1.0
BOUNDARY: str = "testboundary"


class SlowRecognition:
    """Services whose recognition takes a second, as a real model can."""

    @property
    def recognition(self) -> "SlowRecognition":
        return self

    def recognize(self, image: bytes) -> RecognitionResult:
        time.sleep(SLOW_SECONDS)
        return RecognitionResult(status=RecognitionStatus.UNKNOWN, person_name=None, similarity=0.1)


def post_a_photo(url: str) -> None:
    body = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="photo"; filename="a.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n\r\nxx\r\n--{BOUNDARY}--\r\n"
    ).encode()
    request = urllib.request.Request(  # noqa: S310
        f"{url}/api/recognitions",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    urllib.request.urlopen(request).read()  # noqa: S310


def test_a_slow_recognition_does_not_block_the_other_requests() -> None:
    """The model runs on a worker thread, so the server keeps answering while it works."""
    application = create_app(SlowRecognition(), Settings(threshold=THRESHOLD))  # type: ignore[arg-type]

    with serving(application) as url:
        slow = threading.Thread(target=post_a_photo, args=(url,))
        slow.start()
        time.sleep(0.2)  # the recognition is running by now
        started = time.monotonic()
        urllib.request.urlopen(f"{url}/api/health").read()  # noqa: S310
        waited = time.monotonic() - started
        slow.join()

    assert waited < SLOW_SECONDS / 2


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (the name the standard library calls)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (the base name)
        pass


def test_wait_for_server_returns_true_once_the_health_route_answers() -> None:
    server = HTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    answered = wait_for_server(server.server_address[1], timeout=5)

    server.shutdown()
    assert answered is True


def test_wait_for_server_gives_up_when_nothing_listens() -> None:
    assert wait_for_server(free_port(), timeout=0.6) is False
