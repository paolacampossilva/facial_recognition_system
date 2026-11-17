"""Tests of `python -m app`: options, port check and startup, with no server and no browser."""

import runpy
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import torch

from app import launcher
from app.config import Settings
from app.core.errors import LaunchError, ModelLoadError
from app.launcher import (
    LaunchOptions,
    check_model_setup,
    main,
    parse_options,
    port_is_free,
    run,
    wait_for_server,
)
from app.logs import uvicorn_log_config
from tests.app_support import free_port

pytestmark = pytest.mark.unit


class FakeServer:
    """Records how the server was started instead of starting it."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


class FakeBrowser:
    """Records the URLs opened and signals when one is."""

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.opened = threading.Event()

    def __call__(self, url: str) -> None:
        self.urls.append(url)
        self.opened.set()


@pytest.fixture
def busy_port() -> Iterator[int]:
    """A port on the loopback address that something is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        yield holder.getsockname()[1]


@pytest.fixture(autouse=True)
def pretrained_model(default_configuration: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests start no real server, so they must not need the E06 checkpoint on disk."""
    monkeypatch.setenv("FACE_APP_CHECKPOINT_PATH", "")


def test_the_defaults_are_port_8000_with_the_browser_and_without_reload() -> None:
    assert parse_options([]) == LaunchOptions(port=8000, open_browser=True, reload=False)


def test_each_option_is_read() -> None:
    options = parse_options(["--port", "9000", "--no-browser", "--reload"])

    assert options == LaunchOptions(port=9000, open_browser=False, reload=True)


@pytest.mark.parametrize("port", ["0", "80", "1023", "65536", "70000", "-5"])
def test_a_port_outside_the_allowed_range_exits_with_a_usage_error(
    port: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_options(["--port", port])

    assert exit_info.value.code == 2
    assert "--port must be between" in capsys.readouterr().err


def test_a_port_that_is_not_a_number_exits_with_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_options(["--port", "eight"])

    assert exit_info.value.code == 2


def test_there_is_no_option_to_listen_on_the_network() -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_options(["--host", "0.0.0.0"])

    assert exit_info.value.code == 2


def test_a_free_port_is_free_and_a_busy_one_is_not(busy_port: int) -> None:
    assert port_is_free(free_port()) is True
    assert port_is_free(busy_port) is False


def test_the_server_starts_on_the_loopback_address_with_the_given_port() -> None:
    server = FakeServer()
    port = free_port()

    run(LaunchOptions(port=port, open_browser=False), serve=server, open_browser=FakeBrowser())

    ((args, kwargs),) = server.calls
    assert args == ("app.main:create",)
    assert kwargs == {
        "factory": True,
        "host": "127.0.0.1",
        "port": port,
        "reload": False,
        "reload_dirs": None,
        "log_level": "info",
        "log_config": uvicorn_log_config(),
    }


def ready(port: int) -> bool:
    """A server that already answers."""
    return True


def test_the_browser_opens_on_the_right_url() -> None:
    browser = FakeBrowser()
    port = free_port()

    run(LaunchOptions(port=port), serve=FakeServer(), open_browser=browser, wait=ready)

    assert browser.opened.wait(timeout=2)
    assert browser.urls == [f"http://127.0.0.1:{port}"]


def test_the_browser_stays_closed_with_no_browser() -> None:
    browser = FakeBrowser()

    run(
        LaunchOptions(port=free_port(), open_browser=False),
        serve=FakeServer(),
        open_browser=browser,
        wait=ready,
    )

    assert not browser.opened.wait(timeout=0.2)
    assert browser.urls == []


def test_reload_watches_only_the_code_folder() -> None:
    server = FakeServer()

    run(
        LaunchOptions(port=free_port(), open_browser=False, reload=True),
        serve=server,
        open_browser=FakeBrowser(),
    )

    ((_, kwargs),) = server.calls
    assert kwargs["reload"] is True
    assert kwargs["reload_dirs"] == [str(Path(launcher.__file__).resolve().parent)]
    assert kwargs["reload_dirs"][0].endswith("app")  # src/app, never the data folder


def test_a_busy_port_raises_and_nothing_is_started(busy_port: int) -> None:
    server = FakeServer()
    browser = FakeBrowser()

    with pytest.raises(LaunchError, match=str(busy_port)):
        run(LaunchOptions(port=busy_port), serve=server, open_browser=browser, wait=ready)

    assert server.calls == []
    assert not browser.opened.wait(timeout=0.2)


def test_main_returns_zero_when_the_server_runs_and_stops() -> None:
    server = FakeServer()

    code = main(["--no-browser", "--port", str(free_port())], serve=server)

    assert code == 0
    assert len(server.calls) == 1


def test_main_reports_a_busy_port_on_stderr_and_returns_one(
    busy_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    server = FakeServer()

    code = main(["--no-browser", "--port", str(busy_port)], serve=server)

    assert code == 1
    assert f"port {busy_port} is already in use" in capsys.readouterr().err
    assert server.calls == []


def test_python_dash_m_app_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "main", lambda: 7)

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("app", run_name="__main__")

    assert exit_info.value.code == 7


def test_an_invalid_configuration_stops_before_the_server_with_the_variable_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FACE_APP_THRESHOLD", "2")
    server = FakeServer()

    code = main(["--no-browser"], serve=server)

    assert code == 1
    assert capsys.readouterr().err.startswith("error: invalid configuration: FACE_APP_THRESHOLD")
    assert server.calls == []


def test_the_port_comes_from_the_configuration_and_the_option_wins_over_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured, chosen = free_port(), free_port()
    monkeypatch.setenv("FACE_APP_PORT", str(configured))
    server = FakeServer()

    main(["--no-browser"], serve=server)
    main(["--no-browser", "--port", str(chosen)], serve=server)

    assert [kwargs["port"] for _, kwargs in server.calls] == [configured, chosen]


def test_the_configured_log_level_reaches_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACE_APP_LOG_LEVEL", "WARNING")
    server = FakeServer()

    main(["--no-browser", "--port", str(free_port())], serve=server)

    ((_, kwargs),) = server.calls
    assert kwargs["log_level"] == "warning"


def test_a_missing_checkpoint_stops_before_the_server_with_how_to_train_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    missing = tmp_path / "E06" / "best.pt"
    monkeypatch.setenv("FACE_APP_CHECKPOINT_PATH", str(missing))
    server = FakeServer()

    code = main(["--no-browser", "--port", str(free_port())], serve=server)

    printed = capsys.readouterr().err
    assert code == 1
    assert printed.startswith(f"error: checkpoint not found at {missing}")
    assert "scripts/train.py" in printed
    assert "Traceback" not in printed
    assert server.calls == []


def test_a_model_that_fails_to_load_inside_the_server_is_a_message_too(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_server(*args: object, **kwargs: object) -> None:
        raise ModelLoadError("model.pt is not a valid FaceNet checkpoint (RuntimeError)")

    code = main(["--no-browser", "--port", str(free_port())], serve=failing_server)

    assert code == 1
    assert "is not a valid FaceNet checkpoint" in capsys.readouterr().err


def test_an_existing_checkpoint_and_the_pretrained_model_pass_the_check(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"weights")

    check_model_setup(Settings(checkpoint_path=checkpoint, device="cpu"))
    check_model_setup(Settings(checkpoint_path=None, device="cpu"))


@pytest.mark.skipif(torch.cuda.is_available(), reason="this computer has a GPU")
def test_asking_for_cuda_without_a_gpu_is_refused_before_the_server_starts() -> None:
    with pytest.raises(ModelLoadError, match="no GPU"):
        check_model_setup(Settings(checkpoint_path=None, device="cuda"))


def test_the_browser_waits_until_the_server_answers() -> None:
    answers = threading.Event()
    browser = FakeBrowser()

    def wait(port: int) -> bool:
        return answers.wait(timeout=5)

    run(LaunchOptions(port=free_port()), serve=FakeServer(), open_browser=browser, wait=wait)

    assert not browser.opened.wait(timeout=0.3)  # the model is still loading
    answers.set()
    assert browser.opened.wait(timeout=2)


def test_the_browser_stays_closed_when_the_server_never_answers() -> None:
    browser = FakeBrowser()

    run(
        LaunchOptions(port=free_port()),
        serve=FakeServer(),
        open_browser=browser,
        wait=lambda port: False,
    )

    assert not browser.opened.wait(timeout=0.3)
    assert browser.urls == []


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
