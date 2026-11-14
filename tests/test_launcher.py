"""Tests of `python -m app`: options, port check and startup, with no server and no browser."""

import runpy
import socket
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from app import launcher
from app.core.errors import LaunchError
from app.launcher import LaunchOptions, main, parse_options, port_is_free, run

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


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


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
    assert args == ("app.main:app",)
    assert kwargs == {"host": "127.0.0.1", "port": port, "reload": False, "reload_dirs": None}


def test_the_browser_opens_on_the_right_url() -> None:
    browser = FakeBrowser()
    port = free_port()

    run(LaunchOptions(port=port), serve=FakeServer(), open_browser=browser, browser_delay=0)

    assert browser.opened.wait(timeout=2)
    assert browser.urls == [f"http://127.0.0.1:{port}"]


def test_the_browser_stays_closed_with_no_browser() -> None:
    browser = FakeBrowser()

    run(
        LaunchOptions(port=free_port(), open_browser=False),
        serve=FakeServer(),
        open_browser=browser,
        browser_delay=0,
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
        run(LaunchOptions(port=busy_port), serve=server, open_browser=browser, browser_delay=0)

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
