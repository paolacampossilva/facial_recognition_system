"""Starts the application: `python -m app`.

The API and the web interface are one process (FastAPI serves the pages), so this only shortens
the command, opens the browser and explains what went wrong. The server always listens on the
loopback address: the application has no authentication and holds biometric data, so there is
deliberately no option to expose it on the network (spec 005, R14).
"""

import argparse
import socket
import sys
import threading
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

import uvicorn
from pydantic import BaseModel, Field, ValidationError

from app.core.errors import LaunchError

# TODO(#48): the port moves to the Settings class of the configuration module (spec 007).
HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 8000
MIN_PORT: int = 1024
MAX_PORT: int = 65535
APP_IMPORT_PATH: str = "app.main:app"
BROWSER_DELAY_SECONDS: float = 1.0
SOURCE_DIR: Path = Path(__file__).resolve().parent


class LaunchOptions(BaseModel):
    """What the person asked for on the command line."""

    port: int = Field(default=DEFAULT_PORT, ge=MIN_PORT, le=MAX_PORT)
    open_browser: bool = True
    reload: bool = False


def parse_options(argv: Sequence[str] | None = None) -> LaunchOptions:
    """Reads the command line. Exits with a usage message when an option is not valid."""
    parser = argparse.ArgumentParser(
        prog="python -m app", description="Starts the face recognition web application."
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="port (default: %(default)s)"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open the browser automatically"
    )
    parser.add_argument(
        "--reload", action="store_true", help="restart the server when the code changes"
    )
    arguments = parser.parse_args(argv)
    try:
        return LaunchOptions(
            port=arguments.port, open_browser=not arguments.no_browser, reload=arguments.reload
        )
    except ValidationError:
        parser.error(f"--port must be between {MIN_PORT} and {MAX_PORT}")


def port_is_free(port: int) -> bool:
    """Whether nothing is listening on this port of the loopback address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((HOST, port))
        except OSError:
            return False
    return True


def run(
    options: LaunchOptions,
    serve: Callable[..., None] = uvicorn.run,
    open_browser: Callable[[str], object] = webbrowser.open,
    browser_delay: float = BROWSER_DELAY_SECONDS,
) -> None:
    """Starts the server and, after a short delay, opens the browser on it.

    Args:
        options: What the person asked for.
        serve: Starts the server and blocks until it stops (uvicorn's `run`).
        open_browser: Opens a URL in the browser.
        browser_delay: Seconds to wait before opening the browser, so the server is up.

    Raises:
        LaunchError: The port is already in use. Nothing is started.
    """
    if not port_is_free(options.port):
        raise LaunchError(
            f"port {options.port} is already in use; close the other process or use --port"
        )
    if options.open_browser:
        timer = threading.Timer(browser_delay, open_browser, [f"http://{HOST}:{options.port}"])
        timer.daemon = True
        timer.start()
    # Watch only the code: the database in data/private changes on every enrollment.
    reload_dirs = [str(SOURCE_DIR)] if options.reload else None
    serve(
        APP_IMPORT_PATH,
        host=HOST,
        port=options.port,
        reload=options.reload,
        reload_dirs=reload_dirs,
    )


def main(
    argv: Sequence[str] | None = None,
    serve: Callable[..., None] = uvicorn.run,
    open_browser: Callable[[str], object] = webbrowser.open,
) -> int:
    """Edge of the command line: runs the application and turns a launch failure into a message.

    Returns:
        The exit code: 0 when the server ran and stopped, 1 when it could not start.
    """
    options = parse_options(argv)
    try:
        run(options, serve=serve, open_browser=open_browser)
    except LaunchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
