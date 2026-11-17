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
import time
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import uvicorn
from pydantic import BaseModel, Field, ValidationError

from app.config import DEFAULT_PORT, MAX_PORT, MIN_PORT, Settings, load_settings
from app.core.errors import AppError, ConfigError, LaunchError, ModelLoadError
from app.embedder import TRAIN_HINT, pick_device
from app.logs import uvicorn_log_config

HOST: str = "127.0.0.1"
APP_IMPORT_PATH: str = "app.main:create"
# Loading the model can take a while, so the browser waits for the server, up to this long.
READY_TIMEOUT_SECONDS: float = 120.0
READY_POLL_SECONDS: float = 0.25
SOURCE_DIR: Path = Path(__file__).resolve().parent


class LaunchOptions(BaseModel):
    """What the person asked for on the command line, over the configuration (spec 007)."""

    port: int = Field(default=DEFAULT_PORT, ge=MIN_PORT, le=MAX_PORT)
    open_browser: bool = True
    reload: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


def parse_options(
    argv: Sequence[str] | None = None, settings: Settings | None = None
) -> LaunchOptions:
    """Reads the command line. Exits with a usage message when an option is not valid.

    Args:
        argv: The arguments, without the program name. None reads `sys.argv`.
        settings: The configuration, which gives the default port; `--port` wins over it.
    """
    settings = settings or Settings()
    parser = argparse.ArgumentParser(
        prog="python -m app", description="Starts the face recognition web application."
    )
    parser.add_argument(
        "--port", type=int, default=settings.port, help="port (default: %(default)s)"
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
            port=arguments.port,
            open_browser=not arguments.no_browser,
            reload=arguments.reload,
            log_level=settings.log_level,
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


def check_model_setup(settings: Settings) -> None:
    """Catches the usual first-run mistakes before the server starts and loads the model.

    The server calls the application factory itself and shows any error it raises as a traceback,
    so the two mistakes a person is likely to make are checked here, where they become a message.

    Raises:
        ModelLoadError: The device is not available, or the configured checkpoint does not exist.
    """
    pick_device(settings.device)
    checkpoint = settings.checkpoint_path
    if checkpoint is not None and not checkpoint.is_file():
        raise ModelLoadError(
            f"checkpoint not found at {checkpoint}; {TRAIN_HINT}, "
            f"or leave FACE_APP_CHECKPOINT_PATH empty to use the pretrained FaceNet"
        )


# The probe talks to this very computer, so it must never go through a proxy from the environment.
LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_for_server(port: int, timeout: float = READY_TIMEOUT_SECONDS) -> bool:
    """Waits until the server answers its health route.

    Returns:
        True once it answered, False if it did not within `timeout` seconds.
    """
    url = f"http://{HOST}:{port}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with LOCAL_OPENER.open(url, timeout=1) as response:
                if response.status == 200:
                    return True
        except OSError:  # nothing is listening yet, or it is still starting
            pass
        time.sleep(READY_POLL_SECONDS)
    return False


def open_when_ready(
    port: int, open_browser: Callable[[str], object], wait: Callable[[int], bool]
) -> None:
    """Opens the browser on the application once the server answers, and never before."""
    url = f"http://{HOST}:{port}"
    if wait(port):
        open_browser(url)
    else:
        print(f"warning: the server did not answer in time, open {url} yourself", file=sys.stderr)


def run(
    options: LaunchOptions,
    serve: Callable[..., None] = uvicorn.run,
    open_browser: Callable[[str], object] = webbrowser.open,
    wait: Callable[[int], bool] = wait_for_server,
) -> None:
    """Starts the server and opens the browser on it as soon as it answers.

    Args:
        options: What the person asked for.
        serve: Starts the server and blocks until it stops (uvicorn's `run`).
        open_browser: Opens a URL in the browser.
        wait: Waits until the server on a port answers; False means it never did.

    Raises:
        LaunchError: The port is already in use. Nothing is started.
    """
    if not port_is_free(options.port):
        raise LaunchError(
            f"port {options.port} is already in use; close the other process or use --port"
        )
    if options.open_browser:
        opener = threading.Thread(
            target=open_when_ready, args=(options.port, open_browser, wait), daemon=True
        )
        opener.start()
    # Watch only the code: the database in data/private changes on every enrollment.
    reload_dirs = [str(SOURCE_DIR)] if options.reload else None
    serve(
        APP_IMPORT_PATH,
        factory=True,
        host=HOST,
        port=options.port,
        reload=options.reload,
        reload_dirs=reload_dirs,
        log_level=options.log_level.lower(),
        log_config=uvicorn_log_config(),
    )


def main(
    argv: Sequence[str] | None = None,
    serve: Callable[..., None] = uvicorn.run,
    open_browser: Callable[[str], object] = webbrowser.open,
) -> int:
    """Edge of the command line: runs the application and turns a launch failure into a message.

    The configuration is read before anything starts, so a wrong value stops here with the name of
    the variable instead of a traceback from inside the server (spec 007, R6).

    Returns:
        The exit code: 0 when the server ran and stopped, 1 when it could not start (a wrong
        setting, a missing model or a port in use).
    """
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    options = parse_options(argv, settings)
    try:
        check_model_setup(settings)
        run(options, serve=serve, open_browser=open_browser)
    except AppError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
