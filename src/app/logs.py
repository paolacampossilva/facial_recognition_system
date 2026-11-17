"""Logs of the application (spec 007): terminal only, and never a name, a photo or an embedding.

The services pass what they log as `extra` fields. The formatter writes only the fields of an
allow list, so a name passed by mistake is dropped instead of reaching the terminal.
"""

import copy
import logging
import traceback
from types import TracebackType
from typing import Any

from uvicorn.config import LOGGING_CONFIG

APP_LOGGER: str = "app"
# The only extra fields that reach a log line (spec 007, R12). Adding one needs a review.
ALLOWED_FIELDS: tuple[str, ...] = (
    "person_id",
    "embedding_count",
    "deleted_embeddings",
    "status",
    "similarity",
)
LINE_FORMAT: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"
TIME_FORMAT: str = "%H:%M:%S"


class AllowListFormatter(logging.Formatter):
    """`time level module: message field=value`, with only the allowed extra fields."""

    def __init__(self) -> None:
        super().__init__(LINE_FORMAT, datefmt=TIME_FORMAT)

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802 (logging's name)
        # Before the traceback, which `format` appends after this.
        line = super().formatMessage(record)
        values = record.__dict__
        fields = [f"{name}={values[name]}" for name in ALLOWED_FIELDS if name in values]
        return " ".join([line, *fields])

    def formatException(  # noqa: N802 (logging's name)
        self,
        ei: tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None],
    ) -> str:
        """The traceback with the exception type but never its message.

        The message of an exception can hold what it was handling: pydantic's `ValidationError`
        carries the input value, which can be a name or the bytes of a photo (spec 007, R11).
        """
        exc_type, _, tb = ei
        frames = "".join(traceback.format_tb(tb))
        name = exc_type.__qualname__ if exc_type else "unknown error"
        return f"Traceback (most recent call last):\n{frames}{name}"


def configure_logging(level: str) -> None:
    """Sends the application's logs to stderr at `level`. Calling it again replaces the setup.

    Only the `app` loggers are configured: the server configures its own access log.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(AllowListFormatter())
    logger = logging.getLogger(APP_LOGGER)
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False


def uvicorn_log_config() -> dict[str, Any]:
    """The server's own logging setup, with this formatter on its default handler.

    A 500 is logged by the `uvicorn.error` logger, not by the `app` ones, so without this the
    traceback of an unexpected error would reach the terminal with the message of the exception.
    The access log keeps the server's format: it has only the method, the path and the status.
    """
    config = copy.deepcopy(LOGGING_CONFIG)
    config["formatters"]["default"] = {"()": "app.logs.AllowListFormatter"}
    return config
