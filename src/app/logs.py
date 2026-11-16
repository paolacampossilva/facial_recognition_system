"""Logs of the application (spec 007): terminal only, and never a name, a photo or an embedding.

The services pass what they log as `extra` fields. The formatter writes only the fields of an
allow list, so a name passed by mistake is dropped instead of reaching the terminal.
"""

import logging

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
