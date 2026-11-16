"""Configuration of the application (spec 007): one `Settings` read from the environment and `.env`.

Everything that changes from one computer to another (paths, device, port) or that the specs say
must change without touching the code (the threshold) lives here, and no other module reads the
environment. The loopback address, the web folder and the SQLite PRAGMAs are fixed on purpose.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigError

ENV_PREFIX: str = "FACE_APP_"
# Read by `load_settings`, relative to where the application starts (the repository root). The
# tests set it to None so the developer's own .env never changes what they check.
ENV_FILE: Path | None = Path(".env")
DEFAULT_PORT: int = 8000
MIN_PORT: int = 1024
MAX_PORT: int = 65535
BYTES_PER_MB: int = 1024 * 1024


class Settings(BaseSettings):
    """The application settings, each one read from a `FACE_APP_` variable (spec 007, R2)."""

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, env_file_encoding="utf-8")

    # The lowest cosine similarity accepted as the same person, chosen in E08 (decision D08).
    threshold: float = Field(default=0.766, gt=0, le=1)
    database_path: Path = Path("data/private/app.db")
    # The chosen model (decision D16); None is the pretrained FaceNet with no fine-tuning (E01).
    checkpoint_path: Path | None = Path("checkpoints/E06/best.pt")
    device: Literal["auto", "cpu", "cuda"] = "auto"
    port: int = Field(default=DEFAULT_PORT, ge=MIN_PORT, le=MAX_PORT)
    max_upload_mb: int = Field(default=10, ge=1, le=50)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def _empty_means_pretrained(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _level_in_capitals(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @property
    def max_upload_bytes(self) -> int:
        """The largest photo the API accepts, in bytes."""
        return self.max_upload_mb * BYTES_PER_MB


def variable_name(field: str) -> str:
    """The variable a setting is read from; an unknown `.env` line keeps its own name."""
    if field in Settings.model_fields:
        return f"{ENV_PREFIX}{field.upper()}"
    return field.upper()


def describe(error: ValidationError) -> str:
    """One line naming each wrong variable and what it accepts."""
    problems = []
    for detail in error.errors():
        name = variable_name(str(detail["loc"][0])) if detail["loc"] else "configuration"
        reason = "unknown variable" if detail["type"] == "extra_forbidden" else detail["msg"]
        problems.append(f"{name}: {reason}")
    return "invalid configuration: " + "; ".join(problems)


def load_settings() -> Settings:
    """Reads the settings: environment first, then `.env`, then the defaults (spec 007, R3).

    Raises:
        ConfigError: A value is out of range or of the wrong type, or `.env` has an unknown line.
    """
    try:
        return Settings(_env_file=ENV_FILE)
    except ValidationError as error:
        raise ConfigError(describe(error)) from error
