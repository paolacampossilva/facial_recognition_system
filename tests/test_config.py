"""Tests of the configuration of the application (spec 007)."""

from pathlib import Path

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from app import config
from app.api.app import create_app
from app.config import Settings, load_settings, variable_name
from app.core.errors import ConfigError
from app.fakes import FakeServices

pytestmark = pytest.mark.unit

ENV_EXAMPLE: Path = Path(__file__).resolve().parent.parent / ".env.example"


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty .env that `load_settings` reads, in place of the repository's."""
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", path)
    return path


def test_without_env_the_settings_are_the_defaults_of_the_spec() -> None:
    settings = load_settings()

    assert settings.threshold == 0.766
    assert settings.database_path == Path("data/private/app.db")
    assert settings.checkpoint_path == Path("checkpoints/E06/best.pt")
    assert settings.device == "auto"
    assert settings.port == 8000
    assert settings.max_upload_mb == 10
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.log_level == "INFO"


def test_the_threshold_comes_from_the_environment_and_reaches_the_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACE_APP_THRESHOLD", "0.8")

    client = TestClient(create_app(FakeServices(), load_settings()))

    assert client.get("/api/config").json() == {"threshold": 0.8}


def test_the_env_file_is_read(env_file: Path) -> None:
    env_file.write_text(
        "FACE_APP_THRESHOLD=0.7\nFACE_APP_DATABASE_PATH=outro/app.db\nFACE_APP_DEVICE=cpu\n",
        encoding="utf-8",
    )

    settings = load_settings()

    assert (settings.threshold, settings.database_path, settings.device) == (
        0.7,
        Path("outro/app.db"),
        "cpu",
    )


def test_the_environment_wins_over_the_env_file(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file.write_text("FACE_APP_THRESHOLD=0.5\n", encoding="utf-8")
    monkeypatch.setenv("FACE_APP_THRESHOLD", "0.9")

    assert load_settings().threshold == 0.9


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("FACE_APP_THRESHOLD", "2"),
        ("FACE_APP_THRESHOLD", "0"),
        ("FACE_APP_THRESHOLD", "alto"),
        ("FACE_APP_PORT", "80"),
        ("FACE_APP_PORT", "70000"),
        ("FACE_APP_DEVICE", "tpu"),
        ("FACE_APP_MAX_UPLOAD_MB", "0"),
        ("FACE_APP_MAX_UPLOAD_MB", "51"),
        ("FACE_APP_LOG_LEVEL", "VERBOSE"),
    ],
)
def test_a_value_out_of_range_names_its_variable(
    variable: str, value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ConfigError, match=f"^invalid configuration: {variable}: "):
        load_settings()


def test_a_misspelled_variable_in_the_env_file_is_an_error(env_file: Path) -> None:
    env_file.write_text("FACE_APP_TRESHOLD=0.5\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="FACE_APP_TRESHOLD: unknown variable"):
        load_settings()


def test_an_empty_checkpoint_means_the_pretrained_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACE_APP_CHECKPOINT_PATH", "")

    assert load_settings().checkpoint_path is None


def test_the_log_level_is_read_in_any_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FACE_APP_LOG_LEVEL", "warning")

    assert load_settings().log_level == "WARNING"


def test_the_env_example_lists_exactly_the_settings_with_their_defaults() -> None:
    variables = dotenv_values(ENV_EXAMPLE)

    assert set(variables) == {variable_name(field) for field in Settings.model_fields}
    assert Settings(_env_file=ENV_EXAMPLE) == Settings()


def test_a_file_over_the_configured_limit_is_refused() -> None:
    settings = Settings(max_upload_mb=1)
    client = TestClient(create_app(FakeServices(), settings), raise_server_exceptions=False)

    big = b"x" * (settings.max_upload_bytes + 1)
    response = client.post("/api/recognitions", files={"photo": ("a.jpg", big, "image/jpeg")})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
