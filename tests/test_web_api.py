"""API tests with the fake services: contract, error mapping and security headers (spec 005)."""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.errors import to_error_response
from app.config import Settings
from app.core.errors import (
    DuplicatePersonError,
    EmptyGalleryError,
    PersonNotFoundError,
    UploadTooLargeError,
)
from app.fakes import FakeServices
from facial_recognition_system.preprocessing.errors import (
    InvalidImageError,
    MultipleFacesError,
    NoFaceDetectedError,
)

pytestmark = pytest.mark.unit

THRESHOLD: float = 0.766
EVEN_FILE: bytes = b"xx"
ODD_FILE: bytes = b"xxx"
LOOPBACK: str = "http://127.0.0.1"


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        create_app(FakeServices(), Settings(threshold=THRESHOLD)),
        base_url=LOOPBACK,
        raise_server_exceptions=False,
    )


def photo(content: bytes = EVEN_FILE, name: str = "a.jpg") -> tuple[str, tuple[str, bytes, str]]:
    return ("photos", (name, content, "image/jpeg"))


def enroll(client: TestClient, name: str, photos: int = 1):
    files = [photo() for _ in range(photos)]
    return client.post("/api/persons", data={"name": name}, files=files)


def test_health_is_ok(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_config_returns_the_threshold_the_app_was_built_with(client: TestClient) -> None:
    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.json() == {"threshold": THRESHOLD}


def test_the_routes_do_not_log_what_the_services_already_log(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        enroll(client, "Ana")
        client.post("/api/recognitions", files={"photo": ("a.jpg", EVEN_FILE, "image/jpeg")})
        client.delete("/api/persons/1")
    assert [record for record in caplog.records if record.name.startswith("app.api")] == []


def test_enrollment_returns_201_and_the_person_shows_in_the_list(client: TestClient) -> None:
    response = enroll(client, "Ana", photos=3)
    assert response.status_code == 201
    assert response.json() == {"person_id": 1, "name": "Ana", "embedding_count": 3}
    assert client.get("/api/persons").json() == [{"id": 1, "name": "Ana", "embedding_count": 3}]


@pytest.mark.parametrize(
    ("name", "status", "code"),
    [
        ("sem-rosto", 422, "no_face_detected"),
        ("varios-rostos", 422, "multiple_faces"),
        ("invalida", 422, "invalid_image"),
    ],
)
def test_image_errors_of_the_enrollment_have_a_stable_code(
    client: TestClient, name: str, status: int, code: str
) -> None:
    response = enroll(client, name)
    assert response.status_code == status
    assert response.json()["code"] == code
    assert client.get("/api/persons").json() == []


def test_a_repeated_name_ignoring_case_is_a_duplicate_by_name(client: TestClient) -> None:
    enroll(client, "Ana")
    response = enroll(client, "ana")
    assert response.status_code == 409
    assert response.json()["code"] == "duplicate_person"
    assert response.json()["reason"] == "name"


def test_a_similar_face_is_a_duplicate_by_similarity_with_the_existing_name(
    client: TestClient,
) -> None:
    enroll(client, "Ana")
    response = enroll(client, "clone")
    assert response.status_code == 409
    assert response.json()["reason"] == "similarity"
    assert response.json()["person_name"] == "Ana"


@pytest.mark.parametrize("name", ["", "   ", "a" * 101])
def test_an_invalid_name_is_a_validation_error_and_nothing_is_saved(
    client: TestClient, name: str
) -> None:
    response = enroll(client, name)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert client.get("/api/persons").json() == []


def test_more_than_ten_photos_is_a_validation_error(client: TestClient) -> None:
    response = enroll(client, "Ana", photos=11)
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_a_missing_photo_field_is_a_validation_error(client: TestClient) -> None:
    response = client.post("/api/persons", data={"name": "Ana"})
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_a_file_over_the_limit_is_a_validation_error(client: TestClient) -> None:
    big = b"x" * (Settings().max_upload_bytes + 1)
    response = client.post("/api/persons", data={"name": "Ana"}, files=[photo(big)])
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_recognition_with_an_empty_gallery_is_the_empty_gallery_error(client: TestClient) -> None:
    response = client.post("/api/recognitions", files={"photo": ("a.jpg", EVEN_FILE, "image/jpeg")})
    assert response.status_code == 409
    assert response.json()["code"] == "empty_gallery"


def test_recognition_returns_the_name_and_the_similarity(client: TestClient) -> None:
    enroll(client, "Ana")
    response = client.post("/api/recognitions", files={"photo": ("a.jpg", EVEN_FILE, "image/jpeg")})
    assert response.status_code == 200
    assert response.json() == {"status": "recognized", "person_name": "Ana", "similarity": 0.91}


def test_an_unknown_face_carries_no_name(client: TestClient) -> None:
    enroll(client, "Ana")
    response = client.post("/api/recognitions", files={"photo": ("a.jpg", ODD_FILE, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unknown"
    assert body["person_name"] is None
    assert "Ana" not in response.text


def test_deleting_a_person_removes_them_from_the_list(client: TestClient) -> None:
    enroll(client, "Ana", photos=2)
    response = client.delete("/api/persons/1")
    assert response.status_code == 200
    assert response.json() == {"person_id": 1, "deleted_embeddings": 2}
    assert client.get("/api/persons").json() == []


def test_deleting_an_unknown_id_is_person_not_found(client: TestClient) -> None:
    response = client.delete("/api/persons/99")
    assert response.status_code == 404
    assert response.json()["code"] == "person_not_found"


def test_a_non_numeric_id_is_an_invalid_request(client: TestClient) -> None:
    response = client.delete("/api/persons/abc")
    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


class ExplodingServices(FakeServices):
    """Fake whose listing fails with an error that holds a personal detail."""

    def list_persons(self):  # noqa: ANN201
        raise RuntimeError("database /private/data.db password=hunter2 Ana Souza")


def test_an_unexpected_error_is_internal_error_without_leaking_details() -> None:
    client = TestClient(
        create_app(ExplodingServices(), Settings(threshold=THRESHOLD)),
        base_url=LOOPBACK,
        raise_server_exceptions=False,
    )
    response = client.get("/api/persons")
    assert response.status_code == 500
    assert response.json() == {"code": "internal_error", "message": "Unexpected error"}
    assert "hunter2" not in response.text
    assert "Ana Souza" not in response.text


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (NoFaceDetectedError(0.9), 422, "no_face_detected"),
        (MultipleFacesError(2), 422, "multiple_faces"),
        (InvalidImageError(Path("x")), 422, "invalid_image"),
        (DuplicatePersonError("Ana", "name"), 409, "duplicate_person"),
        (EmptyGalleryError(), 409, "empty_gallery"),
        (PersonNotFoundError(1), 404, "person_not_found"),
        (ValueError("bad"), 422, "invalid_request"),
        (UploadTooLargeError(1), 422, "invalid_request"),
        (RuntimeError("boom"), 500, "internal_error"),
    ],
)
def test_each_error_maps_to_its_status_and_code(error: Exception, status: int, code: str) -> None:
    mapped_status, body = to_error_response(error)
    assert mapped_status == status
    assert body.code == code


STATIC_PATHS: list[str] = [
    "/",
    "/app.js",
    "/styles.css",
    "/tokens.css",
    "/assets/icons/sun.svg",
    "/assets/favicon.svg",
]


@pytest.mark.parametrize("path", STATIC_PATHS)
def test_static_files_are_served_with_the_security_headers(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    policy = response.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "script-src 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_the_page_is_portuguese_and_has_no_inline_script_or_style(client: TestClient) -> None:
    html = client.get("/").text
    assert '<html lang="pt-br">' in html
    assert "<style" not in html
    assert "style=" not in html
    assert "<script>" not in html


def test_a_request_for_another_host_name_is_refused(client: TestClient) -> None:
    """A DNS-rebinding page reaches the server under its own name, which is not loopback."""
    response = client.get("/api/persons", headers={"Host": "attacker.example"})

    assert response.status_code == 400


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.1:8123", "localhost:9000"])
def test_the_loopback_names_are_accepted_with_any_port(client: TestClient, host: str) -> None:
    assert client.get("/api/health", headers={"Host": host}).status_code == 200


def test_a_post_from_another_site_is_refused_and_enrolls_nobody(client: TestClient) -> None:
    response = client.post(
        "/api/persons",
        data={"name": "Ana"},
        files=[photo()],
        headers={"Origin": "http://attacker.example"},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "forbidden_origin"
    assert client.get("/api/persons").json() == []


def test_a_delete_from_another_site_is_refused(client: TestClient) -> None:
    enroll(client, "Ana")

    response = client.delete("/api/persons/1", headers={"Origin": "http://attacker.example"})

    assert response.status_code == 403
    assert [person["name"] for person in client.get("/api/persons").json()] == ["Ana"]


def test_an_opaque_origin_is_refused(client: TestClient) -> None:
    """A sandboxed frame sends `Origin: null`."""
    response = client.post(
        "/api/persons", data={"name": "Ana"}, files=[photo()], headers={"Origin": "null"}
    )

    assert response.status_code == 403


def test_the_applications_own_page_may_post_on_any_port() -> None:
    client = TestClient(create_app(FakeServices(), Settings(threshold=THRESHOLD)))

    response = client.post(
        "/api/persons",
        data={"name": "Ana"},
        files=[photo()],
        headers={"Host": "127.0.0.1:8123", "Origin": "http://127.0.0.1:8123"},
    )

    assert response.status_code == 201
