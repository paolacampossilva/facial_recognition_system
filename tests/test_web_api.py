"""API tests with the fake services: contract, error mapping and security headers (spec 005)."""

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx2 import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.app import create_app
from app.api.errors import to_error_response
from app.api.guard import BODY_MARGIN_BYTES, UploadSizeLimit
from app.config import Settings
from app.core.errors import (
    DuplicatePersonError,
    EmptyGalleryError,
    PersonNotFoundError,
    TooManyPhotosError,
    UploadTooLargeError,
)
from app.core.models import MAX_PHOTOS
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


def recognize(client: TestClient, content: bytes = EVEN_FILE):
    return client.post("/api/recognitions", files={"photo": ("a.jpg", content, "image/jpeg")})


EXPECTED_ERRORS: dict[str, tuple[Callable[[TestClient], Response], int, str]] = {
    "no-face": (lambda client: enroll(client, "sem-rosto"), 422, "no_face_detected"),
    "two-faces": (lambda client: enroll(client, "varios-rostos"), 422, "multiple_faces"),
    "not-an-image": (lambda client: enroll(client, "invalida"), 422, "invalid_image"),
    "duplicate": (lambda client: enroll(client, "clone"), 409, "duplicate_person"),
    "empty-name": (lambda client: enroll(client, "   "), 422, "invalid_request"),
    "too-many-photos": (
        lambda client: enroll(client, "Ana", photos=MAX_PHOTOS + 1),
        422,
        "invalid_request",
    ),
    "no-photo": (
        lambda client: client.post("/api/persons", data={"name": "Ana"}),
        422,
        "invalid_request",
    ),
    "file-too-big": (
        lambda client: recognize(client, b"x" * (Settings().max_upload_bytes + 1)),
        422,
        "invalid_request",
    ),
    "empty-gallery": (recognize, 409, "empty_gallery"),
    "unknown-id": (lambda client: client.delete("/api/persons/99"), 404, "person_not_found"),
    "non-numeric-id": (lambda client: client.delete("/api/persons/abc"), 422, "invalid_request"),
}


@pytest.mark.parametrize(
    ("call", "status", "code"), EXPECTED_ERRORS.values(), ids=EXPECTED_ERRORS.keys()
)
def test_an_expected_error_is_answered_without_reaching_the_server(
    call: Callable[[TestClient], Response], status: int, code: str
) -> None:
    """Only the unexpected 500 may reach the server, which logs it as an ERROR with a traceback.

    The default `raise_server_exceptions=True` raises whatever the handlers let escape.
    """
    client = TestClient(
        create_app(FakeServices(), Settings(threshold=THRESHOLD)), base_url=LOOPBACK
    )

    response = call(client)

    assert response.status_code == status
    assert response.json()["code"] == code


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
        (TooManyPhotosError(10), 422, "invalid_request"),
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


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [("GET", "/api/health", 200), ("GET", "/api/config", 200), ("DELETE", "/api/persons/9", 404)],
)
def test_every_api_response_is_marked_private(
    client: TestClient, method: str, path: str, status: int
) -> None:
    response = client.request(method, path)

    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_refusal_is_marked_private_too(client: TestClient) -> None:
    response = client.post(
        "/api/persons", data={"name": "Ana"}, files=[photo()], headers={"Origin": "http://evil"}
    )

    assert response.status_code == 403
    assert response.headers["cache-control"] == "no-store"


def test_the_pages_of_the_interface_are_not_marked_no_store(client: TestClient) -> None:
    assert client.get("/").headers["cache-control"] == "no-cache"


async def run_asgi(
    app: ASGIApp, headers: list[tuple[bytes, bytes]], path: str = "/api/persons"
) -> list[Message]:
    """Calls an ASGI app with a POST to `path` and returns what it sent back."""
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
    await app(scope, receive, send)
    return sent


def test_a_body_declared_bigger_than_ten_photos_is_refused_before_it_is_read() -> None:
    reached: list[bool] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        reached.append(True)

    guard = UploadSizeLimit(downstream, max_photo_bytes=1024)
    declared = str(1024 * MAX_PHOTOS + BODY_MARGIN_BYTES + 1).encode()

    sent = asyncio.run(run_asgi(guard, [(b"content-length", declared)]))

    assert reached == []
    assert sent[0]["status"] == 422
    assert json.loads(sent[1]["body"])["code"] == "invalid_request"


def test_a_recognition_takes_one_photo_so_ten_photos_of_body_are_refused() -> None:
    reached: list[bool] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        reached.append(True)

    one_mib = 1024 * 1024
    guard = UploadSizeLimit(downstream, max_photo_bytes=one_mib)
    ten_photos = str(one_mib * MAX_PHOTOS).encode()

    enrollment = asyncio.run(run_asgi(guard, [(b"content-length", ten_photos)]))
    recognition = asyncio.run(
        run_asgi(guard, [(b"content-length", ten_photos)], path="/api/recognitions")
    )

    assert reached == [True]  # only the enrollment got through
    assert recognition[0]["status"] == 422
    assert enrollment == []


def test_a_body_inside_the_limit_reaches_the_route() -> None:
    reached: list[bool] = []

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        reached.append(True)

    guard = UploadSizeLimit(downstream, max_photo_bytes=1024)

    asyncio.run(run_asgi(guard, [(b"content-length", b"2048")]))

    assert reached == [True]


def test_more_photos_than_the_limit_are_refused_before_any_is_read(client: TestClient) -> None:
    response = enroll(client, "Ana", photos=MAX_PHOTOS + 1)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert client.get("/api/persons").json() == []
