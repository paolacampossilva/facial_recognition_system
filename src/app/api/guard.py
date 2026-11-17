"""Keeps other web pages out of an API that has no login and holds biometric data.

The server listens only on the loopback address, but the browser of the person who uses the
application also opens other sites, and those can send requests to `127.0.0.1` too: a form on
another site can post a photo, and a DNS-rebinding page can read the list. Two checks close that
(spec 005, R14): the `Host` must be the loopback address, and a request that changes something
must come from the application's own page. The same layer keeps the API responses out of the
browser cache and refuses an oversized upload before it is parsed.
"""

import json
from http import HTTPStatus

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import ErrorResponse, to_error_response
from app.core.errors import UploadTooLargeError
from app.core.models import MAX_PHOTOS

# The names the application is reached by. The port can be anything, `--port` chooses it.
LOCAL_HOSTS: list[str] = ["127.0.0.1", "localhost"]
SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})
API_PREFIX: str = "/api/"
ENROLLMENT_PATH: str = "/api/persons"  # the only route that takes more than one photo
# What the multipart framing and the name add to the photos themselves.
BODY_MARGIN_BYTES: int = 64 * 1024
# Names and recognition results must not be cached or sniffed by the browser.
PRIVATE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
}
FORBIDDEN_ORIGIN: ErrorResponse = ErrorResponse(
    code="forbidden_origin", message="The request did not come from the application's own page"
)


class SameOriginOnly:
    """Refuses a request that changes something when its `Origin` is not the page it is sent to.

    Browsers always send `Origin` on a cross-site POST or DELETE, so a page of another site cannot
    leave it out. A request with no `Origin` (a script or a test) is not a browser acting for a
    page, so it passes; the `Host` check of `TrustedHostMiddleware` runs before this one.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in SAFE_METHODS:
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None and origin != f"http://{headers.get('host')}":
                await respond(send, HTTPStatus.FORBIDDEN, FORBIDDEN_ORIGIN)
                return
        await self._app(scope, receive, send)


class UploadSizeLimit:
    """Refuses a request whose declared body is bigger than the photos the route can take.

    An enrollment takes up to ten photos and a recognition takes one.

    Starlette parses the whole multipart body, and keeps every part, before a route reads the first
    one, so the per-photo limit alone only acts after everything was stored. A browser always sends
    `Content-Length` for a form, so this stops an oversized request before it is parsed.
    """

    def __init__(self, app: ASGIApp, max_photo_bytes: int) -> None:
        self._app = app
        self._max_photo_bytes = max_photo_bytes

    def _max_body_bytes(self, path: str) -> int:
        photos = MAX_PHOTOS if path == ENROLLMENT_PATH else 1
        return self._max_photo_bytes * photos + BODY_MARGIN_BYTES

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(API_PREFIX):
            declared = Headers(scope=scope).get("content-length", "0")
            if declared.isdigit() and int(declared) > self._max_body_bytes(scope["path"]):
                status, body = to_error_response(UploadTooLargeError(self._max_photo_bytes))
                await respond(send, status, body)
                return
        await self._app(scope, receive, send)


class ApiResponseHeaders:
    """Marks every API response as private: names and recognition results must not be cached."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(API_PREFIX):
            await self._app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.decode().lower() not in {header.lower() for header in PRIVATE_HEADERS}
                ]
                headers += [
                    (name.lower().encode(), value.encode())
                    for name, value in PRIVATE_HEADERS.items()
                ]
                message = {**message, "headers": headers}
            await send(message)

        await self._app(scope, receive, send_with_headers)


async def respond(send: Send, status: int, error: ErrorResponse) -> None:
    """Sends the error body of the API as a complete response."""
    body = json.dumps(error.model_dump(exclude_none=True)).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
