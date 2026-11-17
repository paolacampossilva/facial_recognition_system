"""Keeps other web pages out of an API that has no login and holds biometric data.

The server listens only on the loopback address, but the browser of the person who uses the
application also opens other sites, and those can send requests to `127.0.0.1` too: a form on
another site can post a photo, and a DNS-rebinding page can read the list. Two checks close that
(spec 005, R14): the `Host` must be the loopback address, and a request that changes something
must come from the application's own page.
"""

import json

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from app.api.errors import ErrorResponse

# The names the application is reached by. The port can be anything, `--port` chooses it.
LOCAL_HOSTS: list[str] = ["127.0.0.1", "localhost"]
SAFE_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS"})
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
                await self._refuse(send)
                return
        await self._app(scope, receive, send)

    @staticmethod
    async def _refuse(send: Send) -> None:
        body = json.dumps(FORBIDDEN_ORIGIN.model_dump(exclude_none=True)).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
