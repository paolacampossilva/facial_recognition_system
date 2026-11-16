"""FastAPI application: the JSON routes plus the static web interface, with security headers."""

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope

from app.api.errors import to_error_response
from app.api.ports import Services
from app.api.routes import build_router
from app.config import Settings

logger = logging.getLogger(__name__)

# Part of the code, not of the configuration (spec 007, R8).
WEB_DIR: Path = Path(__file__).resolve().parent.parent / "web"

WEB_HEADERS: dict[str, str] = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}


class WebFiles(StaticFiles):
    """The static screens, always served with the security headers."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        response.headers.update(WEB_HEADERS)
        return response


def handle_error(_: Request, error: Exception) -> JSONResponse:
    """Edge of the API: turns any error into the single error body."""
    status_code, body = to_error_response(error)
    if status_code >= 500:
        logger.error("unexpected error", exc_info=error)
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


def create_app(services: Services, settings: Settings, web_dir: Path = WEB_DIR) -> FastAPI:
    """Builds the application around the given services and serves the interface from `web_dir`.

    Args:
        services: The enrollment, recognition, deletion and person-list services.
        settings: The configuration (spec 007). The API uses the upload limit and hands the
            threshold to the interface, which shows it on the result (it does not change what the
            services decide).
        web_dir: Folder with the static interface.
    """
    application = FastAPI(title="Face recognition", docs_url=None, redoc_url=None)
    application.state.services = services
    application.state.settings = settings
    application.include_router(build_router())
    application.add_exception_handler(Exception, handle_error)
    application.add_exception_handler(RequestValidationError, handle_error)
    application.mount("/", WebFiles(directory=web_dir, html=True), name="web")
    return application
