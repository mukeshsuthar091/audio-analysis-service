"""Application exceptions and safe error responses."""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.response import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """An expected error with a public-safe code and message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def request_id_for(request: Request) -> str:
    return getattr(request.state, "request_id", "req_unknown")


def error_json(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def install_exception_handlers(app: FastAPI) -> None:
    """Install centralized handlers without exposing internal exception details."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.info(
            "request_failed",
            extra={
                "event_fields": {
                    "request_id": request_id_for(request),
                    "error_code": exc.code,
                    "status_code": exc.status_code,
                }
            },
        )
        return error_json(exc.status_code, exc.code, exc.message, request_id_for(request))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del exc
        return error_json(
            422,
            "INVALID_REQUEST_FIELD",
            "One or more request fields are invalid.",
            request_id_for(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        safe_messages: dict[int, tuple[str, str]] = {
            404: ("NOT_FOUND", "The requested resource was not found."),
            405: ("METHOD_NOT_ALLOWED", "The HTTP method is not allowed."),
        }
        code, message = safe_messages.get(
            exc.status_code, ("HTTP_ERROR", "The request could not be completed.")
        )
        return error_json(exc.status_code, code, message, request_id_for(request))

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        fields: dict[str, Any] = {
            "request_id": request_id_for(request),
            "status_code": 500,
            "error_type": type(exc).__name__,
        }
        logger.exception("unexpected_request_failure", extra={"event_fields": fields})
        return error_json(
            500,
            "INTERNAL_ERROR",
            "An unexpected internal processing error occurred.",
            request_id_for(request),
        )

