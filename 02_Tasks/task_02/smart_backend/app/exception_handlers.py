# app/exception_handlers.py
# Fix from task_01: field variable was used before assignment in validation handler.

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

STATUS_TO_CODE = {
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    429: "rate_limit_exceeded",
    502: "upstream_error",
}


async def http_exception_handler(request: Request, exc: HTTPException):
    error_code = STATUS_TO_CODE.get(exc.status_code, "error")

    logger.warning(
        "HTTP error | status=%d | code=%s | path=%s | detail=%s",
        exc.status_code, error_code, request.url.path, exc.detail,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_code, "detail": exc.detail},
        headers=exc.headers or {},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()

    # Build a clean list of {field, detail} dicts from Pydantic's error list
    formatted = [
        {
            "field":  str(e.get("loc", ["unknown"])[-1]),
            "detail": e.get("msg", "Invalid input").replace("Value error, ", ""),
        }
        for e in errors
    ]

    logger.warning(
        "Validation error | path=%s | errors=%s",
        request.url.path,
        formatted,
    )

    return JSONResponse(
        status_code=422,
        content={
            "error":   "validation_error",
            "details": formatted,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)

    logger.error(
        "Unhandled exception | path=%s | error_type=%s | detail=%s",
        request.url.path, type(exc).__name__, str(exc),
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "An unexpected error occurred"},
    )
