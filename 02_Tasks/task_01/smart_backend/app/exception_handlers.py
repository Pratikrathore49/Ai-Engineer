# app/exception_handlers.py

import logging
from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Override FastAPI's default HTTPException response.
    Maps status codes to error code strings clients can check programmatically.
    """
    status_to_code = {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        429: "rate_limit_exceeded",
        502: "upstream_error",
    }

    error_code = status_to_code.get(exc.status_code, "error")

    logger.warning(
        "HTTP error | status=%d | code=%s | path=%s | detail=%s",
        exc.status_code,
        error_code,
        request.url.path,
        exc.detail
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_code,
            "detail": exc.detail
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Override FastAPI's default 422 response shape.
    FastAPI's default returns a deeply nested 'errors' list — we flatten it to one clean object.
    """
    # Pydantic gives us a list of errors — take the first one and format it cleanly
    errors = exc.errors()
    first_error = errors[0] if errors else {}

    # 'loc' is a tuple like ('body', 'message') or ('body', 'conversation_id')
    field = first_error.get("loc", ["unknown"])[-1]
    message = first_error.get("msg", "Invalid input").replace("Value error, ", "")

    logger.warning(
        "Validation error | path=%s | field=%s | detail=%s",
        request.url.path,
        field,
        message
    )

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": message,
            "field": str(field)
        }
    )

async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for anything that wasn't caught elsewhere.
    Logs the real error internally, returns a safe message to the client.
    Never expose internal error details externally.
    """
    logger.error(
        "Unhandled exception | path=%s | error_type=%s | detail=%s",
        request.url.path,
        type(exc).__name__,
        str(exc),
        exc_info=True   # includes full traceback in the log
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": "An unexpected error occurred"
        }
    )
