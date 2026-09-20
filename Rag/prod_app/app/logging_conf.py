"""Structured, request-scoped logging.

One logger per request carrying request_id so every line for a request
can be joined together when debugging.
"""

import logging
import sys
import uuid

from app.config import settings

_request_id = uuid.uuid4().hex[:12]  # per-process default; overridden per request


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        ))
        logger.addHandler(handler)
        logger.setLevel(settings.log_level.upper())
        logger.propagate = False
    return logger


def log_request(logger: logging.Logger, request_id: str, event: str, **fields) -> None:
    base = {"request_id": request_id, "event": event}
    base.update(fields)
    logger.info(base)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]