# app/logger.py
import os
import logging
import sys


def setup_logging() -> None:
    """
    Configure the root logger once at application startup.
    Call this from main.py before anything else runs.
    """
    log_format ="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%y-%m-%d %H:%M:%S"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging,log_level),
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
        force=True
    )

    # silence noisy thrid-party loggers we don't care about
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)