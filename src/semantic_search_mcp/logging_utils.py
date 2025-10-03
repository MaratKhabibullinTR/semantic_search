import logging
import os
import time
from logging.config import dictConfig
from contextlib import contextmanager

DEFAULT_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
JSON = os.getenv("RAG_LOG_JSON", "0") in {"1", "true", "True"}
LOG_FILE = os.getenv("RAG_LOG_FILE")  # optional path


def setup_logging(
    level: str | None = None, json: bool | None = None, file: str | None = None
):
    level = (level or DEFAULT_LEVEL).upper()
    json = JSON if json is None else json
    file = file or LOG_FILE

    fmt = {
        "()": "logging.Formatter",
        "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }

    json_fmt = {
        "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
        "fmt": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "json_ensure_ascii": False,
    }

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json" if json else "plain",
            "level": level,
        }
    }

    if file:
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": file,
            "maxBytes": 10_000_000,
            "backupCount": 3,
            "formatter": "json" if json else "plain",
            "level": level,
        }

    dictConfig(
        {
            "version": 1,
            "formatters": {"plain": fmt, "json": json_fmt},
            "handlers": handlers,
            "root": {"level": level, "handlers": list(handlers.keys())},
            "disable_existing_loggers": False,
        }
    )


@contextmanager
def time_block(logger: logging.Logger, msg: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0

    logger.info("%s | duration=%.3fs", msg, dt)
