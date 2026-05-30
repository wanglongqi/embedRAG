"""Logging configuration using structlog with JSON or console output.

Provides ``setup_logging()`` for initializing structured logging with
contextual bindings (node type, request ID) and ``get_logger()`` for
acquiring a configured logger instance throughout the application.
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(
    level: str = "INFO",
    fmt: str = "json",
    node_type: str = "query",
) -> None:
    """Configure structlog with JSON or console output.

    Applies common processors (timestamp, log level, stack info, exception
    formatting) and binds the node type to the log context for all subsequent
    log events.

    Args:
        level: Minimum log level (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        fmt: Output format — ``"json"`` for structured JSON lines (default),
            or ``"console"`` for human-readable colour output.
        node_type: A label (``"query"`` or ``"writer"``) bound to every
            log event for operational filtering.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    structlog.contextvars.bind_contextvars(node_type=node_type)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Acquire a configured structlog logger instance.

    Args:
        name: Optional logger name (typically ``__name__``). Falls back
            to the root logger if not provided.

    Returns:
        A structlog ``BoundLogger`` pre-configured by ``setup_logging()``.
    """
    return structlog.get_logger(name)
