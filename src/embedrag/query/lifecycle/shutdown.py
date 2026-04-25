"""Graceful shutdown: drain in-flight queries and release resources."""

from __future__ import annotations

import asyncio

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


async def graceful_shutdown(state) -> None:
    """Graceful shutdown sequence for the query node."""
    drain_seconds = state.config.server.shutdown_drain_seconds
    logger.info("shutdown_start", drain_seconds=drain_seconds)

    # Wait for in-flight queries to drain
    try:
        await asyncio.wait_for(
            _wait_for_drain(state),
            timeout=drain_seconds,
        )
    except TimeoutError:
        logger.warn("shutdown_drain_timeout", seconds=drain_seconds)

    await state.gen_manager.close()
    logger.info("shutdown_complete")


async def _wait_for_drain(state) -> None:
    gen_mgr = state.gen_manager
    while gen_mgr._ref_count > 0:
        await asyncio.sleep(0.05)
