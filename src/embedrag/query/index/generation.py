"""Generation manager: active/standby snapshot generations with atomic swap."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import Manifest
from embedrag.query.index.hotfix import HotfixBuffer
from embedrag.query.retrieval.dense import ShardManager
from embedrag.query.storage import QueryDocStore, ReadOnlySQLitePool

logger = get_logger(__name__)


@dataclass
class GenerationContext:
    """All resources for one snapshot generation.

    Supports multiple embedding spaces: ``shard_managers`` and ``hotfix_buffers``
    are dicts keyed by space name.
    """

    version: str
    shard_managers: dict[str, ShardManager]
    db_pool: ReadOnlySQLitePool
    doc_store: QueryDocStore
    hotfix_buffers: dict[str, HotfixBuffer]
    manifest: Manifest

    @property
    def spaces(self) -> list[str]:
        return list(self.shard_managers.keys())

    def get_shard_manager(self, space: str = "text") -> ShardManager:
        if space not in self.shard_managers:
            raise KeyError(f"Unknown embedding space '{space}'. Available: {self.spaces}")
        return self.shard_managers[space]

    def get_hotfix_buffer(self, space: str = "text") -> HotfixBuffer:
        if space not in self.hotfix_buffers:
            raise KeyError(f"Unknown embedding space '{space}'. Available: {self.spaces}")
        return self.hotfix_buffers[space]

    async def close(self) -> None:
        for sm in self.shard_managers.values():
            sm.shutdown()
        self.db_pool.close()
        for hb in self.hotfix_buffers.values():
            hb.clear()
        logger.info("generation_closed", version=self.version)


class GenerationManager:
    """Manages active/standby generations with ref-counted atomic swap."""

    def __init__(self) -> None:
        self._active: GenerationContext | None = None
        self._ref_count: int = 0
        self._lock = asyncio.Lock()
        self._drain_event = asyncio.Event()
        self._drain_event.set()

    @property
    def active_version(self) -> str:
        return self._active.version if self._active else ""

    @property
    def is_loaded(self) -> bool:
        return self._active is not None

    @property
    def active(self) -> GenerationContext | None:
        return self._active

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[GenerationContext]:
        """Acquire a reference to the active generation for a query.

        This guarantees the generation won't be unloaded while the query runs.
        """
        ctx = self._active
        if ctx is None:
            raise RuntimeError("No active generation loaded")
        self._ref_count += 1
        self._drain_event.clear()
        try:
            yield ctx
        finally:
            self._ref_count -= 1
            if self._ref_count == 0:
                self._drain_event.set()

    async def swap(self, new_ctx: GenerationContext) -> None:
        """Atomically swap to a new generation.

        1. Set new generation as active (new queries see it immediately).
        2. Wait for all in-flight queries on old generation to complete.
        3. Close old generation.
        """
        from embedrag.shared.metrics import ACTIVE_GENERATION, INDEX_VECTOR_COUNT

        async with self._lock:
            old_ctx = self._active
            self._active = new_ctx
            logger.info(
                "generation_swap",
                old=old_ctx.version if old_ctx else "none",
                new=new_ctx.version,
            )

            version_digits = "".join(c for c in new_ctx.version if c.isdigit())
            ACTIVE_GENERATION.set(int(version_digits) if version_digits else 0)
            total_vectors = sum(sm.total_vectors for sm in new_ctx.shard_managers.values())
            INDEX_VECTOR_COUNT.set(total_vectors)

            if old_ctx:
                await self._drain_event.wait()
                await old_ctx.close()

    async def close(self) -> None:
        if self._active:
            await self._drain_event.wait()
            await self._active.close()
            self._active = None
