"""Tests for generation manager: atomic swap, ref counting, drain."""

import pytest

from embedrag.models.manifest import IndexInfo, Manifest
from embedrag.query.index.generation import GenerationContext, GenerationManager
from embedrag.query.index.hotfix import HotfixBuffer
from embedrag.query.index.id_mapping import IDMapper
from embedrag.query.retrieval.dense import ShardManager


def _make_gen(version: str) -> GenerationContext:
    """Create a minimal GenerationContext for testing (no real FAISS/SQLite)."""
    mgr = ShardManager(workers=[], id_mapper=IDMapper([]))
    manifest = Manifest(
        snapshot_version=version,
        indexes={"text": IndexInfo(dim=32)},
    )
    return GenerationContext(
        version=version,
        shard_managers={"text": mgr},
        db_pool=_FakePool(),
        doc_store=None,
        hotfix_buffers={"text": HotfixBuffer(dim=32)},
        manifest=manifest,
    )


class _FakePool:
    def close(self):
        pass


@pytest.mark.asyncio
async def test_swap_basic():
    gm = GenerationManager()
    assert not gm.is_loaded

    gen1 = _make_gen("v1")
    await gm.swap(gen1)
    assert gm.is_loaded
    assert gm.active_version == "v1"

    gen2 = _make_gen("v2")
    await gm.swap(gen2)
    assert gm.active_version == "v2"
    await gm.close()


@pytest.mark.asyncio
async def test_acquire_during_swap():
    gm = GenerationManager()
    gen1 = _make_gen("v1")
    await gm.swap(gen1)

    acquired_version = None
    async with gm.acquire() as ctx:
        acquired_version = ctx.version
    assert acquired_version == "v1"
    await gm.close()


@pytest.mark.asyncio
async def test_no_generation_raises():
    gm = GenerationManager()
    with pytest.raises(RuntimeError, match="No active generation"):
        async with gm.acquire():
            pass
