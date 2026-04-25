"""Query node API routes: /search, /health, /readiness, /admin/*."""

from __future__ import annotations

import time

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from embedrag.logging_setup import get_logger
from embedrag.models.api import (
    ChunkResult,
    DebugDenseHit,
    DebugFusedHit,
    DebugSearchRequest,
    DebugSearchResponse,
    DebugSparseHit,
    DebugTiming,
    HealthResponse,
    HotfixAddRequest,
    HotfixDeleteRequest,
    HotfixResponse,
    MultiSpaceSearchRequest,
    MultiSpaceSearchResponse,
    ReadinessResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    SearchRequest,
    SearchResponse,
    SyncStatusResponse,
    SyncTriggerRequest,
    TextSearchRequest,
)
from embedrag.query.index.hotfix import HotfixChunkData
from embedrag.query.retrieval.dense import DenseResult, DenseRetriever
from embedrag.query.retrieval.fusion import rrf_fuse
from embedrag.query.retrieval.hierarchy_expand import HierarchyExpander
from embedrag.query.retrieval.sparse import SparseResult, SparseRetriever
from embedrag.shared.metrics import (
    DENSE_LATENCY,
    HOTFIX_BUFFER_SIZE,
    SEARCH_COUNT,
    SEARCH_LATENCY,
    SPARSE_LATENCY,
)

logger = get_logger(__name__)
router = APIRouter()


def _get_state(request: Request):
    return request.app.state.query


@router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(status="ok", node_type="query")


@router.get("/api/spaces")
async def list_spaces(request: Request) -> dict:
    """Return available embedding spaces for the active generation."""
    state = _get_state(request)
    gen_mgr = state.gen_manager
    if not gen_mgr.is_loaded or not gen_mgr.active:
        return {"spaces": ["text"]}
    return {"spaces": gen_mgr.active.spaces}


@router.get("/readiness")
async def readiness(request: Request) -> ReadinessResponse:
    state = _get_state(request)
    gen_mgr = state.gen_manager
    if not gen_mgr.is_loaded:
        raise HTTPException(status_code=503, detail="Not ready")
    ctx = gen_mgr.active
    total_vectors = 0
    if ctx:
        for sm in ctx.shard_managers.values():
            total_vectors += sm.total_vectors
    return ReadinessResponse(
        ready=True,
        active_version=gen_mgr.active_version,
        vector_count=total_vectors,
        doc_count=ctx.manifest.db.doc_count if ctx else 0,
    )


async def _run_search_pipeline(
    state,
    query_vec: np.ndarray,
    query_text: str | None,
    top_k: int,
    filters: dict | None,
    expand_context: bool,
    context_depth: int,
    mode: str = "hybrid",
    space: str = "text",
) -> tuple[list[ChunkResult], str, float, float, float, float]:
    """Shared search pipeline used by both /search and /search/text.

    Returns (chunks, score_type, dense_ms, sparse_ms, fusion_ms, pipeline_ms).
    """
    config = state.config
    top_k = min(top_k, config.search.max_top_k)
    t_pipe = time.monotonic()

    async with state.gen_manager.acquire() as ctx:
        shard_mgr = ctx.get_shard_manager(space)
        hotfix = ctx.get_hotfix_buffer(space)

        dense_ms = 0.0
        sparse_ms = 0.0
        dense_results: list[DenseResult] = []
        sparse_results: list[SparseResult] = []

        if mode != "sparse":
            dense_retriever = DenseRetriever(shard_mgr)
            deleted = hotfix._deleted_ids if hotfix else set()
            dense_results, dense_ms = dense_retriever.search(query_vec, top_k * 2, deleted)

            if hotfix and hotfix.size > 0:
                hotfix_results = hotfix.search(query_vec, top_k)
                for cid, score in hotfix_results:
                    dense_results.append(DenseResult(chunk_id=cid, score=score))
                dense_results.sort(key=lambda r: r.score, reverse=True)

        if mode != "dense" and config.search.enable_sparse and query_text:
            sparse_retriever = SparseRetriever(ctx.db_pool)
            sparse_results, sparse_ms = sparse_retriever.search(query_text, top_k * 2, filters)

        # Fusion
        t_fusion = time.monotonic()
        if sparse_results and dense_results:
            fused = rrf_fuse(
                dense_results,
                sparse_results,
                top_k,
                dense_weight=config.search.dense_weight,
                sparse_weight=config.search.sparse_weight,
            )
            chunk_ids = [f.chunk_id for f in fused]
            score_map = {f.chunk_id: f.rrf_score for f in fused}
            score_type = "rrf"
        elif dense_results:
            chunk_ids = [r.chunk_id for r in dense_results[:top_k]]
            score_map = {r.chunk_id: r.score for r in dense_results[:top_k]}
            score_type = "cosine"
        elif sparse_results:
            chunk_ids = [r.chunk_id for r in sparse_results[:top_k]]
            score_map = {r.chunk_id: r.score for r in sparse_results[:top_k]}
            score_type = "bm25"
        else:
            chunk_ids = []
            score_map = {}
            score_type = "rrf"
        fusion_ms = (time.monotonic() - t_fusion) * 1000

        chunks: list[ChunkResult] = []
        for cid in chunk_ids:
            hf_chunk = hotfix.get_chunk(cid) if hotfix else None
            if hf_chunk:
                hf_chunk.score = score_map.get(cid, 0.0)
                chunks.append(hf_chunk)
            else:
                fetched = ctx.doc_store.get_chunks_by_ids([cid])
                for c in fetched:
                    c.score = score_map.get(c.chunk_id, 0.0)
                    chunks.append(c)

        if config.search.enable_hierarchy_expand and expand_context:
            expander = HierarchyExpander(ctx.doc_store)
            chunks = expander.expand(chunks, depth=context_depth)

    pipeline_ms = (time.monotonic() - t_pipe) * 1000
    return chunks, score_type, dense_ms, sparse_ms, fusion_ms, pipeline_ms


@router.post("/search")
async def search(request: Request, body: SearchRequest) -> SearchResponse:
    state = _get_state(request)
    t_total = time.monotonic()

    query_vec = np.array(body.query_embedding, dtype=np.float32)
    chunks, score_type, dense_ms, sparse_ms, fusion_ms, _ = await _run_search_pipeline(
        state,
        query_vec,
        body.query_text,
        body.top_k,
        body.filters,
        body.expand_context,
        body.context_depth,
        space=body.space,
    )

    total_ms = (time.monotonic() - t_total) * 1000
    SEARCH_LATENCY.observe(total_ms / 1000)
    DENSE_LATENCY.observe(dense_ms / 1000)
    SPARSE_LATENCY.observe(sparse_ms / 1000)
    SEARCH_COUNT.labels(status="success").inc()
    return SearchResponse(
        chunks=chunks,
        total=len(chunks),
        score_type=score_type,
        dense_time_ms=round(dense_ms, 2),
        sparse_time_ms=round(sparse_ms, 2),
        fusion_time_ms=round(fusion_ms, 2),
        total_time_ms=round(total_ms, 2),
    )


@router.post("/search/text")
async def search_text(request: Request, body: TextSearchRequest) -> SearchResponse:
    """Text-only search: embeds the query internally, then runs the full pipeline.

    Callers send plain text instead of pre-computed embeddings.
    Set mode to "hybrid" (default), "dense", or "sparse".
    """
    state = _get_state(request)
    t_total = time.monotonic()

    embed_ms = 0.0
    query_vec = np.zeros(0, dtype=np.float32)
    if body.mode != "sparse":
        t_emb = time.monotonic()
        embed_client = state.get_embedding_client(body.space)
        vectors = await embed_client.embed_texts([body.query_text])
        query_vec = vectors[0]
        embed_ms = (time.monotonic() - t_emb) * 1000

    chunks, score_type, dense_ms, sparse_ms, fusion_ms, _ = await _run_search_pipeline(
        state,
        query_vec,
        body.query_text,
        body.top_k,
        body.filters,
        body.expand_context,
        body.context_depth,
        mode=body.mode,
        space=body.space,
    )

    total_ms = (time.monotonic() - t_total) * 1000
    SEARCH_LATENCY.observe(total_ms / 1000)
    DENSE_LATENCY.observe(dense_ms / 1000)
    SPARSE_LATENCY.observe(sparse_ms / 1000)
    SEARCH_COUNT.labels(status="success").inc()
    return SearchResponse(
        chunks=chunks,
        total=len(chunks),
        score_type=score_type,
        embedding_time_ms=round(embed_ms, 2),
        dense_time_ms=round(dense_ms, 2),
        sparse_time_ms=round(sparse_ms, 2),
        fusion_time_ms=round(fusion_ms, 2),
        total_time_ms=round(total_ms, 2),
    )


@router.post("/search/multi")
async def search_multi(request: Request, body: MultiSpaceSearchRequest) -> MultiSpaceSearchResponse:
    """Search multiple embedding spaces and fuse results via late fusion (RRF)."""
    state = _get_state(request)
    config = state.config
    top_k = min(body.top_k, config.search.max_top_k)
    t_total = time.monotonic()

    all_dense_results: list[DenseResult] = []
    per_space_counts: dict[str, int] = {}

    async with state.gen_manager.acquire() as ctx:
        for sq in body.queries:
            shard_mgr = ctx.get_shard_manager(sq.space)
            hotfix = ctx.get_hotfix_buffer(sq.space)
            dense_retriever = DenseRetriever(shard_mgr)
            deleted = hotfix._deleted_ids if hotfix else set()
            query_vec = np.array(sq.query_embedding, dtype=np.float32)

            results, _ = dense_retriever.search(query_vec, top_k * 2, deleted)
            if hotfix and hotfix.size > 0:
                for cid, score in hotfix.search(query_vec, top_k):
                    results.append(DenseResult(chunk_id=cid, score=score))
                results.sort(key=lambda r: r.score, reverse=True)

            for r in results:
                r.score *= sq.weight
            all_dense_results.extend(results)
            per_space_counts[sq.space] = len(results)

        # Sparse retrieval on the text query from the first text-space query
        sparse_results: list[SparseResult] = []
        first_text_query = next((sq for sq in body.queries if sq.query_text and sq.space == "text"), None)
        if first_text_query and config.search.enable_sparse:
            sparse_retriever = SparseRetriever(ctx.db_pool)
            sparse_results, _ = sparse_retriever.search(first_text_query.query_text or "", top_k * 2, body.filters)

        if sparse_results and all_dense_results:
            fused = rrf_fuse(
                all_dense_results,
                sparse_results,
                top_k,
                dense_weight=config.search.dense_weight,
                sparse_weight=config.search.sparse_weight,
            )
            chunk_ids = [f.chunk_id for f in fused]
            score_map = {f.chunk_id: f.rrf_score for f in fused}
        elif all_dense_results:
            seen = {}
            for r in sorted(all_dense_results, key=lambda x: x.score, reverse=True):
                if r.chunk_id not in seen:
                    seen[r.chunk_id] = r.score
            chunk_ids = list(seen.keys())[:top_k]
            score_map = seen
        else:
            chunk_ids = [r.chunk_id for r in sparse_results[:top_k]]
            score_map = {r.chunk_id: r.score for r in sparse_results[:top_k]}

        chunks: list[ChunkResult] = []
        for cid in chunk_ids:
            fetched = ctx.doc_store.get_chunks_by_ids([cid])
            for c in fetched:
                c.score = score_map.get(c.chunk_id, 0.0)
                chunks.append(c)

        if config.search.enable_hierarchy_expand and body.expand_context:
            expander = HierarchyExpander(ctx.doc_store)
            chunks = expander.expand(chunks, depth=body.context_depth)

    total_ms = (time.monotonic() - t_total) * 1000
    return MultiSpaceSearchResponse(
        chunks=chunks,
        total=len(chunks),
        per_space=per_space_counts,
        total_time_ms=round(total_ms, 2),
    )


# ── Debug endpoint ──


@router.post("/api/debug/search")
async def debug_search(request: Request, body: DebugSearchRequest) -> DebugSearchResponse:
    """Full pipeline debug: returns every intermediate result and timing."""
    state = _get_state(request)
    config = state.config
    top_k = min(body.top_k, config.search.max_top_k)
    t_total = time.monotonic()

    embed_ms = 0.0
    query_vec = np.zeros(0, dtype=np.float32)
    if body.mode != "sparse":
        t_emb = time.monotonic()
        embed_client = state.get_embedding_client("text")
        vectors = await embed_client.embed_texts([body.query_text])
        query_vec = vectors[0]
        embed_ms = (time.monotonic() - t_emb) * 1000

    # Build FTS query for debug display
    fts_query_str = ""
    if body.mode != "dense" and config.search.enable_sparse and body.query_text:
        from embedrag.query.retrieval.sparse import SparseRetriever

        tmp_retriever = SparseRetriever.__new__(SparseRetriever)
        fts_segs, short_segs = tmp_retriever._split_segments(body.query_text)
        parts = []
        if fts_segs:
            parts.append(f"FTS: {tmp_retriever._segments_to_fts(fts_segs)}")
        if short_segs:
            parts.append(f"LIKE: {short_segs}")
        fts_query_str = " | ".join(parts)

    async with state.gen_manager.acquire() as ctx:
        dense_results: list[DenseResult] = []
        sparse_results: list[SparseResult] = []
        dense_ms = 0.0
        sparse_ms = 0.0
        space = body.space if hasattr(body, "space") else "text"

        if body.mode != "sparse":
            shard_mgr = ctx.get_shard_manager(space)
            hotfix = ctx.get_hotfix_buffer(space)
            dense_retriever = DenseRetriever(shard_mgr)
            deleted = hotfix._deleted_ids if hotfix else set()
            dense_results, dense_ms = dense_retriever.search(query_vec, top_k * 2, deleted)

            if hotfix and hotfix.size > 0:
                hotfix_hits = hotfix.search(query_vec, top_k)
                for cid, score in hotfix_hits:
                    dense_results.append(DenseResult(chunk_id=cid, score=score))
                dense_results.sort(key=lambda r: r.score, reverse=True)

        if body.mode != "dense" and config.search.enable_sparse and body.query_text:
            sparse_retriever = SparseRetriever(ctx.db_pool)
            sparse_results, sparse_ms = sparse_retriever.search(body.query_text, top_k * 2, body.filters)

        # Fusion
        t_fusion = time.monotonic()
        fused_list = []
        if sparse_results and dense_results:
            fused_list = rrf_fuse(
                dense_results,
                sparse_results,
                top_k,
                dense_weight=config.search.dense_weight,
                sparse_weight=config.search.sparse_weight,
            )
            chunk_ids = [f.chunk_id for f in fused_list]
            score_map = {f.chunk_id: f.rrf_score for f in fused_list}
        elif dense_results:
            chunk_ids = [r.chunk_id for r in dense_results[:top_k]]
            score_map = {r.chunk_id: r.score for r in dense_results[:top_k]}
        elif sparse_results:
            chunk_ids = [r.chunk_id for r in sparse_results[:top_k]]
            score_map = {r.chunk_id: r.score for r in sparse_results[:top_k]}
        else:
            chunk_ids = []
            score_map = {}
        fusion_ms = (time.monotonic() - t_fusion) * 1000

        # Fetch
        t_fetch = time.monotonic()
        hotfix = ctx.get_hotfix_buffer(space)
        chunks: list[ChunkResult] = []
        for cid in chunk_ids:
            hf_chunk = hotfix.get_chunk(cid) if hotfix else None
            if hf_chunk:
                hf_chunk.score = score_map.get(cid, 0.0)
                chunks.append(hf_chunk)
            else:
                fetched = ctx.doc_store.get_chunks_by_ids([cid])
                for c in fetched:
                    c.score = score_map.get(c.chunk_id, 0.0)
                    chunks.append(c)
        fetch_ms = (time.monotonic() - t_fetch) * 1000

        # Hierarchy expansion
        t_expand = time.monotonic()
        if config.search.enable_hierarchy_expand and body.expand_context:
            expander = HierarchyExpander(ctx.doc_store)
            chunks = expander.expand(chunks, depth=body.context_depth)
        expand_ms = (time.monotonic() - t_expand) * 1000

    total_ms = (time.monotonic() - t_total) * 1000

    # Build debug output
    dense_rank_map = {r.chunk_id: i for i, r in enumerate(dense_results)}
    sparse_rank_map = {r.chunk_id: i for i, r in enumerate(sparse_results)}

    debug_dense = [DebugDenseHit(chunk_id=r.chunk_id, score=round(r.score, 6)) for r in dense_results]
    debug_sparse = [DebugSparseHit(chunk_id=r.chunk_id, score=round(r.score, 6)) for r in sparse_results]

    debug_fused = []
    if fused_list:
        for f in fused_list:
            debug_fused.append(
                DebugFusedHit(
                    chunk_id=f.chunk_id,
                    rrf_score=round(f.rrf_score, 6),
                    dense_score=round(f.dense_score, 6),
                    sparse_score=round(f.sparse_score, 6),
                    dense_rank=dense_rank_map.get(f.chunk_id, -1),
                    sparse_rank=sparse_rank_map.get(f.chunk_id, -1),
                )
            )

    if fused_list:
        debug_score_type = "rrf"
    elif dense_results:
        debug_score_type = "cosine"
    elif sparse_results:
        debug_score_type = "bm25"
    else:
        debug_score_type = ""

    return DebugSearchResponse(
        query_text=body.query_text,
        mode=body.mode,
        fts_query=fts_query_str,
        embedding_time_ms=round(embed_ms, 2),
        score_type=debug_score_type,
        dense_results=debug_dense,
        sparse_results=debug_sparse,
        fused_results=debug_fused,
        final_chunks=chunks,
        timing=DebugTiming(
            embedding_ms=round(embed_ms, 2),
            dense_ms=round(dense_ms, 2),
            sparse_ms=round(sparse_ms, 2),
            fusion_ms=round(fusion_ms, 2),
            fetch_ms=round(fetch_ms, 2),
            expand_ms=round(expand_ms, 2),
            total_ms=round(total_ms, 2),
        ),
        config_snapshot={
            "enable_sparse": config.search.enable_sparse,
            "enable_hierarchy_expand": config.search.enable_hierarchy_expand,
            "max_top_k": config.search.max_top_k,
            "context_depth": config.search.context_depth,
            "dense_weight": config.search.dense_weight,
            "sparse_weight": config.search.sparse_weight,
            "embedding_api_format": config.embedding.api_format,
        },
    )


# ── Chunk context endpoint ──


@router.get("/api/chunks/{chunk_id}/neighbors")
async def chunk_neighbors(
    request: Request,
    chunk_id: str,
    before: int = 3,
    after: int = 3,
) -> dict:
    """Get neighboring chunks for context viewing.

    Returns chunks before and after the given chunk under the same parent,
    ordered by seq_in_parent.
    """
    state = _get_state(request)
    before = min(before, 10)
    after = min(after, 10)
    async with state.gen_manager.acquire() as ctx:
        result = ctx.doc_store.get_neighbors(chunk_id, before=before, after=after)
    return result


# ── Admin / Hotfix endpoints ──


@router.post("/admin/hotfix/add")
async def hotfix_add(request: Request, body: HotfixAddRequest) -> HotfixResponse:
    state = _get_state(request)
    space = body.space
    async with state.gen_manager.acquire() as ctx:
        vec = np.array(body.embedding, dtype=np.float32)
        data = HotfixChunkData(
            chunk_id=body.chunk_id,
            doc_id=body.doc_id,
            text=body.text,
            metadata=body.metadata,
        )
        hotfix = ctx.get_hotfix_buffer(space)
        hotfix.add(body.chunk_id, vec, data)
        HOTFIX_BUFFER_SIZE.set(hotfix.size)
        logger.warn("hotfix_add", chunk_id=body.chunk_id, buffer_size=hotfix.size)
        return HotfixResponse(
            operation="add",
            affected=1,
            buffer_size=hotfix.size,
        )


@router.post("/admin/hotfix/delete")
async def hotfix_delete(request: Request, body: HotfixDeleteRequest) -> HotfixResponse:
    state = _get_state(request)
    space = body.space
    async with state.gen_manager.acquire() as ctx:
        hotfix = ctx.get_hotfix_buffer(space)
        for cid in body.chunk_ids:
            hotfix.delete(cid)
        logger.warn(
            "hotfix_delete",
            chunk_ids=body.chunk_ids,
            buffer_deleted=hotfix.deleted_count,
        )
        return HotfixResponse(
            operation="delete",
            affected=len(body.chunk_ids),
            buffer_size=hotfix.size,
        )


@router.post("/admin/sync")
async def trigger_sync(request: Request, body: SyncTriggerRequest | None = None) -> dict:
    """Manually trigger a snapshot sync.

    Supports three modes (checked in order):

    1. ``snapshot_dir`` -- load from a local directory (same as /admin/reload).
    2. ``source_url`` -- one-off sync from an HTTP URL.
    3. Neither -- use the configured background syncer.
    """
    state = _get_state(request)

    snapshot_dir = (body.snapshot_dir if body else "").strip()
    if snapshot_dir:
        from pathlib import Path

        from embedrag.models.manifest import Manifest
        from embedrag.query.lifecycle.startup import load_generation, quick_verify_snapshot

        manifest_path = Path(snapshot_dir) / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"No manifest.json in {snapshot_dir}")

        manifest = Manifest.load(manifest_path)
        current = state.gen_manager.active_version
        if manifest.snapshot_version == current:
            return {"status": "already_loaded", "source": snapshot_dir, "version": current}

        if not quick_verify_snapshot(snapshot_dir, manifest):
            raise HTTPException(status_code=500, detail="Snapshot integrity check failed")

        ctx = load_generation(
            snapshot_dir,
            manifest,
            nprobe=state.config.index.nprobe,
            use_mmap=state.config.index.mmap,
        )
        await state.gen_manager.swap(ctx)
        logger.info(
            "sync_local",
            old_version=current,
            new_version=manifest.snapshot_version,
            dir=snapshot_dir,
        )
        return {"status": "synced", "source": snapshot_dir, "version": manifest.snapshot_version}

    source_url = (body.source_url if body else "").strip()
    if source_url:
        from pathlib import Path

        from embedrag.shared.archive import is_archive_url

        if is_archive_url(source_url):
            from embedrag.models.manifest import Manifest
            from embedrag.query.lifecycle.startup import load_generation, quick_verify_snapshot
            from embedrag.shared.archive import download_and_extract_archive, verify_archive_snapshot

            staging = str(Path(state.config.node.data_dir) / "staging" / "archive_sync")
            snap_dir = download_and_extract_archive(
                source_url,
                staging,
                timeout=state.config.sync.download_timeout_seconds,
            )
            manifest = verify_archive_snapshot(snap_dir)
            current = state.gen_manager.active_version
            if manifest.snapshot_version == current:
                return {"status": "already_loaded", "source": source_url, "version": current}
            if not quick_verify_snapshot(snap_dir, manifest):
                raise HTTPException(status_code=500, detail="Archive snapshot integrity check failed")
            ctx = load_generation(
                snap_dir,
                manifest,
                nprobe=state.config.index.nprobe,
                use_mmap=state.config.index.mmap,
            )
            await state.gen_manager.swap(ctx)
            return {"status": "synced", "source": source_url, "version": manifest.snapshot_version}

        from embedrag.query.sync.downloader import SnapshotDownloader
        from embedrag.query.sync.syncer import SnapshotSyncer
        from embedrag.shared.http_snapshot_client import HttpSnapshotClient

        client = HttpSnapshotClient(source_url, timeout=state.config.sync.download_timeout_seconds)
        staging = str(Path(state.config.node.data_dir) / "staging")
        downloader = SnapshotDownloader(
            client,
            staging,
            concurrency=state.config.sync.download_concurrency,
            timeout=state.config.sync.download_timeout_seconds,
        )
        tmp_syncer = SnapshotSyncer(state, client, downloader)
        swapped = await tmp_syncer.sync_once()
        return {"status": "synced" if swapped else "up_to_date", "source": source_url}

    if state.syncer:
        swapped = await state.syncer.sync_once()
        return {"status": "synced" if swapped else "up_to_date"}
    return {"status": "no_syncer_configured"}


@router.get("/admin/sync/status")
async def sync_status(request: Request) -> SyncStatusResponse:
    """Return current sync status and configuration."""
    state = _get_state(request)
    config = state.config

    if not state.syncer:
        return SyncStatusResponse(
            enabled=config.sync.enabled,
            source=config.sync.source,
            current_version=state.gen_manager.active_version,
        )

    s = state.syncer.status
    return SyncStatusResponse(
        enabled=True,
        source=config.sync.source,
        cron=config.sync.cron,
        poll_interval_seconds=config.sync.poll_interval_seconds,
        last_check_at=s.last_check_at,
        last_sync_at=s.last_sync_at,
        last_result=s.last_result,
        last_version=s.last_version,
        next_check_at=s.next_check_at,
        consecutive_errors=s.consecutive_errors,
        current_version=state.gen_manager.active_version,
    )


@router.post("/admin/reload")
async def reload_snapshot(request: Request, body: dict | None = None) -> dict:
    """Reload a snapshot from the local active directory.

    If ``snapshot_dir`` is provided in the body, load from that path.
    Otherwise, scan the configured data_dir/active/ for the newest snapshot.
    Performs a zero-downtime hot-swap via GenerationManager.
    """
    state = _get_state(request)
    config = state.config

    from pathlib import Path

    from embedrag.models.manifest import Manifest
    from embedrag.query.lifecycle.startup import load_generation, quick_verify_snapshot

    snapshot_dir = (body or {}).get("snapshot_dir", "")

    if not snapshot_dir:
        active_dir = Path(config.node.data_dir) / "active"
        manifests = list(active_dir.glob("*/manifest.json"))
        if not manifests:
            raise HTTPException(status_code=404, detail="No snapshots found in active/")
        manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        snapshot_dir = str(manifests[0].parent)

    manifest_path = Path(snapshot_dir) / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"No manifest.json in {snapshot_dir}")

    manifest = Manifest.load(manifest_path)

    current_version = state.gen_manager.active_version
    if manifest.snapshot_version == current_version:
        return {"status": "already_loaded", "version": current_version}

    if not quick_verify_snapshot(snapshot_dir, manifest):
        raise HTTPException(status_code=500, detail="Snapshot integrity check failed")

    ctx = load_generation(
        snapshot_dir,
        manifest,
        nprobe=config.index.nprobe,
        use_mmap=config.index.mmap,
    )
    await state.gen_manager.swap(ctx)

    logger.info(
        "admin_reload",
        old_version=current_version,
        new_version=manifest.snapshot_version,
        snapshot_dir=snapshot_dir,
    )
    return {
        "status": "reloaded",
        "old_version": current_version,
        "new_version": manifest.snapshot_version,
    }


# ── Rerank proxy ──


@router.post("/api/rerank")
async def rerank_proxy(req: RerankRequest) -> RerankResponse:
    """Proxy rerank requests to an external OpenAI-compatible embeddings API.

    Computes cosine similarity between the query embedding and each text
    embedding, returning results sorted by descending score.  This avoids
    browser CORS issues when the reranker runs on a different origin.
    """
    import math

    import httpx

    url = req.url
    model = req.model
    if not url or not model:
        raise HTTPException(status_code=400, detail="Both 'url' and 'model' are required")
    if not req.texts:
        return RerankResponse(results=[], elapsed_ms=0)

    t0 = time.monotonic()

    all_inputs = [req.query] + req.texts
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json={"model": model, "input": all_inputs})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Reranker returned {resp.status_code}: {resp.text[:300]}")

    data = resp.json()["data"]
    data.sort(key=lambda x: x["index"])
    query_emb = data[0]["embedding"]

    def _cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na * nb > 0 else 0.0

    results = []
    for i, text_data in enumerate(data[1:]):
        score = _cosine(query_emb, text_data["embedding"])
        results.append(RerankResult(index=i, score=score))
    results.sort(key=lambda r: -r.score)

    elapsed = (time.monotonic() - t0) * 1000
    return RerankResponse(results=results, elapsed_ms=elapsed)
