"""Request/Response Pydantic models for both writer and query APIs."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Writer API models ──


class DocumentInput(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    source: str = ""
    doc_type: str = ""
    chunking: str = "auto"
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    metadata: dict = Field(default_factory=dict)
    modality: str = "text"
    content_ref: str = ""


class IngestRequest(BaseModel):
    documents: list[DocumentInput]


class IngestResponse(BaseModel):
    ingested: int
    chunk_count: int
    doc_ids: list[str]


class BuildRequest(BaseModel):
    force_full_rebuild: bool = False


class BuildResponse(BaseModel):
    version: str
    doc_count: int
    chunk_count: int
    vector_count: int
    num_shards: int
    build_time_seconds: float


class PublishResponse(BaseModel):
    version: str
    upload_time_seconds: float
    snapshot_size_bytes: int


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    chunks_deleted: int


# ── Rerank proxy model ──


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    url: str = ""
    model: str = ""


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]
    elapsed_ms: float


# ── Query API models ──


class SearchRequest(BaseModel):
    query_embedding: list[float]
    query_text: str | None = None
    top_k: int = 10
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1
    space: str = "text"


class TextSearchRequest(BaseModel):
    query_text: str
    top_k: int = 10
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1
    mode: str = "hybrid"
    space: str = "text"


class ChunkResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    level: int = 0
    level_type: str = "chunk"
    metadata: dict = Field(default_factory=dict)
    parent_text: str | None = None


class SearchResponse(BaseModel):
    chunks: list[ChunkResult]
    total: int
    score_type: str = "rrf"
    embedding_time_ms: float = 0
    dense_time_ms: float = 0
    sparse_time_ms: float = 0
    fusion_time_ms: float = 0
    total_time_ms: float = 0


class HotfixAddRequest(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict = Field(default_factory=dict)


class HotfixDeleteRequest(BaseModel):
    chunk_ids: list[str]


class HotfixResponse(BaseModel):
    operation: str
    affected: int
    buffer_size: int


class HealthResponse(BaseModel):
    status: str = "ok"
    node_type: str = ""
    version: str = ""


class ReadinessResponse(BaseModel):
    ready: bool
    active_version: str = ""
    vector_count: int = 0
    doc_count: int = 0


# ── Multi-space search models ──


class SpaceQuery(BaseModel):
    """Query for a single embedding space."""

    space: str
    query_embedding: list[float]
    query_text: str | None = None
    weight: float = 1.0


class MultiSpaceSearchRequest(BaseModel):
    """Search multiple spaces and fuse results via late fusion."""

    queries: list[SpaceQuery]
    top_k: int = 10
    fusion: str = "rrf"
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1


class MultiSpaceSearchResponse(BaseModel):
    chunks: list[ChunkResult]
    total: int
    per_space: dict[str, int] = Field(default_factory=dict)
    total_time_ms: float = 0


# ── Debug API models ──


# ── Sync API models ──


class SyncStatusResponse(BaseModel):
    enabled: bool = False
    source: str = ""
    cron: str = ""
    poll_interval_seconds: int = 0
    last_check_at: float = 0
    last_sync_at: float = 0
    last_result: str = "none"
    last_version: str = ""
    next_check_at: float = 0
    consecutive_errors: int = 0
    current_version: str = ""


class SyncTriggerRequest(BaseModel):
    source_url: str = ""
    snapshot_dir: str = ""


# ── Debug API models ──


class DebugSearchRequest(BaseModel):
    query_text: str
    top_k: int = 10
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1
    mode: str = "hybrid"


class DebugDenseHit(BaseModel):
    chunk_id: str
    score: float


class DebugSparseHit(BaseModel):
    chunk_id: str
    score: float


class DebugFusedHit(BaseModel):
    chunk_id: str
    rrf_score: float
    dense_score: float
    sparse_score: float
    dense_rank: int = -1
    sparse_rank: int = -1


class DebugTiming(BaseModel):
    embedding_ms: float = 0
    dense_ms: float = 0
    sparse_ms: float = 0
    fusion_ms: float = 0
    fetch_ms: float = 0
    expand_ms: float = 0
    total_ms: float = 0


class DebugSearchResponse(BaseModel):
    query_text: str
    mode: str
    fts_query: str = ""
    embedding_time_ms: float = 0
    score_type: str = ""
    dense_results: list[DebugDenseHit] = Field(default_factory=list)
    sparse_results: list[DebugSparseHit] = Field(default_factory=list)
    fused_results: list[DebugFusedHit] = Field(default_factory=list)
    final_chunks: list[ChunkResult] = Field(default_factory=list)
    timing: DebugTiming = Field(default_factory=DebugTiming)
    config_snapshot: dict = Field(default_factory=dict)
