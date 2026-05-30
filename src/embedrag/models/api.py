"""Request/Response Pydantic models for both writer and query APIs.

This module defines the data transfer objects (DTOs) that form the API contract
for EmbedRAG. These models ensure type safety, provide automatic validation,
and generate high-quality OpenAPI/Swagger documentation. They are used for
everything from document ingestion and index building to complex hybrid
searches and administrative synchronization tasks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Writer API models ──


class DocumentInput(BaseModel):
    """Input structure for a single document during the ingestion process.

    This model represents the raw data that will be chunked, embedded, and
    stored in the writer's database.

    Attributes:
        doc_id (str): A unique global identifier for the document.
        title (str, optional): The title of the document. Defaults to "".
        text (str): The raw text content of the document to be indexed.
        source (str, optional): An identifier for the document's origin
            (e.g., a URL or file path). Defaults to "".
        doc_type (str, optional): A category for the document (e.g., "wiki",
            "manual"). Useful for filtering. Defaults to "".
        chunking (str, optional): The chunking strategy to use (e.g., "auto",
            "character", "none"). Defaults to "auto".
        chunk_size (int, optional): Override for the default number of
            characters per chunk.
        chunk_overlap (int, optional): Override for the default overlap
            between adjacent chunks.
        metadata (dict, optional): Arbitrary key-value pairs to store with
            the document for filtering and display.
        modality (str, optional): The data modality (e.g., "text", "image").
            Defaults to "text".
        content_ref (str, optional): A reference to external raw content if
            the index only stores metadata/vectors. Defaults to "".
    """

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
    """Request payload for the bulk ingestion endpoint.

    Attributes:
        documents (list[DocumentInput]): A list of one or more documents to
            be processed and stored.
    """

    documents: list[DocumentInput]


class IngestResponse(BaseModel):
    """Success response from the ingestion endpoint.

    Attributes:
        ingested (int): The number of documents successfully processed.
        chunk_count (int): The total number of chunks generated and stored.
        doc_ids (list[str]): The list of document IDs that were processed.
    """

    ingested: int
    chunk_count: int
    doc_ids: list[str]


class BuildRequest(BaseModel):
    """Parameters for triggering a new FAISS index build.

    Attributes:
        force_full_rebuild (bool, optional): If True, ignores incremental
            state and rebuilds the entire index from all documents in the
            database. Defaults to False.
    """

    force_full_rebuild: bool = False


class BuildResponse(BaseModel):
    """Detailed statistics for a successfully completed index build.

    Attributes:
        version (str): The new unique version ID for the built generation.
        doc_count (int): The total number of documents included in the index.
        chunk_count (int): The total number of chunks across all documents.
        vector_count (int): The total number of vectors embedded in FAISS.
        num_shards (int): The number of physical index shards created.
        build_time_seconds (float): The wall-clock time taken for the build.
    """

    version: str
    doc_count: int
    chunk_count: int
    vector_count: int
    num_shards: int
    build_time_seconds: float


class PublishResponse(BaseModel):
    """Metadata for a newly published snapshot generation.

    Attributes:
        version (str): The generation version that was published.
        upload_time_seconds (float): Time taken to transfer files to storage.
        snapshot_size_bytes (int): Total size of the published files.
    """

    version: str
    upload_time_seconds: float
    snapshot_size_bytes: int


class ArchiveRequest(BaseModel):
    """Parameters for creating a portable snapshot archive.

    Attributes:
        format (str, optional): The archive format (e.g., "tar.zst").
        compression_level (int, optional): Zstd compression level (1-22).
            Defaults to 3.
    """

    format: str = "tar.zst"
    compression_level: int = 3


class ArchiveResponse(BaseModel):
    """Location and metadata for a created snapshot archive.

    Attributes:
        version (str): The snapshot version that was archived.
        format (str): The archive format used.
        path (str): The relative path to the archive file on the local disk.
        size_bytes (int): The size of the resulting archive file.
        build_time_seconds (float): Time taken to compress and archive.
    """

    version: str
    format: str
    path: str
    size_bytes: int
    build_time_seconds: float


class DeleteDocumentResponse(BaseModel):
    """Response confirming document deletion.

    Attributes:
        doc_id (str): The identifier of the deleted document.
        chunks_deleted (int): The number of related chunks removed from storage.
    """

    doc_id: str
    chunks_deleted: int


class StatsResponse(BaseModel):
    """Comprehensive health and size statistics for the writer node.

    Attributes:
        doc_count (int): Total documents in the SQLite database.
        chunk_count (int): Total chunks in the SQLite database.
        embedding_spaces (list[str]): Names of configured embedding spaces.
        vectors_per_space (dict[str, int]): Count of vectors stored for each space.
        current_version (str): The version ID of the last successful build.
        db_size_bytes (int): Size of the SQLite database file on disk.
    """

    doc_count: int
    chunk_count: int
    embedding_spaces: list[str]
    vectors_per_space: dict[str, int]
    current_version: str
    db_size_bytes: int


class DocumentSummary(BaseModel):
    """A minimal representation of a document for list views.

    Attributes:
        doc_id (str): Unique document identifier.
        title (str): Document title.
        source (str): Origin identifier.
        doc_type (str): Category/type.
        created_at (str): ISO timestamp of ingestion.
    """

    doc_id: str
    title: str
    source: str
    doc_type: str
    created_at: str


class DocumentListResponse(BaseModel):
    """Paginated list of documents stored on the writer node.

    Attributes:
        documents (list[DocumentSummary]): The list of document summaries.
        total (int): The total number of documents in the system.
        limit (int): The pagination limit used.
        offset (int): The pagination offset used.
    """

    documents: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DocumentDetailResponse(BaseModel):
    """Full detail view of a single document and its structure.

    Attributes:
        doc_id (str): Unique document identifier.
        title (str): Document title.
        source (str): Origin identifier.
        doc_type (str): Category/type.
        metadata (dict): Arbitrary key-value metadata.
        chunk_ids (list[str]): List of all chunk IDs belonging to this document.
    """

    doc_id: str
    title: str
    source: str
    doc_type: str
    metadata: dict
    chunk_ids: list[str]


class BulkDeleteRequest(BaseModel):
    """Parameters for deleting documents in bulk.

    Attributes:
        doc_ids (list[str], optional): A specific list of document IDs to delete.
        doc_type (str, optional): If provided, deletes all documents of this type.
    """

    doc_ids: list[str] = Field(default_factory=list)
    doc_type: str = ""


class BulkDeleteResponse(BaseModel):
    """Summary of a bulk deletion operation.

    Attributes:
        deleted_docs (int): The number of documents successfully removed.
        deleted_chunks (int): The total number of chunks removed.
    """

    deleted_docs: int
    deleted_chunks: int


# ── Rerank proxy model ──


class RerankRequest(BaseModel):
    """Input for an external reranking service.

    Attributes:
        query (str): The search query text.
        texts (list[str]): The candidate texts to be reranked.
        url (str, optional): Override for the reranker service URL.
        model (str, optional): The model name for reranking.
    """

    query: str
    texts: list[str]
    url: str = ""
    model: str = ""


class RerankResult(BaseModel):
    """A single item from a reranking operation.

    Attributes:
        index (int): The index of the text in the original input list.
        score (float): The new relevance score assigned by the reranker.
    """

    index: int
    score: float


class RerankResponse(BaseModel):
    """Result of a cross-encoder reranking operation.

    Attributes:
        results (list[RerankResult]): Results sorted by descending score.
        elapsed_ms (float): Time taken for the reranking operation.
    """

    results: list[RerankResult]
    elapsed_ms: float


# ── Query API models ──


class SearchRequest(BaseModel):
    """Standard vector search request parameters.

    Attributes:
        query_embedding (list[float]): The query vector, pre-embedded in
            the appropriate space.
        query_text (str, optional): The raw query text. Required for hybrid
            and sparse search modes.
        top_k (int, optional): The number of results to return. Defaults to 10.
        filters (dict, optional): Metadata filters to apply (e.g., `{"doc_type": "manual"}`).
        expand_context (bool, optional): If True, retrieves adjacent chunks
            to provide broader context for each hit. Defaults to True.
        context_depth (int, optional): Number of surrounding chunks to
            fetch per result. Defaults to 1.
        space (str, optional): The name of the embedding space to search.
            Defaults to "text".
    """

    query_embedding: list[float]
    query_text: str | None = None
    top_k: int = 10
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1
    space: str = "text"


class TextSearchRequest(BaseModel):
    """Natural language search request where the node handles embedding.

    Attributes:
        query_text (str): The search query in plain text.
        top_k (int, optional): Number of results. Defaults to 10.
        filters (dict, optional): Metadata filters.
        expand_context (bool, optional): Whether to fetch adjacent chunks.
        context_depth (int, optional): Surrounding context window size.
        mode (str, optional): Search algorithm to use ("dense", "sparse",
            "hybrid"). Defaults to "hybrid".
        space (str, optional): The embedding space to target. Defaults to "text".
    """

    query_text: str
    top_k: int = 10
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1
    mode: str = "hybrid"
    space: str = "text"


class ChunkResult(BaseModel):
    """A single retrieved search result (a chunk of a document).

    Attributes:
        chunk_id (str): Unique identifier for the chunk.
        doc_id (str): Identifier of the parent document.
        text (str): The text content of this chunk.
        score (float): The final relevance score (e.g., RRF or Cosine).
        level (int, optional): Hierarchical level (0 for base chunks).
        level_type (str, optional): Name of the level (e.g., "chunk", "section").
        metadata (dict, optional): Metadata associated with the document/chunk.
        parent_text (str, optional): Content of the parent node if context
            expansion was performed.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    level: int = 0
    level_type: str = "chunk"
    metadata: dict = Field(default_factory=dict)
    parent_text: str | None = None


class SearchResponse(BaseModel):
    """The standard response containing search hits and performance timing.

    Attributes:
        chunks (list[ChunkResult]): The ranked list of matching chunks.
        total (int): The total number of hits matching the query.
        score_type (str, optional): The type of scoring used (e.g., "rrf").
        embedding_time_ms (float, optional): Time taken to embed the query.
        dense_time_ms (float, optional): Time taken for the dense FAISS search.
        sparse_time_ms (float, optional): Time taken for the FTS5 sparse search.
        fusion_time_ms (float, optional): Time taken to fuse ranked lists.
        total_time_ms (float, optional): Total end-to-end request latency.
    """

    chunks: list[ChunkResult]
    total: int
    score_type: str = "rrf"
    embedding_time_ms: float = 0
    dense_time_ms: float = 0
    sparse_time_ms: float = 0
    fusion_time_ms: float = 0
    total_time_ms: float = 0


class HotfixAddRequest(BaseModel):
    """Request to add a new chunk to the query node's real-time buffer.

    Attributes:
        chunk_id (str): Unique identifier for the new chunk.
        doc_id (str): Identifier for the parent document.
        text (str): The chunk's text content.
        embedding (list[float]): The pre-calculated embedding vector.
        metadata (dict, optional): Key-value metadata for the chunk.
        space (str, optional): The target embedding space. Defaults to "text".
    """

    chunk_id: str
    doc_id: str
    text: str
    embedding: list[float]
    metadata: dict = Field(default_factory=dict)
    space: str = "text"


class HotfixDeleteRequest(BaseModel):
    """Request to logically delete chunks from the query node's search results.

    Attributes:
        chunk_ids (list[str]): List of identifiers to exclude from search results.
        space (str, optional): The target embedding space. Defaults to "text".
    """

    chunk_ids: list[str]
    space: str = "text"


class HotfixResponse(BaseModel):
    """Confirmation of a real-time hotfix operation.

    Attributes:
        operation (str): The operation type ("add" or "delete").
        affected (int): Number of chunks modified in the buffer.
        buffer_size (int): The new total size of the hotfix buffer for the space.
    """

    operation: str
    affected: int
    buffer_size: int


class HealthResponse(BaseModel):
    """Basic health check status.

    Attributes:
        status (str): The node status ("ok", "starting", etc.).
        node_type (str): Either "query" or "writer".
        version (str): The code version of the running node.
    """

    status: str = "ok"
    node_type: str = ""
    version: str = ""


class ReadinessResponse(BaseModel):
    """Detailed probe to determine if the node can serve traffic.

    Attributes:
        ready (bool): True if the node is fully initialized (e.g., index loaded).
        active_version (str, optional): The snapshot version currently being served.
        vector_count (int, optional): Total vectors available for search.
        doc_count (int, optional): Total documents available for search.
    """

    ready: bool
    active_version: str = ""
    vector_count: int = 0
    doc_count: int = 0


# ── Multi-space search models ──


class SpaceQuery(BaseModel):
    """A sub-query targeting a specific embedding space.

    Attributes:
        space (str): The identifier of the space (e.g., "v1", "v2", "images").
        query_embedding (list[float]): The pre-calculated vector for this space.
        query_text (str, optional): The raw text for sparse path in this space.
        weight (float, optional): Contribution weight during fusion. Defaults to 1.0.
    """

    space: str
    query_embedding: list[float]
    query_text: str | None = None
    weight: float = 1.0


class MultiSpaceSearchRequest(BaseModel):
    """Advanced search across multiple model generations or modalities.

    Attributes:
        queries (list[SpaceQuery]): List of individual space queries to execute.
        top_k (int, optional): Number of final results. Defaults to 10.
        fusion (str, optional): The fusion algorithm (e.g., "rrf").
        filters (dict, optional): Metadata filters applied across all spaces.
        expand_context (bool, optional): Whether to fetch adjacent chunks.
        context_depth (int, optional): Adjacent context window size.
    """

    queries: list[SpaceQuery]
    top_k: int = 10
    fusion: str = "rrf"
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1


class MultiSpaceSearchResponse(BaseModel):
    """The unified results from a multi-space search operation.

    Attributes:
        chunks (list[ChunkResult]): The final ranked and fused hits.
        total (int): The total number of hits found.
        per_space (dict[str, int]): Count of matching documents found per space.
        total_time_ms (float): End-to-end request latency.
    """

    chunks: list[ChunkResult]
    total: int
    per_space: dict[str, int] = Field(default_factory=dict)
    total_time_ms: float = 0


# ── Sync API models ──


class SyncStatusResponse(BaseModel):
    """Real-time monitoring information for the snapshot sync process.

    Attributes:
        enabled (bool): Whether background sync is active.
        source (str): The source type ("object_store" or "http").
        cron (str): The cron expression being used for scheduling.
        poll_interval_seconds (int): Polling interval in use.
        last_check_at (float): Unix timestamp of the last check for updates.
        last_sync_at (float): Unix timestamp of the last successful index swap.
        last_result (str): Outcome of the last sync check.
        last_version (str): The snapshot version found during the last check.
        next_check_at (float): Unix timestamp of the next scheduled sync check.
        consecutive_errors (int): Number of failed sync attempts in a row.
        current_version (str): The snapshot version currently being served.
    """

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
    """Manual request to trigger a snapshot pull or swap.

    Attributes:
        source_url (str, optional): A specific URL to pull a snapshot from,
            bypassing the configured global source.
        snapshot_dir (str, optional): A local directory path to swap to
            directly from the filesystem.
    """

    source_url: str = ""
    snapshot_dir: str = ""


# ── Debug API models ──


class DebugSearchRequest(BaseModel):
    """A detailed search request that returns internal ranking state.

    Attributes:
        query_text (str): The search query in plain text.
        top_k (int, optional): Number of results.
        filters (dict, optional): Metadata filters.
        expand_context (bool, optional): Whether to fetch adjacent chunks.
        context_depth (int, optional): Context window size.
        mode (str, optional): Search algorithm.
        space (str, optional): Target embedding space.
    """

    query_text: str
    top_k: int = 10
    filters: dict | None = None
    expand_context: bool = True
    context_depth: int = 1
    mode: str = "hybrid"
    space: str = "text"


class DebugDenseHit(BaseModel):
    """An intermediate hit from the dense retrieval path.

    Attributes:
        chunk_id (str): Chunk identifier.
        score (float): Original dense similarity score.
    """

    chunk_id: str
    score: float


class DebugSparseHit(BaseModel):
    """An intermediate hit from the sparse retrieval path.

    Attributes:
        chunk_id (str): Chunk identifier.
        score (float): Original sparse relevance score.
    """

    chunk_id: str
    score: float


class DebugFusedHit(BaseModel):
    """Detailed ranking state for a hit after RRF fusion.

    Attributes:
        chunk_id (str): Chunk identifier.
        rrf_score (float): The final calculated RRF score.
        dense_score (float): Score contribution from dense path.
        sparse_score (float): Score contribution from sparse path.
        dense_rank (int): Rank of this chunk in the dense list (-1 if not found).
        sparse_rank (int): Rank of this chunk in the sparse list (-1 if not found).
    """

    chunk_id: str
    rrf_score: float
    dense_score: float
    sparse_score: float
    dense_rank: int = -1
    sparse_rank: int = -1


class DebugTiming(BaseModel):
    """Fine-grained breakdown of search latency components.

    Attributes:
        embedding_ms (float): Latency of external embedding call.
        dense_ms (float): Latency of FAISS search.
        sparse_ms (float): Latency of SQLite FTS5 search.
        fusion_ms (float): Latency of the fusion algorithm.
        fetch_ms (float): Latency of fetching text from the database.
        expand_ms (float): Latency of hierarchical context expansion.
        total_ms (float): Total end-to-end request time.
    """

    embedding_ms: float = 0
    dense_ms: float = 0
    sparse_ms: float = 0
    fusion_ms: float = 0
    fetch_ms: float = 0
    expand_ms: float = 0
    total_ms: float = 0


class DebugSearchResponse(BaseModel):
    """Detailed diagnostic information for a search query.

    Attributes:
        query_text (str): The original search query.
        mode (str): The search mode used.
        fts_query (str): The actual FTS5 query string generated.
        embedding_time_ms (float): Embedding latency.
        score_type (str): The fusion algorithm used.
        dense_results (list[DebugDenseHit]): The raw top hits from dense path.
        sparse_results (list[DebugSparseHit]): The raw top hits from sparse path.
        fused_results (list[DebugFusedHit]): Rank details after fusion.
        final_chunks (list[ChunkResult]): The final hydrated results returned.
        timing (DebugTiming): Detailed latency breakdown.
        config_snapshot (dict): A snapshot of search-relevant config settings.
    """

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
    """Snapshot of relevant configuration settings at query time."""
