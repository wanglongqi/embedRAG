"""Prometheus metrics for both writer and query nodes."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Query node metrics ──

SEARCH_LATENCY = Histogram(
    "embedrag_search_latency_seconds",
    "Search request latency",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0],
)
SEARCH_COUNT = Counter(
    "embedrag_search_total",
    "Total search requests",
    ["status"],
)
DENSE_LATENCY = Histogram(
    "embedrag_dense_search_seconds",
    "Dense retrieval latency",
)
SPARSE_LATENCY = Histogram(
    "embedrag_sparse_search_seconds",
    "Sparse BM25 retrieval latency",
)
ACTIVE_GENERATION = Gauge(
    "embedrag_active_generation",
    "Currently active snapshot generation version number",
)
ACTIVE_QUERIES = Gauge(
    "embedrag_active_queries",
    "Number of in-flight queries",
)
HOTFIX_BUFFER_SIZE = Gauge(
    "embedrag_hotfix_buffer_vectors",
    "Number of vectors in the hotfix buffer",
)
INDEX_VECTOR_COUNT = Gauge(
    "embedrag_index_vector_count",
    "Total vectors in the active index",
)

# ── Writer node metrics ──

INGEST_COUNT = Counter(
    "embedrag_ingest_docs_total",
    "Total documents ingested",
)
BUILD_DURATION = Histogram(
    "embedrag_build_duration_seconds",
    "Snapshot build duration",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
PUBLISH_DURATION = Histogram(
    "embedrag_publish_duration_seconds",
    "Snapshot publish/upload duration",
)
