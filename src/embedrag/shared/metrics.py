"""Prometheus metrics for both writer and query nodes.

This module defines the Prometheus metrics used to monitor the performance,
health, and internal state of EmbedRAG nodes. These metrics can be scraped
by a Prometheus server and used for building Grafana dashboards or setting
up operational alerts.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Query node metrics ──

SEARCH_LATENCY = Histogram(
    "embedrag_search_latency_seconds",
    "Total search request latency from the client perspective.",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0],
)
SEARCH_COUNT = Counter(
    "embedrag_search_total",
    "Total number of search requests processed, partitioned by HTTP status code.",
    ["status"],
)
DENSE_LATENCY = Histogram(
    "embedrag_dense_search_seconds",
    "Latency of the dense (FAISS) retrieval phase.",
)
SPARSE_LATENCY = Histogram(
    "embedrag_sparse_search_seconds",
    "Latency of the sparse (SQLite FTS5) retrieval phase.",
)
ACTIVE_GENERATION = Gauge(
    "embedrag_active_generation",
    "The version string of the currently active snapshot generation.",
)
ACTIVE_QUERIES = Gauge(
    "embedrag_active_queries",
    "The current number of concurrent search requests being processed.",
)
HOTFIX_BUFFER_SIZE = Gauge(
    "embedrag_hotfix_buffer_vectors",
    "The number of vectors currently held in the in-memory hotfix buffer.",
)
INDEX_VECTOR_COUNT = Gauge(
    "embedrag_index_vector_count",
    "The total number of vectors in the active FAISS index.",
)

# ── Writer node metrics ──

INGEST_COUNT = Counter(
    "embedrag_ingest_docs_total",
    "Total number of documents successfully ingested into the writer node.",
)
BUILD_DURATION = Histogram(
    "embedrag_build_duration_seconds",
    "Time taken to build a new snapshot generation.",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
PUBLISH_DURATION = Histogram(
    "embedrag_publish_duration_seconds",
    "Time taken to upload a finished snapshot to the object store.",
)
