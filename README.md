# EmbedRAG

Production-grade Retrieval-Augmented Generation system with read/write split architecture, designed to handle millions of documents at 1000 QPS on a 4C16G machine.

## Architecture

```
                  ┌─────────────┐
                  │  Embedding   │
                  │  Service     │
                  └──────┬───────┘
                         │
┌────────────────────────▼────────────────────────┐
│                  Writer Node (:8001)             │
│  /ingest → chunking → embedding → SQLite (WAL)  │
│  /build  → FAISS index → zstd compress           │
│  /publish → upload snapshot to S3/TOS            │
└────────────────────────┬────────────────────────┘
                         │ snapshot (S3/TOS)
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐ ┌──────────────────────┐
│  Query Node A (:8000)│ │  Query Node B (:8000)│
│  FAISS mmap (ro)     │ │  FAISS mmap (ro)     │
│  SQLite (ro)         │ │  SQLite (ro)         │
│  FTS5 BM25           │ │  FTS5 BM25           │
│  RRF fusion          │ │  RRF fusion          │
│  Hotfix buffer       │ │  Hotfix buffer       │
└──────────────────────┘ └──────────────────────┘
```

**Writer node** ingests documents, chunks text (heading-aware + sliding window), calls an external embedding service, builds sharded FAISS indexes, compresses artifacts with zstd, and publishes snapshots to S3-compatible object storage.

**Query nodes** are stateless readers. On startup they pull the latest snapshot, decompress it, memory-map the FAISS shards, and open SQLite in read-only mode. A background syncer polls for new versions and hot-swaps them with zero downtime. An in-memory hotfix buffer handles emergency writes until the next snapshot overwrites them.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An external embedding service that accepts POST requests and returns float vectors

### Install

```bash
git clone <repo-url> && cd embedRAG
uv venv --python 3.11
uv pip install -e ".[dev]"
```

### Run the Writer

```bash
# With config file
embedrag writer --config config/writer_node.yaml.example --port 8001

# Or with defaults (SQLite at /data/embedrag-writer/db/writer.db)
embedrag writer
```

### Run a Query Node

```bash
embedrag query --config config/query_node.yaml.example --port 8000
```

### Docker

```bash
# Build
docker build --target writer -t embedrag-writer .
docker build --target query  -t embedrag-query  .

# Run
docker run -p 8001:8001 -v /data:/data embedrag-writer
docker run -p 8000:8000 -v /data:/data embedrag-query
```

## API Reference

### Writer Node (default `:8001`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Ingest documents (chunking + embedding) |
| `POST` | `/build` | Build FAISS index + snapshot |
| `POST` | `/publish` | Upload snapshot to object store |

#### POST /ingest

Each document can specify its own chunking strategy via the `chunking` field:

| Strategy | Best For | Behavior |
|----------|----------|----------|
| `auto` (default) | Mixed content | Detects headings -> hierarchy; otherwise sliding window |
| `structured` | Markdown, HTML with headings | Forces heading-aware 3-level tree (document -> section -> paragraph) |
| `plain` | Flat text, logs, code | Sliding window only, no hierarchy |
| `paragraph` | Articles, Q&A, medium-length text | Splits on paragraph boundaries (`\n\n`), merges short ones, no hierarchy |

Per-document `chunk_size` and `chunk_overlap` can also be set (defaults: 512 / 128 tokens).

```json
{
  "documents": [
    {
      "doc_id": "doc_001",
      "title": "Introduction to RAG",
      "text": "# What is RAG?\n\nRetrieval-Augmented Generation...",
      "doc_type": "article",
      "chunking": "structured"
    },
    {
      "doc_id": "faq_042",
      "title": "Password Reset",
      "text": "To reset your password, go to Settings...",
      "doc_type": "faq",
      "chunking": "paragraph"
    },
    {
      "doc_id": "log_99",
      "title": "Error Trace",
      "text": "2024-01-01 ERR connection timeout...",
      "doc_type": "log",
      "chunking": "plain",
      "chunk_size": 256
    }
  ]
}
```

Response:

```json
{
  "ingested": 3,
  "chunk_count": 12,
  "doc_ids": ["doc_001", "faq_042", "log_99"]
}
```

#### POST /build

```json
{"force_full_rebuild": false}
```

Response:

```json
{
  "version": "v0001713800000",
  "doc_count": 100,
  "chunk_count": 1200,
  "vector_count": 1200,
  "num_shards": 4,
  "build_time_seconds": 12.5
}
```

### Query Node (default `:8000`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/readiness` | Readiness probe (503 until loaded) |
| `GET` | `/api/spaces` | List available embedding spaces |
| `POST` | `/search` | Search with pre-computed embedding |
| `POST` | `/search/text` | Text-only search (embedding computed internally) |
| `POST` | `/search/multi` | Cross-space search with late fusion |
| `POST` | `/api/debug/search` | Debug search with full pipeline breakdown |
| `GET` | `/ui/` | WebUI (search, debug, status dashboard) |
| `POST` | `/admin/hotfix/add` | Emergency add chunk |
| `POST` | `/admin/hotfix/delete` | Emergency delete chunk |
| `POST` | `/admin/sync` | Trigger manual snapshot sync (optionally from a URL) |
| `GET` | `/admin/sync/status` | Sync status (last check, next check, errors) |
| `POST` | `/admin/reload` | Hot-swap snapshot from local disk |

#### POST /search (with pre-computed embedding)

For callers that already have a query embedding:

```json
{
  "query_embedding": [0.1, 0.2, ...],
  "query_text": "what is retrieval augmented generation",
  "top_k": 10,
  "filters": {"doc_type": "article"},
  "expand_context": true,
  "context_depth": 1
}
```

#### POST /search/text (text-only, recommended)

Send plain text -- the query node calls the embedding service internally:

```json
{
  "query_text": "what is retrieval augmented generation",
  "top_k": 10,
  "filters": {"doc_type": "article"},
  "mode": "hybrid"
}
```

The `mode` field controls the retrieval pipeline:

| Mode | Behavior |
|------|----------|
| `hybrid` (default) | Dense (FAISS) + Sparse (BM25) merged via RRF |
| `dense` | Dense vector search only |
| `sparse` | BM25 keyword search only (no embedding call needed) |

Response (same for both endpoints):

```json
{
  "chunks": [
    {
      "chunk_id": "a1b2c3d4",
      "doc_id": "doc_001",
      "text": "Retrieval-Augmented Generation combines...",
      "score": 0.92,
      "level": 2,
      "level_type": "paragraph",
      "metadata": {"title": "Introduction to RAG"},
      "parent_text": "# What is RAG?\n\n..."
    }
  ],
  "total": 10,
  "embedding_time_ms": 15.2,
  "dense_time_ms": 2.3,
  "sparse_time_ms": 1.1,
  "fusion_time_ms": 0.1,
  "total_time_ms": 25.0
}
```

## WebUI

The query node ships with a built-in web interface at `http://query-node:8000/ui/`. No build step, no npm -- pure HTML + vanilla JS.

**Three tabs:**

- **Search** -- text input with mode (hybrid/dense/sparse), top_k slider, filters, timing chips, and result cards with parent context expansion.
- **Debug** -- same query input but shows the full pipeline breakdown: the exact FTS5 query string generated, dense/sparse/fused result tables with scores and ranks, a timing waterfall chart, and the active config snapshot.
- **Status** -- health check, active version, vector/document counts, and embedding service connectivity probe.

## Multi-Modal RAG (Named Index Spaces)

EmbedRAG supports multiple modalities (text, image, audio) through **Named Index Spaces**. Each modality has its own embedding model and FAISS index while sharing the SQLite document store.

### Quick example

```yaml
# writer_node.yaml
embedding:
  spaces:
    text:
      service_url: http://text-embed:8080/v1/embeddings
      api_format: openai
      model: text-embedding-large
    image:
      service_url: http://clip-embed:8081/v1/embeddings
      api_format: openai
      model: clip-vit-large
```

```json
POST /ingest
{
  "documents": [
    {"doc_id": "d1", "text": "Hello world", "modality": "text"},
    {"doc_id": "d2", "text": "A cat on a mat", "modality": "image", "content_ref": "cat.jpg"}
  ]
}
```

```json
POST /search/text
{"query_text": "cat", "space": "image", "top_k": 5}
```

Cross-modal fusion is available via `POST /search/multi` for searching multiple spaces with weighted late fusion.

Single-space (text-only) deployments require **zero config changes** — all defaults remain `"text"`.

See [docs/multi-modal.md](docs/multi-modal.md) for the full architecture guide.

## OpenAI-Compatible Embedding

EmbedRAG supports both its native embedding format and the OpenAI API format. Set `api_format: openai` to use OpenAI, Azure OpenAI, or any OpenAI-compatible provider:

```yaml
embedding:
  service_url: https://api.openai.com/v1/embeddings
  api_format: openai
  api_key: sk-your-api-key
  model: text-embedding-3-large
```

Works with vLLM, Ollama, LiteLLM, and other providers that implement the `/v1/embeddings` interface.

## Configuration

Both nodes are configured via YAML files. See:
- `config/writer_node.yaml.example` and `config/query_node.yaml.example` for full annotated examples
- [docs/configuration.md](docs/configuration.md) for the complete reference
- [docs/operations.md](docs/operations.md) for the operational runbook
- [docs/multi-modal.md](docs/multi-modal.md) for the multi-modal architecture guide

### Key Settings

| Section | Field | Default | Description |
|---------|-------|---------|-------------|
| `node.role` | `query` / `writer` | `query` | Node type |
| `node.data_dir` | path | `/data/embedrag` | Root data directory |
| `object_store.provider` | `s3` / `tos` / `minio` | `s3` | Object storage backend |
| `object_store.bucket` | string | `embedrag-data` | Bucket name |
| `index.nprobe` | int | `32` | FAISS search probe count |
| `index.mmap` | bool | `true` | Memory-map FAISS shards |
| `search.enable_sparse` | bool | `true` | Enable BM25 keyword search |
| `search.max_top_k` | int | `100` | Max results per query |
| `snapshot.poll_interval_seconds` | int | `300` | Bootstrap poll interval |
| `sync.enabled` | bool | `false` | Enable background snapshot sync |
| `sync.source` | `object_store` / `http` | `object_store` | Sync source type |
| `sync.http_url` | url | `""` | Base URL for HTTP sync source |
| `sync.cron` | string | `""` | Cron expression (e.g. `"*/10 * * * *"`) |
| `sync.poll_interval_seconds` | int | `300` | Fallback interval if no cron |
| `hotfix.max_vectors` | int | `10000` | Max emergency buffer size |
| `embedding.service_url` | url | `http://localhost:8080/embed` | Embedding service (both nodes) |
| `embedding.api_format` | `embedrag` / `openai` | `embedrag` | Embedding API format |

### Environment Variables

Object store credentials are read from environment variables (configurable names):

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

## Project Structure

```
src/embedrag/
├── config.py                    # Pydantic config with YAML + env vars
├── logging_setup.py             # structlog configuration
├── cli.py                       # CLI entry point
├── models/
│   ├── api.py                   # Request/response Pydantic models
│   ├── chunk.py                 # Document and ChunkNode dataclasses
│   └── manifest.py              # Snapshot manifest v3 (multi-space)
├── shared/
│   ├── checksum.py              # SHA-256 (full, streaming, quick-verify)
│   ├── compression.py           # zstd compress/decompress
│   ├── disk.py                  # Disk space checks, preallocation
│   ├── metrics.py               # Prometheus counters/gauges/histograms
│   └── object_store.py          # S3/TOS/MinIO client wrapper
├── text/
│   ├── normalize.py             # FTS normalization (NFKC, casefold, trad→simp)
│   └── t2s_chars.json           # Traditional→Simplified Chinese char map
├── writer/
│   ├── app.py                   # FastAPI app + lifespan
│   ├── routes.py                # /ingest, /build, /publish, /health
│   ├── storage.py               # SQLite WAL pool (async r/w)
│   ├── schema.py                # DDL: documents, chunks, FTS5, closure, embeddings
│   ├── embedding_client.py      # Async batched embedding calls
│   ├── index_builder.py         # FAISS Flat/IVF_PQ sharded builder
│   ├── snapshot.py              # Packager (zstd + checksums) + publisher
│   └── chunking/
│       ├── splitter.py          # smart_split, heading-aware, sliding window
│       └── hierarchy.py         # Closure table entries
├── webui/
│   ├── index.html               # SPA shell with tab navigation
│   ├── style.css                # Dark/light theme
│   └── app.js                   # Client-side search, debug, status
└── query/
    ├── app.py                   # FastAPI app + lifespan + WebUI mount
    ├── routes.py                # /search, /search/text, /search/multi, /admin/*
    ├── middleware.py             # Request ID, timing, error logging
    ├── storage.py               # Read-only SQLite pool, LRU docstore
    ├── index/
    │   ├── shard.py             # FAISS shard loader (mmap)
    │   ├── id_mapping.py        # FAISS ID -> chunk_id mapper
    │   ├── generation.py        # Ref-counted atomic generation swap
    │   └── hotfix.py            # In-memory emergency write buffer
    ├── retrieval/
    │   ├── dense.py             # Parallel shard search + merge
    │   ├── sparse.py            # FTS5 BM25 search
    │   ├── fusion.py            # Reciprocal Rank Fusion (RRF)
    │   └── hierarchy_expand.py  # Parent context expansion
    ├── lifecycle/
    │   ├── bootstrap.py         # Cold-start: pull + verify + load
    │   ├── startup.py           # Decompress + load generation
    │   ├── shutdown.py          # Graceful drain + cleanup
    │   └── readiness.py         # Phase state machine
    └── sync/
        ├── downloader.py        # Delta download, checksum, retry
        └── syncer.py            # Background poll + hot-swap loop
```

## Multilingual Support (100+ Languages)

EmbedRAG works out-of-the-box with any language. No configuration needed.

### How It Works

| Layer | Approach | Languages |
|-------|----------|-----------|
| **Chunking** | Character-aware tokenizer counts CJK/Thai characters individually, handles Latin/Cyrillic/Arabic by whitespace. Text is split and re-joined losslessly. | All Unicode scripts |
| **FTS5 (Sparse)** | `trigram case_sensitive 0` tokenizer. Matches CJK terms like `机器学习` as exact substrings; Latin/Cyrillic as case-insensitive substrings. BM25 ranking with length normalization. | All Unicode scripts |
| **Dense Search** | Delegated to external embedding service. Use a multilingual model (e.g., `multilingual-e5-large`, `BGE-M3`, `Cohere multilingual-v3.0`). | Depends on embedding model |
| **Query Building** | Query segments are kept as whole phrases (not split per-character). Segments < 3 chars are dropped (trigram minimum); dense search covers those in hybrid mode. | All Unicode scripts |

**Why trigram?** We evaluated `unicode61`, `trigram`, and LLM tokenizer (BPE). `unicode61` treats CJK runs without spaces as single tokens, making substring search like `机器学习` impossible. BPE has space-prefix sensitivity issues (`"learning"` != `" learning"`). `trigram` is the only zero-dependency option that correctly matches CJK terminology, supports case-insensitive Latin, and maintains BM25-quality ranking via FTS5's built-in `rank` function. The 3-character minimum is acceptable because dense vector search covers short queries in hybrid mode.

### Multilingual Ingestion Example

```json
{
  "documents": [
    {
      "doc_id": "cn_001",
      "title": "机器学习简介",
      "text": "# 简介\n\n机器学习是人工智能的一个分支...",
      "chunking": "structured"
    },
    {
      "doc_id": "ar_001",
      "title": "مقدمة في التعلم الآلي",
      "text": "التعلم الآلي هو فرع من الذكاء الاصطناعي...",
      "chunking": "paragraph"
    },
    {
      "doc_id": "mixed_001",
      "title": "Deep Learning概述",
      "text": "深度学习deep learning是一种machine learning方法...",
      "chunking": "auto"
    }
  ]
}
```

### Recommended Embedding Models

For global multilingual deployments, use one of these embedding models:

| Model | Dim | Languages | Notes |
|-------|-----|-----------|-------|
| `multilingual-e5-large` | 1024 | 100+ | Strong all-round, open-source |
| `BGE-M3` | 1024 | 100+ | State-of-the-art, supports dense+sparse+colbert |
| `Cohere embed-multilingual-v3.0` | 1024 | 100+ | Commercial, excellent quality |

## Key Design Decisions

- **FAISS over Milvus**: Runs in-process with mmap, no extra server on a 4C16G machine.
- **SQLite (not Postgres)**: Zero-dependency, WAL for writer concurrency, read-only on query nodes.
- **Sharded indexes**: Each shard is searched in parallel via ThreadPoolExecutor (FAISS releases the GIL).
- **Snapshot-based sync**: Writer produces immutable snapshots; query nodes atomically swap generations.
- **RRF fusion**: Combines dense (FAISS inner product) and sparse (FTS5 trigram) rankings without score calibration.
- **Flexible chunking**: Four strategies (auto/structured/plain/paragraph) handle everything from hierarchical Markdown to flat FAQ text. Each document picks its own strategy at ingest time.
- **Text-only search**: The `/search/text` endpoint embeds the query internally, so callers never need to manage embeddings themselves.
- **Multilingual by default**: Character-aware chunking + FTS5 trigram index ensure CJK/Latin/Cyrillic/Arabic all work without configuration. Dense retrieval quality depends on the external embedding model.

## Testing

```bash
uv run pytest tests/unit/ -v           # 123 unit tests
uv run pytest tests/unit/ --cov=embedrag # with coverage
```
