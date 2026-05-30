# Welcome to EmbedRAG

Production-grade Retrieval-Augmented Generation (RAG) system with a high-performance **read/write split** architecture. Designed to handle millions of documents with sub-second latency (1000+ QPS) on modest hardware (4C16G).

## Why EmbedRAG?

Most RAG systems struggle with scaling ingestion and search simultaneously. EmbedRAG solves this by separating the **Writer Node** (ingestion, chunking, embedding, index building) from the **Query Node** (high-speed retrieval, fusion, search).

- **Extreme Performance**: Engineered for high-throughput search with FAISS and SQLite FTS5.
- **Cost-Efficient**: Achieving 1000 QPS on a single 16GB RAM machine.
- **Production Ready**: Built-in snapshot management, zero-downtime synchronization, and structured logging.
- **Flexible Search**: Hybrid search (dense + sparse) with Reciprocal Rank Fusion (RRF) and hierarchical expansion.
- **Multi-Modal**: Support for heterogeneous embedding spaces (text, images, audio).

## Architecture

EmbedRAG's architecture revolves around a "Build-Publish-Sync" lifecycle:

```mermaid
graph TD
    Embed[Embedding Service] --> Writer
    subgraph "Writer Node (:8001)"
    Writer[Writer] -- "/ingest" --> Chunking
    Chunking --> Embedding
    Embedding --> SQLite[SQLite WAL]
    SQLite -- "/build" --> FAISS[FAISS Index]
    FAISS -- "/publish" --> S3[S3/TOS Storage]
    end
    S3 -- "snapshot" --> Query
    subgraph "Query Node (:8000)"
    Query[Query Node] -- "startup" --> Load[Load Snapshot]
    Load --> Mmap[FAISS mmap]
    Load --> SQLiteRO[SQLite RO]
    Search[/search] --> Fusion[RRF Fusion]
    Fusion --> Results
    end
```

## Documentation Roadmap

Dive into the details of EmbedRAG:

- [**Quick Start**](#quick-start): Get a local cluster running in minutes.
- [**Configuration Guide**](configuration.md): Detailed reference for all YAML settings.
- [**Embedding Setup**](embedding.md): How to connect your favorite embedding models (OpenAI, HuggingFace, etc.).
- [**Multi-Modal RAG**](multi-modal.md): Building RAG systems for more than just text.
- [**Operational Runbook**](operations.md): Deploying, monitoring, and scaling EmbedRAG in production.
- [**Integration Guide**](integration.md): Client libraries and API integration patterns.
- [**API Reference**](api.md): Automatically generated documentation from source code.

---

## Quick Start

### Prerequisites

- **Python**: 3.11 or later.
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or `pip`.
- **Embedding API**: An external service (like OpenAI or a local vLLM/Ollama instance) that returns float vectors.

### 1. Install

```bash
git clone <repo-url> && cd embedRAG
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,docs]"
```

### 2. Start the Writer Node

The writer handles ingestion and index building.

```bash
# Start with default settings (data at /data/embedrag)
embedrag writer --port 8001
```

### 3. Start the Query Node

The query node serves search requests.

```bash
# Start with default settings
embedrag query --port 8000
```

### 4. Search and Explore

Once you've ingested some data via the writer, you can perform hybrid searches:

```bash
curl "http://localhost:8000/search/text?query_text=What+is+EmbedRAG&mode=hybrid"
```
