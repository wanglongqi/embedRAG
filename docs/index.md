# Welcome to EmbedRAG

Production-grade Retrieval-Augmented Generation system with read/write split architecture, designed to handle millions of documents at 1000 QPS on a 4C16G machine.

## Architecture

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

EmbedRAG is built for high-performance, production-grade RAG applications. It separates the ingestion and query layers to allow independent scaling and extreme performance on modest hardware.

## Key Features

- **Read/Write Split**: Dedicated writer node for ingestion and query nodes for search.
- **High Performance**: 1000 QPS on a 4C16G machine for millions of documents.
- **Hybrid Search**: Combines FAISS (dense) and SQLite FTS5 (sparse) with Reciprocal Rank Fusion (RRF).
- **Snapshot-based Sync**: Zero-downtime hot-swapping of search indexes.
- **Multilingual Support**: Character-aware chunking and trigram-based sparse search.
- **Multi-Modal**: Support for multiple embedding spaces (text, image, etc.).

## Getting Started

Check out the [Quick Start](index.md#quick-start) section or dive into the full documentation:

- [Configuration Guide](configuration.md)
- [Embedding Setup](embedding.md)
- [Multi-Modal RAG](multi-modal.md)
- [Operational Runbook](operations.md)
- [Integration Guide](integration.md)

---

## Quick Start (from README)

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- An external embedding service that accepts POST requests and returns float vectors

### Install

```bash
git clone <repo-url> && cd embedRAG
uv venv --python 3.11
uv pip install -e ".[dev,docs]"
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
