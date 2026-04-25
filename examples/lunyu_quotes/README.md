# Example: 论语语录 (Flat-Text Ingestion)

Individual quotes from the **论语** (Analects of Confucius), each treated as an **independent, non-hierarchical document**. This example verifies that EmbedRAG works correctly for flat texts with no parent-child structure.

## Design Choices

- **Chunking**: `plain` -- each quote is short enough to be a single chunk. No splitting, no hierarchy.
- **No hierarchy expansion**: `enable_hierarchy_expand: false`, `context_depth: 0` in query config.
- **1 doc = 1 chunk**: Each numbered saying (e.g. "一之一") becomes one document with one chunk.
- **1 FAISS shard**: Small dataset (~501 quotes) needs only one shard.
- **Embeddings**: 1024-dim via `text-embedding-qwen3-embedding-0.6b`.

## Quick Start

### Option A: Use pre-built snapshot (if `snapshot/` exists)

```bash
uv run embedrag query --config examples/lunyu_quotes/query.yaml

# Search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "学而时习之", "top_k": 5, "mode": "hybrid"}'

# Sparse-only (FTS)
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "仁义", "top_k": 5, "mode": "sparse"}'
```

### Option B: Build from scratch

Requires an OpenAI-compatible embedding service (see `writer.yaml`).

```bash
# 1. Start writer
uv run embedrag writer --config examples/lunyu_quotes/writer.yaml

# 2. Ingest all quotes
uv run python examples/lunyu_quotes/ingest.py

# 3. Copy snapshot
mkdir -p examples/lunyu_quotes/snapshot/active
cp -r /tmp/embedrag-lunyu-quotes/builds/v*/ examples/lunyu_quotes/snapshot/active/

# 4. Start query node
uv run embedrag query --config examples/lunyu_quotes/query.yaml
```

## Files

- `writer.yaml` / `query.yaml` -- Node configs (1 shard, no hierarchy)
- `ingest.py` -- Parses numbered quotes from `data/lunyu/`, each as an independent document
- `snapshot/` -- Pre-built index data (gitignored; build from scratch if missing)
