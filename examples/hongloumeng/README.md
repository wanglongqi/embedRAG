# Example: 红楼梦 (Dream of the Red Chamber)

Full-text indexing of all 120 chapters. Demonstrates **structured chunking** for long novel chapters with hierarchical text.

## Stats

- 120 chapters, ~2362 chunks, 1024-dim embeddings, 3 FAISS shards
- Chunking strategy: `structured` (heading-aware 3-level tree)
- Build time: ~1s, Ingestion time: ~200s (depends on embedding speed)

## Quick Start

### Option A: Use pre-built snapshot (if `snapshot/` directory exists)

```bash
# Start the query node directly (run from project root)
uv run embedrag query --config examples/hongloumeng/query.yaml

# Search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "林黛玉进贾府", "top_k": 5, "mode": "hybrid"}'

# WebUI
open http://localhost:8000/ui/
```

### Option B: Build from scratch

Requires an OpenAI-compatible embedding service at `127.0.0.1:1234`.

```bash
# 1. Start the writer node
uv run embedrag writer --config examples/hongloumeng/writer.yaml

# 2. Run ingestion
uv run python examples/hongloumeng/ingest.py --writer-url http://localhost:8001

# 3. Copy snapshot for the query node
mkdir -p examples/hongloumeng/snapshot/active
cp -r /tmp/embedrag-hlm/builds/v*/ examples/hongloumeng/snapshot/active/

# 4. Start the query node
uv run embedrag query --config examples/hongloumeng/query.yaml
```

## Files

- `writer.yaml` -- Writer node config (embedding via OpenAI-format API)
- `query.yaml` -- Query node config (reads from `snapshot/` directory)
- `ingest.py` -- Ingestion script that reads `data/hongloumeng/` and POSTs to `/ingest`
- `snapshot/` -- Pre-built index data (gitignored; build from scratch if missing)
