# Example: 红楼梦 (Dream of the Red Chamber)

Full-text indexing of all 120 chapters with hybrid dense+sparse retrieval.

## Stats

- 120 chapters, ~1631 chunks, 1024-dim embeddings (`text-embedding-qwen3-embedding-0.6b`), 4 FAISS shards
- Chunking strategy: `paragraph` (split on natural paragraph boundaries `\n\n`)
- Hybrid search: dense (FAISS) + sparse (FTS5 trigram) fused via RRF
- Build time: ~1s, Ingestion time: depends on embedding service speed

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

### Option B: Pull pre-built snapshot from a URL

```bash
# Pull from a GitHub Release or any HTTP URL
uv run embedrag pull https://github.com/<user>/<repo>/releases/download/v1/hongloumeng.tar.zst \
  --output examples/hongloumeng/snapshot/active

# Or pull from a snapshot server (base URL with latest.json)
uv run embedrag pull https://cdn.example.com/snapshots/hongloumeng/ \
  --output examples/hongloumeng/snapshot/active

# Start the query node
uv run embedrag query --config examples/hongloumeng/query.yaml
```

### Option C: Build from scratch

Requires an OpenAI-compatible embedding service (see `writer.yaml` for URL and model).

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

- `writer.yaml` -- Writer node config (embedding service URL and model)
- `query.yaml` -- Query node config (reads from `snapshot/`)
- `ingest.py` -- Ingestion script that reads `data/hongloumeng/` and POSTs to `/ingest`
- `snapshot/` -- Pre-built index data (gitignored; build from scratch or pull if missing)
