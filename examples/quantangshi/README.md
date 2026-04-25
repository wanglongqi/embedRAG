# Example: 全唐诗 (Complete Tang Poems)

All 900 volumes of 全唐诗 (~33,500 poems) indexed as individual documents. Demonstrates handling **large volumes of short, self-contained texts** with complex source structure.

## Stats

- ~33,500 poems from 900 volumes, ~35,000 chunks, 4 FAISS shards
- Chunking strategy: `plain` (each poem is short enough to be a single chunk)
- Hierarchy expansion: **disabled** (poems are standalone, no parent/child context)
- Author bios are ingested as separate documents with `doc_type: author_bio`

## Design Choices

- **Per-poem documents**: Each poem is a single document. Since most Tang poems are under 200 characters, `plain` chunking keeps them as one chunk.
- **No hierarchy**: `enable_hierarchy_expand: false` and `context_depth: 0` in query config -- poems don't benefit from parent-chunk expansion.
- **4 shards**: With 33K+ vectors, 4 shards balance memory usage and search parallelism.
- **Author metadata**: Every poem carries `metadata.author` and `metadata.volume` for filtering.

## Data Pipeline

The data pipeline has two stages:

1. **Download** (`download_poems.py`): Fetches all 900 volumes from Wikisource, parses HTML to extract individual poems with author/title/text, and writes a JSONL file.
2. **Ingest** (`ingest.py`): Reads the JSONL file and sends poems in batches to the writer node.

## Quick Start

### Option A: Use pre-built snapshot (if `snapshot/` directory exists)

```bash
uv run embedrag query --config examples/quantangshi/query.yaml

curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "月落乌啼霜满天", "top_k": 5}'

# Search by author
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "春江花月夜", "top_k": 5, "filters": {"author": "張若虛"}}'
```

### Option B: Build from scratch

Requires an OpenAI-compatible embedding service at `127.0.0.1:1234`.

```bash
# 1. Download poems (takes ~4 minutes, fetches from Wikisource)
python examples/quantangshi/download_poems.py --output data/quantangshi_poems.jsonl

# 2. Start writer
uv run embedrag writer --config examples/quantangshi/writer.yaml

# 3. Ingest all poems (takes ~13 minutes with local embedding)
uv run python examples/quantangshi/ingest.py --writer-url http://localhost:8001

# 4. Copy snapshot
mkdir -p examples/quantangshi/snapshot/active
cp -r /tmp/embedrag-qts/builds/v*/ examples/quantangshi/snapshot/active/

# 5. Start query node
uv run embedrag query --config examples/quantangshi/query.yaml
```

## Files

- `download_poems.py` -- Downloads and parses 全唐诗 from Wikisource into JSONL
- `ingest.py` -- Reads JSONL and ingests poems into writer node
- `writer.yaml` / `query.yaml` -- Node configs (4 shards)
- `snapshot/` -- Pre-built index data (gitignored; build from scratch if missing)
