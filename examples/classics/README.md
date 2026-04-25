# Example: 经典合集 (论语 + 庄子)

Combined indexing of **论语** (20 chapters) and **庄子** (33 chapters) into a single unified index. Demonstrates **ingesting multiple data directories** into one search index, with distinct `doc_type` tags for per-source filtering.

## Stats

- 53 documents, 261 chunks, 1024-dim embeddings (`text-embedding-qwen3-embedding-0.6b`), 2 FAISS shards
- Chunking strategy: `paragraph`

## Design Choices

- **Chunking**: `paragraph` strategy -- both texts are structured as short, self-contained passages separated by blank lines. Paragraph splitting preserves each saying or parable as a unit.
- **Filtering**: Each book gets its own `doc_type` (`classic_lunyu`, `classic_zhuangzi`), so you can search across both or filter to one source.

## Quick Start

### Option A: Use pre-built snapshot (if `snapshot/` directory exists)

```bash
uv run embedrag query --config examples/classics/query.yaml

curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "仁义道德", "top_k": 5, "mode": "hybrid"}'

# Filter to only 庄子
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "逍遥自在", "top_k": 5, "filters": {"doc_type": "classic_zhuangzi"}}'
```

### Option B: Build from scratch

Requires an OpenAI-compatible embedding service (see `writer.yaml`).

```bash
# 1. Start writer
uv run embedrag writer --config examples/classics/writer.yaml

# 2. Ingest both books
uv run python examples/classics/ingest.py

# 3. Copy snapshot
mkdir -p examples/classics/snapshot/active
cp -r /tmp/embedrag-classics/builds/v*/ examples/classics/snapshot/active/

# 4. Start query node
uv run embedrag query --config examples/classics/query.yaml
```

## Hot-Swap Demo

Demonstrates zero-downtime data updates on a running query node:

```bash
# 1. Start the query node
uv run embedrag query --config examples/classics/query.yaml

# 2. In another terminal, run the hot-swap demo
bash examples/classics/hot_swap_demo.sh
```

The script will:
1. Re-ingest data via a temporary writer node
2. Build a new snapshot version
3. Copy it to the query node's active directory
4. Call `POST /admin/reload` to hot-swap -- the query node never goes down

## Files

- `writer.yaml` / `query.yaml` -- Node configs (2 shards, suitable for small datasets)
- `ingest.py` -- Reads from `data/lunyu/` and `data/zhuangzi/`, tags each with its book name and doc_type
- `hot_swap_demo.sh` -- Interactive hot-swap demonstration script
- `snapshot/` -- Pre-built index data (gitignored; build from scratch if missing)
