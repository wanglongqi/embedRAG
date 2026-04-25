# Operations Guide

## Data Flow

```
Ingest → Chunk → Embed → Store (SQLite WAL)
         ↓
Build  → FAISS index + export read-only DB → zstd compress → manifest.json
         ↓
Publish → upload to S3/TOS (latest.json updated atomically)
         ↓
Query nodes poll → delta download → verify → decompress → load → atomic swap
```

## Common Workflows

### Initial Data Load

```bash
# 1. Start writer
embedrag writer --config config/writer_node.yaml

# 2. Ingest documents (can call repeatedly)
curl -X POST http://writer:8001/ingest -H 'Content-Type: application/json' \
  -d '{"documents": [{"doc_id": "d1", "title": "Doc 1", "text": "..."}]}'

# 3. Build index + snapshot
curl -X POST http://writer:8001/build

# 4. Publish to object store
curl -X POST http://writer:8001/publish

# 5. Start query nodes (they bootstrap from latest snapshot)
embedrag query --config config/query_node.yaml
```

### Updating Data

```bash
# Ingest new/updated documents
curl -X POST http://writer:8001/ingest -d '{"documents": [...]}'

# Rebuild (delta-aware: only changed shards are re-uploaded)
curl -X POST http://writer:8001/build
curl -X POST http://writer:8001/publish

# Query nodes auto-detect the new version within poll_interval_seconds
# Or trigger manually:
curl -X POST http://query-node:8000/admin/sync
```

### Emergency Hotfix

When a bad chunk needs to be suppressed immediately (before the next full build):

```bash
# Delete a chunk from search results
curl -X POST http://query-node:8000/admin/hotfix/delete \
  -H 'Content-Type: application/json' \
  -d '{"chunk_ids": ["bad_chunk_id"]}'

# Add an emergency replacement
curl -X POST http://query-node:8000/admin/hotfix/add \
  -H 'Content-Type: application/json' \
  -d '{"chunk_id": "fix_001", "doc_id": "d1", "text": "corrected text", "embedding": [0.1, ...]}'
```

Hotfix changes are merged into search results in-memory and cleared automatically when the next snapshot loads.

## Health Checks

| Endpoint | Node | Use For |
|----------|------|---------|
| `GET /health` | Both | Liveness probe (always 200 if process is up) |
| `GET /readiness` | Query | Readiness probe (503 until a snapshot is loaded) |

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet: {path: /health, port: 8000}
  periodSeconds: 10
readinessProbe:
  httpGet: {path: /readiness, port: 8000}
  initialDelaySeconds: 30
  periodSeconds: 5
```

## Snapshot Lifecycle

Query node data directory layout:

```
/data/embedrag/
├── active/
│   └── v0001713800000/       # currently loaded generation
│       ├── manifest.json
│       ├── index/
│       │   └── text/                  # per-space subdirectory
│       │       ├── shard_0.faiss.zst  (compressed)
│       │       ├── shard_0.faiss      (decompressed, mmap'd)
│       │       └── id_map.msgpack.zst
│       └── db/
│           ├── embedrag.db.zst
│           └── embedrag.db         (read-only)
├── staging/                  # in-progress downloads
└── backup/                   # previous generation (kept for rollback)
```

For multi-modal deployments, each embedding space gets its own subdirectory under `index/`:

```
index/
├── text/
│   ├── shard_0.faiss.zst
│   ├── shard_1.faiss.zst
│   └── id_map.msgpack.zst
└── image/
    ├── shard_0.faiss.zst
    └── id_map.msgpack.zst
```

The manifest (v3) uses `indexes` and `id_maps` dicts keyed by space name. Old v2 manifests (with singular `index` / `id_map`) are loaded transparently as a single `"text"` space.

The syncer keeps the last 2 generations in `active/` and cleans up older ones.

## Monitoring

Prometheus metrics are exposed on the metrics port (default 9090). Key metrics:

- `embedrag_search_latency_seconds` - Search request latency histogram
- `embedrag_search_total` - Total search requests counter
- `embedrag_active_generation` - Currently loaded snapshot version
- `embedrag_hotfix_buffer_size` - Current hotfix buffer entry count
- `embedrag_sync_duration_seconds` - Snapshot sync duration
- `embedrag_shard_load_seconds` - Per-shard FAISS load time

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Query node 503 on `/readiness` | No snapshot loaded | Check object store connectivity and `bootstrap_version` config |
| High search latency | Too many nprobe / shards not mmap'd | Reduce `index.nprobe`, ensure `index.mmap: true` |
| OOM on query node | Index too large for RAM | Enable mmap, reduce `db.cache_size_mb`, use PQ compression |
| Sync not picking up new version | `latest.json` not updated | Ensure `/publish` was called on writer |
| Checksum mismatch during sync | Corrupted upload | Re-run `/publish` on writer; downloader retries 3 times per file |

## Resource Budget (4C16G)

| Component | Memory |
|-----------|--------|
| FAISS shards (mmap) | ~0 RSS (pages loaded on demand by OS) |
| SQLite read-only | 64MB page cache per pool |
| Hotfix buffer | ~40MB for 10K vectors at dim=1024 |
| Python + FastAPI | ~200MB baseline |
| **Total RSS** | **~500MB** (rest available for OS page cache) |
