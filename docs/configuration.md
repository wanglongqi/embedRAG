# Configuration Reference

EmbedRAG uses YAML configuration files with Pydantic validation. All fields have sensible defaults; a minimal config only needs `node.role` set.

## Loading

```bash
embedrag writer --config config/writer_node.yaml
embedrag query  --config config/query_node.yaml
```

Without `--config`, the node starts with all defaults. The config determines node type via `node.role`.

## Shared Sections

These sections appear in both writer and query configs.

### node

```yaml
node:
  role: query          # "query" or "writer" (required distinction)
  node_id: auto        # hostname if "auto", otherwise a custom string
  data_dir: /data/embedrag  # root for all local data (snapshots, db, builds)
```

### object_store

```yaml
object_store:
  provider: s3         # "s3", "tos", or "minio"
  endpoint: ""         # custom endpoint (required for minio/tos)
  bucket: embedrag-data
  prefix: snapshots/   # key prefix inside the bucket
  access_key_env: AWS_ACCESS_KEY_ID    # env var name for access key
  secret_key_env: AWS_SECRET_ACCESS_KEY # env var name for secret key
  region: us-east-1
```

Credentials are never stored in the YAML. The `*_env` fields name environment variables that are read at runtime.

### server

```yaml
server:
  host: 0.0.0.0
  port: 8000           # 8001 for writer
  workers: 1           # uvicorn workers (keep 1 for FAISS mmap)
  readiness_delay_seconds: 0    # delay before reporting ready
  shutdown_drain_seconds: 30    # max wait for in-flight queries
```

### logging

```yaml
logging:
  level: INFO          # DEBUG, INFO, WARNING, ERROR
  format: json         # "json" (structured) or "console" (human-readable)
  access_log: true     # log each HTTP request
```

### metrics

```yaml
metrics:
  enabled: true
  port: 9090           # Prometheus scrape port
```

## Writer-Only Sections

### db

```yaml
db:
  path: ""             # auto-resolved to {data_dir}/db/writer.db if empty
  wal_autocheckpoint: 1000  # WAL checkpoint interval (pages)
  cache_size_mb: 64         # SQLite page cache
```

### index_build

```yaml
index_build:
  num_shards: 4              # FAISS index shards for parallelism
  ivf_nlist: 4096            # IVF cluster count (for datasets > 1K vectors)
  pq_m: 64                  # PQ subquantizer count (for datasets > 50K)
  train_sample_size: 500000  # max vectors for IVF training
  compression: zstd          # "zstd" or "none"
  compression_level: 3       # zstd level (1-19, higher = smaller + slower)
```

Index type is auto-selected based on dataset size:
- `< 1,000` vectors: `IndexFlatIP`
- `1,000 - 50,000`: `IVF{nlist},Flat`
- `> 50,000`: `IVF{nlist},PQ{pq_m}`

## Shared Between Writer and Query

### embedding

```yaml
embedding:
  service_url: http://embedding-service:8080/embed
  api_format: embedrag    # "embedrag" or "openai"
  api_key: ""            # Bearer token for OpenAI format
  model: ""              # model name for OpenAI format
  batch_size: 64         # vectors per API call
  timeout_seconds: 30    # per-batch timeout
  retry_count: 3         # retries on failure
```

On the **writer**, this is used during `/ingest` to embed chunk texts. On the **query** node, it powers the `/search/text` endpoint and the debug panel.

Two API formats are supported:

| Format | Request Body | Response Body |
|--------|-------------|---------------|
| `embedrag` (default) | `{"texts": ["..."]}` | `{"embeddings": [[0.1, ...]]}` |
| `openai` | `{"input": ["..."], "model": "..."}` | `{"data": [{"embedding": [...], "index": 0}]}` |

**OpenAI format example:**

```yaml
embedding:
  service_url: https://api.openai.com/v1/embeddings
  api_format: openai
  api_key: sk-your-api-key
  model: text-embedding-3-large
  batch_size: 64
  timeout_seconds: 30
  retry_count: 3
```

This works with the official OpenAI API, Azure OpenAI, and any OpenAI-compatible embedding provider (e.g., vLLM, Ollama, LiteLLM).

#### Multi-Space Embedding (multi-modal)

To use different embedding models for different modalities (text, image, audio, etc.), use the `spaces` dict:

```yaml
embedding:
  spaces:
    text:
      service_url: http://text-embed:8080/v1/embeddings
      api_format: openai
      model: text-embedding-large
      batch_size: 64
    image:
      service_url: http://clip-embed:8081/v1/embeddings
      api_format: openai
      model: clip-vit-large
      batch_size: 32
      timeout_seconds: 60
```

Each space has its own `EmbeddingSpaceConfig` with the same fields as the top-level embedding block. When `spaces` is set, the top-level fields are ignored.

When `spaces` is omitted (the default), the top-level fields are treated as a single `"text"` space — fully backward compatible.

See [docs/multi-modal.md](multi-modal.md) for the full multi-modal architecture guide.

## Query-Only Sections

### snapshot

```yaml
snapshot:
  bootstrap_version: latest     # "latest" or a specific version string
  poll_interval_seconds: 300    # how often to check for new snapshots
  download_concurrency: 4       # parallel file downloads
  download_timeout_seconds: 600 # per-download timeout
  disk_reserve_bytes: 5368709120  # 5GB reserve after download
```

### sync

Controls background snapshot synchronization.  When enabled, the query node
periodically checks a remote source for new snapshot versions and hot-swaps
automatically.

```yaml
sync:
  enabled: false                  # set to true to activate background sync
  source: object_store            # "object_store" or "http"
  http_url: ""                    # base URL when source=http
  cron: ""                        # 5-field cron expression (e.g. "0 */2 * * *")
  poll_interval_seconds: 300      # fallback interval when cron is empty
  download_concurrency: 4         # parallel file downloads
  download_timeout_seconds: 600   # per-download timeout
```

**Source types:**

| Source | Description |
|--------|-------------|
| `object_store` | Uses the `object_store` config section (S3/MinIO/TOS) |
| `http` | Downloads from `sync.http_url` (any HTTP/HTTPS static file server or CDN) |

**Scheduling:** If `cron` is set (e.g. `"*/10 * * * *"` for every 10 minutes),
it takes priority over `poll_interval_seconds`.  Uses the `croniter` library.

**Manual trigger:** `POST /admin/sync` triggers a one-off sync.  Pass
`{"source_url": "https://..."}` to sync from an arbitrary URL regardless of
the configured source.

**Status:** `GET /admin/sync/status` returns last check time, last result,
next scheduled check, and error counts.

**HTTP source example:**

```yaml
sync:
  enabled: true
  source: http
  http_url: "https://my-cdn.example.com/embedrag/snapshots/"
  cron: "0 */2 * * *"
```

**S3 source example:**

```yaml
sync:
  enabled: true
  source: object_store
  cron: "*/10 * * * *"
object_store:
  provider: s3
  bucket: my-rag-bucket
  prefix: snapshots/
```

### index

```yaml
index:
  num_shards: 4    # must match what the writer built
  nprobe: 32       # FAISS search probes (higher = slower + more accurate)
  mmap: true       # memory-map FAISS indexes (recommended for 4C16G)
```

### search

```yaml
search:
  default_top_k: 10           # default results if not specified in request
  max_top_k: 100              # hard cap on results per query
  enable_sparse: true          # enable FTS5 BM25 keyword search
  enable_hierarchy_expand: true # attach parent chunk text to results
  context_depth: 1             # how many levels up to expand
```

### hotfix

```yaml
hotfix:
  enabled: true
  max_vectors: 10000   # max chunks in the emergency write buffer
```

The hotfix buffer is an in-memory FAISS `IndexFlatIP` on the query node. Writes are merged into search results and cleared when the next snapshot loads.

## Full Examples

See `config/writer_node.yaml.example` and `config/query_node.yaml.example` for complete annotated examples.

## Environment Variable Resolution

Any config field ending in `_env` is treated as an environment variable name:

```yaml
object_store:
  access_key_env: MY_CUSTOM_KEY  # reads os.environ["MY_CUSTOM_KEY"]
```

This pattern keeps secrets out of config files while remaining flexible across environments.
