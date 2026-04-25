# Multi-Modal RAG (Named Index Spaces)

EmbedRAG supports multiple embedding modalities (text, image, audio, etc.) through **Named Index Spaces**. Each modality gets its own embedding model, FAISS index, and id\_map, while sharing the same SQLite document/chunk store and snapshot packaging.

## Architecture

```
                  ┌──────────────┐   ┌──────────────┐
                  │ Text Embed   │   │ Image Embed  │
                  │ Service      │   │ Service      │
                  └──────┬───────┘   └──────┬───────┘
                         │                   │
┌────────────────────────▼───────────────────▼────────┐
│                  Writer Node (:8001)                 │
│  /ingest → chunking → per-space embedding           │
│  SQLite: chunks + chunk_embeddings(space, blob)     │
│  /build  → per-space FAISS shards + id_maps         │
└────────────────────────┬────────────────────────────┘
                         │ snapshot
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐ ┌──────────────────────┐
│  Query Node A        │ │  Query Node B        │
│  ShardManagers:      │ │  ShardManagers:      │
│    text → FAISS      │ │    text → FAISS      │
│    image → FAISS     │ │    image → FAISS     │
│  HotfixBuffers:      │ │  HotfixBuffers:      │
│    text → FlatIP     │ │    text → FlatIP     │
│    image → FlatIP    │ │    image → FlatIP    │
└──────────────────────┘ └──────────────────────┘
```

## Key Concepts

### Embedding Space

An **embedding space** (or just "space") is a named index pipeline. Each space has:

- Its own embedding service configuration (URL, model, API format)
- Its own dimension (e.g. text=1024, image=768)
- Its own FAISS shards under `index/{space}/shard_*.faiss`
- Its own id\_map at `index/{space}/id_map.msgpack`
- Its own `HotfixBuffer` on the query node

All spaces share:
- The SQLite document/chunk store (FTS5 operates on text regardless of modality)
- The hierarchy/closure table
- The snapshot packaging and hot-swap mechanism

### Backward Compatibility

Text-only users see **zero changes**:
- The default space is `"text"`
- All API fields (`space`, `modality`) default to `"text"`
- Old manifests (v2 with `"index"` key) load transparently
- Old configs (flat `embedding:` block) work as a single `"text"` space
- Existing databases auto-migrate (schema v2 → v3)

## Configuration

### Single Space (default, backward compatible)

```yaml
embedding:
  service_url: http://localhost:8080/v1/embeddings
  api_format: openai
  model: text-embedding-large
  batch_size: 64
```

This is equivalent to:

```yaml
embedding:
  spaces:
    text:
      service_url: http://localhost:8080/v1/embeddings
      api_format: openai
      model: text-embedding-large
      batch_size: 64
```

### Multi-Space

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

Each space has its own `EmbeddingSpaceConfig` with independent URL, model, batch size, and timeout.

## Ingestion

Documents specify their modality via the `modality` field:

```json
{
  "documents": [
    {
      "doc_id": "article_001",
      "title": "Introduction to RAG",
      "text": "Retrieval-Augmented Generation combines...",
      "modality": "text"
    },
    {
      "doc_id": "diagram_001",
      "title": "RAG Architecture Diagram",
      "text": "Architecture diagram showing the RAG pipeline",
      "modality": "image",
      "content_ref": "https://example.com/rag_diagram.png"
    }
  ]
}
```

The writer routes each document's chunks to the appropriate space's embedding service. The `text` field is always stored and used for FTS regardless of modality — it can contain a caption or description for non-text content.

### Schema

Embeddings are stored in a separate table from chunks:

```sql
CREATE TABLE chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    space    TEXT NOT NULL DEFAULT 'text',
    embedding BLOB NOT NULL,
    PRIMARY KEY (chunk_id, space)
);
```

This allows a single chunk to have embeddings in multiple spaces (e.g., a document with both text and an embedded image).

## Search

### Single-Space Search (default)

All existing endpoints accept an optional `space` parameter:

```json
POST /search/text
{
  "query_text": "what is RAG?",
  "space": "text",
  "top_k": 10,
  "mode": "hybrid"
}
```

```json
POST /search
{
  "query_embedding": [0.1, 0.2, ...],
  "space": "image",
  "top_k": 10
}
```

### Cross-Space Search (late fusion)

The `/search/multi` endpoint searches multiple spaces and fuses results:

```json
POST /search/multi
{
  "queries": [
    {
      "space": "text",
      "query_embedding": [0.1, 0.2, ...],
      "query_text": "what is RAG?",
      "weight": 1.0
    },
    {
      "space": "image",
      "query_embedding": [0.5, 0.6, ...],
      "weight": 0.5
    }
  ],
  "top_k": 10,
  "fusion": "rrf"
}
```

**How fusion works:**
1. Each space is searched independently with its query embedding
2. Scores are multiplied by the per-query `weight`
3. Dense results from all spaces are merged
4. If the first `text`-space query has `query_text`, sparse (FTS) results are added
5. Dense + sparse results are fused via RRF (or returned as-is if only one type)

### API: List Available Spaces

```
GET /api/spaces
→ {"spaces": ["text", "image"]}
```

The WebUI uses this to populate the space selector dropdown.

## Snapshot Layout

Multi-space snapshots have per-space subdirectories:

```
v0001713800000/
├── manifest.json
├── index/
│   ├── text/
│   │   ├── shard_0.faiss.zst
│   │   ├── shard_1.faiss.zst
│   │   └── id_map.msgpack.zst
│   └── image/
│       ├── shard_0.faiss.zst
│       └── id_map.msgpack.zst
└── db/
    └── embedrag.db.zst
```

### Manifest v3

The manifest uses `indexes` (dict) and `id_maps` (dict) instead of singular `index` / `id_map`:

```json
{
  "manifest_version": 3,
  "indexes": {
    "text": {
      "type": "IVF256,PQ64",
      "dim": 1024,
      "num_shards": 4,
      "total_vectors": 50000,
      "shards": [...]
    },
    "image": {
      "type": "Flat",
      "dim": 768,
      "num_shards": 1,
      "total_vectors": 500,
      "shards": [...]
    }
  },
  "id_maps": {
    "text": {"file": "index/text/id_map.msgpack", ...},
    "image": {"file": "index/image/id_map.msgpack", ...}
  },
  "db": {"file": "db/embedrag.db", ...}
}
```

Old v2 manifests (with `"index"` key) are auto-detected and loaded as a single `"text"` space.

## WebUI

The Search and Debug tabs include a **space selector** dropdown populated from `GET /api/spaces`. For single-space deployments, it shows only "text". For multi-space, all available spaces appear.

## Design Decisions

### Why separate spaces, not a unified embedding?

- Different modalities use different models with different dimensions (text=1024, CLIP=768, audio=512)
- CLIP/SigLIP models that unify text+image work naturally by configuring both spaces to use the same model
- Separate spaces give full flexibility without constraining model choice

### Why not a separate EmbedRAG instance per modality?

- Wastes the shared metadata (documents, chunks, hierarchy, FTS)
- No way to do cross-modal fusion in a single query
- Operational overhead of N deployments

### Why late fusion, not early fusion?

- Late fusion works with any combination of modalities
- No need for a shared embedding space or dimension alignment
- Simple to reason about: each space returns ranked results, then they're merged
- RRF is proven effective for combining heterogeneous rankers
