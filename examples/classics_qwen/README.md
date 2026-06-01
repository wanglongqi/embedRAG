# Example: Classics Qwen (论语 + 庄子)

This example is a variation of the `classics` example, optimized for use with **LM Studio** and the **text-embedding-qwen3-vl-embedding-2b** model. It also uses more granular chunking better suited for classical Chinese texts.

## Key Differences from Base Classics Example

1.  **Model**: Uses `text-embedding-qwen3-vl-embedding-2b` (served via LM Studio).
2.  **Chunking**: Uses a smaller `chunk_size` (250 tokens/characters instead of 512) to ensure segments are more focused and suitable for short classical passages.
3.  **Local Dev**: Configured to work with local LM Studio endpoints by default (`http://127.0.0.1:1234/v1/embeddings`).

## Quick Start

### 1. Prerequisites

*   **LM Studio**: Install and run LM Studio.
*   **Model**: Download and load `text-embedding-qwen3-vl-embedding-2b` in LM Studio.
*   **Server**: Start the Local Server in LM Studio (usually port 1234).

### 2. Build the Index

```bash
# 1. Start the writer node
uv run embedrag writer --config examples/classics_qwen/writer.yaml

# 2. Ingest data (will connect to LM Studio at port 1234)
# Make sure the data/ directory at project root contains lunyu/ and zhuangzi/
uv run python examples/classics_qwen/ingest.py

# 3. Copy the built snapshot to the query directory
# (Replace v1234567890 with the actual version number printed by ingest.py)
mkdir -p examples/classics_qwen/snapshot
cp -r /tmp/embedrag-classics-qwen/v*/ examples/classics_qwen/snapshot/
```

### 3. Query the Index

```bash
# 1. Start the query node
uv run embedrag query --config examples/classics_qwen/query.yaml

# 2. Search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{"query_text": "仁者爱人", "top_k": 5, "mode": "hybrid"}'
```

## Files

*   `writer.yaml`: Configured for LM Studio and Qwen3-VL embedding.
*   `query.yaml`: Configured for the same model and local snapshot path.
*   `ingest.py`: Specialized ingest script with `chunk_size=250`.
