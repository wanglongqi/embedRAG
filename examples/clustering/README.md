# Clustering / Grouping Example

Group a set of inputs (customer complaints, review comments, ...) into a small
number of explainable categories after embedding. The clustering module works
in two co-equal modes:

1. Standalone — a `.jsonl`/`.csv`/`.npy` file, no embedRAG snapshot needed.
2. Integrated — over a loaded query-node generation, with results consumable
   through the index itself.

It is honest about quality (every run reports metrics), picks parameters
automatically (with a transparent sweep you can override), explains each
cluster (distinctive keywords + representative examples), and ships per-algorithm
visualizations.

## 1. Standalone CLI

The bundled `complaints.jsonl` has 30 Chinese complaints spanning shipping,
refunds, login/account, customer-service, and discount/pricing issues.

No embedding service required (falls back to a local TF-IDF representation):

```bash
uv run embedrag cluster \
  --input examples/clustering/complaints.jsonl \
  --algorithm auto \
  -o /tmp/complaints.result.json \
  --viz /tmp/complaints.html
```

Open `/tmp/complaints.html` in a browser for an interactive scatter, cluster
sizes, and a per-cluster keyword/example table.

### Use a real embedding service

To embed text with an OpenAI-compatible endpoint (the same one used elsewhere
in the examples), pass `--embed-url`:

```bash
uv run embedrag cluster \
  --input examples/clustering/complaints.jsonl \
  --embed-url http://127.0.0.1:1234/v1/embeddings \
  --embed-model text-embedding-qwen3-embedding-0.6b \
  --algorithm hdbscan \
  --reduce umap \
  -o /tmp/complaints.result.json --viz /tmp/complaints.html
```

### Auto-label clusters with an LLM (optional)

```bash
uv run embedrag cluster \
  --input examples/clustering/complaints.jsonl \
  --llm-url http://127.0.0.1:1234/v1/chat/completions \
  --llm-model your-chat-model \
  --llm-language zh
```

Without `--llm-*`, clusters are labeled from c-TF-IDF keywords (no external call).

### Precomputed embeddings

If you already have vectors, pass a `.npy` aligned with the rows:

```bash
uv run embedrag cluster --input items.jsonl --embeddings vecs.npy --algorithm kmeans --k 5
```

## 2. Library API

```python
from embedrag.cluster import cluster_items, cluster_vectors, Item

# texts -> TF-IDF -> clusters
result = cluster_items(["...", "..."], algorithm="auto")

# precomputed vectors
result = cluster_vectors(vectors, [Item(id="1", text="...")], algorithm="kmeans", params={"k": 5})

print(result.algorithm, result.n_clusters, result.metrics["silhouette"])
for c in result.clusters:
    print(c.cluster_id, c.label, c.keywords[:5], c.size)
```

## 3. Integrated with the vector store (query node)

With a query node running, cluster the loaded corpus directly (vectors are
reconstructed from the FAISS index; texts come from the snapshot DB):

```bash
# run a clustering job (persists a run under <data_dir>/cluster_runs/)
curl -s -X POST localhost:8000/api/cluster \
  -H 'Content-Type: application/json' \
  -d '{"algorithm":"auto","reduce":"pca","filters":{"doc_type":"complaint"}}' | jq .run_id

# list runs / fetch a run / page members
curl -s localhost:8000/api/clusters | jq .
curl -s localhost:8000/api/clusters/<run_id> | jq '.clusters[] | {cluster_id,label,size}'
curl -s "localhost:8000/api/clusters/<run_id>/members?cluster_id=0&limit=20" | jq .

# search WITHIN a cluster
curl -s -X POST localhost:8000/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query_text":"退款","cluster_run_id":"<run_id>","cluster_id":0}' | jq .

# use a cluster centroid as a query (find items near the cluster)
curl -s -X POST localhost:8000/api/clusters/<run_id>/search \
  -H 'Content-Type: application/json' -d '{"cluster_id":0,"top_k":10}' | jq .
```

### Interactive web view

Open `http://localhost:8000/cluster/` for the dedicated, plotly-powered page:
pick an algorithm, run clustering, explore the 2D projection, click a cluster to
inspect its members, and search by centroid. (Linked from the main UI nav.)

## Notes

- Cluster runs are stored as side files in `<data_dir>/cluster_runs/<run_id>.json`;
  no snapshot DB schema change is involved.
- The FAISS index may be `IVF,PQ` (lossy) at scale; that only affects approximate
  search. Offline clustering over a *writer* DB reads exact vectors from
  `chunk_embeddings` (`--db path/to/writer.db`).
