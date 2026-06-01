# Clustering & Topic Discovery

Grouping a set of texts into themes after embedding them is a recurring task:
clustering customer complaints, triaging review comments, deduplicating FAQs,
or just exploring what's in a corpus. EmbedRAG ships a standalone, reusable
clustering module (`embedrag.cluster`) that works three ways:

1. **A CLI** (`embedrag cluster`) over `.jsonl` / `.csv` / `.npy` files or a
   writer DB.
2. **A Python library** you can import and call directly.
3. **An integrated query-node feature** that clusters the loaded corpus and
   serves an interactive `/cluster/` visualization page.

All three share one modular pipeline.

## The pipeline

```
vectorize → reduce (optional) → cluster → evaluate → explain → label → visualize
```

| Stage | What it does | Options |
|-------|--------------|---------|
| **vectorize** | Turn inputs into vectors | inline embeddings, embedding service, writer DB, FAISS reconstruction, or char-n-gram **TF-IDF** fallback |
| **reduce** | Optional dimensionality reduction | `none` / `pca` / `umap` / `auto` |
| **cluster** | Pluggable backends | `hdbscan`, `kmeans`, `agglomerative`, `dbscan`, optional `leiden`, or `auto` |
| **evaluate** | Internal + external metrics | silhouette, Davies–Bouldin, Calinski–Harabasz, noise ratio; ARI/NMI/V-measure when ground truth is given |
| **explain** | Make clusters interpretable | distinctive c-TF-IDF keywords, medoids, cohesion, separation, per-point attribution |
| **label** | Name each cluster | keyword label by default; optional LLM summary |
| **visualize** | Build view specs | 2D projection scatter, size bars, K-curve / silhouette / similarity matrix |

When you don't pin an algorithm or its parameters, an automatic sweep selects
them using a composite of the internal metrics, so the defaults are sensible
without manual tuning. Pin them whenever you want full control.

## Installation

The core path (TF-IDF + scikit-learn algorithms) needs no extra install.
UMAP-based reduction and the Leiden graph-clustering backend live behind an optional extra:

```bash
uv pip install -e ".[cluster]"   # umap-learn, scipy, igraph, leidenalg
```

## CLI

```bash
embedrag cluster [INPUT SOURCE] [CLUSTERING OPTIONS] [-o run.json] [--viz report.html]
```

Choose exactly one input source:

| Flag | Source |
|------|--------|
| `--input file.jsonl` | `.jsonl` / `.json` / `.csv` of `{id, text, [embedding]}` |
| `--embeddings file.npy` | A precomputed `(N, D)` matrix (pair with `--input` to attach text) |
| `--db writer.db` | Exact vectors from a writer DB's `chunk_embeddings` (with `--filter key=value`) |

If no vectors are present, supply `--embed-url` (OpenAI-compatible) to embed the
text, or let it fall back to a local **TF-IDF** representation (char n-grams, so
it works for CJK without word segmentation).

### Examples

```bash
# Embed text via a service, reduce with UMAP, cluster with HDBSCAN
embedrag cluster --input complaints.jsonl \
  --embed-url http://localhost:1234/v1/embeddings --embed-model my-embed \
  --reduce umap --algorithm hdbscan \
  -o run.json --viz report.html

# No embeddings → local TF-IDF baseline (DBSCAN-style works with raw features)
embedrag cluster --input complaints.jsonl --algorithm hdbscan --reduce umap

# Cluster a filtered slice of a writer DB with exact vectors
embedrag cluster --db /data/embedrag-writer/db/writer.db --filter doc_type=complaint --k 12

# Add natural-language cluster names from an LLM
embedrag cluster --input complaints.jsonl --embed-url … \
  --llm-url http://localhost:1234/v1/chat/completions --llm-model my-chat
```

Key options: `--algorithm` (`auto`/`hdbscan`/`kmeans`/`agglomerative`/`dbscan`/`leiden`),
`--reduce` (`auto`/`none`/`pca`/`umap`), `--no-auto` (disable the sweep),
`--min-cluster-size` / `--k` / `--eps` / `--min-samples` / `--cluster-selection-method` / `--distance-threshold` / `--linkage` / `--knn` / `--resolution` (per-algorithm params), and
`--llm-url` / `--llm-model` for labeling.

## Library API

```python
from embedrag.cluster import cluster_items, cluster_vectors, apply_llm_labels

# Text in, clusters out (embeds or TF-IDF internally)
result = cluster_items(["...", "..."], algorithm="auto")

# Bring your own vectors
result = cluster_vectors(vectors, items, algorithm="hdbscan", reduce="umap")

for c in result.clusters:
    print(c.cluster_id, c.label, c.size, c.keywords[:5])

# Optional: attach LLM-generated names (async)
import asyncio
asyncio.run(apply_llm_labels(result, chat_url="…", model="…"))
```

`cluster_vectors` / `cluster_items` are synchronous and dependency-light, so the
core is easy to unit-test and embed in other pipelines.

## Integrated with the vector store

The query node can cluster **its own loaded corpus** without re-embedding: it
reconstructs vectors directly from the FAISS index (exact for `Flat` /
`IVF,Flat`, approximate for `IVF,PQ`). This is what the cluster API and the
`/cluster/` web page call.

### Persistence (side files)

Runs are stored as standalone JSON files under
`<data_dir>/cluster_runs/<run_id>.json` — **no snapshot DB schema change**. They
can be created, listed, read, and deleted independently of the index lifecycle.
A run produced by the CLI (`-o run.json`) uses the same format, so you can copy
it into `cluster_runs/` to make it appear in the web UI.

### Cluster API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/cluster` | Cluster the loaded corpus (or a `filters` subset) and persist the run |
| `GET` | `/api/clusters` | List persisted run summaries |
| `GET` | `/api/clusters/{run_id}` | Fetch a full run (projection + members) |
| `DELETE` | `/api/clusters/{run_id}` | Delete a run |
| `GET` | `/api/clusters/{run_id}/members` | List a cluster's members |
| `POST` | `/api/clusters/{run_id}/search` | Search using a cluster centroid as the query |

```bash
curl -X POST http://localhost:8000/api/cluster \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"hdbscan","reduce":"umap","auto":false,"persist":true}'
```

### Cluster-aware search

Scope any search to a single cluster by passing the run and cluster id:

```json
POST /search/text
{"query_text": "refund delay", "cluster_run_id": "run-…", "cluster_id": 3}
```

### Visualization

The `/cluster/` page (separate from the zero-dependency `/ui/`) uses Plotly.js
to render a 2D projection scatter, cluster-size bars, and an algorithm-specific
auxiliary chart (K-curve, silhouette, or inter-cluster similarity matrix), plus
a table with a per-cluster member drawer.

## Choosing an approach

- **High-dimensional embeddings + HDBSCAN**: reduce with UMAP first. On raw
  high-dim vectors, density clustering tends to mark most points as noise; UMAP
  typically turns that into clean, well-separated clusters. The reported noise
  ratio and silhouette make this trade-off visible.
- **Known number of groups**: use `kmeans` (or `agglomerative`) with `--k`.
- **No embedding service**: the TF-IDF fallback is a fast baseline; expect
  surface-form grouping rather than deep semantics.

See the [论语 clustering walkthrough](https://github.com/wanglongqi/embedRAG/tree/main/examples/lunyu_quotes/clustering)
for an end-to-end example with real numbers.
