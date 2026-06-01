# Clustering Module — Design Document

## 1. Goals

The clustering module (`embedrag.cluster`) exists to solve a recurring need in
RAG workflows: grouping embedded texts into interpretable themes. It is designed
as a **standalone, reusable library** that does not depend on FastAPI, Pydantic,
or any web framework. The same code powers three surfaces:

1. **CLI** (`embedrag cluster`) — file-based, no server needed.
2. **Python library** — import and call directly from any script or notebook.
3. **Query-node HTTP API** — clusters the loaded corpus in-place and serves an
   interactive `/cluster/` page.

No component deeper than the HTTP layer imports from `embedrag.config` or
`embedrag.models`. The core types are plain dataclasses, not Pydantic models.

---

## 2. Module Map

```
embedrag.cluster
├── __init__.py       # Public API re-exports + cluster_items convenience
├── types.py          # Dataclasses: Item, ClusterInfo, ClusterMember, ClusterResult
├── preprocess.py     # L2 normalization, PCA/UMAP dimensionality reduction
├── algorithms.py     # Pluggable backends (HDBSCAN, DBSCAN, KMeans, Agglomerative, Leiden)
├── evaluate.py       # Internal metrics, external metrics, auto parameter sweep
├── explain.py        # c-TF-IDF keywords, medoids, cohesion, separation, attribution
├── label.py          # Keyword labels + optional LLM-generated topic names
├── pipeline.py       # End-to-end orchestration (public cluster_vectors)
├── source.py         # Input loaders: jsonl/csv/npy file, writer DB, TF-IDF vectorizer
├── store.py          # Side-file persistence (data_dir/cluster_runs/<run_id>.json)
├── visualize.py      # Data-only viz specs consumed by Plotly.js front-end
└── report.py         # Self-contained interactive HTML report (plotly.js CDN)

embedrag.clusterui
├── index.html        # Plotly.js-powered cluster exploration page
├── cluster.css       # Dark-themed styling matching the search UI
└── cluster.js        # Front-end logic: run, load, render, drill-down
```

The module count is 11 .py files + 3 front-end files. Total ~1500 lines of
Python, ~385 lines of JS/CSS/HTML.

---

## 3. Pipeline Architecture

```
vectorize ──> reduce ──> cluster ──> evaluate ──> explain ──> label ──> visualize
    │            │           │            │            │           │           │
    │            │           │            │            │           │           └── build_panels()
    │            │           │            │            │           └── apply_keyword_labels()
    │            │           │            │            └── explain.explain()
    │            │           │            └── select_params() → backend.fit()
    │            │           └── make_backend(name)
    │            └── reduce_dims() → (reduced, method_used)
    └── source.py / TF-IDF / embed service
```

All stages are synchronous and pure (no IO except the optional LLM labeling
step, which is async and layered on top by the caller).

### 3.1 Design decisions per stage

**Vectorize** — Three paths exist:
- *Precomputed vectors*: caller passes `np.ndarray` directly (`cluster_vectors`).
- *File loaders*: `load_items_from_file()` returns `(items, vectors_or_None)`
  from jsonl/csv/json with optional inline embeddings, or `load_vectors_npy()`
- *TF-IDF fallback*: `tfidf_vectors()` uses char-n-gram TfidfVectorizer (2–3
  grams) for CJK without word segmentation, word+bigram for Latin scripts.

**Reduce** — Optional dimensionality reduction before clustering.
- `"none"`: pass through.
- `"pca"`: always available (sklearn.decomposition.PCA).
- `"umap"`: used when installed and n ≤ 50K; otherwise PCA fallback.
- `"auto"`: UMAP if available + n ≤ 50K, else PCA if d > 50, else none.
- Rationale: high-dim embeddings (>768) cause density-based methods to mark
  nearly all points as noise. UMAP typically turns this into clean clusters.

**Cluster** — Pluggable backends registered in `_BACKENDS` dict:
- `hdbscan` (default): density-based, auto cluster count, native noise flag,
  membership probabilities. Auto-calculates `min_cluster_size` = `max(5, √n/2)`.
- `dbscan`: density-based, explicit `eps` estimated via 90th percentile of the
  k-distance curve.
- `kmeans`: centroid-based, uses MiniBatch for n > 20K, FAISS Kmeans for n > 200K.
- `agglomerative`: hierarchical (Ward), with optional dendrogram export when
  n ≤ 4000.
- `leiden`: community detection on a FAISS kNN graph (requires `igraph` +
  `leidenalg`).
- `auto`: runs both HDBSCAN and KMeans with parameter sweeps, picks the best
  by composite score.

**Evaluate** — Quality metrics computed on every run:
- *Internal*: silhouette, Davies–Bouldin, Calinski–Harabasz (sampled at 5K
  points for large sets), noise ratio. No ground truth required.
- *External*: ARI, NMI, V-measure — only when `ground_truth` is passed.
- *Composite score*: `silhouette − 0.5 × noise_ratio` — used by the auto param
  sweep to select the best configuration without supervision.
- *Sweep*: `select_params()` enumerates a grid over the controlling parameter
  (e.g. `min_cluster_size` for HDBSCAN, `k` for KMeans) and records the score
  curve. The caller can inspect `result.sweep`.

**Explain** — Interpretability layer:
- c-TF-IDF: each cluster is treated as a single document; TF-IDF against the
  corpus of clusters yields distinctive keywords.
- Medoid selection: points nearest the centroid (by cosine) serve as
  representative texts.
- Cohesion: mean cosine of members to the centroid (within-cluster tightness).
- Separation: cosine of the centroid to its nearest other centroid.
- Per-member attribution: `self_similarity` (cosine to own centroid) and
  `runner_up_similarity` (best other centroid), so each item can answer "why
  am I in this cluster".

**Label** — Two modes:
- *Keyword labels*: free (derived from top-3 c-TF-IDF terms).
- *LLM labels*: async function calls an OpenAI-compatible chat endpoint with a
  structured prompt, produces `{"label": "≤6 word topic", "summary": "..."}`.
  Falls back to keyword label on failure.

**Visualize** — Data-only JSON specs (no chart-drawing in Python):
- 2D projection (UMAP or PCA, subsampled at 8K).
- Per-algorithm panels declared in `ALGORITHM_PANELS`: scatter, size_bar,
  probability_hist, silhouette, k_curve, similarity_matrix, dendrogram.
- These specs are consumed by the Plotly.js front-end (`/cluster/` page) or
  the standalone HTML report (`--viz`).

---

## 4. Data Types

```python
@dataclass
class Item:
    id: str
    text: str = ""

@dataclass
class ClusterMember:
    id: str
    text: str
    cluster_id: int
    probability: float          # membership confidence
    self_similarity: float      # cosine to own centroid
    runner_up_cluster: int      # best other cluster
    runner_up_similarity: float # similarity to runner-up

@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    label: str                  # human-readable name
    keywords: list[str]
    summary: str
    cohesion: float
    separation: float
    representatives: list[str]  # medoid item ids
    representative_texts: list[str]
    member_ids: list[str]
    centroid: list[float]

@dataclass
class ClusterResult:
    run_id: str
    algorithm: str
    params: dict
    space: str
    created_at: str
    source: str
    n_items: int
    n_clusters: int
    noise_count: int
    clusters: list[ClusterInfo]
    members: list[ClusterMember]
    metrics: dict              # internal + optional external
    sweep: list[dict]          # score curve for auto param search
    projection: dict           # 2D coords for scatter
    viz: list[dict]            # per-algorithm panel specs
```

All dataclasses, no Pydantic — keeps the core importable with zero web deps.
`to_dict()` / `from_dict()` handle numpy→native conversion for JSON.

---

## 5. API Surfaces

### 5.1 Python Library

```python
result = cluster_vectors(vectors, items, algorithm="auto", reduce="auto",
                         auto=True, params=None, ...)
result = cluster_items(texts, algorithm="kmeans", ...)
asyncio.run(apply_llm_labels(result, chat_url="...", model="..."))
```

`cluster_items` is a convenience wrapper that optionally vectorizes via TF-IDF
then calls `cluster_vectors`. The core is always `cluster_vectors`.

### 5.2 CLI

```
embedrag cluster --input file.jsonl [--algorithm hdbscan] [--reduce umap]
                 [-o run.json] [--viz report.html]
       cluster --db writer.db [--filter doc_type=X] [--k 12]
       cluster --embeddings vectors.npy [--input items.jsonl]
```

The CLI handles vector acquisition (file, DB, or embed service), calls
`cluster_vectors`, optionally labels with LLM, writes JSON and/or HTML.

### 5.3 HTTP API

Mounted on the query-node `APIRouter` in `embedrag.query.routes`:

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/cluster` | Cluster loaded corpus → persist run |
| GET | `/api/clusters` | List run summaries |
| GET | `/api/clusters/{run_id}` | Fetch full run |
| DELETE | `/api/clusters/{run_id}` | Delete run |
| GET | `/api/clusters/{run_id}/members` | Page members of a cluster |
| POST | `/api/clusters/{run_id}/search` | Centroid-based search |

Cluster-aware search: `POST /search/text` accepts `cluster_run_id` +
`cluster_id` to restrict results to a single cluster.

### 5.4 Front-end

`/cluster/` serves a Plotly.js page (separate from the lightweight `/ui/`
search page). Three panels:
- 2-D projection scatter (scattergl for >2K points).
- Cluster size bar chart.
- Algorithm-specific auxiliary (K-curve / silhouette / similarity matrix).
- Drill-down drawer: click a cluster row to page through members.

---

## 6. Integration Points

### 6.1 Query node app (`embedrag.query.app`)

Two mount points in `create_query_app()`:
```python
app.mount("/ui", StaticFiles(directory="webui"), name="webui")
app.mount("/cluster", StaticFiles(directory="clusterui"), name="clusterui")
```

### 6.2 Query node routes (`embedrag.query.routes`)

- `run_cluster()`: acquires the generation context, calls
  `_source_generation_vectors()` to reconstruct FAISS vectors + fetch texts,
  then runs `cluster_vectors` in a thread (to avoid blocking the event loop).
- `_source_generation_vectors()`: reconstructs exact vectors from FAISS via
  `ShardManager.reconstruct_all()`, applies filters and max_items sampling.
- `_cluster_member_filter()`: loads a persisted run, builds a `set[str]` of
  chunk ids for the requested cluster, passed to `_run_search_pipeline()`.
- The cluster endpoints use `asyncio.to_thread()` for CPU-bound calls
  (`cluster_vectors`, `list_runs`, `load_run`).

### 6.3 CLI entry point (`embedrag.cli._run_cluster`)

Resolves the input source (file, DB, or npy), optionally embeds text via
`EmbeddingClient` or falls back to TF-IDF, calls `cluster_vectors`, optionally
labels with LLM, writes JSON and/or HTML report.

---

## 7. Persistence Model

Cluster runs are stored as side files—independent of the snapshot DB schema:

```
<data_dir>/cluster_runs/<run_id>.json
```

Each file contains the full `ClusterResult` serialized via `to_dict()`. Schema
is versioned only by the dataclass fields (no formal migration needed — fields
can be added with defaults).

This design was chosen deliberately: clustering is an exploratory, orthogonal
concern. Runs are created, listed, read, and deleted independently of the index
lifecycle. A CLI output (`-o run.json`) can be copied into `cluster_runs/` to
appear in the web UI with zero server changes.

---

## 8. Review Comments

The following findings were identified during a systematic review of the
clustering module (May 2026).

### 8.1 Issues found and fixed

| # | Severity | File | Issue | Fix |
|---|----------|------|-------|-----|
| 1 | High | `mkdocs.yml` | Placeholder `repo_url: your-username` | Fixed to `wanglongqi` |
| 2 | High | `docs/embedding.md` | 5 source-relative links (`../src/embedrag/...`) that break strict mkdocs build | Converted to plain text refs |
| 3 | High | `docs/embedding.md` | Broken anchor link `#b4-sidecar--initcontainer-...` (double dash, slug mismatch) | Corrected to single dash |
| 4 | High | `docs/clustering.md` | Placeholder GitHub URL `your-username` | Fixed to `wanglongqi` |
| 5 | Medium | `src/embedrag/__init__.py` | `__version__ = "0.4.0"` out of sync with `pyproject.toml` `0.6.0` | Synced to `0.6.0` |
| 6 | Medium | `pyproject.toml` | `[dependency-groups.dev]` missing pytest deps – `uv sync --group dev` would fail test runs | Added missing deps |
| 7 | Low | `.readthedocs.yaml` | `extra_requirements` missing `cluster` – UMAP/scipy not installed on RTD | Added `cluster` extra |

### 8.2 Observations (not fixed)

| # | Observation | Module | Detail |
|---|-------------|--------|--------|
| A | Deprecated `logger.warn()` calls | 20+ locations across `cluster/`, `query/`, `writer/`, `shared/` | `logger.warn()` has been deprecated in favour of `logger.warning()` since Python 3.3. The structlog library still supports `warn()` as an alias, so this is cosmetic. |
| B | `select_params()` imports sklearn inside the hot loop | `cluster/evaluate.py:119` | `make_backend()` is called per sweep iteration, which triggers sklearn imports. These are cached by Python after the first import, so the runtime cost is negligible after the first iteration. |
| C | `AgglomerativeBackend` computes linkage matrix in-memory | `cluster/algorithms.py:146-152` | The full linkage matrix for n ≤ 4000 is `O(n²)` memory. At 4000 points this is ~128MB for the 3×(n−1) float64 matrix — acceptable but worth noting for memory-constrained deployments. |
| D | HDBSCAN `min_cluster_size` auto-heuristic | `cluster/algorithms.py:63` | The heuristic `max(5, √n/2)` works well for n ≥ 100 but over-estimates for very small sets (n < 20). The `max(2, min(...))` clamp prevents outright breakage. |
| E | No streaming mode | Whole module | The entire pipeline is in-memory. For datasets >500K items, the projection step (UMAP on full data) would need subsampling or a minibatch approach. The `_MAX_SCATTER_POINTS=8000` limit already handles the projection side. |
| F | LLM labeling uses one HTTP request per cluster | `cluster/label.py:53-76` | No batching. For 50+ clusters this adds latency linearly. Each request has a timeout of 60s, so slow models will compound the delay. The `timeout_seconds` parameter is configurable. |
| G | `api.md` misses 5 cluster submodules | `docs/api.md` | `preprocess.py`, `label.py`, `visualize.py`, `report.py`, `types.py` are not listed under the Clustering section in the API reference docstring. They are still documented via their own docstrings but won't appear in the generated mkdocs output. |
| H | No integration tests for cluster HTTP endpoints | `tests/` | The 18 unit tests cover the core pipeline, serialization, file loading, and store, but there are no tests for the `POST /api/cluster` or `GET /api/clusters` HTTP handlers. Mocking the gen_manager would be sufficient. |

### 8.3 Verification state

All checks pass with zero warnings:

- **ruff**: All checks passed on `src/`
- **mypy**: No issues in 65 source files
- **pytest**: 152/152 passed
- **mkdocs --strict**: Builds with zero warnings

---

## 9. Dependencies

### Core (no extra install)

| Dependency | Used by | Notes |
|------------|---------|-------|
| numpy | All | Vector math |
| scikit-learn | algorithms, evaluate, explain, source | HDBSCAN, DBSCAN, KMeans, Agglomerative, PCA, TfidfVectorizer, metrics |

### Optional (`[cluster]` extra)

| Dependency | Used by | Notes |
|------------|---------|-------|
| umap-learn | preprocess | UMAP reduction |
| scipy.cluster.hierarchy | algorithms | Agglomerative dendrogram |

### Optional (not in extras)

| Dependency | Used by | Notes |
|------------|---------|-------|
| igraph + leidenalg | algorithms | Leiden backend |
| aiohttp | label | LLM labeling |
| faiss | algorithms | KMeans for n > 200K, kNN graph for Leiden |

---

## 10. Future Considerations

1. **Online / streaming clustering**: The current pipeline is batch-only. For
   continuously ingested corpora, an incremental variant (e.g. online FAISS
   k-means or HDBSCAN's approximate_predict) would avoid re-clustering from
   scratch.

2. **Cluster persistence in the snapshot**: Side files are intentionally
   decoupled from the snapshot. If cluster runs become a first-class artifact
   (e.g. referenced across restarts), saving them under `active/vN/clusters/`
   alongside `manifest.json` would tie them to a specific generation.

3. **GPU acceleration**: FAISS KMeans already runs on GPU when `faiss.StandardGpuResources` is configured. The sklearn backends remain CPU-only.

4. **Multi-space clustering**: The current `cluster` endpoint accepts a `space`
   parameter but clusters only one space at a time. Cross-space clustering
   (aligning clusters from text + image embeddings) is future work.

5. **LLM label batching**: Grouping cluster prompts into a single batch request
   would reduce HTTP overhead for runs with many clusters. The prompt format
   would need to support multi-cluster responses.
