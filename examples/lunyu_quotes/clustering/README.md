# Example: Clustering the 论语 quotes

This walks through grouping the **501 论语 quotes** from the
[`examples/lunyu_quotes`](../) snapshot into semantic clusters, end to end. It
exercises the standalone CLI, the persisted side-file store, and the live
`/cluster` web UI — all on the **real 1024-dim `text-embedding-qwen3-embedding-0.6b`
embeddings** that already live in the prebuilt index.

> Prereqs: a built snapshot under `../snapshot/active/v*` (see
> [`../README.md`](../README.md)) and the optional clustering extra:
> `uv pip install -e ".[cluster]"` (pulls `umap-learn` + `scipy`). UMAP needs
> `numpy<2`.

## Why an export step?

The query-node snapshot ships the FAISS index but **not** the
`chunk_embeddings` table (the query export drops it). So instead of re-embedding,
[`export_vectors.py`](export_vectors.py) reconstructs the **exact** vectors
straight from the `Flat` index (exact for `Flat` / `IVF,Flat`) and pairs them
with the quote texts from the snapshot DB:

```bash
uv run python examples/lunyu_quotes/clustering/export_vectors.py
# -> quotes.jsonl  (id, text, chapter)
# -> quotes.npy    (501, 1024) float32, row-aligned
```

## 1. Standalone CLI

Run the real `embedrag cluster` command on the exported files. The recommended
recipe for these dense, short, classical-Chinese embeddings is **UMAP reduction
→ HDBSCAN** (see "Why UMAP" below):

```bash
uv run embedrag cluster \
  --input      examples/lunyu_quotes/clustering/quotes.jsonl \
  --embeddings examples/lunyu_quotes/clustering/quotes.npy \
  --reduce umap --algorithm hdbscan \
  -o   examples/lunyu_quotes/clustering/results/run.json \
  --viz examples/lunyu_quotes/clustering/results/report.html
```

Result on this dataset (11 clusters, silhouette ≈ 0.59, Davies–Bouldin ≈ 0.59):

| cluster | label (keywords) | size |
|---|---|---|
| 君子/小人 | `君子 小人` | 25 |
| 子貢 | `子貢 貢曰` | 29 |
| 子路 | `子路` | 22 |
| 子夏 | `子夏 夏曰` | 12 |
| 孔子 (third-person) | `孔子 對曰` | 20 |
| 有道/無道 | `道 邦有道 邦無道` | 21 |
| 問孝/父母 | `問孝 父母` | 14 |
| 禮/樂 | `禮樂` | 13 |
| 仁者 | `仁者 仁` | 23 |

The embeddings group quotes both **by disciple** (子貢, 子路, 子夏, 顏淵, 曾子)
and **by theme** (仁, 禮樂, 孝, 為政, 有道/無道) — semantics the keyword labels
only partially capture.

Other configs to compare:

```bash
# Broad themes: K-Means with a fixed k
uv run embedrag cluster --input .../quotes.jsonl --embeddings .../quotes.npy \
  --algorithm kmeans --reduce pca --k 8

# Let the auto sweep pick the algorithm + params
uv run embedrag cluster --input .../quotes.jsonl --embeddings .../quotes.npy
```

### No embeddings? It still works

Drop `--embeddings` and the CLI falls back to a char-n-gram **TF-IDF**
representation (no embedding service needed) — handy as a cheap baseline:

```bash
uv run embedrag cluster --input examples/lunyu_quotes/clustering/quotes.jsonl \
  --algorithm hdbscan --reduce umap
```

## 2. Persist a run for the web UI

The query node lists/serves runs from `<data_dir>/cluster_runs/<run_id>.json`.
[`install_run.py`](install_run.py) drops a CLI result into that store (here
`data_dir` is `../snapshot`, per `query.yaml`):

```bash
uv run python examples/lunyu_quotes/clustering/install_run.py \
  examples/lunyu_quotes/clustering/results/run.json
```

Then start the node and browse it:

```bash
uv run embedrag query --config examples/lunyu_quotes/query.yaml
open http://localhost:8000/cluster/
```

## 3. Fully-integrated: cluster from the running node

No export and no CLI — the query node reconstructs vectors from its own FAISS
index, clusters, and persists, all server-side. This is what the `/cluster`
web UI buttons call:

```bash
curl -s -X POST http://localhost:8000/api/cluster \
  -H "Content-Type: application/json" \
  -d '{"algorithm":"hdbscan","reduce":"umap","auto":false,"persist":true}'

curl -s http://localhost:8000/api/clusters            # list persisted runs
```

Search can then be scoped to a cluster by passing `cluster_run_id` + `cluster_id`
to `/search` or `/search/text`.

## Why UMAP → HDBSCAN

On the **raw 1024-dim** vectors, HDBSCAN labels ~86% of points as noise (the
curse of dimensionality). Adding UMAP first flips it:

| config | clusters | noise | silhouette | Davies–Bouldin |
|---|---|---|---|---|
| kmeans k=8 (pca) | 8 | 0% | 0.051 | 3.27 |
| auto → kmeans k=23 | 23 | 0% | 0.059 | 2.75 |
| hdbscan on raw 1024-dim | 3 | **86%** | 0.122 | 1.87 |
| **umap → hdbscan** | 11 | 55% | **0.586** | **0.59** |

The evaluation layer surfaces this honestly (noise ratio, silhouette), and the
preprocessing stage is what fixes it.

## Notes

- Keyword labels lean on a c-TF-IDF char-n-gram fallback, so ubiquitous tokens
  (`子曰`, `：「`, `」`) leak in. For clean names add an LLM pass:
  `--llm-url <openai-compatible-chat-url> --llm-model <name>`.
- `quotes.jsonl`, `quotes.npy`, and `results/` are git-ignored — regenerate them
  with `export_vectors.py` + the CLI.

## Files

- `export_vectors.py` — reconstruct exact vectors + texts from the snapshot
- `install_run.py` — copy a CLI run into the node's `cluster_runs/` store
- `.gitignore` — ignores the regenerable artifacts
