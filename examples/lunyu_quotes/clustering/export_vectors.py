"""Export the 论语 quote embeddings + texts from the prebuilt snapshot.

The query-node snapshot under ``examples/lunyu_quotes/snapshot/active`` ships a
FAISS index but *not* the ``chunk_embeddings`` table (the query export drops it).
For the standalone ``embedrag cluster`` CLI path we therefore reconstruct the
exact vectors straight from the ``Flat`` FAISS index and pair them with the
quote texts pulled from the snapshot DB.

Outputs (written next to this script, both git-ignored):

- ``quotes.jsonl``  -- one ``{"id", "text", "chapter"}`` row per quote
- ``quotes.npy``    -- a ``(N, 1024)`` float32 matrix aligned row-for-row

Reconstruction is exact for ``Flat`` / ``IVF,Flat`` indexes (this example uses
``Flat``). Run it once, then feed the two files to the CLI:

    uv run embedrag cluster \
        --input  examples/lunyu_quotes/clustering/quotes.jsonl \
        --embeddings examples/lunyu_quotes/clustering/quotes.npy \
        --reduce umap --algorithm hdbscan \
        -o   examples/lunyu_quotes/clustering/results/run.json \
        --viz examples/lunyu_quotes/clustering/results/report.html

Usage:
    uv run python examples/lunyu_quotes/clustering/export_vectors.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from embedrag.query.index.id_mapping import IDMapper
from embedrag.query.index.shard import ShardWorker
from embedrag.query.retrieval.dense import ShardManager

HERE = Path(__file__).resolve().parent
SNAPSHOT_ROOT = HERE.parent / "snapshot" / "active"


def _find_snapshot() -> Path:
    candidates = sorted(SNAPSHOT_ROOT.glob("v*"))
    if not candidates:
        raise SystemExit(
            f"No snapshot found under {SNAPSHOT_ROOT}.\nBuild it first (see examples/lunyu_quotes/README.md)."
        )
    return candidates[-1]


def main() -> None:
    snap = _find_snapshot()
    manifest = json.loads((snap / "manifest.json").read_text())
    text_idx = manifest["indexes"]["text"]
    print(f"Snapshot: {snap.name}  type={text_idx['type']} dim={text_idx['dim']} vectors={text_idx['total_vectors']}")

    shard_files = [snap / s["file"] for s in text_idx["shards"]]
    id_map_file = snap / manifest["id_maps"]["text"]["file"]

    workers = [ShardWorker(str(f), use_mmap=False) for f in shard_files]
    mapper = IDMapper.load(str(id_map_file), [w.ntotal for w in workers])
    manager = ShardManager(workers, mapper)

    chunk_ids, vectors = manager.reconstruct_all()
    print(f"Reconstructed {vectors.shape[0]} vectors of dim {vectors.shape[1]}")

    db_path = snap / "db" / "embedrag.db"
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT chunk_id, text, metadata_json FROM chunks").fetchall()
    conn.close()
    text_map = {r["chunk_id"]: r["text"] for r in rows}
    chap_map = {r["chunk_id"]: (json.loads(r["metadata_json"]) or {}).get("chapter_name") for r in rows}

    jsonl_path = HERE / "quotes.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for cid in chunk_ids:
            f.write(
                json.dumps(
                    {"id": cid, "text": text_map.get(cid, ""), "chapter": chap_map.get(cid)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    npy_path = HERE / "quotes.npy"
    np.save(npy_path, vectors.astype("float32"))

    print(f"Wrote {jsonl_path}  ({len(chunk_ids)} rows)")
    print(f"Wrote {npy_path}  {vectors.shape}")


if __name__ == "__main__":
    main()
