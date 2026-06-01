"""Side-file persistence for cluster runs.

A cluster run is stored as a single JSON file under
``<data_dir>/cluster_runs/<run_id>.json``. This deliberately avoids any
snapshot DB schema change: runs can be created, listed, read, and deleted
independently of the index lifecycle.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from embedrag.cluster.types import ClusterResult
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

RUNS_DIRNAME = "cluster_runs"


def runs_dir(data_dir: str) -> Path:
    """Return (and create) the cluster-runs directory under ``data_dir``."""
    p = Path(data_dir) / RUNS_DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_run_id(prefix: str = "run") -> str:
    """Generate a sortable, unique run id."""
    return f"{prefix}-{int(time.time() * 1000)}"


def save_run(data_dir: str, result: ClusterResult) -> str:
    """Persist a cluster run; returns the file path."""
    path = runs_dir(data_dir) / f"{result.run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False)
    logger.info("cluster_run_saved", run_id=result.run_id, path=str(path), clusters=result.n_clusters)
    return str(path)


def load_run(data_dir: str, run_id: str) -> ClusterResult | None:
    """Load a single run by id, or None if missing."""
    path = runs_dir(data_dir) / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return ClusterResult.from_dict(json.load(f))


def delete_run(data_dir: str, run_id: str) -> bool:
    """Delete a run file; returns True if it existed."""
    path = runs_dir(data_dir) / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def list_runs(data_dir: str) -> list[dict]:
    """List run summaries (without the full member/projection payload)."""
    d = runs_dir(data_dir)
    summaries: list[dict] = []
    for path in sorted(d.glob("*.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(
            {
                "run_id": data.get("run_id", path.stem),
                "algorithm": data.get("algorithm", ""),
                "space": data.get("space", "text"),
                "created_at": data.get("created_at", ""),
                "n_items": data.get("n_items", 0),
                "n_clusters": data.get("n_clusters", 0),
                "noise_count": data.get("noise_count", 0),
                "source": data.get("source", ""),
                "metrics": data.get("metrics", {}),
                "clusters": [
                    {
                        "cluster_id": c.get("cluster_id"),
                        "label": c.get("label", ""),
                        "size": c.get("size", 0),
                        "keywords": c.get("keywords", []),
                    }
                    for c in data.get("clusters", [])
                ],
            }
        )
    return summaries
