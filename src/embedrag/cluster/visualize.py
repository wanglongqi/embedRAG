"""Build visualization specs (data-only JSON) rendered by the /cluster page.

The Python side never draws charts; it emits per-algorithm panel data that the
plotly-based front-end (or any consumer) turns into figures. Which panels are
produced depends on the algorithm (see ``ALGORITHM_PANELS``).
"""

from __future__ import annotations

import numpy as np

from embedrag.cluster.algorithms import ClusterAssignment
from embedrag.cluster.preprocess import project_2d
from embedrag.cluster.types import ClusterInfo, Item
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

_MAX_SCATTER_POINTS = 8000
_TEXT_PREVIEW = 160


def build_projection(
    vectors: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    items: list[Item],
) -> dict:
    """Compute 2D coordinates for the scatter plot (subsampled if huge)."""
    n = vectors.shape[0]
    idx = np.arange(n)
    if n > _MAX_SCATTER_POINTS:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(n, _MAX_SCATTER_POINTS, replace=False))

    coords = project_2d(vectors[idx]) if n > 2 else _pad_2d(vectors[idx])
    probs = probabilities[idx] if probabilities is not None else np.ones(len(idx))
    return {
        "x": coords[:, 0].round(4).tolist(),
        "y": coords[:, 1].round(4).tolist(),
        "cluster": [int(labels[i]) for i in idx],
        "id": [items[i].id for i in idx],
        "text": [items[i].text[:_TEXT_PREVIEW] for i in idx],
        "probability": np.round(probs, 4).tolist(),
        "subsampled": bool(n > _MAX_SCATTER_POINTS),
        "total": int(n),
    }


def build_panels(
    panels: list[str],
    vectors: np.ndarray,
    assignment: ClusterAssignment,
    clusters: list[ClusterInfo],
    sim_matrix: dict,
    sweep: list[dict],
) -> list[dict]:
    """Assemble the per-algorithm panel data list."""
    out: list[dict] = []
    labels = assignment.labels
    for panel in panels:
        if panel == "scatter":
            out.append({"type": "scatter", "title": "2D projection", "data": "projection"})
        elif panel == "size_bar":
            out.append(_size_bar(clusters))
        elif panel == "probability_hist" and assignment.probabilities is not None:
            out.append(_probability_hist(assignment.probabilities))
        elif panel == "similarity_matrix" and sim_matrix.get("cluster_ids"):
            out.append({"type": "similarity_matrix", "title": "Inter-cluster similarity", "data": sim_matrix})
        elif panel == "silhouette":
            sp = _silhouette_panel(vectors, labels)
            if sp:
                out.append(sp)
        elif panel == "k_curve" and sweep:
            kp = _k_curve(sweep)
            if kp:
                out.append(kp)
        elif panel == "dendrogram" and assignment.extra.get("linkage"):
            out.append({"type": "dendrogram", "title": "Dendrogram", "data": {"linkage": assignment.extra["linkage"]}})
    return out


def _size_bar(clusters: list[ClusterInfo]) -> dict:
    return {
        "type": "size_bar",
        "title": "Cluster sizes",
        "data": {
            "cluster_ids": [c.cluster_id for c in clusters],
            "labels": [c.label or str(c.cluster_id) for c in clusters],
            "sizes": [c.size for c in clusters],
        },
    }


def _probability_hist(probabilities: np.ndarray) -> dict:
    hist, edges = np.histogram(probabilities, bins=20, range=(0.0, 1.0))
    return {
        "type": "probability_hist",
        "title": "Membership probability",
        "data": {"counts": hist.tolist(), "bin_edges": edges.round(3).tolist()},
    }


def _silhouette_panel(vectors: np.ndarray, labels: np.ndarray, sample: int = 5000) -> dict | None:
    from sklearn.metrics import silhouette_samples

    core = labels != -1
    feats, y = vectors[core], labels[core]
    if len(np.unique(y)) < 2 or feats.shape[0] <= len(np.unique(y)):
        return None
    if feats.shape[0] > sample:
        rng = np.random.RandomState(42)
        sel = rng.choice(feats.shape[0], sample, replace=False)
        feats, y = feats[sel], y[sel]
    try:
        vals = silhouette_samples(feats, y, metric="euclidean")
    except Exception:
        return None
    per_cluster: dict[int, list[float]] = {}
    for cid in np.unique(y):
        per_cluster[int(cid)] = sorted(float(v) for v in vals[y == cid])
    return {
        "type": "silhouette",
        "title": "Silhouette by cluster",
        "data": {"per_cluster": {str(k): v for k, v in per_cluster.items()}, "mean": round(float(vals.mean()), 4)},
    }


def _k_curve(sweep: list[dict]) -> dict | None:
    points = []
    for s in sweep:
        params = s.get("params", {})
        k = params.get("k") or params.get("n_clusters")
        score = s.get("score")
        sil = s.get("metrics", {}).get("silhouette")
        if k is not None and (score is not None or sil is not None):
            points.append({"k": int(k), "score": score, "silhouette": sil})
    if not points:
        return None
    points.sort(key=lambda p: p["k"])
    return {
        "type": "k_curve",
        "title": "Score vs K",
        "data": {
            "k": [p["k"] for p in points],
            "score": [p["score"] for p in points],
            "silhouette": [p["silhouette"] for p in points],
        },
    }


def _pad_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[1] >= 2:
        return vectors[:, :2].astype(np.float32)
    return np.column_stack([vectors[:, 0], np.zeros(vectors.shape[0])]).astype(np.float32)
