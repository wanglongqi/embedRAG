"""Evaluation harness and automatic parameter selection.

This is the "honesty layer": every run reports internal quality metrics so the
result can be judged rather than blindly trusted. ``--auto`` sweeps the key
parameter for an algorithm and returns the full score curve, and the ``auto``
algorithm compares backends on a composite score.
"""

from __future__ import annotations

import numpy as np

from embedrag.cluster.algorithms import ClusterAssignment, make_backend
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


def internal_metrics(vectors: np.ndarray, labels: np.ndarray, sample: int = 5000) -> dict:
    """Compute clustering quality metrics that need no ground truth.

    Silhouette / Davies-Bouldin / Calinski-Harabasz are computed on non-noise
    points only. Returns a dict including ``n_clusters`` and ``noise_ratio``.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    labels = np.asarray(labels)
    n = labels.shape[0]
    noise_mask = labels == -1
    noise_ratio = float(noise_mask.mean()) if n else 0.0
    core = ~noise_mask
    core_labels = labels[core]
    unique = np.unique(core_labels)
    n_clusters = int(len(unique))

    out: dict = {
        "n_clusters": n_clusters,
        "noise_ratio": round(noise_ratio, 4),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }
    if n_clusters < 2 or core.sum() <= n_clusters:
        return out

    feats = vectors[core]
    y = core_labels
    if feats.shape[0] > sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(feats.shape[0], sample, replace=False)
        feats, y = feats[idx], y[idx]
        if len(np.unique(y)) < 2:
            return out
    try:
        out["silhouette"] = round(float(silhouette_score(feats, y, metric="euclidean")), 4)
        out["davies_bouldin"] = round(float(davies_bouldin_score(feats, y)), 4)
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(feats, y)), 2)
    except Exception as exc:
        logger.warn("internal_metrics_failed", error=str(exc))
    return out


def external_metrics(labels: np.ndarray, ground_truth: list) -> dict:
    """Compare predicted labels to ground-truth labels (when available)."""
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    gt = np.asarray(ground_truth)
    return {
        "ari": round(float(adjusted_rand_score(gt, labels)), 4),
        "nmi": round(float(normalized_mutual_info_score(gt, labels)), 4),
        "v_measure": round(float(v_measure_score(gt, labels)), 4),
    }


def composite_score(metrics: dict) -> float:
    """Single objective for model selection: silhouette penalized by noise."""
    sil = metrics.get("silhouette")
    if sil is None or metrics.get("n_clusters", 0) < 2:
        return -1.0
    return float(sil) - 0.5 * float(metrics.get("noise_ratio", 0.0))


def select_params(
    vectors: np.ndarray,
    algorithm: str,
    overrides: dict | None = None,
    auto: bool = True,
) -> tuple[str, ClusterAssignment, dict, list[dict]]:
    """Pick the best parameters for an algorithm (or compare backends for 'auto').

    Returns ``(chosen_algorithm, assignment, chosen_params, sweep)`` where
    ``sweep`` is the list of evaluated candidates (the score curve).
    """
    overrides = dict(overrides or {})

    if algorithm == "auto":
        return _auto_algorithm(vectors, overrides)

    # If the user pinned the controlling parameter, skip the sweep.
    if not auto or _has_pinned_param(algorithm, overrides):
        backend = make_backend(algorithm, **overrides)
        assignment = backend.fit(vectors)
        metrics = internal_metrics(vectors, assignment.labels)
        return algorithm, assignment, backend.params, [{"params": dict(backend.params), "metrics": metrics}]

    grid = _param_grid(algorithm, vectors.shape[0])
    sweep: list[dict] = []
    best = None  # (score, assignment, params)
    for value in grid:
        params = {**overrides, **value}
        backend = make_backend(algorithm, **params)
        try:
            assignment = backend.fit(vectors)
        except Exception as exc:
            logger.warn("sweep_candidate_failed", algorithm=algorithm, params=value, error=str(exc))
            continue
        metrics = internal_metrics(vectors, assignment.labels)
        score = composite_score(metrics)
        sweep.append({"params": dict(backend.params), "metrics": metrics, "score": round(score, 4)})
        if best is None or score > best[0]:
            best = (score, assignment, dict(backend.params))

    if best is None:
        backend = make_backend(algorithm, **overrides)
        assignment = backend.fit(vectors)
        return algorithm, assignment, backend.params, sweep

    return algorithm, best[1], best[2], sweep


def _auto_algorithm(vectors: np.ndarray, overrides: dict) -> tuple[str, ClusterAssignment, dict, list[dict]]:
    """Compare a density backend and a centroid backend, choose the better."""
    candidates = ["hdbscan", "kmeans"]
    best = None  # (score, algo, assignment, params)
    all_sweeps: list[dict] = []
    for algo in candidates:
        algo_name, assignment, params, sweep = select_params(vectors, algo, overrides, auto=True)
        for s in sweep:
            s["algorithm"] = algo_name
        all_sweeps.extend(sweep)
        metrics = internal_metrics(vectors, assignment.labels)
        score = composite_score(metrics)
        if best is None or score > best[0]:
            best = (score, algo_name, assignment, params)
    assert best is not None
    logger.info("auto_algorithm_selected", algorithm=best[1], score=round(best[0], 4))
    return best[1], best[2], best[3], all_sweeps


def _has_pinned_param(algorithm: str, overrides: dict) -> bool:
    pins = {
        "hdbscan": "min_cluster_size",
        "kmeans": "k",
        "agglomerative": "n_clusters",
        "dbscan": "eps",
        "leiden": "resolution",
    }
    key = pins.get(algorithm)
    if key and overrides.get(key):
        return True
    if algorithm == "kmeans" and overrides.get("n_clusters"):
        return True
    if algorithm == "agglomerative" and (
        overrides.get("k") or overrides.get("distance_threshold") or overrides.get("linkage")
    ):
        return True
    if algorithm == "leiden" and overrides.get("knn"):
        return True
    if algorithm == "hdbscan" and (overrides.get("min_samples") or overrides.get("cluster_selection_method")):
        return True
    if algorithm == "dbscan" and overrides.get("min_samples"):
        return True
    return False


def _param_grid(algorithm: str, n: int) -> list[dict]:
    if algorithm == "hdbscan":
        base = max(5, int(round(n**0.5 / 2)))
        values = sorted({max(2, int(base * f)) for f in (0.5, 1.0, 1.5, 2.0, 3.0)})
        return [{"min_cluster_size": v} for v in values if v < n]
    if algorithm in ("kmeans", "agglomerative"):
        hi = max(3, int(round((n / 2) ** 0.5)) * 2)
        ks = sorted({k for k in range(2, hi + 1)})
        if len(ks) > 12:
            ks = list(np.unique(np.linspace(2, hi, 12).astype(int)))
        key = "k" if algorithm == "kmeans" else "n_clusters"
        return [{key: int(k)} for k in ks if k < n]
    if algorithm == "dbscan":
        return [{"eps": None, "min_samples": m} for m in (3, 5, 10)]
    if algorithm == "leiden":
        return [{"resolution": r} for r in (0.5, 1.0, 1.5, 2.0)]
    return [{}]
