"""Pluggable clustering backends behind a single interface.

Each backend takes preprocessed (normalized, optionally reduced) vectors and
produces integer labels (``-1`` == noise) plus optional per-point membership
probabilities. Backends also declare which visualization panels make sense for
them, so the UI/exports can adapt per algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

# Algorithm name -> the viz panels it supports (consumed by visualize.py / UI).
ALGORITHM_PANELS: dict[str, list[str]] = {
    "hdbscan": ["scatter", "probability_hist", "size_bar", "similarity_matrix"],
    "dbscan": ["scatter", "size_bar", "similarity_matrix"],
    "kmeans": ["scatter", "silhouette", "k_curve", "size_bar", "similarity_matrix"],
    "agglomerative": ["scatter", "dendrogram", "size_bar", "similarity_matrix"],
    "leiden": ["scatter", "size_bar", "similarity_matrix"],
}


@dataclass
class ClusterAssignment:
    """Output of a clustering backend."""

    labels: np.ndarray
    probabilities: np.ndarray | None = None
    extra: dict = field(default_factory=dict)  # algo-specific artifacts (e.g. linkage matrix)


class ClusterBackend:
    """Base interface for a clustering algorithm."""

    name: str = "base"

    def __init__(self, **params):
        self.params = params

    def fit(self, vectors: np.ndarray) -> ClusterAssignment:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def panels(self) -> list[str]:
        return ALGORITHM_PANELS.get(self.name, ["scatter", "size_bar"])


class HDBSCANBackend(ClusterBackend):
    """Density-based, auto cluster count, native noise + membership probability."""

    name = "hdbscan"

    def fit(self, vectors: np.ndarray) -> ClusterAssignment:
        from sklearn.cluster import HDBSCAN

        n = vectors.shape[0]
        min_cluster_size = int(self.params.get("min_cluster_size") or max(5, int(round(n**0.5 / 2)) or 5))
        min_cluster_size = max(2, min(min_cluster_size, max(2, n // 2)))
        min_samples = self.params.get("min_samples")
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=int(min_samples) if min_samples else None,
            metric="euclidean",  # vectors are L2-normalized => euclidean ~ cosine
            cluster_selection_method=self.params.get("cluster_selection_method", "eom"),
        )
        labels = model.fit_predict(vectors)
        probs = getattr(model, "probabilities_", None)
        self.params["min_cluster_size"] = min_cluster_size
        return ClusterAssignment(labels=labels, probabilities=probs)


class DBSCANBackend(ClusterBackend):
    """Density-based with explicit eps; good for the no-embedding/TF-IDF path."""

    name = "dbscan"

    def fit(self, vectors: np.ndarray) -> ClusterAssignment:
        from sklearn.cluster import DBSCAN

        eps = float(self.params.get("eps") or _estimate_eps(vectors, int(self.params.get("min_samples", 5))))
        min_samples = int(self.params.get("min_samples", 5))
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(vectors)
        self.params["eps"] = round(eps, 4)
        return ClusterAssignment(labels=labels)


class KMeansBackend(ClusterBackend):
    """Centroid-based (spherical via normalized inputs). Scales via MiniBatch."""

    name = "kmeans"

    def fit(self, vectors: np.ndarray) -> ClusterAssignment:
        n = vectors.shape[0]
        k = int(self.params.get("k") or self.params.get("n_clusters") or max(2, int(round((n / 2) ** 0.5))))
        k = max(2, min(k, n))
        use_faiss = self.params.get("use_faiss", n > 200_000)
        if use_faiss:
            labels, dist = _faiss_kmeans(vectors, k)
        else:
            from sklearn.cluster import KMeans, MiniBatchKMeans

            if n > 20_000:
                model = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=2048)
            else:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(vectors)
            dist = None
        self.params["k"] = k
        probs = None
        if dist is not None:
            # convert nearest-centroid distance to a soft confidence in (0,1]
            probs = 1.0 / (1.0 + dist)
        return ClusterAssignment(labels=labels.astype(int), probabilities=probs)


class AgglomerativeBackend(ClusterBackend):
    """Hierarchical (Ward); supports a dendrogram and threshold/K cut."""

    name = "agglomerative"

    def fit(self, vectors: np.ndarray) -> ClusterAssignment:
        from sklearn.cluster import AgglomerativeClustering

        n = vectors.shape[0]
        n_clusters = self.params.get("n_clusters") or self.params.get("k")
        distance_threshold = self.params.get("distance_threshold")
        if n_clusters is None and distance_threshold is None:
            n_clusters = max(2, int(round((n / 2) ** 0.5)))
        kwargs: dict = {"linkage": self.params.get("linkage", "ward")}
        if distance_threshold is not None:
            kwargs["n_clusters"] = None
            kwargs["distance_threshold"] = float(distance_threshold)
        else:
            nc = int(n_clusters) if n_clusters is not None else max(2, int(round((n / 2) ** 0.5)))
            kwargs["n_clusters"] = max(2, min(nc, n))
        model = AgglomerativeClustering(**kwargs)
        labels = model.fit_predict(vectors)
        extra = {}
        # Linkage matrix for the dendrogram (only feasible at small/medium scale).
        if n <= 4000:
            try:
                from scipy.cluster.hierarchy import linkage

                extra["linkage"] = linkage(vectors, method=self.params.get("linkage", "ward")).tolist()
            except Exception as exc:  # scipy optional
                logger.warn("dendrogram_unavailable", error=str(exc))
        self.params["n_clusters"] = int(len(set(labels)))
        return ClusterAssignment(labels=labels.astype(int), extra=extra)


class LeidenBackend(ClusterBackend):
    """Community detection on a FAISS kNN graph (optional deps)."""

    name = "leiden"

    def fit(self, vectors: np.ndarray) -> ClusterAssignment:
        try:
            import igraph as ig
            import leidenalg
        except ImportError as exc:  # pragma: no cover - optional
            raise RuntimeError("leiden backend needs `igraph` and `leidenalg` installed") from exc

        k = int(self.params.get("knn", 15))
        resolution = float(self.params.get("resolution", 1.0))
        edges, weights = _knn_graph(vectors, k)
        g = ig.Graph(n=vectors.shape[0], edges=edges, edge_attrs={"weight": weights})
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=42,
        )
        labels = np.asarray(part.membership, dtype=int)
        self.params["knn"] = k
        self.params["resolution"] = resolution
        return ClusterAssignment(labels=labels)


_BACKENDS: dict[str, type[ClusterBackend]] = {
    "hdbscan": HDBSCANBackend,
    "dbscan": DBSCANBackend,
    "kmeans": KMeansBackend,
    "agglomerative": AgglomerativeBackend,
    "leiden": LeidenBackend,
}


def available_algorithms() -> list[str]:
    """Names of all registered clustering backends."""
    return list(_BACKENDS.keys())


def make_backend(name: str, **params) -> ClusterBackend:
    """Instantiate a clustering backend by name."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available_algorithms()}")
    return _BACKENDS[name](**params)


def _estimate_eps(vectors: np.ndarray, min_samples: int) -> float:
    """Estimate DBSCAN eps via the median k-distance (k = min_samples)."""
    from sklearn.neighbors import NearestNeighbors

    k = min(min_samples, max(2, vectors.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k).fit(vectors)
    dists, _ = nn.kneighbors(vectors)
    kth = np.sort(dists[:, -1])
    # knee heuristic: use the 90th percentile of k-distances
    return float(np.percentile(kth, 90))


def _faiss_kmeans(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    import faiss

    d = vectors.shape[1]
    km = faiss.Kmeans(d, k, niter=20, seed=42, verbose=False)
    km.train(np.ascontiguousarray(vectors, dtype=np.float32))
    dist, labels = km.index.search(np.ascontiguousarray(vectors, dtype=np.float32), 1)
    return labels.reshape(-1), dist.reshape(-1)


def _knn_graph(vectors: np.ndarray, k: int) -> tuple[list[tuple[int, int]], list[float]]:
    import faiss

    n, d = vectors.shape
    k = min(k + 1, n)
    # L2 normalize vectors to ensure IndexFlatIP computes exact Cosine Similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed_vectors = vectors / norms

    index = faiss.IndexFlatIP(d)
    index.add(np.ascontiguousarray(normed_vectors, dtype=np.float32))
    sims, idx = index.search(np.ascontiguousarray(normed_vectors, dtype=np.float32), k)
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n):
        for j, s in zip(idx[i], sims[i]):
            if j != i and j >= 0:
                edges.append((i, int(j)))
                weights.append(float(max(s, 0.0)))
    return edges, weights
