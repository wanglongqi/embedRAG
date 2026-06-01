"""Vector preprocessing: normalization and optional dimensionality reduction.

The index uses inner-product similarity, so we L2-normalize vectors and treat
cosine == dot-product throughout (consistent with retrieval). Reduction is
optional: PCA is always available; UMAP (better cluster separation) is used
when installed and requested.
"""

from __future__ import annotations

import numpy as np

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized vectors (zero rows left as-is)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def reduce_dims(
    vectors: np.ndarray,
    method: str = "auto",
    n_components: int = 0,
    n_neighbors: int = 15,
) -> tuple[np.ndarray, str]:
    """Optionally reduce dimensionality before clustering.

    Args:
        vectors: (n, d) input, assumed already normalized.
        method: ``"none"``, ``"pca"``, ``"umap"``, or ``"auto"`` (pick by scale/availability).
        n_components: target dims; 0 picks a sensible default per method.
        n_neighbors: UMAP locality parameter.

    Returns ``(reduced_vectors, method_used)``.
    """
    n, d = vectors.shape
    if method == "none" or n < 5 or d <= 2:
        return vectors, "none"

    if method == "auto":
        # UMAP gives better separation but is slow/stochastic; prefer it for
        # mid-size sets when available, else PCA, else nothing.
        if n <= 50_000 and _umap_available():
            method = "umap"
        elif d > 50:
            method = "pca"
        else:
            return vectors, "none"

    if method == "umap":
        if not _umap_available():
            logger.warn("umap_unavailable_fallback_pca")
            method = "pca"
        else:
            return _reduce_umap(vectors, n_components or 10, n_neighbors), "umap"

    if method == "pca":
        return _reduce_pca(vectors, n_components or min(50, d, max(2, n - 1))), "pca"

    raise ValueError(f"Unknown reduction method: {method}")


def _reduce_pca(vectors: np.ndarray, n_components: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_components = max(2, min(n_components, vectors.shape[1], vectors.shape[0] - 1))
    reduced = PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(vectors)
    logger.info("reduce_pca", n_components=n_components)
    return reduced.astype(np.float32)


def _reduce_umap(vectors: np.ndarray, n_components: int, n_neighbors: int) -> np.ndarray:
    import umap

    n_neighbors = min(n_neighbors, max(2, vectors.shape[0] - 1))
    n_components = max(2, min(n_components, vectors.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    reduced = reducer.fit_transform(vectors)
    logger.info("reduce_umap", n_components=n_components, n_neighbors=n_neighbors)
    return np.asarray(reduced, dtype=np.float32)


def project_2d(vectors: np.ndarray) -> np.ndarray:
    """Project to 2D for visualization (UMAP if available, else PCA)."""
    n, d = vectors.shape
    if d <= 2:
        if d == 2:
            return vectors.astype(np.float32)
        return np.column_stack([vectors[:, 0], np.zeros(n)]).astype(np.float32)
    if n <= 50_000 and _umap_available():
        return _reduce_umap(vectors, 2, 15)
    return _reduce_pca(vectors, 2)


def _umap_available() -> bool:
    try:
        import umap  # noqa: F401

        return True
    except Exception:
        return False
