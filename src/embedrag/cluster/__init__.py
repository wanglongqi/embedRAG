"""Standalone + integrated embedding clustering for embedRAG.

Public library API:

- ``cluster_vectors(vectors, items, ...)`` — synchronous core, no network.
- ``cluster_items(texts_or_items, ...)`` — convenience that vectorizes text
  (TF-IDF by default, or via a provided embedder) then clusters.
- ``ClusterResult`` / ``Item`` — typed result and input structures.

See ``embedrag.cluster.source`` for loaders (jsonl/csv/npy/writer DB) and
``embedrag.cluster.store`` for side-file persistence.
"""

from __future__ import annotations

import numpy as np

from embedrag.cluster.pipeline import apply_llm_labels, cluster_vectors
from embedrag.cluster.source import items_from_texts, tfidf_vectors
from embedrag.cluster.types import ClusterInfo, ClusterMember, ClusterResult, Item

__all__ = [
    "cluster_vectors",
    "cluster_items",
    "apply_llm_labels",
    "ClusterResult",
    "ClusterInfo",
    "ClusterMember",
    "Item",
]


def cluster_items(
    inputs: list[str] | list[Item],
    *,
    vectors: np.ndarray | None = None,
    **kwargs,
) -> ClusterResult:
    """Cluster a list of texts (or ``Item``s), vectorizing as needed.

    If ``vectors`` is given it is used directly. Otherwise text is embedded
    with a local TF-IDF representation (no service required). For embedding via
    a service, vectorize first and pass ``vectors``.
    """
    items: list[Item]
    if inputs and isinstance(inputs[0], Item):
        items = [x for x in inputs if isinstance(x, Item)]
    else:
        items = items_from_texts([str(x) for x in inputs])

    if vectors is None:
        source = kwargs.pop("source", "tfidf")
        vectors = tfidf_vectors([it.text for it in items])
    else:
        source = kwargs.pop("source", "passed-in")

    return cluster_vectors(vectors, items, source=source, **kwargs)
