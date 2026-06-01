"""Cluster explainability: keywords, representatives, stats, attribution.

Produces the human-facing description of each cluster:
- distinctive keywords via class-based TF-IDF (c-TF-IDF),
- medoid example texts (points nearest the centroid),
- cohesion (mean cosine to centroid) and separation (nearest other centroid),
- an inter-cluster similarity matrix,
- and per-point "why this cluster" attribution.
"""

from __future__ import annotations

import re

import numpy as np

from embedrag.cluster.preprocess import l2_normalize
from embedrag.cluster.types import ClusterInfo, ClusterMember, Item
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def compute_centroids(vectors: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean (then renormalized) vector per non-noise cluster."""
    centroids: dict[int, np.ndarray] = {}
    for cid in sorted(set(int(x) for x in labels)):
        if cid == -1:
            continue
        members = vectors[labels == cid]
        if members.shape[0] == 0:
            continue
        c = members.mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = c / norm if norm > 0 else c
    return centroids


def explain(
    vectors: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    items: list[Item],
    top_keywords: int = 10,
    top_reps: int = 5,
) -> tuple[list[ClusterInfo], list[ClusterMember], dict[int, np.ndarray], dict]:
    """Build per-cluster info, per-member attribution, and similarity matrix."""
    norm_vecs = l2_normalize(vectors)
    centroids = compute_centroids(norm_vecs, labels)
    texts = [it.text for it in items]
    keywords_by_cluster = ctfidf_keywords(texts, labels, top_keywords)

    cluster_ids = sorted(centroids.keys())
    cmatrix = np.stack([centroids[c] for c in cluster_ids]) if cluster_ids else np.empty((0, 0))

    # similarity matrix between centroids
    sim_matrix: dict = {"cluster_ids": cluster_ids, "matrix": []}
    if len(cluster_ids) > 0:
        sims = cmatrix @ cmatrix.T
        sim_matrix["matrix"] = np.round(sims, 4).tolist()

    clusters: list[ClusterInfo] = []
    for cid in cluster_ids:
        idx = np.where(labels == cid)[0]
        member_vecs = norm_vecs[idx]
        centroid = centroids[cid]
        sims_to_centroid = member_vecs @ centroid
        order = np.argsort(-sims_to_centroid)
        rep_local = order[:top_reps]
        rep_ids = [items[idx[i]].id for i in rep_local]
        rep_texts = [items[idx[i]].text for i in rep_local]

        # separation: highest cosine to any other centroid
        separation = 0.0
        if len(cluster_ids) > 1:
            others = [centroids[o] for o in cluster_ids if o != cid]
            separation = float(max(centroid @ o for o in others))

        clusters.append(
            ClusterInfo(
                cluster_id=int(cid),
                size=int(idx.shape[0]),
                keywords=keywords_by_cluster.get(cid, []),
                cohesion=round(float(sims_to_centroid.mean()), 4),
                separation=round(separation, 4),
                representatives=rep_ids,
                representative_texts=rep_texts,
                member_ids=[items[i].id for i in idx],
                centroid=centroid.astype(float).round(6).tolist(),
            )
        )

    members = _attribute_members(norm_vecs, labels, probabilities, items, centroids, cluster_ids, cmatrix)
    return clusters, members, centroids, sim_matrix


def _attribute_members(
    norm_vecs: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    items: list[Item],
    centroids: dict[int, np.ndarray],
    cluster_ids: list[int],
    cmatrix: np.ndarray,
) -> list[ClusterMember]:
    members: list[ClusterMember] = []
    for i, item in enumerate(items):
        cid = int(labels[i])
        prob = float(probabilities[i]) if probabilities is not None else 1.0
        self_sim = 0.0
        runner_up = -1
        runner_sim = 0.0
        if cluster_ids and cmatrix.shape[0] > 0:
            sims = cmatrix @ norm_vecs[i]
            order = np.argsort(-sims)
            if cid in centroids:
                self_sim = float(norm_vecs[i] @ centroids[cid])
                # runner-up = best centroid that is not own cluster
                for j in order:
                    if cluster_ids[j] != cid:
                        runner_up = int(cluster_ids[j])
                        runner_sim = float(sims[j])
                        break
            else:
                # noise point: nearest centroid is informative
                best = int(order[0])
                runner_up = int(cluster_ids[best])
                runner_sim = float(sims[best])
        members.append(
            ClusterMember(
                id=item.id,
                text=item.text,
                cluster_id=cid,
                probability=round(prob, 4),
                self_similarity=round(self_sim, 4),
                runner_up_cluster=runner_up,
                runner_up_similarity=round(runner_sim, 4),
            )
        )
    return members


def ctfidf_keywords(texts: list[str], labels: np.ndarray, top_n: int = 10) -> dict[int, list[str]]:
    """Distinctive keywords per cluster via class-based TF-IDF (c-TF-IDF).

    Each cluster is treated as a single document. Char bigrams are used for
    CJK-heavy corpora (no word segmentation needed), words otherwise.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    cluster_ids = sorted(c for c in set(int(x) for x in labels) if c != -1)
    if not cluster_ids:
        return {}

    docs = []
    for cid in cluster_ids:
        joined = " ".join(texts[i] for i in range(len(texts)) if int(labels[i]) == cid)
        docs.append(joined)

    if not any(d.strip() for d in docs):
        return {cid: [] for cid in cluster_ids}

    cjk = sum(1 for d in docs if _CJK_RE.search(d)) > len(docs) / 2
    if cjk:
        vec = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2), max_features=5000)
    else:
        vec = CountVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
    try:
        counts = vec.fit_transform(docs).toarray().astype(np.float64)
    except ValueError:
        return {cid: [] for cid in cluster_ids}
    vocab = np.array(vec.get_feature_names_out())

    # c-TF-IDF: tf within class * log(1 + A / f_t)
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    f_t = counts.sum(axis=0)
    avg_words = counts.sum(axis=1).mean()
    idf = np.log(1.0 + avg_words / np.maximum(f_t, 1e-9))
    weights = tf * idf

    result: dict[int, list[str]] = {}
    for row, cid in enumerate(cluster_ids):
        top_idx = np.argsort(-weights[row])[:top_n]
        kws = [str(vocab[j]).strip() for j in top_idx if weights[row, j] > 0]
        result[cid] = [k for k in kws if k]
    return result
