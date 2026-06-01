"""Unit tests for the standalone clustering module."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from embedrag.cluster import cluster_items, cluster_vectors
from embedrag.cluster.algorithms import available_algorithms, make_backend
from embedrag.cluster.evaluate import external_metrics, internal_metrics, select_params
from embedrag.cluster.explain import compute_centroids, ctfidf_keywords
from embedrag.cluster.preprocess import l2_normalize, reduce_dims
from embedrag.cluster.source import load_items_from_file, tfidf_vectors
from embedrag.cluster.store import list_runs, load_run, save_run
from embedrag.cluster.types import ClusterResult, Item


@pytest.fixture
def blobs():
    vecs, y = make_blobs(n_samples=200, centers=4, n_features=16, random_state=0, cluster_std=0.6)
    items = [Item(id=str(i), text=f"text for sample in group {y[i]} number {i}") for i in range(len(vecs))]
    return vecs.astype("float32"), list(y), items


def test_l2_normalize_unit_norm():
    v = np.random.RandomState(0).randn(10, 8).astype("float32")
    out = l2_normalize(v)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_normalize_handles_zero_rows():
    v = np.zeros((3, 4), dtype="float32")
    out = l2_normalize(v)
    assert np.isfinite(out).all()


def test_reduce_pca_shape(blobs):
    vecs, _, _ = blobs
    reduced, method = reduce_dims(l2_normalize(vecs), method="pca", n_components=5)
    assert method == "pca"
    assert reduced.shape == (vecs.shape[0], 5)


def test_available_algorithms():
    algos = available_algorithms()
    for name in ("hdbscan", "kmeans", "agglomerative", "dbscan"):
        assert name in algos


def test_kmeans_backend_respects_k(blobs):
    vecs, _, _ = blobs
    backend = make_backend("kmeans", k=4)
    out = backend.fit(l2_normalize(vecs))
    assert len(set(out.labels)) == 4


def test_cluster_vectors_recovers_blobs(blobs):
    vecs, y, items = blobs
    res = cluster_vectors(vecs, items, algorithm="kmeans", reduce="pca", params={"k": 4}, ground_truth=y)
    assert res.n_clusters == 4
    assert res.metrics["external"]["ari"] > 0.9
    # every item appears as a member
    assert len(res.members) == len(items)
    # clusters carry keywords + representatives
    assert all(c.representatives for c in res.clusters)


def test_auto_algorithm_selection(blobs):
    vecs, y, items = blobs
    res = cluster_vectors(vecs, items, algorithm="auto", reduce="pca")
    assert res.algorithm in ("hdbscan", "kmeans")
    assert res.n_clusters >= 2
    # auto records a sweep / score curve
    assert isinstance(res.sweep, list)


def test_internal_metrics_single_cluster():
    vecs = np.random.RandomState(0).randn(20, 4).astype("float32")
    labels = np.zeros(20, dtype=int)
    m = internal_metrics(vecs, labels)
    assert m["n_clusters"] == 1
    assert m["silhouette"] is None


def test_external_metrics_perfect():
    labels = np.array([0, 0, 1, 1])
    m = external_metrics(labels, [0, 0, 1, 1])
    assert m["ari"] == 1.0
    assert m["nmi"] == 1.0


def test_select_params_pinned_skips_sweep(blobs):
    vecs, _, _ = blobs
    algo, assignment, params, sweep = select_params(l2_normalize(vecs), "kmeans", overrides={"k": 3}, auto=True)
    assert algo == "kmeans"
    assert params["k"] == 3
    assert len(sweep) == 1  # pinned => no sweep


def test_ctfidf_keywords_distinctive():
    texts = ["refund money order"] * 5 + ["shipping delivery late"] * 5
    labels = np.array([0] * 5 + [1] * 5)
    kw = ctfidf_keywords(texts, labels, top_n=5)
    assert 0 in kw and 1 in kw
    assert any("refund" in k or "money" in k or "order" in k for k in kw[0])


def test_compute_centroids_excludes_noise():
    vecs = np.random.RandomState(0).randn(10, 4).astype("float32")
    labels = np.array([-1, -1, 0, 0, 0, 1, 1, 1, 1, 1])
    centroids = compute_centroids(l2_normalize(vecs), labels)
    assert set(centroids.keys()) == {0, 1}


def test_cluster_items_tfidf_path():
    texts = ["cat dog pet animal"] * 6 + ["stock market finance money"] * 6
    res = cluster_items(texts, algorithm="kmeans", params={"k": 2}, reduce="none")
    assert res.n_clusters == 2
    assert res.source.startswith("tfidf")


def test_tfidf_vectors_cjk():
    texts = ["退款 订单 错误", "退款 我要 退钱", "物流 太慢 没到", "快递 一直 没到"]
    vecs = tfidf_vectors(texts)
    assert vecs.shape[0] == 4
    assert vecs.shape[1] > 0


def test_result_roundtrip_serialization(blobs):
    vecs, y, items = blobs
    res = cluster_vectors(vecs, items, algorithm="kmeans", params={"k": 4})
    data = res.to_dict()
    # JSON-serializable (no numpy types leak through)
    raw = json.dumps(data)
    restored = ClusterResult.from_dict(json.loads(raw))
    assert restored.run_id == res.run_id
    assert restored.n_clusters == res.n_clusters
    assert len(restored.members) == len(res.members)


def test_store_save_list_load(tmp_path, blobs):
    vecs, y, items = blobs
    res = cluster_vectors(vecs, items, algorithm="kmeans", params={"k": 4})
    save_run(str(tmp_path), res)
    runs = list_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["run_id"] == res.run_id
    loaded = load_run(str(tmp_path), res.run_id)
    assert loaded is not None
    assert loaded.n_clusters == res.n_clusters


def test_load_items_from_jsonl(tmp_path):
    p = tmp_path / "items.jsonl"
    rows = [{"id": f"r{i}", "text": f"complaint number {i}"} for i in range(5)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items, vectors = load_items_from_file(str(p))
    assert vectors is None
    assert len(items) == 5
    assert items[0].id == "r0"


def test_load_items_from_jsonl_with_embeddings(tmp_path):
    p = tmp_path / "items.jsonl"
    rows = [{"id": f"r{i}", "text": f"t{i}", "embedding": [float(i), float(i + 1)]} for i in range(4)]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items, vectors = load_items_from_file(str(p))
    assert vectors is not None
    assert vectors.shape == (4, 2)
