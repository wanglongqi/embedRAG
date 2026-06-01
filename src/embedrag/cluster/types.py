"""Typed data structures shared across the clustering pipeline.

These are plain dataclasses (not pydantic) so the core ``cluster`` package
stays usable as a standalone library with no web-framework dependency. The
HTTP layer converts them to pydantic models at the edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Item:
    """A single input item to be clustered."""

    id: str
    text: str = ""


@dataclass
class ClusterMember:
    """One item's membership in a cluster."""

    id: str
    text: str
    cluster_id: int
    probability: float = 1.0
    # cosine to own centroid vs the runner-up centroid (attribution)
    self_similarity: float = 0.0
    runner_up_cluster: int = -1
    runner_up_similarity: float = 0.0


@dataclass
class ClusterInfo:
    """A single discovered cluster, with explanation fields."""

    cluster_id: int
    size: int
    label: str = ""
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    cohesion: float = 0.0  # mean cosine of members to the centroid
    separation: float = 0.0  # cosine to the nearest other centroid (lower = better separated)
    representatives: list[str] = field(default_factory=list)  # medoid item ids
    representative_texts: list[str] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)


@dataclass
class ClusterResult:
    """Full result of a clustering run."""

    run_id: str
    algorithm: str
    params: dict[str, Any] = field(default_factory=dict)
    space: str = "text"
    created_at: str = ""
    source: str = ""  # e.g. "jsonl", "snapshot", "passed-in"

    n_items: int = 0
    n_clusters: int = 0
    noise_count: int = 0

    clusters: list[ClusterInfo] = field(default_factory=list)
    members: list[ClusterMember] = field(default_factory=list)

    metrics: dict[str, Any] = field(default_factory=dict)
    sweep: list[dict[str, Any]] = field(default_factory=list)  # auto param-search score curve
    projection: dict[str, Any] = field(default_factory=dict)  # 2D coords for scatter
    viz: list[dict[str, Any]] = field(default_factory=list)  # per-algorithm panel specs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return _to_jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClusterResult:
        """Reconstruct from a previously serialized dict."""
        clusters = [ClusterInfo(**c) for c in data.get("clusters", [])]
        members = [ClusterMember(**m) for m in data.get("members", [])]
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["clusters"] = clusters
        kwargs["members"] = members
        return cls(**kwargs)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to native python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | float):
        return float(obj)
    return obj
