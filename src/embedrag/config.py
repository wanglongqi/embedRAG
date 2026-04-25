"""Configuration for writer and query nodes, loaded from YAML with env var support."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


def _resolve_env(value: str) -> str:
    """If value names an env var, resolve it."""
    return os.environ.get(value, "")


class ObjectStoreConfig(BaseModel):
    provider: Literal["tos", "s3", "minio"] = "s3"
    endpoint: str = ""
    bucket: str = "embedrag-data"
    prefix: str = "snapshots/"
    access_key_env: str = "AWS_ACCESS_KEY_ID"
    secret_key_env: str = "AWS_SECRET_ACCESS_KEY"
    region: str = "us-east-1"

    @property
    def access_key(self) -> str:
        return _resolve_env(self.access_key_env)

    @property
    def secret_key(self) -> str:
        return _resolve_env(self.secret_key_env)


class SnapshotConfig(BaseModel):
    bootstrap_version: str = "latest"
    poll_interval_seconds: int = 300
    download_concurrency: int = 4
    download_timeout_seconds: int = 600
    disk_reserve_bytes: int = 5_368_709_120  # 5GB


class SyncConfig(BaseModel):
    """Background snapshot sync configuration.

    When ``enabled`` is True the query node polls the configured source for
    new snapshot versions and hot-swaps automatically.
    """

    enabled: bool = False
    source: Literal["object_store", "http"] = "object_store"
    http_url: str = ""
    cron: str = ""
    poll_interval_seconds: int = 300
    download_concurrency: int = 4
    download_timeout_seconds: int = 600


class IndexConfig(BaseModel):
    num_shards: int = 4
    nprobe: int = 32
    mmap: bool = True


class SearchConfig(BaseModel):
    default_top_k: int = 10
    max_top_k: int = 100
    enable_sparse: bool = True
    enable_hierarchy_expand: bool = True
    context_depth: int = 1


class HotfixConfig(BaseModel):
    enabled: bool = True
    max_vectors: int = 10_000


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    readiness_delay_seconds: float = 0
    shutdown_drain_seconds: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["json", "console"] = "json"
    access_log: bool = True


class MetricsConfig(BaseModel):
    enabled: bool = True
    port: int = 9090


class NodeConfig(BaseModel):
    role: Literal["query", "writer"] = "query"
    node_id: str = "auto"
    data_dir: str = "/data/embedrag"

    @model_validator(mode="after")
    def _auto_node_id(self) -> NodeConfig:
        if self.node_id == "auto":
            self.node_id = socket.gethostname()
        return self


class DBConfig(BaseModel):
    path: str = ""
    wal_autocheckpoint: int = 1000
    cache_size_mb: int = 64


class EmbeddingSpaceConfig(BaseModel):
    """Config for a single embedding space (e.g. 'text', 'image')."""

    service_url: str = "http://localhost:8080/embed"
    api_format: Literal["embedrag", "openai"] = "embedrag"
    api_key: str = ""
    model: str = ""
    batch_size: int = 64
    timeout_seconds: int = 30
    retry_count: int = 3


class EmbeddingConfig(BaseModel):
    """Embedding configuration with multi-space support.

    If no ``spaces`` dict is provided, the top-level fields are used as
    a single "text" space.  When ``spaces`` is set, each key is a space
    name with its own ``EmbeddingSpaceConfig``.
    """

    service_url: str = "http://localhost:8080/embed"
    api_format: Literal["embedrag", "openai"] = "embedrag"
    api_key: str = ""
    model: str = ""
    batch_size: int = 64
    timeout_seconds: int = 30
    retry_count: int = 3
    spaces: dict[str, EmbeddingSpaceConfig] = Field(default_factory=dict)

    def get_space_config(self, space: str = "text") -> EmbeddingSpaceConfig:
        """Return config for a specific space."""
        if self.spaces and space in self.spaces:
            return self.spaces[space]
        return EmbeddingSpaceConfig(
            service_url=self.service_url,
            api_format=self.api_format,
            api_key=self.api_key,
            model=self.model,
            batch_size=self.batch_size,
            timeout_seconds=self.timeout_seconds,
            retry_count=self.retry_count,
        )

    def get_all_spaces(self) -> list[str]:
        """Return all configured space names."""
        if self.spaces:
            return list(self.spaces.keys())
        return ["text"]


class IndexBuildConfig(BaseModel):
    num_shards: int = 4
    ivf_nlist: int = 4096
    pq_m: int = 64
    train_sample_size: int = 500_000
    compression: Literal["zstd", "none"] = "zstd"
    compression_level: int = 3


# ── Top-level configs ──


class QueryNodeConfig(BaseModel):
    node: NodeConfig = Field(default_factory=lambda: NodeConfig(role="query"))
    object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    hotfix: HotfixConfig = Field(default_factory=HotfixConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class WriterNodeConfig(BaseModel):
    node: NodeConfig = Field(default_factory=lambda: NodeConfig(role="writer"))
    object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index_build: IndexBuildConfig = Field(default_factory=IndexBuildConfig)
    server: ServerConfig = Field(default_factory=lambda: ServerConfig(port=8001))
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @model_validator(mode="after")
    def _resolve_db_path(self) -> WriterNodeConfig:
        if not self.db.path:
            self.db.path = str(Path(self.node.data_dir) / "db" / "writer.db")
        return self


def load_config(path: str | Path) -> QueryNodeConfig | WriterNodeConfig:
    """Load config from YAML file and return the appropriate typed config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    role = raw.get("node", {}).get("role", "query")
    if role == "writer":
        return WriterNodeConfig(**raw)
    return QueryNodeConfig(**raw)


def load_query_config(path: str | Path | None = None) -> QueryNodeConfig:
    if path is None:
        return QueryNodeConfig()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return QueryNodeConfig(**raw)


def load_writer_config(path: str | Path | None = None) -> WriterNodeConfig:
    if path is None:
        return WriterNodeConfig()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return WriterNodeConfig(**raw)
