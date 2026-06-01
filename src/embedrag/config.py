"""Configuration for writer and query nodes, loaded from YAML with env var support.

EmbedRAG uses Pydantic models for configuration, providing automatic validation,
type safety, and the ability to override settings via environment variables.
The configuration is hierarchically structured into logical groups such as
server settings, object store credentials, and search parameters.

Any field ending in `_env` in the YAML configuration is treated as a pointer
to an environment variable, allowing secrets (like AWS keys) to be managed
outside of the version-controlled configuration files.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


def _resolve_env(value: str) -> str:
    """If value names an env var, resolve it.

    Args:
        value (str): The name of the environment variable.

    Returns:
        str: The value of the environment variable, or an empty string if not found.
    """
    return os.environ.get(value, "")


class ObjectStoreConfig(BaseModel):
    """Configuration for S3-compatible object storage providers.

    This class manages connection details for services like AWS S3, Google Cloud
    Storage (in S3-compatibility mode), MinIO, and ByteDance TOS. It is used
    by the writer to upload snapshots and by the query node to download them.

    Attributes:
        provider (Literal["tos", "s3", "minio"]): The storage provider type.
            Defaults to "s3".
        endpoint (str): Custom endpoint URL (e.g., "http://localhost:9000" for MinIO).
        bucket (str): The name of the bucket where snapshots are stored.
        prefix (str): A key prefix (folder) within the bucket for all EmbedRAG data.
        access_key_env (str): Name of the environment variable holding the access key.
        secret_key_env (str): Name of the environment variable holding the secret key.
        region (str): The AWS region identifier (e.g., "us-east-1").
    """

    provider: Literal["tos", "s3", "minio"] = "s3"
    endpoint: str = ""
    bucket: str = "embedrag-data"
    prefix: str = "snapshots/"
    access_key_env: str = "AWS_ACCESS_KEY_ID"
    secret_key_env: str = "AWS_SECRET_ACCESS_KEY"
    region: str = "us-east-1"

    @property
    def access_key(self) -> str:
        """str: The resolved access key from the environment."""
        return _resolve_env(self.access_key_env)

    @property
    def secret_key(self) -> str:
        """str: The resolved secret key from the environment."""
        return _resolve_env(self.secret_key_env)


class SnapshotConfig(BaseModel):
    """Configuration for local snapshot management on the query node.

    Attributes:
        bootstrap_version (str): The version ID to load on startup. Use "latest"
            to automatically pull the most recent version from the source.
        poll_interval_seconds (int): How often to check for new snapshots when
            using basic polling. Defaults to 300.
        download_concurrency (int): Max number of concurrent downloads for
            shards and data files. Defaults to 4.
        download_timeout_seconds (int): Timeout for individual file downloads.
            Defaults to 600.
        disk_reserve_bytes (int): Minimum free disk space (in bytes) to maintain
            on the data partition. Defaults to 5GB.
    """

    bootstrap_version: str = "latest"
    poll_interval_seconds: int = 300
    download_concurrency: int = 4
    download_timeout_seconds: int = 600
    disk_reserve_bytes: int = 5_368_709_120  # 5GB


class SyncConfig(BaseModel):
    """Background snapshot synchronization configuration.

    When enabled, the query node will periodically poll the source (either
    object storage or an HTTP server) for newer snapshot versions and
    automatically perform zero-downtime hot-swaps.

    Attributes:
        enabled (bool): Whether to activate background synchronization.
        source (Literal["object_store", "http"]): The metadata source type.
        http_url (str): The base URL for fetching `latest.json` if source is "http".
        cron (str): An optional 5-field cron expression for scheduling checks.
        poll_interval_seconds (int): Interval between checks if cron is not set.
        download_concurrency (int): Max concurrent downloads for new snapshots.
        download_timeout_seconds (int): Timeout for sync downloads.
    """

    enabled: bool = False
    source: Literal["object_store", "http"] = "object_store"
    http_url: str = ""
    cron: str = ""
    poll_interval_seconds: int = 300
    download_concurrency: int = 4
    download_timeout_seconds: int = 600


class IndexConfig(BaseModel):
    """Configuration for FAISS index loading on the query node.

    Attributes:
        num_shards (int): The expected number of shards in the index.
        nprobe (int): Number of IVF cells to visit during search. Higher
            values increase recall but also increase latency. Defaults to 32.
        mmap (bool): Whether to use memory-mapped loading for FAISS indexes.
            Required for handling indexes larger than available RAM. Defaults to True.
    """

    num_shards: int = 4
    nprobe: int = 32
    mmap: bool = True


class SearchConfig(BaseModel):
    """High-level retrieval and ranking configuration.

    Attributes:
        default_top_k (int): Default number of results to return if not specified.
        max_top_k (int): Absolute maximum results allowed per request.
        enable_sparse (bool): Whether to include the keyword (BM25) search path.
        enable_hierarchy_expand (bool): Whether to automatically fetch parent
            context for retrieved chunks.
        context_depth (int): How many levels of parent context to traverse.
        dense_weight (float): Multiplier for dense scores in RRF fusion.
        sparse_weight (float): Multiplier for sparse scores in RRF fusion.
    """

    default_top_k: int = 10
    max_top_k: int = 100
    enable_sparse: bool = True
    enable_hierarchy_expand: bool = True
    context_depth: int = 1
    dense_weight: float = 1.0
    sparse_weight: float = 1.0


class HotfixConfig(BaseModel):
    """Configuration for real-time incremental updates (hotfixes).

    Hotfixes allow inserting or deleting documents in the query node's
    memory between snapshot updates.

    Attributes:
        enabled (bool): Whether to enable the hotfix buffer.
        max_vectors (int): The maximum number of vectors to keep in the
            in-memory hotfix index. Defaults to 10,000.
    """

    enabled: bool = True
    max_vectors: int = 10_000


class ServerConfig(BaseModel):
    """Web server (FastAPI/Uvicorn) configuration.

    Attributes:
        host (str): The interface to bind the server to.
        port (int): The port to listen on.
        workers (int): Number of worker processes. Note: For FAISS mmap
            stability, 1 worker is recommended.
        readiness_delay_seconds (float): Time to wait before reporting the
            node as ready during startup.
        shutdown_drain_seconds (int): Maximum time to wait for active
            requests to complete during shutdown.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    readiness_delay_seconds: float = 0
    shutdown_drain_seconds: int = 30


class LoggingConfig(BaseModel):
    """Structured logging configuration.

    Attributes:
        level (str): Log level (DEBUG, INFO, WARNING, ERROR).
        format (Literal["json", "console"]): The log output format. Use "json"
             for production/ELK stacks and "console" for local development.
        access_log (bool): Whether to enable HTTP access logs.
    """

    level: str = "INFO"
    format: Literal["json", "console"] = "json"
    access_log: bool = True


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration.

    Attributes:
        enabled (bool): Whether to start the Prometheus metrics exporter.
        port (int): The port to expose the /metrics endpoint on. Defaults to 9090.
    """

    enabled: bool = True
    port: int = 9090


class NodeConfig(BaseModel):
    """Basic node identification and role configuration.

    Attributes:
        role (Literal["query", "writer"]): The operating mode of the node.
        node_id (str): A unique identifier for this node instance. Use "auto"
            to use the machine's hostname.
        data_dir (str): The local directory used for storing databases,
            shards, and temporary build files.
        port (int): Port override. If 0, uses ServerConfig.port.
    """

    role: Literal["query", "writer"] = "query"
    node_id: str = "auto"
    data_dir: str = "/data/embedrag"
    port: int = 0

    @model_validator(mode="after")
    def _auto_node_id(self) -> NodeConfig:
        """Automatically set node_id to hostname if "auto"."""
        if self.node_id == "auto":
            self.node_id = socket.gethostname()
        return self


class DBConfig(BaseModel):
    """Configuration for the writer's SQLite metadata database.

    Attributes:
        path (str): File path to the database. If empty, it's auto-resolved
            relative to NodeConfig.data_dir.
        wal_autocheckpoint (int): SQLite WAL checkpoint interval in pages.
        cache_size_mb (int): SQLite page cache size in megabytes.
    """

    path: str = ""
    wal_autocheckpoint: int = 1000
    cache_size_mb: int = 64


class EmbeddingSpaceConfig(BaseModel):
    """Configuration for a specific embedding space/model.

    Attributes:
        service_url (str): The endpoint of the external embedding service.
        api_format (Literal["embedrag", "openai"]): The API protocol used
            by the service.
        api_key (str): Optional API key for the service.
        model (str): The model identifier to send in the request.
        batch_size (int): Max number of texts to send in a single batch.
        timeout_seconds (int): Request timeout in seconds.
        retry_count (int): Number of retry attempts on network failure.
    """

    service_url: str = "http://localhost:8080/embed"
    api_format: Literal["embedrag", "openai"] = "embedrag"
    api_key: str = ""
    model: str = ""
    batch_size: int = 64
    timeout_seconds: int = 30
    retry_count: int = 3


class EmbeddingConfig(BaseModel):
    """Root configuration for all embedding services.

    EmbedRAG supports multiple "spaces" (e.g., one for text, one for images).
    If the `spaces` dictionary is empty, the top-level fields are used
    to define a single default "text" space.

    Attributes:
        service_url (str): Default service URL.
        api_format (Literal["embedrag", "openai"]): Default API format.
        api_key (str): Default API key.
        model (str): Default model name.
        batch_size (int): Default batch size.
        timeout_seconds (int): Default timeout.
        retry_count (int): Default retry count.
        spaces (dict[str, EmbeddingSpaceConfig]): Dictionary mapping space
            names to their specific configurations.
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
        """Get the configuration for a specific embedding space.

        If the space is not found in the `spaces` dictionary, returns a
        configuration built from the default top-level fields.

        Args:
            space (str, optional): The name of the space to retrieve.
                Defaults to "text".

        Returns:
            EmbeddingSpaceConfig: The configuration for the requested space.
        """
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
        """Get a list of all configured space names.

        Returns:
            list[str]: A list of space identifiers.
        """
        if self.spaces:
            return list(self.spaces.keys())
        return ["text"]


class IndexBuildConfig(BaseModel):
    """Configuration for building FAISS indexes on the writer node.

    Attributes:
        num_shards (int): Number of shards to split the index into.
        ivf_nlist (int): Number of IVF centroids to train.
        pq_m (int): Number of PQ sub-vectors.
        train_sample_size (int): Maximum number of vectors to use for training.
        compression (Literal["zstd", "none"]): Compression algorithm for shards.
        compression_level (int): Zstd compression level (1-22).
    """

    num_shards: int = 4
    ivf_nlist: int = 4096
    pq_m: int = 64
    train_sample_size: int = 500_000
    compression: Literal["zstd", "none"] = "zstd"
    compression_level: int = 3


class LLMConfig(BaseModel):
    """Configuration for LLM integration (e.g., cluster labeling, summary generation).

    Attributes:
        service_url (str): The endpoint of the external LLM chat service.
        api_key (str): Optional API key for the service.
        api_key_env (str): Name of environment variable containing API key.
        model (str): The model identifier to send in the request.
        language (str): The target language for generation (e.g. "auto", "Chinese").
        timeout_seconds (int): Request timeout in seconds.
    """

    service_url: str = ""
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    model: str = ""
    language: str = "auto"
    timeout_seconds: int = 60

    @property
    def api_key_resolved(self) -> str:
        """str: The resolved API key from env or direct config."""
        if self.api_key:
            return self.api_key
        return _resolve_env(self.api_key_env)


# ── Top-level configs ──


class QueryNodeConfig(BaseModel):
    """The root configuration for an EmbedRAG Query Node.

    Attributes:
        node (NodeConfig): Basic node settings.
        object_store (ObjectStoreConfig): Snapshot source settings.
        snapshot (SnapshotConfig): Local snapshot management.
        sync (SyncConfig): Background update settings.
        index (IndexConfig): FAISS loading parameters.
        search (SearchConfig): Retrieval and fusion settings.
        hotfix (HotfixConfig): Real-time update buffer.
        embedding (EmbeddingConfig): Embedding service settings.
        llm (LLMConfig): Default LLM configuration for cluster labeling/chat.
        server (ServerConfig): FastAPI server settings.
        logging (LoggingConfig): Logging settings.
        metrics (MetricsConfig): Monitoring settings.
    """

    node: NodeConfig = Field(default_factory=lambda: NodeConfig(role="query"))
    object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    sync: SyncConfig = Field(default_factory=SyncConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    hotfix: HotfixConfig = Field(default_factory=HotfixConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


class WriterNodeConfig(BaseModel):
    """The root configuration for an EmbedRAG Writer Node.

    Attributes:
        node (NodeConfig): Basic node settings.
        object_store (ObjectStoreConfig): Snapshot upload settings.
        db (DBConfig): Metadata database settings.
        embedding (EmbeddingConfig): Embedding service settings.
        llm (LLMConfig): Default LLM configuration for cluster labeling/chat.
        index_build (IndexBuildConfig): FAISS build parameters.
        server (ServerConfig): FastAPI server settings (defaults to port 8001).
        logging (LoggingConfig): Logging settings.
        metrics (MetricsConfig): Monitoring settings.
    """

    node: NodeConfig = Field(default_factory=lambda: NodeConfig(role="writer"))
    object_store: ObjectStoreConfig = Field(default_factory=ObjectStoreConfig)
    db: DBConfig = Field(default_factory=DBConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    index_build: IndexBuildConfig = Field(default_factory=IndexBuildConfig)
    server: ServerConfig = Field(default_factory=lambda: ServerConfig(port=8001))
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @model_validator(mode="after")
    def _resolve_db_path(self) -> WriterNodeConfig:
        """Resolve the database path relative to data_dir if not specified."""
        if not self.db.path:
            self.db.path = str(Path(self.node.data_dir) / "db" / "writer.db")
        return self


def load_config(path: str | Path) -> QueryNodeConfig | WriterNodeConfig:
    """Load a configuration file and return the appropriate node config.

    The function reads the `node.role` field to decide whether to return
    a `QueryNodeConfig` or a `WriterNodeConfig`.

    Args:
        path (str | Path): Path to the YAML configuration file.

    Returns:
        QueryNodeConfig | WriterNodeConfig: The loaded and validated config.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    role = raw.get("node", {}).get("role", "query")
    if role == "writer":
        return WriterNodeConfig(**raw)
    return QueryNodeConfig(**raw)


def load_query_config(path: str | Path | None = None) -> QueryNodeConfig:
    """Load configuration for a query node.

    Args:
        path (str | Path, optional): Path to the YAML file. If None,
            returns the default configuration.

    Returns:
        QueryNodeConfig: The validated query node configuration.
    """
    if path is None:
        return QueryNodeConfig()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return QueryNodeConfig(**raw)


def load_writer_config(path: str | Path | None = None) -> WriterNodeConfig:
    """Load configuration for a writer node.

    Args:
        path (str | Path, optional): Path to the YAML file. If None,
            returns the default configuration.

    Returns:
        WriterNodeConfig: The validated writer node configuration.
    """
    if path is None:
        return WriterNodeConfig()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return WriterNodeConfig(**raw)
