"""Writer node FastAPI application.

This module defines the web application and runtime state management for the
EmbedRAG Writer Node. The writer node is responsible for the "write" side of
the system: ingesting documents, managing the persistent SQLite database,
communicating with external embedding services, and building the final FAISS
indexes that are published as snapshots.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from embedrag.config import WriterNodeConfig, load_writer_config
from embedrag.logging_setup import get_logger, setup_logging
from embedrag.writer.embedding_client import EmbeddingClient
from embedrag.writer.storage import WriterSQLitePool

logger = get_logger(__name__)


class WriterState:
    """Holds all runtime state for the writer node.

    This class serves as a central registry for shared resources such as
    database connection pools and embedding clients. It is initialized once
    during the application startup and made available to all API routes
    via the `app.state` object.

    Attributes:
        config (WriterNodeConfig): The validated configuration for this node.
        db (WriterSQLitePool): The connection pool to the primary SQLite database.
        embedding_clients (dict[str, EmbeddingClient]): A mapping of embedding
            space names to their respective API clients.
        build_dir (Path): The local directory where new index versions are built
            before being published.
        current_version (str): The version ID of the most recently built index.
    """

    def __init__(self, config: WriterNodeConfig):
        """Initialize the writer state.

        Args:
            config (WriterNodeConfig): The writer node configuration.
        """
        self.config = config
        self.db = WriterSQLitePool(
            db_path=config.db.path,
            wal_autocheckpoint=config.db.wal_autocheckpoint,
            cache_size_mb=config.db.cache_size_mb,
        )
        self.embedding_clients: dict[str, EmbeddingClient] = {}
        for space in config.embedding.get_all_spaces():
            space_cfg = config.embedding.get_space_config(space)
            self.embedding_clients[space] = EmbeddingClient(space_cfg)
        self.build_dir = Path(config.node.data_dir) / "builds"
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.current_version: str = ""
        self._last_manifest = None

    def get_embedding_client(self, space: str = "text") -> EmbeddingClient:
        """Retrieve the embedding client for a specific space.

        Args:
            space (str, optional): The identifier of the embedding space.
                Defaults to "text".

        Returns:
            EmbeddingClient: The client configured for the requested space.

        Raises:
            KeyError: If no client is configured for the given space name.
        """
        if space in self.embedding_clients:
            return self.embedding_clients[space]
        available = list(self.embedding_clients.keys())
        raise KeyError(f"No embedding client for space '{space}'. Available: {available}")

    @property
    def last_manifest(self):
        """The manifest from the most recent successful build."""
        return self._last_manifest

    @last_manifest.setter
    def last_manifest(self, val):
        self._last_manifest = val

    async def close(self) -> None:
        """Gracefully shut down all database connections and network clients.

        This method should be called during the application's shutdown sequence.
        """
        for client in self.embedding_clients.values():
            await client.close()
        self.db.close()


@asynccontextmanager
async def writer_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manages the lifecycle of the Writer FastAPI application.

    This context manager handles the startup and shutdown phases. On startup,
    it loads the configuration, initializes the `WriterState`, and sets up
    structured logging. On shutdown, it ensures that all resources (DB
    connections, clients) are released properly.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is returned to the FastAPI framework to start serving.
    """
    config_path = app.state.config_path
    config = load_writer_config(config_path)
    setup_logging(level=config.logging.level, fmt=config.logging.format, node_type="writer")
    state = WriterState(config)
    app.state.writer = state
    logger.info("writer_started", data_dir=config.node.data_dir)
    yield
    await state.close()
    logger.info("writer_stopped")


def create_writer_app(config_path: str | None = None) -> FastAPI:
    """Factory function to create and configure the Writer FastAPI application.

    This function sets up the basic FastAPI app, attaches the lifespan manager,
    and registers all functional routes.

    Args:
        config_path (str, optional): An optional file path to a YAML
            configuration file.

    Returns:
        FastAPI: The fully configured web application instance.
    """
    app = FastAPI(title="EmbedRAG Writer", version="0.5.1", lifespan=writer_lifespan)
    app.state.config_path = config_path

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        """Prometheus metrics endpoint."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    from embedrag.writer.routes import router

    app.include_router(router)
    return app
