"""Writer node FastAPI application."""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from embedrag.config import WriterNodeConfig, load_writer_config
from embedrag.logging_setup import get_logger, setup_logging
from embedrag.writer.embedding_client import EmbeddingClient
from embedrag.writer.storage import WriterSQLitePool

logger = get_logger(__name__)


class WriterState:
    """Holds all runtime state for the writer node."""

    def __init__(self, config: WriterNodeConfig):
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
        if space in self.embedding_clients:
            return self.embedding_clients[space]
        raise KeyError(f"No embedding client for space '{space}'. Available: {list(self.embedding_clients.keys())}")

    @property
    def last_manifest(self):
        return self._last_manifest

    @last_manifest.setter
    def last_manifest(self, val):
        self._last_manifest = val

    async def close(self) -> None:
        for client in self.embedding_clients.values():
            await client.close()
        self.db.close()


@asynccontextmanager
async def writer_lifespan(app: FastAPI) -> AsyncIterator[None]:
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
    app = FastAPI(title="EmbedRAG Writer", version="0.4.0", lifespan=writer_lifespan)
    app.state.config_path = config_path

    from embedrag.writer.routes import router
    app.include_router(router)
    return app
