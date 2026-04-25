"""Query node FastAPI application with lifespan for bootstrap and shutdown."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from embedrag.config import QueryNodeConfig, load_query_config
from embedrag.logging_setup import get_logger, setup_logging
from embedrag.query.index.generation import GenerationManager
from embedrag.query.middleware import RequestContextMiddleware
from embedrag.writer.embedding_client import EmbeddingClient

logger = get_logger(__name__)


class QueryState:
    """Holds all runtime state for the query node."""

    def __init__(self, config: QueryNodeConfig):
        self.config = config
        self.gen_manager = GenerationManager()
        self.embedding_clients: dict[str, EmbeddingClient] = {}
        for space in config.embedding.get_all_spaces():
            space_cfg = config.embedding.get_space_config(space)
            self.embedding_clients[space] = EmbeddingClient(space_cfg)
        self.syncer: Any = None  # set by lifecycle/bootstrap

    def get_embedding_client(self, space: str = "text") -> EmbeddingClient:
        if space not in self.embedding_clients:
            available = list(self.embedding_clients.keys())
            raise KeyError(f"No embedding client for space '{space}'. Available: {available}")
        return self.embedding_clients[space]

    async def close(self) -> None:
        await self.gen_manager.close()
        for client in self.embedding_clients.values():
            await client.close()


@asynccontextmanager
async def query_lifespan(app: FastAPI) -> AsyncIterator[None]:
    config_path = app.state.config_path
    config = load_query_config(config_path)
    setup_logging(level=config.logging.level, fmt=config.logging.format, node_type="query")

    state = QueryState(config)
    app.state.query = state

    from embedrag.query.lifecycle.bootstrap import BootstrapError, bootstrap_query_node

    try:
        await bootstrap_query_node(state)
    except BootstrapError as exc:
        logger.critical("startup_aborted", reason=str(exc))
        import sys

        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"STARTUP FAILED: {exc}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)
        raise

    if not state.gen_manager.is_loaded:
        msg = "Bootstrap completed but no generation loaded -- this should not happen"
        logger.critical("startup_aborted", reason=msg)
        raise RuntimeError(msg)

    # Start background syncer if configured
    if config.sync.enabled:
        from pathlib import Path as _Path

        from embedrag.query.sync.downloader import SnapshotDownloader
        from embedrag.query.sync.syncer import SnapshotSyncer

        if config.sync.source == "http":
            if not config.sync.http_url:
                logger.warn("sync_disabled_no_url", reason="sync.http_url is empty")
            else:
                from embedrag.shared.http_snapshot_client import HttpSnapshotClient

                h_client = HttpSnapshotClient(
                    config.sync.http_url,
                    timeout=config.sync.download_timeout_seconds,
                )
                staging = str(_Path(config.node.data_dir) / "staging")
                downloader = SnapshotDownloader(
                    h_client,
                    staging,
                    concurrency=config.sync.download_concurrency,
                    timeout=config.sync.download_timeout_seconds,
                )
                state.syncer = SnapshotSyncer(
                    state,
                    h_client,
                    downloader,
                    cron_expr=config.sync.cron,
                    poll_interval=config.sync.poll_interval_seconds,
                )
                state.syncer.start()
        else:
            from embedrag.shared.object_store import ObjectStoreClient

            try:
                o_client = ObjectStoreClient(config.object_store)
                staging = str(_Path(config.node.data_dir) / "staging")
                downloader = SnapshotDownloader(
                    o_client,
                    staging,
                    concurrency=config.sync.download_concurrency,
                    timeout=config.sync.download_timeout_seconds,
                )
                state.syncer = SnapshotSyncer(
                    state,
                    o_client,
                    downloader,
                    cron_expr=config.sync.cron,
                    poll_interval=config.sync.poll_interval_seconds,
                )
                state.syncer.start()
            except Exception:
                logger.exception("sync_init_failed")

    logger.info("query_started", version=state.gen_manager.active_version)
    yield

    # Graceful shutdown
    if state.syncer:
        state.syncer.stop()
    from embedrag.query.lifecycle.shutdown import graceful_shutdown

    await graceful_shutdown(state)
    logger.info("query_stopped")


def create_query_app(config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="EmbedRAG Query", version="0.5.1", lifespan=query_lifespan)
    app.state.config_path = config_path
    app.add_middleware(RequestContextMiddleware)

    from starlette.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from starlette.responses import Response

    @app.get("/sw.js", include_in_schema=False)
    async def _no_service_worker():
        return Response(status_code=204)

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> PlainTextResponse:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    from embedrag.query.routes import router

    app.include_router(router)

    # Mount WebUI static files
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    webui_dir = Path(__file__).parent.parent / "webui"
    if webui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(webui_dir), html=True), name="webui")

    return app
