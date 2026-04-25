"""HTTP client for calling an external embedding service.

Supports two API formats:
- "embedrag": POST {"texts": [...]} -> {"embeddings": [[...], ...]}
- "openai":  POST {"input": [...], "model": "..."} -> {"data": [{"embedding": [...], "index": 0}, ...]}
"""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
import numpy as np

from embedrag.config import EmbeddingConfig, EmbeddingSpaceConfig
from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Async client that calls an external embedding service in batches.

    Accepts either an ``EmbeddingSpaceConfig`` (single space) or the legacy
    ``EmbeddingConfig`` (top-level fields used as a single space).
    """

    def __init__(self, config: EmbeddingSpaceConfig | EmbeddingConfig):
        self._url = config.service_url
        self._api_format = config.api_format
        self._api_key = config.api_key
        self._model = config.model
        self._batch_size = config.batch_size
        self._timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        self._retry_count = config.retry_count
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts, batching as configured."""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = await self._embed_batch(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if self._api_format == "openai":
            return await self._embed_batch_openai(texts)
        return await self._embed_batch_embedrag(texts)

    async def _embed_batch_embedrag(self, texts: list[str]) -> list[np.ndarray]:
        session = await self._get_session()
        payload = {"texts": texts}

        for attempt in range(self._retry_count):
            try:
                async with session.post(self._url, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    vectors = data.get("embeddings", data.get("vectors", []))
                    return [np.array(v, dtype=np.float32) for v in vectors]
            except Exception as e:
                logger.warn(
                    "embedding_retry",
                    attempt=attempt + 1,
                    error=str(e),
                    batch_size=len(texts),
                )
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Unreachable")

    async def _embed_batch_openai(self, texts: list[str]) -> list[np.ndarray]:
        session = await self._get_session()
        payload: dict = {"input": texts}
        if self._model:
            payload["model"] = self._model

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        for attempt in range(self._retry_count):
            try:
                async with session.post(
                    self._url, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    items = sorted(data["data"], key=lambda x: x["index"])
                    return [
                        np.array(item["embedding"], dtype=np.float32)
                        for item in items
                    ]
            except Exception as e:
                logger.warn(
                    "embedding_retry_openai",
                    attempt=attempt + 1,
                    error=str(e),
                    batch_size=len(texts),
                )
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        raise RuntimeError("Unreachable")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
