"""Tests for EmbeddingClient with embedrag and openai API formats."""

from __future__ import annotations

import numpy as np
import pytest

from embedrag.config import EmbeddingConfig
from embedrag.writer.embedding_client import EmbeddingClient


class _FakeResponse:
    """Minimal aiohttp response mock."""

    def __init__(self, data: dict, status: int = 200):
        self._data = data
        self.status = status

    async def json(self):
        return self._data

    def raise_for_status(self):
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeSession:
    """Minimal aiohttp session mock that records calls."""

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_url = None
        self.last_json = None
        self.last_headers = None
        self.closed = False

    def post(self, url, json=None, headers=None):
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        return self._response

    async def close(self):
        self.closed = True


@pytest.fixture
def embedrag_config():
    return EmbeddingConfig(
        service_url="http://test:8080/embed",
        api_format="embedrag",
        batch_size=2,
        timeout_seconds=5,
        retry_count=1,
    )


@pytest.fixture
def openai_config():
    return EmbeddingConfig(
        service_url="https://api.openai.com/v1/embeddings",
        api_format="openai",
        api_key="sk-test-key",
        model="text-embedding-3-large",
        batch_size=2,
        timeout_seconds=5,
        retry_count=1,
    )


class TestEmbedragFormat:
    @pytest.mark.asyncio
    async def test_single_batch(self, embedrag_config):
        client = EmbeddingClient(embedrag_config)
        fake_resp = _FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
        fake_session = _FakeSession(fake_resp)
        client._session = fake_session

        result = await client.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert fake_session.last_url == "http://test:8080/embed"
        assert fake_session.last_json == {"texts": ["hello", "world"]}
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.2])
        np.testing.assert_array_almost_equal(result[1], [0.3, 0.4])

    @pytest.mark.asyncio
    async def test_batching(self, embedrag_config):
        """With batch_size=2 and 3 texts, should make 2 calls."""
        call_count = 0

        class _CountingSession(_FakeSession):
            def post(self, url, json=None, headers=None):
                nonlocal call_count
                call_count += 1
                batch_size = len(json["texts"])
                vecs = [[float(i)] * 4 for i in range(batch_size)]
                return _FakeResponse({"embeddings": vecs})

        client = EmbeddingClient(embedrag_config)
        client._session = _CountingSession(_FakeResponse({}))

        result = await client.embed_texts(["a", "b", "c"])
        assert len(result) == 3
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_vectors_key_fallback(self, embedrag_config):
        client = EmbeddingClient(embedrag_config)
        fake_resp = _FakeResponse({"vectors": [[1.0, 2.0]]})
        client._session = _FakeSession(fake_resp)

        result = await client.embed_texts(["test"])
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])


class TestOpenAIFormat:
    @pytest.mark.asyncio
    async def test_single_batch(self, openai_config):
        client = EmbeddingClient(openai_config)
        fake_resp = _FakeResponse(
            {
                "data": [
                    {"embedding": [0.5, 0.6], "index": 0},
                    {"embedding": [0.7, 0.8], "index": 1},
                ],
                "model": "text-embedding-3-large",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        )
        fake_session = _FakeSession(fake_resp)
        client._session = fake_session

        result = await client.embed_texts(["hello", "world"])

        assert len(result) == 2
        assert fake_session.last_url == "https://api.openai.com/v1/embeddings"
        assert fake_session.last_json == {
            "input": ["hello", "world"],
            "model": "text-embedding-3-large",
        }
        assert fake_session.last_headers["Authorization"] == "Bearer sk-test-key"
        assert fake_session.last_headers["Content-Type"] == "application/json"
        np.testing.assert_array_almost_equal(result[0], [0.5, 0.6])
        np.testing.assert_array_almost_equal(result[1], [0.7, 0.8])

    @pytest.mark.asyncio
    async def test_index_reordering(self, openai_config):
        """OpenAI may return items out of order; client should sort by index."""
        client = EmbeddingClient(openai_config)
        fake_resp = _FakeResponse(
            {
                "data": [
                    {"embedding": [0.9, 0.9], "index": 1},
                    {"embedding": [0.1, 0.1], "index": 0},
                ],
            }
        )
        client._session = _FakeSession(fake_resp)

        result = await client.embed_texts(["first", "second"])
        np.testing.assert_array_almost_equal(result[0], [0.1, 0.1])
        np.testing.assert_array_almost_equal(result[1], [0.9, 0.9])

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        config = EmbeddingConfig(
            service_url="http://local-llm:8080/v1/embeddings",
            api_format="openai",
            model="local-model",
            batch_size=2,
            retry_count=1,
        )
        client = EmbeddingClient(config)
        fake_resp = _FakeResponse(
            {
                "data": [{"embedding": [1.0], "index": 0}],
            }
        )
        fake_session = _FakeSession(fake_resp)
        client._session = fake_session

        await client.embed_texts(["test"])
        assert "Authorization" not in (fake_session.last_headers or {})

    @pytest.mark.asyncio
    async def test_no_model(self):
        config = EmbeddingConfig(
            service_url="http://local:8080/v1/embeddings",
            api_format="openai",
            batch_size=2,
            retry_count=1,
        )
        client = EmbeddingClient(config)
        fake_resp = _FakeResponse(
            {
                "data": [{"embedding": [1.0], "index": 0}],
            }
        )
        session = _FakeSession(fake_resp)
        client._session = session

        await client.embed_texts(["test"])
        assert "model" not in session.last_json


class TestClientLifecycle:
    @pytest.mark.asyncio
    async def test_close(self, embedrag_config):
        client = EmbeddingClient(embedrag_config)
        fake_resp = _FakeResponse({"embeddings": [[1.0]]})
        client._session = _FakeSession(fake_resp)

        await client.close()
        assert client._session.closed
