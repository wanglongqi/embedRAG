"""Tests for configuration loading and validation."""

import os
from pathlib import Path

import yaml

from embedrag.config import QueryNodeConfig, WriterNodeConfig, load_config


def _write_yaml(path: Path, data: dict) -> str:
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


class TestQueryConfig:
    def test_defaults(self):
        cfg = QueryNodeConfig()
        assert cfg.node.role == "query"
        assert cfg.server.port == 8000
        assert cfg.index.num_shards == 4

    def test_embedding_config_on_query(self):
        cfg = QueryNodeConfig()
        assert cfg.embedding.service_url == "http://localhost:8080/embed"

    def test_load_from_yaml(self, tmp_path):
        data = {
            "node": {"role": "query", "data_dir": "/tmp/test"},
            "server": {"port": 9000},
            "search": {"default_top_k": 20},
            "embedding": {"service_url": "http://emb:8080/embed"},
        }
        path = _write_yaml(tmp_path / "q.yaml", data)
        cfg = load_config(path)
        assert isinstance(cfg, QueryNodeConfig)
        assert cfg.server.port == 9000
        assert cfg.search.default_top_k == 20
        assert cfg.embedding.service_url == "http://emb:8080/embed"


class TestEmbeddingConfig:
    def test_default_format_is_embedrag(self):
        cfg = QueryNodeConfig()
        assert cfg.embedding.api_format == "embedrag"
        assert cfg.embedding.api_key == ""
        assert cfg.embedding.model == ""

    def test_openai_format_from_yaml(self, tmp_path):
        data = {
            "node": {"role": "query"},
            "embedding": {
                "service_url": "https://api.openai.com/v1/embeddings",
                "api_format": "openai",
                "api_key": "sk-test-key",
                "model": "text-embedding-3-large",
            },
        }
        path = _write_yaml(tmp_path / "q_openai.yaml", data)
        cfg = load_config(path)
        assert isinstance(cfg, QueryNodeConfig)
        assert cfg.embedding.api_format == "openai"
        assert cfg.embedding.api_key == "sk-test-key"
        assert cfg.embedding.model == "text-embedding-3-large"
        assert cfg.embedding.service_url == "https://api.openai.com/v1/embeddings"

    def test_openai_format_on_writer(self, tmp_path):
        data = {
            "node": {"role": "writer"},
            "embedding": {
                "service_url": "https://api.openai.com/v1/embeddings",
                "api_format": "openai",
                "api_key": "sk-writer-key",
                "model": "text-embedding-ada-002",
            },
        }
        path = _write_yaml(tmp_path / "w_openai.yaml", data)
        cfg = load_config(path)
        assert isinstance(cfg, WriterNodeConfig)
        assert cfg.embedding.api_format == "openai"
        assert cfg.embedding.model == "text-embedding-ada-002"

    def test_invalid_format_rejected(self):
        from pydantic import ValidationError
        import pytest
        with pytest.raises(ValidationError):
            QueryNodeConfig(embedding={"api_format": "invalid"})

    def test_embedrag_format_explicit(self, tmp_path):
        data = {
            "node": {"role": "query"},
            "embedding": {
                "service_url": "http://my-service:8080/embed",
                "api_format": "embedrag",
            },
        }
        path = _write_yaml(tmp_path / "q_mini.yaml", data)
        cfg = load_config(path)
        assert cfg.embedding.api_format == "embedrag"
        assert cfg.embedding.batch_size == 64


class TestWriterConfig:
    def test_defaults(self):
        cfg = WriterNodeConfig()
        assert cfg.node.role == "writer"
        assert cfg.server.port == 8001

    def test_db_path_auto_resolved(self):
        cfg = WriterNodeConfig(
            node={"role": "writer", "data_dir": "/data/wr"},
        )
        assert "writer.db" in cfg.db.path

    def test_load_from_yaml(self, tmp_path):
        data = {
            "node": {"role": "writer", "data_dir": "/tmp/wr"},
            "embedding": {"service_url": "http://localhost:1234/embed"},
        }
        path = _write_yaml(tmp_path / "w.yaml", data)
        cfg = load_config(path)
        assert isinstance(cfg, WriterNodeConfig)
        assert cfg.embedding.service_url == "http://localhost:1234/embed"
