"""Shared pytest fixtures for EmbedRAG tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_documents():
    """100 sample documents with mixed types."""
    docs = []
    for i in range(50):
        docs.append(
            {
                "doc_id": f"faq_{i:03d}",
                "title": f"FAQ Question {i}",
                "text": (
                    f"Q: What is item {i}? "
                    f"A: Item {i} is a frequently asked question about topic {i}."
                ),
                "doc_type": "faq",
                "source": "test",
            }
        )
    for i in range(30):
        docs.append(
            {
                "doc_id": f"article_{i:03d}",
                "title": f"Article about Topic {i}",
                "text": (
                    f"# Introduction\n\nThis article discusses topic {i} in detail.\n\n"
                    f"## Background\n\nThe background of topic {i} involves several key concepts. "
                    f"First, we need to understand the fundamentals. "
                    f"{'Lorem ipsum dolor sit amet. ' * 20}\n\n"
                    f"## Analysis\n\nOur analysis shows that topic {i} has multiple dimensions. "
                    f"{'Consectetur adipiscing elit. ' * 20}\n\n"
                    f"## Conclusion\n\nIn conclusion, topic {i} is important because "
                    f"it affects many areas."
                ),
                "doc_type": "article",
                "source": "test",
            }
        )
    for i in range(20):
        docs.append(
            {
                "doc_id": f"manual_{i:03d}",
                "title": f"Technical Manual {i}",
                "text": f"# Technical Manual {i}\n\n"
                f"## Chapter 1: Installation\n\n"
                f"To install the system, follow these steps. "
                f"{'Step by step instructions for installation. ' * 30}\n\n"
                f"### Section 1.1: Prerequisites\n\n"
                f"Before installing, ensure you have the following. "
                f"{'Prerequisite details and requirements. ' * 20}\n\n"
                f"## Chapter 2: Configuration\n\n"
                f"Configure the system using the following parameters. "
                f"{'Configuration parameter descriptions. ' * 30}\n\n"
                f"## Chapter 3: Troubleshooting\n\n"
                f"Common issues and their resolutions. "
                f"{'Troubleshooting guide content. ' * 20}",
                "doc_type": "manual",
                "source": "test",
            }
        )
    return docs


@pytest.fixture
def random_vectors():
    """Generate random 1024-dim vectors for testing."""

    def _generate(n: int, dim: int = 1024) -> np.ndarray:
        rng = np.random.RandomState(42)
        vecs = rng.randn(n, dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms  # normalize for IP metric

    return _generate
