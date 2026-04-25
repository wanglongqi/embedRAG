"""Tests for hotfix API models and multi-space support."""

from embedrag.models.api import HotfixAddRequest, HotfixDeleteRequest


class TestHotfixModels:
    def test_hotfix_add_defaults(self):
        req = HotfixAddRequest(
            chunk_id="c-001",
            doc_id="d-001",
            text="test content",
            embedding=[0.1] * 512,
        )
        assert req.chunk_id == "c-001"
        assert req.doc_id == "d-001"
        assert req.text == "test content"
        assert req.space == "text"
        assert req.metadata == {}

    def test_hotfix_add_custom_space(self):
        req = HotfixAddRequest(
            chunk_id="c-002",
            doc_id="d-002",
            text="image data",
            embedding=[0.5] * 256,
            space="image",
            metadata={"title": "test.png"},
        )
        assert req.space == "image"
        assert req.metadata["title"] == "test.png"

    def test_hotfix_delete_defaults(self):
        req = HotfixDeleteRequest(chunk_ids=["c-001", "c-002"])
        assert req.chunk_ids == ["c-001", "c-002"]
        assert req.space == "text"

    def test_hotfix_delete_custom_space(self):
        req = HotfixDeleteRequest(chunk_ids=["c-003"], space="audio")
        assert req.space == "audio"

    def test_hotfix_delete_multiple(self):
        req = HotfixDeleteRequest(
            chunk_ids=["c-001", "c-002", "c-003", "c-004"],
            space="video",
        )
        assert len(req.chunk_ids) == 4
        assert req.space == "video"
