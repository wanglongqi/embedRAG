"""Tests for closure table building."""

from embedrag.models.chunk import ChunkNode
from embedrag.writer.chunking.hierarchy import build_closure_entries


def test_single_node():
    chunks = [ChunkNode(chunk_id="a", doc_id="d1", text="hello")]
    entries = build_closure_entries(chunks)
    assert ("a", "a", 0) in entries


def test_parent_child():
    parent = ChunkNode(chunk_id="p", doc_id="d1", text="parent")
    child = ChunkNode(chunk_id="c", doc_id="d1", text="child", parent_chunk_id="p")
    entries = build_closure_entries([parent, child])
    assert ("p", "p", 0) in entries
    assert ("c", "c", 0) in entries
    assert ("p", "c", 1) in entries


def test_three_levels():
    root = ChunkNode(chunk_id="r", doc_id="d1", text="root")
    mid = ChunkNode(chunk_id="m", doc_id="d1", text="mid", parent_chunk_id="r")
    leaf = ChunkNode(chunk_id="l", doc_id="d1", text="leaf", parent_chunk_id="m")
    entries = build_closure_entries([root, mid, leaf])
    assert ("r", "l", 2) in entries
    assert ("m", "l", 1) in entries
    assert ("r", "m", 1) in entries
