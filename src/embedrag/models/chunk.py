"""Core document and chunk data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    doc_id: str
    title: str = ""
    source: str = ""
    doc_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkNode:
    chunk_id: str
    doc_id: str
    text: str
    parent_chunk_id: Optional[str] = None
    level: int = 0  # 0=doc, 1=section, 2=paragraph, 3=chunk
    level_type: str = "chunk"  # document | section | paragraph | chunk
    seq_in_parent: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    children: list["ChunkNode"] = field(default_factory=list)
