"""Core document and chunk data models.

This module defines the internal data structures used to represent documents
and their hierarchical decompositions within the EmbedRAG system. These models
are foundational for the chunking, embedding, and retrieval processes,
allowing for a rich, tree-like representation of content.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """Represents a complete, logical document ingested into the system.

    A `Document` is the top-level unit of information. It contains the raw
    content and global metadata that applies to all of its child chunks.

    Attributes:
        doc_id (str): A globally unique identifier for the document.
        title (str, optional): The human-readable title of the document.
        source (str, optional): The origin of the document (e.g., a URL,
            filepath, or database key).
        doc_type (str, optional): A category or classification for the
            document (e.g., 'technical_manual', 'news_article').
        metadata (dict): A dictionary of arbitrary key-value pairs stored
            with the document.
    """

    doc_id: str
    title: str = ""
    source: str = ""
    doc_type: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkNode:
    """A single node in the document's hierarchical chunk tree.

    EmbedRAG represents documents as trees of chunks. A `ChunkNode` can represent
    anything from the entire document at the root level down to a single
    sentence or fixed-size window at the leaf level. This hierarchical structure
    enables features like "hierarchical expansion," where small chunks are
    retrieved but their parent context (e.g., the containing paragraph or section)
    is also returned to the user.

    Attributes:
        chunk_id (str): A globally unique identifier for this specific chunk.
        doc_id (str): The identifier of the parent `Document`.
        text (str): The text content of this chunk.
        parent_chunk_id (str, optional): The ID of the parent node in the
            hierarchy. If None, this is a root node.
        level (int): The depth in the tree. Conventionally: 0=document,
            1=section, 2=paragraph, 3=leaf chunk. Defaults to 0.
        level_type (str): A descriptive name for the level (e.g., 'chunk',
            'section', 'document'). Defaults to 'chunk'.
        seq_in_parent (int): The 0-indexed position of this chunk among its
            siblings under the same parent. Defaults to 0.
        metadata (dict): Key-value pairs specific to this chunk.
        embedding (list[float], optional): The pre-calculated vector
            embedding for this chunk's text.
        children (list[ChunkNode]): A list of child `ChunkNode` instances.
    """

    chunk_id: str
    doc_id: str
    text: str
    parent_chunk_id: str | None = None
    level: int = 0
    level_type: str = "chunk"
    seq_in_parent: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None
    children: list[ChunkNode] = field(default_factory=list)
