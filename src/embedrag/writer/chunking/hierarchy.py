"""Build closure table entries from a list of ChunkNodes with parent links."""

from __future__ import annotations

from embedrag.models.chunk import ChunkNode


def build_closure_entries(chunks: list[ChunkNode]) -> list[tuple[str, str, int]]:
    """Build closure table entries from a flat list of chunks with parent_chunk_id.

    Returns list of (ancestor_id, descendant_id, depth) tuples.
    Every node is its own ancestor at depth 0.
    """
    parent_map: dict[str, str | None] = {}
    for c in chunks:
        parent_map[c.chunk_id] = c.parent_chunk_id

    entries: list[tuple[str, str, int]] = []
    for c in chunks:
        entries.append((c.chunk_id, c.chunk_id, 0))

        ancestor_id = c.parent_chunk_id
        depth = 1
        visited: set[str] = {c.chunk_id}
        while ancestor_id and ancestor_id not in visited:
            entries.append((ancestor_id, c.chunk_id, depth))
            visited.add(ancestor_id)
            ancestor_id = parent_map.get(ancestor_id)
            depth += 1

    return entries
