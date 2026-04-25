"""Mapping between FAISS internal IDs and chunk IDs."""

from __future__ import annotations

from pathlib import Path

import msgpack

from embedrag.logging_setup import get_logger

logger = get_logger(__name__)


class IDMapper:
    """Maps (shard_index, faiss_position) -> chunk_id.

    Each shard has its own local ID space starting at 0.
    The id_map file stores a flat mapping of global_offset -> chunk_id.
    We reconstruct per-shard mappings from it.
    """

    def __init__(self, shard_id_maps: list[dict[int, str]]):
        self._maps = shard_id_maps

    @classmethod
    def load(cls, id_map_path: str, shard_sizes: list[int]) -> "IDMapper":
        """Load from a msgpack file and split into per-shard maps."""
        with open(id_map_path, "rb") as f:
            flat_map: dict = msgpack.unpack(f, strict_map_key=False)

        flat_map = {int(k): v for k, v in flat_map.items()}
        sorted_keys = sorted(flat_map.keys())

        shard_maps: list[dict[int, str]] = []
        offset = 0
        for size in shard_sizes:
            shard_map: dict[int, str] = {}
            for local_idx in range(size):
                global_idx = offset + local_idx
                if global_idx in flat_map:
                    shard_map[local_idx] = flat_map[global_idx]
            shard_maps.append(shard_map)
            offset += size

        logger.info(
            "id_map_loaded",
            total_entries=len(flat_map),
            shards=len(shard_maps),
        )
        return cls(shard_maps)

    def resolve(self, shard_idx: int, faiss_ids: list[int]) -> list[str]:
        """Resolve FAISS positions to chunk IDs for a given shard."""
        shard_map = self._maps[shard_idx]
        return [shard_map.get(fid, "") for fid in faiss_ids]

    def resolve_single(self, shard_idx: int, faiss_id: int) -> str:
        return self._maps[shard_idx].get(faiss_id, "")
