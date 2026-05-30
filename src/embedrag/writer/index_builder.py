"""FAISS IVF_PQ index builder: training, sharding, and serialization.

This module provides the logic for constructing sharded FAISS indexes from
document embeddings. To achieve high performance and support datasets that
exceed single-machine memory, EmbedRAG splits the vector space into multiple
independent shards. This builder handles the automatic selection of the
most appropriate index type (e.g., Flat, IVF, or PQ) based on the dataset size
and performs deterministic sharding to ensure consistency.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import faiss
import msgpack
import numpy as np

from embedrag.config import IndexBuildConfig
from embedrag.logging_setup import get_logger
from embedrag.models.manifest import IndexInfo, ShardEntry

logger = get_logger(__name__)


class IndexBuilder:
    """Builds sharded FAISS IVF_PQ indexes from chunk embeddings.

    The builder orchestrates the entire index creation pipeline:
    1. Distributing chunk IDs and embeddings into shards using consistent hashing.
    2. Analyzing the dataset size to select an optimal FAISS index factory string.
    3. Training IVF centroids and PQ sub-quantizers if necessary.
    4. Building and serializing individual shards to disk.
    5. Generating the corresponding ID mapping files.
    """

    def __init__(self, config: IndexBuildConfig, dim: int = 1024):
        """Initialize the IndexBuilder.

        Args:
            config (IndexBuildConfig): Configuration parameters for the build
                process, including shard count and quantization settings.
            dim (int, optional): The dimensionality of the input vectors.
                Defaults to 1024.
        """
        self._config = config
        self._dim = dim

    def build(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        output_dir: str,
        space: str = "text",
    ) -> tuple[IndexInfo, str]:
        """Build a complete sharded FAISS index for a specific embedding space.

        This is the primary entry point for index construction. It takes a
        flat list of embeddings and their IDs, partitions them, builds the
        physical shard files, and returns the metadata required for the manifest.

        Args:
            chunk_ids (list[str]): A list of globally unique chunk identifiers.
            embeddings (np.ndarray): A 2D float32 numpy array of shape (N, dim)
                containing the vectors to be indexed.
            output_dir (str): The root directory where the built files will
                be stored.
            space (str, optional): The name of the embedding space (e.g., 'text').
                Files will be stored in a sub-folder named after the space.
                Defaults to "text".

        Returns:
            tuple[IndexInfo, str]: A tuple containing:
                - IndexInfo: A dataclass describing the built index (type, shards, etc.).
                - str: The filesystem path to the generated ID mapping file.
        """
        n_vectors = len(chunk_ids)
        assert embeddings.shape == (
            n_vectors,
            self._dim,
        ), f"Shape mismatch: {embeddings.shape} vs ({n_vectors}, {self._dim})"

        output = Path(output_dir) / "index" / space
        output.mkdir(parents=True, exist_ok=True)

        logger.info("index_build_start", space=space, n_vectors=n_vectors, dim=self._dim)
        t0 = time.monotonic()

        num_shards = self._config.num_shards
        index_type = self._determine_index_type(n_vectors)

        shard_assignments = self._assign_shards(chunk_ids, num_shards)
        shard_entries: list[ShardEntry] = []
        global_id_map: dict[int, str] = {}
        global_offset = 0

        for shard_idx in range(num_shards):
            mask = shard_assignments == shard_idx
            shard_ids = [cid for cid, m in zip(chunk_ids, mask) if m]
            shard_vecs = embeddings[mask]

            if len(shard_ids) == 0:
                continue

            index = self._build_single_shard(shard_vecs, index_type, n_vectors)

            shard_file = f"shard_{shard_idx}.faiss"
            shard_path = output / shard_file
            faiss.write_index(index, str(shard_path))

            for local_idx, cid in enumerate(shard_ids):
                global_id_map[global_offset + local_idx] = cid
            global_offset += len(shard_ids)

            shard_entries.append(
                ShardEntry(
                    file=f"index/{space}/{shard_file}",
                    raw_size=shard_path.stat().st_size,
                    num_vectors=len(shard_ids),
                )
            )
            logger.info(
                "shard_built",
                space=space,
                shard=shard_idx,
                vectors=len(shard_ids),
                size_mb=round(shard_path.stat().st_size / 1024 / 1024, 1),
            )

        str_id_map = {str(k): v for k, v in global_id_map.items()}
        id_map_path = str(output / "id_map.msgpack")
        with open(id_map_path, "wb") as f:
            msgpack.pack(str_id_map, f)

        elapsed = time.monotonic() - t0
        logger.info("index_build_done", space=space, elapsed_s=round(elapsed, 1), shards=len(shard_entries))

        index_info = IndexInfo(
            type=index_type,
            dim=self._dim,
            metric="IP",
            nprobe_default=32,
            num_shards=len(shard_entries),
            total_vectors=n_vectors,
            shards=shard_entries,
        )
        return index_info, id_map_path

    def _determine_index_type(self, n_vectors: int) -> str:
        """Heuristically choose the optimal FAISS index factory string.

        The selection logic is:
            - < 1,000 vectors: 'Flat' (Exact search, no training).
            - < 50,000 vectors: 'IVF{n},Flat' (Inverted file with exact residuals).
            - >= 50,000 vectors: 'IVF{n},PQ{m}' (Product Quantization for high compression).

        Args:
            n_vectors (int): The total number of vectors in the dataset.

        Returns:
            str: A FAISS factory string compatible with `faiss.index_factory`.
        """
        if n_vectors < 1_000:
            return "Flat"
        nlist = min(self._config.ivf_nlist, max(4, int(n_vectors**0.5)))
        pq_m = self._config.pq_m
        if n_vectors < 50_000 or self._dim < pq_m:
            return f"IVF{nlist},Flat"
        return f"IVF{nlist},PQ{pq_m}"

    def _build_single_shard(self, vectors: np.ndarray, index_type: str, total_vectors: int) -> faiss.Index:
        """Construct, train, and populate a single FAISS index shard.

        Args:
            vectors (np.ndarray): The subset of vectors to be added to this shard.
            index_type (str): The FAISS factory string to use.
            total_vectors (int): The total number of vectors across all shards
                (used to determine training samples).

        Returns:
            faiss.Index: A fully built and trained FAISS index object.
        """
        n = vectors.shape[0]
        dim = vectors.shape[1]

        if index_type == "Flat":
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            return index

        index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)

        train_size = min(n, self._config.train_sample_size)
        if train_size < n:
            rng = np.random.RandomState(42)
            indices = rng.choice(n, train_size, replace=False)
            train_data = vectors[indices]
        else:
            train_data = vectors

        nlist = int(index_type.split("IVF")[1].split(",")[0])
        if train_size < nlist:
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            return index

        index.train(train_data)
        index.add(vectors)
        return index

    def _assign_shards(self, chunk_ids: list[str], num_shards: int) -> np.ndarray:
        """Deterministically assign each chunk ID to a shard index.

        Uses MD5 hashing to ensure that a given chunk ID always maps to the
        same shard, which is critical for consistent ID mapping and
        incremental updates.

        Args:
            chunk_ids (list[str]): The list of IDs to assign.
            num_shards (int): The total number of shards.

        Returns:
            np.ndarray: An int32 array of shard assignments (0 to num_shards-1).
        """
        assignments = np.zeros(len(chunk_ids), dtype=np.int32)
        for i, cid in enumerate(chunk_ids):
            h = int(hashlib.md5(cid.encode()).hexdigest(), 16)
            assignments[i] = h % num_shards
        return assignments
