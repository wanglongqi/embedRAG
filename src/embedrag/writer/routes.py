"""Writer API routes: /ingest, /build, /publish, /health."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Request

from embedrag.logging_setup import get_logger
from embedrag.models.api import (
    ArchiveRequest,
    ArchiveResponse,
    BuildRequest,
    BuildResponse,
    DeleteDocumentResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PublishResponse,
)
from embedrag.models.chunk import ChunkNode, Document
from embedrag.models.manifest import IndexInfo
from embedrag.shared.object_store import ObjectStoreClient
from embedrag.writer.chunking.hierarchy import build_closure_entries
from embedrag.writer.chunking.splitter import split_document
from embedrag.writer.index_builder import IndexBuilder
from embedrag.writer.snapshot import SnapshotPackager, SnapshotPublisher

logger = get_logger(__name__)
router = APIRouter()


def _get_state(request: Request) -> Any:
    return request.app.state.writer


@router.get("/health")
async def health(request: Request) -> HealthResponse:
    state = _get_state(request)
    return HealthResponse(
        status="ok",
        node_type="writer",
        version=state.current_version,
    )


@router.post("/ingest")
async def ingest(request: Request, body: IngestRequest) -> IngestResponse:
    state = _get_state(request)
    all_chunks: list[ChunkNode] = []
    docs: list[Document] = []
    doc_titles: dict[str, str] = {}
    space_chunks: dict[str, list[ChunkNode]] = {}

    for doc_input in body.documents:
        doc = Document(
            doc_id=doc_input.doc_id,
            title=doc_input.title,
            source=doc_input.source,
            doc_type=doc_input.doc_type,
            metadata=doc_input.metadata,
        )
        docs.append(doc)
        doc_titles[doc.doc_id] = doc.title

        chunks = split_document(
            doc_input.text,
            doc_input.doc_id,
            strategy=doc_input.chunking,
            chunk_size=doc_input.chunk_size or 512,
            overlap=doc_input.chunk_overlap or 128,
        )
        space = doc_input.modality
        for c in chunks:
            c.metadata["doc_type"] = doc_input.doc_type
            c.metadata["chunking"] = doc_input.chunking
            c.metadata["modality"] = space
        all_chunks.extend(chunks)
        space_chunks.setdefault(space, []).extend(chunks)

    # Embed per-space
    for space, chunks_in_space in space_chunks.items():
        embed_client = state.get_embedding_client(space)
        texts = [c.text for c in chunks_in_space]
        embeddings = await embed_client.embed_texts(texts)
        for chunk, emb in zip(chunks_in_space, embeddings):
            chunk.embedding = emb.tolist()

    doc_ids = [d.doc_id for d in docs]
    await state.db.cleanup_before_upsert(doc_ids)
    await state.db.insert_documents_batch(docs)

    for space, chunks_in_space in space_chunks.items():
        await state.db.insert_chunks_batch(chunks_in_space, space=space)

    await state.db.insert_fts_batch(all_chunks, doc_titles)

    closure = build_closure_entries(all_chunks)
    await state.db.insert_closure_batch(closure)

    logger.info(
        "ingest_complete",
        docs=len(docs),
        chunks=len(all_chunks),
        spaces=list(space_chunks.keys()),
    )
    return IngestResponse(
        ingested=len(docs),
        chunk_count=len(all_chunks),
        doc_ids=[d.doc_id for d in docs],
    )


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str) -> DeleteDocumentResponse:
    """Delete a document and all its chunks, FTS entries, and closure rows."""
    state = _get_state(request)
    chunks_deleted = await state.db.delete_document(doc_id)
    logger.info("document_deleted", doc_id=doc_id, chunks_deleted=chunks_deleted)
    return DeleteDocumentResponse(doc_id=doc_id, chunks_deleted=chunks_deleted)


@router.post("/build")
async def build(request: Request, body: BuildRequest = BuildRequest()) -> BuildResponse:
    state = _get_state(request)
    config = state.config
    t0 = time.monotonic()

    build_dir = str(state.build_dir / "current")
    Path(build_dir).mkdir(parents=True, exist_ok=True)

    spaces = await state.db.get_embedding_spaces()
    if not spaces:
        raise ValueError("No chunks with embeddings found")

    space_index_infos: dict[str, IndexInfo] = {}
    space_id_map_paths: dict[str, str] = {}
    total_vectors = 0

    for space in spaces:
        pairs = await state.db.get_all_chunks_with_embeddings(space)
        if not pairs:
            continue
        chunk_ids = [cid for cid, _ in pairs]
        embeddings = np.stack([emb for _, emb in pairs])
        builder = IndexBuilder(config.index_build, dim=embeddings.shape[1])
        index_info, id_map_path = builder.build(chunk_ids, embeddings, build_dir, space=space)
        space_index_infos[space] = index_info
        space_id_map_paths[space] = id_map_path
        total_vectors += index_info.total_vectors

    doc_count = await state.db.get_doc_count()
    chunk_count = await state.db.get_chunk_count()
    version_num = int(time.time())
    version = f"v{version_num:010d}"

    snapshot_dir = str(state.build_dir / version)
    db_export_path = str(Path(snapshot_dir) / "db" / "embedrag.db")
    Path(db_export_path).parent.mkdir(parents=True, exist_ok=True)
    exported_doc_count, exported_chunk_count = state.db.export_query_db(db_export_path)

    packager = SnapshotPackager(compression_level=config.index_build.compression_level)
    manifest = packager.package(
        build_dir=build_dir,
        output_dir=snapshot_dir,
        space_index_infos=space_index_infos,
        space_id_map_paths=space_id_map_paths,
        db_path=db_export_path,
        doc_count=exported_doc_count,
        chunk_count=exported_chunk_count,
        version=version,
        previous_manifest=state.last_manifest,
    )

    state.current_version = version
    state.last_manifest = manifest
    elapsed = time.monotonic() - t0

    logger.info(
        "build_complete",
        version=version,
        spaces=spaces,
        docs=doc_count,
        chunks=chunk_count,
        vectors=total_vectors,
        elapsed_s=round(elapsed, 1),
    )
    return BuildResponse(
        version=version,
        doc_count=doc_count,
        chunk_count=chunk_count,
        vector_count=total_vectors,
        num_shards=sum(idx.num_shards for idx in space_index_infos.values()),
        build_time_seconds=round(elapsed, 2),
    )


@router.post("/publish")
async def publish(request: Request) -> PublishResponse:
    state = _get_state(request)
    if not state.current_version or not state.last_manifest:
        raise ValueError("No build available to publish. Run /build first.")

    config = state.config
    client = ObjectStoreClient(config.object_store)
    publisher = SnapshotPublisher(client)

    snapshot_dir = str(state.build_dir / state.current_version)
    elapsed = publisher.publish(snapshot_dir, state.last_manifest)

    snapshot_size = state.last_manifest.total_compressed_size

    logger.info(
        "publish_complete",
        version=state.current_version,
        size_mb=round(snapshot_size / 1024 / 1024, 1),
        elapsed_s=round(elapsed, 1),
    )
    return PublishResponse(
        version=state.current_version,
        upload_time_seconds=round(elapsed, 2),
        snapshot_size_bytes=snapshot_size,
    )


VALID_ARCHIVE_FORMATS = {"tar.zst", "tar.zstd", "tar.gz", "tgz", "tar"}


@router.post("/build/archive")
async def build_archive(request: Request, body: ArchiveRequest = ArchiveRequest()) -> ArchiveResponse:
    """Create a downloadable archive (.tar.zst, .tar.gz, .tgz, .tar) from the latest build.

    Requires a prior ``/build`` call.  The archive is written next to the
    snapshot directory and can be served directly or uploaded to a CDN /
    GitHub Release.
    """
    state = _get_state(request)
    if not state.current_version or not state.last_manifest:
        raise ValueError("No build available. Run /build first.")

    fmt = body.format.lower().lstrip(".")
    if fmt not in VALID_ARCHIVE_FORMATS:
        raise ValueError(f"Unsupported format {body.format!r}. Choose from: {', '.join(sorted(VALID_ARCHIVE_FORMATS))}")

    from embedrag.shared.archive import create_snapshot_archive

    t0 = time.monotonic()
    snapshot_dir = str(state.build_dir / state.current_version)
    ext = fmt if fmt != "tgz" else "tar.gz"
    archive_name = f"{state.current_version}.{ext}"
    archive_path = str(state.build_dir / archive_name)

    size = create_snapshot_archive(
        snapshot_dir,
        archive_path,
        format=fmt,
        compression_level=body.compression_level,
    )
    elapsed = time.monotonic() - t0

    logger.info(
        "archive_built",
        version=state.current_version,
        format=fmt,
        path=archive_path,
        size_mb=round(size / 1024 / 1024, 2),
        elapsed_s=round(elapsed, 1),
    )
    return ArchiveResponse(
        version=state.current_version,
        format=fmt,
        path=archive_path,
        size_bytes=size,
        build_time_seconds=round(elapsed, 2),
    )
