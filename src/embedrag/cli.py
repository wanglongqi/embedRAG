"""CLI entry point for embedrag writer/query node.

This module provides the primary command-line interface (CLI) for managing
EmbedRAG nodes. It uses a sub-command pattern to provide different
functionalities such as starting servers, downloading remote snapshots, and
performing data migrations between versions.

The `embedrag` command is the single entry point for both operators (running
production nodes) and developers (migrating data or testing snapshots).
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """The main entry point for the `embedrag` command-line tool.

    This function parses command-line arguments using `argparse` and dispatches
    to the appropriate sub-command handler.

    Available Sub-commands:
        writer: Starts the Writer Node server for ingestion and indexing.
        query: Starts the Query Node server for serving search traffic.
        migrate: Upgrades local data/manifests to the current version.
        pull: Downloads and extracts snapshots from a remote URL.
    """
    parser = argparse.ArgumentParser(prog="embedrag", description="EmbedRAG server")
    sub = parser.add_subparsers(dest="command")

    # Writer
    wp = sub.add_parser("writer", help="Run the writer node")
    wp.add_argument("--config", "-c", default=None, help="Path to writer config YAML")
    wp.add_argument("--host", default="0.0.0.0")
    wp.add_argument("--port", type=int, default=8001)

    # Query
    qp = sub.add_parser("query", help="Run the query node")
    qp.add_argument("--config", "-c", default=None, help="Path to query config YAML")
    qp.add_argument("--host", default="0.0.0.0")
    qp.add_argument("--port", type=int, default=8000)

    # Migrate
    mp = sub.add_parser("migrate", help="Upgrade a snapshot to latest schema + manifest v3")
    mp.add_argument("path", help="Path to snapshot directory or embedrag.db file")
    mp.add_argument("--dry-run", action="store_true", help="Show current version without modifying")

    # Cluster
    cp = sub.add_parser("cluster", help="Cluster a set of inputs (standalone file or a writer DB)")
    src = cp.add_argument_group("input source (choose one)")
    src.add_argument("--input", help="Input file: .jsonl / .json / .csv of {id, text, [embedding]}")
    src.add_argument("--embeddings", help="Optional .npy of precomputed vectors aligned with --input rows")
    src.add_argument("--db", help="Writer SQLite DB path (reads exact vectors from chunk_embeddings)")
    cp.add_argument("--text-field", default="text", help="Field name for text (default: text)")
    cp.add_argument("--id-field", default="id", help="Field name for id (default: id)")
    cp.add_argument("--embedding-field", default="embedding", help="Field name for inline embedding")
    cp.add_argument("--space", default="text", help="Embedding space (DB source, default: text)")
    cp.add_argument("--filter", action="append", default=[], help="DB filter key=value (e.g. doc_type=complaint)")
    cp.add_argument("--algorithm", default="auto", help="auto|hdbscan|kmeans|agglomerative|dbscan|leiden")
    cp.add_argument("--reduce", default="auto", help="auto|none|pca|umap")
    cp.add_argument("--no-auto", action="store_true", help="Disable automatic parameter sweep")
    cp.add_argument("--min-cluster-size", type=int, help="HDBSCAN min_cluster_size")
    cp.add_argument("--k", type=int, help="KMeans/Agglomerative cluster count")
    cp.add_argument("--eps", type=float, help="DBSCAN eps")
    cp.add_argument("--min-samples", type=int, help="HDBSCAN/DBSCAN min_samples (density cluster core threshold)")
    cp.add_argument(
        "--cluster-selection-method",
        choices=["eom", "leaf"],
        help="HDBSCAN cluster selection method (Excess of Mass or Leaf)",
    )
    cp.add_argument("--distance-threshold", type=float, help="Agglomerative linkage distance merge threshold")
    cp.add_argument("--linkage", choices=["ward", "complete", "average", "single"], help="Agglomerative linkage type")
    cp.add_argument("--knn", type=int, help="Leiden k-nearest neighbors (default: 15)")
    cp.add_argument("--resolution", type=float, help="Leiden resolution parameter (default: 1.0)")
    cp.add_argument("--embed-url", help="OpenAI-compatible embeddings URL (vectorize text)")
    cp.add_argument("--embed-model", default="", help="Embedding model name")
    cp.add_argument("--embed-key", default="", help="Embedding API key")
    cp.add_argument("--llm-url", help="OpenAI-compatible chat URL for cluster labeling")
    cp.add_argument("--llm-model", default="", help="LLM model name")
    cp.add_argument("--llm-key", default="", help="LLM API key")
    cp.add_argument("--llm-language", default="auto", help="Label language (default: auto)")
    cp.add_argument("-o", "--output", help="Write result JSON to this path")
    cp.add_argument("--viz", help="Write a self-contained interactive HTML report to this path")

    # Pull
    pp = sub.add_parser("pull", help="Download a snapshot from a URL (GitHub Release, CDN, etc.)")
    pp.add_argument("url", help="Snapshot URL: archive (.tar.zst) or base URL with latest.json")
    pp.add_argument(
        "--output",
        "-o",
        default="./snapshot/active",
        help="Output directory (default: ./snapshot/active)",
    )
    pp.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Download timeout in seconds (default: 600)",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "cluster":
        _run_cluster(args)
        return

    if args.command == "pull":
        _run_pull(args.url, output=args.output, timeout=args.timeout)
        return

    if args.command == "migrate":
        _run_migrate(args.path, dry_run=args.dry_run)
        return

    import uvicorn

    if args.command == "writer":
        from embedrag.config import load_writer_config
        from embedrag.writer.app import create_writer_app

        writer_config = load_writer_config(args.config)
        app = create_writer_app(config_path=args.config)
        port = args.port if args.port != 8001 else (writer_config.server.port or 8001)
        uvicorn.run(app, host=args.host, port=port)
    elif args.command == "query":
        from embedrag.config import load_query_config
        from embedrag.query.app import create_query_app

        query_config = load_query_config(args.config)
        app = create_query_app(config_path=args.config)
        port = args.port if args.port != 8000 else (query_config.server.port or 8000)
        uvicorn.run(app, host=args.host, port=port)


def _run_cluster(args) -> None:
    """Run a clustering job from the CLI (standalone file or a writer DB).

    Resolves a vector source, optionally embeds text via a service or falls
    back to TF-IDF, clusters, optionally labels with an LLM, and writes JSON
    and/or a self-contained HTML report.
    """
    import asyncio

    import numpy as np

    from embedrag.cluster import apply_llm_labels, cluster_vectors, report
    from embedrag.cluster.source import (
        load_items_from_file,
        load_vectors_npy,
        read_writer_db,
        tfidf_vectors,
    )

    filters: dict = {}
    for kv in args.filter:
        if "=" in kv:
            k, v = kv.split("=", 1)
            filters[k.strip()] = v.strip()

    items = None
    vectors = None
    source_label = ""

    if args.db:
        items, vectors = read_writer_db(args.db, space=args.space, filters=filters)
        source_label = "writer-db"
    elif args.input:
        items, vectors = load_items_from_file(
            args.input,
            text_field=args.text_field,
            id_field=args.id_field,
            embedding_field=args.embedding_field,
        )
        source_label = "file"
        if args.embeddings:
            items, vectors = load_vectors_npy(args.embeddings, items)
            source_label = "file+npy"
    elif args.embeddings:
        items, vectors = load_vectors_npy(args.embeddings)
        source_label = "npy"
    else:
        print("Error: provide one of --input, --db, or --embeddings", file=sys.stderr)
        sys.exit(1)

    # If we still have no vectors, vectorize the text: embed service or TF-IDF.
    if vectors is None:
        texts = [it.text for it in items]
        if args.embed_url:
            from embedrag.config import EmbeddingSpaceConfig
            from embedrag.writer.embedding_client import EmbeddingClient

            cfg = EmbeddingSpaceConfig(
                service_url=args.embed_url, api_format="openai", model=args.embed_model, api_key=args.embed_key
            )
            client = EmbeddingClient(cfg)

            async def _embed():
                try:
                    return await client.embed_texts(texts)
                finally:
                    await client.close()

            vecs = asyncio.run(_embed())
            vectors = np.stack(vecs).astype("float32")
            source_label += "+embed"
        else:
            print("No embeddings found; using local TF-IDF representation.")
            vectors = tfidf_vectors(texts)
            source_label += "+tfidf"

    params: dict = {}
    if args.min_cluster_size:
        params["min_cluster_size"] = args.min_cluster_size
    if args.k:
        params["k"] = args.k
        params["n_clusters"] = args.k
    if args.eps:
        params["eps"] = args.eps
    if args.knn:
        params["knn"] = args.knn
    if args.resolution:
        params["resolution"] = args.resolution
    if args.min_samples:
        params["min_samples"] = args.min_samples
    if args.cluster_selection_method:
        params["cluster_selection_method"] = args.cluster_selection_method
    if args.distance_threshold:
        params["distance_threshold"] = args.distance_threshold
    if args.linkage:
        params["linkage"] = args.linkage

    result = cluster_vectors(
        vectors,
        items,
        algorithm=args.algorithm,
        reduce=args.reduce,
        auto=not args.no_auto,
        params=params,
        space=args.space,
        source=source_label,
    )

    if args.llm_url:
        print("Labeling clusters with LLM...")
        asyncio.run(
            apply_llm_labels(
                result, chat_url=args.llm_url, model=args.llm_model, api_key=args.llm_key, language=args.llm_language
            )
        )

    print(f"\nAlgorithm: {result.algorithm}  params: {result.params}")
    print(f"Clusters: {result.n_clusters}   Noise: {result.noise_count}   Items: {result.n_items}")
    print(
        f"Metrics: silhouette={result.metrics.get('silhouette')} "
        f"davies_bouldin={result.metrics.get('davies_bouldin')} noise_ratio={result.metrics.get('noise_ratio')}"
    )
    print("\nClusters:")
    for c in result.clusters:
        print(f"  [{c.cluster_id}] {c.label}  (size={c.size}, cohesion={c.cohesion})  kw: {', '.join(c.keywords[:6])}")

    if args.output:
        import json

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nWrote result JSON: {args.output}")

    if args.viz:
        report.write_html(result, args.viz)
        print(f"Wrote HTML report: {args.viz}")


def _run_pull(url: str, output: str = "./snapshot/active", timeout: int = 600) -> None:
    """Download a snapshot from a URL and prepare it for a query node.

    This function handles the complexities of downloading, extracting, and
    verifying snapshots. It supports both direct archive links (e.g., .tar.zst)
    and base URLs that follow the standard EmbedRAG layout with a `latest.json`
    index file.

    Args:
        url (str): The source URL. Can be a .tar.zst/gz archive or a base URL.
        output (str, optional): The local directory where the snapshot will be
            stored. Defaults to "./snapshot/active".
        timeout (int, optional): Network timeout for the download in seconds.
            Defaults to 600.
    """
    import shutil
    from pathlib import Path

    from embedrag.shared.archive import is_archive_url

    output_path = Path(output)

    if is_archive_url(url):
        from embedrag.shared.archive import download_and_extract_archive, verify_archive_snapshot

        print(f"Downloading archive: {url}")
        staging = str(output_path.parent / ".pull_staging")
        try:
            snap_dir = download_and_extract_archive(url, staging, timeout=timeout)
            manifest = verify_archive_snapshot(snap_dir)
            print(f"  Version:  {manifest.snapshot_version}")
            print(f"  Docs:     {manifest.db.doc_count}")
            print(f"  Chunks:   {manifest.db.chunk_count}")
            for space, idx in manifest.indexes.items():
                print(f"  Vectors:  {idx.total_vectors} ({space}, {idx.dim}-dim, {idx.num_shards} shards)")

            target = output_path / manifest.snapshot_version
            if target.exists():
                shutil.rmtree(str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            Path(snap_dir).rename(target)
            print(f"\nSnapshot ready at: {target}")
        finally:
            staging_path = Path(staging)
            if staging_path.exists():
                shutil.rmtree(str(staging_path), ignore_errors=True)
    else:
        from embedrag.models.manifest import Manifest
        from embedrag.shared.http_snapshot_client import HttpSnapshotClient

        print(f"Fetching from snapshot server: {url}")
        client = HttpSnapshotClient(url, timeout=timeout)
        latest = client.get_json("latest.json")
        if not latest:
            print("Error: No latest.json found at the given URL.")
            print("  If this is a direct archive URL, make sure it ends with .tar.zst or .tar.gz")
            sys.exit(1)

        version = latest.get("version", "")
        if not version:
            print("Error: latest.json has no 'version' field.")
            sys.exit(1)

        print(f"  Latest version: {version}")

        manifest_data = client.get_json(f"{version}/manifest.json")
        if not manifest_data:
            print(f"Error: Could not fetch {version}/manifest.json")
            sys.exit(1)

        manifest = Manifest.from_dict(manifest_data)
        print(f"  Docs:     {manifest.db.doc_count}")
        print(f"  Chunks:   {manifest.db.chunk_count}")
        for space, idx in manifest.indexes.items():
            print(f"  Vectors:  {idx.total_vectors} ({space}, {idx.dim}-dim, {idx.num_shards} shards)")

        target = output_path / version
        target.mkdir(parents=True, exist_ok=True)
        manifest.save(str(target / "manifest.json"))

        files_to_download = manifest.all_compressed_files()
        print(f"\nDownloading {len(files_to_download)} files...")

        for i, file_path in enumerate(files_to_download, 1):
            remote_key = f"{version}/{file_path}"
            local_path = str(target / file_path)
            print(f"  [{i}/{len(files_to_download)}] {file_path}")
            client.download_file(remote_key, local_path)

        print(f"\nSnapshot ready at: {target}")

    print("Start the query node with:")
    print("  uv run embedrag query --config <your-query.yaml>")
    print(f"  (set data_dir to point to: {output_path})")


def _run_migrate(path: str, dry_run: bool = False) -> None:
    """Upgrade a snapshot directory (manifest + DB) to the latest version.

    This utility ensures backward compatibility by upgrading data structures
    created with older versions of EmbedRAG to the current schema. It can
    handle upgrading SQLite schemas and restructuring manifest files (e.g.,
    converting v2 manifests to the v3 per-space format).

    Args:
        path (str): Path to the snapshot directory or a direct .db file.
        dry_run (bool, optional): If True, only prints what would be done
            without modifying any files. Defaults to False.
    """
    import json
    import shutil
    import sqlite3
    from pathlib import Path

    from embedrag.writer.schema import CURRENT_SCHEMA_VERSION, get_schema_version, initialize_schema

    target = Path(path)

    # Determine snapshot_dir and db_path
    if target.is_dir():
        snapshot_dir = target
        db_path = target / "db" / "embedrag.db"
    elif target.name.endswith(".db"):
        db_path = target
        snapshot_dir = target.parent.parent  # db/embedrag.db -> snapshot_dir
    else:
        print(f"Error: {path} is not a snapshot directory or .db file")
        sys.exit(1)

    manifest_path = snapshot_dir / "manifest.json"
    has_manifest = manifest_path.exists()
    has_db = db_path.exists()

    if not has_db and not has_manifest:
        print(f"Error: no embedrag.db or manifest.json found at {path}")
        sys.exit(1)

    # ── Report current state ──
    print(f"Snapshot dir: {snapshot_dir}")

    db_version = 0
    if has_db:
        conn = sqlite3.connect(str(db_path))
        db_version = get_schema_version(conn)
        conn.close()
        print(f"DB schema:    v{db_version} (target: v{CURRENT_SCHEMA_VERSION})")
    else:
        print(f"DB:           not found at {db_path}")

    manifest_version = 0
    if has_manifest:
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        manifest_version = manifest_data.get("manifest_version", 0)
        needs_manifest_upgrade = "index" in manifest_data and "indexes" not in manifest_data
        print(f"Manifest:     v{manifest_version}" + (" (needs v3 upgrade)" if needs_manifest_upgrade else " (ok)"))
    else:
        needs_manifest_upgrade = False
        print(f"Manifest:     not found at {manifest_path}")

    db_needs_upgrade = has_db and db_version < CURRENT_SCHEMA_VERSION

    if not db_needs_upgrade and not needs_manifest_upgrade:
        print("\nAlready up to date. Nothing to do.")
        return

    if dry_run:
        print("\nDry run: would perform the following:")
        if db_needs_upgrade:
            print(f"  - Migrate DB schema v{db_version} -> v{CURRENT_SCHEMA_VERSION}")
        if needs_manifest_upgrade:
            print(f"  - Convert manifest v{manifest_version} -> v3 (index->indexes, id_map->id_maps)")
        return

    # ── Perform migration ──
    if db_needs_upgrade:
        backup = str(db_path) + ".bak"
        print(f"\nBacking up DB to {backup} ...")
        shutil.copy2(str(db_path), backup)

        conn = sqlite3.connect(str(db_path))
        print(f"Migrating DB v{db_version} -> v{CURRENT_SCHEMA_VERSION} ...")
        try:
            initialize_schema(conn)
            conn.execute("VACUUM")
            conn.close()
            print("DB migration complete.")
        except Exception as e:
            conn.close()
            print(f"DB migration FAILED: {e}")
            shutil.copy2(backup, str(db_path))
            print("Restored DB from backup.")
            sys.exit(1)

    if needs_manifest_upgrade:
        backup = str(manifest_path) + ".bak"
        print(f"\nBacking up manifest to {backup} ...")
        shutil.copy2(str(manifest_path), backup)

        print("Upgrading manifest to v3 ...")
        _upgrade_manifest_v2_to_v3(manifest_data, manifest_path)
        print("Manifest upgrade complete.")

    print("\nDone.")


def _upgrade_manifest_v2_to_v3(data: dict, manifest_path) -> None:
    """Convert a v2 manifest to v3: restructure keys and move files to per-space dirs."""
    import json
    from pathlib import Path

    snapshot_dir = Path(manifest_path).parent

    def _move_file(old_rel: str, new_rel: str) -> None:
        old = snapshot_dir / old_rel
        new = snapshot_dir / new_rel
        if old.exists() and not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)

    # Convert index -> indexes
    if "index" in data and "indexes" not in data:
        data["indexes"] = {"text": data.pop("index")}

    # Convert id_map -> id_maps
    if "id_map" in data and "id_maps" not in data:
        data["id_maps"] = {"text": data.pop("id_map")}

    # Relocate files from index/ to index/text/ and update paths
    for space, idx in data.get("indexes", {}).items():
        for shard in idx.get("shards", []):
            for key in ("file", "compressed_file"):
                old_path = shard.get(key, "")
                if old_path and f"/{space}/" not in old_path:
                    new_path = old_path.replace("index/", f"index/{space}/", 1)
                    _move_file(old_path, new_path)
                    shard[key] = new_path

    for space, id_map in data.get("id_maps", {}).items():
        for key in ("file", "compressed_file"):
            old_path = id_map.get(key, "")
            if old_path and f"/{space}/" not in old_path:
                new_path = old_path.replace("index/", f"index/{space}/", 1)
                _move_file(old_path, new_path)
                id_map[key] = new_path

    data["manifest_version"] = 3
    data["schema_version"] = 3

    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
