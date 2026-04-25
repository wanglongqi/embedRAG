"""CLI entry point for embedrag writer/query node."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
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

    # Pull
    pp = sub.add_parser("pull", help="Download a snapshot from a URL (GitHub Release, CDN, etc.)")
    pp.add_argument("url", help="Snapshot URL: archive (.tar.zst) or base URL with latest.json")
    pp.add_argument(
        "--output", "-o", default="./snapshot/active",
        help="Output directory (default: ./snapshot/active)",
    )
    pp.add_argument(
        "--timeout", type=int, default=600,
        help="Download timeout in seconds (default: 600)",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "pull":
        _run_pull(args.url, output=args.output, timeout=args.timeout)
        return

    if args.command == "migrate":
        _run_migrate(args.path, dry_run=args.dry_run)
        return

    import uvicorn

    if args.command == "writer":
        from embedrag.writer.app import create_writer_app

        app = create_writer_app(config_path=args.config)
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "query":
        from embedrag.query.app import create_query_app

        app = create_query_app(config_path=args.config)
        uvicorn.run(app, host=args.host, port=args.port)


def _run_pull(url: str, output: str = "./snapshot/active", timeout: int = 600) -> None:
    """Download a snapshot from a URL and place it ready for a query node.

    Supports two URL types:
    - Direct archive: .tar.zst, .tar.gz, .tgz, .tar (e.g. GitHub Release asset)
    - Base URL: standard embedrag snapshot layout with latest.json
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

    Accepts either a snapshot directory (containing manifest.json + db/) or
    a direct path to a embedrag.db file.
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
        print(
            f"Manifest:     v{manifest_version}"
            + (" (needs v3 upgrade)" if needs_manifest_upgrade else " (ok)")
        )
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
            print(
                f"  - Convert manifest v{manifest_version} -> v3 (index->indexes, id_map->id_maps)"
            )
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
