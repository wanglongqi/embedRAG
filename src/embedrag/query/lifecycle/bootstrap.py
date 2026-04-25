"""Cold-start bootstrap: from config to ready."""

from __future__ import annotations

from pathlib import Path

from embedrag.logging_setup import get_logger
from embedrag.models.manifest import Manifest
from embedrag.query.lifecycle.startup import load_generation, quick_verify_snapshot
from embedrag.shared.disk import check_disk_space
from embedrag.shared.object_store import ObjectStoreClient

logger = get_logger(__name__)


class BootstrapError(RuntimeError):
    """Raised when the query node cannot load any snapshot on startup."""


async def bootstrap_query_node(state) -> None:
    """Bootstrap the query node from cold start.

    1. Check if active snapshot exists locally.
    2. If yes, quick-verify and load it.
    3. If no, download from object store and load.

    Raises BootstrapError on failure so the process exits with a clear message
    instead of starting in a broken state.
    """
    config = state.config
    data_dir = Path(config.node.data_dir)
    active_dir = data_dir / "active"
    staging_dir = data_dir / "staging"
    backup_dir = data_dir / "backup"

    for d in [active_dir, staging_dir, backup_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Check for existing active snapshot
    active_manifest_path = _find_active_manifest(active_dir)
    if active_manifest_path:
        manifest = Manifest.load(active_manifest_path)
        snap_dir = str(active_manifest_path.parent)
        if quick_verify_snapshot(snap_dir, manifest):
            logger.info("bootstrap_local", version=manifest.snapshot_version)
            ctx = load_generation(
                snap_dir,
                manifest,
                nprobe=config.index.nprobe,
                use_mmap=config.index.mmap,
            )
            await state.gen_manager.swap(ctx)
            return
        else:
            logger.error(
                "bootstrap_local_corrupt",
                version=manifest.snapshot_version,
                snapshot_dir=snap_dir,
            )
            raise BootstrapError(
                f"Local snapshot {manifest.snapshot_version} at {snap_dir} "
                f"failed integrity check. Delete or re-download it."
            )

    # No local snapshot -- try cold start from object store
    logger.info("bootstrap_cold_start", data_dir=str(active_dir))
    try:
        client = ObjectStoreClient(config.object_store)
        version = _resolve_version(client, config.snapshot.bootstrap_version)
        if not version:
            raise BootstrapError(
                "No snapshot version found in object store (latest.json missing or empty). "
                "Publish a snapshot first or place one locally."
            )

        snap_dir = str(staging_dir / version)
        await _download_snapshot(client, version, snap_dir, config)

        manifest = Manifest.load(Path(snap_dir) / "manifest.json")

        # Move to active
        target = active_dir / version
        if target.exists():
            import shutil

            shutil.rmtree(str(target))
        Path(snap_dir).rename(target)

        ctx = load_generation(
            str(target),
            manifest,
            nprobe=config.index.nprobe,
            use_mmap=config.index.mmap,
        )
        await state.gen_manager.swap(ctx)
        logger.info("bootstrap_complete", version=version)
    except BootstrapError:
        raise
    except Exception as exc:
        logger.exception("bootstrap_failed")
        raise BootstrapError(
            f"Failed to bootstrap: {exc}\n"
            f"  Hint: place a snapshot at {active_dir}/<version>/manifest.json "
            f"or configure a valid object_store."
        ) from exc


def _find_active_manifest(active_dir: Path) -> Path | None:
    """Find the most recent manifest.json in active directory."""
    manifests = list(active_dir.glob("*/manifest.json"))
    if not manifests:
        return None
    manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests[0]


def _resolve_version(client: ObjectStoreClient, bootstrap_version: str) -> str | None:
    if bootstrap_version == "latest":
        latest = client.get_json("latest.json")
        if latest:
            return latest.get("version", "")
        return None
    return bootstrap_version


async def _download_snapshot(
    client: ObjectStoreClient, version: str, snap_dir: str, config
) -> None:
    """Download a full snapshot from object store."""
    Path(snap_dir).mkdir(parents=True, exist_ok=True)

    # Download manifest first
    client.download_file(f"{version}/manifest.json", f"{snap_dir}/manifest.json")
    manifest = Manifest.load(f"{snap_dir}/manifest.json")

    # Disk check
    needed = manifest.total_compressed_size + manifest.total_raw_size
    ok, available = check_disk_space(snap_dir, needed, config.snapshot.disk_reserve_bytes)
    if not ok:
        raise RuntimeError(f"Insufficient disk: need {needed} + reserve, have {available}")

    # Download all compressed files (per-space shards + id_maps)
    for idx_info in manifest.indexes.values():
        for shard in idx_info.shards:
            remote = f"{version}/{shard.compressed_file}"
            local = f"{snap_dir}/{shard.compressed_file}"
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            client.download_file(remote, local)

    remote_db = f"{version}/{manifest.db.compressed_file}"
    local_db = f"{snap_dir}/{manifest.db.compressed_file}"
    Path(local_db).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(remote_db, local_db)

    for idm in manifest.id_maps.values():
        remote_idmap = f"{version}/{idm.compressed_file}"
        local_idmap = f"{snap_dir}/{idm.compressed_file}"
        Path(local_idmap).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(remote_idmap, local_idmap)
