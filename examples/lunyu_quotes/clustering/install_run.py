"""Install a CLI cluster run into the query node's side-file store.

The ``embedrag cluster`` CLI writes a result JSON (``-o``). The query node lists
and serves persisted runs from ``<data_dir>/cluster_runs/<run_id>.json`` (see
``embedrag.cluster.store``). Copying the CLI output there makes the run show up
in the ``/cluster`` web UI without re-running anything server-side.

For this example ``data_dir`` is ``examples/lunyu_quotes/snapshot`` (from
``query.yaml``), so runs land in ``snapshot/cluster_runs/``.

Usage:
    uv run python examples/lunyu_quotes/clustering/install_run.py results/run.json
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "snapshot"  # matches node.data_dir in query.yaml


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "run.json"
    if not src.is_absolute():
        src = (Path.cwd() / src) if (Path.cwd() / src).exists() else (HERE / src)
    if not src.exists():
        raise SystemExit(f"Run JSON not found: {src}")

    run_id = json.loads(src.read_text(encoding="utf-8")).get("run_id")
    if not run_id:
        raise SystemExit(f"{src} has no run_id field")

    runs_dir = DATA_DIR / "cluster_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dst = runs_dir / f"{run_id}.json"
    shutil.copyfile(src, dst)

    print(f"Installed run '{run_id}' -> {dst}")
    print("Start the query node and open the cluster UI to browse it:")
    print("  uv run embedrag query --config examples/lunyu_quotes/query.yaml")
    print("  open http://localhost:8000/cluster/")


if __name__ == "__main__":
    main()
