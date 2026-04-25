#!/usr/bin/env bash
# Hot-swap demo: zero-downtime data reload on the query node.
#
# This script demonstrates:
# 1. Start a query node with the current classics snapshot
# 2. Re-ingest with modified data (add extra metadata)
# 3. Build a new snapshot version
# 4. Copy new snapshot to query node's active directory
# 5. Hot-swap via POST /admin/reload -- zero downtime
#
# Prerequisites:
#   - An embedding service running at http://127.0.0.1:1234
#   - An existing classics snapshot (run ingest.py first if missing)
#
# Usage:
#   bash examples/classics/hot_swap_demo.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUERY_URL="http://localhost:8000"
WRITER_URL="http://localhost:8001"

echo "=== Hot-Swap Demo ==="
echo ""

# ── Step 1: Verify query node is running ──
echo "Step 1: Checking query node..."
if ! curl -sf "$QUERY_URL/health" > /dev/null 2>&1; then
    echo "  Query node not running. Start it first:"
    echo "    uv run embedrag query --config examples/classics/query.yaml"
    echo ""
    echo "  Then run this script again."
    exit 1
fi

VERSION_BEFORE=$(curl -sf "$QUERY_URL/readiness" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_version','?'))")
echo "  Current version: $VERSION_BEFORE"

# ── Step 2: Start writer and re-ingest ──
echo ""
echo "Step 2: Starting writer node in background..."
uv run embedrag writer --config "$SCRIPT_DIR/writer.yaml" &
WRITER_PID=$!
sleep 3

if ! curl -sf "$WRITER_URL/health" > /dev/null 2>&1; then
    echo "  Writer failed to start. Check logs."
    kill "$WRITER_PID" 2>/dev/null || true
    exit 1
fi
echo "  Writer running (PID=$WRITER_PID)"

echo ""
echo "Step 3: Ingesting data..."
uv run python "$SCRIPT_DIR/ingest.py" --writer-url "$WRITER_URL"

# ── Step 4: Copy new snapshot to query node ──
SNAPSHOT_DIR="$SCRIPT_DIR/snapshot"
ACTIVE_DIR="$SNAPSHOT_DIR/active"
BUILD_DIR="/tmp/embedrag-classics/builds"

echo ""
echo "Step 4: Copying new snapshot to query node..."
LATEST_BUILD=$(ls -td "$BUILD_DIR"/v* 2>/dev/null | head -1)
if [ -z "$LATEST_BUILD" ]; then
    echo "  No build found. The build step may have failed."
    kill "$WRITER_PID" 2>/dev/null || true
    exit 1
fi

VERSION_NEW=$(basename "$LATEST_BUILD")
echo "  New version: $VERSION_NEW"

mkdir -p "$ACTIVE_DIR"
cp -r "$LATEST_BUILD" "$ACTIVE_DIR/$VERSION_NEW"
echo "  Copied to $ACTIVE_DIR/$VERSION_NEW"

# ── Step 5: Hot-swap ──
echo ""
echo "Step 5: Triggering hot-swap via /admin/reload..."
RELOAD_RESP=$(curl -sf -X POST "$QUERY_URL/admin/reload" \
    -H "Content-Type: application/json" \
    -d "{\"snapshot_dir\": \"$ACTIVE_DIR/$VERSION_NEW\"}")
echo "  Response: $RELOAD_RESP"

VERSION_AFTER=$(curl -sf "$QUERY_URL/readiness" | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_version','?'))")
echo ""
echo "=== Result ==="
echo "  Before: $VERSION_BEFORE"
echo "  After:  $VERSION_AFTER"

if [ "$VERSION_BEFORE" != "$VERSION_AFTER" ]; then
    echo "  Hot-swap successful! Zero-downtime data update."
else
    echo "  Version unchanged (may be same data)."
fi

# ── Step 6: Verify search works ──
echo ""
echo "Step 6: Verifying search..."
curl -sf -X POST "$QUERY_URL/search/text" \
    -H "Content-Type: application/json" \
    -d '{"query_text": "仁义", "top_k": 3, "mode": "hybrid"}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'  Found {data[\"total\"]} results')
for c in data['chunks'][:3]:
    text = c['text'][:60].replace(chr(10), ' ')
    print(f'    [{c[\"score\"]:.4f}] {text}...')
"

# Cleanup
echo ""
echo "Stopping writer..."
kill "$WRITER_PID" 2>/dev/null || true
wait "$WRITER_PID" 2>/dev/null || true
echo "Done."
