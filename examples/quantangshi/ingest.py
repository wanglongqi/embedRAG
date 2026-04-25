"""Ingest 全唐诗 poems from JSONL into EmbedRAG writer node.

Each poem becomes a single document with 'plain' chunking (short texts).

Usage:
    python examples/quantangshi/ingest.py [--writer-url http://localhost:8001] [--batch-size 50]
    python examples/quantangshi/ingest.py --input data/quantangshi_poems.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "quantangshi_poems.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Ingest 全唐诗 into EmbedRAG")
    parser.add_argument("--writer-url", default="http://localhost:8001")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="JSONL file produced by download_poems.py")
    parser.add_argument("--batch-size", type=int, default=50, help="Poems per /ingest call")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        print("Run download_poems.py first.")
        sys.exit(1)

    poems = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                poems.append(json.loads(line))

    print(f"Loaded {len(poems)} poems from {args.input}")

    try:
        resp = requests.get(f"{args.writer_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Writer healthy: {resp.json()}")
    except Exception as e:
        print(f"Writer unreachable at {args.writer_url}: {e}")
        sys.exit(1)

    total_ingested = 0
    total_chunks = 0
    t_start = time.time()

    for i in range(0, len(poems), args.batch_size):
        batch = poems[i : i + args.batch_size]
        documents = []

        for p in batch:
            author = p.get("author", "").strip()
            title = p.get("title", "").strip()
            text = p.get("text", "").strip()
            vol = p.get("volume", 0)
            annotation = p.get("annotation", "").strip()
            bio = p.get("author_bio", "").strip()

            if not text:
                continue

            full_title = f"{title}" if not author else f"{author}·{title}"
            doc_id = f"qts_v{vol:03d}_{title[:20]}"

            full_text = text
            if annotation:
                full_text += f"\n\n【注】{annotation}"

            documents.append(
                {
                    "doc_id": doc_id,
                    "title": full_title,
                    "text": full_text,
                    "doc_type": "tang_poem",
                    "chunking": "plain",
                    "source": f"全唐诗·卷{vol:03d}",
                    "metadata": {
                        "author": author,
                        "volume": vol,
                        "collection": "全唐诗",
                    },
                }
            )

            if bio:
                documents.append(
                    {
                        "doc_id": f"qts_bio_{author[:10]}",
                        "title": f"{author}（传记）",
                        "text": bio,
                        "doc_type": "author_bio",
                        "chunking": "plain",
                        "source": f"全唐诗·卷{vol:03d}",
                        "metadata": {
                            "author": author,
                            "volume": vol,
                            "collection": "全唐诗",
                        },
                    }
                )

        if not documents:
            continue

        desc = f"poems {i + 1}-{i + len(batch)}"
        if (i // args.batch_size) % 20 == 0 or i + args.batch_size >= len(poems):
            print(f"  Ingesting {desc}...", end=" ", flush=True)

        t_batch = time.time()
        try:
            resp = requests.post(
                f"{args.writer_url}/ingest",
                json={"documents": documents},
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            total_ingested += result["ingested"]
            total_chunks += result["chunk_count"]
            elapsed = time.time() - t_batch
            if (i // args.batch_size) % 20 == 0 or i + args.batch_size >= len(poems):
                msg = f"OK ({result['ingested']} docs, " f"{result['chunk_count']} chunks, {elapsed:.1f}s)"
                print(msg)
        except Exception as e:
            print(f"FAILED at {desc}: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"  Response: {e.response.text[:500]}")

    total_elapsed = time.time() - t_start
    print("\n=== Ingestion Complete ===")
    print(f"  Documents: {total_ingested}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Time:      {total_elapsed:.1f}s")

    print("\nBuilding index...", flush=True)
    t_build = time.time()
    try:
        resp = requests.post(f"{args.writer_url}/build", json={}, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        be = time.time() - t_build
        print(
            f"Build complete: version={result['version']}, "
            f"docs={result['doc_count']}, chunks={result['chunk_count']}, "
            f"vectors={result['vector_count']}, shards={result['num_shards']}, "
            f"time={be:.1f}s"
        )
    except Exception as e:
        print(f"Build FAILED: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text[:500]}")


if __name__ == "__main__":
    main()
