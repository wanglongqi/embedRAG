"""Ingest all 120 chapters of 红楼梦 into EmbedRAG writer node.

Usage:
    python examples/hongloumeng/ingest.py [--writer-url http://localhost:8001] [--batch-size 5]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "hongloumeng"


def extract_chapter_title(text: str, chapter_num: int) -> str:
    """Extract the chapter title line (e.g. '第一回　甄士隱夢幻識通靈...')."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(f"第") and "回" in line[:10]:
            return line
    return f"第{chapter_num}回"


def clean_text(text: str) -> str:
    """Remove navigation lines and public domain notices."""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("", "回目录　下一回", "上一回　回目录　下一回", "上一回　回目录"):
            continue
        if "公有领域" in stripped or "Public domain" in stripped:
            continue
        if stripped.startswith("此清朝作品"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def main():
    parser = argparse.ArgumentParser(description="Ingest 红楼梦 into EmbedRAG")
    parser.add_argument("--writer-url", default="http://localhost:8001")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of chapters per /ingest call")
    args = parser.parse_args()

    chapter_files = sorted(DATA_DIR.glob("chapter_*.txt"))
    if not chapter_files:
        print(f"No chapter files found in {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(chapter_files)} chapters in {DATA_DIR}")

    # Check writer health
    try:
        resp = requests.get(f"{args.writer_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Writer node healthy: {resp.json()}")
    except Exception as e:
        print(f"Writer node unreachable at {args.writer_url}: {e}")
        sys.exit(1)

    total_ingested = 0
    total_chunks = 0
    t_start = time.time()

    for i in range(0, len(chapter_files), args.batch_size):
        batch_files = chapter_files[i : i + args.batch_size]
        documents = []

        for fpath in batch_files:
            chapter_num = int(fpath.stem.split("_")[1])
            raw_text = fpath.read_text(encoding="utf-8")
            text = clean_text(raw_text)
            title = extract_chapter_title(text, chapter_num)

            documents.append({
                "doc_id": f"hlm_ch{chapter_num:03d}",
                "title": title,
                "text": text,
                "doc_type": "novel_chapter",
                "chunking": "structured",
                "source": "红楼梦",
                "metadata": {
                    "book": "红楼梦",
                    "chapter": chapter_num,
                    "author": "曹雪芹",
                },
            })

        payload = {"documents": documents}
        batch_desc = f"chapters {i+1}-{i+len(batch_files)}"
        print(f"Ingesting {batch_desc}...", end=" ", flush=True)

        t_batch = time.time()
        try:
            resp = requests.post(
                f"{args.writer_url}/ingest",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            total_ingested += result["ingested"]
            total_chunks += result["chunk_count"]
            elapsed = time.time() - t_batch
            print(f"OK ({result['ingested']} docs, {result['chunk_count']} chunks, {elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"  Response: {e.response.text[:500]}")

    total_elapsed = time.time() - t_start
    print(f"\n=== Ingestion Complete ===")
    print(f"  Documents: {total_ingested}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Time:      {total_elapsed:.1f}s")

    # Build index
    print(f"\nBuilding FAISS index...", flush=True)
    t_build = time.time()
    try:
        resp = requests.post(f"{args.writer_url}/build", json={}, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        build_elapsed = time.time() - t_build
        print(f"Build complete:")
        print(f"  Version:  {result['version']}")
        print(f"  Docs:     {result['doc_count']}")
        print(f"  Chunks:   {result['chunk_count']}")
        print(f"  Vectors:  {result['vector_count']}")
        print(f"  Shards:   {result['num_shards']}")
        print(f"  Time:     {build_elapsed:.1f}s")
    except Exception as e:
        print(f"Build FAILED: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text[:500]}")


if __name__ == "__main__":
    main()
