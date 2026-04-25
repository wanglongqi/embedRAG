"""Ingest 论语 + 庄子 into a single EmbedRAG index.

Demonstrates reading from multiple data directories into one unified index.
Each source gets a distinct doc_type so they can be filtered at query time.

Usage:
    python examples/classics/ingest.py [--writer-url http://localhost:8001]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent

SOURCES = [
    {
        "data_dir": PROJECT_ROOT / "data" / "lunyu",
        "book": "论语",
        "author": "孔子及弟子",
        "doc_type": "classic_lunyu",
        "doc_prefix": "lunyu",
        "chunking": "paragraph",
        "title_pattern": re.compile(r"^(.+第[一二三四五六七八九十百]+)$"),
    },
    {
        "data_dir": PROJECT_ROOT / "data" / "zhuangzi",
        "book": "庄子",
        "author": "庄周",
        "doc_type": "classic_zhuangzi",
        "doc_prefix": "zhuangzi",
        "chunking": "paragraph",
        "title_pattern": re.compile(r"^(.+第[一二三四五六七八九十百]+)$"),
    },
]

NOISE_LINES = {
    "姊妹计划: 数据项", "註疏", "返回頁首", "Public domainPublic domainfalsefalse",
}


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped in NOISE_LINES:
            continue
        if stripped.startswith("此") and "公有领域" in stripped:
            continue
        if "Public domain" in stripped:
            continue
        if re.match(r"^[◄►←→]", stripped):
            continue
        if re.match(r"^(上一卷|下一卷|卷\d+→|←卷\d+)$", stripped):
            continue
        if stripped.startswith("汉语普通话朗读") or stripped.startswith("此录音"):
            continue
        if stripped.startswith("更多有声文献") or stripped.startswith("收听本文"):
            continue
        if stripped.startswith("（完整的外部链接）"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def extract_title(text: str, pattern: re.Pattern, chapter_num: int, book: str) -> str:
    for line in text.split("\n")[:10]:
        stripped = line.strip()
        m = pattern.search(stripped)
        if m:
            return m.group(1)
    return f"{book} 第{chapter_num}章"


def main():
    parser = argparse.ArgumentParser(description="Ingest 论语+庄子 into EmbedRAG")
    parser.add_argument("--writer-url", default="http://localhost:8001")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    try:
        resp = requests.get(f"{args.writer_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Writer healthy: {resp.json()}")
    except Exception as e:
        print(f"Writer unreachable: {e}")
        sys.exit(1)

    total_docs = 0
    total_chunks = 0
    t_start = time.time()

    for src in SOURCES:
        data_dir = src["data_dir"]
        files = sorted(data_dir.glob("chapter_*.txt"))
        if not files:
            print(f"WARNING: No files in {data_dir}")
            continue

        print(f"\n--- {src['book']} ({len(files)} files from {data_dir.name}/) ---")

        documents = []
        for fpath in files:
            chapter_num = int(fpath.stem.split("_")[1])
            raw = fpath.read_text(encoding="utf-8")
            text = clean_text(raw)
            title = extract_title(text, src["title_pattern"], chapter_num, src["book"])

            documents.append({
                "doc_id": f"{src['doc_prefix']}_ch{chapter_num:03d}",
                "title": title,
                "text": text,
                "doc_type": src["doc_type"],
                "chunking": src["chunking"],
                "source": src["book"],
                "metadata": {
                    "book": src["book"],
                    "author": src["author"],
                    "chapter": chapter_num,
                },
            })

        for i in range(0, len(documents), args.batch_size):
            batch = documents[i : i + args.batch_size]
            desc = f"{src['book']} {i+1}-{i+len(batch)}"
            print(f"  Ingesting {desc}...", end=" ", flush=True)
            t = time.time()
            try:
                resp = requests.post(
                    f"{args.writer_url}/ingest", json={"documents": batch}, timeout=300
                )
                resp.raise_for_status()
                r = resp.json()
                total_docs += r["ingested"]
                total_chunks += r["chunk_count"]
                print(f"OK ({r['ingested']} docs, {r['chunk_count']} chunks, {time.time()-t:.1f}s)")
            except Exception as e:
                print(f"FAILED: {e}")

    elapsed = time.time() - t_start
    print(f"\n=== Ingestion Complete ===")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Time:      {elapsed:.1f}s")

    print(f"\nBuilding index...", flush=True)
    t = time.time()
    try:
        resp = requests.post(f"{args.writer_url}/build", json={}, timeout=600)
        resp.raise_for_status()
        r = resp.json()
        print(f"Build complete: version={r['version']}, docs={r['doc_count']}, "
              f"chunks={r['chunk_count']}, vectors={r['vector_count']}, "
              f"shards={r['num_shards']}, time={time.time()-t:.1f}s")
    except Exception as e:
        print(f"Build FAILED: {e}")


if __name__ == "__main__":
    main()
