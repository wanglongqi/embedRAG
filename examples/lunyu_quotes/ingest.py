"""Ingest 论语 quotes as independent flat texts (no hierarchy).

Each numbered saying (e.g. "一之一", "一之二") becomes a separate document
with ``plain`` chunking and no parent-child relationships. This demonstrates
that EmbedRAG works correctly for non-hierarchical, short, independent texts.

Usage:
    python examples/lunyu_quotes/ingest.py [--writer-url http://localhost:8001]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "lunyu"

NOISE_LINES = {
    "姊妹计划: 数据项",
    "註疏",
    "返回頁首",
    "Public domainPublic domainfalsefalse",
}

CHAPTER_NAMES = {
    1: "學而",
    2: "爲政",
    3: "八佾",
    4: "里仁",
    5: "公冶長",
    6: "雍也",
    7: "述而",
    8: "泰伯",
    9: "子罕",
    10: "鄉黨",
    11: "先進",
    12: "顏淵",
    13: "子路",
    14: "憲問",
    15: "衛靈公",
    16: "季氏",
    17: "陽貨",
    18: "微子",
    19: "子張",
    20: "堯曰",
}

QUOTE_NUM = re.compile(r"^[一二三四五六七八九十百]+之[一二三四五六七八九十百]+$")


def parse_quotes(chapter_num: int, text: str) -> list[dict]:
    """Parse a chapter file into individual quotes."""
    lines = text.split("\n")
    quotes = []
    current_id = None
    current_lines: list[str] = []

    def flush():
        if current_id and current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                quotes.append(
                    {
                        "quote_id": current_id,
                        "text": body,
                    }
                )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in NOISE_LINES:
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
        if stripped.startswith("此") and "公有领域" in stripped:
            continue

        if QUOTE_NUM.match(stripped):
            flush()
            current_id = stripped
            current_lines = []
        elif current_id:
            current_lines.append(stripped)
        elif re.search(r"第[一二三四五六七八九十百]+$", stripped):
            continue
        elif stripped in ("論語", "序說"):
            continue

    flush()
    return quotes


def main():
    parser = argparse.ArgumentParser(description="Ingest 论语 quotes as flat texts")
    parser.add_argument("--writer-url", default="http://localhost:8001")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    try:
        resp = requests.get(f"{args.writer_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Writer healthy: {resp.json()}")
    except Exception as e:
        print(f"Writer unreachable: {e}")
        sys.exit(1)

    all_docs = []
    for fpath in sorted(DATA_DIR.glob("chapter_*.txt")):
        chapter_num = int(fpath.stem.split("_")[1])
        raw = fpath.read_text(encoding="utf-8")
        quotes = parse_quotes(chapter_num, raw)
        chapter_name = CHAPTER_NAMES.get(chapter_num, f"第{chapter_num}篇")

        for q in quotes:
            doc_id = f"lunyu_{chapter_num:02d}_{q['quote_id']}"
            all_docs.append(
                {
                    "doc_id": doc_id,
                    "title": f"論語·{chapter_name}·{q['quote_id']}",
                    "text": q["text"],
                    "doc_type": "lunyu_quote",
                    "chunking": "plain",
                    "source": "論語",
                    "metadata": {
                        "book": "論語",
                        "chapter": chapter_num,
                        "chapter_name": chapter_name,
                        "quote_id": q["quote_id"],
                    },
                }
            )

    print(f"Parsed {len(all_docs)} quotes from {len(list(DATA_DIR.glob('chapter_*.txt')))} chapters")

    total_docs = 0
    total_chunks = 0
    t_start = time.time()

    for i in range(0, len(all_docs), args.batch_size):
        batch = all_docs[i : i + args.batch_size]
        desc = f"quotes {i + 1}-{i + len(batch)}"
        print(f"  Ingesting {desc}...", end=" ", flush=True)
        t = time.time()
        try:
            resp = requests.post(f"{args.writer_url}/ingest", json={"documents": batch}, timeout=300)
            resp.raise_for_status()
            r = resp.json()
            total_docs += r["ingested"]
            total_chunks += r["chunk_count"]
            print(f"OK ({r['ingested']} docs, {r['chunk_count']} chunks, {time.time() - t:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")

    elapsed = time.time() - t_start
    print("\n=== Ingestion Complete ===")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Time:      {elapsed:.1f}s")
    print("  (Each quote is 1 doc = 1 chunk, no hierarchy)")

    print("\nBuilding index...", flush=True)
    t = time.time()
    try:
        resp = requests.post(f"{args.writer_url}/build", json={}, timeout=600)
        resp.raise_for_status()
        r = resp.json()
        print(
            f"Build complete: version={r['version']}, docs={r['doc_count']}, "
            f"chunks={r['chunk_count']}, vectors={r['vector_count']}, "
            f"shards={r['num_shards']}, time={time.time() - t:.1f}s"
        )
    except Exception as e:
        print(f"Build FAILED: {e}")


if __name__ == "__main__":
    main()
