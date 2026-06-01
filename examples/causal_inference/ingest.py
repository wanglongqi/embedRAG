"""Ingest two causal-inference books (Chinese + English) into a mixed-language index.

Books:
  1. Judea Pearl "为什么：关于因果关系的新科学" (zh)
  2. Matheus Facure "Causal Inference in Python" (en)

Usage:
    # Start the writer first:
    #   uv run embedrag writer --config examples/causal_inference/writer.yaml
    # Then run:
    python examples/causal_inference/ingest.py [--writer-url http://localhost:8001]
"""

import argparse
import re
import sys
import time
import warnings
import zipfile
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DOWNLOADS = Path.home() / "Downloads"

PEARL_EPUB = DOWNLOADS / "Judea Pearl_ Dana MacKenzie - 为什么：关于因果关系的新科学.epub"
PYTHON_EPUB = DOWNLOADS / "Causal Inference in Python.epub"

_MULTI_NEWLINE = re.compile(r"\n{3,}")
_FIGURE_NOISE = re.compile(r"Figure\s+\d+-\d+\.\s*\n?", re.IGNORECASE)


def parse_epub_ebooklib(path: Path) -> list[dict]:
    """Parse epub using ebooklib (works for well-formed epubs)."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        html = item.get_content().decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 100:
            continue
        title_tag = soup.find(["h1", "h2"])
        title = title_tag.get_text(strip=True) if title_tag else item.get_name()
        chapters.append({"title": title, "text": text, "source_file": item.get_name()})
    return chapters


def parse_epub_zipfile(path: Path, chapter_pattern: str = "/ch") -> list[dict]:
    """Fallback parser for epubs with broken OPF manifests."""
    chapters = []
    with zipfile.ZipFile(path) as zf:
        xhtml_files = sorted(n for n in zf.namelist() if n.endswith((".xhtml", ".html")) and chapter_pattern in n)
        for name in xhtml_files:
            html = zf.read(name).decode("utf-8", errors="replace")
            soup = BeautifulSoup(html, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            if len(text) < 100:
                continue
            title_tag = soup.find(["h1", "h2"])
            title = title_tag.get_text(strip=True) if title_tag else name
            chapters.append({"title": title, "text": text, "source_file": name})
    return chapters


def clean_text(text: str) -> str:
    text = _FIGURE_NOISE.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def load_pearl() -> list[dict]:
    """Load 为什么 (Judea Pearl, Chinese)."""
    if not PEARL_EPUB.exists():
        print(f"  NOT FOUND: {PEARL_EPUB}")
        return []
    print(f"  Parsing: {PEARL_EPUB.name}")
    chapters = parse_epub_ebooklib(PEARL_EPUB)
    documents = []
    for i, ch in enumerate(chapters):
        text = clean_text(ch["text"])
        if len(text) < 200:
            continue
        documents.append(
            {
                "doc_id": f"pearl_zh_{i:03d}",
                "title": ch["title"],
                "text": text,
                "doc_type": "book_chapter",
                "chunking": "structured",
                "source": "为什么：关于因果关系的新科学",
                "metadata": {
                    "book": "为什么：关于因果关系的新科学",
                    "author": "Judea Pearl, Dana MacKenzie",
                    "language": "zh",
                    "chapter_index": i,
                },
            }
        )
    return documents


def load_python_causal() -> list[dict]:
    """Load Causal Inference in Python (English)."""
    if not PYTHON_EPUB.exists():
        print(f"  NOT FOUND: {PYTHON_EPUB}")
        return []
    print(f"  Parsing: {PYTHON_EPUB.name}")
    try:
        chapters = parse_epub_ebooklib(PYTHON_EPUB)
    except Exception:
        chapters = parse_epub_zipfile(PYTHON_EPUB)
    documents = []
    for i, ch in enumerate(chapters):
        text = clean_text(ch["text"])
        if len(text) < 200:
            continue
        documents.append(
            {
                "doc_id": f"facure_en_{i:03d}",
                "title": ch["title"],
                "text": text,
                "doc_type": "book_chapter",
                "chunking": "structured",
                "source": "Causal Inference in Python",
                "metadata": {
                    "book": "Causal Inference in Python",
                    "author": "Matheus Facure",
                    "language": "en",
                    "chapter_index": i,
                },
            }
        )
    return documents


def ingest_batch(url: str, documents: list[dict], batch_size: int = 3):
    total_ingested = 0
    total_chunks = 0
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        titles = [d["title"][:30] for d in batch]
        print(
            f"  Ingesting batch {i // batch_size + 1} ({len(batch)} docs: {titles})...",
            end=" ",
            flush=True,
        )
        t0 = time.time()
        resp = requests.post(f"{url}/ingest", json={"documents": batch}, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        total_ingested += result["ingested"]
        total_chunks += result["chunk_count"]
        print(f"OK ({result['chunk_count']} chunks, {time.time() - t0:.1f}s)")
    return total_ingested, total_chunks


def main():
    parser = argparse.ArgumentParser(description="Ingest causal inference books")
    parser.add_argument("--writer-url", default="http://localhost:8001")
    parser.add_argument("--batch-size", type=int, default=3)
    args = parser.parse_args()

    try:
        resp = requests.get(f"{args.writer_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"Writer healthy: {resp.json()}")
    except Exception as e:
        print(f"Writer unreachable at {args.writer_url}: {e}")
        sys.exit(1)

    t_start = time.time()
    all_docs = []

    print("\n[1/2] Loading 为什么：关于因果关系的新科学...")
    all_docs.extend(load_pearl())

    print("\n[2/2] Loading Causal Inference in Python...")
    all_docs.extend(load_python_causal())

    if not all_docs:
        print("No documents loaded. Check epub paths.")
        sys.exit(1)

    zh_count = sum(1 for d in all_docs if d["metadata"]["language"] == "zh")
    en_count = sum(1 for d in all_docs if d["metadata"]["language"] == "en")
    print(f"\nLoaded {len(all_docs)} documents ({zh_count} zh, {en_count} en)")

    print("\nIngesting...")
    total_docs, total_chunks = ingest_batch(args.writer_url, all_docs, args.batch_size)

    print("\n=== Ingestion Complete ===")
    print(f"  Documents: {total_docs}")
    print(f"  Chunks:    {total_chunks}")
    print(f"  Time:      {time.time() - t_start:.1f}s")

    print("\nBuilding FAISS index...", flush=True)
    t_build = time.time()
    try:
        resp = requests.post(f"{args.writer_url}/build", json={}, timeout=600)
        resp.raise_for_status()
        result = resp.json()
        print("Build complete:")
        print(f"  Version:  {result['version']}")
        print(f"  Docs:     {result['doc_count']}")
        print(f"  Chunks:   {result['chunk_count']}")
        print(f"  Vectors:  {result['vector_count']}")
        print(f"  Shards:   {result['num_shards']}")
        print(f"  Time:     {time.time() - t_build:.1f}s")
    except Exception as e:
        print(f"Build FAILED: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"  Response: {e.response.text[:500]}")


if __name__ == "__main__":
    main()
