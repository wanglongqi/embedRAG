"""Download 全唐诗 from Wikisource and extract individual poems as JSONL.

Each poem becomes a JSON object with: author, title, text, volume, annotation.

Usage:
    python examples/quantangshi/download_poems.py [--start 1] [--end 900] [--output data/quantangshi_poems.jsonl]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Biography keywords that typically appear in author bio paragraphs
_BIO_MARKERS = re.compile(r"(字[^\s，]{1,4}|號[^\s，]{1,6}|人\s*[。，]|詩\s*\d+\s*首|"
                          r"貞觀|開元|天寶|大曆|元和|長慶|會昌)")

NOISE = {
    "返回頁首", "上一卷", "下一卷", "全唐詩", "姊妹计划: 数据项", "目录",
}


def fetch_volume_html(vol_num: int) -> str:
    encoded_vol = urllib.parse.quote(f"卷{vol_num:03d}")
    url = f"https://zh.wikisource.org/wiki/%E5%85%A8%E5%94%90%E8%A9%A9/{encoded_vol}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; miniRAG-QTS/1.0)"
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def extract_content_block(html: str) -> str:
    start = html.find('class="mw-content-ltr mw-parser-output"')
    if start == -1:
        return ""
    start = html.find(">", start) + 1
    end = html.find('<div class="printfooter"', start)
    if end == -1:
        end = len(html)
    return html[start:end]


def strip_html(s: str) -> str:
    s = re.sub(r"<sup[^>]*>.*?</sup>", "", s, flags=re.DOTALL)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    for old, new in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")]:
        s = s.replace(old, new)
    return s


def is_noise(line: str) -> bool:
    if not line:
        return True
    if line in NOISE:
        return True
    if "公有领域" in line or "Public domain" in line:
        return True
    if re.match(r"^[◄►←→↑]", line):
        return True
    if re.match(r"^(←卷|卷\d+→)", line):
        return True
    if re.match(r"^全唐詩\s+卷", line):
        return True
    if line.startswith("作者："):
        return True
    return False


def _clean_lines(raw_html: str) -> list[str]:
    """Strip HTML, remove noise, remove [编辑] artifacts."""
    text = strip_html(raw_html)
    lines = []
    for l in text.split("\n"):
        l = l.strip()
        l = l.replace("[编辑]", "").strip()
        if l and not is_noise(l):
            lines.append(l)
    return lines


def _extract_bio(lines: list[str]) -> tuple[str, int]:
    """If the first line looks like an author bio, return (bio, start_index)."""
    if not lines:
        return "", 0
    first = lines[0]
    if len(first) > 25 and ("。" in first or _BIO_MARKERS.search(first)):
        return first, 1
    return "", 0


def _heading_is_author(heading_text: str, following_html: str) -> bool:
    """Determine if an h2 heading is an author name (vs. a poem group title).

    Strategy:
    1. If the text after the heading contains biography markers -> author.
    2. If the heading itself contains poem-title markers -> poem title.
    3. Short headings (<=6 chars) are likely author names.
    """
    following = strip_html(following_html)[:200]
    if _BIO_MARKERS.search(following):
        return True

    poem_markers = re.compile(r"[首篇章曲歌行賦序詠贈送別題遊過宿登]")
    if poem_markers.search(heading_text) and len(heading_text) > 4:
        return False

    if re.search(r"\d", heading_text):
        return False

    return len(heading_text) <= 10


def parse_volume(html: str, vol_num: int) -> list[dict]:
    content = extract_content_block(html)
    if not content:
        return []

    content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)

    # Split by h2/h3 headings, keeping the delimiters
    parts = re.split(r"(<h[23][^>]*>.*?</h[23]>)", content, flags=re.DOTALL)

    # Extract default author from the header table "作者：<name>".
    # Multi-author volumes list all names space-separated; only use
    # the header if it looks like a single author (short, no spaces
    # within the name portion).
    default_author = ""
    author_m = re.search(r"作者[：:]</span>(.*?)(?:\n|</td>)", content, re.DOTALL)
    if author_m:
        raw_author = strip_html(author_m.group(1)).strip()
        # Single author: typically <= 10 chars with no internal spaces
        if raw_author and len(raw_author) <= 10 and " " not in raw_author:
            default_author = raw_author

    if not default_author and parts:
        preamble = strip_html(parts[0]).strip()
        m = re.search(r"卷[一二三四五六七八九十百零○\d]+\s+(.+?)(?:\n|$)", preamble)
        if m:
            candidate = m.group(1).strip()
            # Exclude volume navigation artifacts
            if (candidate and len(candidate) <= 15
                    and not candidate.startswith("全唐詩")
                    and not candidate.startswith("卷")
                    and "→" not in candidate):
                default_author = candidate

    # Two-pass: first classify h2 headings, then extract poems
    # Build a structured list: (type, heading_text, content_html)
    sections: list[tuple[str, str, str]] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        h2 = re.search(r"<h2[^>]*>(.*?)</h2>", part, re.DOTALL)
        h3 = re.search(r"<h3[^>]*>(.*?)</h3>", part, re.DOTALL)

        if h2:
            raw = strip_html(h2.group(1)).replace("[编辑]", "").strip()
            following = parts[i + 1] if i + 1 < len(parts) else ""
            if raw and raw != "目录":
                # If the header gives a single author, all h2s are poem titles.
                # Multi-author volumes have no header author; classify by heuristic.
                if default_author:
                    sections.append(("poem_group", raw, ""))
                elif _heading_is_author(raw, following):
                    sections.append(("author", raw, ""))
                else:
                    sections.append(("poem_group", raw, ""))
        elif h3:
            raw = strip_html(h3.group(1)).replace("[编辑]", "").strip()
            if raw:
                sections.append(("poem", raw, ""))
        else:
            sections.append(("text", "", part))
        i += 1

    # Now walk sections and build poems
    poems: list[dict] = []
    current_author = default_author
    current_title = ""
    text_parts: list[str] = []

    def flush():
        nonlocal current_title, text_parts
        if not current_title:
            text_parts = []
            return

        raw_html = "".join(text_parts)
        text_parts = []

        anns = re.findall(r"〈(.*?)〉", raw_html, re.DOTALL)
        annotation = " ".join(strip_html(a).strip() for a in anns if a.strip())
        raw_html = re.sub(r"〈.*?〉", "", raw_html, flags=re.DOTALL)

        lines = _clean_lines(raw_html)
        bio, start = _extract_bio(lines)
        poem_text = "\n".join(lines[start:])

        if poem_text.strip():
            poems.append({
                "author": current_author,
                "title": current_title,
                "text": poem_text.strip(),
                "volume": vol_num,
                "annotation": annotation,
                "author_bio": bio,
            })

    for stype, heading, content_html in sections:
        if stype == "author":
            flush()
            current_author = heading
            current_title = ""
        elif stype == "poem_group":
            flush()
            current_title = heading
        elif stype == "poem":
            flush()
            current_title = heading
        else:
            text_parts.append(content_html)

    flush()
    return poems


def main():
    parser = argparse.ArgumentParser(description="Download 全唐诗 poems as JSONL")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=900)
    parser.add_argument("--output", default="data/quantangshi_poems.jsonl")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between requests (seconds)")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_poems = 0
    failed = 0

    with open(output, "w", encoding="utf-8") as f:
        for vol in range(args.start, args.end + 1):
            try:
                html = fetch_volume_html(vol)
                poems = parse_volume(html, vol)
                for p in poems:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                total_poems += len(poems)
                if vol % 50 == 0 or vol == args.end:
                    print(f"  卷{vol:03d}: {len(poems)} poems (total so far: {total_poems})")
                time.sleep(args.delay)
            except Exception as e:
                print(f"  卷{vol:03d}: FAILED ({e})")
                failed += 1
                time.sleep(1)

    print(f"\n=== Download Complete ===")
    print(f"  Volumes: {args.start}-{args.end}")
    print(f"  Total poems: {total_poems}")
    print(f"  Failed volumes: {failed}")
    print(f"  Output: {output}")


if __name__ == "__main__":
    main()
