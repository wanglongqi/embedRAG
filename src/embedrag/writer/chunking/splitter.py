"""Multi-strategy text splitter: heading-aware, sliding window, and short text.

Designed for multilingual content (100+ languages). Key design:
- Token counting uses character-aware segmentation, not whitespace splitting.
- CJK characters (Chinese, Japanese, Korean) each count as ~1 token.
- Latin/Cyrillic/Arabic words are counted normally by whitespace.
- Text splitting preserves original characters -- no lossy re-joining.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Optional

from embedrag.models.chunk import ChunkNode

HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+)$|^(<h([1-6])>)(.+?)(</h\d>)$",
    re.MULTILINE | re.IGNORECASE,
)

_CJK_RANGES = (
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2B73F),  # CJK Extension C
    (0x2B740, 0x2B81F),  # CJK Extension D
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0x3040, 0x309F),    # Hiragana
    (0x30A0, 0x30FF),    # Katakana
    (0xAC00, 0xD7AF),    # Hangul Syllables
    (0x1100, 0x11FF),    # Hangul Jamo
)

_THAI_RANGE = (0x0E00, 0x0E7F)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 128


def _is_cjk_char(cp: int) -> bool:
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _is_ideographic(char: str) -> bool:
    """True for characters in scripts where each char is roughly one token."""
    cp = ord(char)
    return _is_cjk_char(cp) or (_THAI_RANGE[0] <= cp <= _THAI_RANGE[1])


def _generate_chunk_id() -> str:
    return uuid.uuid4().hex[:16]


def _count_tokens_approx(text: str) -> int:
    """Approximate token count, aware of CJK and mixed scripts.

    - Each CJK/Thai character counts as ~1.0 token.
    - Whitespace-delimited words count as ~1.3 tokens each.
    - Mixed text (e.g. Chinese with English terms) is handled correctly.
    """
    count = 0
    in_word = False
    for ch in text:
        if _is_ideographic(ch):
            if in_word:
                count += 1
                in_word = False
            count += 1
        elif ch.isspace():
            if in_word:
                count += 1
                in_word = False
        else:
            in_word = True
    if in_word:
        count += 1
    return max(1, int(count * 1.0))


def split_by_headings(
    text: str,
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkNode]:
    """Split structured text (Markdown/HTML) by headings into a hierarchy.

    Builds a tree: document -> sections (by heading) -> chunks (by size).
    """
    sections = _extract_sections(text)
    if not sections:
        return split_sliding_window(text, doc_id, chunk_size=chunk_size, overlap=overlap)

    doc_node = ChunkNode(
        chunk_id=_generate_chunk_id(),
        doc_id=doc_id,
        text=text[:500],
        level=0,
        level_type="document",
    )
    all_chunks = [doc_node]

    for seq, (heading_level, heading_text, section_body) in enumerate(sections):
        section_node = ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=heading_text,
            parent_chunk_id=doc_node.chunk_id,
            level=1,
            level_type="section",
            seq_in_parent=seq,
            metadata={"heading_level": heading_level},
        )
        all_chunks.append(section_node)

        paragraphs = _split_text_by_size(section_body, chunk_size, overlap)
        for pseq, para_text in enumerate(paragraphs):
            chunk_node = ChunkNode(
                chunk_id=_generate_chunk_id(),
                doc_id=doc_id,
                text=para_text,
                parent_chunk_id=section_node.chunk_id,
                level=2,
                level_type="paragraph",
                seq_in_parent=pseq,
            )
            all_chunks.append(chunk_node)

    return all_chunks


def split_sliding_window(
    text: str,
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    parent_chunk_id: Optional[str] = None,
) -> list[ChunkNode]:
    """Split long plain text using a sliding window with overlap.

    Also creates a parent node representing the full text.
    """
    if _count_tokens_approx(text) <= chunk_size:
        return [ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=text,
            parent_chunk_id=parent_chunk_id,
            level=2 if parent_chunk_id else 0,
            level_type="chunk",
        )]

    if not parent_chunk_id:
        parent = ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=text[:500],
            level=0,
            level_type="document",
        )
        parent_chunk_id = parent.chunk_id
        chunks = [parent]
    else:
        chunks = []

    parts = _split_text_by_size(text, chunk_size, overlap)
    for seq, part in enumerate(parts):
        chunks.append(ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=part,
            parent_chunk_id=parent_chunk_id,
            level=3,
            level_type="chunk",
            seq_in_parent=seq,
        ))

    return chunks


def split_by_paragraphs(
    text: str,
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkNode]:
    """Split by paragraph boundaries (double newlines), merging short paragraphs.

    Best for medium-length plain text (articles, descriptions, Q&A) where
    paragraph breaks are meaningful but there are no headings.
    """
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not raw_paras:
        return [ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=text,
            level=0,
            level_type="chunk",
        )]

    merged: list[str] = []
    current = ""
    for para in raw_paras:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if _count_tokens_approx(candidate) > chunk_size and current:
            merged.append(current)
            current = para
        else:
            current = candidate
    if current:
        merged.append(current)

    if len(merged) == 1:
        return [ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=merged[0],
            level=0,
            level_type="chunk",
        )]

    chunks: list[ChunkNode] = []
    for seq, para_text in enumerate(merged):
        chunks.append(ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=para_text,
            level=0,
            level_type="chunk",
            seq_in_parent=seq,
        ))
    return chunks


def smart_split(
    text: str,
    doc_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkNode]:
    """Choose splitting strategy based on content.

    - Short text (< chunk_size tokens): single chunk
    - Has headings: heading-aware split
    - Long plain text: sliding window
    """
    if HEADING_PATTERN.search(text):
        return split_by_headings(text, doc_id, chunk_size, overlap)

    token_est = _count_tokens_approx(text)
    if token_est <= chunk_size:
        return [ChunkNode(
            chunk_id=_generate_chunk_id(),
            doc_id=doc_id,
            text=text,
            level=0,
            level_type="chunk",
        )]

    return split_sliding_window(text, doc_id, chunk_size, overlap)


def split_document(
    text: str,
    doc_id: str,
    strategy: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[ChunkNode]:
    """Dispatch to the requested chunking strategy.

    Strategies:
        auto       - detect headings vs. plain text (default)
        structured - force heading-aware split (Markdown/HTML)
        plain      - flat sliding window, no hierarchy
        paragraph  - split by paragraph boundaries, no hierarchy
    """
    if strategy == "structured":
        return split_by_headings(text, doc_id, chunk_size, overlap)
    if strategy == "plain":
        return split_sliding_window(text, doc_id, chunk_size, overlap)
    if strategy == "paragraph":
        return split_by_paragraphs(text, doc_id, chunk_size, overlap)
    return smart_split(text, doc_id, chunk_size, overlap)


def _extract_sections(text: str) -> list[tuple[int, str, str]]:
    """Extract (heading_level, heading_text, body) from structured text."""
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        if m.group(1):  # Markdown heading
            level = len(m.group(1))
            heading = m.group(2).strip()
        else:  # HTML heading
            level = int(m.group(4))
            heading = m.group(5).strip()

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((level, heading, body))

    return sections


def _tokenize_multilingual(text: str) -> list[str]:
    """Segment text into token-like units for chunking.

    Each CJK/Thai character becomes its own token.
    Whitespace-delimited words stay as single tokens.
    Whitespace between tokens is preserved as separate tokens.
    Rejoining via ''.join(tokens[a:b]) reconstructs the original text.
    """
    tokens: list[str] = []
    buf: list[str] = []

    def flush_buf():
        if buf:
            tokens.append("".join(buf))
            buf.clear()

    for ch in text:
        if _is_ideographic(ch):
            flush_buf()
            tokens.append(ch)
        elif ch.isspace():
            flush_buf()
            tokens.append(ch)
        else:
            buf.append(ch)
    flush_buf()
    return tokens


def _split_text_by_size(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into chunks of approximately chunk_size tokens with overlap.

    Works for all scripts: Latin, CJK, Arabic, Cyrillic, Thai, mixed, etc.
    Text is reconstructed losslessly via ''.join() instead of ' '.join().
    """
    tokens = _tokenize_multilingual(text)
    content_tokens = [t for t in tokens if not t.isspace()]
    if not content_tokens:
        return [text] if text.strip() else []

    # Build index: map content token positions to original token positions
    content_positions: list[int] = []
    for i, t in enumerate(tokens):
        if not t.isspace():
            content_positions.append(i)

    n = len(content_positions)
    step = max(1, chunk_size - overlap)
    parts: list[str] = []
    i = 0
    while i < n:
        end = min(i + chunk_size, n)
        start_pos = content_positions[i]
        end_pos = content_positions[end - 1] + 1 if end > 0 else start_pos + 1
        part = "".join(tokens[start_pos:end_pos]).strip()
        if part:
            parts.append(part)
        if end >= n:
            break
        i += step

    return parts if parts else [text]
