"""Text normalization for FTS: NFKC, diacritic stripping, trad-to-simp Chinese.

The normalized form is used only for indexing/searching in the shadow
``text_norm`` FTS5 column. Original text is preserved for display.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

_T2S_PATH = Path(__file__).parent / "t2s_chars.json"

_t2s_map: dict[str, str] | None = None


def _load_t2s() -> dict[str, str]:
    global _t2s_map
    if _t2s_map is None:
        with open(_T2S_PATH, encoding="utf-8") as f:
            _t2s_map = json.load(f)
    return _t2s_map


def _strip_diacritics(text: str) -> str:
    """Remove combining marks (accents/diacritics) after NFKD decomposition."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def normalize_for_fts(text: str) -> str:
    """Normalize text for FTS indexing and query matching.

    Steps:
    1. NFKC normalization (fullwidth -> halfwidth, compatibility forms)
    2. Casefold (aggressive lowercasing, handles German sharp-s etc.)
    3. Strip diacritics (e -> e for cafe, u -> u for uber)
    4. Traditional -> Simplified Chinese character mapping
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _strip_diacritics(text)
    t2s = _load_t2s()
    return text.translate(str.maketrans(t2s))  # type: ignore[arg-type]


@lru_cache(maxsize=256)
def normalize_query(text: str) -> str:
    """Cached version for query-time normalization (few unique queries)."""
    return normalize_for_fts(text)
