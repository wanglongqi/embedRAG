"""Multilingual tests: CJK, Arabic, Cyrillic, Thai, mixed-script content."""

from embedrag.writer.chunking.splitter import (
    _count_tokens_approx,
    _split_text_by_size,
    _tokenize_multilingual,
    smart_split,
    split_by_headings,
    split_by_paragraphs,
    split_document,
)
from embedrag.query.retrieval.sparse import SparseRetriever


# ── Token counting ──


class TestTokenCounting:
    def test_chinese_text(self):
        text = "机器学习是人工智能的一个分支"
        count = _count_tokens_approx(text)
        assert count == len(text)  # each CJK char = 1 token

    def test_japanese_mixed(self):
        text = "東京タワーはとても高いです"
        count = _count_tokens_approx(text)
        assert count >= 10

    def test_korean(self):
        text = "한국어 테스트 문장입니다"
        count = _count_tokens_approx(text)
        assert count >= 3

    def test_english(self):
        text = "The quick brown fox jumps over the lazy dog"
        count = _count_tokens_approx(text)
        assert count == 9

    def test_mixed_chinese_english(self):
        text = "深度学习deep learning是机器学习的分支"
        count = _count_tokens_approx(text)
        assert count >= 12

    def test_arabic(self):
        text = "هذا نص باللغة العربية للاختبار"
        count = _count_tokens_approx(text)
        assert count >= 5

    def test_cyrillic(self):
        text = "Это тестовое предложение на русском языке"
        count = _count_tokens_approx(text)
        assert count >= 5

    def test_empty(self):
        assert _count_tokens_approx("") == 1
        assert _count_tokens_approx("   ") == 1

    def test_thai(self):
        text = "ภาษาไทยไม่มีช่องว่างระหว่างคำ"
        count = _count_tokens_approx(text)
        assert count >= 10


# ── Tokenizer ──


class TestMultilingualTokenizer:
    def test_chinese_chars_are_individual_tokens(self):
        tokens = _tokenize_multilingual("你好世界")
        content = [t for t in tokens if not t.isspace()]
        assert content == ["你", "好", "世", "界"]

    def test_lossless_rejoin(self):
        texts = [
            "Hello 你好 World 世界",
            "深度学习 deep learning",
            "Привет мир",
            "مرحبا بالعالم",
            "東京 is great! 素晴らしい",
        ]
        for text in texts:
            tokens = _tokenize_multilingual(text)
            assert "".join(tokens) == text, f"Lossy rejoin for: {text}"

    def test_mixed_script_tokens(self):
        tokens = _tokenize_multilingual("Hello你好World")
        content = [t for t in tokens if not t.isspace()]
        assert "Hello" in content
        assert "你" in content
        assert "好" in content
        assert "World" in content


# ── Splitting ──


class TestMultilingualSplitting:
    def test_chinese_text_splits(self):
        text = "机器学习" * 200
        chunks = _split_text_by_size(text, chunk_size=50, overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) > 0
        rejoined = "".join(chunks)
        assert "机器学习" in rejoined

    def test_chinese_no_space_corruption(self):
        text = "自然语言处理是人工智能领域的重要方向"
        chunks = _split_text_by_size(text, chunk_size=5, overlap=1)
        for chunk in chunks:
            assert " " not in chunk or chunk.strip() == chunk

    def test_japanese_preserves_text(self):
        text = "日本語のテスト文章です。これは二番目の文です。"
        chunks = _split_text_by_size(text, chunk_size=10, overlap=2)
        assert len(chunks) >= 1
        for chunk in chunks:
            for ch in chunk:
                if not ch.isspace():
                    assert ch in text

    def test_mixed_chinese_english_split(self):
        text = "深度学习deep learning是一种machine learning方法"
        chunks = _split_text_by_size(text, chunk_size=8, overlap=2)
        assert len(chunks) >= 1

    def test_arabic_rtl_preserves(self):
        text = "هذا نص عربي طويل للاختبار " * 30
        chunks = _split_text_by_size(text, chunk_size=20, overlap=5)
        assert len(chunks) > 1

    def test_smart_split_chinese_markdown(self):
        text = "# 简介\n\n这是一个关于机器学习的文档。\n\n## 方法\n\n深度学习是一种方法。"
        chunks = smart_split(text, "cn_doc")
        level_types = [c.level_type for c in chunks]
        assert "document" in level_types
        assert "section" in level_types

    def test_paragraph_split_chinese(self):
        text = "第一段内容在这里。\n\n第二段内容在这里。\n\n第三段内容在这里。"
        chunks = split_by_paragraphs(text, "cn_para", chunk_size=5)
        assert len(chunks) >= 2

    def test_split_document_plain_chinese(self):
        text = "中文纯文本" * 100
        chunks = split_document(text, "cn_plain", strategy="plain", chunk_size=50)
        assert len(chunks) > 1

    def test_korean_paragraph_split(self):
        text = "첫 번째 단락입니다.\n\n두 번째 단락입니다.\n\n세 번째 단락입니다."
        chunks = split_by_paragraphs(text, "kr_para", chunk_size=5)
        assert len(chunks) >= 2


# ── FTS5 Trigram Query Building ──


def _build_fts(text: str) -> str:
    """Build an FTS query string using the production split+build pipeline."""
    retriever = SparseRetriever.__new__(SparseRetriever)
    fts_segs, _ = retriever._split_segments(text)
    if not fts_segs:
        return ""
    return SparseRetriever._segments_to_fts(fts_segs)


class TestFTSQueryBuilding:
    """Tests for _split_segments + _segments_to_fts with trigram tokenizer."""

    def test_chinese_term_kept_whole(self):
        q = _build_fts("机器学习方法")
        assert '"机器学习方法"' in q or "机器学习方法" in q

    def test_english_words_as_segments(self):
        q = _build_fts("machine learning")
        assert '"machine"' in q
        assert '"learning"' in q
        assert "OR" in q

    def test_mixed_cjk_english(self):
        q = _build_fts("深度学习 deep learning")
        assert "深度学习" in q
        assert '"deep"' in q or '"learning"' in q

    def test_arabic_terms(self):
        q = _build_fts("البحث النصي")
        assert q
        assert "OR" in q

    def test_cyrillic_terms(self):
        q = _build_fts("поиск текста")
        assert '"поиск"' in q
        assert '"текста"' in q

    def test_short_query_dropped(self):
        assert _build_fts("ab") == ""
        assert _build_fts("机器") == ""

    def test_3char_minimum(self):
        q = _build_fts("abc")
        assert '"abc"' in q
        q = _build_fts("机器学")
        assert '"机器学"' in q

    def test_special_chars_stripped(self):
        q = _build_fts('test "quoted" (parens)')
        assert "test" in q
        assert "quoted" in q
        assert "parens" in q

    def test_empty_query(self):
        assert _build_fts("") == ""
        assert _build_fts("   ") == ""

    def test_mixed_short_and_long(self):
        q = _build_fts("AI 机器学习")
        assert "机器学习" in q
        assert "AI" not in q  # too short, goes to LIKE path

    def test_cjk_sliding_windows(self):
        q = _build_fts("谁栽到地下死了")
        assert '"谁栽到地下死了"' in q
        assert '"谁栽到"' in q
        assert '"地下死"' in q

    def test_short_cjk_no_windows(self):
        q = _build_fts("机器学")
        assert q == '"机器学"'


# ── End-to-end FTS5 trigram search ──


import sqlite3
import pytest


class TestFTS5TrigramEndToEnd:
    """Full roundtrip: create trigram FTS table, insert docs, search."""

    @pytest.fixture
    def fts_conn(self):
        from embedrag.text.normalize import normalize_query

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5("
            "  chunk_id UNINDEXED, text, text_norm, title, title_norm, tags,"
            "  tokenize='trigram case_sensitive 0'"
            ")"
        )
        docs = [
            ("c1", "机器学习是人工智能的一个分支，通过数据让计算机从经验中学习", "AI入门", ""),
            ("c2", "深度学习是机器学习的子领域，使用神经网络进行特征学习", "深度学习", ""),
            ("c3", "自然语言处理是人工智能的重要方向", "NLP", ""),
            ("c4", "学习英语需要每天坚持练习", "英语", ""),
            ("c5", "这台机器运行很稳定", "硬件", ""),
            ("c6", "Machine learning is a subset of artificial intelligence", "ML", ""),
            ("c7", "Deep learning uses neural networks for feature extraction", "DL", ""),
            ("c8", "Natural language processing handles human language", "NLP", ""),
            ("c9", "東京タワーは観光名所として有名です", "東京", ""),
            ("c10", "기계학습은 인공지능의 한 분야입니다", "ML", ""),
            ("c11", "我们使用BERT模型进行fine-tuning，效果很好", "BERT", ""),
        ]
        for cid, text, title, tags in docs:
            text_norm = normalize_query(text)
            title_norm = normalize_query(title)
            conn.execute(
                "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?)",
                (cid, text, text_norm, title, title_norm, tags),
            )
        conn.commit()
        yield conn
        conn.close()

    def _search(self, conn, query: str) -> set[str]:
        """Search using the same query builder as production code."""
        retriever = SparseRetriever.__new__(SparseRetriever)
        fts_segs, _ = retriever._split_segments(query)
        if not fts_segs:
            return set()
        fts_query = SparseRetriever._segments_to_fts(fts_segs)
        if not fts_query:
            return set()
        try:
            rows = conn.execute(
                "SELECT chunk_id FROM chunks_fts WHERE text_norm MATCH ? ORDER BY rank LIMIT 10",
                (fts_query,),
            ).fetchall()
            return {r["chunk_id"] for r in rows}
        except Exception:
            return set()

    def test_chinese_term_recall(self, fts_conn):
        hits = self._search(fts_conn, "机器学习")
        assert "c1" in hits
        assert "c2" in hits
        assert "c5" not in hits  # only has 机器, not 机器学习

    def test_chinese_ai_term(self, fts_conn):
        hits = self._search(fts_conn, "人工智能")
        assert "c1" in hits
        assert "c3" in hits

    def test_english_phrase(self, fts_conn):
        hits = self._search(fts_conn, "machine learning")
        assert "c6" in hits

    def test_english_single_word(self, fts_conn):
        hits = self._search(fts_conn, "neural")
        assert "c7" in hits

    def test_japanese(self, fts_conn):
        hits = self._search(fts_conn, "東京タワー")
        assert "c9" in hits

    def test_korean(self, fts_conn):
        hits = self._search(fts_conn, "인공지능")
        assert "c10" in hits

    def test_mixed_bert(self, fts_conn):
        hits = self._search(fts_conn, "BERT")
        assert "c11" in hits

    def test_mixed_fine_tuning(self, fts_conn):
        hits = self._search(fts_conn, "fine-tuning")
        assert "c11" in hits

    def test_no_false_positive_short_substring(self, fts_conn):
        hits = self._search(fts_conn, "机器学习")
        assert "c4" not in hits  # c4 has 学习 but not 机器学习

    def test_case_insensitive(self, fts_conn):
        hits = self._search(fts_conn, "bert")
        assert "c11" in hits

    def test_short_query_no_crash(self, fts_conn):
        hits = self._search(fts_conn, "AI")
        # 2 chars < trigram min, returns empty -- no crash
        assert isinstance(hits, set)
