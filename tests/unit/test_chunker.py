"""Tests for the hierarchical text splitter."""

from embedrag.writer.chunking.splitter import (
    smart_split,
    split_by_headings,
    split_by_paragraphs,
    split_document,
    split_sliding_window,
)


class TestSmartSplit:
    def test_short_text_single_chunk(self):
        text = "This is a short FAQ answer."
        chunks = smart_split(text, "doc_001")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].level_type == "chunk"

    def test_markdown_headings_detected(self):
        text = (
            "# Introduction\n\nSome intro text here.\n\n"
            "## Section A\n\nContent of section A with enough words to matter.\n\n"
            "## Section B\n\nContent of section B also with enough words."
        )
        chunks = smart_split(text, "doc_002")
        level_types = [c.level_type for c in chunks]
        assert "document" in level_types
        assert "section" in level_types

    def test_long_plain_text_sliding_window(self):
        text = " ".join(["word"] * 2000)
        chunks = smart_split(text, "doc_003", chunk_size=200, overlap=50)
        assert len(chunks) > 1
        doc_chunks = [c for c in chunks if c.level_type == "document"]
        assert len(doc_chunks) == 1


class TestSplitByHeadings:
    def test_extracts_sections(self):
        text = (
            "# Title\n\nIntro paragraph.\n\n"
            "## First Section\n\nFirst section content with several words.\n\n"
            "## Second Section\n\nSecond section content with several words."
        )
        chunks = split_by_headings(text, "doc_h1")
        sections = [c for c in chunks if c.level_type == "section"]
        assert len(sections) >= 2
        section_texts = [s.text for s in sections]
        assert "First Section" in section_texts
        assert "Second Section" in section_texts

    def test_parent_child_links(self):
        text = "# Title\n\nIntro.\n\n## Sub\n\nContent here."
        chunks = split_by_headings(text, "doc_h2")
        doc_node = [c for c in chunks if c.level_type == "document"][0]
        sections = [c for c in chunks if c.level_type == "section"]
        for s in sections:
            assert s.parent_chunk_id == doc_node.chunk_id

    def test_no_headings_falls_back_to_sliding(self):
        text = " ".join(["word"] * 1000)
        chunks = split_by_headings(text, "doc_h3", chunk_size=200)
        assert len(chunks) > 1


class TestSlidingWindow:
    def test_overlap_creates_redundancy(self):
        words = [f"w{i}" for i in range(500)]
        text = " ".join(words)
        chunks = split_sliding_window(text, "doc_sw", chunk_size=100, overlap=30)
        leaf_chunks = [c for c in chunks if c.level_type == "chunk"]
        assert len(leaf_chunks) > 1
        if len(leaf_chunks) >= 2:
            t0_words = set(leaf_chunks[0].text.split())
            t1_words = set(leaf_chunks[1].text.split())
            assert len(t0_words & t1_words) > 0

    def test_single_chunk_for_short_text(self):
        text = "Short text here."
        chunks = split_sliding_window(text, "doc_sw2")
        assert len(chunks) == 1


class TestSplitByParagraphs:
    def test_splits_on_double_newlines(self):
        text = (
            "First paragraph with some content and enough words to exceed the size.\n\n"
            "Second paragraph with different content and also enough words here.\n\n"
            "Third paragraph closes things out with yet more words for the test."
        )
        chunks = split_by_paragraphs(text, "doc_para", chunk_size=8)
        assert len(chunks) >= 2
        assert all(c.level_type == "chunk" for c in chunks)

    def test_merges_short_paragraphs(self):
        text = "A.\n\nB.\n\nC."
        chunks = split_by_paragraphs(text, "doc_short", chunk_size=500)
        assert len(chunks) == 1
        assert "A." in chunks[0].text and "C." in chunks[0].text

    def test_no_hierarchy_created(self):
        text = (
            "Para one with enough words to be meaningful.\n\n"
            "Para two also has enough words to matter here.\n\n"
            "Para three fills things out with more content."
        )
        chunks = split_by_paragraphs(text, "doc_flat", chunk_size=10)
        for c in chunks:
            assert c.parent_chunk_id is None

    def test_empty_text(self):
        chunks = split_by_paragraphs("  ", "doc_empty")
        assert len(chunks) == 1


class TestSplitDocument:
    def test_auto_delegates_to_smart_split(self):
        text = "# Heading\n\nBody text."
        auto = split_document(text, "d1", strategy="auto")
        assert any(c.level_type == "section" for c in auto)

    def test_structured_forces_heading_split(self):
        text = "No headings here, just plain text with enough words " * 20
        chunks = split_document(text, "d2", strategy="structured")
        assert len(chunks) >= 1

    def test_plain_forces_sliding_window(self):
        text = "# Heading\n\nBody.\n\n## Sub\n\nMore body text."
        chunks = split_document(text, "d3", strategy="plain")
        for c in chunks:
            assert c.level_type != "section"

    def test_paragraph_strategy(self):
        text = "First para.\n\nSecond para.\n\nThird para."
        chunks = split_document(text, "d4", strategy="paragraph", chunk_size=5)
        assert len(chunks) >= 2
