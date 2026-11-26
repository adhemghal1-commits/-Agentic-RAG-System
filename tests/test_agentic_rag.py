"""Unit and integration tests for the Agentic RAG System."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import AgenticRAG, StandardRAG
from src.config import AgentConfig, LLMConfig
from src.pipeline.chunker import chunk_pages
from src.pipeline.cleaner import clean_text
from src.pipeline.extractor import ExtractedPage, extract_document
from src.pipeline.vector_store import VectorStore
from src.tools.analysis_tools import analyze_data, calculate


@pytest.fixture()
def sample_store(tmp_path: Path) -> VectorStore:
    pages = [
        ExtractedPage(
            doc_id="lec1",
            filename="Lec1_Intro_DS.pdf",
            page=1,
            text=clean_text(
                "Data Structures organize data efficiently in memory. "
                "Arrays provide O(1) random access by index, while linked lists allow O(1) insertion at the head."
            ),
            extraction_method="text",
        ),
        ExtractedPage(
            doc_id="lec2",
            filename="Lec2_Trees.pdf",
            page=4,
            text=clean_text(
                "A Binary Search Tree (BST) maintains the invariant that left child keys are smaller "
                "and right child keys are larger. Average search takes O(log n) time."
            ),
            extraction_method="text",
        ),
        ExtractedPage(
            doc_id="lec5",
            filename="Lec5_Heaps.pdf",
            page=2,
            text=clean_text(
                "A Binary Heap is a complete binary tree. In a min-heap, the parent key is less than or equal to "
                "its children keys, so the minimum element is always at the root. "
                "In a max-heap, the largest element is at the root. Heapify takes O(log n) time and build-heap takes O(n)."
            ),
            extraction_method="text",
        ),
    ]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50, min_chunk_chars=20)
    docs_meta = [
        {"filename": "Lec1_Intro_DS.pdf", "pages_extracted": 1, "ocr_pages": 0, "chunks": 1, "total_words": 28},
        {"filename": "Lec2_Trees.pdf", "pages_extracted": 1, "ocr_pages": 0, "chunks": 1, "total_words": 28},
        {"filename": "Lec5_Heaps.pdf", "pages_extracted": 1, "ocr_pages": 0, "chunks": 1, "total_words": 48},
    ]
    store = VectorStore(tmp_path / "vs")
    store.build(chunks=chunks, documents_meta=docs_meta, svd_components=8)
    return store


def test_clean_text_normalizes_and_rejoins_hyphens():
    raw = "Algo-\nrithm   analysis\n\n\nPage 3 of 10\nNext   line"
    cleaned = clean_text(raw)
    assert "Algorithm analysis" in cleaned
    assert "Page 3 of 10" not in cleaned
    assert "   " not in cleaned


def test_vector_store_save_load_and_search(sample_store: VectorStore, tmp_path: Path):
    reloaded = VectorStore(tmp_path / "vs")
    assert reloaded.load() is True
    assert reloaded.is_ready

    hybrid = reloaded.search("min-heap root element", top_k=2, strategy="hybrid")
    assert len(hybrid) >= 1
    assert hybrid[0].chunk.filename == "Lec5_Heaps.pdf"

    filtered = reloaded.search("O(1)", top_k=2, doc_filter="Lec1")
    assert len(filtered) == 1
    assert filtered[0].chunk.filename == "Lec1_Intro_DS.pdf"


def test_calculate_tool_and_safety():
    res = calculate("log2(1024) + 1")
    assert res.success is True
    assert res.data["result"] == 11.0

    bad = calculate("__import__('os').system('echo hi')")
    assert bad.success is False


def test_analyze_data_operations(sample_store: VectorStore):
    stats_res = analyze_data("corpus_stats", store=sample_store)
    assert stats_res.success is True
    assert len(stats_res.data) == 3

    freq_res = analyze_data("topic_frequency", store=sample_store, target="heap")
    assert freq_res.success is True
    assert freq_res.data.get("Lec5_Heaps.pdf", 0) >= 2

    comp_res = analyze_data("complexity_table", values=[10, 100])
    assert comp_res.success is True
    assert len(comp_res.data) == 2


def test_standard_vs_agentic_rag(sample_store: VectorStore):
    std = StandardRAG(sample_store).run("What is a min-heap?")
    assert "Lec5_Heaps.pdf" in std.answer

    cfg = AgentConfig(llm=LLMConfig(provider="local"))
    agent = AgenticRAG(sample_store, config=cfg)

    comp_resp = agent.run("Compare arrays and min-heap")
    assert comp_resp.query_type == "comparison"
    assert len(comp_resp.plan) >= 2
    assert "refine_and_search" in comp_resp.tools_used
    assert len(comp_resp.sources) >= 2
    assert "[Source:" in comp_resp.answer

    calc_resp = agent.run("Calculate log2(1024) + 1 for a binary heap")
    assert "calculate" in calc_resp.tools_used
    assert "11" in calc_resp.answer


def test_extract_plain_and_csv(tmp_path: Path):
    txt = tmp_path / "notes.txt"
    txt.write_text("Stacks follow LIFO order.\nQueues follow FIFO order.", encoding="utf-8")
    pages = extract_document(txt)
    assert len(pages) == 1
    assert "LIFO" in pages[0].text

