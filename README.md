# Agentic RAG System

An intelligent **Agentic Retrieval-Augmented Generation (RAG)** system built in Python and deployed with **Streamlit**. It processes multi-format documents (including scanned/image-based PDFs via automatic OCR), builds a persistent hybrid vector + BM25 knowledge base, and uses a planning-capable agent with tools to answer complex multi-step questions with source citations.

## Standard RAG vs. Agentic RAG

| Dimension | Standard RAG | Agentic RAG |
| :--- | :--- | :--- |
| **Workflow** | Fixed 1-shot `Retrieve -> Generate` | Dynamic `Plan -> Act -> Observe -> Refine -> Synthesize` |
| **Query Handling** | Uses raw user query as-is | Classifies intent, decomposes multi-hop/comparison queries, refines low-relevance queries |
| **Retrieval** | Single top-k semantic lookup | Hybrid (Dense LSA + BM25 Reciprocal Rank Fusion), document filtering, multi-query fusion |
| **Tool Usage** | None | `search_knowledge_base`, `refine_and_search`, `list_documents`, `summarize_document`, `analyze_data`, `calculate` |
| **Complex Questions** | Often misses sub-parts or calculations | Breaks down sub-tasks, invokes tools, and synthesizes cited answers |

## Project Structure

```text
├── app.py                      # Streamlit web interface (Q&A, Comparison, KB Pipeline, Architecture)
├── requirements.txt
├── files/                      # Source documents (4 Data Structures lecture PDFs + user uploads)
├── scripts/
│   └── build_kb.py             # CLI script to run the 5-stage ingestion pipeline
├── src/
│   ├── config.py               # Pipeline, LLM, and Agent configuration
│   ├── pipeline/
│   │   ├── extractor.py        # PDF/TXT/MD/CSV/JSON extraction + RapidOCR fallback & disk cache
│   │   ├── cleaner.py          # Unicode normalization, hyphen-break repair, boilerplate removal
│   │   ├── chunker.py          # Recursive sentence/paragraph-aware chunking with metadata
│   │   ├── embedder.py         # Local TF-IDF + SVD dense embeddings (+ optional Gemini embeddings)
│   │   ├── vector_store.py     # Persistent numpy + BM25 hybrid vector store
│   │   └── ingest.py           # 5-stage pipeline orchestrator & telemetry
│   ├── tools/
│   │   ├── registry.py         # ToolSpec, ToolResult, ToolRegistry
│   │   ├── kb_tools.py         # search_knowledge_base, refine_and_search, list_documents, summarize_document
│   │   └── analysis_tools.py   # Safe math/Big-O calculator & pandas/numpy data analysis tool
│   └── agent/
│       ├── llm_client.py       # Gemini, OpenAI, and zero-key Local Extractive synthesizer
│       ├── standard_rag.py     # 1-shot Standard RAG baseline
│       └── agentic_rag.py      # Planning + ReAct Agentic RAG engine
└── tests/
    └── test_agentic_rag.py     # Unit & integration test suite
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Build the Knowledge Base** (processes all documents in `files/`):
   ```bash
   python scripts/build_kb.py
   ```
   *(You can also build/rebuild and upload new files directly inside the Streamlit UI.)*
3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
4. **Run tests:**
   ```bash
   pytest -v
   ```

