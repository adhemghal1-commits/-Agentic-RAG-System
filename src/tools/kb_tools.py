"""Knowledge-base tools: search, query refinement, document listing, and summarization."""

from __future__ import annotations

from collections import Counter

from src.pipeline.vector_store import SearchResult, VectorStore
from src.tools.registry import ToolResult, ToolSpec


def _results_to_sources(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "chunk_id": r.chunk.chunk_id,
            "filename": r.chunk.filename,
            "page": r.chunk.page,
            "score": r.score,
            "semantic_score": r.semantic_score,
            "bm25_score": r.bm25_score,
            "strategy": r.strategy,
            "text": r.chunk.text,
            "keywords": r.chunk.keywords,
        }
        for r in results
    ]


def build_kb_tools(store: VectorStore) -> list[ToolSpec]:
    """Create knowledge-base tools bound to a VectorStore instance."""

    def search_knowledge_base(
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
        strategy: str = "hybrid",
    ) -> ToolResult:
        if not store.is_ready:
            return ToolResult(
                tool_name="search_knowledge_base",
                success=False,
                summary="Knowledge base is empty. Run ingestion first.",
            )
        results = store.search(query=query, top_k=top_k, strategy=strategy, doc_filter=doc_filter)
        if not results:
            return ToolResult(
                tool_name="search_knowledge_base",
                success=True,
                summary=f"No matching passages found for query '{query}'.",
                data=[],
                sources=[],
            )
        top_score = results[0].score
        snippets = [
            f"[{r.chunk.filename}, p.{r.chunk.page}, score={r.score:.3f}] {r.chunk.text[:220]}..."
            for r in results[:3]
        ]
        return ToolResult(
            tool_name="search_knowledge_base",
            success=True,
            summary=(
                f"Retrieved {len(results)} chunks (top score={top_score:.3f}) for '{query}'.\n"
                + "\n".join(snippets)
            ),
            data={"query": query, "top_score": top_score, "count": len(results)},
            sources=_results_to_sources(results),
        )

    def refine_and_search(
        original_query: str,
        sub_queries: list[str],
        top_k_per_query: int = 4,
        strategy: str = "hybrid",
    ) -> ToolResult:
        if not store.is_ready:
            return ToolResult(
                tool_name="refine_and_search",
                success=False,
                summary="Knowledge base is empty. Run ingestion first.",
            )
        merged: dict[str, SearchResult] = {}
        for sq in sub_queries:
            for r in store.search(query=sq, top_k=top_k_per_query, strategy=strategy):
                existing = merged.get(r.chunk.chunk_id)
                if existing is None or r.score > existing.score:
                    merged[r.chunk.chunk_id] = r

        ranked = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        sources = _results_to_sources(ranked)
        return ToolResult(
            tool_name="refine_and_search",
            success=True,
            summary=(
                f"Refined '{original_query}' into {len(sub_queries)} sub-queries "
                f"({sub_queries}) -> {len(ranked)} unique chunks."
            ),
            data={"sub_queries": sub_queries, "unique_chunks": len(ranked)},
            sources=sources,
        )

    def list_documents() -> ToolResult:
        if not store.is_ready:
            return ToolResult(
                tool_name="list_documents",
                success=False,
                summary="Knowledge base is empty.",
            )
        if store.documents_meta:
            lines = [
                f"- {d['filename']}: {d['pages_extracted']} pages, {d['chunks']} chunks, {d['total_words']} words"
                for d in store.documents_meta
            ]
            return ToolResult(
                tool_name="list_documents",
                success=True,
                summary="Indexed documents:\n" + "\n".join(lines),
                data=store.documents_meta,
            )
        names = sorted({c.filename for c in store.chunks})
        return ToolResult(
            tool_name="list_documents",
            success=True,
            summary=f"Indexed documents: {', '.join(names)}",
            data=[{"filename": n} for n in names],
        )

    def summarize_document(filename: str, focus_topic: str | None = None) -> ToolResult:
        if not store.is_ready:
            return ToolResult(
                tool_name="summarize_document",
                success=False,
                summary="Knowledge base is empty.",
            )
        doc_chunks = [c for c in store.chunks if filename.lower() in c.filename.lower()]
        if not doc_chunks:
            return ToolResult(
                tool_name="summarize_document",
                success=False,
                summary=f"No document matching '{filename}' found in knowledge base.",
            )

        actual_name = doc_chunks[0].filename
        if focus_topic:
            focused = store.search(query=focus_topic, top_k=6, doc_filter=actual_name)
            selected_chunks = [r.chunk for r in focused]
            sources = _results_to_sources(focused)
        else:
            # Sample evenly across the document (start, middle, end)
            step = max(1, len(doc_chunks) // 6)
            selected_chunks = doc_chunks[::step][:6]
            sources = [
                {
                    "chunk_id": c.chunk_id,
                    "filename": c.filename,
                    "page": c.page,
                    "score": 1.0,
                    "semantic_score": 1.0,
                    "bm25_score": 0.0,
                    "strategy": "document_sample",
                    "text": c.text,
                    "keywords": c.keywords,
                }
                for c in selected_chunks
            ]

        kw_counts = Counter(kw for c in doc_chunks for kw in c.keywords)
        top_topics = [kw for kw, _ in kw_counts.most_common(10)]
        pages_covered = sorted({c.page for c in doc_chunks})
        excerpt_lines = [f"[p.{c.page}] {c.text[:260]}" for c in selected_chunks]

        summary = (
            f"Document '{actual_name}' ({len(pages_covered)} pages, {len(doc_chunks)} chunks). "
            f"Key topics: {', '.join(top_topics)}.\nRepresentative excerpts:\n"
            + "\n".join(excerpt_lines)
        )
        return ToolResult(
            tool_name="summarize_document",
            success=True,
            summary=summary,
            data={"filename": actual_name, "top_topics": top_topics, "chunks_total": len(doc_chunks)},
            sources=sources,
        )

    return [
        ToolSpec(
            name="search_knowledge_base",
            description="Search indexed documents using hybrid, semantic, or BM25 retrieval.",
            parameters={
                "query": "str",
                "top_k": "int (default 5)",
                "doc_filter": "str | None",
                "strategy": "'hybrid' | 'semantic' | 'bm25'",
            },
            fn=search_knowledge_base,
        ),
        ToolSpec(
            name="refine_and_search",
            description="Decompose or refine a complex query into sub-queries and merge deduplicated results.",
            parameters={"original_query": "str", "sub_queries": "list[str]", "top_k_per_query": "int"},
            fn=refine_and_search,
        ),
        ToolSpec(
            name="list_documents",
            description="List all documents in the knowledge base with page, chunk, and word counts.",
            parameters={},
            fn=list_documents,
        ),
        ToolSpec(
            name="summarize_document",
            description="Summarize a specific document's key topics and representative excerpts.",
            parameters={"filename": "str", "focus_topic": "str | None"},
            fn=summarize_document,
        ),
    ]

