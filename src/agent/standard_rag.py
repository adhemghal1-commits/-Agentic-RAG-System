"""Standard (Naive) RAG baseline: single-shot retrieval followed by single generation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.agent.llm_client import LLMClient, extractive_synthesize
from src.pipeline.vector_store import VectorStore


@dataclass
class StandardRAGResponse:
    """Output of the Standard RAG baseline."""

    query: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    provider: str = "local"


class StandardRAG:
    """Classic 1-shot Retrieve-Then-Generate RAG without planning, tools, or query refinement."""

    def __init__(self, store: VectorStore, llm_client: LLMClient | None = None) -> None:
        self.store = store
        self.llm = llm_client or LLMClient()

    def run(self, query: str, top_k: int = 5, strategy: str = "semantic") -> StandardRAGResponse:
        t0 = time.perf_counter()
        results = self.store.search(query=query, top_k=top_k, strategy=strategy)
        sources = [
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

        provider = self.llm.active_provider
        if provider == "local":
            answer = extractive_synthesize(query, sources)
        else:
            context = "\n\n".join(
                f"[{s['filename']}, p. {s['page']}] {s['text']}" for s in sources
            )
            system_prompt = (
                "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                "Cite sources inline as [Source: filename, p. X]. If the context is insufficient, say so."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
            answer = self.llm.generate(system_prompt, user_prompt)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return StandardRAGResponse(
            query=query,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            provider=provider,
        )

