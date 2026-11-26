"""Persistent local vector store supporting Dense, BM25, and Hybrid RRF retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.pipeline.chunker import DocumentChunk
from src.pipeline.embedder import LocalHybridEmbedder

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+-]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class SearchResult:
    """Single retrieved chunk with similarity/relevance scores."""

    chunk: DocumentChunk
    score: float
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    strategy: str = "hybrid"

    def citation(self) -> str:
        return f"[Source: {self.chunk.filename}, p. {self.chunk.page}]"


class VectorStore:
    """Persistent vector + lexical store backed by numpy arrays and JSON metadata."""

    def __init__(self, store_dir: Path) -> None:
        self.store_dir = Path(store_dir)
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self.embedder: LocalHybridEmbedder | None = None
        self.bm25: BM25Okapi | None = None
        self.documents_meta: list[dict] = []

    @property
    def is_ready(self) -> bool:
        return len(self.chunks) > 0 and self.embedder is not None and self.embeddings.shape[0] == len(self.chunks)

    def build(
        self,
        chunks: list[DocumentChunk],
        documents_meta: list[dict],
        svd_components: int = 128,
    ) -> None:
        """Build dense + BM25 indices from chunks and persist to disk."""
        self.chunks = chunks
        self.documents_meta = documents_meta

        texts = [c.text for c in chunks]
        self.embedder = LocalHybridEmbedder(n_components=svd_components)
        self.embeddings = self.embedder.fit_transform(texts)

        tokenized = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        self.save()

    def save(self) -> None:
        """Persist chunks, embeddings, embedder, and document metadata to disk."""
        self.store_dir.mkdir(parents=True, exist_ok=True)
        with open(self.store_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)
        with open(self.store_dir / "documents.json", "w", encoding="utf-8") as f:
            json.dump(self.documents_meta, f, ensure_ascii=False, indent=2)
        np.savez_compressed(self.store_dir / "embeddings.npz", embeddings=self.embeddings)
        if self.embedder is not None:
            self.embedder.save(self.store_dir / "embedder.pkl")

    def load(self) -> bool:
        """Load persisted store from disk. Returns True if successful."""
        chunks_path = self.store_dir / "chunks.json"
        emb_path = self.store_dir / "embeddings.npz"
        embedder_path = self.store_dir / "embedder.pkl"
        docs_path = self.store_dir / "documents.json"

        if not (chunks_path.exists() and emb_path.exists() and embedder_path.exists()):
            return False

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = [DocumentChunk.from_dict(d) for d in json.load(f)]
        if docs_path.exists():
            with open(docs_path, "r", encoding="utf-8") as f:
                self.documents_meta = json.load(f)
        self.embeddings = np.load(emb_path)["embeddings"]
        self.embedder = LocalHybridEmbedder.load(embedder_path)

        tokenized = [_tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        doc_filter: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve top_k chunks using 'hybrid' (RRF), 'semantic', or 'bm25'."""
        if not self.is_ready or not query.strip():
            return []

        n = len(self.chunks)
        # Filter mask if doc_filter is provided (substring match, case-insensitive)
        if doc_filter:
            flt = doc_filter.lower()
            valid_indices = [i for i, c in enumerate(self.chunks) if flt in c.filename.lower()]
        else:
            valid_indices = list(range(n))

        if not valid_indices:
            return []

        q_vec = self.embedder.embed_queries([query])[0]
        sem_scores_all = self.embeddings @ q_vec

        if self.bm25 is not None:
            bm25_scores_all = np.array(self.bm25.get_scores(_tokenize(query)), dtype=np.float32)
        else:
            bm25_scores_all = np.zeros(n, dtype=np.float32)

        # Normalize BM25 to [0, 1] for display & fusion
        max_bm25 = float(bm25_scores_all.max()) if bm25_scores_all.size > 0 else 0.0
        bm25_norm = bm25_scores_all / max_bm25 if max_bm25 > 0 else bm25_scores_all

        if strategy == "semantic":
            ranked = sorted(valid_indices, key=lambda i: float(sem_scores_all[i]), reverse=True)[:top_k]
            return [
                SearchResult(
                    chunk=self.chunks[i],
                    score=float(sem_scores_all[i]),
                    semantic_score=float(sem_scores_all[i]),
                    bm25_score=float(bm25_norm[i]),
                    strategy="semantic",
                )
                for i in ranked
            ]

        if strategy == "bm25":
            ranked = sorted(valid_indices, key=lambda i: float(bm25_norm[i]), reverse=True)[:top_k]
            return [
                SearchResult(
                    chunk=self.chunks[i],
                    score=float(bm25_norm[i]),
                    semantic_score=float(sem_scores_all[i]),
                    bm25_score=float(bm25_norm[i]),
                    strategy="bm25",
                )
                for i in ranked
            ]

        # Hybrid: Reciprocal Rank Fusion (RRF) + weighted score tie-breaker
        sem_ranked = sorted(valid_indices, key=lambda i: float(sem_scores_all[i]), reverse=True)
        bm25_ranked = sorted(valid_indices, key=lambda i: float(bm25_norm[i]), reverse=True)

        k_rrf = 60.0
        rrf_scores: dict[int, float] = {i: 0.0 for i in valid_indices}
        for rank, idx in enumerate(sem_ranked):
            rrf_scores[idx] += 1.0 / (k_rrf + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            if bm25_norm[idx] > 0:
                rrf_scores[idx] += 1.0 / (k_rrf + rank + 1)

        # Scale RRF into [0, 1] relative to max possible RRF (2 / 61)
        max_possible = 2.0 / (k_rrf + 1.0)
        ranked = sorted(valid_indices, key=lambda i: rrf_scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                chunk=self.chunks[i],
                score=round(min(1.0, rrf_scores[i] / max_possible), 4),
                semantic_score=round(float(sem_scores_all[i]), 4),
                bm25_score=round(float(bm25_norm[i]), 4),
                strategy="hybrid",
            )
            for i in ranked
        ]

