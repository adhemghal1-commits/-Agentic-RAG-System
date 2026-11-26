"""Sentence- and paragraph-aware recursive text chunking with metadata."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, asdict

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}")

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "have", "been", "were", "they",
    "this", "that", "with", "from", "will", "would", "there", "their", "what",
    "about", "which", "when", "make", "like", "time", "just", "know", "take",
    "into", "year", "your", "some", "could", "them", "than", "then", "look",
    "only", "come", "its", "over", "also", "back", "after", "use", "two",
    "how", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "most", "each", "more", "very", "such", "where",
    "used", "using", "does", "example", "slide", "lecture",
})


@dataclass
class DocumentChunk:
    """A single indexed chunk of a document with provenance metadata."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    char_count: int
    word_count: int
    keywords: list[str]
    extraction_method: str = "text"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentChunk":
        return cls(**data)


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Extract top domain-relevant keywords from a chunk."""
    tokens = [w.lower() for w in _WORD_RE.findall(text)]
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _split_text_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into chunks respecting paragraph and sentence boundaries."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    units = [u.strip() for u in _SENTENCE_SPLIT_RE.split(text) if u.strip()]
    if not units:
        return [text[i : i + chunk_size] for i in range(0, len(text), max(1, chunk_size - chunk_overlap))]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for unit in units:
        unit_len = len(unit)
        # If a single unit exceeds chunk_size, hard-slice it cleanly.
        if unit_len > chunk_size:
            if current_parts:
                chunks.append(" ".join(current_parts).strip())
                current_parts = []
                current_len = 0
            step = max(1, chunk_size - chunk_overlap)
            for start in range(0, unit_len, step):
                part = unit[start : start + chunk_size].strip()
                if part:
                    chunks.append(part)
            continue

        projected = current_len + (1 if current_parts else 0) + unit_len
        if projected <= chunk_size:
            current_parts.append(unit)
            current_len = projected
        else:
            chunks.append(" ".join(current_parts).strip())
            # Build overlap from tail of current_parts
            overlap_parts: list[str] = []
            overlap_len = 0
            for prev in reversed(current_parts):
                if overlap_len + len(prev) + 1 <= chunk_overlap:
                    overlap_parts.insert(0, prev)
                    overlap_len += len(prev) + 1
                else:
                    break
            current_parts = overlap_parts + [unit]
            current_len = sum(len(p) for p in current_parts) + max(0, len(current_parts) - 1)

    if current_parts:
        chunks.append(" ".join(current_parts).strip())

    return chunks


def chunk_pages(
    pages: list,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 60,
) -> list[DocumentChunk]:
    """Chunk a list of ExtractedPage objects into DocumentChunk records."""
    all_chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for page_obj in pages:
        raw_text = (page_obj.text or "").strip()
        if not raw_text:
            continue

        pieces = _split_text_recursive(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for piece in pieces:
            if len(piece) < min_chunk_chars and len(pieces) > 1:
                continue
            words = piece.split()
            chunk_id = f"{page_obj.doc_id}::p{page_obj.page}::c{chunk_idx}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=page_obj.doc_id,
                    filename=page_obj.filename,
                    page=page_obj.page,
                    chunk_index=chunk_idx,
                    text=piece,
                    char_count=len(piece),
                    word_count=len(words),
                    keywords=extract_keywords(piece),
                    extraction_method=page_obj.extraction_method,
                )
            )
            chunk_idx += 1

    return all_chunks

