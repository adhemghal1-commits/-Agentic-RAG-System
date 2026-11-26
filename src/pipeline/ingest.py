"""End-to-end ingestion pipeline: Extraction -> Cleaning -> Chunking -> Embedding -> Storage."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from src.config import PipelineConfig
from src.pipeline.chunker import DocumentChunk, chunk_pages
from src.pipeline.extractor import ExtractedPage, extract_document
from src.pipeline.vector_store import VectorStore

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}


@dataclass
class PipelineStats:
    """Detailed execution telemetry for each stage of the ingestion pipeline."""

    documents_processed: int
    total_pages_extracted: int
    ocr_pages: int
    text_pages: int
    total_chunks: int
    embedding_dim: int
    extraction_seconds: float
    chunking_seconds: float
    embedding_storage_seconds: float
    total_seconds: float
    documents_meta: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


class IngestionPipeline:
    """Orchestrates document extraction, cleaning, chunking, embedding, and vector storage."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def list_source_files(self) -> list[Path]:
        if not self.config.files_dir.exists():
            return []
        return sorted(
            p
            for p in self.config.files_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def run(
        self,
        progress_cb: Callable[[str], None] | None = None,
    ) -> tuple[VectorStore, PipelineStats]:
        """Run the full 5-stage pipeline on all supported files in config.files_dir."""
        t_start = time.perf_counter()
        source_files = self.list_source_files()
        if not source_files:
            raise FileNotFoundError(f"No supported documents found in {self.config.files_dir}")

        # Stage 1 & 2: Extraction + Cleaning
        t_ext = time.perf_counter()
        all_pages: list[ExtractedPage] = []
        doc_pages_map: dict[str, list[ExtractedPage]] = {}

        for file_path in source_files:
            if progress_cb:
                progress_cb(f"Extracting {file_path.name}...")
            pages = extract_document(
                file_path,
                ocr_cache_dir=self.config.ocr_cache_dir,
                ocr_dpi=self.config.ocr_dpi,
                max_ocr_pages=self.config.max_ocr_pages_per_doc,
                progress_cb=progress_cb,
            )
            all_pages.extend(pages)
            doc_pages_map[file_path.name] = pages
        extraction_seconds = round(time.perf_counter() - t_ext, 3)

        # Stage 3: Chunking
        t_chk = time.perf_counter()
        if progress_cb:
            progress_cb("Chunking extracted pages...")
        all_chunks: list[DocumentChunk] = chunk_pages(
            all_pages,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_chars=self.config.min_chunk_chars,
        )
        chunking_seconds = round(time.perf_counter() - t_chk, 3)

        # Build per-document metadata summary
        chunks_by_doc: dict[str, list[DocumentChunk]] = {}
        for ch in all_chunks:
            chunks_by_doc.setdefault(ch.filename, []).append(ch)

        documents_meta: list[dict] = []
        for file_path in source_files:
            pages = doc_pages_map.get(file_path.name, [])
            doc_chunks = chunks_by_doc.get(file_path.name, [])
            total_words = sum(c.word_count for c in doc_chunks)
            ocr_p = sum(1 for p in pages if p.extraction_method == "ocr")
            documents_meta.append(
                {
                    "filename": file_path.name,
                    "size_kb": round(file_path.stat().st_size / 1024, 1),
                    "pages_extracted": len(pages),
                    "ocr_pages": ocr_p,
                    "chunks": len(doc_chunks),
                    "total_words": total_words,
                }
            )

        # Stage 4 & 5: Embedding Generation + Storage
        t_emb = time.perf_counter()
        if progress_cb:
            progress_cb("Generating embeddings and building vector store...")
        store = VectorStore(self.config.vector_store_dir)
        store.build(
            chunks=all_chunks,
            documents_meta=documents_meta,
            svd_components=self.config.svd_components,
        )
        embedding_storage_seconds = round(time.perf_counter() - t_emb, 3)

        total_seconds = round(time.perf_counter() - t_start, 3)
        ocr_pages = sum(1 for p in all_pages if p.extraction_method == "ocr")
        text_pages = len(all_pages) - ocr_pages
        emb_dim = int(store.embeddings.shape[1]) if store.embeddings.size > 0 else 0

        stats = PipelineStats(
            documents_processed=len(source_files),
            total_pages_extracted=len(all_pages),
            ocr_pages=ocr_pages,
            text_pages=text_pages,
            total_chunks=len(all_chunks),
            embedding_dim=emb_dim,
            extraction_seconds=extraction_seconds,
            chunking_seconds=chunking_seconds,
            embedding_storage_seconds=embedding_storage_seconds,
            total_seconds=total_seconds,
            documents_meta=documents_meta,
        )
        return store, stats

