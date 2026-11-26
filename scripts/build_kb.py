"""CLI script to build and persist the Knowledge Base from documents in files/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.pipeline import IngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Agentic RAG Knowledge Base.")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--max-ocr-pages", type=int, default=None, help="Optional cap on OCR pages per PDF for quick testing.")
    args = parser.parse_args()

    cfg = PipelineConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        max_ocr_pages_per_doc=args.max_ocr_pages,
    )
    pipeline = IngestionPipeline(cfg)
    store, stats = pipeline.run(progress_cb=lambda msg: print(f"[pipeline] {msg}", flush=True))

    print("\n=== Ingestion Complete ===")
    print(f"Documents processed : {stats.documents_processed}")
    print(f"Total pages         : {stats.total_pages_extracted} (OCR: {stats.ocr_pages}, Text: {stats.text_pages})")
    print(f"Total chunks        : {stats.total_chunks}")
    print(f"Embedding dimension : {stats.embedding_dim}")
    print(f"Timings (s)         : extract={stats.extraction_seconds}, chunk={stats.chunking_seconds}, embed+save={stats.embedding_storage_seconds}, total={stats.total_seconds}")
    for doc in stats.documents_meta:
        print(f"  - {doc['filename']}: {doc['pages_extracted']} pages, {doc['chunks']} chunks, {doc['total_words']} words")


if __name__ == "__main__":
    main()

