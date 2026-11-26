"""Central configuration for the Agentic RAG System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / "files"
DATA_DIR = BASE_DIR / "data"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"


@dataclass
class PipelineConfig:
    """Configuration for document processing, chunking, and vector storage."""

    files_dir: Path = FILES_DIR
    ocr_cache_dir: Path = OCR_CACHE_DIR
    vector_store_dir: Path = VECTOR_STORE_DIR
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_chars: int = 60
    svd_components: int = 128
    ocr_dpi: int = 150
    max_ocr_pages_per_doc: int | None = None


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""

    provider: str = "auto"  # "auto", "local", "gemini", "openai"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2

    def resolved_provider(self) -> str:
        """Resolve 'auto' to an available provider based on API keys."""
        if self.provider != "auto":
            return self.provider
        if self.api_key:
            return "gemini" if self.model.startswith("gemini") else "openai"
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "local"


@dataclass
class AgentConfig:
    """Configuration for the Agentic RAG loop."""

    max_steps: int = 6
    default_top_k: int = 5
    retrieval_strategy: str = "hybrid"  # "hybrid", "semantic", "bm25"
    min_relevance_threshold: float = 0.15
    llm: LLMConfig = field(default_factory=LLMConfig)

