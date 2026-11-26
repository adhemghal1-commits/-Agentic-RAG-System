"""Multi-format document text extraction with automatic OCR fallback and disk caching."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from src.pipeline.cleaner import clean_text


@dataclass
class ExtractedPage:
    """Extracted and cleaned text for a single page or section of a document."""

    doc_id: str
    filename: str
    page: int
    text: str
    extraction_method: str  # "text", "ocr", "csv", "plain"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractedPage":
        return cls(**data)


_OCR_ENGINE = None


def _resize_image_for_ocr(img_bytes: bytes, max_dim: int = 1280) -> bytes:
    """Downscale oversized slide images before OCR for faster CPU inference."""
    from PIL import Image

    im = Image.open(io.BytesIO(img_bytes))
    w, h = im.size
    if max(w, h) <= max_dim:
        return img_bytes
    scale = max_dim / float(max(w, h))
    resized = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


def _get_ocr_engine():
    """Lazy-load RapidOCR engine once per process."""
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _doc_id_for_path(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    short_hash = hashlib.md5(str(path.name).encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{short_hash}"


def _cache_key(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extract_pdf(
    path: Path,
    ocr_cache_dir: Path | None = None,
    ocr_dpi: int = 150,
    max_ocr_pages: int | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> list[ExtractedPage]:
    """Extract text from PDF, using RapidOCR on image-only/scanned pages."""
    doc_id = _doc_id_for_path(path)
    cache_file = None
    if ocr_cache_dir is not None:
        ocr_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = ocr_cache_dir / f"{path.stem}_{_cache_key(path)}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return [ExtractedPage.from_dict(item) for item in cached]

    pages: list[ExtractedPage] = []
    # Prefer PyMuPDF (fitz) when available for both text extraction and fast page rendering
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        total_pages = len(doc)
        ocr_count = 0

        for idx in range(total_pages):
            page_num = idx + 1
            page = doc[idx]
            raw_text = clean_text(page.get_text("text") or "")
            if len(raw_text) >= 25:
                pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        filename=path.name,
                        page=page_num,
                        text=raw_text,
                        extraction_method="text",
                    )
                )
                continue

            if max_ocr_pages is not None and ocr_count >= max_ocr_pages:
                continue

            if progress_cb and (page_num == 1 or page_num % 10 == 0 or page_num == total_pages):
                progress_cb(f"OCR {path.name}: page {page_num}/{total_pages}")

            pix = page.get_pixmap(dpi=ocr_dpi)
            img_bytes = pix.tobytes("png")
            ocr_engine = _get_ocr_engine()
            result, _ = ocr_engine(img_bytes)
            ocr_count += 1

            lines = [item[1] for item in (result or []) if len(item) >= 2 and item[1]]
            cleaned = clean_text("\n".join(lines))
            if cleaned:
                pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        filename=path.name,
                        page=page_num,
                        text=cleaned,
                        extraction_method="ocr",
                    )
                )
        doc.close()
    except ImportError:
        # Fallback to pypdf + embedded images if PyMuPDF is not installed
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        ocr_count = 0
        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = clean_text(page.extract_text() or "")
            if len(raw_text) >= 25:
                pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        filename=path.name,
                        page=page_num,
                        text=raw_text,
                        extraction_method="text",
                    )
                )
                continue
            if page.images and (max_ocr_pages is None or ocr_count < max_ocr_pages):
                ocr_engine = _get_ocr_engine()
                page_lines: list[str] = []
                for img in page.images:
                    img_data = _resize_image_for_ocr(img.data)
                    result, _ = ocr_engine(img_data)
                    page_lines.extend(
                        item[1] for item in (result or []) if len(item) >= 2 and item[1]
                    )
                ocr_count += 1
                if progress_cb and (page_num == 1 or page_num % 5 == 0 or page_num == total_pages):
                    progress_cb(f"OCR {path.name}: page {page_num}/{total_pages}")
                cleaned = clean_text("\n".join(page_lines))
                if cleaned:
                    pages.append(
                        ExtractedPage(
                            doc_id=doc_id,
                            filename=path.name,
                            page=page_num,
                            text=cleaned,
                            extraction_method="ocr",
                        )
                    )

    if cache_file is not None and pages and max_ocr_pages is None:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in pages], f, ensure_ascii=False, indent=2)

    return pages


def _extract_plain(path: Path) -> list[ExtractedPage]:
    doc_id = _doc_id_for_path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    cleaned = clean_text(raw)
    if not cleaned:
        return []
    return [
        ExtractedPage(
            doc_id=doc_id,
            filename=path.name,
            page=1,
            text=cleaned,
            extraction_method="plain",
        )
    ]


def _extract_csv(path: Path) -> list[ExtractedPage]:
    doc_id = _doc_id_for_path(path)
    raw = path.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        return []
    header = rows[0]
    lines = [", ".join(header)]
    for row in rows[1:]:
        pairs = [f"{col}: {val}" for col, val in zip(header, row)]
        lines.append(" | ".join(pairs))
    cleaned = clean_text("\n".join(lines))
    return [
        ExtractedPage(
            doc_id=doc_id,
            filename=path.name,
            page=1,
            text=cleaned,
            extraction_method="csv",
        )
    ]


def extract_document(
    path: Path,
    ocr_cache_dir: Path | None = None,
    ocr_dpi: int = 150,
    max_ocr_pages: int | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> list[ExtractedPage]:
    """Extract cleaned pages from a PDF, TXT, MD, JSON, or CSV file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(
            path,
            ocr_cache_dir=ocr_cache_dir,
            ocr_dpi=ocr_dpi,
            max_ocr_pages=max_ocr_pages,
            progress_cb=progress_cb,
        )
    if suffix == ".csv":
        return _extract_csv(path)
    if suffix in {".txt", ".md", ".json"}:
        return _extract_plain(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")

