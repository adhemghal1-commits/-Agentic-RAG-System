"""Text cleaning and normalization utilities for extracted document text."""

from __future__ import annotations

import re
import unicodedata


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_SLIDE_FOOTER_RE = re.compile(
    r"(?im)^\s*(?:page\s*\d+(?:\s*of\s*\d+)?|\d+\s*/\s*\d+)\s*$"
)


def clean_text(text: str) -> str:
    """Normalize and clean extracted text while preserving paragraph structure.

    Steps:
    1. Unicode NFKC normalization.
    2. Remove control characters.
    3. Rejoin hyphenated words split across line breaks.
    4. Strip isolated page-number / slide-counter lines.
    5. Collapse excessive spaces and blank lines.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        if _SLIDE_FOOTER_RE.match(line):
            continue
        line = _MULTI_SPACE_RE.sub(" ", line).strip()
        cleaned_lines.append(line)

    joined = "\n".join(cleaned_lines)
    joined = _MULTI_NEWLINE_RE.sub("\n\n", joined)
    return joined.strip()

