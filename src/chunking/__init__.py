"""Módulo de chunking: segmentación estructural de textos normativos."""

from .chunker import chunk_document, chunk_corpus, split_large_text
from .patterns import (
    PAGE_MARKER_RE,
    TITLE_RE,
    CHAPTER_RE,
    ARTICLE_RE,
    ANNEX_RE,
    DISPOSITION_RE,
    LEGAL_HEADER_RE,
    is_normative_header,
)

__all__ = [
    "chunk_document",
    "chunk_corpus",
    "split_large_text",
    "PAGE_MARKER_RE",
    "TITLE_RE",
    "CHAPTER_RE",
    "ARTICLE_RE",
    "ANNEX_RE",
    "DISPOSITION_RE",
    "LEGAL_HEADER_RE",
    "is_normative_header",
]
