"""Módulo de ingesta: lectura y procesamiento inicial de documentos PDF."""

from .extractor import extract_text_with_pymupdf, clean_text, process_corpus
from .loader import load_corpus_csv, load_processed_texts

__all__ = [
    "extract_text_with_pymupdf",
    "clean_text",
    "process_corpus",
    "load_corpus_csv",
    "load_processed_texts",
]
