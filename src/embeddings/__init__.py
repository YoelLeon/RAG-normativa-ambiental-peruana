"""Módulo de embeddings: generación de vectores semánticos con BAAI/bge-m3."""

from .encoder import EmbeddingEncoder, encode_texts
from .indexer import index_chunks, load_chunks_jsonl

__all__ = [
    "EmbeddingEncoder",
    "encode_texts",
    "index_chunks",
    "load_chunks_jsonl",
]
