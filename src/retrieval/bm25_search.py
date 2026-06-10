"""
Configuración A — Búsqueda lexical con BM25 (rank-bm25).

Útil para consultas con números de norma, artículos específicos
o términos legales exactos que la búsqueda semántica puede pasar por alto.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank-bm25 no instalado: pip install rank-bm25")


def _tokenize(text: str) -> list[str]:
    """Tokenización simple: minúsculas, solo palabras alfanuméricas (incluye tildes y ñ)."""
    return re.findall(r"[a-záéíóúüñ0-9]+", text.lower())


class BM25Retriever:
    """
    Índice BM25 sobre el corpus de chunks.

    Uso
    ---
    retriever = BM25Retriever(chunks_df)
    results = retriever.search("calidad del aire DS 003-2017", top_k=5)
    """

    def __init__(self, chunks_df: pd.DataFrame) -> None:
        self.chunks_df = chunks_df.reset_index(drop=True)
        corpus_textos = self.chunks_df["texto"].fillna("").tolist()
        tokenized = [_tokenize(t) for t in corpus_textos]
        self._index = BM25Okapi(tokenized)
        print(f"Índice BM25 construido con {len(tokenized)} documentos.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Búsqueda lexical BM25.

        Retorna lista de dicts con campos:
          id_chunk, id_documento, titulo_documento, texto (primeros 300 chars),
          score_bm25, score_vectorial, score_rrf, configuracion.
        """
        query_tokens = _tokenize(query)
        scores = self._index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[dict] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            row = self.chunks_df.iloc[idx]
            results.append(
                {
                    "id_chunk": row.get("id_chunk", ""),
                    "id_documento": row.get("id_documento", ""),
                    "titulo_documento": row.get("titulo_documento", ""),
                    "seccion": row.get("seccion", ""),
                    "texto": row.get("texto", ""),
                    "score_bm25": float(scores[idx]),
                    "score_vectorial": 0.0,
                    "score_rrf": 0.0,
                    "configuracion": "A_BM25",
                }
            )
        return results
