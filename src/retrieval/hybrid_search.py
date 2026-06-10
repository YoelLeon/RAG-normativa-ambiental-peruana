"""
Configuración C — Búsqueda híbrida: BM25 + Vectorial fusionados con RRF.

Reciprocal Rank Fusion (RRF) combina los rankings de ambos métodos sin
necesidad de normalizar sus puntuaciones individualmente.

Referencia: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient

COLLECTION_NAME = "normativa_ambiental_chunks_v1"
RRF_K = 60  # Parámetro estándar de RRF


# ─────────────────────────────────────────────
# RRF puro (función reutilizable)
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """
    Fusiona múltiples rankings con Reciprocal Rank Fusion.

    Parámetros
    ----------
    rankings : Lista de listas de IDs, cada una ordenada por relevancia descendente.
    k        : Constante de suavizado (60 por defecto según la literatura).

    Retorna
    -------
    Dict {id: score_rrf} con puntuaciones fusionadas.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)
    return dict(rrf_scores)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ0-9]+", text.lower())


# ─────────────────────────────────────────────
# HybridRetriever
# ─────────────────────────────────────────────

class HybridRetriever:
    """
    Búsqueda híbrida BM25 + Vectorial con fusión RRF.

    Uso
    ---
    retriever = HybridRetriever(bm25_retriever, vector_retriever, chunks_df)
    results = retriever.search("límites de ruido ambiental", top_k=5)
    """

    def __init__(
        self,
        bm25_retriever,
        vector_retriever,
        chunks_df: pd.DataFrame,
        rrf_k: int = RRF_K,
    ) -> None:
        self._bm25 = bm25_retriever
        self._vec = vector_retriever
        self._chunks_df = chunks_df.reset_index(drop=True)
        # Índice rápido id_chunk → fila del DataFrame
        self._chunk_index: dict[str, dict] = {
            row["id_chunk"]: row.to_dict()
            for _, row in self._chunks_df.iterrows()
        }
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        top_k_candidatos: int = 20,
    ) -> list[dict]:
        """
        Búsqueda híbrida con RRF.

        Parámetros
        ----------
        query             : Texto de la consulta.
        top_k             : Número de resultados finales a retornar.
        top_k_candidatos  : Candidatos tomados de cada método antes de fusionar.

        Retorna
        -------
        Lista de dicts con campos de metadatos y puntuaciones BM25, vectorial y RRF.
        """
        # ── Candidatos BM25 ──
        bm25_results = self._bm25.search(query, top_k=top_k_candidatos)
        bm25_ranking = [r["id_chunk"] for r in bm25_results]
        bm25_score_map = {r["id_chunk"]: r["score_bm25"] for r in bm25_results}

        # ── Candidatos vectoriales ──
        vec_results = self._vec.search(query, top_k=top_k_candidatos)
        vec_ranking = [r["id_chunk"] for r in vec_results]
        vec_score_map = {r["id_chunk"]: r["score_vectorial"] for r in vec_results}
        vec_payload_map = {r["id_chunk"]: r for r in vec_results}

        # ── Fusión RRF ──
        rrf_scores = reciprocal_rank_fusion(
            [bm25_ranking, vec_ranking], k=self._rrf_k
        )
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for chunk_id, rrf_score in sorted_chunks[:top_k]:
            # Recuperar payload del índice local o del resultado vectorial
            if chunk_id in self._chunk_index:
                payload = self._chunk_index[chunk_id]
            elif chunk_id in vec_payload_map:
                payload = vec_payload_map[chunk_id]
            else:
                continue

            results.append(
                {
                    "id_chunk": chunk_id,
                    "id_documento": payload.get("id_documento", ""),
                    "titulo_documento": payload.get("titulo_documento", ""),
                    "seccion": payload.get("seccion", ""),
                    "texto": payload.get("texto", ""),
                    "score_bm25": bm25_score_map.get(chunk_id, 0.0),
                    "score_vectorial": vec_score_map.get(chunk_id, 0.0),
                    "score_rrf": rrf_score,
                    "configuracion": "C_HIBRIDA_RRF",
                }
            )
        return results
