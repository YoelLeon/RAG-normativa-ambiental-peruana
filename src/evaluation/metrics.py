"""
Métricas de evaluación de recuperación de información.

  Recall@K    : ¿Cuántos documentos esperados aparecen en los top-K resultados?
  Precision@K : ¿Qué fracción de los top-K resultados es relevante?
  MRR         : Inverso de la posición del primer resultado relevante.
"""

from __future__ import annotations

import pandas as pd


def get_docs_esperados(doc_esperado_str: str | float) -> list[str]:
    """
    Parsea el campo doc_esperado del banco de preguntas.
    Acepta IDs separados por coma. Retorna lista vacía si es NaN o 'NINGUNO'.
    """
    if pd.isna(doc_esperado_str) or str(doc_esperado_str).strip().upper() == "NINGUNO":
        return []
    return [d.strip() for d in str(doc_esperado_str).split(",") if d.strip()]


def calcular_recall_at_k(
    docs_recuperados: list[dict],
    docs_esperados: list[str],
    k: int = 5,
) -> float:
    """
    Recall@K: fracción de documentos esperados que aparecen en los top-K resultados.

    Si hay múltiples documentos esperados (pregunta multi-documento),
    se cuenta cuántos de ellos están en los top-K (capped a 1.0).
    """
    if not docs_esperados:
        return 0.0
    recuperados = {r["id_documento"] for r in docs_recuperados[:k]}
    hits = len(set(docs_esperados) & recuperados)
    return min(hits / len(docs_esperados), 1.0)


def calcular_precision_at_k(
    docs_recuperados: list[dict],
    docs_esperados: list[str],
    k: int = 5,
) -> float:
    """
    Precision@K: proporción de resultados top-K que son relevantes.
    """
    if not docs_recuperados or not docs_esperados:
        return 0.0
    recuperados = [r["id_documento"] for r in docs_recuperados[:k]]
    hits = sum(1 for d in recuperados if d in docs_esperados)
    return hits / min(k, len(recuperados))


def calcular_mrr(
    docs_recuperados: list[dict],
    docs_esperados: list[str],
) -> float:
    """
    MRR (Mean Reciprocal Rank): inverso de la posición del primer resultado relevante.

    Retorna 0.0 si ningún resultado relevante aparece en la lista.
    """
    if not docs_esperados:
        return 0.0
    for rank, r in enumerate(docs_recuperados, start=1):
        if r["id_documento"] in docs_esperados:
            return 1.0 / rank
    return 0.0
