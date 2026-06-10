"""Módulo de evaluación: métricas de recuperación y análisis de resultados."""

from .metrics import (
    calcular_recall_at_k,
    calcular_precision_at_k,
    calcular_mrr,
    get_docs_esperados,
)
from .evaluator import Evaluator

__all__ = [
    "calcular_recall_at_k",
    "calcular_precision_at_k",
    "calcular_mrr",
    "get_docs_esperados",
    "Evaluator",
]
