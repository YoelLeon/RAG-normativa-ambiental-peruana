"""
Evaluador de las tres configuraciones de recuperación (A, B, C)
sobre el banco de preguntas con documentos de referencia.

Genera:
  - DataFrame con resultados por pregunta y configuración.
  - Tablas resumen: global, por tipo de pregunta, por dificultad.
  - Exporta CSV y JSON a experiments/resultados/.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .metrics import (
    calcular_recall_at_k,
    calcular_precision_at_k,
    calcular_mrr,
    get_docs_esperados,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


class Evaluator:
    """
    Evalúa múltiples estrategias de recuperación sobre un banco de preguntas.

    Uso
    ---
    evaluator = Evaluator(preguntas_df, results_dir)
    evaluator.add_config("A_BM25", bm25_retriever.search)
    evaluator.add_config("B_VECTORIAL", vector_retriever.search)
    evaluator.add_config("C_HIBRIDA_RRF", hybrid_retriever.search)
    summary = evaluator.run(top_k=5)
    evaluator.save_results()
    """

    def __init__(
        self,
        preguntas_df: pd.DataFrame,
        results_dir: Optional[Path] = None,
        top_k: int = 5,
    ) -> None:
        self._preguntas_df = preguntas_df
        self._configs: dict[str, Callable] = {}
        self._results: list[dict] = []
        self._top_k = top_k
        self._results_dir = results_dir

        if results_dir:
            results_dir.mkdir(parents=True, exist_ok=True)

    def add_config(self, name: str, search_fn: Callable) -> None:
        """Registra una configuración de búsqueda con nombre y función."""
        self._configs[name] = search_fn

    def run(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """
        Ejecuta la evaluación sobre todas las configuraciones registradas.

        Excluye automáticamente las preguntas trampa del cálculo de métricas
        (las evalúa por separado para detectar alucinaciones).

        Retorna
        -------
        DataFrame con métricas agregadas por configuración.
        """
        k = top_k or self._top_k
        self._results = []

        preguntas_eval = self._preguntas_df[
            self._preguntas_df["tipo_pregunta"] != "trampa_alucinacion"
        ].copy()
        preguntas_trampa = self._preguntas_df[
            self._preguntas_df["tipo_pregunta"] == "trampa_alucinacion"
        ].copy()

        print(f"Preguntas de evaluación: {len(preguntas_eval)}")
        print(f"Preguntas trampa: {len(preguntas_trampa)}")

        for config_nombre, search_fn in self._configs.items():
            print(f"\nEvaluando: {config_nombre}")
            for _, fila in tqdm(preguntas_eval.iterrows(), total=len(preguntas_eval)):
                pregunta = fila["pregunta"]
                docs_esperados = get_docs_esperados(fila.get("doc_esperado", ""))

                t0 = time.time()
                try:
                    fragmentos = search_fn(pregunta, top_k=k)
                except Exception as exc:
                    print(f"  Error en {config_nombre}: {exc}")
                    fragmentos = []
                latencia = time.time() - t0

                self._results.append(
                    {
                        "configuracion": config_nombre,
                        "id_pregunta": fila.get("id_pregunta", ""),
                        "pregunta": pregunta,
                        "tipo_pregunta": fila.get("tipo_pregunta", ""),
                        "dificultad": fila.get("dificultad", ""),
                        "doc_esperado": fila.get("doc_esperado", ""),
                        "docs_recuperados": ", ".join(
                            r["id_documento"] for r in fragmentos
                        ),
                        "recall_at_5": calcular_recall_at_k(fragmentos, docs_esperados, k),
                        "precision_at_5": calcular_precision_at_k(fragmentos, docs_esperados, k),
                        "mrr": calcular_mrr(fragmentos, docs_esperados),
                        "latencia_seg": round(latencia, 3),
                    }
                )

            # Resumen por configuración
            config_rows = [r for r in self._results if r["configuracion"] == config_nombre]
            print(
                f"  Recall@{k}:    {np.mean([r['recall_at_5'] for r in config_rows]):.3f}"
            )
            print(
                f"  Precision@{k}: {np.mean([r['precision_at_5'] for r in config_rows]):.3f}"
            )
            print(f"  MRR:          {np.mean([r['mrr'] for r in config_rows]):.3f}")

        return self.summary()

    def summary(self) -> pd.DataFrame:
        """Tabla resumen de métricas agregadas por configuración."""
        if not self._results:
            raise RuntimeError("Ejecuta run() antes de llamar a summary().")

        df = pd.DataFrame(self._results)
        return (
            df.groupby("configuracion")
            .agg(
                Recall_5=("recall_at_5", "mean"),
                Precision_5=("precision_at_5", "mean"),
                MRR=("mrr", "mean"),
                Latencia_prom=("latencia_seg", "mean"),
                N_preguntas=("id_pregunta", "count"),
            )
            .round(4)
            .reset_index()
        )

    def summary_by_type(self) -> pd.DataFrame:
        """Métricas desagregadas por tipo de pregunta."""
        df = pd.DataFrame(self._results)
        return (
            df.groupby(["configuracion", "tipo_pregunta"])
            .agg(
                Recall_5=("recall_at_5", "mean"),
                MRR=("mrr", "mean"),
                N=("id_pregunta", "count"),
            )
            .round(4)
            .reset_index()
        )

    def summary_by_difficulty(self) -> pd.DataFrame:
        """Métricas desagregadas por dificultad."""
        df = pd.DataFrame(self._results)
        return (
            df.groupby(["configuracion", "dificultad"])
            .agg(
                Recall_5=("recall_at_5", "mean"),
                MRR=("mrr", "mean"),
                N=("id_pregunta", "count"),
            )
            .round(4)
            .reset_index()
        )

    def save_results(self, tag: str = "") -> None:
        """
        Guarda los resultados detallados y las tablas resumen en results_dir.
        """
        if not self._results_dir:
            print("results_dir no configurado. Resultados no guardados.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{tag}" if tag else ""

        df = pd.DataFrame(self._results)

        # CSV detallado
        csv_path = self._results_dir / f"evaluacion_detallada{suffix}_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"CSV detallado guardado: {csv_path}")

        # JSON resumen
        json_path = self._results_dir / f"resumen_metricas{suffix}_{timestamp}.json"
        resumen = {
            "timestamp": timestamp,
            "global": self.summary().to_dict(orient="records"),
            "por_tipo": self.summary_by_type().to_dict(orient="records"),
            "por_dificultad": self.summary_by_difficulty().to_dict(orient="records"),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resumen, f, ensure_ascii=False, indent=2)
        print(f"JSON resumen guardado: {json_path}")
