"""
Generación de embeddings semánticos con BAAI/bge-m3 (FlagEmbedding).

Clase principal: EmbeddingEncoder — wrapper singleton para el modelo.
Función auxiliar: encode_texts — interfaz rápida sin instanciar manualmente.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_DEFAULT_MODEL = "BAAI/bge-m3"
_VECTOR_DIM = 1024


class EmbeddingEncoder:
    """
    Wrapper sobre BGEM3FlagModel para generar embeddings densos normalizados.

    Uso
    ---
    encoder = EmbeddingEncoder()
    vectors = encoder.encode(["texto 1", "texto 2"])          # shape (2, 1024)
    query_vec = encoder.encode(["consulta"], is_query=True)[0] # shape (1024,)
    """

    _instance: Optional["EmbeddingEncoder"] = None

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        use_fp16: bool = True,
        batch_size_docs: int = 4,
        batch_size_query: int = 1,
        max_length: int = 512,
    ) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding no está instalado.\n"
                "Instala con: pip install FlagEmbedding==1.2.11"
            )

        self.model_name = model_name
        self.batch_size_docs = batch_size_docs
        self.batch_size_query = batch_size_query
        self.max_length = max_length

        print(f"Cargando modelo {model_name} …")
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        print("Modelo cargado correctamente.")

    @classmethod
    def get_instance(cls, **kwargs) -> "EmbeddingEncoder":
        """Singleton: devuelve la instancia existente o crea una nueva."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> np.ndarray:
        """
        Genera embeddings densos y normalizados (norma L2 = 1).

        Parámetros
        ----------
        texts    : Lista de textos a codificar.
        is_query : True para consultas (batch_size=1); False para documentos.

        Retorna
        -------
        numpy array float32 de forma (len(texts), 1024).
        """
        batch_size = self.batch_size_query if is_query else self.batch_size_docs

        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = np.array(output["dense_vecs"], dtype=np.float32)

        # Normalización L2
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


# ─────────────────────────────────────────────
# Función de conveniencia
# ─────────────────────────────────────────────

def encode_texts(
    texts: list[str],
    is_query: bool = False,
    encoder: Optional[EmbeddingEncoder] = None,
) -> np.ndarray:
    """
    Codifica una lista de textos usando el encoder singleton.

    Si no se pasa un encoder, usa EmbeddingEncoder.get_instance().
    """
    enc = encoder or EmbeddingEncoder.get_instance()
    return enc.encode(texts, is_query=is_query)
