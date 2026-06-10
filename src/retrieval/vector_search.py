"""
Configuración B — Búsqueda semántica vectorial con Qdrant + BAAI/bge-m3.

Recupera fragmentos por similitud coseno sobre vectores de 1024 dimensiones.
Soporta filtros opcionales por metadatos (tema, tipo_norma, estado_vigencia, etc.).
"""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient

COLLECTION_NAME = "normativa_ambiental_chunks_v1"


class VectorRetriever:
    """
    Búsqueda semántica sobre la colección Qdrant.

    Uso
    ---
    retriever = VectorRetriever(client, encoder)
    results = retriever.search("calidad del aire", top_k=5)
    results = retriever.search("residuos sólidos", filtros={"tema_principal": "Residuos sólidos"})
    """

    def __init__(
        self,
        client: QdrantClient,
        encoder,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._encoder = encoder
        self._collection = collection_name

    def search(
        self,
        query: str,
        top_k: int = 5,
        filtros: Optional[dict] = None,
    ) -> list[dict]:
        """
        Búsqueda semántica vectorial.

        Parámetros
        ----------
        query   : Texto de la consulta.
        top_k   : Número de resultados a retornar.
        filtros : Dict opcional con pares campo:valor para filtrar por metadatos.
                  Ejemplo: {"tema_principal": "Calidad ambiental"}

        Retorna
        -------
        Lista de dicts con campos:
          id_chunk, id_documento, titulo_documento, texto,
          score_bm25, score_vectorial, score_rrf, configuracion.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_vector = self._encoder.encode([query], is_query=True)[0]

        query_filter = None
        if filtros:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filtros.items()
            ]
            query_filter = Filter(must=conditions)

        hits = self._client.query_points(
            collection_name=self._collection,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        ).points

        results: list[dict] = []
        for hit in hits:
            p = hit.payload or {}
            results.append(
                {
                    "id_chunk": p.get("id_chunk", ""),
                    "id_documento": p.get("id_documento", ""),
                    "titulo_documento": p.get("titulo_documento", ""),
                    "seccion": p.get("seccion", ""),
                    "texto": p.get("texto", ""),
                    "score_bm25": 0.0,
                    "score_vectorial": float(hit.score),
                    "score_rrf": 0.0,
                    "configuracion": "B_VECTORIAL",
                }
            )
        return results
