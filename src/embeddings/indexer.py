"""
Indexación de chunks en Qdrant con embeddings BAAI/bge-m3.

Soporta dos modos:
  - "memory" : Qdrant en RAM (sin persistencia), útil para pruebas.
  - "docker"  : Qdrant corriendo en localhost:6333.
  - path      : Qdrant con persistencia en disco (recomendado para Colab / Drive).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kw):  # type: ignore
        return x


COLLECTION_NAME = "normativa_ambiental_chunks_v1"
VECTOR_DIM = 1024
BATCH_SIZE = 32


# ─────────────────────────────────────────────
# Carga de chunks
# ─────────────────────────────────────────────

def load_chunks_jsonl(jsonl_path: Path) -> list[dict]:
    """Carga los chunks desde un archivo JSONL."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL no encontrado: {jsonl_path}")

    chunks: list[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"Chunks cargados: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────
# Conexión a Qdrant
# ─────────────────────────────────────────────

def get_qdrant_client(
    mode: str = "memory",
    host: str = "localhost",
    port: int = 6333,
    storage_path: Optional[Union[str, Path]] = None,
):
    """
    Crea y retorna un cliente Qdrant.

    Parámetros
    ----------
    mode         : "memory" | "docker" | "disk"
    host / port  : Usados solo en modo "docker".
    storage_path : Directorio de persistencia en modo "disk".
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError("qdrant-client no instalado: pip install qdrant-client")

    if mode == "memory":
        client = QdrantClient(":memory:")
        print("Qdrant en modo memoria (sin persistencia).")
    elif mode == "disk" and storage_path:
        # Eliminar lock si existe (relevante en Colab/Drive)
        lock = Path(storage_path) / ".lock"
        if lock.exists():
            lock.unlink()
        client = QdrantClient(path=str(storage_path))
        print(f"Qdrant con persistencia en disco: {storage_path}")
    else:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port)
        print(f"Qdrant conectado a {host}:{port}")

    return client


# ─────────────────────────────────────────────
# Creación de colección
# ─────────────────────────────────────────────

def create_collection(
    client,
    collection_name: str = COLLECTION_NAME,
    vector_dim: int = VECTOR_DIM,
    recreate: bool = False,
) -> None:
    """
    Crea la colección vectorial en Qdrant si no existe.

    recreate=True borra y recrea la colección (útil al reiniciar experimentos).
    """
    from qdrant_client.models import Distance, VectorParams

    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            client.delete_collection(collection_name)
            print(f"Colección '{collection_name}' eliminada.")
        else:
            print(f"Colección '{collection_name}' ya existe. Omitiendo creación.")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Colección '{collection_name}' creada con dim={vector_dim}.")


# ─────────────────────────────────────────────
# Indexación de chunks
# ─────────────────────────────────────────────

def index_chunks(
    chunks: list[dict],
    client,
    encoder,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
    recreate: bool = False,
) -> None:
    """
    Genera embeddings para todos los chunks y los indexa en Qdrant.

    Parámetros
    ----------
    chunks          : Lista de dicts con campo "texto" y metadatos.
    client          : Cliente Qdrant activo.
    encoder         : EmbeddingEncoder con método .encode().
    collection_name : Nombre de la colección Qdrant.
    batch_size      : Tamaño de lote para la generación de embeddings.
    recreate        : Si True, borra y recrea la colección antes de indexar.
    """
    from qdrant_client.models import PointStruct

    create_collection(client, collection_name, recreate=recreate)

    texts = [c.get("texto", "") for c in chunks]
    print(f"Generando embeddings para {len(texts)} chunks …")

    start = time.time()
    all_vectors: list[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embeddings"):
        batch = texts[i : i + batch_size]
        vecs = encoder.encode(batch, is_query=False)
        all_vectors.extend(vecs)

    elapsed = time.time() - start
    print(f"Embeddings generados en {elapsed:.1f} s.")

    # Subir a Qdrant en lotes
    print("Indexando en Qdrant …")
    points: list[PointStruct] = []
    for idx, (chunk, vec) in enumerate(zip(chunks, all_vectors)):
        payload = {k: v for k, v in chunk.items() if k != "texto"}
        payload["texto"] = chunk.get("texto", "")
        points.append(
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload=payload,
            )
        )

    for i in tqdm(range(0, len(points), batch_size), desc="Indexando"):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + batch_size],
        )

    total = client.count(collection_name=collection_name).count
    print(f"Indexación completa. Puntos en colección: {total}")
