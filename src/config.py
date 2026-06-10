"""
Configuración central del proyecto RAG — Normativa Ambiental Peruana.

Todas las rutas son relativas a ROOT_DIR (raíz del repositorio).
Nunca hay rutas hardcodeadas a Google Drive ni rutas absolutas de máquina.

Para cargar variables de entorno (.env en local, Secrets en Colab):
    from src.config import settings
    api_key = settings.groq_api_key
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Carga de .env (local). En Colab las vars ya vienen del entorno.
# python-dotenv ignora silenciosamente si el archivo no existe.
# ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    # Busca .env en ROOT_DIR (dos niveles arriba de este archivo)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass  # En Colab no es necesario; las vars vienen de userdata/os.environ


# ─────────────────────────────────────────────────────────────
# Resolución de ROOT_DIR
# ─────────────────────────────────────────────────────────────
def _resolve_root() -> Path:
    """
    Determina ROOT_DIR de forma robusta para local, Colab y Streamlit.

    Orden de prioridad:
      1. Variable de entorno RAG_ROOT_DIR (override explícito).
      2. Inferido desde la ubicación de este archivo (src/config.py → ../.).
    """
    env_root = os.environ.get("RAG_ROOT_DIR")
    if env_root:
        return Path(env_root).resolve()
    # src/config.py está en <root>/src/config.py  →  parent.parent = root
    return Path(__file__).resolve().parent.parent


ROOT_DIR = _resolve_root()

# ─────────────────────────────────────────────────────────────
# Rutas estándar del proyecto (todas relativas a ROOT_DIR)
# ─────────────────────────────────────────────────────────────
DATA_DIR         = ROOT_DIR / "data"
RAW_DIR          = DATA_DIR / "raw"
PROCESSED_DIR    = DATA_DIR / "processed"
CHUNKS_DIR       = DATA_DIR / "chunks"
METADATA_DIR     = DATA_DIR / "metadata"
EMBEDDINGS_DIR   = DATA_DIR / "embeddings"   # para guardar .npy si se quiere

QDRANT_STORAGE   = ROOT_DIR / "qdrant_storage"
REPORTS_DIR      = ROOT_DIR / "experiments" / "resultados"

# Archivos clave
CORPUS_CSV       = METADATA_DIR / "corpus_normativo_ambiental.csv"
CORPUS_EXTRACTION_CSV = METADATA_DIR / "corpus_normativo_ambiental_con_extraccion.csv"
CHUNKS_JSONL     = CHUNKS_DIR / "chunks_normativa_v1.jsonl"
CHUNKS_CSV       = CHUNKS_DIR / "chunks_normativa_v1.csv"
PREGUNTAS_CSV    = METADATA_DIR / "banco_preguntas_evaluacion.csv"

# ─────────────────────────────────────────────────────────────
# Parámetros del pipeline
# ─────────────────────────────────────────────────────────────

# Chunking
MAX_WORDS     = 450
OVERLAP_WORDS = 80
MIN_WORDS     = 25

# Embeddings
MODEL_NAME      = "BAAI/bge-m3"
EMBEDDING_BACKEND = "bge_m3_flagembedding"   # o "sentence_transformers"
VECTOR_DIM      = 1024
BATCH_SIZE      = 1       # conservador para BGE-M3 en CPU; subir en GPU
MAX_LENGTH      = 512
PASSAGE_PREFIX  = ""      # BGE-M3 no requiere prefijo
QUERY_PREFIX    = ""

# Qdrant
COLLECTION_NAME = "normativa_ambiental_chunks_v1"
QDRANT_MODE     = "disk"  # "memory" | "disk" | "docker"
QDRANT_HOST     = "localhost"
QDRANT_PORT     = 6333

# Recuperación
TOP_K               = 5
TOP_K_CANDIDATOS    = 20
RRF_K               = 60

# Generación
GROQ_MODEL   = "llama-3.1-8b-instant"
MAX_TOKENS   = 1024


# ─────────────────────────────────────────────────────────────
# Acceso a variables de entorno sensibles
# ─────────────────────────────────────────────────────────────

class _Settings:
    """Acceso centralizado a variables de entorno con mensajes claros."""

    @property
    def groq_api_key(self) -> str:
        # 1. Variable de entorno estándar (cargada por dotenv en local
        #    o por Colab Secrets via userdata / os.environ)
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY no encontrada.\n"
                "  · Local: define la variable en el archivo .env\n"
                "  · Colab:  añádela en Secrets (icono 🔑) con nombre GROQ_API_KEY"
            )
        return key


settings = _Settings()
