"""
Chunking estructural de documentos normativos peruanos.

Respeta artículos, capítulos, títulos y anexos como unidades de segmentación.
Aplica solapamiento solo cuando un bloque supera MAX_WORDS.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .patterns import (
    PAGE_MARKER_RE,
    TITLE_RE,
    CHAPTER_RE,
    ARTICLE_RE,
    ANNEX_RE,
    DISPOSITION_RE,
    LEGAL_HEADER_RE,
)

# ─────────────────────────────────────────────
# Parámetros por defecto
# ─────────────────────────────────────────────
MAX_WORDS = 450
OVERLAP_WORDS = 80
MIN_WORDS = 25


# ─────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────

def split_large_text(
    text: str,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:
    """
    Divide un bloque de texto largo en subchunks con solapamiento de palabras.
    Se aplica únicamente cuando el bloque supera max_words.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def _make_chunk_id(id_documento: str, orden: int, suborden: int) -> str:
    return f"{id_documento}_C{orden:04d}_{suborden:02d}"


def _detect_section(line: str) -> Optional[str]:
    """Detecta el tipo de sección de la línea."""
    stripped = line.strip()
    if LEGAL_HEADER_RE.match(stripped):
        return "encabezado_legal"
    if TITLE_RE.match(stripped):
        return "titulo"
    if CHAPTER_RE.match(stripped):
        return "capitulo"
    if DISPOSITION_RE.match(stripped):
        return "disposicion"
    if ANNEX_RE.match(stripped):
        return "anexo"
    if ARTICLE_RE.match(stripped):
        return "articulo"
    return None


def _create_chunk_record(
    row: dict,
    text: str,
    tipo_chunk: str,
    seccion: str,
    titulo_seccion: str,
    capitulo: str,
    pagina_aprox: int,
    orden_chunk: int,
    suborden_chunk: int,
) -> dict:
    """Construye el diccionario de metadatos de un chunk."""
    id_chunk = _make_chunk_id(str(row["id_documento"]), orden_chunk, suborden_chunk)
    return {
        "id_chunk": id_chunk,
        "id_documento": row.get("id_documento", ""),
        "archivo_pdf": row.get("archivo_pdf", ""),
        "archivo_txt": row.get("archivo_txt", f"{row.get('id_documento', '')}.txt"),
        "titulo_documento": row.get("titulo_documento", ""),
        "tipo_norma": row.get("tipo_norma", ""),
        "numero_norma": row.get("numero_norma", ""),
        "entidad_emisora": row.get("entidad_emisora", ""),
        "fecha_publicacion": row.get("fecha_publicacion", ""),
        "tema_principal": row.get("tema_principal", ""),
        "subtema": row.get("subtema", ""),
        "fuente_oficial": row.get("fuente_oficial", ""),
        "url_fuente": row.get("url_fuente", ""),
        "estado_vigencia": row.get("estado_vigencia", "no_verificado"),
        "tipo_chunk": tipo_chunk,
        "seccion": seccion,
        "titulo_seccion": titulo_seccion,
        "capitulo": capitulo,
        "pagina_aprox": pagina_aprox,
        "texto": text,
        "n_palabras": len(text.split()),
        "orden_chunk": orden_chunk,
        "suborden_chunk": suborden_chunk,
    }


# ─────────────────────────────────────────────
# Chunking de un documento
# ─────────────────────────────────────────────

def chunk_document(
    text: str,
    row: dict,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    min_words: int = MIN_WORDS,
) -> list[dict]:
    """
    Segmenta el texto de un documento normativo en chunks estructurales.

    Prioriza la división por artículos, capítulos, títulos y anexos.
    Cuando un bloque supera max_words, aplica subdivisión con solapamiento.

    Parámetros
    ----------
    text          : Texto completo del documento (con marcadores de página).
    row           : Dict / Series con metadatos del documento.
    max_words     : Tamaño máximo de chunk en palabras.
    overlap_words : Solapamiento al subdividir bloques grandes.
    min_words     : Tamaño mínimo para conservar un chunk.

    Retorna
    -------
    Lista de dicts con texto y metadatos de cada chunk.
    """
    lines = text.split("\n")
    chunks: list[dict] = []

    current_lines: list[str] = []
    current_section = "inicio"
    current_titulo = ""
    current_capitulo = ""
    current_tipo = "general"
    current_page = 1
    orden = 0

    def flush_block() -> None:
        nonlocal orden
        block_text = "\n".join(current_lines).strip()
        if len(block_text.split()) < min_words:
            return
        subblocks = split_large_text(block_text, max_words, overlap_words)
        for sub_idx, sub in enumerate(subblocks):
            if len(sub.split()) < min_words:
                continue
            orden += 1
            chunks.append(
                _create_chunk_record(
                    row=row,
                    text=sub,
                    tipo_chunk=current_tipo,
                    seccion=current_section,
                    titulo_seccion=current_titulo,
                    capitulo=current_capitulo,
                    pagina_aprox=current_page,
                    orden_chunk=orden,
                    suborden_chunk=sub_idx,
                )
            )

    for line in lines:
        # Detectar marcador de página
        pm = PAGE_MARKER_RE.match(line.strip())
        if pm:
            current_page = int(pm.group(1))
            continue

        section_type = _detect_section(line)

        if section_type == "titulo":
            flush_block()
            current_lines = [line]
            current_section = line.strip()
            current_titulo = line.strip()
            current_tipo = "titulo"

        elif section_type == "capitulo":
            flush_block()
            current_lines = [line]
            current_section = line.strip()
            current_capitulo = line.strip()
            current_tipo = "capitulo"

        elif section_type in ("articulo", "disposicion", "anexo", "encabezado_legal"):
            flush_block()
            current_lines = [line]
            current_section = line.strip()
            current_tipo = section_type

        else:
            current_lines.append(line)

    # Flush del último bloque
    flush_block()
    return chunks


# ─────────────────────────────────────────────
# Chunking del corpus completo
# ─────────────────────────────────────────────

def chunk_corpus(
    df: pd.DataFrame,
    processed_dir: Path,
    chunks_dir: Path,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
    min_words: int = MIN_WORDS,
    output_jsonl: str = "chunks_normativa_v1.jsonl",
    output_csv: str = "chunks_normativa_v1.csv",
) -> list[dict]:
    """
    Procesa todo el corpus, genera chunks estructurales y los guarda en disco.

    Salidas
    -------
    chunks_dir/output_jsonl : Un chunk por línea JSON.
    chunks_dir/output_csv   : Tabla con todos los chunks.

    Retorna lista de todos los chunks generados.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []

    for _, row in df.iterrows():
        id_doc = str(row["id_documento"])
        txt_name = row.get("archivo_txt", f"{id_doc}.txt")
        if not txt_name or str(txt_name) == "nan":
            txt_name = f"{id_doc}.txt"
        txt_path = processed_dir / str(txt_name)

        if not txt_path.exists():
            print(f"  [OMITIDO] Sin texto para {id_doc}: {txt_path}")
            continue

        text = txt_path.read_text(encoding="utf-8")
        doc_chunks = chunk_document(
            text=text,
            row=row.to_dict() if hasattr(row, "to_dict") else dict(row),
            max_words=max_words,
            overlap_words=overlap_words,
            min_words=min_words,
        )
        all_chunks.extend(doc_chunks)
        print(f"  {id_doc}: {len(doc_chunks)} chunks")

    # Guardar JSONL
    jsonl_path = chunks_dir / output_jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Guardar CSV
    csv_path = chunks_dir / output_csv
    pd.DataFrame(all_chunks).to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\nTotal chunks generados: {len(all_chunks)}")
    print(f"JSONL guardado en: {jsonl_path}")
    print(f"CSV guardado en: {csv_path}")

    return all_chunks
