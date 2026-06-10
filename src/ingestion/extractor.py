"""
Extracción de texto de PDFs usando PyMuPDF.
Implementa limpieza básica y preserva marcadores de página para chunking estructural.
"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import fitz  # PyMuPDF
except ImportError:
    raise ImportError(
        "PyMuPDF no está instalado. Instala con: pip install pymupdf"
    )


# ─────────────────────────────────────────────
# Limpieza de texto
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Limpieza básica del texto extraído de PDFs.
    - Normaliza saltos de línea.
    - Elimina espacios redundantes por línea.
    - Colapsa líneas vacías consecutivas a una sola.
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    cleaned: list[str] = []
    prev_empty = False
    for line in lines:
        empty = line == ""
        if empty and prev_empty:
            continue
        cleaned.append(line)
        prev_empty = empty

    return "\n".join(cleaned).strip()


# ─────────────────────────────────────────────
# Extracción PyMuPDF
# ─────────────────────────────────────────────

def extract_text_with_pymupdf(pdf_path: Path) -> dict:
    """
    Extrae el texto de un PDF con PyMuPDF.

    Devuelve un dict con:
      status         : 'OK' | 'PARCIAL' | 'ERROR'
      text           : texto completo (con marcadores de página)
      pages          : total de páginas
      pages_with_text: páginas con texto
      characters     : total de caracteres
      words          : total de palabras
      error          : mensaje de error si hubo alguno
    """
    result = {
        "status": "OK",
        "text": "",
        "pages": 0,
        "pages_with_text": 0,
        "empty_pages": 0,
        "characters": 0,
        "words": 0,
        "method": "PyMuPDF",
        "error": None,
    }

    try:
        doc = fitz.open(str(pdf_path))
        result["pages"] = len(doc)
        page_texts: list[str] = []

        for page_num, page in enumerate(doc, start=1):
            raw = page.get_text("text")
            cleaned = clean_text(raw)
            if cleaned:
                result["pages_with_text"] += 1
                page_texts.append(f"{'=' * 40} PÁGINA {page_num} {'=' * 40}\n{cleaned}")
            else:
                result["empty_pages"] += 1

        doc.close()
        result["text"] = "\n\n".join(page_texts)
        result["characters"] = len(result["text"])
        result["words"] = len(result["text"].split())

        if result["pages_with_text"] == 0:
            result["status"] = "SIN_TEXTO"
        elif result["empty_pages"] > result["pages"] * 0.3:
            result["status"] = "PARCIAL"

    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)

    return result


# ─────────────────────────────────────────────
# Procesamiento del corpus completo
# ─────────────────────────────────────────────

def process_corpus(
    df: pd.DataFrame,
    raw_dir: Path,
    processed_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """
    Itera sobre el corpus CSV, extrae el texto de cada PDF y guarda .txt en processed_dir.

    Parámetros
    ----------
    df            : DataFrame con el CSV del corpus (columnas: id_documento, archivo_pdf).
    raw_dir       : Directorio con los PDFs originales.
    processed_dir : Directorio de destino para los .txt generados.
    force         : Si True, re-extrae aunque ya exista el .txt.

    Retorna
    -------
    DataFrame con columnas de resultado por documento.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for _, row in df.iterrows():
        id_doc = str(row["id_documento"])
        archivo_pdf = str(row["archivo_pdf"])
        pdf_path = raw_dir / archivo_pdf
        txt_filename = f"{id_doc}.txt"
        txt_path = processed_dir / txt_filename

        record: dict = {
            "id_documento": id_doc,
            "archivo_pdf": archivo_pdf,
            "archivo_txt": txt_filename,
        }

        if not pdf_path.exists():
            record.update(
                status="PDF_NO_ENCONTRADO",
                pages=0,
                pages_with_text=0,
                empty_pages=0,
                characters=0,
                words=0,
                method="PyMuPDF",
                error=f"No existe: {pdf_path}",
            )
            records.append(record)
            continue

        if txt_path.exists() and not force:
            existing = txt_path.read_text(encoding="utf-8")
            record.update(
                status="YA_EXISTE",
                pages=0,
                pages_with_text=0,
                empty_pages=0,
                characters=len(existing),
                words=len(existing.split()),
                method="PyMuPDF",
                error=None,
            )
            records.append(record)
            continue

        extraction = extract_text_with_pymupdf(pdf_path)
        if extraction["status"] in ("OK", "PARCIAL") and extraction["text"]:
            txt_path.write_text(extraction["text"], encoding="utf-8")

        record.update(
            status=extraction["status"],
            pages=extraction["pages"],
            pages_with_text=extraction["pages_with_text"],
            empty_pages=extraction["empty_pages"],
            characters=extraction["characters"],
            words=extraction["words"],
            method=extraction["method"],
            error=extraction["error"],
        )
        records.append(record)

    return pd.DataFrame(records)
