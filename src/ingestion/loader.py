"""
Funciones de carga del corpus CSV y de los textos procesados.
"""

from pathlib import Path

import pandas as pd


def load_corpus_csv(csv_path: Path) -> pd.DataFrame:
    """
    Carga el CSV del corpus normativo y valida las columnas obligatorias.

    Lanza ValueError si faltan columnas obligatorias.
    """
    required_columns = [
        "id_documento",
        "archivo_pdf",
        "titulo_documento",
        "tipo_norma",
        "numero_norma",
        "entidad_emisora",
        "fecha_publicacion",
        "tema_principal",
        "subtema",
        "fuente_oficial",
        "url_fuente",
        "estado_vigencia",
    ]

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV no encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en el CSV: {missing}")

    return df


def load_processed_texts(
    df: pd.DataFrame,
    processed_dir: Path,
    skip_missing: bool = True,
) -> dict[str, str]:
    """
    Carga los textos extraídos (.txt) de processed_dir para cada documento del corpus.

    Parámetros
    ----------
    df            : DataFrame del corpus (necesita columna id_documento).
    processed_dir : Directorio con los .txt generados por el extractor.
    skip_missing  : Si True, omite documentos sin .txt; si False, lanza FileNotFoundError.

    Retorna
    -------
    Dict {id_documento: texto}.
    """
    texts: dict[str, str] = {}

    for _, row in df.iterrows():
        id_doc = str(row["id_documento"])
        # Preferir archivo_txt si existe la columna
        if "archivo_txt" in row and pd.notna(row["archivo_txt"]):
            txt_name = str(row["archivo_txt"])
        else:
            txt_name = f"{id_doc}.txt"

        txt_path = processed_dir / txt_name
        if not txt_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Texto no encontrado: {txt_path}")

        texts[id_doc] = txt_path.read_text(encoding="utf-8")

    return texts
