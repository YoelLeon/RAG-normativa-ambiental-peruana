"""
Expresiones regulares para detectar la estructura de documentos normativos peruanos.

Soporta formas como:
  TÍTULO I / TÍTULO PRELIMINAR
  CAPÍTULO II
  Artículo 1º.- / Artículo 1.-
  DISPOSICIONES COMPLEMENTARIAS FINALES
  ANEXO Nº 01
  DECRETO SUPREMO / RESOLUCIÓN MINISTERIAL / LEY
"""

import re

# Marcador de página insertado durante la extracción
PAGE_MARKER_RE = re.compile(
    r"^=+\s*PÁGINA\s+(\d+)\s*=+$",
    re.IGNORECASE,
)

# Título (TÍTULO I, TÍTULO PRELIMINAR, etc.)
TITLE_RE = re.compile(
    r"^(T[ÍI]TULO\s+(?:PRELIMINAR|[IVXLCDM]+|\d+).*)$",
    re.IGNORECASE,
)

# Capítulo
CHAPTER_RE = re.compile(
    r"^(CAP[ÍI]TULO\s+(?:[IVXLCDM]+|\d+).*)$",
    re.IGNORECASE,
)

# Artículo (acepta ordinal con º o sin él, guiones y rayas)
ARTICLE_RE = re.compile(
    r"^(Art[íi]culo|Articulo)\s+\d+[A-Za-z0-9]*[°º]?\s*[.\-–—º]*\s*.*$",
    re.IGNORECASE,
)

# Anexo
ANNEX_RE = re.compile(
    r"^(ANEXO\s*(?:N[°º]\s*)?(?:\d+|[IVXLCDM]+)?.*)$",
    re.IGNORECASE,
)

# Disposiciones complementarias / finales / transitorias / modificatorias
DISPOSITION_RE = re.compile(
    r"^((DISPOSICI[ÓO]N|DISPOSICIONES)\s+"
    r"(COMPLEMENTARIAS|FINALES|TRANSITORIAS|MODIFICATORIAS).*)$",
    re.IGNORECASE,
)

# Encabezado legal (tipo de norma)
LEGAL_HEADER_RE = re.compile(
    r"^(DECRETO SUPREMO|RESOLUCI[ÓO]N MINISTERIAL|LEY"
    r"|DECRETO LEGISLATIVO|RESOLUCI[ÓO]N DIRECTORAL).*$",
    re.IGNORECASE,
)

# Todos los patrones de sección en orden de jerarquía
_SECTION_PATTERNS = [
    LEGAL_HEADER_RE,
    TITLE_RE,
    CHAPTER_RE,
    DISPOSITION_RE,
    ANNEX_RE,
    ARTICLE_RE,
]


def is_normative_header(line: str) -> bool:
    """Devuelve True si la línea corresponde a un encabezado normativo reconocido."""
    stripped = line.strip()
    return any(p.match(stripped) for p in _SECTION_PATTERNS)
