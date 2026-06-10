"""
Generación de respuestas RAG con Groq (LLaMA 3.1-8b-instant).

Reglas estrictas del sistema:
  - Solo responder con los fragmentos recuperados.
  - No inventar artículos, fechas, números de norma ni entidades.
  - Citar siempre el documento fuente.
  - Si la información no está en el corpus: declararlo explícitamente.
"""

from __future__ import annotations

import os
import time
from typing import Optional

GROQ_MODEL = "llama-3.1-8b-instant"
MAX_TOKENS = 1024

PROMPT_TEMPLATE = """\
Eres un asistente especializado en normativa ambiental peruana.
Responde la pregunta usando ÚNICAMENTE la información de los fragmentos proporcionados.

REGLAS ESTRICTAS:
- Si la respuesta no está en los fragmentos, responde exactamente:
  "La información no se encuentra en los documentos disponibles."
- No inventes artículos, fechas, números de norma ni entidades.
- Cita siempre el documento fuente (id_documento y título).
- Sé claro, preciso y estructurado.

FRAGMENTOS RECUPERADOS:
{contexto}

PREGUNTA:
{pregunta}

RESPUESTA:"""


def build_context(fragmentos: list[dict], max_chars_per_fragment: int = 800) -> str:
    """
    Construye el bloque de contexto a partir de los fragmentos recuperados.

    Trunca el texto de cada fragmento a max_chars_per_fragment para evitar
    superar el contexto del LLM.
    """
    partes: list[str] = []
    for i, f in enumerate(fragmentos, 1):
        texto = f.get("texto", "")
        if len(texto) > max_chars_per_fragment:
            texto = texto[:max_chars_per_fragment] + " [...]"
        partes.append(
            f"[Fragmento {i}]\n"
            f"Documento: {f.get('id_documento', '')} — {f.get('titulo_documento', '')}\n"
            f"Sección: {f.get('seccion', 'N/A')}\n"
            f"Texto: {texto}"
        )
    return "\n\n".join(partes)


class RAGGenerator:
    """
    Generador de respuestas RAG usando Groq como proveedor LLM.

    Uso
    ---
    gen = RAGGenerator(api_key="gsk_...")
    result = gen.generate("¿Qué son los ECA del aire?", fragmentos)
    print(result["respuesta"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        max_tokens: int = MAX_TOKENS,
    ) -> None:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq no instalado: pip install groq")

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "API key de Groq no proporcionada. "
                "Pásala como argumento o define la variable de entorno GROQ_API_KEY."
            )

        self._client = Groq(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def generate(
        self,
        pregunta: str,
        fragmentos: list[dict],
        max_chars_per_fragment: int = 800,
    ) -> dict:
        """
        Genera una respuesta fundamentada en los fragmentos recuperados.

        Parámetros
        ----------
        pregunta              : Pregunta del usuario.
        fragmentos            : Lista de dicts con texto y metadatos (salida del retriever).
        max_chars_per_fragment: Truncado de texto por fragmento para controlar el contexto.

        Retorna
        -------
        Dict con:
          respuesta     : Texto generado por el LLM.
          tokens_usados : Total de tokens consumidos.
          latencia_seg  : Tiempo de respuesta en segundos.
          modelo        : Nombre del modelo usado.
        """
        if not fragmentos:
            return {
                "respuesta": "No se encontraron fragmentos relevantes para esta consulta.",
                "tokens_usados": 0,
                "latencia_seg": 0.0,
                "modelo": self.model,
            }

        contexto = build_context(fragmentos, max_chars_per_fragment)
        prompt = PROMPT_TEMPLATE.format(contexto=contexto, pregunta=pregunta)

        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
            )
            latencia = time.time() - t0
            respuesta_texto = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0
        except Exception as exc:
            latencia = time.time() - t0
            respuesta_texto = f"Error al generar respuesta: {exc}"
            tokens = 0

        return {
            "respuesta": respuesta_texto,
            "tokens_usados": tokens,
            "latencia_seg": round(latencia, 3),
            "modelo": self.model,
        }
