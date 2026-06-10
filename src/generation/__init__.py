"""Módulo de generación: respuestas con LLM Groq (LLaMA 3.1-8b-instant)."""

from .generator import RAGGenerator, build_context

__all__ = ["RAGGenerator", "build_context"]
