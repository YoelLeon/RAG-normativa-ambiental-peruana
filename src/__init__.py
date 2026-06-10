"""
src — Módulos del sistema RAG para normativa ambiental peruana.

Submódulos:
  ingestion   : Extracción de texto de PDFs y carga del corpus.
  chunking    : Segmentación estructural de textos normativos.
  embeddings  : Generación de embeddings (BAAI/bge-m3) e indexación en Qdrant.
  retrieval   : Búsqueda BM25, vectorial e híbrida (RRF).
  generation  : Generación de respuestas con Groq (LLaMA 3.1-8b-instant).
  evaluation  : Métricas de recuperación (Recall@K, MRR, Precision@K).
"""
