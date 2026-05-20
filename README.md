# RAG para la Consulta de Normativa Ambiental Peruana

Proyecto de investigación desarrollado para el curso de **Tópicos Avanzados de Base de Datos**.

## Título del proyecto

**Optimización y evaluación de una arquitectura RAG con búsqueda híbrida y bases de datos vectoriales para la consulta semántica de normativa ambiental peruana**

## Descripción general

Este proyecto propone el diseño, implementación y evaluación de un prototipo basado en **Generación Aumentada por Recuperación** (*Retrieval-Augmented Generation*, RAG) para la consulta semántica de normativa ambiental peruana.

La solución busca procesar documentos oficiales relacionados con normas ambientales, extraer su contenido, segmentarlo mediante **chunking estructural**, generar representaciones vectoriales mediante **embeddings** y almacenarlas en una **base de datos vectorial**. Posteriormente, el sistema permitirá realizar consultas en lenguaje natural y recuperar fragmentos relevantes mediante una estrategia de **búsqueda híbrida**, combinando recuperación semántica, búsqueda lexical y metadatos documentales.

A diferencia de un chatbot genérico, el enfoque principal del proyecto no se limita a generar respuestas en lenguaje natural. El objetivo central es evaluar cómo diferentes estrategias de recuperación pueden mejorar la precisión, trazabilidad y confiabilidad de las respuestas generadas a partir de normativa ambiental peruana.

## Problema de investigación

La normativa ambiental peruana se encuentra distribuida en múltiples documentos, fuentes oficiales y formatos, lo que dificulta la búsqueda rápida de información específica, vigente y sustentada. Esta situación puede afectar la consulta de normas relacionadas con gestión ambiental, calidad ambiental, evaluación ambiental, residuos sólidos, fiscalización, estándares de calidad ambiental y límites máximos permisibles.

Frente a este problema, el proyecto plantea una arquitectura RAG capaz de recuperar fragmentos normativos relevantes y generar respuestas basadas en documentos oficiales, reduciendo el riesgo de respuestas no sustentadas o alucinadas.

## Objetivo general

Diseñar, implementar y evaluar una arquitectura RAG con búsqueda híbrida y bases de datos vectoriales para mejorar la recuperación y consulta semántica de normativa ambiental peruana.

## Objetivos específicos

- Construir un corpus inicial de documentos oficiales de normativa ambiental peruana.
- Extraer y limpiar el texto contenido en los documentos seleccionados.
- Aplicar chunking estructural respetando la organización normativa de los documentos.
- Generar embeddings de los fragmentos normativos.
- Almacenar los embeddings y metadatos en una base de datos vectorial.
- Implementar mecanismos de búsqueda semántica, lexical e híbrida.
- Generar respuestas sustentadas en los fragmentos recuperados.
- Evaluar el desempeño del sistema mediante métricas de recuperación, calidad de respuesta y rendimiento.

## Flujo general de la arquitectura

```text
Documentos oficiales
        ↓
Extracción y limpieza de texto
        ↓
Chunking estructural
        ↓
Generación de embeddings
        ↓
Base de datos vectorial
        ↓
Búsqueda híbrida
        ↓
Recuperación de fragmentos relevantes
        ↓
Generación de respuesta
        ↓
Respuesta sustentada en documentos oficiales
