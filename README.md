# RAG para Normativa Ambiental Peruana

Proyecto de investigaciÃ³n para el curso de TÃ³picos Avanzados de Base de Datos.

## TÃ­tulo

OptimizaciÃ³n y evaluaciÃ³n de una arquitectura RAG con bÃºsqueda hÃ­brida y bases de datos vectoriales para la consulta semÃ¡ntica de normativa ambiental peruana.

## Objetivo

DiseÃ±ar, implementar y evaluar un prototipo RAG capaz de recuperar fragmentos relevantes de normativa ambiental peruana y generar respuestas sustentadas en documentos oficiales.

## Flujo general

PDFs oficiales â†’ extracciÃ³n de texto â†’ chunking estructural â†’ embeddings â†’ base vectorial â†’ bÃºsqueda hÃ­brida â†’ generaciÃ³n de respuesta â†’ evaluaciÃ³n.

## Corpus

El corpus inicial estarÃ¡ compuesto por documentos oficiales de normativa ambiental peruana registrados en:

data/metadata/corpus_normativo_ambiental.csv

## Estructura del proyecto

- data/raw: PDFs originales o renombrados.
- data/processed: textos extraÃ­dos.
- data/chunks: fragmentos generados.
- data/metadata: CSV del corpus.
- notebooks: pruebas y experimentos.
- src: cÃ³digo modular del sistema.
- app: demo del sistema.
- experiments: resultados de evaluaciÃ³n.
- docs: documentos del proyecto.

## Estado actual

- [x] Tema definido
- [x] Estado del arte organizado
- [x] Corpus v1 construido
- [ ] ExtracciÃ³n de texto
- [ ] Chunking estructural
- [ ] Embeddings
- [ ] Base vectorial
- [ ] BÃºsqueda hÃ­brida
- [ ] GeneraciÃ³n RAG
- [ ] EvaluaciÃ³n
