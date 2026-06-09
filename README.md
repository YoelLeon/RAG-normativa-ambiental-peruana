# RAG con Búsqueda Híbrida para Normativa Ambiental Peruana

**Curso:** Tópicos Avanzados de Base de Datos  
**Tipo de trabajo:** Artículo / propuesta de investigación aplicada  
**Dominio:** Normativa ambiental peruana  
**Enfoque técnico:** RAG, búsqueda híbrida, bases de datos vectoriales, métricas de recuperación y optimización

---

## Descripción del proyecto

Este proyecto diseña, implementa y evalúa una arquitectura de **Generación Aumentada por Recuperación (RAG)** orientada a la consulta semántica de normativa ambiental peruana. El aporte central no es la interfaz conversacional, sino la comparación experimental entre tres estrategias de recuperación sobre un corpus de documentos oficiales:

- **Configuración A:** Búsqueda lexical con BM25
- **Configuración B:** Búsqueda vectorial semántica (Qdrant + BAAI/bge-m3)
- **Configuración C:** Búsqueda híbrida (BM25 + Vectorial + RRF)

Las respuestas se generan con **Groq (LLaMA 3.1-8b-instant)** y se restringen estrictamente a los fragmentos recuperados del corpus indexado.

---

## Resultados experimentales

Evaluación realizada sobre **38 preguntas** de un banco de consultas con documentos de referencia asignados manualmente (top-k = 5).

### Tabla comparativa de configuraciones

| Configuración    | Recall@5 | Precision@5 | MRR    | Latencia promedio |
|------------------|----------|-------------|--------|-------------------|
| A — BM25         | 0.715    | 0.563       | 0.651  | 0.058 s           |
| B — Vectorial    | 0.779    | 0.721       | 0.803  | 0.834 s           |
| **C — Híbrida RRF** | **0.792** | **0.674** | **0.809** | **0.763 s**   |

**Mejor configuración por MRR: C_HIBRIDA_RRF**

### Resultados por tipo de pregunta (Recall@5 / MRR)

| Tipo de pregunta   | BM25        | Vectorial   | Híbrida RRF |
|--------------------|-------------|-------------|-------------|
| Aplicada           | 0.625 / 0.531 | 0.875 / 0.875 | 0.875 / 0.812 |
| Conceptual         | 1.000 / 0.833 | 1.000 / 1.000 | 1.000 / 1.000 |
| Factual            | 1.000 / 0.854 | 1.000 / 0.938 | 1.000 / 1.000 |
| Multi-documento    | 0.310 / 0.476 | 0.369 / 0.571 | 0.440 / 0.607 |
| Procedimental      | 0.400 / 0.400 | 0.400 / 0.400 | 0.400 / 0.400 |

### Resultados por dificultad (Recall@5 / MRR)

| Dificultad | BM25        | Vectorial   | Híbrida RRF |
|------------|-------------|-------------|-------------|
| Alta       | 0.700 / 0.756 | 0.817 / 0.867 | 0.806 / 0.817 |
| Baja       | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Media      | 0.712 / 0.564 | 0.742 / 0.750 | 0.773 / 0.795 |

### Evaluación de alucinación (preguntas trampa)

El sistema fue evaluado con 2 preguntas trampa sobre documentos que **no existen en el corpus**. Las tres configuraciones respondieron correctamente con "La información no se encuentra en los documentos disponibles", sin inventar artículos, fechas ni entidades.

### Latencia de generación de respuestas (configuración híbrida)

| Métrica         | Valor    |
|-----------------|----------|
| Latencia promedio | 7.944 s |
| Latencia P95    | 15.232 s |

> La latencia alta en generación se debe al LLM (Groq API), no al pipeline de recuperación. La recuperación híbrida tiene una latencia promedio de 0.763 s.

---

## Corpus documental

El corpus fue construido con **30 documentos oficiales públicos** de normativa ambiental peruana.

| Tema principal          | Documentos |
|-------------------------|------------|
| Calidad ambiental       | 17         |
| Gestión ambiental       | 6          |
| Evaluación ambiental    | 3          |
| Residuos sólidos        | 2          |
| Biodiversidad           | 1          |
| Fiscalización ambiental | 1          |

| Tipo de norma           | Documentos |
|-------------------------|------------|
| Decreto Supremo         | 14         |
| Resolución Ministerial  | 8          |
| Otro                    | 5          |
| Ley                     | 2          |
| Norma técnica           | 1          |

**Fuentes:** MINAM, SINIA, Diario Oficial El Peruano, SPIJ  
**Extracción exitosa:** 26 documentos OK / 4 parciales (PDFs con tablas o baja densidad de texto)

---

## Arquitectura del sistema

```
PDFs oficiales + corpus CSV
        ↓
Extracción de texto con PyMuPDF
        ↓
Limpieza básica y normalización
        ↓
Chunking estructural (artículos, capítulos, incisos)
        ↓
Embeddings con BAAI/bge-m3  →  vectores de 1024 dimensiones
        ↓
Indexación en Qdrant con metadatos (34 s para 1057 chunks)
        ↓
Búsqueda vectorial + BM25
        ↓
Fusión híbrida con RRF  (k=60)
        ↓
Top-k = 5 fragmentos relevantes
        ↓
Generación con Groq (LLaMA 3.1-8b-instant)
        ↓
Respuesta sustentada con fuentes
```

**Total de chunks generados:** 1057 (de 30 documentos)  
**Parámetros de chunking:** MAX_WORDS=450, OVERLAP_WORDS=80, MIN_WORDS=25  
**Dimensión de embeddings:** 1024 (similitud coseno)  
**Colección Qdrant:** `normativa_ambiental_chunks_v1`

---

## Stack tecnológico

| Componente          | Tecnología                          |
|---------------------|-------------------------------------|
| Lenguaje            | Python 3.11                         |
| IDE / entorno       | VS Code (NB01–03) + Google Colab (NB04–05) |
| Extracción PDF      | PyMuPDF 1.24.9 + pdfplumber 0.11.4 (respaldo) |
| Chunking            | Implementación propia con regex     |
| Embeddings          | BAAI/bge-m3 (FlagEmbedding 1.2.11) |
| Base vectorial      | Qdrant 1.10.1                       |
| Búsqueda lexical    | BM25 (rank-bm25 0.2.2)              |
| Fusión híbrida      | Reciprocal Rank Fusion (RRF)        |
| LLM generativo      | Groq — LLaMA 3.1-8b-instant        |
| Interfaz demo       | Streamlit 1.37.0                    |
| Evaluación          | pandas + scikit-learn               |
| Control de versiones| GitHub                              |

> **Cambio respecto a la propuesta original:** el LLM fue migrado de Gemini API a **Groq (llama-3.1-8b-instant)** por restricción regional de Perú. Los notebooks NB04 y NB05 se ejecutan en Google Colab por requerimiento de GPU para embeddings. La API `client.search()` de Qdrant fue reemplazada por `client.query_points()` por deprecación.

---

## Estructura del repositorio

```
rag-normativa-ambiental-peruana/
│
├── data/
│   ├── raw/                        # PDFs oficiales (30 documentos)
│   ├── processed/                  # Textos extraídos (.txt por documento)
│   ├── chunks/
│   │   ├── chunks_normativa_v1.jsonl
│   │   └── chunks_normativa_v1.csv
│   └── metadata/
│       ├── corpus_normativo_ambiental.csv
│       ├── corpus_normativo_ambiental_con_extraccion.csv
│       └── banco_preguntas_evaluacion.csv
│
├── notebooks/
│   ├── 01_verificacion_corpus.ipynb      # Validación del corpus y metadatos
│   ├── 02_extraccion_texto.ipynb         # Extracción de texto con PyMuPDF
│   ├── 03_chunking.ipynb                 # Chunking estructural normativo
│   ├── 04_embeddings_indexacion.ipynb    # Embeddings + indexación en Qdrant
│   └── 05_pruebas_rag.ipynb              # Búsqueda híbrida, RAG y evaluación
│
├── experiments/
│   └── resultados/                       # CSVs y JSONs de evaluación
│
├── app/
│   └── demo_streamlit.py                 # Demo visual del sistema
│
├── .env.example                          # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks — descripción y entorno

| NB  | Nombre                         | Función                                              | Entorno        |
|-----|--------------------------------|------------------------------------------------------|----------------|
| 01  | `01_verificacion_corpus`       | Valida CSV, PDFs existentes y columnas obligatorias  | Local / VS Code |
| 02  | `02_extraccion_texto`          | Extrae texto de PDFs con PyMuPDF                     | Local / VS Code |
| 03  | `03_chunking`                  | Chunking estructural con regex normativos            | Local / VS Code |
| 04  | `04_embeddings_indexacion`     | Genera embeddings e indexa en Qdrant                 | Google Colab (GPU) |
| 05  | `05_pruebas_rag`               | BM25, vectorial, híbrida, generación y evaluación    | Google Colab (GPU) |

---

## Instalación y uso

### Requisitos previos

- Python 3.11
- Docker (para Qdrant local en NB01–03)
- Cuenta en [Groq Console](https://console.groq.com) para obtener API key gratuita
- Google Drive (para NB04–05 en Colab)

### Instalación local (NB01–NB03)

```bash
git clone https://github.com/<tu-usuario>/rag-normativa-ambiental-peruana.git
cd rag-normativa-ambiental-peruana
pip install -r requirements.txt
```

Crear el archivo `.env` en la raíz:

```
GROQ_API_KEY=tu_clave_aqui
```

Levantar Qdrant con Docker:

```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Instalación en Google Colab (NB04–NB05)

Cada notebook incluye una celda de instalación. Ejecutarla antes de continuar:

```bash
!pip install -q --upgrade pip
!pip uninstall -y transformers FlagEmbedding torchvision torchaudio torchcodec
!pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers==4.44.2
!pip install -q FlagEmbedding==1.2.11
!pip install -q rank-bm25 groq qdrant-client pandas numpy tqdm scikit-learn
```

La API key de Groq se carga desde **Colab Secrets** (no escribirla directamente en el código).

### Orden de ejecución

```
NB01 → NB02 → NB03 → NB04 (Colab) → NB05 (Colab)
```

Cada notebook genera artefactos que el siguiente consume. No saltear pasos.

### Demo Streamlit

```bash
streamlit run app/demo_streamlit.py
```

---

## Variables de entorno

| Variable       | Descripción                        | Dónde obtenerla                     |
|----------------|------------------------------------|-------------------------------------|
| `GROQ_API_KEY` | API key para LLaMA 3.1 vía Groq   | https://console.groq.com (gratuita) |

---

## Metadatos del corpus

Cada chunk conserva los siguientes campos:

`id_chunk`, `id_documento`, `archivo_pdf`, `archivo_txt`, `titulo_documento`, `tipo_norma`, `numero_norma`, `entidad_emisora`, `fecha_publicacion`, `tema_principal`, `subtema`, `fuente_oficial`, `url_fuente`, `estado_vigencia`, `seccion`, `texto`, `n_palabras`, `pagina_aprox`

El campo `estado_vigencia` puede tomar los valores: `vigente`, `modificada`, `derogada`, `no_verificado`.

---

## Reglas técnicas del proyecto

- No responder desde memoria del LLM: la respuesta debe basarse únicamente en los fragmentos recuperados.
- No afirmar vigencia normativa si no está verificada en una fuente oficial.
- Cada respuesta debe mostrar documento y fragmento fuente (trazabilidad completa).
- No usar chunking puramente arbitrario: se aplica chunking estructural respetando artículos, capítulos e incisos.
- No subir claves API al repositorio. Usar `.env` y mantenerlo en `.gitignore`.

---

## Referencias

1. MINAM — Normas y documentos legales: https://www.gob.pe/institucion/minam/normas-legales  
2. SINIA — Compendio de la legislación ambiental peruana: https://sinia.minam.gob.pe/normas/compendio-legislacion-ambiental-peruana  
3. Diario Oficial El Peruano: https://diariooficial.elperuano.pe/normas/normasactualizadas  
4. SPIJ: https://spijweb.minjus.gob.pe/  
5. PyMuPDF: https://pymupdf.readthedocs.io/  
6. BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3  
7. Qdrant: https://qdrant.tech/  
8. rank-bm25: https://pypi.org/project/rank-bm25/  
9. Groq: https://console.groq.com  
10. Streamlit: https://docs.streamlit.io/
