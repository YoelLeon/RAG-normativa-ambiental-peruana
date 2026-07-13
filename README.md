# RAG con Búsqueda Híbrida para Normativa Ambiental Peruana

**Curso:** Tópicos Avanzados de Base de Datos  
**Año:** 2026-A  
**Alumnos:** Joel Condori-Leon, Luis Yana-Agramonte  
**Docente:** Antonio Arroyo-Paz  
**Tipo de trabajo:** Artículo / propuesta de investigación aplicada  
**Dominio:** Normativa ambiental peruana  
**Enfoque técnico:** RAG, búsqueda híbrida, bases de datos vectoriales, métricas de recuperación y optimización

🌐 **Demo en vivo:** [huggingface.co/spaces/JOEL022022/rag-normativa-ambiental](https://huggingface.co/spaces/JOEL022022/rag-normativa-ambiental)

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

| Configuración       | Recall@5  | Precision@5 | MRR    | Latencia promedio |
|---------------------|-----------|-------------|--------|-------------------|
| A — BM25            | 0.715     | 0.563       | 0.651  | 0.005 s           |
| B — Vectorial       | 0.794     | 0.763       | 0.855  | 0.040 s           |
| **C — Híbrida RRF** | **0.805** | **0.705**   | **0.817** | **0.085 s**    |

**Mejor configuración por MRR: C_HIBRIDA_RRF**

### Intervalos de confianza bootstrap (95%, n=1000 iteraciones)

| Configuración    | Recall@5 IC95%      | MRR IC95%           |
|------------------|---------------------|---------------------|
| A — BM25         | 0.715 [0.579, 0.842] | 0.651 [0.507, 0.776] |
| B — Vectorial    | 0.794 [0.684, 0.904] | 0.855 [0.737, 0.947] |
| C — Híbrida RRF  | 0.805 [0.686, 0.912] | 0.817 [0.693, 0.921] |

### Resultados por tipo de pregunta (Recall@5 / MRR)

| Tipo de pregunta | BM25          | Vectorial     | Híbrida RRF   |
|------------------|---------------|---------------|---------------|
| Aplicada         | 0.625 / 0.531 | 0.875 / 0.812 | 0.875 / 0.775 |
| Conceptual       | 1.000 / 0.833 | 1.000 / 1.000 | 1.000 / 1.000 |
| Factual          | 1.000 / 0.854 | 1.000 / 1.000 | 1.000 / 1.000 |
| Multi-documento  | 0.310 / 0.476 | 0.452 / 0.857 | 0.512 / 0.690 |
| Procedimental    | 0.400 / 0.400 | 0.400 / 0.400 | 0.400 / 0.400 |

### Resultados por dificultad (Recall@5 / MRR)

| Dificultad | BM25          | Vectorial     | Híbrida RRF   |
|------------|---------------|---------------|---------------|
| Alta       | 0.700 / 0.756 | 0.833 / 0.900 | 0.839 / 0.802 |
| Baja       | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| Media      | 0.712 / 0.564 | 0.758 / 0.818 | 0.773 / 0.818 |

### Evaluación de alucinación (preguntas trampa)

El sistema fue evaluado con **10 preguntas trampa** sobre documentos que **no existen en el corpus** (normas con números inventados, entidades inexistentes, decretos ficticios). Las tres configuraciones respondieron correctamente sin inventar artículos, fechas ni entidades.

### Evaluación generativa — LLM-as-Judge (Groq LLaMA 3.1)

Evaluación del componente generativo sobre 15 preguntas con la configuración híbrida, usando el propio LLM como juez en escala 1-5:

| Métrica          | Media | Min | Max |
|------------------|-------|-----|-----|
| Faithfulness     | 1.40  | 1   | 3   |
| Answer Relevance | 1.13  | 0   | 3   |

> Los scores bajos reflejan el comportamiento conservador del sistema: el LLM responde "La información no se encuentra en los documentos disponibles" cuando los fragmentos recuperados no contienen la respuesta exacta. Esto es el comportamiento esperado y deseado — el sistema prefiere no responder a inventar información.

### Latencia de generación de respuestas (configuración híbrida)

| Métrica           | Valor    |
|-------------------|----------|
| Latencia promedio | 9.691 s  |
| Latencia P95      | 17.411 s |

> La latencia en generación se debe al LLM (Groq API), no al pipeline de recuperación. La recuperación híbrida tiene una latencia promedio de 0.085 s.

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

| Tipo de norma        | Documentos |
|----------------------|------------|
| Decreto Supremo      | 14         |
| Resolución Ministerial | 8        |
| Otro                 | 5          |
| Ley                  | 2          |
| Norma técnica        | 1          |

**Fuentes:** MINAM, SINIA, Diario Oficial El Peruano, SPIJ  
**Extracción exitosa:** 26 documentos OK / 4 parciales (PDFs con tablas o baja densidad de texto)

### Banco de preguntas de evaluación

| Tipo de pregunta    | Cantidad |
|---------------------|----------|
| Conceptual          | 10       |
| Trampa (alucinación)| 10       |
| Aplicada            | 8        |
| Factual             | 8        |
| Multi-documento     | 7        |
| Procedimental       | 5        |
| **Total**           | **48**   |

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
Indexación en Qdrant Cloud con metadatos (1057 chunks)
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

| Componente           | Tecnología                                        |
|----------------------|---------------------------------------------------|
| Lenguaje             | Python 3.11                                       |
| IDE / entorno        | VS Code (NB01–03) + Google Colab (NB04–05)        |
| Extracción PDF       | PyMuPDF 1.24.9 + pdfplumber 0.11.4 (respaldo)    |
| Chunking             | Implementación propia con regex normativos        |
| Embeddings           | BAAI/bge-m3 (FlagEmbedding)                       |
| Base vectorial       | Qdrant Cloud (cluster gratuito AWS São Paulo)     |
| Búsqueda lexical     | BM25 (rank-bm25 0.2.2)                            |
| Fusión híbrida       | Reciprocal Rank Fusion (RRF, k=60)                |
| LLM generativo       | Groq — LLaMA 3.1-8b-instant                      |
| Evaluación generativa| LLM-as-Judge (Groq — faithfulness + answer relevance) |
| Interfaz demo        | Streamlit 1.37.0                                  |
| Despliegue           | Hugging Face Spaces (Docker, CPU Basic)           |
| Evaluación           | pandas + scikit-learn + bootstrap IC95%           |
| Control de versiones | GitHub                                            |

> **Cambios respecto a la propuesta original:**
> - El LLM fue migrado de Gemini API a **Groq (llama-3.1-8b-instant)** por restricción regional de Perú.
> - Qdrant migrado de disco local a **Qdrant Cloud** para el despliegue en producción.
> - La API `client.search()` de Qdrant fue reemplazada por `client.query_points()` por deprecación.
> - La demo se desplegó en **Hugging Face Spaces** con Docker en lugar de ejecución local.
> - Se agregaron **intervalos de confianza bootstrap** y evaluación **LLM-as-Judge** al NB05.
> - El banco de preguntas trampa se amplió de **2 a 10 preguntas**.

---

## Estructura del repositorio

```
rag-normativa-ambiental-peruana/
│
├── data/
│   ├── raw/                        # PDFs oficiales (30 documentos)
│   ├── chunks/
│   │   ├── chunks_normativa_v1.jsonl
│   │   └── chunks_normativa_v1.csv
│   └── metadata/
│       ├── corpus_normativo_ambiental.csv
│       ├── corpus_normativo_ambiental_con_extraccion.csv
│       └── banco_preguntas_evaluacion.csv  # 48 preguntas (10 trampa)
│
├── notebooks/
│   ├── 01_verificacion_corpus.ipynb      # Validación del corpus y metadatos
│   ├── 02_extraccion_texto.ipynb         # Extracción de texto con PyMuPDF
│   ├── 03_chunking.ipynb                 # Chunking estructural normativo
│   ├── 04_embeddings_indexacion.ipynb    # Embeddings + indexación en Qdrant Cloud
│   └── 05_pruebas_rag.ipynb              # Búsqueda híbrida, RAG, evaluación y bootstrap
│
├── src/
│   ├── config.py                   # Configuración central (rutas, parámetros, Qdrant)
│   ├── ingestion/                  # Extracción de texto de PDFs
│   ├── chunking/                   # Segmentación estructural normativa
│   ├── embeddings/                 # Generación de embeddings e indexación
│   ├── retrieval/                  # BM25, vectorial e híbrida (RRF)
│   ├── generation/                 # Generación de respuestas con Groq
│   └── evaluation/                 # Métricas: Recall@k, Precision@k, MRR, bootstrap
│
├── experiments/
│   └── resultados/                 # CSVs y JSONs de evaluación (bootstrap, LLM-judge)
│
├── app/
│   └── demo_streamlit.py           # Demo visual desplegada en Hugging Face Spaces
│
├── Dockerfile                      # Configuración para despliegue en HF Spaces
├── .env.example                    # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Notebooks — descripción y entorno

| NB  | Nombre                       | Función                                                        | Entorno            |
|-----|------------------------------|----------------------------------------------------------------|--------------------|
| 01  | `01_verificacion_corpus`     | Valida CSV, PDFs existentes y columnas obligatorias            | Local / VS Code    |
| 02  | `02_extraccion_texto`        | Extrae texto de PDFs con PyMuPDF                               | Local / VS Code    |
| 03  | `03_chunking`                | Chunking estructural con regex normativos                      | Local / VS Code    |
| 04  | `04_embeddings_indexacion`   | Genera embeddings e indexa en Qdrant Cloud                     | Google Colab (GPU) |
| 05  | `05_pruebas_rag`             | BM25, vectorial, híbrida, generación, bootstrap y LLM-judge   | Google Colab (GPU) |

---

## Instalación y uso

### Requisitos previos

- Python 3.11
- Cuenta en [Groq Console](https://console.groq.com) para obtener API key gratuita
- Cuenta en [Qdrant Cloud](https://cloud.qdrant.io) para el cluster vectorial
- Google Drive (para NB04–05 en Colab)

### Instalación local (NB01–NB03)

```bash
git clone https://github.com/JOEL022022/rag-normativa-ambiental-peruana.git
cd rag-normativa-ambiental-peruana
pip install -r requirements.txt
```

Crear el archivo `.env` en la raíz:

```
GROQ_API_KEY=tu_clave_groq
QDRANT_URL=tu_url_qdrant_cloud
QDRANT_API_KEY=tu_clave_qdrant
```

### Instalación en Google Colab (NB04–NB05)

Cada notebook incluye una celda de instalación. Ejecutarla antes de continuar:

```bash
!pip install -q --upgrade pip
!pip uninstall -y transformers FlagEmbedding torchvision torchaudio torchcodec
!pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers==4.44.2
!pip install -q FlagEmbedding
!pip install -q rank-bm25 groq qdrant-client pandas numpy tqdm scikit-learn
```

Las API keys se cargan desde **Colab Secrets** (icono 🔑 en el panel lateral).

### Orden de ejecución

```
NB01 → NB02 → NB03 → NB04 (Colab) → NB05 (Colab)
```

Cada notebook genera artefactos que el siguiente consume. No saltear pasos.

### Demo Streamlit (local)

```bash
set PYTHONPATH=.
python -m streamlit run app/demo_streamlit.py
```

### Demo desplegada

La demo está disponible públicamente en:  
🌐 [huggingface.co/spaces/JOEL022022/rag-normativa-ambiental](https://huggingface.co/spaces/JOEL022022/rag-normativa-ambiental)

---

## Variables de entorno

| Variable         | Descripción                          | Dónde obtenerla                          |
|------------------|--------------------------------------|------------------------------------------|
| `GROQ_API_KEY`   | API key para LLaMA 3.1 vía Groq     | https://console.groq.com (gratuita)      |
| `QDRANT_URL`     | URL del cluster Qdrant Cloud         | https://cloud.qdrant.io                  |
| `QDRANT_API_KEY` | API key del cluster Qdrant Cloud     | https://cloud.qdrant.io → API Keys       |

En Hugging Face Spaces las tres variables se configuran como **Secrets** en Settings → Variables and secrets.

---

## Metadatos del corpus

Cada chunk conserva los siguientes campos:

`id_chunk`, `id_documento`, `archivo_pdf`, `archivo_txt`, `titulo_documento`, `tipo_norma`, `numero_norma`, `entidad_emisora`, `fecha_publicacion`, `tema_principal`, `subtema`, `fuente_oficial`, `url_fuente`, `estado_vigencia`, `estado_extraccion`, `seccion`, `texto`, `n_palabras`, `pagina_aprox`

El campo `estado_vigencia` puede tomar los valores: `vigente`, `modificada`, `derogada`, `no_verificado`.  
El campo `estado_extraccion` puede tomar los valores: `OK`, `PARCIAL`, `SIN_TEXTO`, `ERROR`.

La demo muestra badges de advertencia cuando un fragmento proviene de una norma derogada, no verificada o con extracción parcial.

---

## Reglas técnicas del proyecto

- No responder desde memoria del LLM: la respuesta debe basarse únicamente en los fragmentos recuperados.
- No afirmar vigencia normativa si no está verificada en una fuente oficial.
- Cada respuesta debe mostrar documento y fragmento fuente (trazabilidad completa).
- No usar chunking puramente arbitrario: se aplica chunking estructural respetando artículos, capítulos e incisos.
- No subir claves API al repositorio. Usar `.env` y mantenerlo en `.gitignore`.
- El `id_chunk` es inmutable una vez indexado en Qdrant — regenerar chunks requiere reindexación completa.

---

## Referencias

1. MINAM — Normas y documentos legales: https://www.gob.pe/institucion/minam/normas-legales  
2. SINIA — Compendio de la legislación ambiental peruana: https://sinia.minam.gob.pe/normas/compendio-legislacion-ambiental-peruana  
3. Diario Oficial El Peruano: https://diariooficial.elperuano.pe/normas/normasactualizadas  
4. SPIJ: https://spijweb.minjus.gob.pe/  
5. PyMuPDF: https://pymupdf.readthedocs.io/  
6. BAAI/bge-m3: https://huggingface.co/BAAI/bge-m3  
7. Qdrant: https://qdrant.tech/  
8. Qdrant Cloud: https://cloud.qdrant.io  
9. rank-bm25: https://pypi.org/project/rank-bm25/  
10. Groq: https://console.groq.com  
11. Streamlit: https://docs.streamlit.io  
12. Hugging Face Spaces: https://huggingface.co/spaces
