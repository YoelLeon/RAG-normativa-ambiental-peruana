"""
Demo Streamlit — RAG para Normativa Ambiental Peruana
=====================================================
Ejecutar:
    streamlit run app/demo_streamlit.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# ── Raíz del proyecto en sys.path ──
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Importar config (esto también carga el .env si existe)
from src.config import (
    CHUNKS_JSONL, QDRANT_STORAGE, COLLECTION_NAME,
    TOP_K, settings
)

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG · Normativa Ambiental Peruana",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.app-header {
    background: linear-gradient(135deg, #0072BC 0%, #00AEEF 60%, #39B54A 100%);
    color: white; padding: 1.6rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
}
.app-header h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
.app-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 0.95rem; }

.fragment-card {
    background: #f0f8ff; border-left: 4px solid #00AEEF;
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 0.8rem;
    font-size: 0.88rem; color: #1a2a3a;
}
.fragment-card .frag-header { font-weight: 600; color: #0072BC; margin-bottom: 0.4rem; }
.fragment-card .frag-scores { margin-top: 0.5rem; font-size: 0.78rem; color: #4a7a9b; }

.answer-box {
    background: white; border: 1.5px solid #b3d9f5; border-radius: 10px;
    padding: 1.2rem 1.4rem; color: #1a2a3a; line-height: 1.65; font-size: 0.95rem;
}
.config-badge { display:inline-block; padding:.25rem .7rem; border-radius:20px;
    font-size:.78rem; font-weight:600; margin-right:.4rem; }
.badge-bm25      { background:#fff3cd; color:#856404; }
.badge-vectorial { background:#cce8f4; color:#0072BC; }
.badge-hibrida   { background:#d6f0da; color:#1a7a2a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🌿 Plataforma Ambiental · Consulta Normativa</h1>
    <p>Consulta semántica sobre 30 documentos oficiales — MINAM, SINIA, El Peruano, SPIJ</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Inicialización cacheada de recursos
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo de embeddings…")
def load_encoder():
    from src.embeddings.encoder import EmbeddingEncoder
    return EmbeddingEncoder()


@st.cache_resource(show_spinner="Conectando a Qdrant…")
def load_qdrant_client():
    from src.embeddings.indexer import get_qdrant_client
    return get_qdrant_client(mode="disk", storage_path=QDRANT_STORAGE)


@st.cache_resource(show_spinner="Cargando chunks e índice BM25…")
def load_retrievers():
    """
    Carga chunks, construye BM25 y conecta vectorial.
    Cacheado con @st.cache_resource: se construye UNA sola vez por sesión
    del servidor Streamlit, resolviendo el problema de reconstrucción de BM25.
    """
    import pandas as pd
    from src.embeddings.indexer import load_chunks_jsonl
    from src.retrieval.bm25_search import BM25Retriever
    from src.retrieval.vector_search import VectorRetriever
    from src.retrieval.hybrid_search import HybridRetriever

    chunks = load_chunks_jsonl(CHUNKS_JSONL)
    chunks_df = pd.DataFrame(chunks)

    encoder = load_encoder()
    qdrant = load_qdrant_client()

    bm25 = BM25Retriever(chunks_df)           # construido en memoria, cacheado
    vec = VectorRetriever(qdrant, encoder)
    hybrid = HybridRetriever(bm25, vec, chunks_df)

    return bm25, vec, hybrid


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    # La key viene del .env (local) o del entorno (Colab).
    # Si no está en el entorno, el usuario la puede ingresar aquí.
    env_key = os.environ.get("GROQ_API_KEY", "")
    groq_key = st.text_input(
        "API Key de Groq",
        value=env_key,
        type="password",
        help="Si definiste GROQ_API_KEY en .env ya está cargada automáticamente.",
    )
    if groq_key and not env_key:
        os.environ["GROQ_API_KEY"] = groq_key  # inyectar para que settings la vea

    st.markdown("---")
    st.markdown("### 🔍 Búsqueda")

    config_display = st.selectbox(
        "Configuración de recuperación",
        ["C — Híbrida RRF (recomendada)", "B — Vectorial semántica", "A — BM25 lexical"],
    )
    config_map = {
        "C — Híbrida RRF (recomendada)": "hybrid",
        "B — Vectorial semántica": "vector",
        "A — BM25 lexical": "bm25",
    }
    selected_config = config_map[config_display]
    top_k = st.slider("Fragmentos a recuperar (top-k)", 1, 10, TOP_K)
    show_fragments = st.checkbox("Mostrar fragmentos recuperados", value=True)
    show_scores = st.checkbox("Mostrar puntuaciones", value=False)

    st.markdown("---")
    st.markdown("### 📊 Resultados experimentales")
    st.markdown("""
| Config | Recall@5 | MRR |
|--------|----------|-----|
| A BM25 | 0.715 | 0.651 |
| B Vect | 0.779 | 0.803 |
| **C RRF** | **0.792** | **0.809** |
    """)


# ─────────────────────────────────────────────
# Carga de recursos
# ─────────────────────────────────────────────
resources_ok = False
try:
    bm25_ret, vec_ret, hybrid_ret = load_retrievers()
    resources_ok = True
except Exception as exc:
    st.error(
        f"⚠️ No se pudieron cargar los recursos: {exc}\n\n"
        "Verifica que qdrant_storage y el JSONL de chunks existan."
    )


# ─────────────────────────────────────────────
# Área principal
# ─────────────────────────────────────────────
st.markdown("#### 💬 Realiza una consulta")

preguntas_ejemplo = [
    "¿Cuáles son los Estándares de Calidad Ambiental para el aire según el DS 003-2017-MINAM?",
    "¿Qué obligaciones tiene el generador de residuos sólidos peligrosos?",
    "¿Cuál es el procedimiento para obtener la Certificación Ambiental?",
    "¿Qué es el Sistema de Evaluación del Impacto Ambiental (SEIA)?",
    "¿Qué sanciones establece la Ley de Fiscalización Ambiental?",
]

with st.expander("🗂 Preguntas de ejemplo", expanded=False):
    for ej in preguntas_ejemplo:
        if st.button(ej, key=f"btn_{ej[:30]}"):
            st.session_state["pregunta_input"] = ej

pregunta = st.text_area(
    "Escribe tu consulta sobre normativa ambiental peruana:",
    value=st.session_state.get("pregunta_input", ""),
    height=90,
    placeholder="Ejemplo: ¿Cuáles son los límites permisibles de ruido ambiental?",
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    buscar = st.button("🔍 Consultar", type="primary", disabled=not resources_ok)
with col_clear:
    if st.button("✕ Limpiar"):
        st.session_state["pregunta_input"] = ""
        st.rerun()


# ─────────────────────────────────────────────
# Ejecución
# ─────────────────────────────────────────────
if buscar and pregunta.strip():
    if not os.environ.get("GROQ_API_KEY"):
        st.warning("⚠️ Ingresa tu API Key de Groq en el panel lateral.")
        st.stop()

    retriever_fn = {
        "bm25":   lambda q, k: bm25_ret.search(q, top_k=k),
        "vector": lambda q, k: vec_ret.search(q, top_k=k),
        "hybrid": lambda q, k: hybrid_ret.search(q, top_k=k),
    }[selected_config]

    badge_html = {
        "bm25":   '<span class="config-badge badge-bm25">A — BM25</span>',
        "vector": '<span class="config-badge badge-vectorial">B — Vectorial</span>',
        "hybrid": '<span class="config-badge badge-hibrida">C — Híbrida RRF</span>',
    }[selected_config]

    with st.spinner("Recuperando fragmentos…"):
        t0 = time.time()
        fragmentos = retriever_fn(pregunta, top_k)
        t_ret = time.time() - t0

    with st.spinner("Generando respuesta con LLaMA 3.1…"):
        from src.generation.generator import RAGGenerator
        gen = RAGGenerator()   # toma la key de os.environ vía settings
        resultado = gen.generate(pregunta, fragmentos)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fragmentos", len(fragmentos))
    c2.metric("Latencia recuperación", f"{t_ret:.3f} s")
    c3.metric("Latencia generación", f"{resultado['latencia_seg']} s")
    c4.metric("Tokens", resultado["tokens_usados"])

    st.markdown("---")
    st.markdown(f"**Configuración:** {badge_html}", unsafe_allow_html=True)
    st.markdown("#### 📝 Respuesta")
    st.markdown(f'<div class="answer-box">{resultado["respuesta"]}</div>', unsafe_allow_html=True)

    if show_fragments and fragmentos:
        st.markdown("---")
        st.markdown(f"#### 📄 Fragmentos recuperados (top-{top_k})")
        for i, frag in enumerate(fragmentos, 1):
            scores_html = ""
            if show_scores:
                scores_html = (
                    f'<div class="frag-scores">'
                    f'BM25: {frag.get("score_bm25", 0):.4f} | '
                    f'Vectorial: {frag.get("score_vectorial", 0):.4f} | '
                    f'RRF: {frag.get("score_rrf", 0):.5f}</div>'
                )
            texto = frag.get("texto", "")

            # ── Punto 12: badge de estado_vigencia ───────────────────────────
            vigencia = str(frag.get("estado_vigencia", "no_verificado")).lower()
            vigencia_config = {
                "vigente":       ("#d4edda", "#155724", "✅ vigente"),
                "modificada":    ("#fff3cd", "#856404", "⚠️ modificada"),
                "derogada":      ("#f8d7da", "#721c24", "❌ derogada"),
                "no_verificado": ("#e2e3e5", "#383d41", "❓ no verificado"),
            }
            bg, fg, label = vigencia_config.get(
                vigencia, ("#e2e3e5", "#383d41", f"❓ {vigencia}")
            )
            vigencia_badge = (
                f'<span style="background:{bg};color:{fg};padding:.15rem .5rem;'
                f'border-radius:4px;font-size:.75rem;font-weight:600;">{label}</span>'
            )

            # ── Punto 11: advertencia de extracción parcial ───────────────────
            parcial_badge = ""
            if str(frag.get("estado_extraccion", "")).upper() == "PARCIAL":
                parcial_badge = (
                    '<span style="background:#fff3cd;color:#856404;padding:.15rem .5rem;'
                    'border-radius:4px;font-size:.75rem;font-weight:600;margin-left:.3rem;">'
                    '⚠️ extracción parcial</span>'
                )

            st.markdown(f"""
            <div class="fragment-card">
                <div class="frag-header">
                    [{i}] {frag.get("id_documento","")} — {frag.get("titulo_documento","")}
                    <span style="color:#5a7a6a;font-weight:400;font-size:.82rem;">
                      · {frag.get("seccion","")[:80]}
                    </span>
                    <span style="margin-left:.5rem;">{vigencia_badge}{parcial_badge}</span>
                </div>
                {texto[:500]}{"…" if len(texto) > 500 else ""}
                {scores_html}
            </div>
            """, unsafe_allow_html=True)

            # ── Punto 12: advertencia extra para normas no vigentes ───────────
            if vigencia == "derogada":
                st.warning(
                    f"⚠️ Este fragmento proviene de una norma **derogada** "
                    f"({frag.get('titulo_documento', '')}). "
                    "Verifica la normativa vigente antes de usar esta información."
                )
            elif vigencia == "no_verificado":
                st.caption(
                    "❓ La vigencia de este documento no ha sido verificada. "
                    "Consulta el SPIJ o el Diario Oficial El Peruano para confirmarla."
                )

elif buscar and not pregunta.strip():
    st.warning("Escribe una consulta antes de buscar.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:.8rem;'>"
    "Plataforma Ambiental · RAG Normativa Peruana — BAAI/bge-m3 · Qdrant · BM25 · RRF · Groq LLaMA 3.1"
    "</p>",
    unsafe_allow_html=True,
)
