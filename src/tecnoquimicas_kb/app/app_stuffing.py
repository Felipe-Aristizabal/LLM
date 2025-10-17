import os, streamlit as st
from dotenv import load_dotenv
from tecnoquimicas_kb.rag.stuffing import *

load_dotenv()

st.set_page_config(page_title="TQ Q&A (Stuffing)", layout="wide")
st.title("Asistente Q&A – Tecnoquímicas (Modo Stuffing)")

with st.sidebar:
    st.header("Modelo")
    provider = st.selectbox("Provider", ["gemini", "ollama"], index=0)
    os.environ["MODEL_PROVIDER"] = provider
    if provider == "gemini":
        os.environ["GEN_MODEL_ID"] = st.text_input(
            "Gemini model id", value=os.getenv("GEN_MODEL_ID", "gemini-2.5-pro")
        )
        st.caption("Requiere GOOGLE_API_KEY en .env")
    else:
        os.environ["OLLAMA_MODEL_ID"] = st.text_input(
            "Ollama model id", value=os.getenv("OLLAMA_MODEL_ID", "gemma3:4b")
        )
        st.caption("Asegura `ollama pull gemma3:4b` o `gemma3:270m`.")

    st.header("Datos")
    data_dir = st.text_input(
        "Directorio de chunks limpios", value="src/tecnoquimicas_kb/data/clean"
    )

if "docs" not in st.session_state:
    with st.spinner("Cargando documentos..."):
        st.session_state["docs"] = load_all_docs(data_dir)

st.success(f"Documentos cargados: {len(st.session_state['docs'])}")

tab1, tab2, tab3 = st.tabs(["Resumen", "FAQ", "Q&A"])

with tab1:
    st.subheader("Resumen (onboarding)")
    if st.button("Generar resumen"):
        with st.spinner("Generando..."):
            out = summarize_stuffing(st.session_state["docs"])
            st.write(out)

with tab2:
    st.subheader("Preguntas frecuentes")
    n = st.slider("Cantidad de FAQs", 5, 20, 10)
    if st.button("Generar FAQs"):
        with st.spinner("Generando..."):
            out = faq_stuffing(st.session_state["docs"], n=n)
            st.write(out)

with tab3:
    st.subheader("Pregunta y Respuesta")
    q = st.text_input("Escribe tu pregunta:", "")
    if st.button("Responder"):
        if not q.strip():
            st.warning("Escribe una pregunta.")
        else:
            with st.spinner("Generando..."):
                out = qa_stuffing(st.session_state["docs"], q)
                st.markdown(out)
