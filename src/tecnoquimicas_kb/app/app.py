# streamlit run src/tecnoquimicas_kb/app/app.py
import os, streamlit as st
from tecnoquimicas_kb.rag.chains import load_index, summarize, make_faq, answer

st.set_page_config(page_title="TQ Q&A", layout="wide")
st.title("Asistente Q&A – Tecnoquímicas (Módulo 1)")

with st.sidebar:
    st.header("Modelo")
    provider = st.selectbox("Provider", ["gemini", "ollama"], index=0)
    os.environ["MODEL_PROVIDER"] = provider
    if provider == "gemini":
        os.environ["GEN_MODEL_ID"] = st.text_input("Gemini model id", value=os.getenv("GEN_MODEL_ID", "gemini-2.5-pro"))
        st.caption("Requiere GOOGLE_API_KEY en entorno.")
    else:
        os.environ["OLLAMA_MODEL_ID"] = st.text_input("Ollama model id", value=os.getenv("OLLAMA_MODEL_ID", "gemma3:4b"))
        st.caption("Asegura `ollama run gemma3:4b` o `ollama pull gemma3:4b`.")

    st.header("Embeddings")
    embp = st.selectbox("Embed provider", ["local","google","ollama"], index=["local","google","ollama"].index(os.getenv("EMBED_PROVIDER","local")))
    os.environ["EMBED_PROVIDER"] = embp
    if embp == "google":
        st.caption("Usará `text-embedding-004` (Google).")
    elif embp == "ollama":
        os.environ["OLLAMA_EMBED_MODEL"] = st.text_input("Ollama embed model", value=os.getenv("OLLAMA_EMBED_MODEL","nomic-embed-text"))

st.session_state.setdefault("vs", load_index())

tab1, tab2, tab3 = st.tabs(["Resumen", "FAQ", "Q&A"])

with tab1:
    if st.button("Generar resumen (onboarding)"):
        with st.spinner("Generando..."):
            st.write(summarize(st.session_state["vs"]))

with tab2:
    n = st.slider("Cantidad de FAQs", 5, 20, 10)
    if st.button("Generar FAQs"):
        with st.spinner("Generando..."):
            st.write(make_faq(st.session_state["vs"], n=n))

with tab3:
    q = st.text_input("Escribe tu pregunta:", "")
    if st.button("Responder"):
        if not q.strip():
            st.warning("Escribe una pregunta.")
        else:
            with st.spinner("Buscando en la base de conocimiento..."):
                ans, docs = answer(st.session_state["vs"], q)
                st.markdown(ans)
                with st.expander("Ver fragments recuperados"):
                    for i, d in enumerate(docs, 1):
                        st.write(f"#{i}", d.metadata)
                        st.write(d.page_content[:900] + ("..." if len(d.page_content)>900 else ""))
