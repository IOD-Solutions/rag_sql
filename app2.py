# app.py — Streamlit Chat RAG (BM25 + rerank + LLM avec citations)
import os, sys
from pathlib import Path
import streamlit as st
import rag_core_local

# --- Diagnostics au démarrage (logs Render) ---
print("ES_CLOUD_ID set:", bool(os.getenv("ES_CLOUD_ID")), file=sys.stderr, flush=True)
print("ES_API_KEY set:", bool(os.getenv("ES_API_KEY")), file=sys.stderr, flush=True)
print("ES_URL:", os.getenv("ES_URL"), file=sys.stderr, flush=True)
print("rag_core file:", getattr(rag_core, "__file__", "?"), file=sys.stderr, flush=True)
try:
    client = rag_core.es()
    info = client.info()
    print("ES_OK:", info.body["cluster_name"], info.body["version"]["number"], file=sys.stderr, flush=True)
except Exception as e:
    print("ES_FAIL:", repr(e), file=sys.stderr, flush=True)

st.set_page_config(page_title="Super RAG", page_icon="🧠", layout="wide")
st.title("🧠 Super RAG — Chat")

with st.sidebar:
    st.header("⚙️ Réglages")

    using_cloud = bool(os.getenv("ES_CLOUD_ID")) and bool(os.getenv("ES_API_KEY"))
    read_only   = os.getenv("READONLY") == "1"

    if using_cloud:
        st.caption("🔗 Elasticsearch: **Elastic Cloud (ES_CLOUD_ID)**")
        es_url = None  # on n'utilise pas ES_URL en mode cloud
    else:
        es_url = st.text_input("Elasticsearch URL (dev)", value=os.getenv("ES_URL", "http://localhost:9200"))
        rag_core.ES_URL = es_url

    index_name = st.text_input("Index", value=rag_core.INDEX_NAME)
    data_root  = st.text_input("Dossier des documents", value=str(rag_core.DATA_ROOT))
    use_llm    = st.checkbox("Réponse LLM (citations)", value=True)
    model      = st.selectbox("Modèle OpenAI", ["gpt-4o-mini", "gpt-4o"], index=0)
    openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
    st.caption("🔑 OPENAI_API_KEY " + ("✅ détectée" if openai_key_set else "❌ manquante"))

    rag_core.INDEX_NAME = index_name
    rag_core.DATA_ROOT  = Path(data_root)

    if read_only:
        st.caption("🔒 Lecture seule : indexation désactivée (pré-indexation conseillée).")
    else:
        if st.button("📚 (Re)indexer maintenant"):
            with st.spinner("Indexation en cours..."):
                try:
                    added = rag_core.index_folder(Path(data_root))
                    st.success(f"(Ré)indexés: {added} fichiers" if added else "Index à jour")
                except Exception as e:
                    st.error(f"Erreur d'indexation: {e}")

st.divider()

q = st.text_input("Ta question")
go = st.button("Poser la question")

if go and q.strip():
    with st.spinner("Recherche + rerank en cours…"):
        try:
            os.environ["OPENAI_MODEL"] = model  # optionnel
            answer = rag_core.ask(q.strip(), k_bm25=12, k_final=6, use_llm=use_llm)
        except Exception as e:
            answer = f"❌ Erreur: {e}"

    st.markdown(answer)
    mode = "Elastic Cloud" if using_cloud else f"URL: {es_url}"
    st.caption(f"Index: `{index_name}` • ES: {mode} • Dossier: {data_root}")
else:
    st.info("Entre une question puis clique sur **Poser la question**.")
