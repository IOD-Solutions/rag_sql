# app.py — Streamlit Chat RAG (backend local: SQLite/Whoosh)
import os
from pathlib import Path
import streamlit as st
# IMPORTANT : utilise rag_core_local (pas Elasticsearch)
import rag_core_local as rag_core

import sys,urllib.request, zipfile

def ensure_assets():
    mode = os.getenv("RAG_MODE", "sqlite")
    if mode == "sqlite":
        db_path = os.getenv("RAG_DB_PATH", "/tmp/rag.db")
        url = os.getenv("RAG_DB_URL")
        if url and not os.path.exists(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            print(f"[preflight] Downloading rag.db -> {db_path}", file=sys.stderr, flush=True)
            urllib.request.urlretrieve(url, db_path)
    elif mode == "whoosh":
        idx_dir = os.getenv("WHOOSH_INDEX_DIR", "/tmp/whoosh_index")
        url = os.getenv("WHOOSH_INDEX_URL")
        if url and not os.path.isdir(idx_dir):
            os.makedirs(idx_dir, exist_ok=True)
            tmpzip = "/tmp/whoosh_index.zip"
            print(f"[preflight] Downloading whoosh index", file=sys.stderr, flush=True)
            urllib.request.urlretrieve(url, tmpzip)
            with zipfile.ZipFile(tmpzip) as z: z.extractall(idx_dir)
            os.remove(tmpzip)

ensure_assets()

st.set_page_config(page_title="Super RAG", page_icon="🧠", layout="wide")
st.title("🧠 Super RAG — Chat (local)")

with st.sidebar:
    st.header("⚙️ Réglages")

    mode = os.getenv("RAG_MODE", "sqlite")
    st.caption(f"🔧 Backend: **{mode}**")
    use_llm = st.checkbox("Réponse LLM (citations)", value=True)
    model   = st.selectbox("Modèle OpenAI", ["gpt-4o-mini", "gpt-4o"], index=0)
    openai_key_set = bool(os.getenv("OPENAI_API_KEY"))
    st.caption("🔑 OPENAI_API_KEY " + ("✅ détectée" if openai_key_set else "❌ manquante"))

    # Indexation locale (optionnelle sur Render si tu fournis RAG_DB_URL/WHOOSH_INDEX_URL)
    data_root = st.text_input("Dossier local à indexer (si présent)", value=str(rag_core.DATA_ROOT))
    if st.button("📚 (Re)indexer depuis ce dossier"):
        with st.spinner("Indexation en cours..."):
            try:
                added = rag_core.index_folder(Path(data_root))
                st.success(f"(Ré)indexés: {added} fichiers" if added else "Index à jour")
            except Exception as e:
                st.error(f"Erreur d'indexation: {e}")

st.divider()

q = st.text_input("Ta question")
if st.button("Poser la question") and q.strip():
    with st.spinner("Recherche + rerank en cours…"):
        if use_llm:
            os.environ["OPENAI_MODEL"] = model
        try:
            answer = rag_core.ask(q.strip(), k_bm25=12, k_final=6, use_llm=use_llm)
        except Exception as e:
            answer = f"❌ Erreur: {e}"
    st.markdown(answer)
else:
    st.info("Entre une question puis clique sur **Poser la question**.")
