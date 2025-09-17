#!/usr/bin/env bash
set -e

# --- SQLite mode ---
if [ "${RAG_MODE}" = "sqlite" ]; then
  if [ -n "${RAG_DB_URL}" ] && [ -n "${RAG_DB_PATH}" ] && [ ! -f "${RAG_DB_PATH}" ]; then
    echo "🔽 Downloading rag.db..."
    mkdir -p "$(dirname "${RAG_DB_PATH}")"
    curl -fsSL "${RAG_DB_URL}" -o "${RAG_DB_PATH}"
    echo "✅ rag.db downloaded to ${RAG_DB_PATH}"
  fi
fi

# --- Whoosh mode (optionnel) ---
if [ "${RAG_MODE}" = "whoosh" ]; then
  if [ -n "${WHOOSH_INDEX_URL}" ] && [ -n "${WHOOSH_INDEX_DIR}" ] && [ ! -d "${WHOOSH_INDEX_DIR}" ]; then
    echo "🔽 Downloading whoosh_index.zip..."
    mkdir -p "${WHOOSH_INDEX_DIR}"
    curl -fsSL "${WHOOSH_INDEX_URL}" -o /tmp/whoosh_index.zip
    unzip -q /tmp/whoosh_index.zip -d "${WHOOSH_INDEX_DIR}"
    echo "✅ Whoosh index ready at ${WHOOSH_INDEX_DIR}"
  fi
fi
