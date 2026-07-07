"""
app.py — Streamlit frontend for the Document AI Assistant.

Talks to the FastAPI backend (see ../backend/main.py) to:
  1. Upload a PDF/TXT document.
  2. Ask questions about it in a chat interface.
  3. Show streaming answers with a "view sources" panel per response.
"""

import json
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Document AI Assistant", page_icon="📄", layout="wide")

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "doc_meta" not in st.session_state:
    st.session_state.doc_meta = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources}

# --------------------------------------------------------------------------
# Sidebar: upload
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document AI Assistant")
    st.caption("Powered by Gemini 2.5 Flash + LangChain + ChromaDB. "
               "Upload a document, then ask questions grounded in its content.")

    uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded is not None:
        if st.button("Process document", type="primary", use_container_width=True):
            with st.spinner("Extracting, chunking, and embedding..."):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue())}
                    resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                    if resp.status_code == 200:
                        st.session_state.doc_meta = resp.json()
                        st.session_state.doc_id = st.session_state.doc_meta["doc_id"]
                        st.session_state.messages = []
                        st.success(f"Processed '{uploaded.name}' into "
                                   f"{st.session_state.doc_meta['num_chunks']} chunks.")
                    else:
                        st.error(f"Upload failed: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Can't reach backend at {BACKEND_URL}. Is it running?")

    if st.session_state.doc_meta:
        st.divider()
        st.markdown("**Current document**")
        meta = st.session_state.doc_meta
        st.write(f"📄 {meta['filename']}")
        st.write(f"Pages: {meta['num_pages']} · Chunks: {meta['num_chunks']}")
        if st.button("Clear / upload a new document", use_container_width=True):
            st.session_state.doc_id = None
            st.session_state.doc_meta = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    top_k = st.slider("Chunks to retrieve (top-k)", min_value=1, max_value=10, value=4)
    use_streaming = st.toggle("Stream response", value=True)

# --------------------------------------------------------------------------
# Main: chat interface
# --------------------------------------------------------------------------
st.header("Chat with your document")

if not st.session_state.doc_id:
    st.info("👈 Upload and process a document in the sidebar to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("🔎 View sources"):
                    for s in msg["sources"]:
                        st.markdown(f"**Page {s['page']} · Chunk {s['chunk_index']}**")
                        st.caption(s["snippet"])
                        st.divider()

    query = st.chat_input("Ask a question about the document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            sources = []

            payload = {
                "doc_id": st.session_state.doc_id,
                "query": query,
                "top_k": top_k,
                "stream": use_streaming,
            }

            try:
                if use_streaming:
                    with requests.post(
                        f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=120
                    ) as resp:
                        if resp.status_code != 200:
                            st.error(f"Chat failed: {resp.text}")
                        else:
                            for line in resp.iter_lines(decode_unicode=True):
                                if not line:
                                    continue
                                event = json.loads(line)
                                if event["type"] == "token":
                                    full_answer += event["content"]
                                    placeholder.markdown(full_answer + "▌")
                                elif event["type"] == "sources":
                                    sources = event["sources"]
                                elif event["type"] == "error":
                                    st.error(event["message"])
                            placeholder.markdown(full_answer)
                else:
                    resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        full_answer = data["answer"]
                        sources = data["sources"]
                        placeholder.markdown(full_answer)
                    else:
                        st.error(f"Chat failed: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach backend at {BACKEND_URL}. Is it running?")

            if sources:
                with st.expander("🔎 View sources"):
                    for s in sources:
                        st.markdown(f"**Page {s['page']} · Chunk {s['chunk_index']}**")
                        st.caption(s["snippet"])
                        st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
        })
