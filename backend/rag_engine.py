"""
rag_engine.py
-------------
Core RAG logic built on LangChain:
  - PyMuPDF (fitz) for PDF text extraction (page-aware)
  - LangChain's RecursiveCharacterTextSplitter for chunking
  - Gemini embeddings (via langchain-google-genai) for vectors
  - ChromaDB (via langchain-chroma) as the vector store
  - Gemini 2.5 Flash (via langchain-google-genai) for grounded generation
  - A simple LCEL chain (prompt | llm) ties retrieval + generation together
"""

import os
import uuid
from typing import List, Dict, Any, Generator

import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

# --------------------------------------------------------------------------
# Gemini models (embeddings + chat) and the Chroma vector store
# --------------------------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

llm = ChatGoogleGenerativeAI(
    model=CHAT_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

vectorstore = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# --------------------------------------------------------------------------
# 1. Text extraction (PyMuPDF, page-aware)
# --------------------------------------------------------------------------

def extract_pages(file_bytes: bytes, filename: str) -> List[Document]:
    """Return one LangChain Document per page, tagged with its page number."""
    lower = filename.lower()
    docs: List[Document] = []

    if lower.endswith(".pdf"):
        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        try:
            for i in range(len(pdf)):
                text = pdf[i].get_text()
                docs.append(Document(page_content=text, metadata={"page": i + 1}))
        finally:
            pdf.close()
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"page": 1}))

    return docs


# --------------------------------------------------------------------------
# 2. Chunking (LangChain's RecursiveCharacterTextSplitter, page-preserving)
# --------------------------------------------------------------------------

def chunk_document(page_docs: List[Document], doc_id: str, filename: str) -> List[Document]:
    """Split page-level documents into overlapping chunks. LangChain's
    splitter carries each source Document's metadata (page number) forward
    onto every chunk derived from it, so page attribution stays accurate."""
    split_docs = splitter.split_documents(page_docs)

    chunks: List[Document] = []
    idx = 0
    for d in split_docs:
        if not d.page_content.strip():
            continue
        d.metadata["doc_id"] = doc_id
        d.metadata["filename"] = filename
        d.metadata["chunk_index"] = idx
        chunks.append(d)
        idx += 1
    return chunks


# --------------------------------------------------------------------------
# 3 & 4. Embedding + storage (handled together by the Chroma vector store)
# --------------------------------------------------------------------------

def store_document(doc_id: str, chunks: List[Document]) -> None:
    if not chunks:
        return
    ids = [f"{doc_id}_{c.metadata['chunk_index']}" for c in chunks]
    vectorstore.add_documents(chunks, ids=ids)  # embeds via `embeddings` internally


# --------------------------------------------------------------------------
# 5. Retrieval
# --------------------------------------------------------------------------

def retrieve(query: str, doc_id: str, k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    results = vectorstore.similarity_search_with_score(
        query, k=k, filter={"doc_id": doc_id}
    )
    hits = []
    for doc, score in results:
        hits.append({
            "chunk_id": f"{doc_id}_{doc.metadata.get('chunk_index')}",
            "text": doc.page_content,
            "page": doc.metadata.get("page"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "distance": float(score),
        })
    return hits


# --------------------------------------------------------------------------
# 6. Generation (grounded, with citations) via an LCEL chain
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a document \
the user uploaded. Only use the provided context to answer. If the answer is not \
in the context, say you don't know based on the document.

When you use information from a chunk, cite it inline like [Page P] using the page \
number given for that chunk. Be concise and accurate. Do not invent page numbers."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

rag_chain = prompt | llm  # simple LCEL retrieval-augmented-generation chain


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Chunk {c['chunk_index']} | Page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, chunks: List[Dict[str, Any]]) -> str:
    context = _build_context(chunks)
    resp = rag_chain.invoke({"context": context, "question": query})
    return resp.content


def generate_answer_stream(query: str, chunks: List[Dict[str, Any]]) -> Generator[str, None, None]:
    """Yields raw text tokens as they arrive from Gemini."""
    context = _build_context(chunks)
    for token_chunk in rag_chain.stream({"context": context, "question": query}):
        if token_chunk.content:
            yield token_chunk.content


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def new_doc_id() -> str:
    return uuid.uuid4().hex[:12]


def document_chunk_count(doc_id: str) -> int:
    res = vectorstore.get(where={"doc_id": doc_id})
    return len(res["ids"]) if res and res.get("ids") else 0
