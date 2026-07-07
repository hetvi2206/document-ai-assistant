"""
main.py — FastAPI backend for the Document AI Assistant.
Stack: FastAPI + LangChain + Gemini 2.5 Flash + Gemini Embeddings + ChromaDB + PyMuPDF.

Endpoints:
  POST /upload      -> upload a PDF/TXT, extract+chunk+embed+store
  GET  /documents    -> list uploaded documents
  POST /chat         -> ask a question about a document (streaming or not)
  GET  /health       -> simple healthcheck
"""

import json
import traceback
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag_engine as rag

app = FastAPI(title="Document AI Assistant API (Gemini + LangChain)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory registry of uploaded documents
DOCUMENTS: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    doc_id: str
    query: str
    top_k: Optional[int] = None
    stream: Optional[bool] = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_file"
    if not (filename.lower().endswith(".pdf") or filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    try:
        file_bytes = await file.read()
        page_docs = rag.extract_pages(file_bytes, filename)

        doc_id = rag.new_doc_id()
        chunks = rag.chunk_document(page_docs, doc_id, filename)
        if not chunks:
            raise HTTPException(status_code=400, detail="No extractable text found in file.")

        rag.store_document(doc_id, chunks)

        DOCUMENTS[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "num_pages": len(page_docs),
            "num_chunks": len(chunks),
        }
        return DOCUMENTS[doc_id]
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")


@app.get("/documents")
def list_documents():
    return {"documents": list(DOCUMENTS.values())}


def _sources_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "chunk_index": c["chunk_index"],
            "page": c["page"],
            "snippet": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
        }
        for c in chunks
    ]


@app.post("/chat")
def chat(req: ChatRequest):
    if req.doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Unknown doc_id. Upload a document first.")

    k = req.top_k or rag.DEFAULT_TOP_K
    chunks = rag.retrieve(req.query, req.doc_id, k=k)

    if not req.stream:
        answer = rag.generate_answer(req.query, chunks) if chunks else (
            "I couldn't find relevant content in this document to answer that."
        )
        return {"answer": answer, "sources": _sources_from_chunks(chunks)}

    # Streaming response: NDJSON lines of {"type": "token"/"sources", ...}
    def event_stream():
        try:
            if not chunks:
                msg = "I couldn't find relevant content in this document to answer that."
                yield json.dumps({"type": "token", "content": msg}) + "\n"
            else:
                for token in rag.generate_answer_stream(req.query, chunks):
                    yield json.dumps({"type": "token", "content": token}) + "\n"
            yield json.dumps({"type": "sources", "sources": _sources_from_chunks(chunks)}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
