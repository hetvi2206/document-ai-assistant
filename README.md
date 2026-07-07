# Document AI Assistant (RAG)

Upload a PDF or text file, then ask questions about it and get answers grounded
in the document, with page/chunk citations. Built entirely on free/low-cost
components — no OpenAI dependency.

## At a glance

| | |
|---|---|
| Focus | Document AI Assistant |
| Core AI task | RAG: chunk → embed → retrieve → generate |
| Primary output | Grounded answers with citations |

## Tech stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Gemini 2.5 Flash (free-tier API) |
| Embeddings | Gemini Embedding (`gemini-embedding-2`, via `langchain-google-genai`) |
| Vector DB | ChromaDB (local, embedded, persisted to disk) |
| PDF parsing | PyMuPDF (`fitz`) |
| Text chunking | LangChain's `RecursiveCharacterTextSplitter` |
| RAG framework | LangChain (LCEL chain: prompt \| llm, over Chroma retrieval) |
| Environment | Python 3.12 |

## Project structure

```
rag-assistant/
├── backend/
│   ├── main.py           # FastAPI app: /upload, /chat, /documents
│   ├── rag_engine.py      # extraction, chunking, embeddings, retrieval, generation
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── app.py             # Streamlit chat UI
│   └── requirements.txt
└── README.md
```

## How chunking / retrieval works

**1. Extraction (PyMuPDF)** — `fitz` opens the PDF and pulls text page by
page. Each page becomes a LangChain `Document` with `metadata={"page": N}`
(a `.txt` upload is treated as a single page). Page-level granularity here is
what makes accurate page citations possible later.

**2. Chunking (LangChain `RecursiveCharacterTextSplitter`)** — The list of
page `Document`s is passed to `splitter.split_documents(...)`. LangChain's
splitter tries to break on paragraph → sentence → word boundaries (in that
order) before falling back to a hard character cut, and — importantly —
**carries each source page's metadata forward** onto every chunk derived from
it. Default settings: `chunk_size=1000` characters, `chunk_overlap=150`
characters, so context isn't lost across chunk boundaries. Each chunk also
gets a `doc_id` (scopes it to the uploaded file) and a `chunk_index`.

**3. Embedding (Gemini Embedding)** — Each chunk is embedded with Google's
`gemini-embedding-2` model via `GoogleGenerativeAIEmbeddings`.

**4. Storage (ChromaDB)** — Chunks (as LangChain `Document`s, with their
embeddings computed automatically) are added to a persistent Chroma
collection on disk, tagged with metadata `{doc_id, filename, page,
chunk_index}`.

**5. Retrieval** — On each chat query, `vectorstore.similarity_search_with_score`
embeds the query and returns the top-`k` most similar chunks (`k` is
adjustable in the UI, default 4), filtered to the active `doc_id` so multiple
uploaded documents never bleed into each other's answers.

**6. Generation (Gemini 2.5 Flash + LCEL)** — Retrieved chunks are formatted
into a context block labeled `[Chunk N | Page P]` and fed into a small LCEL
chain (`ChatPromptTemplate | ChatGoogleGenerativeAI`). The system prompt
instructs the model to answer **only** from that context and cite pages
inline, e.g. `... as shown in the results [Page 3]`. The backend separately
returns the raw chunk metadata + a text snippet for each source, so the
frontend's "view sources" panel doesn't depend on what the model chose to
cite in prose.

Answers can stream: `/chat` returns newline-delimited JSON
(`{"type": "token", ...}` chunks from `rag_chain.stream(...)`, then a final
`{"type": "sources", ...}`), which the Streamlit app reads incrementally to
show the answer typing out, followed by the source chunks.

## Setup

### 1. Get a free Gemini API key

Go to [Google AI Studio](https://aistudio.google.com/apikey) and create a
free API key.

### 2. Backend

```bash
cd backend
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your GOOGLE_API_KEY
uvicorn main:app --reload --port 8000
```

The API will be live at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

### 3. Frontend

In a second terminal:

```bash
cd frontend
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000   # optional, this is the default
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Using it

1. In the sidebar, upload a `.pdf` or `.txt` file and click **Process document**.
2. Wait for the "Processed into N chunks" confirmation.
3. Ask a question in the chat box.
4. The answer streams in; expand **🔎 View sources** below it to see which
   page(s) and chunk(s) were used to ground the answer.
5. Adjust **top-k** in the sidebar to retrieve more/fewer chunks, or toggle
   streaming off if you'd rather get the full answer at once.

## API reference (backend)

- `POST /upload` — multipart form with `file`. Returns
  `{doc_id, filename, num_pages, num_chunks}`.
- `GET /documents` — list of documents processed in this session.
- `POST /chat` — JSON body `{doc_id, query, top_k?, stream?}`.
  - `stream=false` → `{answer, sources: [{chunk_index, page, snippet}]}`
  - `stream=true` → NDJSON stream of `{"type": "token", "content": "..."}`
    lines, ending with `{"type": "sources", "sources": [...]}` then
    `{"type": "done"}`.
