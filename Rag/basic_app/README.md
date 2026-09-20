# Basic RAG (Part 1) — FastAPI + ChromaDB

Part 1 ka running app. Ik chroma + embedding + LLM API use karte hain. Simple, no LangChain.

## Project structure

```
basic_app/
├── requirements.txt     # dependencies (pin kar ke rakho)
├── .env.example         # copy to .env, apne keys daalo
├── config.py            # saare env vars yaha jaate hain
├── chunking.py          # recursive chunker (khud likha, taaki samjh aaye)
├── vector_store.py      # embeddings + ChromaDB access (swap point)
├── ingest.py            # PDF -> text -> clean -> chunk -> embed -> store
├── rag.py               # retrieve -> context -> prompt -> LLM
├── main.py              # FastAPI endpoints (ingest + query)
└── data/
    ├── docs/            # yaha ingress PDFs daalo
    └── chroma/          # persistent DB (auto banega)
```

## Setup

```bash
cd Rag/basic_app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env     # aur API key daalo
```

Manually the embed API works with OpenAI, or any OpenAI-compatible endpoint
(Ollama/LM Studio) by changing `OPENAI_BASE_URL`.

## Run the indexes

```bash
# 1) Apne PDFs data/docs/ mein rakh ke:
python ingest.py

# 2) API start karo:
uvicorn main:app --reload
```

## Try it

Docs: http://127.0.0.1:8000/docs

- `GET /health` — system status, kitne chunks indexed.
- `POST /ingest` — file upload se naya PDF index karo.
- `POST /query` — query bhejo, answer + sources get karo.

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the company leave policy?"}'
```

## File-by-file explanation

### config.py
Saare settings ek jagah, .env se. `Settings` class env vars ko typed fields mein
load karti hai. Agar key missing ho to startup par fast error (fail-fast), runtime
mein silent break nahi.

### chunking.py
`recursive_chunk(text, chunk_size, overlap)`:
- Paragraph (`\n\n`) se split try karo, phir sentence (`. `), phir word.
- Overlap re-attach karta hai previous chunk ka tail — sentence/phrase boundary par
  info loss avoid karne ke liye.
- Input: raw text. Output: list of chunk strings.
- Isi as a roop LangChain ka `RecursiveCharacterTextSplitter` — lekinn hum khud
  likh kar samajh rahe hain ki ye abstract kya karta hai.

### vector_store.py
`VectorStore` class wrapping:
- `embed(text)` — OpenAI embeddings API, text -> 1536-d array (same model docs/query).
- `store_chunks(chunks, metadatas, ids)` — har chunk embed karke Chroma collection
  mein upsert (doc va metadata ke saath).
- `search(query, top_k)` — query embed kar ke collection.query; result dict twin
  documents, metadatas, distances.
- Agar baad mein Qdrant/Pinecone switch karna hai, sirf yahi file change hogi.

### ingest.py
`ingest_pdf(store, file_path)`:
- `load_pdf` — pypdf se pages extract (page number + text).
- `clean_text` — whitespace normalize, `-\n` hyphen breaks fix.
- `recursive_chunk` — chunks banao.
- metadata banao (doc_id, source, page, embedding_model) — citations/filtering ke liye.
- `store_chunks` mein daalo.
- `ingest_all()` — data/docs/ ke saare PDFs par same loop (CLI: `python ingest.py`).

### rag.py
`RAGEngine.answer(query)` — inference pipeline ka controller:
1. `store.search` — top-k chunks.
2. `build_context` — raw search result ko `[Source N: file page N]` labeled block
   mein convert karta hai; budget limit (max_chars).
3. Prompt = system template + context + query.
4. LLM call (low temperature, max tokens).
5. Returns `{answer, sources, distance_scores}` — answer + source metadata for UI.

### main.py
FastAPI app:
- `GET /health` — kaam ka system check + chunk count.
- `POST /ingest` — file upload -> data/docs -> index (File upload).
- `POST /query` — `QueryRequest` (pydantic) validate karta hai khaali query par 400.

## Data flow (query time)

```
POST /query {"query": "leave policy?"}
  -> VectorStore.search: embed query -> chroma.query -> top-5 (documents, distances)
  -> RAGEngine.build_context: labeled "[Source 1: leave_policy.pdf page 2]\n..."
  -> Prompt: system("answer ONLY from context, cite [Source N]") + context + question
  -> LLM (temperature 0.2) -> answer string
  -> {"answer": "...", "sources": [{"source": "...", "page": n}, ...]}
```

## Common first-run issues

| Symptom | Cause | Fix |
|---|---|---|
| `Missing API key` | .env nahi bana | `copy .env.example .env` + key daalo |
| `ModuleNotFoundError` | deps install nahi | `pip install -r requirements.txt` |
| Empty context / "not found" answers | PDF scanned hai (no text) | OCR-backed loader (Part 2/3) |
| Duplicate chunks | ingest baar baar chalaya | upsert re-run delete nahi karta; prod mein versioning (Part 3) |
| High token usage | top_k/chunk_size bade | `.env` mein `TOP_K`, `CHUNK_SIZE` tune karo |

## Exercises

1. Kisi query ke `distance_scores` JSON mein dekho — poor answers ke liye scores kaise hote hain?
2. Bina context ("I don't know" prompt) wali query — kya LLM ab bhi arti hai?
3. `TOP_K=1` aur `TOP_K=10` same query par compare karo.
4. Metadata mein `department` add karke `where` filter try karo.
5. Naya PDF `data/docs/` mein daal kar `ingest.py` re-run karo — answer meinsa bhi update hota hai?