# RAG Engineer Learning Guide — Part 1: RAG Fundamentals + Basic RAG

> **Trainer's Note:** Aap already jaante ho — Python, FastAPI, REST APIs, LLM APIs, prompts, caching, logging. Ye sab theek hai. Ab RAG wahi cheez hai jo aapne pehle banayi thi (FastAPI backend, DB query, response), bas usme ek **intelligent search layer** add hota hai jo LLM ko sahi context dekar accurate answers deta hai. Isn't that simple.
>
> Sabse pehle ek promise: **RAG = Vector DB + LLM ye nahi hai.** Ye ek *system* hai jisme har component ki apni responsibility hai. Har component ko alag se samjhenge, phir jodenge.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Why RAG is Needed](#2-why-rag-is-needed)
3. [LLM Knowledge vs External/Private Knowledge](#3-llm-knowledge-vs-externalprivate-knowledge)
4. [The Complete RAG Pipeline](#4-the-complete-rag-pipeline)
5. [Naive RAG](#5-naive-rag)
6. [RAG Architecture](#6-rag-architecture)
7. [Document Ingestion Pipeline](#7-document-ingestion-pipeline)
8. [Document Loading](#8-document-loading)
9. [Supported Document Types and Loaders](#9-supported-document-types-and-loaders)
10. [Text Cleaning / Preprocessing](#10-text-cleaning--preprocessing)
11. [Text Splitting](#11-text-splitting)
12. [Chunking and Why Chunking Matters](#12-chunking-and-why-chunking-matters)
13. [Fixed-Size Chunking](#13-fixed-size-chunking)
14. [Recursive Chunking](#14-recursive-chunking)
15. [Semantic Chunking](#15-semantic-chunking)
16. [Chunk Size and Overlap](#16-chunk-size-and-overlap)
17. [Embeddings](#17-embeddings)
18. [How Embedding Models Work (High Level)](#18-how-embedding-models-work-high-level)
19. [Similarity and Cosine Similarity](#19-similarity-and-cosine-similarity)
20. [Vector Databases](#20-vector-databases)
21. [ChromaDB](#21-chromadb)
22. [Storing Embeddings and Metadata](#22-storing-embeddings-and-metadata)
23. [Indexing](#23-indexing)
24. [Retrieval](#24-retrieval)
25. [Top-K Retrieval](#25-top-k-retrieval)
26. [Similarity Search](#26-similarity-search)
27. [Context Construction](#27-context-construction)
28. [Prompt Construction for RAG](#28-prompt-construction-for-rag)
29. [LLM Generation](#29-llm-generation)
30. [Complete End-to-End RAG Flow](#30-complete-end-to-end-rag-flow)

---

## 1. What is RAG?

### Concept
**RAG (Retrieval-Augmented Generation)** ek technique hai jisme LLM generate karne se *pehle* relevant documents ko kisi store se **retrieve** karta hai, aur phir us retrieved content ko as **context** use karke answer generate karta hai.

Formula (yaad rakho):
```
RAG = Retrieve + Augment + Generate

Retrieve: sahi content dhoondo (from your docs)
Augment:  us content ko prompt mein add karo
Generate: LLM se grounded answer banao
```

### Why?
Q: Pyar se technically — LLM sirf wahi jaanta hai jo training data mein tha (ya jo aap prompt mein diye). Kisi company ka private leave policy LLM ne kabhi dekha hi nahi. RAG us private content ko retrieval ke through LLM ke saamne rakh deta hai.

### How it works (mental model)
Ek FastAPI app from your own experience:

- **Bina RAG:** Aap user ka query lo, directly LLM API pe bhejo, answer lo. LLM apni training knowledge se answer karega.
- **RAG ke saath:** Aap query ko pehle apne "document search" mein daalo (jaise SQL query), top matches lo, unhe prompt mein paste karo, phir LLM se answer ki request karo.

### Example
Company ke paas 100 PDFs hain. User: *"What is the company's leave policy?"*
- Bina RAG: LLM bahana banayega ya galat answer dega.
- RAG: PDFs se relevant chunks retrieve ho gaye ("Earned leave = 24 days/year..."), LLM ne wahi context padh kar answer diya.

### Interview
**Q: RAG ko ek line mein define karo.**
**A:** RAG ek framework hai jo information retrieval aur LLM generation ko combine karta hai — pehle external documents se relevant chunks retrieve kare, unhe context ki tarah prompt mein inject kare, aur phir LLM se factually grounded answer generate karwaye.

---

## 2. Why RAG is Needed

### Concept
LLM ki teen fundamental problems hain jinhe retrieval solve karti hai:

| Problem | LLM ki default behaviour | RAG ka solution |
|---|---|---|
| **Fresh knowledge** | Training ke baad ka data nahi jaanta | Naye docs retrieve & inject |
| **Private knowledge** | Company/ikh-private data nahi dekha | Unhi docs se answer |
| **Grounding / hallucination** | Fakta ghar bana deta hai | Context se answer, source milta hai |

### Why?
Think about these three needs yourself. Ye hi RAG ki **business justification** hai. Companies RAG isliye lagati hain kyunki:
1. Model ko har baar retrain/fine-tune nahi karna (bahut costly).
2. Data centre mein har din badalta hai (new policy, new product docs).
3. Misinformation par trust nahi (support, legal, healthcare mein koi bhi hallucination galat hai).

### How it works (in contrast to Fine-tuning)
- **Fine-tuning:** Model ke weights change karo (costly, slow to update, specialists chahiye).
- **RAG:** Weights change nahi hote. Bas kaha se answer lena hai wo change hota hai. Naya document aaya → ingest karo → done.

### Example
Banks ke customer support mein daily koi bonus scheme change hota hai. Fine-tune har din nahi kar sakte. RAG mein scheme ka PDF upload kiya → us din se LLM us scheme ke baare mein sahi batata hai.

### Interview
**Q: Fine-tuning vs RAG — kab kya?**
**A:** RAG tab jab knowledge frequently change hoti hai / private hai / grounding chahiye. Fine-tuning tab jab aapko model ka **style, format, tone, domain terminology** change karni ho. Ye exclusive nahi hain — production mein dono often combine karte hain (fine-tuned model jo RAG context use kare).

### Practice
Apne ghar mein 5 docs (e.g., ek recipe, ek CV, koi notice) socho. Agar LLM se direct pucho to wo kis-kis cheez mein galat/sos betaega? Likhkar note karo. Ye RAG ki "why" ki duniya ka intuition hai.

---

## 3. LLM Knowledge vs External/Private Knowledge

### Concept
Two types of knowledge:

1. **Parametric knowledge** — model ke weights mein stored (training ke time learn hua). Fixed, stale.
2. **Non-parametric / contextual knowledge** — jo aap runtime par prompt mein dete ho (RAG context). Fresh, custom, verifiable.

### Why?
Ye distinction interview mein sabse pehle puchha jata hai. Difference samajhna matlab aap samajh gaye RAG kyun exist karta hai.

### How it works
```
LLM internals: weights (parametric) + input tokens (context)
                             ↓
Aapka context window = parametric memory + jo aapne abhi inject kiya
```

### Example
- Parametric: "Delhi India ki capital hai" — training mein seekha.
- External: "Aapki company ka leave policy" — ye sirf aapki PDF mein hai, LLM ke paas nahi. Isliye aapko wo PDF retrieve karke context mein dalni hogi.

### Interview
**Q: Parametric vs non-parametric knowledge?**
**A:** Parametric = knowledge baked in model weights during training (stale, general). Non-parametric = knowledge supplied at inference time in the prompt (fresh, specific, auditable). RAG non-parametric knowledge inject karta hai.

---

## 4. The Complete RAG Pipeline

### Concept
Pura RAG system do phases mein split hota hai:

```
 PHASE 1: INGESTION (offline, document grooming)
 Documents → Load → Clean → Chunk → Embed → Store in Vector DB

 PHASE 2: INFERENCE (online, query time)
 Query → Embed query → Retrieve top-k chunks → Construct context
       → Build prompt → LLM → Answer (+ sources)
```

### Why?
Dono phases ka goal alag hai:
- **Ingestion:** Data ko searchable form mein pre-process karna (ek baar hota hai, per document).
- **Inference:** User ke query ke liye sahi context ke saath answer dena (har query par hota hai).

Ye distinction yaad rakho — RAG "one-shot" nahi hai, ye **two-stage system** hai.

### How it works — flow diagram
```
[DOCUMENTS]
    │  (load)
    ▼
[TEXT] ──clean──► [CLEAN TEXT]
    │  (split)
    ▼
[CHUNKS] ──embed──► [EMBEDDINGS]
    │  (store)
    ▼
[VECTOR DB]   ◄────── (requirement: fast similarity search)

=================== QUERY TIME ===================

[USER QUERY]
    │  (embed)
    ▼
[QUERY VECTOR] ──similarity search──► [TOP-K CHUNKS]
    │                                       │
    │                    (context construction)
    │                                       │
    ▼                                       ▼
[PROMPT = system + context + query] ──► [LLM] ──► [ANSWER + SOURCES]
```

### Example
Company ka HR bot. Document upload hone ke baad ingestion phase chalta hai. Jab user sawal puchta hai, inference phase chalta hai. Dono independent processes hain.

### Interview
**Q: RAG ke major phases?**
**A:** Do — (1) Ingestion: load, clean, chunk, embed, index. (2) Inference/retrieval-augmented generation: embed query, retrieve top-k, construct context, generate. Retrieval quality *table* dono phases ke decisions (chunking, embeddings, retrieval params) se tilti hai.

### Practice
Ek sheet banao, upar pipeline ke har box ka naam likho. Har box ke aage likho: "Offline hai ya Online?" — ye exercise hi aapko architecture samjha degi.

---

## 5. Naive RAG

### Concept
**Naive RAG** = simplest RAG. Query embed karo → vector DB se top-k nikalo → context banao → LLM. Koi extra step nahi: no reranking, no query rewriting, no hybrid search.

### Why?
Naive RAG ko pehle understand karna isliye zaroori hai kyunki:
1. Ye "baseline" hota hai — comparison ka anchor.
2. Interview mein "Har advanced technique kis problem solve karti hai" — uska answer hamesha "jo naive RAG mein problem hai".
3. Production mein bhi bahut jagah naive RAG hi kaafi hota hai when data is clean.

### How it works — naive RAG steps (exactly)
1. Query aaya (string).
2. Query ko vector mein embed kiya (same model jo chunks ke liye).
3. Vector DB mein similarity search → top-k chunks.
4. Selected chunks ko ek context string mein join kiya.
5. Context + query ko prompt mein daala.
6. LLM se answer nikala.

### Example
Aapne 5 PDFs diye, user poochta hai "How many leave days per year?". Naive RAG top-3 chunks dega jo query se semantically close hain, context banayega, LLM answer dega.

### Interview
**Q: Naive RAG kya hai aur iski limitation?**
**A:** Naive/baseline RAG = simple single-shot retrieve-then-generate without any optimization. Limitations: keyword nuances miss karta hai, irrelevant chunks pick kar sakta hai, multiple queries/variations handle nahi karta, reordering ya filtering nahi karta. Advanced RAG (Part 2) inme se har ek problem ke liye aata hai.

---

## 6. RAG Architecture

### Concept
Architecture = component inventory + unke boundaries + data flow. RAG ke core components (har system mein):

| Component | Responsibility | Analogy (aap jaante ho) |
|---|---|---|
| **Loader / Parser** | Doc → raw text | File I/O adapter |
| **Preprocessor** | Clean text | Data validation layer |
| **Splitter** | Text → chunks | Business logic |
| **Embedding model** | Text → vector | Feature encoder |
| **Vector DB** | Storing/searching vectors | Database with indices |
| **Retriever** | Query → relevant chunks | SELECT ... WHERE semantic_match |
| **Prompt builder** | Context + query → LLM prompt | Request body builder |
| **LLM** | Prompt → answer | External API call |
| **Orchestrator** | Sab ko sequence mein chalao | FastAPI route handler |

### Why?
Architecture samajhna = boundaries samajhna. Har component swap-able hone chahiye (e.g., ChromaDB ko anusar Pinecone se replace karo, embedding model badlo) bina pipeline toot ke.

### How it works (core principle)
```
Definition: har component ek pure-ish transform hai:
  input ──┐      ┌── output (verifiable)
          │      │
DOCUMENT ─►LOAD      ► TEXT
TEXT     ─►CHUNK     ► CHUNKS
CHUNK    ─►EMBED     ► VECTOR
VECTOR+  ─►SEARCH    ► RANKED CHUNKS
QUERY
CHUNKS   ─►PROMISE   ► PROMPT
PROMPT   ─►LLM       ► ANSWER
```

### Interview
**Q: Production RAG system ke core building blocks?**
**A:** Ingestion pipeline (parser → cleaner → chunker → embedder → indexer), vector store + metadata database, retriever (top-k + optional hybrid/rerank), context formatter, prompt template, LLM, aur ek orchestration layer (FastAPI) plus observability stack (logging, tracing, eval).

### Practice
Ghar ke dusre learning module ki tarah, in components ka ek apna diagram draw karo. Har component ke baad "input type" aur "output type" likho. Data flow type-safe hona chahiye.

---

## 7. Document Ingestion Pipeline

### Concept
**Ingestion** = data ko apne vector database ke andar searchable form mein daalne ka complete flow. Built once, run on:
- document upload
- document update
- scheduled sync (new files)
- startup (existing corpus)

### Why?
Retrieval utna hi achha hai jitna ingestion. Agar chunking galat hai ya metadata missing hai — retriever kabhi bhi sahi chunks nahi dega. **Garbage in, garbage out** — isse ye samajho.

### How it works — ingestion steps
```
1. RECEIVE   — file mili (PDF/CSV/DOCX), basic check (size, type)
2. PARSE     — bytes → text (PyPDF, unstructured, pypdfium)
3. CLEAN     — remove unwanted headers/footers/links, normalize whitespace
4. CHUNK     — split text into logical pieces
5. ANNOTATE  — attach metadata (source filename, page, section, timestamp, doc_id)
6. EMBED     — each chunk → vector (same embedding model)
7. STORE     — vectors + original text + metadata in vector DB
8. INDEX     — ensure index built for fast search
```

### Example
Company ka HR bot. Koi new "Parental Leave Policy 2026.pdf" upload karta hai. Ingestion pipeline usko chunk kar ke vector DB mein daal deta hai. Usi second se RAG us policy ka jawab de sakta hai — bina model retrain kiye.

### Interview
**Q: Ingestion pipeline kab chalta hai aur isme kya hota hai?**
**A:** Offline/event-driven — document upload, update, ya scheduled sync par. Steps: parse → clean → chunk → metadata attach → embed → store/index. Quality decisions (chunk size, overlap, metadata schema) yahi fix hote hain aur retrieval quality ko directly dictate karte hain.

### Practice
Ek PDF liya, manually uske content ko 4 chunks mein tod kar likho aur har chunk ke liye metadata (filename, page) socho. Ab socho: user ka query "page 3 par kya hai?" — bina page metadata ke wo kaise answer dunga? → inference: metadata kyun zaroori.

---

## 8. Document Loading

### Concept
**Loading** = file bytes ko readable text mein convert karna. Ye ingestion ka first step hai.

```python
# Concept: raw bytes → string
raw_bytes: bytes      # file from disk/upload
text: str             # loaders ye karte hain
```

### Why?
LLMs aur embeddings text par chalti hain, files par nahi. Har file format apna structure — PDF mein tables/columns, DOCX mein paragraphs, HTML mein tags. Loader wo structure tod kar plain text banata hai.

### How it works — PDF ko example lo
PDF text extraction ke liye common tools:
- `PyPDF2` / `pypdf` — basic text extraction
- `pdfplumber` — tables better handle karta hai
- `pymupdf` (fitz) — fastest, page-level control
- `unstructured` — PDF/DOCX/HTML/PPTX/MD sab handle karta hai

**Unstructured split** (Excel/PPTX) — formatting ke liye extra parsers.

### Example
```python
import pypdf

def load_pdf(file_path: str) -> list[dict]:
    """Input: PDF path. Output: [{"page": 0, "text": "..."}, ...]"""
    reader = pypdf.PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": text})
    return pages
```
`pypdf` get_text() output kabhi-kabhi garbled/khaali aati hai (scanned PDFs). Later: OCR (tesseract, or Unstructured for complex layouts).

### Interview
**Q: PDF text extraction ke options?**
**A:** pypdf/PyPDF2 (light), pdfplumber (tables), PyMuPDF (speed), Unstructured (multi-format + OCR fallback). Choice data type par depend: scanned docs → OCR; complex tables → layout-aware parsing. Extraction errors RAG mein silent failures produce karte hain — validation zaroori.

---

## 9. Supported Document Types and Loaders

### Concept
Different formats → different loaders. Ye table basically interview + project me syllabus hai.

| Format | Loader (Python) | Catch |
|---|---|---|
| PDF | `pypdf`, `pdfplumber`, `PyMuPDF`, `unstructured` | scanned → OCR |
| DOCX | `python-docx` | tables handled manually |
| HTML | `BeautifulSoup` | strip tags |
| Markdown | built-in | keep headers (useful for chunking!) |
| CSV/Excel | `pandas` | row vs cell granularity |
| JSON | `json` + custom | keep keys as metadata |
| TXT | `read_text` | trivial |

### Why?
Format parsers ek data source triage problem hai. Prospective interview mein in-memory knowledge nahi, but choices par rationale chahiye.

### How it works — rule of thumb
- **Most important rule:** Metadata ko throw mat karo. Page numbers, headings, table headers, filename — ye retrieval filtering ke liye gold hai.

### Example — DOCX
```python
from docx import Document

def load_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)
```

### Interview
**Q: RAG mein filename/page/metadata kyun store karte hain?**
**A:** Metadata enables (1) source attribution/citations, (2) hybrid search filters (date, doc_type, department), (3) parent-child/sentence-window retrieval (Part 2). Barely text store karo, data samajh revenue pair jail.

### Practice
Ek `.docx` + ek `.html` download karo, dono ko text mein load karo aur apne code se print karo. Difference observe karo: html mein tags kyu `</p>` `\n` mein end ho rahe hain.

---

## 10. Text Cleaning / Preprocessing

### Concept
Raw text se noise remove karna before chunking. Garbage web/PDF text often aisa hota hai:
- repeated headers/footers (har page ke repeated)
- page numbers, URLs, email signatures
- extra whitespace / malformed unicode (`\u2019` isle `'`)
- HTML entities, watermark text

### Why?
Noise se do problem:
1. Embedding quality — noise words vector ko pollute karte hain.
2. Chunk boundaries glitch — footer text chunk ke beech aake semantic unity todta hai.
3. LLM context — waste tokens, answer mein "Page 5" jaise artifacts.

### How it works — small but careful steps
```python
import re

def clean_text(text: str) -> str:
    # 1. normalize unicode (e.g., curly quotes)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # 2. collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 3. collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # 4. strip page-number-like artifacts
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    return text.strip()
```
**Warning:** Aggressive cleaning dangerous hai. Emails, amounts, product codes delete mat karo. Cleaning context-aware honi chahiye.

### Interview
**Q: RAG mein text cleaning why important? Cleaning over-do to masla hai — how do you balance?**
**A:** Clean structural noise (repeated footers, double whitespace, OCR artifacts) but preserve semantic content (numbers, codes, names). Over-cleaning embeds mein information loss karta hai aur chunk integrity todta hai. Balance corpus distribution normal hai.

---

## 11. Text Splitting

### Concept
Lamba text (file level) ko chhote, searchable, self-contained pieces mein todna = **splitting**. Osa pieces hi "chunks" hain.

### Why?
- Embedding models normal mein ~512-8192 token axis hota hai (input limit).
- LLM context window limited.
- Retrieval needs **granularity** — aap pura 500-page policy search nahi karte, aap specific relevant section retrieve karte ho.

### How it works — before/after
```
BEFORE: "The company provides earned leave of 24 days per year. In addition,
          sick leave of 12 days per year applies from the first day of joining.
          Employees may carry forward up to 10 days..."

AFTER (split on sentences/paragraphs):
  chunk1: "The company provides earned leave of 24 days per year."
  chunk2: "In addition, sick leave of 12 days per year applies..."
  chunk3: "Employees may carry forward up to 10 days..."
```

### Example
User: "sick leave kitni hai" → chunk2 retrieve hoga, warna pura doc embed hota to bhi answer galat granularity ka hota.

### Interview
**Q: Splitting vs chunking — same?**
**A:** Haan, "text splitting" aur "chunking" same process ke naam hain — long text ko smaller searchable units mein todna. "Chunk size", "overlap" jaise parameters isi configuration se aate hain.

---

## 12. Chunking and Why Chunking Matters

### Concept
Chunking = **the single most impactful parameter** in ingestion. Sahi chunk = self-contained, focused, retrievable, verifiable unit. Galat chunk = garbage retrieval.

### Why?
Chunking quality directly controls:
1. **Retrieval precision** — sahi detail milti hai ya nahi.
2. **Retrieval recall** — sahi info chunk mein hai ya ke baad gep main khoya.
3. **Context waste** — too-big chunks → irrelevant detail LLM context mein (cost + noise).
4. **Answer quality** — chunk jitna self-contained, LLM utna better inference.

### How it works — the chunking trade-off
```
TOO SMALL  → chunk lacks context, loses meaning, retrieval fails precision
TOO LARGE  → dilutes relevance, waste tokens, top-k deletes detail

SWEET SPOT → self-contained unit of thought (1-2 paragraphs, ~200-500 tokens typical)
```

### Example
Policy doc: "Leave Policy" chapter ka paragraph pura chunk ho (self contained). Agar 5-page section ek chunk → query "joining ke baad sick leave kab eligible?" → chunk ka relevant line kisi unrelated line ke saath enfold ho jaata hai, retrieval top-k mein hi miss ho jata hai.

### Interview
**Q: Chunk size selection kaise karthe?**
**A:** Embedding model ke context limit ke andar, retrieval unit ko "one coherent idea" banao. Start ~350-500 tokens, evaluation (Part 3) par tune karo. Chunking ka tough rule: content type se select — tables vs chat logs vs policies.

### Practice
Apne favourite text (article/report) ko 3 chunking strategies (fixed-500, recursive, sentence) se tod kar compare karo: konsa chunk apne aap mein readable, self-contained hai.

---

## 13. Fixed-Size Chunking

### Concept
Fixed-size chunking = text ko exact token/character count ke basis par kato, i.e., "har 200 tokens ka chunk banao".

```python
def fixed_chunk(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    tokens = text.split()  # simplified; real embedding tokenizers byte-pair hote hain
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i+size]))
        i += size - overlap  # overlap prevents info loss at boundaries
    return chunks
```

### Why?
- Simple, deterministic, cheap, compute-time negligible.
- Good quick baseline to benchmark retrieval against.

### How it works
`size` token window ko slide karo, `overlap` window ke beech rkhne wale prev chunk ka tail hai.

### Limitations (important!)
- Boundaries text ki semantics ignore karte — sentence/module ke beech mein seedha kaat sakte hain. "The company's policy on sick leave is **generous — but** the cap applies..." → do chunks mein toot jata hai.

### Interview
**Q: Fixed-size chunking downside?**
**A:** Deterministic but semantics-agnostic — sentence/thought ke beech mein kat sakta hai, deliberately contextual retrieval hurt karta hai. Baseline ke liye okay, production quality ke liye recursive/semantic better.

---

## 14. Recursive Chunking

### Concept
Recursive character/text splitter `langchain` sabse popular chunker: text ko **natural boundaries** (`\n\n`, `\n`, `.`, space) par recursively split karta hai, har divider ke saath lenfit to target chunk size.

### Why?
Text structure (paragraphs, sentences) reflect meaning. Paragraph boundary par chunking = smarter than fixed-size kyunki chunk apni logical unit ban jaata hai.

### How it works
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""],  # priority order
)
chunks = splitter.split_text(long_text)
```
**Mechanism:** Ya to `chunk_size` hit limit → split boundary par tod do. Yam najdik separator dhundo and split. Recursion = order mein try karo, smallest tak fallback.

### Example
PDF paragraph `\n\n` par alg; paragraphs bade hain to sentences `. ` par; abhi bhi lamba to spaces par.

### Interview
**Q: Recursive chunking kya hai?**
**A:** Natural text separators (paragraph → sentence → word) ko priority order mein recursively apply karta hai taaki chunk boundaries semantic units (mostly) coincide karein while keeping approximate size. Fixed-size vs recursive — recursive practical winner hai most text data ke liye.

---

## 15. Semantic Chunking

### Concept
**Semantic chunking** = text ko meaning/topic ke baandh par split kara (statistics/embedding based), token count par nahi. Idea: text position after which meaning naturally changes → break boundary.

### Why?
Fixed/recursive size-based chunking mein semantic break insert ho sakta hai. Semantic chunker "conversation topic shift" detect karta hai.

### How it works — high level (LLM + embedding based)
1. Sentences ko embed karo.
2. Adjacent sentence embeddings ki cosine similarity measure karo.
3. Similarity gap = semantic boundary → split kar do.
4. Ya LLM se "paragraphs/points" batwao (Langech aise options).

(LlamaIndex `SemanticSplitterNodeParser`, LangChain langchain_experimental semantic chunkers isi idea pe built hain.)

### Limitations
- Slower/costlier (every sentence embed).
- Pagano do intermediate: chunk size variable → some chunks too large/small.

### Interview
**Q: Semantic vs recursive chunking?**
**A:** Recursive size-based with structural boundaries; semantic content-based grouping. Semantic better-rich topical data, but slower + costlier + variance. Production often: recursive as default, semantic where data is long-form prosaic/technical.

---

## 16. Chunk Size and Overlap

### Concept
Two parameters that define partitioning:
- **chunk_size**: tokens/characters per chunk (typical 200-1000 tokens; common 350-500).
- **overlap**: adjacent chunks shared tokens (typical 10-20%).

### Why?
Overlap na rakho to context join points (section boundary sentences, tables dividing) chunk ke beech split hokar info khote hain.

### How it works — overlap demo
```
chunk1: [A B C D E]
chunk2:        [E F G H I]   ← E shared (overlap window)
```
Query "E" se related ho to dono chunks retrieve ho sakte hain → coverage upar.

### Trade-offs (major table NOTE)
| Parameter | Too small | Too large |
|---|---|---|
| chunk_size | no context, poor similarity | cost, noise, diluted relevance |
| overlap | info gaps at boundaries | duplicate vectors, more cost, redundant |
| top-k | may miss answer | context overload, cost |

### Example
Company policy where "eligibility" and "cap" do paragraphs adjoining hain — overlap ki wajah se dono ka key sentence kisi ek chunk me available.

### Interview
**Q: chunk_size/chunk_overlap/top-k ka trade-off batao.**
**A:** Chunk_size balances embedding context vs granularity; overlap prevents boundary loss but adds redundancy/cost; top-k balances recall vs context budget. These must be jointly tuned via eval, never separately.

---

## 17. Embeddings

### Concept
**Embedding** = text → list of numbers (vector) jisme semantics encoded ho. Similar text → similar vectors (nearby in vector space).

```
"dog"     → [0.02, -0.31, 0.87, ...]
"puppy"   → [0.03, -0.29, 0.85, ...]  (dog se close)
"bicycle" → [0.91, 0.44, -0.12, ...]  (door rakha)
```

### Why?
LLM/computer text ko samjhta nahi. Embeddings text ko numeric space mein map karte hain → similarity or computed numerically (cosine). Niche ka pura RAG retrieval isi "text ki numeric proximity" par depend karta hai.

### Why "same model" if important?
Query aur document dono ko **same embedding model** se embed karo (same model → same vector space). Alag model → incommensurable vectors.

### How it works
- Input: text (chunk ya query).
- Output: fixed-dimension vector (OpenAI text-embedding-3-small = 1536 dims, BGE = 1024, MiniLM = 384).
- Aise model bhi exist hote hain — `text-embedding-3-large`, `bge-large`, `e5` etc.
- Dim larger → more expressive, but more storage & compute.

### Example (OpenAI)
```python
from openai import OpenAI
client = OpenAI()

def embed(text: str) -> list[float]:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return resp.data[0].embedding   # → [1536 floats]
```

### Interview
**Q: Embedding jin queries aur documents ko same model why?**
**A:** Both embed to same latent space, math (similarity) valid hota hai. Cross-model embeddings meaningless similarity dete hain. Also store embedding *model + version* in metadata for reproducibility.

### Practice
"python", "java", "snake", "banana" ko embed karo aur cosine similarities print karo. Observe: "python" kise closer hai — "java" (programming) ya "snake" (animal)? Ye word-sense ambiguity RAG mein realistic problem hai.

---

## 18. How Embedding Models Work (High Level)

### Concept
Embedding model = special transformer trained specifically to place similar-meaning text nearby in vector space. Training signal: **contrastive learning** (pairs: similar = close, dissimilar = apart).

### Why? (mental model)
Aapko detail mein mat jaana (Part 1 ka point), but basic mechanism samajhna zaroori: taaki aap ko pata ho *kya* embed vectors encode karte hain.

### How it works — stages
1. **Tokenizer:** text → tokens (subwords).
2. **Transformer encoder:** tokens pass through attention layers; each layer words ke context mix home (jisse word's meaning kontext-dependent hota hai).
3. **Pooling:** final hidden states ko pool (mean / CLS token) → single fixed-size vector.
4. **Normalize:** vectors often L2-normalized (unit length) — cosine becomes dot product.

Key insight: **same word, different contexts → different vectors** (contextual embeddings). "bank" in "river bank" vs "investment bank."

### Interview
**Q: A model trained on pairs (query, relevant doc) vs general embedding model — difference?**
**A:** General embedding models (BGE, E5, OpenAI) trained on massive diverse pairs (dataset mixture). Domain-specific fine-tuning (e.g., legal/medical) makes model's similarity space tuned to that domain → better retrieval in niche corpora.

---

## 19. Similarity and Cosine Similarity

### Concept
**Cosine similarity** = angle between two vectors ka cosine. Range: -1 (opposite) to +1 (same direction). For embeddings typically 0 to 1.

```
cosine_similarity(A, B) = (A · B) / (||A|| · ||B||)
```
- Exactly analogous to: how aligned two directions are (magnitude ignored).

### Why "cosine" and not "Euclidean distance"?
- Embeddings magnitude-type irrelevant (longer text naturally larger magnitude), angle reflects semantic direction.
- Cosine is bounded → comparable ranking.
- With normalized embeddings, cosine ≡ dot product (faster).

### Example
```python
import numpy as np

def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
```

### Interview
**Q: Dot product vs cosine similarity — kab choose?**
**A:** Normalized embeddings (common) par primarily same. Cosine: scale-invariant, direction-only — default for semantic similarity. Dot product: if embedding magnitude carries info (e.g., frequency), or as optimization after L2 norm.

---

## 20. Vector Databases

### Concept
**Vector DB** = special database designed for storing vectors + fast similarity search (ANN — Approximate Nearest Neighbor). Examples: ChromaDB, Qdrant, Pinecone, Weaviate, Milvus, FAISS (library not full DB).

### Why? — "Why DB and not just array?"
1. **Speed:** Brute-force comparing millions of embeddings too slow. Vector DB uses ANN index (like HNSW, IVF) → milliseconds.
2. **Scalability:** 100M+ vectors, distributed.
3. **Metadata + filters** alongside.
4. **Upsert, delete, persistent, concurrency-safe.**
5. **CRUD on documents** (aapka doc delete ho to vector bhi delete).

### How it works — ANN intuition (HNSW)
HNSW (Hierarchical Navigable Small World) = multi-layer graph. Top layer sparse connections (fast far navigation), lower layer dense (precise neighbors). Query = start top, greedily traverse down → find approximate nearest neighbors very fast. Trade-off: **approximate** (recall slightly <100%) vs exact search.

### Interview
**Q: Vector DB vs plain numpy array similarity?**
**A:** Arrays naive O(N×D) scan lagta hai — 1M vectors impractical. Vector DB with ANN index (HNSW/IVF) sub-linear-ish search + persistence + filtering + concurrency. For tiny demos array okay; production → dedicated DB.

---

## 21. ChromaDB

### Concept
ChromaDB = open-source, embedded (in-process, like SQLite) vector database. Python-first, no server needed. Ideal for learning & prototyping; scales to moderate workloads (prod possible with care).

### Why choose for learning?
- Zero infra: `pip install chromadb`, `client = chromadb.PersistentClient(path=...)`.
- Document/metadata embedding API built-in.
- Simple: collection = table.

### How it works — core API
```python
import chromadb

client = chromadb.PersistentClient(path="./data/chroma")   # persists to disk
collection = client.get_or_create_collection(
    name="company_docs",
    metadata={"hnsw:space": "cosine"},   # distance metric
)

# add chunks
collection.add(
    ids=["chunk_001"],               # unique id
    documents=["The company offers 24 days of earned leave per year."],
    metadatas=[{"source": "leave_policy.pdf", "page": 2}],
    embeddings=None,                 # or pass explicit embeddings
)
```
If `embeddings=None` + `documents` given → Chroma needs a default embedding function. For explicit control prefer passing your own `embeddings`.

### Interview
**Q: ChromaDB vs Pinecone/Qdrant?**
**A:** Chroma: embedded, simple, zero-ops — great for dev/prototype/small prod. Qdrant: server, features (filters, quantization, replication) — scaler for prod. Pinecone: managed cloud — ease but cost/vendor lock. Choice: control + scale aspect.

### Practice
ChromaDB install karo, ek collection banao, 3 documents add karo, query karo. Observe `ids`, `distances`, `metadatas`, `documents` output — ye aapke retrieval ka raw material hai.

---

## 22. Storing Embeddings and Metadata

### Concept
Each record in collection = vector + text + metadata:
```
{ "id": "c-001",
  "embedding": [0.1, -0.2, ...],
  "document":  "Earned leave... 24 days...",
  "metadata":  {"source": "leave_policy.pdf", "page": 2, "section": "Earned Leave", "doc_type": "HR policy"} }
```

### Why metadata?
- **Filtering** retrieval (date range, source).
- **Citations** ("Source: leave_policy.pdf, page 2").
- **Parent-child** (Part 2) mapping.
- **Cleanup** (delete all chunks of doc X).

### How it works — good metadata schema
```
BIDEN: doc_id (unique doc), source filename, page, section/header,
created_at (ingest time), chunk_index, embedding_model+version
```

### Example
```python
collection.add(
    ids=[f"{doc_id}-{i}"],
    documents=[chunks],
    metadatas=[{"doc_id": doc_id, "source": "leave_policy.pdf", "page": p, "chunk_index": i,
                "embedding_model": "text-embedding-3-small", "ingested_at": "2026-09-20"}],
)
```

### Interview
**Q: Retrieval metadata filters kyun use karte? example?**
**A:** Reduces search space, improves precision, enables citations/auditing. Example: Bol koi pura policy ka bhi ye hai, user sirf "2026 india policy" se — filter `year=2026` + `region=india` → only matching chunks.

---

## 23. Indexing

### Concept
**Index** = data structure that makes search fast. Vector DB creation time par index built hota hai; vectors insert hote hi index maintains hota hai. In Chroma, default index = HNSW (with configurable `hnsw:M` (connections) and `hnsw:ef_construction`).

### Why?
Bina index: exact scan → O(N×D). Index → search in milliseconds even on millions of vectors.

### How it works — mental model of HNSW
- Multiple linked "layers" of graphs.
- Layer 0: every item + nearest neighbors (dense).
- Higher layers: fewer items (sparse shortcuts).
- Query: enter at highest layer, descend greedily → distance shrinks → exit at layer 0 with approximate nearest neighbors.

### "Approximate" and recall
ANN parameter trade-offs: `ef_search` (beam width at query) — higher recall but slower. `M` (max edges/node) — higher index quality but memory. So "recall" quality depends on index config + data dimensionality.

### Interview
**Q: ANN ka matlab kya hota hai? recall trade-off?**
**A:** Approximate Nearest Neighbor — exact search nai but "good enough near" — sublinear cost. Trade-off precision vs speed/memory. Recall evaluation on dataset required (Part 3).

---

## 24. Retrieval

### Concept
**Retrieval** = user query ke sabase relevant chunks vector DB se lana. Ye RAG ka "knowledge finding" phase — quality ka 80% yahan decide hota hai.

### Why?
LLM wahi answer de sakta hai jo context mein diya ho. Agar retrieval galat chunks de — LLM ke paas sahi info nahi → hallucination ya "context nahi mila". **Better retrieval = better answer** — this is the core RAG principle.

### How it works
```
Query (string)
  → embed → query_vector
  → vector DB ANN search
  → top-k most similar (by cosine)
  → retrieve chunks + metadata (+ distances)
```
Distances roll me: low distance = high relevance. These distances help you judge confidence later.

### Example
Query "sick leave kitne days?" → query vector → search → chunks sabase aligned → those chunks included in context. Take note: search is in **embedding space**, instruction-following bhi yahi hai.

### Interview
**Q: Retrieval quality RAG quality par kya impact?**
**A:** Retrieval is the highest-leverage stage — if irrelevant chunks retrieved, even best LLM generates hallucinated/stock answer. Most advanced RAG techniques (Part 2) improve retrieval first.

---

## 25. Top-K Retrieval

### Concept
`top_k` = kitne most-similar items retrieve karne hain. Typical 3-10.

### Why?
- Too low (k=1): exact info missed.
- Too high: context overwhelmed (cost, noise, LLM confused by unrelated chunks).

### How it works
```
results = collection.query(query_embeddings=[qv], n_results=3)
# → top 3 chunk texts + metadata + distances
```

### Trade-off framework
```
top_k ↑ → recall ↑ (miss nahi karega) but precision ↓ + tokens ↑ + cost ↑
top_k ↓ → precision ↑ + cheap but risk: sahi answer miss ho
```
Optimal k depends on chunk size + context budget: loosely `K = context_window / avg_chunk_tokens`.

### Example
Chunk ~300 tokens, context budget ~3000 → top_k ≈ 8-10.

### Interview
**Q: top_k tune kaise karo?**
**A:** Budget reasoning (context window, chunk size) + eval-driven tuning (find k where recall saturates before extra noise). Individual docs ke experiments par depend mat karo — evaluate on realistic query set.

---

## 26. Similarity Search

### Concept
Similarity search in vector space = compare query vector with chunk vectors → rank by distance/similarity → return best.

### How it works
```
query vector (1536 dims)
         │ dot/cosine with every candidate (via ANN index)
         ▼
sorted candidates by distance → top-k
```

### Important: "Similar ≠ Relevant"
Embedding similarity finds **semantic closeness**, not shared keywords. Har context me relevance = topic match + sufficient detail + negation handling etc. Failure modes:
- Synonym nuance (Part 2 query expansion).
- Long-winded chunks dilute similarity.
- Multi-topic docs.

### Example code (ChromaDB)
```python
res = collection.query(
    query_embeddings=[embed("leave policy sick days")],
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
for dist, meta, doc in zip(res["distances"][0], res["metadatas"][0], res["documents"][0]):
    print(f"{dist:.3f} | {meta['source']} | {doc[:80]}")
```

### Interview
**Q: Similarity search limitations — gorge anyway bulljati baat?**
**A:** Vector similarity captures overall semantic direction — exact-match, negation, multi-hop reasoning missed ho sakte hain. Naive RAG's main bottlenecks → Part 2's techniques (hybrid search BM25, rerankers, multi-query) target inhi problems ko.

---

## 27. Context Construction

### Concept
Retrieved chunks ko ek **formatted context block** mein assemble karna jo prompt mein jaayega.

### Why?
- LLM ke liye structured, deduped, ranked context helps.
- Source tracking for citations.
- Control token budget.

### How it works
```python
def build_context(results: dict, limit_chars: int = 3000) -> tuple[str, list[dict]]:
    parts, sources = [], []
    used = 0
    for dist, meta, doc in zip(results["distances"][0], results["metadatas"][0], results["documents"][0]):
        if used + len(doc) > limit_chars:
            break
        parts.append(f"[Source {len(parts)+1}: {meta.get('source','?')} page {meta.get('page','?')}]\n{doc}")
        sources.append(meta)
        used += len(doc)
    return "\n\n".join(parts), sources
```
Duplicate chunks dedupe (overlap) idea. Order by relevance.

### Interview
**Q: Context construction best practice?**
**A:** Deduplicate overlapping chunks, rank by distance, label sources, truncate to budget, keep structure consistent with prompt template. Context budget management = cost + quality control.

---

## 28. Prompt Construction for RAG

### Concept
RAG prompt = system instructions + context + query saath mein template. Ye "Augment" step hai.

### Why?
Prompt ki structure defines LLM ki behaviours: kya use karna hai (context only), kya karna when unknown ("say I don't know"), citation format, tone, length.

### Example — production-style template
```python
SYSTEM = """You are a helpful assistant for {company}.
Answer ONLY using the provided context.
If the context does not contain the answer, say: "I don't have this information in the documents."
Always cite sources in format [Source N].
Be concise and factual. Provide enough detail but cap at {max_length} words."""

def build_prompt(query: str, context: str) -> str:
    return f"""{SYSTEM.format(company="Acme Corp", max_length=200)}

Context:
{context}

Question:
{query}

Answer:"""
```

### Why these instructions matter
- "Use only context" → hallucination reduction.
- Fallback behavior → no made-up answers.
- Citation format → trust + verifiability.

### Interview
**Q: RAG prompt ke core elements?**
**A:** System instructions (scope, persona, constraints, citation format, "I don't know" fallback), context block (retrieved chunks with source labels), user query. Design trade-off: stricter = less hallucination but some over-refusal; evaluate balanced via tests.

---

## 29. LLM Generation

### Concept
Final step: prompt → LLM API → answer. RAG ke generation phase mein LLM "reader/aggregator" role karta hai: context se relevant info extract karke answer bulata hai.

```python
from openai import OpenAI
client = OpenAI()

def generate(prompt: str, temperature: float = 0.2) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,   # low → factual, deterministic
    )
    return resp.choices[0].message.content
```

### Why temperature low?
RAG answers factual/grounded hone chahiye → low temperature (0-0.3). High temperature → creativity → hallucination push.

### Generation often splits into:
1. **Pre-LLM checks** (context has relevant facts? below threshold → skip LLM, respond "not found").
2. **LLM call**.
3. **Post-LLM validation** (is answer grounded in context? citations valid?) — Part 3 detail.

### Interview
**Q: RAG generation LLM param choices?**
**A:** temperature 0-0.3 for grounded factual answers; max_tokens bounded; determinism for testability in eval vs variability trade-off. Also consider max output length to control cost/latency.

---

## 30. Complete End-to-End RAG Flow

Putting everything together with the company example:

**Scenario:** Acme Corp ke paas 100 PDFs. User: *"What is the company's leave policy?"*

### Ingestion (offline/one-time)
```
100 PDFs
  → pypdf   → page texts
  → clean   → normalized text
  → split   → ~2,500 chunks (500 tokens, 100 overlap)
  → embed   → vector each (text-embedding-3-small, 1536 dims)
  → add     → ChromaDB collection "company_docs"
```

### Inference (per query)
```
1. Query: "What is the company's leave policy?"
2. Embed query           → query_vector
3. Similarity search     → top-5 chunks (distances included)
4. Build context         → labeled blocks + source metadata
5. Build prompt          → system + context + query
6. LLM call              → temperature 0.2
7. Response              → grounded answer + [Source 1: leave_policy.pdf p.2]
```

### Code — the full naive RAG, one flow
```python
import chromadb
import pypdf
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="./data/chroma")
coll = chroma.get_collection("company_docs")

def answer(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    qv = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
    res = coll.query(query_embeddings=[qv], n_results=top_k,
                     include=["documents", "metadatas", "distances"])
    ctx = "\n\n".join(
        f"[Source {i}: {m.get('source')} p.{m.get('page')}]\n{d}"
        for i, (d, m) in enumerate(zip(res["documents"][0], res["metadatas"][0]), 1)
    )
    prompt = f"""You are a helpful assistant for Acme Corp.
Answer ONLY using the provided context. If the context lacks the answer, say so
and mention it is not in the documents. Cite like [Source N].

Context:
{ctx}

Question:
{query}

Answer:"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content, res["metadatas"][0]

answer_text, sources = answer("What is the company's leave policy?")
print(answer_text, "\nSources:", sources)
```

### Interview
**Q: End-to-end RAG flow bolo (classes).**
**A:** Two pipelines. Ingestion: documents → parse → clean → chunk → embed → index. Inference: query → embed → top-k ANN → context assembly → prompt (system+context+query) → LLM → validated grounded answer with citations. Identify: retrieval quality + generation grounding are the two main levers.

---

## Common Beginner Mistakes

1. **Query aur chunks ko alag embedding model se embed karna** — vector spaces mismatch → garbage similarity.
2. **Chunk size sochna hi nahi** — pura document ek embedding → retrieval granularity zero.
3. **Overlap ignoring** — sentence/phrase boundary par info loss.
4. **Metadata/freshness not stored** — citations aur filtering impossible.
5. **top_k arbitrarily set** (k=1 ya k=100) — either miss or context spam.
6. **Same-prompt context order mixed** — LLM ko ranking ki info nahi, confusion.
7. **Temperature high** — hallucinated style answers.
8. **"I don't know" fallback absent** — LLM confidently fabricates.
9. **Clean text not validated before embedding** — OCR garbage embedded forever.
10. **No distance/score inspection** — irrelevant chunk pickup silently accepted.
11. **Ingesting on every run** — duplicate vectors, stale versions, cost.
12. **Wrong distance metric** — Chroma default 'cosine' good, but must confirm.

## Debugging Checklist

- [ ] Query embed = same model/dim as chunk embed?
- [ ] Collections ki `distances` print karke dekh rahe ho — scores sane?
- [ ] Retrieved chunks **semantically** relevant? (Human eyeball)
- [ ] Chunk size/app — self-contained, boundaries clean?
- [ ] Metadata filters jo expected hai wo chunks pakad rahe?
- [ ] Context boundary: context+query < LLM window (token count)?
- [ ] If answer misses fact — is fact present in indexed docs at all?
- [ ] Re-index after doc update? Duplicate chunks check.
- [ ] Ollama/API errors surfaced properly (logging)?
- [ ] Async issues — collection writes committed before query?

---

## 25 Interview Questions — Part 1 (with answers)

**1. RAG kya hai?** A: Retrieval-Augmented Generation — external docs se relevant chunks retrieve karke context inject karta hai, phir LLM grounded answer generate karta hai.

**2. RAG kyun needed, fine-tuning nahi?** A: Fresh/private data + grounding. RAG: no re-training, update instant, verifiable sources. Fine-tune: style/format, expensive.

**3. Parametric vs non-parametric knowledge?** A: Parametric in weights (training), non-parametric supplied at inference (prompt context). RAG works with non-parametric.

**4. RAG core phases?** A: Ingestion (load→clean→chunk→embed→index) + Inference (embed query→retrieve→context→prompt→generate).

**5. Naive vs advanced RAG?** A: Naive = single retrieve-then-generate; advanced adds query rewriting, hybrid search, rerank, self/corrective loops.

**6. Vector DB kya problem solve karta hai?** A: Fast approximate (ANN) similarity search over millions of vectors + filtering + persistence + concurrency.

**7. Chunking kya role?** A: Retrieval granularity. Poor chunking → poor retrieval → poor answers. Most important ingestion parameter.

**8. Chunk size/overlap trade-off?** A: Size: small=high precision/low context, large=recall up/cost+noise. Overlap: boundary info loss prevention (cost +).

**9. Fixed vs recursive vs semantic chunking?** A: fixed=numeric deterministic; recursive=natural boundaries (paragraphs/sentences); semantic=meaning-based split — each more complex/costly.

**10. Cosine similarity kya?** A: Cosine of angle between vectors, scale invariant, [-1,1], high = semantically similar. With normalized embeddings = dot product.

**11. Embeddings same model why?** A: Same vector space. Cross-model vectors incomparable — cosine meaningless.

**12. ANN vs exact search?** A: Exact = O(N·D) expensive; ANN (HNSW) = approximate, sublinear-ish, acceptable recall trade-off.

**13. HNSW mental model?** A: Multi-layer graph; sparse top layers for fast navigation, dense bottom for precision; descend layers.

**14. top_k set kaise?** A: Context budget (chunk size × k ≤ LLM window) + eval-driven: find recall saturation vs noise threshold.

**15. Retrieval quality aur answer quality ka relation?** A: Direct — LLM ka answer sirf context ke saath grounded. Poor retrieval = hallucination/missed info regardless of LLM strength.

**16. Metadata store kyun?** A: Filtering, grounding/citations, parent-child retrieval, cleanup.

**17. ChromaDB vs Pinecone?** A: Chroma embedded/simple/OSS; Pinecone managed/cloud/scalable. Dev vs prod scale-vendored choice.

**18. Embedding model compatible dims why?** A: Vector dimension consistent, cosine math valid; also dedupe storage considerations.

**19. Similarity ≠ relevance difference?** A: Semantic closeness vs true task relevance. Negation, multi-doc weaving, keyword misses break naive similarity.

**20. "context sufficient nahi" detect kaise?** A: Distance threshold, chunk count, LLM judgement prompt, answer-quality metrics (faithfulness) — Part 3 detailed.

**21. Context token budgeting?** A: System+context+query ≤ window; keep system short, context capped; monitor.

**22. Temperature in RAG?** A: 0-0.3 for grounded deterministic factual answers; avoid randomness in eval.

**23. Document update/delete cycle RAG mein?** A: Versioned ingestion; old doc chunks delete/upsert by doc_id; schemas for traceability.

**24. Two pipelines demo?** A: Explain ingestion vs inference with company examples (one-time indexing, per-query retrieval).

**25. "RAG = vector DB + LLM" — ye enough hai?** A: Nahin — system hai: retrieval quality, context/prompt design, eval, monitoring all critical. Vector+LLM is baseline only.

---

## Practice Exercises (karo khud, answers apne code se)

1. **Mini corpus:** 3-5 text files (policies/recipes) ingest karo, 5 realistic queries kar ke top-3 chunks print karo with distances. Krishna ko samjho: kya sab queries ke liye sahi chunks mile?
2. **Chunk experiment:** Ek hi document ko fixed-200/500/overlap-variants se chunk karo → same query ka retrieval compare karo. Kya boundaries affect retrieval?
3. **Cosine manual:** Embeddings lo, cosine manually calculate karo, Chroma ke distances se verify karo.
4. **Metadata filters:** Collection mein `department`, `year` metadata daalo, `where={"year": 2026}` se query karo — filter ka effect observe karo.
5. **"I don't know" test:** Apne corpus se bahar ki query lo (e.g., "Kalkulator mein..." irrelevant). Prompt fallback sahi trigger hua? Distance kaisa tha?
6. **Top-k sweep:** k = 1,3,5,10 same query pe → answer quality/latency subtleties note karo.
7. **Debug checklist:** Intentional bug fix karo (different embedding models for query vs doc) → retrieval distance pattern note karo.

---

> **Ye sab concepts teor hain.** Next step: `04_basic_rag_app.md` mein complete working FastAPI + ChromaDB app bana hua hai — har function explain kiya hai, aap usse apne machine mein chala ke dekh sakte ho.

**PART 1 COMPLETE — Say \`continue\` when you are ready for PART 2.**