# =============================================================================
# FILE: 05_real_world_mini_project.py
# TOPIC: Mini Project — AI-Powered Document Q&A with FastAPI
#
# This combines everything from Module 1 into a real application.
# You'll recognize this pattern in every RAG system you ever build.
#
# What this builds:
#   - A FastAPI server with two endpoints
#   - POST /upload  → accepts text, creates embeddings, stores in memory
#   - POST /ask     → takes a question, finds relevant text, answers with LLM
#
# This is a SIMPLIFIED RAG. Module 3 covers the full production version.
#
# pip install openai fastapi uvicorn numpy python-dotenv pydantic
#
# Run: uvicorn 05_real_world_mini_project:app --reload
# =============================================================================

import os
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Mini Document Q&A", version="1.0.0")


# =============================================================================
# DATA MODELS (Pydantic — you know this from FastAPI)
# =============================================================================


class DocumentInput(BaseModel):
    title: str
    content: str


class QuestionInput(BaseModel):
    question: str
    stream: bool = False  # whether to stream the response
    top_k: int = 3  # how many document chunks to retrieve


class DocumentChunk(BaseModel):
    title: str
    content: str
    chunk_index: int


# =============================================================================
# IN-MEMORY VECTOR STORE
# In production: replace this with ChromaDB, Pinecone, or pgvector
# The interface stays the same — just the storage backend changes
# =============================================================================


class InMemoryVectorStore:
    """
    A dead-simple vector store.
    Stores (text, embedding) pairs and supports similarity search.

    Think of this as a MongoDB collection where instead of
    querying by field values, you query by semantic similarity.
    """

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embeddings: list[list[float]] = []

    def add(self, chunk: DocumentChunk, embedding: list[float]):
        self.chunks.append(chunk)
        self.embeddings.append(embedding)

    def search(
        self, query_embedding: list[float], top_k: int = 3
    ) -> list[tuple[DocumentChunk, float]]:
        """Return top_k most similar chunks with their scores."""

        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)

        scores = []
        for i, emb in enumerate(self.embeddings):
            emb_vec = np.array(emb)
            # Cosine similarity
            similarity = float(
                np.dot(query_vec, emb_vec)
                / (np.linalg.norm(query_vec) * np.linalg.norm(emb_vec))
            )
            scores.append((self.chunks[i], similarity))

        # Sort by similarity, return top K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def count(self) -> int:
        return len(self.chunks)

    def clear(self):
        self.chunks.clear()
        self.embeddings.clear()


# Single global vector store (in production: use a proper DB)
vector_store = InMemoryVectorStore()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_embedding(text: str) -> list[float]:
    """Get embedding for a text string."""
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks.

    Why overlap? If an important sentence is at the boundary of two chunks,
    overlap ensures it appears fully in at least one chunk.

    chunk_size: number of words per chunk
    overlap: words shared between consecutive chunks
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap)
        start += chunk_size - overlap

    return chunks


def build_rag_prompt(
    question: str, retrieved_chunks: list[tuple[DocumentChunk, float]]
) -> list[dict]:
    """
    Build the prompt for the LLM using retrieved context.

    This is the "augmented generation" part of RAG.
    The model's answer is grounded in real retrieved text, not hallucinated.
    """

    # Format retrieved chunks into context
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk.title} (relevance: {score:.2f})]\n{chunk.content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = """You are a helpful assistant that answers questions based on provided document context.

RULES:
1. Only answer based on the provided context below.
2. If the answer is not in the context, say "I don't have enough information to answer that."
3. Always mention which source(s) you used.
4. Be concise and accurate.

CONTEXT:
{context}""".format(
        context=context
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/")
async def health_check():
    return {
        "status": "running",
        "documents_indexed": vector_store.count(),
        "message": "POST /upload to add documents, POST /ask to query them",
    }


@app.post("/upload")
async def upload_document(doc: DocumentInput):
    """
    Upload a document. It gets chunked and embedded, then stored.

    Example request:
    POST /upload
    {
        "title": "Python Guide",
        "content": "Python is a high-level programming language... (long text)"
    }
    """

    # Step 1: Split into chunks (large docs don't fit in context window)
    chunks = chunk_text(doc.content, chunk_size=200, overlap=20)

    if not chunks:
        raise HTTPException(status_code=400, detail="Document content is empty")

    # Step 2: Embed each chunk and store
    embedded_count = 0
    for i, chunk_text_content in enumerate(chunks):

        # Skip very short chunks (not useful)
        if len(chunk_text_content.split()) < 10:
            continue

        chunk = DocumentChunk(
            title=doc.title, content=chunk_text_content, chunk_index=i
        )

        # Get embedding for this chunk
        embedding = get_embedding(chunk_text_content)

        # Store in vector store
        vector_store.add(chunk, embedding)
        embedded_count += 1

    return {
        "message": f"Document '{doc.title}' uploaded successfully",
        "chunks_created": embedded_count,
        "total_indexed": vector_store.count(),
    }


@app.post("/ask")
async def ask_question(query: QuestionInput):
    """
    Ask a question. Retrieves relevant chunks, then uses LLM to answer.

    Example request:
    POST /ask
    {
        "question": "What is the context window in LLMs?",
        "stream": false,
        "top_k": 3
    }
    """

    if vector_store.count() == 0:
        raise HTTPException(
            status_code=400, detail="No documents indexed. Use POST /upload first."
        )

    # Step 1: Embed the question
    question_embedding = get_embedding(query.question)

    # Step 2: Retrieve the most relevant chunks
    retrieved = vector_store.search(question_embedding, top_k=query.top_k)

    if not retrieved:
        raise HTTPException(status_code=500, detail="Retrieval failed")

    # Step 3: Build the prompt with retrieved context
    messages = build_rag_prompt(query.question, retrieved)

    # Step 4a: Streaming response
    if query.stream:

        def generate():
            stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                stream=True,
                temperature=0.3,  # low temp for factual Q&A
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Step 4b: Non-streaming response
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.3
    )

    answer = response.choices[0].message.content

    # Return answer + metadata about what was retrieved (useful for debugging)
    return {
        "question": query.question,
        "answer": answer,
        "sources_used": [
            {
                "title": chunk.title,
                "relevance_score": round(score, 4),
                "chunk_preview": chunk.content[:100] + "...",
            }
            for chunk, score in retrieved
        ],
        "tokens_used": {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    }


@app.delete("/clear")
async def clear_documents():
    """Clear all indexed documents."""
    count = vector_store.count()
    vector_store.clear()
    return {"message": f"Cleared {count} document chunks"}


# =============================================================================
# HOW TO TEST THIS (run in a separate terminal after starting the server)
# =============================================================================

TEST_INSTRUCTIONS = """
HOW TO TEST THIS API:

1. Start the server:
   uvicorn 05_real_world_mini_project:app --reload

2. Check it's running:
   curl http://localhost:8000/

3. Upload a document (use your own content here):
   curl -X POST http://localhost:8000/upload \\
     -H "Content-Type: application/json" \\
     -d '{
       "title": "LLM Guide",
       "content": "Large Language Models are AI systems trained on massive text datasets. They use the Transformer architecture with attention mechanisms. The context window determines how much text the model can process at once. GPT-4o has a 128000 token context window. Temperature controls randomness in outputs. Embeddings convert text to vectors for semantic similarity search. RAG combines retrieval with generation to reduce hallucinations. Fine-tuning adapts models to specific domains."
     }'

4. Ask a question:
   curl -X POST http://localhost:8000/ask \\
     -H "Content-Type: application/json" \\
     -d '{"question": "What is the context window of GPT-4o?", "stream": false}'

5. Try the API docs:
   Open http://localhost:8000/docs in your browser (FastAPI Swagger UI)
"""

if __name__ == "__main__":
    print(TEST_INSTRUCTIONS)
    print(
        "Starting server... run with: uvicorn 05_real_world_mini_project:app --reload"
    )
