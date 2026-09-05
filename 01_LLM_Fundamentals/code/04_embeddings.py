# =============================================================================
# FILE: 04_embeddings.py
# TOPIC: Embeddings — turning text into vectors for semantic search
#
# This is the foundation of RAG (Module 3). Master this here.
#
# pip install openai numpy python-dotenv
# =============================================================================

import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------------------------------------------
# PART 1: Getting your first embedding
# Just like a database returns rows, the embedding API returns vectors
# -----------------------------------------------------------------------------

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """
    Get an embedding for a piece of text.
    
    Models:
    - text-embedding-3-small: 1536 dimensions, cheap ($0.02/1M tokens)
    - text-embedding-3-large: 3072 dimensions, better quality, more expensive
    - text-embedding-ada-002: older, 1536 dims, still widely used
    """
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


def first_embedding_demo():
    """See what an embedding actually looks like."""
    
    text = "Python is a great programming language for AI"
    embedding = get_embedding(text)
    
    print(f"Text: '{text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 10 values: {[round(v, 4) for v in embedding[:10]]}")
    print(f"Last 10 values:  {[round(v, 4) for v in embedding[-10:]]}")
    print()
    print("These 1536 numbers represent the 'meaning' of that sentence")
    print("Similar sentences will have similar sets of numbers")


# -----------------------------------------------------------------------------
# PART 2: Cosine similarity — the core operation of semantic search
# -----------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value between -1 and 1.
    1.0 = identical direction (same meaning)
    0.0 = perpendicular (unrelated)
    -1.0 = opposite direction (opposite meaning)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    # Dot product divided by product of magnitudes
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def similarity_demo():
    """
    Show that similar texts have high cosine similarity.
    This is the "aha moment" for understanding why RAG works.
    """
    
    # A set of sentences with varying similarity
    sentences = {
        "anchor":   "I love programming in Python",
        "similar1": "Python is my favorite coding language",
        "similar2": "Writing software in Python is enjoyable",
        "related":  "I enjoy software development",
        "different":"The weather is sunny today",
        "opposite": "I hate coding and programming",
    }
    
    # Get all embeddings (batch them to save API calls)
    print("Getting embeddings for all sentences...")
    embeddings = {}
    for key, text in sentences.items():
        embeddings[key] = get_embedding(text)
    
    # Compare each sentence to the anchor
    anchor = embeddings["anchor"]
    anchor_text = sentences["anchor"]
    
    print(f"\nAnchor sentence: '{anchor_text}'")
    print("-" * 60)
    print(f"{'Sentence':<12} | {'Similarity':>10} | Text")
    print("-" * 60)
    
    for key, emb in embeddings.items():
        if key == "anchor":
            continue
        similarity = cosine_similarity(anchor, emb)
        print(f"{key:<12} | {similarity:>10.4f} | {sentences[key]}")
    
    print()
    print("Notice: 'similar1' and 'similar2' score highest,")
    print("        'different' scores low,")
    print("        'related' scores in the middle")


# -----------------------------------------------------------------------------
# PART 3: Building a tiny semantic search engine
# This is EXACTLY what happens inside a RAG system's retrieval step.
# Master this and RAG will make complete sense.
# -----------------------------------------------------------------------------

class TinySemanticSearch:
    """
    A mini semantic search engine.
    In production, you'd use a vector database (Pinecone, ChromaDB, etc.)
    but this shows the core concept without extra dependencies.
    
    Think of this like a simplified version of MongoDB Atlas Search,
    but for semantic meaning instead of keyword matching.
    """
    
    def __init__(self):
        self.documents = []      # stores original text
        self.embeddings = []     # stores vectors
    
    def add_documents(self, docs: list[str]):
        """Add documents to the search index."""
        print(f"Indexing {len(docs)} documents...")
        
        # In production: batch these calls. OpenAI allows up to 2048 inputs at once.
        for doc in docs:
            embedding = get_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(embedding)
        
        print(f"Indexed {len(self.documents)} documents total.")
    
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Find the top_k most semantically similar documents to the query.
        
        This is the retrieval step in RAG:
        1. Embed the query
        2. Compare to all document embeddings
        3. Return the most similar ones
        """
        query_embedding = get_embedding(query)
        
        # Calculate similarity to all documents
        similarities = []
        for i, doc_emb in enumerate(self.embeddings):
            score = cosine_similarity(query_embedding, doc_emb)
            similarities.append({
                "document": self.documents[i],
                "score": score,
                "rank": i
            })
        
        # Sort by similarity score, highest first
        similarities.sort(key=lambda x: x["score"], reverse=True)
        
        return similarities[:top_k]


def semantic_search_demo():
    """
    Build and query a tiny knowledge base.
    This is the retrieval step of a RAG pipeline.
    """
    
    # A small knowledge base about AI topics
    knowledge_base = [
        "Python is the most popular language for machine learning and AI development.",
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "The Transformer architecture was introduced in 2017 and revolutionized NLP.",
        "RAG stands for Retrieval Augmented Generation — combining search with LLMs.",
        "Fine-tuning adapts a pre-trained model to a specific task or domain.",
        "Vector databases like Pinecone, Weaviate, and ChromaDB store embeddings efficiently.",
        "Temperature controls the randomness of LLM output generation.",
        "RLHF uses human feedback to align language models with human preferences.",
        "Prompt engineering is the art of crafting effective inputs for LLMs.",
    ]
    
    # Build the search index
    searcher = TinySemanticSearch()
    searcher.add_documents(knowledge_base)
    
    # Run some test queries
    queries = [
        "How do I build a web API in Python?",               # should match FastAPI
        "What is the best way to get an LLM to behave?",     # should match RLHF + prompt eng
        "How do LLMs store and retrieve knowledge?",         # should match RAG + vector DBs
    ]
    
    print("\n=== Semantic Search Results ===\n")
    
    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 50)
        
        results = searcher.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  #{i} (score: {result['score']:.4f}): {result['document']}")
        
        print()


# -----------------------------------------------------------------------------
# PART 4: Batch embeddings — the efficient way to embed many documents
# In production you'll always embed in batches to reduce API calls
# -----------------------------------------------------------------------------

def batch_embeddings_demo():
    """
    Batch multiple texts in a single API call.
    Much more efficient than calling get_embedding() one at a time.
    """
    
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks with many layers",
        "Natural language processing deals with text and speech",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through rewards",
    ]
    
    # ONE API call for ALL texts — much cheaper and faster
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts  # pass a list, not a single string
    )
    
    # Results come back in the same order as input
    embeddings = [item.embedding for item in response.data]
    
    print(f"Batched {len(texts)} texts in 1 API call")
    print(f"Total tokens used: {response.usage.total_tokens}")
    print(f"Got {len(embeddings)} embeddings, each {len(embeddings[0])} dimensions")
    
    # Compare all pairs
    print("\nPairwise similarities (all AI-related, so all should be moderately high):")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  '{texts[i][:30]}...' vs '{texts[j][:30]}...' = {sim:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("PART 1: Your First Embedding")
    print("=" * 60)
    first_embedding_demo()
    
    print("\n" + "=" * 60)
    print("PART 2: Cosine Similarity Demo")
    print("=" * 60)
    similarity_demo()
    
    print("\n" + "=" * 60)
    print("PART 3: Semantic Search Engine")
    print("=" * 60)
    semantic_search_demo()
    
    print("\n" + "=" * 60)
    print("PART 4: Batch Embeddings")
    print("=" * 60)
    batch_embeddings_demo()
