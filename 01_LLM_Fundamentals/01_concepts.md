# Module 1: LLM Fundamentals

> **Your Trainer's Note:** You already know how APIs work, how servers process requests, and how to build full-stack apps. That is a massive advantage. We will connect every LLM concept back to things you already understand. By the end of this module, you will know *how* LLMs work, *why* they behave the way they do, and *how to use them* in real applications.

---

## Table of Contents

1. [What is an LLM?](#1-what-is-an-llm)
2. [How LLMs are Trained](#2-how-llms-are-trained)
3. [Tokens — The Currency of LLMs](#3-tokens--the-currency-of-llms)
4. [The Transformer Architecture (Simplified)](#4-the-transformer-architecture-simplified)
5. [Attention Mechanism](#5-attention-mechanism)
6. [Context Window](#6-context-window)
7. [Temperature, Top-P, Top-K](#7-temperature-top-p-top-k)
8. [Types of LLMs](#8-types-of-llms)
9. [LLM APIs — How You Will Actually Use Them](#9-llm-apis--how-you-will-actually-use-them)
10. [Embeddings — Turning Text into Numbers](#10-embeddings--turning-text-into-numbers)
11. [Hallucinations and Why They Happen](#11-hallucinations-and-why-they-happen)
12. [Key Terms Cheat Sheet](#12-key-terms-cheat-sheet)

---

## 1. What is an LLM?

An **LLM (Large Language Model)** is a deep learning model trained on massive amounts of text data to understand and generate human language.

Think of it like this:

> You have a FastAPI endpoint. You send it a request (prompt), it processes it, and returns a response. An LLM is essentially a very sophisticated function:
> `f(text_input) → text_output`

But internally, instead of your business logic, there are **billions of parameters** (weights) that were tuned during training to predict the most likely next word/token.

### Real-World Examples of LLMs
| Model | Made By | Notes |
|---|---|---|
| GPT-4o | OpenAI | Most popular, used via API |
| Claude 3.5 Sonnet | Anthropic | Strong reasoning, large context |
| Gemini 1.5 Pro | Google | 1M token context window |
| Llama 3.1 | Meta | Open source, you can run locally |
| Mistral 7B | Mistral AI | Lightweight, open source |

---

## 2. How LLMs are Trained

Training happens in stages. You don't need to train one yourself (costs millions of dollars), but you MUST understand this to ace interviews and build smart applications.

### Stage 1: Pre-training
- The model reads **trillions of tokens** from the internet, books, code, Wikipedia, etc.
- It learns one simple task: **predict the next token**
- Example: given `"The capital of France is"` → predict `"Paris"`
- This is called **self-supervised learning** — no human labeling needed
- After this, the model has broad world knowledge but doesn't follow instructions well

### Stage 2: Fine-tuning (Supervised Fine-Tuning / SFT)
- The model is trained on **curated (prompt, response) pairs**
- Example: `{"prompt": "Summarize this article", "response": "The article discusses..."}`
- This teaches the model to follow instructions
- You can do this yourself with tools like Hugging Face + LoRA (we'll cover this later)

### Stage 3: RLHF (Reinforcement Learning from Human Feedback)
- Human raters score multiple model responses
- A **reward model** is trained on those scores
- The LLM is then tuned to maximize the reward model's score
- This is why ChatGPT feels helpful and safe — it was RLHF'd extensively

```
Pre-training → SFT → RLHF → Deployed Model
   (broad        (follows    (helpful,
   knowledge)    instructions) harmless)
```

---

## 3. Tokens — The Currency of LLMs

This is **critical** — everything in LLM development revolves around tokens.

### What is a Token?
A token is a chunk of text. It's NOT always a word. The tokenizer breaks text into pieces:

```
"Hello world"        → ["Hello", " world"]           = 2 tokens
"Unbelievable"       → ["Un", "believ", "able"]       = 3 tokens
"ChatGPT"            → ["Chat", "G", "PT"]            = 3 tokens
"I love Python"      → ["I", " love", " Python"]      = 3 tokens
```

**Rule of thumb:** 1 token ≈ 0.75 words, or ~4 characters in English

### Why Tokens Matter for You as a Developer
- **Cost**: OpenAI charges per 1000 tokens (input + output)
- **Speed**: More tokens = slower response
- **Context limit**: Every model has a max token limit (context window)
- **Optimization**: You need to write prompts that are token-efficient

### Token Counting Example
```python
# Using tiktoken (OpenAI's tokenizer library)
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")
text = "Hello, I am building an AI application with FastAPI."
tokens = encoder.encode(text)

print(f"Token count: {len(tokens)}")   # ~11 tokens
print(f"Tokens: {tokens}")             # list of integer IDs
```

---

## 4. The Transformer Architecture (Simplified)

Every modern LLM is built on the **Transformer** architecture, introduced in the 2017 paper *"Attention is All You Need"*.

You don't need to implement one, but you need to understand the key components.

### High-Level Architecture

```
Input Text
    ↓
[Tokenizer] → converts text to token IDs
    ↓
[Embedding Layer] → converts each token ID to a vector (e.g., 768 dimensions)
    ↓
[Positional Encoding] → adds position info (token 1, token 2, etc.)
    ↓
[N × Transformer Blocks]
    ├── [Multi-Head Self-Attention] ← THE CORE
    ├── [Feed Forward Network]
    └── [Layer Normalization + Residual Connections]
    ↓
[Output Layer] → predicts probability over entire vocabulary
    ↓
Next Token (e.g., "Paris")
```

### Key Insight — Why Transformers Beat Everything Before
Before Transformers, we had RNNs/LSTMs. They processed text **sequentially** (word by word). The problem: by the time you reach word 100, the model had "forgotten" word 1.

Transformers process **all tokens simultaneously** and let every token "look at" every other token. This is the attention mechanism.

---

## 5. Attention Mechanism

This is the most important concept in modern AI. Let me explain it like you're a developer.

### The Core Idea
When processing the word `"bank"` in:
- `"I went to the river bank"` → should focus on "river"
- `"I went to the bank to deposit money"` → should focus on "deposit", "money"

Attention lets each token **dynamically decide** which other tokens are most relevant.

### How it Works (Conceptually)
For each token, the model computes three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I offer?"
- **Value (V)**: "What's my actual content?"

Attention score = how well a Query matches a Key → used to weight the Values

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```

Don't panic about the math. The intuition is:
> Token A asks "who is relevant to me?" (Query)
> Token B says "here's what I'm about" (Key)
> If they match, Token A borrows information from Token B (Value)

### Multi-Head Attention
Instead of doing this once, the model does it **multiple times in parallel** (e.g., 12 heads in GPT-2, 96 in GPT-4). Each head can learn a different type of relationship:
- Head 1 might focus on grammar
- Head 2 might focus on semantic meaning
- Head 3 might focus on coreference (he/she/it → who?)

---

## 6. Context Window

The **context window** is the maximum number of tokens an LLM can "see" at once — both your input AND its output combined.

### Why It Matters
```
Context Window = Input Tokens + Output Tokens

If model has 8,000 token context:
- You send 6,000 tokens of input (a long document)
- Model can only output ~2,000 tokens
```

### Real Context Windows (as of 2025)
| Model | Context Window |
|---|---|
| GPT-4o | 128,000 tokens (~96,000 words) |
| Claude 3.5 Sonnet | 200,000 tokens |
| Gemini 1.5 Pro | 1,000,000 tokens |
| Llama 3.1 70B | 128,000 tokens |

### The "Lost in the Middle" Problem
Research shows LLMs perform best with information at the **beginning** or **end** of the context. Information buried in the middle gets "forgotten". This directly impacts how you design RAG systems (we'll cover in Module 3).

---

## 7. Temperature, Top-P, Top-K

These are parameters you control when calling LLM APIs. Understanding them is essential for building good AI products.

### Temperature
Controls **randomness / creativity** in the output.

```
Temperature = 0.0  →  Always picks the most likely token (deterministic)
Temperature = 0.7  →  Balanced (default for most use cases)
Temperature = 1.0  →  More creative/random
Temperature = 2.0  →  Very random, often incoherent
```

**Real-world usage:**
- Chatbots: `temperature = 0.7`
- Code generation: `temperature = 0.0` or `0.1` (you want correct, deterministic code)
- Creative writing: `temperature = 0.9 - 1.2`
- Data extraction / classification: `temperature = 0.0`

### Top-K
Instead of considering ALL possible next tokens, only consider the **top K most likely** ones.
- `top_k = 1` → same as temperature 0 (greedy)
- `top_k = 50` → choose from the 50 most likely tokens

### Top-P (Nucleus Sampling)
Instead of a fixed K, pick the smallest group of tokens whose combined probability adds up to P.
- `top_p = 0.9` → consider tokens that together make up 90% probability mass
- More dynamic than top-K

### In Practice
You usually just tune **temperature**. OpenAI recommends not changing both temperature and top_p simultaneously.

```python
# OpenAI API example
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a poem about Python"}],
    temperature=0.9,    # creative
    max_tokens=200,     # limit output length
    top_p=1.0,          # default
)
```

---

## 8. Types of LLMs

### By Access
| Type | Description | Examples |
|---|---|---|
| **Proprietary/Closed** | API-only, you don't see weights | GPT-4o, Claude, Gemini |
| **Open Source** | Download and run yourself | Llama 3, Mistral, Phi-3 |
| **Open Weights** | Weights public but license restricted | Some Meta models |

### By Capability
| Type | Description | Use Case |
|---|---|---|
| **Base Model** | Just pre-trained, not instruction-tuned | Research, fine-tuning |
| **Instruction-tuned** | Follows instructions (chat models) | Most applications |
| **Code models** | Specialized for code | GitHub Copilot (CodeX) |
| **Multimodal** | Text + images (+ audio/video) | GPT-4o, Gemini |
| **Embedding models** | Text → vector | RAG, search, similarity |

### By Size
Size is measured in **parameters** (the weights):
- **Small** (1B-7B): Phi-3, Mistral 7B — run on your laptop
- **Medium** (13B-34B): Llama 3.1 8B+, CodeLlama — need a good GPU
- **Large** (70B+): Llama 3.1 70B, Mixtral 8x22B — need server-grade GPU
- **Very Large** (unknown, est. 1T+): GPT-4, Gemini Ultra — only via API

---

## 9. LLM APIs — How You Will Actually Use Them

As an AI engineer with FastAPI knowledge, this is where you start building immediately.

### The Message Format
Every major LLM API uses a **chat message format** with roles:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "How do I install it?"}  # new message
]
```

- **system**: Instructions to the model (behavior, persona, constraints)
- **user**: Human's message
- **assistant**: Previous AI responses (for conversation history)

### OpenAI API (Industry Standard)
Most important to learn. Almost every other provider has an OpenAI-compatible API.

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Explain async/await in Python"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)

# Token usage (important for cost tracking!)
print(f"Input tokens: {response.usage.prompt_tokens}")
print(f"Output tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")
```

### Streaming Responses
Critical for building good UX — users see text appear word by word instead of waiting.

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True  # Enable streaming
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 10. Embeddings — Turning Text into Numbers

Embeddings are a **foundational concept for RAG** (Module 3). Understand this well.

### What Are Embeddings?
An embedding converts text into a **high-dimensional vector** (list of numbers) that captures semantic meaning.

```
"dog"     → [0.2, -0.5, 0.8, 0.1, ...]   # 1536 numbers
"puppy"   → [0.21, -0.49, 0.79, 0.12, ...] # very similar vector!
"cat"     → [0.18, -0.3, 0.6, 0.05, ...]  # similar, but different
"car"     → [-0.8, 0.3, -0.2, 0.9, ...]   # very different
```

The key insight: **similar meanings → similar vectors**

### Cosine Similarity
To compare two embeddings, we use cosine similarity (ranges from -1 to 1):
- `1.0` = identical meaning
- `0.9` = very similar
- `0.0` = unrelated
- `-1.0` = opposite meaning

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dimensions
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

emb1 = get_embedding("I love programming in Python")
emb2 = get_embedding("Python is my favorite coding language")
emb3 = get_embedding("I enjoy eating pizza")

print(cosine_similarity(emb1, emb2))  # ~0.93 (very similar)
print(cosine_similarity(emb1, emb3))  # ~0.3 (unrelated)
```

**This is the foundation of RAG** — you embed your documents, embed the user's question, find the most similar documents, and give them to the LLM as context.

---

## 11. Hallucinations and Why They Happen

**Hallucination** = when an LLM generates confident-sounding but factually incorrect information.

### Why It Happens
LLMs are trained to predict the **most statistically likely next token**, not to "tell the truth". If a plausible-sounding but wrong answer has a high probability in the training data, the model will generate it.

Classic examples:
- Making up fake research papers with real-sounding authors
- Inventing API functions that don't exist
- Getting historical dates wrong with complete confidence

### How to Mitigate (This is Your Job as an AI Engineer)
1. **RAG (Retrieval Augmented Generation)**: Give the model real facts as context (Module 3)
2. **Grounding**: Tell the model "only answer from the provided context"
3. **Temperature = 0**: More deterministic, less creative hallucination
4. **Verification chains**: Use a second LLM call to verify the first answer
5. **Structured outputs**: Force JSON output — harder to hallucinate structure
6. **Citation prompting**: Ask the model to cite sources from provided context

---

## 12. Key Terms Cheat Sheet

| Term | Simple Definition |
|---|---|
| **LLM** | Large language model — a neural network trained on text |
| **Token** | Basic unit of text (~0.75 words) |
| **Context Window** | Max tokens the model can see at once |
| **Temperature** | Controls randomness (0 = deterministic, 1+ = creative) |
| **Embedding** | Text converted to a vector of numbers |
| **Hallucination** | LLM making up confident-sounding false information |
| **Prompt** | The input you give to an LLM |
| **System Prompt** | Instructions that define the model's behavior |
| **Inference** | Running a trained model to get an output |
| **Fine-tuning** | Further training a pre-trained model on specific data |
| **RLHF** | Training technique using human feedback to improve models |
| **Transformer** | The neural network architecture all modern LLMs use |
| **Attention** | Mechanism that lets tokens relate to each other |
| **Parameters** | The learnable weights inside the model (billions of them) |
| **Top-P / Top-K** | Sampling parameters controlling token selection |

---

## What's Next?

You now understand **how LLMs work**, **how to call them**, and **key concepts** every interviewer will ask about.

**Next module: Prompt Engineering** — learning to speak to LLMs effectively. This is the highest ROI skill for an AI engineer, directly impacts the quality of every application you build.

> **Study tip:** Before moving on, make sure you can explain tokens, context window, temperature, and embeddings out loud without looking at notes. These come up in EVERY AI engineering interview.
