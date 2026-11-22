# app/rag/pipeline.py
#
# The full RAG pipeline — retrieve, prompt, generate, cite.
#
# WHAT MAKES THIS DIFFERENT FROM REGULAR CHAT:
#   Regular /chat:  system_prompt is fixed ("you are a helpful assistant")
#                   LLM answers from its training data
#                   No document context
#
#   /rag/ask:       system_prompt is BUILT DYNAMICALLY from retrieved chunks
#                   LLM is instructed to answer ONLY from the provided context
#                   Sources are tracked and returned to the user
#
# THE GROUNDED PROMPT STRUCTURE:
#   We build a system prompt that contains the actual chunk text.
#   The LLM reads this context and answers from it — not from memory.
#   This is called "in-context learning" or "grounded generation."
#
#   System message:
#       "You are a document assistant. Answer ONLY using the context below..."
#       "[Source: filename.pdf, chunk 3]"
#       "...actual chunk text..."
#       "[Source: filename.pdf, chunk 7]"
#       "...actual chunk text..."
#
#   User message:
#       "What does the document say about error handling?"
#
# WHY NOT CALL THE LLM WHEN THERE ARE NO CHUNKS?
#   If no relevant chunks are found, the LLM has no document context.
#   It would either hallucinate an answer or say "I don't know" —
#   but it would still cost tokens and latency.
#   Better to return a clean "no information found" response immediately.
#   This is the fail-fast principle applied to RAG.

import logging

from sqlalchemy.orm import Session
from openai import OpenAIError

from app.chat.llm import call_llm
from app.core.conversations import add_message, create_conversation, get_conversation
from app.documents.retriever import retrieve_chunks
from app.documents.schemas import RetrievedChunk
from app.rag.schemas import RAGResponse, SourceCitation

logger = logging.getLogger(__name__)

# We use Groq for the LLM call — same as the chat feature.
# WHY not reuse generate_llm_response() from chat/llm.py?
#   chat/llm.py hardcodes tools and the chat system prompt.
#   RAG needs NO tools and a completely different system prompt.
#   We make a direct, clean LLM call here with full control.


def _build_system_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Build a grounded system prompt from retrieved chunks.

    Each chunk is labeled with its source filename and chunk index
    so the LLM knows where each piece of information came from.
    The LLM is explicitly instructed to ONLY use this context.

    WHY BUILD THE PROMPT DYNAMICALLY?
        A fixed system prompt like "you are a helpful assistant" gives the
        LLM no document content to work with. By injecting the actual chunk
        text into the system prompt, we're giving the LLM a temporary
        "working memory" of the most relevant parts of your documents.
        This is the core mechanism of RAG.
    """
    context_blocks = []

    for chunk in chunks:
        block = (
            f"[Source: {chunk.filename}, chunk {chunk.chunk_index}]\n"
            f"{chunk.content}"
        )
        context_blocks.append(block)

    context_text = "\n\n".join(context_blocks)

    return (
        "You are a document assistant. Your job is to answer questions "
        "based strictly on the document context provided below.\n\n"
        "Rules:\n"
        "- Answer ONLY using the information in the context below.\n"
        "- If the answer is not in the context, say: "
        "'I cannot find this information in the provided documents.'\n"
        "- Always mention which source (filename and chunk number) "
        "your answer comes from.\n"
        "- Be concise and accurate.\n\n"
        "CONTEXT:\n"
        "────────────────────────────────────────\n"
        f"{context_text}\n"
        "────────────────────────────────────────"
    )


def ask(
    question: str,
    db: Session,
    document_id: str | None = None,
    conversation_id: str | None = None,
    top_k: int = 5,
) -> RAGResponse:
    """
    Full RAG pipeline — retrieve relevant chunks, build grounded
    prompt, call LLM, return answer with citations.

    Args:
        question:        the user's natural language question
        db:              SQLAlchemy session for pgvector search
        document_id:     optional — restrict search to one document
        conversation_id: optional — save to existing Redis conversation
        top_k:           max chunks to retrieve

    Returns:
        RAGResponse with answer, conversation_id, and sources
    """
    logger.info(
        "RAG pipeline started | question_preview='%s' | document_id=%s",
        question[:50],
        document_id,
    )

    # ── Step 1: Get or create conversation ────────────────────────
    # We always save the Q&A to Redis so follow-up questions work.
    # If the user passes a conversation_id we append to that conversation.
    # If not, we create a new one.
    conv_id = conversation_id or create_conversation()
    logger.debug("Using conversation_id=%s", conv_id)

    # ── Step 2: Retrieve relevant chunks from pgvector ────────────
    chunks = retrieve_chunks(
        query=question,
        db=db,
        document_id=document_id,
        top_k=top_k,
    )

    # ── Step 3: Handle no results — don't call the LLM ────────────
    # WHY this check matters:
    #   If the user asks about something completely unrelated to their
    #   documents, we have two options:
    #     a) Call the LLM anyway — it has no context, wastes tokens,
    #        and might hallucinate an answer that sounds like it came
    #        from the document but didn't.
    #     b) Return a clean "not found" response immediately — honest,
    #        fast, free (no LLM call), and prevents hallucination.
    #   We choose (b). This is a deliberate product decision.
    if not chunks:
        logger.info(
            "No relevant chunks found | question_preview='%s'",
            question[:50],
        )

        no_info_answer = (
            "I could not find relevant information in your documents "
            "to answer this question. Please make sure you have uploaded "
            "documents related to this topic, or try rephrasing your question."
        )

        # Still save to conversation so the user has a record.
        # Wrapped in try/except (just like Step 7) because if the user
        # passed a wrong/nonexistent conversation_id, add_message() raises KeyError.
        try:
            add_message(conv_id, "user", question)
            add_message(conv_id, "assistant", no_info_answer)
        except Exception as exc:
            logger.warning("Failed to save to Redis | error=%s", exc)

        return RAGResponse(
            answer=no_info_answer,
            conversation_id=conv_id,
            sources=[],
            chunks_used=0,
        )

    logger.info(
        "Chunks retrieved | count=%d | top_score=%.4f",
        len(chunks),
        chunks[0].similarity_score,
    )

    # ── Step 4: Build the grounded system prompt ──────────────────
    system_prompt = _build_system_prompt(chunks)

    # ──old Step 5: Build messages for the LLM call ───────────────────
    # We include conversation history so follow-up questions work.
    # The system prompt changes every call (it contains the retrieved
    # chunks) so we build it fresh each time.
    #
    # WHY NOT use get_conversation() from conversations.py here?
    #   We could — but for RAG the context window fills up fast.
    #   Each message contains the full chunk context in the system prompt.
    #   For now we keep it simple: single turn with fresh system prompt.
    #   Multi-turn RAG with history is a natural next improvement.
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": question},
    # ]

    # ── Step 5: Load conversation history & build messages ────────
    # WHY load history ONLY if `conversation_id` was passed?
    #   In Step 1, we created `conv_id = conversation_id or create_conversation()`.
    #   We intentionally check `conversation_id` (the argument from the user),
    #   NOT `conv_id` — because a brand-new conversation has no history yet,
    #   so checking Redis for it would be a wasted network call.
    history = []
    if conversation_id:
        try:
            # Fetch previous turns: [{"role": "user", ...}, {"role": "assistant", ...}]
            history = get_conversation(conversation_id)

            # WHY check the last message?
            #   Right now Step 7 saves the current question *after* the LLM call.
            #   However, if a previous request crashed midway or another caller
            #   already saved the user message, we don't want to send the exact
            #   same question twice in a row.
            #   `history[-1]` gets the last item in the list, and checking `if history`
            #   first prevents an IndexError when the list is empty (`[]`).
            if (
                history
                and history[-1].get("role") == "user"
                and history[-1].get("content") == question
            ):
                # Slice everything EXCEPT the last item
                history = history[:-1]

        except KeyError:
            # WHY catch KeyError?
            #   Look at `conversations.py`: `get_conversation()` raises `KeyError`
            #   when the conversation_id doesn't exist (e.g. wrong ID or expired
            #   after 24h TTL). Instead of crashing, we treat it as a fresh start.
            logger.debug(
                "Conversation not found, starting fresh | conversation_id=%s",
                conversation_id,
            )
            history = []

        except Exception as exc:
            # WHY also catch Exception?
            #   `conversations.py` can raise `redis.RedisError` if Redis is down
            #   or unreachable. Answering the user's question with fresh context
            #   is better than failing the whole API call ("graceful degradation").
            logger.warning(
                "Failed to load conversation history | conversation_id=%s | error=%s",
                conversation_id,
                exc,
            )
            history = []

    # Build the final message list in chronological order:
    # 1. Fresh system prompt (contains chunks relevant to the NEW question)
    # 2. Unpacked past messages (*history) — sliding window applied
    # 3. Current question at the very end
    #
    # WHY sliding window?
    #   Full history + chunk context can exceed the model's context window.
    #   We keep only the last MAX_HISTORY_MESSAGES messages so long
    #   conversations never cause a context_length_exceeded error.
    #   The trade-off: very old turns are forgotten — acceptable for most use cases.
    MAX_HISTORY_MESSAGES = 10  # 5 user + 5 assistant turns
    if len(history) > MAX_HISTORY_MESSAGES:
        logger.debug(
            "History trimmed | original=%d | kept=%d",
            len(history),
            MAX_HISTORY_MESSAGES,
        )
        history = history[-MAX_HISTORY_MESSAGES:]

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": question},
    ]
    # ── Step 6: Call the LLM ──────────────────────────────────────
    # response = client.chat.completions.create(
    #     model=GROQ_MODEL,
    #     messages=messages,
    #     # WHY tool_choice="none"?
    #     #   RAG doesn't use tools. The LLM should only read the
    #     #   context and generate an answer — no function calls.
    #     #   Explicitly disabling tools avoids accidental tool calls.
    #     tool_choice="none",
    # )
    # call_llm() uses the shared Groq client from chat/llm.py.
    # No duplicate client, no hardcoded tools, full control over messages.
    try:
        logger.info("RAG LLM call | chunks_in_context=%d", len(chunks))
        answer = call_llm(messages)

    except OpenAIError as exc:
        logger.error(
            "RAG LLM call failed | error_type=%s | error=%s",
            type(exc).__name__,
            str(exc),
        )
        raise

    # ── Step 7: Save to Redis conversation ───────────────────────
    # Saving both the question and answer lets us support follow-up
    # questions in the same conversation later.
    try:
        add_message(conv_id, "user", question)
        add_message(conv_id, "assistant", answer)
    except Exception as exc:
        logger.warning("Failed to save to Redis | error=%s", exc)
    # don't re-raise — the answer is more important than the history

    # ── Step 8: Build sources list for the response ───────────────
    # We return which chunks were actually used so the client can
    # show citations — "this answer came from report.pdf, chunk 3"
    sources = [
        SourceCitation(
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            similarity_score=chunk.similarity_score,
        )
        for chunk in chunks
    ]

    logger.info(
        "RAG pipeline complete | conversation_id=%s | chunks_used=%d",
        conv_id,
        len(chunks),
    )

    return RAGResponse(
        answer=answer,
        conversation_id=conv_id,
        sources=sources,
        chunks_used=len(chunks),
    )
