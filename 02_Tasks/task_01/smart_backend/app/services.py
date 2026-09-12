from app.cache import get_cached_response, set_cached_response

from app.conversations import add_message, create_conversation, get_conversation

from app.llm import generate_llm_response, stream_llm_response

from app.schemas import ChatRequest, ChatResponse


def generate_reply(request: ChatRequest) -> ChatResponse:

    # 1. Get or create conversation
    if request.conversation_id:

        conversation_id = request.conversation_id

        try:
            history = get_conversation(conversation_id)

        except KeyError:
            raise ValueError("Conversation not found")

    else:

        conversation_id = create_conversation()

        history = get_conversation(conversation_id)

    # 2. Check cache
    cached_response = get_cached_response(request.message)

    if cached_response:

        # Save user message
        add_message(conversation_id, "user", request.message)

        # Save assistant message
        add_message(conversation_id, "assistant", cached_response.reply)

        return cached_response.model_copy(
            update={
                "cached": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "conversation_id": conversation_id,
            }
        )

    # 3. Add current user message to history
    add_message(conversation_id, "user", request.message)

    # 4. Get updated history
    history = get_conversation(conversation_id)

    # 5. Call LLM
    response = generate_llm_response(history)

    # 6. Extract assistant reply
    reply = response.choices[0].message.content

    # 7. Save assistant message
    add_message(conversation_id, "assistant", reply)

    # 8. Create response
    chat_response = ChatResponse(
        reply=reply,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model=response.model,
        cached=False,
        conversation_id=conversation_id,
    )

    # 9. Store in cache
    set_cached_response(request.message, chat_response)

    return chat_response


def stream_reply(request: ChatRequest):

    # 1. Get or create conversation
    if request.conversation_id:

        conversation_id = request.conversation_id

        try:
            history = get_conversation(conversation_id)

        except KeyError:
            raise ValueError("Conversation not found")

    else:

        conversation_id = create_conversation()

        history = get_conversation(conversation_id)

    # 2. Add user message
    add_message(conversation_id, "user", request.message)

    # 3. Get updated history
    history = get_conversation(conversation_id)

    # 4. Call LLM
    stream = stream_llm_response(history)

    # 5. Collect complete assistant response
    assistant_reply = ""

    for chunk in stream:

        token = chunk.choices[0].delta.content

        if token:
            assistant_reply += token
            yield token

    # 6. Save assistant response
    add_message(conversation_id, "assistant", assistant_reply)
