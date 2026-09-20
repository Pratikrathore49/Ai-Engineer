# app/services.py

import logging

from app.cache import get_cached_response, set_cached_response
from app.conversations import add_message, create_conversation, get_conversation
from app.llm import generate_llm_response, generate_final_llm_response, stream_llm_response
from app.schemas import ChatRequest, ChatResponse
from app.tools import execute_tool

logger = logging.getLogger(__name__)

def generate_reply(request: ChatRequest) -> ChatResponse:

    # 1. Get or create conversation
    conversation_id = request.conversation_id or create_conversation()

    # Log the incoming request (preview only — no full message content)
    message_preview = request.message[:50]
    logger.info(
        "Request received | conversation_id=%s | message_preview='%s'",
        conversation_id,
        message_preview
    )

    # 2. Check cache
    cached_response = get_cached_response(request.message)

    if cached_response:
        logger.info(
            "Cache hit | conversation_id=%s | message_preview='%s'",
            conversation_id,
            message_preview
        )
        add_message(conversation_id, "user", request.message)
        add_message(conversation_id, "assistant", cached_response.reply)

        return cached_response.model_copy(
            update={
                "cached": True,
                "input_tokens": 0,
                "output_tokens": 0,
                "conversation_id": conversation_id
            }
        )

    # 3. Add user message
    add_message(conversation_id, "user", request.message)

    # 4. Get conversation history
    history = get_conversation(conversation_id)

    # 5. First LLM call
    logger.debug("Calling LLM | conversation_id=%s | history_length=%d", conversation_id, len(history))
    response = generate_llm_response(history)

    assistant_message = response.choices[0].message
    tool_used = False

    # 6. Check whether LLM requested a tool
    if assistant_message.tool_calls:
        tool_used = True

        assistant_message_dict = {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        }

        history.append(assistant_message_dict)

        # 7. Execute every requested tool
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments

            tool_result = execute_tool(tool_name, arguments)

            # 8. Add tool result to messages
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_result)
            })

        # 9. Second LLM call
        logger.debug("Calling LLM for final answer | conversation_id=%s", conversation_id)
        final_response = generate_final_llm_response(history)

        reply = final_response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens + final_response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens + final_response.usage.completion_tokens
        model = final_response.model

    else:
        reply = assistant_message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        model = response.model

    # 10. Save assistant response
    add_message(conversation_id, "assistant", reply)

    # 11. Create API response
    chat_response = ChatResponse(
        reply=reply,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        cached=False,
        conversation_id=conversation_id,
        tool_used=tool_used
    )

    logger.info(
        "Response sent | conversation_id=%s | tool_used=%s | input_tokens=%d | output_tokens=%d",
        conversation_id,
        tool_used,
        input_tokens,
        output_tokens
    )

    # 12. Cache response
    set_cached_response(request.message, chat_response)

    return chat_response

def stream_reply(request: ChatRequest):
    conversation_id = request.conversation_id or create_conversation()
    message_preview = request.message[:50]

    logger.info(
        "Stream request received | conversation_id=%s | message_preview='%s'",
        conversation_id,
        message_preview
    )

    add_message(conversation_id, "user", request.message)
    history = get_conversation(conversation_id)

    logger.debug("Starting LLM stream | conversation_id=%s | history_length=%d", conversation_id, len(history))
    stream = stream_llm_response(history)

    assistant_reply = ""

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            assistant_reply += token
            yield token

    add_message(conversation_id, "assistant", assistant_reply)
    logger.info(
        "Stream complete | conversation_id=%s | reply_length=%d",
        conversation_id,
        len(assistant_reply)
    )
