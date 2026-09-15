from app.cache import (get_cached_response,set_cached_response)

from app.conversations import (add_message,create_conversation,get_conversation)

from app.llm import(generate_llm_response,generate_final_llm_response,stream_llm_response)

from app.schemas import ChatRequest,ChatResponse

from app.tools import execute_tool

def generate_reply(request:ChatRequest) -> ChatResponse:

    #1. Get or create conversation
    conversation_id = (request.conversation_id or create_conversation())
    #2. Check cache
    cached_response = get_cached_response(request.message)
    
    if cached_response:
        add_message(
        conversation_id,"user",request.message
        )
    
        add_message(conversation_id,"assistant",cached_response.reply)

        return cached_response.model_copy(
            update={
                "cached":True,
                "input_tokens":0,
                "output_tokens":0,
                "conversation_id":conversation_id
            }
        )
    # 3. Add User message
    add_message(
        conversation_id,
        "user",
        request.message
    )

    #4. Get conversation history
    history = get_conversation(conversation_id)

    #5. First LLM call
    response = generate_llm_response(history)

    assistant_message = response.choices[0].message

    tool_used = False

    # 6. Check Whether LLM requested a tool

    if assistant_message.tool_calls:

        tool_used = True
        #Convert assistant message to dictionary 
        assistant_message_dict ={
            "role":"assistant",
            "content":assistant_message.content,
            "tool_calls":[
                {
                    "id":tool_call.id,
                    "type":"function",
                    "function":{
                        "name":tool_call.function.name,
                        "arguments":tool_call.function.arguments
                    }
                }
                for tool_call in assistant_message.tool_calls
            ]
        }

        #Add assistant's tool request
        history.append(assistant_message_dict)

        #7. Execute every requested tool
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            arguments = tool_call.function.arguments

            tool_result = execute_tool(tool_name,arguments)

            # 8. Add tool result to messages
            history.append(
                {
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "content":str(tool_result)
                }
            )
        
        #9. Second LLM Call
        final_response = generate_final_llm_response(history)

        reply = final_response.choices[0].message.content

        input_tokens = response.usage.prompt_tokens + final_response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens + final_response.usage.completion_tokens
        model = final_response.model
    
    else:
        # No tool required
        reply = assistant_message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        model = response.model
    
    #10. Save assistant response
    add_message(
        conversation_id,
        "assistant",
        reply
    )

    # 11. Create API response
    chat_response = ChatResponse(
        reply=reply,
        input_tokens =input_tokens,
        output_tokens = output_tokens,
        model = model,
        cached=False,
        conversation_id=conversation_id,
        tool_used=tool_used
    )

    #12.Cache response
    set_cached_response(
        request.message,
        chat_response
    )

    return chat_response



def stream_reply(request:ChatRequest):
    # Get or create conversation
    conversation_id =(request.conversation_id or create_conversation())

    #Add user message
    add_message(conversation_id,"user",request.message)

    #Get history
    history = get_conversation(conversation_id)

    #call LLM
    stream = stream_llm_response(history)

    #Collect assistant response
    assistant_reply =""

    for chunk in stream:
        token = chunk.choices[0].delta.content
        
        if token:
            assistant_reply += token

            yield token
    
    # Save assistant response
    add_message(
        conversation_id,"assistant",assistant_reply
    )