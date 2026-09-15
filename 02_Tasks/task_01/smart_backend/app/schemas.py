from pydantic import BaseModel,Field,field_validator

class ChatRequest(BaseModel):
    message:str=Field(
        ...,
        min_length=1,
        max_length=2000
    )

    conversation_id: str | None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be empty")

        return value

class ChatResponse(BaseModel):
    conversation_id:str
    reply:str
    input_tokens:int
    output_tokens:int
    model:str
    cached:bool
    tool_used:bool

class CacheStats(BaseModel):
    total_items:int
    cache_hits:int
    cache_misses:int