import uuid as uuid_module
from pydantic import BaseModel, Field, field_validator

class CacheRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )
    conversation_id:str| None = None

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must be between 1 and 2000 characters")
        return value
    
    @field_validator("conversation_id")
    @classmethod
    def validate_conversation_id(cls, value:str | None) -> str | None:
        if value is None:
            return value
        try:
            uuid_module.UUID(value)
        except ValueError:
            raise ValueError("conversation_id must be a valid UUID")
        return value

class ChatResponse(BaseModel):
    conversation_id:str
    reply:str
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool
    tool_used: bool

class CacheStats(BaseModel):
    total_items: int
    cache_hits: int
    cache_misses: int