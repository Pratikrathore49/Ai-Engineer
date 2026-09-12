import uuid

# In-memory conversation storage
_conversations: dict[str, list[dict[str, str]]] = {}


def create_conversation() -> str:
    """
    Create a new conversation and return its UUID.
    """
    conversation_id = str(uuid.uuid4())

    _conversations[conversation_id] = []

    return conversation_id


def get_conversation(conversation_id: str) -> list[dict[str, str]]:
    """
    Return conversation history.

    Raises KeyError if conversation does not exist.
    """
    if conversation_id not in _conversations:
        raise KeyError(conversation_id)

    return _conversations[conversation_id]


def add_message(conversation_id: str, role: str, content: str) -> None:
    """
    Add a message to a conversation.
    """
    if conversation_id not in _conversations:
        raise KeyError(conversation_id)

    _conversations[conversation_id].append({"role": role, "content": content})
