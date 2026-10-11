# app/conversations.py
#
# Redis-backed conversation store using Redis Lists.
#
# WHY THIS IS BETTER THAN A DICT:
#   Old approach:  _conversations = {}   → lost on every server restart
#   New approach:  Redis Lists           → persists across restarts, shared
#                                          across instances, auto-expires
#
# DATA MODEL:
#   key   → "conversation:{uuid4}"
#   value → Redis List where each element is a JSON-encoded message dict
#            e.g.  '{"role": "user", "content": "hello"}'
#   TTL   → 24 hours (86400 seconds) — old conversations auto-clean
#
# REDIS COMMANDS USED:
#   RPUSH key value   → append one item to the right end of the list
#   LRANGE key 0 -1   → retrieve ALL items from index 0 to the last (-1)
#   EXPIRE key secs   → set/refresh TTL on the key
#   EXISTS key        → check whether the key exists at all
#   DEL key           → delete the key + its entire list
#
# SERIALIZATION:
#   Same rule as cache.py — Redis stores strings, not dicts.
#   Each message is json.dumps'd before RPUSH and json.loads'd after LRANGE.

import json
import logging
import uuid

import redis

from app.redis_client import redis_client

logger = logging.getLogger(__name__)

CONVERSATION_TTL_SECONDS = 86_400   # 24 hours


# ------------------------------------------------------------------ #
#  Helper: build the Redis key                                        #
# ------------------------------------------------------------------ #

def _make_conversation_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}"


# ------------------------------------------------------------------ #
#  Public interface (same signatures as the old dict-based store)     #
# ------------------------------------------------------------------ #

def create_conversation() -> str:
    """
    Generate a new UUID, create an empty list in Redis, and return the ID.

    We RPUSH a sentinel empty message then immediately delete it — actually
    the simpler approach is to just set the TTL key and rely on the first
    real message to create the list.  We use EXISTS checks on lookup instead.
    """
    conversation_id = str(uuid.uuid4())
    key = _make_conversation_key(conversation_id)

    # We don't push anything yet — the list is created on first RPUSH.
    # But we DO set a placeholder so the key exists and we can detect
    # "conversation was created" vs "conversation never existed".
    # A clean way: store an empty JSON array as a string at a helper key.
    # Simpler way used here: let add_message create it, track existence
    # by checking the key OR by convention (404 on unknown ID).
    #
    # We store a tiny marker so /conversations/{id} can return an empty
    # list without a 404 for brand-new conversations.
    redis_client.set(
        f"conversation_meta:{conversation_id}",
        "created",
        ex=CONVERSATION_TTL_SECONDS,
    )

    logger.debug("Conversation created | conversation_id=%s", conversation_id)
    return conversation_id


def _conversation_exists(conversation_id: str) -> bool:
    """
,Return True if this conversation was ever created.
    Checks both the meta marker and the message list.
    """
    meta_key = f"conversation_meta:{conversation_id}"
    msg_key  = _make_conversation_key(conversation_id)
    return bool(
        redis_client.exists(meta_key) or redis_client.exists(msg_key)
    )


def get_conversation(conversation_id: str) -> list[dict[str, str]]:
    """
    Return the full message list for a conversation.

    Raises KeyError if the conversation does not exist (consistent with
    the old dict-based interface so services.py doesn't need to change).
    """
    if not _conversation_exists(conversation_id):
        raise KeyError(conversation_id)

    key = _make_conversation_key(conversation_id)

    try:
        # LRANGE key 0 -1 → all elements, oldest first
        raw_messages = redis_client.lrange(key, 0, -1)
    except redis.RedisError as exc:
        logger.error(
            "Conversation GET failed | conversation_id=%s | error=%s",
            conversation_id,
            exc,
        )
        raise

    messages = []
    for raw in raw_messages:
        try:
            messages.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            logger.warning(
                "Skipping malformed message | conversation_id=%s | error=%s",
                conversation_id,
                exc,
            )

    return messages


def add_message(conversation_id: str, role: str, content: str) -> None:
    """
    Append one message to a conversation.

    Each message is stored as a JSON string in a Redis List.
    The TTL is refreshed on every write so active conversations
    don't expire mid-session.
    """
    if not _conversation_exists(conversation_id):
        raise KeyError(conversation_id)

    key = _make_conversation_key(conversation_id)
    message = json.dumps({"role": role, "content": content})

    try:
        # RPUSH appends to the right end — preserves chronological order
        redis_client.rpush(key, message)
        # Refresh TTL so active conversations stay alive
        redis_client.expire(key, CONVERSATION_TTL_SECONDS)
        # Also refresh the meta key TTL
        redis_client.expire(
            f"conversation_meta:{conversation_id}",
            CONVERSATION_TTL_SECONDS,
        )
        logger.debug(
            "Message added | conversation_id=%s | role=%s", conversation_id, role
        )
    except redis.RedisError as exc:
        logger.error(
            "Message add failed | conversation_id=%s | error=%s",
            conversation_id,
            exc,
        )
        raise


def delete_conversation(conversation_id: str) -> None:
    """
    Delete a conversation and its meta key.

    Raises KeyError if the conversation does not exist.
    """
    if not _conversation_exists(conversation_id):
        raise KeyError(conversation_id)

    try:
        redis_client.delete(_make_conversation_key(conversation_id))
        redis_client.delete(f"conversation_meta:{conversation_id}")
        logger.debug("Conversation deleted | conversation_id=%s", conversation_id)
    except redis.RedisError as exc:
        logger.error(
            "Conversation delete failed | conversation_id=%s | error=%s",
            conversation_id,
            exc,
        )
        raise
