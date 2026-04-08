import json
import uuid

import redis


class ConversationMemory:
    def __init__(self, redis_url: str | None = None, max_turns: int = 5, ttl: int = 1800):
        self.max_turns = max_turns
        self.ttl = ttl  # 30 minutes default

        if redis_url:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            self.redis_client = None
            self._local_store: dict[str, list[dict]] = {}

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def create_session(self) -> str:
        return str(uuid.uuid4())

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Get conversation history for a session."""
        if self.redis_client:
            data = self.redis_client.get(self._key(session_id))
            if data:
                return json.loads(data)
            return []
        else:
            return self._local_store.get(session_id, [])

    def add_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        """Add a conversation turn (user + assistant)."""
        history = self.get_history(session_id)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "model", "content": assistant_message})

        # Keep only last N turns (each turn = 2 messages)
        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        if self.redis_client:
            self.redis_client.setex(
                self._key(session_id),
                self.ttl,
                json.dumps(history, ensure_ascii=False),
            )
        else:
            self._local_store[session_id] = history

    def clear_session(self, session_id: str) -> None:
        """Clear a session's conversation history."""
        if self.redis_client:
            self.redis_client.delete(self._key(session_id))
        else:
            self._local_store.pop(session_id, None)
