import json
import uuid

from redis.asyncio import Redis


class ConversationMemory:
    def __init__(self, redis_client: Redis, max_turns: int = 5, ttl: int = 1800):
        if redis_client is None:
            raise ValueError(
                "ConversationMemory requires a Redis client. "
                "Set REDIS_URL and pass a redis.asyncio.Redis instance."
            )
        self.redis = redis_client
        self.max_turns = max_turns
        self.ttl = ttl  # 30 minutes default

    def _key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    def create_session(self) -> str:
        return str(uuid.uuid4())

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Get conversation history for a session."""
        data = await self.redis.get(self._key(session_id))
        if data:
            return json.loads(data)
        return []

    async def add_turn(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Add a conversation turn (user + assistant)."""
        history = await self.get_history(session_id)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "model", "content": assistant_message})

        max_messages = self.max_turns * 2
        if len(history) > max_messages:
            history = history[-max_messages:]

        await self.redis.setex(
            self._key(session_id),
            self.ttl,
            json.dumps(history, ensure_ascii=False),
        )

    async def clear_session(self, session_id: str) -> None:
        """Clear a session's conversation history."""
        await self.redis.delete(self._key(session_id))
