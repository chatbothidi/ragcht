from src.memory import ConversationMemory


async def test_conversation_memory_redis(fake_redis):
    """Test conversation memory backed by Redis (fakeredis)."""
    memory = ConversationMemory(redis_client=fake_redis, max_turns=3)

    session_id = memory.create_session()
    assert await memory.get_history(session_id) == []

    await memory.add_turn(session_id, "안녕하세요", "안녕하세요! 무엇을 도와드릴까요?")
    history = await memory.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "model"

    for i in range(5):
        await memory.add_turn(session_id, f"질문 {i}", f"답변 {i}")

    history = await memory.get_history(session_id)
    assert len(history) == 6  # max_turns=3 → 6 messages

    await memory.clear_session(session_id)
    assert await memory.get_history(session_id) == []


async def test_memory_session_isolation(fake_redis):
    """Concurrent-style sessions must not leak history into each other."""
    memory = ConversationMemory(redis_client=fake_redis, max_turns=5)

    s1 = memory.create_session()
    s2 = memory.create_session()

    await memory.add_turn(s1, "세션1 질문", "세션1 답변")
    await memory.add_turn(s2, "세션2 질문", "세션2 답변")

    h1 = await memory.get_history(s1)
    h2 = await memory.get_history(s2)

    assert h1[0]["content"] == "세션1 질문"
    assert h2[0]["content"] == "세션2 질문"
    assert h1 != h2


def test_memory_requires_redis():
    """ConversationMemory should fail-fast when Redis client is missing."""
    import pytest

    with pytest.raises(ValueError):
        ConversationMemory(redis_client=None)
