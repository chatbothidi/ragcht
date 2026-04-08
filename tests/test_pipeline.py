from src.memory import ConversationMemory


def test_conversation_memory_local():
    """Test conversation memory with in-memory store."""
    memory = ConversationMemory(redis_url=None, max_turns=3)

    session_id = memory.create_session()
    assert memory.get_history(session_id) == []

    memory.add_turn(session_id, "안녕하세요", "안녕하세요! 무엇을 도와드릴까요?")
    history = memory.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "model"

    # Test max turns
    for i in range(5):
        memory.add_turn(session_id, f"질문 {i}", f"답변 {i}")

    history = memory.get_history(session_id)
    assert len(history) == 6  # max_turns=3, so 3*2=6 messages

    # Test clear
    memory.clear_session(session_id)
    assert memory.get_history(session_id) == []
