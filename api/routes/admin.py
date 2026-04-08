from fastapi import APIRouter, Depends

from api.dependencies import get_memory
from src.memory import ConversationMemory

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.delete("/session/{session_id}")
async def clear_session(
    session_id: str,
    memory: ConversationMemory = Depends(get_memory),
):
    memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
