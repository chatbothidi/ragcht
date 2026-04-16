from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import get_memory
from src.memory import ConversationMemory

router = APIRouter(prefix="/admin", tags=["admin"])

DOCUMENTS_DIR = Path(__file__).parent.parent.parent / "data" / "documents"


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


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a document file by filename."""
    # Search in all subdirectories
    for file_path in DOCUMENTS_DIR.rglob(filename):
        if file_path.is_file():
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type="application/octet-stream",
            )
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
