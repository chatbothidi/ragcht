from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import setup_middleware
from api.routes import admin, chat, search

app = FastAPI(
    title="Medical & Event RAG Chatbot",
    description="의료 문서와 행사 문서 기반 RAG 챗봇 API",
    version="1.0.0",
)

setup_middleware(app)

app.include_router(chat.router)
app.include_router(search.router)
app.include_router(admin.router)

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))
