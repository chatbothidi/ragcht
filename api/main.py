import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_pipeline, get_redis
from api.middleware import setup_middleware
from api.routes import admin, chat, search

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = get_redis()
    pong = await redis.ping()
    logger.info("Redis connection OK (ping=%s)", pong)

    logger.info("Initializing pipeline components...")
    pipeline = get_pipeline()
    logger.info("Warming up (first query)...")
    try:
        await pipeline.query(question="테스트", session_id="warmup")
    except Exception:
        pass
    logger.info("Pipeline ready.")
    yield

    await redis.aclose()


app = FastAPI(
    title="Medical & Event RAG Chatbot",
    description="의료 문서와 행사 문서 기반 RAG 챗봇 API",
    version="1.0.0",
    lifespan=lifespan,
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
