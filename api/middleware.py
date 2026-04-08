import time

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = structlog.get_logger()


def setup_middleware(app: FastAPI) -> None:
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error("request_failed", path=request.url.path, error=str(e))
            return JSONResponse(
                status_code=500,
                content={"detail": "내부 서버 오류가 발생했습니다."},
            )

        duration = time.time() - start
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration=round(duration, 3),
        )
        return response
