import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import get_pipeline
from src.models import ChatRequest, ChatResponseModel
from src.rag_pipeline import RAGPipeline

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponseModel)
async def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    if request.stream:
        return StreamingResponse(
            _stream_response(pipeline, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = pipeline.query(
        question=request.query,
        session_id=request.session_id,
        source_type_filter=request.source_type_filter,
    )

    return ChatResponseModel(
        answer=result.answer,
        sources=[
            {
                "source_file": s.source_file,
                "source_type": s.source_type,
                "page_number": s.page_number,
                "relevance_score": s.relevance_score,
                "excerpt": s.excerpt,
            }
            for s in result.sources
        ],
        query=result.query,
        rewritten_query=result.rewritten_query,
        session_id=result.session_id,
    )


async def _stream_response(pipeline: RAGPipeline, request: ChatRequest):
    for event in pipeline.query_stream(
        question=request.query,
        session_id=request.session_id,
        source_type_filter=request.source_type_filter,
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
