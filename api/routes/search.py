from fastapi import APIRouter, Depends

from api.dependencies import get_retriever
from src.hybrid_retriever import HybridRetriever
from src.models import SearchRequest, SearchResponseModel

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponseModel)
async def search(
    request: SearchRequest,
    retriever: HybridRetriever = Depends(get_retriever),
):
    results = await retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        source_type_filter=request.source_type_filter,
    )

    return SearchResponseModel(
        results=results,
        query=request.query,
        total=len(results),
    )
