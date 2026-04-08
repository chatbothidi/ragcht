from pydantic import BaseModel, Field
from dataclasses import dataclass
from typing import Any


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    source_type_filter: str | None = Field(
        None, description="Filter by document type: 'medical' or 'event'"
    )
    stream: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 7
    source_type_filter: str | None = None


class IndexRequest(BaseModel):
    directory: str = "./data/documents"


@dataclass
class SourceCitation:
    source_file: str
    source_type: str
    page_number: int | None
    relevance_score: float
    excerpt: str


@dataclass
class RAGResponse:
    answer: str
    sources: list[SourceCitation]
    query: str
    rewritten_query: str | None
    session_id: str


class ChatResponseModel(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    query: str
    rewritten_query: str | None = None
    session_id: str


class SearchResponseModel(BaseModel):
    results: list[dict[str, Any]]
    query: str
    total: int


@dataclass
class LoadedDocument:
    text: str
    source_file: str
    source_type: str
    pages: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DocumentChunk:
    text: str
    source_file: str
    source_type: str
    chunk_index: int
    page_number: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] | None = None
