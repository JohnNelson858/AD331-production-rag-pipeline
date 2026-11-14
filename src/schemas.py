from typing import Literal

from pydantic import BaseModel, Field


class Turn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    answer: str = Field(default="", max_length=1500)


class RetrievedChunk(BaseModel):
    chunk_id: str
    title: str
    content: str
    source: Literal["internal_doc", "web"]
    location: str
    dense_rank: int | None = None
    distance: float | None = None
    rerank_logit: float | None = None
    score: float | None = None
    rerank_rank: int | None = None


class RAGResponse(BaseModel):
    answer: str
    citations: list[str]
    route_taken: Literal["INTERNAL_RAG", "WEB_FALLBACK"]
    confidence_score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    transformed_query: str
    rewrite_status: Literal["unchanged", "rewritten", "fallback"]
    stage1: list[RetrievedChunk]
    stage2: list[RetrievedChunk]
    web_sources: list[RetrievedChunk]
