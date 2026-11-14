from pathlib import Path
from typing import Sequence

from openai import OpenAI
from tavily import TavilyClient

from .generator import generate
from .reranker import THRESHOLD, Reranker, confidence, route_for_score
from .retriever import Retriever
from .rewriter import rewrite
from .schemas import RAGResponse, RetrievedChunk, Turn
from .web_fallback import search_web


class Pipeline:
    def __init__(
        self, docs: Path, openai: OpenAI, model: str, tavily: TavilyClient | None
    ) -> None:
        self.retriever = Retriever(docs)
        self.reranker = Reranker()
        self.openai = openai
        self.model = model
        self.tavily = tavily

    def ask(self, question: str, history: Sequence[Turn] = ()) -> RAGResponse:
        query, rewrite_status = rewrite(question, history, self.openai, self.model)
        candidates = self.retriever.search(query, k=20)
        ranked = self.reranker.rank(query, candidates, k=5)
        score = confidence(ranked)
        web: list[RetrievedChunk] = []
        route = route_for_score(score)
        if route == "INTERNAL_RAG":
            evidence = ranked[:3]
        else:
            web = search_web(query, self.tavily) if self.tavily is not None else []
            evidence = web
        answer, citations = generate(query, evidence, self.openai, self.model)
        return RAGResponse(
            answer=answer,
            citations=citations,
            route_taken=route,
            confidence_score=score,
            threshold=THRESHOLD,
            transformed_query=query,
            rewrite_status=rewrite_status,
            stage1=candidates,
            stage2=ranked,
            web_sources=web,
        )
