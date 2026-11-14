import math
from typing import Literal

from sentence_transformers import CrossEncoder
from torch.nn import Identity

from .schemas import RetrievedChunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
THRESHOLD = 0.55


def sigmoid(logit: float) -> float:
    if not math.isfinite(logit):
        raise ValueError("The cross-encoder returned a non-finite logit.")
    exp = math.exp(-abs(logit))
    return 1.0 / (1.0 + exp) if logit >= 0 else exp / (1.0 + exp)


class Reranker:
    def __init__(self) -> None:
        self.model = CrossEncoder(RERANKER_MODEL, device="cpu")

    def rank(self, query: str, candidates: list[RetrievedChunk], k: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []
        logits = self.model.predict(
            [(query, chunk.content) for chunk in candidates],
            activation_fn=Identity(),
            batch_size=8,
            show_progress_bar=False,
        )
        scored = [
            chunk.model_copy(update={"rerank_logit": float(logit), "score": sigmoid(float(logit))})
            for chunk, logit in zip(candidates, logits, strict=True)
        ]
        scored.sort(key=lambda chunk: -(chunk.rerank_logit or 0.0))
        return [
            chunk.model_copy(update={"rerank_rank": rank})
            for rank, chunk in enumerate(scored[:k], 1)
        ]


def confidence(ranked: list[RetrievedChunk]) -> float:
    return (ranked[0].score or 0.0) if ranked else 0.0


def route_for_score(score: float, threshold: float = THRESHOLD) -> Literal["INTERNAL_RAG", "WEB_FALLBACK"]:
    if not (0.0 <= score <= 1.0 and 0.0 <= threshold <= 1.0):
        raise ValueError("Score and threshold must be finite numbers between zero and one.")
    return "INTERNAL_RAG" if score >= threshold else "WEB_FALLBACK"
