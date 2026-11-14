from unittest.mock import Mock

import pytest

from src.reranker import Reranker, confidence, route_for_score, sigmoid
from src.schemas import RetrievedChunk


@pytest.mark.parametrize(("logit", "expected"), [(-2, 0.119202922), (0, 0.5), (2, 0.880797078)])
def test_sigmoid_logit_normalization(logit, expected):
    actual = sigmoid(logit)
    assert 0 < actual < 1
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(("score", "route"), [
    (0.42, "WEB_FALLBACK"), (0.55, "INTERNAL_RAG"), (0.81, "INTERNAL_RAG"),
])
def test_threshold_routes_top_score_including_boundary(score, route):
    chunk = RetrievedChunk(chunk_id="doc-1", title="Policy", content="Evidence",
                           source="internal_doc", location="docs/policy.md", score=score)
    assert route_for_score(confidence([chunk])) == route


def test_reranking_reorders_results_without_losing_dense_rank():
    candidates = [
        RetrievedChunk(chunk_id=f"doc-{i}", title=f"Document {i}", content=f"Content {i}",
                       source="internal_doc", location=f"docs/{i}.md", dense_rank=i)
        for i in range(1, 4)
    ]
    reranker = Reranker.__new__(Reranker)
    reranker.model = Mock()
    reranker.model.predict.return_value = [-2.0, 2.0, 0.0]
    ranked = reranker.rank("question", candidates, k=2)
    assert [(r.chunk_id, r.dense_rank, r.rerank_rank) for r in ranked] == [
        ("doc-2", 2, 1), ("doc-3", 3, 2)
    ]
    assert confidence(ranked) == pytest.approx(0.880797078)
    assert candidates[1].score is None
