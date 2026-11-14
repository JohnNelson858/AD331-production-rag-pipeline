"""Offline Step 3 smoke check. Run as a module from the project root."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.generator import ABSTAIN, validate_synthesis
from src.reranker import route_for_score, sigmoid
from src.retriever import corpus_fingerprint, read_chunks
from src.rewriter import keeps_protected_terms
from src.schemas import RetrievedChunk


def main() -> None:
    assert sigmoid(-2) < 0.5 < sigmoid(2)
    assert route_for_score(0.42) == "WEB_FALLBACK"
    assert route_for_score(0.55) == route_for_score(0.81) == "INTERNAL_RAG"

    question = "Does 3.4 support network_mode with --dry-run for INC-42?"
    assert keeps_protected_terms(
        question, "Does version 3.4 support network_mode with --dry-run for INC-42?"
    )
    assert not keeps_protected_terms(question, question.replace("INC-42", "INC-420"))
    assert not keeps_protected_terms(question, question.replace(" --dry-run", ""))
    assert not keeps_protected_terms(question, question.replace("3.4", "3.4.1"))

    with TemporaryDirectory() as folder:
        path = Path(folder) / "policy.md"
        path.write_text("# Policy\n\n## Retention\nKeep daily backups for seven days.\n")
        first = read_chunks(Path(folder))
        assert len(first) == 1 and corpus_fingerprint(first) == corpus_fingerprint(read_chunks(Path(folder)))
        path.write_text("# Policy\n\n## Retention\nKeep daily backups for eight days.\n")
        second = read_chunks(Path(folder))
        assert first[0].chunk_id == second[0].chunk_id
        assert corpus_fingerprint(first) != corpus_fingerprint(second)

    internal = RetrievedChunk(
        chunk_id="doc-backups", title="Backups", content="Keep daily backups for seven days.",
        source="internal_doc", location="docs/backup_and_recovery.md"
    )
    web = RetrievedChunk(
        chunk_id="web-1", title="Documentation", content="The example runs in read-only mode.",
        source="web", location="https://example.com/docs"
    )
    good = {"text": "The example runs in read-only mode.", "source_id": 2,
            "quote": "The example runs in read-only mode."}

    def check(*claims: dict) -> tuple[str, list[str]]:
        return validate_synthesis(json.dumps({"claims": list(claims)}), [internal, web])

    assert check(good) == ("The example runs in read-only mode. [1]", [web.location])
    assert check({**good, "source_id": 1}) == (ABSTAIN, [])
    assert check({**good, "source_id": 3}) == (ABSTAIN, [])
    assert check({**good, "quote": "This passage was made up."}) == (ABSTAIN, [])
    assert check({**good, "text": "[1]"}) == (ABSTAIN, [])
    assert check({**good, "text": "<b>Output</b>"}) == (ABSTAIN, [])
    print("Step 3 checks passed: sigmoid and routing, rewrite terms, corpus hash, citation validation.")


if __name__ == "__main__":
    main()
