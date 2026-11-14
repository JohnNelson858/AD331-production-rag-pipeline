import json

import httpx
import pytest
from openai import APITimeoutError

from src.rewriter import rewrite
from src.schemas import Turn

HISTORY = [Turn(question="Tell me about Kubernetes Worker Nodes.")]
FOLLOW_UP = "What are its system requirements?"


def test_conversational_pronoun_resolution(model_client):
    client, respond = model_client
    respond("What are the system requirements for Kubernetes Worker Nodes?")
    result, status = rewrite(FOLLOW_UP, HISTORY, client, "test-model")
    assert status == "rewritten"
    assert "Kubernetes Worker Nodes" in result
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == FOLLOW_UP
    assert "Kubernetes Worker Nodes" in payload["history"][0]["question"]


def test_timeout_keeps_original_question(model_client):
    client, _ = model_client
    client.chat.completions.create.side_effect = APITimeoutError(
        request=httpx.Request("POST", "https://example.test")
    )
    assert rewrite(FOLLOW_UP, HISTORY, client, "test-model") == (FOLLOW_UP, "fallback")
    client.with_options.assert_called_once_with(timeout=7.0, max_retries=0)


def test_rewrite_that_loses_exact_flags_and_versions_is_rejected(model_client):
    client, respond = model_client
    original = "Can it use --dry-run with version 3.4 for INC-42?"
    respond("Can Kubernetes Worker Nodes use dry run with version 3.4.1 for INC-420?")
    assert rewrite(original, HISTORY, client, "test-model") == (original, "fallback")


def test_blank_input_does_not_call_openai(model_client):
    client, _ = model_client
    with pytest.raises(ValueError, match="Enter a question"):
        rewrite("  ", HISTORY, client, "test-model")
    client.with_options.assert_not_called()
