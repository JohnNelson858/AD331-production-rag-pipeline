from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def model_client():
    """Mock the provider at the completion boundary, keeping application code real."""
    client = Mock()
    client.with_options.return_value = client

    def respond(content, finish_reason="stop"):
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )]
        )

    return client, respond
