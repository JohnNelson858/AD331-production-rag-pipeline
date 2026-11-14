# Step 5: Pipeline QA

Run from the project root: `uv run --no-sync python -m pytest -v`.

The suite exercises local application logic. It substitutes fixed responses at the OpenAI, Tavily and retrieval model boundaries. The pronoun test checks that the rewriter sends the right history to OpenAI and accepts a valid standalone rewrite. Routing tests exercise the real branching code with controlled scores. The untrusted-source test verifies that the snippet remains data and that a simulated unsupported output is rejected; it does not measure how a live model reacts to arbitrary instructions or prove semantic grounding.

The live workbench still needs an internal query, a contextual follow-up, and an out-of-domain question in **Live providers** mode. Record actual route, score and citations for each. Offline preview values and mocked scores cannot be used to calibrate the 0.55 threshold.
