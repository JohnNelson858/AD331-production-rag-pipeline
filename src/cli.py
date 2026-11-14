import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from .pipeline import Pipeline
from .schemas import Turn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Harbor Cloud RAG question.")
    parser.add_argument("question")
    parser.add_argument("--previous-question", help="Optional context for a follow-up.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key or key == "your_openai_key":
        parser.error("Set OPENAI_API_KEY locally before running a live request.")
    tv_key = os.getenv("TAVILY_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    tavily = TavilyClient(api_key=tv_key) if tv_key and tv_key != "your_tavily_key" else None
    history = [Turn(question=args.previous_question)] if args.previous_question else []
    pipeline = Pipeline(root / "docs", OpenAI(api_key=key), model, tavily)
    print(pipeline.ask(args.question, history).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
