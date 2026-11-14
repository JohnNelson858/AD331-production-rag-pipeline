import logging
import os
import secrets
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .preview import EXAMPLES, preview
from .schemas import RAGResponse, Turn

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger(__name__)
live_lock = Lock()

app = FastAPI(title="Harbor Cloud RAG")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class QueryRequest(BaseModel):
    question: str = Field(max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=3)
    mode: Literal["preview", "live"] = "preview"


def live_settings() -> tuple[str, str, str]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    tavily = os.getenv("TAVILY_API_KEY", "").strip()
    return (
        key if key != "your_openai_key" else "",
        tavily if tavily != "your_tavily_key" else "",
        os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


@lru_cache(maxsize=1)
def live_pipeline():
    from openai import OpenAI
    from tavily import TavilyClient

    from .pipeline import Pipeline

    key, tavily, model = live_settings()
    return Pipeline(ROOT / "docs", OpenAI(api_key=key), model, TavilyClient(api_key=tavily))


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(
        STATIC / "index.html",
        headers={"Content-Security-Policy": "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"},
    )


@app.get("/healthz")
def health():
    return {"status": "ok"}


def access_required() -> bool:
    return os.getenv("RENDER", "").lower() == "true" or os.getenv("REQUIRE_LIVE_ACCESS") == "1"


@app.get("/api/config")
def config():
    key, tavily, _ = live_settings()
    required = access_required()
    return {
        "live_available": bool(key and tavily and (not required or os.getenv("LIVE_ACCESS_CODE"))),
        "live_access_required": required,
        "examples": EXAMPLES,
    }


@app.post("/api/query")
def ask(data: QueryRequest, x_live_access: str = Header(default="")):
    question = data.question.strip()
    if not question:
        raise HTTPException(422, "Enter a question before running the pipeline.")
    start = time.perf_counter()
    if data.mode == "preview":
        try:
            result = preview(question, data.history)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
    else:
        if access_required():
            code = os.getenv("LIVE_ACCESS_CODE", "")
            if not code:
                raise HTTPException(503, "Live access is not configured on this server.")
            if not secrets.compare_digest(x_live_access.encode(), code.encode()):
                raise HTTPException(403, "Enter the live access code to run a provider request.")
        key, tavily, _ = live_settings()
        if not key or not tavily:
            raise HTTPException(503, "Live mode needs OpenAI and Tavily keys on the server.")
        if not live_lock.acquire(blocking=False):
            raise HTTPException(429, "The live pipeline is busy. Try again after the current run.")
        try:
            result = live_pipeline().ask(question, data.history)
        except Exception as exc:
            log.warning("Live pipeline failed: %s", type(exc).__name__)
            raise HTTPException(502, "The live pipeline could not complete. Check the server and provider settings.") from None
        finally:
            live_lock.release()

    return {"mode": data.mode, "elapsed_ms": round((time.perf_counter() - start) * 1000),
            "result": RAGResponse.model_validate(result).model_dump()}
