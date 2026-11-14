FROM ghcr.io/astral-sh/uv:0.13.6 AS uv
FROM python:3.11-slim
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    HF_HOME=/var/cache/harbor/huggingface HF_HUB_DISABLE_XET=1 \
    CHROMA_DIR=/var/cache/harbor/chroma REQUIRE_LIVE_ACCESS=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY docs ./docs
RUN useradd --create-home --uid 10001 harbor \
    && mkdir -p /var/cache/harbor \
    && chown harbor:harbor /var/cache/harbor
USER harbor
EXPOSE 8000
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
