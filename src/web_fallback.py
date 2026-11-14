from urllib.parse import urlsplit

from tavily import TavilyClient

from .schemas import RetrievedChunk


def search_web(query: str, client: TavilyClient) -> list[RetrievedChunk]:
    response = client.search(query=query, max_results=3, search_depth="basic", timeout=8.0)
    chunks = []
    seen = set()
    for row in response.get("results", [])[:3]:
        url = str(row.get("url") or "").strip()
        content = str(row.get("content") or "").strip()
        try:
            parsed = urlsplit(url)
            valid = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not any(c.isspace() for c in url)
            )
        except ValueError:
            valid = False
        if not valid or not content or url in seen:
            continue
        seen.add(url)
        chunks.append(
            RetrievedChunk(
                chunk_id=f"web-{len(chunks) + 1}",
                title=str(row.get("title") or "Web result")[:200],
                content=content[:1800],
                source="web",
                location=url,
            )
        )
    return chunks
