import hashlib
import json
import os
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .schemas import RetrievedChunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def read_chunks(directory: Path) -> list[RetrievedChunk]:
    chunks = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8-sig")
        parts = re.split(r"^##[ \t]+(.+)$", text, flags=re.MULTILINE)
        title = parts[0].splitlines()[0].lstrip("# ") if parts[0].splitlines() else path.stem
        for heading, body in zip(parts[1::2], parts[2::2]):
            if not body.strip():
                continue
            digest = hashlib.sha256(f"{path.name}\n{heading}".encode()).hexdigest()[:16]
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"doc-{digest}",
                    title=f"{title} / {heading.strip()}",
                    content=f"{title}\n{heading.strip()}\n{body.strip()}",
                    source="internal_doc",
                    location=f"docs/{path.name} — {heading.strip()}",
                )
            )
    return chunks


def corpus_fingerprint(chunks: list[RetrievedChunk]) -> str:
    records = sorted((c.chunk_id, c.title, c.content, c.location) for c in chunks)
    payload = json.dumps(["markdown-v1-cosine", EMBEDDING_MODEL, records], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Retriever:
    def __init__(self, directory: Path) -> None:
        chunks = read_chunks(directory)
        if not chunks:
            raise ValueError(f"No Markdown sections found in {directory}.")
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self.by_id) != len(chunks):
            raise ValueError("Duplicate Markdown section headings in a document.")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        cache_dir = Path(os.getenv("CHROMA_DIR") or str(directory.parent / ".chroma"))
        self.client = chromadb.PersistentClient(path=str(cache_dir))
        fingerprint = corpus_fingerprint(chunks)
        self.collection = self.client.get_or_create_collection(
            name=f"harbor-{fingerprint[:40]}",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
        )
        indexed = set(self.collection.get(include=[])["ids"])
        if indexed != set(self.by_id):
            embeddings = self.encoder.encode(
                [chunk.content for chunk in chunks], normalize_embeddings=True
            ).tolist()
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.content for chunk in chunks],
                metadatas=[{"title": chunk.title, "location": chunk.location} for chunk in chunks],
            )
            stale = indexed - set(self.by_id)
            if stale:
                self.collection.delete(ids=sorted(stale))

    def search(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        embedding = self.encoder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=min(k, len(self.by_id)),
            include=["distances"],
        )
        distances = result["distances"]
        if distances is None:
            raise RuntimeError("Chroma returned no distance data.")
        return [
            self.by_id[chunk_id].model_copy(
                update={"dense_rank": rank, "distance": float(distance)}
            )
            for rank, (chunk_id, distance) in enumerate(
                zip(result["ids"][0], distances[0], strict=True), 1
            )
        ]
