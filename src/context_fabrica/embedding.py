from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import import_module
from math import sqrt
from typing import Literal, Protocol


EmbedderName = Literal["auto", "fastembed", "sentence-transformers", "hash"]
DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 120) -> list[Chunk]:
    normalized = text.strip()
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(normalized):
        target_end = min(len(normalized), start + max_chars)
        end = _chunk_boundary(normalized, start=start, target_end=target_end, max_chars=max_chars)
        chunks.append(Chunk(text=normalized[start:end], chunk_index=index))
        if end == len(normalized):
            break
        next_start = max(end - overlap, start + 1)
        while next_start < len(normalized) and normalized[next_start].isspace():
            next_start += 1
        start = next_start
        index += 1
    return chunks


def _chunk_boundary(text: str, *, start: int, target_end: int, max_chars: int) -> int:
    if target_end >= len(text):
        return len(text)

    window_start = max(start + max_chars // 2, start + 1)
    paragraph = text.rfind("\n\n", window_start, target_end)
    if paragraph != -1:
        return paragraph

    line = text.rfind("\n", window_start, target_end)
    if line != -1:
        return line

    whitespace = text.rfind(" ", window_start, target_end)
    if whitespace != -1:
        return whitespace

    return target_end


class HashEmbedder:
    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        tokens = text.lower().split()
        if not tokens:
            return values

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            magnitude = (digest[5] / 255.0) + 0.1
            values[slot] += sign * magnitude

        norm = sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL) -> None:
        SentenceTransformer = import_module("sentence_transformers").SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> list[float]:
        return list(self._model.encode(text, normalize_embeddings=True))


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL) -> None:
        TextEmbedding = import_module("fastembed").TextEmbedding
        self._model = TextEmbedding(model_name=model_name)
        self.dimensions = len(list(next(self._model.embed(["dimension probe"]))))

    def embed(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))


def build_default_embedder(
    dimensions: int = 384,
    prefer_transformer: bool = False,
    *,
    embedder: EmbedderName = "auto",
    model_name: str | None = None,
) -> Embedder:
    """Build an embedder for local memory retrieval.

    ``auto`` prefers FastEmbed, then sentence-transformers, and only falls
    back to hashing if optional local embedding dependencies are unavailable.
    Pass ``embedder="hash"`` to explicitly request the deterministic fallback.
    """
    if embedder == "hash":
        return HashEmbedder(dimensions=dimensions)

    if embedder in {"auto", "fastembed"}:
        try:
            return FastEmbedEmbedder(model_name=model_name or DEFAULT_FASTEMBED_MODEL)
        except ModuleNotFoundError:
            if embedder == "fastembed":
                raise

    if embedder in {"auto", "sentence-transformers"} or prefer_transformer:
        try:
            return SentenceTransformerEmbedder(model_name=model_name or DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        except ModuleNotFoundError:
            if embedder == "sentence-transformers" or prefer_transformer:
                raise

    return HashEmbedder(dimensions=dimensions)
