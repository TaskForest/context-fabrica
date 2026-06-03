from src.context_fabrica import embedding
from src.context_fabrica.embedding import FastEmbedEmbedder, HashEmbedder, build_default_embedder, chunk_text


def test_chunk_text_splits_long_text() -> None:
    chunks = chunk_text("alpha " * 400, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0


def test_chunk_text_preserves_formatting() -> None:
    chunks = chunk_text("def deploy():\n    return True\n\nclass Worker:\n    pass", max_chars=32, overlap=0)
    assert "\n" in chunks[0].text
    assert " ".join(chunks[0].text.split()) != chunks[0].text


def test_hash_embedder_returns_unit_length_vectors() -> None:
    embedder = HashEmbedder(dimensions=16)
    vector = embedder.embed("AuthService depends on TokenSigner")
    assert len(vector) == 16
    assert round(sum(value * value for value in vector), 5) == 1.0


def test_build_default_embedder_hash_is_explicit() -> None:
    embedder = build_default_embedder(embedder="hash", dimensions=32)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dimensions == 32


def test_fastembed_dimensions_are_inferred(monkeypatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            yield [0.1, 0.2, 0.3, 0.4, 0.5]

    class FakeModule:
        TextEmbedding = FakeTextEmbedding

    monkeypatch.setattr(embedding, "import_module", lambda name: FakeModule)
    embedder = FastEmbedEmbedder(model_name="fake-model")

    assert embedder.dimensions == 5
    assert embedder.embed("hello") == [0.1, 0.2, 0.3, 0.4, 0.5]
