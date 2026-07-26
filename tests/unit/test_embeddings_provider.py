"""
Testy providera embeddingów — bez wczytywania prawdziwego modelu.

Sens: pominięty albo pomylony prefiks E5 nie powoduje żadnego błędu. Wszystko
działa, tylko retrieval jest gorszy, i to o kilka punktów. Taki błąd jest
niewykrywalny bez pomiaru, więc musi go pilnować test.
"""

from __future__ import annotations

import asyncio

import pytest

from tsl_rag.core.embeddings import LocalSentenceTransformerEmbeddings
from tsl_rag.core.settings import Settings

pytestmark = pytest.mark.unit


class _StubModel:
    """Podstawka pod SentenceTransformer — zapamiętuje, co dostała do zakodowania."""

    def __init__(self, dimensions: int = 4) -> None:
        self.seen: list[list[str]] = []
        self._dimensions = dimensions

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimensions

    def encode(self, texts, **_: object):
        self.seen.append(list(texts))
        return [[0.1] * self._dimensions for _ in texts]


def _provider(**overrides) -> tuple[LocalSentenceTransformerEmbeddings, _StubModel]:
    settings = Settings(
        postgres_dsn="postgresql+asyncpg://u:p@localhost:5433/db",
        embedding_provider="local",
        local_embed_model="stub/model",
        local_embed_dimensions=4,
        **overrides,
    )
    provider = LocalSentenceTransformerEmbeddings(settings)
    stub = _StubModel()
    provider._model = stub
    return provider, stub


def test_query_gets_query_prefix():
    provider, stub = _provider()
    asyncio.run(provider.embed_query("ile godzin"))
    assert stub.seen == [["query: ile godzin"]]


def test_documents_get_passage_prefix():
    provider, stub = _provider()
    asyncio.run(provider.embed_documents(["Artykuł 6", "Artykuł 7"]))
    assert stub.seen == [["passage: Artykuł 6", "passage: Artykuł 7"]]


def test_prefixes_can_be_disabled_for_models_that_dont_use_them():
    """
    bge-m3 nie używa prefiksów. Porównanie go z E5 bez wyzerowania prefiksów
    mierzyłoby dwie różne rzeczy naraz.
    """
    provider, stub = _provider(local_embed_query_prefix="", local_embed_passage_prefix="")
    asyncio.run(provider.embed_query("ile godzin"))
    asyncio.run(provider.embed_documents(["Artykuł 6"]))
    assert stub.seen == [["ile godzin"], ["Artykuł 6"]]


def test_dimension_mismatch_raises_on_load():
    """
    Niezgodność wymiarów modelu z konfiguracją musi paść przy ładowaniu,
    a nie przy zapisie do kolumny vector(n) w środku ingestu.
    """
    settings = Settings(
        postgres_dsn="postgresql+asyncpg://u:p@localhost:5433/db",
        embedding_provider="local",
        local_embed_model="stub/model",
        local_embed_dimensions=1024,
    )
    provider = LocalSentenceTransformerEmbeddings(settings)
    provider._model = None

    class _Loader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return 768

    import sentence_transformers

    original = sentence_transformers.SentenceTransformer
    sentence_transformers.SentenceTransformer = _Loader  # type: ignore[misc, assignment]
    try:
        with pytest.raises(ValueError, match="768 wymiarów"):
            provider._load()
    finally:
        sentence_transformers.SentenceTransformer = original  # type: ignore[misc]
