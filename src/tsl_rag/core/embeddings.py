"""
Providery embeddingów.

Wydzielone z `llm_client.py`, bo tamten modelował wszystko jako klienta
zgodnego z OpenAI, a lokalny `sentence-transformers` żadnym klientem nie jest —
to model w tym samym procesie. Nazwa `get_llm_client()` dla embeddingów była
przy okazji myląca: sugerowała model językowy, a chodziło o wektory.

Podział na embeddingi i chat pozostaje ostry (CLAUDE.md §3):
- ten moduł        → WYŁĄCZNIE embeddingi (`embedding_provider`)
- `llm_client.py`  → WYŁĄCZNIE generacja (`chat_provider`)

Zapytanie a dokument
--------------------
`embed_query` i `embed_documents` są rozdzielone, bo modele z rodziny E5
wymagają różnych prefiksów ("query: " i "passage: "). Pominięcie ich nie
powoduje błędu — po prostu obniża jakość, i to niewidocznie. To najgorszy
możliwy rodzaj pomyłki w tym miejscu, dlatego prefiksy są jawną konfiguracją,
a nie zgadywane z nazwy modelu.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Protocol

from loguru import logger
from openai import AsyncOpenAI

from tsl_rag.core.settings import Settings, get_settings


class EmbeddingProvider(Protocol):
    """Minimalny kontrakt, jakiego potrzebuje retrieval i ingest."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbeddings:
    """
    Embeddingi przez API zgodne z OpenAI — Ollama (lokalny serwer) albo OpenAI.

    Ollama wystawia /v1 kompatybilne z SDK OpenAI, więc oba przypadki różnią
    się wyłącznie base_url i kluczem.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._is_ollama = settings.embedding_provider == "ollama"

        if self._is_ollama:
            self._client = AsyncOpenAI(
                base_url=f"{settings.ollama_base_url}/v1",
                api_key="ollama",  # Ollama ignoruje wartość, SDK wymaga niepustej
            )
            self._model = settings.ollama_embed_model
            self._dimensions = settings.ollama_embed_dimensions
        else:
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY nie jest ustawiony")
            self._client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
            self._model = settings.openai_embedding_model
            self._dimensions = settings.openai_embedding_dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _kwargs(self, payload: str | list[str]) -> dict:
        kwargs: dict = {"model": self._model, "input": payload}
        if not self._is_ollama:
            kwargs["dimensions"] = self._dimensions
        return kwargs

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(**self._kwargs(text))
        return list(response.data[0].embedding)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(**self._kwargs(texts))
        return [list(d.embedding) for d in response.data]


class LocalSentenceTransformerEmbeddings:
    """
    Embeddingi w tym samym procesie, na CPU, przez `sentence-transformers`.

    To jest ścieżka odcinająca system od lokalnej Ollamy: brak rate limitu,
    brak klucza API w runtime, brak ryzyka, że model zniknie z darmowej puli.
    Embedding zapytania jest liczony przy KAŻDYM pytaniu, więc jest to
    najbardziej wrażliwy na awarię punkt pipeline'u (docs/PROVIDERS.md).

    Model ładowany leniwie, przy pierwszym użyciu, i trzymany na stałe —
    wczytanie wag to sekundy, więc nie może dziać się per zapytanie.
    `encode` jest synchroniczne i obciąża CPU, dlatego leci przez
    `asyncio.to_thread`, żeby nie blokować pętli zdarzeń API.
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.local_embed_model
        self._dimensions = settings.local_embed_dimensions
        self._query_prefix = settings.local_embed_query_prefix
        self._passage_prefix = settings.local_embed_passage_prefix
        self._batch_size = settings.local_embed_batch_size
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Ładowanie modelu embeddingów: {self._model_name} (CPU)")
            self._model = SentenceTransformer(self._model_name, device="cpu")
            # Nazwa metody zmieniła się w nowszych wersjach sentence-transformers;
            # stara wersja jeszcze działa, ale ostrzega przy każdym starcie.
            read_dimension = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            actual = read_dimension()
            if actual != self._dimensions:
                raise ValueError(
                    f"Model {self._model_name} zwraca {actual} wymiarów, a konfiguracja "
                    f"mówi {self._dimensions}. Popraw LOCAL_EMBED_DIMENSIONS — "
                    f"niezgodność wymiarów oznacza błąd zapisu do kolumny vector(n)."
                )
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(v) for v in vector] for vector in vectors]

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._encode, [f"{self._query_prefix}{text}"])
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{self._passage_prefix}{t}" for t in texts]
        return await asyncio.to_thread(self._encode, prefixed)


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """
    Zwraca provider embeddingów wg `embedding_provider`.

    Cache'owane, bo w wariancie lokalnym obiekt trzyma wczytany model.
    Po zmianie konfiguracji w testach wołaj `get_embedding_provider.cache_clear()`.
    """
    settings = get_settings()
    if settings.embedding_provider == "local":
        return LocalSentenceTransformerEmbeddings(settings)
    return OpenAICompatibleEmbeddings(settings)
