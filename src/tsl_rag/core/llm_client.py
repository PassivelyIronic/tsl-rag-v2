"""
Klient generacji odpowiedzi.

Embeddingi mieszkają w `core/embeddings.py` i mają własny przełącznik.
To rozdzielenie jest twardą zasadą repo (CLAUDE.md §3): `embedding_provider`
i `chat_provider` są niezależne, bo embedding musi być najstabilniejszym
elementem układu, a generacja może iść przez darmowy model w chmurze.

Do commita, w którym powstał `core/embeddings.py`, funkcja pobierająca klienta
embeddingów nazywała się `get_llm_client()` i mieszkała tutaj — nazwa sugerowała
model językowy, a chodziło o wektory.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from tsl_rag.core.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_chat_client(settings: Settings) -> AsyncOpenAI:
    """
    Klient dla GENERACJI odpowiedzi (RAGGenerator).
    Sterowany przez `chat_provider`:
    - "ollama"     (lokalny)
    - "openai"     (cloud, płatny)
    - "openrouter" (cloud, darmowe modele ":free" dostępne bez karty)

    Celowo rozdzielony od get_llm_client — embedding i chat mogą teraz
    iść przez różnych providerów jednocześnie (np. embeddingi lokalnie
    przez Ollamę, generacja przez darmowy model na OpenRouter).
    """
    if settings.chat_provider == "ollama":
        return AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",
        )

    if settings.chat_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY nie jest ustawiony")
        return AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key.get_secret_value(),
            # Opcjonalne, ale zalecane przez OpenRouter — identyfikują apkę
            # na openrouter.ai/rankings, nie wpływają na działanie.
            default_headers={
                "HTTP-Referer": "https://github.com/PassivelyIronic/TSL_RAG",
                "X-Title": "TSL_RAG",
            },
        )

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY nie jest ustawiony")

    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    )
