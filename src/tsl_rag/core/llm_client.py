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

from dataclasses import dataclass

from openai import AsyncOpenAI

from tsl_rag.core.settings import ChatProvider, Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class ChatTarget:
    """Jedno ogniwo łańcucha fallbacku: provider + konkretny model."""

    provider: ChatProvider
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def resolve_chat_chain(settings: Settings) -> list[ChatTarget]:
    """
    Uporządkowana lista celów: najpierw provider główny, potem zapasowe.

    Pierwsze ogniwo pochodzi zawsze z `chat_provider` + `active_llm_model`,
    więc konfiguracja bez łańcucha zachowuje się dokładnie jak przedtem.
    Kolejne czytane są z `chat_fallback_chain` w formacie
    "provider:model,provider:model".

    Duplikaty są usuwane: powtórzony cel nie dodaje odporności, a wydłuża
    czas do zwrócenia błędu użytkownikowi o kolejny timeout.
    """
    chain = [ChatTarget(settings.chat_provider, settings.active_llm_model)]
    seen = {str(chain[0])}

    for entry in settings.chat_fallback_chain.split(","):
        entry = entry.strip()
        if not entry:
            continue
        provider, _, model = entry.partition(":")
        provider, model = provider.strip(), model.strip()
        if not model:
            raise ValueError(
                f"Nieprawidłowy wpis w CHAT_FALLBACK_CHAIN: {entry!r}. "
                "Oczekiwany format to 'provider:model', np. 'openrouter:openai/gpt-4o-mini'."
            )
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Nieznany provider w CHAT_FALLBACK_CHAIN: {provider!r}. "
                f"Dozwolone: {', '.join(sorted(_SUPPORTED_PROVIDERS))}."
            )
        target = ChatTarget(provider, model)  # type: ignore[arg-type]
        if str(target) not in seen:
            seen.add(str(target))
            chain.append(target)

    return chain


_SUPPORTED_PROVIDERS: frozenset[str] = frozenset({"ollama", "openai", "openrouter", "gemini"})


def get_chat_client_for(provider: ChatProvider, settings: Settings) -> AsyncOpenAI:
    """
    Klient dla WSKAZANEGO providera, niezależnie od `settings.chat_provider`.

    Potrzebne łańcuchowi fallbacku: kolejne ogniwo bywa u innego providera
    niż skonfigurowany główny, a `Settings` jest niemutowalny w ścieżce zapytania.
    """
    if provider == "ollama":
        return AsyncOpenAI(base_url=f"{settings.ollama_base_url}/v1", api_key="ollama")

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY nie jest ustawiony")
        return AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=settings.openrouter_api_key.get_secret_value(),
            default_headers={
                "HTTP-Referer": "https://github.com/PassivelyIronic/TSL_RAG",
                "X-Title": "TSL_RAG",
            },
        )

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY nie jest ustawiony")
        return AsyncOpenAI(
            base_url=GEMINI_OPENAI_BASE_URL,
            api_key=settings.gemini_api_key.get_secret_value(),
        )

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY nie jest ustawiony")
    return AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())


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

    Gemini wystawia warstwę zgodną z OpenAI, więc nie potrzeba osobnego SDK.
    Używany jako MODEL REFERENCYJNY w ewaluacji, nie jako runtime dla
    użytkownika końcowego.
    """
    return get_chat_client_for(settings.chat_provider, settings)
