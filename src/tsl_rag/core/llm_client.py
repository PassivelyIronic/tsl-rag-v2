from __future__ import annotations

from openai import AsyncOpenAI

from tsl_rag.core.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_client(settings: Settings) -> AsyncOpenAI:
    """
    Klient dla EMBEDDINGÓW (retrieval, ingest, health-check).
    Sterowany przez `embedding_provider`:
    - "ollama" (lokalny, base_url=localhost:11434/v1)
    - "openai" (cloud)

    Ollama udostępnia endpoint /v1 kompatybilny z OpenAI SDK.
    Zmiana providera = 1 linia w .env, zero zmian w kodzie.
    """
    if settings.embedding_provider == "ollama":
        return AsyncOpenAI(
            base_url=f"{settings.ollama_base_url}/v1",
            api_key="ollama",  # Ollama ignoruje tę wartość, SDK wymaga niepustej
        )

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY nie jest ustawiony")

    return AsyncOpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
    )


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


async def get_embedding(
    text: str,
    settings: Settings,
    client: AsyncOpenAI,
) -> list[float]:
    model = (
        settings.ollama_embed_model
        if settings.embedding_provider == "ollama"
        else settings.openai_embedding_model
    )
    kwargs: dict = {"model": model, "input": text}
    if settings.embedding_provider == "openai":
        kwargs["dimensions"] = settings.openai_embedding_dimensions

    response = await client.embeddings.create(**kwargs)
    return response.data[0].embedding


async def get_embeddings_batch(
    texts: list[str],
    settings: Settings,
    client: AsyncOpenAI,
    batch_size: int = 32,
) -> list[list[float]]:
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        model = (
            settings.ollama_embed_model
            if settings.embedding_provider == "ollama"
            else settings.openai_embedding_model
        )
        kwargs: dict = {"model": model, "input": batch}
        if settings.embedding_provider == "openai":
            kwargs["dimensions"] = settings.openai_embedding_dimensions

        response = await client.embeddings.create(**kwargs)
        all_embeddings.extend([d.embedding for d in response.data])

    return all_embeddings
