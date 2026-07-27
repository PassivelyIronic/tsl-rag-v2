from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from tsl_rag.api.routers.health import router as health_router
from tsl_rag.api.routers.query import router as query_router
from tsl_rag.core.llm_client import resolve_chat_chain
from tsl_rag.core.logging import configure_logging
from tsl_rag.core.settings import get_settings
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Jeden HybridRetriever na proces, tworzony przy starcie.

    Wcześniej każdy request budował go od zera: nowy pool asyncpg, ponowne
    wczytanie całego korpusu z bazy, przebudowa indeksu BM25 i inicjalizacja
    cross-encodera. Komentarz w retriever.py obiecywał "budowane raz przy
    starcie", co w ścieżce HTTP nie było prawdą.

    Awaria bazy przy starcie NIE wywala procesu — retriever zostaje None,
    /ready zwraca 503, a /query komunikat po polsku. Proces, który nie wstaje
    przy chwilowo niedostępnej bazie, jest gorszy dla probe'ów niż proces
    raportujący swój stan.
    """
    settings = get_settings()
    configure_logging(settings)
    app.state.retriever = None

    retriever = HybridRetriever()
    try:
        await retriever.__aenter__()
        await retriever.warmup()
        app.state.retriever = retriever
        logger.info("Retriever gotowy (pool + indeks BM25 + cross-encoder)")
    except Exception as exc:
        logger.error(f"Nie udało się zainicjalizować retrievera: {exc}")

    # Raz na proces, bo trzyma stan bezpiecznika łańcucha fallbacku.
    # Tworzenie per request zerowałoby licznik porażek per ogniwo.
    app.state.generator = RAGGenerator(settings)

    chain = resolve_chat_chain(settings)
    logger.info(
        f"App start | env={settings.app_env} | "
        f"embedding={settings.embedding_provider} | "
        f"chat={settings.chat_provider} | model={settings.active_llm_model} | "
        f"łańcuch fallbacku: {' → '.join(str(t) for t in chain)}"
    )

    yield

    if app.state.retriever is not None:
        await retriever.__aexit__(None, None, None)
        logger.info("Retriever zamknięty")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="TSL-RAG API",
        description="EU Transport & Logistics compliance RAG system",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(query_router)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"service": "tsl-rag", "status": "running"}

    return app


app = create_app()
