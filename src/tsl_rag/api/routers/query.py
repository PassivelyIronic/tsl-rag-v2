from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field

from tsl_rag.api.auth import verify_api_key
from tsl_rag.core.documents import DOCUMENT_REGISTRY
from tsl_rag.core.models import DocumentType, QueryResponse
from tsl_rag.core.settings import get_settings
from tsl_rag.generation.cache import AnswerCache
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever
from tsl_rag.service import answer_query

router = APIRouter(prefix="/query", tags=["query"])

# Komunikaty dla użytkownika końcowego — po polsku i mówiące, co zrobić.
# Użytkownik docelowy jest nietechniczny, stacktrace nic mu nie mówi
# (CLAUDE.md §1).
_MSG_NOT_READY = (
    "System nie ma teraz połączenia z bazą dokumentów prawnych. "
    "Odczekaj chwilę i zadaj pytanie ponownie. Jeśli to się powtarza, "
    "baza danych prawdopodobnie nie jest uruchomiona."
)
_MSG_GENERATION_FAILED = (
    "Nie udało się uzyskać odpowiedzi od modelu językowego. "
    "Zwykle znaczy to, że darmowy limit zapytań został chwilowo wyczerpany "
    "albo usługa jest przeciążona. Spróbuj ponownie za kilka minut."
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)
    top_k: int = Field(default_factory=lambda: get_settings().retrieval_top_k, ge=1, le=50)
    rerank_top_n: int = Field(
        default_factory=lambda: get_settings().retrieval_rerank_top_n, ge=1, le=20
    )
    filter_document_type: str | None = None
    filter_contains_penalty: bool | None = None
    debug: bool = False  # zwraca raw chunks w odpowiedzi


def get_retriever(request: Request) -> HybridRetriever:
    """
    Zwraca retriever utworzony raz w lifespanie aplikacji (patrz api/app.py).

    Poprzednia wersja tworzyła nowy HybridRetriever per request, a osobna,
    nieużywana funkcja o tej nazwie wołała __aenter__ bez __aexit__, czyli
    wyciekałaby pool połączeń przy każdym wywołaniu.
    """
    retriever: HybridRetriever | None = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_MSG_NOT_READY,
        )
    return retriever


def get_cache(request: Request) -> AnswerCache | None:
    """Cache z lifespanu. None = wyłączony w konfiguracji."""
    return getattr(request.app.state, "cache", None)


def get_generator(request: Request) -> RAGGenerator:
    """
    Generator utworzony raz na proces, nie per request.

    Powód jest ten sam co przy retrieverze, ale skutek inny: RAGGenerator
    trzyma bezpiecznik łańcucha fallbacku, czyli licznik porażek per ogniwo.
    Nowa instancja przy każdym pytaniu zerowałaby ten licznik, więc bezpiecznik
    nigdy by się nie otworzył i każde zapytanie płaciłoby pełny timeout
    na providerze, o którym już wiadomo, że nie odpowiada.
    """
    generator: RAGGenerator | None = getattr(request.app.state, "generator", None)
    if generator is None:
        # Awaryjnie: brak w lifespanie nie może wywalić zapytania, tylko
        # kosztuje utratę stanu bezpiecznika.
        logger.warning("Brak generatora w app.state — tworzę doraźnie, bezpiecznik bez historii")
        return RAGGenerator()
    return generator


@router.post("", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
async def query_rag(
    request: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    generator: Annotated[RAGGenerator, Depends(get_generator)],
    cache: Annotated[AnswerCache | None, Depends(get_cache)],
) -> QueryResponse:
    """
    Główny endpoint RAG.
    Przyjmuje pytanie, zwraca odpowiedź z cytowaniami.

    `settings` NIE jest wstrzykiwane przez `Depends(get_settings)`. Funkcja jest
    owinięta w `lru_cache`, więc FastAPI nie potrafi jej zintrospektować i traktuje
    adnotację `Settings` (model pydantica) jako POLE CIAŁA żądania. Skutek: ciało
    zostaje zagnieżdżone w `{"request": …, "settings": …}`, a każde normalne
    zapytanie dostaje 422. Nie wychodziło to w testach kontenera, bo sprawdzały
    wyłącznie odpowiedzi 401 — te zapadają przed walidacją ciała.
    """
    settings = get_settings()
    doc_type = None
    if request.filter_document_type:
        try:
            doc_type = DocumentType(request.filter_document_type)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Nieprawidłowy document_type '{request.filter_document_type}'. "
                f"Dopuszczalne: {[e.value for e in DocumentType]}",
            ) from None

    # Logika wspólna z trybem in-process (`tsl_rag.service`), żeby UI na
    # Streamlit Cloud i API nie miały dwóch osobnych, rozjeżdżających się kopii.
    # Router odpowiada wyłącznie za tłumaczenie wyjątków na kody HTTP.
    try:
        return await answer_query(
            request.query,
            retriever=retriever,
            generator=generator,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n,
            filter_document_type=doc_type,
            filter_contains_penalty=request.filter_contains_penalty,
            include_chunks=request.debug,
            cache=cache,
        )
    except HTTPException:
        raise
    except Exception as exc:
        # Rozdzielenie 503 od 502 po stronie retrievalu i generacji straciłoby
        # sens przy wspólnej funkcji, więc rozstrzyga typ wyjątku: problem
        # z bazą to niegotowość, reszta to awaria generacji.
        if isinstance(exc, (asyncpg.PostgresError, OSError)):
            logger.error(f"Retrieval nieudany: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_MSG_NOT_READY,
            ) from exc
        logger.error(f"Generacja nieudana ({settings.active_llm_model}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_MSG_GENERATION_FAILED,
        ) from exc


@router.get("/documents")
async def get_documents() -> dict[str, str]:
    """
    Zwraca listę obsługiwanych dokumentów, żeby UI nie miało ich zaszytych
    na sztywno. Klucze to identyfikatory używane w cytowaniach.
    """
    return {doc_id.lower(): meta["title"] for doc_id, meta in DOCUMENT_REGISTRY.items()}
