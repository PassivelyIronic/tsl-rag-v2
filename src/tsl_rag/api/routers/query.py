from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel, Field

from tsl_rag.core.models import (
    DocumentChunk,
    DocumentType,
    QueryResponse,
    RetrievalRequest,
    RetrievedChunk,
)
from tsl_rag.core.observability import stage
from tsl_rag.core.settings import Settings, get_settings
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever

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


@router.post("", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    retriever: Annotated[HybridRetriever, Depends(get_retriever)],
    generator: Annotated[RAGGenerator, Depends(get_generator)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    """
    Główny endpoint RAG.
    Przyjmuje pytanie, zwraca odpowiedź z cytowaniami.
    """
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

    retrieval_request = RetrievalRequest(
        query=request.query,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        filter_document_type=doc_type,
        filter_contains_penalty=request.filter_contains_penalty,
    )

    # Span nadrzędny dla całego zapytania. Bez niego retrieval i generacja
    # trafiałyby do dwóch osobnych śladów, a `trace_id` w logach nie spinałby
    # jednego pytania w całość — czyli bramka Fazy 4 nie byłaby spełniona.
    # Długość zapytania zamiast jego treści: atrybut spanu z pytaniem
    # użytkownika wyciekłby do kolektora razem z danymi, których nie musi znać.
    with stage("query", query_length=len(request.query), debug=request.debug):
        try:
            results = await retriever.retrieve(retrieval_request)
        except Exception as exc:
            logger.error(f"Retrieval nieudany: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=_MSG_NOT_READY,
            ) from exc

        if not results:
            logger.warning(f"Brak wyników retrievalu dla: '{request.query[:60]}'")

        try:
            response = await generator.generate(request.query, results)
        except Exception as exc:
            logger.error(f"Generacja nieudana ({settings.active_llm_model}): {exc}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=_MSG_GENERATION_FAILED,
            ) from exc

    if request.debug:
        response.retrieved_chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=r.chunk.chunk_id,
                    content=r.chunk.text,
                    metadata=r.chunk.metadata,
                ),
                dense_score=r.dense_score,
                bm25_score=r.bm25_score,
                hybrid_score=r.rrf_score,
                rerank_score=r.rerank_score,
            )
            for r in results
        ]

    return response


@router.get("/documents")
async def get_documents() -> dict[str, str]:
    """
    Zwraca listę obsługiwanych dokumentów, żeby UI nie miało ich zaszytych
    na sztywno. Klucze to identyfikatory używane w cytowaniach.
    """
    from tsl_rag.ingestion.cli import DOCUMENT_REGISTRY

    return {doc_id.lower(): meta["title"] for doc_id, meta in DOCUMENT_REGISTRY.items()}
