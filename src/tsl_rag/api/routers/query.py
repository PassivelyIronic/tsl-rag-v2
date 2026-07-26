from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from tsl_rag.core.llm_client import get_embedding, get_llm_client
from tsl_rag.core.models import QueryResponse, RetrievalRequest
from tsl_rag.core.settings import Settings, get_settings
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000)
    top_k: int = Field(default=20, ge=1, le=50)
    rerank_top_n: int = Field(default=5, ge=1, le=20)
    filter_document_type: str | None = None
    filter_contains_penalty: bool | None = None
    debug: bool = False  # zwraca raw chunks w odpowiedzi


class HealthResponse(BaseModel):
    status: str
    postgres: str
    ollama: str


async def get_retriever() -> HybridRetriever:
    """Dependency: tworzy i zwraca HybridRetriever."""
    retriever = HybridRetriever()
    await retriever.__aenter__()
    return retriever


@router.post("", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    """
    Główny endpoint RAG.
    Przyjmuje pytanie, zwraca odpowiedź z cytowaniami.
    """
    from tsl_rag.core.models import DocumentType

    doc_type = None
    if request.filter_document_type:
        try:
            doc_type = DocumentType(request.filter_document_type)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid document_type '{request.filter_document_type}'. "
                f"Valid: {[e.value for e in DocumentType]}",
            ) from None

    retrieval_request = RetrievalRequest(
        query=request.query,
        top_k=request.top_k,
        rerank_top_n=request.rerank_top_n,
        filter_document_type=doc_type,
        filter_contains_penalty=request.filter_contains_penalty,
    )

    async with HybridRetriever() as retriever:
        results = await retriever.retrieve(retrieval_request)

    if not results:
        logger.warning(f"No retrieval results for query: '{request.query[:60]}'")

    generator = RAGGenerator()
    response = await generator.generate(request.query, results)
    if request.debug:
        from tsl_rag.core.models import DocumentChunk, RetrievedChunk

        response.retrieved_chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=r.chunk.chunk_id,  # type: ignore[arg-type]
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


@router.get("/health", response_model=HealthResponse)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Sprawdza połączenia z zależnościami."""
    postgres_status = "ok"
    ollama_status = "ok"

    try:
        raw_dsn = str(settings.postgres_dsn).replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn=raw_dsn)
        await conn.fetchval("SELECT 1")
        await conn.close()
    except Exception as e:
        postgres_status = f"error: {e}"

    try:
        client = get_llm_client(settings)
        await get_embedding("health check", settings, client)
    except Exception as e:
        ollama_status = f"error: {e}"

    overall = "ok" if postgres_status == "ok" and ollama_status == "ok" else "degraded"
    return HealthResponse(status=overall, postgres=postgres_status, ollama=ollama_status)


@router.get("/documents")
async def get_documents() -> dict[str, str]:
    """
    Zwraca listę obsługiwanych dokumentów.
    Dzięki temu UI nie musi mieć ich zaszytych na sztywno.
    """
    from tsl_rag.ingestion.cli import DOCUMENT_REGISTRY

    # Tworzymy prosty słownik { "ID_DOKUMENTU": "Tytuł dokumentu" }
    return {doc_id: meta["title"] for doc_id, meta in DOCUMENT_REGISTRY.items()}
