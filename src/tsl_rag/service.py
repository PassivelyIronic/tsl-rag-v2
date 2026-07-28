"""
Wspólna ścieżka zapytania dla API i dla UI w trybie in-process.

Po co osobny moduł: na Streamlit Community Cloud działa JEDEN proces, więc UI
nie ma do czego wysłać żądania HTTP. Bez tej warstwy trzeba by powielić w `ui.py`
budowę `RetrievalRequest`, obsługę pustego retrievalu i doklejanie chunków —
czyli dwie kopie tej samej logiki, które rozjadą się przy pierwszej zmianie,
i to rozjadą się cicho, bo obie „działają".

Rozdział API/UI z `PLAN.md` zostaje w mocy: to jest wspólna funkcja, a nie
zlanie warstw. FastAPI dalej jest osobnym serwisem i to on idzie do K8s;
tryb in-process jest ustępstwem wobec konkretnego darmowego hostingu.
"""

from __future__ import annotations

from loguru import logger

from tsl_rag.core.models import (
    DocumentChunk,
    DocumentType,
    QueryResponse,
    RetrievalRequest,
    RetrievedChunk,
)
from tsl_rag.core.observability import stage
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever


async def answer_query(
    query: str,
    *,
    retriever: HybridRetriever,
    generator: RAGGenerator,
    top_k: int,
    rerank_top_n: int,
    filter_document_type: DocumentType | None = None,
    filter_contains_penalty: bool | None = None,
    include_chunks: bool = False,
) -> QueryResponse:
    """
    Retrieval + generacja dla jednego pytania.

    Span nadrzędny jest tutaj, a nie w routerze, żeby tryb in-process też
    produkował kompletny ślad — inaczej wdrożenie na Streamlicie traciłoby
    całą observability z Fazy 4.

    Długość zapytania zamiast treści jako atrybut spanu: pytanie użytkownika
    nie ma powodu wyciekać do kolektora.
    """
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        filter_document_type=filter_document_type,
        filter_contains_penalty=filter_contains_penalty,
    )

    with stage("query", query_length=len(query)):
        results = await retriever.retrieve(request)
        if not results:
            logger.warning(f"Brak wyników retrievalu dla: '{query[:60]}'")
        response = await generator.generate(query, results)

    if include_chunks:
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
