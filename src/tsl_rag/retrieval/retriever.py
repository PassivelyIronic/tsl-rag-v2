"""
Hybrid retriever: Dense (pgvector) + BM25 (rank-bm25) → RRF fusion → rerank.

Architektura
------------
1. Dense search  — cosine similarity w pgvector (top_k kandydatów)
2. BM25 search   — in-memory rank-bm25 zbudowany ze wszystkich chunków
3. RRF fusion    — Reciprocal Rank Fusion łączy obie listy rankingowe
4. Cross-encoder — reranker na top_n wynikach z RRF

BM25 index jest budowany raz przy starcie (lazy, przy pierwszym zapytaniu).
Dla 200 chunków zajmuje ~2MB RAM — bez problemu dla naszego korpusu.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import asyncpg
from loguru import logger
from rank_bm25 import BM25Okapi

from tsl_rag.core.llm_client import get_embedding, get_llm_client
from tsl_rag.core.models import (
    Chunk,
    DocumentMetadata,
    DocumentType,
    LegalHierarchyLevel,
    RetrievalRequest,
)
from tsl_rag.core.settings import get_settings
from tsl_rag.retrieval.reranker import CrossEncoderReranker

# Typy wewnętrzne


@dataclass
class RetrievalResult:
    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


# RRF constant — standard wartość z literatury
_RRF_K = 60

# SQL do dense search
_DENSE_SQL = """
SELECT
    chunk_id, document_id, document_type, title, jurisdiction,
    chapter, article, paragraph, hierarchy_level,
    contains_table, contains_penalty, is_definition,
    page_start, page_end, text,
    1 - (embedding <=> $1::vector) AS cosine_score
FROM document_chunks
WHERE embedding IS NOT NULL
  AND ($2::text IS NULL OR document_id   = ANY($2::text[]))
  AND ($3::text IS NULL OR document_type = $3)
  AND ($4::bool IS NULL OR contains_penalty = $4)
ORDER BY cosine_score DESC
LIMIT $5;
"""

_ALL_CHUNKS_SQL = """
SELECT
    chunk_id, document_id, document_type, title, jurisdiction,
    chapter, article, paragraph, hierarchy_level,
    contains_table, contains_penalty, is_definition,
    page_start, page_end, text
FROM document_chunks
WHERE embedding IS NOT NULL
ORDER BY chunk_id;
"""


class HybridRetriever:
    """
    Hybrid retriever z lazy-loaded BM25 i cross-encoder rerankerem.

    Usage
    -----
    async with HybridRetriever() as retriever:
        results = await retriever.retrieve(request)
    """

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None
        self._reranker: CrossEncoderReranker | None = None

        # BM25 state — budowane raz, lazy
        self._bm25_index: BM25Okapi | None = None
        self._bm25_chunks: list[Chunk] = []
        self._bm25_lock = asyncio.Lock()

    async def __aenter__(self) -> HybridRetriever:
        settings = get_settings()
        raw_dsn = str(settings.postgres_dsn).replace("postgresql+asyncpg://", "postgresql://")
        self._pool = await asyncpg.create_pool(dsn=raw_dsn, min_size=2, max_size=10)
        self._reranker = CrossEncoderReranker(settings.reranker_model)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._pool:
            await self._pool.close()

    # Public API

    async def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        """
        Główna metoda. Zwraca listę RetrievalResult posortowaną malejąco
        po final_score (rerank jeśli dostępny, RRF w przeciwnym razie).
        """
        settings = get_settings()
        client = get_llm_client(settings)

        # 1. Embed query
        query_embedding = await get_embedding(request.query, settings, client)

        # 2. Dense search (pgvector)
        dense_results = await self._dense_search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filter_doc_ids=request.filter_document_ids,
            filter_doc_type=request.filter_document_type,
            filter_penalty=request.filter_contains_penalty,
        )

        # 3. BM25 search
        bm25_results = await self._bm25_search(
            query=request.query,
            top_k=request.top_k,
        )

        # 4. RRF fusion
        fused = _reciprocal_rank_fusion(dense_results, bm25_results)

        # Bierzemy top_k po fuzji do rerankingu
        candidates = fused[: request.top_k]

        # 5. Cross-encoder rerank
        if candidates and self._reranker:
            candidates = self._apply_rerank(
                query=request.query,
                results=candidates,
                top_n=request.rerank_top_n,
            )

        logger.info(
            f"retrieve('{request.query[:60]}...') → "
            f"dense={len(dense_results)}, bm25={len(bm25_results)}, "
            f"fused={len(fused)}, final={len(candidates)}"
        )
        return candidates

    # Dense search

    async def _dense_search(
        self,
        query_vector: list[float],
        top_k: int,
        filter_doc_ids: list[str] | None,
        filter_doc_type: DocumentType | None,
        filter_penalty: bool | None,
    ) -> list[RetrievalResult]:
        assert self._pool, "Use inside async context manager"

        embedding_str = "[" + ",".join(f"{v:.8f}" for v in query_vector) + "]"
        doc_type_val = filter_doc_type.value if filter_doc_type else None

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _DENSE_SQL,
                embedding_str,
                filter_doc_ids,
                doc_type_val,
                filter_penalty,
                top_k,
            )

        return [
            RetrievalResult(
                chunk=_row_to_chunk(row),
                dense_score=float(row["cosine_score"]),
            )
            for row in rows
        ]

    # BM25 search

    async def _ensure_bm25_index(self) -> None:
        """Buduje BM25 index przy pierwszym wywołaniu (lazy)."""
        async with self._bm25_lock:
            if self._bm25_index is not None:
                return

            assert self._pool, "Use inside async context manager"
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(_ALL_CHUNKS_SQL)

            self._bm25_chunks = [_row_to_chunk(row) for row in rows]
            tokenized = [_tokenize(c.text) for c in self._bm25_chunks]
            self._bm25_index = BM25Okapi(tokenized)
            logger.info(f"BM25 index built: {len(self._bm25_chunks)} chunks")

    async def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        await self._ensure_bm25_index()

        tokens = _tokenize(query)
        scores = self._bm25_index.get_scores(tokens)  # type: ignore[union-attr]

        # Pobierz indeksy top_k malejąco
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=self._bm25_chunks[i],
                bm25_score=float(scores[i]),
            )
            for i in top_indices
            if scores[i] > 0  # pomiń zerowe dopasowania
        ]

    # Rerank

    def _apply_rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int,
    ) -> list[RetrievalResult]:
        texts = [r.chunk.text for r in results]
        ranked = self._reranker.rerank(query, texts, top_n=top_n)  # type: ignore[union-attr]

        # Mapuj wyniki rerankera z powrotem na RetrievalResult
        reranked: list[RetrievalResult] = []
        for r in ranked:
            result = results[r.index]
            result.rerank_score = r.score
            reranked.append(result)

        return reranked


# RRF fusion


def _reciprocal_rank_fusion(
    dense: list[RetrievalResult],
    bm25: list[RetrievalResult],
    k: int = _RRF_K,
) -> list[RetrievalResult]:
    """
    Łączy dwie listy rankingowe przez Reciprocal Rank Fusion.
    score(d) = Σ 1 / (k + rank(d))
    """
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievalResult] = {}

    for rank, result in enumerate(dense):
        cid = result.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        by_id[cid] = result

    for rank, result in enumerate(bm25):
        cid = result.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in by_id:
            by_id[cid] = result
        else:
            # Uzupełnij bm25_score w istniejącym rekordzie
            by_id[cid].bm25_score = result.bm25_score

    # Przypisz rrf_score i posortuj
    for cid, rrf_score in scores.items():
        by_id[cid].rrf_score = rrf_score

    return sorted(by_id.values(), key=lambda r: r.rrf_score, reverse=True)


def _tokenize(text: str) -> list[str]:
    """Prosty tokenizer: lowercase + split po non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _row_to_chunk(row: asyncpg.Record) -> Chunk:
    metadata = DocumentMetadata(
        document_id=row["document_id"],
        document_type=DocumentType(row["document_type"]),
        title=row["title"] or "",
        jurisdiction=row["jurisdiction"] or "EU",
        chapter=row["chapter"],
        article=row["article"],
        paragraph=row["paragraph"],
        hierarchy_level=LegalHierarchyLevel(row["hierarchy_level"])
        if row["hierarchy_level"]
        else LegalHierarchyLevel.PARAGRAPH,
        contains_table=row["contains_table"] or False,
        contains_penalty=row["contains_penalty"] or False,
        is_definition=row["is_definition"] or False,
        page_start=row["page_start"],
        page_end=row["page_end"],
    )
    return Chunk(
        chunk_id=row["chunk_id"],
        text=row["text"],
        metadata=metadata,
    )
