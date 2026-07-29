"""
Testy integracyjne: wymagają działającego Postgresa z zaindeksowanym korpusem.

Pomijane automatycznie, gdy bazy nie ma — CI nie stawia Postgresa, a test,
który czerwieni się z powodu braku infrastruktury, uczy ignorowania czerwonego CI.

Uruchomienie:  uv run pytest -m integration
"""

from __future__ import annotations

import pytest

from tsl_rag.core.models import RetrievalRequest
from tsl_rag.retrieval.retriever import HybridRetriever

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


# Zasięg funkcji, nie modułu: pytest-asyncio tworzy pętlę zdarzeń per test,
# a pool asyncpg należy do pętli, w której powstał. Fixture modułowa dawała
# "cannot perform operation: another operation is in progress". Model
# embeddingów jest cache'owany globalnie, więc powtórny warmup jest tani.
@pytest.fixture
async def retriever():
    r = HybridRetriever()
    try:
        await r.__aenter__()
        await r.warmup()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Brak bazy albo korpusu: {type(exc).__name__}: {str(exc)[:120]}")
    yield r
    await r.__aexit__(None, None, None)


async def test_corpus_is_indexed(retriever):
    """Korpus ma tyle chunków, ile mówi dokumentacja — 438 z 13 dokumentów."""
    async with retriever._pool.acquire() as conn:
        chunks = await conn.fetchval("SELECT count(*) FROM document_chunks")
        docs = await conn.fetchval("SELECT count(DISTINCT document_id) FROM document_chunks")
    assert chunks == 438
    assert docs == 13


async def test_retrieval_returns_ranked_chunks(retriever):
    results = await retriever.retrieve(
        RetrievalRequest(query="Jaki jest maksymalny dzienny czas prowadzenia pojazdu?")
    )
    assert len(results) == 5  # rerank_top_n
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True), "wyniki muszą być posortowane malejąco"


async def test_expected_document_is_retrieved(retriever):
    """Pytanie o czas jazdy musi wciągnąć rozporządzenie 561/2006."""
    results = await retriever.retrieve(
        RetrievalRequest(query="Jaki jest maksymalny dzienny czas prowadzenia pojazdu?")
    )
    assert "ec_561_2006" in {r.chunk.metadata.document_id for r in results}


async def test_query_without_diacritics_still_matches(retriever):
    """
    Użytkownik nietechniczny często pisze bez ogonków — tokenizer BM25 składa
    diakrytyki po obu stronach, więc obie formy mają trafiać w to samo.
    """
    with_diacritics = await retriever.retrieve(RetrievalRequest(query="przewóz kabotażowy"))
    without = await retriever.retrieve(RetrievalRequest(query="przewoz kabotazowy"))
    assert {r.chunk.chunk_id for r in with_diacritics} & {r.chunk.chunk_id for r in without}
