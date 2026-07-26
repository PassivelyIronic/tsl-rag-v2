from __future__ import annotations

import json
from collections.abc import Sequence

import asyncpg
from loguru import logger
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from tsl_rag.core.llm_client import get_embeddings_batch, get_llm_client
from tsl_rag.core.models import Chunk
from tsl_rag.core.settings import Settings, get_settings

DEFAULT_BATCH_SIZE = 16

UPSERT_SQL = """
INSERT INTO document_chunks (
    chunk_id, document_id, document_type, title, jurisdiction,
    chapter, article, paragraph, hierarchy_level,
    contains_table, contains_penalty, is_definition,
    page_start, page_end,
    text, embedding, metadata
)
VALUES (
    $1,  $2,  $3,  $4,  $5,
    $6,  $7,  $8,  $9,
    $10, $11, $12,
    $13, $14,
    $15, $16::vector, $17
)
ON CONFLICT (chunk_id) DO UPDATE SET
    text      = EXCLUDED.text,
    embedding = EXCLUDED.embedding,
    metadata  = EXCLUDED.metadata;
"""


class ChunkEmbedder:
    """
    Embeds a list of Chunk objects and persists them to pgvector.

    Usage
    -----
    async with ChunkEmbedder() as embedder:
        stats = await embedder.embed_and_store(chunks)
        print(stats)
    """

    def __init__(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.batch_size = batch_size
        self._pool: asyncpg.Pool | None = None
        self._settings: Settings | None = None  # ← inicjalizowane w __aenter__
        self._client: AsyncOpenAI | None = None  # ← inicjalizowane w __aenter__

    async def __aenter__(self) -> ChunkEmbedder:
        self._settings = get_settings()
        self._client = get_llm_client(self._settings)

        raw_dsn = str(self._settings.postgres_dsn).replace("postgresql+asyncpg://", "postgresql://")
        self._pool = await asyncpg.create_pool(
            dsn=raw_dsn,
            min_size=2,
            max_size=10,
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._pool:
            await self._pool.close()

    async def embed_and_store(self, chunks: Sequence[Chunk]) -> dict:
        if not chunks:
            logger.warning("embed_and_store called with empty chunk list")
            return {"total": 0, "stored": 0, "failed": 0}

        assert self._settings and self._client, "Call inside async context manager"

        batches = _make_batches(list(chunks), self.batch_size)
        stored = 0
        failed = 0

        logger.info(
            f"Embedding {len(chunks)} chunks " f"in {len(batches)} batches (size={self.batch_size})"
        )

        for batch in tqdm(batches, desc="Embedding", unit="batch"):
            texts = [c.text for c in batch]
            try:
                embeddings = await get_embeddings_batch(
                    texts,
                    self._settings,
                    self._client,
                )
            except Exception as exc:
                logger.error(f"Embedding batch failed: {exc}")
                failed += len(batch)
                continue

            if len(embeddings) != len(batch):
                logger.error(
                    f"Embedding count mismatch: " f"got {len(embeddings)}, expected {len(batch)}"
                )
                failed += len(batch)
                continue

            for chunk, emb in zip(batch, embeddings, strict=False):
                chunk.embedding = emb

            n = await self._upsert_batch(batch)
            stored += n
            failed += len(batch) - n

        stats = {"total": len(chunks), "stored": stored, "failed": failed}
        logger.info(f"Embedding complete: {stats}")
        return stats

    # pgvector upsert

    async def _upsert_batch(self, batch: Sequence[Chunk]) -> int:
        assert self._pool is not None, "Call inside async context manager"

        records = [_chunk_to_record(c) for c in batch if c.embedding]
        if not records:
            return 0

        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(UPSERT_SQL, records)
            return len(records)
        except Exception as exc:
            logger.error(f"pgvector upsert failed: {exc}")
            return 0


# Helpers


def _make_batches(items: list[Chunk], size: int) -> list[list[Chunk]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _chunk_to_record(chunk: Chunk) -> tuple:
    m = chunk.metadata
    embedding_str = "[" + ",".join(f"{v:.8f}" for v in chunk.embedding) + "]"
    metadata_json = json.dumps(
        {
            "source_file": m.title,
            "hierarchy_level": m.hierarchy_level.value,
        }
    )
    return (
        chunk.chunk_id,  # $1
        m.document_id,  # $2
        m.document_type.value,  # $3
        m.title,  # $4
        m.jurisdiction,  # $5
        m.chapter,  # $6
        m.article,  # $7
        m.paragraph,  # $8
        m.hierarchy_level.value,  # $9
        m.contains_table,  # $10
        m.contains_penalty,  # $11
        m.is_definition,  # $12
        m.page_start,  # $13
        m.page_end,  # $14
        chunk.text,  # $15
        embedding_str,  # $16
        metadata_json,  # $17
    )
