from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from tsl_rag.core.settings import get_settings


class DocumentType(StrEnum):
    EU_REGULATION = "eu_regulation"
    DIRECTIVE = "directive"
    AETR_AGREEMENT = "aetr_agreement"
    PENALTY_TARIFF = "penalty_tariff"
    NATIONAL_LAW = "national_law"


class LegalHierarchyLevel(StrEnum):
    CHAPTER = "chapter"
    ARTICLE = "article"
    PARAGRAPH = "paragraph"
    SUBPARAGRAPH = "subparagraph"
    TABLE = "table"
    ANNEX = "annex"


class DocumentMetadata(BaseModel):
    document_id: str
    document_type: DocumentType
    title: str
    source_url: str | None = None
    jurisdiction: str = "EU"
    chapter: str | None = None
    article: str | None = None
    paragraph: str | None = None
    hierarchy_level: LegalHierarchyLevel = LegalHierarchyLevel.PARAGRAPH
    contains_table: bool = False
    contains_penalty: bool = False
    is_definition: bool = False
    page_start: int | None = None  # ← NOWE
    page_end: int | None = None  # ← NOWE


class Chunk(BaseModel):  # ← NOWA KLASA
    chunk_id: str = ""
    text: str
    metadata: DocumentMetadata
    embedding: list[float] | None = Field(default=None, repr=False)
    token_count: int | None = None


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    embedding: list[float] | None = None
    token_count: int | None = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float | None = None


class RetrievalRequest(BaseModel):
    """
    Domyślne top_k / rerank_top_n pochodzą z Settings, nie z hardkodu —
    dzięki temu RETRIEVAL_TOP_K i RETRIEVAL_RERANK_TOP_N w .env faktycznie
    działają. Wcześniej te dwa pola były zaszyte na 20/5, a odpowiadające
    im pola w Settings nie były czytane przez nikogo.
    """

    query: str
    top_k: int = Field(default_factory=lambda: get_settings().retrieval_top_k)
    rerank_top_n: int = Field(default_factory=lambda: get_settings().retrieval_rerank_top_n)
    filter_document_ids: list[str] | None = None
    filter_document_type: DocumentType | None = None
    filter_contains_penalty: bool | None = None


class Citation(BaseModel):
    document_id: str
    document_title: str
    article: str | None
    paragraph: str | None
    chunk_id: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    model_used: str
    latency_ms: int
    has_answer: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
