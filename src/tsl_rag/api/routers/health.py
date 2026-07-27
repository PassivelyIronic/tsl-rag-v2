"""
Probe'y stanu aplikacji.

Rozdzielone celowo, zgodnie z semantyką probe'ów Kubernetesa:

- /health (liveness)  — czy proces żyje. Nie sprawdza zależności zewnętrznych,
  bo restart poda nie naprawi niedostępnej bazy ani przeciążonego providera.
- /ready (readiness)  — czy proces jest w stanie obsłużyć zapytanie: baza
  odpowiada, retriever zainicjalizowany, provider embeddingów odpowiada.
  Zwraca 503, gdy którykolwiek warunek nie jest spełniony.

Wcześniej istniał jeden endpoint /query/health mieszający oba znaczenia,
z polem "ollama" zaszytym w schemacie odpowiedzi — nazwa przestawała być
prawdziwa w momencie przełączenia EMBEDDING_PROVIDER na cokolwiek innego.
"""

from __future__ import annotations

from typing import Annotated, Literal

import asyncpg
from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel

from tsl_rag.core.embeddings import get_embedding_provider
from tsl_rag.core.observability import metrics_payload
from tsl_rag.core.settings import Settings, get_settings

router = APIRouter(tags=["health"])


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]
    embedding_provider: str
    chat_provider: str
    chat_model: str


@router.get("/health", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Liveness: proces odpowiada. Bez odpytywania zależności."""
    return LivenessResponse(status="ok")


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """
    Metryki w formacie Prometheusa.

    Poza schematem OpenAPI, bo to endpoint dla scrapera, nie dla klienta API —
    w dokumentacji tylko zaśmiecałby listę operacji.
    """
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(
    request: Request,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadinessResponse:
    """Readiness: baza + retriever + provider embeddingów."""
    checks: dict[str, str] = {}

    # 1. Postgres
    try:
        raw_dsn = str(settings.postgres_dsn).replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(dsn=raw_dsn, timeout=5)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"

    # 2. Retriever — zainicjalizowany w lifespanie
    retriever = getattr(request.app.state, "retriever", None)
    checks["retriever"] = "ok" if retriever is not None else "error: nie zainicjalizowany"

    # 3. Provider embeddingów — liczony przy KAŻDYM zapytaniu, więc jego
    #    niedostępność oznacza brak gotowości, nie tylko degradację.
    try:
        await get_embedding_provider().embed_query("health check")
        checks["embeddings"] = "ok"
    except Exception as exc:
        checks["embeddings"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if all_ok else "not_ready",
        checks=checks,
        embedding_provider=settings.embedding_provider,
        chat_provider=settings.chat_provider,
        chat_model=settings.active_llm_model,
    )
