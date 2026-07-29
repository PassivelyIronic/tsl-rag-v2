"""
Testy e2e: pełna aplikacja przez HTTP, z atrapą generacji.

Generacja jest podmieniana na atrapę celowo. Prawdziwe wywołanie modelu
kosztowałoby zapytanie z dziennego limitu (~50), a wynik zależałby od dostępności
providera — czyli test byłby wolny i losowo czerwony. Sprawdzamy tu montaż
aplikacji: routing, autoryzację, kształt odpowiedzi i cache, a nie jakość modelu.
Jakość mierzy `evals/`.

Uruchomienie:  uv run pytest -m e2e
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tsl_rag.core.models import Citation, QueryResponse

pytestmark = pytest.mark.e2e

_ANSWER = "Dzienny czas prowadzenia to 9 godzin. [ec_561_2006 | Art. 6(1)]"


@pytest.fixture
def client(monkeypatch):
    """Aplikacja z podmienioną generacją i retrievalem — bez bazy i bez providera."""

    async def _fake_answer(query, **kwargs):
        response = QueryResponse(
            query=query,
            answer=_ANSWER,
            citations=[
                Citation(
                    document_id="ec_561_2006",
                    document_title="Rozporządzenie 561/2006",
                    article="6(1)",
                    paragraph=None,
                    chunk_id="ec_561_2006::0001",
                )
            ],
            retrieved_chunks=[],
            model_used="atrapa",
            latency_ms=1,
            has_answer=True,
        )
        cache = kwargs.get("cache")
        if cache is not None:
            key = f"k::{query}"
            if (hit := cache.get(key)) is not None:
                return hit
            cache.put(key, response)
        return response

    import tsl_rag.api.routers.query as query_module

    monkeypatch.setattr(query_module, "answer_query", _fake_answer)

    from tsl_rag.api.app import create_app

    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        # PO wejściu w kontekst: lifespan właśnie się wykonał i nadpisałby
        # to, co ustawione wcześniej.
        app.state.retriever = object()  # obecność wystarcza — retrieval jest zaatrapowany
        app.state.generator = object()
        yield c


def test_health_is_open(client):
    """Liveness nie może zależeć od bazy ani od sekretu — inaczej probe kłamie."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_metrics_endpoint_serves_prometheus(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]


def test_documents_endpoint_lists_corpus(client):
    r = client.get("/query/documents")
    assert r.status_code == 200
    assert "ec_561_2006" in r.json()


def test_query_returns_answer_with_citation(client):
    r = client.post("/query", json={"query": "Ile godzin dziennie moze jechac kierowca?"})
    assert r.status_code == 200
    body = r.json()
    assert body["has_answer"] is True
    assert body["citations"][0]["document_id"] == "ec_561_2006"


def test_query_rejects_too_short_input(client):
    """Walidacja wejścia to 422, nie 500 — pusty ekran nie mówi użytkownikowi nic."""
    assert client.post("/query", json={"query": "hm"}).status_code == 422


def test_query_requires_password_when_configured(monkeypatch, client):
    from pydantic import SecretStr

    from tsl_rag.core.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "api_password", SecretStr("tajne"), raising=False)

    r = client.post("/query", json={"query": "Ile godzin dziennie moze jechac kierowca?"})
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"]

    r_ok = client.post(
        "/query",
        json={"query": "Ile godzin dziennie moze jechac kierowca?"},
        headers={"X-API-Key": "tajne"},
    )
    assert r_ok.status_code == 200
