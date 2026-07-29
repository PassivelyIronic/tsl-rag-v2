import pytest

from tsl_rag.core.models import Citation, QueryResponse
from tsl_rag.core.settings import Settings
from tsl_rag.generation.cache import AnswerCache, cache_key

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(postgres_dsn="postgresql+asyncpg://u:p@localhost:5433/db", **overrides)


def _response(answer: str = "Odpowiedź.", *, has_answer: bool = True, cited: bool = True):
    return QueryResponse(
        query="q",
        answer=answer,
        citations=[
            Citation(
                document_id="ec_561_2006",
                document_title="Rozporządzenie 561/2006",
                article="6",
                paragraph=None,
                chunk_id="ec_561_2006::0001",
            )
        ]
        if cited
        else [],
        retrieved_chunks=[],
        model_used="m",
        latency_ms=10,
        has_answer=has_answer,
    )


def test_hit_and_miss():
    cache = AnswerCache()
    assert cache.get("k") is None
    cache.put("k", _response())
    assert cache.get("k") is not None
    assert (cache.hits, cache.misses) == (1, 1)


def test_refusal_is_not_cached():
    """
    Odmowa bywa skutkiem chwilowego stanu — wyczerpanego limitu albo
    przeciążenia providera. Zapisanie jej na dobę utrwala najgorszą
    możliwą odpowiedź.
    """
    cache = AnswerCache()
    cache.put("k", _response("Nie mogę odpowiedzieć…", has_answer=False))
    assert cache.get("k") is None


def test_answer_without_citation_is_not_cached():
    """Cytowanie jest funkcją krytyczną — odpowiedź bez niego to porażka."""
    cache = AnswerCache()
    cache.put("k", _response(cited=False))
    assert cache.get("k") is None


def test_entry_expires_after_ttl():
    cache = AnswerCache(ttl_s=60.0)
    cache.put("k", _response(), now=0.0)
    assert cache.get("k", now=59.0) is not None
    assert cache.get("k", now=61.0) is None


def test_lru_evicts_oldest():
    cache = AnswerCache(max_entries=2)
    for k in ("a", "b"):
        cache.put(k, _response())
    cache.get("a")  # "a" staje się świeższe
    cache.put("c", _response())
    assert cache.get("a") is not None
    assert cache.get("b") is None  # najstarsze nieużywane wypadło


def test_key_depends_on_query():
    s = _settings()
    assert cache_key("pytanie A", s) != cache_key("pytanie B", s)


def test_key_ignores_case_and_whitespace():
    s = _settings()
    assert cache_key("Ile godzin?", s) == cache_key("  ile   GODZIN?  ", s)


@pytest.mark.parametrize(
    "override",
    [
        {"rrf_k": 60},
        {"llm_system_prefix": "/no_think"},
        {"retrieval_rerank_top_n": 3},
        {"max_context_chars": 8000},
        {"llm_temperature": 0.7},
    ],
)
def test_key_changes_with_config(override):
    """
    Sedno klucza: zmiana konfiguracji MUSI unieważniać cache. Inaczej po zmianie
    modelu albo stałej fuzji system serwowałby odpowiedzi z poprzedniego
    ustawienia, a pomiar „po zmianie" pokazywałby stan sprzed niej.
    """
    base = _settings()
    assert cache_key("pytanie", base) != cache_key("pytanie", _settings(**override))


def test_key_separator_prevents_field_collision():
    """
    Pola sklejane bajtem zerowym, nie spacją: przy spacji dwa różne zestawy
    konfiguracji mogłyby dać ten sam ciąg wejściowy i ten sam klucz.
    """
    a = cache_key("q", _settings(llm_system_prefix="a b", openrouter_chat_model="c"))
    b = cache_key("q", _settings(llm_system_prefix="a", openrouter_chat_model="b c"))
    assert a != b
