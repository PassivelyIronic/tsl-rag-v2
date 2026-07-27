import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from tsl_rag.core.llm_client import ChatTarget, resolve_chat_chain
from tsl_rag.core.settings import Settings
from tsl_rag.generation.fallback import CircuitBreaker, FailureKind, classify_failure

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(postgres_dsn="postgresql+asyncpg://u:p@localhost:5433/db", **overrides)


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(code, request=request)
    return APIStatusError("boom", response=response, body=None)


# --- klasyfikacja awarii ---


@pytest.mark.parametrize("code", [400, 404])
def test_deterministic_errors(code):
    """
    400 (zły slug) i 404 (model wycofany z darmowej puli) dają ten sam wynik
    przy każdej próbie — ponawianie na tym samym ogniwie to czysta strata czasu.
    """
    assert classify_failure(_status_error(code)) is FailureKind.DETERMINISTIC


@pytest.mark.parametrize("code", [429, 500, 502, 503])
def test_transient_errors(code):
    assert classify_failure(_status_error(code)) is FailureKind.TRANSIENT


@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors(code):
    assert classify_failure(_status_error(code)) is FailureKind.AUTH


def test_network_errors_are_transient():
    request = httpx.Request("POST", "https://example.test/")
    assert classify_failure(APIConnectionError(request=request)) is FailureKind.TRANSIENT
    assert classify_failure(APITimeoutError(request=request)) is FailureKind.TRANSIENT


def test_unknown_exception():
    assert classify_failure(RuntimeError("co to")) is FailureKind.UNKNOWN


# --- łańcuch ---


def test_chain_without_config_is_single_target():
    """Brak łańcucha = zachowanie sprzed jego wprowadzenia."""
    chain = resolve_chat_chain(_settings())
    assert len(chain) == 1
    assert chain[0] == ChatTarget("ollama", "mistral:7b-instruct-q4_K_M")


def test_chain_puts_primary_provider_first():
    chain = resolve_chat_chain(_settings(chat_fallback_chain="ollama:zapas"))
    assert chain[0].provider == "ollama"
    assert chain[0].model == "mistral:7b-instruct-q4_K_M"
    assert chain[1] == ChatTarget("ollama", "zapas")


def test_chain_keeps_model_names_containing_colons():
    """Slugi Ollamy mają dwukropek w nazwie ('mistral:7b'), więc dzielimy raz."""
    chain = resolve_chat_chain(_settings(chat_fallback_chain="ollama:mistral:7b-instruct"))
    assert chain[1] == ChatTarget("ollama", "mistral:7b-instruct")


def test_chain_deduplicates():
    """
    Powtórzone ogniwo nie dodaje odporności, a wydłuża czas do komunikatu
    o błędzie o kolejny timeout.
    """
    chain = resolve_chat_chain(
        _settings(chat_fallback_chain="ollama:mistral:7b-instruct-q4_K_M,ollama:zapas,ollama:zapas")
    )
    assert [str(t) for t in chain] == ["ollama:mistral:7b-instruct-q4_K_M", "ollama:zapas"]


def test_chain_ignores_blank_entries():
    chain = resolve_chat_chain(_settings(chat_fallback_chain=" , ollama:zapas , "))
    assert len(chain) == 2


def test_chain_rejects_entry_without_model():
    with pytest.raises(ValueError, match="provider:model"):
        resolve_chat_chain(_settings(chat_fallback_chain="openrouter"))


def test_chain_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Nieznany provider"):
        resolve_chat_chain(_settings(chat_fallback_chain="wymyslony:model"))


# --- bezpiecznik ---


def test_breaker_opens_after_threshold():
    breaker = CircuitBreaker(failures=3, cooldown_s=60.0)
    for _ in range(2):
        breaker.record_failure("a", now=0.0)
    assert not breaker.is_open("a", now=0.0)

    breaker.record_failure("a", now=0.0)
    assert breaker.is_open("a", now=0.0)


def test_breaker_closes_after_cooldown():
    breaker = CircuitBreaker(failures=1, cooldown_s=60.0)
    breaker.record_failure("a", now=0.0)
    assert breaker.is_open("a", now=59.0)
    assert not breaker.is_open("a", now=61.0)


def test_breaker_gives_clean_slate_after_cooldown():
    """
    Po cooldownie licznik startuje od zera — inaczej jedna przejściowa awaria
    wykluczałaby ogniwo do końca życia procesu, bo próg byłby już przekroczony.
    """
    breaker = CircuitBreaker(failures=2, cooldown_s=60.0)
    breaker.record_failure("a", now=0.0)
    breaker.record_failure("a", now=0.0)
    assert breaker.is_open("a", now=0.0)

    assert not breaker.is_open("a", now=61.0)
    breaker.record_failure("a", now=61.0)
    assert not breaker.is_open("a", now=61.0)


def test_success_resets_failures():
    breaker = CircuitBreaker(failures=2, cooldown_s=60.0)
    breaker.record_failure("a", now=0.0)
    breaker.record_success("a")
    breaker.record_failure("a", now=0.0)
    assert not breaker.is_open("a", now=0.0)


def test_breaker_is_per_target():
    breaker = CircuitBreaker(failures=1, cooldown_s=60.0)
    breaker.record_failure("a", now=0.0)
    assert breaker.is_open("a", now=0.0)
    assert not breaker.is_open("b", now=0.0)
