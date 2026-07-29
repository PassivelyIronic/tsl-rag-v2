"""
Pętla przełączania w RAGGenerator, na atrapach klienta.

Cała wartość łańcucha jest w ścieżkach awaryjnych, a te nie występują
w normalnym przebiegu — więc muszą być wymuszone tutaj, nie wypatrywane
w produkcji.
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from tsl_rag.core.llm_client import ChatTarget
from tsl_rag.core.settings import Settings
from tsl_rag.generation import generator as generator_module
from tsl_rag.generation.generator import _ALL_FAILED_MESSAGE, RAGGenerator

pytestmark = pytest.mark.unit


def _settings(**overrides) -> Settings:
    return Settings(postgres_dsn="postgresql+asyncpg://u:p@localhost:5433/db", **overrides)


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://example.test/")
    return APIStatusError("boom", response=httpx.Response(code, request=request), body=None)


class _FakeClient:
    """Klient, który dla danego modelu zwraca zaplanowany wynik."""

    def __init__(self, plan: dict[str, object]):
        self.plan = plan
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model: str, **_: object):
        self.calls.append(model)
        outcome = self.plan[model]
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=outcome), finish_reason="stop")
            ],
            usage=None,
        )


@pytest.fixture
def fake_client(monkeypatch):
    holder: dict[str, _FakeClient] = {}

    def _install(plan: dict[str, object]) -> _FakeClient:
        client = _FakeClient(plan)
        holder["client"] = client
        monkeypatch.setattr(
            generator_module, "get_chat_client_for", lambda provider, settings: client
        )
        return client

    return _install


async def _run(chain: list[ChatTarget], settings: Settings):
    gen = RAGGenerator()
    return await gen._generate_with_fallback("pytanie", [], chain, settings)


async def test_first_target_succeeds_without_switching(fake_client):
    client = fake_client({"a": "Odpowiedź. [ec_561_2006 | Art. 6]"})
    answer, has_answer, model, switches = await _run(
        [ChatTarget("ollama", "a"), ChatTarget("ollama", "b")], _settings()
    )
    assert has_answer and switches == 0 and model == "a"
    assert client.calls == ["a"]  # ogniwo zapasowe nietknięte


async def test_switches_to_next_target_on_429(fake_client):
    """Bramka Fazy 3: zepsuty pierwszy provider → odpowiedź z drugiego."""
    client = fake_client({"a": _status_error(429), "b": "Odpowiedź z zapasu."})
    answer, has_answer, model, switches = await _run(
        [ChatTarget("openrouter", "a"), ChatTarget("ollama", "b")], _settings()
    )
    assert answer == "Odpowiedź z zapasu."
    assert has_answer and switches == 1 and model == "b"
    assert client.calls == ["a", "b"]


@pytest.mark.parametrize("code", [400, 404, 500])
async def test_switches_on_every_failing_status(fake_client, code):
    fake_client({"a": _status_error(code), "b": "Zapas."})
    _, has_answer, model, _ = await _run(
        [ChatTarget("openrouter", "a"), ChatTarget("ollama", "b")], _settings()
    )
    assert has_answer and model == "b"


async def test_empty_answer_triggers_fallback(fake_client):
    """
    Zmierzone: nemotron zwrócił pustą treść w 2 z 21 pytań (run_015), bo
    rozumowanie zjadało budżet max_tokens. Oddanie tej pustki użytkownikowi
    przy skonfigurowanym modelu zapasowym marnowałoby posiadaną odporność.
    """
    client = fake_client({"a": "   ", "b": "Konkretna odpowiedź."})
    answer, has_answer, model, switches = await _run(
        [ChatTarget("openrouter", "a"), ChatTarget("ollama", "b")], _settings()
    )
    assert answer == "Konkretna odpowiedź."
    assert has_answer and switches == 1 and model == "b"
    assert client.calls == ["a", "b"]


async def test_all_targets_failing_gives_polish_message(fake_client):
    fake_client({"a": _status_error(429), "b": _status_error(503)})
    answer, has_answer, _, _ = await _run(
        [ChatTarget("openrouter", "a"), ChatTarget("ollama", "b")], _settings()
    )
    assert not has_answer
    assert answer == _ALL_FAILED_MESSAGE
    # Użytkownik jest nietechniczny — żadnych kodów HTTP ani nazw wyjątków
    assert "429" not in answer and "Error" not in answer


async def test_refusal_is_not_treated_as_failure(fake_client):
    """
    Odmowa to poprawna odpowiedź modelu, nie awaria. Przełączanie na kolejne
    ogniwo w jej wyniku szukałoby modelu skłonnego halucynować — dokładnie
    odwrotnie do zasady, że halucynacja jest gorsza niż odmowa.
    """
    client = fake_client(
        {"a": "Nie mogę odpowiedzieć na to pytanie na podstawie dostępnych dokumentów.", "b": "X."}
    )
    _, has_answer, model, switches = await _run(
        [ChatTarget("openrouter", "a"), ChatTarget("ollama", "b")], _settings()
    )
    assert not has_answer  # odmowa
    assert model == "a" and switches == 0
    assert client.calls == ["a"]


async def test_open_breaker_skips_target(fake_client):
    client = fake_client({"a": _status_error(429), "b": "Zapas."})
    settings = _settings(chat_breaker_failures=1)
    gen = RAGGenerator(settings)
    chain = [ChatTarget("openrouter", "a"), ChatTarget("ollama", "b")]

    await gen._generate_with_fallback("q", [], chain, settings)
    assert client.calls == ["a", "b"]

    # Drugie zapytanie: ogniwo "a" jest już odcięte, więc nie jest wołane
    await gen._generate_with_fallback("q", [], chain, settings)
    assert client.calls == ["a", "b", "b"]


async def test_missing_api_key_in_backup_does_not_kill_request(monkeypatch):
    """
    Ogniwo zapasowe bez klucza jest pomijane, a nie wywala zapytania, które
    ogniwo główne obsłużyłoby poprawnie.
    """
    calls: list[str] = []

    def _client_for(provider, settings):
        if provider == "openai":
            raise ValueError("OPENAI_API_KEY nie jest ustawiony")
        client = _FakeClient({"dobry": "Odpowiedź."})
        return client

    monkeypatch.setattr(generator_module, "get_chat_client_for", _client_for)
    gen = RAGGenerator()
    _, has_answer, model, _ = await gen._generate_with_fallback(
        "q", [], [ChatTarget("openai", "zly"), ChatTarget("ollama", "dobry")], _settings()
    )
    assert has_answer and model == "dobry"
    assert calls == []
