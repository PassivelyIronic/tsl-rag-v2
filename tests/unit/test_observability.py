import pytest
from prometheus_client import REGISTRY

from tsl_rag.core.observability import (
    current_trace_id,
    metrics_payload,
    record_answer,
    record_fallback_switch,
    record_provider_error,
    stage,
)

pytestmark = pytest.mark.unit


def _sample(name: str, **labels) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return value if value is not None else 0.0


def test_stage_records_duration():
    before = _sample("tsl_rag_stage_duration_seconds_count", stage="test_etap")
    with stage("test_etap"):
        pass
    assert _sample("tsl_rag_stage_duration_seconds_count", stage="test_etap") == before + 1


def test_stage_records_duration_even_on_error():
    """
    Etap, który się wywrócił, też musi trafić do histogramu — inaczej dashboard
    pokazuje wyłącznie udane wywołania i awaria wygląda na spadek ruchu,
    a nie na awarię.
    """
    before = _sample("tsl_rag_stage_duration_seconds_count", stage="test_blad")
    with pytest.raises(RuntimeError):
        with stage("test_blad"):
            raise RuntimeError("bum")
    assert _sample("tsl_rag_stage_duration_seconds_count", stage="test_blad") == before + 1


def test_stage_does_not_swallow_exception():
    """Instrumentacja nie zmienia zachowania — wyjątek leci dalej."""
    with pytest.raises(ValueError, match="oryginalny"):
        with stage("test_przepuszcza"):
            raise ValueError("oryginalny")


def test_stage_attributes_accept_none():
    """None jako atrybut spanu wywala OTel — filtrujemy go, zamiast pilnować u wołających."""
    with stage("test_none", cos=None, inne=5):
        pass


def test_trace_id_is_set_inside_span_and_empty_outside():
    assert current_trace_id() == ""
    with stage("test_trace"):
        inside = current_trace_id()
    assert len(inside) == 32
    assert int(inside, 16) != 0


def test_nested_stages_share_trace_id():
    """
    Sedno bramki Fazy 4: retrieval i generacja muszą trafić do JEDNEGO śladu,
    inaczej `trace_id` w logach nie spina jednego pytania w całość.
    """
    with stage("rodzic"):
        outer = current_trace_id()
        with stage("dziecko"):
            inner = current_trace_id()
    assert outer == inner


def test_provider_error_counter():
    before = _sample(
        "tsl_rag_provider_errors_total", provider="openrouter", model="m", kind="transient"
    )
    record_provider_error("openrouter", "m", "transient")
    assert (
        _sample("tsl_rag_provider_errors_total", provider="openrouter", model="m", kind="transient")
        == before + 1
    )


def test_fallback_switch_counter():
    before = _sample("tsl_rag_fallback_switches_total", from_target="a", to_target="b")
    record_fallback_switch("a", "b")
    assert _sample("tsl_rag_fallback_switches_total", from_target="a", to_target="b") == before + 1


def test_answer_outcome_counter():
    before = _sample("tsl_rag_answers_total", outcome="refused")
    record_answer("refused")
    assert _sample("tsl_rag_answers_total", outcome="refused") == before + 1


def test_metrics_payload_is_prometheus_text():
    record_answer("answered")
    payload, content_type = metrics_payload()
    assert "text/plain" in content_type
    assert b"tsl_rag_answers_total" in payload
    assert b"tsl_rag_stage_duration_seconds" in payload
