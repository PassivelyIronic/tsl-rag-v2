"""
OpenTelemetry + Prometheus: jedno miejsce, w którym powstaje tracer i metryki.

Dlaczego to jest w kodzie od początku, a nie doklejane później: etapy retrievalu
mierzone są dziś wyłącznie sumarycznie, więc pytanie „gdzie poszły 24 sekundy"
nie ma odpowiedzi bez ręcznego dorzucania print-ów. Rozbicie na spany jest też
warunkiem canary po metrykach w przyszłym projekcie K8s (PLAN.md Faza 6).

Trzy zasady, które sterują tym modułem:

1. **Brak kolektora nie może psuć aplikacji.** Domyślnie tracing idzie do
   eksportera pustego (`none`), czyli spany powstają, ale nigdzie nie lecą.
   Aplikacja dla jednego użytkownika nie może wymagać Jaegera do uruchomienia.
2. **Metryki nie mogą zawierać danych o wysokiej kardynalności.** Etykietą jest
   provider, model i rodzaj błędu — nigdy treść zapytania ani `chunk_id`.
   Prometheus z etykietą per zapytanie przewraca się na pamięci.
3. **Instrumentacja nie zmienia zachowania.** Wyjątek w eksporterze nie ma prawa
   przerwać odpowiedzi dla użytkownika.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from loguru import logger
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from tsl_rag.core.settings import Settings, get_settings

# --- Metryki Prometheusa ---
#
# Kubełki histogramu dobrane pod ZMIERZONE latencje tego systemu, nie domyślne
# z biblioteki: retrieval bez rerankingu to ~0.1 s, generacja na darmowym modelu
# 15-25 s. Domyślne kubełki (do 10 s) wrzucałyby każdą generację do +Inf,
# czyli nie dałoby się odróżnić 15 s od 60 s — a to jest różnica między
# „wolno" a „użytkownik zamknął kartę".
_STAGE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0)

stage_duration = Histogram(
    "tsl_rag_stage_duration_seconds",
    "Czas trwania etapu przetwarzania zapytania",
    labelnames=("stage",),
    buckets=_STAGE_BUCKETS,
)

provider_errors = Counter(
    "tsl_rag_provider_errors_total",
    "Błędy providera generacji, w rozbiciu na rodzaj awarii",
    labelnames=("provider", "model", "kind"),
)

fallback_switches = Counter(
    "tsl_rag_fallback_switches_total",
    "Przełączenia na kolejne ogniwo łańcucha fallbacku",
    labelnames=("from_target", "to_target"),
)

answers_total = Counter(
    "tsl_rag_answers_total",
    "Zakończone zapytania w rozbiciu na wynik",
    labelnames=("outcome",),  # answered | refused | all_providers_failed
)

_TRACER_NAME = "tsl_rag"


@lru_cache(maxsize=1)
def _setup_tracing() -> trace.Tracer:
    """
    Konfiguruje TracerProvider raz na proces.

    Bez argumentu i z cache'em, bo `Settings` jest modelem pydantica i nie jest
    hashowalny — `lru_cache` po nim wywaliłby się przy pierwszym wywołaniu.

    `otel_exporter`:
      - "none"    — spany powstają, nigdzie nie lecą (domyślne)
      - "console" — wypisywane na stdout, do debugowania bez kolektora
      - "otlp"    — wysyłane do `otel_endpoint`
    """
    settings: Settings = get_settings()
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "deployment.environment": settings.app_env,
        }
    )
    provider = TracerProvider(resource=resource)

    if settings.otel_exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif settings.otel_exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint))
            )
            logger.info(f"OpenTelemetry: eksport OTLP do {settings.otel_endpoint}")
        except Exception as exc:  # noqa: BLE001
            # Zasada 1: brak kolektora nie psuje aplikacji.
            logger.error(f"OTLP niedostępny ({exc}) — spany zostają lokalne")

    trace.set_tracer_provider(provider)
    return trace.get_tracer(_TRACER_NAME)


def get_tracer() -> trace.Tracer:
    return _setup_tracing()


@contextmanager
def stage(name: str, **attributes: object) -> Iterator[Span]:
    """
    Span + histogram Prometheusa dla jednego etapu, w jednym wywołaniu.

    Trzymane razem celowo: etap mierzony w tracingu, a pominięty w metrykach,
    daje dashboard niezgodny ze śladem i to rozjeżdżanie się wychodzi dopiero
    przy diagnozowaniu awarii, czyli w najgorszym momencie.
    """
    t0 = time.monotonic()
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)  # type: ignore[arg-type]
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            span.record_exception(exc)
            raise
        finally:
            stage_duration.labels(stage=name).observe(time.monotonic() - t0)


def current_trace_id() -> str:
    """
    Identyfikator śladu do wpięcia w logi — spina jedno zapytanie w całość.

    Pusty string, gdy nie ma aktywnego spanu: log bez trace_id jest lepszy
    niż log, który się nie zapisał.
    """
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return ""
    return format(context.trace_id, "032x")


def record_provider_error(provider: str, model: str, kind: str) -> None:
    provider_errors.labels(provider=provider, model=model, kind=kind).inc()


def record_fallback_switch(from_target: str, to_target: str) -> None:
    fallback_switches.labels(from_target=from_target, to_target=to_target).inc()


def record_answer(outcome: str) -> None:
    answers_total.labels(outcome=outcome).inc()


def metrics_payload() -> tuple[bytes, str]:
    """Zawartość odpowiedzi dla `/metrics`."""
    return generate_latest(), CONTENT_TYPE_LATEST
