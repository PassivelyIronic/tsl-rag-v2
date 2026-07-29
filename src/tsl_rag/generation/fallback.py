"""
Klasyfikacja awarii providera i bezpiecznik dla łańcucha fallbacku.

Wydzielone z generatora, bo bezpiecznik trzyma stan MIĘDZY zapytaniami,
a generator jest bezstanowy. Trzymanie licznika porażek w generatorze
oznaczałoby, że bezpiecznik resetuje się przy każdym pytaniu i nie chroni
przed niczym.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError


class FailureKind(StrEnum):
    """Rodzaj awarii — rozstrzyga, czy ponawianie ma sens."""

    DETERMINISTIC = "deterministic"  # 400/404 — ten sam request da ten sam błąd
    TRANSIENT = "transient"  # 429/5xx/timeout — może minąć
    AUTH = "auth"  # 401/403 — konfiguracja, nie awaria
    EMPTY = "empty"  # odpowiedź bez treści
    UNKNOWN = "unknown"


def classify_failure(exc: BaseException) -> FailureKind:
    """
    Mapuje wyjątek providera na rodzaj awarii.

    Rozróżnienie ma jedną konsekwencję praktyczną: przy DETERMINISTIC nie ma
    sensu ponawiać (zły slug modelu albo model wycofany z puli zwrócą to samo
    przy każdej próbie), przy TRANSIENT ma — ale na KOLEJNYM ogniwie, bo
    zmierzone 429 z OpenRoutera to przeciążenie upstreamu, którego backoff
    na tym samym providerze nie omija.
    """
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status in (400, 404):
            return FailureKind.DETERMINISTIC
        if status in (401, 403):
            return FailureKind.AUTH
        if status == 429 or status >= 500:
            return FailureKind.TRANSIENT
        return FailureKind.UNKNOWN
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return FailureKind.TRANSIENT
    return FailureKind.UNKNOWN


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    open_until: float = 0.0


@dataclass
class CircuitBreaker:
    """
    Pomija ogniwo, które zawiodło `failures` razy z rzędu, przez `cooldown_s`.

    Świadomie prosty: liczy porażki Z RZĘDU, a każdy sukces zeruje licznik.
    Przy jednym użytkowniku i kilku zapytaniach dziennie okno przesuwne albo
    półotwarty stan byłyby maszynerią bez pokrycia w ruchu.
    """

    failures: int = 3
    cooldown_s: float = 60.0
    _state: dict[str, _BreakerState] = field(default_factory=dict)

    def is_open(self, target: str, *, now: float | None = None) -> bool:
        state = self._state.get(target)
        if state is None:
            return False
        current = time.monotonic() if now is None else now
        if state.open_until > current:
            return True
        if state.open_until:
            # Cooldown minął — ogniwo dostaje czystą kartę, żeby jedna
            # przejściowa awaria nie wykluczała go do końca życia procesu.
            state.open_until = 0.0
            state.consecutive_failures = 0
        return False

    def record_success(self, target: str) -> None:
        self._state.pop(target, None)

    def record_failure(self, target: str, *, now: float | None = None) -> None:
        current = time.monotonic() if now is None else now
        state = self._state.setdefault(target, _BreakerState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= self.failures:
            state.open_until = current + self.cooldown_s
            logger.warning(
                f"Bezpiecznik OTWARTY dla {target}: {state.consecutive_failures} "
                f"porażek z rzędu, pomijane przez {self.cooldown_s:.0f} s"
            )
