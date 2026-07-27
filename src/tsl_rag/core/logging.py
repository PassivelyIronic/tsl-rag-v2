"""
Konfiguracja logów: poziom z ustawień, `trace_id` w każdym rekordzie.

Do tej pory `log_level` istniał w `Settings`, ale nic go nie czytało — loguru
pracował na domyślnym sinku, więc ustawienie `LOG_LEVEL=DEBUG` nie robiło nic.
To ta sama klasa błędu co martwe wagi RRF przed commitem 92ab634: konfiguracja
istnieje, wygląda na działającą i nie jest podłączona.

`trace_id` wchodzi do rekordu przez `patch`, a nie przez ręczne dopisywanie
w każdym wywołaniu — inaczej wystarczy jeden log bez identyfikatora, żeby
zerwać ciągłość śladu dokładnie tam, gdzie coś poszło nie tak.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from loguru import logger

from tsl_rag.core.observability import current_trace_id
from tsl_rag.core.settings import Settings

_HUMAN_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[trace_id]: <8}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _json_sink(message: Any) -> None:
    """
    Jedna linia JSON na rekord, na stdout.

    Stdout, nie plik: aplikacja ma być dobrze zachowującym się tenantem —
    zbieranie logów należy do platformy, nie do procesu (CLAUDE.md §7).
    """
    record = message.record
    payload = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
        "trace_id": record["extra"].get("trace_id", ""),
    }
    if record["exception"] is not None:
        payload["exception"] = repr(record["exception"].value)
    print(json.dumps(payload, ensure_ascii=False), file=sys.stdout)


def _patch_trace_id(record: Any) -> None:
    # Skrócony do 8 znaków w formacie czytelnym dla człowieka — pełny i tak
    # jest w JSON-ie, a w konsoli 32 znaki wypychają treść komunikatu za krawędź.
    trace_id = current_trace_id()
    record["extra"]["trace_id"] = trace_id[:8] if trace_id else "-"
    record["extra"]["trace_id_full"] = trace_id


def configure_logging(settings: Settings) -> None:
    """Wywoływane raz przy starcie aplikacji, przed pierwszym logiem."""
    logger.remove()
    logger.configure(patcher=_patch_trace_id)

    if settings.log_json:
        logger.add(_json_sink, level=settings.log_level, format="{message}")
    else:
        logger.add(sys.stderr, level=settings.log_level, format=_HUMAN_FORMAT)

    logger.debug(f"Logi skonfigurowane: poziom={settings.log_level}, json={settings.log_json}")
