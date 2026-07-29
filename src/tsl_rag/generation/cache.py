"""
Cache odpowiedzi na powtarzające się pytania.

Po co: darmowy model ma ~50 zapytań na dobę, a hosting usypia aplikację po
bezczynności. Powtórzone pytanie nie ma powodu zjadać limitu ani czekać
kilku sekund na generację.

Dwie decyzje, które są tu istotne:

1. **Klucz zawiera konfigurację, nie samo pytanie.** Model, prefiks systemowy,
   wagi RRF, `rrf_k`, `top_k` i limit kontekstu wchodzą do hasza. Bez tego
   zmiana modelu albo stałej fuzji serwowałaby odpowiedzi wygenerowane przez
   poprzednią konfigurację — i to bez żadnego sygnału, że coś jest nie tak.
   To najgroźniejszy wariant błędu w tym projekcie: pomiar „po zmianie", który
   w rzeczywistości pokazuje stan sprzed niej.
2. **Odmowy NIE są cache'owane.** Odmowa bywa skutkiem chwilowego stanu —
   wyczerpanego limitu, przeciążenia providera, gorszego losowania kontekstu.
   Zapisanie jej na godziny utrwala najgorszą możliwą odpowiedź. Cache'ujemy
   wyłącznie odpowiedzi z treścią i z cytowaniem.

Cache jest **w pamięci procesu**. Przy jednym użytkowniku i jednej replice to
wystarcza, a wariant w bazie dokładałby zapis do ścieżki zapytania i tabelę
do utrzymania. Restart czyści cache — to akceptowalne, bo koszt pudła to jedno
zapytanie do providera.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass

from loguru import logger

from tsl_rag.core.models import QueryResponse
from tsl_rag.core.settings import Settings

# Separator pól w materiale hasza. Bajt zerowy, nie spacja: wartości pól
# (nazwy modeli, prefiks systemowy) mogą zawierać spacje, więc przy spacji
# dwa różne zestawy konfiguracji potrafiłyby dać ten sam ciąg wejściowy,
# a tym samym ten sam klucz cache.
_SEP = chr(0)


def cache_key(query: str, settings: Settings) -> str:
    """
    Hasz z pytania ORAZ konfiguracji, która wpływa na odpowiedź.

    Normalizacja pytania jest celowo minimalna — małe litery i zwinięte białe
    znaki. Dalej idąca (np. składanie diakrytyków) zlewałaby pytania, które
    retrieval traktuje inaczej, więc cache oddawałby odpowiedź na inne pytanie.
    """
    normalized = " ".join(query.lower().split())
    material = _SEP.join(
        [
            normalized,
            settings.chat_provider,
            settings.active_llm_model,
            settings.llm_system_prefix,
            settings.llm_reasoning_effort,
            str(settings.llm_temperature),
            str(settings.rrf_k),
            str(settings.bm25_weight),
            str(settings.retrieval_top_k),
            str(settings.retrieval_rerank_top_n),
            str(settings.max_context_chars),
            settings.active_embedding_model,
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass
class _Entry:
    response: QueryResponse
    stored_at: float


class AnswerCache:
    """
    Cache LRU z czasem życia wpisu.

    Zwykła klasa, NIE dataclass: FastAPI introspektuje dataclassy jak modele
    ciała żądania, więc wstrzyknięcie przez `Depends` kończyło się odpowiedzią
    422 z `loc=["body","request"]` — cache lądował w schemacie jako drugie pole
    ciała obok właściwego zapytania.
    """

    def __init__(self, max_entries: int = 128, ttl_s: float = 86_400.0) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, *, now: float | None = None) -> QueryResponse | None:
        current = time.monotonic() if now is None else now
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if current - entry.stored_at > self.ttl_s:
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.response

    def put(self, key: str, response: QueryResponse, *, now: float | None = None) -> None:
        """
        Zapisuje wyłącznie użyteczną odpowiedź.

        Warunek `has_answer` odsiewa odmowy i komunikaty o awarii providera,
        a brak cytowań traktujemy jak brak odpowiedzi — cytowanie jest w tym
        systemie funkcją krytyczną, nie ozdobą.
        """
        if not response.has_answer or not response.citations:
            return
        current = time.monotonic() if now is None else now
        self._entries[key] = _Entry(response=response, stored_at=current)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            evicted, _ = self._entries.popitem(last=False)
            logger.debug(f"Cache: usunięto najstarszy wpis {evicted[:12]}…")

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
