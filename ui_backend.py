"""
Most między Streamlitem a rdzeniem systemu — tryb in-process.

Streamlit Community Cloud uruchamia JEDEN proces, więc nie ma tam osobnego
serwisu FastAPI, do którego UI mogłoby wysłać żądanie HTTP. Ten moduł daje
`ui.py` to samo, co dawało API, wołając wprost `tsl_rag.service.answer_query` —
czyli dokładnie tę samą funkcję, której używa router.

Rozdział API/UI z PLAN.md nie znika: FastAPI zostaje osobnym serwisem i to on
idzie do K8s. In-process jest ustępstwem wobec jednego darmowego hostingu,
zamkniętym w tym pliku.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import streamlit as st


def bridge_secrets_to_env() -> None:
    """
    Przepisuje `st.secrets` do zmiennych środowiskowych.

    Konieczne, bo `Settings` czyta wyłącznie ze środowiska i z `.env`, a na
    Streamlit Cloud sekrety są dostępne tylko przez `st.secrets`. Wywoływane
    PRZED pierwszym `get_settings()`, inaczej cache ustawień utrwali stan
    bez sekretów i aplikacja zgłosi brak DSN mimo poprawnej konfiguracji.

    Istniejące zmienne środowiskowe mają pierwszeństwo — uruchomienie lokalne
    z `.env` nie ma być nadpisywane przez plik sekretów.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 — brak pliku sekretów lokalnie to norma
        return
    for key, value in secrets.items():
        if isinstance(value, (str, int, float, bool)) and key not in os.environ:
            os.environ[key] = str(value)


@st.cache_resource(show_spinner="Wczytuję bazę przepisów…")
def get_engine() -> dict[str, Any]:
    """
    Retriever i generator utworzone RAZ na proces, nie na zapytanie.

    `st.cache_resource`, a nie `cache_data`: to są obiekty z połączeniami
    i wczytanym modelem, a nie dane do serializacji. Bez tego każde pytanie
    budowałoby od nowa pool połączeń, indeks BM25 i wczytywało model
    embeddingów — czyli 8 sekund zmierzone w Fazie 4, przy każdym pytaniu.

    Pętla zdarzeń też jest trzymana tutaj. Streamlit wywołuje skrypt
    synchronicznie, a retriever i generator są asynchroniczne; `asyncio.run()`
    per pytanie zamykałby pętlę razem z poolem asyncpg, który do niej należy.
    """
    from tsl_rag.core.settings import get_settings
    from tsl_rag.generation.generator import RAGGenerator
    from tsl_rag.retrieval.retriever import HybridRetriever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    retriever = HybridRetriever()
    loop.run_until_complete(retriever.__aenter__())
    loop.run_until_complete(retriever.warmup())

    settings = get_settings()
    return {
        "loop": loop,
        "retriever": retriever,
        "generator": RAGGenerator(settings),
        "settings": settings,
    }


def ask(query: str, *, top_k: int, rerank_top_n: int, debug: bool = False) -> dict:
    """
    Zadaje pytanie w tym samym procesie. Zwraca słownik o kształcie
    identycznym z odpowiedzią API, żeby `ui.py` nie musiało rozróżniać trybów.
    """
    from tsl_rag.service import answer_query

    engine = get_engine()
    response = engine["loop"].run_until_complete(
        answer_query(
            query,
            retriever=engine["retriever"],
            generator=engine["generator"],
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            include_chunks=debug,
        )
    )
    return response.model_dump(mode="json")


def documents() -> dict[str, str]:
    from tsl_rag.core.documents import DOCUMENT_REGISTRY

    return {doc_id.lower(): meta["title"] for doc_id, meta in DOCUMENT_REGISTRY.items()}
