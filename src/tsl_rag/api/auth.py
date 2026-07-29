"""
Autoryzacja API współdzielonym hasłem.

Świadomie najprostsza z możliwych: jeden sekret w zmiennej środowiskowej,
przekazywany nagłówkiem. Bez użytkowników, ról i tokenów, bo system ma jedną
osobę korzystającą, a każdy z tych mechanizmów dokładałby stan do utrzymania
bez żadnego zysku.

Po co w ogóle: publiczny URL bez autoryzacji to zaproszenie do wypalenia
darmowych limitów providera przez pierwszego bota, który go znajdzie —
i to bez śladu w rachunku, bo limity są dzienne, a nie kwotowe.

Domyślnie WYŁĄCZONA (`api_password` puste), żeby uruchomienie lokalne
i `pytest` nie wymagały ustawiania sekretu. Włącza się przez ustawienie hasła,
co jest zarazem jedynym sensownym miejscem decyzji: lokalnie niepotrzebna,
w publicznym wdrożeniu obowiązkowa.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status
from loguru import logger
from pydantic import SecretStr

from tsl_rag.core.settings import get_settings

API_KEY_HEADER = "X-API-Key"

_MSG_MISSING = "Brak hasła dostępu. Dodaj nagłówek X-API-Key z hasłem otrzymanym od administratora."
_MSG_WRONG = "Nieprawidłowe hasło dostępu."


def check_api_key(provided: str | None, expected: SecretStr | None) -> None:
    """
    Czysta logika sprawdzenia — bez typów FastAPI, więc testowalna wprost.

    Wydzielona z zależności celowo. Wcześniej `verify_api_key` przyjmowało
    `settings: Settings | None = None` „dla testowalności" i to był realny błąd:
    FastAPI widzi w sygnaturze zależności model pydantica i traktuje go jako
    POLE CIAŁA żądania. Ciało /query robiło się zagnieżdżone
    (`{"request": …, "settings": …}`), więc każde normalne zapytanie dostawało
    422. Nie wychodziło to w testach, bo sprawdzały tylko ścieżkę 401, która
    zapada przed walidacją ciała.

    Porównanie przez `secrets.compare_digest`, nie `==`: zwykłe porównanie
    stringów kończy się na pierwszym różniącym się bajcie, więc czas odpowiedzi
    zdradza, ile znaków hasła się zgadza.
    """
    if not expected:
        return  # autoryzacja wyłączona — uruchomienie lokalne

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_MSG_MISSING,
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    if not secrets.compare_digest(provided, expected.get_secret_value()):
        # Bez logowania podanej wartości — trafiłaby do stdout, a stamtąd
        # do agregatora logów, czyli sekret wyciekłby przez własną telemetrię.
        logger.warning("Odrzucone żądanie: nieprawidłowe hasło dostępu")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_MSG_WRONG,
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """
    Zależność FastAPI. Sygnatura zawiera WYŁĄCZNIE typy, które FastAPI rozumie
    jako parametry żądania — żadnych modeli pydantica, patrz `check_api_key`.
    """
    check_api_key(x_api_key, get_settings().api_password)
