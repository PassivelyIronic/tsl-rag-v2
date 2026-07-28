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

from tsl_rag.core.settings import Settings, get_settings

API_KEY_HEADER = "X-API-Key"

_MSG_MISSING = "Brak hasła dostępu. Dodaj nagłówek X-API-Key z hasłem otrzymanym od administratora."
_MSG_WRONG = "Nieprawidłowe hasło dostępu."


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
    settings: Settings | None = None,
) -> None:
    """
    Zależność FastAPI: przepuszcza żądanie albo zwraca 401 po polsku.

    Porównanie przez `secrets.compare_digest`, nie `==`: zwykłe porównanie
    stringów kończy się na pierwszym różniącym się bajcie, więc czas odpowiedzi
    zdradza, ile znaków hasła się zgadza. Przy sekrecie w zmiennej środowiskowej
    to realna, choć wolna, ścieżka odgadnięcia.
    """
    settings = settings or get_settings()
    expected = settings.api_password

    if not expected:
        return  # autoryzacja wyłączona — uruchomienie lokalne

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_MSG_MISSING,
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    if not secrets.compare_digest(x_api_key, expected.get_secret_value()):
        # Bez logowania podanej wartości — trafiłaby do stdout, a stamtąd
        # do agregatora logów, czyli sekret wyciekłby przez własną telemetrię.
        logger.warning("Odrzucone żądanie: nieprawidłowe hasło dostępu")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_MSG_WRONG,
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )
