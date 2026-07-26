"""
Wymuszenie UTF-8 na wyjściu konsoli.

Powód: domyślne kodowanie konsoli na polskim Windowsie to cp1250, które nie
zna znaków spoza swojej strony kodowej. Wypisanie strzałki "→" albo "✓"
kończyło się wtedy UnicodeEncodeError i przerwaniem całego ingestu — po
przetworzeniu części plików, bez informacji, że reszta nie została zrobiona.
Awaria wyglądała na błąd pipeline'u, a była błędem wypisywania tekstu.

errors="replace" jest tu świadome: gorszy znak zapytania w wyjściu niż
wywalony proces wsadowy.
"""

from __future__ import annotations

import sys


def ensure_utf8_output() -> None:
    """Przełącza stdout/stderr na UTF-8, jeśli jeszcze nim nie są."""
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        if encoding and encoding.lower().replace("-", "") == "utf8":
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
