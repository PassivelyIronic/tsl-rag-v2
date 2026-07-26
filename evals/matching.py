"""
Dopasowywanie oczekiwanych faktów do tekstu.

Wspólne dla oceny keyword-match (`run_evals`) i weryfikacji datasetu względem
korpusu (`verify_dataset`), żeby obie odpowiadały na to samo pytanie w ten
sam sposób.

Dwa problemy, które to rozwiązuje:

1. **Zapis liczb.** Taryfikatory i akty zapisują kwoty raz jako "12 000",
   raz "12000", a model w odpowiedzi dopisuje "zł". Bez normalizacji poprawna
   odpowiedź nie dopasowywała się z powodu spacji.
2. **Fragmenty liczbowe łapały większe liczby.** Oczekiwany fakt "200"
   dopasowywał się do "2000" i "1200", a "50" do "150" i "1500" — czyli
   ocena potwierdzała fakt, którego w odpowiedzi nie było. Przy taryfikatorach,
   gdzie kwoty są rzędu 50-12000, to nie jest przypadek brzegowy.
"""

from __future__ import annotations

import re
import unicodedata

_MANUAL_FOLD = str.maketrans({"ł": "l", "Ł": "l"})
_NUMERIC_FACT_RE = re.compile(r"^[\d\s.,]+$")


def fold_diacritics(text: str) -> str:
    """Sprowadza polskie znaki do ASCII (tak jak tokenizer BM25)."""
    folded = unicodedata.normalize("NFKD", text.translate(_MANUAL_FOLD))
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def normalize(text: str, *, fold: bool = False) -> str:
    """
    Małe litery, zwinięte białe znaki, scalone spacje wewnątrz liczb.

    Zwijanie białych znaków jest konieczne, bo ekstrakcja z PDF-a zostawia
    podwójne spacje przy liczbach — korpus zawiera "9  godzin", nie "9 godzin".
    """
    out = text.lower()
    if fold:
        out = fold_diacritics(out)
    out = " ".join(out.split())
    # Spacja między cyframi ("12 000" -> "12000") — nie jest różnicą znaczącą.
    return re.sub(r"(?<=\d) (?=\d)", "", out)


def fact_matches(fact: str, text: str, *, fold: bool = False) -> bool:
    """
    Czy oczekiwany fakt występuje w tekście.

    Fakty czysto liczbowe dopasowywane są z granicą cyfry, żeby "200"
    nie trafiało w "2000". Pozostałe jako zwykły podciąg.
    """
    needle = normalize(fact, fold=fold)
    haystack = normalize(text, fold=fold)
    if not needle:
        return False

    if _NUMERIC_FACT_RE.match(needle):
        return re.search(rf"(?<!\d){re.escape(needle)}(?!\d)", haystack) is not None
    return needle in haystack


def count_matches(facts: list[str], text: str, *, fold: bool = False) -> int:
    return sum(1 for f in facts if fact_matches(f, text, fold=fold))
