"""
Ocena odpowiedzi przez podobieństwo embeddingów.

Po co, skoro jest keyword-match
-------------------------------
Ocena keyword-match sprawdza, czy fragment oczekiwanej odpowiedzi występuje
w tekście DOSŁOWNIE. Karze więc poprawne parafrazy: "dziewięć godzin" nie
dopasuje się do oczekiwanego "9 godzin", a "kara wynosi 50 zł" do "grzywna
w wysokości 50 złotych". Model referencyjny (`gpt-4o-mini`) uzyskał tą metodą
`answer_score` = 0.653 — praktycznie tyle co model dziesięciokrotnie mniejszy,
co jest silną przesłanką, że mierzymy metrykę, a nie jakość.

Po co, skoro jest LLM-as-a-judge
--------------------------------
Sędzia rozumie znaczenie lepiej i potrafi uzasadnić ocenę, ale kosztuje,
zależy od zewnętrznego limitu i **nie jest deterministyczny** — a rozrzut
metryk zależnych od LLM-a między przebiegami identycznego kodu zmierzyliśmy
w tym repo na 0.133. Ten scorer jest darmowy, powtarzalny i korzysta z modelu,
który i tak jest już wczytany do retrievalu.

Czego ten scorer NIE zrobi
--------------------------
Nie wykryje odpowiedzi płynnej i błędnej. Zdanie "dzienny czas jazdy to
11 godzin" jest semantycznie bardzo blisko "9 godzin" — obie mówią o tym samym
przepisie tymi samymi słowami, różnią się jedną liczbą. Dlatego liczby są tu
sprawdzane osobno, dosłownie, a wynik semantyczny jest metryką OBOK
keyword-matcha, nie zamiast niego.
"""

from __future__ import annotations

import re

from evals.matching import fact_matches

# Fakt uznajemy za obecny, gdy podobieństwo przekracza ten próg. Wartość dobrana
# tak, żeby parafraza tego samego faktu przechodziła, a inne zdanie z tego samego
# aktu prawnego — nie. Do skalibrowania na danych, gdy uzbiera się więcej przebiegów.
SIMILARITY_THRESHOLD = 0.80

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_HAS_DIGIT = re.compile(r"\d")


def split_into_sentences(text: str) -> list[str]:
    """
    Dzieli odpowiedź na zdania. Porównujemy fakt ze ZDANIAMI, nie z całą
    odpowiedzią, bo pojedynczy fakt utopiony w długim akapicie daje niskie
    podobieństwo do całości i wyglądałby na nieobecny.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _cosine(a: list[float], b: list[float]) -> float:
    # Wektory z providera są znormalizowane (normalize_embeddings=True),
    # więc iloczyn skalarny jest już cosinusem. Normalizujemy mimo to,
    # żeby scorer nie zależał od cudzego ustawienia.
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def score_answer(
    key_facts: list[str],
    answer: str,
    embed,
) -> dict:
    """
    Zwraca ocenę semantyczną odpowiedzi względem oczekiwanych faktów.

    `embed` to funkcja async przyjmująca listę tekstów i zwracająca wektory —
    w praktyce `get_embedding_provider().embed_query`, opakowane w batch.
    Używamy embed_query dla OBU stron: porównanie jest symetryczne (fakt do
    zdania), a nie zapytanie do dokumentu, więc prefiks "passage: " byłby tu
    niewłaściwy.

    Fakty zawierające liczbę sprawdzamy DOSŁOWNIE, bo różnica między "9 godzin"
    a "11 godzin" jest semantycznie znikoma, a merytorycznie decydująca.
    """
    if not key_facts:
        return {"semantic_score": 1.0, "per_fact": []}

    sentences = split_into_sentences(answer)
    if not sentences:
        return {
            "semantic_score": 0.0,
            "per_fact": [{"fact": f, "similarity": 0.0, "present": False} for f in key_facts],
        }

    vectors = await embed(key_facts + sentences)
    fact_vectors = vectors[: len(key_facts)]
    sentence_vectors = vectors[len(key_facts) :]

    per_fact: list[dict] = []
    for fact, fact_vector in zip(key_facts, fact_vectors, strict=True):
        best = max(_cosine(fact_vector, sv) for sv in sentence_vectors)

        if _HAS_DIGIT.search(fact):
            # Fakt liczbowy: podobieństwo semantyczne nie wystarcza, liczba musi
            # wystąpić. fact_matches pilnuje granicy cyfry, więc "50" nie trafia
            # w "1500".
            present = fact_matches(fact, answer)
            kind = "liczbowy"
        else:
            present = best >= SIMILARITY_THRESHOLD
            kind = "opisowy"

        per_fact.append(
            {"fact": fact, "similarity": round(best, 3), "present": present, "kind": kind}
        )

    hits = sum(1 for f in per_fact if f["present"])
    return {"semantic_score": round(hits / len(key_facts), 3), "per_fact": per_fact}
