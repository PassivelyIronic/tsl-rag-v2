"""
Weryfikacja golden datasetu względem korpusu w bazie.

Sprawdza dla każdego pytania, czy fragmenty `expected_answer` faktycznie
występują w tekście dokumentów wskazanych w `expected_docs`.

Po co to istnieje
-----------------
Pytania są generowane nad korpusem PDF przez model, poza repo. Model potrafi
przypisać fakt do niewłaściwego aktu albo sformułować oczekiwaną odpowiedź
słowami, których w dokumencie nie ma. Jedno i drugie jest gorsze niż brak
pytania: od momentu scalenia mierzylibyśmy system względem nieprawdy,
a niepowodzenia wyglądałyby na winę retrievalu albo modelu.

Ten skrypt nie zastępuje przeczytania `source_note` przez człowieka —
sprawdza obecność ciągów znaków, nie sens prawny. Wyłapuje jednak dwie
najczęstsze klasy błędu: zły dokument w `expected_docs` i oczekiwaną
odpowiedź sformułowaną inaczej niż w źródle.

Uruchomienie:
  uv run python -m evals.verify_dataset
  uv run python -m evals.verify_dataset --json path/do/nowych_pytan.json
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg
import typer

from evals.golden_dataset.questions import GoldenQuestion, load_dataset
from evals.matching import fact_matches, normalize
from tsl_rag.core.console import ensure_utf8_output
from tsl_rag.core.settings import get_settings

ensure_utf8_output()

app = typer.Typer(add_completion=False)

_DOC_TEXT_SQL = """
SELECT document_id, string_agg(text, ' ' ORDER BY chunk_id) AS full_text
FROM document_chunks
GROUP BY document_id;
"""


def _prepare(text: str) -> str:
    """
    Normalizacja przed porównaniem — ta sama, której używa ocena odpowiedzi.

    Miękkie łączniki (U+00AD) **wraz z następującym po nich białym znakiem**
    są usuwane, bo ekstrakcja z PDF-a rozrywa nimi słowa w miejscu podziału
    wiersza: korpus zawiera "przynaj\xad mniej" i "wyko\xad rzystać".
    Usunięcie samego znaku nie scala słowa — zostaje spacja w środku.

    To tylko obejście po stronie pomiaru. Sam korpus nadal jest tym zepsuty
    (1258 wystąpień w 307 z 444 chunków), co psuje tokenizację BM25 —
    naprawa w ingeście jest zadaniem Fazy 1 w PLAN.md.
    """
    return normalize(re.sub(r"\xad\s*", "", text), fold=True)


async def _load_corpus_texts() -> dict[str, str]:
    settings = get_settings()
    dsn = str(settings.postgres_dsn).replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(dsn=dsn)
    try:
        rows = await conn.fetch(_DOC_TEXT_SQL)
    finally:
        await conn.close()
    return {r["document_id"]: _prepare(r["full_text"]) for r in rows}


def _check(
    questions: list[GoldenQuestion],
    corpus: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Zwraca (pytania_z_problemami, pytania_ok)."""
    problems: list[dict] = []
    fine: list[dict] = []

    for q in questions:
        if q.category == "out_of_scope":
            # Odwrotny warunek: fakty NIE mają być w korpusie. Nie sprawdzamy
            # tego automatycznie, bo pytanie jest z definicji bez oczekiwanej
            # treści — ocenia się je odmową.
            continue

        missing_docs = [d for d in q.expected_docs if d not in corpus]
        haystack = " ".join(corpus.get(d, "") for d in q.expected_docs)
        not_found = [f for f in q.key_facts if not fact_matches(f, haystack, fold=True)]

        # Czy fragment znalazłby się w JAKIMKOLWIEK innym dokumencie? Jeśli tak,
        # to sygnał, że expected_docs wskazuje zły akt, a nie że fakt nie istnieje.
        elsewhere: dict[str, list[str]] = {}
        for fact in not_found:
            hits = [
                doc
                for doc, text in corpus.items()
                if doc not in q.expected_docs and fact_matches(fact, text, fold=True)
            ]
            if hits:
                elsewhere[fact] = hits

        record = {
            "id": q.id,
            "category": q.category,
            "expected_docs": q.expected_docs,
            "source_note": q.source_note,
            "facts_total": len(q.key_facts),
            "facts_missing": not_found,
            "found_in_other_docs": elsewhere,
            "unknown_docs": missing_docs,
        }
        (problems if (not_found or missing_docs) else fine).append(record)

    return problems, fine


def _report(problems: list[dict], fine: list[dict], total: int) -> None:
    print(f"\n{'=' * 78}")
    print("WERYFIKACJA GOLDEN DATASETU WZGLĘDEM KORPUSU")
    print(f"{'=' * 78}")
    print(f"  Pytań sprawdzonych     : {len(problems) + len(fine)} (z {total}, bez out_of_scope)")
    print(f"  Wszystkie fakty w źródle: {len(fine)}")
    print(f"  Do przejrzenia          : {len(problems)}")

    if not problems:
        print("\n  Każdy fragment oczekiwanej odpowiedzi występuje we wskazanym dokumencie.")
        print("=" * 78)
        return

    print(f"\n{'-' * 78}")
    for p in problems:
        print(f"\n  {p['id']}  [{p['category']}]")
        print(f"    expected_docs : {p['expected_docs']}")
        print(f"    source_note   : {p['source_note'] or '(brak)'}")
        if p["unknown_docs"]:
            print(f"    !! dokumenty nieobecne w korpusie: {p['unknown_docs']}")
        for fact in p["facts_missing"]:
            where = p["found_in_other_docs"].get(fact)
            if where:
                print(f"    - {fact!r} NIE ma w tych dokumentach, ale JEST w: {where}")
            else:
                print(f"    - {fact!r} nie występuje w żadnym dokumencie korpusu")
    print(f"\n{'=' * 78}")
    print("  Fragment obecny w innym dokumencie = prawdopodobnie zły expected_docs.")
    print("  Fragment nieobecny nigdzie = odpowiedź sformułowana inaczej niż źródło")
    print("  albo pochodzi z części PDF-a, która nie weszła do korpusu.")
    print("=" * 78)


@app.command()
def main(
    json_path: Path = typer.Option(  # noqa: B008
        None,
        "--json",
        help="Alternatywny plik z pytaniami (domyślnie golden dataset repo)",
    ),
    output: Path = typer.Option(None, "--output", "-o", help="Zapisz raport do JSON"),  # noqa: B008
) -> None:
    """Sprawdza, czy oczekiwane odpowiedzi mają oparcie w korpusie."""
    if json_path:
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        items = raw["questions"] if isinstance(raw, dict) else raw
        questions = [
            GoldenQuestion(
                id=i.get("id", ""),
                question=i["question"],
                expected_answer=i.get("expected_answer", ""),
                expected_docs=i.get("expected_docs", []),
                category=i["category"],
                variant=i.get("variant", "standard"),
                expected_articles=i.get("expected_articles", []),
                source_note=i.get("source_note", ""),
            )
            for i in items
        ]
    else:
        questions = load_dataset()

    corpus = asyncio.run(_load_corpus_texts())
    problems, fine = _check(questions, corpus)
    _report(problems, fine, len(questions))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"problems": problems, "ok": fine}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nRaport zapisany: {output}")

    if problems:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
