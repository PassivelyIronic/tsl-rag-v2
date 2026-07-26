from __future__ import annotations

import re
import time
from textwrap import dedent

from loguru import logger

from tsl_rag.core.llm_client import get_chat_client
from tsl_rag.core.models import Citation, QueryResponse
from tsl_rag.core.settings import Settings, get_settings
from tsl_rag.retrieval.retriever import RetrievalResult

SYSTEM_PROMPT = dedent("""\
    Jesteś specjalistycznym asystentem prawnym ds. zgodności z przepisami
    transportu i logistyki w UE. Odpowiadasz WYŁĄCZNIE po polsku.
    Twoim JEDYNYM źródłem wiedzy są fragmenty dokumentów podane poniżej.
    NIE wolno Ci korzystać z żadnej wiedzy zewnętrznej ani domysłów.

    ZASADY BEZWZGLĘDNE:
    1. Odpowiadaj TYLKO na podstawie podanego kontekstu.
    2. Po każdym fakcie MUSISZ dodać cytowanie w formacie:
       [id_dokumentu | Art. X] lub [id_dokumentu | ust. Y]
       Przykład: "Dzienny czas jazdy nie może przekroczyć 9 godzin. [ec_561_2006 | Art. 6]"
    3. Jeśli kontekst CAŁKOWICIE nie pozwala na udzielenie odpowiedzi, napisz TYLKO I WYŁĄCZNIE:
       "Nie mogę odpowiedzieć na to pytanie na podstawie dostępnych dokumentów."
       BEZWZGLĘDNY ZAKAZ: Jeśli udzieliłeś jakiejkolwiek odpowiedzi (nawet częściowej), NIE dodawaj tej frazy na końcu!
    4. Nie zmieniaj znaczenia prawnego podczas parafrazowania.
    5. Gdy przepisy różnych dokumentów są sprzeczne, podaj OBA i wskaż
       który ma pierwszeństwo (Rozporządzenie UE > Dyrektywa > AETR).
    6. Zawsze podawaj dokładne liczby (godziny, odległości, kary) — bez zaokrągleń.

    PRZYKŁADY POPRAWNYCH ODPOWIEDZI:
    - "Dzienny czas prowadzenia pojazdu nie może przekroczyć 9 godzin. [ec_561_2006 | Art. 6(1)]"
    - "Kara za to naruszenie wynosi od 500 do 2000 PLN. [tariff_driver_2022 | ust. 3]"
    - "Nie mogę odpowiedzieć na to pytanie na podstawie dostępnych dokumentów."
""")

_NO_ANSWER_MARKER = "Nie mogę odpowiedzieć"


def _reasoning_kwargs(settings: Settings) -> dict:
    """
    Parametr sterujący rozumowaniem, w formie właściwej dla providera.

    OpenRouter przyjmuje ujednolicone `reasoning: {"effort": ...}` w ciele
    żądania, reszta providerów zgodnych z OpenAI — `reasoning_effort`.
    Puste ustawienie oznacza, że nie wysyłamy nic i zostaje zachowanie domyślne.
    """
    effort = settings.llm_reasoning_effort
    if not effort:
        return {}
    if settings.chat_provider == "openrouter":
        return {"extra_body": {"reasoning": {"effort": effort}}}
    return {"reasoning_effort": effort}


class RAGGenerator:
    """
    Generuje odpowiedź na podstawie pytania i listy RetrievalResult.

    Usage
    -----
    generator = RAGGenerator()
    response  = await generator.generate(query, retrieval_results)
    """

    async def generate(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> QueryResponse:
        t0 = time.monotonic()
        settings = get_settings()
        client = get_chat_client(settings)

        context_block, used_results = _build_context(results, settings.max_context_chars)

        user_message = _build_user_message(query, context_block)

        logger.debug(f"Calling LLM for query: '{query[:80]}'")
        response = await client.chat.completions.create(
            model=settings.active_llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            **_reasoning_kwargs(settings),
        )

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()

        if not answer:
            # Model rozumujący potrafi zużyć cały budżet max_tokens na łańcuch
            # rozumowania i zwrócić PUSTĄ treść. Zmierzone na
            # nvidia/nemotron-nano-9b-v2:free: 455 z 621 tokenów wyjścia poszło
            # na reasoning. Pusta odpowiedź jest dla użytkownika gorsza niż
            # odmowa — wygląda jak zawieszenie systemu, a nie jak brak danych.
            usage = getattr(response, "usage", None)
            logger.error(
                f"Model {settings.active_llm_model} zwrócił pustą treść "
                f"(finish_reason={choice.finish_reason}, usage={usage})"
            )
            answer = (
                "Nie udało się przygotować odpowiedzi — model językowy zwrócił pustą "
                "treść. Spróbuj zadać pytanie jeszcze raz, najlepiej krócej."
            )
            has_answer = False
        else:
            has_answer = _NO_ANSWER_MARKER not in answer
        latency_ms = int((time.monotonic() - t0) * 1000)

        citations = _extract_citations(answer, used_results)

        logger.info(
            f"generate() → has_answer={has_answer}, "
            f"citations={len(citations)}, latency={latency_ms}ms"
        )

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=[],  # wypełniane przez API layer jeśli debug=True
            model_used=settings.active_llm_model,
            latency_ms=latency_ms,
            has_answer=has_answer,
            metadata={
                "chunks_in_context": len(used_results),
                "context_chars": len(context_block),
            },
        )


def _build_context(
    results: list[RetrievalResult],
    max_context_chars: int,
) -> tuple[str, list[RetrievalResult]]:
    """
    Buduje blok kontekstu z chunków.
    Przycina do max_context_chars (Settings), zachowując najlepiej ocenione
    chunki — lista wchodzi tu już posortowana po final_score.
    Zwraca (context_text, użyte_results).
    """
    lines: list[str] = []
    used_results: list[RetrievalResult] = []
    total_chars = 0

    for result in results:
        chunk = result.chunk
        m = chunk.metadata

        header_parts = [m.document_id]
        if m.article:
            header_parts.append(f"Art. {m.article}")
        if m.paragraph:
            header_parts.append(f"§{m.paragraph}")
        header = " | ".join(header_parts)

        block = f"[{header}]\n{chunk.text}\n"

        if total_chars + len(block) > max_context_chars:
            logger.debug(f"Context limit reached at chunk {chunk.chunk_id}")
            break

        lines.append(block)
        used_results.append(result)
        total_chars += len(block)

    return "\n".join(lines), used_results


def _build_user_message(query: str, context: str) -> str:
    return dedent(f"""\
        KONTEKST (Akty prawne):
        --------
        {context}
        --------

        PYTANIE: {query}

        Odpowiedz w języku polskim, opierając się rygorystycznie na powyższym kontekście. Na końcu zdań dodaj cytowania w wymaganym formacie.
    """)


# Numer jednostki z cytowania: "Art. 6(1)" → "6(1)", "Artykuł 11" → "11",
# "ust. 3" → "3". Bierzemy wszystko po słowie kluczowym do końca fragmentu.
# Dłuższe formy przed krótszymi — inaczej "art" złapałoby prefiks "Artykuł"
# i przechwyciło resztę słowa ("ykuł") jako numer.
_ARTICLE_RE = re.compile(r"(?:artykuł|artykul|art\.?)\s*([\w().\-–/]+)", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"(?:ust\.?|§|par\.?)\s*([\w().\-–/]+)", re.IGNORECASE)


def _article_number(value: str | None) -> str | None:
    """
    "Artykuł 6" → "6", "Art. 8a" → "8a", "6(1)" → "6(1)".

    Metadane chunków trzymają pełny nagłówek z dokumentu, a cytowanie
    w odpowiedzi zawiera zwykle sam numer. Bez sprowadzenia obu do wspólnej
    postaci dopasowanie chunka po artykule nigdy nie trafiało.
    """
    if not value:
        return None
    match = re.search(r"([\w()./\-–]+)\s*$", value.strip())
    return match.group(1).lower() if match else None


def _extract_citations(
    answer: str,
    used_results: list[RetrievalResult],
) -> list[Citation]:
    """
    Parsuje cytowania z formatu [doc_id | Art. X] w tekście odpowiedzi.

    Źródłem prawdy dla numeru artykułu jest TO, CO NAPISAŁ MODEL, a nie
    metadane pierwszego chunka danego dokumentu. Poprzednia wersja brała
    `candidates[0].chunk.metadata.article`, więc odpowiedź cytująca
    "[ec_561_2006 | Art. 6]" była raportowana jako Artykuł 11, jeśli tylko
    chunk z artykułem 11 znalazł się wyżej w retrievalu. Przy zasadzie,
    że cytowanie niewłaściwego przepisu jest porażką (CLAUDE.md §5.6),
    zgłaszanie innego artykułu niż wskazany w tekście jest poważniejsze
    niż brak cytowania — bo wygląda na potwierdzone metadanymi.
    """
    by_doc: dict[str, list[RetrievalResult]] = {}
    for r in used_results:
        did = r.chunk.metadata.document_id
        by_doc.setdefault(did, []).append(r)

    pattern = re.compile(r"\[([^\]]+)\]")
    seen: set[tuple[str, str | None, str | None]] = set()
    citations: list[Citation] = []

    for match in pattern.finditer(answer):
        raw = match.group(1)
        parts = [p.strip() for p in raw.split("|")]
        doc_id = parts[0].lower().replace(" ", "_")

        candidates = by_doc.get(doc_id, [])
        if not candidates:
            # Model zacytował dokument, którego nie było w kontekście —
            # nie ma czym tego potwierdzić, więc nie zgłaszamy cytowania.
            continue

        locator = " | ".join(parts[1:]) if len(parts) > 1 else ""
        article_match = _ARTICLE_RE.search(locator)
        paragraph_match = _PARAGRAPH_RE.search(locator)
        article = article_match.group(1) if article_match else None
        paragraph = paragraph_match.group(1) if paragraph_match else None

        # Chunk potwierdzający: ten z pasującym artykułem, jeśli jest.
        # Porównanie po samym numerze, bo metadane trzymają pełny nagłówek
        # ("Artykuł 6"), a z odpowiedzi wyciągamy sam numer ("6").
        chunk = next(
            (
                c.chunk
                for c in candidates
                if article and _article_number(c.chunk.metadata.article) == _article_number(article)
            ),
            candidates[0].chunk,
        )

        # Deduplikacja po (dokument, artykuł, ustęp), nie po surowym tekście —
        # "[x | Art. 6]" i "[x | Art. 6(1)]" to dwa różne cytowania, ale dwa
        # razy ten sam zapis nie ma się powtarzać na liście źródeł.
        key = (doc_id, article, paragraph)
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            Citation(
                document_id=doc_id,
                document_title=chunk.metadata.title,
                article=article or chunk.metadata.article,
                paragraph=paragraph or chunk.metadata.paragraph,
                chunk_id=chunk.chunk_id,
            )
        )

    return citations
