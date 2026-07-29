from __future__ import annotations

import re
import time
from textwrap import dedent

from loguru import logger

from tsl_rag.core.llm_client import ChatTarget, get_chat_client_for, resolve_chat_chain
from tsl_rag.core.models import Citation, QueryResponse
from tsl_rag.core.observability import (
    record_answer,
    record_fallback_switch,
    record_provider_error,
    stage,
)
from tsl_rag.core.settings import Settings, get_settings
from tsl_rag.generation.fallback import CircuitBreaker, FailureKind, classify_failure
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


def _reasoning_kwargs(settings: Settings, provider: str | None = None) -> dict:
    """
    Parametr sterujący rozumowaniem, w formie właściwej dla providera.

    OpenRouter przyjmuje ujednolicone `reasoning: {"effort": ...}` w ciele
    żądania, reszta providerów zgodnych z OpenAI — `reasoning_effort`.
    Puste ustawienie oznacza, że nie wysyłamy nic i zostaje zachowanie domyślne.

    `provider` podaje łańcuch fallbacku, bo ogniwo zapasowe bywa u innego
    providera niż `settings.chat_provider` — wysłanie wtedy formy OpenRoutera
    do OpenAI kończy się błędem 400 i niepotrzebnym przejściem dalej.
    """
    effort = settings.llm_reasoning_effort
    if not effort:
        return {}
    if (provider or settings.chat_provider) == "openrouter":
        return {"extra_body": {"reasoning": {"effort": effort}}}
    return {"reasoning_effort": effort}


def _system_prompt(settings: Settings) -> str:
    """
    System prompt z opcjonalnym prefiksem sterującym z konfiguracji.

    Prefiks jest osobną linią przed promptem, bo modele z rodziny Nemotron
    oczekują `/no_think` jako samodzielnego tokenu na początku wiadomości
    systemowej — wklejony w środek zdania przestaje działać.
    """
    prefix = settings.llm_system_prefix.strip()
    if not prefix:
        return SYSTEM_PROMPT
    return f"{prefix}\n{SYSTEM_PROMPT}"


_ALL_FAILED_MESSAGE = (
    "Usługa modelu językowego jest chwilowo niedostępna u wszystkich "
    "skonfigurowanych dostawców. Spróbuj ponownie za kilka minut."
)


class RAGGenerator:
    """
    Generuje odpowiedź na podstawie pytania i listy RetrievalResult.

    Przechodzi po łańcuchu (provider, model) z konfiguracji: przy awarii
    jednego ogniwa próbuje kolejnego, zamiast zwracać użytkownikowi błąd.
    Bezpiecznik żyje w instancji, więc generator ma być tworzony RAZ na proces
    — tak jak HybridRetriever. Nowa instancja per request zerowałaby licznik
    porażek i bezpiecznik nie chroniłby przed niczym.

    Usage
    -----
    generator = RAGGenerator()
    response  = await generator.generate(query, retrieval_results)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        # Ustawienia wstrzykiwane, bo próg bezpiecznika czytany jest tutaj,
        # a ścieżka zapytania dostaje Settings argumentem. Bez tego test
        # podający własny próg konfigurowałby co innego, niż mierzy.
        settings = settings or get_settings()
        self._breaker = CircuitBreaker(
            failures=settings.chat_breaker_failures,
            cooldown_s=settings.chat_breaker_cooldown_s,
        )

    async def generate(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> QueryResponse:
        t0 = time.monotonic()
        settings = get_settings()

        context_block, used_results = _build_context(results, settings.max_context_chars)
        user_message = _build_user_message(query, context_block)
        messages = [
            {"role": "system", "content": _system_prompt(settings)},
            {"role": "user", "content": user_message},
        ]

        chain = resolve_chat_chain(settings)
        with stage("generate", chain_length=len(chain)) as span:
            answer, has_answer, model_used, switches = await self._generate_with_fallback(
                query, messages, chain, settings
            )
            span.set_attribute("model_used", model_used)
            span.set_attribute("fallback_switches", switches)

        latency_ms = int((time.monotonic() - t0) * 1000)
        citations = _extract_citations(answer, used_results)

        if answer == _ALL_FAILED_MESSAGE:
            record_answer("all_providers_failed")
        elif has_answer:
            record_answer("answered")
        else:
            record_answer("refused")

        logger.info(
            f"generate() → has_answer={has_answer}, model={model_used}, "
            f"citations={len(citations)}, przełączeń={switches}, latency={latency_ms}ms"
        )

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieved_chunks=[],  # wypełniane przez API layer jeśli debug=True
            model_used=model_used,
            latency_ms=latency_ms,
            has_answer=has_answer,
            metadata={
                "chunks_in_context": len(used_results),
                "context_chars": len(context_block),
                "fallback_switches": switches,
                "chain": [str(t) for t in chain],
            },
        )

    async def _generate_with_fallback(
        self,
        query: str,
        messages: list[dict],
        chain: list[ChatTarget],
        settings: Settings,
    ) -> tuple[str, bool, str, int]:
        """
        Przechodzi po ogniwach łańcucha. Zwraca (odpowiedź, has_answer, model, przełączenia).

        Pusta treść jest traktowana jak awaria i przełącza na kolejne ogniwo.
        Powód jest zmierzony: nvidia/nemotron-nano-9b-v2:free zwrócił pustkę
        w 2 z 21 pytań (run_015), bo łańcuch rozumowania zjadał cały budżet
        max_tokens. Zwrócenie tej pustki użytkownikowi, gdy skonfigurowano
        model zapasowy, byłoby marnowaniem odporności, którą się ma.
        """
        switches = 0
        last_error: str | None = None

        for index, target in enumerate(chain):
            if self._breaker.is_open(str(target)):
                logger.warning(f"Pomijam {target} — bezpiecznik otwarty")
                continue

            if index > 0:
                switches += 1
                record_fallback_switch(str(chain[index - 1]), str(target))
                logger.warning(
                    f"Fallback: przełączam na {target} "
                    f"(ogniwo {index + 1}/{len(chain)}, powód: {last_error})"
                )

            try:
                client = get_chat_client_for(target.provider, settings)
            except ValueError as exc:
                # Brak klucza dla ogniwa zapasowego nie może wywalić zapytania,
                # które ogniwo główne obsłużyłoby poprawnie.
                last_error = f"konfiguracja: {exc}"
                logger.error(f"Ogniwo {target} nieużywalne — {exc}")
                self._breaker.record_failure(str(target))
                continue

            # Span per PRÓBĘ, nie per zapytanie: przy fallbacku ślad ma pokazać,
            # ile czasu zjadło ogniwo, które i tak zawiodło. To jest ta liczba,
            # której brakuje przy diagnozowaniu "dlaczego odpowiedź szła minutę".
            try:
                with stage(
                    "llm_call", provider=target.provider, model=target.model, attempt=index + 1
                ):
                    logger.debug(f"Wywołanie {target} dla zapytania: '{query[:80]}'")
                    response = await client.chat.completions.create(
                        model=target.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=settings.llm_temperature,
                        max_tokens=settings.llm_max_tokens,
                        **_reasoning_kwargs(settings, target.provider),
                    )
            except Exception as exc:  # noqa: BLE001 — klasyfikujemy niżej
                kind = classify_failure(exc)
                last_error = f"{kind.value}: {type(exc).__name__}"
                logger.error(f"Ogniwo {target} zawiodło [{kind.value}]: {str(exc)[:200]}")
                record_provider_error(target.provider, target.model, kind.value)
                self._breaker.record_failure(str(target))
                continue

            choice = response.choices[0]
            answer = (choice.message.content or "").strip()

            if not answer:
                usage = getattr(response, "usage", None)
                last_error = FailureKind.EMPTY.value
                logger.error(
                    f"Ogniwo {target} zwróciło pustą treść "
                    f"(finish_reason={choice.finish_reason}, usage={usage})"
                )
                record_provider_error(target.provider, target.model, FailureKind.EMPTY.value)
                self._breaker.record_failure(str(target))
                continue

            self._breaker.record_success(str(target))
            return answer, _NO_ANSWER_MARKER not in answer, target.model, switches

        # Wszystkie ogniwa padły. Komunikat po polsku, bo trafia wprost
        # do nietechnicznego użytkownika.
        logger.error(f"Wszystkie ogniwa łańcucha zawiodły. Ostatni powód: {last_error}")
        model_used = chain[-1].model if chain else "brak"
        return _ALL_FAILED_MESSAGE, False, model_used, switches


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
    że cytowanie niewłaściwego przepisu jest porażką,
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
