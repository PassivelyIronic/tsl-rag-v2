from __future__ import annotations

import time
from textwrap import dedent

from loguru import logger

from tsl_rag.core.llm_client import get_chat_client
from tsl_rag.core.models import Citation, QueryResponse
from tsl_rag.core.settings import get_settings
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

# Max tokenów kontekstu (zachowawczo — mistral 7b ma 8k okno)
_MAX_CONTEXT_CHARS = 12_000


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

        context_block, used_results = _build_context(results)

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
        )

        answer = response.choices[0].message.content or ""
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
) -> tuple[str, list[RetrievalResult]]:
    """
    Buduje blok kontekstu z chunków.
    Przycina do _MAX_CONTEXT_CHARS, zachowując najlepiej ocenione chunki.
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

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
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


def _extract_citations(
    answer: str,
    used_results: list[RetrievalResult],
) -> list[Citation]:
    """
    Parsuje cytowania z formatu [doc_id | Art. X] w tekście odpowiedzi.
    Mapuje z powrotem na pełne metadane chunka.
    """
    import re

    by_doc: dict[str, list[RetrievalResult]] = {}
    for r in used_results:
        did = r.chunk.metadata.document_id
        by_doc.setdefault(did, []).append(r)

    pattern = re.compile(r"\[([^\]]+)\]")
    seen: set[str] = set()
    citations: list[Citation] = []

    for match in pattern.finditer(answer):
        raw = match.group(1)
        if raw in seen:
            continue
        seen.add(raw)

        parts = [p.strip() for p in raw.split("|")]
        doc_id = parts[0].lower().replace(" ", "_")

        candidates = by_doc.get(doc_id, [])
        chunk = candidates[0].chunk if candidates else None

        if chunk is None:
            continue

        citations.append(
            Citation(
                document_id=doc_id,
                document_title=chunk.metadata.title,
                article=chunk.metadata.article,
                paragraph=chunk.metadata.paragraph,
                chunk_id=chunk.chunk_id,
            )
        )

    return citations
