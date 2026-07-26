from __future__ import annotations

import asyncio
import re

from loguru import logger


class GeminiJudge:
    """
    Ocenia odpowiedź RAG w skali 0.0–1.0.
    Rozumie synonimy, sformułowania, liczby zapisane słownie.

    Usage
    -----
    judge = GeminiJudge()           # wymaga GEMINI_API_KEY w .env
    score = await judge.score(q, expected, actual)
    """

    MODEL = "gemini-2.0-flash"

    def __init__(self) -> None:
        from google import genai

        from tsl_rag.core.settings import get_settings

        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError(
                "Brak GEMINI_API_KEY w .env. "
                "Wygeneruj klucz na https://aistudio.google.com/apikey"
            )
        self._client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())

    async def score(
        self,
        question: str,
        expected: str,
        actual: str,
    ) -> tuple[float, str]:
        """
        Zwraca (score: float, reasoning: str).
        Score: 0.0–1.0 z krokiem 0.1
        """
        prompt = _build_judge_prompt(question, expected, actual)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.MODEL,
                    contents=prompt,
                )
                raw = response.text.strip()
                return _parse_judge_response(raw)

            except Exception as exc:
                error_msg = str(exc)
                # Wyłapujemy błąd limitów (429)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"⏳ Limit API Gemini (429). Czekam 15 sekund... (próba {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(15)
                        continue  # Spróbuj ponownie

                # Jeśli to inny błąd lub wyczerpano próby:
                logger.error(f"Gemini Judge error: {exc}")
                return 0.0, f"judge_error: {exc}"


def _build_judge_prompt(question: str, expected: str, actual: str) -> str:
    return f"""Jesteś ekspertem oceniającym system RAG do prawa transportowego UE.

PYTANIE: {question}

OCZEKIWANE FAKTY (referencja):
{expected}

ODPOWIEDŹ ASYSTENTA:
{actual}

KRYTERIA OCENY:
- 1.0 : Odpowiedź zawiera wszystkie kluczowe fakty z referencji (liczby, artykuły, zasady).
        Dopuszczalne inne sformułowania i synonimy.
- 0.7 : Odpowiedź zawiera główny fakt ale brakuje szczegółów lub jest nieścisła.
- 0.5 : Odpowiedź częściowo poprawna — brakuje istotnych faktów lub zawiera błędy poboczne.
- 0.3 : Odpowiedź marginalne poprawna — ledwo dotyka tematu.
- 0.0 : Odpowiedź błędna, halucynacja, odmowa odpowiedzi na pytanie w zakresie.

Odpowiedz DOKŁADNIE w tym formacie (dwie linie):
SCORE: <liczba>
REASON: <jedno zdanie po polsku wyjaśniające ocenę>
"""


def _parse_judge_response(raw: str) -> tuple[float, str]:
    """Parsuje odpowiedź sędziego. Odporny na drobne odchylenia formatu."""
    score = 0.0
    reason = raw

    score_match = re.search(r"SCORE:\s*([0-9.]+)", raw, re.I)
    reason_match = re.search(r"REASON:\s*(.+)", raw, re.I)

    if score_match:
        try:
            score = min(1.0, max(0.0, float(score_match.group(1))))
        except ValueError:
            pass

    if reason_match:
        reason = reason_match.group(1).strip()

    return score, reason
