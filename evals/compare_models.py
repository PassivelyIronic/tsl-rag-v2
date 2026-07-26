"""
Porównanie darmowych modeli OpenRouter na golden dataset — wybór modelu
generacji do wersji chmurowej TSL_RAG (dla mamy, bez lokalnego GPU).

Wymaga (bez zmian względem obecnego setupu):
  - Lokalnie działającej Ollamy — embeddingi przy retrieval nadal idą tędy
  - Lokalnie działającego Postgresa z zaindeksowanym korpusem
  - OPENROUTER_API_KEY w .env (https://openrouter.ai/keys, bez karty)

Testowany jest WYŁĄCZNIE etap generacji — retrieval (BM25 + dense + rerank)
zostaje identyczny dla każdego modelu, więc różnice w wyniku pochodzą
tylko z jakości LLM-a.

WAŻNE — limity darmowego tieru OpenRouter (stan: lipiec 2026, źródło:
openrouter.ai/docs — sprawdź przed dużym runem, bo się zmienia):
  - 20 zapytań / minutę
  - 50 zapytań / DZIEŃ — WSPÓLNE dla wszystkich darmowych modeli naraz,
    nie per model. Zużyjesz 30 na jednym modelu, zostaje 20 na resztę dnia.
  - Jednorazowe doładowanie $10 na koncie -> 1000/dzień, limit nie wygasa.
    Przy 10 dokumentach i rzadkim użyciu mamy to i tak grosze realnego
    zużycia, ale rozwiązuje problem ciasnego limitu na czas testów.
  - Darmowe modele bywają zdejmowane/podmieniane bez ostrzeżenia.
    Zweryfikuj aktualną listę: https://openrouter.ai/models?max_price=0

Przy 15 pytaniach w GOLDEN_DATASET:
  - 1 model  = 15 wywołań
  - 3 modele = 45 wywołań (mieści się w dziennym limicie 50 bez doładowania)
  - 4+ modeli w jednym dniu = przekroczysz free cap bez doładowania —
    użyj --resume i rozłóż na kilka dni, albo doładuj $10.

Uruchomienie:
  uv run python -m evals.compare_models
  uv run python -m evals.compare_models --models "deepseek/deepseek-chat-v3.1:free,qwen/qwen3-235b-a22b:free"
  uv run python -m evals.compare_models --use-judge     # ocena Gemini zamiast keyword-match
  uv run python -m evals.compare_models --resume        # pomija modele już zapisane w --output
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from loguru import logger

from evals.golden_dataset.questions import GOLDEN_DATASET, GoldenQuestion
from evals.run_evals import _aggregate, _print_result, evaluate_question
from tsl_rag.core.settings import get_settings
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever

app = typer.Typer(add_completion=False)

# Kandydat domyślny — JEDEN, i to ten, który faktycznie zwrócił wyniki
# w tym projekcie (evals/results/model_comparison.json).
#
# Poprzednia lista domyślna zawierała trzy slugi, z których żaden nie działał:
#   deepseek/deepseek-chat-v3.1:free  → 404, wypadł z darmowej puli
#   qwen/qwen3-235b-a22b:free         → 404, wypadł z darmowej puli
#   meta-llama/llama-3.3-70b:free     → 400, urwane "-instruct" w slugu
# Domyślne uruchomienie spalało więc dzienny limit na trzy pewne porażki.
#
# Nowych kandydatów dopisuj świadomie przez --models, po sprawdzeniu sluga
# w zakładce API modelu (nazwa w katalogu ≠ slug API), a nie z pamięci.
DEFAULT_CANDIDATES = [
    "nvidia/nemotron-nano-9b-v2:free",
]

# Sekundy między wywołaniami generacji — pokrywa limit OpenRouter (20/min,
# czyli >=3s) oraz, jeśli --use-judge, limit Gemini (ten sam interwał co
# w evals/run_evals.py).
_SLEEP_BETWEEN_CALLS = 5.0

_MAX_RETRIES = 3


async def _evaluate_question_safe(
    question: GoldenQuestion,
    retriever: HybridRetriever,
    generator: RAGGenerator,
    judge,
) -> dict:
    """
    evaluate_question() z retry — jedno nieudane pytanie (429/5xx/timeout,
    darmowe modele bywają niestabilne) nie ma wywalać całego przebiegu
    modelu, tylko zostać zapisane jako porażka tego pytania.
    """
    last_error = ""
    for attempt in range(_MAX_RETRIES):
        try:
            return await evaluate_question(question, retriever, generator, judge)
        except Exception as exc:
            last_error = str(exc)
            transient = any(
                token in last_error for token in ("429", "500", "502", "503", "timeout", "Timeout")
            )
            if attempt < _MAX_RETRIES - 1 and transient:
                wait = 15 * (attempt + 1)
                logger.warning(
                    f"  ⏳ Błąd ({last_error[:80]}), czekam {wait}s "
                    f"(próba {attempt + 1}/{_MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
                continue
            logger.error(f"  ✗ Pytanie nieudane po {attempt + 1} próbach: {last_error[:120]}")
            break

    return {
        "question": question.question,
        "category": question.category,
        "expected_docs": question.expected_docs,
        "answer_score": 0.0,
        "citation_hit_rate": 0.0,
        "has_answer": False,
        "correctly_refused": False,
        "incorrectly_refused": False,
        "latency_ms": 0,
        "cited_docs": [],
        "judge_reasoning": None,
        "answer_preview": f"[BŁĄD: {last_error[:150]}]",
    }


def _configure_openrouter_model(model_id: str) -> None:
    """Przełącza globalny Settings na dany model OpenRouter (cache_clear,
    bo get_settings() jest @lru_cache)."""
    os.environ["CHAT_PROVIDER"] = "openrouter"
    os.environ["OPENROUTER_CHAT_MODEL"] = model_id
    get_settings.cache_clear()
    get_settings()  # rzuci ValidationError jeśli brak OPENROUTER_API_KEY


async def evaluate_model(
    model_id: str,
    retriever: HybridRetriever,
    judge,
) -> dict:
    _configure_openrouter_model(model_id)
    generator = RAGGenerator()

    results_list: list[dict] = []
    for i, question in enumerate(GOLDEN_DATASET):
        logger.info(f"  [{model_id}] [{i + 1}/{len(GOLDEN_DATASET)}] {question.question[:60]}")
        result = await _evaluate_question_safe(question, retriever, generator, judge)
        results_list.append(result)
        _print_result(result)

        if i < len(GOLDEN_DATASET) - 1:
            await asyncio.sleep(_SLEEP_BETWEEN_CALLS)

    return {"model": model_id, "summary": _aggregate(results_list), "results": results_list}


async def run_comparison(
    models: list[str],
    output_path: Path,
    use_judge: bool,
    resume: bool,
) -> None:
    total_calls = len(models) * len(GOLDEN_DATASET)
    logger.info(
        f"Porównanie {len(models)} modeli x {len(GOLDEN_DATASET)} pytań "
        f"= {total_calls} wywołań generacji."
    )
    if total_calls > 50:
        logger.warning(
            f"{total_calls} wywołań przekracza darmowy dzienny limit OpenRouter "
            "(50/dzień bez doładowania, WSPÓLNE dla wszystkich modeli). "
            "Doładuj $10 na koncie (limit -> 1000/dzień) albo testuj mniej "
            "modeli na raz i douruchamiaj kolejne dni z --resume."
        )

    all_results: dict[str, dict] = {}
    if resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        all_results = existing.get("models", {})
        logger.info(f"Wznawiam — już przetestowane: {list(all_results.keys())}")

    judge = None
    if use_judge:
        from evals.judge import GeminiJudge

        judge = GeminiJudge()
        logger.info(f"LLM Judge aktywny: {GeminiJudge.MODEL}")
    else:
        logger.info("Tryb keyword-match (bez --use-judge)")

    async with HybridRetriever() as retriever:
        for model_id in models:
            if resume and model_id in all_results and "error" not in all_results[model_id]:
                logger.info(f"Pomijam {model_id} (już w wynikach, --resume)")
                continue

            logger.info(f"\n{'=' * 70}\nMODEL: {model_id}\n{'=' * 70}")
            try:
                all_results[model_id] = await evaluate_model(model_id, retriever, judge)
            except Exception as exc:
                logger.error(f"Model {model_id} przerwany: {exc}")
                all_results[model_id] = {"model": model_id, "error": str(exc)}

            # zapis po każdym modelu — postęp przeżywa błąd/limit w kolejnym
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps({"models": all_results}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    _print_comparison_table(all_results)
    logger.info(f"Wyniki zapisane: {output_path}")


def _print_comparison_table(all_results: dict[str, dict]) -> None:
    print(f"\n{'=' * 92}")
    print("PORÓWNANIE MODELI")
    print(f"{'=' * 92}")
    print(
        f"{'model':42s} {'answer':>8s} {'cite_rec':>9s} "
        f"{'cite_prec':>10s} {'refusal':>8s} {'latency':>10s}"
    )
    print("-" * 92)

    rankable: list[tuple[str, float, float]] = []
    for model_id, r in all_results.items():
        if "error" in r:
            print(f"{model_id:42s}  BŁĄD: {r['error'][:40]}")
            continue
        s = r["summary"]
        refusal = s["refusal_precision"]
        refusal_str = f"{refusal:.2f}" if refusal is not None else "—"
        print(
            f"{model_id:42s} {s['avg_answer_score']:8.3f} "
            f"{s['avg_citation_hit_rate']:9.3f} "
            f"{s.get('avg_citation_precision', 0.0):10.3f} {refusal_str:>8s} "
            f"{s['avg_latency_ms']:9d}ms"
        )
        rankable.append((model_id, s["avg_answer_score"], s["avg_citation_hit_rate"]))

    if rankable:
        rankable.sort(key=lambda x: (x[1], x[2]), reverse=True)
        print("\nRanking (answer_score, potem citation_hit):")
        for i, (model_id, score, cite) in enumerate(rankable, 1):
            print(f"  {i}. {model_id}  (answer={score:.3f}, citation={cite:.3f})")
    print("=" * 92)


@app.command()
def main(
    models: str = typer.Option(
        ",".join(DEFAULT_CANDIDATES),
        "--models",
        "-m",
        help="Lista modeli OpenRouter, ID rozdzielone przecinkiem",
    ),
    output: Path = typer.Option(
        Path("evals/results/model_comparison.json"), "--output", "-o"
    ),
    use_judge: bool = typer.Option(
        False, "--use-judge", "-j", help="Gemini jako LLM-as-a-judge zamiast keyword-match"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Pomiń modele już zapisane w pliku --output"
    ),
) -> None:
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    asyncio.run(run_comparison(model_list, output, use_judge, resume))


if __name__ == "__main__":
    app()
