"""
Eval harness z opcjonalnym LLM-as-a-Judge (Gemini).

Uruchomienie:
  uv run python -m evals.run_evals --output evals/results/run_005.json
  uv run python -m evals.run_evals --output evals/results/run_005.json --use-judge
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from loguru import logger

from evals.golden_dataset.questions import GOLDEN_DATASET, GoldenQuestion
from evals.matching import count_matches
from tsl_rag.core.console import ensure_utf8_output
from tsl_rag.core.models import RetrievalRequest
from tsl_rag.core.settings import get_settings
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever

if TYPE_CHECKING:
    # Tylko do adnotacji. Import w runtime jest leniwy (dopiero przy
    # --use-judge), żeby przebieg bez sędziego nie wymagał google-genai
    # ani klucza Gemini — istotne dla joba wsadowego w CI.
    from evals.judge import GeminiJudge

ensure_utf8_output()

app = typer.Typer(add_completion=False)

REFUSAL_PHRASES = [
    "nie mogę odpowiedzieć",
    "nie ma w tekście",
    "nie jest wskazane w",
    "brak informacji",
    "kontekst nie zawiera",
]


def _is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(p in lower for p in REFUSAL_PHRASES)


async def evaluate_question(
    question: GoldenQuestion,
    retriever: HybridRetriever,
    generator: RAGGenerator,
    judge: GeminiJudge | None = None,
) -> dict:
    t0 = time.monotonic()

    # top_k / rerank_top_n z Settings, nie z hardkodu — eval musi mierzyć
    # tę samą konfigurację retrievalu, którą uruchamia API. Wcześniej było
    # tu zaszyte 20/5 niezależnie od .env.
    request = RetrievalRequest(query=question.question)
    results = await retriever.retrieve(request)
    response = await generator.generate(question.question, results)
    latency_ms = int((time.monotonic() - t0) * 1000)

    is_out_of_scope = question.category == "out_of_scope"
    is_refused = _is_refusal(response.answer) or not response.has_answer

    judge_reasoning = None

    if is_out_of_scope:
        answer_score = 1.0 if is_refused else 0.0
        judge_reasoning = (
            "out_of_scope: odmowa poprawna" if is_refused else "out_of_scope: halucynacja"
        )

    elif judge is not None:
        # LLM-as-a-Judge
        answer_score, judge_reasoning = await judge.score(
            question=question.question,
            expected=question.expected_answer,
            actual=response.answer,
        )
        logger.debug(f"  Judge: {answer_score:.1f} — {judge_reasoning}")

    else:
        # Fallback: keyword match na fragmentach z expected_answer.
        # Dopasowanie przez evals.matching, nie zwykłe "in" — fakty liczbowe
        # wymagają granicy cyfry, inaczej oczekiwane "200" potwierdza się
        # przez "2000" w odpowiedzi (kwoty w taryfikatorach: 50-12000).
        key_facts = question.key_facts
        fact_hits = count_matches(key_facts, response.answer)
        answer_score = fact_hits / len(key_facts) if key_facts else 1.0

    cited_docs = {c.document_id for c in response.citations}
    retrieved_docs = {r.chunk.metadata.document_id for r in results}
    expected_set = set(question.expected_docs)

    # Recall cytowań: ile oczekiwanych dokumentów model zacytował.
    citation_hit = len(cited_docs & expected_set) / len(expected_set) if expected_set else 1.0

    # Precyzja cytowań: ile z zacytowanych dokumentów było oczekiwanych.
    # Bez tej metryki model cytujący wszystko, co dostał w kontekście,
    # dostaje recall 1.0 i wygląda na lepszy niż jest.
    citation_precision = len(cited_docs & expected_set) / len(cited_docs) if cited_docs else 1.0

    # Czy retriever w ogóle podał właściwy dokument. To jest ten pomiar,
    # który pozwala odróżnić "retriever nie znalazł" od "model zignorował
    # kontekst" — bez niego citation_hit = 0 jest nieinterpretowalne.
    retrieval_recall = (
        len(retrieved_docs & expected_set) / len(expected_set) if expected_set else 1.0
    )

    return {
        "id": question.id,
        "question": question.question,
        "category": question.category,
        "variant": question.variant,
        "expected_docs": question.expected_docs,
        "answer_score": round(answer_score, 3),
        "citation_hit_rate": round(citation_hit, 3),
        "citation_precision": round(citation_precision, 3),
        "retrieval_recall": round(retrieval_recall, 3),
        "failure_stage": _classify_failure(retrieval_recall, citation_hit, expected_set),
        "has_answer": response.has_answer,
        "correctly_refused": is_refused and is_out_of_scope,
        "incorrectly_refused": is_refused and not is_out_of_scope,
        "latency_ms": latency_ms,
        "cited_docs": sorted(cited_docs),
        "retrieved_docs": sorted(retrieved_docs),
        "judge_reasoning": judge_reasoning,
        "answer_preview": response.answer[:200],
    }


def _classify_failure(
    retrieval_recall: float,
    citation_hit: float,
    expected_set: set[str],
) -> str | None:
    """
    Wskazuje etap, na którym pytanie się wywróciło.

    Sedno problemu z poprzednią wersją harnessu: przy citation_hit = 0 nie
    było jak stwierdzić, czy retriever nie podał właściwego dokumentu, czy
    podał, a model go zignorował. Bez tego każda optymalizacja jest zgadywaniem.
    """
    if not expected_set:
        return None  # out_of_scope — oceniane odmową, nie cytowaniami
    if retrieval_recall == 0.0:
        return "retrieval"  # właściwego dokumentu nie było w kontekście
    if citation_hit == 0.0:
        return "generation"  # dokument był w kontekście, model go nie zacytował
    if citation_hit < 1.0:
        return "generation_partial"
    return None


async def run_evaluation(output_path: Path | None, use_judge: bool) -> None:
    judge: GeminiJudge | None = None
    if use_judge:
        from evals.judge import GeminiJudge as _GeminiJudge

        judge = _GeminiJudge()
        logger.info(f"LLM Judge aktywny: {judge.MODEL}")
    else:
        logger.info("Tryb keyword-match (bez --use-judge)")

    judge_model = judge.MODEL if judge else "keyword_match"

    results_list: list[dict] = []

    async with HybridRetriever() as retriever:
        generator = RAGGenerator()
        for i, question in enumerate(GOLDEN_DATASET):
            logger.info(f"[{i + 1}/{len(GOLDEN_DATASET)}] {question.question[:70]}")
            result = await evaluate_question(question, retriever, generator, judge)
            results_list.append(result)
            _print_result(result)

            if use_judge and i < len(GOLDEN_DATASET) - 1:
                logger.info("⏳ Czekam 5 sekund na zresetowanie limitów Gemini API...")
                await asyncio.sleep(5)

    summary = _aggregate(results_list)
    _print_summary(summary, judge_model)

    output = {
        "summary": summary,
        "results": results_list,
        "judge_model": judge_model,
        "config": _config_snapshot(),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Wyniki zapisane: {output_path}")


def _aggregate(results: list[dict]) -> dict:
    n = len(results)
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    def _avg(items: list[dict], key: str) -> float:
        vals = [i[key] for i in items if i.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    per_cat = {
        cat: {
            "count": len(items),
            "avg_answer_score": _avg(items, "answer_score"),
            "avg_citation_hit": _avg(items, "citation_hit_rate"),
            "avg_citation_precision": _avg(items, "citation_precision"),
            "avg_retrieval_recall": _avg(items, "retrieval_recall"),
        }
        for cat, items in categories.items()
    }

    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]
    in_scope = [r for r in results if r["category"] != "out_of_scope"]

    # Rozkład etapów porażki — odpowiada na pytanie "co naprawiać najpierw":
    # retriever czy prompt/model.
    failure_stages: dict[str, int] = {}
    for r in results:
        stage = r.get("failure_stage")
        if stage:
            failure_stages[stage] = failure_stages.get(stage, 0) + 1

    return {
        "total_questions": n,
        "avg_answer_score": round(sum(r["answer_score"] for r in results) / n, 3),
        "avg_citation_hit_rate": round(sum(r["citation_hit_rate"] for r in results) / n, 3),
        "avg_citation_precision": _avg(results, "citation_precision"),
        "avg_retrieval_recall": _avg(results, "retrieval_recall"),
        "failure_stages": failure_stages,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / n),
        "refusal_precision": round(
            sum(1 for r in out_of_scope if r["correctly_refused"]) / len(out_of_scope), 3
        )
        if out_of_scope
        else None,
        "false_refusal_rate": round(
            sum(1 for r in in_scope if r["incorrectly_refused"]) / len(in_scope), 3
        )
        if in_scope
        else None,
        "per_category": per_cat,
    }


def _config_snapshot() -> dict:
    """
    Zapisuje konfigurację, która wyprodukowała te liczby.

    Bez tego wynik w evals/results/ jest nieinterpretowalny po tygodniu:
    nie wiadomo, jaki model, jakie wagi RRF i jakie top_k stały za metryką.
    Zasada "zero niezweryfikowanych liczb" wymaga, żeby liczbie towarzyszyły
    warunki pomiaru (CLAUDE.md §5.2).
    """
    s = get_settings()
    embedding_model = (
        s.ollama_embed_model if s.embedding_provider == "ollama" else s.openai_embedding_model
    )
    return {
        "embedding_provider": s.embedding_provider,
        "embedding_model": embedding_model,
        "chat_provider": s.chat_provider,
        "chat_model": s.active_llm_model,
        "retrieval_top_k": s.retrieval_top_k,
        "retrieval_rerank_top_n": s.retrieval_rerank_top_n,
        "bm25_weight": s.bm25_weight,
        "dense_weight": s.dense_weight,
        "reranker_model": s.reranker_model,
        "max_context_chars": s.max_context_chars,
    }


def _print_result(r: dict) -> None:
    ok = "✓" if r["answer_score"] >= 0.7 and r["citation_hit_rate"] >= 0.5 else "✗"
    reason = f" → {r['judge_reasoning']}" if r.get("judge_reasoning") else ""
    stage = f" [{r['failure_stage']}]" if r.get("failure_stage") else ""
    print(
        f"  {ok} [{r['category']:15s}] "
        f"fact={r['answer_score']:.2f} cite={r['citation_hit_rate']:.2f} "
        f"ret={r.get('retrieval_recall', 0):.2f} "
        f"{r['latency_ms']}ms{stage}{reason}"
    )


def _print_summary(s: dict, judge_model: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"EVALUATION SUMMARY  [{judge_model}]")
    print(f"{'=' * 72}")
    print(f"  Questions        : {s['total_questions']}")
    print(f"  Answer score     : {s['avg_answer_score']:.3f}")
    print(f"  Citation recall  : {s['avg_citation_hit_rate']:.3f}")
    print(f"  Citation prec.   : {s['avg_citation_precision']:.3f}")
    print(f"  Retrieval recall : {s['avg_retrieval_recall']:.3f}")
    print(f"  Refusal prec.    : {s['refusal_precision']}")
    print(f"  False refusals   : {s['false_refusal_rate']}")
    print(f"  Avg latency      : {s['avg_latency_ms']} ms")
    if s["failure_stages"]:
        stages = ", ".join(f"{k}={v}" for k, v in sorted(s["failure_stages"].items()))
        print(f"  Etap porażki     : {stages}")
    print("\n  Per category:")
    for cat, v in s["per_category"].items():
        print(
            f"    {cat:20s} n={v['count']} "
            f"fact={v['avg_answer_score']:.2f} "
            f"cite={v['avg_citation_hit']:.2f} "
            f"prec={v['avg_citation_precision']:.2f} "
            f"ret={v['avg_retrieval_recall']:.2f}"
        )
    print("=" * 72)


@app.command()
def main(
    output: Path = typer.Option(None, "--output", "-o"),
    use_judge: bool = typer.Option(
        False, "--use-judge", "-j", help="Użyj Gemini jako LLM-as-a-Judge"
    ),
) -> None:
    asyncio.run(run_evaluation(output, use_judge))


if __name__ == "__main__":
    app()
