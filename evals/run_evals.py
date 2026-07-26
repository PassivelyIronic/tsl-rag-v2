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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from loguru import logger

from evals.golden_dataset.questions import GOLDEN_DATASET, GoldenQuestion
from tsl_rag.core.models import RetrievalRequest
from tsl_rag.generation.generator import RAGGenerator
from tsl_rag.retrieval.retriever import HybridRetriever

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
    judge=None,  # GeminiJudge | None
) -> dict:
    t0 = time.monotonic()

    request = RetrievalRequest(query=question.question, top_k=20, rerank_top_n=5)
    results = await retriever.retrieve(request)
    response = await generator.generate(question.question, results)
    latency_ms = int((time.monotonic() - t0) * 1000)

    answer_lower = response.answer.lower()
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
        # Fallback: keyword match
        key_facts = [f.strip() for f in question.expected_answer.lower().split(",")]
        fact_hits = sum(1 for f in key_facts if f in answer_lower)
        answer_score = fact_hits / len(key_facts) if key_facts else 1.0

    cited_docs = {c.document_id for c in response.citations}
    expected_set = set(question.expected_docs)
    citation_hit = len(cited_docs & expected_set) / len(expected_set) if expected_set else 1.0

    return {
        "question": question.question,
        "category": question.category,
        "expected_docs": question.expected_docs,
        "answer_score": round(answer_score, 3),
        "citation_hit_rate": round(citation_hit, 3),
        "has_answer": response.has_answer,
        "correctly_refused": is_refused and is_out_of_scope,
        "incorrectly_refused": is_refused and not is_out_of_scope,
        "latency_ms": latency_ms,
        "cited_docs": list(cited_docs),
        "judge_reasoning": judge_reasoning,
        "answer_preview": response.answer[:200],
    }


async def run_evaluation(output_path: Path | None, use_judge: bool) -> None:
    judge = None
    if use_judge:
        from evals.judge import GeminiJudge

        judge = GeminiJudge()
        logger.info(f"LLM Judge aktywny: {GeminiJudge.MODEL}")
    else:
        logger.info("Tryb keyword-match (bez --use-judge)")

    results_list: list[dict] = []

    async with HybridRetriever() as retriever:
        generator = RAGGenerator()
        for i, question in enumerate(GOLDEN_DATASET):
            logger.info(f"[{i+1}/{len(GOLDEN_DATASET)}] {question.question[:70]}")
            result = await evaluate_question(question, retriever, generator, judge)
            results_list.append(result)
            _print_result(result)

            if use_judge and i < len(GOLDEN_DATASET) - 1:
                logger.info("⏳ Czekam 5 sekund na zresetowanie limitów Gemini API...")
                await asyncio.sleep(5)

    summary = _aggregate(results_list)
    _print_summary(summary, use_judge)

    output = {
        "summary": summary,
        "results": results_list,
        "judge_model": GeminiJudge.MODEL if use_judge else "keyword_match",
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

    per_cat = {
        cat: {
            "count": len(items),
            "avg_answer_score": round(sum(i["answer_score"] for i in items) / len(items), 3),
            "avg_citation_hit": round(sum(i["citation_hit_rate"] for i in items) / len(items), 3),
        }
        for cat, items in categories.items()
    }

    out_of_scope = [r for r in results if r["category"] == "out_of_scope"]
    in_scope = [r for r in results if r["category"] != "out_of_scope"]

    return {
        "total_questions": n,
        "avg_answer_score": round(sum(r["answer_score"] for r in results) / n, 3),
        "avg_citation_hit_rate": round(sum(r["citation_hit_rate"] for r in results) / n, 3),
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


def _print_result(r: dict) -> None:
    ok = "✓" if r["answer_score"] >= 0.7 and r["citation_hit_rate"] >= 0.5 else "✗"
    reason = f" → {r['judge_reasoning']}" if r.get("judge_reasoning") else ""
    print(
        f"  {ok} [{r['category']:15s}] "
        f"fact={r['answer_score']:.2f} cite={r['citation_hit_rate']:.2f} "
        f"{r['latency_ms']}ms{reason}"
    )


def _print_summary(s: dict, use_judge: bool) -> None:
    mode = f"Gemini {GeminiJudge.MODEL if use_judge else ''}" if use_judge else "keyword-match"
    print(f"\n{'='*65}")
    print(f"EVALUATION SUMMARY  [{mode}]")
    print(f"{'='*65}")
    print(f"  Questions      : {s['total_questions']}")
    print(f"  Answer score   : {s['avg_answer_score']:.3f}")
    print(f"  Citation hit   : {s['avg_citation_hit_rate']:.3f}")
    print(f"  Refusal prec.  : {s['refusal_precision']}")
    print(f"  False refusals : {s['false_refusal_rate']}")
    print(f"  Avg latency    : {s['avg_latency_ms']} ms")
    print("\n  Per category:")
    for cat, v in s["per_category"].items():
        print(
            f"    {cat:20s} n={v['count']} fact={v['avg_answer_score']:.2f} cite={v['avg_citation_hit']:.2f}"
        )
    print("=" * 65)


from evals.judge import GeminiJudge as _GJ  # noqa: E402

GeminiJudge = _GJ


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
