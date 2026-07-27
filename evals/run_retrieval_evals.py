"""
Ewaluacja SAMEGO retrievalu — bez wywołania modelu generującego.

Po co osobne narzędzie
----------------------
Pomiar wariancji (`evals/results/run_010`–`run_012`) pokazał, że metryki
zależne od generacji rozrzucają się między przebiegami identycznego kodu
o więcej, niż wynosi efekt typowej zmiany w retrievalu — przy 15 pytaniach
`citation_hit_rate` różnił się o 0.133. Nie da się na nich niczego bramkować.

Metryki retrievalu były w tych samych przebiegach identyczne, bo nie
przechodzą przez LLM. Ten skrypt mierzy wyłącznie je i dzięki temu:

- jest **powtarzalny** — ta sama konfiguracja daje ten sam wynik,
- nie zużywa limitów płatnego ani darmowego API generacji,
- trwa sekundy, nie minuty, więc nadaje się do pracy iteracyjnej,
- rozdziela etapy: dense, BM25, po fuzji RRF i po rerankingu, więc odpowiada
  na pytanie, czy reranker poprawia kolejność i który retriever wnosi trafienie.

Zależności
----------
Po Fazie 2 embeddingi liczą się lokalnie (`sentence-transformers`, CPU), więc
skrypt nie potrzebuje ŻADNEJ usługi zewnętrznej ani klucza API. Wymaga tylko
Postgresa z zaindeksowanym korpusem, czyli **nadaje się na bramkę w CI** —
wystarczy serwis Postgresa i ingest w jobie.

Progi bramkujące trzymane są w `evals/thresholds.yaml`, nie we fladze komendy,
żeby CI i człowiek patrzyli na te same liczby, a ich zmiana była widoczna
w diffie (zasada #1 z CLAUDE.md).

Uruchomienie:
  uv run python -m evals.run_retrieval_evals
  uv run python -m evals.run_retrieval_evals --output evals/results/retrieval_001.json
  uv run python -m evals.run_retrieval_evals --no-gate     # sam pomiar, bez bramki
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
from evals.matching import count_matches
from evals.retrieval_metrics import first_hit_position, recall_at_k, reciprocal_rank
from tsl_rag.core.console import ensure_utf8_output
from tsl_rag.core.models import RetrievalRequest
from tsl_rag.core.settings import get_settings
from tsl_rag.retrieval.retriever import HybridRetriever, RetrievalResult

ensure_utf8_output()

app = typer.Typer(add_completion=False)

# Mierzymy przy pełnym oknie kandydatów: top_k = rerank_top_n = 20. Dzięki temu
# etap "po rerankingu" zawiera ten sam ZBIÓR dokumentów co "po RRF", tylko
# w innej kolejności — czyli różnica metryk opisuje wyłącznie jakość
# porządkowania rerankera, a nie to, że obciął listę.
_EVAL_TOP_K = 20
_K_VALUES = (5, 10, 20)

_STAGES = ("dense", "bm25", "fused", "reranked")


def _doc_ids(results: list[RetrievalResult]) -> list[str]:
    return [r.chunk.metadata.document_id for r in results]


def fact_recall_at_k(
    results: list[RetrievalResult],
    key_facts: list[str],
    k: int,
) -> float:
    """
    Ile oczekiwanych faktów faktycznie stoi w TREŚCI pobranych chunków.

    Po co obok recall@k: recall po dokumentach mówi tylko, że właściwy AKT
    wszedł do kontekstu — nie że wszedł właściwy PRZEPIS. Widać to było przy
    modelu referencyjnym, gdzie 3 z 6 błędnych odmów miało recall = 1.00:
    model dostał właściwe rozporządzenie i mimo to odmówił, bo w pięciu
    chunkach nie było akurat tego artykułu.

    Ta metryka jest też jedynym zabezpieczeniem przed graniem pod recall
    dokumentowy. Ograniczenie liczby chunków na dokument mechanicznie zwiększa
    liczbę RÓŻNYCH dokumentów w top-5, więc podnosi recall@k niemal
    tautologicznie — a jednocześnie może wyrzucić z kontekstu wiersz z kwotą.
    Dopiero fact recall pokazuje, po której stronie wychodzi bilans.

    Dopasowanie przez evals.matching, czyli tak samo jak ocena odpowiedzi:
    fakty liczbowe z granicą cyfry ("200" nie trafia w "2000"), składanie
    diakrytyków włączone, bo korpus bywa zapisany niekonsekwentnie.
    """
    if not key_facts:
        return 1.0
    context = "\n".join(r.chunk.text for r in results[:k])
    return count_matches(key_facts, context, fold=True) / len(key_facts)


async def evaluate_question_retrieval(
    question: GoldenQuestion,
    retriever: HybridRetriever,
) -> dict:
    t0 = time.monotonic()
    request = RetrievalRequest(
        query=question.question,
        top_k=_EVAL_TOP_K,
        rerank_top_n=_EVAL_TOP_K,
    )
    stages = await retriever.retrieve_stages(request)
    latency_ms = int((time.monotonic() - t0) * 1000)

    by_stage = {
        "dense": _doc_ids(stages.dense),
        "bm25": _doc_ids(stages.bm25),
        "fused": _doc_ids(stages.fused),
        "reranked": _doc_ids(stages.final),
    }
    results_by_stage = {
        "dense": stages.dense,
        "bm25": stages.bm25,
        "fused": stages.fused,
        "reranked": stages.final,
    }

    record: dict = {
        "id": question.id,
        "category": question.category,
        "variant": question.variant,
        "expected_docs": question.expected_docs,
        "latency_ms": latency_ms,
        "stages": {},
    }

    for stage, doc_ids in by_stage.items():
        record["stages"][stage] = {
            **{
                f"recall@{k}": round(recall_at_k(doc_ids, question.expected_docs, k), 3)
                for k in _K_VALUES
            },
            **{
                f"fact_recall@{k}": round(
                    fact_recall_at_k(results_by_stage[stage], question.key_facts, k), 3
                )
                for k in _K_VALUES
            },
            "rr": round(reciprocal_rank(doc_ids, question.expected_docs), 3),
            "first_hit": first_hit_position(doc_ids, question.expected_docs),
            "candidates": len(doc_ids),
        }

    return record


def _aggregate(records: list[dict]) -> dict:
    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    summary: dict = {"questions": len(records), "stages": {}, "per_category": {}}

    for stage in _STAGES:
        summary["stages"][stage] = {
            **{
                f"recall@{k}": avg([r["stages"][stage][f"recall@{k}"] for r in records])
                for k in _K_VALUES
            },
            **{
                f"fact_recall@{k}": avg([r["stages"][stage][f"fact_recall@{k}"] for r in records])
                for k in _K_VALUES
            },
            "mrr": avg([r["stages"][stage]["rr"] for r in records]),
        }

    categories: dict[str, list[dict]] = {}
    for r in records:
        categories.setdefault(r["category"], []).append(r)

    for category, items in sorted(categories.items()):
        summary["per_category"][category] = {
            "count": len(items),
            "recall@5_fused": avg([i["stages"]["fused"]["recall@5"] for i in items]),
            "recall@5_reranked": avg([i["stages"]["reranked"]["recall@5"] for i in items]),
            "fact_recall@5_fused": avg([i["stages"]["fused"]["fact_recall@5"] for i in items]),
            "mrr_reranked": avg([i["stages"]["reranked"]["rr"] for i in items]),
        }

    # Pytania, w których żaden etap nie znalazł oczekiwanego dokumentu.
    # To są porażki retrievalu w najczystszej postaci: żadna zmiana promptu
    # ani modelu generującego ich nie naprawi.
    summary["misses"] = [r["id"] for r in records if r["stages"]["fused"]["recall@20"] == 0.0]
    return summary


def _config_snapshot() -> dict:
    s = get_settings()
    return {
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.active_embedding_model,
        "reranker_model": s.reranker_model,
        "bm25_weight": s.bm25_weight,
        "dense_weight": s.dense_weight,
        "eval_top_k": _EVAL_TOP_K,
    }


def _print_report(summary: dict) -> None:
    print(f"\n{'=' * 78}")
    print("EWALUACJA RETRIEVALU (bez generacji)")
    print(f"{'=' * 78}")
    print(f"  Pytań: {summary['questions']}  (bez out_of_scope — brak oczekiwanych dokumentów)")
    print()
    print(
        f"  {'etap':10s} {'recall@5':>9s} {'recall@10':>10s} {'recall@20':>10s} "
        f"{'MRR':>7s} {'fakty@5':>8s} {'fakty@20':>9s}"
    )
    print(f"  {'-' * 70}")
    for stage in _STAGES:
        s = summary["stages"][stage]
        print(
            f"  {stage:10s} {s['recall@5']:9.3f} {s['recall@10']:10.3f} "
            f"{s['recall@20']:10.3f} {s['mrr']:7.3f} "
            f"{s['fact_recall@5']:8.3f} {s['fact_recall@20']:9.3f}"
        )
    print(
        "\n  'fakty@k' = ile oczekiwanych faktów stoi w TREŚCI k pobranych chunków.\n"
        "  recall@k mówi tylko, że wszedł właściwy AKT — nie że wszedł właściwy PRZEPIS."
    )

    print("\n  Per kategoria (recall@5):")
    print(
        f"  {'kategoria':20s} {'n':>3s} {'po RRF':>8s} {'po rerank':>10s} {'MRR':>7s} {'fakty@5':>8s}"
    )
    print(f"  {'-' * 61}")
    for category, v in summary["per_category"].items():
        print(
            f"  {category:20s} {v['count']:3d} {v['recall@5_fused']:8.3f} "
            f"{v['recall@5_reranked']:10.3f} {v['mrr_reranked']:7.3f} "
            f"{v['fact_recall@5_fused']:8.3f}"
        )

    if summary["misses"]:
        print(
            f"\n  Pytania, w których oczekiwany dokument NIE wszedł nawet do top-20 ({len(summary['misses'])}):"
        )
        for qid in summary["misses"]:
            print(f"    - {qid}")

    fused5 = summary["stages"]["fused"]["recall@5"]
    rerank5 = summary["stages"]["reranked"]["recall@5"]
    delta = rerank5 - fused5
    verdict = "poprawia" if delta > 0.001 else ("pogarsza" if delta < -0.001 else "nie zmienia")
    print(f"\n  Reranker {verdict} recall@5: {fused5:.3f} → {rerank5:.3f} ({delta:+.3f})")
    print("=" * 78)


_THRESHOLDS_PATH = Path(__file__).with_name("thresholds.yaml")


def _check_thresholds(summary: dict, thresholds_path: Path) -> tuple[bool, list[str]]:
    """
    Porównuje wynik z progami z `evals/thresholds.yaml`.

    Zwraca (czy_przeszło, linie_raportu). Progi są w pliku, a nie we fladze
    komendy, żeby CI i człowiek patrzyli dokładnie na te same liczby, a ich
    zmiana była widoczna w diffie.
    """
    import yaml

    config = yaml.safe_load(thresholds_path.read_text(encoding="utf-8"))
    limits = config["retrieval"]
    stage = summary["stages"]["reranked"]

    checks = [
        ("recall@5", stage["recall@5"], limits["min_recall_at_5"]),
        ("recall@10", stage["recall@10"], limits["min_recall_at_10"]),
        ("MRR", stage["mrr"], limits["min_mrr"]),
    ]

    lines: list[str] = []
    passed = True
    for name, actual, minimum in checks:
        ok = actual >= minimum
        passed = passed and ok
        mark = "OK  " if ok else "PONIŻEJ"
        lines.append(f"  {mark:8s} {name:10s} {actual:.3f}  (próg {minimum:.3f})")
    return passed, lines


async def run(output_path: Path | None, check_thresholds: bool) -> int:
    # out_of_scope nie ma oczekiwanych dokumentów — recall byłby zawsze 1.0
    # i tylko zawyżałby średnie.
    questions = [q for q in GOLDEN_DATASET if q.expected_docs]
    logger.info(f"Ewaluacja retrievalu na {len(questions)} pytaniach z {len(GOLDEN_DATASET)}")

    records: list[dict] = []
    async with HybridRetriever() as retriever:
        await retriever.warmup()
        for i, question in enumerate(questions, start=1):
            record = await evaluate_question_retrieval(question, retriever)
            records.append(record)
            hit = record["stages"]["reranked"]["first_hit"]
            mark = "OK " if hit and hit <= 5 else "..."
            logger.info(
                f"[{i}/{len(questions)}] {mark} {question.id} "
                f"(pierwsze trafienie po rerankingu: {hit})"
            )

    summary = _aggregate(records)
    _print_report(summary)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {"summary": summary, "results": records, "config": _config_snapshot()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(f"Wyniki zapisane: {output_path}")

    if not check_thresholds:
        return 0

    passed, lines = _check_thresholds(summary, _THRESHOLDS_PATH)
    print(f"\n  Progi z {_THRESHOLDS_PATH.name}:")
    for line in lines:
        print(line)

    if not passed:
        print(
            "\n  BRAMKA NIESPEŁNIONA. Zgodnie z zasadą #1 w CLAUDE.md progu NIE obniża się,\n"
            "  żeby przebieg przeszedł — spadek jest wynikiem do zaraportowania."
        )
        return 1

    print("\n  Bramka spełniona.")
    return 0


@app.command()
def main(
    output: Path = typer.Option(None, "--output", "-o"),  # noqa: B008
    no_gate: bool = typer.Option(
        False,
        "--no-gate",
        help="Pomiń sprawdzenie progów z evals/thresholds.yaml (tylko pomiar)",
    ),
) -> None:
    """Mierzy retrieval na golden datasecie, bez wywoływania modelu generującego."""
    exit_code = asyncio.run(run(output, check_thresholds=not no_gate))
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
