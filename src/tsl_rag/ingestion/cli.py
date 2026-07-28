"""
CLI do ingestion: PDF → parse → chunk → embed → pgvector.

Użycie:
    uv run python -m tsl_rag.ingestion.cli ingest data/raw/EC_561_2006.pdf \
        --doc-id ec_561_2006 \
        --doc-type eu_regulation \
        --title "Regulation (EC) No 561/2006"

    # Wszystkie PDFy naraz:
    uv run python -m tsl_rag.ingestion.cli ingest-all data/raw/
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from loguru import logger

from tsl_rag.core.console import ensure_utf8_output
from tsl_rag.core.models import DocumentType
from tsl_rag.core.settings import get_settings
from tsl_rag.ingestion.chunkers.legal_chunker import LegalChunker
from tsl_rag.ingestion.embedders.embedder import ChunkEmbedder
from tsl_rag.ingestion.parsers.legal_pdf_parser import LegalPDFParser

ensure_utf8_output()

app = typer.Typer(
    name="tsl-ingest",
    help="TSL-RAG ingestion pipeline: PDF → pgvector",
    add_completion=False,
)

from tsl_rag.core.documents import DOCUMENT_REGISTRY  # noqa: E402  (re-eksport)

# TARIFF_EMPLOYER_2022 celowo NIE MA wpisu w rejestrze.
#
# Plan przewidywał dodanie go jako czternastego dokumentu, w założeniu, że
# jest cicho pomijany przez przeoczenie. Weryfikacja pokazała coś innego:
# TARIFF_EMPLOYER_2022.pdf jest bajtowo identyczny z TARIFF_COMPANY_2022.pdf
# (md5 bc6f6cb87eaa7f19e27a163522f0fdde dla obu). To ten sam plik pobrany
# dwa razy pod inną nazwą, nie osobny akt prawny.
#
# Dodanie go do rejestru wstrzykuje 15 chunków o tekście identycznym
# z tariff_company_2022. Dwa identyczne chunki konkurują wtedy w retrievalu,
# zajmują dwa miejsca w kontekście zamiast jednego i wypychają z niego inne
# dokumenty — co uderza wprost w kategorię "penalty" z golden dataset, gdzie
# model już teraz cytuje tariff_company_2022 zamiast tariff_driver_2022.
#
# Potwierdzone też po stronie źródła: oba pliki to **Załącznik nr 3** do ustawy
# o transporcie drogowym (kary pieniężne dla przewoźnika, czyli pracodawcy).
# "COMPANY" i "EMPLOYER" to dwie nazwy nadane temu samemu załącznikowi, a nie
# dwa różne taryfikatory. Nie ma więc czego dodatkowo szukać ani pobierać.
#
# Dla porządku, odwzorowanie taryfikatorów na załączniki ustawy:
#   Załącznik nr 1 → tariff_driver_2022   (grzywny dla kierowcy)
#   Załącznik nr 3 → tariff_company_2022  (kary dla przewoźnika/pracodawcy)
#   Załącznik nr 4 → tariff_manager_2022  (kary dla zarządzającego transportem)
_KNOWN_DUPLICATES = {
    "TARIFF_EMPLOYER_2022": (
        "identyczny bajtowo z TARIFF_COMPANY_2022.pdf — oba to Załącznik nr 3"
    ),
}


@app.command()
def ingest(
    pdf_path: Path = typer.Argument(..., help="Ścieżka do pliku PDF"),  # noqa: B008
    doc_id: str = typer.Option(..., "--doc-id", help="Unikalny ID dokumentu, np. ec_561_2006"),
    doc_type: str = typer.Option(
        ..., "--doc-type", help="Typ: eu_regulation | directive | aetr_agreement | penalty_tariff"
    ),
    title: str = typer.Option(..., "--title", help="Pełna nazwa dokumentu"),
    jurisdiction: str = typer.Option("EU", "--jurisdiction"),
    batch_size: int = typer.Option(16, "--batch-size", help="Chunki per batch do Ollamy"),
) -> None:
    """Przetwarza jeden plik PDF i zapisuje chunks do pgvector."""
    if not pdf_path.exists():
        typer.echo(f"ERROR: Plik nie istnieje: {pdf_path}", err=True)
        raise typer.Exit(1)

    try:
        document_type = DocumentType(doc_type)
    except ValueError:
        valid = [e.value for e in DocumentType]
        typer.echo(f"ERROR: Nieprawidłowy doc-type '{doc_type}'. Dopuszczalne: {valid}", err=True)
        raise typer.Exit(1) from None

    asyncio.run(
        _ingest_one(
            pdf_path=pdf_path,
            doc_id=doc_id,
            document_type=document_type,
            title=title,
            jurisdiction=jurisdiction,
            batch_size=batch_size,
        )
    )


@app.command("ingest-all")
def ingest_all(
    data_dir: Path = typer.Argument(..., help="Katalog z plikami PDF (data/raw/)"),  # noqa: B008
    batch_size: int = typer.Option(16, "--batch-size"),
) -> None:
    """Przetwarza wszystkie PDFy z katalogu wg DOCUMENT_REGISTRY."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        typer.echo(f"ERROR: Brak plików PDF w {data_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Znaleziono {len(pdfs)} plików PDF\n")

    # Jeden asyncio.run() na CAŁĄ pętlę, nie per plik. Poprzednia wersja
    # otwierała i zamykała event loop 14 razy, co na Windows (ProactorEventLoop)
    # kończyło się serią "RuntimeError: Event loop is closed" z httpx.AsyncClient
    # przy zamykaniu. Efekt był kosmetyczny, ale zaśmiecał wyjście na tyle,
    # że trudno było odróżnić prawdziwy błąd ingestu od hałasu.
    summary = asyncio.run(_ingest_all_async(pdfs, batch_size))

    typer.echo(
        f"\nGotowe. Plików: {summary['files_ok']}/{len(pdfs)} "
        f"(pominięte: {summary['files_skipped']}), "
        f"chunki: stored={summary['stored']}, failed={summary['failed']}"
    )
    if summary["failed"]:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Pipeline (async core)
# ---------------------------------------------------------------------------


def _parse_and_chunk(
    pdf_path: Path,
    doc_id: str,
    document_type: DocumentType,
    title: str,
    jurisdiction: str = "EU",
) -> list:
    """Etapy synchroniczne: PDF → elementy → chunki. Bez sieci i bez bazy."""
    settings = get_settings()

    parser = LegalPDFParser(doc_type=document_type)
    elements = parser.parse(pdf_path)
    if not elements:
        logger.warning(f"[{doc_id}] Parser zwrócił 0 elementów — pomijam")
        return []

    chunker = LegalChunker(
        document_id=doc_id,
        document_type=document_type,
        document_title=title,
        jurisdiction=jurisdiction,
        # Parametry z Settings — wcześniej chunker zawsze jechał na swoich
        # stałych modułowych, a CHUNK_SIZE/CHUNK_OVERLAP w .env nic nie robiły.
        max_tokens=settings.chunker_max_tokens,
        min_tokens=settings.chunker_min_tokens,
        overlap_tokens=settings.chunker_overlap_tokens,
    )
    chunks = chunker.chunk(elements)
    if not chunks:
        logger.warning(f"[{doc_id}] Chunker zwrócił 0 chunków — pomijam")
    return chunks


async def _ingest_one(
    pdf_path: Path,
    doc_id: str,
    document_type: DocumentType,
    title: str,
    jurisdiction: str = "EU",
    batch_size: int = 16,
) -> dict:
    chunks = _parse_and_chunk(pdf_path, doc_id, document_type, title, jurisdiction)
    if not chunks:
        return {"total": 0, "stored": 0, "failed": 0}

    async with ChunkEmbedder(batch_size=batch_size) as embedder:
        stats = await embedder.embed_and_store(chunks)

    _echo_file_stats(doc_id, stats)
    return stats


async def _ingest_all_async(pdfs: list[Path], batch_size: int) -> dict:
    """
    Ingest wielu plików w JEDNEJ pętli zdarzeń i na JEDNYM połączeniu
    do bazy oraz jednym kliencie embeddingów — zamiast otwierania ich
    od nowa dla każdego pliku.
    """
    summary = {"files_ok": 0, "files_skipped": 0, "total": 0, "stored": 0, "failed": 0}

    async with ChunkEmbedder(batch_size=batch_size) as embedder:
        for pdf in pdfs:
            stem = pdf.stem.upper()
            if stem not in DOCUMENT_REGISTRY:
                # Rozróżniamy "pominięty świadomie" od "pominięty, bo ktoś
                # zapomniał wpisu" — bez tego pierwszy przypadek wygląda
                # w logu jak przeoczenie i ktoś go kiedyś "naprawi".
                reason = _KNOWN_DUPLICATES.get(stem)
                if reason:
                    typer.echo(f"  SKIP  {pdf.name} — pomijany świadomie: {reason}")
                else:
                    typer.echo(f"  SKIP  {pdf.name} — brak wpisu w DOCUMENT_REGISTRY")
                summary["files_skipped"] += 1
                continue

            meta = DOCUMENT_REGISTRY[stem]
            doc_id = stem.lower()
            typer.echo(f"  →  {pdf.name}")

            chunks = _parse_and_chunk(
                pdf_path=pdf,
                doc_id=doc_id,
                document_type=meta["doc_type"],
                title=meta["title"],
            )
            if not chunks:
                summary["files_skipped"] += 1
                continue

            stats = await embedder.embed_and_store(chunks)
            _echo_file_stats(doc_id, stats)

            summary["files_ok"] += 1
            for k in ("total", "stored", "failed"):
                summary[k] += stats.get(k, 0)

    return summary


def _echo_file_stats(doc_id: str, stats: dict) -> None:
    typer.echo(
        f"    ✓ {doc_id}: "
        f"{stats['stored']}/{stats['total']} chunks zapisanych"
        + (f", {stats['failed']} błędów" if stats["failed"] else "")
    )


if __name__ == "__main__":
    app()
