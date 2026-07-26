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

from tsl_rag.core.models import DocumentType
from tsl_rag.core.settings import get_settings
from tsl_rag.ingestion.chunkers.legal_chunker import LegalChunker
from tsl_rag.ingestion.embedders.embedder import ChunkEmbedder
from tsl_rag.ingestion.parsers.legal_pdf_parser import LegalPDFParser

app = typer.Typer(
    name="tsl-ingest",
    help="TSL-RAG ingestion pipeline: PDF → pgvector",
    add_completion=False,
)

DOCUMENT_REGISTRY: dict[str, dict] = {
    # --- STARE PLIKI ---
    "EC_561_2006": {"doc_type": DocumentType.EU_REGULATION, "title": "Regulation (EC) No 561/2006"},
    "EU_2020_1054": {"doc_type": DocumentType.EU_REGULATION, "title": "Regulation (EU) 2020/1054"},
    "DIRECTIVE_2002_15": {"doc_type": DocumentType.DIRECTIVE, "title": "Directive 2002/15/EC"},
    "EU_2016_403": {"doc_type": DocumentType.EU_REGULATION, "title": "Regulation (EU) 2016/403"},
    "AETR": {"doc_type": DocumentType.AETR_AGREEMENT, "title": "AETR Agreement"},
    # --- NOWE PLIKI ---
    "EU_165_2014": {
        "doc_type": DocumentType.EU_REGULATION,
        "title": "Rozporządzenie (UE) 165/2014 (Tachografy)",
    },
    "EU_1071_2009": {
        "doc_type": DocumentType.EU_REGULATION,
        "title": "Rozporządzenie (WE) 1071/2009 (Zawód przewoźnika)",
    },
    "EU_1072_2009": {
        "doc_type": DocumentType.EU_REGULATION,
        "title": "Rozporządzenie (WE) 1072/2009 (Kabotaż i rynek)",
    },
    "DIRECTIVE_2020_1057": {
        "doc_type": DocumentType.DIRECTIVE,
        "title": "Dyrektywa (UE) 2020/1057 (Delegowanie kierowców)",
    },
    "PL_DRIVER_HOURS_ACT": {
        "doc_type": DocumentType.NATIONAL_LAW,
        "title": "Ustawa o czasie pracy kierowców (PL)",
    },
    "TARIFF_DRIVER_2022": {
        "doc_type": DocumentType.PENALTY_TARIFF,
        "title": "Taryfikator dla kierowcy (2022)",
    },
    "TARIFF_COMPANY_2022": {
        "doc_type": DocumentType.PENALTY_TARIFF,
        "title": "Taryfikator dla przedsiębiorcy (2022)",
    },
    "TARIFF_MANAGER_2022": {
        "doc_type": DocumentType.PENALTY_TARIFF,
        "title": "Taryfikator dla zarządzającego (2022)",
    },
    "TARIFF_EMPLOYER_2022": {
        "doc_type": DocumentType.PENALTY_TARIFF,
        "title": "Taryfikator dla pracodawcy (2022)",
    },
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
