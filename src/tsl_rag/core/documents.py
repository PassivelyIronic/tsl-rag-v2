"""
Rejestr dokumentów korpusu: identyfikator → typ i tytuł.

Mieszka w `core`, a nie w `ingestion/cli.py`, bo czyta go także API
(`/query/documents`, żeby UI nie miało listy zaszytej na sztywno). Import
z `ingestion.cli` ciągnął przy tym cały moduł ingestu razem z parserami PDF,
czyli `unstructured`, `pdfplumber` i `pymupdf` — zależnościami, których obraz
produkcyjny API nie potrzebuje i które wnosiły do niego torchvision, opencv
oraz spacy.
"""

from __future__ import annotations

from tsl_rag.core.models import DocumentType

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
}
