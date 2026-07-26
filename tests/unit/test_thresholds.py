"""
Testy bramki progowej.

Sens: plik z progami jest jedynym miejscem, w którym zapisane jest, co uznajemy
za regresję. Literówka w kluczu albo próg wpisany jako tekst zamiast liczby
sprawiłyby, że bramka cicho przepuszcza wszystko — a wyglądałaby na działającą.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from evals.run_retrieval_evals import _THRESHOLDS_PATH, _check_thresholds

pytestmark = pytest.mark.unit


def _summary(recall5: float, recall10: float, mrr: float) -> dict:
    return {"stages": {"reranked": {"recall@5": recall5, "recall@10": recall10, "mrr": mrr}}}


def test_thresholds_file_exists_and_parses():
    config = yaml.safe_load(_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    limits = config["retrieval"]
    for key in ("min_recall_at_5", "min_recall_at_10", "min_mrr"):
        assert isinstance(limits[key], float), f"{key} musi być liczbą, nie {type(limits[key])}"
        assert 0.0 <= limits[key] <= 1.0


def test_gate_passes_on_current_measurement():
    """Wartości z retrieval_009_norerank_bm25_0.5.json muszą przechodzić."""
    passed, _ = _check_thresholds(_summary(0.938, 0.969, 0.874), _THRESHOLDS_PATH)
    assert passed


def test_gate_fails_when_recall_drops():
    passed, lines = _check_thresholds(_summary(0.80, 0.969, 0.874), _THRESHOLDS_PATH)
    assert not passed
    assert any("PONIŻEJ" in line and "recall@5" in line for line in lines)


def test_gate_fails_when_mrr_drops():
    passed, _ = _check_thresholds(_summary(0.938, 0.969, 0.50), _THRESHOLDS_PATH)
    assert not passed


def test_thresholds_stay_below_measured_baseline(tmp_path: Path):
    """
    Progi mają leżeć PONIŻEJ pomiaru odniesienia, ale nie dowolnie nisko —
    próg na poziomie 0.5 przepuszczałby regresję o połowę i byłby atrapą.
    """
    config = yaml.safe_load(_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    limits = config["retrieval"]
    measured = {"min_recall_at_5": 0.938, "min_recall_at_10": 0.969, "min_mrr": 0.874}
    for key, value in measured.items():
        assert limits[key] <= value, f"{key}: próg powyżej pomiaru — bramka nie przejdzie nigdy"
        assert value - limits[key] <= 0.05, (
            f"{key}: próg o {value - limits[key]:.3f} poniżej pomiaru — za luźny, "
            f"przepuści realną regresję"
        )
