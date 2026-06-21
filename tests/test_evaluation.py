"""Classification metrics helpers (Phase F)."""

from __future__ import annotations

from sparkles.models.evaluation import f1_macro_weighted, per_class_rows


def test_f1_macro_weighted_perfect() -> None:
    y = [0, 0, 1, 1]
    pred = [0, 0, 1, 1]
    macro, weighted = f1_macro_weighted(y, pred)
    assert macro == 1.0
    assert weighted == 1.0


def test_f1_macro_weighted_imbalanced() -> None:
    # Always predict majority class 0
    y = [0, 0, 0, 1]
    pred = [0, 0, 0, 0]
    macro, weighted = f1_macro_weighted(y, pred)
    assert macro < weighted
    assert weighted > 0.0


def test_per_class_rows_skips_summary() -> None:
    report = {
        "stop_loss": {"precision": 0.5, "recall": 1.0, "f1-score": 0.67, "support": 2},
        "macro avg": {"precision": 0.5, "recall": 0.5, "f1-score": 0.5, "support": 4},
    }
    rows = per_class_rows(report)
    assert len(rows) == 1
    assert rows[0]["class"] == "stop_loss"
    assert rows[0]["f1"] == 0.67
