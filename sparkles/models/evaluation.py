"""Classification metrics helpers (ML expansion Phase F)."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import classification_report, f1_score


def classification_report_dict(
    y_true: Any,
    y_pred: Any,
    *,
    labels: list[int],
    target_names: list[str],
) -> dict[str, Any]:
    """sklearn classification_report as a dict (per-class + macro/weighted avg)."""
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )


def f1_macro_weighted(y_true: Any, y_pred: Any) -> tuple[float, float]:
    """Return (macro_f1, weighted_f1) with zero_division=0."""
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return macro, weighted


_SUMMARY_ROWS = frozenset({"accuracy", "macro avg", "weighted avg"})


def per_class_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Outcome rows from a classification_report dict (excludes summary rows)."""
    rows: list[dict[str, Any]] = []
    for name, stats in report.items():
        if name in _SUMMARY_ROWS:
            continue
        if isinstance(stats, dict) and "f1-score" in stats:
            rows.append(
                {
                    "class": str(name),
                    "precision": float(stats["precision"]),
                    "recall": float(stats["recall"]),
                    "f1": float(stats["f1-score"]),
                    "support": int(stats["support"]),
                },
            )
    return rows
