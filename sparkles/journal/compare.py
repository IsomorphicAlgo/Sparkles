"""Join personal trade CSV to exported ``predictions.parquet`` by session date."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd

from sparkles.config.schema import ExperimentConfig


def resolve_journal_csv_path(
    cfg: ExperimentConfig,
    base_dir: Path | None,
) -> Path | None:
    """Return absolute path to journal CSV, or None if not configured."""
    raw = cfg.journal.csv_path
    if not raw or not str(raw).strip():
        return None
    p = Path(raw)
    if not p.is_absolute():
        root = Path.cwd() if base_dir is None else base_dir
        p = root / p
    return p


def load_and_normalize_journal(path: Path, symbol_filter: str) -> pd.DataFrame:
    """Load CSV; normalize ``entry_date`` and optional ``symbol`` filter."""
    df = pd.read_csv(path)
    if df.empty:
        return df

    colmap = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}

    def pick(*names: str) -> str | None:
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    entry_col = pick("entry_date", "date", "open_date", "entry")
    if entry_col is None:
        raise ValueError(
            "Journal CSV needs entry_date, date, open_date, or entry column",
        )

    out = df.rename(columns={entry_col: "entry_date"})
    out["entry_date"] = pd.to_datetime(out["entry_date"], errors="coerce").dt.date
    if out["entry_date"].isna().any():
        raise ValueError("Journal CSV has invalid entry_date values")

    sym_col = pick("symbol", "ticker")
    if sym_col is not None:
        symu = symbol_filter.upper()
        s = out[sym_col].astype(str).str.upper().str.strip()
        mask = s.eq(symu) | s.eq("") | s.eq("NAN")
        out = out.loc[mask].copy()

    return cast(pd.DataFrame, out)


def aggregate_predictions_by_session(
    pred: pd.DataFrame,
    *,
    split_filter: str | None,
) -> pd.DataFrame:
    """One row per session_date for joining to journal entry_date."""
    d = pred.copy()
    if split_filter is not None and "split" in d.columns:
        d = d[d["split"] == split_filter]
    if d.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "pred_n",
                "y_true_first",
                "y_pred_mode",
                "max_proba_mean",
            ],
        )

    rows: list[dict[str, Any]] = []
    for sd, sub in d.groupby("session_date", sort=True):
        mode = sub["y_pred"].mode(dropna=True)
        ypm = mode.iloc[0] if len(mode) else sub["y_pred"].iloc[0]
        row: dict[str, Any] = {
            "session_date": sd,
            "pred_n": int(len(sub)),
            "y_true_first": sub["y_true"].iloc[0],
            "y_pred_mode": ypm,
        }
        if "max_proba" in sub.columns:
            row["max_proba_mean"] = float(sub["max_proba"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_journal_compare(
    cfg: ExperimentConfig,
    run_dir: Path,
    *,
    split_filter: str | None = "val",
    base_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Merge journal rows with aggregated predictions; write ``journal_compare.csv``."""
    jpath = resolve_journal_csv_path(cfg, base_dir)
    if jpath is None or not jpath.is_file():
        raise FileNotFoundError(
            "Set journal.csv_path in YAML to an existing CSV file.",
        )

    pred_path = run_dir / "predictions.parquet"
    if not pred_path.is_file():
        raise FileNotFoundError(
            f"Missing {pred_path.name}; run training with "
            "train.export_predictions: val or all.",
        )

    pred = pd.read_parquet(pred_path)
    if "session_date" not in pred.columns:
        raise ValueError("predictions.parquet missing session_date column")

    journal = load_and_normalize_journal(jpath, cfg.symbol)
    if journal.empty:
        merged = pd.DataFrame()
    else:
        by_day = aggregate_predictions_by_session(pred, split_filter=split_filter)
        merged = journal.merge(
            by_day,
            how="left",
            left_on="entry_date",
            right_on="session_date",
        )
        merged["model_matched"] = merged["pred_n"].notna() & (merged["pred_n"] > 0)

    out_csv = run_dir / "journal_compare.csv"
    merged.to_csv(out_csv, index=False)
    return merged, out_csv
