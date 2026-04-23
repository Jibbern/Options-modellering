"""Dividend-assumption metadata helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import parse_date
from .store import empty_dataset_payload, load_dataset_history, register_dataset_file, row_to_payload

COLUMN_ALIASES = {
    "ticker": "ticker",
    "snapshot_date": "snapshot_date",
    "date": "snapshot_date",
    "dividend_yield": "dividend_yield",
    "yield": "dividend_yield",
    "expected_dividend_date": "expected_dividend_date",
    "dividend_date": "expected_dividend_date",
    "source": "source",
    "source_url": "source_url",
    "acquisition_method": "acquisition_method",
    "notes": "notes",
    "registered_at": "registered_at",
}


def register_dividends_file(
    ticker: str,
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register a manual dividends CSV or JSON file."""

    return register_dataset_file(
        "dividends",
        ticker,
        path,
        column_aliases=COLUMN_ALIASES,
        default_values={"acquisition_method": "manual"},
        data_root=data_root,
    )


def get_dividend_assumption(
    ticker: str,
    snapshot_date: date | str | None,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the latest dividend assumption on or before the snapshot date."""

    snapshot = parse_date(snapshot_date)
    if snapshot is None:
        return empty_dataset_payload("dividends")
    history = load_dataset_history("dividends", ticker, data_root)
    if history.empty:
        return empty_dataset_payload("dividends")
    frame = history.loc[history["snapshot_date"] <= pd.Timestamp(snapshot)].copy()
    if frame.empty:
        return empty_dataset_payload("dividends")
    row = frame.sort_values(["snapshot_date", "registered_at"]).iloc[-1]
    payload = row_to_payload("dividends", row)
    payload["matched"] = True
    payload["matched_snapshot_date"] = payload.get("snapshot_date")
    return payload
