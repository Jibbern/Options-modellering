"""Expected-move metadata registration and lookup helpers."""

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
    "expiry_date": "expiry_date",
    "expiry": "expiry_date",
    "expected_move": "expected_move_abs",
    "expected_move_abs": "expected_move_abs",
    "expected_move_dollars": "expected_move_abs",
    "expected_move_pct": "expected_move_pct",
    "expected_move_percent": "expected_move_pct",
    "lower": "lower_bound",
    "lower_bound": "lower_bound",
    "upper": "upper_bound",
    "upper_bound": "upper_bound",
    "iv": "implied_volatility",
    "implied_volatility": "implied_volatility",
    "source": "source",
    "source_url": "source_url",
    "acquisition_method": "acquisition_method",
    "notes": "notes",
    "registered_at": "registered_at",
}


def register_expected_move_file(
    ticker: str,
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register a manual expected-move CSV or JSON file."""

    return register_dataset_file(
        "expected_move",
        ticker,
        path,
        column_aliases=COLUMN_ALIASES,
        default_values={"acquisition_method": "manual"},
        data_root=data_root,
    )


def get_expected_move(
    ticker: str,
    snapshot_date: date | str | None,
    *,
    expiry_date: date | str | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the best available expected-move row on or before a snapshot date."""

    snapshot = parse_date(snapshot_date)
    expiry = parse_date(expiry_date) if expiry_date is not None else None
    if snapshot is None:
        return empty_dataset_payload("expected_move")
    history = load_dataset_history("expected_move", ticker, data_root)
    if history.empty:
        return empty_dataset_payload("expected_move")
    frame = history.loc[history["snapshot_date"] <= pd.Timestamp(snapshot)].copy()
    if expiry is not None:
        frame = frame.loc[frame["expiry_date"] == pd.Timestamp(expiry)]
    if frame.empty:
        return empty_dataset_payload("expected_move")
    row = frame.sort_values(["snapshot_date", "registered_at"]).iloc[-1]
    payload = row_to_payload("expected_move", row)
    payload["matched"] = True
    payload["matched_snapshot_date"] = payload.get("snapshot_date")
    return payload
