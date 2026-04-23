"""Analyst-note metadata helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import parse_date
from .store import load_dataset_history, register_dataset_file, row_to_payload

COLUMN_ALIASES = {
    "ticker": "ticker",
    "note_date": "note_date",
    "date": "note_date",
    "category": "category",
    "title": "title",
    "body": "body",
    "text": "body",
    "note": "body",
    "source": "source",
    "source_url": "source_url",
    "acquisition_method": "acquisition_method",
    "notes": "notes",
    "registered_at": "registered_at",
}


def register_notes_file(
    ticker: str,
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register a manual notes CSV or JSON file."""

    return register_dataset_file(
        "notes",
        ticker,
        path,
        column_aliases=COLUMN_ALIASES,
        default_values={"acquisition_method": "manual", "category": "general"},
        data_root=data_root,
    )


def get_recent_notes(
    ticker: str,
    snapshot_date: date | str | None,
    *,
    limit: int = 3,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent notes on or before the snapshot date."""

    snapshot = parse_date(snapshot_date)
    if snapshot is None:
        return []
    history = load_dataset_history("notes", ticker, data_root)
    if history.empty:
        return []
    frame = history.loc[history["note_date"] <= pd.Timestamp(snapshot)].copy()
    if frame.empty:
        return []
    frame = frame.sort_values(["note_date", "registered_at"], ascending=[False, False]).head(limit)
    return [row_to_payload("notes", row) for _, row in frame.iterrows()]
