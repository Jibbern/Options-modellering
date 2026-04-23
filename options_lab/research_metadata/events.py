"""Event and earnings metadata helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import parse_date
from .store import empty_dataset_payload, load_dataset_history, register_dataset_file, row_to_payload

COLUMN_ALIASES = {
    "ticker": "ticker",
    "event_date": "event_date",
    "date": "event_date",
    "earnings_date": "event_date",
    "event_time": "event_time",
    "time": "event_time",
    "earnings_time": "event_time",
    "event_type": "event_type",
    "type": "event_type",
    "source": "source",
    "source_url": "source_url",
    "acquisition_method": "acquisition_method",
    "notes": "notes",
    "registered_at": "registered_at",
}


def register_events_file(
    ticker: str,
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register a manual events or earnings CSV/JSON file."""

    return register_dataset_file(
        "events",
        ticker,
        path,
        column_aliases=COLUMN_ALIASES,
        default_values={"acquisition_method": "manual", "event_type": "event"},
        data_root=data_root,
    )


def list_events(
    ticker: str,
    *,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List all registered events for a ticker."""

    history = load_dataset_history("events", ticker, data_root)
    if history.empty:
        return []
    return [row_to_payload("events", row) for _, row in history.sort_values(["event_date", "registered_at"]).iterrows()]


def get_nearest_event(
    ticker: str,
    snapshot_date: date | str | None,
    *,
    expiry_date: date | str | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return the next event on or after the snapshot date, if any."""

    snapshot = parse_date(snapshot_date)
    expiry = parse_date(expiry_date) if expiry_date is not None else None
    if snapshot is None:
        return empty_dataset_payload("events")
    history = load_dataset_history("events", ticker, data_root)
    if history.empty:
        return empty_dataset_payload("events")
    frame = history.loc[history["event_date"] >= pd.Timestamp(snapshot)].copy()
    if frame.empty:
        return empty_dataset_payload("events")
    row = frame.sort_values(["event_date", "registered_at"]).iloc[0]
    payload = row_to_payload("events", row)
    payload["matched"] = True
    payload["occurs_before_expiry"] = None if expiry is None else parse_date(payload["event_date"]) <= expiry
    return payload
