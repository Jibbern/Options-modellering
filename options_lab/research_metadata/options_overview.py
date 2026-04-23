"""Options overview and vol-regime metadata helpers."""

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
    "iv": "implied_volatility",
    "implied_volatility": "implied_volatility",
    "historic_volatility": "historic_volatility",
    "historical_volatility": "historic_volatility",
    "hv": "historic_volatility",
    "iv_rank": "iv_rank",
    "iv_percentile": "iv_percentile",
    "iv_hv_ratio": "iv_hv_ratio",
    "put_call_volume_ratio": "put_call_volume_ratio",
    "put_call_open_interest_ratio": "put_call_open_interest_ratio",
    "total_call_volume": "total_call_volume",
    "total_put_volume": "total_put_volume",
    "total_call_open_interest": "total_call_open_interest",
    "total_put_open_interest": "total_put_open_interest",
    "earnings_date": "earnings_date",
    "source": "source",
    "source_url": "source_url",
    "acquisition_method": "acquisition_method",
    "notes": "notes",
    "registered_at": "registered_at",
}


def register_options_overview_file(
    ticker: str,
    path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register a manual options-overview CSV or JSON file."""

    return register_dataset_file(
        "options_overview",
        ticker,
        path,
        column_aliases=COLUMN_ALIASES,
        default_values={"acquisition_method": "manual"},
        data_root=data_root,
    )


def get_options_overview(
    ticker: str,
    snapshot_date: date | str | None,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the latest available options-overview row on or before snapshot date."""

    snapshot = parse_date(snapshot_date)
    if snapshot is None:
        return empty_dataset_payload("options_overview")
    history = load_dataset_history("options_overview", ticker, data_root)
    if history.empty:
        return empty_dataset_payload("options_overview")
    frame = history.loc[history["snapshot_date"] <= pd.Timestamp(snapshot)].copy()
    if frame.empty:
        return empty_dataset_payload("options_overview")
    row = frame.sort_values(["snapshot_date", "registered_at"]).iloc[-1]
    payload = row_to_payload("options_overview", row)
    payload["matched"] = True
    payload["matched_snapshot_date"] = payload.get("snapshot_date")
    return payload
