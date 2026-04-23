"""Local option-chain snapshot discovery helpers.

This module is the canonical home for local snapshot lookup across analysis and
ingestion flows. It intentionally sits outside any removed non-canonical workflow.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from .io import load_chain
from .research_metadata.catalog import discover_chain_snapshots
from .utils import clean_string, parse_date


def _normalize_date_list(values: Iterable[date | str] | date | str | None) -> list[date]:
    if values is None:
        return []
    if isinstance(values, (date, str)):
        values = [values]
    normalized: list[date] = []
    for value in values:
        parsed = parse_date(value)
        if parsed is None:
            raise ValueError(f"Expected a valid date, got: {value!r}")
        normalized.append(parsed)
    return normalized


def list_snapshot_slices(
    ticker: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return one row per local chain slice discovered for a ticker."""

    records = discover_chain_snapshots(ticker, data_root)
    if not records:
        return pd.DataFrame(
            columns=[
                "file_path",
                "storage_location",
                "snapshot_scope",
                "snapshot_date",
                "expiry_date",
                "days_to_expiry",
                "has_sidecar_metadata",
                "contract_count",
                "usable_quote_count",
                "usable_quote_coverage_pct",
                "quote_usable",
                "quote_usability_gate_pct",
                "snapshot_expiry_count",
                "snapshot_dates_for_expiry",
                "comparison_ready",
            ]
        )

    frame = pd.DataFrame(records)
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce").dt.normalize()
    frame["expiry_date"] = pd.to_datetime(frame["expiry_date"], errors="coerce").dt.normalize()
    frame["days_to_expiry"] = (frame["expiry_date"] - frame["snapshot_date"]).dt.days
    frame["snapshot_expiry_count"] = frame.groupby("snapshot_date")["expiry_date"].transform("nunique")
    frame["snapshot_dates_for_expiry"] = frame.groupby("expiry_date")["snapshot_date"].transform("nunique")
    frame["comparison_ready"] = frame["snapshot_dates_for_expiry"].fillna(0).astype(int) >= 2
    return frame.sort_values(["snapshot_date", "expiry_date", "file_path"]).reset_index(drop=True)


def available_snapshot_dates(
    ticker: str,
    data_root: str | Path | None = None,
) -> list[str]:
    """Return all exact local snapshot dates available for a ticker."""

    slices = list_snapshot_slices(ticker, data_root)
    if slices.empty:
        return []
    return [
        pd.Timestamp(value).date().isoformat()
        for value in slices["snapshot_date"].dropna().sort_values().unique().tolist()
    ]


def available_expiries_for_snapshot(
    ticker: str,
    snapshot_date: date | str,
    data_root: str | Path | None = None,
) -> list[str]:
    """Return expiries available for one exact snapshot date."""

    slices = snapshot_slices_for_date(ticker, snapshot_date, data_root)
    return [
        pd.Timestamp(value).date().isoformat()
        for value in slices["expiry_date"].dropna().sort_values().unique().tolist()
    ]


def comparison_ready_expiries(
    ticker: str,
    data_root: str | Path | None = None,
) -> list[str]:
    """Return expiries that exist across at least two local snapshot dates."""

    slices = list_snapshot_slices(ticker, data_root)
    if slices.empty:
        return []
    ready = slices.loc[slices["comparison_ready"], "expiry_date"].dropna().sort_values().unique().tolist()
    return [pd.Timestamp(value).date().isoformat() for value in ready]


def snapshot_slices_for_date(
    ticker: str,
    snapshot_date: date | str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Return all chain slices for one exact local snapshot date."""

    requested = parse_date(snapshot_date)
    if requested is None:
        raise ValueError(f"snapshot_date must be a valid date, got: {snapshot_date!r}")
    slices = list_snapshot_slices(ticker, data_root)
    if slices.empty:
        raise ValueError(f"No local chain snapshots were found for {clean_string(ticker).upper()}.")
    matched = slices.loc[slices["snapshot_date"] == pd.Timestamp(requested)].copy()
    if matched.empty:
        available = ", ".join(available_snapshot_dates(ticker, data_root)) or "none"
        raise ValueError(
            f"No local chain snapshot exists for {clean_string(ticker).upper()} on "
            f"{requested.isoformat()}. Available snapshot dates: {available}."
        )
    return matched.sort_values(["expiry_date", "file_path"]).reset_index(drop=True)


def snapshot_slice_for_expiry(
    ticker: str,
    snapshot_date: date | str,
    expiry_date: date | str,
    data_root: str | Path | None = None,
) -> pd.Series:
    """Return one exact local `(snapshot_date, expiry_date)` chain slice."""

    requested_expiry = parse_date(expiry_date)
    if requested_expiry is None:
        raise ValueError(f"expiry_date must be a valid date, got: {expiry_date!r}")
    slices = snapshot_slices_for_date(ticker, snapshot_date, data_root)
    matched = slices.loc[slices["expiry_date"] == pd.Timestamp(requested_expiry)]
    if matched.empty:
        available = ", ".join(available_expiries_for_snapshot(ticker, snapshot_date, data_root)) or "none"
        raise ValueError(
            f"No local chain slice exists for {clean_string(ticker).upper()} on "
            f"{parse_date(snapshot_date).isoformat()} with expiry {requested_expiry.isoformat()}. "
            f"Available expiries for that snapshot: {available}."
        )
    return matched.iloc[0]


def build_chain_panel(
    ticker: str,
    snapshot_dates: Iterable[date | str] | date | str | None = None,
    expiries: Iterable[date | str] | date | str | None = None,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Build a normalized contract-level panel across local chain slices."""

    slices = list_snapshot_slices(ticker, data_root)
    if slices.empty:
        return pd.DataFrame(
            columns=[
                "file_path",
                "storage_location",
                "snapshot_date",
                "expiry_date",
                "days_to_expiry",
                "option_type",
                "strike",
                "moneyness",
                "iv",
                "delta",
                "mid",
                "bid",
                "ask",
                "last",
                "volume",
                "open_interest",
                "spot_price",
                "has_quote",
                "quote_count",
            ]
        )

    requested_snapshots = _normalize_date_list(snapshot_dates)
    requested_expiries = _normalize_date_list(expiries)
    if requested_snapshots:
        slices = slices.loc[slices["snapshot_date"].isin([pd.Timestamp(value) for value in requested_snapshots])]
    if requested_expiries:
        slices = slices.loc[slices["expiry_date"].isin([pd.Timestamp(value) for value in requested_expiries])]
    if slices.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for row in slices.itertuples():
        chain = load_chain(
            row.file_path,
            prices_data_root=data_root,
            rates_data_root=data_root,
            research_data_root=data_root,
        )
        frame = chain.contracts.copy()
        frame["file_path"] = str(row.file_path)
        frame["storage_location"] = row.storage_location
        frame["days_to_expiry"] = row.days_to_expiry
        frame["spot_price"] = chain.metadata.spot_price
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if panel.empty:
        return panel
    panel["snapshot_date"] = pd.to_datetime(panel["snapshot_date"], errors="coerce").dt.normalize()
    panel["expiry_date"] = pd.to_datetime(panel["expiry_date"], errors="coerce").dt.normalize()
    panel["last_trade"] = pd.to_datetime(panel["last_trade"], errors="coerce").dt.normalize()
    ordered_columns = [
        "file_path",
        "storage_location",
        "ticker",
        "snapshot_date",
        "expiry_date",
        "days_to_expiry",
        "option_type",
        "strike",
        "moneyness",
        "iv",
        "delta",
        "mid",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "spot_price",
        "has_quote",
        "quote_count",
        "last_trade",
        "raw_row",
    ]
    available_columns = [column for column in ordered_columns if column in panel.columns]
    remainder = [column for column in panel.columns if column not in available_columns]
    return panel[available_columns + remainder].sort_values(
        ["snapshot_date", "expiry_date", "option_type", "strike"]
    ).reset_index(drop=True)
