"""Helpers for selecting a locally stored risk-free rate for an option horizon."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..utils import parse_date
from .rate_store import (
    FRED_SOURCE,
    SERIES_DECIMAL_COLUMN,
    SERIES_PERCENT_COLUMN,
    get_latest_rates_snapshot as load_latest_rates_snapshot,
    load_merged_rates,
)


@dataclass(frozen=True)
class RiskFreeRateMatch:
    """Resolved local Treasury-rate match for a snapshot/expiry pair."""

    snapshot_date: date
    expiry_date: date
    days_to_expiry: int
    series_used: str
    matched_date: date
    rate_percent: float
    rate_decimal: float
    source: str = FRED_SOURCE
    used_prior_date: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["snapshot_date"] = self.snapshot_date.isoformat()
        payload["expiry_date"] = self.expiry_date.isoformat()
        payload["matched_date"] = self.matched_date.isoformat()
        return payload


def select_series_for_days(days_to_expiry: int) -> str:
    """Map days to expiry to the configured Treasury constant-maturity series."""

    if days_to_expiry < 0:
        raise ValueError("expiry_date must be on or after snapshot_date.")
    if days_to_expiry <= 45:
        return "DGS1MO"
    if days_to_expiry <= 135:
        return "DGS3MO"
    if days_to_expiry <= 270:
        return "DGS6MO"
    return "DGS1"


def get_risk_free_rate(
    snapshot_date: date | str,
    expiry_date: date | str,
    data_root: str | Path | None = None,
) -> RiskFreeRateMatch:
    """Select the latest available local Treasury rate on or before the snapshot date.

    Prior-business-day fallback is explicit because FRED series do not publish
    weekend or market-holiday observations.
    """

    snapshot = parse_date(snapshot_date)
    expiry = parse_date(expiry_date)
    if snapshot is None or expiry is None:
        raise ValueError("snapshot_date and expiry_date must be valid dates.")

    days_to_expiry = (expiry - snapshot).days
    series_id = select_series_for_days(days_to_expiry)
    merged = load_merged_rates(data_root)
    if merged.empty:
        raise LookupError("Merged FRED rate table is empty.")

    percent_column = SERIES_PERCENT_COLUMN[series_id]
    decimal_column = SERIES_DECIMAL_COLUMN[series_id]
    lookup_date = pd.Timestamp(snapshot)
    eligible = merged.loc[
        (merged["date"] <= lookup_date) & merged[decimal_column].notna(),
        ["date", percent_column, decimal_column],
    ].sort_values("date")
    if eligible.empty:
        raise LookupError(
            f"No {series_id} rate was available on or before {snapshot.isoformat()}."
        )
    match = eligible.iloc[-1]
    matched_date = pd.Timestamp(match["date"]).date()
    return RiskFreeRateMatch(
        snapshot_date=snapshot,
        expiry_date=expiry,
        days_to_expiry=days_to_expiry,
        series_used=series_id,
        matched_date=matched_date,
        rate_percent=float(match[percent_column]),
        rate_decimal=float(match[decimal_column]),
        used_prior_date=matched_date != snapshot,
    )


def get_latest_rates_snapshot(data_root: str | Path | None = None) -> dict:
    """Load the fast-read JSON snapshot of the latest known rate values."""

    return load_latest_rates_snapshot(data_root)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for one-off local rate selection checks."""

    parser = argparse.ArgumentParser(description="Select the best local FRED risk-free rate for an option horizon.")
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--expiry-date", required=True)
    parser.add_argument("--data-root", help="Override the Options data root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the one-off local rate-selector CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    match = get_risk_free_rate(
        snapshot_date=args.snapshot_date,
        expiry_date=args.expiry_date,
        data_root=args.data_root,
    )
    print(json.dumps(match.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
