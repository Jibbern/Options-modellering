"""Helpers for selecting a local historical spot price for a snapshot date."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ..utils import parse_date
from .price_store import load_price_history


@dataclass(frozen=True)
class SpotPriceMatch:
    """Resolved local spot-price match for one snapshot date."""

    ticker: str
    snapshot_date: date
    matched_date: date
    close_price: float
    source: str
    used_prior_date: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["snapshot_date"] = self.snapshot_date.isoformat()
        payload["matched_date"] = self.matched_date.isoformat()
        return payload


def get_latest_price_date(
    ticker: str,
    data_root: str | Path | None = None,
) -> date | None:
    """Return the latest local trading date with a non-null close."""

    history = load_price_history(ticker, data_root)
    if history.empty:
        return None
    valid = history.dropna(subset=["close"]).sort_values("date")
    if valid.empty:
        return None
    return pd.Timestamp(valid.iloc[-1]["date"]).date()


def get_underlying_spot(
    ticker: str,
    snapshot_date: date | str,
    data_root: str | Path | None = None,
) -> SpotPriceMatch:
    """Return the latest available close on or before a snapshot date.

    Weekend and holiday snapshot dates fall back to the most recent prior
    trading session because the options lab is built around end-of-day
    snapshots, not intraday pricing.
    """

    snapshot = parse_date(snapshot_date)
    if snapshot is None:
        raise ValueError("snapshot_date must be a valid date.")

    history = load_price_history(ticker, data_root)
    if history.empty:
        raise FileNotFoundError(
            f"No local historical price store exists for {ticker}. Run the downloader or import a manual CSV first."
        )

    eligible = history.loc[
        (history["date"] <= pd.Timestamp(snapshot)) & history["close"].notna(),
        ["date", "close", "source"],
    ].sort_values("date")
    if eligible.empty:
        raise LookupError(
            f"No closing price for {ticker.upper()} was available on or before {snapshot.isoformat()}."
        )

    match = eligible.iloc[-1]
    matched_date = pd.Timestamp(match["date"]).date()
    return SpotPriceMatch(
        ticker=ticker.upper(),
        snapshot_date=snapshot,
        matched_date=matched_date,
        close_price=float(match["close"]),
        source=str(match["source"]),
        used_prior_date=matched_date != snapshot,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the small CLI parser for one-off local spot checks."""

    parser = argparse.ArgumentParser(description="Select the best local historical close for a snapshot date.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--data-root", help="Override the Options data root.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the one-off local spot-selector CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    match = get_underlying_spot(
        ticker=args.ticker,
        snapshot_date=args.snapshot_date,
        data_root=args.data_root,
    )
    print(json.dumps(match.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
