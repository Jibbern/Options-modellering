"""Downloader for the local Nasdaq-backed historical price store."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..utils import clean_string, parse_date
from .price_store import (
    PRICE_SOURCE_NASDAQ,
    build_manifest,
    combine_price_history,
    ensure_price_structure,
    isoformat_utc,
    load_price_history,
    normalize_rows,
    save_manifest,
    save_price_history,
    utc_now,
    write_raw_response,
    write_source_notes,
)

NASDAQ_ENDPOINT = "https://api.nasdaq.com/api/quote/{ticker}/historical"
DEFAULT_START_DATE = date(1990, 1, 1)
DEFAULT_LIMIT = 5000
REQUEST_TIMEOUT_SECONDS = 30


def build_headers(ticker: str) -> dict[str, str]:
    """Build browser-like headers for the Nasdaq historical endpoint."""

    ticker_lower = clean_string(ticker).lower()
    return {
        "accept": "application/json, text/plain, */*",
        "accept-language": "en-US,en;q=0.9",
        "origin": "https://www.nasdaq.com",
        "referer": f"https://www.nasdaq.com/market-activity/stocks/{ticker_lower}/historical",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }


def _coerce_total_records(payload: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    data = payload.get("data") or {}
    total = data.get("totalRecords") or data.get("totalrecords") or data.get("total")
    if total is None:
        return len(rows)
    text = clean_string(total).replace(",", "")
    try:
        return int(float(text))
    except ValueError:
        return len(rows)


def extract_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int, str | None]:
    """Extract row data from the known Nasdaq JSON/XHR response shapes."""

    data = payload.get("data")
    if not isinstance(data, dict):
        message = clean_string(payload.get("message")) or clean_string(payload.get("statusMessage"))
        return [], 0, message or "Nasdaq response did not include a data object."

    tables = [
        data.get("tradesTable"),
        data.get("historicalTable"),
        data.get("table"),
    ]
    rows: list[dict[str, Any]] = []
    for table in tables:
        if isinstance(table, dict) and isinstance(table.get("rows"), list):
            rows = [row for row in table["rows"] if isinstance(row, dict)]
            break

    message = clean_string(payload.get("message")) or clean_string(data.get("message"))
    total_records = _coerce_total_records(payload, rows)
    return rows, total_records, message or None


def _build_error_payload(
    *,
    status_code: int | None = None,
    text: str | None = None,
    error: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {}
    if status_code is not None:
        document["status_code"] = status_code
    if text is not None:
        document["response_text"] = text
    if error is not None:
        document["error"] = error
    if payload is not None:
        document["payload"] = payload
    return document


def resolve_request_window(
    ticker: str,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
    data_root: str | Path | None = None,
    full_refresh: bool = False,
) -> tuple[date, date, str]:
    """Resolve the historical download window for a full or incremental refresh."""

    end_date = parse_date(end) or pd.Timestamp.now(tz="UTC").date()
    start_date = parse_date(start)
    if start_date is None:
        if full_refresh:
            start_date = DEFAULT_START_DATE
            mode = "full_refresh"
        else:
            latest_date = None
            history = load_price_history(ticker, data_root)
            if not history.empty:
                valid = history.dropna(subset=["close"]).sort_values("date")
                if not valid.empty:
                    latest_date = pd.Timestamp(valid.iloc[-1]["date"]).date()
            if latest_date is not None:
                start_date = max(latest_date - timedelta(days=14), DEFAULT_START_DATE)
                mode = "incremental"
            else:
                start_date = DEFAULT_START_DATE
                mode = "full_refresh"
    else:
        mode = "custom_window"

    if start_date > end_date:
        raise ValueError("start date must be on or before end date.")
    return start_date, end_date, mode


def download_nasdaq_prices(
    ticker: str,
    data_root: str | Path | None = None,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
    full_refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download Nasdaq historical prices into the local per-ticker store."""

    ticker_upper = clean_string(ticker).upper()
    start_date, end_date, mode = resolve_request_window(
        ticker_upper,
        start=start,
        end=end,
        data_root=data_root,
        full_refresh=full_refresh,
    )
    active_session = session or requests.Session()
    downloaded_at = utc_now()
    raw_files: list[str] = []
    all_rows: list[dict[str, Any]] = []
    page = 0
    offset = 0
    total_records: int | None = None
    endpoint = NASDAQ_ENDPOINT.format(ticker=ticker_upper.lower())

    while True:
        # Nasdaq exposes historical rows behind a JSON/XHR endpoint rather than
        # a stable static HTML table, so we page through that response directly.
        request_params = {
            "assetclass": "stocks",
            "fromdate": start_date.isoformat(),
            "todate": end_date.isoformat(),
            "limit": DEFAULT_LIMIT,
            "offset": offset,
        }
        try:
            response = active_session.get(
                endpoint,
                params=request_params,
                headers=build_headers(ticker_upper),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            raw_path = write_raw_response(
                ticker_upper,
                _build_error_payload(error=str(exc)),
                downloaded_at=downloaded_at,
                request_params=request_params,
                data_root=data_root,
                suffix=f"error_p{page:03d}",
            )
            raw_files.append(str(raw_path))
            raise RuntimeError(
                f"Nasdaq historical download failed for {ticker_upper}: {exc}"
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raw_path = write_raw_response(
                ticker_upper,
                _build_error_payload(
                    status_code=response.status_code,
                    text=response.text,
                    error="Response was not valid JSON.",
                ),
                downloaded_at=downloaded_at,
                request_params=request_params,
                data_root=data_root,
                suffix=f"error_p{page:03d}",
            )
            raw_files.append(str(raw_path))
            raise RuntimeError(
                f"Nasdaq historical download returned non-JSON content for {ticker_upper}."
            ) from exc

        raw_path = write_raw_response(
            ticker_upper,
            payload,
            downloaded_at=downloaded_at,
            request_params=request_params,
            data_root=data_root,
            suffix=None if page == 0 else f"p{page:03d}",
        )
        raw_files.append(str(raw_path))

        if response.status_code >= 400:
            raise RuntimeError(
                f"Nasdaq historical download failed for {ticker_upper} with HTTP {response.status_code}."
            )

        rows, page_total_records, message = extract_rows(payload)
        if page == 0:
            total_records = page_total_records
        if not rows and page == 0:
            detail = message or "Nasdaq returned no historical rows for the requested window."
            raise RuntimeError(f"{detail} Manual CSV import is the supported fallback.")

        all_rows.extend(rows)
        page += 1
        if not rows:
            break
        if total_records is not None:
            if offset + len(rows) >= total_records:
                break
        elif len(rows) < DEFAULT_LIMIT:
            break
        offset += len(rows)

    incoming = normalize_rows(
        all_rows,
        ticker=ticker_upper,
        source=PRICE_SOURCE_NASDAQ,
        downloaded_at=downloaded_at,
    )
    if incoming.empty or incoming["close"].notna().sum() == 0:
        raise RuntimeError(
            f"Nasdaq historical download for {ticker_upper} did not yield usable close prices. Manual CSV import is the supported fallback."
        )

    existing = load_price_history(ticker_upper, data_root)
    combined = combine_price_history(existing, incoming, ticker=ticker_upper)
    saved_files = save_price_history(ticker_upper, combined, data_root)

    source_notes = {
        "ticker": ticker_upper,
        "primary_source": PRICE_SOURCE_NASDAQ,
        "transport": "direct_json_xhr",
        "endpoint": endpoint,
        "request_headers": {
            key: value for key, value in build_headers(ticker_upper).items() if key != "user-agent"
        },
        "manual_import_directory": str(ensure_price_structure(ticker_upper, data_root)["raw_manual"]),
        "notes": [
            "The downloader targets Nasdaq historical quotes JSON/XHR rather than scraping HTML.",
            "If Nasdaq returns unavailable data or blocks the request, drop a manual CSV into raw/manual and normalize it with normalize_manual_price_file(...).",
        ],
        "latest_downloaded_at": isoformat_utc(downloaded_at),
    }
    source_notes_path = write_source_notes(ticker_upper, source_notes, data_root)

    manifest = build_manifest(
        ticker=ticker_upper,
        data_root=data_root,
        raw_files=raw_files,
        request_window={
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "full_refresh": bool(full_refresh),
            "resolved_mode": mode,
        },
        history=combined,
        saved_files=saved_files,
        source_notes_path=source_notes_path,
        download_mode=mode,
    )
    manifest_path = save_manifest(ticker_upper, manifest, data_root)
    manifest["manifest_path"] = manifest_path
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for historical-price refresh commands."""

    parser = argparse.ArgumentParser(description="Download and store Nasdaq historical daily prices.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--start", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--data-root", help="Override the Options data root.")
    parser.add_argument("--full-refresh", action="store_true", help="Fetch a full available history window.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Nasdaq historical-price downloader CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = download_nasdaq_prices(
        ticker=args.ticker,
        data_root=args.data_root,
        start=args.start,
        end=args.end,
        full_refresh=args.full_refresh,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
