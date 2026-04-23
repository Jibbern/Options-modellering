"""Downloader for the local FRED-backed nominal risk-free-rate store."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ..utils import parse_date
from .rate_store import (
    FRED_SERIES,
    FRED_SOURCE,
    build_manifest,
    build_merged_table,
    combine_series_history,
    ensure_fred_structure,
    isoformat_utc,
    load_series_history,
    save_latest_snapshot_files,
    save_manifest,
    save_merged_table,
    save_series_history,
    write_raw_response,
)

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredApiKeyError(RuntimeError):
    """Raised when the FRED API key is missing."""


def get_fred_api_key() -> str:
    """Read the FRED API key from the environment or fail clearly."""

    api_key = os.getenv("FRED_API_KEY", "").strip()
    if api_key:
        return api_key
    raise FredApiKeyError(
        "FRED_API_KEY is not set. Set it before running the downloader. "
        "PowerShell example: $env:FRED_API_KEY='your_api_key_here'"
    )


def normalize_series_ids(series_ids: list[str] | tuple[str, ...] | None) -> list[str]:
    """Validate and de-duplicate requested FRED series IDs."""

    if not series_ids:
        return list(FRED_SERIES)
    normalized = []
    for series_id in series_ids:
        candidate = str(series_id).upper()
        if candidate not in FRED_SERIES:
            raise ValueError(f"Unsupported FRED series: {series_id}")
        if candidate not in normalized:
            normalized.append(candidate)
    return normalized


def determine_request_window(
    existing: pd.DataFrame,
    *,
    start: date | str | None,
    end: date | str | None,
    full_refresh: bool,
) -> tuple[str | None, str | None]:
    """Resolve the API observation window for a refresh request."""

    start_date = parse_date(start)
    end_date = parse_date(end)
    if start_date and end_date and start_date > end_date:
        raise ValueError("start must be on or before end.")
    if full_refresh:
        return (
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
        )
    if start_date or end_date:
        return (
            start_date.isoformat() if start_date else None,
            end_date.isoformat() if end_date else None,
        )
    if existing.empty or existing["date"].dropna().empty:
        return None, None
    last_date = pd.to_datetime(existing["date"]).max().date()
    incremental_start = last_date - timedelta(days=7)
    return incremental_start.isoformat(), None


def fetch_series_observations(
    series_id: str,
    *,
    api_key: str,
    start: str | None = None,
    end: str | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch one series payload from the official FRED observations API."""

    request_params = {
        "series_id": series_id.upper(),
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if start:
        request_params["observation_start"] = start
    if end:
        request_params["observation_end"] = end
    client = session or requests.Session()
    response = client.get(FRED_OBSERVATIONS_URL, params=request_params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    safe_request_params = {key: value for key, value in request_params.items() if key != "api_key"}
    return payload, safe_request_params


def normalize_fred_payload(
    series_id: str,
    payload: dict[str, Any],
    *,
    downloaded_at: datetime,
) -> pd.DataFrame:
    """Normalize one raw FRED payload into the internal daily-rate schema."""

    rows: list[dict[str, Any]] = []
    for observation in payload.get("observations", []):
        value = str(observation.get("value", "")).strip()
        if value in {"", "."}:
            rate_percent = None
            rate_decimal = None
            status = "missing"
        else:
            rate_percent = float(value)
            rate_decimal = rate_percent / 100.0
            status = "observed"
        rows.append(
            {
                "date": observation.get("date"),
                "series_id": series_id.upper(),
                "rate_percent": rate_percent,
                "rate_decimal": rate_decimal,
                "source": FRED_SOURCE,
                "downloaded_at": isoformat_utc(downloaded_at),
                "observation_status": status,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "series_id",
                "rate_percent",
                "rate_decimal",
                "source",
                "downloaded_at",
                "observation_status",
            ]
        )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["downloaded_at"] = pd.to_datetime(frame["downloaded_at"], errors="coerce", utc=True)
    frame = frame.sort_values(["date", "downloaded_at"]).drop_duplicates(subset=["date"], keep="last")
    return frame.reset_index(drop=True)


def download_fred_rates(
    data_root: str | Path | None = None,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
    series_ids: list[str] | tuple[str, ...] | None = None,
    full_refresh: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Download configured Treasury series into the local FRED store."""

    ensure_fred_structure(data_root)
    api_key = get_fred_api_key()
    selected_series = normalize_series_ids(series_ids)
    downloaded_at = datetime.now(timezone.utc).replace(microsecond=0)
    raw_paths: dict[str, str] = {}
    updated_frames: dict[str, pd.DataFrame] = {}
    series_write_results: dict[str, dict[str, str]] = {}
    last_request_window: dict[str, Any] = {
        "start": parse_date(start).isoformat() if parse_date(start) else None,
        "end": parse_date(end).isoformat() if parse_date(end) else None,
        "full_refresh": full_refresh,
    }

    for series_id in selected_series:
        existing = load_series_history(series_id, data_root)
        request_start, request_end = determine_request_window(
            existing,
            start=start,
            end=end,
            full_refresh=full_refresh,
        )
        payload, request_params = fetch_series_observations(
            series_id,
            api_key=api_key,
            start=request_start,
            end=request_end,
            session=session,
        )
        raw_path = write_raw_response(
            series_id,
            payload,
            downloaded_at=downloaded_at,
            request_params=request_params,
            data_root=data_root,
        )
        incoming = normalize_fred_payload(
            series_id,
            payload,
            downloaded_at=downloaded_at,
        )
        combined = combine_series_history(
            existing,
            incoming,
            series_id,
            replace_existing=full_refresh,
        )
        series_write_results[series_id] = save_series_history(series_id, combined, data_root)
        updated_frames[series_id] = combined
        raw_paths[series_id] = str(raw_path)
        last_request_window = {
            "start": request_start,
            "end": request_end,
            "full_refresh": full_refresh,
        }

    all_series_frames = {
        series_id: updated_frames.get(series_id, load_series_history(series_id, data_root))
        for series_id in FRED_SERIES
    }
    merged = build_merged_table(all_series_frames)
    merged_paths = save_merged_table(merged, data_root)
    latest_paths = save_latest_snapshot_files(all_series_frames, data_root)
    manifest = build_manifest(
        data_root=data_root,
        raw_paths=raw_paths,
        series_frames=all_series_frames,
        series_write_results=series_write_results,
        merged_frame=merged,
        latest_paths=latest_paths,
        merged_paths=merged_paths,
        requested_series=selected_series,
        request_window=last_request_window,
    )
    manifest["manifest_path"] = save_manifest(manifest, data_root)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for local FRED refresh commands."""

    parser = argparse.ArgumentParser(description="Download nominal Treasury constant-maturity rates from FRED.")
    parser.add_argument("--data-root", help="Override the Options data root.")
    parser.add_argument("--start", help="Observation start date in YYYY-MM-DD.")
    parser.add_argument("--end", help="Observation end date in YYYY-MM-DD.")
    parser.add_argument("--full-refresh", action="store_true", help="Replace local history for the requested series.")
    parser.add_argument(
        "--series",
        action="append",
        help="Series to download. May be passed multiple times. Defaults to all configured series.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the FRED downloader CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = download_fred_rates(
            data_root=args.data_root,
            start=args.start,
            end=args.end,
            series_ids=args.series,
            full_refresh=args.full_refresh,
        )
    except FredApiKeyError as exc:
        parser.exit(status=2, message=f"{exc}\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
