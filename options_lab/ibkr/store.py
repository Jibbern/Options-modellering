"""Local persistence helpers for IBKR delayed snapshots and chains."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..persistence import write_dataframe_csv, write_dataframe_parquet_if_available, write_json
from ..utils import clean_string, ensure_directory, parse_date
from .models import (
    ChainFetchResult,
    ChainRow,
    ConnectionSettings,
    IbkrSpotMatch,
    OptionQuoteSnapshot,
    OptionSnapshotFetchResult,
    UnderlyingQuoteSnapshot,
    isoformat_utc,
)


def options_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return options_root() / "data"


def ibkr_root(ticker: str, data_root: str | Path | None = None) -> Path:
    base = Path(data_root) if data_root is not None else default_data_root()
    return base / clean_string(ticker).upper() / "ibkr"


def ensure_ibkr_structure(ticker: str, data_root: str | Path | None = None) -> dict[str, Path]:
    root = ibkr_root(ticker, data_root)
    return {
        "root": ensure_directory(root),
        "snapshots": ensure_directory(root / "snapshots"),
        "underlying_raw": ensure_directory(root / "snapshots" / "underlying" / "raw"),
        "underlying_normalized": ensure_directory(root / "snapshots" / "underlying" / "normalized"),
        "option_quotes_raw": ensure_directory(root / "snapshots" / "option_quotes" / "raw"),
        "option_quotes_normalized": ensure_directory(root / "snapshots" / "option_quotes" / "normalized"),
        "chains_raw": ensure_directory(root / "chains" / "raw"),
        "chains_normalized": ensure_directory(root / "chains" / "normalized"),
        "metadata": ensure_directory(root / "metadata"),
    }


def timestamp_slug(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _underlying_file_stem(snapshot: UnderlyingQuoteSnapshot) -> str:
    mode = clean_string(snapshot.market_data_mode).lower()
    return f"ibkr_{clean_string(snapshot.ticker).lower()}_underlying_{timestamp_slug(snapshot.snapshot_timestamp)}_{mode}"


def _option_file_stem(ticker: str, snapshot_timestamp: datetime, expiry_date: str, market_data_mode: str) -> str:
    mode = clean_string(market_data_mode).lower()
    return (
        f"ibkr_{clean_string(ticker).lower()}_options_exp_{expiry_date}_"
        f"{timestamp_slug(snapshot_timestamp)}_{mode}"
    )


def _chain_file_stem(ticker: str, fetched_at: datetime, market_data_mode: str) -> str:
    mode = clean_string(market_data_mode).lower()
    return f"ibkr_{clean_string(ticker).lower()}_chain_{timestamp_slug(fetched_at)}_{mode}"


def _metadata_path(ticker: str, name: str, data_root: str | Path | None = None) -> Path:
    return ibkr_root(ticker, data_root) / "metadata" / name


def _source_notes_payload(ticker: str, last_run: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ticker": clean_string(ticker).upper(),
        "primary_source": "ibkr",
        "policy": {
            "delayed_only": True,
            "allowed_market_data_modes": ["delayed", "delayed_frozen"],
            "forbidden_market_data_modes": ["live", "frozen", "regulatory_snapshot"],
        },
        "last_run": last_run or {},
    }


def _write_source_notes(ticker: str, payload: dict[str, Any], data_root: str | Path | None = None) -> str:
    path = _metadata_path(ticker, "source_notes.json", data_root)
    write_json(payload, path)
    return str(path)


def record_request_failure(
    ticker: str,
    *,
    request_type: str,
    market_data_mode: str,
    connection: ConnectionSettings,
    error_message: str,
    warnings: list[str] | None = None,
    diagnostics: dict[str, Any] | None = None,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    ensure_ibkr_structure(ticker, data_root)
    manifest_name = {
        "underlying": "underlying_manifest.json",
        "chain": "chain_manifest.json",
        "option_snapshot": "option_snapshot_manifest.json",
    }.get(clean_string(request_type).lower(), f"{clean_string(request_type).lower()}_manifest.json")
    manifest_path = _metadata_path(ticker, manifest_name, data_root)
    payload = {
        "generated_at": isoformat_utc(datetime.now(timezone.utc)),
        "ticker": clean_string(ticker).upper(),
        "status": "failed",
        "request_type": clean_string(request_type).lower(),
        "market_data_mode": clean_string(market_data_mode).lower(),
        "request": {
            "connection": connection.to_dict(),
        },
        "error": error_message,
        "warnings": list(warnings or []),
    }
    if diagnostics:
        payload["failure_stage"] = diagnostics.get("failure_stage")
        payload["diagnostics"] = diagnostics
    write_json(payload, manifest_path)
    source_notes_path = _write_source_notes(
        ticker,
        _source_notes_payload(
            ticker,
            last_run={
                "type": clean_string(request_type).lower(),
                "status": "failed",
                "market_data_mode": clean_string(market_data_mode).lower(),
                "error": error_message,
                "warnings": list(warnings or []),
            },
        ),
        data_root,
    )
    return {
        "manifest_path": str(manifest_path),
        "source_notes_path": source_notes_path,
    }


def _save_frame(frame: pd.DataFrame, csv_path: Path, parquet_path: Path) -> dict[str, str]:
    write_dataframe_csv(frame, csv_path, index=False)
    parquet_result = write_dataframe_parquet_if_available(frame, parquet_path, index=False)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "parquet_written": str(parquet_result.written),
        "parquet_note": parquet_result.note or "",
    }


def save_underlying_snapshot(
    snapshot: UnderlyingQuoteSnapshot,
    *,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    structure = ensure_ibkr_structure(snapshot.ticker, data_root)
    stem = _underlying_file_stem(snapshot)
    record = snapshot.to_record()
    frame = pd.DataFrame([record])

    raw_path = structure["underlying_raw"] / f"{stem}.json"
    csv_path = structure["underlying_normalized"] / f"{stem}.csv"
    parquet_path = structure["underlying_normalized"] / f"{stem}.parquet"
    write_json(
        {
            "snapshot": record,
            "connection": snapshot.connection.to_dict(),
        },
        raw_path,
    )
    saved = _save_frame(frame, csv_path, parquet_path)

    manifest = {
        "generated_at": isoformat_utc(datetime.now(timezone.utc)),
        "ticker": clean_string(snapshot.ticker).upper(),
        "market_data_mode": clean_string(snapshot.market_data_mode).lower(),
        "request": {
            "timestamp": isoformat_utc(snapshot.snapshot_timestamp),
            "connection": snapshot.connection.to_dict(),
        },
        "raw_json": str(raw_path),
        "normalized_files": {
            "csv": saved["csv"],
            "parquet": saved["parquet"],
            "parquet_written": bool(saved["parquet_written"] == "True"),
            "parquet_note": saved["parquet_note"] or None,
        },
        "field_availability": {
            "missing_fields": list(snapshot.missing_fields),
            "warnings": list(snapshot.warnings),
        },
        "resolved_underlying_contract": snapshot.resolved_underlying.to_dict() if snapshot.resolved_underlying else None,
        "latest_snapshot": record,
    }
    manifest_path = _metadata_path(snapshot.ticker, "underlying_manifest.json", data_root)
    write_json(manifest, manifest_path)
    source_notes_path = _write_source_notes(
        snapshot.ticker,
        _source_notes_payload(
            snapshot.ticker,
            last_run={
                "type": "underlying",
                "timestamp": isoformat_utc(snapshot.snapshot_timestamp),
                "market_data_mode": clean_string(snapshot.market_data_mode).lower(),
                "warnings": list(snapshot.warnings),
            },
        ),
        data_root,
    )
    return {
        "raw_json": str(raw_path),
        "normalized_csv": saved["csv"],
        "normalized_parquet": saved["parquet"],
        "manifest_path": str(manifest_path),
        "source_notes_path": source_notes_path,
    }


def _option_snapshot_frame(quotes: list[OptionQuoteSnapshot]) -> pd.DataFrame:
    return pd.DataFrame([quote.to_record() for quote in quotes])


def _chain_compatible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()
    result["strike"] = frame["strike"]
    result["moneyness"] = None
    result["bid"] = frame["bid"]
    result["mid"] = frame["mid"]
    result["ask"] = frame["ask"]
    result["last"] = frame["last"]
    result["change"] = None
    result["pct_change"] = None
    result["volume"] = frame["volume"]
    result["open_interest"] = frame["open_interest"]
    result["oi_change"] = None
    result["iv"] = frame["implied_volatility"]
    result["delta"] = frame["delta"]
    result["type"] = frame["option_type"]
    result["last_trade"] = None
    return result


def _coverage_metric(frame: pd.DataFrame, field: str) -> dict[str, Any]:
    total = int(len(frame.index))
    available_count = int(frame[field].notna().sum()) if field in frame.columns else 0
    return {
        "available_count": available_count,
        "missing_count": max(total - available_count, 0),
        "coverage_pct": round((available_count / total) * 100.0, 1) if total else 0.0,
    }


def _coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    tracked_fields = ("bid", "ask", "implied_volatility", "volume", "open_interest")
    overall = {field: _coverage_metric(frame, field) for field in tracked_fields}
    by_expiry: dict[str, Any] = {}
    if "expiry_date" in frame.columns:
        for expiry, group in frame.groupby("expiry_date", dropna=False):
            expiry_key = parse_date(expiry).isoformat() if parse_date(expiry) else "unknown"
            by_expiry[expiry_key] = {
                "contract_count": int(len(group.index)),
                **{field: _coverage_metric(group, field) for field in tracked_fields},
            }
    return {
        "contract_count": int(len(frame.index)),
        "overall": overall,
        "by_expiry": by_expiry,
    }


def save_option_snapshot(
    quotes: list[OptionQuoteSnapshot] | OptionSnapshotFetchResult,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    result_payload = quotes if isinstance(quotes, OptionSnapshotFetchResult) else None
    diagnostics = result_payload.diagnostics if result_payload is not None else None
    quotes_list = result_payload.quotes if result_payload is not None else quotes
    if not quotes_list:
        raise ValueError("At least one option quote snapshot is required.")
    snapshot_scope = clean_string(getattr(diagnostics, "snapshot_scope", None)).lower() or "filtered_slice"
    storage_location = "ibkr_full_quoted_snapshot" if snapshot_scope == "full_chain" else "ibkr_chain_snapshot"
    ticker = quotes_list[0].ticker
    structure = ensure_ibkr_structure(ticker, data_root)
    frame = _option_snapshot_frame(quotes_list)
    snapshot_timestamp = quotes_list[0].snapshot_timestamp
    mode = clean_string(quotes_list[0].market_data_mode).lower()
    stem = _option_file_stem(ticker, snapshot_timestamp, "multi", mode)
    raw_path = structure["option_quotes_raw"] / f"{stem}.json"
    csv_path = structure["option_quotes_normalized"] / f"{stem}.csv"
    parquet_path = structure["option_quotes_normalized"] / f"{stem}.parquet"
    write_json(
        {
            "quotes": frame.to_dict(orient="records"),
            "connection": quotes_list[0].connection.to_dict(),
            "diagnostics": diagnostics.to_dict() if diagnostics is not None else None,
        },
        raw_path,
    )
    saved = _save_frame(frame, csv_path, parquet_path)

    chain_slice_files: list[str] = []
    for expiry_date, expiry_frame in frame.groupby("expiry_date", dropna=False):
        expiry_text = parse_date(expiry_date).isoformat() if parse_date(expiry_date) else "unknown"
        slice_stem = _option_file_stem(ticker, snapshot_timestamp, expiry_text, mode)
        slice_path = structure["option_quotes_normalized"] / f"{slice_stem}.csv"
        chain_frame = _chain_compatible_frame(expiry_frame)
        write_dataframe_csv(chain_frame, slice_path, index=False)
        sidecar_payload = {
            "ticker": clean_string(ticker).upper(),
            "snapshot_date": snapshot_timestamp.date().isoformat(),
            "snapshot_time": snapshot_timestamp.astimezone(timezone.utc).strftime("%H:%M:%S"),
            "expiry_date": expiry_text if expiry_text != "unknown" else None,
            "spot_price": float(expiry_frame["under_price"].dropna().iloc[0]) if expiry_frame["under_price"].notna().any() else None,
            "spot_price_source": "ibkr_delayed",
            "source": "ibkr",
            "market_data_mode": mode,
            "snapshot_scope": snapshot_scope,
            "storage_location": storage_location,
            "quote_count": int(len(expiry_frame.index)),
            "strike_count": int(expiry_frame["strike"].dropna().nunique()),
        }
        write_json(sidecar_payload, slice_path.with_suffix(".metadata.json"))
        chain_slice_files.append(str(slice_path))

    manifest = {
        "generated_at": isoformat_utc(datetime.now(timezone.utc)),
        "ticker": clean_string(ticker).upper(),
        "market_data_mode": mode,
        "request": {
            "timestamp": isoformat_utc(snapshot_timestamp),
                "connection": quotes_list[0].connection.to_dict(),
        },
        "raw_json": str(raw_path),
        "normalized_files": {
            "csv": saved["csv"],
            "parquet": saved["parquet"],
            "parquet_written": bool(saved["parquet_written"] == "True"),
            "parquet_note": saved["parquet_note"] or None,
        },
        "chain_slice_files": chain_slice_files,
        "request_filters": diagnostics.contract_match.to_dict() if diagnostics is not None else None,
        "snapshot_scope": snapshot_scope,
        "storage_location": storage_location,
        "discovered_expiries": list(getattr(diagnostics, "discovered_expiries", []) or []),
        "strike_count_by_expiry": dict(getattr(diagnostics, "strike_count_by_expiry", {}) or {}),
        "attempted_contract_count": int(getattr(diagnostics, "attempted_contract_count", len(quotes_list)) or 0),
        "persisted_quote_count": int(len(frame)),
        "coverage_summary": _coverage_summary(frame),
        "field_availability": {
            "missing_fields": sorted({field for quote in quotes_list for field in quote.missing_fields}),
            "missing_field_counts": (getattr(diagnostics, "delayed_field_summary", {}) or {}).get("missing_field_counts", {}),
            "warnings": sorted({warning for quote in quotes_list for warning in quote.warnings}),
        },
        "row_count": int(len(frame)),
        "expiries": sorted({value for value in frame["expiry_date"].dropna().tolist()}),
        "diagnostics": diagnostics.to_dict() if diagnostics is not None else None,
    }
    manifest_path = _metadata_path(ticker, "option_snapshot_manifest.json", data_root)
    write_json(manifest, manifest_path)
    source_notes_path = _write_source_notes(
        ticker,
        _source_notes_payload(
            ticker,
            last_run={
                "type": "option_snapshot",
                "timestamp": isoformat_utc(snapshot_timestamp),
                "market_data_mode": mode,
                "row_count": int(len(frame)),
            },
        ),
        data_root,
    )
    return {
        "raw_json": str(raw_path),
        "normalized_csv": saved["csv"],
        "normalized_parquet": saved["parquet"],
        "manifest_path": str(manifest_path),
        "source_notes_path": source_notes_path,
        "chain_slice_files": chain_slice_files,
    }


def save_chain_rows(
    rows: list[ChainRow] | ChainFetchResult,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    result_payload = rows if isinstance(rows, ChainFetchResult) else None
    diagnostics = result_payload.diagnostics if result_payload is not None else None
    row_list = result_payload.rows if result_payload is not None else rows
    if not row_list:
        raise ValueError("At least one chain row is required.")
    ticker = row_list[0].ticker
    structure = ensure_ibkr_structure(ticker, data_root)
    frame = pd.DataFrame([row.to_record() for row in row_list])
    fetched_at = row_list[0].fetched_at
    mode = clean_string(row_list[0].market_data_mode).lower()
    stem = _chain_file_stem(ticker, fetched_at, mode)
    raw_path = structure["chains_raw"] / f"{stem}.json"
    csv_path = structure["chains_normalized"] / f"{stem}.csv"
    parquet_path = structure["chains_normalized"] / f"{stem}.parquet"
    write_json(
        {
            "rows": frame.to_dict(orient="records"),
            "connection": row_list[0].connection.to_dict(),
            "diagnostics": diagnostics.to_dict() if diagnostics is not None else None,
        },
        raw_path,
    )
    saved = _save_frame(frame, csv_path, parquet_path)
    manifest = {
        "generated_at": isoformat_utc(datetime.now(timezone.utc)),
        "ticker": clean_string(ticker).upper(),
        "market_data_mode": mode,
        "snapshot_scope": "chain_universe",
        "request": {
            "timestamp": isoformat_utc(fetched_at),
                "connection": row_list[0].connection.to_dict(),
        },
        "raw_json": str(raw_path),
        "normalized_files": {
            "csv": saved["csv"],
            "parquet": saved["parquet"],
            "parquet_written": bool(saved["parquet_written"] == "True"),
            "parquet_note": saved["parquet_note"] or None,
        },
        "row_count": int(len(frame)),
        "discovered_expiries": sorted({value for value in frame["expiry_date"].dropna().tolist()}),
        "strike_count_by_expiry": {
            str(expiry): int(group["strike"].dropna().nunique())
            for expiry, group in frame.groupby("expiry_date", dropna=True)
        },
        "underlying_conid": frame["underlying_conid"].dropna().iloc[0] if frame["underlying_conid"].notna().any() else None,
        "diagnostics": diagnostics.to_dict() if diagnostics is not None else None,
    }
    manifest_path = _metadata_path(ticker, "chain_manifest.json", data_root)
    write_json(manifest, manifest_path)
    source_notes_path = _write_source_notes(
        ticker,
        _source_notes_payload(
            ticker,
            last_run={
                "type": "chain",
                "timestamp": isoformat_utc(fetched_at),
                "market_data_mode": mode,
                "row_count": int(len(frame)),
            },
        ),
        data_root,
    )
    return {
        "raw_json": str(raw_path),
        "normalized_csv": saved["csv"],
        "normalized_parquet": saved["parquet"],
        "manifest_path": str(manifest_path),
        "source_notes_path": source_notes_path,
    }


def save_full_chain_snapshot_run(
    ticker: str,
    *,
    market_data_mode: str,
    connection: ConnectionSettings,
    underlying_files: dict[str, Any],
    chain_files: dict[str, Any],
    option_snapshot_files: dict[str, Any],
    option_snapshot: OptionSnapshotFetchResult,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    ensure_ibkr_structure(ticker, data_root)
    diagnostics = option_snapshot.diagnostics
    delayed_field_summary = diagnostics.delayed_field_summary or {}
    warnings = sorted({warning for quote in option_snapshot.quotes for warning in quote.warnings})
    manifest = {
        "generated_at": isoformat_utc(datetime.now(timezone.utc)),
        "ticker": clean_string(ticker).upper(),
        "status": "succeeded",
        "request_type": "full_chain_snapshot",
        "market_data_mode": clean_string(market_data_mode).lower(),
        "request": {
            "connection": connection.to_dict(),
        },
        "snapshot_scope": clean_string(diagnostics.snapshot_scope).lower() or "full_chain",
        "discovered_expiries": list(diagnostics.discovered_expiries),
        "strike_count_by_expiry": dict(diagnostics.strike_count_by_expiry),
        "attempted_contract_count": int(diagnostics.attempted_contract_count or 0),
        "persisted_quote_count": int(len(option_snapshot.quotes)),
        "coverage_summary": _coverage_summary(pd.DataFrame([quote.to_record() for quote in option_snapshot.quotes])),
        "field_availability": {
            "missing_fields": list(delayed_field_summary.get("missing_fields") or []),
            "missing_field_counts": dict(delayed_field_summary.get("missing_field_counts") or {}),
            "warnings": list(delayed_field_summary.get("warnings") or warnings),
        },
        "files": {
            "underlying": underlying_files,
            "chain": chain_files,
            "option_snapshot": option_snapshot_files,
        },
        "diagnostics": diagnostics.to_dict(),
        "warnings": warnings,
    }
    manifest_path = _metadata_path(ticker, "full_chain_snapshot_manifest.json", data_root)
    write_json(manifest, manifest_path)
    source_notes_path = _write_source_notes(
        ticker,
        _source_notes_payload(
            ticker,
            last_run={
                "type": "full_chain_snapshot",
                "status": "succeeded",
                "market_data_mode": clean_string(market_data_mode).lower(),
                "attempted_contract_count": int(diagnostics.attempted_contract_count or 0),
                "persisted_quote_count": int(len(option_snapshot.quotes)),
                "warnings": warnings,
            },
        ),
        data_root,
    )
    return {
        "manifest_path": str(manifest_path),
        "source_notes_path": source_notes_path,
    }


def load_underlying_snapshots(ticker: str, data_root: str | Path | None = None) -> pd.DataFrame:
    normalized_dir = ensure_ibkr_structure(ticker, data_root)["underlying_normalized"]
    frames: list[pd.DataFrame] = []
    for path in sorted(normalized_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["snapshot_timestamp"] = pd.to_datetime(combined["snapshot_timestamp"], errors="coerce", utc=True)
    combined["bid"] = pd.to_numeric(combined["bid"], errors="coerce")
    combined["ask"] = pd.to_numeric(combined["ask"], errors="coerce")
    combined["last"] = pd.to_numeric(combined["last"], errors="coerce")
    combined["close"] = pd.to_numeric(combined["close"], errors="coerce")
    combined["mid"] = pd.to_numeric(combined["mid"], errors="coerce")
    combined["mark"] = pd.to_numeric(combined["mark"], errors="coerce")
    combined["ticker"] = combined["ticker"].astype(str).str.upper()
    return combined.sort_values("snapshot_timestamp").reset_index(drop=True)


def get_underlying_spot(
    ticker: str,
    snapshot_date: date | str,
    data_root: str | Path | None = None,
    *,
    require_same_day: bool = False,
) -> IbkrSpotMatch:
    requested = parse_date(snapshot_date)
    if requested is None:
        raise ValueError("snapshot_date must be a valid date.")
    history = load_underlying_snapshots(ticker, data_root)
    if history.empty:
        raise FileNotFoundError(f"No local IBKR underlying snapshots exist for {clean_string(ticker).upper()}.")

    requested_ts = pd.Timestamp(requested)
    same_day_rows = history.loc[history["snapshot_timestamp"].dt.date == requested].copy()
    cutoff = datetime.combine(requested, time.max, tzinfo=timezone.utc)
    eligible = same_day_rows if require_same_day else history.loc[history["snapshot_timestamp"] <= cutoff].copy()
    if eligible.empty:
        if require_same_day:
            raise LookupError(
                f"No same-day delayed IBKR underlying snapshot for {clean_string(ticker).upper()} was available on {requested.isoformat()}."
            )
        raise LookupError(
            f"No delayed IBKR underlying snapshot for {clean_string(ticker).upper()} was available on or before {requested.isoformat()}."
        )

    def _first_usable_field(row: pd.Series) -> str | None:
        for column in ["last", "mid", "mark", "close"]:
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.notna(value):
                return column
        return None

    eligible["working_price"] = eligible["last"].fillna(eligible["mid"]).fillna(eligible["mark"]).fillna(eligible["close"])
    eligible["field_used"] = eligible.apply(_first_usable_field, axis=1)
    eligible = eligible.dropna(subset=["working_price"]).sort_values("snapshot_timestamp")
    if eligible.empty:
        if require_same_day:
            raise LookupError(
                f"Same-day delayed IBKR underlying snapshots for {clean_string(ticker).upper()} existed, but none contained a usable price field."
            )
        raise LookupError(
            f"Delayed IBKR underlying snapshots for {clean_string(ticker).upper()} existed, but none contained a usable price field."
        )
    match = eligible.iloc[-1]
    matched_timestamp = pd.Timestamp(match["snapshot_timestamp"]).to_pydatetime().astimezone(timezone.utc)
    matched_date = matched_timestamp.date()
    return IbkrSpotMatch(
        ticker=clean_string(ticker).upper(),
        requested_date=requested,
        matched_timestamp=matched_timestamp,
        matched_date=matched_date,
        close_price=float(match["working_price"]),
        source="ibkr_delayed",
        market_data_mode=str(match["market_data_mode"]),
        field_used=clean_string(match.get("field_used")).lower() or None,
        used_prior_date=matched_date != requested,
    )
