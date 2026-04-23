"""Shared local-store helpers for research-oriented options metadata datasets."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..persistence import (
    make_json_safe,
    write_dataframe_csv,
    write_dataframe_parquet_if_available,
    write_json,
)
from ..utils import clean_string, ensure_directory, normalize_column_name, parse_date, parse_number

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "expected_move": {
        "columns": [
            "ticker",
            "snapshot_date",
            "expiry_date",
            "expected_move_abs",
            "expected_move_pct",
            "lower_bound",
            "upper_bound",
            "implied_volatility",
            "source",
            "source_url",
            "acquisition_method",
            "notes",
            "registered_at",
        ],
        "date_columns": ["snapshot_date", "expiry_date"],
        "numeric_columns": [
            "expected_move_abs",
            "expected_move_pct",
            "lower_bound",
            "upper_bound",
            "implied_volatility",
        ],
        "required_columns": ["snapshot_date", "expiry_date"],
        "dedupe_keys": ["snapshot_date", "expiry_date", "source"],
        "primary_date": "snapshot_date",
        "manifest_name": "expected_move_manifest.json",
    },
    "options_overview": {
        "columns": [
            "ticker",
            "snapshot_date",
            "implied_volatility",
            "historic_volatility",
            "iv_rank",
            "iv_percentile",
            "iv_hv_ratio",
            "put_call_volume_ratio",
            "put_call_open_interest_ratio",
            "total_call_volume",
            "total_put_volume",
            "total_call_open_interest",
            "total_put_open_interest",
            "earnings_date",
            "source",
            "source_url",
            "acquisition_method",
            "notes",
            "registered_at",
        ],
        "date_columns": ["snapshot_date", "earnings_date"],
        "numeric_columns": [
            "implied_volatility",
            "historic_volatility",
            "iv_rank",
            "iv_percentile",
            "iv_hv_ratio",
            "put_call_volume_ratio",
            "put_call_open_interest_ratio",
            "total_call_volume",
            "total_put_volume",
            "total_call_open_interest",
            "total_put_open_interest",
        ],
        "required_columns": ["snapshot_date"],
        "dedupe_keys": ["snapshot_date", "source"],
        "primary_date": "snapshot_date",
        "manifest_name": "options_overview_manifest.json",
    },
    "events": {
        "columns": [
            "ticker",
            "event_date",
            "event_time",
            "event_type",
            "source",
            "source_url",
            "acquisition_method",
            "notes",
            "registered_at",
        ],
        "date_columns": ["event_date"],
        "numeric_columns": [],
        "required_columns": ["event_date", "event_type"],
        "dedupe_keys": ["event_date", "event_type", "source"],
        "primary_date": "event_date",
        "manifest_name": "events_manifest.json",
    },
    "dividends": {
        "columns": [
            "ticker",
            "snapshot_date",
            "dividend_yield",
            "expected_dividend_date",
            "source",
            "source_url",
            "acquisition_method",
            "notes",
            "registered_at",
        ],
        "date_columns": ["snapshot_date", "expected_dividend_date"],
        "numeric_columns": ["dividend_yield"],
        "required_columns": ["snapshot_date"],
        "dedupe_keys": ["snapshot_date", "source"],
        "primary_date": "snapshot_date",
        "manifest_name": "dividends_manifest.json",
    },
    "notes": {
        "columns": [
            "ticker",
            "note_date",
            "category",
            "title",
            "body",
            "source",
            "source_url",
            "acquisition_method",
            "notes",
            "registered_at",
        ],
        "date_columns": ["note_date"],
        "numeric_columns": [],
        "required_columns": ["note_date", "title"],
        "dedupe_keys": ["note_date", "title", "source"],
        "primary_date": "note_date",
        "manifest_name": "notes_manifest.json",
    },
}


def options_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return options_root() / "data"


def ticker_root(ticker: str, data_root: str | Path | None = None) -> Path:
    base = Path(data_root) if data_root is not None else default_data_root()
    return base / clean_string(ticker).upper()


def option_chains_root(ticker: str, data_root: str | Path | None = None) -> Path:
    return ticker_root(ticker, data_root) / "option_chains"


def options_metadata_root(ticker: str, data_root: str | Path | None = None) -> Path:
    return ticker_root(ticker, data_root) / "options_metadata"


def dataset_root(
    ticker: str,
    dataset: str,
    data_root: str | Path | None = None,
) -> Path:
    return options_metadata_root(ticker, data_root) / dataset


def ensure_dataset_structure(
    ticker: str,
    dataset: str,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Create the standard raw/normalized/metadata folder set for one dataset."""

    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported research metadata dataset: {dataset}")
    base = dataset_root(ticker, dataset, data_root)
    return {
        "root": ensure_directory(base),
        "raw": ensure_directory(base / "raw"),
        "normalized": ensure_directory(base / "normalized"),
        "metadata": ensure_directory(base / "metadata"),
    }


def ensure_ticker_metadata_structure(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Create the preferred top-level metadata structure for a ticker."""

    ticker_dir = ensure_directory(ticker_root(ticker, data_root))
    chains_dir = ensure_directory(option_chains_root(ticker, data_root))
    metadata_dir = ensure_directory(options_metadata_root(ticker, data_root))
    result = {
        "ticker_root": ticker_dir,
        "option_chains": chains_dir,
        "options_metadata": metadata_dir,
    }
    for dataset in DATASET_SPECS:
        result[dataset] = ensure_dataset_structure(ticker, dataset, data_root)["root"]
    return result


def normalized_csv_path(
    ticker: str,
    dataset: str,
    data_root: str | Path | None = None,
) -> Path:
    return dataset_root(ticker, dataset, data_root) / "normalized" / f"{dataset}.csv"


def normalized_parquet_path(
    ticker: str,
    dataset: str,
    data_root: str | Path | None = None,
) -> Path:
    return dataset_root(ticker, dataset, data_root) / "normalized" / f"{dataset}.parquet"


def dataset_manifest_path(
    ticker: str,
    dataset: str,
    data_root: str | Path | None = None,
) -> Path:
    manifest_name = DATASET_SPECS[dataset]["manifest_name"]
    return dataset_root(ticker, dataset, data_root) / "metadata" / manifest_name


def catalog_path(ticker: str, data_root: str | Path | None = None) -> Path:
    return options_metadata_root(ticker, data_root) / "catalog.json"


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def isoformat_utc(value: datetime | None = None) -> str:
    timestamp = value or utc_now()
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_slug(value: datetime | None = None) -> str:
    timestamp = value or utc_now()
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _normalize_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "rows", "items", "data"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        return [payload]
    raise ValueError("JSON metadata input must be an object or a list of objects.")


def load_input_records(path: str | Path) -> list[dict[str, Any]]:
    """Load CSV or JSON metadata input as a list of records."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Metadata input file not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(
            input_path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
        return frame.to_dict(orient="records")
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return _normalize_json_payload(payload)
    raise ValueError(f"Unsupported metadata input type: {input_path.suffix}")


def _canonicalize_records(
    records: list[dict[str, Any]],
    *,
    column_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    aliases = column_aliases or {}
    frame.columns = [
        aliases.get(normalize_column_name(column), normalize_column_name(column))
        for column in frame.columns
    ]
    return frame


def _prepare_dataset_frame(
    dataset: str,
    frame: pd.DataFrame | None,
    *,
    ticker: str,
    default_values: dict[str, Any] | None = None,
) -> pd.DataFrame:
    spec = DATASET_SPECS[dataset]
    columns = spec["columns"]
    clean_ticker = clean_string(ticker).upper()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)

    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            result[column] = None

    defaults = default_values or {}
    for key, value in defaults.items():
        if key in result.columns:
            result[key] = result[key].replace("", pd.NA).fillna(value)

    result["ticker"] = result["ticker"].replace("", pd.NA).fillna(clean_ticker)
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result["source"] = result["source"].replace("", pd.NA).fillna("manual_input")
    result["source_url"] = result["source_url"].replace("", pd.NA)
    result["acquisition_method"] = result["acquisition_method"].replace("", pd.NA).fillna("manual")
    result["notes"] = result["notes"].replace("", pd.NA)
    result["registered_at"] = result["registered_at"].replace("", pd.NA).fillna(isoformat_utc())

    for column in spec["date_columns"]:
        result[column] = result[column].apply(parse_date)
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.normalize()
    for column in spec["numeric_columns"]:
        result[column] = result[column].apply(parse_number)
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["registered_at"] = pd.to_datetime(result["registered_at"], errors="coerce", utc=True)

    for column in columns:
        if column in spec["date_columns"] or column == "registered_at" or column in spec["numeric_columns"]:
            continue
        result[column] = result[column].map(clean_string)
        result[column] = result[column].replace("", pd.NA)

    for required in spec["required_columns"]:
        result = result[result[required].notna()]

    ordered = result[columns].sort_values(spec["dedupe_keys"] + ["registered_at"]).drop_duplicates(
        subset=spec["dedupe_keys"],
        keep="last",
    )
    return ordered.reset_index(drop=True)


def load_dataset_history(
    dataset: str,
    ticker: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load one normalized dataset history from the local CSV baseline."""

    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported research metadata dataset: {dataset}")
    path = normalized_csv_path(ticker, dataset, data_root)
    if not path.exists():
        return pd.DataFrame(columns=DATASET_SPECS[dataset]["columns"])
    frame = pd.read_csv(path)
    return _prepare_dataset_frame(dataset, frame, ticker=ticker)


def combine_dataset_history(
    dataset: str,
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame | None,
    *,
    ticker: str,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = incoming.copy() if incoming is not None else pd.DataFrame()
    elif incoming is None or incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    return _prepare_dataset_frame(dataset, combined, ticker=ticker)


def save_dataset_history(
    dataset: str,
    ticker: str,
    frame: pd.DataFrame,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    """Persist one normalized dataset history to CSV and optional parquet."""

    ensure_dataset_structure(ticker, dataset, data_root)
    clean_frame = _prepare_dataset_frame(dataset, frame, ticker=ticker)
    csv_path = normalized_csv_path(ticker, dataset, data_root)
    parquet_path = normalized_parquet_path(ticker, dataset, data_root)
    save_frame = clean_frame.copy()
    spec = DATASET_SPECS[dataset]
    for column in spec["date_columns"]:
        if column in save_frame.columns:
            save_frame[column] = pd.to_datetime(save_frame[column], errors="coerce").dt.strftime("%Y-%m-%d")
    save_frame["registered_at"] = pd.to_datetime(save_frame["registered_at"], errors="coerce", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    write_dataframe_csv(save_frame, csv_path, index=False)
    parquet_result = write_dataframe_parquet_if_available(save_frame, parquet_path, index=False)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "parquet_written": str(parquet_result.written),
        "parquet_note": parquet_result.note or "",
    }


def copy_raw_input(
    dataset: str,
    ticker: str,
    source_path: str | Path,
    data_root: str | Path | None = None,
) -> Path:
    """Copy the original metadata input file into the dataset raw folder."""

    structure = ensure_dataset_structure(ticker, dataset, data_root)
    input_path = Path(source_path)
    destination = structure["raw"] / input_path.name
    if destination.exists():
        destination = structure["raw"] / f"{input_path.stem}_{timestamp_slug()}{input_path.suffix}"
    shutil.copy2(input_path, destination)
    return destination


def build_dataset_manifest(
    dataset: str,
    ticker: str,
    history: pd.DataFrame,
    *,
    raw_file: str,
    saved_files: dict[str, str],
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact manifest for one normalized metadata dataset."""

    spec = DATASET_SPECS[dataset]
    primary_date = spec["primary_date"]
    clean_history = _prepare_dataset_frame(dataset, history, ticker=ticker)
    source_values = sorted({str(value) for value in clean_history["source"].dropna().tolist()})
    coverage_dates = []
    if primary_date in clean_history.columns:
        coverage_dates = [
            pd.Timestamp(value).date().isoformat()
            for value in clean_history[primary_date].dropna().sort_values().unique().tolist()
        ]
    manifest = {
        "generated_at": isoformat_utc(),
        "ticker": clean_string(ticker).upper(),
        "dataset": dataset,
        "data_root": str(dataset_root(ticker, dataset, data_root)),
        "raw_file": raw_file,
        "normalized_files": {
            "csv": saved_files["csv"],
            "parquet": saved_files["parquet"],
            "parquet_written": bool(saved_files["parquet_written"] == "True"),
            "parquet_note": saved_files["parquet_note"] or None,
        },
        "row_count": int(len(clean_history)),
        "sources": source_values,
        "coverage_dates": coverage_dates,
        "min_date": coverage_dates[0] if coverage_dates else None,
        "max_date": coverage_dates[-1] if coverage_dates else None,
    }
    if dataset == "expected_move":
        manifest["expiry_dates"] = [
            pd.Timestamp(value).date().isoformat()
            for value in clean_history["expiry_date"].dropna().sort_values().unique().tolist()
        ]
    if dataset == "events":
        manifest["event_types"] = sorted({str(value) for value in clean_history["event_type"].dropna().tolist()})
    if dataset == "notes":
        manifest["categories"] = sorted({str(value) for value in clean_history["category"].dropna().tolist()})
    return manifest


def save_dataset_manifest(
    dataset: str,
    ticker: str,
    manifest: dict[str, Any],
    data_root: str | Path | None = None,
) -> Path:
    ensure_dataset_structure(ticker, dataset, data_root)
    return write_json(manifest, dataset_manifest_path(ticker, dataset, data_root))


def register_dataset_file(
    dataset: str,
    ticker: str,
    source_path: str | Path,
    *,
    column_aliases: dict[str, str] | None = None,
    default_values: dict[str, Any] | None = None,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Register one local CSV/JSON metadata file into the normalized store."""

    ensure_ticker_metadata_structure(ticker, data_root)
    raw_copy = copy_raw_input(dataset, ticker, source_path, data_root)
    records = load_input_records(source_path)
    normalized_input = _canonicalize_records(records, column_aliases=column_aliases)
    incoming = _prepare_dataset_frame(
        dataset,
        normalized_input,
        ticker=ticker,
        default_values=default_values,
    )
    existing = load_dataset_history(dataset, ticker, data_root)
    combined = combine_dataset_history(dataset, existing, incoming, ticker=ticker)
    saved_files = save_dataset_history(dataset, ticker, combined, data_root)
    manifest = build_dataset_manifest(
        dataset,
        ticker,
        combined,
        raw_file=str(raw_copy),
        saved_files=saved_files,
        data_root=data_root,
    )
    manifest_path_value = save_dataset_manifest(dataset, ticker, manifest, data_root)
    from .catalog import update_ticker_catalog

    catalog = update_ticker_catalog(ticker, data_root)
    return {
        "dataset": dataset,
        "ticker": clean_string(ticker).upper(),
        "raw_file": str(raw_copy),
        "manifest_path": str(manifest_path_value),
        "row_count": manifest["row_count"],
        "catalog_path": str(catalog_path(ticker, data_root)),
        "catalog": catalog,
    }


def empty_dataset_payload(dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "matched": False,
    }


def row_to_payload(dataset: str, row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    spec = DATASET_SPECS[dataset]
    for column in spec["date_columns"]:
        value = payload.get(column)
        if value is None or pd.isna(value):
            payload[column] = None
        elif not isinstance(value, str):
            payload[column] = pd.Timestamp(value).date().isoformat()
    registered_at = payload.get("registered_at")
    if registered_at is not None and not isinstance(registered_at, str) and not pd.isna(registered_at):
        payload["registered_at"] = pd.Timestamp(registered_at).strftime("%Y-%m-%dT%H:%M:%SZ")
    return make_json_safe(payload)
