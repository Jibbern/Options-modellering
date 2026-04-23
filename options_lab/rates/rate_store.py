"""Local file-store helpers for FRED-backed risk-free rates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..persistence import (
    write_dataframe_csv,
    write_dataframe_parquet_if_available,
    write_json,
)
from ..utils import ensure_directory

FRED_SERIES = ("DGS1MO", "DGS3MO", "DGS6MO", "DGS1")
FRED_SOURCE = "FRED"
SERIES_PREFIX = {
    "DGS1MO": "dgs1mo",
    "DGS3MO": "dgs3mo",
    "DGS6MO": "dgs6mo",
    "DGS1": "dgs1",
}
SERIES_DECIMAL_COLUMN = {
    series_id: f"{prefix}_decimal" for series_id, prefix in SERIES_PREFIX.items()
}
SERIES_PERCENT_COLUMN = {
    series_id: f"{prefix}_percent" for series_id, prefix in SERIES_PREFIX.items()
}

NORMALIZED_COLUMNS = [
    "date",
    "series_id",
    "rate_percent",
    "rate_decimal",
    "source",
    "downloaded_at",
    "observation_status",
]


def options_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return options_root() / "data"


def fred_root(data_root: str | Path | None = None) -> Path:
    base = Path(data_root) if data_root is not None else default_data_root()
    return base / "risk_free" / "fred"


def ensure_fred_structure(data_root: str | Path | None = None) -> dict[str, Path]:
    root = fred_root(data_root)
    directories = {
        "root": ensure_directory(root),
        "raw": ensure_directory(root / "raw"),
        "normalized": ensure_directory(root / "normalized"),
        "merged": ensure_directory(root / "merged"),
        "metadata": ensure_directory(root / "metadata"),
    }
    return directories


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def isoformat_utc(value: datetime | None = None) -> str:
    timestamp = value or utc_now()
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_slug(value: datetime | None = None) -> str:
    timestamp = value or utc_now()
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def raw_response_path(
    series_id: str,
    *,
    downloaded_at: datetime | None = None,
    data_root: str | Path | None = None,
) -> Path:
    return fred_root(data_root) / "raw" / f"{series_id.upper()}_{timestamp_slug(downloaded_at)}.json"


def normalized_csv_path(series_id: str, data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "normalized" / f"{series_id.upper()}.csv"


def normalized_parquet_path(series_id: str, data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "normalized" / f"{series_id.upper()}.parquet"


def merged_csv_path(data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "merged" / "fred_treasury_constant_maturity_daily.csv"


def merged_parquet_path(data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "merged" / "fred_treasury_constant_maturity_daily.parquet"


def current_snapshot_csv_path(data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "merged" / "current_risk_free_snapshot.csv"


def manifest_json_path(data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "metadata" / "download_manifest.json"


def latest_rates_json_path(data_root: str | Path | None = None) -> Path:
    return fred_root(data_root) / "metadata" / "latest_rates.json"


def empty_series_history(series_id: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["series_id"] = pd.Series(dtype="object")
    return frame


def _prepare_series_frame(frame: pd.DataFrame, series_id: str | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_series_history(series_id or "")
    result = frame.copy()
    for column in NORMALIZED_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["downloaded_at"] = pd.to_datetime(result["downloaded_at"], errors="coerce", utc=True)
    result["rate_percent"] = pd.to_numeric(result["rate_percent"], errors="coerce")
    result["rate_decimal"] = pd.to_numeric(result["rate_decimal"], errors="coerce")
    result["series_id"] = result["series_id"].fillna(series_id)
    result["source"] = result["source"].fillna(FRED_SOURCE)
    if series_id:
        result["series_id"] = series_id.upper()
    result = result[NORMALIZED_COLUMNS].sort_values(["date", "downloaded_at"]).drop_duplicates(
        subset=["date"], keep="last"
    )
    return result.reset_index(drop=True)


def load_series_history(
    series_id: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load one normalized series history from the local CSV store."""

    path = normalized_csv_path(series_id, data_root)
    if not path.exists():
        return empty_series_history(series_id)
    frame = pd.read_csv(path)
    return _prepare_series_frame(frame, series_id)


def save_series_history(
    series_id: str,
    frame: pd.DataFrame,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    """Persist one normalized rate history to CSV and optional Parquet."""

    ensure_fred_structure(data_root)
    clean_frame = _prepare_series_frame(frame, series_id)
    csv_path = normalized_csv_path(series_id, data_root)
    parquet_path = normalized_parquet_path(series_id, data_root)
    save_frame = clean_frame.copy()
    save_frame["date"] = save_frame["date"].dt.strftime("%Y-%m-%d")
    save_frame["downloaded_at"] = save_frame["downloaded_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    write_dataframe_csv(save_frame, csv_path, index=False)
    parquet_result = write_dataframe_parquet_if_available(save_frame, parquet_path, index=False)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "parquet_written": str(parquet_result.written),
        "parquet_note": parquet_result.note or "",
    }


def write_raw_response(
    series_id: str,
    payload: dict[str, Any],
    *,
    downloaded_at: datetime,
    request_params: dict[str, Any],
    data_root: str | Path | None = None,
) -> Path:
    ensure_fred_structure(data_root)
    path = raw_response_path(series_id, downloaded_at=downloaded_at, data_root=data_root)
    document = {
        "downloaded_at": isoformat_utc(downloaded_at),
        "series_id": series_id.upper(),
        "request_params": request_params,
        "api_response": payload,
    }
    write_json(document, path)
    return path


def combine_series_history(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    series_id: str,
    *,
    replace_existing: bool = False,
) -> pd.DataFrame:
    if replace_existing:
        combined = incoming.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    return _prepare_series_frame(combined, series_id)


def build_merged_table(series_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = pd.DataFrame(columns=["date"])
    for series_id in FRED_SERIES:
        frame = _prepare_series_frame(series_frames.get(series_id, empty_series_history(series_id)), series_id)
        prefix = SERIES_PREFIX[series_id]
        percent_column = f"{prefix}_percent"
        decimal_column = f"{prefix}_decimal"
        downloaded_column = f"{prefix}_downloaded_at"
        if frame.empty:
            subset = pd.DataFrame(columns=["date", percent_column, decimal_column, downloaded_column])
        else:
            subset = frame[["date", "rate_percent", "rate_decimal", "downloaded_at"]].rename(
                columns={
                    "rate_percent": percent_column,
                    "rate_decimal": decimal_column,
                    "downloaded_at": downloaded_column,
                }
            )
        if merged.empty:
            merged = subset.copy()
        else:
            merged = merged.merge(subset, on="date", how="outer")
    if merged.empty:
        merged = pd.DataFrame(columns=["date"])
    download_columns = [f"{SERIES_PREFIX[series_id]}_downloaded_at" for series_id in FRED_SERIES]
    for column in download_columns:
        if column not in merged.columns:
            merged[column] = pd.NaT
        merged[column] = pd.to_datetime(merged[column], errors="coerce", utc=True)
    merged["downloaded_at"] = merged[download_columns].max(axis=1)
    merged = merged.drop(columns=download_columns)
    ordered_columns = ["date"]
    for series_id in FRED_SERIES:
        ordered_columns.append(SERIES_PERCENT_COLUMN[series_id])
        ordered_columns.append(SERIES_DECIMAL_COLUMN[series_id])
        for column in ordered_columns[1:]:
            if column not in merged.columns:
                merged[column] = pd.NA
    ordered_columns.append("downloaded_at")
    merged = merged[ordered_columns].sort_values("date").reset_index(drop=True)
    return merged


def save_merged_table(
    frame: pd.DataFrame,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    """Persist the merged convenience table to CSV and optional Parquet."""

    ensure_fred_structure(data_root)
    merged = frame.copy()
    csv_path = merged_csv_path(data_root)
    parquet_path = merged_parquet_path(data_root)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "downloaded_at" in merged.columns:
        merged["downloaded_at"] = pd.to_datetime(merged["downloaded_at"], errors="coerce", utc=True).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    write_dataframe_csv(merged, csv_path, index=False)
    parquet_result = write_dataframe_parquet_if_available(merged, parquet_path, index=False)
    return {
        "csv": str(csv_path),
        "parquet": str(parquet_path),
        "parquet_written": str(parquet_result.written),
        "parquet_note": parquet_result.note or "",
    }


def load_merged_rates(data_root: str | Path | None = None) -> pd.DataFrame:
    """Load the merged daily rate table from local CSV storage."""

    path = merged_csv_path(data_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Merged FRED rate file not found at {path}. Run the downloader first."
        )
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["downloaded_at"] = pd.to_datetime(frame["downloaded_at"], errors="coerce", utc=True)
    for series_id in FRED_SERIES:
        for column in (SERIES_PERCENT_COLUMN[series_id], SERIES_DECIMAL_COLUMN[series_id]):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                frame[column] = pd.NA
    return frame.sort_values("date").reset_index(drop=True)


def build_latest_snapshot(series_frames: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], pd.DataFrame]:
    generated_at = isoformat_utc()
    payload: dict[str, Any] = {
        "generated_at": generated_at,
        "source": FRED_SOURCE,
        "series": {},
    }
    row: dict[str, Any] = {
        "generated_at": generated_at,
    }
    for series_id in FRED_SERIES:
        prefix = SERIES_PREFIX[series_id]
        frame = _prepare_series_frame(series_frames.get(series_id, empty_series_history(series_id)), series_id)
        valid = frame.dropna(subset=["rate_decimal"]).sort_values(["date", "downloaded_at"])
        if valid.empty:
            payload["series"][series_id] = None
            row[f"{prefix}_date"] = None
            row[f"{prefix}_percent"] = None
            row[f"{prefix}_decimal"] = None
            continue
        latest = valid.iloc[-1]
        payload["series"][series_id] = {
            "series_id": series_id,
            "matched_date": latest["date"].date().isoformat(),
            "rate_percent": None if pd.isna(latest["rate_percent"]) else float(latest["rate_percent"]),
            "rate_decimal": None if pd.isna(latest["rate_decimal"]) else float(latest["rate_decimal"]),
            "downloaded_at": latest["downloaded_at"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": latest["source"],
        }
        row[f"{prefix}_date"] = latest["date"].date().isoformat()
        row[f"{prefix}_percent"] = None if pd.isna(latest["rate_percent"]) else float(latest["rate_percent"])
        row[f"{prefix}_decimal"] = None if pd.isna(latest["rate_decimal"]) else float(latest["rate_decimal"])
    return payload, pd.DataFrame([row])


def save_latest_snapshot_files(
    series_frames: dict[str, pd.DataFrame],
    data_root: str | Path | None = None,
) -> dict[str, str]:
    """Write fast-read files for the latest known rate per configured series."""

    ensure_fred_structure(data_root)
    payload, snapshot_frame = build_latest_snapshot(series_frames)
    latest_json = latest_rates_json_path(data_root)
    snapshot_csv = current_snapshot_csv_path(data_root)
    write_json(payload, latest_json)
    write_dataframe_csv(snapshot_frame, snapshot_csv, index=False)
    return {
        "latest_json": str(latest_json),
        "current_snapshot_csv": str(snapshot_csv),
    }


def get_latest_rates_snapshot(data_root: str | Path | None = None) -> dict[str, Any]:
    """Load the fast-read latest-rates JSON snapshot."""

    path = latest_rates_json_path(data_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Latest-rates snapshot not found at {path}. Run the downloader first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(
    *,
    data_root: str | Path | None,
    raw_paths: dict[str, str],
    series_frames: dict[str, pd.DataFrame],
    series_write_results: dict[str, dict[str, str]],
    merged_frame: pd.DataFrame,
    latest_paths: dict[str, str],
    merged_paths: dict[str, str],
    requested_series: list[str],
    request_window: dict[str, Any],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "generated_at": isoformat_utc(),
        "source": FRED_SOURCE,
        "data_root": str(fred_root(data_root)),
        "requested_series": requested_series,
        "request_window": request_window,
        "raw_files": raw_paths,
        "series": {},
        "merged_files": merged_paths,
        "latest_files": latest_paths,
    }
    for series_id in FRED_SERIES:
        frame = _prepare_series_frame(series_frames.get(series_id, empty_series_history(series_id)), series_id)
        valid = frame.dropna(subset=["rate_decimal"]).sort_values(["date", "downloaded_at"])
        manifest["series"][series_id] = {
            "normalized_csv": str(normalized_csv_path(series_id, data_root)),
            "normalized_parquet": str(normalized_parquet_path(series_id, data_root)),
            "normalized_parquet_written": bool(
                series_write_results.get(series_id, {}).get("parquet_written") == "True"
            ),
            "normalized_parquet_note": series_write_results.get(series_id, {}).get("parquet_note") or None,
            "row_count": int(len(frame)),
            "non_null_row_count": int(frame["rate_decimal"].notna().sum()),
            "min_date": None if frame.empty else frame["date"].min().date().isoformat(),
            "max_date": None if frame.empty else frame["date"].max().date().isoformat(),
            "latest_non_null_date": None if valid.empty else valid.iloc[-1]["date"].date().isoformat(),
            "latest_raw_file": raw_paths.get(series_id),
        }
    manifest["merged"] = {
        "row_count": int(len(merged_frame)),
        "min_date": None if merged_frame.empty else pd.to_datetime(merged_frame["date"]).min().date().isoformat(),
        "max_date": None if merged_frame.empty else pd.to_datetime(merged_frame["date"]).max().date().isoformat(),
        "parquet_written": bool(merged_paths.get("parquet_written") == "True"),
        "parquet_note": merged_paths.get("parquet_note") or None,
    }
    return manifest


def save_manifest(
    manifest: dict[str, Any],
    data_root: str | Path | None = None,
) -> str:
    ensure_fred_structure(data_root)
    path = manifest_json_path(data_root)
    write_json(manifest, path)
    return str(path)
