"""Local file-store helpers for historical daily stock prices."""

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
from ..utils import clean_string, ensure_directory, normalize_column_name, parse_date

PRICE_SOURCE_NASDAQ = "nasdaq_historical_quotes"
PRICE_SOURCE_MANUAL = "manual_import"
NORMALIZED_COLUMNS = [
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
    "source",
    "downloaded_at",
]

PRICE_COLUMN_ALIASES = {
    "date": "date",
    "time": "date",
    "trade_date": "date",
    "open": "open",
    "opening_price": "open",
    "high": "high",
    "day_high": "high",
    "low": "low",
    "day_low": "low",
    "close": "close",
    "close_last": "close",
    "close_last_sale": "close",
    "close_last_sale_price": "close",
    "close_last_price": "close",
    "close_last_trade": "close",
    "close_last_trade_price": "close",
    "last": "close",
    "latest": "close",
    "last_sale": "close",
    "volume": "volume",
    "share_volume": "volume",
    "adj_close": "adj_close",
    "adjusted_close": "adj_close",
    "adjusted_close_last": "adj_close",
    "adjusted_close_last_sale": "adj_close",
}


def options_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return options_root() / "data"


def historical_prices_root(
    ticker: str,
    data_root: str | Path | None = None,
) -> Path:
    base = Path(data_root) if data_root is not None else default_data_root()
    return base / clean_string(ticker).upper() / "historical_prices"


def ensure_price_structure(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    root = historical_prices_root(ticker, data_root)
    return {
        "root": ensure_directory(root),
        "raw": ensure_directory(root / "raw"),
        "raw_manual": ensure_directory(root / "raw" / "manual"),
        "normalized": ensure_directory(root / "normalized"),
        "merged": ensure_directory(root / "merged"),
        "metadata": ensure_directory(root / "metadata"),
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def isoformat_utc(value: datetime | None = None) -> str:
    timestamp = value or utc_now()
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp_slug(value: datetime | None = None) -> str:
    timestamp = value or utc_now()
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def raw_response_path(
    ticker: str,
    *,
    downloaded_at: datetime | None = None,
    suffix: str | None = None,
    data_root: str | Path | None = None,
) -> Path:
    root = historical_prices_root(ticker, data_root)
    slug = timestamp_slug(downloaded_at)
    ending = f"_{suffix}" if suffix else ""
    return root / "raw" / f"nasdaq_{clean_string(ticker).lower()}_historical_{slug}{ending}.json"


def normalized_csv_path(ticker: str, data_root: str | Path | None = None) -> Path:
    root = historical_prices_root(ticker, data_root)
    return root / "normalized" / f"{clean_string(ticker).lower()}_daily_prices.csv"


def normalized_parquet_path(ticker: str, data_root: str | Path | None = None) -> Path:
    root = historical_prices_root(ticker, data_root)
    return root / "normalized" / f"{clean_string(ticker).lower()}_daily_prices.parquet"


def merged_csv_path(ticker: str, data_root: str | Path | None = None) -> Path:
    root = historical_prices_root(ticker, data_root)
    return root / "merged" / f"{clean_string(ticker).lower()}_daily_prices_merged.csv"


def merged_parquet_path(ticker: str, data_root: str | Path | None = None) -> Path:
    root = historical_prices_root(ticker, data_root)
    return root / "merged" / f"{clean_string(ticker).lower()}_daily_prices_merged.parquet"


def manifest_json_path(ticker: str, data_root: str | Path | None = None) -> Path:
    root = historical_prices_root(ticker, data_root)
    return root / "metadata" / "download_manifest.json"


def source_notes_json_path(ticker: str, data_root: str | Path | None = None) -> Path:
    root = historical_prices_root(ticker, data_root)
    return root / "metadata" / "source_notes.json"


def empty_price_history(ticker: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["ticker"] = pd.Series(dtype="object")
    frame["source"] = pd.Series(dtype="object")
    return frame


def _parse_price_date(value: Any) -> pd.Timestamp | pd.NaT:
    parsed = parse_date(value)
    if parsed is not None:
        return pd.Timestamp(parsed)
    text = clean_string(value)
    if not text:
        return pd.NaT
    parsed_ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed_ts):
        return pd.NaT
    return pd.Timestamp(parsed_ts).normalize()


def _parse_price_number(value: Any) -> float | None:
    text = clean_string(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"n/a", "na", "null", "none", "--", "-", "."}:
        return None
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    if text.startswith("(") and text.endswith(")"):
        text = f"-{text[1:-1]}"
    try:
        return float(text)
    except ValueError:
        return None


def _parse_volume(value: Any) -> int | None:
    number = _parse_price_number(value)
    if number is None:
        return None
    return int(round(number))


def _rename_columns(raw_frame: pd.DataFrame) -> pd.DataFrame:
    renamed = raw_frame.copy()
    renamed.columns = [
        PRICE_COLUMN_ALIASES.get(normalize_column_name(column), normalize_column_name(column))
        for column in raw_frame.columns
    ]
    for column in NORMALIZED_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = None
    return renamed


def prepare_price_history(
    frame: pd.DataFrame | None,
    ticker: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return empty_price_history(ticker)

    result = frame.copy()
    for column in NORMALIZED_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result["ticker"] = result["ticker"].fillna(clean_string(ticker).upper()).replace("", clean_string(ticker).upper())
    result["date"] = result["date"].apply(_parse_price_date)
    result["downloaded_at"] = pd.to_datetime(result["downloaded_at"], errors="coerce", utc=True)
    for column in ("open", "high", "low", "close", "adj_close"):
        result[column] = result[column].apply(_parse_price_number)
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["volume"] = result["volume"].apply(_parse_volume)
    result["volume"] = pd.array(result["volume"], dtype="Int64")
    result["source"] = result["source"].replace("", pd.NA).fillna(PRICE_SOURCE_NASDAQ)
    result = result.dropna(subset=["date"])
    result["ticker"] = result["ticker"].astype(str).str.upper()
    result = result[NORMALIZED_COLUMNS].sort_values(["date", "downloaded_at"]).drop_duplicates(
        subset=["date"], keep="last"
    )
    return result.reset_index(drop=True)


def normalize_rows(
    rows: list[dict[str, Any]],
    *,
    ticker: str,
    source: str,
    downloaded_at: datetime | None = None,
) -> pd.DataFrame:
    if not rows:
        return empty_price_history(ticker)
    raw_frame = pd.DataFrame(rows)
    renamed = _rename_columns(raw_frame)
    renamed["ticker"] = clean_string(ticker).upper()
    renamed["source"] = source
    renamed["downloaded_at"] = isoformat_utc(downloaded_at)
    return prepare_price_history(renamed, ticker)


def load_price_history(
    ticker: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load local historical prices from CSV storage."""

    merged_path = merged_csv_path(ticker, data_root)
    normalized_path = normalized_csv_path(ticker, data_root)
    if merged_path.exists():
        frame = pd.read_csv(merged_path)
        return prepare_price_history(frame, ticker)
    if normalized_path.exists():
        frame = pd.read_csv(normalized_path)
        return prepare_price_history(frame, ticker)
    return empty_price_history(ticker)


def combine_price_history(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    *,
    ticker: str,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = incoming.copy()
    elif incoming is None or incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    return prepare_price_history(combined, ticker)


def save_price_history(
    ticker: str,
    frame: pd.DataFrame,
    data_root: str | Path | None = None,
) -> dict[str, str]:
    """Persist normalized historical prices to CSV and optional Parquet."""

    ensure_price_structure(ticker, data_root)
    clean_frame = prepare_price_history(frame, ticker)
    normalized_csv = normalized_csv_path(ticker, data_root)
    normalized_parquet = normalized_parquet_path(ticker, data_root)
    merged_csv = merged_csv_path(ticker, data_root)
    merged_parquet = merged_parquet_path(ticker, data_root)

    save_frame = clean_frame.copy()
    save_frame["date"] = pd.to_datetime(save_frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    save_frame["downloaded_at"] = pd.to_datetime(save_frame["downloaded_at"], errors="coerce", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    write_dataframe_csv(save_frame, normalized_csv, index=False)
    normalized_parquet_result = write_dataframe_parquet_if_available(save_frame, normalized_parquet, index=False)
    write_dataframe_csv(save_frame, merged_csv, index=False)
    merged_parquet_result = write_dataframe_parquet_if_available(save_frame, merged_parquet, index=False)
    return {
        "normalized_csv": str(normalized_csv),
        "normalized_parquet": str(normalized_parquet),
        "normalized_parquet_written": str(normalized_parquet_result.written),
        "normalized_parquet_note": normalized_parquet_result.note or "",
        "merged_csv": str(merged_csv),
        "merged_parquet": str(merged_parquet),
        "merged_parquet_written": str(merged_parquet_result.written),
        "merged_parquet_note": merged_parquet_result.note or "",
    }


def write_raw_response(
    ticker: str,
    payload: dict[str, Any],
    *,
    downloaded_at: datetime,
    request_params: dict[str, Any],
    data_root: str | Path | None = None,
    suffix: str | None = None,
) -> Path:
    ensure_price_structure(ticker, data_root)
    path = raw_response_path(
        ticker,
        downloaded_at=downloaded_at,
        suffix=suffix,
        data_root=data_root,
    )
    document = {
        "downloaded_at": isoformat_utc(downloaded_at),
        "ticker": clean_string(ticker).upper(),
        "request_params": request_params,
        "api_response": payload,
    }
    write_json(document, path)
    return path


def write_source_notes(
    ticker: str,
    payload: dict[str, Any],
    data_root: str | Path | None = None,
) -> str:
    ensure_price_structure(ticker, data_root)
    path = source_notes_json_path(ticker, data_root)
    write_json(payload, path)
    return str(path)


def build_manifest(
    *,
    ticker: str,
    data_root: str | Path | None,
    raw_files: list[str],
    request_window: dict[str, Any],
    history: pd.DataFrame,
    saved_files: dict[str, str],
    source_notes_path: str,
    download_mode: str,
) -> dict[str, Any]:
    """Build a manifest describing the local historical-price store state."""

    clean_history = prepare_price_history(history, ticker)
    valid = clean_history.dropna(subset=["close"]).sort_values(["date", "downloaded_at"])
    return {
        "generated_at": isoformat_utc(),
        "ticker": clean_string(ticker).upper(),
        "data_root": str(historical_prices_root(ticker, data_root)),
        "download_mode": download_mode,
        "request_window": request_window,
        "raw_files": raw_files,
        "normalized_files": {
            "csv": saved_files["normalized_csv"],
            "parquet": saved_files["normalized_parquet"],
            "parquet_written": bool(saved_files.get("normalized_parquet_written") == "True"),
            "parquet_note": saved_files.get("normalized_parquet_note") or None,
        },
        "merged_files": {
            "csv": saved_files["merged_csv"],
            "parquet": saved_files["merged_parquet"],
            "parquet_written": bool(saved_files.get("merged_parquet_written") == "True"),
            "parquet_note": saved_files.get("merged_parquet_note") or None,
        },
        "source_notes": source_notes_path,
        "row_count": int(len(clean_history)),
        "non_null_close_row_count": int(clean_history["close"].notna().sum()),
        "min_date": None if clean_history.empty else clean_history["date"].min().date().isoformat(),
        "max_date": None if clean_history.empty else clean_history["date"].max().date().isoformat(),
        "latest_close_date": None if valid.empty else valid.iloc[-1]["date"].date().isoformat(),
    }


def save_manifest(
    ticker: str,
    manifest: dict[str, Any],
    data_root: str | Path | None = None,
) -> str:
    ensure_price_structure(ticker, data_root)
    path = manifest_json_path(ticker, data_root)
    write_json(manifest, path)
    return str(path)


def normalize_manual_price_file(
    path: str | Path,
    ticker: str,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Normalize one manually exported historical-price CSV into the local store."""

    manual_path = Path(path)
    if not manual_path.exists():
        raise FileNotFoundError(f"Manual price file was not found: {manual_path}")
    raw_frame = pd.read_csv(
        manual_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
    )
    if raw_frame.empty:
        raise ValueError(f"Manual price file was empty: {manual_path}")

    downloaded_at = utc_now()
    incoming = normalize_rows(
        raw_frame.to_dict(orient="records"),
        ticker=ticker,
        source=PRICE_SOURCE_MANUAL,
        downloaded_at=downloaded_at,
    )
    if incoming.empty or incoming["close"].notna().sum() == 0:
        raise ValueError(
            f"Manual price file could not be normalized into usable price rows: {manual_path}"
        )

    existing = load_price_history(ticker, data_root)
    combined = combine_price_history(existing, incoming, ticker=ticker)
    save_price_history(ticker, combined, data_root)

    source_notes = {
        "ticker": clean_string(ticker).upper(),
        "primary_source": PRICE_SOURCE_NASDAQ,
        "manual_import_directory": str(ensure_price_structure(ticker, data_root)["raw_manual"]),
        "last_manual_import": {
            "source_file": str(manual_path),
            "imported_at": isoformat_utc(downloaded_at),
            "row_count": int(len(incoming)),
        },
    }
    write_source_notes(ticker, source_notes, data_root)
    return combined
