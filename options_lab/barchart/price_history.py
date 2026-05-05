"""Import manually downloaded Barchart price-history CSV exports."""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..prices.price_store import (
    combine_price_history,
    ensure_price_structure,
    isoformat_utc,
    load_price_history,
    normalize_rows,
    save_manifest,
    save_price_history,
    write_source_notes,
)
from ..persistence import write_json
from ..utils import clean_string

BARCHART_PRICE_HISTORY_SOURCE = "barchart_price_history"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _copy_raw_price_file(source_path: Path, raw_manual_dir: Path) -> Path:
    destination = raw_manual_dir / source_path.name
    if destination.exists():
        try:
            if destination.read_bytes() == source_path.read_bytes():
                return destination
        except OSError:
            pass
        destination = raw_manual_dir / f"{source_path.stem}_{_timestamp_slug()}{source_path.suffix}"
    shutil.copy2(source_path, destination)
    return destination


def _clean_price_rows(raw_frame: pd.DataFrame) -> pd.DataFrame:
    first_col = raw_frame.columns[0]
    first_values = raw_frame[first_col].astype(str).str.strip()
    footer_mask = first_values.str.startswith("Downloaded from Barchart.com", na=False)
    blank_mask = raw_frame.apply(lambda row: all(not clean_string(value) for value in row), axis=1)
    return raw_frame.loc[~footer_mask & ~blank_mask].copy()


def import_barchart_price_history_csv(
    ticker: str,
    csv_path: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy and merge one Barchart price-history CSV into the local price store."""

    source_path = Path(csv_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Barchart price-history CSV was not found: {source_path}")
    structure = ensure_price_structure(ticker, data_root)
    copied_raw = _copy_raw_price_file(source_path, structure["raw_manual"])
    raw_frame = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    clean_frame = _clean_price_rows(raw_frame)
    imported_at = datetime.now(timezone.utc).replace(microsecond=0)
    incoming = normalize_rows(
        clean_frame.to_dict(orient="records"),
        ticker=ticker,
        source=BARCHART_PRICE_HISTORY_SOURCE,
        downloaded_at=imported_at,
    )
    if incoming.empty or incoming["close"].notna().sum() == 0:
        raise ValueError(f"Barchart price-history CSV did not contain usable close rows: {source_path}")

    existing = load_price_history(ticker, data_root)
    combined = combine_price_history(existing, incoming, ticker=ticker)
    saved = save_price_history(ticker, combined, data_root)
    source_notes_path = write_source_notes(
        ticker,
        {
            "ticker": clean_string(ticker).upper(),
            "primary_source": BARCHART_PRICE_HISTORY_SOURCE,
            "manual_import_directory": str(structure["raw_manual"]),
            "last_manual_import": {
                "source_file": str(source_path),
                "raw_copy": str(copied_raw),
                "imported_at": isoformat_utc(imported_at),
                "row_count": int(len(incoming.index)),
            },
        },
        data_root,
    )
    manifest = {
        "generated_at": isoformat_utc(),
        "ticker": clean_string(ticker).upper(),
        "source": BARCHART_PRICE_HISTORY_SOURCE,
        "raw_csv_path": str(copied_raw),
        "rows_raw": int(len(raw_frame.index)),
        "rows_after_footer_cleanup": int(len(clean_frame.index)),
        "rows_imported": int(len(incoming.index)),
        "normalized_files": {"csv": saved["normalized_csv"], "parquet": saved["normalized_parquet"]},
        "merged_files": {"csv": saved["merged_csv"], "parquet": saved["merged_parquet"]},
        "source_notes": source_notes_path,
    }
    manifest_path = structure["metadata"] / f"barchart_price_history_import_{_timestamp_slug()}.json"
    write_json(manifest, manifest_path)
    return {
        "command": "import-barchart-price-history",
        "ticker": clean_string(ticker).upper(),
        "source": BARCHART_PRICE_HISTORY_SOURCE,
        "raw_csv_path": str(copied_raw),
        "normalized_csv": saved["normalized_csv"],
        "merged_csv": saved["merged_csv"],
        "manifest_path": str(manifest_path),
        "row_count": int(len(incoming.index)),
    }
