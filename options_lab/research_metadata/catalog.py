"""Coverage catalog helpers for option chains and research metadata."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..metadata import coerce_metadata_dict, infer_metadata_from_filename, load_sidecar_metadata
from ..persistence import write_json
from ..utils import clean_string, normalize_column_name, parse_number
from .store import (
    DATASET_SPECS,
    catalog_path,
    dataset_manifest_path,
    ensure_ticker_metadata_structure,
    load_dataset_history,
    option_chains_root,
    ticker_root,
)

MIN_USABLE_QUOTE_COVERAGE_PCT = 20.0

_QUOTE_COLUMN_ALIASES = {
    "bid": "bid",
    "mid": "mid",
    "ask": "ask",
    "latest": "last",
    "last": "last",
    "type": "option_type",
    "strike": "strike",
}


@lru_cache(maxsize=512)
def _quote_usability_for_csv(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception:
        return {
            "contract_count": 0,
            "usable_quote_count": 0,
            "usable_quote_coverage_pct": 0.0,
            "quote_usable": False,
        }

    normalized_columns = [
        _QUOTE_COLUMN_ALIASES.get(normalize_column_name(column), normalize_column_name(column))
        for column in frame.columns
    ]
    frame = frame.copy()
    frame.columns = normalized_columns
    for column in ["option_type", "strike", "bid", "mid", "ask", "last"]:
        if column not in frame.columns:
            frame[column] = ""

    option_type = frame["option_type"].astype(str).str.strip().str.lower()
    strike = frame["strike"].map(parse_number)
    contracts = frame.loc[option_type.isin({"call", "put"}) & strike.notna()].copy()
    if contracts.empty:
        return {
            "contract_count": 0,
            "usable_quote_count": 0,
            "usable_quote_coverage_pct": 0.0,
            "quote_usable": False,
        }

    def _has_quote(row: pd.Series) -> bool:
        for field in ["bid", "mid", "ask", "last"]:
            if parse_number(row.get(field)) is not None:
                return True
        return False

    usable_quote_count = int(contracts.apply(_has_quote, axis=1).sum())
    contract_count = int(len(contracts.index))
    coverage_pct = round((usable_quote_count / contract_count) * 100.0, 1) if contract_count else 0.0
    return {
        "contract_count": contract_count,
        "usable_quote_count": usable_quote_count,
        "usable_quote_coverage_pct": coverage_pct,
        "quote_usable": coverage_pct >= MIN_USABLE_QUOTE_COVERAGE_PCT,
    }


def _source_priority(location: str, *, quote_usable: bool) -> int:
    normalized = clean_string(location).lower()
    if normalized == "ibkr_full_quoted_snapshot":
        return 0 if quote_usable else 4
    if normalized == "barchart_options_screener":
        return 1 if quote_usable else 5
    if normalized == "preferred_option_chains":
        return 2
    if normalized == "ibkr_chain_snapshot":
        return 3 if quote_usable else 6
    if normalized == "legacy_ticker_root":
        return 7
    if normalized == "ibkr_chain_universe":
        return 8
    return 9


def discover_chain_snapshots(
    ticker: str,
    data_root: str | Path | None = None,
    *,
    dedupe: bool = True,
) -> list[dict[str, Any]]:
    """Discover option-chain CSV snapshots in preferred and legacy ticker locations."""

    clean_ticker = clean_string(ticker).upper()
    preferred_dir = option_chains_root(clean_ticker, data_root)
    barchart_options_dir = ticker_root(clean_ticker, data_root) / "options" / "barchart" / "normalized"
    ibkr_dir = ticker_root(clean_ticker, data_root) / "ibkr" / "chains" / "normalized"
    ibkr_option_snapshot_dir = ticker_root(clean_ticker, data_root) / "ibkr" / "snapshots" / "option_quotes" / "normalized"
    legacy_dir = ticker_root(clean_ticker, data_root)
    candidates: list[tuple[Path, str, dict[str, Any]]] = []
    if preferred_dir.exists():
        for path in preferred_dir.glob("*.csv"):
            candidates.append((path, "preferred_option_chains", {}))
    if barchart_options_dir.exists():
        for path in barchart_options_dir.glob("*.csv"):
            sidecar = coerce_metadata_dict(load_sidecar_metadata(path))
            if sidecar.get("snapshot_date") and sidecar.get("expiry_date"):
                candidates.append((path, "barchart_options_screener", sidecar))
    if ibkr_dir.exists():
        for path in ibkr_dir.glob("*.csv"):
            sidecar = coerce_metadata_dict(load_sidecar_metadata(path))
            if sidecar.get("snapshot_date") and sidecar.get("expiry_date"):
                scope = clean_string(sidecar.get("snapshot_scope")).lower()
                location = clean_string(sidecar.get("storage_location")).lower() or (
                    "ibkr_chain_universe" if scope == "chain_universe" else "ibkr_chain_snapshot"
                )
                candidates.append((path, location, sidecar))
    if ibkr_option_snapshot_dir.exists():
        for path in ibkr_option_snapshot_dir.glob("*.csv"):
            sidecar = coerce_metadata_dict(load_sidecar_metadata(path))
            if sidecar.get("snapshot_date") and sidecar.get("expiry_date"):
                scope = clean_string(sidecar.get("snapshot_scope")).lower()
                location = clean_string(sidecar.get("storage_location")).lower() or (
                    "ibkr_full_quoted_snapshot" if scope == "full_chain" else "ibkr_chain_snapshot"
                )
                candidates.append((path, location, sidecar))
    if legacy_dir.exists():
        for path in legacy_dir.glob("*.csv"):
            candidates.append((path, "legacy_ticker_root", {}))

    discovered: list[dict[str, Any]] = []
    for path, location, preloaded_sidecar in candidates:
        inferred = coerce_metadata_dict(infer_metadata_from_filename(path))
        sidecar = preloaded_sidecar or coerce_metadata_dict(load_sidecar_metadata(path))
        snapshot_date = sidecar.get("snapshot_date") or inferred.get("snapshot_date")
        expiry_date = sidecar.get("expiry_date") or inferred.get("expiry_date")
        quote_summary = _quote_usability_for_csv(str(path))
        priority = _source_priority(location, quote_usable=bool(quote_summary["quote_usable"]))
        discovered.append(
            {
                "file_path": str(path),
                "storage_location": location,
                "snapshot_scope": clean_string(sidecar.get("snapshot_scope")).lower() or None,
                "snapshot_date": snapshot_date.isoformat() if snapshot_date else None,
                "expiry_date": expiry_date.isoformat() if expiry_date else None,
                "has_sidecar_metadata": Path(path).with_suffix(".metadata.json").exists(),
                "contract_count": int(quote_summary["contract_count"]),
                "usable_quote_count": int(quote_summary["usable_quote_count"]),
                "usable_quote_coverage_pct": float(quote_summary["usable_quote_coverage_pct"]),
                "quote_usable": bool(quote_summary["quote_usable"]),
                "quote_usability_gate_pct": MIN_USABLE_QUOTE_COVERAGE_PCT,
                "_priority": priority,
            }
        )

    if dedupe:
        deduped: dict[tuple[str | None, str | None, str], dict[str, Any]] = {}
        for item in sorted(discovered, key=lambda entry: (entry["_priority"], entry["file_path"])):
            key = (item.get("snapshot_date"), item.get("expiry_date"), clean_ticker)
            if key[0] is None and key[1] is None:
                key = (None, None, Path(item["file_path"]).name.lower())
            existing = deduped.get(key)
            if existing is None or item["_priority"] < existing["_priority"]:
                deduped[key] = item
        discovered = list(deduped.values())

    snapshots = []
    for item in discovered:
        payload = dict(item)
        payload.pop("_priority", None)
        snapshots.append(payload)
    snapshots.sort(key=lambda item: ((item.get("snapshot_date") or ""), (item.get("expiry_date") or ""), item["file_path"]))
    return snapshots


def _dataset_coverage(dataset: str, ticker: str, data_root: str | Path | None = None) -> dict[str, Any]:
    history = load_dataset_history(dataset, ticker, data_root)
    spec = DATASET_SPECS[dataset]
    primary_date = spec["primary_date"]
    coverage_dates = [
        pd.Timestamp(value).date().isoformat()
        for value in history[primary_date].dropna().sort_values().unique().tolist()
    ] if primary_date in history.columns else []
    sources = sorted({str(value) for value in history["source"].dropna().tolist()}) if "source" in history.columns else []
    payload: dict[str, Any] = {
        "dataset": dataset,
        "row_count": int(len(history)),
        "coverage_dates": coverage_dates,
        "min_date": coverage_dates[0] if coverage_dates else None,
        "max_date": coverage_dates[-1] if coverage_dates else None,
        "sources": sources,
        "manifest_path": str(dataset_manifest_path(ticker, dataset, data_root)),
    }
    if dataset == "expected_move" and not history.empty:
        payload["expiry_dates"] = [
            pd.Timestamp(value).date().isoformat()
            for value in history["expiry_date"].dropna().sort_values().unique().tolist()
        ]
    if dataset == "events" and not history.empty:
        payload["event_types"] = sorted({str(value) for value in history["event_type"].dropna().tolist()})
    return payload


def build_ticker_catalog(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build a ticker-level metadata catalog with coverage across datasets."""

    clean_ticker = clean_string(ticker).upper()
    chain_snapshots = discover_chain_snapshots(clean_ticker, data_root)
    datasets = {
        dataset: _dataset_coverage(dataset, clean_ticker, data_root)
        for dataset in DATASET_SPECS
    }
    return {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ticker": clean_ticker,
        "chain_snapshots": chain_snapshots,
        "chain_snapshot_count": len(chain_snapshots),
        "datasets": datasets,
    }


def update_ticker_catalog(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild and persist the ticker-level research metadata catalog."""

    ensure_ticker_metadata_structure(ticker, data_root)
    catalog = build_ticker_catalog(ticker, data_root)
    write_json(catalog, catalog_path(ticker, data_root))
    return catalog


def load_ticker_catalog(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load the persisted ticker catalog, rebuilding it if missing."""

    path = catalog_path(ticker, data_root)
    if not path.exists():
        return update_ticker_catalog(ticker, data_root)
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_summary(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a compact coverage summary suitable for embedding in report metadata."""

    catalog = load_ticker_catalog(ticker, data_root)
    return {
        "chain_snapshot_count": catalog.get("chain_snapshot_count", 0),
        "datasets": {
            dataset: {
                "row_count": details.get("row_count", 0),
                "min_date": details.get("min_date"),
                "max_date": details.get("max_date"),
            }
            for dataset, details in catalog.get("datasets", {}).items()
        },
    }
