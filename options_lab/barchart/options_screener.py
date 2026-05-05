"""Import manually downloaded Barchart Options Screener CSV exports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ..persistence import write_dataframe_csv, write_json
from ..utils import clean_string, ensure_directory, normalize_column_name, parse_date, parse_int, parse_number

BARCHART_OPTIONS_SOURCE = "barchart_options_screener"
BARCHART_OPTIONS_TRUST = "manually_downloaded_barchart"

EXPECTED_FIELDS = {
    "symbol",
    "type",
    "price",
    "latest",
    "exp_date",
    "dte",
    "strike",
    "moneyness",
    "bid",
    "ask",
    "mid",
    "volume",
    "open_int",
    "iv",
    "iv_rank",
    "iv_pctl",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
    "itm_prob",
    "otm_prob",
    "profit_prob",
}


@dataclass(frozen=True)
class BarchartOptionsImportResult:
    """Files and counts produced by one Barchart options import."""

    ticker: str
    source: str
    trust_level: str
    snapshot_date: str
    raw_csv_path: str
    normalized_output_paths: list[str]
    manifest_path: str
    rows_raw: int
    rows_after_footer_cleanup: int
    rows_for_ticker: int
    rows_model_eligible: int


def barchart_options_root(ticker: str, data_root: str | Path | None = None) -> Path:
    base = Path(data_root) if data_root is not None else Path(__file__).resolve().parents[2] / "data"
    return base / clean_string(ticker).upper() / "options" / "barchart"


def ensure_barchart_options_structure(ticker: str, data_root: str | Path | None = None) -> dict[str, Path]:
    root = barchart_options_root(ticker, data_root)
    return {
        "root": ensure_directory(root),
        "raw": ensure_directory(root / "raw"),
        "normalized": ensure_directory(root / "normalized"),
        "manifests": ensure_directory(root / "manifests"),
    }


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")


def _safe_copy_raw(source_path: Path, destination_dir: Path) -> Path:
    destination = destination_dir / source_path.name
    if destination.exists():
        try:
            if source_path.read_bytes() == destination.read_bytes():
                return destination
        except OSError:
            pass
        destination = destination_dir / f"{source_path.stem}_{_timestamp_slug()}{source_path.suffix}"
    shutil.copy2(source_path, destination)
    return destination


def _column_alias(name: str) -> str:
    normalized = normalize_column_name(name)
    aliases = {
        "price": "underlying_price",
        "price_": "underlying_price",
        "latest": "last",
        "exp_date": "expiration",
        "open_int": "open_interest",
        "iv": "implied_volatility",
        "iv_pctl": "iv_percentile",
        "itm_prob": "itm_probability",
        "otm_prob": "otm_probability",
        "profit_prob": "profit_probability",
        "be_bid": "barchart_be_bid",
        "be_ask": "barchart_be_ask",
        "be_mid": "barchart_be_mid",
    }
    return aliases.get(normalized, normalized)


def _downloaded_at_from_footer(raw_frame: pd.DataFrame) -> str | None:
    pattern = re.compile(
        r"Downloaded from Barchart\.com as of (?P<date>\d{2}-\d{2}-\d{4}) (?P<time>\d{1,2}:\d{2})(?P<ampm>am|pm) (?P<tz>[A-Z]+)",
        re.IGNORECASE,
    )
    for row in raw_frame.astype(str).to_dict(orient="records"):
        text = " ".join(clean_string(value) for value in row.values() if clean_string(value))
        match = pattern.search(text)
        if not match:
            continue
        parsed = datetime.strptime(
            f"{match.group('date')} {match.group('time')}{match.group('ampm').lower()}",
            "%m-%d-%Y %I:%M%p",
        )
        return f"{parsed.strftime('%Y-%m-%dT%H:%M:%S')} {match.group('tz').upper()}"
    return None


def _clean_contract_rows(raw_frame: pd.DataFrame) -> pd.DataFrame:
    if raw_frame.empty:
        return raw_frame.copy()
    first_col = raw_frame.columns[0]
    first_values = raw_frame[first_col].astype(str).str.strip()
    footer_mask = first_values.str.startswith("Downloaded from Barchart.com", na=False)
    blank_mask = raw_frame.apply(lambda row: all(not clean_string(value) for value in row), axis=1)
    return raw_frame.loc[~footer_mask & ~blank_mask].copy()


def _parse_decimal(value: Any) -> float | None:
    return parse_number(value)


def _parse_percent(value: Any) -> float | None:
    return parse_number(value, percent=True)


def _contract_label(strike: float | None, option_type: str, expiration: str | None) -> str:
    expiry = parse_date(expiration)
    expiry_label = expiry.strftime("%b-%y") if expiry else "Unknown"
    strike_label = f"{float(strike):g}" if strike is not None else "?"
    type_label = "C" if clean_string(option_type).lower() == "call" else "P"
    return f"{strike_label}{type_label} {expiry_label}"


def _moneyness_bucket(moneyness: float | None) -> str:
    if moneyness is None:
        return "unknown"
    distance = abs(float(moneyness))
    if distance >= 0.30:
        return "deep_itm" if moneyness > 0 else "deep_otm"
    if distance <= 0.05:
        return "atm"
    return "itm" if moneyness > 0 else "otm"


def _quality_flags(row: dict[str, Any], *, snapshot_date: str) -> list[str]:
    flags: list[str] = []
    bid = row.get("bid")
    ask = row.get("ask")
    mid = row.get("mid")
    last = row.get("last")
    iv = row.get("implied_volatility")
    volume = row.get("volume")
    oi = row.get("open_interest")
    spread_pct = row.get("spread_pct_of_mid")
    if bid is None:
        flags.append("missing_bid")
    if ask is None:
        flags.append("missing_ask")
    if mid is None:
        flags.append("missing_mid")
    if iv is None or float(iv) <= 0:
        flags.append("invalid_iv")
    if volume is not None and int(volume) == 0:
        flags.append("zero_volume")
    if oi is not None and int(oi) == 0:
        flags.append("zero_open_interest")
    if last is None:
        flags.append("stale_last")
    elif float(last) == 0:
        flags.extend(["latest_zero", "stale_last"])
    if spread_pct is not None:
        if float(spread_pct) > 0.50:
            flags.append("very_wide_spread")
        elif float(spread_pct) > 0.25:
            flags.append("wide_spread")
    if iv is not None and float(iv) > 2.0:
        flags.append("high_iv")
    expiration = parse_date(row.get("expiration"))
    dte = row.get("dte")
    snap = parse_date(snapshot_date)
    if expiration is not None and snap is not None and dte is not None:
        actual_dte = max((expiration - snap).days, 0)
        if abs(actual_dte - int(dte)) > 1:
            flags.append("fallback_dte_mismatch")
    if row.get("strike") is None or row.get("underlying_price") is None:
        flags.append("suspicious_row")
    return list(dict.fromkeys(flags))


def _liquidity_bucket(row: dict[str, Any]) -> str:
    flags = set(clean_string(row.get("quality_flags")).split(";")) if clean_string(row.get("quality_flags")) else set()
    spread_pct = row.get("spread_pct_of_mid")
    oi = int(row.get("open_interest") or 0)
    volume = int(row.get("volume") or 0)
    bid = _parse_decimal(row.get("bid"))
    ask = _parse_decimal(row.get("ask"))
    if {"missing_bid", "missing_ask", "missing_mid", "invalid_iv"} & flags:
        return "stale_or_wide"
    if ask is None or float(ask) <= 0 or bid is None or float(bid) < 0:
        return "stale_or_wide"
    if spread_pct is None or float(spread_pct) > 0.40 or oi < 5:
        return "stale_or_wide"
    if float(spread_pct) <= 0.10 and oi >= 100 and volume >= 10:
        return "liquid"
    if float(spread_pct) <= 0.20 and oi >= 25:
        return "usable"
    if float(spread_pct) <= 0.40 or oi >= 5:
        return "thin"
    return "stale_or_wide"


def _normalize_options_frame(
    raw_frame: pd.DataFrame,
    *,
    ticker: str,
    snapshot_date: str,
    downloaded_at: str | None,
    source: str,
    trust_level: str,
    include_puts: bool,
    min_ask: float,
    min_iv: float,
    min_dte: int,
    max_dte: int,
    min_open_interest: int | None,
    allow_zero_volume: bool,
    entry_mode: str,
) -> pd.DataFrame:
    clean_ticker = clean_string(ticker).upper()
    frame = raw_frame.copy()
    frame.columns = [_column_alias(column) for column in frame.columns]
    for column in [
        "symbol",
        "type",
        "underlying_price",
        "last",
        "expiration",
        "dte",
        "strike",
        "moneyness",
        "bid",
        "ask",
        "mid",
        "volume",
        "open_interest",
        "implied_volatility",
        "iv_rank",
        "iv_percentile",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "itm_probability",
        "otm_probability",
        "profit_probability",
        "barchart_be_bid",
        "barchart_be_ask",
        "barchart_be_mid",
        "links",
    ]:
        if column not in frame.columns:
            frame[column] = None

    rows: list[dict[str, Any]] = []
    for _, raw_row in frame.iterrows():
        symbol = clean_string(raw_row.get("symbol")).upper()
        option_type = clean_string(raw_row.get("type")).lower()
        if option_type in {"c", "calls"}:
            option_type = "call"
        if option_type in {"p", "puts"}:
            option_type = "put"
        if symbol != clean_ticker or option_type not in {"call", "put"}:
            continue
        if option_type == "put" and not include_puts:
            continue

        expiration_date = parse_date(raw_row.get("expiration"))
        strike = _parse_decimal(raw_row.get("strike"))
        underlying = _parse_decimal(raw_row.get("underlying_price"))
        bid = _parse_decimal(raw_row.get("bid"))
        ask = _parse_decimal(raw_row.get("ask"))
        mid = _parse_decimal(raw_row.get("mid"))
        if mid is None and bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2.0
        last = _parse_decimal(raw_row.get("last"))
        volume = parse_int(raw_row.get("volume"))
        open_interest = parse_int(raw_row.get("open_interest"))
        iv = _parse_percent(raw_row.get("implied_volatility"))
        moneyness = _parse_percent(raw_row.get("moneyness"))
        dte = parse_int(raw_row.get("dte"))
        if dte is None and expiration_date is not None and parse_date(snapshot_date) is not None:
            dte = max((expiration_date - parse_date(snapshot_date)).days, 0)
        spread = float(ask) - float(bid) if ask is not None and bid is not None else None
        spread_pct = float(spread) / float(mid) if spread is not None and mid and float(mid) > 0 else None

        entry_premium_realistic = (
            float(mid) + 0.25 * (float(ask) - float(bid))
            if mid is not None and ask is not None and bid is not None
            else mid
        )
        selected_entry = {
            "mid": mid,
            "ask": ask,
            "realistic": entry_premium_realistic,
        }.get(clean_string(entry_mode).lower(), mid)

        row = {
            "ticker": clean_ticker,
            "source": source,
            "trust": trust_level,
            "snapshot_date": snapshot_date,
            "downloaded_at": downloaded_at,
            "contract_symbol": "",
            "contract_label": _contract_label(strike, option_type, expiration_date.isoformat() if expiration_date else None),
            "option_type": option_type,
            "expiration": expiration_date.isoformat() if expiration_date else None,
            "dte": dte,
            "strike": strike,
            "underlying_price": underlying,
            "moneyness": moneyness,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "last": last,
            "volume": volume,
            "open_interest": open_interest,
            "implied_volatility": iv,
            "iv": iv,
            "iv_rank": _parse_percent(raw_row.get("iv_rank")),
            "iv_percentile": _parse_percent(raw_row.get("iv_percentile")),
            "delta": _parse_decimal(raw_row.get("delta")),
            "gamma": _parse_decimal(raw_row.get("gamma")),
            "theta": _parse_decimal(raw_row.get("theta")),
            "vega": _parse_decimal(raw_row.get("vega")),
            "rho": _parse_decimal(raw_row.get("rho")),
            "itm_probability": _parse_percent(raw_row.get("itm_probability")),
            "otm_probability": _parse_percent(raw_row.get("otm_probability")),
            "profit_probability": _parse_percent(raw_row.get("profit_probability")),
            "spread": spread,
            "spread_pct_of_mid": spread_pct,
            "entry_premium_mid": mid,
            "entry_premium_ask": ask,
            "entry_premium_realistic": entry_premium_realistic,
            "entry_premium_selected": selected_entry,
            "entry_price_mode": clean_string(entry_mode).lower() or "mid",
            "exit_premium_conservative": bid,
            "breakeven_mid": float(strike) + float(mid) if strike is not None and mid is not None and option_type == "call" else (float(strike) - float(mid) if strike is not None and mid is not None else None),
            "breakeven_ask": float(strike) + float(ask) if strike is not None and ask is not None and option_type == "call" else (float(strike) - float(ask) if strike is not None and ask is not None else None),
            "barchart_be_bid": _parse_decimal(raw_row.get("barchart_be_bid")),
            "barchart_be_ask": _parse_decimal(raw_row.get("barchart_be_ask")),
            "barchart_be_mid": _parse_decimal(raw_row.get("barchart_be_mid")),
            "moneyness_decimal": moneyness,
            "moneyness_bucket": _moneyness_bucket(moneyness),
            "links": clean_string(raw_row.get("links")) or None,
        }
        flags = _quality_flags(row, snapshot_date=snapshot_date)
        row["quality_flags"] = ";".join(flags)
        row["liquidity_bucket"] = _liquidity_bucket(row)
        row["model_eligible"] = bool(
            symbol == clean_ticker
            and (option_type == "call" or include_puts)
            and ask is not None
            and float(ask) > float(min_ask)
            and mid is not None
            and float(mid) > 0
            and iv is not None
            and float(iv) > float(min_iv)
            and dte is not None
            and int(dte) >= int(min_dte)
            and int(dte) <= int(max_dte)
            and strike is not None
            and float(strike) > 0
            and underlying is not None
            and float(underlying) > 0
            and (min_open_interest is None or int(open_interest or 0) >= int(min_open_interest))
            and (allow_zero_volume or int(volume or 0) > 0)
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["expiration", "option_type", "strike"]).reset_index(drop=True)


def import_barchart_options_csv(
    ticker: str,
    csv_path: str | Path,
    *,
    snapshot_date: str,
    data_root: str | Path | None = None,
    entry_mode: str = "mid",
    calls_only: bool = True,
    include_puts: bool = False,
    min_ask: float = 0.0,
    min_iv: float = 0.0001,
    min_dte: int = 1,
    max_dte: int = 900,
    min_open_interest: int | None = None,
    allow_zero_volume: bool = True,
    source: str = BARCHART_OPTIONS_SOURCE,
    trust_level: str = BARCHART_OPTIONS_TRUST,
) -> BarchartOptionsImportResult:
    """Copy and normalize one manually downloaded Barchart Options Screener CSV."""

    source_path = Path(csv_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Barchart options CSV was not found: {source_path}")
    snapshot = parse_date(snapshot_date)
    if snapshot is None:
        raise ValueError(f"snapshot_date must be a valid date, got: {snapshot_date!r}")
    include_puts = bool(include_puts or not calls_only)
    entry_mode_clean = clean_string(entry_mode).lower() or "mid"
    if entry_mode_clean not in {"mid", "ask", "realistic"}:
        raise ValueError("entry_mode must be one of: mid, ask, realistic")

    structure = ensure_barchart_options_structure(ticker, data_root)
    copied_raw = _safe_copy_raw(source_path, structure["raw"])
    raw_frame = pd.read_csv(source_path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    downloaded_at = _downloaded_at_from_footer(raw_frame)
    clean_frame = _clean_contract_rows(raw_frame)
    ticker_frame = clean_frame.copy()
    ticker_frame.columns = [_column_alias(column) for column in ticker_frame.columns]
    ticker_symbols = ticker_frame.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()
    ticker_types = ticker_frame.get("type", pd.Series(dtype=str)).astype(str).str.lower()
    ticker_rows = ticker_frame.loc[ticker_symbols.eq(clean_string(ticker).upper())].copy()
    source_calls_count = int(ticker_types.loc[ticker_symbols.eq(clean_string(ticker).upper())].eq("call").sum())
    source_puts_count = int(ticker_types.loc[ticker_symbols.eq(clean_string(ticker).upper())].eq("put").sum())

    normalized = _normalize_options_frame(
        clean_frame,
        ticker=ticker,
        snapshot_date=snapshot.isoformat(),
        downloaded_at=downloaded_at,
        source=source,
        trust_level=trust_level,
        include_puts=include_puts,
        min_ask=min_ask,
        min_iv=min_iv,
        min_dte=min_dte,
        max_dte=max_dte,
        min_open_interest=min_open_interest,
        allow_zero_volume=allow_zero_volume,
        entry_mode=entry_mode_clean,
    )
    if normalized.empty:
        raise ValueError(f"No Barchart options rows for {clean_string(ticker).upper()} could be normalized from: {source_path}")

    output_paths: list[str] = []
    for expiry_text, expiry_frame in normalized.groupby("expiration", dropna=True):
        expiry = parse_date(expiry_text)
        expiry_slug = expiry.isoformat() if expiry else "unknown"
        stem = f"barchart_{clean_string(ticker).lower()}_options_exp_{expiry_slug}_{snapshot.isoformat()}"
        output_path = structure["normalized"] / f"{stem}.csv"
        write_dataframe_csv(expiry_frame, output_path, index=False)
        spot_values = pd.to_numeric(expiry_frame.get("underlying_price"), errors="coerce").dropna()
        sidecar = {
            "ticker": clean_string(ticker).upper(),
            "snapshot_date": snapshot.isoformat(),
            "expiry_date": expiry_slug if expiry else None,
            "spot_price": float(spot_values.iloc[0]) if not spot_values.empty else None,
            "spot_price_source": source,
            "source": source,
            "trust": trust_level,
            "extra": {
                "trust": trust_level,
                "entry_price_mode": entry_mode_clean,
            },
            "snapshot_scope": source,
            "storage_location": source,
            "quote_count": int(len(expiry_frame.index)),
            "strike_count": int(pd.to_numeric(expiry_frame.get("strike"), errors="coerce").dropna().nunique()),
            "entry_price_mode": entry_mode_clean,
            "raw_csv_path": str(copied_raw),
            "downloaded_at": downloaded_at,
        }
        write_json(sidecar, output_path.with_suffix(".metadata.json"))
        output_paths.append(str(output_path))

    fields_detected = list(raw_frame.columns)
    normalized_aliases = {normalize_column_name(column) for column in raw_frame.columns}
    manifest = {
        "ticker": clean_string(ticker).upper(),
        "source": source,
        "trust_level": trust_level,
        "raw_csv_path": str(copied_raw),
        "normalized_output_paths": output_paths,
        "snapshot_date": snapshot.isoformat(),
        "downloaded_at": downloaded_at,
        "rows_raw": int(len(raw_frame.index)),
        "rows_after_footer_cleanup": int(len(clean_frame.index)),
        "rows_for_ticker": int(len(ticker_rows.index)),
        "calls_count": source_calls_count,
        "puts_count": source_puts_count,
        "rows_model_eligible": int(normalized["model_eligible"].sum()),
        "expiries": sorted(str(value) for value in normalized["expiration"].dropna().unique().tolist()),
        "min_expiry": normalized["expiration"].dropna().min(),
        "max_expiry": normalized["expiration"].dropna().max(),
        "min_strike": float(pd.to_numeric(normalized["strike"], errors="coerce").min()),
        "max_strike": float(pd.to_numeric(normalized["strike"], errors="coerce").max()),
        "fields_detected": fields_detected,
        "missing_expected_fields": sorted(EXPECTED_FIELDS - normalized_aliases),
        "warnings": sorted(
            {
                flag
                for flags in normalized["quality_flags"].dropna().astype(str)
                for flag in flags.split(";")
                if flag
            }
        ),
        "filters": {
            "calls_only": bool(calls_only),
            "include_puts": bool(include_puts),
            "min_ask": float(min_ask),
            "min_iv": float(min_iv),
            "min_dte": int(min_dte),
            "max_dte": int(max_dte),
            "min_open_interest": min_open_interest,
            "allow_zero_volume": bool(allow_zero_volume),
            "entry_mode": entry_mode_clean,
        },
    }
    manifest_path = structure["manifests"] / f"barchart_options_import_{snapshot.isoformat()}.json"
    write_json(manifest, manifest_path)

    return BarchartOptionsImportResult(
        ticker=clean_string(ticker).upper(),
        source=source,
        trust_level=trust_level,
        snapshot_date=snapshot.isoformat(),
        raw_csv_path=str(copied_raw),
        normalized_output_paths=output_paths,
        manifest_path=str(manifest_path),
        rows_raw=int(len(raw_frame.index)),
        rows_after_footer_cleanup=int(len(clean_frame.index)),
        rows_for_ticker=int(len(ticker_rows.index)),
        rows_model_eligible=int(normalized["model_eligible"].sum()),
    )
