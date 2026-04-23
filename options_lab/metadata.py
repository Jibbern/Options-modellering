"""Metadata resolution helpers for option snapshots and generated reports."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .persistence import make_json_safe
from .utils import (
    DEFAULT_DIVIDEND_YIELD,
    DEFAULT_RISK_FREE_RATE,
    clean_string,
    coalesce,
    parse_date,
    parse_number,
)

FILENAME_PATTERN = re.compile(
    r"(?P<ticker>[a-z0-9]+)-options-exp-(?P<expiry>\d{4}-\d{2}-\d{2})-"
    r"(?P<cycle>monthly|weekly).*?(?P<snapshot>\d{2}-\d{2}-\d{4})$",
    re.IGNORECASE,
)

METADATA_ALIASES = {
    "spot_source": "spot_price_source",
    "spot_matched_date": "spot_price_matched_date",
    "risk_free_source": "risk_free_rate_source",
    "risk_free_matched_date": "risk_free_rate_matched_date",
}


@dataclass(frozen=True)
class SnapshotMetadata:
    """Resolved metadata for one option-chain snapshot.

    Sidecar metadata can override locally resolved values on purpose. The lab is
    built for thesis-driven work where the user may have cleaner context than
    what can be inferred from a CSV export or a local store.
    """

    ticker: str | None = None
    expiry_date: date | None = None
    snapshot_date: date | None = None
    snapshot_time: str | None = None
    cycle: str | None = None
    spot_price: float | None = None
    spot_price_source: str | None = None
    spot_price_matched_date: date | None = None
    spot_price_note: str | None = None
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    risk_free_rate_source: str | None = None
    risk_free_rate_series: str | None = None
    risk_free_rate_matched_date: date | None = None
    risk_free_rate_note: str | None = None
    dividend_yield: float = DEFAULT_DIVIDEND_YIELD
    source: str | None = None
    notes: str | None = None
    research_context: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expiry_date"] = (
            self.expiry_date.isoformat() if self.expiry_date else None
        )
        payload["snapshot_date"] = (
            self.snapshot_date.isoformat() if self.snapshot_date else None
        )
        payload["spot_price_matched_date"] = (
            self.spot_price_matched_date.isoformat()
            if self.spot_price_matched_date
            else None
        )
        payload["risk_free_rate_matched_date"] = (
            self.risk_free_rate_matched_date.isoformat()
            if self.risk_free_rate_matched_date
            else None
        )
        return make_json_safe(payload)


def sidecar_metadata_path(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_suffix(".metadata.json")


def infer_metadata_from_filename(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    match = FILENAME_PATTERN.search(path.stem)
    if not match:
        return {}
    groups = match.groupdict()
    return {
        "ticker": groups["ticker"].upper(),
        "expiry_date": parse_date(groups["expiry"]),
        "snapshot_date": parse_date(groups["snapshot"]),
        "cycle": groups["cycle"].lower(),
    }


def load_sidecar_metadata(csv_path: str | Path) -> dict[str, Any]:
    path = sidecar_metadata_path(csv_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def coerce_metadata_dict(payload: dict[str, Any] | SnapshotMetadata | None) -> dict[str, Any]:
    """Normalize metadata payloads into the internal field naming scheme."""

    if payload is None:
        return {}
    if isinstance(payload, SnapshotMetadata):
        return payload.to_dict()
    result = dict(payload)
    for alias, canonical in METADATA_ALIASES.items():
        if alias in result and canonical not in result:
            result[canonical] = result[alias]
    for key in (
        "expiry_date",
        "snapshot_date",
        "spot_price_matched_date",
        "risk_free_rate_matched_date",
    ):
        if key in result:
            result[key] = parse_date(result[key])
    for key in ("spot_price", "risk_free_rate", "dividend_yield"):
        if key in result:
            result[key] = parse_number(result[key])
    if "ticker" in result and result["ticker"] is not None:
        result["ticker"] = clean_string(result["ticker"]).upper()
    if "cycle" in result and result["cycle"] is not None:
        result["cycle"] = clean_string(result["cycle"]).lower()
    if "snapshot_time" in result and result["snapshot_time"] is not None:
        result["snapshot_time"] = clean_string(result["snapshot_time"])
    return result


def _deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_metadata(
    csv_path: str | Path,
    *,
    metadata_override: dict[str, Any] | SnapshotMetadata | None = None,
    spot_price: float | None = None,
    prices_data_root: str | Path | None = None,
    rates_data_root: str | Path | None = None,
    research_data_root: str | Path | None = None,
) -> SnapshotMetadata:
    """Resolve snapshot metadata with explicit, local-store, and fallback precedence.

    Spot precedence:
    explicit override -> sidecar metadata -> future source-file field ->
    local historical-price store -> near-money heuristic in ``load_chain``.

    Risk-free precedence:
    explicit override -> sidecar metadata -> local FRED store -> default 0.04.

    Research-context precedence:
    explicit override -> sidecar ``research_context`` -> local normalized
    research-metadata stores -> omitted/empty sections.

    Dividend-yield precedence:
    explicit override -> sidecar ``dividend_yield`` -> research-context dividend
    assumption -> default 0.0.
    """

    inferred = coerce_metadata_dict(infer_metadata_from_filename(csv_path))
    sidecar = coerce_metadata_dict(load_sidecar_metadata(csv_path))
    override = coerce_metadata_dict(metadata_override)

    extra: dict[str, Any] = {}
    for payload in (inferred, sidecar, override):
        extra.update(payload.get("extra", {}))

    snapshot_date = coalesce(
        override.get("snapshot_date"),
        sidecar.get("snapshot_date"),
        inferred.get("snapshot_date"),
    )
    snapshot_time = coalesce(
        override.get("snapshot_time"),
        sidecar.get("snapshot_time"),
        inferred.get("snapshot_time"),
    )
    expiry_date = coalesce(
        override.get("expiry_date"),
        sidecar.get("expiry_date"),
        inferred.get("expiry_date"),
    )
    ticker = coalesce(
        override.get("ticker"),
        sidecar.get("ticker"),
        inferred.get("ticker"),
    )

    local_research_context: dict[str, Any] = {}
    if ticker is not None and snapshot_date is not None:
        try:
            from .research_metadata import resolve_research_context

            local_research_context = resolve_research_context(
                ticker=ticker,
                snapshot_date=snapshot_date.isoformat(),
                expiry_date=expiry_date.isoformat() if expiry_date else None,
                data_root=research_data_root,
            )
        except (FileNotFoundError, LookupError):
            local_research_context = {}

    # Sidecars and explicit overrides are intentionally allowed to outrank the
    # local store. This keeps one-off snapshot context reproducible without
    # forcing users to rewrite the shared ticker-level metadata catalog.
    research_context = dict(local_research_context)
    if isinstance(sidecar.get("research_context"), dict):
        research_context = _deep_merge_dicts(research_context, sidecar["research_context"])
    if isinstance(override.get("research_context"), dict):
        research_context = _deep_merge_dicts(research_context, override["research_context"])

    explicit_risk_free_rate = coalesce(
        override.get("risk_free_rate"),
        sidecar.get("risk_free_rate"),
        inferred.get("risk_free_rate"),
    )
    explicit_risk_free_source = coalesce(
        override.get("risk_free_rate_source"),
        sidecar.get("risk_free_rate_source"),
        inferred.get("risk_free_rate_source"),
    )
    explicit_risk_free_series = coalesce(
        override.get("risk_free_rate_series"),
        sidecar.get("risk_free_rate_series"),
        inferred.get("risk_free_rate_series"),
    )
    explicit_risk_free_matched_date = coalesce(
        override.get("risk_free_rate_matched_date"),
        sidecar.get("risk_free_rate_matched_date"),
        inferred.get("risk_free_rate_matched_date"),
    )
    explicit_risk_free_note = coalesce(
        override.get("risk_free_rate_note"),
        sidecar.get("risk_free_rate_note"),
        inferred.get("risk_free_rate_note"),
    )
    explicit_spot_price = coalesce(
        spot_price,
        override.get("spot_price"),
        sidecar.get("spot_price"),
        inferred.get("spot_price"),
    )
    explicit_spot_source = coalesce(
        override.get("spot_price_source"),
        sidecar.get("spot_price_source"),
        inferred.get("spot_price_source"),
    )
    explicit_spot_matched_date = coalesce(
        override.get("spot_price_matched_date"),
        sidecar.get("spot_price_matched_date"),
        inferred.get("spot_price_matched_date"),
    )
    explicit_spot_note = coalesce(
        override.get("spot_price_note"),
        sidecar.get("spot_price_note"),
        inferred.get("spot_price_note"),
    )

    resolved_spot_price = explicit_spot_price
    resolved_spot_source = explicit_spot_source
    resolved_spot_matched_date = explicit_spot_matched_date
    resolved_spot_note = explicit_spot_note

    if resolved_spot_price is None and ticker is not None and snapshot_date is not None:
        try:
            from .ibkr import get_underlying_spot as get_ibkr_underlying_spot

            match = get_ibkr_underlying_spot(
                ticker=ticker,
                snapshot_date=snapshot_date,
                data_root=research_data_root,
            )
            resolved_spot_price = match.close_price
            resolved_spot_source = match.source
            resolved_spot_matched_date = match.matched_date
            if match.used_prior_date:
                resolved_spot_note = (
                    "Used the latest available delayed IBKR underlying snapshot on or before the requested snapshot date."
                )
            else:
                resolved_spot_note = coalesce(
                    resolved_spot_note,
                    "Resolved spot from the local delayed IBKR underlying snapshot store.",
                )
        except (FileNotFoundError, LookupError, ValueError):
            pass

    if resolved_spot_price is None and ticker is not None and snapshot_date is not None:
        try:
            from .prices import get_underlying_spot

            match = get_underlying_spot(
                ticker=ticker,
                snapshot_date=snapshot_date,
                data_root=prices_data_root,
            )
            resolved_spot_price = match.close_price
            resolved_spot_source = match.source
            resolved_spot_matched_date = match.matched_date
            if match.used_prior_date:
                resolved_spot_note = (
                    "Used the latest available prior trading-day close because the snapshot date "
                    "was a non-trading day or was missing from the local historical-price store."
                )
        except (FileNotFoundError, LookupError):
            resolved_spot_note = coalesce(
                resolved_spot_note,
                "No local historical price store was available for this snapshot; falling back to moneyness if needed.",
            )

    if resolved_spot_price is not None and spot_price is not None:
        resolved_spot_source = coalesce(resolved_spot_source, "explicit_override")
        resolved_spot_matched_date = coalesce(resolved_spot_matched_date, snapshot_date)
    elif resolved_spot_price is not None and (
        override.get("spot_price") is not None
        or sidecar.get("spot_price") is not None
        or inferred.get("spot_price") is not None
    ):
        resolved_spot_source = coalesce(resolved_spot_source, "metadata_explicit")
        resolved_spot_matched_date = coalesce(resolved_spot_matched_date, snapshot_date)

    risk_free_rate = explicit_risk_free_rate
    risk_free_rate_source = explicit_risk_free_source
    risk_free_rate_series = explicit_risk_free_series
    risk_free_rate_matched_date = explicit_risk_free_matched_date
    risk_free_rate_note = explicit_risk_free_note

    if risk_free_rate is None and snapshot_date is not None and expiry_date is not None:
        try:
            from .rates import get_risk_free_rate

            match = get_risk_free_rate(
                snapshot_date=snapshot_date,
                expiry_date=expiry_date,
                data_root=rates_data_root,
            )
            risk_free_rate = match.rate_decimal
            risk_free_rate_source = "fred_local_store"
            risk_free_rate_series = match.series_used
            risk_free_rate_matched_date = match.matched_date
            if match.used_prior_date:
                risk_free_rate_note = (
                    "Used the latest available prior Treasury observation because the "
                    "snapshot date was a non-business day or missing in FRED."
                )
        except (FileNotFoundError, LookupError):
            risk_free_rate_note = coalesce(
                risk_free_rate_note,
                "No local FRED risk-free store was available for this snapshot; using the default fallback rate.",
            )

    if risk_free_rate is None:
        risk_free_rate = DEFAULT_RISK_FREE_RATE
        risk_free_rate_source = coalesce(risk_free_rate_source, "default_fallback")
    elif explicit_risk_free_rate is not None:
        risk_free_rate_source = coalesce(risk_free_rate_source, "metadata_explicit")

    return SnapshotMetadata(
        ticker=ticker,
        expiry_date=expiry_date,
        snapshot_date=snapshot_date,
        snapshot_time=snapshot_time,
        cycle=coalesce(
            override.get("cycle"),
            sidecar.get("cycle"),
            inferred.get("cycle"),
        ),
        spot_price=resolved_spot_price,
        spot_price_source=resolved_spot_source,
        spot_price_matched_date=resolved_spot_matched_date,
        spot_price_note=resolved_spot_note,
        risk_free_rate=risk_free_rate,
        risk_free_rate_source=risk_free_rate_source,
        risk_free_rate_series=risk_free_rate_series,
        risk_free_rate_matched_date=risk_free_rate_matched_date,
        risk_free_rate_note=risk_free_rate_note,
        dividend_yield=coalesce(
            override.get("dividend_yield"),
            sidecar.get("dividend_yield"),
            inferred.get("dividend_yield"),
            parse_number(
                ((research_context.get("dividend_assumption") or {}).get("dividend_yield"))
                if isinstance(research_context.get("dividend_assumption"), dict)
                else None
            ),
            DEFAULT_DIVIDEND_YIELD,
        ),
        source=coalesce(
            override.get("source"),
            sidecar.get("source"),
            inferred.get("source"),
        ),
        notes=coalesce(
            override.get("notes"),
            sidecar.get("notes"),
            inferred.get("notes"),
        ),
        research_context=research_context,
        extra=extra,
    )


def infer_spot_from_moneyness(contracts: pd.DataFrame) -> tuple[float | None, str | None]:
    if contracts.empty:
        return None, None
    frame = contracts.copy()
    frame = frame.dropna(subset=["strike", "option_type", "moneyness"])
    if frame.empty:
        return None, None
    frame["abs_moneyness"] = frame["moneyness"].abs()
    near_money = frame.sort_values("abs_moneyness").head(8)
    near_money = near_money[near_money["abs_moneyness"] <= 0.10]
    if near_money.empty:
        near_money = frame.sort_values("abs_moneyness").head(4)
    estimates: list[float] = []
    for row in near_money.itertuples():
        if row.option_type == "call":
            estimate = row.strike * (1.0 + row.moneyness)
        else:
            estimate = row.strike * (1.0 - row.moneyness)
        if estimate > 0:
            estimates.append(float(estimate))
    if not estimates:
        return None, None
    method = (
        "Estimated from near-money moneyness because no explicit underlying spot "
        "was available in the CSV or sidecar metadata."
    )
    return float(pd.Series(estimates).median()), method
