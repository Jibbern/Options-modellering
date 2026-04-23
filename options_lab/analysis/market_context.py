"""Analysis-side local market-context resolution for canonical bundle workflows."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..ibkr import get_underlying_spot as get_ibkr_underlying_spot
from ..io import OptionChain, load_chain
from ..prices import get_underlying_spot as get_price_store_spot
from ..rates import get_risk_free_rate
from ..research_metadata import resolve_research_context
from ..research_metadata.catalog import MIN_USABLE_QUOTE_COVERAGE_PCT, discover_chain_snapshots
from ..utils import DEFAULT_RISK_FREE_RATE, clean_string, parse_date


@dataclass(frozen=True)
class ResolvedChainInput:
    """One resolved local chain slice selected for analysis."""

    scope: str
    fallback_level: str
    file_path: str
    storage_location: str
    source_snapshot_date: str
    expiry_date: str
    quote_usable: bool
    usable_quote_coverage_pct: float
    usable_quote_count: int
    contract_count: int
    snapshot_distance_days: int
    chosen_reason: str
    source_quality: str
    source_trust_label: str
    source_quality_note: str
    rejected_same_day_ibkr_file: str | None = None
    rejected_same_day_ibkr_coverage_pct: float | None = None
    chain: OptionChain | None = None


@dataclass
class MarketContextResolution:
    """Resolved canonical local market context for one analysis run."""

    ticker: str
    snapshot_date: date
    target_date: date | None
    resolved_chain_inputs: list[ResolvedChainInput] = field(default_factory=list)
    selection_scope: dict[str, Any] = field(default_factory=dict)
    chain_source_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_context_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    spot_price: float | None = None
    spot_price_source: str | None = None
    spot_price_matched_date: date | None = None
    spot_price_field_used: str | None = None
    spot_price_used_prior_date: bool = False
    spot_price_note: str | None = None
    spot_quality_note: str | None = None
    ibkr_same_day_spot_attempted: bool = False
    ibkr_same_day_spot_rejected_reason: str | None = None
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    risk_free_rate_source: str | None = None
    risk_free_rate_series: str | None = None
    risk_free_rate_matched_date: date | None = None
    risk_free_rate_note: str | None = None
    research_context: dict[str, Any] = field(default_factory=dict)
    research_context_expiry_used: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _SpotContext:
    price: float | None
    source: str | None
    matched_date: date | None
    field_used: str | None
    used_prior_date: bool
    note: str | None
    quality_note: str | None
    ibkr_same_day_attempted: bool
    ibkr_same_day_rejected_reason: str | None


_NEARBY_SOURCE_RANK = {
    "preferred_option_chains": 0,
    "ibkr_full_quoted_snapshot": 1,
    "ibkr_chain_snapshot": 2,
    "legacy_ticker_root": 3,
}


def _slice_sort_key(row: pd.Series, requested_snapshot: date) -> tuple[int, int, str]:
    snapshot_date = pd.Timestamp(row["snapshot_date"]).date()
    distance_days = abs((snapshot_date - requested_snapshot).days)
    source_rank = _NEARBY_SOURCE_RANK.get(clean_string(row.get("storage_location")).lower(), 9)
    return distance_days, source_rank, snapshot_date.isoformat()


def _chosen_reason(fallback_level: str, row: pd.Series) -> str:
    mapping = {
        "exact_same_day_ibkr_full_quoted": "Same-day quote-usable IBKR full quoted slice won the source precedence.",
        "same_day_fallback_from_sparse_ibkr": "Same-day IBKR full quoted slice was too sparse, so the same-day local option_chains slice was used instead.",
        "exact_same_day_preferred_option_chain": "Same-day preferred local option_chains slice was used because no quote-usable same-day IBKR full quoted slice existed.",
        "exact_same_day_local_quoted": "Same-day quoted local slice was used because it was the best available exact-date source.",
        "nearest_quoted_fallback": "No exact same-day usable slice existed for this expiry, so the nearest usable local quoted slice was selected.",
        "nearest_sparse_fallback": "No usable quoted slice existed for this expiry, so the nearest sparse local slice was selected as a last resort.",
        "same_day_sparse_last_resort": "No usable same-day slice existed for this expiry, so the sparse same-day slice was used as a last resort.",
    }
    return mapping.get(fallback_level, f"Resolved local chain source from {clean_string(row.get('storage_location')).lower()}.")


def _source_quality(snapshot_distance_days: int, quote_usable: bool) -> str:
    if int(snapshot_distance_days) <= 0:
        return "same_day_quoted" if bool(quote_usable) else "same_day_sparse"
    return "prior_day_quoted" if bool(quote_usable) else "prior_day_sparse"


def _source_trust_label(source_quality: str) -> str:
    mapping = {
        "same_day_quoted": "trusted_quoted",
        "prior_day_quoted": "quoted_prior_day",
        "same_day_sparse": "fallback_only",
        "prior_day_sparse": "fallback_only",
        "structure_only": "structure_only",
    }
    return mapping.get(clean_string(source_quality).lower(), "fallback_only")


def _source_quality_note(*, source_quality: str, fallback_level: str, usable_quote_coverage_pct: float) -> str:
    quality = clean_string(source_quality).lower()
    fallback = clean_string(fallback_level).lower()
    coverage_text = f"{float(usable_quote_coverage_pct):.1f}% usable quote coverage"
    if quality == "same_day_quoted":
        return f"Same-day quoted source was used for pricing with {coverage_text}."
    if quality == "prior_day_quoted":
        return f"Nearest prior quoted source was used for pricing with {coverage_text} because no trusted same-day slice won."
    if quality == "same_day_sparse":
        return f"Same-day slice remained sparse ({coverage_text}) and is usable only as a cautious fallback."
    if quality == "prior_day_sparse":
        if fallback.startswith("nearest_"):
            return f"Nearest prior sparse slice was used as a last-resort fallback with {coverage_text}."
        return f"Prior-day sparse slice was used cautiously with {coverage_text}."
    return "Source was present structurally but was not trustworthy enough for quoted pricing."


def _analysis_trust_rollup(resolved_inputs: list[ResolvedChainInput]) -> tuple[str, str, dict[str, int]]:
    counts = {
        "same_day_quoted": 0,
        "same_day_sparse": 0,
        "prior_day_quoted": 0,
        "prior_day_sparse": 0,
        "structure_only": 0,
    }
    for item in resolved_inputs:
        counts[clean_string(item.source_quality).lower()] = counts.get(clean_string(item.source_quality).lower(), 0) + 1
    trusted_count = counts["same_day_quoted"] + counts["prior_day_quoted"]
    sparse_count = counts["same_day_sparse"] + counts["prior_day_sparse"]
    if trusted_count == len(resolved_inputs) and counts["same_day_quoted"] == len(resolved_inputs):
        return "high", "All resolved expiries were priced from same-day quoted sources.", counts
    if sparse_count == 0 and trusted_count > 0:
        return "medium", "Pricing used quoted sources, but one or more expiries came from prior-day fallbacks.", counts
    if trusted_count > 0:
        return "cautious", "The run mixes quoted expiries with sparse fallback expiries, so treat rankings more cautiously.", counts
    return "low", "Resolved expiries rely on sparse fallback sources only, so pricing confidence is limited.", counts


def _resolve_chain_row_for_expiry(group: pd.DataFrame, requested_snapshot: date) -> tuple[pd.Series, str, str, pd.Series | None]:
    requested_ts = pd.Timestamp(requested_snapshot)
    working = group.copy()
    working["snapshot_distance_days"] = (
        working["snapshot_date"].map(lambda value: abs((pd.Timestamp(value).date() - requested_snapshot).days))
    )
    same_day = working.loc[working["snapshot_date"] == requested_ts].copy()
    non_universe = working.loc[working["storage_location"] != "ibkr_chain_universe"].copy()
    same_day_sparse_ibkr = same_day.loc[
        (same_day["storage_location"] == "ibkr_full_quoted_snapshot") & (~same_day["quote_usable"].fillna(False))
    ].copy()

    exact_ibkr = same_day.loc[
        (same_day["storage_location"] == "ibkr_full_quoted_snapshot") & (same_day["quote_usable"].fillna(False))
    ].copy()
    if not exact_ibkr.empty:
        chosen = exact_ibkr.sort_values(
            ["usable_quote_coverage_pct", "usable_quote_count", "contract_count"],
            ascending=[False, False, False],
        ).iloc[0]
        return chosen, "exact_snapshot", "exact_same_day_ibkr_full_quoted", None

    exact_preferred = same_day.loc[same_day["storage_location"] == "preferred_option_chains"].copy()
    if not exact_preferred.empty:
        chosen = exact_preferred.sort_values(
            ["usable_quote_coverage_pct", "usable_quote_count", "contract_count"],
            ascending=[False, False, False],
        ).iloc[0]
        rejected = same_day_sparse_ibkr.sort_values(
            ["usable_quote_coverage_pct", "usable_quote_count", "contract_count"],
            ascending=[False, False, False],
        ).iloc[0] if not same_day_sparse_ibkr.empty else None
        fallback_level = "same_day_fallback_from_sparse_ibkr" if rejected is not None else "exact_same_day_preferred_option_chain"
        return chosen, "exact_snapshot", fallback_level, rejected

    exact_other_quoted = same_day.loc[
        (same_day["quote_usable"].fillna(False)) & (same_day["storage_location"] != "ibkr_chain_universe")
    ].copy()
    if not exact_other_quoted.empty:
        exact_other_quoted = exact_other_quoted.assign(
            _source_rank=exact_other_quoted["storage_location"].map(lambda value: _NEARBY_SOURCE_RANK.get(clean_string(value).lower(), 9))
        )
        chosen = exact_other_quoted.sort_values(
            ["_source_rank", "usable_quote_coverage_pct", "usable_quote_count", "contract_count"],
            ascending=[True, False, False, False],
        ).iloc[0]
        return chosen, "exact_snapshot", "exact_same_day_local_quoted", None

    nearby_usable = non_universe.loc[
        (non_universe["snapshot_date"] != requested_ts) & (non_universe["quote_usable"].fillna(False))
    ].copy()
    if not nearby_usable.empty:
        nearby_usable = nearby_usable.assign(
            _sort_key=nearby_usable.apply(lambda row: _slice_sort_key(row, requested_snapshot), axis=1)
        ).sort_values("_sort_key")
        chosen = nearby_usable.iloc[0]
        return chosen, "nearest_snapshot_fallback", "nearest_quoted_fallback", None

    if same_day_sparse_ibkr.empty and same_day.loc[same_day["storage_location"] != "ibkr_chain_universe"].empty and non_universe.empty:
        raise ValueError("No usable local chain slices were available after excluding chain-universe metadata files.")

    fallback_pool = same_day.loc[same_day["storage_location"] != "ibkr_chain_universe"].copy()
    fallback_level = "same_day_sparse_last_resort"
    if fallback_pool.empty:
        fallback_pool = non_universe.copy()
        fallback_level = "nearest_sparse_fallback"
    fallback_pool = fallback_pool.assign(
        _sort_key=fallback_pool.apply(lambda row: _slice_sort_key(row, requested_snapshot), axis=1)
    ).sort_values("_sort_key")
    chosen = fallback_pool.iloc[0]
    rejected = same_day_sparse_ibkr.sort_values(
        ["usable_quote_coverage_pct", "usable_quote_count", "contract_count"],
        ascending=[False, False, False],
    ).iloc[0] if not same_day_sparse_ibkr.empty else None
    scope = "exact_snapshot" if clean_string(fallback_level).startswith("same_day") else "nearest_snapshot_fallback"
    return chosen, scope, fallback_level, rejected


def _resolve_spot_context(ticker: str, snapshot_date: date, *, data_root: str | Path | None) -> _SpotContext:
    ibkr_rejected_reason: str | None = None
    try:
        match = get_ibkr_underlying_spot(
            ticker=ticker,
            snapshot_date=snapshot_date,
            data_root=data_root,
            require_same_day=True,
        )
        field_used = clean_string(match.field_used).lower() or "last"
        return _SpotContext(
            price=float(match.close_price),
            source=clean_string(match.source) or "ibkr_delayed",
            matched_date=match.matched_date,
            field_used=field_used,
            used_prior_date=bool(match.used_prior_date),
            note=f"Resolved spot from same-day delayed IBKR spot using the {field_used} field.",
            quality_note="Same-day delayed IBKR spot was usable and won over local historical prices.",
            ibkr_same_day_attempted=True,
            ibkr_same_day_rejected_reason=None,
        )
    except (FileNotFoundError, LookupError, ValueError) as exc:
        ibkr_rejected_reason = str(exc)

    try:
        match = get_price_store_spot(ticker=ticker, snapshot_date=snapshot_date, data_root=data_root)
        if match.used_prior_date:
            note = (
                "Same-day delayed IBKR spot was unavailable or unusable, so the latest prior local historical price close was used."
            )
            quality_note = (
                "Spot fell back to a prior-date local historical close because same-day delayed IBKR spot was unavailable or unusable."
            )
        else:
            note = "Same-day delayed IBKR spot was unavailable or unusable, so the local historical price close was used."
            quality_note = "Spot fell back cleanly to a same-day local historical close."
        return _SpotContext(
            price=float(match.close_price),
            source=clean_string(match.source) or "historical_price_store",
            matched_date=match.matched_date,
            field_used="close",
            used_prior_date=bool(match.used_prior_date),
            note=note,
            quality_note=quality_note,
            ibkr_same_day_attempted=True,
            ibkr_same_day_rejected_reason=ibkr_rejected_reason,
        )
    except (FileNotFoundError, LookupError, ValueError):
        return _SpotContext(
            price=None,
            source=None,
            matched_date=None,
            field_used=None,
            used_prior_date=False,
            note="No same-day delayed IBKR spot or local historical-price match was available for the requested snapshot date.",
            quality_note="Spot could not be resolved from either same-day delayed IBKR or local historical-price data.",
            ibkr_same_day_attempted=True,
            ibkr_same_day_rejected_reason=ibkr_rejected_reason,
        )


def _resolve_risk_free_context(snapshot_date: date, target_date: date | None, *, data_root: str | Path | None) -> tuple[float, str, str | None, date | None, str | None]:
    if target_date is not None:
        try:
            match = get_risk_free_rate(snapshot_date=snapshot_date, expiry_date=target_date, data_root=data_root)
            note = (
                "Used the latest available prior Treasury observation because the requested snapshot date was missing in the local FRED store."
                if match.used_prior_date
                else None
            )
            return float(match.rate_decimal), "fred_local_store", clean_string(match.series_used) or None, match.matched_date, note
        except (FileNotFoundError, LookupError, ValueError):
            pass
    return DEFAULT_RISK_FREE_RATE, "default_fallback", None, None, "No local FRED risk-free match was available, so the default fallback rate was used."


def _resolve_research_context_for_inputs(
    ticker: str,
    snapshot_date: date,
    resolved_chain_inputs: list[ResolvedChainInput],
    *,
    data_root: str | Path | None,
) -> tuple[dict[str, Any], str | None]:
    if not resolved_chain_inputs:
        return {}, None
    expiry_candidates = sorted(
        [
            parse_date(item.expiry_date)
            for item in resolved_chain_inputs
            if parse_date(item.expiry_date) is not None
        ]
    )
    chosen_expiry = expiry_candidates[0] if expiry_candidates else None
    context = resolve_research_context(
        ticker=ticker,
        snapshot_date=snapshot_date.isoformat(),
        expiry_date=chosen_expiry.isoformat() if chosen_expiry is not None else None,
        data_root=data_root,
    )
    return context, chosen_expiry.isoformat() if chosen_expiry is not None else None


def resolve_market_context(
    *,
    ticker: str,
    snapshot_date: date | str,
    target_date: date | str | None = None,
    data_root: str | Path | None = None,
    minimum_quote_coverage_pct: float = MIN_USABLE_QUOTE_COVERAGE_PCT,
) -> MarketContextResolution:
    """Resolve canonical local chain, spot, risk-free, and metadata context."""

    ticker_label = clean_string(ticker).upper()
    requested_snapshot = parse_date(snapshot_date)
    if requested_snapshot is None:
        raise ValueError(f"snapshot_date must be a valid date, got: {snapshot_date!r}")
    resolved_target_date = parse_date(target_date) if target_date is not None else None
    if float(minimum_quote_coverage_pct) != float(MIN_USABLE_QUOTE_COVERAGE_PCT):
        raise ValueError("minimum_quote_coverage_pct currently follows the canonical 20% gate only.")

    raw_records = discover_chain_snapshots(ticker_label, data_root=data_root, dedupe=False)
    slices = pd.DataFrame(raw_records)
    if slices.empty:
        raise ValueError(f"No local chain snapshots were found for {ticker_label}.")
    slices["snapshot_date"] = pd.to_datetime(slices["snapshot_date"], errors="coerce").dt.normalize()
    slices["expiry_date"] = pd.to_datetime(slices["expiry_date"], errors="coerce").dt.normalize()

    pricing_candidates = slices.loc[slices["storage_location"] != "ibkr_chain_universe"].copy()
    if pricing_candidates.empty:
        raise ValueError(f"No local quoted chain slices were available for {ticker_label}; only chain-universe metadata exists.")

    resolved_inputs: list[ResolvedChainInput] = []
    warnings: list[str] = []
    for expiry_value, group in pricing_candidates.groupby("expiry_date", dropna=True):
        chosen_row, scope, fallback_level, rejected_sparse_ibkr = _resolve_chain_row_for_expiry(group, requested_snapshot)
        chain = load_chain(
            chosen_row["file_path"],
            prices_data_root=data_root,
            rates_data_root=data_root,
            research_data_root=data_root,
        )
        rejected_file = str(rejected_sparse_ibkr["file_path"]) if rejected_sparse_ibkr is not None else None
        rejected_coverage = float(rejected_sparse_ibkr["usable_quote_coverage_pct"]) if rejected_sparse_ibkr is not None else None
        resolved_inputs.append(
            ResolvedChainInput(
                scope=scope,
                fallback_level=fallback_level,
                file_path=str(chosen_row["file_path"]),
                storage_location=clean_string(chosen_row.get("storage_location")).lower(),
                source_snapshot_date=pd.Timestamp(chosen_row["snapshot_date"]).date().isoformat(),
                expiry_date=pd.Timestamp(expiry_value).date().isoformat(),
                quote_usable=bool(chosen_row.get("quote_usable")),
                usable_quote_coverage_pct=float(chosen_row.get("usable_quote_coverage_pct") or 0.0),
                usable_quote_count=int(chosen_row.get("usable_quote_count") or 0),
                contract_count=int(chosen_row.get("contract_count") or 0),
                snapshot_distance_days=int(chosen_row.get("snapshot_distance_days") or abs((pd.Timestamp(chosen_row["snapshot_date"]).date() - requested_snapshot).days)),
                chosen_reason=_chosen_reason(fallback_level, chosen_row),
                source_quality=_source_quality(
                    int(chosen_row.get("snapshot_distance_days") or abs((pd.Timestamp(chosen_row["snapshot_date"]).date() - requested_snapshot).days)),
                    bool(chosen_row.get("quote_usable")),
                ),
                source_trust_label=_source_trust_label(
                    _source_quality(
                        int(chosen_row.get("snapshot_distance_days") or abs((pd.Timestamp(chosen_row["snapshot_date"]).date() - requested_snapshot).days)),
                        bool(chosen_row.get("quote_usable")),
                    )
                ),
                source_quality_note=_source_quality_note(
                    source_quality=_source_quality(
                        int(chosen_row.get("snapshot_distance_days") or abs((pd.Timestamp(chosen_row["snapshot_date"]).date() - requested_snapshot).days)),
                        bool(chosen_row.get("quote_usable")),
                    ),
                    fallback_level=fallback_level,
                    usable_quote_coverage_pct=float(chosen_row.get("usable_quote_coverage_pct") or 0.0),
                ),
                rejected_same_day_ibkr_file=rejected_file,
                rejected_same_day_ibkr_coverage_pct=rejected_coverage,
                chain=chain,
            )
        )
        if rejected_file:
            warnings.append(
                f"Rejected sparse same-day IBKR full quoted slice for expiry {pd.Timestamp(expiry_value).date().isoformat()} because usable quote coverage was {rejected_coverage:.1f}%."
            )
        if clean_string(fallback_level).startswith("nearest_"):
            warnings.append(
                f"Used a nearest local chain fallback for expiry {pd.Timestamp(expiry_value).date().isoformat()} because no usable same-day slice existed."
            )

    resolved_inputs.sort(key=lambda item: (item.expiry_date, item.source_snapshot_date, item.file_path))
    analysis_trust_level, analysis_trust_note, trust_counts = _analysis_trust_rollup(resolved_inputs)
    spot_context = _resolve_spot_context(
        ticker_label,
        requested_snapshot,
        data_root=data_root,
    )
    risk_free_rate, risk_free_source, risk_free_series, risk_free_matched_date, risk_free_note = _resolve_risk_free_context(
        requested_snapshot,
        resolved_target_date,
        data_root=data_root,
    )
    research_context, research_expiry_used = _resolve_research_context_for_inputs(
        ticker_label,
        requested_snapshot,
        resolved_inputs,
        data_root=data_root,
    )
    dividend_yield = float(
        ((research_context.get("dividend_assumption") or {}).get("dividend_yield"))
        or 0.0
    )
    for item in resolved_inputs:
        if item.chain is None:
            continue
        item.chain.metadata = replace(
            item.chain.metadata,
            spot_price=spot_context.price if spot_context.price is not None else item.chain.metadata.spot_price,
            spot_price_source=spot_context.source or item.chain.metadata.spot_price_source,
            spot_price_matched_date=spot_context.matched_date or item.chain.metadata.spot_price_matched_date,
            spot_price_note=spot_context.note or item.chain.metadata.spot_price_note,
            risk_free_rate=risk_free_rate,
            risk_free_rate_source=risk_free_source or item.chain.metadata.risk_free_rate_source,
            risk_free_rate_series=risk_free_series or item.chain.metadata.risk_free_rate_series,
            risk_free_rate_matched_date=risk_free_matched_date or item.chain.metadata.risk_free_rate_matched_date,
            risk_free_rate_note=risk_free_note or item.chain.metadata.risk_free_rate_note,
            dividend_yield=dividend_yield,
            research_context=dict(research_context),
        )

    chain_source_summary = pd.DataFrame(
        [
            {
                "ticker": ticker_label,
                "requested_snapshot_date": requested_snapshot.isoformat(),
                "expiry_date": item.expiry_date,
                "scope": item.scope,
                "fallback_level": item.fallback_level,
                "storage_location": item.storage_location,
                "source_snapshot_date": item.source_snapshot_date,
                "source_snapshot_file": item.file_path,
                "quote_usable": item.quote_usable,
                "usable_quote_coverage_pct": item.usable_quote_coverage_pct,
                "usable_quote_count": item.usable_quote_count,
                "contract_count": item.contract_count,
                "snapshot_distance_days": item.snapshot_distance_days,
                "source_quality": item.source_quality,
                "source_trust_label": item.source_trust_label,
                "source_quality_note": item.source_quality_note,
                "chosen_reason": item.chosen_reason,
                "rejected_same_day_ibkr_file": item.rejected_same_day_ibkr_file,
                "rejected_same_day_ibkr_coverage_pct": item.rejected_same_day_ibkr_coverage_pct,
            }
            for item in resolved_inputs
        ]
    )
    market_context_summary = pd.DataFrame(
        [
            {
                "ticker": ticker_label,
                "requested_snapshot_date": requested_snapshot.isoformat(),
                "target_date": resolved_target_date.isoformat() if resolved_target_date is not None else None,
                "research_context_expiry_used": research_expiry_used,
                "resolved_expiry_count": len(resolved_inputs),
                "exact_requested_expiry_count": int(sum(1 for item in resolved_inputs if item.scope == "exact_snapshot")),
                "used_nearest_snapshot_fallback": any(item.scope == "nearest_snapshot_fallback" for item in resolved_inputs),
                "used_same_day_fallback_from_sparse_ibkr": any(item.fallback_level == "same_day_fallback_from_sparse_ibkr" for item in resolved_inputs),
                "analysis_trust_level": analysis_trust_level,
                "analysis_trust_note": analysis_trust_note,
                "same_day_quoted_expiry_count": trust_counts.get("same_day_quoted", 0),
                "same_day_sparse_expiry_count": trust_counts.get("same_day_sparse", 0),
                "prior_day_quoted_expiry_count": trust_counts.get("prior_day_quoted", 0),
                "prior_day_sparse_expiry_count": trust_counts.get("prior_day_sparse", 0),
                "trusted_expiry_count": trust_counts.get("same_day_quoted", 0) + trust_counts.get("prior_day_quoted", 0),
                "fallback_only_expiry_count": trust_counts.get("same_day_sparse", 0) + trust_counts.get("prior_day_sparse", 0),
                "spot_price": spot_context.price,
                "spot_price_source": spot_context.source,
                "spot_price_matched_date": spot_context.matched_date.isoformat() if spot_context.matched_date is not None else None,
                "spot_field_used": spot_context.field_used,
                "spot_used_prior_date": bool(spot_context.used_prior_date),
                "spot_price_note": spot_context.note,
                "spot_quality_note": spot_context.quality_note,
                "ibkr_same_day_spot_attempted": bool(spot_context.ibkr_same_day_attempted),
                "ibkr_same_day_spot_rejected_reason": spot_context.ibkr_same_day_rejected_reason,
                "risk_free_rate": risk_free_rate,
                "risk_free_rate_source": risk_free_source,
                "risk_free_rate_series": risk_free_series,
                "risk_free_rate_matched_date": risk_free_matched_date.isoformat() if risk_free_matched_date is not None else None,
                "risk_free_rate_note": risk_free_note,
                "expected_move_matched": bool((research_context.get("expected_move") or {}).get("matched")),
                "expected_move_pct": (research_context.get("expected_move") or {}).get("expected_move_pct"),
                "options_overview_iv_rank": (research_context.get("options_overview") or {}).get("iv_rank"),
                "options_overview_iv_percentile": (research_context.get("options_overview") or {}).get("iv_percentile"),
                "nearest_event_date": (research_context.get("nearest_event") or {}).get("event_date"),
                "nearest_event_type": (research_context.get("nearest_event") or {}).get("event_type"),
                "dividend_yield": (research_context.get("dividend_assumption") or {}).get("dividend_yield"),
                "note_count": len(research_context.get("notes") or []),
            }
        ]
    )
    selection_scope = {
        "requested_snapshot_date": requested_snapshot.isoformat(),
        "resolved_expiry_count": len(resolved_inputs),
        "exact_requested_expiry_count": int(sum(1 for item in resolved_inputs if item.scope == "exact_snapshot")),
        "used_nearby_snapshot_fallback": any(item.scope == "nearest_snapshot_fallback" for item in resolved_inputs),
        "used_same_day_fallback_from_sparse_ibkr": any(item.fallback_level == "same_day_fallback_from_sparse_ibkr" for item in resolved_inputs),
        "source_snapshot_files": [item.file_path for item in resolved_inputs],
        "source_snapshot_storage_locations": sorted({item.storage_location for item in resolved_inputs}),
        "source_snapshot_dates": sorted({item.source_snapshot_date for item in resolved_inputs}),
        "used_full_quoted_ibkr_same_date": any(
            item.scope == "exact_snapshot" and item.storage_location == "ibkr_full_quoted_snapshot"
            for item in resolved_inputs
        ),
        "analysis_trust_level": analysis_trust_level,
        "analysis_trust_note": analysis_trust_note,
        "same_day_quoted_expiry_count": trust_counts.get("same_day_quoted", 0),
        "same_day_sparse_expiry_count": trust_counts.get("same_day_sparse", 0),
        "prior_day_quoted_expiry_count": trust_counts.get("prior_day_quoted", 0),
        "prior_day_sparse_expiry_count": trust_counts.get("prior_day_sparse", 0),
        "trusted_expiry_count": trust_counts.get("same_day_quoted", 0) + trust_counts.get("prior_day_quoted", 0),
        "fallback_only_expiry_count": trust_counts.get("same_day_sparse", 0) + trust_counts.get("prior_day_sparse", 0),
        "rejected_sparse_same_day_ibkr_files": [item.rejected_same_day_ibkr_file for item in resolved_inputs if item.rejected_same_day_ibkr_file],
    }
    return MarketContextResolution(
        ticker=ticker_label,
        snapshot_date=requested_snapshot,
        target_date=resolved_target_date,
        resolved_chain_inputs=resolved_inputs,
        selection_scope=selection_scope,
        chain_source_summary=chain_source_summary,
        market_context_summary=market_context_summary,
        spot_price=spot_context.price,
        spot_price_source=spot_context.source,
        spot_price_matched_date=spot_context.matched_date,
        spot_price_field_used=spot_context.field_used,
        spot_price_used_prior_date=bool(spot_context.used_prior_date),
        spot_price_note=spot_context.note,
        spot_quality_note=spot_context.quality_note,
        ibkr_same_day_spot_attempted=bool(spot_context.ibkr_same_day_attempted),
        ibkr_same_day_spot_rejected_reason=spot_context.ibkr_same_day_rejected_reason,
        risk_free_rate=risk_free_rate,
        risk_free_rate_source=risk_free_source,
        risk_free_rate_series=risk_free_series,
        risk_free_rate_matched_date=risk_free_matched_date,
        risk_free_rate_note=risk_free_note,
        research_context=research_context,
        research_context_expiry_used=research_expiry_used,
        warnings=warnings,
    )
