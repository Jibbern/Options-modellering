"""Official IBKR option-chain discovery helpers."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable

from ..utils import clean_string, parse_date
from .connection import DelayedOnlyIbkrSession
from .contracts import build_stock_contract, resolved_underlying_from_contract
from .models import (
    ChainDiscoveryDiagnostics,
    ChainFetchResult,
    ChainRow,
    ConnectionSettings,
    ContractMatchDiagnostics,
    normalize_market_data_mode,
)


def _unique_sorted_strings(values: Iterable[str | None]) -> list[str]:
    items = {clean_string(value).upper() for value in values if clean_string(value)}
    return sorted(items)


def _available_expiries(rows: Iterable[dict]) -> list[str]:
    expiries = {
        parsed.isoformat()
        for row in rows
        for value in row.get("expirations", [])
        if (parsed := parse_date(value)) is not None
    }
    return sorted(expiries)


def _available_strikes(rows: Iterable[dict]) -> list[float]:
    strikes = {float(value) for row in rows for value in row.get("strikes", [])}
    return sorted(strikes)


def _nearest_expiries(requested: list[str], available: list[str], limit: int = 3) -> list[str]:
    requested_dates = [parse_date(value) for value in requested if parse_date(value)]
    available_dates = [parse_date(value) for value in available if parse_date(value)]
    if not requested_dates or not available_dates:
        return available[:limit]
    target = requested_dates[0]
    ordered = sorted(available_dates, key=lambda item: (abs((item - target).days), item))
    return [value.isoformat() for value in ordered[:limit]]


def fetch_option_chain(
    ticker: str,
    *,
    settings: ConnectionSettings,
    market_data_mode: str = "delayed",
    exchange: str | None = None,
    currency: str = "USD",
    timeout: float = 10.0,
    include_all_exchanges: bool = False,
) -> ChainFetchResult:
    mode = normalize_market_data_mode(market_data_mode)
    requested_exchange = clean_string(exchange).upper() or None
    with DelayedOnlyIbkrSession(settings, timeout=timeout) as session:
        underlying_contract = build_stock_contract(ticker, exchange=exchange or "SMART", currency=currency)
        details = session.request_contract_details(underlying_contract, timeout=timeout)
        resolved_underlying = None
        if details:
            resolved_underlying = resolved_underlying_from_contract(details[0].contract)
            resolved_contract = details[0].contract
            raw_rows = session.request_option_parameters(
                symbol=resolved_contract.symbol,
                underlying_sec_type=resolved_contract.secType,
                underlying_conid=resolved_contract.conId,
                fut_fop_exchange="",
                timeout=timeout,
            )
        else:
            raw_rows = []

    raw_rows = [dict(row) for row in raw_rows]
    raw_exchanges_seen = _unique_sorted_strings(row.get("exchange") for row in raw_rows)
    raw_trading_classes_seen = _unique_sorted_strings(row.get("trading_class") for row in raw_rows)

    if resolved_underlying is not None and resolved_underlying.conid is not None:
        after_underlying = [
            row
            for row in raw_rows
            if row.get("underlying_conid") in {None, resolved_underlying.conid}
            or str(row.get("underlying_conid")) == str(resolved_underlying.conid)
        ]
    else:
        after_underlying = list(raw_rows)

    if requested_exchange and not include_all_exchanges:
        after_exchange = [
            row
            for row in after_underlying
            if (clean_string(row.get("exchange")).upper() or None) == requested_exchange
        ]
    else:
        after_exchange = list(after_underlying)

    after_trading_class = list(after_exchange)

    chain_rows: list[ChainRow] = []
    fetched_at = datetime.now(timezone.utc)
    for row in after_trading_class:
        row_exchange = clean_string(row.get("exchange")).upper() or None
        trading_class = clean_string(row.get("trading_class")).upper() or None
        multiplier = str(row.get("multiplier") or "") or None
        expirations = sorted({parsed.isoformat() for value in row.get("expirations", []) if (parsed := parse_date(value)) is not None})
        strikes = sorted({float(value) for value in row.get("strikes", [])})
        for expiry in expirations:
            for strike in strikes:
                for option_type in ("call", "put"):
                    chain_rows.append(
                        ChainRow(
                            ticker=clean_string(ticker).upper(),
                            underlying_conid=resolved_underlying.conid if resolved_underlying else None,
                            fetched_at=fetched_at,
                            market_data_mode=mode,
                            exchange=row_exchange,
                            trading_class=trading_class,
                            multiplier=multiplier,
                            expiry_date=expiry,
                            strike=float(strike),
                            option_type=option_type,
                            currency=clean_string(currency).upper() or "USD",
                            source="ibkr",
                            connection=settings,
                        )
                    )

    row_counts = {
        "raw_opt_param_rows": len(raw_rows),
        "after_underlying_selection": len(after_underlying),
        "after_exchange_filter": len(after_exchange),
        "after_trading_class_filter": len(after_trading_class),
        "after_expiry_strike_normalization": len(chain_rows),
        "final_chain_rows": len(chain_rows),
    }
    failure_stage = None
    if not details:
        failure_stage = "underlying_qualification"
    elif not raw_rows:
        failure_stage = "no_raw_option_parameters"
    elif not after_underlying:
        failure_stage = "underlying_conid_filter"
    elif requested_exchange and not include_all_exchanges and not after_exchange:
        failure_stage = "exchange_filter"
    elif not chain_rows:
        failure_stage = "expiry_strike_normalization"

    diagnostics = ChainDiscoveryDiagnostics(
        requested={
            "ticker": clean_string(ticker).upper(),
            "exchange": requested_exchange,
            "currency": clean_string(currency).upper() or "USD",
            "market_data_mode": mode,
            "include_all_exchanges": bool(include_all_exchanges),
        },
        resolved_underlying=resolved_underlying,
        raw_option_parameter_rows=raw_rows,
        raw_exchanges_seen=raw_exchanges_seen,
        raw_trading_classes_seen=raw_trading_classes_seen,
        available_expiries=_available_expiries(after_underlying or raw_rows),
        available_strike_count=len(_available_strikes(after_underlying or raw_rows)),
        available_strike_sample=_available_strikes(after_underlying or raw_rows)[:10],
        row_counts=row_counts,
        selected_exchange=requested_exchange,
        failure_stage=failure_stage,
    )
    return ChainFetchResult(rows=chain_rows, diagnostics=diagnostics)


def select_contract_candidates_with_diagnostics(
    chain_rows: Iterable[ChainRow],
    *,
    expiries: Iterable[str] | None = None,
    right: str = "both",
    min_strike: float | None = None,
    max_strike: float | None = None,
    strikes: Iterable[float] | None = None,
    around_spot: int | None = None,
    spot_price: float | None = None,
    max_contracts: int | None = None,
) -> tuple[list[ChainRow], ContractMatchDiagnostics]:
    original = list(chain_rows)
    available_expiries = sorted({str(row.expiry_date) for row in original if row.expiry_date})
    available_strikes = sorted({float(row.strike) for row in original})
    requested_expiries = sorted(
        {parsed.isoformat() for value in (expiries or []) if (parsed := parse_date(value)) is not None}
    )
    requested_expiry_exists = None
    if requested_expiries:
        requested_expiry_exists = any(value in available_expiries for value in requested_expiries)

    selected = list(original)
    row_counts = {"initial_candidates": len(selected)}
    failure_stage = None

    if requested_expiries:
        selected = [row for row in selected if parse_date(row.expiry_date) and parse_date(row.expiry_date).isoformat() in requested_expiries]
    row_counts["after_expiry_filter"] = len(selected)
    if failure_stage is None and row_counts["initial_candidates"] > 0 and requested_expiries and not selected:
        failure_stage = "expiry_filter"

    right_text = clean_string(right).lower()
    if right_text in {"call", "put"}:
        selected = [row for row in selected if row.option_type == right_text]
    row_counts["after_right_filter"] = len(selected)
    if failure_stage is None and row_counts["after_expiry_filter"] > 0 and not selected:
        failure_stage = "right_filter"

    if min_strike is not None:
        selected = [row for row in selected if float(row.strike) >= float(min_strike)]
    row_counts["after_min_strike_filter"] = len(selected)
    if failure_stage is None and row_counts["after_right_filter"] > 0 and not selected and min_strike is not None:
        failure_stage = "min_strike_filter"

    if max_strike is not None:
        selected = [row for row in selected if float(row.strike) <= float(max_strike)]
    row_counts["after_max_strike_filter"] = len(selected)
    if failure_stage is None and row_counts["after_min_strike_filter"] > 0 and not selected and max_strike is not None:
        failure_stage = "max_strike_filter"

    explicit_strikes = {float(value) for value in (strikes or [])}
    if explicit_strikes:
        selected = [row for row in selected if float(row.strike) in explicit_strikes]
    row_counts["after_explicit_strike_filter"] = len(selected)
    if failure_stage is None and row_counts["after_max_strike_filter"] > 0 and not selected and explicit_strikes:
        failure_stage = "explicit_strike_filter"

    around_spot_skipped_reason = None
    if around_spot is not None and spot_price is None:
        around_spot_skipped_reason = "No usable underlying spot was available, so around-spot narrowing was skipped."
    elif around_spot is not None and spot_price is not None:
        by_expiry_and_type: dict[tuple[str, str], list[ChainRow]] = defaultdict(list)
        for row in selected:
            by_expiry_and_type[(str(row.expiry_date), row.option_type)].append(row)
        narrowed: list[ChainRow] = []
        for group_rows in by_expiry_and_type.values():
            group_rows = sorted(group_rows, key=lambda row: (abs(float(row.strike) - float(spot_price)), float(row.strike)))
            narrowed.extend(group_rows[: max(int(around_spot), 0)])
        selected = narrowed
    row_counts["after_around_spot_filter"] = len(selected)
    if failure_stage is None and row_counts["after_explicit_strike_filter"] > 0 and not selected and around_spot is not None and spot_price is not None:
        failure_stage = "around_spot_filter"

    selected = sorted(
        selected,
        key=lambda row: (str(row.expiry_date), float(row.strike), row.option_type, clean_string(row.exchange)),
    )
    if max_contracts is not None:
        selected = selected[: max(int(max_contracts), 0)]
    row_counts["final_selected"] = len(selected)
    if failure_stage is None and row_counts["after_around_spot_filter"] > 0 and not selected:
        failure_stage = "max_contracts_filter"
    if failure_stage is None and not selected and row_counts["initial_candidates"] == 0:
        failure_stage = "chain_rows_empty"

    diagnostics = ContractMatchDiagnostics(
        requested_expiries=requested_expiries,
        requested_right=right_text or "both",
        requested_min_strike=float(min_strike) if min_strike is not None else None,
        requested_max_strike=float(max_strike) if max_strike is not None else None,
        requested_strikes=sorted(explicit_strikes),
        around_spot=around_spot,
        max_contracts=max_contracts,
        requested_expiry_exists=requested_expiry_exists,
        available_expiries=available_expiries,
        nearest_expiries=_nearest_expiries(requested_expiries, available_expiries),
        available_strike_count=len(available_strikes),
        available_strike_sample=available_strikes[:10],
        spot_price_used=float(spot_price) if spot_price is not None else None,
        around_spot_skipped_reason=around_spot_skipped_reason,
        row_counts=row_counts,
        selected_contracts=[
            {
                "expiry_date": str(row.expiry_date),
                "strike": float(row.strike),
                "option_type": row.option_type,
                "exchange": row.exchange,
                "trading_class": row.trading_class,
            }
            for row in selected
        ],
        failure_stage=failure_stage,
    )
    return selected, diagnostics


def select_contract_candidates(
    chain_rows: Iterable[ChainRow],
    *,
    expiries: Iterable[str] | None = None,
    right: str = "both",
    min_strike: float | None = None,
    max_strike: float | None = None,
    strikes: Iterable[float] | None = None,
    around_spot: int | None = None,
    spot_price: float | None = None,
    max_contracts: int | None = None,
) -> list[ChainRow]:
    selected, _ = select_contract_candidates_with_diagnostics(
        chain_rows,
        expiries=expiries,
        right=right,
        min_strike=min_strike,
        max_strike=max_strike,
        strikes=strikes,
        around_spot=around_spot,
        spot_price=spot_price,
        max_contracts=max_contracts,
    )
    return selected
