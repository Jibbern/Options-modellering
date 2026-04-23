"""Delayed-only IBKR market-data fetchers for underlying and option quotes."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import time
from typing import Any

from ..utils import clean_string, parse_date
from .chains import fetch_option_chain, select_contract_candidates_with_diagnostics
from .connection import DelayedOnlyIbkrSession
from .contracts import build_option_contract, build_stock_contract, contract_to_dict, resolved_underlying_from_contract
from .models import (
    ConnectionSettings,
    ContractMatchDiagnostics,
    FullChainSnapshotFetchResult,
    OptionQuoteSnapshot,
    OptionSnapshotDiagnostics,
    OptionSnapshotFetchResult,
    UnderlyingQuoteSnapshot,
    normalize_market_data_mode,
    validate_effective_market_data_type,
)


def _warnings_from_result(result: dict[str, Any], missing_fields: list[str], mode: str) -> list[str]:
    warnings = [error["error_string"] for error in result.get("errors", [])]
    accepted, reason = validate_effective_market_data_type(result.get("market_data_type_code"))
    if not accepted:
        warnings.append(reason)
    if missing_fields:
        warnings.append(
            f"Delayed field availability was partial for this request: missing {', '.join(sorted(missing_fields))}."
        )
    if mode == "delayed_frozen":
        warnings.append("Delayed-frozen mode can surface last known delayed values after the market closes.")
    return sorted(dict.fromkeys(warnings))


def _strike_count_by_expiry(chain_rows) -> dict[str, int]:
    counts: dict[str, set[float]] = {}
    for row in chain_rows:
        expiry = parse_date(getattr(row, "expiry_date", None))
        if expiry is None:
            continue
        counts.setdefault(expiry.isoformat(), set()).add(float(row.strike))
    return {expiry: len(strikes) for expiry, strikes in sorted(counts.items())}


def _coverage_summary_from_quotes(quotes: list[OptionQuoteSnapshot]) -> dict[str, Any]:
    tracked_fields = ("bid", "ask", "implied_volatility", "volume", "open_interest")
    total = len(quotes)
    overall: dict[str, Any] = {}
    for field in tracked_fields:
        available_count = sum(1 for quote in quotes if getattr(quote, field) is not None)
        overall[field] = {
            "available_count": available_count,
            "missing_count": max(total - available_count, 0),
            "coverage_pct": round((available_count / total) * 100.0, 1) if total else 0.0,
        }
    return {
        "contract_count": total,
        "overall": overall,
    }


def _finalize_option_snapshot_result(
    snapshots: list[OptionQuoteSnapshot],
    *,
    diagnostics: OptionSnapshotDiagnostics | None = None,
) -> OptionSnapshotFetchResult:
    missing_fields = sorted({field for quote in snapshots for field in quote.missing_fields})
    missing_field_counts = {
        field: sum(1 for quote in snapshots if field in quote.missing_fields)
        for field in missing_fields
    }
    final_diagnostics = replace(
        diagnostics or OptionSnapshotDiagnostics(),
        selected_contract_count=len(snapshots),
        final_selected_expiries=sorted({quote.expiry_date for quote in snapshots if quote.expiry_date}),
        final_selected_strikes=sorted({float(quote.strike) for quote in snapshots}),
        final_selected_exchanges=sorted({clean_string(quote.exchange).upper() for quote in snapshots if clean_string(quote.exchange)}),
        final_selected_trading_classes=sorted({clean_string(quote.trading_class).upper() for quote in snapshots if clean_string(quote.trading_class)}),
        delayed_field_summary={
            "missing_fields": missing_fields,
            "missing_field_counts": missing_field_counts,
            "warnings": sorted({warning for quote in snapshots for warning in quote.warnings}),
            "effective_market_data_types": sorted({quote.market_data_type_code for quote in snapshots if quote.market_data_type_code is not None}),
            "coverage_summary": _coverage_summary_from_quotes(snapshots),
        },
        failure_stage=(diagnostics.failure_stage if diagnostics is not None and not snapshots else None),
    )
    return OptionSnapshotFetchResult(quotes=snapshots, diagnostics=final_diagnostics)


def _build_quote_snapshot(
    *,
    ticker: str,
    row,
    requested_contract,
    resolved_contract,
    result: dict[str, Any],
    underlying: UnderlyingQuoteSnapshot,
    market_data_mode: str,
    exchange: str,
    settings: ConnectionSettings,
) -> OptionQuoteSnapshot:
    ticks = result.get("ticks", {})
    bid = ticks.get("bid")
    ask = ticks.get("ask")
    last = ticks.get("last")
    close = ticks.get("close")
    mark = ticks.get("mark")
    mid = None
    if bid is not None and ask is not None:
        mid = round((float(bid) + float(ask)) / 2.0, 6)
    elif mark is not None:
        mid = float(mark)
    missing_fields = [
        name
        for name, value in {
            "bid": bid,
            "ask": ask,
            "last": last,
            "open_interest": ticks.get("open_interest"),
            "implied_volatility": ticks.get("implied_volatility"),
        }.items()
        if value is None
    ]
    warnings = _warnings_from_result(result, missing_fields, market_data_mode)
    return OptionQuoteSnapshot(
        ticker=clean_string(ticker).upper(),
        snapshot_timestamp=datetime.now(timezone.utc),
        market_data_mode=market_data_mode,
        market_data_type_code=result.get("market_data_type_code"),
        expiry_date=parse_date(row.expiry_date).isoformat(),
        strike=float(row.strike),
        option_type=row.option_type,
        conid=getattr(resolved_contract, "conId", None),
        local_symbol=getattr(resolved_contract, "localSymbol", None),
        trading_class=row.trading_class,
        exchange=row.exchange or exchange,
        bid=float(bid) if bid is not None else None,
        ask=float(ask) if ask is not None else None,
        last=float(last) if last is not None else None,
        mid=float(mid) if mid is not None else None,
        mark=float(mark) if mark is not None else None,
        close=float(close) if close is not None else None,
        volume=float(ticks.get("option_volume")) if ticks.get("option_volume") is not None else float(ticks.get("volume")) if ticks.get("volume") is not None else None,
        open_interest=float(ticks.get("open_interest")) if ticks.get("open_interest") is not None else None,
        implied_volatility=float(ticks.get("implied_volatility")) if ticks.get("implied_volatility") is not None else None,
        historical_volatility=float(ticks.get("historical_volatility")) if ticks.get("historical_volatility") is not None else None,
        delta=float(ticks.get("delta")) if ticks.get("delta") is not None else None,
        gamma=float(ticks.get("gamma")) if ticks.get("gamma") is not None else None,
        theta=float(ticks.get("theta")) if ticks.get("theta") is not None else None,
        vega=float(ticks.get("vega")) if ticks.get("vega") is not None else None,
        option_price=float(ticks.get("option_price")) if ticks.get("option_price") is not None else None,
        pv_dividend=float(ticks.get("pv_dividend")) if ticks.get("pv_dividend") is not None else None,
        under_price=float(ticks.get("under_price")) if ticks.get("under_price") is not None else underlying.mid or underlying.mark or underlying.last or underlying.close,
        source="ibkr",
        warnings=warnings,
        missing_fields=missing_fields,
        connection=settings,
        requested_contract=contract_to_dict(requested_contract),
        resolved_contract=contract_to_dict(resolved_contract),
    )


def _quote_information_score(quote: OptionQuoteSnapshot) -> int:
    return sum(
        1
        for value in (
            quote.bid,
            quote.ask,
            quote.implied_volatility,
            quote.volume,
            quote.open_interest,
        )
        if value is not None
    )


def _quote_is_sparse(quote: OptionQuoteSnapshot) -> bool:
    return _quote_information_score(quote) <= 1


def _merge_quote_snapshots(original: OptionQuoteSnapshot, retry: OptionQuoteSnapshot) -> OptionQuoteSnapshot:
    merged_values: dict[str, Any] = {}
    for field in (
        "market_data_type_code",
        "bid",
        "ask",
        "last",
        "mid",
        "mark",
        "close",
        "volume",
        "open_interest",
        "implied_volatility",
        "historical_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        "option_price",
        "pv_dividend",
        "under_price",
    ):
        retry_value = getattr(retry, field)
        merged_values[field] = retry_value if retry_value is not None else getattr(original, field)
    if merged_values.get("mid") is None and merged_values.get("bid") is not None and merged_values.get("ask") is not None:
        merged_values["mid"] = round((float(merged_values["bid"]) + float(merged_values["ask"])) / 2.0, 6)
    merged_missing_fields = [
        field
        for field, value in {
            "bid": merged_values.get("bid"),
            "ask": merged_values.get("ask"),
            "last": merged_values.get("last"),
            "open_interest": merged_values.get("open_interest"),
            "implied_volatility": merged_values.get("implied_volatility"),
        }.items()
        if value is None
    ]
    preferred_timestamp = retry.snapshot_timestamp if _quote_information_score(retry) > _quote_information_score(original) else original.snapshot_timestamp
    return replace(
        original,
        snapshot_timestamp=preferred_timestamp,
        warnings=sorted(dict.fromkeys([*original.warnings, *retry.warnings])),
        missing_fields=merged_missing_fields,
        **merged_values,
    )


def _collect_batch_snapshots(
    session: DelayedOnlyIbkrSession,
    *,
    ticker: str,
    market_data_mode: str,
    contexts: list[dict[str, Any]],
    underlying: UnderlyingQuoteSnapshot,
    exchange: str,
    settings: ConnectionSettings,
    snapshot_wait_seconds: float,
) -> list[OptionQuoteSnapshot]:
    batch_results = session.collect_market_snapshots(
        [context["resolved_contract"] for context in contexts],
        market_data_mode=market_data_mode,
        generic_tick_list="100,101,104,106,221",
        wait_seconds=snapshot_wait_seconds,
    )
    snapshots: list[OptionQuoteSnapshot] = []
    for context, result in zip(contexts, batch_results, strict=False):
        snapshots.append(
            _build_quote_snapshot(
                ticker=ticker,
                row=context["row"],
                requested_contract=context["requested_contract"],
                resolved_contract=context["resolved_contract"],
                result=result,
                underlying=underlying,
                market_data_mode=market_data_mode,
                exchange=exchange,
                settings=settings,
            )
        )
    return snapshots


def _collect_option_quotes_from_chain_rows(
    ticker: str,
    *,
    settings: ConnectionSettings,
    market_data_mode: str,
    chain_rows,
    underlying: UnderlyingQuoteSnapshot,
    timeout: float,
    exchange: str,
    currency: str,
    diagnostics: OptionSnapshotDiagnostics | None = None,
    resolve_contract_details: bool = True,
    snapshot_wait_seconds: float | None = None,
    retry_sparse_quotes_once: bool = False,
    sparse_retry_wait_seconds: float = 2.0,
) -> OptionSnapshotFetchResult:
    mode = normalize_market_data_mode(market_data_mode)
    snapshot_wait = snapshot_wait_seconds if snapshot_wait_seconds is not None else max(2.0, timeout / 2.0)
    request_contexts: list[dict[str, Any]] = []
    with DelayedOnlyIbkrSession(settings, timeout=timeout) as session:
        for row in chain_rows:
            requested_contract = build_option_contract(
                ticker,
                expiry_date=parse_date(row.expiry_date).isoformat(),
                strike=float(row.strike),
                option_type=row.option_type,
                exchange=row.exchange or exchange,
                currency=row.currency or currency,
                trading_class=row.trading_class,
                multiplier=row.multiplier,
            )
            detail = requested_contract
            if resolve_contract_details:
                details = session.request_contract_details(requested_contract, timeout=timeout)
                detail = details[0].contract if details else requested_contract
            request_contexts.append(
                {
                    "row": row,
                    "requested_contract": requested_contract,
                    "resolved_contract": detail,
                }
            )
        snapshots = _collect_batch_snapshots(
            session,
            ticker=ticker,
            market_data_mode=mode,
            contexts=request_contexts,
            underlying=underlying,
            exchange=exchange,
            settings=settings,
            snapshot_wait_seconds=snapshot_wait,
        )
        if retry_sparse_quotes_once:
            sparse_indices = [index for index, quote in enumerate(snapshots) if _quote_is_sparse(quote)]
            if sparse_indices:
                if sparse_retry_wait_seconds > 0:
                    time.sleep(sparse_retry_wait_seconds)
                retry_contexts = [request_contexts[index] for index in sparse_indices]
                retry_snapshots = _collect_batch_snapshots(
                    session,
                    ticker=ticker,
                    market_data_mode=mode,
                    contexts=retry_contexts,
                    underlying=underlying,
                    exchange=exchange,
                    settings=settings,
                    snapshot_wait_seconds=snapshot_wait,
                )
                retry_iter = iter(retry_snapshots)
                merged_snapshots: list[OptionQuoteSnapshot] = []
                for index, quote in enumerate(snapshots):
                    if index in sparse_indices:
                        merged_snapshots.append(_merge_quote_snapshots(quote, next(retry_iter)))
                    else:
                        merged_snapshots.append(quote)
                snapshots = merged_snapshots

    return _finalize_option_snapshot_result(snapshots, diagnostics=diagnostics)


def _combine_option_snapshot_batches(
    batch_results: list[OptionSnapshotFetchResult],
    *,
    diagnostics: OptionSnapshotDiagnostics,
) -> OptionSnapshotFetchResult:
    snapshots: list[OptionQuoteSnapshot] = []
    for result in batch_results:
        snapshots.extend(result.quotes)
    return _finalize_option_snapshot_result(snapshots, diagnostics=diagnostics)


def _resolve_underlying_contract(
    session: DelayedOnlyIbkrSession,
    *,
    ticker: str,
    exchange: str,
    currency: str,
    primary_exchange: str | None = None,
    timeout: float,
):
    contract = build_stock_contract(
        ticker,
        exchange=exchange,
        currency=currency,
        primary_exchange=primary_exchange,
    )
    details = session.request_contract_details(contract, timeout=timeout)
    if not details:
        raise LookupError(f"IBKR returned no underlying contract details for {clean_string(ticker).upper()}.")
    resolved_contract = details[0].contract
    return resolved_contract, resolved_underlying_from_contract(resolved_contract)


def fetch_underlying_snapshot(
    ticker: str,
    *,
    settings: ConnectionSettings,
    market_data_mode: str = "delayed",
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: str | None = None,
    timeout: float = 8.0,
) -> UnderlyingQuoteSnapshot:
    mode = normalize_market_data_mode(market_data_mode)
    with DelayedOnlyIbkrSession(settings, timeout=timeout) as session:
        resolved_contract, resolved_underlying = _resolve_underlying_contract(
            session,
            ticker=ticker,
            exchange=exchange,
            currency=currency,
            primary_exchange=primary_exchange,
            timeout=timeout,
        )
        result = session.collect_market_snapshot(
            resolved_contract,
            market_data_mode=mode,
            generic_tick_list="221",
            wait_seconds=timeout,
        )
    ticks = result.get("ticks", {})
    bid = ticks.get("bid")
    ask = ticks.get("ask")
    last = ticks.get("last")
    close = ticks.get("close")
    mark = ticks.get("mark")
    mid = None
    if bid is not None and ask is not None:
        mid = round((float(bid) + float(ask)) / 2.0, 6)
    elif mark is not None:
        mid = float(mark)
    missing_fields = [name for name, value in {"bid": bid, "ask": ask, "last": last, "close": close}.items() if value is None]
    warnings = _warnings_from_result(result, missing_fields, mode)
    return UnderlyingQuoteSnapshot(
        ticker=clean_string(ticker).upper(),
        snapshot_timestamp=datetime.now(timezone.utc),
        market_data_mode=mode,
        market_data_type_code=result.get("market_data_type_code"),
        bid=float(bid) if bid is not None else None,
        ask=float(ask) if ask is not None else None,
        last=float(last) if last is not None else None,
        close=float(close) if close is not None else None,
        mid=float(mid) if mid is not None else None,
        mark=float(mark) if mark is not None else None,
        exchange=clean_string(exchange).upper() or "SMART",
        primary_exchange=clean_string(primary_exchange).upper() if primary_exchange else None,
        currency=clean_string(currency).upper() or "USD",
        source="ibkr",
        warnings=warnings,
        missing_fields=missing_fields,
        connection=settings,
        resolved_underlying=resolved_underlying,
    )


def fetch_option_snapshots(
    ticker: str,
    *,
    settings: ConnectionSettings,
    market_data_mode: str = "delayed",
    expiries: list[str] | None = None,
    right: str = "both",
    min_strike: float | None = None,
    max_strike: float | None = None,
    strikes: list[float] | None = None,
    around_spot: int | None = None,
    max_contracts: int | None = None,
    exchange: str = "SMART",
    currency: str = "USD",
    timeout: float = 8.0,
    include_all_exchanges: bool = False,
) -> OptionSnapshotFetchResult:
    mode = normalize_market_data_mode(market_data_mode)
    underlying = fetch_underlying_snapshot(
        ticker,
        settings=settings,
        market_data_mode=mode,
        exchange=exchange,
        currency=currency,
        timeout=max(3.0, timeout / 2.0),
    )
    chain_result = fetch_option_chain(
        ticker,
        settings=settings,
        market_data_mode=mode,
        exchange=exchange,
        currency=currency,
        timeout=timeout,
        include_all_exchanges=include_all_exchanges,
    )
    selected, match_diagnostics = select_contract_candidates_with_diagnostics(
        chain_result.rows,
        expiries=expiries,
        right=right,
        min_strike=min_strike,
        max_strike=max_strike,
        strikes=strikes,
        around_spot=around_spot,
        spot_price=underlying.mid or underlying.mark or underlying.last or underlying.close,
        max_contracts=max_contracts,
    )

    if not selected:
        diagnostics = OptionSnapshotDiagnostics(
            underlying_snapshot=underlying.to_record(),
            resolved_underlying=underlying.resolved_underlying,
            chain_diagnostics=chain_result.diagnostics,
            contract_match=match_diagnostics,
            snapshot_scope="filtered_slice",
            discovered_expiries=sorted({parse_date(row.expiry_date).isoformat() for row in selected if parse_date(row.expiry_date) is not None}),
            strike_count_by_expiry=_strike_count_by_expiry(selected),
            attempted_contract_count=0,
            selected_contract_count=0,
            final_selected_expiries=[],
            final_selected_strikes=[],
            final_selected_exchanges=[],
            final_selected_trading_classes=[],
            delayed_field_summary={"missing_fields": [], "warnings": []},
            failure_stage=chain_result.diagnostics.failure_stage or match_diagnostics.failure_stage,
        )
        return OptionSnapshotFetchResult(quotes=[], diagnostics=diagnostics)

    diagnostics = OptionSnapshotDiagnostics(
        underlying_snapshot=underlying.to_record(),
        resolved_underlying=underlying.resolved_underlying,
        chain_diagnostics=chain_result.diagnostics,
        contract_match=match_diagnostics,
        snapshot_scope="filtered_slice",
        discovered_expiries=sorted({parse_date(row.expiry_date).isoformat() for row in selected if parse_date(row.expiry_date) is not None}),
        strike_count_by_expiry=_strike_count_by_expiry(selected),
        attempted_contract_count=len(selected),
        failure_stage=None,
    )
    return _collect_option_quotes_from_chain_rows(
        ticker,
        settings=settings,
        market_data_mode=mode,
        chain_rows=selected,
        underlying=underlying,
        timeout=timeout,
        exchange=exchange,
        currency=currency,
        diagnostics=diagnostics,
        resolve_contract_details=True,
    )


def fetch_full_chain_snapshot(
    ticker: str,
    *,
    settings: ConnectionSettings,
    market_data_mode: str = "delayed",
    exchange: str = "SMART",
    currency: str = "USD",
    timeout: float = 8.0,
    per_expiry_timeout: float = 90.0,
    retry_sparse_quotes_once: bool = True,
    sparse_retry_wait_seconds: float = 3.0,
    include_all_exchanges: bool = False,
) -> FullChainSnapshotFetchResult:
    mode = normalize_market_data_mode(market_data_mode)
    underlying = fetch_underlying_snapshot(
        ticker,
        settings=settings,
        market_data_mode=mode,
        exchange=exchange,
        currency=currency,
        timeout=max(3.0, timeout / 2.0),
    )
    chain_result = fetch_option_chain(
        ticker,
        settings=settings,
        market_data_mode=mode,
        exchange=exchange,
        currency=currency,
        timeout=timeout,
        include_all_exchanges=include_all_exchanges,
    )
    discovered_expiries = sorted(
        {
            parse_date(row.expiry_date).isoformat()
            for row in chain_result.rows
            if parse_date(row.expiry_date) is not None
        }
    )
    diagnostics = OptionSnapshotDiagnostics(
        underlying_snapshot=underlying.to_record(),
        resolved_underlying=underlying.resolved_underlying,
        chain_diagnostics=chain_result.diagnostics,
        contract_match=ContractMatchDiagnostics(
            requested_expiries=discovered_expiries,
            requested_right="both",
            row_counts={"final_selected_contracts": len(chain_result.rows)},
        ),
        snapshot_scope="full_chain",
        discovered_expiries=discovered_expiries,
        strike_count_by_expiry=_strike_count_by_expiry(chain_result.rows),
        attempted_contract_count=len(chain_result.rows),
        failure_stage=chain_result.diagnostics.failure_stage if not chain_result.rows else None,
    )
    if not chain_result.rows:
        option_snapshot = OptionSnapshotFetchResult(quotes=[], diagnostics=diagnostics)
    else:
        expiry_batches: dict[str, list[Any]] = {}
        for row in chain_result.rows:
            expiry = parse_date(row.expiry_date)
            expiry_key = expiry.isoformat() if expiry is not None else "unknown"
            expiry_batches.setdefault(expiry_key, []).append(row)
        batch_results: list[OptionSnapshotFetchResult] = []
        for _expiry, batch_rows in sorted(expiry_batches.items()):
            batch_results.append(
                _collect_option_quotes_from_chain_rows(
                    ticker,
                    settings=settings,
                    market_data_mode=mode,
                    chain_rows=batch_rows,
                    underlying=underlying,
                    timeout=timeout,
                    exchange=exchange,
                    currency=currency,
                    diagnostics=diagnostics,
                    resolve_contract_details=False,
                    snapshot_wait_seconds=per_expiry_timeout,
                    retry_sparse_quotes_once=retry_sparse_quotes_once,
                    sparse_retry_wait_seconds=sparse_retry_wait_seconds,
                )
            )
        option_snapshot = _combine_option_snapshot_batches(batch_results, diagnostics=diagnostics)
    return FullChainSnapshotFetchResult(
        underlying=underlying,
        chain=chain_result,
        option_snapshot=option_snapshot,
    )
