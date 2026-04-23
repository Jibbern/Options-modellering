"""Historical replay / case-study computation for local option ideas."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..io import load_chain
from ..rates import get_risk_free_rate
from ..research_metadata import get_nearest_event
from ..snapshots import snapshot_slice_for_expiry
from ..strategies import StrategyPosition, build_strategy
from ..utils import CONTRACT_MULTIPLIER, clean_string, finite_or_none, parse_date
from .scenario import load_snapshot_chain


CHECKPOINT_ORDER = ["entry", "1w", "1m", "event", "post_event", "expiry"]
BULLISH_STRATEGIES = {
    "long_stock",
    "long_call",
    "bull_call_spread",
    "covered_call",
    "cash_secured_put",
}
BEARISH_STRATEGIES = {"long_put", "bear_put_spread"}


@dataclass
class HistoricalReplayComputation:
    """Computed rows, plots, and metadata for one historical replay case study."""

    ticker: str
    snapshot_date: date
    expiry_date: date
    strategy_name: str
    source_snapshot_file: str
    premium_mode: str
    comparison_capital: float
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    resolved_metadata: dict[str, Any]
    research_context: dict[str, Any]
    warnings: list[str]
    status: str
    shareability_status: str
    valuation_source_rollup: dict[str, int]
    available_checkpoints: list[str]
    what_this_case_shows: str
    replay_defaults: dict[str, Any]
    case_summary: pd.DataFrame
    checkpoint_replay: pd.DataFrame
    expected_move_vs_actual: pd.DataFrame
    driver_decomposition: pd.DataFrame
    compare_vs_stock: pd.DataFrame
    local_history: pd.DataFrame
    position: StrategyPosition = field(repr=False)


@dataclass
class ReplayEventContext:
    """Lightweight event-context payload used by replay analysis."""

    summary: pd.DataFrame
    warnings: list[str]


def make_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return float(numeric)


def _prepared_price_history(ticker: str, data_root: str | Path | None = None) -> pd.DataFrame:
    from ..prices.price_store import load_price_history

    history = load_price_history(ticker, data_root)
    if history.empty:
        return history
    frame = history.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.dropna(subset=["date", "close"]).reset_index(drop=True)


def _latest_close_on_or_before(history: pd.DataFrame, target_date) -> dict[str, Any] | None:
    target = parse_date(target_date)
    if history.empty or target is None:
        return None
    eligible = history.loc[history["date"] <= pd.Timestamp(target)].copy()
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    return {
        "matched_date": pd.Timestamp(row["date"]).date().isoformat(),
        "close": float(row["close"]),
        "source": row.get("source"),
    }


def _latest_close_before(history: pd.DataFrame, target_date) -> dict[str, Any] | None:
    target = parse_date(target_date)
    if history.empty or target is None:
        return None
    eligible = history.loc[history["date"] < pd.Timestamp(target)].copy()
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    return {
        "matched_date": pd.Timestamp(row["date"]).date().isoformat(),
        "close": float(row["close"]),
        "source": row.get("source"),
    }


def _next_close_on_or_after(history: pd.DataFrame, target_date) -> dict[str, Any] | None:
    target = parse_date(target_date)
    if history.empty or target is None:
        return None
    eligible = history.loc[history["date"] >= pd.Timestamp(target)].copy()
    if eligible.empty:
        return None
    row = eligible.iloc[0]
    return {
        "matched_date": pd.Timestamp(row["date"]).date().isoformat(),
        "close": float(row["close"]),
        "source": row.get("source"),
    }


def _next_close_after(history: pd.DataFrame, target_date) -> dict[str, Any] | None:
    target = parse_date(target_date)
    if history.empty or target is None:
        return None
    eligible = history.loc[history["date"] > pd.Timestamp(target)].copy()
    if eligible.empty:
        return None
    row = eligible.iloc[0]
    return {
        "matched_date": pd.Timestamp(row["date"]).date().isoformat(),
        "close": float(row["close"]),
        "source": row.get("source"),
    }


def _normalize_event_time(value: object) -> str:
    text = clean_string(value).lower().replace(" ", "_")
    if text in {"after_close", "after_market_close", "after_hours", "amc"}:
        return "after_close"
    if text in {"before_open", "before_market_open", "before_bell", "bmo"}:
        return "before_open"
    return "unknown"


def _event_window(history: pd.DataFrame, event_date, event_time: object) -> dict[str, object]:
    normalized_time = _normalize_event_time(event_time)
    event_day = parse_date(event_date)
    if event_day is None:
        return {"event_time_normalized": normalized_time, "matched": False}
    if normalized_time == "after_close":
        pre = _latest_close_on_or_before(history, event_day)
        post = _next_close_after(history, event_day)
    elif normalized_time == "before_open":
        pre = _latest_close_before(history, event_day)
        post = _latest_close_on_or_before(history, event_day)
    else:
        pre = _latest_close_before(history, event_day)
        post = _next_close_on_or_after(history, event_day)
    if pre is None or post is None:
        return {
            "event_time_normalized": normalized_time,
            "matched": False,
            "pre_event": pre,
            "post_event": post,
        }
    signed_move_abs = post["close"] - pre["close"]
    signed_move_pct = signed_move_abs / pre["close"] if pre["close"] else None
    return {
        "event_time_normalized": normalized_time,
        "matched": True,
        "pre_event": pre,
        "post_event": post,
        "signed_move_abs": signed_move_abs,
        "signed_move_pct": signed_move_pct,
        "realized_gap_abs": abs(signed_move_abs),
        "realized_gap_pct": abs(signed_move_pct) if signed_move_pct is not None else None,
    }


def _replay_event_context(
    ticker: str,
    *,
    snapshot_date,
    expiry_date=None,
    data_root: str | Path | None = None,
) -> ReplayEventContext:
    """Resolve nearest-event context for replay without depending on removed workflow layers."""

    snapshot = parse_date(snapshot_date)
    expiry = parse_date(expiry_date) if expiry_date is not None else None
    if snapshot is None:
        raise ValueError(f"snapshot_date must be a valid date, got: {snapshot_date!r}")
    nearest_event = get_nearest_event(ticker, snapshot, expiry_date=expiry, data_root=data_root)
    if not nearest_event.get("matched"):
        return ReplayEventContext(
            summary=pd.DataFrame(
                [
                    {
                        "snapshot_date": snapshot.isoformat(),
                        "expiry_date": expiry.isoformat() if expiry else None,
                        "event_date": None,
                        "event_time": None,
                        "event_type": None,
                        "days_to_event": None,
                        "event_in_horizon": False,
                        "realized_gap_pct": None,
                    }
                ]
            ),
            warnings=["No local upcoming event was available on or after the requested snapshot date."],
        )
    history = _prepared_price_history(ticker, data_root)
    window = _event_window(history, nearest_event.get("event_date"), nearest_event.get("event_time"))
    event_date = parse_date(nearest_event.get("event_date"))
    payload = {
        "snapshot_date": snapshot.isoformat(),
        "expiry_date": expiry.isoformat() if expiry else None,
        "event_date": nearest_event.get("event_date"),
        "event_time": nearest_event.get("event_time"),
        "event_type": nearest_event.get("event_type"),
        "days_to_event": (event_date - snapshot).days if event_date else None,
        "event_in_horizon": nearest_event.get("occurs_before_expiry"),
        "event_time_normalized": window.get("event_time_normalized"),
        "pre_event_date": (window.get("pre_event") or {}).get("matched_date"),
        "pre_event_close": (window.get("pre_event") or {}).get("close"),
        "post_event_date": (window.get("post_event") or {}).get("matched_date"),
        "post_event_close": (window.get("post_event") or {}).get("close"),
        "realized_gap_abs": window.get("realized_gap_abs"),
        "realized_gap_pct": window.get("realized_gap_pct"),
        "signed_move_abs": window.get("signed_move_abs"),
        "signed_move_pct": window.get("signed_move_pct"),
    }
    warnings: list[str] = []
    if not window.get("matched"):
        warnings.append("The nearest event was found, but there were not enough local prices to compute the event window move.")
    return ReplayEventContext(summary=pd.DataFrame([payload]), warnings=warnings)


def _average_entry_iv(position: StrategyPosition) -> float | None:
    values = [float(leg.base_iv) for leg in position.option_legs if leg.base_iv is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _strategy_build_kwargs(
    strategy_name: str,
    *,
    spot_price: float | None,
    premium_mode: str,
    contract_selector: dict[str, Any] | None,
    long_selector: dict[str, Any] | None,
    short_selector: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "spot_price": spot_price,
        "premium_mode": premium_mode,
    }
    if strategy_name in {"long_call", "long_put"}:
        payload["contract_selector"] = dict(contract_selector or {})
    elif strategy_name in {"bull_call_spread", "bear_put_spread"}:
        payload["long_selector"] = dict(long_selector or contract_selector or {})
        payload["short_selector"] = dict(short_selector or {})
    elif strategy_name in {"covered_call", "cash_secured_put"}:
        payload["contract_selector"] = dict(short_selector or contract_selector or {})
    return payload


def _exact_chain_value(
    position: StrategyPosition,
    *,
    target_date,
    data_root: str | Path | None = None,
) -> dict[str, Any] | None:
    expiry = position.expiry_date
    if expiry is None:
        return None
    try:
        slice_row = snapshot_slice_for_expiry(position.ticker, target_date, expiry, data_root)
    except ValueError:
        return None
    chain = load_chain(
        slice_row["file_path"],
        prices_data_root=data_root,
        rates_data_root=data_root,
        research_data_root=data_root,
    )
    if chain.spot_price is None:
        return None
    total = 0.0
    observed_ivs: list[float] = []
    for leg in position.legs:
        if leg.asset_type == "stock":
            total += int(leg.quantity) * float(chain.spot_price)
            continue
        matches = chain.contracts.loc[
            (chain.contracts["option_type"] == leg.option_type)
            & (chain.contracts["strike"] == float(leg.strike))
        ].copy()
        if matches.empty:
            return None
        contract = chain.to_contract(matches.iloc[0])
        total += int(leg.quantity) * CONTRACT_MULTIPLIER * contract.premium(mode=position.premium_mode)
        if contract.iv is not None:
            observed_ivs.append(float(contract.iv))
    return {
        "value": float(total),
        "source": "exact_local_later_chain",
        "matched_date": parse_date(target_date).isoformat(),
        "spot": float(chain.spot_price),
        "spot_matched_date": chain.metadata.spot_price_matched_date.isoformat() if chain.metadata.spot_price_matched_date else None,
        "spot_source": chain.metadata.spot_price_source,
        "iv_assumption": float(sum(observed_ivs) / len(observed_ivs)) if observed_ivs else None,
        "rate_assumption": float(chain.metadata.risk_free_rate),
        "source_snapshot_file": slice_row["file_path"],
    }


def _modeled_value(
    position: StrategyPosition,
    *,
    target_date,
    data_root: str | Path | None = None,
) -> dict[str, Any] | None:
    spot_context = _latest_close_on_or_before(_prepared_price_history(position.ticker, data_root), target_date)
    if spot_context is None:
        return None
    current_rate = position.risk_free_rate
    if position.expiry_date is not None and parse_date(target_date) is not None:
        try:
            current_rate = get_risk_free_rate(parse_date(target_date), position.expiry_date, data_root=data_root).rate_decimal
        except (FileNotFoundError, LookupError):
            current_rate = position.risk_free_rate
    value = float(
        position.mark_to_market_value(
            [float(spot_context["close"])],
            valuation_date=parse_date(target_date),
            iv_shift=0.0,
            risk_free_rate=current_rate,
            dividend_yield=position.dividend_yield,
        )[0]
    )
    return {
        "value": float(value),
        "source": "approximate_model_estimate",
        "matched_date": parse_date(target_date).isoformat(),
        "spot": float(spot_context["close"]),
        "spot_matched_date": spot_context["matched_date"],
        "spot_source": spot_context.get("source"),
        "iv_assumption": _average_entry_iv(position),
        "rate_assumption": float(current_rate),
        "iv_assumption_source": "entry_iv_static",
    }


def _decomposition_rows(
    position: StrategyPosition,
    *,
    checkpoint_label: str,
    valuation_date: date,
    final_spot: float | None,
    selected_value: float | None,
    selected_source: str,
    observed_iv: float | None,
    current_rate: float | None,
    event_context_note: str | None,
    event_gap_pct: float | None,
) -> list[dict[str, Any]]:
    if final_spot is None or selected_value is None:
        return []
    entry_rate = position.risk_free_rate
    entry_iv = _average_entry_iv(position)
    spot_only_value = float(
        position.mark_to_market_value(
            [final_spot],
            valuation_date=position.snapshot_date,
            iv_shift=0.0,
            risk_free_rate=entry_rate,
            dividend_yield=position.dividend_yield,
        )[0]
    )
    time_value = float(
        position.mark_to_market_value(
            [final_spot],
            valuation_date=valuation_date,
            iv_shift=0.0,
            risk_free_rate=entry_rate,
            dividend_yield=position.dividend_yield,
        )[0]
    )
    iv_shift = 0.0 if observed_iv is None or entry_iv is None else float(observed_iv) - float(entry_iv)
    iv_value = float(
        position.mark_to_market_value(
            [final_spot],
            valuation_date=valuation_date,
            iv_shift=iv_shift,
            risk_free_rate=entry_rate,
            dividend_yield=position.dividend_yield,
        )[0]
    )
    effective_rate = float(current_rate if current_rate is not None else entry_rate)
    rate_value = float(
        position.mark_to_market_value(
            [final_spot],
            valuation_date=valuation_date,
            iv_shift=iv_shift,
            risk_free_rate=effective_rate,
            dividend_yield=position.dividend_yield,
        )[0]
    )
    rows = [
        {"checkpoint": checkpoint_label, "component": "stock_move_effect", "effect": spot_only_value - float(position.initial_outlay)},
        {"checkpoint": checkpoint_label, "component": "time_decay_effect", "effect": time_value - spot_only_value},
        {"checkpoint": checkpoint_label, "component": "iv_change_effect", "effect": iv_value - time_value},
        {"checkpoint": checkpoint_label, "component": "rate_effect", "effect": rate_value - iv_value},
        {"checkpoint": checkpoint_label, "component": "structure_residual_effect", "effect": float(selected_value) - rate_value},
    ]
    for row in rows:
        row["valuation_source"] = selected_source
        row["event_context_note"] = event_context_note
        row["event_gap_pct"] = event_gap_pct
        row["matched_date"] = valuation_date.isoformat()
    return rows


def _checkpoint_spec_rows(
    position: StrategyPosition,
    *,
    event_result,
) -> list[dict[str, Any]]:
    expiry = position.expiry_date
    if expiry is None:
        raise ValueError("Historical replay requires an option expiry date.")
    rows = [
        {
            "checkpoint": "entry",
            "requested_date": position.snapshot_date.isoformat(),
            "target_date": position.snapshot_date,
            "clamped_to_expiry": False,
            "event_context_note": "Entry snapshot.",
        }
    ]
    for label, days in [("1w", 7), ("1m", 30)]:
        requested = position.snapshot_date + timedelta(days=days)
        target_date = min(requested, expiry)
        rows.append(
            {
                "checkpoint": label,
                "requested_date": requested.isoformat(),
                "target_date": target_date,
                "clamped_to_expiry": target_date != requested,
                "event_context_note": "Calendar replay checkpoint.",
            }
        )
    if event_result.summary is not None and not event_result.summary.empty:
        event_row = event_result.summary.iloc[0].to_dict()
        event_date = parse_date(event_row.get("event_date"))
        post_event_date = parse_date(event_row.get("post_event_date"))
        if event_date is not None and event_date >= position.snapshot_date and event_date <= expiry:
            rows.append(
                {
                    "checkpoint": "event",
                    "requested_date": event_date.isoformat(),
                    "target_date": event_date,
                    "clamped_to_expiry": False,
                    "event_context_note": "Replay checkpoint anchored to the event date. For after-close events this is the pre-event close.",
                }
            )
        if post_event_date is not None and post_event_date >= position.snapshot_date and post_event_date <= expiry:
            rows.append(
                {
                    "checkpoint": "post_event",
                    "requested_date": post_event_date.isoformat(),
                    "target_date": post_event_date,
                    "clamped_to_expiry": False,
                    "event_context_note": "Replay checkpoint anchored to the first matched post-event close.",
                }
            )
    rows.append(
        {
            "checkpoint": "expiry",
            "requested_date": expiry.isoformat(),
            "target_date": expiry,
            "clamped_to_expiry": False,
            "event_context_note": "Expiry outcome checkpoint.",
        }
    )
    order = {label: index for index, label in enumerate(CHECKPOINT_ORDER)}
    return sorted(rows, key=lambda item: order.get(item["checkpoint"], 99))


def _normalize_strategy_label(name: str) -> str:
    return clean_string(name).lower().replace("-", "_")


def _compare_vs_stock_rows(
    position: StrategyPosition,
    checkpoint_rows: pd.DataFrame,
    *,
    comparison_capital: float,
) -> pd.DataFrame:
    if checkpoint_rows.empty:
        return pd.DataFrame()
    stock_capital = position.entry_spot * CONTRACT_MULTIPLIER
    strategy_scale = (comparison_capital / position.capital_required) if comparison_capital > 0 and position.capital_required > 0 else 1.0
    stock_scale = (comparison_capital / stock_capital) if stock_capital else 1.0
    affordable_units = int(comparison_capital // position.capital_required) if comparison_capital > 0 and position.capital_required > 0 else None
    fully_implementable = affordable_units is not None and affordable_units >= 1 if position.capital_required > 0 else None
    budget_note = None
    if position.capital_required > 0:
        if affordable_units is not None and affordable_units >= 1:
            budget_note = f"At ${comparison_capital:,.0f}, this strategy funds {affordable_units} whole unit" + ("s." if affordable_units != 1 else ".")
        else:
            budget_note = f"Not fully implementable at ${comparison_capital:,.0f}; one full unit needs ${position.capital_required:,.2f}."
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoint_rows.to_dict(orient="records"):
        selected_value = make_float(checkpoint.get("selected_value"))
        modeled_estimated_value = make_float(checkpoint.get("modeled_estimated_value"))
        spot = make_float(checkpoint.get("matched_stock_close"))
        if selected_value is None or spot is None:
            continue
        stock_value = float(spot * CONTRACT_MULTIPLIER)
        stock_profit_loss = stock_value - stock_capital
        strategy_profit_loss = make_float(checkpoint.get("selected_profit_loss")) or (selected_value - position.initial_outlay)
        modeled_profit_loss = None if modeled_estimated_value is None else modeled_estimated_value - position.initial_outlay
        common = {
            "checkpoint": checkpoint.get("checkpoint"),
            "requested_date": checkpoint.get("requested_date"),
            "matched_stock_date": checkpoint.get("matched_stock_date"),
            "matched_stock_close": spot,
            "valuation_source": checkpoint.get("valuation_source"),
            "value_quality": checkpoint.get("value_quality"),
            "comparison_capital": comparison_capital,
            "unit_capital_required": position.capital_required,
            "affordable_units": affordable_units,
            "fully_implementable_with_budget": fully_implementable,
            "budget_note": budget_note,
        }
        rows.append(
            {
                **common,
                "mode": "equal_capital",
                "strategy_value": round(float(selected_value * strategy_scale), 4),
                "modeled_strategy_value": round(float(modeled_estimated_value * strategy_scale), 4) if modeled_estimated_value is not None else None,
                "strategy_profit_loss": round(float(strategy_profit_loss * strategy_scale), 4),
                "modeled_strategy_profit_loss": round(float(modeled_profit_loss * strategy_scale), 4) if modeled_profit_loss is not None else None,
                "strategy_return_on_comparison_capital": finite_or_none((strategy_profit_loss * strategy_scale) / comparison_capital if comparison_capital else None),
                "stock_value": round(float(stock_value * stock_scale), 4),
                "stock_profit_loss": round(float(stock_profit_loss * stock_scale), 4),
                "stock_return_on_comparison_capital": finite_or_none((stock_profit_loss * stock_scale) / comparison_capital if comparison_capital else None),
                "strategy_minus_stock_pnl": round(float((strategy_profit_loss * strategy_scale) - (stock_profit_loss * stock_scale)), 4),
            }
        )
        rows.append(
            {
                **common,
                "mode": "share_equivalent",
                "strategy_value": round(float(selected_value), 4),
                "modeled_strategy_value": round(float(modeled_estimated_value), 4) if modeled_estimated_value is not None else None,
                "strategy_profit_loss": round(float(strategy_profit_loss), 4),
                "modeled_strategy_profit_loss": round(float(modeled_profit_loss), 4) if modeled_profit_loss is not None else None,
                "strategy_return_on_comparison_capital": None,
                "stock_value": round(float(stock_value), 4),
                "stock_profit_loss": round(float(stock_profit_loss), 4),
                "stock_return_on_comparison_capital": None,
                "strategy_minus_stock_pnl": round(float(strategy_profit_loss - stock_profit_loss), 4),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["__order"] = frame["checkpoint"].map(lambda value: CHECKPOINT_ORDER.index(value) if value in CHECKPOINT_ORDER else 99)
    frame = frame.sort_values(["mode", "__order", "matched_stock_date"], ascending=[True, True, False], kind="mergesort").drop(columns=["__order"])
    return frame.reset_index(drop=True)


def _expected_actual_rows(
    checkpoint_rows: pd.DataFrame,
    *,
    entry_spot: float,
    expected_move: dict[str, Any],
    event_result,
) -> pd.DataFrame:
    if checkpoint_rows.empty:
        return pd.DataFrame()
    expected_move_pct = make_float(expected_move.get("expected_move_pct"))
    expected_move_abs = make_float(expected_move.get("expected_move_abs"))
    event_row = event_result.summary.iloc[0].to_dict() if event_result.summary is not None and not event_result.summary.empty else {}
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoint_rows.to_dict(orient="records"):
        final_spot = make_float(checkpoint.get("matched_stock_close"))
        if final_spot is None or not entry_spot:
            actual_move_abs = None
            actual_move_pct = None
        else:
            actual_move_abs = float(final_spot - entry_spot)
            actual_move_pct = float(actual_move_abs / entry_spot)
        note = clean_string(checkpoint.get("event_context_note"))
        if clean_string(checkpoint.get("checkpoint")) == "post_event" and event_row:
            note = "This row uses the first matched post-event close from local prices."
        rows.append(
            {
                "checkpoint": checkpoint.get("checkpoint"),
                "requested_date": checkpoint.get("requested_date"),
                "matched_date": checkpoint.get("matched_stock_date"),
                "expected_move_pct_at_entry": expected_move_pct,
                "expected_move_abs_at_entry": expected_move_abs,
                "actual_move_pct": actual_move_pct,
                "actual_move_abs": actual_move_abs,
                "realized_move_exceeded_expected": (
                    abs(float(actual_move_pct)) > abs(float(expected_move_pct))
                    if expected_move_pct is not None and actual_move_pct is not None
                    else None
                ),
                "event_before_expiry": event_row.get("event_in_horizon"),
                "event_gap_pct": event_row.get("realized_gap_pct"),
                "event_gap_abs": event_row.get("realized_gap_abs"),
                "strategy_profit_loss": checkpoint.get("selected_profit_loss"),
                "note": note,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["__order"] = frame["checkpoint"].map(lambda value: CHECKPOINT_ORDER.index(value) if value in CHECKPOINT_ORDER else 99)
    return frame.sort_values(["__order", "matched_date"], ascending=[True, False], kind="mergesort").drop(columns=["__order"]).reset_index(drop=True)


def _what_this_case_shows(
    position: StrategyPosition,
    *,
    anchor_row: dict[str, Any] | None,
    compare_anchor: dict[str, Any] | None,
    iv_effect: float | None,
    expected_move_pct: float | None,
) -> str:
    if not anchor_row:
        return "Local replay data was too sparse to produce a strong historical read, so this case study is mainly a setup-and-assumptions sheet."
    actual_move_pct = make_float(anchor_row.get("actual_move_pct"))
    pnl = make_float(anchor_row.get("selected_profit_loss"))
    direction_text = "The stock direction was broadly neutral."
    if actual_move_pct is not None:
        if position.name in BULLISH_STRATEGIES:
            direction_text = "The stock moved in the thesis direction." if actual_move_pct > 0 else "The stock moved against the thesis direction."
        elif position.name in BEARISH_STRATEGIES:
            direction_text = "The stock moved in the thesis direction." if actual_move_pct < 0 else "The stock moved against the thesis direction."
        elif actual_move_pct > 0:
            direction_text = "The stock moved higher over the replay window."
        elif actual_move_pct < 0:
            direction_text = "The stock moved lower over the replay window."
    timing_text = "Timing was good enough for the structure." if pnl is not None and pnl > 0 else "Timing and path still mattered enough to keep the structure under pressure."
    if expected_move_pct is not None and actual_move_pct is not None:
        if abs(actual_move_pct) > abs(expected_move_pct):
            timing_text += " Realized movement exceeded the entry expected move."
        else:
            timing_text += " Realized movement stayed inside the entry expected move."
    iv_text = ""
    if iv_effect is not None:
        if iv_effect > 0:
            iv_text = " IV changes helped the position."
        elif iv_effect < 0:
            iv_text = " IV changes hurt the position."
        else:
            iv_text = " IV changes were not a large driver in the available replay path."
    stock_text = ""
    if compare_anchor is not None:
        diff = make_float(compare_anchor.get("strategy_minus_stock_pnl"))
        if diff is not None:
            if diff > 0:
                stock_text = " The structure outperformed long stock on the chosen comparison basis."
            elif diff < 0:
                stock_text = " Long stock would have been the stronger simple benchmark on the chosen comparison basis."
            else:
                stock_text = " The structure and long stock landed near the same outcome on the chosen comparison basis."
    structure_text = ""
    if position.name in {"bull_call_spread", "bear_put_spread", "covered_call"}:
        structure_text = " The structure's cap matters here because it can trade away upside for a cheaper or more defensive entry."
    elif position.name in {"long_call", "long_put"}:
        structure_text = " The option structure still needed enough movement before expiry because time value does not wait."
    elif position.name == "cash_secured_put":
        structure_text = " The cash-secured put is still a bullish, collateral-backed structure rather than a free-premium trade."
    return f"{direction_text} {timing_text}{iv_text}{stock_text}{structure_text}".strip()


def collect_local_replay_history(
    output_root: str | Path,
    *,
    ticker: str,
    strategy_name: str,
) -> pd.DataFrame:
    root = Path(output_root)
    if not root.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for metadata_path in root.rglob("report_metadata.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if clean_string(payload.get("report_kind")) != "replay":
            continue
        if clean_string(payload.get("ticker") or (payload.get("metadata") or {}).get("ticker")).upper() != clean_string(ticker).upper():
            continue
        replay_payload = payload.get("replay") or {}
        payload_strategy = payload.get("strategy_name") or replay_payload.get("strategy_name")
        if _normalize_strategy_label(payload_strategy) != _normalize_strategy_label(strategy_name):
            continue
        if metadata_path.parent.name == "metadata":
            bundle_dir = metadata_path.parent.parent
            summary_path = bundle_dir / "tables" / "case_summary.csv"
            if not summary_path.exists():
                summary_path = bundle_dir / "tables" / "summary.csv"
            source_dir = bundle_dir
        else:
            summary_path = metadata_path.parent / "case_summary.csv"
            if not summary_path.exists():
                summary_path = metadata_path.parent / "summary.csv"
            source_dir = metadata_path.parent
        if not summary_path.exists():
            continue
        try:
            frame = pd.read_csv(summary_path)
        except Exception:
            continue
        if frame.empty:
            continue
        row = frame.iloc[0].to_dict()
        row["source_report_dir"] = str(source_dir)
        row["generated_at"] = payload.get("generated_at")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "snapshot_date" in frame.columns:
        frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    if "generated_at" in frame.columns:
        frame["generated_at"] = pd.to_datetime(frame["generated_at"], errors="coerce", utc=True)
    order_columns = [column for column in ["snapshot_date", "generated_at"] if column in frame.columns]
    if order_columns:
        frame = frame.sort_values(order_columns, ascending=[False] * len(order_columns), kind="mergesort")
    if "snapshot_date" in frame.columns:
        frame["snapshot_date"] = frame["snapshot_date"].dt.date.astype(str)
    if "generated_at" in frame.columns:
        frame["generated_at"] = frame["generated_at"].astype(str)
    return frame.reset_index(drop=True)


def _build_replay_core(
    ticker: str,
    *,
    snapshot_date: date | str,
    expiry_date: date | str,
    strategy_name: str,
    data_root: str | Path | None = None,
    premium_mode: str = "mid",
    spot_price: float | None = None,
    comparison_capital: float = 1000.0,
    contract_selector: dict[str, Any] | None = None,
    long_selector: dict[str, Any] | None = None,
    short_selector: dict[str, Any] | None = None,
) -> HistoricalReplayComputation:
    """Build one historical replay / case-study bundle from local data."""

    chain = load_snapshot_chain(
        ticker,
        snapshot_date=snapshot_date,
        expiry_date=expiry_date,
        data_root=data_root,
        spot_price=spot_price,
    )
    position = build_strategy(
        strategy_name,
        chain,
        **_strategy_build_kwargs(
            strategy_name,
            spot_price=spot_price,
            premium_mode=premium_mode,
            contract_selector=contract_selector,
            long_selector=long_selector,
            short_selector=short_selector,
        ),
    )
    event_result = _replay_event_context(
        ticker,
        snapshot_date=position.snapshot_date,
        expiry_date=position.expiry_date,
        data_root=data_root,
    )
    warnings = list(position.warnings) + list(event_result.warnings)
    expected_move = (position.resolved_metadata.get("research_context") or {}).get("expected_move") or {}
    checkpoint_specs = _checkpoint_spec_rows(position, event_result=event_result)
    checkpoint_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    value_sources: dict[str, int] = {}
    history = _prepared_price_history(position.ticker, data_root)

    for spec in checkpoint_specs:
        checkpoint = clean_string(spec.get("checkpoint"))
        requested_date = parse_date(spec.get("requested_date"))
        target_date = parse_date(spec.get("target_date"))
        spot_context = _latest_close_on_or_before(history, target_date)
        exact_value = _exact_chain_value(position, target_date=target_date, data_root=data_root)
        modeled_value = _modeled_value(position, target_date=target_date, data_root=data_root)
        selected = exact_value or modeled_value
        selected_source = clean_string((selected or {}).get("source")) or "insufficient_data"
        value_sources[selected_source] = value_sources.get(selected_source, 0) + 1
        final_spot = make_float((selected or {}).get("spot")) if selected is not None else make_float((spot_context or {}).get("close"))
        selected_value = make_float((selected or {}).get("value"))
        actual_move_abs = None
        actual_move_pct = None
        if final_spot is not None and position.entry_spot:
            actual_move_abs = float(final_spot - position.entry_spot)
            actual_move_pct = float(actual_move_abs / position.entry_spot)
        normalized_scale = (comparison_capital / position.capital_required) if comparison_capital > 0 and position.capital_required > 0 else 1.0
        selected_profit_loss = None if selected_value is None else float(selected_value - position.initial_outlay)
        event_gap_pct = None
        if event_result.summary is not None and not event_result.summary.empty:
            event_summary_row = event_result.summary.iloc[0].to_dict()
            event_gap_pct = make_float(event_summary_row.get("realized_gap_pct"))
        decomposition_rows.extend(
            _decomposition_rows(
                position,
                checkpoint_label=checkpoint,
                valuation_date=target_date,
                final_spot=final_spot,
                selected_value=selected_value,
                selected_source=selected_source,
                observed_iv=make_float((exact_value or {}).get("iv_assumption")),
                current_rate=make_float((selected or {}).get("rate_assumption")),
                event_context_note=clean_string(spec.get("event_context_note")) or None,
                event_gap_pct=event_gap_pct,
            )
        )
        checkpoint_rows.append(
            {
                "checkpoint": checkpoint,
                "requested_date": requested_date.isoformat() if requested_date else None,
                "matched_stock_date": (spot_context or {}).get("matched_date") or (selected or {}).get("spot_matched_date"),
                "matched_stock_close": final_spot,
                "actual_move_abs": actual_move_abs,
                "actual_move_pct": actual_move_pct,
                "modeled_estimated_value": make_float((modeled_value or {}).get("value")),
                "exact_observed_value": make_float((exact_value or {}).get("value")),
                "selected_value": selected_value,
                "selected_profit_loss": selected_profit_loss,
                "selected_return_on_capital": finite_or_none(selected_profit_loss / position.capital_required if selected_profit_loss is not None and position.capital_required else None),
                "normalized_profit_loss": None if selected_profit_loss is None else float(selected_profit_loss * normalized_scale),
                "return_on_comparison_capital": finite_or_none((selected_profit_loss * normalized_scale) / comparison_capital if selected_profit_loss is not None and comparison_capital else None),
                "valuation_source": selected_source,
                "value_quality": (
                    "exact"
                    if selected_source == "exact_local_later_chain"
                    else ("modeled" if selected_source == "approximate_model_estimate" else "missing")
                ),
                "clamped_to_expiry": bool(spec.get("clamped_to_expiry")),
                "valuation_date": target_date.isoformat() if target_date else None,
                "iv_assumption": make_float((selected or {}).get("iv_assumption")),
                "rate_assumption": make_float((selected or {}).get("rate_assumption")),
                "event_context_note": spec.get("event_context_note"),
                "expected_move_pct_at_entry": make_float(expected_move.get("expected_move_pct")),
                "expected_move_abs_at_entry": make_float(expected_move.get("expected_move_abs")),
                "source_snapshot_file": None if exact_value is None else exact_value.get("source_snapshot_file"),
            }
        )

    checkpoint_replay = pd.DataFrame(checkpoint_rows)
    if not checkpoint_replay.empty:
        checkpoint_replay["__order"] = checkpoint_replay["checkpoint"].map(lambda value: CHECKPOINT_ORDER.index(value) if value in CHECKPOINT_ORDER else 99)
        checkpoint_replay = checkpoint_replay.sort_values(["__order", "matched_stock_date"], ascending=[True, False], kind="mergesort").drop(columns=["__order"]).reset_index(drop=True)

    compare_vs_stock = _compare_vs_stock_rows(position, checkpoint_replay, comparison_capital=comparison_capital)
    expected_move_vs_actual = _expected_actual_rows(
        checkpoint_replay,
        entry_spot=position.entry_spot,
        expected_move=expected_move,
        event_result=event_result,
    )
    driver_decomposition = pd.DataFrame(decomposition_rows)
    if not driver_decomposition.empty:
        driver_decomposition["__checkpoint_order"] = driver_decomposition["checkpoint"].map(lambda value: CHECKPOINT_ORDER.index(value) if value in CHECKPOINT_ORDER else 99)
        driver_decomposition = driver_decomposition.sort_values(["__checkpoint_order", "matched_date", "component"], ascending=[True, False, True], kind="mergesort").drop(columns=["__checkpoint_order"]).reset_index(drop=True)

    anchor_label = "expiry" if not checkpoint_replay.empty and "expiry" in checkpoint_replay["checkpoint"].tolist() else (checkpoint_replay.iloc[-1]["checkpoint"] if not checkpoint_replay.empty else None)
    anchor_row = None if not anchor_label or checkpoint_replay.empty else checkpoint_replay.loc[checkpoint_replay["checkpoint"] == anchor_label].iloc[-1].to_dict()
    compare_anchor = None
    if anchor_label and not compare_vs_stock.empty:
        anchor_matches = compare_vs_stock.loc[
            (compare_vs_stock["checkpoint"] == anchor_label)
            & (compare_vs_stock["mode"] == "equal_capital")
        ]
        if not anchor_matches.empty:
            compare_anchor = anchor_matches.iloc[-1].to_dict()
    iv_effect = None
    if anchor_label and not driver_decomposition.empty:
        iv_rows = driver_decomposition.loc[
            (driver_decomposition["checkpoint"] == anchor_label)
            & (driver_decomposition["component"] == "iv_change_effect")
        ]
        if not iv_rows.empty:
            iv_effect = make_float(iv_rows.iloc[-1]["effect"])
    what_this_case_shows = _what_this_case_shows(
        position,
        anchor_row=anchor_row,
        compare_anchor=compare_anchor,
        iv_effect=iv_effect,
        expected_move_pct=make_float(expected_move.get("expected_move_pct")),
    )

    exact_count = int((checkpoint_replay["valuation_source"] == "exact_local_later_chain").sum()) if not checkpoint_replay.empty else 0
    modeled_count = int((checkpoint_replay["valuation_source"] == "approximate_model_estimate").sum()) if not checkpoint_replay.empty else 0
    missing_count = int((checkpoint_replay["valuation_source"] == "insufficient_data").sum()) if not checkpoint_replay.empty else 0
    status = "ok" if exact_count and not modeled_count and not missing_count else "partial"
    if checkpoint_replay.empty or (exact_count == 0 and modeled_count == 0):
        status = "insufficient_data"
        warnings.append("No exact later chain snapshots or modelable replay checkpoints were available for this case study.")
    elif exact_count == 0:
        warnings.append("No exact later local chain snapshots were available for this replay, so later checkpoints use modeled estimates with entry IV held static.")

    case_summary = pd.DataFrame(
        [
            {
                "ticker": position.ticker,
                "snapshot_date": position.snapshot_date.isoformat(),
                "expiry_date": position.expiry_date.isoformat() if position.expiry_date else None,
                "strategy_name": position.name,
                "status": status,
                "comparison_capital": comparison_capital,
                "checkpoint_count": int(len(checkpoint_replay)),
                "exact_chain_checkpoint_count": exact_count,
                "modeled_checkpoint_count": modeled_count,
                "insufficient_checkpoint_count": missing_count,
                "anchor_checkpoint": anchor_label,
                "anchor_matched_date": None if anchor_row is None else anchor_row.get("matched_stock_date"),
                "anchor_valuation_source": None if anchor_row is None else anchor_row.get("valuation_source"),
                "anchor_profit_loss": None if anchor_row is None else anchor_row.get("selected_profit_loss"),
                "anchor_return_on_capital": None if anchor_row is None else anchor_row.get("selected_return_on_capital"),
                "anchor_normalized_profit_loss": None if anchor_row is None else anchor_row.get("normalized_profit_loss"),
                "anchor_return_on_comparison_capital": None if anchor_row is None else anchor_row.get("return_on_comparison_capital"),
                "anchor_strategy_minus_stock_pnl": None if compare_anchor is None else compare_anchor.get("strategy_minus_stock_pnl"),
                "expected_move_pct_at_entry": make_float(expected_move.get("expected_move_pct")),
                "actual_move_pct_anchor": None if anchor_row is None else anchor_row.get("actual_move_pct"),
                "expected_minus_actual_move_pct_anchor": (
                    None
                    if anchor_row is None
                    or make_float(expected_move.get("expected_move_pct")) is None
                    or make_float(anchor_row.get("actual_move_pct")) is None
                    else float(make_float(expected_move.get("expected_move_pct")) - make_float(anchor_row.get("actual_move_pct")))
                ),
                "what_this_case_shows": what_this_case_shows,
            }
        ]
    )

    return HistoricalReplayComputation(
        ticker=position.ticker,
        snapshot_date=position.snapshot_date,
        expiry_date=position.expiry_date or parse_date(expiry_date),
        strategy_name=position.name,
        source_snapshot_file=str(position.resolved_metadata.get("source_snapshot_file") or chain.source_path),
        premium_mode=position.premium_mode,
        comparison_capital=float(comparison_capital),
        spot_price=float(position.entry_spot),
        risk_free_rate=float(position.risk_free_rate),
        dividend_yield=float(position.dividend_yield),
        resolved_metadata=dict(position.resolved_metadata),
        research_context=dict(position.resolved_metadata.get("research_context") or {}),
        warnings=list(dict.fromkeys(clean_string(item) for item in warnings if clean_string(item))),
        status=status,
        shareability_status="Mostly Self-Contained",
        valuation_source_rollup=value_sources,
        available_checkpoints=[row["checkpoint"] for row in checkpoint_rows],
        what_this_case_shows=what_this_case_shows,
        replay_defaults={
            "anchor_checkpoint": anchor_label,
            "comparison_mode": "equal_capital",
        },
        case_summary=case_summary,
        checkpoint_replay=checkpoint_replay,
        expected_move_vs_actual=expected_move_vs_actual,
        driver_decomposition=driver_decomposition,
        compare_vs_stock=compare_vs_stock,
        local_history=pd.DataFrame(),
        position=position,
    )


def build_replay_analysis(*args, **kwargs) -> HistoricalReplayComputation:
    """Build one canonical replay analysis result."""

    return _build_replay_core(*args, **kwargs)
