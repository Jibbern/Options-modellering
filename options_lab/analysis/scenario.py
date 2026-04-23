"""Multi-strategy scenario dashboard orchestration for one snapshot slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..io import OptionChain, load_chain
from ..pricing import intrinsic_value, price_option
from ..scenarios import compare_positions
from ..snapshots import snapshot_slice_for_expiry
from ..strategies import StrategyPosition, build_strategy
from ..utils import CONTRACT_MULTIPLIER, build_stock_grid, clean_string, finite_or_none, horizon_to_days


SUPPORTED_SCENARIO_STRATEGIES = [
    "long_stock",
    "long_call",
    "bull_call_spread",
    "long_put",
    "bear_put_spread",
    "covered_call",
    "cash_secured_put",
]

DEFAULT_SPOT_CASES = {
    "far_bear": -0.30,
    "bear": -0.15,
    "flat": 0.0,
    "bull": 0.15,
    "strong_bull": 0.30,
}

DEFAULT_IV_CASES = {
    "iv_down": -0.10,
    "iv_unchanged": 0.0,
    "iv_up": 0.10,
}

DEFAULT_HORIZON_SPECS = [
    ("entry", 0),
    ("1w", 7),
    ("1m", 30),
    ("3m", 90),
    ("6m", 180),
]

REPRESENTATIVE_HORIZON_LABEL = "1m"
REPRESENTATIVE_IV_CASE = "iv_unchanged"
CAPITAL_SIZING_MODE = "hybrid"


@dataclass
class ScenarioDashboardComputation:
    """Computed tables and metadata for one multi-strategy scenario dashboard."""

    ticker: str
    snapshot_date: date
    expiry_date: date
    source_snapshot_file: str
    spot_price: float
    premium_mode: str
    risk_free_rate: float
    dividend_yield: float
    resolved_metadata: dict[str, Any]
    research_context: dict[str, Any]
    warnings: list[str]
    status: str
    shareability_status: str
    comparison_capital: float
    capital_sizing_mode: str
    featured_focus_strategy: str
    available_strategies: list[str]
    omitted_strategies: list[dict[str, str]]
    scenario_defaults: dict[str, Any]
    forward_defaults: dict[str, Any]
    decision_hints: dict[str, Any]
    replay_defaults: dict[str, Any]
    valuation_defaults: dict[str, Any]
    what_matters_most: str
    executive_summary: pd.DataFrame
    strategy_summary: pd.DataFrame
    named_scenarios: pd.DataFrame
    stock_relative: pd.DataFrame
    spot_time_grid: pd.DataFrame
    spot_iv_grid: pd.DataFrame
    forward_quick_scenarios: pd.DataFrame
    forward_spot_time_grid: pd.DataFrame
    forward_spot_iv_grid: pd.DataFrame
    forward_time_iv_grid: pd.DataFrame
    valuation_explanation: pd.DataFrame
    positions: list[StrategyPosition] = field(default_factory=list, repr=False)


def load_snapshot_chain(
    ticker: str,
    *,
    snapshot_date: date | str,
    expiry_date: date | str,
    data_root: str | Path | None = None,
    spot_price: float | None = None,
    metadata_override: dict[str, Any] | None = None,
) -> OptionChain:
    """Load one exact local snapshot slice for the scenario dashboard."""

    slice_row = snapshot_slice_for_expiry(ticker, snapshot_date, expiry_date, data_root)
    return load_chain(
        slice_row["file_path"],
        metadata_override=metadata_override,
        spot_price=spot_price,
        prices_data_root=data_root,
        rates_data_root=data_root,
        research_data_root=data_root,
    )


def _dedupe(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        text = clean_string(value)
        if text and text not in unique:
            unique.append(text)
    return unique


def _strategy_selector_payload(
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


def _dashboard_horizon_specs(expiry_days: int) -> list[tuple[str, int]]:
    specs = list(DEFAULT_HORIZON_SPECS)
    specs.append(("expiry", max(expiry_days, 0)))
    resolved: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for label, days in specs:
        key = (label, max(int(days), 0))
        if key in seen:
            continue
        seen.add(key)
        resolved.append(key)
    return resolved


def _representative_horizon(expiry_days: int) -> tuple[str, int]:
    requested_days = horizon_to_days(REPRESENTATIVE_HORIZON_LABEL)
    if expiry_days <= 0:
        return ("expiry", 0)
    if expiry_days < requested_days:
        return ("expiry", expiry_days)
    return (REPRESENTATIVE_HORIZON_LABEL, requested_days)


def _scenario_case_prices(spot_price: float, overrides: dict[str, float] | None = None) -> dict[str, float]:
    spot_cases = dict(DEFAULT_SPOT_CASES)
    for label, value in (overrides or {}).items():
        if clean_string(label):
            spot_cases[clean_string(label)] = float(value)
    return {
        label: round(float(spot_price) * (1.0 + float(move_pct)), 4)
        for label, move_pct in spot_cases.items()
    }


def _iv_case_points(overrides: dict[str, float] | None = None) -> dict[str, float]:
    payload = dict(DEFAULT_IV_CASES)
    for label, value in (overrides or {}).items():
        if clean_string(label):
            payload[clean_string(label)] = float(value)
    return payload


def _valuation_point(
    position: StrategyPosition,
    *,
    context_expiry_days: int,
    horizon_label: str,
    requested_days: int,
    spot_price: float,
    iv_case: str,
    iv_shift: float,
) -> dict[str, Any]:
    if horizon_label == "expiry" and position.expiry_date is None:
        valuation_date = position.snapshot_date + timedelta(days=max(context_expiry_days, 0))
        clamped = False
    else:
        valuation_date, clamped = position.valuation_date_for_horizon(requested_days)
    effective_days = max((valuation_date - position.snapshot_date).days, 0)
    estimated_value = float(
        position.mark_to_market_value(
            [spot_price],
            valuation_date=valuation_date,
            iv_shift=iv_shift,
        )[0]
    )
    profit_loss = estimated_value - position.initial_outlay
    return {
        "strategy": position.name,
        "horizon": horizon_label,
        "requested_days": int(requested_days),
        "effective_days": int(effective_days),
        "clamped_to_expiry": bool(clamped),
        "valuation_date": valuation_date.isoformat(),
        "spot_price": round(float(spot_price), 4),
        "iv_case": iv_case,
        "iv_shift": float(iv_shift),
        "estimated_value": round(float(estimated_value), 4),
        "profit_loss": round(float(profit_loss), 4),
        "return_on_capital": finite_or_none(profit_loss / position.capital_required if position.capital_required else None),
    }


def _grid_rows(
    position: StrategyPosition,
    *,
    context_expiry_days: int,
    spot_prices: Iterable[float],
    horizon_specs: Iterable[tuple[str, int]],
    iv_cases: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_label, requested_days in horizon_specs:
        for iv_case, iv_shift in iv_cases.items():
            for spot_price in spot_prices:
                rows.append(
                    _valuation_point(
                        position,
                        context_expiry_days=context_expiry_days,
                        horizon_label=horizon_label,
                        requested_days=requested_days,
                        spot_price=float(spot_price),
                        iv_case=iv_case,
                        iv_shift=float(iv_shift),
                    )
                )
    return rows


def _leg_summary(position: StrategyPosition) -> str:
    labels = [clean_string(leg.label) for leg in position.legs if clean_string(leg.label)]
    return " | ".join(labels)


def _position_warning_text(position: StrategyPosition) -> str | None:
    for text in list(position.warnings) + list(position.notes):
        cleaned = clean_string(text)
        if cleaned:
            return cleaned
    return None


def _budget_fields(position: StrategyPosition, comparison_capital: float) -> dict[str, Any]:
    unit_capital_required = float(position.capital_required) if position.capital_required > 0 else None
    if unit_capital_required is None or comparison_capital <= 0:
        affordable_units = None
        fully_implementable = None
        budget_note = None
    else:
        affordable_units = int(comparison_capital // unit_capital_required)
        fully_implementable = affordable_units >= 1
        if fully_implementable:
            budget_note = (
                f"At ${comparison_capital:,.0f}, this structure can fund {affordable_units} whole unit"
                + ("s." if affordable_units != 1 else ".")
            )
        else:
            budget_note = (
                f"Not fully implementable at ${comparison_capital:,.0f}; one full unit needs "
                f"${unit_capital_required:,.2f}."
            )
    return {
        "comparison_capital": round(float(comparison_capital), 4),
        "unit_capital_required": round(float(unit_capital_required), 4) if unit_capital_required is not None else None,
        "affordable_units": affordable_units,
        "fully_implementable_with_budget": fully_implementable,
        "budget_note": budget_note,
    }


def _named_scenario_row(
    frame: pd.DataFrame,
    *,
    strategy: str,
    spot_case: str,
    horizon: str,
    iv_case: str,
) -> pd.Series | None:
    matches = frame.loc[
        (frame["strategy"] == strategy)
        & (frame["spot_case"] == spot_case)
        & (frame["horizon"] == horizon)
        & (frame["iv_case"] == iv_case)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _build_strategy_summary(
    positions: list[StrategyPosition],
    named_scenarios: pd.DataFrame,
    *,
    representative_horizon: str,
    representative_iv_case: str,
    comparison_capital: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position in positions:
        summary = position.summary_metrics()
        budget_fields = _budget_fields(position, comparison_capital)
        bear_row = _named_scenario_row(named_scenarios, strategy=position.name, spot_case="bear", horizon=representative_horizon, iv_case=representative_iv_case)
        base_row = _named_scenario_row(named_scenarios, strategy=position.name, spot_case="flat", horizon=representative_horizon, iv_case=representative_iv_case)
        bull_row = _named_scenario_row(named_scenarios, strategy=position.name, spot_case="bull", horizon=representative_horizon, iv_case=representative_iv_case)
        rows.append(
            {
                "strategy": position.name,
                "status": "included",
                "expiry_date": position.expiry_date.isoformat() if position.expiry_date else None,
                "leg_summary": _leg_summary(position),
                "capital_required": summary.get("capital_required"),
                "max_loss": summary.get("max_loss"),
                "max_gain": summary.get("max_gain"),
                "break_even": summary.get("break_even"),
                "premium_paid": summary.get("premium_paid"),
                "premium_received": summary.get("premium_received"),
                "initial_outlay": summary.get("initial_outlay"),
                "selected_horizon": representative_horizon,
                "selected_iv_case": representative_iv_case,
                "selected_estimated_value": base_row.get("estimated_value") if base_row is not None else None,
                "selected_profit_loss": base_row.get("profit_loss") if base_row is not None else None,
                "selected_comparison_estimated_value": base_row.get("equal_capital_estimated_value") if base_row is not None else None,
                "selected_comparison_profit_loss": base_row.get("equal_capital_profit_loss") if base_row is not None else None,
                "selected_return_on_comparison_capital": base_row.get("return_on_comparison_capital") if base_row is not None else None,
                "bear_profit_loss": bear_row.get("profit_loss") if bear_row is not None else None,
                "base_profit_loss": base_row.get("profit_loss") if base_row is not None else None,
                "bull_profit_loss": bull_row.get("profit_loss") if bull_row is not None else None,
                "bear_comparison_profit_loss": bear_row.get("equal_capital_profit_loss") if bear_row is not None else None,
                "base_comparison_profit_loss": base_row.get("equal_capital_profit_loss") if base_row is not None else None,
                "bull_comparison_profit_loss": bull_row.get("equal_capital_profit_loss") if bull_row is not None else None,
                "capital_efficiency": finite_or_none(
                    (bull_row.get("equal_capital_profit_loss") / comparison_capital)
                    if bull_row is not None and comparison_capital
                    else None
                ),
                "warning_or_note": _position_warning_text(position),
                **budget_fields,
            }
        )
    return pd.DataFrame(rows)


def _build_stock_relative(
    positions: list[StrategyPosition],
    *,
    spot_grid: np.ndarray,
    horizon: tuple[str, int],
    iv_case: str,
    iv_shift: float,
    comparison_capital: float,
) -> pd.DataFrame:
    comparison = compare_positions(
        positions,
        mode="both",
        spot_grid=spot_grid,
        horizon=horizon[1],
        iv_shift=iv_shift,
        comparison_capital=comparison_capital,
    )
    if comparison.empty:
        return comparison
    comparison["horizon"] = horizon[0]
    comparison["iv_case"] = iv_case
    baseline = comparison.loc[comparison["strategy"] == "long_stock", ["mode", "spot_price", "estimated_value", "profit_loss"]].rename(
        columns={
            "estimated_value": "stock_estimated_value",
            "profit_loss": "stock_profit_loss",
        }
    )
    merged = comparison.merge(baseline, on=["mode", "spot_price"], how="left")
    merged["stock_relative_difference"] = (
        pd.to_numeric(merged["profit_loss"], errors="coerce")
        - pd.to_numeric(merged["stock_profit_loss"], errors="coerce")
    ).round(4)
    return merged


def _merge_equal_capital_forward_context(
    rows_frame: pd.DataFrame,
    *,
    positions: list[StrategyPosition],
    spot_prices: Iterable[float],
    horizon: tuple[str, int],
    iv_case: str,
    iv_shift: float,
    comparison_capital: float,
) -> pd.DataFrame:
    if rows_frame.empty:
        return rows_frame
    stock_relative = _build_stock_relative(
        positions,
        spot_grid=np.asarray(list(spot_prices), dtype=float),
        horizon=horizon,
        iv_case=iv_case,
        iv_shift=iv_shift,
        comparison_capital=comparison_capital,
    )
    if stock_relative.empty:
        return rows_frame
    equal_capital = stock_relative.loc[stock_relative["mode"] == "equal_capital"].copy()
    if equal_capital.empty:
        return rows_frame
    equal_capital = equal_capital.rename(
        columns={
            "estimated_value": "comparison_estimated_value",
            "profit_loss": "comparison_profit_loss",
            "return_on_comparison_capital": "return_on_comparison_capital",
        }
    )
    keep_columns = [
        column
        for column in [
            "strategy",
            "spot_price",
            "horizon",
            "iv_case",
            "comparison_estimated_value",
            "comparison_profit_loss",
            "return_on_comparison_capital",
            "comparison_capital",
            "unit_capital_required",
            "affordable_units",
            "fully_implementable_with_budget",
            "budget_note",
            "stock_estimated_value",
            "stock_profit_loss",
            "stock_relative_difference",
        ]
        if column in equal_capital.columns
    ]
    merged = rows_frame.merge(
        equal_capital[keep_columns],
        on=["strategy", "spot_price", "horizon", "iv_case"],
        how="left",
    )
    if "stock_profit_loss" in merged.columns and "comparison_capital" in merged.columns:
        merged["stock_return_on_comparison_capital"] = merged.apply(
            lambda row: finite_or_none(
                (
                    pd.to_numeric(row.get("stock_profit_loss"), errors="coerce")
                    / pd.to_numeric(row.get("comparison_capital"), errors="coerce")
                )
                if pd.notna(pd.to_numeric(row.get("stock_profit_loss"), errors="coerce"))
                and pd.notna(pd.to_numeric(row.get("comparison_capital"), errors="coerce"))
                and float(pd.to_numeric(row.get("comparison_capital"), errors="coerce")) > 0
                else None
            ),
            axis=1,
        )
    return merged


def _attach_spot_case_labels(
    frame: pd.DataFrame,
    *,
    spot_case_lookup: dict[float, str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    enriched["spot_case"] = enriched["spot_price"].map(
        lambda value: spot_case_lookup.get(round(float(value), 4))
        if pd.notna(pd.to_numeric(value, errors="coerce"))
        else None
    )
    return enriched


def _forward_grid_frame(
    positions: list[StrategyPosition],
    *,
    context_expiry_days: int,
    spot_prices: Iterable[float],
    horizon_specs: Iterable[tuple[str, int]],
    iv_cases: dict[str, float],
    comparison_capital: float,
    spot_case_lookup: dict[float, str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    spot_values = [round(float(value), 4) for value in spot_prices]
    for horizon in horizon_specs:
        for iv_case, iv_shift in iv_cases.items():
            actual_rows: list[dict[str, Any]] = []
            for position in positions:
                actual_rows.extend(
                    _grid_rows(
                        position,
                        context_expiry_days=context_expiry_days,
                        spot_prices=spot_values,
                        horizon_specs=[horizon],
                        iv_cases={iv_case: iv_shift},
                    )
                )
            actual_frame = pd.DataFrame(actual_rows)
            if actual_frame.empty:
                continue
            actual_frame = _merge_equal_capital_forward_context(
                actual_frame,
                positions=positions,
                spot_prices=spot_values,
                horizon=horizon,
                iv_case=iv_case,
                iv_shift=iv_shift,
                comparison_capital=comparison_capital,
            )
            if spot_case_lookup:
                actual_frame = _attach_spot_case_labels(actual_frame, spot_case_lookup=spot_case_lookup)
            frames.append(actual_frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _executive_summary_row(
    strategy_summary: pd.DataFrame,
    stock_relative: pd.DataFrame,
    *,
    ticker: str,
    snapshot_date: date,
    expiry_date: date,
    status: str,
    spot_price: float,
    risk_free_rate: float,
    representative_horizon: tuple[str, int],
    comparison_capital: float,
) -> pd.DataFrame:
    quick = strategy_summary.copy()
    quick["bull_comparison_profit_loss"] = pd.to_numeric(quick["bull_comparison_profit_loss"], errors="coerce")
    quick["base_comparison_profit_loss"] = pd.to_numeric(quick["base_comparison_profit_loss"], errors="coerce")
    quick["bear_comparison_profit_loss"] = pd.to_numeric(quick["bear_comparison_profit_loss"], errors="coerce")
    best_bull = quick.sort_values("bull_comparison_profit_loss", ascending=False).iloc[0] if not quick.empty else None
    best_flat = quick.sort_values("base_comparison_profit_loss", ascending=False).iloc[0] if not quick.empty else None
    best_downside = quick.sort_values("bear_comparison_profit_loss", ascending=False).iloc[0] if not quick.empty else None
    stock_rows = stock_relative.loc[
        (stock_relative["strategy"] != "long_stock")
        & (stock_relative["mode"] == "equal_capital")
    ].copy()
    stock_rows["stock_relative_difference"] = pd.to_numeric(stock_rows["stock_relative_difference"], errors="coerce")
    best_relative = stock_rows.sort_values("stock_relative_difference", ascending=False).iloc[0] if not stock_rows.empty else None
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "snapshot_date": snapshot_date.isoformat(),
                "expiry_date": expiry_date.isoformat(),
                "status": status,
                "spot_price": spot_price,
                "risk_free_rate": risk_free_rate,
                "comparison_capital": comparison_capital,
                "included_strategy_count": int(len(strategy_summary.index)),
                "representative_horizon": representative_horizon[0],
                "representative_days": int(representative_horizon[1]),
                "best_bull_strategy": best_bull.get("strategy") if best_bull is not None else None,
                "best_bull_profit_loss": best_bull.get("bull_comparison_profit_loss") if best_bull is not None else None,
                "best_flat_strategy": best_flat.get("strategy") if best_flat is not None else None,
                "best_flat_profit_loss": best_flat.get("base_comparison_profit_loss") if best_flat is not None else None,
                "best_defensive_strategy": best_downside.get("strategy") if best_downside is not None else None,
                "best_defensive_profit_loss": best_downside.get("bear_comparison_profit_loss") if best_downside is not None else None,
                "best_equal_capital_relative_strategy": best_relative.get("strategy") if best_relative is not None else None,
                "best_equal_capital_relative_diff": best_relative.get("stock_relative_difference") if best_relative is not None else None,
            }
        ]
    )


def _pick_hint(
    frame: pd.DataFrame,
    *,
    column: str,
    ascending: bool,
) -> dict[str, Any] | None:
    if frame.empty or column not in frame.columns:
        return None
    working = frame.copy()
    working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=[column])
    if working.empty:
        return None
    row = working.sort_values(column, ascending=ascending, kind="mergesort").iloc[0]
    return {
        "strategy": row.get("strategy"),
        "value": row.get(column),
    }


def _decision_hints(
    strategy_summary: pd.DataFrame,
    stock_relative: pd.DataFrame,
) -> dict[str, Any]:
    relative = stock_relative.loc[
        (stock_relative["mode"] == "equal_capital")
        & (stock_relative["strategy"] != "long_stock")
    ].copy() if not stock_relative.empty else pd.DataFrame()
    return {
        "best_bull_case": _pick_hint(strategy_summary, column="bull_comparison_profit_loss", ascending=False),
        "best_flat_case": _pick_hint(strategy_summary, column="base_comparison_profit_loss", ascending=False),
        "best_downside_control": _pick_hint(strategy_summary, column="bear_comparison_profit_loss", ascending=False),
        "best_capital_efficiency": _pick_hint(strategy_summary, column="capital_efficiency", ascending=False),
        "lowest_max_loss": _pick_hint(strategy_summary, column="max_loss", ascending=True),
        "best_vs_stock": _pick_hint(relative, column="stock_relative_difference", ascending=False),
    }


def _featured_focus_strategy(positions: list[StrategyPosition]) -> str:
    for position in positions:
        if position.name != "long_stock":
            return position.name
    return positions[0].name if positions else "long_stock"


def _valuation_default_horizon_label(representative_horizon: tuple[str, int]) -> str:
    return "entry" if representative_horizon[0] == "expiry" else representative_horizon[0]


def _entry_delta_estimate(position: StrategyPosition) -> float | None:
    total_delta = 0.0
    has_any = False
    for leg in position.legs:
        if leg.asset_type == "stock":
            total_delta += float(leg.quantity)
            has_any = True
            continue
        if leg.delta is None:
            return None
        total_delta += float(leg.quantity) * CONTRACT_MULTIPLIER * float(leg.delta)
        has_any = True
    return round(float(total_delta), 4) if has_any else None


def _valuation_components(
    position: StrategyPosition,
    *,
    valuation_date: date,
    spot_price: float,
    iv_shift: float,
) -> dict[str, Any]:
    stock_leg_value = 0.0
    option_intrinsic_value = 0.0
    option_modeled_value = 0.0
    for leg in position.legs:
        if leg.asset_type == "stock":
            stock_leg_value += float(leg.quantity) * float(spot_price)
            continue
        if leg.option_type is None or leg.strike is None or leg.expiry_date is None:
            continue
        option_intrinsic_value += (
            float(leg.quantity)
            * CONTRACT_MULTIPLIER
            * intrinsic_value(float(spot_price), float(leg.strike), str(leg.option_type))
        )
        sigma = max(float(leg.base_iv or 0.0) + float(iv_shift), 1e-6)
        option_modeled_value += (
            float(leg.quantity)
            * CONTRACT_MULTIPLIER
            * price_option(
                spot=float(spot_price),
                strike=float(leg.strike),
                time_to_expiry=max((leg.expiry_date - valuation_date).days, 0) / 365.0,
                iv=sigma,
                risk_free_rate=float(position.risk_free_rate),
                dividend_yield=float(position.dividend_yield),
                option_type=str(leg.option_type),
            )
        )
    modeled_value = stock_leg_value + option_modeled_value
    return {
        "stock_leg_value": round(float(stock_leg_value), 4),
        "option_intrinsic_value": round(float(option_intrinsic_value), 4),
        "option_modeled_value": round(float(option_modeled_value), 4),
        "option_extrinsic_value": round(float(option_modeled_value - option_intrinsic_value), 4),
        "modeled_value": round(float(modeled_value), 4),
    }


def _build_valuation_explanation(
    positions: list[StrategyPosition],
    named_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    if not positions or named_scenarios.empty:
        return pd.DataFrame()
    position_map = {position.name: position for position in positions}
    rows: list[dict[str, Any]] = []
    for row in named_scenarios.to_dict(orient="records"):
        strategy_name = clean_string(row.get("strategy"))
        position = position_map.get(strategy_name)
        valuation_date_text = clean_string(row.get("valuation_date"))
        if position is None or not valuation_date_text:
            continue
        valuation_date = date.fromisoformat(valuation_date_text)
        spot_price = float(row.get("spot_price") or 0.0)
        iv_shift = float(row.get("iv_shift") or 0.0)
        components = _valuation_components(
            position,
            valuation_date=valuation_date,
            spot_price=spot_price,
            iv_shift=iv_shift,
        )
        rows.append(
            {
                "strategy": strategy_name,
                "spot_case": row.get("spot_case"),
                "horizon": row.get("horizon"),
                "iv_case": row.get("iv_case"),
                "valuation_date": valuation_date.isoformat(),
                "modeled_value": components["modeled_value"],
                "profit_loss_now": row.get("profit_loss"),
                "payoff_at_expiry_same_spot": round(float(position.payoff_at_expiry([spot_price])[0]), 4),
                "stock_leg_value": components["stock_leg_value"],
                "option_intrinsic_value": components["option_intrinsic_value"],
                "option_modeled_value": components["option_modeled_value"],
                "option_extrinsic_value": components["option_extrinsic_value"],
                "entry_delta_estimate": _entry_delta_estimate(position),
                "clamped_to_expiry": row.get("clamped_to_expiry"),
            }
        )
    return pd.DataFrame(rows)


def _what_matters_most(
    *,
    status: str,
    comparison_capital: float,
    featured_focus_strategy: str,
    executive_summary: pd.DataFrame,
    decision_hints: dict[str, Any],
) -> str:
    if status == "insufficient_data":
        return (
            "Local chain coverage was too sparse to build a full option set, so start with the stock baseline, "
            "warnings, and any buildable structures before leaning on the comparison rankings."
        )
    executive = executive_summary.iloc[0].to_dict() if not executive_summary.empty else {}
    best_bull = clean_string(executive.get("best_bull_strategy") or (decision_hints.get("best_bull_case") or {}).get("strategy"))
    best_flat = clean_string(executive.get("best_flat_strategy") or (decision_hints.get("best_flat_case") or {}).get("strategy"))
    best_vs_stock = clean_string(executive.get("best_equal_capital_relative_strategy") or (decision_hints.get("best_vs_stock") or {}).get("strategy"))
    if best_bull and best_flat and best_vs_stock:
        return (
            f"Under the ${float(comparison_capital):,.0f} normalized lens, {best_bull.replace('_', ' ').title()} leads the bullish path, "
            f"{best_flat.replace('_', ' ').title()} handles flatter paths best, and {best_vs_stock.replace('_', ' ').title()} "
            f"looks strongest versus simply owning the stock. Use the {featured_focus_strategy.replace('_', ' ').title()} spot x time heatmap "
            "to judge how much timing risk sits underneath that headline ranking."
        )
    return (
        f"Start with the four featured visuals, then use the ${float(comparison_capital):,.0f} normalized tables to confirm "
        "which structure wins only when the path, timing, and affordability all line up."
    )


def _build_scenario_core(
    chain: OptionChain,
    *,
    strategies: Iterable[str] | None = None,
    premium_mode: str = "mid",
    spot_price: float | None = None,
    contract_selector: dict[str, Any] | None = None,
    long_selector: dict[str, Any] | None = None,
    short_selector: dict[str, Any] | None = None,
    spot_case_moves: dict[str, float] | None = None,
    iv_case_points: dict[str, float] | None = None,
    stock_grid_points: int = 21,
    comparison_capital: float = 1000.0,
) -> ScenarioDashboardComputation:
    """Build one multi-strategy scenario dashboard context from an exact chain slice."""

    ticker = clean_string(chain.ticker).upper() or "UNKNOWN"
    snapshot_date = chain.metadata.snapshot_date
    expiry_date = chain.metadata.expiry_date
    if snapshot_date is None or expiry_date is None:
        raise ValueError("Scenario dashboards require a chain with both snapshot_date and expiry_date.")
    spot = float(spot_price or chain.spot_price or chain.metadata.spot_price or 0.0)
    if spot <= 0:
        raise ValueError("Scenario dashboards require a positive spot price.")
    if comparison_capital <= 0:
        raise ValueError("Scenario dashboards require a positive comparison_capital.")

    requested_strategies = [clean_string(name).lower() for name in (strategies or SUPPORTED_SCENARIO_STRATEGIES)]
    requested_strategies = [name for name in requested_strategies if name]
    if "long_stock" not in requested_strategies:
        requested_strategies = ["long_stock"] + requested_strategies

    included: list[StrategyPosition] = []
    omitted: list[dict[str, str]] = []
    warnings: list[str] = list(chain.warnings)

    for strategy_name in requested_strategies:
        if strategy_name not in SUPPORTED_SCENARIO_STRATEGIES:
            omitted.append({"strategy": strategy_name, "reason": "unsupported_strategy"})
            continue
        try:
            position = build_strategy(
                strategy_name,
                chain,
                **_strategy_selector_payload(
                    strategy_name,
                    spot_price=spot_price,
                    premium_mode=premium_mode,
                    contract_selector=contract_selector,
                    long_selector=long_selector,
                    short_selector=short_selector,
                ),
            )
            included.append(position)
            warnings.extend(position.warnings)
        except ValueError as exc:
            omitted.append({"strategy": strategy_name, "reason": clean_string(exc)})

    if not included:
        raise ValueError("No strategies could be constructed for the requested dashboard.")

    included = sorted(included, key=lambda position: (position.name != "long_stock", SUPPORTED_SCENARIO_STRATEGIES.index(position.name)))
    unique_warnings = _dedupe(warnings)
    expiry_days = max((expiry_date - snapshot_date).days, 0)
    scenario_spot_cases = _scenario_case_prices(spot, spot_case_moves)
    spot_case_lookup = {
        round(float(spot_price), 4): label
        for label, spot_price in scenario_spot_cases.items()
    }
    effective_spot_moves = dict(DEFAULT_SPOT_CASES)
    effective_spot_moves.update(spot_case_moves or {})
    named_spot_values = [scenario_spot_cases[label] for label in scenario_spot_cases]
    numeric_spot_grid = build_stock_grid(spot, points=stock_grid_points)
    named_iv_cases = _iv_case_points(iv_case_points)
    horizon_specs = _dashboard_horizon_specs(expiry_days)
    representative_horizon = _representative_horizon(expiry_days)

    named_rows: list[dict[str, Any]] = []
    spot_time_rows: list[dict[str, Any]] = []
    spot_iv_rows: list[dict[str, Any]] = []
    for position in included:
        named_case_rows = _grid_rows(
            position,
            context_expiry_days=expiry_days,
            spot_prices=named_spot_values,
            horizon_specs=horizon_specs,
            iv_cases=named_iv_cases,
        )
        for row in named_case_rows:
            spot_case = next(
                (
                    label
                    for label, case_spot in scenario_spot_cases.items()
                    if abs(float(case_spot) - float(row["spot_price"])) < 1e-6
                ),
                None,
            )
            row["spot_case"] = spot_case
            row["spot_move_pct"] = effective_spot_moves.get(spot_case)
            row["scenario_name"] = f"{spot_case}_{row['horizon']}_{row['iv_case']}"
            named_rows.append(row)

        spot_time_rows.extend(
            _grid_rows(
                position,
                context_expiry_days=expiry_days,
                spot_prices=numeric_spot_grid,
                horizon_specs=horizon_specs,
                iv_cases={REPRESENTATIVE_IV_CASE: named_iv_cases[REPRESENTATIVE_IV_CASE]},
            )
        )
        spot_iv_rows.extend(
            _grid_rows(
                position,
                context_expiry_days=expiry_days,
                spot_prices=numeric_spot_grid,
                horizon_specs=[representative_horizon],
                iv_cases=named_iv_cases,
            )
        )

    named_scenarios = pd.DataFrame(named_rows)
    spot_time_grid = pd.DataFrame(spot_time_rows)
    spot_iv_grid = pd.DataFrame(spot_iv_rows)
    stock_relative = _build_stock_relative(
        included,
        spot_grid=numeric_spot_grid,
        horizon=representative_horizon,
        iv_case=REPRESENTATIVE_IV_CASE,
        iv_shift=named_iv_cases[REPRESENTATIVE_IV_CASE],
        comparison_capital=comparison_capital,
    )
    named_relative_frames: list[pd.DataFrame] = []
    for horizon_label, horizon_days in horizon_specs:
        for iv_case, iv_shift in named_iv_cases.items():
            named_relative_frames.append(
                _build_stock_relative(
                    included,
                    spot_grid=np.asarray(named_spot_values, dtype=float),
                    horizon=(horizon_label, horizon_days),
                    iv_case=iv_case,
                    iv_shift=iv_shift,
                    comparison_capital=comparison_capital,
                )
            )
    named_stock_relative = pd.concat(named_relative_frames, ignore_index=True) if named_relative_frames else pd.DataFrame()

    if not named_scenarios.empty and not named_stock_relative.empty:
        named_relative = named_stock_relative[
            [
                "strategy",
                "mode",
                "spot_price",
                "horizon",
                "iv_case",
                "estimated_value",
                "profit_loss",
                "return_on_comparison_capital",
                "comparison_capital",
                "unit_capital_required",
                "affordable_units",
                "fully_implementable_with_budget",
                "budget_note",
                "stock_estimated_value",
                "stock_profit_loss",
                "stock_relative_difference",
            ]
        ].copy()
        named_relative = named_relative.rename(
            columns={
                "estimated_value": "equal_capital_estimated_value",
                "profit_loss": "equal_capital_profit_loss",
            }
        )
        named_scenarios = named_scenarios.merge(
            named_relative.loc[named_relative["mode"] == "equal_capital"].drop(columns=["mode"]),
            on=["strategy", "spot_price", "horizon", "iv_case"],
            how="left",
        )
        if "stock_profit_loss" in named_scenarios.columns and "comparison_capital" in named_scenarios.columns:
            named_scenarios["stock_return_on_comparison_capital"] = named_scenarios.apply(
                lambda row: finite_or_none(
                    (
                        pd.to_numeric(row.get("stock_profit_loss"), errors="coerce")
                        / pd.to_numeric(row.get("comparison_capital"), errors="coerce")
                    )
                    if pd.notna(pd.to_numeric(row.get("stock_profit_loss"), errors="coerce"))
                    and pd.notna(pd.to_numeric(row.get("comparison_capital"), errors="coerce"))
                    and float(pd.to_numeric(row.get("comparison_capital"), errors="coerce")) > 0
                    else None
                ),
                axis=1,
            )

    strategy_summary = _build_strategy_summary(
        included,
        named_scenarios,
        representative_horizon=representative_horizon[0],
        representative_iv_case=REPRESENTATIVE_IV_CASE,
        comparison_capital=comparison_capital,
    )
    decision_hints = _decision_hints(strategy_summary, stock_relative)
    featured_focus_strategy = _featured_focus_strategy(included)

    option_strategy_count = sum(1 for position in included if position.name != "long_stock")
    resolved_metadata = included[0].resolved_metadata if included else {}
    status = "ok"
    if option_strategy_count == 0:
        status = "insufficient_data"
    elif omitted or any("fallback" in item.lower() for item in unique_warnings):
        status = "partial"
    executive_summary = _executive_summary_row(
        strategy_summary,
        stock_relative,
        ticker=ticker,
        snapshot_date=snapshot_date,
        expiry_date=expiry_date,
        status=status,
        spot_price=spot,
        risk_free_rate=float(chain.metadata.risk_free_rate or 0.0),
        representative_horizon=representative_horizon,
        comparison_capital=comparison_capital,
    )
    valuation_default_horizon = _valuation_default_horizon_label(representative_horizon)
    valuation_explanation = _build_valuation_explanation(included, named_scenarios)
    forward_quick_scenarios = named_scenarios.loc[
        (named_scenarios.get("horizon") == representative_horizon[0])
        & (named_scenarios.get("iv_case") == REPRESENTATIVE_IV_CASE)
        & (named_scenarios.get("spot_case").isin(["bear", "flat", "bull"]))
    ].copy() if not named_scenarios.empty else pd.DataFrame()
    forward_spot_time_grid = _forward_grid_frame(
        included,
        context_expiry_days=expiry_days,
        spot_prices=numeric_spot_grid,
        horizon_specs=horizon_specs,
        iv_cases=named_iv_cases,
        comparison_capital=comparison_capital,
    )
    forward_spot_iv_grid = forward_spot_time_grid.copy()
    forward_time_iv_grid = _forward_grid_frame(
        included,
        context_expiry_days=expiry_days,
        spot_prices=named_spot_values,
        horizon_specs=horizon_specs,
        iv_cases=named_iv_cases,
        comparison_capital=comparison_capital,
        spot_case_lookup=spot_case_lookup,
    )
    replay_defaults = {
        "spot_case": "flat" if "flat" in scenario_spot_cases else next(iter(scenario_spot_cases.keys()), None),
        "iv_case": REPRESENTATIVE_IV_CASE,
        "horizon": representative_horizon[0],
        "compare_mode": "equal_capital",
        "focus_strategy": featured_focus_strategy,
    }
    forward_defaults = {
        "strategy": featured_focus_strategy,
        "metric": "profit_loss",
        "mode": "spot_time",
        "fixed_iv_case": REPRESENTATIVE_IV_CASE,
        "fixed_horizon": representative_horizon[0],
        "fixed_spot_case": "flat" if "flat" in scenario_spot_cases else next(iter(scenario_spot_cases.keys()), None),
        "comparison_capital_label": f"${float(comparison_capital):,.0f}",
    }
    valuation_defaults = {
        "strategy": featured_focus_strategy,
        "spot_case": "flat" if "flat" in scenario_spot_cases else next(iter(scenario_spot_cases.keys()), None),
        "horizon": valuation_default_horizon,
        "iv_case": REPRESENTATIVE_IV_CASE,
    }
    what_matters_most = _what_matters_most(
        status=status,
        comparison_capital=comparison_capital,
        featured_focus_strategy=featured_focus_strategy,
        executive_summary=executive_summary,
        decision_hints=decision_hints,
    )
    scenario_defaults = {
        "spot_cases": {label: {"move_pct": move_pct, "spot_price": scenario_spot_cases[label]} for label, move_pct in {**DEFAULT_SPOT_CASES, **(spot_case_moves or {})}.items() if label in scenario_spot_cases},
        "iv_cases": {label: {"iv_shift": shift} for label, shift in named_iv_cases.items()},
        "horizons": [{"label": label, "requested_days": days} for label, days in horizon_specs],
        "representative_horizon": {"label": representative_horizon[0], "requested_days": representative_horizon[1]},
        "representative_iv_case": {"label": REPRESENTATIVE_IV_CASE, "iv_shift": named_iv_cases[REPRESENTATIVE_IV_CASE]},
        "comparison_capital": round(float(comparison_capital), 4),
        "capital_sizing_mode": CAPITAL_SIZING_MODE,
        "compare_vs_stock_modes": ["equal_capital", "share_equivalent"],
        "spot_time_display_order": ["entry", "1w", "1m", "3m", "6m", "expiry"],
        "spot_iv_display_order": ["iv_down", "iv_unchanged", "iv_up"],
    }
    shareability_status = "mostly_self_contained"

    return ScenarioDashboardComputation(
        ticker=ticker,
        snapshot_date=snapshot_date,
        expiry_date=expiry_date,
        source_snapshot_file=str(chain.source_path),
        spot_price=spot,
        premium_mode=premium_mode,
        risk_free_rate=float(chain.metadata.risk_free_rate or 0.0),
        dividend_yield=float(chain.metadata.dividend_yield or 0.0),
        resolved_metadata=dict(resolved_metadata),
        research_context=dict((resolved_metadata or {}).get("research_context") or {}),
        warnings=unique_warnings,
        status=status,
        shareability_status=shareability_status,
        comparison_capital=float(comparison_capital),
        capital_sizing_mode=CAPITAL_SIZING_MODE,
        featured_focus_strategy=featured_focus_strategy,
        available_strategies=[position.name for position in included],
        omitted_strategies=omitted,
        scenario_defaults=scenario_defaults,
        forward_defaults=forward_defaults,
        decision_hints=decision_hints,
        replay_defaults=replay_defaults,
        valuation_defaults=valuation_defaults,
        what_matters_most=what_matters_most,
        executive_summary=executive_summary,
        strategy_summary=strategy_summary,
        named_scenarios=named_scenarios,
        stock_relative=stock_relative,
        spot_time_grid=spot_time_grid,
        spot_iv_grid=spot_iv_grid,
        forward_quick_scenarios=forward_quick_scenarios,
        forward_spot_time_grid=forward_spot_time_grid,
        forward_spot_iv_grid=forward_spot_iv_grid,
        forward_time_iv_grid=forward_time_iv_grid,
        valuation_explanation=valuation_explanation,
        positions=included,
    )


def build_scenario_analysis(*args, **kwargs) -> ScenarioDashboardComputation:
    """Build one canonical scenario analysis result."""

    if args and isinstance(args[0], OptionChain):
        return _build_scenario_core(*args, **kwargs)
    if "chain" in kwargs and isinstance(kwargs["chain"], OptionChain):
        chain = kwargs.pop("chain")
        return _build_scenario_core(chain, **kwargs)
    chain = load_snapshot_chain(
        kwargs.pop("ticker"),
        snapshot_date=kwargs.pop("snapshot_date"),
        expiry_date=kwargs.pop("expiry_date"),
        data_root=kwargs.pop("data_root", None),
        spot_price=kwargs.get("spot_price"),
        metadata_override=kwargs.pop("metadata_override", None),
    )
    return _build_scenario_core(chain, **kwargs)
