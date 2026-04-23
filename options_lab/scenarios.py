"""Scenario-table and comparison helpers for strategy analysis."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .strategies import StrategyPosition
from .utils import DEFAULT_HORIZONS, build_stock_grid, horizon_to_days


def resolve_horizons(horizons: Iterable[str | int] | None) -> list[tuple[str, int]]:
    """Normalize user-facing horizon inputs into labels and day counts."""

    values = list(horizons or DEFAULT_HORIZONS)
    resolved: list[tuple[str, int]] = []
    for value in values:
        label = str(value)
        resolved.append((label, horizon_to_days(value)))
    return resolved


def build_case_template(
    spot_price: float,
    *,
    bear_pct: float = -0.20,
    base_pct: float = 0.0,
    bull_pct: float = 0.20,
) -> dict[str, float]:
    """Build a simple bear/base/bull spot template around the current spot."""

    return {
        "bear": round(spot_price * (1.0 + bear_pct), 4),
        "base": round(spot_price * (1.0 + base_pct), 4),
        "bull": round(spot_price * (1.0 + bull_pct), 4),
    }


def scenario_table(
    strategy: StrategyPosition,
    spot_grid=None,
    horizons: Iterable[str | int] | None = None,
    iv_shocks: Iterable[float] | None = None,
    pricing_inputs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Estimate position values across spot, horizon, and IV scenarios."""

    spots = list(spot_grid if spot_grid is not None else build_stock_grid(strategy.entry_spot, points=21))
    horizon_specs = resolve_horizons(horizons)
    shocks = list(iv_shocks if iv_shocks is not None else [0.0])
    pricing = pricing_inputs or {}
    rows: list[dict[str, Any]] = []
    for horizon_label, requested_days in horizon_specs:
        valuation_date, clamped = strategy.valuation_date_for_horizon(requested_days)
        effective_days = (valuation_date - strategy.snapshot_date).days
        for iv_shift in shocks:
            values = strategy.mark_to_market_value(
                spots,
                valuation_date=valuation_date,
                iv_shift=iv_shift,
                risk_free_rate=pricing.get("risk_free_rate"),
                dividend_yield=pricing.get("dividend_yield"),
            )
            profits = values - strategy.initial_outlay
            for spot_price, estimated_value, profit_loss in zip(spots, values, profits):
                rows.append(
                    {
                        "strategy": strategy.name,
                        "horizon": horizon_label,
                        "requested_days": requested_days,
                        "effective_days": effective_days,
                        "clamped_to_expiry": clamped,
                        "valuation_date": valuation_date.isoformat(),
                        "spot_price": round(float(spot_price), 4),
                        "iv_shift": float(iv_shift),
                        "estimated_value": round(float(estimated_value), 4),
                        "profit_loss": round(float(profit_loss), 4),
                        "return_on_capital": (
                            round(float(profit_loss / strategy.capital_required), 6)
                            if strategy.capital_required
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compare_positions(
    positions: list[StrategyPosition],
    mode: str = "both",
    *,
    spot_grid=None,
    horizon: str | int = "1m",
    iv_shift: float = 0.0,
    comparison_capital: float | None = None,
    pricing_inputs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Compare multiple positions on a share-equivalent or equal-capital basis."""

    if not positions:
        return pd.DataFrame()
    spots = list(spot_grid if spot_grid is not None else build_stock_grid(positions[0].entry_spot, points=41))
    pricing = pricing_inputs or {}
    horizon_days = horizon_to_days(horizon)
    reference_capital = float(comparison_capital or 0.0)
    if reference_capital <= 0:
        reference_capital = next((position.capital_required for position in positions if position.capital_required > 0), 1.0)
    requested_modes = {
        "both": ["share_equivalent", "equal_capital"],
        "share_equivalent": ["share_equivalent"],
        "equal_capital": ["equal_capital"],
    }.get(mode, [mode])

    rows: list[dict[str, Any]] = []
    for current_mode in requested_modes:
        for position in positions:
            valuation_date, clamped = position.valuation_date_for_horizon(horizon_days)
            base_values = position.mark_to_market_value(
                spots,
                valuation_date=valuation_date,
                iv_shift=iv_shift,
                risk_free_rate=pricing.get("risk_free_rate"),
                dividend_yield=pricing.get("dividend_yield"),
            )
            base_profits = base_values - position.initial_outlay
            if current_mode == "equal_capital" and position.capital_required > 0:
                scale_factor = reference_capital / position.capital_required
            else:
                scale_factor = 1.0
            unit_capital_required = float(position.capital_required) if position.capital_required > 0 else None
            affordable_units = (
                int(reference_capital // position.capital_required)
                if reference_capital > 0 and position.capital_required > 0
                else None
            )
            fully_implementable = (
                affordable_units is not None and affordable_units >= 1
                if current_mode == "equal_capital"
                else None
            )
            if current_mode == "equal_capital" and reference_capital > 0 and unit_capital_required:
                if affordable_units is not None and affordable_units >= 1:
                    budget_note = (
                        f"At ${reference_capital:,.0f}, this structure can fund {affordable_units} whole unit"
                        + ("s." if affordable_units != 1 else ".")
                    )
                else:
                    budget_note = (
                        f"Not fully implementable at ${reference_capital:,.0f}; one full unit needs "
                        f"${unit_capital_required:,.2f}."
                    )
            else:
                budget_note = None
            for spot_price, estimated_value, profit_loss in zip(spots, base_values, base_profits):
                rows.append(
                    {
                        "strategy": position.name,
                        "mode": current_mode,
                        "scale_factor": round(float(scale_factor), 6),
                        "horizon_days": horizon_days,
                        "valuation_date": valuation_date.isoformat(),
                        "clamped_to_expiry": clamped,
                        "spot_price": round(float(spot_price), 4),
                        "comparison_capital": round(float(reference_capital), 4),
                        "unit_capital_required": round(float(unit_capital_required), 4) if unit_capital_required is not None else None,
                        "affordable_units": affordable_units,
                        "fully_implementable_with_budget": fully_implementable,
                        "budget_note": budget_note,
                        "estimated_value": round(float(estimated_value * scale_factor), 4),
                        "profit_loss": round(float(profit_loss * scale_factor), 4),
                        "return_on_comparison_capital": (
                            round(float((profit_loss * scale_factor) / reference_capital), 6)
                            if current_mode == "equal_capital" and reference_capital > 0
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)
