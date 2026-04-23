"""Canonical strategy analysis builder."""

from __future__ import annotations

from .models import StrategyAnalysisComputation
from ..strategies import StrategyPosition


def build_strategy_analysis(
    strategy: StrategyPosition,
    *,
    spot_grid: list[float] | None = None,
    horizons: list[str | int] | None = None,
    iv_shocks: list[float] | None = None,
    comparison_positions: list[StrategyPosition] | None = None,
    comparison_mode: str = "both",
) -> StrategyAnalysisComputation:
    """Wrap one strategy as an analysis-first run with explicit supporting views."""

    return StrategyAnalysisComputation(
        strategy=strategy,
        spot_grid=spot_grid,
        horizons=horizons,
        iv_shocks=iv_shocks,
        comparison_positions=list(comparison_positions or []),
        comparison_mode=comparison_mode,
    )
