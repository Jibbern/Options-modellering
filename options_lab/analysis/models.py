"""Canonical analysis-layer dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..strategies import StrategyPosition


@dataclass
class StrategyAnalysisComputation:
    """Analysis-first wrapper for one strategy plus requested supporting views."""

    strategy: StrategyPosition
    spot_grid: list[float] | None = None
    horizons: list[str | int] | None = None
    iv_shocks: list[float] | None = None
    comparison_positions: list[StrategyPosition] = field(default_factory=list)
    comparison_mode: str = "both"


@dataclass(frozen=True)
class PathHorizonSpec:
    """Canonical named horizon used by analysis-first path workflows."""

    label: str
    requested_days: int


@dataclass(frozen=True)
class RequiredPathSummaryRecord:
    """One summarized required-path row for a candidate or family representative."""

    summary_scope: str
    summary_label: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    goal: str
    iv_variant_kind: str
    iv_variant: str
    first_cleared_horizon: str | None
    required_stock_price_at_target: float | None
    assumed_stock_price_at_target: float | None
    path_gap_at_target: float | None
    required_path_difficulty: str
    unreached: bool
    clamped_to_expiry: bool
    target_beyond_expiry: bool


@dataclass(frozen=True)
class AssumedPathTraceRecord:
    """One modeled checkpoint along the active assumed stock + IV path."""

    trace_scope: str
    series_label: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    horizon: str
    requested_days: int
    spot_price: float
    iv_shift_points: float
    modeled_value: float | None
    profit_loss: float | None
    return_on_comparison_capital: float | None
    stock_modeled_value: float | None
    stock_profit_loss: float | None
    stock_return_on_comparison_capital: float | None
    difference_vs_stock: float | None
    difference_vs_stock_return_pct: float | None
    benchmark_note: str
    worst_interim_profit_loss_to_date: float | None
    drawdown_from_peak_to_date: float | None
    clamped_to_expiry: bool
    target_beyond_expiry: bool


@dataclass
class AnalysisBundle:
    """One canonical analysis bundle on disk."""

    analysis_kind: str
    bundle_dir: Path
    manifest_path: Path
    ticker: str
    snapshot_date: str
    run_slug: str
    file_map: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class IVPathTraceRecord:
    """One IV-path checkpoint used to compare active and preset IV paths."""

    trace_scope: str
    iv_path_name: str
    variant_kind: str
    horizon: str
    requested_days: int
    iv_shift_points: float
    delta_from_entry_iv_shift: float | None


@dataclass(frozen=True)
class CompareVsStockPathRecord:
    """One explicit compare-vs-stock checkpoint along an assumed path."""

    trace_scope: str
    series_label: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    horizon: str
    requested_days: int
    strategy_modeled_value: float | None
    strategy_profit_loss: float | None
    strategy_return_on_comparison_capital: float | None
    stock_modeled_value: float | None
    stock_profit_loss: float | None
    stock_return_on_comparison_capital: float | None
    delta_profit_loss_vs_stock: float | None
    delta_return_pct_vs_stock: float | None
    benchmark_note: str
    clamped_to_expiry: bool
    target_beyond_expiry: bool


@dataclass(frozen=True)
class IVPathSensitivitySummaryRecord:
    """One IV-path sensitivity summary row for a candidate or family representative."""

    summary_scope: str
    summary_label: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    stock_path_name: str
    active_iv_path_name: str
    best_iv_variant: str
    worst_iv_variant: str
    active_profit_loss: float | None
    active_difference_vs_stock: float | None
    active_return_on_comparison_capital: float | None
    best_profit_loss: float | None
    worst_profit_loss: float | None
    pnl_sensitivity_range: float | None
    best_difference_vs_stock: float | None
    worst_difference_vs_stock: float | None
    difference_vs_stock_range: float | None
    best_return_on_comparison_capital: float | None
    worst_return_on_comparison_capital: float | None
    return_sensitivity_range: float | None
    iv_risk: str
    sensitivity_note: str


@dataclass(frozen=True)
class PathRiskSummaryRecord:
    """One path-risk summary row for a candidate or family representative."""

    summary_scope: str
    summary_label: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    goal: str
    stock_path_name: str
    iv_path_name: str
    required_path_difficulty: str
    first_cleared_horizon: str | None
    path_gap_at_target: float | None
    timing_risk: str
    iv_risk: str
    success_dependency: str
    max_downside: float | None
    worst_interim_profit_loss: float | None
    worst_drawdown_from_peak: float | None
    benchmark_edge: float | None
    benchmark_return_edge: float | None
    benchmark_note: str
    confidence_label: str
    coverage_flags: str
    target_beyond_expiry: bool


@dataclass(frozen=True)
class StockPathExampleRecord:
    """One checkpoint along a generated stock path example."""

    path_id: str
    path_kind: str
    path_name: str
    representative_bucket: str
    selection_reason: str
    is_representative: bool
    date: str
    requested_days: int
    step_index: int
    spot_price: float
    return_pct: float | None


@dataclass(frozen=True)
class IVPathExampleRecord:
    """One checkpoint along a generated IV path example."""

    iv_path_id: str
    iv_path_name: str
    representative_bucket: str
    selection_reason: str
    is_representative: bool
    date: str
    requested_days: int
    step_index: int
    iv_shift_points: float


@dataclass(frozen=True)
class PathPairSummaryRecord:
    """One summary row for a stock-path / IV-path pair."""

    path_pair_id: str
    stock_path_id: str
    iv_path_id: str
    stock_path_name: str
    iv_path_name: str
    stock_path_kind: str
    representative_bucket: str
    selection_reason: str
    is_representative: bool
    terminal_stock_price: float | None
    terminal_iv_shift_points: float | None
    final_profit_loss: float | None
    final_difference_vs_stock: float | None
    goal_reached: bool
    outperformed_stock: bool
    goal_success_rate: float | None


@dataclass(frozen=True)
class ValuationOverPathRecord:
    """One daily mark-to-market checkpoint for a candidate under a path pair."""

    path_pair_id: str
    representative_bucket: str
    selection_reason: str
    path_scope: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    expiry_date: str
    strike_label: str
    date: str
    requested_days: int
    step_index: int
    spot_price: float
    iv_shift_points: float
    modeled_value: float | None
    profit_loss: float | None
    return_on_comparison_capital: float | None
    stock_modeled_value: float | None
    stock_profit_loss: float | None
    stock_return_on_comparison_capital: float | None
    difference_vs_stock: float | None
    difference_vs_stock_return_pct: float | None
    benchmark_note: str
    worst_interim_profit_loss_to_date: float | None
    drawdown_from_peak_to_date: float | None
    max_favorable_profit_to_date: float | None
    success_status: str
    goal_reached: bool
    outperformed_stock: bool
    clamped_to_expiry: bool
    target_beyond_expiry: bool


@dataclass(frozen=True)
class CompareVsStockOverPathRecord:
    """One explicit compare-vs-stock checkpoint for a path pair."""

    path_pair_id: str
    representative_bucket: str
    selection_reason: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    date: str
    requested_days: int
    step_index: int
    strategy_profit_loss: float | None
    stock_profit_loss: float | None
    delta_profit_loss_vs_stock: float | None
    strategy_return_on_comparison_capital: float | None
    stock_return_on_comparison_capital: float | None
    delta_return_pct_vs_stock: float | None
    benchmark_note: str


@dataclass(frozen=True)
class RepresentativePathSummaryRecord:
    """One representative path-pair selection summary row."""

    path_pair_id: str
    stock_path_id: str
    iv_path_id: str
    stock_path_name: str
    iv_path_name: str
    representative_bucket: str
    selection_reason: str
    top_candidate_success_status: str
    stock_benchmark_status: str
    terminal_stock_price: float | None
    terminal_iv_shift_points: float | None
    final_profit_loss: float | None
    final_difference_vs_stock: float | None


@dataclass(frozen=True)
class PathComparisonRecord:
    """One strike or expiry comparison row under a fixed representative path pair."""

    comparison_scope: str
    path_pair_id: str
    representative_bucket: str
    selection_reason: str
    strategy_family: str
    strike_label: str
    expiry_date: str
    best_candidate_label: str
    objective_score: float | None
    profit_loss: float | None
    return_on_comparison_capital: float | None
    difference_vs_stock: float | None
    difference_vs_stock_return_pct: float | None
    benchmark_note: str
    required_path_difficulty: str
    timing_risk: str
    iv_risk: str
    success_dependency: str
    source_trust_label: str
    source_quality_note: str
    weak_horizon_fit: bool
    target_beyond_expiry: bool
    clamped_to_expiry: bool


@dataclass(frozen=True)
class RequiredVsAssumedPathSummaryRecord:
    """One comparison row for required path versus assumed and representative paths."""

    comparison_scope: str
    candidate_slug: str
    candidate_label: str
    strategy_family: str
    goal: str
    assumed_path_name: str
    representative_path_pair_id: str
    representative_bucket: str
    first_cleared_horizon: str | None
    required_path_difficulty: str
    assumed_path_gap_at_target: float | None
    representative_path_gap_at_target: float | None
    representative_terminal_stock_price: float | None
    representative_goal_reached: bool
