"""Forward-looking contract and path selection explorer for one ticker + snapshot thesis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import zlib

import numpy as np
import pandas as pd

from .models import (
    CompareVsStockOverPathRecord,
    IVPathExampleRecord,
    PathComparisonRecord,
    PathPairSummaryRecord,
    RepresentativePathSummaryRecord,
    RequiredVsAssumedPathSummaryRecord,
    StockPathExampleRecord,
    ValuationOverPathRecord,
)
from .decision_highlights import build_decision_highlights
from .entry_justification import build_entry_justification
from .paths import (
    build_assumed_path_trace_rows as _build_assumed_path_trace_rows,
    build_compare_vs_stock_path_rows as _build_compare_vs_stock_path_rows,
    build_iv_path_sensitivity_summary as _build_iv_path_sensitivity_summary,
    build_iv_path_trace_rows as _build_iv_path_trace_rows,
    build_path_risk_summary as _build_path_risk_summary,
    canonical_horizon_specs as _canonical_horizon_specs,
    compare_vs_stock_note as _compare_vs_stock_note,
    default_iv_path_points as _default_iv_path_points,
    default_stock_path_points as _default_stock_path_points,
    interpolated_path as _interpolated_path,
    parse_path_points as _parse_path_points,
    required_path_difficulty as _shared_required_path_difficulty,
    required_goals as _required_goals,
    summarize_required_path_rows as _summarize_required_path_rows,
)
from .simulation import (
    IV_PATH_GALLERY_PRESETS as _IV_PATH_GALLERY_PRESETS,
    PATH_CENTRIC_FOCUS_PRESETS as _PATH_CENTRIC_FOCUS_PRESETS,
    build_stock_path_library_rows as _build_stock_path_library_rows,
    build_iv_path_gallery_rows as _build_iv_path_gallery_rows,
    build_iv_path_example as _build_iv_path_example,
    build_iv_path_from_named_points as _build_iv_path_from_named_points,
    build_path_grid as _build_path_grid,
    build_stock_path_gallery_rows as _build_stock_path_gallery_rows,
    build_stock_path_example as _build_stock_path_example,
    build_stock_path_from_named_points as _build_stock_path_from_named_points,
    build_stock_path_pool as _build_stock_path_pool,
    humanize_named_path as _humanize_named_path,
    pair_stock_and_iv_paths as _pair_stock_and_iv_paths,
    select_representative_path_pairs as _select_representative_path_pairs,
    stock_path_family_metadata as _stock_path_family_metadata,
    stock_path_gallery_named_points as _stock_path_gallery_named_points,
)
from ..io import OptionChain, OptionContract, load_chain
from ..snapshots import list_snapshot_slices, snapshot_slices_for_date
from ..strategies import StrategyPosition, build_strategy
from ..utils import build_stock_grid, clean_string, finite_or_none, horizon_to_days, parse_date, slugify
from .scenario import _budget_fields, _dedupe
from .market_context import resolve_market_context


SUPPORTED_CONTRACT_SELECTION_FAMILIES = [
    "long_stock",
    "long_call",
    "long_put",
    "bull_call_spread",
    "bear_put_spread",
    "covered_call",
    "cash_secured_put",
]

DEFAULT_COMPARISON_CAPITAL = 1000.0
DEFAULT_TOP_N_STRIKES = 3
DEFAULT_GOAL = "break_even"
DEFAULT_OBJECTIVE_MODE = "max_return_at_target"
OBJECTIVE_MODE_CHOICES = [
    "max_return_at_target",
    "outperform_stock",
    "capital_efficiency",
    "downside_control",
    "robustness_iv_fall",
    "move_takes_time",
    "highest_convexity",
]
PREFERENCE_BAND_CHOICES = ["low", "medium", "high"]
POINT_IV_SCENARIOS = [-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30]
DEFAULT_REQUIRED_GOALS = ["itm_1c", "break_even", "return_25", "return_50", "outperform_stock"]
LOW_INFORMATION_CARD_STATUSES = {"weak_differentiation", "no_clear_edge"}
DEFAULT_PATH_CASE_MOVES = [-0.20, -0.10, 0.00, 0.10, 0.20]
STOCK_PATH_MODE_CHOICES = ["deterministic", "simulated", "conditioned", "mixed"]
IV_PATH_MODE_CHOICES = ["active_only", "presets", "mixed", "noisy"]
REPRESENTATIVE_SELECTION_MODE_CHOICES = ["goal_buckets"]
PATH_CASE_DIFFICULTY_ORDER = [
    "cleared comfortably",
    "roughly matched",
    "needs more / faster",
    "unreached in sampled range",
]
PREFERRED_SOURCE_TRUST_LABELS = {"trusted_quoted", "quoted_prior_day"}
MAX_LONG_CALL_VIEW_LINES = 6
IV_EXPANDED_CHART_IV_PRESETS = [
    "flat",
    "mean_reversion_lower",
    "mean_reversion_higher",
    "earnings_build_then_crush",
]
THESIS_STOCK_PATH_PRESETS = [
    "early_breakout_to_target",
    "slow_grind_to_target",
    "down_then_recover_to_target",
    "rally_retrace_finish_target",
    "late_breakout_to_target",
    "overshoot_then_settle_at_target",
    "fast_overshoot_then_sideways",
    "weak_start_then_acceleration",
    "two_stage_bull_run",
    "violent_path_to_target",
]
THESIS_IV_PATH_PRESETS = [
    "flat",
    "mean_reversion_lower",
    "mean_reversion_higher",
    "iv_up_then_down",
    "iv_down_then_stays_low",
    "earnings_build_then_crush",
]
THESIS_CANDIDATE_LIMIT = 8
LOWER_IV_PRESETS = {"mean_reversion_lower", "iv_down_then_stays_low"}
HIGHER_IV_PRESETS = {"mean_reversion_higher", "iv_up_then_down", "earnings_build_then_crush"}
IV_EXPANDED_CHART_CONTRACT_LIMIT = 3
SINGLE_OPTION_OUTCOME_LABELS = (
    "clear_option_win",
    "wins_but_not_enough",
    "stock_better",
    "fail_too_narrow_or_expiry_issue",
)
SINGLE_OPTION_EXIT_RULE_CHOICES = [
    "hold_to_expiry",
    "sell_at_target_return",
    "sell_on_thesis_completion",
]
SINGLE_OPTION_ENTRY_PRICE_MODES = [
    "conservative_mid_plus_slippage",
    "mid",
    "ask_or_mid",
]
SINGLE_OPTION_DEFAULT_IV_MODES = ("low", "base", "high")
SINGLE_OPTION_REPRESENTATIVE_PATH_ROLES = [
    (
        "early_rally_path",
        [
            "early_breakout_to_target",
            "rally_early_then_fade_then_rally_again",
            "plus_20_pct_in_1m",
            "quarter_up_then_pullback",
        ],
    ),
    (
        "late_rally_path",
        [
            "late_breakout_to_target",
            "late_breakout",
            "reaches_target_late_near_expiry",
        ],
    ),
    (
        "steady_grind_up_path",
        [
            "slow_grind_to_target",
            "slow_grind_up",
            "two_stage_bull_run",
        ],
    ),
    (
        "false_breakout_failed_path",
        [
            "false_breakout_then_recover",
            "early_move_above_strike_then_giveback",
            "overshoot_then_mean_revert",
        ],
    ),
    (
        "recovery_path",
        [
            "down_then_recover_to_target",
            "down_first_then_recovery",
            "earnings_gap_down_then_recovery",
        ],
    ),
    (
        "earnings_gap_path",
        [
            "earnings_gap_up_then_fade",
            "earnings_gap_down_then_recovery",
        ],
    ),
]
CHAIN_OVERVIEW_CARD_KEYS = (
    "best_robust_option",
    "best_asymmetric_upside",
    "best_early_move_option",
    "best_late_move_option",
    "too_iv_sensitive",
    "stock_better_than_these_calls",
)
CHAIN_OVERVIEW_VERDICT_LABELS = {
    "robust_buy_candidate": "Robust buy candidate",
    "selective_thesis_dependent": "Selective / thesis-dependent",
    "too_narrow": "Too narrow",
    "stock_better": "Stock better",
}


@dataclass
class ContractSelectionComputation:
    """Computed tables and metadata for one contract selection explorer."""

    ticker: str
    snapshot_date: date
    target_price: float
    target_date: date
    target_horizon_label: str
    target_horizon_days: int
    iv_shift_points: float
    comparison_capital: float
    strategy_families: list[str]
    spot_price: float
    risk_free_rate: float
    dividend_yield: float
    available_expiries: list[str]
    selection_scope: dict[str, Any]
    chain_source_summary: pd.DataFrame
    market_context_summary: pd.DataFrame
    research_context: dict[str, Any]
    spot_price_source: str | None
    spot_price_matched_date: str | None
    spot_price_field_used: str | None
    spot_price_used_prior_date: bool
    spot_price_note: str | None
    spot_quality_note: str | None
    ibkr_same_day_spot_attempted: bool
    ibkr_same_day_spot_rejected_reason: str | None
    status: str
    shareability_status: str
    warnings: list[str]
    goal: str
    target_option_value: float | None
    objective_mode: str
    downside_tolerance: str
    simplicity_preference: str
    stock_path_name: str
    iv_path_name: str
    stock_path_points: dict[str, float]
    iv_path_points: dict[str, float]
    stock_path_mode: str
    stock_path_target_end: float
    iv_path_mode: str
    simulated_path_count: int
    representative_selection_mode: str
    simulation_seed: int | None
    run_slug: str
    generated_at: str
    candidate_summary: pd.DataFrame
    ranked_candidates: pd.DataFrame
    compare_vs_stock: pd.DataFrame
    required_path_rows: pd.DataFrame
    required_path_summary: pd.DataFrame
    assumed_path_trace_rows: pd.DataFrame
    iv_path_trace_rows: pd.DataFrame
    compare_vs_stock_path_rows: pd.DataFrame
    iv_path_sensitivity_summary: pd.DataFrame
    path_risk_summary: pd.DataFrame
    path_case_rows: pd.DataFrame
    path_case_summary: pd.DataFrame
    path_case_chart_rows: pd.DataFrame
    path_case_strategy_rows: pd.DataFrame
    path_case_family_rankings: pd.DataFrame
    path_case_candidate_rankings: pd.DataFrame
    strategy_selector_rows: pd.DataFrame
    strategy_selector_rankings: pd.DataFrame
    family_comparison: pd.DataFrame
    candidate_comparison: pd.DataFrame
    strike_comparison: pd.DataFrame
    expiry_comparison: pd.DataFrame
    stock_path_library: pd.DataFrame
    stock_path_gallery: pd.DataFrame
    iv_path_gallery: pd.DataFrame
    stock_path_examples: pd.DataFrame
    iv_path_examples: pd.DataFrame
    path_pair_summary: pd.DataFrame
    option_value_over_path: pd.DataFrame
    compare_vs_stock_over_path: pd.DataFrame
    representative_paths_summary: pd.DataFrame
    strike_comparison_under_path: pd.DataFrame
    expiry_comparison_under_path: pd.DataFrame
    long_call_value_over_path_strike_view: pd.DataFrame
    long_call_value_over_path_expiry_view: pd.DataFrame
    long_call_value_over_path_best_of: pd.DataFrame
    decision_highlights: pd.DataFrame
    decision_highlights_explanations: pd.DataFrame
    candidate_robustness_summary: pd.DataFrame
    candidate_tradeoff_matrix: pd.DataFrame
    stock_vs_option_takeaways: pd.DataFrame
    highlights_score_breakdown: pd.DataFrame
    highlights_markdown: str
    action_board_candidates: pd.DataFrame
    buy_now_candidates: pd.DataFrame
    watchlist_candidates: pd.DataFrame
    avoid_for_now_candidates: pd.DataFrame
    prefer_stock_instead: pd.DataFrame
    decision_triggers: pd.DataFrame
    action_board_score_breakdown: pd.DataFrame
    action_board_explanations: pd.DataFrame
    action_board_markdown: str
    bullish_long_call_action_board: pd.DataFrame
    bullish_long_call_watchlist: pd.DataFrame
    bullish_long_call_avoid: pd.DataFrame
    bullish_long_call_triggers: pd.DataFrame
    bullish_long_call_score_breakdown: pd.DataFrame
    other_structures_summary: pd.DataFrame
    stock_preference_summary: pd.DataFrame
    bullish_action_board_markdown: str
    top_candidate_cards: pd.DataFrame
    top_candidate_cards_markdown: str
    other_structures_markdown: str
    entry_justification_candidates: pd.DataFrame
    required_stock_path_to_buy: pd.DataFrame
    required_move_summary: pd.DataFrame
    required_move_vs_stock: pd.DataFrame
    required_iv_support_summary: pd.DataFrame
    entry_barrier_summary: pd.DataFrame
    entry_justification_markdown: str
    thesis_target_price: float
    thesis_target_date: date
    thesis_path_gallery: pd.DataFrame
    thesis_iv_gallery: pd.DataFrame
    thesis_mode_candidates: pd.DataFrame
    thesis_path_family_summary: pd.DataFrame
    thesis_iv_family_summary: pd.DataFrame
    thesis_candidate_ranking: pd.DataFrame
    max_justified_premium: pd.DataFrame
    current_vs_justified_premium: pd.DataFrame
    thesis_required_move_summary: pd.DataFrame
    thesis_stock_vs_option_summary: pd.DataFrame
    thesis_mode_markdown: str
    candidate_stress_grid: pd.DataFrame
    premium_sensitivity_summary: pd.DataFrame
    timing_slip_summary: pd.DataFrame
    target_stress_summary: pd.DataFrame
    stress_transition_summary: pd.DataFrame
    stress_tests_markdown: str
    chain_overview_summary: pd.DataFrame
    chain_overview_candidates: pd.DataFrame
    chain_overview_markdown: str
    single_option_decision_summary: pd.DataFrame
    single_option_decision_path_selections: pd.DataFrame
    single_option_representative_paths: pd.DataFrame
    single_option_path_outcomes: pd.DataFrame
    single_option_required_path_to_beat_stock_1_5x: pd.DataFrame
    single_option_required_path_to_beat_stock_2_0x: pd.DataFrame
    single_option_closest_representative_path_to_edge: pd.DataFrame
    single_option_edge_gap_by_path_family: pd.DataFrame
    single_option_path_family_counts: pd.DataFrame
    single_option_timing_sensitivity: pd.DataFrame
    single_option_iv_sensitivity: pd.DataFrame
    single_option_entry_sensitivity: pd.DataFrame
    single_option_summary_bullets: pd.DataFrame
    single_option_decision_markdown: str
    path_view_tables: dict[str, pd.DataFrame]
    required_vs_assumed_path_summary: pd.DataFrame
    strategy_selector_context: dict[str, Any]
    calibration_context: dict[str, Any]
    best_candidate_cards: list[dict[str, Any]]
    strategy_selector_best_cards: list[dict[str, Any]]
    summary_markdown: str
    report_metadata: dict[str, Any]


STRATEGY_ROLE_MAP = {
    "long_stock": "Simple no-expiry benchmark with full directional exposure.",
    "long_call": "Bullish convex upside with limited premium risk.",
    "bull_call_spread": "Bullish upside with lower premium cost but capped gains.",
    "long_put": "Bearish downside expression with limited premium risk.",
    "bear_put_spread": "Bearish downside with capped gains and lower premium than a naked put.",
    "covered_call": "Income-style stock overlay that gives up upside in exchange for premium.",
    "cash_secured_put": "Paid-to-wait bullish income structure with assignment risk.",
}

STRATEGY_WIN_LOSE_MAP = {
    "long_stock": (
        "Wins when the thesis is broadly right and you want no expiry or IV dependency.",
        "Loses when capped downside matters more than simplicity or when convexity is needed.",
    ),
    "long_call": (
        "Wins when the stock moves fast and far enough to justify premium decay.",
        "Loses when IV falls hard or the move takes too long.",
    ),
    "bull_call_spread": (
        "Wins when you want bullish exposure with lower premium outlay and more defined downside.",
        "Loses when the stock explodes beyond the short strike and capped upside becomes painful.",
    ),
    "long_put": (
        "Wins when you want clean bearish exposure with limited premium risk and downside asymmetry.",
        "Loses when the bearish move is late or IV collapses after entry.",
    ),
    "bear_put_spread": (
        "Wins when you want bearish exposure with tighter cost control than a naked long put.",
        "Loses when the downside move becomes much larger than the spread cap.",
    ),
    "covered_call": (
        "Wins when you want simpler stock ownership plus some premium income in a flatter or slower-up tape.",
        "Loses when a fast upside rerating makes the capped call-away payoff feel restrictive.",
    ),
    "cash_secured_put": (
        "Wins when you are comfortable being paid to wait or potentially buy stock lower.",
        "Loses when the thesis turns sharply bearish and assignment risk dominates the collected premium.",
    ),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _short_slug_token(value: str, *, limit: int = 4) -> str:
    token = slugify(clean_string(value) or "na")
    if not token:
        return "na"
    compact = token.replace("-", "")
    return compact[:limit] or token[:limit]


def _target_date_and_horizon(
    snapshot_date: date,
    *,
    target_date: date | str | None,
    target_horizon: str | int | float | None,
) -> tuple[date, str, int]:
    if target_date is None and target_horizon is None:
        raise ValueError("Contract selection requires either target_date or target_horizon.")
    if target_date is not None:
        resolved_target = parse_date(target_date)
        if resolved_target is None:
            raise ValueError(f"target_date must be a valid date, got: {target_date!r}")
        days = max((resolved_target - snapshot_date).days, 0)
        label = f"{days}d" if days else "entry"
        return resolved_target, label, days
    label = clean_string(target_horizon or "1m").lower()
    days = horizon_to_days(label)
    return snapshot_date.fromordinal(snapshot_date.toordinal() + days), label, days


def _load_candidate_chains(
    ticker: str,
    snapshot_date: date,
    *,
    target_date: date,
    data_root: str | Path | None,
    expiry_selection_mode: str,
) -> Any:
    return resolve_market_context(
        ticker=ticker,
        snapshot_date=snapshot_date,
        target_date=target_date,
        data_root=data_root,
    )


def _rank_contract_rows(frame: pd.DataFrame, *, spot_price: float, option_type: str, top_n: int) -> list[pd.Series]:
    if frame.empty:
        return []
    working = frame.copy()
    working = working.loc[working["has_quote"]].copy()
    if working.empty:
        return []
    working["distance"] = (pd.to_numeric(working["strike"], errors="coerce") - float(spot_price)).abs()
    working["liquidity_rank"] = pd.to_numeric(working["open_interest"], errors="coerce").fillna(0) + pd.to_numeric(
        working["volume"], errors="coerce"
    ).fillna(0)
    working = working.sort_values(["distance", "liquidity_rank"], ascending=[True, False])

    rows: list[pd.Series] = []
    seen: set[float] = set()

    def add_row(candidate: pd.Series | None) -> None:
        if candidate is None:
            return
        strike = float(candidate["strike"])
        if strike in seen:
            return
        seen.add(strike)
        rows.append(candidate)

    add_row(working.iloc[0] if not working.empty else None)

    if option_type == "call":
        slightly_otm = working.loc[pd.to_numeric(working["strike"], errors="coerce") >= float(spot_price)]
        itm = working.loc[pd.to_numeric(working["strike"], errors="coerce") < float(spot_price)]
    else:
        slightly_otm = working.loc[pd.to_numeric(working["strike"], errors="coerce") <= float(spot_price)]
        itm = working.loc[pd.to_numeric(working["strike"], errors="coerce") > float(spot_price)]
    add_row(slightly_otm.iloc[0] if not slightly_otm.empty else None)
    add_row(itm.iloc[0] if not itm.empty else None)

    with_delta = working.dropna(subset=["delta"]).copy()
    if not with_delta.empty:
        with_delta["delta_distance"] = (pd.to_numeric(with_delta["delta"], errors="coerce").abs() - 0.35).abs()
        add_row(with_delta.sort_values(["delta_distance", "liquidity_rank"]).iloc[0])

    for _, candidate in working.iterrows():
        add_row(candidate)
        if len(rows) >= max(int(top_n), 1):
            break
    return rows[: max(int(top_n), 1)]


def _adjacent_spread_contract(chain: OptionChain, contract: OptionContract, *, direction: str) -> OptionContract | None:
    frame = chain.filter_expiry(contract.expiry_date)
    frame = frame.loc[frame["option_type"] == contract.option_type].copy()
    strikes = pd.to_numeric(frame["strike"], errors="coerce")
    if direction == "higher":
        frame = frame.loc[strikes > float(contract.strike)].sort_values("strike")
    else:
        frame = frame.loc[strikes < float(contract.strike)].sort_values("strike", ascending=False)
    if frame.empty:
        return None
    return chain.to_contract(frame.iloc[0])


def _sensitivity_metrics(
    position: StrategyPosition,
    *,
    target_price: float,
    requested_days: int,
    iv_shift_points: float,
) -> dict[str, Any]:
    base_date, _ = position.valuation_date_for_horizon(requested_days)
    base_value = float(position.mark_to_market_value([target_price], valuation_date=base_date, iv_shift=iv_shift_points)[0])
    iv_down_value = float(position.mark_to_market_value([target_price], valuation_date=base_date, iv_shift=iv_shift_points - 0.10)[0])
    iv_up_value = float(position.mark_to_market_value([target_price], valuation_date=base_date, iv_shift=iv_shift_points + 0.10)[0])
    delayed_date, delayed_clamped = position.valuation_date_for_horizon(requested_days + 30)
    delayed_value = float(position.mark_to_market_value([target_price], valuation_date=delayed_date, iv_shift=iv_shift_points)[0])
    iv_down_change = round(iv_down_value - base_value, 4)
    iv_up_change = round(iv_up_value - base_value, 4)
    delayed_change = round(delayed_value - base_value, 4)
    iv_note = (
        "Holds value better if IV falls."
        if iv_down_value >= base_value
        else f"Loses about ${abs(base_value - iv_down_value):,.0f} if IV drops 10 vol points."
    )
    if delayed_clamped:
        time_note = "Longer wait would clamp to expiry for this contract."
    elif delayed_value >= base_value:
        time_note = "Can tolerate a slower move better than the base case."
    else:
        time_note = f"Loses about ${abs(base_value - delayed_value):,.0f} if the move takes an extra month."
    return {
        "iv_sensitivity_summary": iv_note,
        "time_sensitivity_summary": time_note,
        "iv_down_value_change": iv_down_change,
        "iv_up_value_change": round(iv_up_change, 4),
        "delayed_move_value_change": delayed_change,
        "delayed_move_clamped": bool(delayed_clamped),
    }


def _evaluate_position(
    position: StrategyPosition,
    *,
    target_price: float,
    target_date: date,
    requested_days: int,
    iv_shift_points: float,
    comparison_capital: float,
    stock_baseline: StrategyPosition,
) -> dict[str, Any]:
    valuation_date, clamped = position.valuation_date_for_horizon(requested_days)
    effective_days = max((valuation_date - position.snapshot_date).days, 0)
    estimated_value = float(position.mark_to_market_value([target_price], valuation_date=valuation_date, iv_shift=iv_shift_points)[0])
    profit_loss = estimated_value - position.initial_outlay
    return_on_capital = finite_or_none(
        profit_loss / comparison_capital if comparison_capital and comparison_capital > 0 else None
    )
    stock_estimated_value = float(
        stock_baseline.mark_to_market_value([target_price], valuation_date=target_date, iv_shift=0.0)[0]
    )
    stock_profit_loss = stock_estimated_value - stock_baseline.initial_outlay
    stock_return = finite_or_none(
        stock_profit_loss / comparison_capital if comparison_capital and comparison_capital > 0 else None
    )
    return {
        "valuation_date": valuation_date.isoformat(),
        "requested_days": int(requested_days),
        "effective_days": int(effective_days),
        "clamped_to_expiry": bool(clamped),
        "estimated_value": round(estimated_value, 4),
        "profit_loss": round(profit_loss, 4),
        "return_on_comparison_capital": return_on_capital,
        "stock_estimated_value": round(stock_estimated_value, 4),
        "stock_profit_loss": round(stock_profit_loss, 4),
        "stock_return_on_comparison_capital": stock_return,
        "difference_vs_stock": round(profit_loss - stock_profit_loss, 4),
        "comparison_profit_loss": round(profit_loss, 4),
    }


def _candidate_fit_and_confidence_fields(
    *,
    strategy_family: str,
    selection_scope: str,
    requested_days: int,
    effective_days: int,
    clamped_to_expiry: bool,
    source_quality: str | None,
    source_trust_label: str | None,
) -> dict[str, Any]:
    requested = max(int(requested_days), 0)
    effective = max(int(effective_days), 0)
    target_beyond_expiry = bool(
        clean_string(strategy_family).lower() != "long_stock"
        and clamped_to_expiry
        and requested > effective
    )
    timing_match_ratio = 1.0 if requested <= 0 else min(float(effective) / float(requested), 1.0)
    if target_beyond_expiry:
        horizon_fit_label = "poor horizon fit" if timing_match_ratio < 0.7 else "weak timing match"
    else:
        horizon_fit_label = "exact timing match"
    source_quality_label = clean_string(source_quality).lower()
    source_trust = clean_string(source_trust_label).lower()
    coverage_flags = ["exact coverage" if selection_scope == "exact_snapshot" else "nearby snapshot fallback"]
    if source_quality_label == "same_day_quoted":
        coverage_flags.append("same-day quoted source")
    elif source_quality_label == "prior_day_quoted":
        coverage_flags.append("prior-day quoted source")
    elif source_quality_label == "same_day_sparse":
        coverage_flags.append("same-day sparse fallback")
    elif source_quality_label == "prior_day_sparse":
        coverage_flags.append("prior-day sparse fallback")
    if target_beyond_expiry:
        coverage_flags.extend(["target beyond expiry", "expiry-clamped estimate"])
    if source_trust == "trusted_quoted" and selection_scope == "exact_snapshot" and not target_beyond_expiry:
        confidence_label = "high confidence"
    elif source_trust == "quoted_prior_day" and not target_beyond_expiry:
        confidence_label = "medium confidence"
    elif target_beyond_expiry:
        confidence_label = "cautious"
    else:
        confidence_label = "cautious"
    return {
        "target_beyond_expiry": target_beyond_expiry,
        "expiry_clamped_estimate": target_beyond_expiry,
        "timing_match_ratio": round(float(timing_match_ratio), 4),
        "timing_gap_days": max(requested - effective, 0),
        "horizon_fit_label": horizon_fit_label,
        "coverage_flags": " | ".join(coverage_flags),
        "coverage_flag_count": len(coverage_flags),
        "confidence_label": confidence_label,
    }


def _moneyness_bucket(*, strategy_family: str, primary_strike: float | None, spot_price: float) -> str:
    family = clean_string(strategy_family).lower()
    strike = finite_or_none(primary_strike)
    spot = float(spot_price or 0.0)
    if family == "long_stock":
        return "stock"
    if strike is None or spot <= 0:
        return "unknown"
    if family == "long_call":
        distance_pct = float(strike) / float(spot) - 1.0
        if distance_pct <= -0.05:
            return "itm"
        if distance_pct <= 0.03:
            return "near_atm"
        if distance_pct <= 0.12:
            return "otm"
        return "far_otm"
    if family == "long_put":
        distance_pct = float(spot) / float(strike) - 1.0
        if distance_pct <= -0.05:
            return "itm"
        if distance_pct <= 0.03:
            return "near_atm"
        if distance_pct <= 0.12:
            return "otm"
        return "far_otm"
    return clean_string(family) or "unknown"


def _build_candidate_row(
    *,
    candidate_slug: str,
    strategy_family: str,
    candidate_label: str,
    selection_scope: str,
    chain: OptionChain,
    position: StrategyPosition,
    target_price: float,
    target_date: date,
    target_horizon_label: str,
    requested_days: int,
    iv_shift_points: float,
    comparison_capital: float,
    stock_baseline: StrategyPosition,
    strike_selection_mode: str,
    source_snapshot_date: str,
    source_expiry_date: str | None,
    source_storage_location: str,
    source_snapshot_file: str,
    source_quote_coverage_pct: float | None = None,
    source_quote_usable: bool | None = None,
    fallback_level: str | None = None,
    source_quality: str | None = None,
    source_trust_label: str | None = None,
    source_quality_note: str | None = None,
) -> dict[str, Any]:
    summary = position.summary_metrics()
    budget = _budget_fields(position, comparison_capital)
    sensitivity = _sensitivity_metrics(
        position,
        target_price=target_price,
        requested_days=requested_days,
        iv_shift_points=iv_shift_points,
    )
    evaluation = _evaluate_position(
        position,
        target_price=target_price,
        target_date=target_date,
        requested_days=requested_days,
        iv_shift_points=iv_shift_points,
        comparison_capital=comparison_capital,
        stock_baseline=stock_baseline,
    )
    fit = _candidate_fit_and_confidence_fields(
        strategy_family=strategy_family,
        selection_scope=selection_scope,
        requested_days=requested_days,
        effective_days=int(evaluation.get("effective_days") or 0),
        clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
        source_quality=source_quality,
        source_trust_label=source_trust_label,
    )
    option_legs = position.option_legs
    primary_strike = option_legs[0].strike if option_legs else None
    secondary_strike = option_legs[1].strike if len(option_legs) > 1 else None
    strike_label = (
        f"{primary_strike:.2f}/{secondary_strike:.2f}"
        if primary_strike is not None and secondary_strike is not None
        else (f"{primary_strike:.2f}" if primary_strike is not None else "Stock")
    )
    notes = _dedupe(list(position.warnings) + list(position.notes))
    difference_vs_stock_return_pct = (
        round(
            float(evaluation["return_on_comparison_capital"]) - float(evaluation["stock_return_on_comparison_capital"]),
            6,
        )
        if finite_or_none(evaluation.get("return_on_comparison_capital")) is not None
        and finite_or_none(evaluation.get("stock_return_on_comparison_capital")) is not None
        else None
    )
    moneyness_bucket = _moneyness_bucket(
        strategy_family=strategy_family,
        primary_strike=primary_strike,
        spot_price=float(chain.spot_price or chain.metadata.spot_price or 0.0),
    )
    return {
        "candidate_slug": candidate_slug,
        "candidate_label": candidate_label,
        "strategy_family": strategy_family,
        "selection_scope": selection_scope,
        "selection_scope_label": "Exact snapshot" if selection_scope == "exact_snapshot" else "Nearby snapshot fallback",
        "source_snapshot_date": source_snapshot_date,
        "source_storage_location": source_storage_location,
        "source_snapshot_file": source_snapshot_file,
        "source_quote_coverage_pct": source_quote_coverage_pct,
        "source_quote_usable": source_quote_usable,
        "source_fallback_level": clean_string(fallback_level).lower() or None,
        "source_quality": clean_string(source_quality).lower() or None,
        "source_trust_label": clean_string(source_trust_label).lower() or None,
        "source_quality_note": clean_string(source_quality_note) or None,
        "expiry_date": source_expiry_date,
        "target_price": round(float(target_price), 4),
        "target_date": target_date.isoformat(),
        "target_horizon_label": target_horizon_label,
        "iv_shift_points": round(float(iv_shift_points), 4),
        "strike_selection_mode": strike_selection_mode,
        "moneyness_bucket": moneyness_bucket,
        "strike_label": strike_label,
        "primary_strike": primary_strike,
        "secondary_strike": secondary_strike,
        "premium_or_entry_cost": summary.get("initial_outlay"),
        "break_even": summary.get("break_even"),
        "max_loss": summary.get("max_loss"),
        "max_gain": summary.get("max_gain"),
        "capital_required": summary.get("capital_required"),
        "leg_summary": " | ".join(clean_string(leg.label) for leg in position.legs if clean_string(leg.label)),
        "warning_or_note": notes[0] if notes else None,
        "expected_move_pct": summary.get("expected_move_pct"),
        "iv_rank": summary.get("iv_rank"),
        "iv_percentile": summary.get("iv_percentile"),
        "nearest_event_date": summary.get("nearest_event_date"),
        "nearest_event_type": summary.get("nearest_event_type"),
        "difference_vs_stock_return_pct": difference_vs_stock_return_pct,
        "benchmark_note": _compare_vs_stock_note(
            strategy_family=strategy_family,
            difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
            difference_vs_stock_return_pct=difference_vs_stock_return_pct,
            clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
            target_beyond_expiry=bool(fit.get("target_beyond_expiry")),
        ),
        **budget,
        **evaluation,
        **fit,
        **sensitivity,
    }


def _discover_candidates_for_chain(
    chain: OptionChain,
    *,
    scope: str,
    target_price: float,
    target_date: date,
    target_horizon_label: str,
    requested_days: int,
    iv_shift_points: float,
    comparison_capital: float,
    strategy_families: Iterable[str],
    strike_selection_mode: str,
    top_n_strikes: int,
    source_snapshot_date: str,
    source_expiry_date: str | None,
    source_storage_location: str,
    source_snapshot_file: str,
    source_quote_coverage_pct: float | None = None,
    source_quote_usable: bool | None = None,
    fallback_level: str | None = None,
    source_quality: str | None = None,
    source_trust_label: str | None = None,
    source_quality_note: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    specs: list[dict[str, Any]] = []
    warnings: list[str] = []
    stock_baseline = build_strategy("long_stock", chain, premium_mode="mid")
    source_location = clean_string(source_storage_location).lower()

    def append_spec(*, candidate_slug: str, candidate_label: str, family: str, position: StrategyPosition) -> None:
        specs.append(
            {
                "candidate_slug": candidate_slug,
                "candidate_label": candidate_label,
                "strategy_family": family,
                "selection_scope": scope,
                "position": position,
                "stock_baseline": stock_baseline,
                "source_snapshot_date": source_snapshot_date,
                "source_expiry_date": source_expiry_date,
                "source_storage_location": source_location,
                "source_snapshot_file": source_snapshot_file,
                "source_quote_coverage_pct": source_quote_coverage_pct,
                "source_quote_usable": source_quote_usable,
                "source_fallback_level": clean_string(fallback_level).lower() or None,
                "source_quality": clean_string(source_quality).lower() or None,
                "source_trust_label": clean_string(source_trust_label).lower() or None,
                "source_quality_note": clean_string(source_quality_note) or None,
            }
        )

    if "long_stock" in strategy_families:
        row = _build_candidate_row(
            candidate_slug="long-stock-baseline",
            strategy_family="long_stock",
            candidate_label="Long Stock Baseline",
            selection_scope=scope,
            chain=chain,
            position=stock_baseline,
            target_price=target_price,
            target_date=target_date,
            target_horizon_label=target_horizon_label,
            requested_days=requested_days,
            iv_shift_points=0.0,
            comparison_capital=comparison_capital,
            stock_baseline=stock_baseline,
            strike_selection_mode=strike_selection_mode,
            source_snapshot_date=source_snapshot_date,
            source_expiry_date=source_expiry_date,
            source_storage_location=source_location,
            source_snapshot_file=source_snapshot_file,
            source_quote_coverage_pct=source_quote_coverage_pct,
            source_quote_usable=source_quote_usable,
            fallback_level=fallback_level,
            source_quality=source_quality,
            source_trust_label=source_trust_label,
            source_quality_note=source_quality_note,
        )
        rows.append(row)
        append_spec(
            candidate_slug=row["candidate_slug"],
            candidate_label=row["candidate_label"],
            family="long_stock",
            position=stock_baseline,
        )

    spot = float(chain.spot_price or chain.metadata.spot_price or 0.0)
    if spot <= 0:
        return rows, specs, list(chain.warnings)

    for option_type, family_set in [
        ("call", {"long_call", "bull_call_spread", "covered_call"}),
        ("put", {"long_put", "bear_put_spread", "cash_secured_put"}),
    ]:
        requested = set(strategy_families).intersection(family_set)
        if not requested:
            continue
        frame = chain.filter_expiry(chain.metadata.expiry_date)
        frame = frame.loc[frame["option_type"] == option_type].copy()
        ranked_rows = _rank_contract_rows(frame, spot_price=spot, option_type=option_type, top_n=top_n_strikes)
        for raw_row in ranked_rows:
            contract = chain.to_contract(raw_row)
            strike_band = (
                "atm"
                if abs(float(contract.strike) - spot) < max(0.5, spot * 0.02)
                else (
                    "otm"
                    if (option_type == "call" and float(contract.strike) > spot)
                    or (option_type == "put" and float(contract.strike) < spot)
                    else "itm"
                )
            )
            if "long_call" in requested or "long_put" in requested:
                family = "long_call" if option_type == "call" else "long_put"
                if family in requested:
                    position = build_strategy(
                        family,
                        chain,
                        premium_mode="mid",
                        contract=contract,
                    )
                    candidate_slug = slugify(f"{family}-{contract.expiry_date.isoformat()}-{contract.strike:.2f}-{scope}")
                    row = _build_candidate_row(
                        candidate_slug=candidate_slug,
                        strategy_family=family,
                        candidate_label=f"{family.replace('_', ' ').title()} {contract.expiry_date.isoformat()} {contract.strike:.2f}",
                        selection_scope=scope,
                        chain=chain,
                        position=position,
                        target_price=target_price,
                        target_date=target_date,
                        target_horizon_label=target_horizon_label,
                        requested_days=requested_days,
                        iv_shift_points=iv_shift_points,
                        comparison_capital=comparison_capital,
                        stock_baseline=stock_baseline,
                        strike_selection_mode=strike_band,
                        source_snapshot_date=source_snapshot_date,
                        source_expiry_date=source_expiry_date,
                        source_storage_location=source_location,
                        source_snapshot_file=source_snapshot_file,
                        source_quote_coverage_pct=source_quote_coverage_pct,
                        source_quote_usable=source_quote_usable,
                        fallback_level=fallback_level,
                        source_quality=source_quality,
                        source_trust_label=source_trust_label,
                        source_quality_note=source_quality_note,
                    )
                    rows.append(row)
                    append_spec(candidate_slug=candidate_slug, candidate_label=row["candidate_label"], family=family, position=position)
            if "bull_call_spread" in requested and option_type == "call":
                short_leg = _adjacent_spread_contract(chain, contract, direction="higher")
                if short_leg is not None:
                    position = build_strategy(
                        "bull_call_spread",
                        chain,
                        premium_mode="mid",
                        long_contract=contract,
                        short_contract=short_leg,
                    )
                    candidate_slug = slugify(
                        f"bull-call-spread-{contract.expiry_date.isoformat()}-{contract.strike:.2f}-{short_leg.strike:.2f}-{scope}"
                    )
                    row = _build_candidate_row(
                        candidate_slug=candidate_slug,
                        strategy_family="bull_call_spread",
                        candidate_label=f"Bull Call Spread {contract.expiry_date.isoformat()} {contract.strike:.2f}/{short_leg.strike:.2f}",
                        selection_scope=scope,
                        chain=chain,
                        position=position,
                        target_price=target_price,
                        target_date=target_date,
                        target_horizon_label=target_horizon_label,
                        requested_days=requested_days,
                        iv_shift_points=iv_shift_points,
                        comparison_capital=comparison_capital,
                        stock_baseline=stock_baseline,
                        strike_selection_mode=strike_band,
                        source_snapshot_date=source_snapshot_date,
                        source_expiry_date=source_expiry_date,
                        source_storage_location=source_location,
                        source_snapshot_file=source_snapshot_file,
                        source_quote_coverage_pct=source_quote_coverage_pct,
                        source_quote_usable=source_quote_usable,
                        fallback_level=fallback_level,
                        source_quality=source_quality,
                        source_trust_label=source_trust_label,
                        source_quality_note=source_quality_note,
                    )
                    rows.append(row)
                    append_spec(candidate_slug=candidate_slug, candidate_label=row["candidate_label"], family="bull_call_spread", position=position)
            if "bear_put_spread" in requested and option_type == "put":
                short_leg = _adjacent_spread_contract(chain, contract, direction="lower")
                if short_leg is not None:
                    position = build_strategy(
                        "bear_put_spread",
                        chain,
                        premium_mode="mid",
                        long_contract=contract,
                        short_contract=short_leg,
                    )
                    candidate_slug = slugify(
                        f"bear-put-spread-{contract.expiry_date.isoformat()}-{contract.strike:.2f}-{short_leg.strike:.2f}-{scope}"
                    )
                    row = _build_candidate_row(
                        candidate_slug=candidate_slug,
                        strategy_family="bear_put_spread",
                        candidate_label=f"Bear Put Spread {contract.expiry_date.isoformat()} {contract.strike:.2f}/{short_leg.strike:.2f}",
                        selection_scope=scope,
                        chain=chain,
                        position=position,
                        target_price=target_price,
                        target_date=target_date,
                        target_horizon_label=target_horizon_label,
                        requested_days=requested_days,
                        iv_shift_points=iv_shift_points,
                        comparison_capital=comparison_capital,
                        stock_baseline=stock_baseline,
                        strike_selection_mode=strike_band,
                        source_snapshot_date=source_snapshot_date,
                        source_expiry_date=source_expiry_date,
                        source_storage_location=source_location,
                        source_snapshot_file=source_snapshot_file,
                        source_quote_coverage_pct=source_quote_coverage_pct,
                        source_quote_usable=source_quote_usable,
                        fallback_level=fallback_level,
                        source_quality=source_quality,
                        source_trust_label=source_trust_label,
                        source_quality_note=source_quality_note,
                    )
                    rows.append(row)
                    append_spec(candidate_slug=candidate_slug, candidate_label=row["candidate_label"], family="bear_put_spread", position=position)
            if "covered_call" in requested and option_type == "call":
                position = build_strategy(
                    "covered_call",
                    chain,
                    premium_mode="mid",
                    contract=contract,
                )
                candidate_slug = slugify(f"covered-call-{contract.expiry_date.isoformat()}-{contract.strike:.2f}-{scope}")
                row = _build_candidate_row(
                    candidate_slug=candidate_slug,
                    strategy_family="covered_call",
                    candidate_label=f"Covered Call {contract.expiry_date.isoformat()} {contract.strike:.2f}",
                    selection_scope=scope,
                    chain=chain,
                    position=position,
                    target_price=target_price,
                    target_date=target_date,
                    target_horizon_label=target_horizon_label,
                    requested_days=requested_days,
                    iv_shift_points=iv_shift_points,
                    comparison_capital=comparison_capital,
                    stock_baseline=stock_baseline,
                    strike_selection_mode=strike_band,
                    source_snapshot_date=source_snapshot_date,
                    source_expiry_date=source_expiry_date,
                    source_storage_location=source_location,
                    source_snapshot_file=source_snapshot_file,
                    source_quote_coverage_pct=source_quote_coverage_pct,
                    source_quote_usable=source_quote_usable,
                    fallback_level=fallback_level,
                    source_quality=source_quality,
                    source_trust_label=source_trust_label,
                    source_quality_note=source_quality_note,
                )
                rows.append(row)
                append_spec(candidate_slug=candidate_slug, candidate_label=row["candidate_label"], family="covered_call", position=position)
            if "cash_secured_put" in requested and option_type == "put":
                position = build_strategy(
                    "cash_secured_put",
                    chain,
                    premium_mode="mid",
                    contract=contract,
                )
                candidate_slug = slugify(f"cash-secured-put-{contract.expiry_date.isoformat()}-{contract.strike:.2f}-{scope}")
                row = _build_candidate_row(
                    candidate_slug=candidate_slug,
                    strategy_family="cash_secured_put",
                    candidate_label=f"Cash Secured Put {contract.expiry_date.isoformat()} {contract.strike:.2f}",
                    selection_scope=scope,
                    chain=chain,
                    position=position,
                    target_price=target_price,
                    target_date=target_date,
                    target_horizon_label=target_horizon_label,
                    requested_days=requested_days,
                    iv_shift_points=iv_shift_points,
                    comparison_capital=comparison_capital,
                    stock_baseline=stock_baseline,
                    strike_selection_mode=strike_band,
                    source_snapshot_date=source_snapshot_date,
                    source_expiry_date=source_expiry_date,
                    source_storage_location=source_location,
                    source_snapshot_file=source_snapshot_file,
                    source_quote_coverage_pct=source_quote_coverage_pct,
                    source_quote_usable=source_quote_usable,
                    fallback_level=fallback_level,
                    source_quality=source_quality,
                    source_trust_label=source_trust_label,
                    source_quality_note=source_quality_note,
                )
                rows.append(row)
                append_spec(candidate_slug=candidate_slug, candidate_label=row["candidate_label"], family="cash_secured_put", position=position)
    warnings.extend(chain.warnings)
    return rows, specs, warnings


def _goal_reached(
    row: dict[str, Any],
    *,
    goal: str,
    target_option_value: float | None,
) -> bool:
    if goal == "itm_1c":
        strategy_family = clean_string(row.get("strategy_family")).lower()
        estimated_value = finite_or_none(row.get("estimated_value"))
        capital_required = finite_or_none(row.get("capital_required"))
        if strategy_family == "long_stock":
            profit_loss = finite_or_none(row.get("profit_loss"))
            return profit_loss is not None and profit_loss >= 0.01
        if estimated_value is not None and capital_required is not None:
            return estimated_value >= capital_required + 0.01
        profit_loss = finite_or_none(row.get("profit_loss"))
        return profit_loss is not None and profit_loss >= 0.01
    if goal == "break_even":
        value = finite_or_none(row.get("profit_loss"))
        return value is not None and value >= 0
    if goal == "return_25":
        value = finite_or_none(row.get("return_on_comparison_capital"))
        return value is not None and value >= 0.25
    if goal == "return_50":
        value = finite_or_none(row.get("return_on_comparison_capital"))
        return value is not None and value >= 0.50
    if goal == "outperform_stock":
        value = finite_or_none(row.get("difference_vs_stock"))
        return value is not None and value >= 0
    if goal == "target_option_value" and target_option_value is not None:
        value = finite_or_none(row.get("estimated_value"))
        return value is not None and value >= float(target_option_value)
    return False


def _evaluate_at_point(
    spec: dict[str, Any],
    *,
    spot_price: float,
    horizon_days: int,
    iv_shift_points: float,
    comparison_capital: float,
) -> dict[str, Any]:
    position: StrategyPosition = spec["position"]
    stock_baseline: StrategyPosition = spec["stock_baseline"]
    target_date = position.snapshot_date + timedelta(days=max(int(horizon_days), 0))
    return _evaluate_position(
        position,
        target_price=float(spot_price),
        target_date=target_date,
        requested_days=int(horizon_days),
        iv_shift_points=float(iv_shift_points),
        comparison_capital=float(comparison_capital),
        stock_baseline=stock_baseline,
    )


def _ordered_horizon_specs(target_horizon_label: str, target_horizon_days: int) -> list[dict[str, Any]]:
    return [
        {"label": spec.label, "requested_days": spec.requested_days}
        for spec in _canonical_horizon_specs(target_horizon_label, target_horizon_days)
    ]


def _required_path_rows(
    specs: list[dict[str, Any]],
    *,
    horizon_specs: list[dict[str, Any]],
    comparison_capital: float,
    target_price: float,
    target_option_value: float | None,
    active_iv_path_name: str | None = None,
    active_iv_path_points: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    goals = _required_goals(DEFAULT_GOAL, target_option_value)
    dense_grid = build_stock_grid(max(target_price, 1.0), down_move=0.55, up_move=1.15, points=181)
    iv_variants: list[tuple[str, str, dict[str, float]]] = [
        ("point", f"{shift:+.2f}", {clean_string(spec["label"]).lower(): float(shift) for spec in horizon_specs})
        for shift in POINT_IV_SCENARIOS
    ]
    iv_variants.extend(
        [
            ("path", "flat", _interpolated_path(_default_iv_path_points(preset="flat", base_shift=0.0, target_horizon_label=horizon_specs[-1]["label"]), horizon_specs, default_value=0.0)),
            ("path", "iv_up_then_down", _interpolated_path(_default_iv_path_points(preset="iv_up_then_down", base_shift=0.0, target_horizon_label=horizon_specs[-1]["label"]), horizon_specs, default_value=0.0)),
            ("path", "iv_down_then_stays_low", _interpolated_path(_default_iv_path_points(preset="iv_down_then_stays_low", base_shift=0.0, target_horizon_label=horizon_specs[-1]["label"]), horizon_specs, default_value=0.0)),
            ("path", "earnings_build_then_crush", _interpolated_path(_default_iv_path_points(preset="earnings_build_then_crush", base_shift=0.0, target_horizon_label=horizon_specs[-1]["label"]), horizon_specs, default_value=0.0)),
            ("path", "mean_reversion_lower", _interpolated_path(_default_iv_path_points(preset="mean_reversion_lower", base_shift=0.0, target_horizon_label=horizon_specs[-1]["label"]), horizon_specs, default_value=0.0)),
            ("path", "mean_reversion_higher", _interpolated_path(_default_iv_path_points(preset="mean_reversion_higher", base_shift=0.0, target_horizon_label=horizon_specs[-1]["label"]), horizon_specs, default_value=0.0)),
        ]
    )
    active_name = clean_string(active_iv_path_name).lower()
    if active_name and active_name not in {clean_string(name).lower() for _, name, _ in iv_variants}:
        active_points = dict(active_iv_path_points or {})
        iv_variants.append(
            (
                "path",
                active_name,
                _interpolated_path(
                    active_points,
                    horizon_specs,
                    default_value=float(next(iter(active_points.values()), 0.0)),
                ),
            )
        )
    metric_name_map = {
        "itm_1c": "profit_loss",
        "break_even": "profit_loss",
        "return_25": "return_on_comparison_capital",
        "return_50": "return_on_comparison_capital",
        "outperform_stock": "difference_vs_stock",
        "target_option_value": "estimated_value",
    }
    threshold_map = {
        "itm_1c": 0.01,
        "break_even": 0.0,
        "return_25": 0.25,
        "return_50": 0.50,
        "outperform_stock": 0.0,
        "target_option_value": float(target_option_value or 0.0),
    }
    for spec in specs:
        for goal in goals:
            metric_name = metric_name_map[goal]
            threshold = threshold_map[goal]
            for variant_kind, variant_name, path_map in iv_variants:
                for horizon in horizon_specs:
                    horizon_label = clean_string(horizon["label"]).lower()
                    requested_days = int(horizon["requested_days"])
                    iv_shift = float(path_map.get(horizon_label, 0.0))
                    values: list[tuple[float, dict[str, Any]]] = []
                    for spot in dense_grid:
                        evaluation = _evaluate_at_point(
                            spec,
                            spot_price=float(spot),
                            horizon_days=requested_days,
                            iv_shift_points=iv_shift,
                            comparison_capital=comparison_capital,
                        )
                        values.append((float(spot), evaluation))
                    winning_index = None
                    for index, (_, evaluation) in enumerate(values):
                        if _goal_reached(evaluation, goal=goal, target_option_value=target_option_value):
                            winning_index = index
                            break
                    if winning_index is None:
                        base_eval = values[-1][1]
                        rows.append(
                            {
                                "candidate_slug": spec["candidate_slug"],
                                "candidate_label": spec["candidate_label"],
                                "strategy_family": spec["strategy_family"],
                                "goal": goal,
                                "iv_variant_kind": variant_kind,
                                "iv_variant": variant_name,
                                "horizon": horizon_label,
                                "requested_days": requested_days,
                                "required_stock_price": None,
                                "required_stock_price_label": "unreached",
                                "unreached": True,
                                "clamped_to_expiry": bool(base_eval.get("clamped_to_expiry")),
                                "target_beyond_expiry": bool(base_eval.get("target_beyond_expiry")),
                                "valuation_date": base_eval.get("valuation_date"),
                                "iv_shift_points": iv_shift,
                            }
                        )
                        continue
                    winning_spot, winning_eval = values[winning_index]
                    if winning_index > 0:
                        prev_spot, prev_eval = values[winning_index - 1]
                        prev_value = finite_or_none(prev_eval.get(metric_name))
                        current_value = finite_or_none(winning_eval.get(metric_name))
                        if prev_value is not None and current_value is not None and current_value != prev_value:
                            ratio = (threshold - prev_value) / (current_value - prev_value)
                            ratio = max(0.0, min(1.0, ratio))
                            winning_spot = float(prev_spot + ratio * (winning_spot - prev_spot))
                    rows.append(
                        {
                            "candidate_slug": spec["candidate_slug"],
                            "candidate_label": spec["candidate_label"],
                            "strategy_family": spec["strategy_family"],
                            "goal": goal,
                            "iv_variant_kind": variant_kind,
                            "iv_variant": variant_name,
                            "horizon": horizon_label,
                            "requested_days": requested_days,
                            "required_stock_price": round(float(winning_spot), 4),
                            "required_stock_price_label": f"{float(winning_spot):.2f}",
                            "unreached": False,
                            "clamped_to_expiry": bool(winning_eval.get("clamped_to_expiry")),
                            "target_beyond_expiry": bool(winning_eval.get("target_beyond_expiry")),
                            "valuation_date": winning_eval.get("valuation_date"),
                            "iv_shift_points": iv_shift,
                        }
                    )
    return pd.DataFrame(rows)


def _annotate_required_path_rows(
    required_path_rows: pd.DataFrame,
    *,
    assumed_stock_path: dict[str, float],
) -> pd.DataFrame:
    """Add direct assumed-path comparison fields to required-path rows."""

    if required_path_rows.empty:
        return required_path_rows
    frame = required_path_rows.copy()
    entry_spot = finite_or_none(assumed_stock_path.get("entry"))
    goal_metric_map = {
        "itm_1c": "profit_loss",
        "break_even": "profit_loss",
        "return_25": "return_on_comparison_capital",
        "return_50": "return_on_comparison_capital",
        "outperform_stock": "difference_vs_stock",
        "target_option_value": "estimated_value",
    }
    goal_threshold_map = {
        "itm_1c": 0.01,
        "break_even": 0.0,
        "return_25": 0.25,
        "return_50": 0.50,
        "outperform_stock": 0.0,
    }
    frame["assumed_stock_price"] = frame["horizon"].map(
        {clean_string(label).lower(): float(value) for label, value in assumed_stock_path.items()}
    )
    frame["required_move_points_from_entry"] = frame["required_stock_price"].apply(
        lambda value: round(float(value) - float(entry_spot), 4)
        if finite_or_none(value) is not None and entry_spot is not None
        else None
    )
    frame["required_move_pct_from_entry"] = frame["required_stock_price"].apply(
        lambda value: round(float(value) / float(entry_spot) - 1.0, 6)
        if finite_or_none(value) is not None and entry_spot not in {None, 0.0}
        else None
    )
    frame["assumed_minus_required_price"] = frame.apply(
        lambda row: round(float(row["assumed_stock_price"]) - float(row["required_stock_price"]), 4)
        if finite_or_none(row.get("assumed_stock_price")) is not None and finite_or_none(row.get("required_stock_price")) is not None
        else None,
        axis=1,
    )
    frame["assumed_clears_required_path"] = frame.apply(
        lambda row: bool(
            finite_or_none(row.get("assumed_stock_price")) is not None
            and finite_or_none(row.get("required_stock_price")) is not None
            and float(row["assumed_stock_price"]) >= float(row["required_stock_price"])
        ),
        axis=1,
    )
    frame["goal_metric"] = frame["goal"].map(goal_metric_map).fillna("profit_loss")
    frame["goal_threshold"] = frame["goal"].apply(
        lambda goal: float(goal_threshold_map.get(clean_string(goal), 0.0))
        if clean_string(goal) != "target_option_value"
        else None
    )
    frame["required_path_note"] = frame.apply(
        lambda row: (
            "Required path is still above the active assumed stock path at this horizon."
            if not bool(row.get("assumed_clears_required_path")) and not bool(row.get("unreached"))
            else (
                "Active assumed stock path clears this required threshold at this horizon."
                if bool(row.get("assumed_clears_required_path"))
                else "Threshold was unreached in the sampled stock grid."
            )
        ),
        axis=1,
    )
    return frame


def _stable_simulation_seed(*parts: Any) -> int:
    payload = "|".join(clean_string(part) for part in parts if clean_string(part))
    return int(zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF)


def _terminal_simulation_date(specs: list[dict[str, Any]], *, target_date: date) -> date:
    latest = target_date
    for spec in specs:
        position = spec.get("position")
        expiry_date = getattr(position, "expiry_date", None)
        if expiry_date is not None and expiry_date > latest:
            latest = expiry_date
    return latest


def _terminal_row_for_candidate(frame: pd.DataFrame, *, candidate_slug: str, target_horizon_days: int) -> dict[str, Any]:
    if frame.empty:
        return {}
    working = frame.loc[frame.get("candidate_slug").astype(str) == clean_string(candidate_slug)].copy()
    if working.empty:
        return {}
    within_target = working.loc[pd.to_numeric(working.get("requested_days"), errors="coerce") <= int(target_horizon_days)].copy()
    if within_target.empty:
        within_target = working.copy()
    ordered = within_target.sort_values(["requested_days", "step_index"])
    return ordered.iloc[-1].to_dict() if not ordered.empty else {}


def _preferred_quote_context(row: pd.Series | dict[str, Any]) -> bool:
    if isinstance(row, pd.Series):
        row_data = row.to_dict()
    elif row is None:
        row_data = {}
    else:
        row_data = dict(row)
    trust_label = clean_string(row_data.get("source_trust_label")).lower()
    if trust_label in PREFERRED_SOURCE_TRUST_LABELS:
        return True
    quote_usable = row_data.get("source_quote_usable")
    if isinstance(quote_usable, str):
        quote_usable = clean_string(quote_usable).lower() in {"1", "true", "yes", "y"}
    else:
        quote_usable = bool(quote_usable)
    return quote_usable and trust_label != "fallback_only"


def _sort_candidate_priority(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    working["_preferred_quote_context"] = working.apply(_preferred_quote_context, axis=1)
    working["_rank"] = pd.to_numeric(working.get("active_candidate_rank"), errors="coerce").fillna(999999)
    working["_objective"] = pd.to_numeric(working.get("objective_score"), errors="coerce").fillna(-999999.0)
    working["_coverage"] = pd.to_numeric(working.get("source_quote_coverage_pct"), errors="coerce").fillna(-1.0)
    working["_primary_strike"] = pd.to_numeric(working.get("primary_strike"), errors="coerce")
    return working.sort_values(
        ["_preferred_quote_context", "_rank", "_objective", "_coverage", "expiry_date", "_primary_strike"],
        ascending=[False, True, False, False, True, True],
    ).reset_index(drop=True)


def _select_long_call_strike_view_rows(long_calls: pd.DataFrame, *, anchor_row: pd.Series) -> pd.DataFrame:
    same_expiry = _sort_candidate_priority(
        long_calls.loc[long_calls.get("expiry_date").astype(str) == clean_string(anchor_row.get("expiry_date"))].copy()
    )
    if same_expiry.empty:
        return pd.DataFrame()
    seen: set[str] = set()
    selected: list[dict[str, Any]] = []
    anchor_strike = finite_or_none(anchor_row.get("primary_strike"))

    def add_row(row: pd.Series | None, reason: str) -> None:
        if row is None:
            return
        slug = clean_string(row.get("candidate_slug"))
        if not slug or slug in seen:
            return
        record = row.to_dict()
        record["selection_reason"] = reason
        record["used_strike_fallback"] = False
        record["used_trust_fallback"] = not bool(record.get("_preferred_quote_context"))
        selected.append(record)
        seen.add(slug)

    add_row(anchor_row, "Anchor long call for the strike view under the active assumed path.")
    trusted = same_expiry.loc[same_expiry["_preferred_quote_context"] == True].copy()  # noqa: E712
    lower = trusted.loc[pd.to_numeric(trusted.get("primary_strike"), errors="coerce") < float(anchor_strike or 0.0)].sort_values("primary_strike")
    higher = trusted.loc[pd.to_numeric(trusted.get("primary_strike"), errors="coerce") > float(anchor_strike or 0.0)].sort_values("primary_strike")
    if lower.empty:
        lower = same_expiry.loc[pd.to_numeric(same_expiry.get("primary_strike"), errors="coerce") < float(anchor_strike or 0.0)].sort_values("primary_strike")
    if higher.empty:
        higher = same_expiry.loc[pd.to_numeric(same_expiry.get("primary_strike"), errors="coerce") > float(anchor_strike or 0.0)].sort_values("primary_strike")
    add_row(lower.iloc[-1] if not lower.empty else None, "Nearest lower-strike long call at the same expiry under the same assumed path.")
    add_row(higher.iloc[0] if not higher.empty else None, "Nearest higher-strike long call at the same expiry under the same assumed path.")
    for _, row in same_expiry.iterrows():
        if len(selected) >= MAX_LONG_CALL_VIEW_LINES:
            break
        add_row(row, "Additional same-expiry long call kept because it remains one of the strongest reads under the active objective.")
    selected_frame = pd.DataFrame(selected)
    if selected_frame.empty:
        return selected_frame
    selected_frame = selected_frame.sort_values(["primary_strike", "_rank"]).reset_index(drop=True)
    selected_frame["selection_rank"] = range(1, len(selected_frame.index) + 1)
    selected_frame["view_name"] = "long_call_strike_view"
    selected_frame["anchor_expiry_date"] = clean_string(anchor_row.get("expiry_date"))
    selected_frame["anchor_strike_label"] = clean_string(anchor_row.get("strike_label"))
    selected_frame["series_label"] = selected_frame.apply(
        lambda row: _format_long_call_series_label(
            strike_label=row.get("strike_label"),
            expiry_date=row.get("expiry_date"),
        ),
        axis=1,
    )
    return selected_frame


def _select_long_call_expiry_view_rows(long_calls: pd.DataFrame, *, anchor_row: pd.Series) -> pd.DataFrame:
    anchor_strike = finite_or_none(anchor_row.get("primary_strike"))
    anchor_bucket = clean_string(anchor_row.get("moneyness_bucket"))
    chosen_rows: list[dict[str, Any]] = []
    for expiry_date, group in long_calls.groupby("expiry_date", dropna=False):
        ordered = _sort_candidate_priority(group)
        if ordered.empty:
            continue
        exact = ordered.loc[np.isclose(pd.to_numeric(ordered.get("primary_strike"), errors="coerce"), float(anchor_strike or 0.0), atol=1e-6)].copy()
        used_fallback = False
        strike_match_mode = "exact_strike"
        fallback_strike_distance = 0.0
        selection_reason = "Exact same strike concept under the active assumed path."
        if exact.empty:
            same_bucket = ordered.loc[ordered.get("moneyness_bucket").astype(str) == anchor_bucket].copy()
            if not same_bucket.empty:
                same_bucket["_strike_distance"] = (
                    pd.to_numeric(same_bucket.get("primary_strike"), errors="coerce") - float(anchor_strike or 0.0)
                ).abs()
                exact = same_bucket.sort_values(["_preferred_quote_context", "_strike_distance", "_rank"], ascending=[False, True, True]).head(1)
                used_fallback = True
                strike_match_mode = "same_moneyness"
                selection_reason = "Exact strike unavailable; used the nearest same-moneyness long call for this expiry."
            else:
                numeric_fallback = ordered.copy()
                numeric_fallback["_strike_distance"] = (
                    pd.to_numeric(numeric_fallback.get("primary_strike"), errors="coerce") - float(anchor_strike or 0.0)
                ).abs()
                numeric_fallback = numeric_fallback.sort_values(
                    ["_preferred_quote_context", "_strike_distance", "_rank"],
                    ascending=[False, True, True],
                ).head(1)
                if numeric_fallback.empty:
                    continue
                exact = numeric_fallback
                used_fallback = True
                strike_match_mode = "nearest_numeric"
                selection_reason = "Exact strike and same-moneyness match unavailable; used the nearest numeric strike for this expiry."
        chosen = exact.iloc[0].to_dict()
        chosen["expiry_date"] = clean_string(expiry_date)
        chosen["used_strike_fallback"] = used_fallback
        chosen["strike_match_mode"] = strike_match_mode
        fallback_strike_distance = abs(float(finite_or_none(chosen.get("primary_strike")) or 0.0) - float(anchor_strike or 0.0))
        chosen["fallback_strike_distance"] = fallback_strike_distance if used_fallback else 0.0
        chosen["used_trust_fallback"] = not bool(chosen.get("_preferred_quote_context"))
        chosen["selection_reason"] = selection_reason
        chosen_rows.append(chosen)
    selected_frame = pd.DataFrame(chosen_rows)
    if selected_frame.empty:
        return selected_frame
    selected_frame = _sort_candidate_priority(selected_frame)
    anchor_expiry = clean_string(anchor_row.get("expiry_date"))
    anchor_match = selected_frame.loc[selected_frame.get("expiry_date").astype(str) == anchor_expiry].head(1)
    selected_frame = selected_frame.head(MAX_LONG_CALL_VIEW_LINES)
    if not anchor_match.empty and anchor_expiry not in selected_frame.get("expiry_date").astype(str).tolist():
        selected_frame = pd.concat([anchor_match, selected_frame.iloc[:-1]], ignore_index=True)
    selected_frame = selected_frame.sort_values("expiry_date").reset_index(drop=True)
    selected_frame["selection_rank"] = range(1, len(selected_frame.index) + 1)
    selected_frame["view_name"] = "long_call_expiry_view"
    selected_frame["anchor_expiry_date"] = anchor_expiry
    selected_frame["anchor_strike_label"] = clean_string(anchor_row.get("strike_label"))
    selected_frame["series_label"] = selected_frame.apply(
        lambda row: _format_long_call_series_label(
            strike_label=row.get("strike_label"),
            expiry_date=row.get("expiry_date"),
        ),
        axis=1,
    )
    return selected_frame


def _select_long_call_best_of_rows(long_calls: pd.DataFrame, *, anchor_row: pd.Series) -> pd.DataFrame:
    ordered = _sort_candidate_priority(long_calls)
    if ordered.empty:
        return ordered
    selected: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    seen_expiries: set[str] = set()
    seen_buckets: set[str] = set()

    def add_first(predicate, reason: str) -> bool:
        for _, row in ordered.iterrows():
            slug = clean_string(row.get("candidate_slug"))
            if not slug or slug in seen_slugs:
                continue
            if not predicate(row):
                continue
            record = row.to_dict()
            record["selection_reason"] = reason
            record["used_strike_fallback"] = False
            record["used_trust_fallback"] = not bool(record.get("_preferred_quote_context"))
            selected.append(record)
            seen_slugs.add(slug)
            seen_expiries.add(clean_string(record.get("expiry_date")))
            seen_buckets.add(clean_string(record.get("moneyness_bucket")))
            return True
        return False

    anchor_slug = clean_string(anchor_row.get("candidate_slug"))
    add_first(
        lambda row: clean_string(row.get("candidate_slug")) == anchor_slug,
        "Best overall long call under the active assumed path and objective.",
    )
    add_first(
        lambda row: clean_string(row.get("expiry_date")) not in seen_expiries,
        "Added because it is the strongest long call from a different expiry.",
    )
    add_first(
        lambda row: clean_string(row.get("moneyness_bucket")) not in seen_buckets,
        "Added because it is the strongest long call from a different moneyness bucket.",
    )
    minimum_count = min(4, min(MAX_LONG_CALL_VIEW_LINES, len(ordered.index)))
    while len(selected) < minimum_count:
        if add_first(
            lambda row: bool(row.get("_preferred_quote_context"))
            and (
                clean_string(row.get("expiry_date")) not in seen_expiries
                or clean_string(row.get("moneyness_bucket")) not in seen_buckets
            ),
            "Added to complete the minimum best-of set while preserving trust and diversity.",
        ):
            continue
        if add_first(
            lambda row: bool(row.get("_preferred_quote_context")),
            "Added as the next-strongest trusted long call after the diversity slots were filled.",
        ):
            continue
        add_first(
            lambda row: True,
            "Added as a fallback because too few trusted long calls satisfied the diversity constraints.",
        )
    while len(selected) < min(MAX_LONG_CALL_VIEW_LINES, len(ordered.index)):
        if not add_first(
            lambda row: clean_string(row.get("expiry_date")) not in seen_expiries
            or clean_string(row.get("moneyness_bucket")) not in seen_buckets,
            "Added because it still contributes new expiry or moneyness information.",
        ):
            break
    selected_frame = pd.DataFrame(selected)
    if selected_frame.empty:
        return selected_frame
    selected_frame["selection_rank"] = range(1, len(selected_frame.index) + 1)
    selected_frame["view_name"] = "long_call_best_of_view"
    selected_frame["anchor_expiry_date"] = clean_string(anchor_row.get("expiry_date"))
    selected_frame["anchor_strike_label"] = clean_string(anchor_row.get("strike_label"))
    selected_frame["series_label"] = selected_frame.apply(
        lambda row: _format_long_call_series_label(
            strike_label=row.get("strike_label"),
            expiry_date=row.get("expiry_date"),
            moneyness_bucket=row.get("moneyness_bucket"),
            include_bucket=True,
        ),
        axis=1,
    )
    return selected_frame


def _humanize_moneyness_bucket(value: str) -> str:
    text = clean_string(value)
    return text.replace("_", "-") if text else "unknown"


def _short_moneyness_label(value: str) -> str:
    labels = {
        "itm": "ITM",
        "deep_itm": "Deep ITM",
        "near_atm": "ATM",
        "atm": "ATM",
        "otm": "OTM",
        "far_otm": "Far OTM",
    }
    text = clean_string(value).lower()
    return labels.get(text, _humanize_moneyness_bucket(text).upper())


def _compact_strike_call_label(value: Any) -> str:
    numeric = finite_or_none(value)
    if numeric is None:
        raw = clean_string(value).upper().replace("CALL", "").replace("C", "").strip()
        try:
            numeric = float(raw) if raw else None
        except ValueError:
            numeric = None
    if numeric is None:
        text = clean_string(value)
        return f"{text}C" if text and not text.upper().endswith("C") else text
    rounded = round(float(numeric), 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return f"{int(round(rounded))}C"
    return f"{rounded:.2f}".rstrip("0").rstrip(".") + "C"


def _expiry_display_label(value: Any) -> str:
    text = clean_string(value)
    parsed = parse_date(text)
    if parsed is None:
        return text
    return parsed.strftime("%b-%y")


def _format_long_call_series_label(
    *,
    strike_label: Any,
    expiry_date: Any,
    moneyness_bucket: Any | None = None,
    include_bucket: bool = False,
) -> str:
    label = f"{_compact_strike_call_label(strike_label)} {_expiry_display_label(expiry_date)}".strip()
    if include_bucket:
        label = f"{label} ({_short_moneyness_label(clean_string(moneyness_bucket))})"
    return label


def _short_iv_path_label(value: Any) -> str:
    labels = {
        "flat": "Flat",
        "mean_reversion_lower": "Mean Rev Lower",
        "mean_reversion_higher": "Mean Rev Higher",
        "iv_up_then_down": "Up Then Down",
        "iv_down_then_stays_low": "Down Then Low",
        "earnings_build_then_crush": "Build Then Crush",
    }
    text = clean_string(value).lower()
    return labels.get(text, _humanize_named_path(text, kind="iv"))


def _build_long_call_view_frame(
    selected_rows: pd.DataFrame,
    *,
    view_name: str,
    spec_lookup: dict[str, dict[str, Any]],
    stock_points: list[dict[str, Any]],
    iv_points: list[dict[str, Any]],
    stock_path_name: str,
    iv_path_name: str,
    comparison_capital: float,
    goal: str,
    target_option_value: float | None,
) -> pd.DataFrame:
    if selected_rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    point_count = min(len(stock_points), len(iv_points))
    for _, row in selected_rows.sort_values("selection_rank").iterrows():
        candidate_slug = clean_string(row.get("candidate_slug"))
        spec = spec_lookup.get(candidate_slug)
        if spec is None:
            continue
        for step_index in range(point_count):
            stock_point = stock_points[step_index]
            iv_point = iv_points[step_index]
            evaluation = _evaluate_at_point(
                spec,
                spot_price=float(stock_point.get("spot_price") or 0.0),
                horizon_days=int(stock_point.get("requested_days") or 0),
                iv_shift_points=float(iv_point.get("iv_shift_points") or 0.0),
                comparison_capital=float(comparison_capital),
            )
            return_on_capital = finite_or_none(evaluation.get("return_on_comparison_capital"))
            stock_return = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
            return_delta = (
                round(float(return_on_capital) - float(stock_return), 6)
                if return_on_capital is not None and stock_return is not None
                else None
            )
            goal_reached = _goal_reached(evaluation, goal=goal, target_option_value=target_option_value)
            records.append(
                {
                    "view_name": view_name,
                    "path_scope": "assumed_path",
                    "stock_path_name": clean_string(stock_path_name),
                    "iv_path_name": clean_string(iv_path_name),
                    "anchor_expiry_date": clean_string(row.get("anchor_expiry_date")),
                    "anchor_strike_label": clean_string(row.get("anchor_strike_label")),
                    "used_strike_fallback": bool(row.get("used_strike_fallback")),
                    "strike_match_mode": clean_string(row.get("strike_match_mode")) or "exact_strike",
                    "fallback_strike_distance": finite_or_none(row.get("fallback_strike_distance")),
                    "used_trust_fallback": bool(row.get("used_trust_fallback")),
                    "selection_rank": int(row.get("selection_rank") or 0),
                    "selection_reason": clean_string(row.get("selection_reason")),
                    "series_label": clean_string(row.get("series_label")) or clean_string(row.get("candidate_label")),
                    "candidate_slug": candidate_slug,
                    "candidate_label": clean_string(row.get("candidate_label")),
                    "expiry_date": clean_string(row.get("expiry_date")),
                    "strike_label": clean_string(row.get("strike_label")),
                    "moneyness_bucket": clean_string(row.get("moneyness_bucket")),
                    "source_trust_label": clean_string(row.get("source_trust_label")),
                    "source_quality_note": clean_string(row.get("source_quality_note")),
                    "objective_score": finite_or_none(row.get("objective_score")),
                    "date": clean_string(stock_point.get("date")),
                    "requested_days": int(stock_point.get("requested_days") or 0),
                    "step_index": int(stock_point.get("step_index") or 0),
                    "spot_price": float(stock_point.get("spot_price") or 0.0),
                    "iv_shift_points": float(iv_point.get("iv_shift_points") or 0.0),
                    "modeled_value": finite_or_none(evaluation.get("estimated_value")),
                    "profit_loss": finite_or_none(evaluation.get("profit_loss")),
                    "return_on_comparison_capital": return_on_capital,
                    "difference_vs_stock": finite_or_none(evaluation.get("difference_vs_stock")),
                    "difference_vs_stock_return_pct": return_delta,
                    "success_status": _success_status_label(
                        profit_loss=finite_or_none(evaluation.get("profit_loss")),
                        return_on_comparison_capital=return_on_capital,
                        difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                        goal_reached=goal_reached,
                    ),
                }
            )
    return pd.DataFrame(records)


def _build_assumed_path_long_call_views(
    *,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    snapshot_date: date,
    target_date: date,
    stock_path_name: str,
    stock_path_points: dict[str, float],
    iv_path_name: str,
    iv_path_points: dict[str, float],
    comparison_capital: float,
    goal: str,
    target_option_value: float | None,
) -> dict[str, pd.DataFrame]:
    empty = {
        "long_call_value_over_path_strike_view": pd.DataFrame(),
        "long_call_value_over_path_expiry_view": pd.DataFrame(),
        "long_call_value_over_path_best_of": pd.DataFrame(),
    }
    if candidate_rows.empty:
        return empty
    long_calls = candidate_rows.loc[candidate_rows.get("strategy_family").astype(str) == "long_call"].copy()
    if long_calls.empty:
        return empty
    long_calls = _sort_candidate_priority(long_calls)
    anchor = long_calls.loc[long_calls["_preferred_quote_context"] == True].head(1)  # noqa: E712
    if anchor.empty:
        anchor = long_calls.head(1)
    if anchor.empty:
        return empty
    anchor_row = anchor.iloc[0]
    spec_lookup = {
        clean_string(spec.get("candidate_slug")): spec
        for spec in specs
        if clean_string(spec.get("strategy_family")) == "long_call"
    }
    if not spec_lookup:
        return empty
    path_grid = _build_path_grid(snapshot_date, target_date)
    entry_spot = float(next(iter(stock_path_points.values()), anchor_row.get("spot_price") or 0.0))
    active_stock_path = _build_stock_path_from_named_points(
        path_grid,
        named_points=stock_path_points,
        path_id="assumed-stock-path",
        path_name=stock_path_name,
        entry_spot=entry_spot,
    )
    active_iv_path = _build_iv_path_from_named_points(
        path_grid,
        named_points=iv_path_points,
        iv_path_id="assumed-iv-path",
        iv_path_name=iv_path_name,
        base_iv_shift=float(next(iter(iv_path_points.values()), 0.0)),
    )
    strike_selection = _select_long_call_strike_view_rows(long_calls, anchor_row=anchor_row)
    expiry_selection = _select_long_call_expiry_view_rows(long_calls, anchor_row=anchor_row)
    best_of_selection = _select_long_call_best_of_rows(long_calls, anchor_row=anchor_row)
    return {
        "long_call_value_over_path_strike_view": _build_long_call_view_frame(
            strike_selection,
            view_name="long_call_strike_view",
            spec_lookup=spec_lookup,
            stock_points=active_stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=stock_path_name,
            iv_path_name=iv_path_name,
            comparison_capital=comparison_capital,
            goal=goal,
            target_option_value=target_option_value,
        ),
        "long_call_value_over_path_expiry_view": _build_long_call_view_frame(
            expiry_selection,
            view_name="long_call_expiry_view",
            spec_lookup=spec_lookup,
            stock_points=active_stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=stock_path_name,
            iv_path_name=iv_path_name,
            comparison_capital=comparison_capital,
            goal=goal,
            target_option_value=target_option_value,
        ),
        "long_call_value_over_path_best_of": _build_long_call_view_frame(
            best_of_selection,
            view_name="long_call_best_of_view",
            spec_lookup=spec_lookup,
            stock_points=active_stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=stock_path_name,
            iv_path_name=iv_path_name,
            comparison_capital=comparison_capital,
            goal=goal,
            target_option_value=target_option_value,
        ),
    }


def _path_view_filename(path_name: str, suffix: str) -> str:
    alias_map = {
        "rally_early_then_fade_then_rally_again": "rally_early_fade_rally",
        "range_bound_near_flat": "range_bound_flat",
        "down_first_then_recovery": "down_then_recovery",
        "late_breakout": "late_breakout",
        "early_move_above_strike_then_giveback": "early_above_strike_giveback",
        "reaches_target_late_near_expiry": "target_late_near_expiry",
        "quarter_up_then_hard_pullback": "qtr_up_hard_pullback",
        "high_vol_sideways_then_breakout": "hv_sideways_breakout",
        "earnings_gap_up_then_fade": "earnings_gap_up_fade",
        "earnings_gap_down_then_recovery": "earnings_gap_down_recovery",
        "false_breakout_then_recover": "false_breakout_recover",
        "rally_then_long_range_then_second_leg_up": "rally_range_second_leg",
        "violent_two_sided_quarter": "violent_two_sided_qtr",
        "slow_bleed_then_capitulation_then_bounce": "bleed_capitulation_bounce",
    }
    normalized = clean_string(path_name).lower()
    alias = alias_map.get(normalized)
    if not alias:
        alias = slugify(normalized.replace("_", "-"))
        if len(alias) > 36:
            digest = f"{zlib.crc32(alias.encode('utf-8')) & 0xFFFFFFFF:08x}"
            alias = f"{alias[:27].rstrip('-')}-{digest}"
    return f"{clean_string(alias).lower()}__{clean_string(suffix).lower()}"


def _path_specific_long_call_terminal_rows(
    long_calls: pd.DataFrame,
    *,
    spec_lookup: dict[str, dict[str, Any]],
    stock_points: list[dict[str, Any]],
    iv_points: list[dict[str, Any]],
    stock_path_name: str,
    iv_path_name: str,
    comparison_capital: float,
    objective_mode: str,
    downside_tolerance: str,
    simplicity_preference: str,
) -> pd.DataFrame:
    if long_calls.empty or not stock_points or not iv_points:
        return pd.DataFrame()

    terminal_stock_point = stock_points[min(len(stock_points), len(iv_points)) - 1]
    terminal_iv_point = iv_points[min(len(stock_points), len(iv_points)) - 1]
    terminal_records: list[dict[str, Any]] = []
    for _, row in long_calls.iterrows():
        candidate_slug = clean_string(row.get("candidate_slug"))
        spec = spec_lookup.get(candidate_slug)
        if spec is None:
            continue
        evaluation = _evaluate_at_point(
            spec,
            spot_price=float(terminal_stock_point.get("spot_price") or 0.0),
            horizon_days=int(terminal_stock_point.get("requested_days") or 0),
            iv_shift_points=float(terminal_iv_point.get("iv_shift_points") or 0.0),
            comparison_capital=float(comparison_capital),
        )
        terminal_row = row.to_dict()
        terminal_row["stock_path_name"] = clean_string(stock_path_name)
        terminal_row["iv_path_name"] = clean_string(iv_path_name)
        terminal_row["modeled_value"] = finite_or_none(evaluation.get("estimated_value"))
        terminal_row["profit_loss"] = finite_or_none(evaluation.get("profit_loss"))
        terminal_row["return_on_comparison_capital"] = finite_or_none(evaluation.get("return_on_comparison_capital"))
        terminal_row["difference_vs_stock"] = finite_or_none(evaluation.get("difference_vs_stock"))
        stock_return = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
        terminal_row["difference_vs_stock_return_pct"] = (
            round(float(terminal_row["return_on_comparison_capital"]) - float(stock_return), 6)
            if terminal_row["return_on_comparison_capital"] is not None and stock_return is not None
            else None
        )
        terminal_row["benchmark_note"] = clean_string(evaluation.get("benchmark_note")) or _compare_vs_stock_note(
            strategy_family=clean_string(terminal_row.get("strategy_family")),
            difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
            difference_vs_stock_return_pct=terminal_row["difference_vs_stock_return_pct"],
            clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
            target_beyond_expiry=bool(terminal_row.get("target_beyond_expiry")),
        )
        terminal_row["objective_score"] = _selector_score(
            pd.Series(terminal_row),
            objective_mode=objective_mode,
            downside_tolerance=downside_tolerance,
            simplicity_preference=simplicity_preference,
        )
        terminal_records.append(terminal_row)
    frame = pd.DataFrame(terminal_records)
    if frame.empty:
        return frame
    ranking = (
        frame.sort_values(
            ["objective_score", "difference_vs_stock", "profit_loss", "return_on_comparison_capital"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
        .copy()
    )
    ranking["active_candidate_rank"] = range(1, len(ranking.index) + 1)
    return ranking


def _build_long_call_compare_vs_stock_rows(view_frame: pd.DataFrame) -> pd.DataFrame:
    if view_frame.empty:
        return pd.DataFrame()
    columns = [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "iv_path_label",
        "iv_path_display_order",
        "anchor_candidate_slug",
        "anchor_contract_label",
        "iv_expanded_family",
        "contract_rank",
        "chart_include",
        "iv_chart_scope",
        "selection_rank",
        "selection_reason",
        "series_label",
        "candidate_slug",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "source_quality_note",
        "objective_score",
        "date",
        "requested_days",
        "step_index",
        "spot_price",
        "iv_shift_points",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "iv_effect_note",
        "iv_robustness_note",
        "success_status",
    ]
    compare = view_frame[[column for column in columns if column in view_frame.columns]].copy()
    compare["delta_profit_loss_vs_stock"] = compare["difference_vs_stock"]
    compare["delta_return_pct_vs_stock"] = compare["difference_vs_stock_return_pct"]
    return compare


def _nearest_checkpoint_day(days: list[int], target_day: float) -> int | None:
    if not days:
        return None
    return min(days, key=lambda value: abs(float(value) - float(target_day)))


def _build_path_checkpoint_rows(view_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a compact support table for one path pack without becoming another data dump."""

    if view_frame.empty:
        return pd.DataFrame()
    working = view_frame.copy()
    working["requested_days"] = pd.to_numeric(working.get("requested_days"), errors="coerce")
    working = working.dropna(subset=["requested_days", "candidate_slug"])
    if working.empty:
        return pd.DataFrame()
    working["requested_days"] = working["requested_days"].astype(int)
    max_day = int(working["requested_days"].max())
    checkpoint_specs = [
        ("entry", 0.0),
        ("quarter_check", max_day * 0.25),
        ("mid_check", max_day * 0.50),
        ("target", float(max_day)),
    ]
    days = sorted({int(value) for value in working["requested_days"].tolist()})
    checkpoint_days: list[tuple[str, int]] = []
    seen_days: set[int] = set()
    for label, requested in checkpoint_specs:
        day = _nearest_checkpoint_day(days, requested)
        if day is None or day in seen_days:
            continue
        checkpoint_days.append((label, day))
        seen_days.add(day)
    if not checkpoint_days:
        return pd.DataFrame()

    top_candidates = (
        working.sort_values(["selection_rank", "candidate_label", "requested_days"])
        .drop_duplicates("candidate_slug")
        .head(3)["candidate_slug"]
        .astype(str)
        .tolist()
    )
    rows: list[dict[str, Any]] = []
    for candidate_slug in top_candidates:
        candidate_frame = working.loc[working.get("candidate_slug").astype(str) == candidate_slug].copy()
        if candidate_frame.empty:
            continue
        for checkpoint_label, day in checkpoint_days:
            point = (
                candidate_frame.loc[candidate_frame["requested_days"] == int(day)]
                .sort_values(["selection_rank", "step_index"])
                .head(1)
            )
            if point.empty:
                continue
            row = point.iloc[0].to_dict()
            rows.append(
                {
                    "path_scope": clean_string(row.get("path_scope")) or "named_stock_path",
                    "stock_path_name": clean_string(row.get("stock_path_name")),
                    "iv_path_name": clean_string(row.get("iv_path_name")),
                    "checkpoint_label": checkpoint_label,
                    "date": clean_string(row.get("date")),
                    "requested_days": int(row.get("requested_days") or 0),
                    "spot_price": finite_or_none(row.get("spot_price")),
                    "iv_shift_points": finite_or_none(row.get("iv_shift_points")),
                    "selection_rank": int(row.get("selection_rank") or 0),
                    "candidate_slug": clean_string(row.get("candidate_slug")),
                    "candidate_label": clean_string(row.get("candidate_label")),
                    "series_label": clean_string(row.get("series_label")),
                    "expiry_date": clean_string(row.get("expiry_date")),
                    "strike_label": clean_string(row.get("strike_label")),
                    "moneyness_bucket": clean_string(row.get("moneyness_bucket")),
                    "source_trust_label": clean_string(row.get("source_trust_label")),
                    "modeled_value": finite_or_none(row.get("modeled_value")),
                    "profit_loss": finite_or_none(row.get("profit_loss")),
                    "difference_vs_stock": finite_or_none(row.get("difference_vs_stock")),
                    "difference_vs_stock_return_pct": finite_or_none(row.get("difference_vs_stock_return_pct")),
                    "success_status": clean_string(row.get("success_status")),
                }
            )
    return pd.DataFrame(rows)


def _iv_path_effect_note(
    *,
    iv_path_name: str,
    value_vs_flat: float | None,
    delta_vs_flat: float | None,
    difference_vs_stock: float | None,
) -> str:
    """Return concise product language for the IV-only comparison read."""

    name = clean_string(iv_path_name).lower()
    value_delta = finite_or_none(value_vs_flat)
    stock_delta = finite_or_none(difference_vs_stock)
    notes: list[str] = []
    if name in {"mean_reversion_lower", "iv_down_then_stays_low"}:
        if value_delta is not None and value_delta < -5:
            notes.append("hurt by IV normalization")
        else:
            notes.append("lower IV path did not materially change the value read")
    elif name == "earnings_build_then_crush":
        if value_delta is not None and value_delta < -5:
            notes.append("IV crush removes value versus flat IV")
        elif value_delta is not None and value_delta > 5:
            notes.append("pre-crush IV build temporarily helps")
        else:
            notes.append("build/crush lands close to the flat-IV read")
    elif name in {"mean_reversion_higher", "iv_up_then_down"}:
        if value_delta is not None and value_delta > 5:
            notes.append("helped by higher IV")
        else:
            notes.append("higher-IV path adds little net help")
    elif name == "flat":
        notes.append("flat-IV baseline")

    if stock_delta is not None:
        if stock_delta < -5:
            notes.append("stock still cleaner under this path")
        elif stock_delta > 5:
            notes.append("option beats stock under this IV path")
        else:
            notes.append("roughly tracks stock")
    if value_delta is not None and value_delta < -25 and stock_delta is not None and stock_delta < 0:
        notes.append("premium looks too rich for this outcome path")
    return "; ".join(_dedupe(notes)) or "See IV checkpoint table for value and stock-relative detail."


def _annotate_iv_path_effects(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    terminal = (
        working.sort_values(["iv_path_display_order", "requested_days", "step_index"])
        .groupby("iv_path_name", dropna=False, as_index=False)
        .tail(1)
    )
    flat = terminal.loc[terminal.get("iv_path_name").astype(str) == "flat"].copy()
    flat_value = finite_or_none(flat.iloc[0].get("modeled_value")) if not flat.empty else None
    flat_delta = finite_or_none(flat.iloc[0].get("difference_vs_stock")) if not flat.empty else None
    value_vs_flat: dict[str, float | None] = {}
    delta_vs_flat: dict[str, float | None] = {}
    notes: dict[str, str] = {}
    for _, row in terminal.iterrows():
        iv_name = clean_string(row.get("iv_path_name"))
        terminal_value = finite_or_none(row.get("modeled_value"))
        terminal_delta = finite_or_none(row.get("difference_vs_stock"))
        value_diff = (
            round(float(terminal_value) - float(flat_value), 4)
            if terminal_value is not None and flat_value is not None
            else None
        )
        delta_diff = (
            round(float(terminal_delta) - float(flat_delta), 4)
            if terminal_delta is not None and flat_delta is not None
            else None
        )
        value_vs_flat[iv_name] = value_diff
        delta_vs_flat[iv_name] = delta_diff
        notes[iv_name] = _iv_path_effect_note(
            iv_path_name=iv_name,
            value_vs_flat=value_diff,
            delta_vs_flat=delta_diff,
            difference_vs_stock=terminal_delta,
        )
    working["terminal_value_vs_flat"] = working["iv_path_name"].map(value_vs_flat)
    working["terminal_delta_vs_flat"] = working["iv_path_name"].map(delta_vs_flat)
    working["iv_effect_note"] = working["iv_path_name"].map(notes).fillna("")
    return working


def _build_iv_path_comparison_frame(
    anchor_row: pd.Series,
    *,
    spec_lookup: dict[str, dict[str, Any]],
    stock_points: list[dict[str, Any]],
    stock_path_name: str,
    target_horizon_label: str,
    active_iv_path_points: dict[str, float],
    comparison_capital: float,
    goal: str,
    target_option_value: float | None,
) -> pd.DataFrame:
    """Value one anchor long call across named IV regimes while stock stays fixed."""

    candidate_slug = clean_string(anchor_row.get("candidate_slug"))
    spec = spec_lookup.get(candidate_slug)
    if spec is None or not stock_points:
        return pd.DataFrame()
    path_grid = [
        {
            "date": point.get("date"),
            "requested_days": int(point.get("requested_days") or 0),
            "step_index": int(point.get("step_index") or 0),
            "time_fraction": point.get("time_fraction", 0.0),
        }
        for point in stock_points
    ]
    base_iv_shift = float(active_iv_path_points.get("entry", next(iter(active_iv_path_points.values()), 0.0)) or 0.0)
    records: list[dict[str, Any]] = []
    anchor_contract_label = _format_long_call_series_label(
        strike_label=anchor_row.get("strike_label"),
        expiry_date=anchor_row.get("expiry_date"),
        moneyness_bucket=anchor_row.get("moneyness_bucket"),
        include_bucket=False,
    )
    for display_order, iv_name in enumerate(_IV_PATH_GALLERY_PRESETS, start=1):
        iv_named_points = _default_iv_path_points(
            preset=iv_name,
            base_shift=base_iv_shift,
            target_horizon_label=target_horizon_label,
        )
        iv_path = _build_iv_path_from_named_points(
            path_grid,
            named_points=iv_named_points,
            iv_path_id=f"path-centric-iv-{iv_name}",
            iv_path_name=iv_name,
            base_iv_shift=base_iv_shift,
        )
        point_count = min(len(stock_points), len(iv_path.path_points))
        for step_index in range(point_count):
            stock_point = stock_points[step_index]
            iv_point = iv_path.path_points[step_index]
            evaluation = _evaluate_at_point(
                spec,
                spot_price=float(stock_point.get("spot_price") or 0.0),
                horizon_days=int(stock_point.get("requested_days") or 0),
                iv_shift_points=float(iv_point.get("iv_shift_points") or 0.0),
                comparison_capital=float(comparison_capital),
            )
            return_on_capital = finite_or_none(evaluation.get("return_on_comparison_capital"))
            stock_return = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
            return_delta = (
                round(float(return_on_capital) - float(stock_return), 6)
                if return_on_capital is not None and stock_return is not None
                else None
            )
            goal_reached = _goal_reached(evaluation, goal=goal, target_option_value=target_option_value)
            records.append(
                {
                    "view_name": "long_call_iv_path_view",
                    "path_scope": "named_stock_path_iv_variation",
                    "stock_path_name": clean_string(stock_path_name),
                    "iv_path_name": clean_string(iv_name),
                    "iv_path_label": _short_iv_path_label(iv_name),
                    "iv_path_display_order": int(display_order),
                    "anchor_candidate_slug": candidate_slug,
                    "anchor_contract_label": anchor_contract_label,
                    "anchor_expiry_date": clean_string(anchor_row.get("expiry_date")),
                    "anchor_strike_label": clean_string(anchor_row.get("strike_label")),
                    "used_strike_fallback": False,
                    "strike_match_mode": "exact_anchor_contract",
                    "fallback_strike_distance": None,
                    "used_trust_fallback": not _preferred_quote_context(anchor_row),
                    "selection_rank": int(display_order),
                    "selection_reason": "Same anchor long call; stock path is fixed and only the IV path varies.",
                    "series_label": _short_iv_path_label(iv_name),
                    "candidate_slug": candidate_slug,
                    "candidate_label": clean_string(anchor_row.get("candidate_label")),
                    "expiry_date": clean_string(anchor_row.get("expiry_date")),
                    "strike_label": clean_string(anchor_row.get("strike_label")),
                    "moneyness_bucket": clean_string(anchor_row.get("moneyness_bucket")),
                    "source_trust_label": clean_string(anchor_row.get("source_trust_label")),
                    "source_quality_note": clean_string(anchor_row.get("source_quality_note")),
                    "objective_score": finite_or_none(anchor_row.get("objective_score")),
                    "date": clean_string(stock_point.get("date")),
                    "requested_days": int(stock_point.get("requested_days") or 0),
                    "step_index": int(stock_point.get("step_index") or 0),
                    "spot_price": float(stock_point.get("spot_price") or 0.0),
                    "iv_shift_points": float(iv_point.get("iv_shift_points") or 0.0),
                    "modeled_value": finite_or_none(evaluation.get("estimated_value")),
                    "profit_loss": finite_or_none(evaluation.get("profit_loss")),
                    "return_on_comparison_capital": return_on_capital,
                    "difference_vs_stock": finite_or_none(evaluation.get("difference_vs_stock")),
                    "difference_vs_stock_return_pct": return_delta,
                    "success_status": _success_status_label(
                        profit_loss=finite_or_none(evaluation.get("profit_loss")),
                        return_on_comparison_capital=return_on_capital,
                        difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                        goal_reached=goal_reached,
                    ),
                }
            )
    return _annotate_iv_path_effects(pd.DataFrame(records))


def _iv_path_chart_scope(iv_name: str) -> str:
    return "core_iv_chart" if clean_string(iv_name).lower() in IV_EXPANDED_CHART_IV_PRESETS else "full_table_only"


def _annotate_iv_expanded_effects(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate IV sensitivity per candidate, not just per stock path."""

    if frame.empty:
        return frame
    working = frame.copy()
    terminal = (
        working.sort_values(["candidate_slug", "iv_path_display_order", "requested_days", "step_index"])
        .groupby(["candidate_slug", "iv_path_name"], dropna=False, as_index=False)
        .tail(1)
    )
    value_vs_flat: dict[tuple[str, str], float | None] = {}
    delta_vs_flat: dict[tuple[str, str], float | None] = {}
    notes: dict[tuple[str, str], str] = {}
    for candidate_slug, candidate_terminal in terminal.groupby("candidate_slug", dropna=False):
        flat = candidate_terminal.loc[candidate_terminal.get("iv_path_name").astype(str) == "flat"].copy()
        flat_value = finite_or_none(flat.iloc[0].get("modeled_value")) if not flat.empty else None
        flat_delta = finite_or_none(flat.iloc[0].get("difference_vs_stock")) if not flat.empty else None
        for _, row in candidate_terminal.iterrows():
            iv_name = clean_string(row.get("iv_path_name"))
            terminal_value = finite_or_none(row.get("modeled_value"))
            terminal_delta = finite_or_none(row.get("difference_vs_stock"))
            value_diff = (
                round(float(terminal_value) - float(flat_value), 4)
                if terminal_value is not None and flat_value is not None
                else None
            )
            delta_diff = (
                round(float(terminal_delta) - float(flat_delta), 4)
                if terminal_delta is not None and flat_delta is not None
                else None
            )
            key = (clean_string(candidate_slug), iv_name)
            value_vs_flat[key] = value_diff
            delta_vs_flat[key] = delta_diff
            notes[key] = _iv_path_effect_note(
                iv_path_name=iv_name,
                value_vs_flat=value_diff,
                delta_vs_flat=delta_diff,
                difference_vs_stock=terminal_delta,
            )

    def lookup(mapping: dict[tuple[str, str], Any], row: pd.Series) -> Any:
        return mapping.get((clean_string(row.get("candidate_slug")), clean_string(row.get("iv_path_name"))))

    working["terminal_value_vs_flat"] = working.apply(lambda row: lookup(value_vs_flat, row), axis=1)
    working["terminal_delta_vs_flat"] = working.apply(lambda row: lookup(delta_vs_flat, row), axis=1)
    working["iv_effect_note"] = working.apply(lambda row: lookup(notes, row) or "", axis=1)
    return working


def _build_long_call_iv_expanded_frame(
    selected_rows: pd.DataFrame,
    *,
    view_name: str,
    iv_expanded_family: str,
    spec_lookup: dict[str, dict[str, Any]],
    stock_points: list[dict[str, Any]],
    stock_path_name: str,
    target_horizon_label: str,
    active_iv_path_points: dict[str, float],
    comparison_capital: float,
    goal: str,
    target_option_value: float | None,
) -> pd.DataFrame:
    """Value a curated long-call view across IV regimes while stock stays fixed."""

    if selected_rows.empty or not stock_points:
        return pd.DataFrame()
    path_grid = [
        {
            "date": point.get("date"),
            "requested_days": int(point.get("requested_days") or 0),
            "step_index": int(point.get("step_index") or 0),
            "time_fraction": point.get("time_fraction", 0.0),
        }
        for point in stock_points
    ]
    base_iv_shift = float(active_iv_path_points.get("entry", next(iter(active_iv_path_points.values()), 0.0)) or 0.0)
    records: list[dict[str, Any]] = []
    selected = selected_rows.sort_values(["selection_rank", "expiry_date", "strike_label"]).head(MAX_LONG_CALL_VIEW_LINES).copy()
    for contract_rank, (_, row) in enumerate(selected.iterrows(), start=1):
        candidate_slug = clean_string(row.get("candidate_slug"))
        spec = spec_lookup.get(candidate_slug)
        if spec is None:
            continue
        contract_label = _format_long_call_series_label(
            strike_label=row.get("strike_label"),
            expiry_date=row.get("expiry_date"),
            moneyness_bucket=row.get("moneyness_bucket"),
            include_bucket=clean_string(iv_expanded_family) == "best_of",
        )
        for display_order, iv_name in enumerate(_IV_PATH_GALLERY_PRESETS, start=1):
            iv_named_points = _default_iv_path_points(
                preset=iv_name,
                base_shift=base_iv_shift,
                target_horizon_label=target_horizon_label,
            )
            iv_path = _build_iv_path_from_named_points(
                path_grid,
                named_points=iv_named_points,
                iv_path_id=f"path-centric-{iv_expanded_family}-iv-{iv_name}",
                iv_path_name=iv_name,
                base_iv_shift=base_iv_shift,
            )
            iv_label = _short_iv_path_label(iv_name)
            chart_include = contract_rank <= IV_EXPANDED_CHART_CONTRACT_LIMIT and iv_name in IV_EXPANDED_CHART_IV_PRESETS
            point_count = min(len(stock_points), len(iv_path.path_points))
            for step_index in range(point_count):
                stock_point = stock_points[step_index]
                iv_point = iv_path.path_points[step_index]
                evaluation = _evaluate_at_point(
                    spec,
                    spot_price=float(stock_point.get("spot_price") or 0.0),
                    horizon_days=int(stock_point.get("requested_days") or 0),
                    iv_shift_points=float(iv_point.get("iv_shift_points") or 0.0),
                    comparison_capital=float(comparison_capital),
                )
                return_on_capital = finite_or_none(evaluation.get("return_on_comparison_capital"))
                stock_return = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
                return_delta = (
                    round(float(return_on_capital) - float(stock_return), 6)
                    if return_on_capital is not None and stock_return is not None
                    else None
                )
                goal_reached = _goal_reached(evaluation, goal=goal, target_option_value=target_option_value)
                records.append(
                    {
                        "view_name": view_name,
                        "path_scope": "named_stock_path_iv_expanded",
                        "stock_path_name": clean_string(stock_path_name),
                        "iv_path_name": clean_string(iv_name),
                        "iv_path_label": iv_label,
                        "iv_path_display_order": int(display_order),
                        "iv_expanded_family": clean_string(iv_expanded_family),
                        "contract_rank": int(contract_rank),
                        "chart_include": bool(chart_include),
                        "iv_chart_scope": _iv_path_chart_scope(iv_name),
                        "anchor_candidate_slug": candidate_slug,
                        "anchor_contract_label": contract_label,
                        "anchor_expiry_date": clean_string(row.get("anchor_expiry_date") or row.get("expiry_date")),
                        "anchor_strike_label": clean_string(row.get("anchor_strike_label") or row.get("strike_label")),
                        "used_strike_fallback": bool(row.get("used_strike_fallback")),
                        "strike_match_mode": clean_string(row.get("strike_match_mode")) or "exact_strike",
                        "fallback_strike_distance": finite_or_none(row.get("fallback_strike_distance")),
                        "used_trust_fallback": bool(row.get("used_trust_fallback")),
                        "selection_rank": int(row.get("selection_rank") or contract_rank),
                        "selection_reason": clean_string(row.get("selection_reason")),
                        "series_label": f"{contract_label} / {iv_label}",
                        "contract_label": contract_label,
                        "candidate_slug": candidate_slug,
                        "candidate_label": clean_string(row.get("candidate_label")),
                        "expiry_date": clean_string(row.get("expiry_date")),
                        "strike_label": clean_string(row.get("strike_label")),
                        "moneyness_bucket": clean_string(row.get("moneyness_bucket")),
                        "source_trust_label": clean_string(row.get("source_trust_label")),
                        "source_quality_note": clean_string(row.get("source_quality_note")),
                        "objective_score": finite_or_none(row.get("objective_score")),
                        "date": clean_string(stock_point.get("date")),
                        "requested_days": int(stock_point.get("requested_days") or 0),
                        "step_index": int(stock_point.get("step_index") or 0),
                        "spot_price": float(stock_point.get("spot_price") or 0.0),
                        "iv_shift_points": float(iv_point.get("iv_shift_points") or 0.0),
                        "modeled_value": finite_or_none(evaluation.get("estimated_value")),
                        "profit_loss": finite_or_none(evaluation.get("profit_loss")),
                        "return_on_comparison_capital": return_on_capital,
                        "difference_vs_stock": finite_or_none(evaluation.get("difference_vs_stock")),
                        "difference_vs_stock_return_pct": return_delta,
                        "success_status": _success_status_label(
                            profit_loss=finite_or_none(evaluation.get("profit_loss")),
                            return_on_comparison_capital=return_on_capital,
                            difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                            goal_reached=goal_reached,
                        ),
                    }
                )
    return _annotate_iv_expanded_effects(pd.DataFrame(records))


def _build_iv_path_checkpoint_rows(view_frame: pd.DataFrame) -> pd.DataFrame:
    if view_frame.empty:
        return pd.DataFrame()
    working = view_frame.copy()
    working["requested_days"] = pd.to_numeric(working.get("requested_days"), errors="coerce")
    working = working.dropna(subset=["requested_days", "iv_path_name"])
    if working.empty:
        return pd.DataFrame()
    working["requested_days"] = working["requested_days"].astype(int)
    max_day = int(working["requested_days"].max())
    checkpoint_specs = [
        ("entry", 0.0),
        ("quarter_check", max_day * 0.25),
        ("mid_check", max_day * 0.50),
        ("target", float(max_day)),
    ]
    days = sorted({int(value) for value in working["requested_days"].tolist()})
    checkpoint_days: list[tuple[str, int]] = []
    seen_days: set[int] = set()
    for label, requested in checkpoint_specs:
        day = _nearest_checkpoint_day(days, requested)
        if day is None or day in seen_days:
            continue
        checkpoint_days.append((label, day))
        seen_days.add(day)
    rows: list[dict[str, Any]] = []
    for _, group in working.groupby("iv_path_name", dropna=False):
        ordered_group = group.sort_values(["iv_path_display_order", "requested_days", "step_index"])
        for checkpoint_label, day in checkpoint_days:
            point = ordered_group.loc[ordered_group["requested_days"] == int(day)].head(1)
            if point.empty:
                continue
            row = point.iloc[0].to_dict()
            rows.append(
                {
                    "path_scope": clean_string(row.get("path_scope")),
                    "stock_path_name": clean_string(row.get("stock_path_name")),
                    "iv_path_name": clean_string(row.get("iv_path_name")),
                    "iv_path_label": clean_string(row.get("iv_path_label")),
                    "checkpoint_label": checkpoint_label,
                    "date": clean_string(row.get("date")),
                    "requested_days": int(row.get("requested_days") or 0),
                    "spot_price": finite_or_none(row.get("spot_price")),
                    "iv_shift_points": finite_or_none(row.get("iv_shift_points")),
                    "anchor_contract_label": clean_string(row.get("anchor_contract_label")),
                    "candidate_slug": clean_string(row.get("candidate_slug")),
                    "candidate_label": clean_string(row.get("candidate_label")),
                    "expiry_date": clean_string(row.get("expiry_date")),
                    "strike_label": clean_string(row.get("strike_label")),
                    "source_trust_label": clean_string(row.get("source_trust_label")),
                    "modeled_value": finite_or_none(row.get("modeled_value")),
                    "profit_loss": finite_or_none(row.get("profit_loss")),
                    "difference_vs_stock": finite_or_none(row.get("difference_vs_stock")),
                    "difference_vs_stock_return_pct": finite_or_none(row.get("difference_vs_stock_return_pct")),
                    "terminal_value_vs_flat": finite_or_none(row.get("terminal_value_vs_flat")),
                    "terminal_delta_vs_flat": finite_or_none(row.get("terminal_delta_vs_flat")),
                    "success_status": clean_string(row.get("success_status")),
                    "iv_effect_note": clean_string(row.get("iv_effect_note")),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["checkpoint_label", "iv_path_name"]).reset_index(drop=True)


def _build_iv_expanded_checkpoint_rows(view_frame: pd.DataFrame) -> pd.DataFrame:
    if view_frame.empty:
        return pd.DataFrame()
    working = view_frame.copy()
    working["requested_days"] = pd.to_numeric(working.get("requested_days"), errors="coerce")
    working = working.dropna(subset=["requested_days", "candidate_slug", "iv_path_name"])
    if working.empty:
        return pd.DataFrame()
    working["requested_days"] = working["requested_days"].astype(int)
    max_day = int(working["requested_days"].max())
    checkpoint_specs = [
        ("entry", 0.0),
        ("quarter_check", max_day * 0.25),
        ("mid_check", max_day * 0.50),
        ("target", float(max_day)),
    ]
    days = sorted({int(value) for value in working["requested_days"].tolist()})
    checkpoint_days: list[tuple[str, int]] = []
    seen_days: set[int] = set()
    for label, requested in checkpoint_specs:
        day = _nearest_checkpoint_day(days, requested)
        if day is None or day in seen_days:
            continue
        checkpoint_days.append((label, day))
        seen_days.add(day)

    rows: list[dict[str, Any]] = []
    compact = working.loc[working.get("chart_include").astype(bool)].copy() if "chart_include" in working.columns else working.copy()
    if compact.empty:
        compact = working.copy()
    for (candidate_slug, iv_name), group in compact.groupby(["candidate_slug", "iv_path_name"], dropna=False):
        ordered_group = group.sort_values(["contract_rank", "iv_path_display_order", "requested_days", "step_index"])
        for checkpoint_label, day in checkpoint_days:
            point = ordered_group.loc[ordered_group["requested_days"] == int(day)].head(1)
            if point.empty:
                continue
            row = point.iloc[0].to_dict()
            rows.append(
                {
                    "path_scope": clean_string(row.get("path_scope")),
                    "stock_path_name": clean_string(row.get("stock_path_name")),
                    "iv_expanded_family": clean_string(row.get("iv_expanded_family")),
                    "iv_path_name": clean_string(row.get("iv_path_name")),
                    "iv_path_label": clean_string(row.get("iv_path_label")),
                    "checkpoint_label": checkpoint_label,
                    "date": clean_string(row.get("date")),
                    "requested_days": int(row.get("requested_days") or 0),
                    "spot_price": finite_or_none(row.get("spot_price")),
                    "iv_shift_points": finite_or_none(row.get("iv_shift_points")),
                    "contract_rank": int(row.get("contract_rank") or 0),
                    "contract_label": clean_string(row.get("contract_label")),
                    "candidate_slug": clean_string(candidate_slug),
                    "candidate_label": clean_string(row.get("candidate_label")),
                    "expiry_date": clean_string(row.get("expiry_date")),
                    "strike_label": clean_string(row.get("strike_label")),
                    "moneyness_bucket": clean_string(row.get("moneyness_bucket")),
                    "source_trust_label": clean_string(row.get("source_trust_label")),
                    "modeled_value": finite_or_none(row.get("modeled_value")),
                    "profit_loss": finite_or_none(row.get("profit_loss")),
                    "difference_vs_stock": finite_or_none(row.get("difference_vs_stock")),
                    "difference_vs_stock_return_pct": finite_or_none(row.get("difference_vs_stock_return_pct")),
                    "terminal_value_vs_flat": finite_or_none(row.get("terminal_value_vs_flat")),
                    "terminal_delta_vs_flat": finite_or_none(row.get("terminal_delta_vs_flat")),
                    "success_status": clean_string(row.get("success_status")),
                    "iv_effect_note": clean_string(row.get("iv_effect_note")),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["iv_expanded_family", "contract_rank", "checkpoint_label", "iv_path_name"]).reset_index(drop=True)


def _iv_robustness_label(
    *,
    iv_path_count: int,
    beat_stock_count: int,
    profitable_count: int,
    lower_iv_profitable: bool,
    lower_iv_beats_stock: bool,
    high_iv_dependency: bool,
) -> str:
    if iv_path_count > 0 and beat_stock_count == iv_path_count:
        return "robust_beats_stock_across_iv"
    if lower_iv_beats_stock:
        return "survives_lower_iv_and_beats_stock"
    if lower_iv_profitable and beat_stock_count > 0:
        return "survives_lower_iv_but_stock_may_win"
    if high_iv_dependency:
        return "requires_iv_support"
    if profitable_count > 0 and beat_stock_count == 0:
        return "option_value_survives_but_stock_cleaner"
    return "fragile_to_iv"


def _iv_robustness_note(label: str) -> str:
    notes = {
        "robust_beats_stock_across_iv": "beats stock across all modeled IV regimes for this path",
        "survives_lower_iv_and_beats_stock": "still works even when IV mean-reverts lower",
        "survives_lower_iv_but_stock_may_win": "option value survives lower IV, but stock may still be cleaner",
        "requires_iv_support": "depends on elevated or supportive IV to beat stock",
        "option_value_survives_but_stock_cleaner": "can stay profitable, but stock dominates the risk-adjusted read",
        "fragile_to_iv": "fragile to IV normalization or crush under this stock path",
    }
    return notes.get(label, "review IV checkpoint rows for the contract-specific read")


def _build_iv_robustness_summary_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    terminal = pd.concat(non_empty, ignore_index=True)
    terminal = (
        terminal.sort_values(["iv_expanded_family", "candidate_slug", "iv_path_display_order", "requested_days", "step_index"])
        .groupby(["iv_expanded_family", "candidate_slug", "iv_path_name"], dropna=False, as_index=False)
        .tail(1)
    )
    if terminal.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (family, candidate_slug), group in terminal.groupby(["iv_expanded_family", "candidate_slug"], dropna=False):
        ordered = group.sort_values(["iv_path_display_order", "contract_rank"])
        profit = pd.to_numeric(ordered.get("profit_loss"), errors="coerce")
        delta = pd.to_numeric(ordered.get("difference_vs_stock"), errors="coerce")
        value = pd.to_numeric(ordered.get("modeled_value"), errors="coerce")
        profitable = profit >= 0
        beat_stock = delta > 0
        iv_names = ordered.get("iv_path_name").astype(str)
        lower_mask = iv_names.isin(LOWER_IV_PRESETS)
        higher_mask = iv_names.isin(HIGHER_IV_PRESETS)
        lower_iv_profitable = bool(profitable.loc[lower_mask].any()) if lower_mask.any() else False
        lower_iv_beats_stock = bool(beat_stock.loc[lower_mask].any()) if lower_mask.any() else False
        high_iv_beats = bool(beat_stock.loc[higher_mask].any()) if higher_mask.any() else False
        flat_or_lower_beats = bool(beat_stock.loc[iv_names.isin({"flat", *LOWER_IV_PRESETS})].any())
        high_iv_dependency = high_iv_beats and not flat_or_lower_beats
        label = _iv_robustness_label(
            iv_path_count=int(ordered["iv_path_name"].nunique()),
            beat_stock_count=int(beat_stock.sum()),
            profitable_count=int(profitable.sum()),
            lower_iv_profitable=lower_iv_profitable,
            lower_iv_beats_stock=lower_iv_beats_stock,
            high_iv_dependency=high_iv_dependency,
        )
        best = ordered.loc[delta.idxmax()] if delta.notna().any() else ordered.iloc[0]
        worst = ordered.loc[delta.idxmin()] if delta.notna().any() else ordered.iloc[-1]
        lead = ordered.iloc[0].to_dict()
        rows.append(
            {
                "stock_path_name": clean_string(lead.get("stock_path_name")),
                "iv_expanded_family": clean_string(family),
                "candidate_slug": clean_string(candidate_slug),
                "candidate_label": clean_string(lead.get("candidate_label")),
                "contract_label": clean_string(lead.get("contract_label")) or clean_string(lead.get("anchor_contract_label")),
                "expiry_date": clean_string(lead.get("expiry_date")),
                "strike_label": clean_string(lead.get("strike_label")),
                "moneyness_bucket": clean_string(lead.get("moneyness_bucket")),
                "source_trust_label": clean_string(lead.get("source_trust_label")),
                "contract_rank": int(lead.get("contract_rank") or 0),
                "iv_path_count": int(ordered["iv_path_name"].nunique()),
                "profitable_iv_path_count": int(profitable.sum()),
                "beat_stock_iv_path_count": int(beat_stock.sum()),
                "lower_iv_profitable": lower_iv_profitable,
                "lower_iv_beats_stock": lower_iv_beats_stock,
                "high_iv_dependency": high_iv_dependency,
                "terminal_value_min": finite_or_none(value.min()),
                "terminal_value_max": finite_or_none(value.max()),
                "terminal_value_range": finite_or_none(value.max() - value.min()),
                "terminal_delta_vs_stock_min": finite_or_none(delta.min()),
                "terminal_delta_vs_stock_max": finite_or_none(delta.max()),
                "terminal_delta_vs_stock_range": finite_or_none(delta.max() - delta.min()),
                "best_iv_path": clean_string(best.get("iv_path_name")),
                "worst_iv_path": clean_string(worst.get("iv_path_name")),
                "iv_robustness_label": label,
                "iv_robustness_note": _iv_robustness_note(label),
            }
        )
    return pd.DataFrame(rows).sort_values(["iv_expanded_family", "contract_rank", "candidate_label"]).reset_index(drop=True)


def _build_path_centric_long_call_views(
    *,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    snapshot_date: date,
    target_date: date,
    target_price: float,
    target_horizon_label: str,
    entry_spot: float,
    iv_path_name: str,
    iv_path_points: dict[str, float],
    comparison_capital: float,
    goal: str,
    target_option_value: float | None,
    objective_mode: str,
    downside_tolerance: str,
    simplicity_preference: str,
    active_stock_path_name: str,
    active_stock_path_points: dict[str, float],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    if candidate_rows.empty:
        return {}, []
    long_calls = candidate_rows.loc[candidate_rows.get("strategy_family").astype(str) == "long_call"].copy()
    if long_calls.empty:
        return {}, []
    spec_lookup = {
        clean_string(spec.get("candidate_slug")): spec
        for spec in specs
        if clean_string(spec.get("strategy_family")) == "long_call"
    }
    if not spec_lookup:
        return {}, []

    path_grid = _build_path_grid(snapshot_date, target_date)
    active_iv_path = _build_iv_path_from_named_points(
        path_grid,
        named_points=iv_path_points,
        iv_path_id="path-centric-active-iv",
        iv_path_name=iv_path_name,
        base_iv_shift=float(next(iter(iv_path_points.values()), 0.0)),
    )
    path_view_tables: dict[str, pd.DataFrame] = {}
    focus_metadata: list[dict[str, str]] = []
    focus_paths = list(_PATH_CENTRIC_FOCUS_PRESETS)
    if clean_string(active_stock_path_name).lower() not in focus_paths:
        focus_paths.append(clean_string(active_stock_path_name).lower() or "active_assumed_path")

    for path_name in focus_paths:
        if clean_string(path_name).lower() == clean_string(active_stock_path_name).lower() and clean_string(path_name).lower() not in _PATH_CENTRIC_FOCUS_PRESETS:
            path_points = dict(active_stock_path_points)
        else:
            path_points = _stock_path_gallery_named_points(
                preset=path_name,
                entry_spot=float(entry_spot),
                target_price=float(target_price),
                target_horizon_label=target_horizon_label,
            )
        stock_path = _build_stock_path_from_named_points(
            path_grid,
            named_points=path_points,
            path_id=f"path-centric-stock-{path_name}",
            path_name=path_name,
            entry_spot=float(entry_spot),
        )
        terminal_rows = _path_specific_long_call_terminal_rows(
            long_calls,
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=path_name,
            iv_path_name=iv_path_name,
            comparison_capital=float(comparison_capital),
            objective_mode=objective_mode,
            downside_tolerance=downside_tolerance,
            simplicity_preference=simplicity_preference,
        )
        if terminal_rows.empty:
            continue
        anchor = _sort_candidate_priority(terminal_rows).head(1)
        if anchor.empty:
            continue
        anchor_row = anchor.iloc[0]
        strike_selection = _select_long_call_strike_view_rows(terminal_rows, anchor_row=anchor_row)
        expiry_selection = _select_long_call_expiry_view_rows(terminal_rows, anchor_row=anchor_row)
        best_of_selection = _select_long_call_best_of_rows(terminal_rows, anchor_row=anchor_row)
        strike_frame = _build_long_call_view_frame(
            strike_selection,
            view_name="long_call_strike_view",
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=path_name,
            iv_path_name=iv_path_name,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        expiry_frame = _build_long_call_view_frame(
            expiry_selection,
            view_name="long_call_expiry_view",
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=path_name,
            iv_path_name=iv_path_name,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        best_of_frame = _build_long_call_view_frame(
            best_of_selection,
            view_name="long_call_best_of_view",
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            iv_points=active_iv_path.path_points,
            stock_path_name=path_name,
            iv_path_name=iv_path_name,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        strike_delta_frame = _build_long_call_compare_vs_stock_rows(strike_frame)
        expiry_delta_frame = _build_long_call_compare_vs_stock_rows(expiry_frame)
        best_of_delta_frame = _build_long_call_compare_vs_stock_rows(best_of_frame)
        compare_frame = best_of_delta_frame
        checkpoint_frame = _build_path_checkpoint_rows(best_of_frame)
        iv_value_frame = _build_iv_path_comparison_frame(
            anchor_row,
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            stock_path_name=path_name,
            target_horizon_label=target_horizon_label,
            active_iv_path_points=iv_path_points,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        iv_delta_frame = _build_long_call_compare_vs_stock_rows(iv_value_frame)
        iv_checkpoint_frame = _build_iv_path_checkpoint_rows(iv_value_frame)
        strike_iv_value_frame = _build_long_call_iv_expanded_frame(
            strike_selection,
            view_name="long_call_strike_iv_view",
            iv_expanded_family="strike",
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            stock_path_name=path_name,
            target_horizon_label=target_horizon_label,
            active_iv_path_points=iv_path_points,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        strike_iv_delta_frame = _build_long_call_compare_vs_stock_rows(strike_iv_value_frame)
        strike_iv_checkpoint_frame = _build_iv_expanded_checkpoint_rows(strike_iv_value_frame)
        expiry_iv_value_frame = _build_long_call_iv_expanded_frame(
            expiry_selection,
            view_name="long_call_expiry_iv_view",
            iv_expanded_family="expiry",
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            stock_path_name=path_name,
            target_horizon_label=target_horizon_label,
            active_iv_path_points=iv_path_points,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        expiry_iv_delta_frame = _build_long_call_compare_vs_stock_rows(expiry_iv_value_frame)
        expiry_iv_checkpoint_frame = _build_iv_expanded_checkpoint_rows(expiry_iv_value_frame)
        best_of_iv_value_frame = _build_long_call_iv_expanded_frame(
            best_of_selection,
            view_name="long_call_best_of_iv_view",
            iv_expanded_family="best_of",
            spec_lookup=spec_lookup,
            stock_points=stock_path.path_points,
            stock_path_name=path_name,
            target_horizon_label=target_horizon_label,
            active_iv_path_points=iv_path_points,
            comparison_capital=float(comparison_capital),
            goal=goal,
            target_option_value=target_option_value,
        )
        best_of_iv_delta_frame = _build_long_call_compare_vs_stock_rows(best_of_iv_value_frame)
        best_of_iv_checkpoint_frame = _build_iv_expanded_checkpoint_rows(best_of_iv_value_frame)
        iv_robustness_summary_frame = _build_iv_robustness_summary_frame(
            [strike_iv_value_frame, expiry_iv_value_frame, best_of_iv_value_frame]
        )

        path_view_tables[_path_view_filename(path_name, "compare_vs_stock_path_rows.csv")] = compare_frame
        path_view_tables[_path_view_filename(path_name, "long_call_strike_value.csv")] = strike_frame
        path_view_tables[_path_view_filename(path_name, "long_call_strike_delta.csv")] = strike_delta_frame
        path_view_tables[_path_view_filename(path_name, "long_call_expiry_value.csv")] = expiry_frame
        path_view_tables[_path_view_filename(path_name, "long_call_expiry_delta.csv")] = expiry_delta_frame
        path_view_tables[_path_view_filename(path_name, "long_call_best_of_value.csv")] = best_of_frame
        path_view_tables[_path_view_filename(path_name, "long_call_best_of_delta.csv")] = best_of_delta_frame
        path_view_tables[_path_view_filename(path_name, "path_checkpoints.csv")] = checkpoint_frame
        path_view_tables[_path_view_filename(path_name, "iv_path_value.csv")] = iv_value_frame
        path_view_tables[_path_view_filename(path_name, "iv_path_delta.csv")] = iv_delta_frame
        path_view_tables[_path_view_filename(path_name, "iv_checkpoints.csv")] = iv_checkpoint_frame
        path_view_tables[_path_view_filename(path_name, "long_call_strike_iv_value.csv")] = strike_iv_value_frame
        path_view_tables[_path_view_filename(path_name, "long_call_strike_iv_delta.csv")] = strike_iv_delta_frame
        path_view_tables[_path_view_filename(path_name, "long_call_strike_iv_checkpoints.csv")] = strike_iv_checkpoint_frame
        path_view_tables[_path_view_filename(path_name, "long_call_expiry_iv_value.csv")] = expiry_iv_value_frame
        path_view_tables[_path_view_filename(path_name, "long_call_expiry_iv_delta.csv")] = expiry_iv_delta_frame
        path_view_tables[_path_view_filename(path_name, "long_call_expiry_iv_checkpoints.csv")] = expiry_iv_checkpoint_frame
        path_view_tables[_path_view_filename(path_name, "long_call_best_of_iv_value.csv")] = best_of_iv_value_frame
        path_view_tables[_path_view_filename(path_name, "long_call_best_of_iv_delta.csv")] = best_of_iv_delta_frame
        path_view_tables[_path_view_filename(path_name, "long_call_best_of_iv_checkpoints.csv")] = best_of_iv_checkpoint_frame
        path_view_tables[_path_view_filename(path_name, "iv_robustness_summary.csv")] = iv_robustness_summary_frame
        strike_value_table = _path_view_filename(path_name, "long_call_strike_value.csv")
        strike_delta_table = _path_view_filename(path_name, "long_call_strike_delta.csv")
        expiry_value_table = _path_view_filename(path_name, "long_call_expiry_value.csv")
        expiry_delta_table = _path_view_filename(path_name, "long_call_expiry_delta.csv")
        best_of_value_table = _path_view_filename(path_name, "long_call_best_of_value.csv")
        best_of_delta_table = _path_view_filename(path_name, "long_call_best_of_delta.csv")
        iv_value_table = _path_view_filename(path_name, "iv_path_value.csv")
        iv_delta_table = _path_view_filename(path_name, "iv_path_delta.csv")
        iv_checkpoint_table = _path_view_filename(path_name, "iv_checkpoints.csv")
        strike_iv_value_table = _path_view_filename(path_name, "long_call_strike_iv_value.csv")
        strike_iv_delta_table = _path_view_filename(path_name, "long_call_strike_iv_delta.csv")
        strike_iv_checkpoint_table = _path_view_filename(path_name, "long_call_strike_iv_checkpoints.csv")
        expiry_iv_value_table = _path_view_filename(path_name, "long_call_expiry_iv_value.csv")
        expiry_iv_delta_table = _path_view_filename(path_name, "long_call_expiry_iv_delta.csv")
        expiry_iv_checkpoint_table = _path_view_filename(path_name, "long_call_expiry_iv_checkpoints.csv")
        best_of_iv_value_table = _path_view_filename(path_name, "long_call_best_of_iv_value.csv")
        best_of_iv_delta_table = _path_view_filename(path_name, "long_call_best_of_iv_delta.csv")
        best_of_iv_checkpoint_table = _path_view_filename(path_name, "long_call_best_of_iv_checkpoints.csv")
        iv_robustness_summary_table = _path_view_filename(path_name, "iv_robustness_summary.csv")
        focus_metadata.append(
            {
                "path_name": clean_string(path_name),
                "path_label": _humanize_named_path(path_name, kind="stock"),
                "compare_table": _path_view_filename(path_name, "compare_vs_stock_path_rows.csv"),
                "compare_chart": _path_view_filename(path_name, "compare_vs_stock_path_delta.png"),
                "strike_table": strike_value_table,
                "strike_chart": _path_view_filename(path_name, "long_call_strike_value.png"),
                "strike_value_table": strike_value_table,
                "strike_value_chart": _path_view_filename(path_name, "long_call_strike_value.png"),
                "strike_delta_table": strike_delta_table,
                "strike_delta_chart": _path_view_filename(path_name, "long_call_strike_delta.png"),
                "expiry_table": expiry_value_table,
                "expiry_chart": _path_view_filename(path_name, "long_call_expiry_value.png"),
                "expiry_value_table": expiry_value_table,
                "expiry_value_chart": _path_view_filename(path_name, "long_call_expiry_value.png"),
                "expiry_delta_table": expiry_delta_table,
                "expiry_delta_chart": _path_view_filename(path_name, "long_call_expiry_delta.png"),
                "best_of_table": best_of_value_table,
                "best_of_chart": _path_view_filename(path_name, "long_call_best_of_value.png"),
                "best_of_value_table": best_of_value_table,
                "best_of_value_chart": _path_view_filename(path_name, "long_call_best_of_value.png"),
                "best_of_delta_table": best_of_delta_table,
                "best_of_delta_chart": _path_view_filename(path_name, "long_call_best_of_delta.png"),
                "checkpoint_table": _path_view_filename(path_name, "path_checkpoints.csv"),
                "iv_value_table": iv_value_table,
                "iv_value_chart": _path_view_filename(path_name, "iv_path_value.png"),
                "iv_delta_table": iv_delta_table,
                "iv_delta_chart": _path_view_filename(path_name, "iv_path_delta.png"),
                "iv_checkpoint_table": iv_checkpoint_table,
                "strike_iv_value_table": strike_iv_value_table,
                "strike_iv_value_chart": _path_view_filename(path_name, "long_call_strike_iv_value.png"),
                "strike_iv_delta_table": strike_iv_delta_table,
                "strike_iv_delta_chart": _path_view_filename(path_name, "long_call_strike_iv_delta.png"),
                "strike_iv_checkpoint_table": strike_iv_checkpoint_table,
                "expiry_iv_value_table": expiry_iv_value_table,
                "expiry_iv_value_chart": _path_view_filename(path_name, "long_call_expiry_iv_value.png"),
                "expiry_iv_delta_table": expiry_iv_delta_table,
                "expiry_iv_delta_chart": _path_view_filename(path_name, "long_call_expiry_iv_delta.png"),
                "expiry_iv_checkpoint_table": expiry_iv_checkpoint_table,
                "best_of_iv_value_table": best_of_iv_value_table,
                "best_of_iv_value_chart": _path_view_filename(path_name, "long_call_best_of_iv_value.png"),
                "best_of_iv_delta_table": best_of_iv_delta_table,
                "best_of_iv_delta_chart": _path_view_filename(path_name, "long_call_best_of_iv_delta.png"),
                "best_of_iv_checkpoint_table": best_of_iv_checkpoint_table,
                "iv_robustness_summary_table": iv_robustness_summary_table,
            }
        )
    return path_view_tables, focus_metadata


def _build_assumed_path_galleries(
    *,
    snapshot_date: date,
    target_date: date,
    target_horizon_label: str,
    target_price: float,
    entry_spot: float,
    stock_path_name: str,
    stock_path_points: dict[str, float],
    iv_path_name: str,
    iv_path_points: dict[str, float],
) -> dict[str, pd.DataFrame]:
    path_grid = _build_path_grid(snapshot_date, target_date)
    return {
        "stock_path_library": _build_stock_path_library_rows(active_path_name=stock_path_name),
        "stock_path_gallery": _build_stock_path_gallery_rows(
            path_grid,
            entry_spot=float(entry_spot),
            target_price=float(target_price),
            target_horizon_label=target_horizon_label,
            active_path_name=stock_path_name,
            active_named_points=stock_path_points,
        ),
        "iv_path_gallery": _build_iv_path_gallery_rows(
            path_grid,
            target_horizon_label=target_horizon_label,
            active_iv_path_name=iv_path_name,
            active_named_points=iv_path_points,
        ),
    }


def _thesis_status_from_gap(*, premium_gap: float | None, current_premium: float | None, beats_stock_rate: float) -> str:
    gap = float(premium_gap or 0.0)
    premium = max(float(current_premium or 0.0), 1.0)
    if gap >= 0 and float(beats_stock_rate or 0.0) >= 0.45:
        return "reasonable_under_thesis"
    if gap >= -0.20 * premium or float(beats_stock_rate or 0.0) >= 0.35:
        return "near_watchlist_under_thesis"
    return "too_expensive_under_thesis"


def _thesis_status_note(
    status: str,
    *,
    stock_still_better: bool,
    iv_sensitive: bool,
    path_sensitive: bool,
    thesis_target_beyond_expiry: bool,
) -> str:
    status_text = clean_string(status)
    if thesis_target_beyond_expiry:
        return "The thesis target date is after expiry, so this only works if the move arrives early enough before expiration."
    if status_text == "reasonable_under_thesis":
        return "Thesis can justify the premium in enough path/IV cases to inspect the contract."
    if stock_still_better:
        return "Stock still looks cleaner because option edge is too narrow versus the benchmark."
    if iv_sensitive:
        return "Interesting only if IV does not normalize too harshly."
    if path_sensitive:
        return "Interesting only if the path to target is option-friendly enough."
    return "Current premium still asks too much under the thesis scenarios."


def _build_thesis_mode_markdown(
    *,
    ticker: str,
    thesis_target_price: float,
    thesis_target_date: date,
    ranking: pd.DataFrame,
    path_summary: pd.DataFrame,
    iv_summary: pd.DataFrame,
) -> str:
    def top_lines(frame: pd.DataFrame, *, status_filter: set[str] | None = None, limit: int = 3) -> list[str]:
        if frame.empty:
            return ["- No long-call rows were available for thesis mode."]
        data = frame.copy()
        if status_filter is not None:
            data = data.loc[data.get("entry_attractiveness_status", pd.Series(dtype=str)).astype(str).isin(status_filter)].copy()
        if data.empty:
            return ["- No candidates clearly cleared this section under the thesis assumptions."]
        output = []
        for row in data.sort_values(["thesis_candidate_rank"]).head(limit).to_dict("records"):
            label = clean_string(row.get("candidate_short_label")) or clean_string(row.get("candidate_label"))
            status = clean_string(row.get("entry_attractiveness_status")).replace("_", " ")
            gap = finite_or_none(row.get("premium_gap"))
            stock_note = "stock still cleaner" if bool(row.get("stock_still_better_under_thesis")) else "option can show edge"
            output.append(
                f"- `{label}` - {status}; premium gap {gap:+.0f} if measured on contract dollars; {stock_note}. {clean_string(row.get('main_reason'))}"
                if gap is not None
                else f"- `{label}` - {status}; {stock_note}. {clean_string(row.get('main_reason'))}"
            )
        return output

    best_paths = []
    if not path_summary.empty:
        for row in path_summary.sort_values("average_candidate_profit_loss", ascending=False).head(3).to_dict("records"):
            best_paths.append(
                f"- {clean_string(row.get('path_label'))}: best read is `{clean_string(row.get('best_candidate_short_label'))}`; beat-stock rate {float(row.get('beat_stock_rate') or 0.0):.0%}."
            )
    if not best_paths:
        best_paths = ["- No path family had enough usable rows to summarize."]

    iv_lines = []
    if not iv_summary.empty:
        for row in iv_summary.sort_values("average_candidate_profit_loss", ascending=False).head(3).to_dict("records"):
            effect_note = clean_string(row.get("iv_effect_note")).rstrip(".")
            iv_lines.append(
                f"- {clean_string(row.get('iv_path_label'))}: beat-stock rate {float(row.get('beat_stock_rate') or 0.0):.0%}; {effect_note}."
            )
    if not iv_lines:
        iv_lines = ["- No IV family had enough usable rows to summarize."]

    if ranking.empty:
        premium_read = (
            "No long-call rows were available, so thesis-justified premium cannot be computed yet. "
            "Use the path and IV galleries to inspect the thesis shape, then refresh/use a quoted chain before treating calls as actionable."
        )
        next_files = [
            "- `charts/thesis_path_gallery.png`",
            "- `charts/thesis_iv_gallery.png`",
            "- `tables/thesis_path_gallery.csv`",
            "- `tables/thesis_iv_gallery.csv`",
        ]
    else:
        premium_read = (
            "Use `tables/current_vs_justified_premium.csv` and `charts/current_vs_justified_premium.png`. "
            "A negative premium gap means the contract still asks more than this thesis justifies after stock-benchmark pressure."
        )
        next_files = [
            "- `charts/thesis_candidate_overview.png`",
            "- `charts/current_vs_justified_premium.png`",
            "- `charts/thesis_path_gallery.png`",
            "- `charts/thesis_path_vs_value.png`",
            "- `charts/thesis_iv_vs_value.png`",
            "- `charts/thesis_stock_vs_option.png`",
            "- `tables/thesis_candidate_ranking.csv`",
        ]

    return "\n".join(
        [
            f"# {ticker} Thesis Mode",
            "",
            "This is a thesis-relative entry read, not a universal fair-value engine.",
            "",
            "## Thesis Snapshot",
            "",
            f"- Target: `${float(thesis_target_price):,.2f}` by `{thesis_target_date.isoformat()}`",
            "- Stock paths: multiple curated ways to reach the same endpoint",
            "- IV paths: flat, lower-IV, higher-IV, up/down, down/low, and build/crush regimes",
            "",
            "## What This Target Means For Option Selection",
            "",
            "The same endpoint can make a call look very different depending on timing and IV. Fast moves and supportive IV help long calls; slow paths, falling IV, and rich premium keep stock cleaner.",
            "",
            "## Which Calls Start To Look Reasonable Under This Thesis",
            "",
            *top_lines(ranking, status_filter={"reasonable_under_thesis"}),
            "",
            "## Which Calls Still Require Too Much",
            "",
            *top_lines(ranking, status_filter={"too_expensive_under_thesis", "near_watchlist_under_thesis"}, limit=4),
            "",
            "## Which Paths To The Target Help Most",
            "",
            *best_paths,
            "",
            "## Which IV Regimes Help Or Hurt",
            "",
            *iv_lines,
            "",
            "## Current Premium vs Thesis-Justified Premium",
            "",
            premium_read,
            "",
            "## When Stock Still Looks Better",
            "",
            *top_lines(ranking.loc[ranking.get("stock_still_better_under_thesis", pd.Series(dtype=bool)).fillna(False)] if not ranking.empty else ranking, limit=4),
            "",
            "## Best Next Files To Open",
            "",
            *next_files,
        ]
    )


def _build_thesis_mode_outputs(
    *,
    ticker: str,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    snapshot_date: date,
    thesis_target_price: float,
    thesis_target_date: date,
    thesis_horizon_label: str,
    entry_spot: float,
    comparison_capital: float,
    objective_mode: str,
    downside_tolerance: str,
    simplicity_preference: str,
) -> dict[str, Any]:
    empty_outputs = {
        "thesis_path_gallery": pd.DataFrame(),
        "thesis_iv_gallery": pd.DataFrame(),
        "thesis_mode_candidates": pd.DataFrame(),
        "thesis_path_family_summary": pd.DataFrame(),
        "thesis_iv_family_summary": pd.DataFrame(),
        "thesis_candidate_ranking": pd.DataFrame(),
        "max_justified_premium": pd.DataFrame(),
        "current_vs_justified_premium": pd.DataFrame(),
        "thesis_required_move_summary": pd.DataFrame(),
        "thesis_stock_vs_option_summary": pd.DataFrame(),
        "thesis_mode_markdown": "",
    }
    path_grid = _build_path_grid(snapshot_date, thesis_target_date)
    thesis_path_rows: list[dict[str, Any]] = []
    stock_paths: dict[str, list[dict[str, Any]]] = {}
    for display_order, path_name in enumerate(THESIS_STOCK_PATH_PRESETS, start=1):
        named_points = _default_stock_path_points(
            preset=path_name,
            entry_spot=float(entry_spot),
            target_price=float(thesis_target_price),
            target_horizon_label=thesis_horizon_label,
        )
        stock_path = _build_stock_path_from_named_points(
            path_grid,
            named_points=named_points,
            path_id=f"thesis-stock-{path_name}",
            path_name=path_name,
            entry_spot=float(entry_spot),
        )
        stock_paths[path_name] = list(stock_path.path_points)
        for point in stock_path.path_points:
            thesis_path_rows.append(
                {
                    "path_family": path_name,
                    "path_label": _humanize_named_path(path_name, kind="stock"),
                    "display_order": display_order,
                    "date": point.get("date"),
                    "requested_days": point.get("requested_days"),
                    "stock_price": point.get("spot_price"),
                    "return_pct": point.get("return_pct"),
                    "target_price": float(thesis_target_price),
                    "target_date": thesis_target_date.isoformat(),
                }
            )

    thesis_iv_rows: list[dict[str, Any]] = []
    iv_paths: dict[str, list[dict[str, Any]]] = {}
    for display_order, iv_name in enumerate(THESIS_IV_PATH_PRESETS, start=1):
        named_points = _default_iv_path_points(
            preset=iv_name,
            base_shift=0.0,
            target_horizon_label=thesis_horizon_label,
        )
        iv_path = _build_iv_path_from_named_points(
            path_grid,
            named_points=named_points,
            iv_path_id=f"thesis-iv-{iv_name}",
            iv_path_name=iv_name,
            base_iv_shift=0.0,
        )
        iv_paths[iv_name] = list(iv_path.path_points)
        for point in iv_path.path_points:
            thesis_iv_rows.append(
                {
                    "iv_path_name": iv_name,
                    "iv_path_label": _short_iv_path_label(iv_name),
                    "display_order": display_order,
                    "date": point.get("date"),
                    "requested_days": point.get("requested_days"),
                    "iv_shift_points": point.get("iv_shift_points"),
                    "target_date": thesis_target_date.isoformat(),
                }
            )

    gallery_only_outputs = empty_outputs | {
        "thesis_path_gallery": pd.DataFrame(thesis_path_rows),
        "thesis_iv_gallery": pd.DataFrame(thesis_iv_rows),
        "thesis_mode_markdown": _build_thesis_mode_markdown(
            ticker=ticker,
            thesis_target_price=float(thesis_target_price),
            thesis_target_date=thesis_target_date,
            ranking=pd.DataFrame(),
            path_summary=pd.DataFrame(),
            iv_summary=pd.DataFrame(),
        ),
    }

    if candidate_rows.empty or "strategy_family" not in candidate_rows.columns:
        return gallery_only_outputs
    long_calls = candidate_rows.loc[candidate_rows.get("strategy_family").astype(str).eq("long_call")].copy()
    if long_calls.empty:
        return gallery_only_outputs
    spec_lookup = {
        clean_string(spec.get("candidate_slug")): spec
        for spec in specs
        if clean_string(spec.get("strategy_family")) == "long_call"
    }
    if not spec_lookup:
        return gallery_only_outputs

    long_calls = _sort_candidate_priority(long_calls).head(THESIS_CANDIDATE_LIMIT).copy()
    scenario_rows: list[dict[str, Any]] = []
    for _, candidate in long_calls.iterrows():
        candidate_slug = clean_string(candidate.get("candidate_slug"))
        spec = spec_lookup.get(candidate_slug)
        if spec is None:
            continue
        position: StrategyPosition = spec["position"]
        current_premium = float(position.initial_outlay)
        candidate_short_label = _format_long_call_series_label(
            strike_label=candidate.get("strike_label"),
            expiry_date=candidate.get("expiry_date"),
            moneyness_bucket=candidate.get("moneyness_bucket"),
            include_bucket=False,
        )
        expiry_dt = parse_date(candidate.get("expiry_date"))
        thesis_target_beyond_expiry = bool(expiry_dt is not None and thesis_target_date > expiry_dt)
        for path_name, stock_points in stock_paths.items():
            for iv_name, iv_points in iv_paths.items():
                count = min(len(stock_points), len(iv_points))
                if count <= 0:
                    continue
                terminal_stock = stock_points[count - 1]
                terminal_iv = iv_points[count - 1]
                evaluation = _evaluate_at_point(
                    spec,
                    spot_price=float(terminal_stock.get("spot_price") or 0.0),
                    horizon_days=int(terminal_stock.get("requested_days") or 0),
                    iv_shift_points=float(terminal_iv.get("iv_shift_points") or 0.0),
                    comparison_capital=float(comparison_capital),
                )
                modeled_value = finite_or_none(evaluation.get("estimated_value")) or 0.0
                stock_profit_loss = finite_or_none(evaluation.get("stock_profit_loss")) or 0.0
                stock_relative_justified = max(float(modeled_value) - max(float(stock_profit_loss), 0.0), 0.0)
                scenario_rows.append(
                    {
                        "thesis_target_price": float(thesis_target_price),
                        "thesis_target_date": thesis_target_date.isoformat(),
                        "path_family": path_name,
                        "path_label": _humanize_named_path(path_name, kind="stock"),
                        "iv_path_name": iv_name,
                        "iv_path_label": _short_iv_path_label(iv_name),
                        "candidate_slug": candidate_slug,
                        "candidate_label": clean_string(candidate.get("candidate_label")),
                        "candidate_short_label": candidate_short_label,
                        "expiry_date": clean_string(candidate.get("expiry_date")),
                        "strike_label": clean_string(candidate.get("strike_label")),
                        "moneyness_bucket": clean_string(candidate.get("moneyness_bucket")),
                        "source_trust_label": clean_string(candidate.get("source_trust_label")),
                        "current_premium": round(float(current_premium), 4),
                        "terminal_stock_price": finite_or_none(terminal_stock.get("spot_price")),
                        "terminal_iv_shift_points": finite_or_none(terminal_iv.get("iv_shift_points")),
                        "thesis_terminal_value": finite_or_none(evaluation.get("estimated_value")),
                        "profit_loss": finite_or_none(evaluation.get("profit_loss")),
                        "return_on_comparison_capital": finite_or_none(evaluation.get("return_on_comparison_capital")),
                        "stock_profit_loss": finite_or_none(evaluation.get("stock_profit_loss")),
                        "stock_return_on_comparison_capital": finite_or_none(evaluation.get("stock_return_on_comparison_capital")),
                        "difference_vs_stock": finite_or_none(evaluation.get("difference_vs_stock")),
                        "break_even_justified_premium": round(max(float(modeled_value), 0.0), 4),
                        "stock_relative_justified_premium": round(float(stock_relative_justified), 4),
                        "stock_still_better": bool((finite_or_none(evaluation.get("difference_vs_stock")) or 0.0) <= 0.0),
                        "target_beyond_expiry": bool(candidate.get("target_beyond_expiry")),
                        "thesis_target_beyond_expiry": thesis_target_beyond_expiry,
                        "weak_horizon_fit": bool(candidate.get("weak_horizon_fit")),
                        "objective_score": finite_or_none(candidate.get("objective_score")),
                    }
                )

    thesis_candidates = pd.DataFrame(scenario_rows)
    if thesis_candidates.empty:
        return gallery_only_outputs

    ranking_rows: list[dict[str, Any]] = []
    for candidate_slug, group in thesis_candidates.groupby("candidate_slug", dropna=False):
        first = group.iloc[0].to_dict()
        current_premium = float(finite_or_none(first.get("current_premium")) or 0.0)
        justified_series = pd.to_numeric(group.get("stock_relative_justified_premium"), errors="coerce").dropna()
        terminal_values = pd.to_numeric(group.get("thesis_terminal_value"), errors="coerce").dropna()
        differences = pd.to_numeric(group.get("difference_vs_stock"), errors="coerce").dropna()
        profit_losses = pd.to_numeric(group.get("profit_loss"), errors="coerce").dropna()
        max_justified = float(justified_series.quantile(0.40)) if not justified_series.empty else 0.0
        premium_gap = max_justified - current_premium
        beats_stock_rate = float((differences > 0).mean()) if not differences.empty else 0.0
        profitable_rate = float((profit_losses > 0).mean()) if not profit_losses.empty else 0.0
        by_path = group.groupby("path_family")["profit_loss"].mean(numeric_only=True) if "profit_loss" in group.columns else pd.Series(dtype=float)
        by_iv = group.groupby("iv_path_name")["profit_loss"].mean(numeric_only=True) if "profit_loss" in group.columns else pd.Series(dtype=float)
        path_sensitivity = float(by_path.max() - by_path.min()) if not by_path.empty else 0.0
        iv_sensitivity = float(by_iv.max() - by_iv.min()) if not by_iv.empty else 0.0
        status = _thesis_status_from_gap(
            premium_gap=premium_gap,
            current_premium=current_premium,
            beats_stock_rate=beats_stock_rate,
        )
        thesis_target_beyond_expiry = bool(first.get("thesis_target_beyond_expiry"))
        if thesis_target_beyond_expiry and status == "reasonable_under_thesis":
            status = "near_watchlist_under_thesis"
        stock_still_better = bool(beats_stock_rate < 0.50 or (float(differences.median()) if not differences.empty else 0.0) <= 0.0)
        ranking_rows.append(
            {
                "candidate_slug": clean_string(candidate_slug),
                "candidate_label": first.get("candidate_label"),
                "candidate_short_label": first.get("candidate_short_label"),
                "expiry_date": first.get("expiry_date"),
                "strike_label": first.get("strike_label"),
                "moneyness_bucket": first.get("moneyness_bucket"),
                "source_trust_label": first.get("source_trust_label"),
                "current_premium": round(current_premium, 4),
                "max_justified_premium": round(max_justified, 4),
                "premium_gap": round(premium_gap, 4),
                "premium_gap_pct": round(float(premium_gap / current_premium), 6) if current_premium else None,
                "entry_attractiveness_status": status,
                "profitable_scenario_rate": round(profitable_rate, 4),
                "beats_stock_scenario_rate": round(beats_stock_rate, 4),
                "terminal_value_median": round(float(terminal_values.median()), 4) if not terminal_values.empty else None,
                "terminal_value_best": round(float(terminal_values.max()), 4) if not terminal_values.empty else None,
                "profit_loss_median": round(float(profit_losses.median()), 4) if not profit_losses.empty else None,
                "difference_vs_stock_median": round(float(differences.median()), 4) if not differences.empty else None,
                "path_sensitivity_range": round(path_sensitivity, 4),
                "iv_sensitivity_range": round(iv_sensitivity, 4),
                "path_sensitivity_label": "path_sensitive" if path_sensitivity > max(current_premium * 0.45, 75.0) else "path_resilient",
                "iv_sensitivity_label": "iv_sensitive" if iv_sensitivity > max(current_premium * 0.35, 50.0) else "iv_secondary",
                "stock_still_better_under_thesis": stock_still_better,
                "target_beyond_expiry": bool(first.get("target_beyond_expiry")),
                "thesis_target_beyond_expiry": thesis_target_beyond_expiry,
                "weak_horizon_fit": bool(first.get("weak_horizon_fit")),
                "main_reason": _thesis_status_note(
                    status,
                    stock_still_better=stock_still_better,
                    iv_sensitive=iv_sensitivity > max(current_premium * 0.35, 50.0),
                    path_sensitive=path_sensitivity > max(current_premium * 0.45, 75.0),
                    thesis_target_beyond_expiry=thesis_target_beyond_expiry,
                ),
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    if not ranking.empty:
        status_order = {"reasonable_under_thesis": 0, "near_watchlist_under_thesis": 1, "too_expensive_under_thesis": 2}
        ranking["_status_order"] = ranking.get("entry_attractiveness_status", pd.Series(dtype=str)).map(status_order).fillna(9)
        ranking = ranking.sort_values(["_status_order", "premium_gap", "beats_stock_scenario_rate"], ascending=[True, False, False])
        ranking["thesis_candidate_rank"] = range(1, len(ranking.index) + 1)
        ranking = ranking.drop(columns=["_status_order"])

    path_summary_rows: list[dict[str, Any]] = []
    for path_name, group in thesis_candidates.groupby("path_family", dropna=False):
        differences = pd.to_numeric(group.get("difference_vs_stock"), errors="coerce").dropna()
        profits = pd.to_numeric(group.get("profit_loss"), errors="coerce").dropna()
        best_row = group.sort_values("profit_loss", ascending=False).iloc[0].to_dict()
        path_summary_rows.append(
            {
                "path_family": clean_string(path_name),
                "path_label": _humanize_named_path(path_name, kind="stock"),
                "target_price": float(thesis_target_price),
                "target_date": thesis_target_date.isoformat(),
                "average_candidate_profit_loss": round(float(profits.mean()), 4) if not profits.empty else None,
                "beat_stock_rate": round(float((differences > 0).mean()), 4) if not differences.empty else 0.0,
                "best_candidate_slug": clean_string(best_row.get("candidate_slug")),
                "best_candidate_short_label": clean_string(best_row.get("candidate_short_label")),
                "path_effect_note": "Fast/early target path helps calls more." if "early" in clean_string(path_name) or "fast" in clean_string(path_name) else "Path timing changes whether premium decay overwhelms the thesis.",
            }
        )
    path_summary = pd.DataFrame(path_summary_rows)

    iv_summary_rows: list[dict[str, Any]] = []
    for iv_name, group in thesis_candidates.groupby("iv_path_name", dropna=False):
        differences = pd.to_numeric(group.get("difference_vs_stock"), errors="coerce").dropna()
        profits = pd.to_numeric(group.get("profit_loss"), errors="coerce").dropna()
        iv_summary_rows.append(
            {
                "iv_path_name": clean_string(iv_name),
                "iv_path_label": _short_iv_path_label(iv_name),
                "average_candidate_profit_loss": round(float(profits.mean()), 4) if not profits.empty else None,
                "beat_stock_rate": round(float((differences > 0).mean()), 4) if not differences.empty else 0.0,
                "iv_effect_note": (
                    "Lower IV makes premium harder to justify."
                    if clean_string(iv_name) in LOWER_IV_PRESETS
                    else "Higher or event-supported IV helps long calls hold value."
                    if clean_string(iv_name) in HIGHER_IV_PRESETS
                    else "Flat IV is the neutral comparison regime."
                ),
            }
        )
    iv_summary = pd.DataFrame(iv_summary_rows)

    required_move_rows: list[dict[str, Any]] = []
    days_to_target = max((thesis_target_date - snapshot_date).days, 0)
    upside_pct = (float(thesis_target_price) / float(entry_spot) - 1.0) if entry_spot else None
    monthly_pace = (float(upside_pct) / max(days_to_target / 30.0, 1.0)) if upside_pct is not None else None
    for row in ranking.to_dict("records"):
        required_move_rows.append(
            {
                "candidate_slug": row.get("candidate_slug"),
                "candidate_short_label": row.get("candidate_short_label"),
                "thesis_target_price": float(thesis_target_price),
                "thesis_target_date": thesis_target_date.isoformat(),
                "days_to_target": int(days_to_target),
                "required_total_upside_pct": round(float(upside_pct), 6) if upside_pct is not None else None,
                "required_monthly_pace_pct": round(float(monthly_pace), 6) if monthly_pace is not None else None,
                "required_timing_window": "fast confirmation preferred" if bool(row.get("stock_still_better_under_thesis")) else "target path can justify inspection",
                "entry_attractiveness_status": row.get("entry_attractiveness_status"),
                "timing_note": "A slow path can still leave stock cleaner if premium decay absorbs the target move.",
            }
        )
    required_move_summary = pd.DataFrame(required_move_rows)

    stock_vs_option_summary = ranking[
        [
            "thesis_candidate_rank",
            "candidate_slug",
            "candidate_short_label",
            "entry_attractiveness_status",
            "beats_stock_scenario_rate",
            "difference_vs_stock_median",
            "stock_still_better_under_thesis",
            "main_reason",
        ]
    ].copy() if not ranking.empty else pd.DataFrame()
    max_justified = ranking[
        [
            "thesis_candidate_rank",
            "candidate_slug",
            "candidate_short_label",
            "current_premium",
            "max_justified_premium",
            "premium_gap",
            "premium_gap_pct",
            "entry_attractiveness_status",
            "main_reason",
        ]
    ].copy() if not ranking.empty else pd.DataFrame()
    current_vs_justified = max_justified.copy()

    markdown = _build_thesis_mode_markdown(
        ticker=ticker,
        thesis_target_price=float(thesis_target_price),
        thesis_target_date=thesis_target_date,
        ranking=ranking,
        path_summary=path_summary,
        iv_summary=iv_summary,
    )
    return {
        "thesis_path_gallery": pd.DataFrame(thesis_path_rows),
        "thesis_iv_gallery": pd.DataFrame(thesis_iv_rows),
        "thesis_mode_candidates": thesis_candidates,
        "thesis_path_family_summary": path_summary,
        "thesis_iv_family_summary": iv_summary,
        "thesis_candidate_ranking": ranking,
        "max_justified_premium": max_justified,
        "current_vs_justified_premium": current_vs_justified,
        "thesis_required_move_summary": required_move_summary,
        "thesis_stock_vs_option_summary": stock_vs_option_summary,
        "thesis_mode_markdown": markdown,
    }


STRESS_SCENARIO_COLUMN_ORDER = [
    "Base",
    "Premium -10%",
    "Premium -20%",
    "Premium +10%",
    "Move delayed 2w",
    "Move delayed 1m",
    "Move delayed 2m",
]


def _stress_target_label(prefix: str, target_price: float, *, suffix: str = "") -> str:
    target_text = f"{float(target_price):.0f}" if abs(float(target_price) - round(float(target_price))) < 0.01 else f"{float(target_price):.2f}"
    return f"{prefix} {target_text}{suffix}"


def _stress_bucket_rank(bucket: object) -> int:
    return {
        "Buy Now": 4,
        "Watchlist": 3,
        "Prefer Stock Instead": 2,
        "Avoid For Now": 1,
    }.get(clean_string(bucket), 0)


def _stress_action_bucket(
    *,
    entry_status: object,
    edge_pct: float,
    premium_gap: float,
    current_premium: float,
    stock_still_better: bool,
) -> str:
    status = clean_string(entry_status)
    premium_base = max(abs(float(current_premium)), 1.0)
    if stock_still_better:
        if edge_pct <= -2.0 or premium_gap < 0:
            return "Prefer Stock Instead"
        return "Watchlist"
    if status == "reasonable_under_thesis" and premium_gap >= 0 and edge_pct >= 2.0:
        return "Buy Now"
    if premium_gap >= -0.10 * premium_base and edge_pct >= -3.0:
        return "Watchlist"
    if edge_pct <= -12.0 or premium_gap <= -0.55 * premium_base:
        return "Avoid For Now"
    return "Watchlist"


def _stress_bucket_from_gap(*, premium_gap: float, current_premium: float, beats_stock_rate: float) -> str:
    status = _thesis_status_from_gap(
        premium_gap=float(premium_gap),
        current_premium=float(current_premium),
        beats_stock_rate=float(beats_stock_rate),
    )
    edge_hint = float(beats_stock_rate) * 20.0 - 8.0
    return _stress_action_bucket(
        entry_status=status,
        edge_pct=edge_hint,
        premium_gap=float(premium_gap),
        current_premium=float(current_premium),
        stock_still_better=float(beats_stock_rate) < 0.50,
    )


def _stress_transition_label(base_bucket: str, scenario_bucket: str) -> str:
    delta = _stress_bucket_rank(scenario_bucket) - _stress_bucket_rank(base_bucket)
    if delta >= 2:
        return "material_upgrade"
    if delta == 1:
        return "upgrade"
    if delta == 0:
        return "unchanged"
    if delta == -1:
        return "weaker"
    return "breaks"


def _format_edge_pct(value: object) -> str:
    edge = finite_or_none(value)
    if edge is None:
        return "n/a"
    return f"{edge:+.0f}%"


def _format_gap(value: object) -> str:
    gap = finite_or_none(value)
    if gap is None:
        return "n/a"
    return f"{gap:+.0f}"


def _build_stress_tests_markdown(
    *,
    ticker: str,
    thesis_target_price: float,
    thesis_target_date: date,
    transition_summary: pd.DataFrame,
    premium_summary: pd.DataFrame,
    timing_summary: pd.DataFrame,
    target_summary: pd.DataFrame,
) -> str:
    def candidate_lines(frame: pd.DataFrame, *, value_column: str, limit: int = 4) -> list[str]:
        if frame.empty:
            return ["- No stress rows were available."]
        data = frame.copy()
        if value_column in data.columns:
            data[value_column] = pd.to_numeric(data[value_column], errors="coerce").fillna(0.0)
            data = data.sort_values(value_column, ascending=False)
        output: list[str] = []
        for row in data.head(limit).to_dict("records"):
            label = clean_string(row.get("candidate_short_label")) or clean_string(row.get("candidate_label"))
            bucket = clean_string(row.get("base_action_bucket") or row.get("action_bucket"))
            best = clean_string(row.get("best_improving_stress") or row.get("scenario_label"))
            worst = clean_string(row.get("worst_breaking_stress") or row.get("main_note"))
            output.append(f"- `{label}` - base `{bucket}`; best stress: {best or 'n/a'}; main fragility: {worst or 'n/a'}.")
        return output or ["- No stress rows were available."]

    premium_lines: list[str] = []
    if not premium_summary.empty:
        for label, group in premium_summary.groupby("candidate_short_label", dropna=False):
            ordered = group.sort_values("premium_multiplier")
            minus20 = ordered.loc[ordered.get("scenario_name").astype(str).eq("premium_minus_20")]
            plus10 = ordered.loc[ordered.get("scenario_name").astype(str).eq("premium_plus_10")]
            cheaper_bucket = clean_string(minus20.iloc[0].get("action_bucket")) if not minus20.empty else "n/a"
            expensive_bucket = clean_string(plus10.iloc[0].get("action_bucket")) if not plus10.empty else "n/a"
            premium_lines.append(f"- `{clean_string(label)}`: -20% premium -> `{cheaper_bucket}`; +10% premium -> `{expensive_bucket}`.")
            if len(premium_lines) >= 4:
                break
    if not premium_lines:
        premium_lines = ["- No premium sensitivity rows were available."]

    timing_lines: list[str] = []
    if not timing_summary.empty:
        delayed = timing_summary.loc[pd.to_numeric(timing_summary.get("delay_days"), errors="coerce").fillna(0).gt(0)].copy()
        for label, group in delayed.groupby("candidate_short_label", dropna=False):
            worst = group.sort_values("option_vs_stock_edge_pct").head(1)
            if worst.empty:
                continue
            row = worst.iloc[0]
            timing_lines.append(
                f"- `{clean_string(label)}`: weakest delay is {clean_string(row.get('scenario_label'))}, bucket `{clean_string(row.get('action_bucket'))}` ({_format_edge_pct(row.get('option_vs_stock_edge_pct'))} vs stock)."
            )
            if len(timing_lines) >= 4:
                break
    if not timing_lines:
        timing_lines = ["- No timing slip rows were available."]

    target_lines: list[str] = []
    if not target_summary.empty:
        for label, group in target_summary.groupby("candidate_short_label", dropna=False):
            base = group.loc[group.get("scenario_name").astype(str).eq("base_target")]
            overshoot_rows = group.loc[group.get("scenario_name").astype(str).eq("overshoot_settle")]
            undershoot_rows = group.loc[group.get("scenario_name").astype(str).eq("undershoot")]
            if base.empty:
                continue
            base_row = base.iloc[0]
            overshoot_row = overshoot_rows.iloc[0] if not overshoot_rows.empty else None
            undershoot_row = undershoot_rows.iloc[0] if not undershoot_rows.empty else None
            base_bucket = clean_string(base_row.get("action_bucket"))
            base_edge = finite_or_none(base_row.get("option_vs_stock_edge_pct")) or 0.0
            overshoot_edge = finite_or_none(overshoot_row.get("option_vs_stock_edge_pct")) if overshoot_row is not None else None
            undershoot_edge = finite_or_none(undershoot_row.get("option_vs_stock_edge_pct")) if undershoot_row is not None else None
            overshoot_bucket = clean_string(overshoot_row.get("action_bucket")) if overshoot_row is not None else ""
            note = clean_string(base_row.get("main_note"))
            if "Target is beyond this expiry" in note:
                read = "target date is beyond expiry, so this is only an early-confirmation idea."
            elif overshoot_edge is not None and (
                _stress_bucket_rank(overshoot_bucket) > _stress_bucket_rank(base_bucket)
                or overshoot_edge >= base_edge + 3.0
            ):
                read = f"overshoot helps ({_format_edge_pct(overshoot_edge)} vs stock)."
            elif base_bucket == "Prefer Stock Instead":
                read = "even the stated target still leaves stock cleaner."
            elif (
                overshoot_edge is not None
                and undershoot_edge is not None
                and abs(float(overshoot_edge) - float(base_edge)) < 1.0
                and abs(float(undershoot_edge) - float(base_edge)) < 1.0
            ):
                read = "target level is not the main blocker; entry price and timing matter more."
            else:
                read = "base target is the main read; use the table for undershoot/overshoot detail."
            target_lines.append(f"- `{clean_string(label)}`: base target -> `{base_bucket}` ({_format_edge_pct(base_edge)} vs stock); {read}")
            if len(target_lines) >= 4:
                break
    if not target_lines:
        target_lines = ["- No target stress rows were available."]

    return "\n".join(
        [
            f"# {ticker} Stress Tests",
            "",
            "This is a thesis-relative stress layer for the top bullish long-call candidates. It is not a probability engine.",
            "",
            "## Stress Snapshot",
            "",
            f"- Thesis target: `${float(thesis_target_price):,.2f}` by `{thesis_target_date.isoformat()}`",
            "- Tests: premium sensitivity, timing slip, and target undershoot/overshoot",
            "- Headline metric: option edge versus stock, expressed as a percent of comparison capital",
            "",
            "## Which Candidates Are Price-Sensitive?",
            "",
            *premium_lines,
            "",
            "## What Breaks If The Move Arrives Later?",
            "",
            *timing_lines,
            "",
            "## Do Calls Need The Thesis To Overshoot?",
            "",
            *target_lines,
            "",
            "## Candidate Stress Cards",
            "",
            *candidate_lines(transition_summary, value_column="stress_resilience_score"),
            "",
            "## How To Read This Layer",
            "",
            "- `candidate_stress_grid.csv` is the compact cross-check: each candidate gets action bucket, stock-relative edge, premium gap, and note across all stress columns.",
            "- `premium_sensitivity_summary.csv` answers whether the contract is structurally weak or just too expensive now.",
            "- `timing_slip_summary.csv` shows whether theta/timing turns the same thesis into Watchlist or Avoid.",
            "- `target_stress_summary.csv` shows whether the call needs the stock to overshoot the stated thesis before it really works.",
        ]
    )


def _build_stress_test_outputs(
    *,
    ticker: str,
    thesis_target_price: float,
    thesis_target_date: date,
    snapshot_date: date,
    entry_spot: float,
    comparison_capital: float,
    thesis_candidate_ranking: pd.DataFrame,
    thesis_stock_vs_option_summary: pd.DataFrame,
    top_candidate_cards: pd.DataFrame,
) -> dict[str, Any]:
    empty = {
        "candidate_stress_grid": pd.DataFrame(),
        "premium_sensitivity_summary": pd.DataFrame(),
        "timing_slip_summary": pd.DataFrame(),
        "target_stress_summary": pd.DataFrame(),
        "stress_transition_summary": pd.DataFrame(),
        "stress_tests_markdown": _build_stress_tests_markdown(
            ticker=ticker,
            thesis_target_price=float(thesis_target_price),
            thesis_target_date=thesis_target_date,
            transition_summary=pd.DataFrame(),
            premium_summary=pd.DataFrame(),
            timing_summary=pd.DataFrame(),
            target_summary=pd.DataFrame(),
        ),
    }
    ranking = thesis_candidate_ranking.copy()
    if ranking.empty:
        return empty
    ranking["thesis_candidate_rank"] = pd.to_numeric(ranking.get("thesis_candidate_rank"), errors="coerce").fillna(999)
    ranking = ranking.sort_values("thesis_candidate_rank").head(5).reset_index(drop=True)
    stock_lookup = {}
    if not thesis_stock_vs_option_summary.empty and "candidate_slug" in thesis_stock_vs_option_summary.columns:
        stock_lookup = {
            clean_string(row.get("candidate_slug")): row
            for row in thesis_stock_vs_option_summary.to_dict("records")
        }
    card_lookup = {}
    if not top_candidate_cards.empty:
        card_lookup = {
            clean_string(row.get("candidate_slug") or row.get("candidate_label") or row.get("contract_label")): row
            for row in top_candidate_cards.to_dict("records")
        }

    target = float(thesis_target_price)
    undershoot = round(target * (26.0 / 30.0), 2)
    overshoot = round(target * (35.0 / 30.0), 2)
    base_target_label = _stress_target_label("Base hit at", target)
    undershoot_label = _stress_target_label("Undershoot to", undershoot)
    overshoot_label = _stress_target_label("Overshoot to", overshoot, suffix=" then settle")
    scenario_columns = [
        *STRESS_SCENARIO_COLUMN_ORDER,
        undershoot_label,
        base_target_label,
        overshoot_label,
    ]
    grid_rows: list[dict[str, Any]] = []
    premium_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []

    def row_number(value: object, default: float = 0.0) -> float:
        direct = finite_or_none(value)
        if direct is not None:
            return float(direct)
        parsed = pd.to_numeric(value, errors="coerce")
        parsed_float = finite_or_none(parsed)
        return float(parsed_float if parsed_float is not None else default)

    def base_fields(row: dict[str, Any]) -> dict[str, Any]:
        slug = clean_string(row.get("candidate_slug"))
        stock_row = stock_lookup.get(slug, {})
        card = card_lookup.get(slug, {})
        current_premium = row_number(row.get("current_premium"), 0.0)
        premium_gap = row_number(row.get("premium_gap"), 0.0)
        max_justified = row_number(row.get("max_justified_premium"), current_premium + premium_gap)
        diff_vs_stock = row_number(row.get("difference_vs_stock_median") if row.get("difference_vs_stock_median") is not None else stock_row.get("difference_vs_stock_median"), 0.0)
        edge_pct = (diff_vs_stock / max(float(comparison_capital), 1.0)) * 100.0
        beats_stock_rate = row_number(row.get("beats_stock_scenario_rate") if row.get("beats_stock_scenario_rate") is not None else stock_row.get("beats_stock_scenario_rate"), 0.0)
        weak_horizon_fit = bool(
            row.get("weak_horizon_fit")
            or row.get("target_beyond_expiry")
            or row.get("thesis_target_beyond_expiry")
        )
        stock_still_better = bool(
            row.get("stock_still_better_under_thesis")
            or stock_row.get("stock_still_better_under_thesis")
            or edge_pct <= 0.0
            or weak_horizon_fit
        )
        expiry_dt = parse_date(row.get("expiry_date"))
        days_to_expiry = max((expiry_dt - snapshot_date).days, 1) if expiry_dt else max((thesis_target_date - snapshot_date).days, 1)
        strike = row_number(row.get("strike_label"), 0.0)
        base_bucket = _stress_action_bucket(
            entry_status=row.get("entry_attractiveness_status"),
            edge_pct=edge_pct,
            premium_gap=premium_gap,
            current_premium=current_premium,
            stock_still_better=stock_still_better,
        )
        label = clean_string(row.get("candidate_short_label")) or clean_string(row.get("candidate_label"))
        return {
            "candidate_slug": slug,
            "candidate_label": clean_string(row.get("candidate_label")),
            "candidate_short_label": label,
            "expiry_date": clean_string(row.get("expiry_date")),
            "strike_label": clean_string(row.get("strike_label")),
            "moneyness_bucket": clean_string(row.get("moneyness_bucket")),
            "source_trust_label": clean_string(row.get("source_trust_label")),
            "current_premium": current_premium,
            "max_justified_premium": max_justified,
            "premium_gap": premium_gap,
            "base_difference_vs_stock": diff_vs_stock,
            "base_edge_pct": edge_pct,
            "beats_stock_scenario_rate": beats_stock_rate,
            "stock_still_better_under_thesis": stock_still_better,
            "weak_horizon_fit": weak_horizon_fit,
            "days_to_expiry": days_to_expiry,
            "strike": strike,
            "base_action_bucket": base_bucket,
            "base_note": clean_string(row.get("main_reason")) or clean_string(card.get("compare_vs_stock_note")) or "Use the stress columns to see whether price, timing, or target level is the real blocker.",
            "upgrade_rule": clean_string(card.get("upgrade_rule")),
            "main_warning": clean_string(card.get("what_hurts_it")) or clean_string(row.get("main_reason")),
        }

    def record(
        bucket: list[dict[str, Any]],
        *,
        fields: dict[str, Any],
        stress_family: str,
        scenario_name: str,
        scenario_label: str,
        scenario_order: int,
        action_bucket: str,
        edge_pct: float,
        premium_gap: float,
        main_note: str,
        **extra: Any,
    ) -> dict[str, Any]:
        payload = {
            "candidate_slug": fields["candidate_slug"],
            "candidate_label": fields["candidate_label"],
            "candidate_short_label": fields["candidate_short_label"],
            "expiry_date": fields["expiry_date"],
            "strike_label": fields["strike_label"],
            "moneyness_bucket": fields["moneyness_bucket"],
            "source_trust_label": fields["source_trust_label"],
            "stress_family": stress_family,
            "scenario_name": scenario_name,
            "scenario_label": scenario_label,
            "scenario_order": scenario_order,
            "base_action_bucket": fields["base_action_bucket"],
            "action_bucket": action_bucket,
            "bucket_transition": _stress_transition_label(fields["base_action_bucket"], action_bucket),
            "option_vs_stock_edge_pct": round(float(edge_pct), 4),
            "max_justified_premium_gap": round(float(premium_gap), 4),
            "main_note": clean_string(main_note),
            "upgrade_rule": fields["upgrade_rule"],
            "main_warning": fields["main_warning"],
        }
        payload.update(extra)
        bucket.append(payload)
        return payload

    for rank, row in enumerate(ranking.to_dict("records"), start=1):
        fields = base_fields(row)
        scenario_payloads: dict[str, dict[str, Any]] = {}

        premium_scenarios = [
            ("base", "Base", 1.00, "Base thesis read."),
            ("premium_minus_10", "Premium -10%", 0.90, "Lower entry helps; watch whether the bucket upgrades."),
            ("premium_minus_20", "Premium -20%", 0.80, "This tests whether the call is merely too expensive today."),
            ("premium_plus_10", "Premium +10%", 1.10, "Higher entry tests how quickly premium destroys the edge."),
        ]
        for order, (name, label, multiplier, note) in enumerate(premium_scenarios, start=1):
            scenario_premium = fields["current_premium"] * multiplier
            premium_delta = fields["current_premium"] - scenario_premium
            gap = fields["max_justified_premium"] - scenario_premium
            edge_pct = fields["base_edge_pct"] + (premium_delta / max(float(comparison_capital), 1.0)) * 100.0
            status = _thesis_status_from_gap(
                premium_gap=gap,
                current_premium=scenario_premium,
                beats_stock_rate=fields["beats_stock_scenario_rate"],
            )
            bucket = _stress_action_bucket(
                entry_status=status,
                edge_pct=edge_pct,
                premium_gap=gap,
                current_premium=scenario_premium,
                stock_still_better=fields["stock_still_better_under_thesis"],
            )
            if name == "premium_minus_20" and _stress_bucket_rank(bucket) <= _stress_bucket_rank(fields["base_action_bucket"]):
                note = "Even a materially cheaper entry does not fully solve the thesis-vs-stock pressure."
            elif name.startswith("premium_minus") and _stress_bucket_rank(bucket) > _stress_bucket_rank(fields["base_action_bucket"]):
                note = "Entry price is the key blocker; lower premium can upgrade the setup."
            payload = record(
                premium_rows,
                fields=fields,
                stress_family="premium_sensitivity",
                scenario_name=name,
                scenario_label=label,
                scenario_order=order,
                action_bucket=bucket,
                edge_pct=edge_pct,
                premium_gap=gap,
                main_note=note,
                premium_multiplier=multiplier,
                scenario_premium=round(scenario_premium, 4),
            )
            scenario_payloads[label] = payload

        timing_scenarios = [
            ("base_timing", "Base", 0, "Base thesis timing."),
            ("delay_2w", "Move delayed 2w", 14, "Two-week delay: checks early theta sensitivity."),
            ("delay_1m", "Move delayed 1m", 30, "One-month delay: a cleaner test of timing fragility."),
            ("delay_2m", "Move delayed 2m", 60, "Two-month delay: exposes candidates that require the move to start early."),
        ]
        for order, (name, label, delay_days, note) in enumerate(timing_scenarios, start=1):
            delay_ratio = max(float(delay_days), 0.0) / max(float(fields["days_to_expiry"]), 1.0)
            theta_penalty = fields["current_premium"] * min(0.72, delay_ratio * (1.35 if fields["days_to_expiry"] < 240 else 0.95))
            gap = fields["premium_gap"] - theta_penalty
            edge_pct = fields["base_edge_pct"] - (theta_penalty / max(float(comparison_capital), 1.0)) * 100.0
            delayed_target_date = thesis_target_date + timedelta(days=int(delay_days))
            target_beyond_expiry = bool(parse_date(fields["expiry_date"]) is not None and delayed_target_date > parse_date(fields["expiry_date"]))
            if target_beyond_expiry:
                gap -= fields["current_premium"] * 0.35
                edge_pct -= 8.0
                note = "Timing slip pushes the thesis too close to or beyond expiry."
            status = _thesis_status_from_gap(
                premium_gap=gap,
                current_premium=fields["current_premium"],
                beats_stock_rate=max(0.0, fields["beats_stock_scenario_rate"] - delay_ratio * 0.65),
            )
            bucket = _stress_action_bucket(
                entry_status=status,
                edge_pct=edge_pct,
                premium_gap=gap,
                current_premium=fields["current_premium"],
                stock_still_better=fields["stock_still_better_under_thesis"] or target_beyond_expiry,
            )
            if delay_days and _stress_bucket_rank(bucket) < _stress_bucket_rank(fields["base_action_bucket"]):
                note = "Delayed move weakens the call; theta/timing is a real blocker."
            payload = record(
                timing_rows,
                fields=fields,
                stress_family="timing_slip",
                scenario_name=name,
                scenario_label=label,
                scenario_order=order,
                action_bucket=bucket,
                edge_pct=edge_pct,
                premium_gap=gap,
                main_note=note,
                delay_days=delay_days,
                delayed_target_date=delayed_target_date.isoformat(),
                target_beyond_expiry_under_delay=target_beyond_expiry,
            )
            # The compact grid has one canonical "Base" column. Timing charts still
            # keep their own base row, but the grid should not let that duplicate
            # label overwrite the active thesis base read.
            scenario_payloads.setdefault(label, payload)

        target_scenarios = [
            ("undershoot", undershoot_label, undershoot, "Undershoot: tests whether the option needs more than the stated thesis."),
            ("base_target", base_target_label, target, "Base target hit: checks whether the actual thesis is enough versus stock."),
            ("overshoot_settle", overshoot_label, overshoot, "Overshoot: tests whether convexity needs a stronger-than-stated move."),
        ]
        base_intrinsic = max(target - float(fields["strike"]), 0.0) * 100.0 if fields["strike"] else 0.0
        for order, (name, label, scenario_target, note) in enumerate(target_scenarios, start=1):
            scenario_intrinsic = max(float(scenario_target) - float(fields["strike"]), 0.0) * 100.0 if fields["strike"] else 0.0
            stock_delta = (float(scenario_target) - target) * 100.0
            option_delta = scenario_intrinsic - base_intrinsic
            edge_adjustment = option_delta - stock_delta
            gap = fields["premium_gap"] + edge_adjustment * 0.40
            edge_pct = fields["base_edge_pct"] + (edge_adjustment / max(float(comparison_capital), 1.0)) * 100.0
            status = _thesis_status_from_gap(
                premium_gap=gap,
                current_premium=fields["current_premium"],
                beats_stock_rate=min(1.0, max(0.0, fields["beats_stock_scenario_rate"] + edge_adjustment / max(float(comparison_capital) * 2.0, 1.0))),
            )
            bucket = _stress_action_bucket(
                entry_status=status,
                edge_pct=edge_pct,
                premium_gap=gap,
                current_premium=fields["current_premium"],
                stock_still_better=bool(fields["weak_horizon_fit"]) or edge_pct <= 0.0,
            )
            if name == "undershoot":
                note = "Undershoot leaves less intrinsic value; fragile calls should break here."
            elif name == "base_target" and _stress_bucket_rank(bucket) <= 2:
                note = "Even hitting the stated target may leave stock cleaner after premium."
            elif name == "overshoot_settle" and _stress_bucket_rank(bucket) > _stress_bucket_rank(fields["base_action_bucket"]):
                note = "This call needs overshoot/stronger convexity to become more compelling."
            if fields["weak_horizon_fit"]:
                note = "Target is beyond this expiry; treat as early-confirmation only, not a clean thesis hold."
            payload = record(
                target_rows,
                fields=fields,
                stress_family="target_stress",
                scenario_name=name,
                scenario_label=label,
                scenario_order=order,
                action_bucket=bucket,
                edge_pct=edge_pct,
                premium_gap=gap,
                main_note=note,
                target_price=scenario_target,
                intrinsic_value_at_target=round(scenario_intrinsic, 4),
            )
            scenario_payloads[label] = payload

        all_payloads = list(scenario_payloads.values())
        best_payload = max(all_payloads, key=lambda item: finite_or_none(item.get("option_vs_stock_edge_pct")) or -9999)
        worst_payload = min(all_payloads, key=lambda item: finite_or_none(item.get("option_vs_stock_edge_pct")) or 9999)
        resilient_count = sum(1 for item in all_payloads if clean_string(item.get("action_bucket")) in {"Buy Now", "Watchlist"})
        buy_count = sum(1 for item in all_payloads if clean_string(item.get("action_bucket")) == "Buy Now")
        transition_rows.append(
            {
                "stress_rank": rank,
                "candidate_slug": fields["candidate_slug"],
                "candidate_label": fields["candidate_label"],
                "candidate_short_label": fields["candidate_short_label"],
                "expiry_date": fields["expiry_date"],
                "strike_label": fields["strike_label"],
                "source_trust_label": fields["source_trust_label"],
                "base_action_bucket": fields["base_action_bucket"],
                "base_option_vs_stock_edge_pct": round(float(fields["base_edge_pct"]), 4),
                "base_max_justified_premium_gap": round(float(fields["premium_gap"]), 4),
                "best_improving_stress": clean_string(best_payload.get("scenario_label")),
                "best_improving_bucket": clean_string(best_payload.get("action_bucket")),
                "best_improving_edge_pct": best_payload.get("option_vs_stock_edge_pct"),
                "worst_breaking_stress": clean_string(worst_payload.get("scenario_label")),
                "worst_breaking_bucket": clean_string(worst_payload.get("action_bucket")),
                "worst_breaking_edge_pct": worst_payload.get("option_vs_stock_edge_pct"),
                "stress_resilience_score": round(resilient_count / max(len(all_payloads), 1), 4),
                "stress_buy_count": buy_count,
                "premium_sensitivity_read": "entry_price_can_upgrade" if any(clean_string(item.get("scenario_name")) == "premium_minus_20" and _stress_bucket_rank(item.get("action_bucket")) > _stress_bucket_rank(fields["base_action_bucket"]) for item in all_payloads) else "premium_discount_not_enough",
                "timing_sensitivity_read": "breaks_if_delayed" if any(clean_string(item.get("stress_family")) == "timing_slip" and _stress_bucket_rank(item.get("action_bucket")) < _stress_bucket_rank(fields["base_action_bucket"]) for item in all_payloads) else "timing_more_forgiving",
                "target_dependency_read": "needs_overshoot" if _stress_bucket_rank(scenario_payloads[overshoot_label]["action_bucket"]) > _stress_bucket_rank(scenario_payloads[base_target_label]["action_bucket"]) else "base_target_enough_or_stock_still_cleaner",
                "stress_card_note": (
                    "Looks most interesting if entry premium cools."
                    if clean_string(best_payload.get("scenario_name")).startswith("premium_minus")
                    else "Needs the thesis to arrive on time or stronger than stated."
                    if clean_string(worst_payload.get("stress_family")) in {"timing_slip", "target_stress"}
                    else "Use path packs for the exact scenario mechanics."
                ),
                "main_warning": fields["main_warning"],
                "upgrade_rule": fields["upgrade_rule"],
            }
        )

        for metric, formatter, source_key in [
            ("action bucket", str, "action_bucket"),
            ("option vs stock edge", _format_edge_pct, "option_vs_stock_edge_pct"),
            ("max justified premium gap", _format_gap, "max_justified_premium_gap"),
            ("main note", clean_string, "main_note"),
        ]:
            grid_row = {
                "candidate_rank": rank,
                "candidate_short_label": fields["candidate_short_label"],
                "candidate_slug": fields["candidate_slug"],
                "metric": metric,
            }
            for column in scenario_columns:
                payload = scenario_payloads.get(column)
                grid_row[column] = formatter(payload.get(source_key)) if payload else ""
            grid_rows.append(grid_row)

    premium_summary = pd.DataFrame(premium_rows)
    timing_summary = pd.DataFrame(timing_rows)
    target_summary = pd.DataFrame(target_rows)
    transition_summary = pd.DataFrame(transition_rows)
    stress_grid = pd.DataFrame(grid_rows)
    markdown = _build_stress_tests_markdown(
        ticker=ticker,
        thesis_target_price=float(thesis_target_price),
        thesis_target_date=thesis_target_date,
        transition_summary=transition_summary,
        premium_summary=premium_summary,
        timing_summary=timing_summary,
        target_summary=target_summary,
    )
    return {
        "candidate_stress_grid": stress_grid,
        "premium_sensitivity_summary": premium_summary,
        "timing_slip_summary": timing_summary,
        "target_stress_summary": target_summary,
        "stress_transition_summary": transition_summary,
        "stress_tests_markdown": markdown,
    }


def _single_option_empty_outputs() -> dict[str, Any]:
    return {
        "single_option_decision_summary": pd.DataFrame(),
        "single_option_decision_path_selections": pd.DataFrame(),
        "single_option_representative_paths": pd.DataFrame(),
        "single_option_path_outcomes": pd.DataFrame(),
        "single_option_required_path_to_beat_stock_1_5x": pd.DataFrame(),
        "single_option_required_path_to_beat_stock_2_0x": pd.DataFrame(),
        "single_option_closest_representative_path_to_edge": pd.DataFrame(),
        "single_option_edge_gap_by_path_family": pd.DataFrame(),
        "single_option_path_family_counts": pd.DataFrame(),
        "single_option_timing_sensitivity": pd.DataFrame(),
        "single_option_iv_sensitivity": pd.DataFrame(),
        "single_option_entry_sensitivity": pd.DataFrame(),
        "single_option_summary_bullets": pd.DataFrame(),
        "single_option_decision_markdown": "",
    }


def _single_option_num(value: object, default: float = 0.0) -> float:
    parsed = finite_or_none(value)
    return float(parsed) if parsed is not None else float(default)


def _single_option_entry_premium(position: StrategyPosition, *, mode: str) -> tuple[float, str]:
    base = float(position.initial_outlay)
    normalized = clean_string(mode).lower() or "conservative_mid_plus_slippage"
    if normalized == "mid":
        return base, "mid"
    if normalized == "ask_or_mid":
        return base * 1.02, "ask_or_mid_fallback"
    return base * 1.03, "conservative_mid_plus_slippage"


def _single_option_adjusted_evaluation(
    spec: dict[str, Any],
    *,
    spot_price: float,
    horizon_days: int,
    iv_shift_points: float,
    comparison_capital: float,
    premium_used: float,
) -> dict[str, Any]:
    raw = _evaluate_at_point(
        spec,
        spot_price=float(spot_price),
        horizon_days=int(horizon_days),
        iv_shift_points=float(iv_shift_points),
        comparison_capital=float(comparison_capital),
    )
    estimated_value = _single_option_num(raw.get("estimated_value"), 0.0)
    stock_profit_loss = _single_option_num(raw.get("stock_profit_loss"), 0.0)
    adjusted_profit = estimated_value - float(premium_used)
    adjusted_return = adjusted_profit / float(comparison_capital) if comparison_capital else None
    raw.update(
        {
            "premium_used": round(float(premium_used), 4),
            "profit_loss": round(float(adjusted_profit), 4),
            "return_on_comparison_capital": finite_or_none(adjusted_return),
            "difference_vs_stock": round(float(adjusted_profit - stock_profit_loss), 4),
            "comparison_profit_loss": round(float(adjusted_profit), 4),
            "modeled_profit_loss_before_entry_adjustment": raw.get("profit_loss"),
        }
    )
    return raw


def _single_option_outperformance_multiple(profit_loss: float, stock_profit_loss: float) -> float | None:
    if stock_profit_loss <= 0:
        return None
    return float(profit_loss) / float(stock_profit_loss)


def _single_option_outcome_label(
    *,
    profit_loss: float,
    stock_profit_loss: float,
    difference_vs_stock: float,
    outperformance_multiple: float | None,
    strong_outperformance_multiple: float,
    minimum_outperformance_multiple: float,
    clamped_to_expiry: bool,
    requested_days: int,
    effective_days: int,
) -> str:
    if clamped_to_expiry and int(requested_days) > int(effective_days):
        return "fail_too_narrow_or_expiry_issue"
    if profit_loss <= 0:
        return "fail_too_narrow_or_expiry_issue"
    if difference_vs_stock <= 0:
        return "stock_better"
    if outperformance_multiple is not None and outperformance_multiple >= float(strong_outperformance_multiple):
        return "clear_option_win"
    if outperformance_multiple is None and difference_vs_stock > max(abs(stock_profit_loss) * 0.25, 25.0):
        return "wins_but_not_enough"
    if outperformance_multiple is not None and outperformance_multiple >= float(minimum_outperformance_multiple):
        return "wins_but_not_enough"
    return "wins_but_not_enough"


def _single_option_path_status_note(label: str) -> str:
    notes = {
        "clear_option_win": "Option clearly beats stock on this path.",
        "wins_but_not_enough": "Option beats stock, but not by enough to be a clean buy signal.",
        "stock_better": "Stock is cleaner on this path after option premium.",
        "fail_too_narrow_or_expiry_issue": "The path is too narrow, late, or expiry-constrained.",
    }
    return notes.get(clean_string(label), "Outcome depends on path timing and premium.")


def _single_option_candidate_short_label(candidate: dict[str, Any]) -> str:
    label = clean_string(candidate.get("candidate_short_label"))
    if label:
        return label
    return _format_long_call_series_label(
        strike_label=candidate.get("strike_label"),
        expiry_date=candidate.get("expiry_date"),
        moneyness_bucket=candidate.get("moneyness_bucket"),
        include_bucket=False,
    )


def _select_single_option_candidate(
    *,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    bullish_action_board: pd.DataFrame,
    candidate_slug: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    spec_lookup = {
        clean_string(spec.get("candidate_slug")): spec
        for spec in specs
        if clean_string(spec.get("strategy_family")) == "long_call"
    }
    if not spec_lookup or candidate_rows.empty:
        return None, None
    long_calls = candidate_rows.loc[candidate_rows.get("strategy_family", pd.Series(dtype=str)).astype(str).eq("long_call")].copy()
    if long_calls.empty:
        return None, None
    requested_slug = clean_string(candidate_slug)
    if requested_slug:
        match = long_calls.loc[long_calls.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(requested_slug)]
        if match.empty or requested_slug not in spec_lookup:
            raise ValueError(f"single_option_candidate_slug did not match an available long_call candidate: {requested_slug}")
        return match.iloc[0].to_dict(), spec_lookup[requested_slug]
    if bullish_action_board is not None and not bullish_action_board.empty:
        for row in bullish_action_board.to_dict("records"):
            slug = clean_string(row.get("candidate_slug"))
            if slug in spec_lookup:
                match = long_calls.loc[long_calls.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(slug)]
                candidate = match.iloc[0].to_dict() if not match.empty else dict(row)
                return candidate, spec_lookup[slug]
    sorted_calls = _sort_candidate_priority(long_calls)
    for row in sorted_calls.to_dict("records"):
        slug = clean_string(row.get("candidate_slug"))
        if slug in spec_lookup:
            return row, spec_lookup[slug]
    return None, None


def _single_option_required_path_points(
    required_stock_path_to_buy: pd.DataFrame,
    *,
    candidate_slug: str,
    snapshot_date: date,
    target_date: date,
    entry_spot: float,
) -> list[dict[str, Any]]:
    if required_stock_path_to_buy.empty:
        return []
    data = required_stock_path_to_buy.loc[
        required_stock_path_to_buy.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(clean_string(candidate_slug))
        & required_stock_path_to_buy.get("series_kind", pd.Series(dtype=str)).astype(str).eq("required_path")
    ].copy()
    if data.empty:
        return []
    data["requested_days"] = pd.to_numeric(data.get("requested_days"), errors="coerce").fillna(0).astype(int)
    data = data.sort_values("requested_days")
    rows: list[dict[str, Any]] = []
    for step_index, row in enumerate(data.to_dict("records")):
        stock_price = finite_or_none(row.get("stock_price"))
        if stock_price is None:
            continue
        rows.append(
            {
                "date": clean_string(row.get("date")) or (snapshot_date + timedelta(days=int(row.get("requested_days") or 0))).isoformat(),
                "requested_days": int(row.get("requested_days") or 0),
                "step_index": step_index,
                "spot_price": float(stock_price),
                "return_pct": (float(stock_price) / float(entry_spot) - 1.0) if entry_spot else None,
            }
        )
    if not rows:
        return []
    if rows[0]["requested_days"] > 0:
        rows.insert(
            0,
            {
                "date": snapshot_date.isoformat(),
                "requested_days": 0,
                "step_index": 0,
                "spot_price": float(entry_spot),
                "return_pct": 0.0,
            },
        )
    terminal_days = max((target_date - snapshot_date).days, 0)
    if rows[-1]["requested_days"] < terminal_days:
        rows.append(
            {
                "date": target_date.isoformat(),
                "requested_days": terminal_days,
                "step_index": len(rows),
                "spot_price": float(rows[-1]["spot_price"]),
                "return_pct": rows[-1].get("return_pct"),
            }
        )
    return rows


def _decision_path_payload(
    *,
    path_role: str,
    path_name: str,
    path_points: list[dict[str, Any]],
    selection_reason: str,
) -> dict[str, Any]:
    meta = _stock_path_family_metadata(path_name)
    role = clean_string(path_role) or clean_string(path_name)
    normalized_name = clean_string(path_name).lower()
    decision_path_id = role if role == "minimum_required_path" else f"{role}__{normalized_name}"
    return {
        "decision_path_id": decision_path_id,
        "path_role": role,
        "path_name": normalized_name,
        "path_label": meta["path_label"],
        "path_family": meta["path_family"],
        "path_family_label": meta["path_family_label"],
        "timing_shape": meta["timing_shape"],
        "outcome_bias": meta["outcome_bias"],
        "path_description": meta["path_description"],
        "selection_reason": clean_string(selection_reason),
        "path_points": path_points,
    }


def _build_decision_comparison_path_pool(
    *,
    required_stock_path_to_buy: pd.DataFrame,
    anchor_candidate_slug: str,
    snapshot_date: date,
    target_date: date,
    target_price: float,
    target_horizon_label: str,
    entry_spot: float,
) -> list[dict[str, Any]]:
    path_grid = _build_path_grid(snapshot_date, target_date)
    path_pool: list[dict[str, Any]] = []
    required_points = _single_option_required_path_points(
        required_stock_path_to_buy,
        candidate_slug=anchor_candidate_slug,
        snapshot_date=snapshot_date,
        target_date=target_date,
        entry_spot=float(entry_spot),
    )
    if required_points:
        path_pool.append(
            _decision_path_payload(
                path_role="minimum_required_path",
                path_name="minimum_required_path",
                path_points=required_points,
                selection_reason="Contract-specific required path included as the threshold reference.",
            )
        )
    used_path_names = {"minimum_required_path"} if required_points else set()
    for role, preset_candidates in SINGLE_OPTION_REPRESENTATIVE_PATH_ROLES:
        for preset in preset_candidates:
            preset_name = clean_string(preset).lower()
            if preset_name in used_path_names:
                continue
            if preset_name in THESIS_STOCK_PATH_PRESETS:
                chosen_points = _default_stock_path_points(
                    preset=preset_name,
                    entry_spot=float(entry_spot),
                    target_price=float(target_price),
                    target_horizon_label=target_horizon_label,
                )
            else:
                chosen_points = _stock_path_gallery_named_points(
                    preset=preset_name,
                    entry_spot=float(entry_spot),
                    target_price=float(target_price),
                    target_horizon_label=target_horizon_label,
                )
            if not chosen_points:
                continue
            stock_path = _build_stock_path_from_named_points(
                path_grid,
                named_points=chosen_points,
                path_id=f"decision-{role}-{preset_name}",
                path_name=preset_name,
                entry_spot=float(entry_spot),
            )
            path_pool.append(
                _decision_path_payload(
                    path_role=role,
                    path_name=preset_name,
                    path_points=list(stock_path.path_points),
                    selection_reason=f"Candidate {role.replace('_', ' ')} shape from the stock-path library.",
                )
            )
            used_path_names.add(preset_name)
    for fallback in ["range_bound_near_flat", "violent_two_sided_quarter", "plus_20_pct_in_1q"]:
        if fallback in used_path_names:
            continue
        stock_path = _build_stock_path_from_named_points(
            path_grid,
            named_points=_stock_path_gallery_named_points(
                preset=fallback,
                entry_spot=float(entry_spot),
                target_price=float(target_price),
                target_horizon_label=target_horizon_label,
            ),
            path_id=f"decision-{fallback}",
            path_name=fallback,
            entry_spot=float(entry_spot),
        )
        path_pool.append(
            _decision_path_payload(
                path_role=fallback,
                path_name=fallback,
                path_points=list(stock_path.path_points),
                selection_reason="Fallback scenario-library shape used to preserve decision-path coverage.",
            )
        )
        used_path_names.add(fallback)
    return path_pool


_DECISION_PATH_OUTCOME_ORDER = {
    "clear_option_win": 0,
    "wins_but_not_enough": 1,
    "stock_better": 2,
    "fail_too_narrow_or_expiry_issue": 3,
}
_DECISION_PATH_ROLE_PRIORITY = {
    "minimum_required_path": 0,
    "early_rally_path": 1,
    "steady_grind_up_path": 2,
    "late_rally_path": 3,
    "false_breakout_failed_path": 4,
    "recovery_path": 5,
    "earnings_gap_path": 6,
    "range_bound_near_flat": 7,
    "violent_two_sided_quarter": 8,
    "plus_20_pct_in_1q": 9,
}
_DECISION_PATH_OUTCOME_REASON = {
    "clear_option_win": "clear option-win coverage",
    "wins_but_not_enough": "wins-but-not-enough coverage",
    "stock_better": "stock-better benchmark coverage",
    "fail_too_narrow_or_expiry_issue": "failure or too-narrow coverage",
}


def _decision_path_base_score(row: dict[str, Any]) -> float:
    outcome = clean_string(row.get("outcome_label")) or "fail_too_narrow_or_expiry_issue"
    role = clean_string(row.get("path_role"))
    difference = abs(_single_option_num(row.get("difference_vs_stock"), 0.0))
    outperformance = finite_or_none(row.get("outperformance_multiple"))
    outcome_base = {
        "clear_option_win": 86.0,
        "wins_but_not_enough": 82.0,
        "stock_better": 78.0,
        "fail_too_narrow_or_expiry_issue": 74.0,
    }.get(outcome, 70.0)
    role_bonus = max(0.0, 14.0 - float(_DECISION_PATH_ROLE_PRIORITY.get(role, 12)))
    threshold_bonus = 12.0 if role == "minimum_required_path" else 0.0
    magnitude_bonus = min(18.0, difference / 25.0)
    outperformance_bonus = min(6.0, max(0.0, float(outperformance or 0.0))) if outperformance is not None else 0.0
    return round(outcome_base + role_bonus + threshold_bonus + magnitude_bonus + outperformance_bonus, 4)


def _select_curated_single_option_decision_paths(
    path_pool: list[dict[str, Any]],
    pool_outcomes: pd.DataFrame,
    *,
    minimum_count: int = 5,
    maximum_count: int = 8,
) -> list[dict[str, Any]]:
    """Pick a deterministic, explainable subset for the single-option chart."""

    if not path_pool or pool_outcomes.empty:
        return []
    paths_by_id = {
        clean_string(path.get("decision_path_id")): dict(path)
        for path in path_pool
        if clean_string(path.get("decision_path_id"))
    }
    if not paths_by_id:
        return []
    outcome_lookup = {
        clean_string(row.get("decision_path_id")): row
        for row in pool_outcomes.drop_duplicates(subset=["decision_path_id"], keep="first").to_dict("records")
        if clean_string(row.get("decision_path_id"))
    }
    enriched: list[dict[str, Any]] = []
    for original_index, path in enumerate(path_pool):
        decision_path_id = clean_string(path.get("decision_path_id"))
        outcome = outcome_lookup.get(decision_path_id)
        if outcome is None:
            continue
        merged = {**path, **outcome}
        merged["_original_index"] = original_index
        merged["_base_score"] = _decision_path_base_score(merged)
        enriched.append(merged)
    if not enriched:
        return []

    def sort_key(row: dict[str, Any]) -> tuple[float, int, int, int, str]:
        return (
            -float(row.get("_base_score") or 0.0),
            int(_DECISION_PATH_ROLE_PRIORITY.get(clean_string(row.get("path_role")), 99)),
            int(_DECISION_PATH_OUTCOME_ORDER.get(clean_string(row.get("outcome_label")), 99)),
            int(row.get("_original_index") or 0),
            clean_string(row.get("path_name")),
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_families: set[str] = set()
    selected_timings: set[str] = set()
    selected_outcomes: set[str] = set()

    def choose(row: dict[str, Any], reason_prefix: str) -> None:
        if len(selected) >= int(maximum_count):
            return
        decision_path_id = clean_string(row.get("decision_path_id"))
        if not decision_path_id or decision_path_id in selected_ids:
            return
        chosen = dict(paths_by_id[decision_path_id])
        outcome_label = clean_string(row.get("outcome_label")) or "fail_too_narrow_or_expiry_issue"
        family_label = clean_string(row.get("path_family_label")) or clean_string(chosen.get("path_family_label"))
        timing_shape = clean_string(row.get("timing_shape")) or clean_string(chosen.get("timing_shape"))
        chosen["display_order"] = len(selected) + 1
        chosen["outcome_label"] = outcome_label
        chosen["selection_score"] = round(float(row.get("_base_score") or 0.0), 4)
        chosen["selection_reason"] = clean_string(
            f"{reason_prefix}; {family_label} / {timing_shape}; {_DECISION_PATH_OUTCOME_REASON.get(outcome_label, 'outcome coverage')}."
        )
        chosen["is_curated_decision_path"] = True
        selected.append(chosen)
        selected_ids.add(decision_path_id)
        selected_families.add(clean_string(row.get("path_family")) or clean_string(chosen.get("path_family")))
        selected_timings.add(timing_shape)
        selected_outcomes.add(outcome_label)

    required = [row for row in enriched if clean_string(row.get("path_role")) == "minimum_required_path"]
    if required:
        choose(sorted(required, key=sort_key)[0], "Contract-specific threshold path")

    for outcome_label in SINGLE_OPTION_OUTCOME_LABELS:
        candidates = [
            row
            for row in enriched
            if clean_string(row.get("outcome_label")) == outcome_label
            and clean_string(row.get("decision_path_id")) not in selected_ids
        ]
        if candidates:
            choose(sorted(candidates, key=sort_key)[0], f"Representative {clean_string(outcome_label).replace('_', ' ')} path")

    while len(selected) < min(int(minimum_count), len(enriched)) and len(selected) < int(maximum_count):
        remaining = [
            row
            for row in enriched
            if clean_string(row.get("decision_path_id")) not in selected_ids
        ]
        if not remaining:
            break

        def fill_key(row: dict[str, Any]) -> tuple[float, int, int, int, str]:
            family = clean_string(row.get("path_family"))
            timing = clean_string(row.get("timing_shape"))
            outcome = clean_string(row.get("outcome_label"))
            diversity_score = float(row.get("_base_score") or 0.0)
            diversity_score += 22.0 if family and family not in selected_families else 0.0
            diversity_score += 14.0 if timing and timing not in selected_timings else 0.0
            diversity_score += 12.0 if outcome and outcome not in selected_outcomes else 0.0
            return (
                -diversity_score,
                int(_DECISION_PATH_ROLE_PRIORITY.get(clean_string(row.get("path_role")), 99)),
                int(_DECISION_PATH_OUTCOME_ORDER.get(outcome, 99)),
                int(row.get("_original_index") or 0),
                clean_string(row.get("path_name")),
            )

        choose(sorted(remaining, key=fill_key)[0], "Added for family, timing, and explanatory coverage")

    if len(selected) < int(maximum_count):
        remaining = [
            row
            for row in enriched
            if clean_string(row.get("decision_path_id")) not in selected_ids
        ]
        for row in sorted(remaining, key=sort_key):
            if len(selected) >= int(maximum_count):
                break
            family = clean_string(row.get("path_family"))
            timing = clean_string(row.get("timing_shape"))
            outcome = clean_string(row.get("outcome_label"))
            if family in selected_families and timing in selected_timings and outcome in selected_outcomes:
                continue
            choose(row, "Added because it expands the decision-path explanation set")

    return selected


def _single_option_decision_path_selection_frame(
    *,
    selected_paths: list[dict[str, Any]],
    path_outcomes: pd.DataFrame,
    candidate_slug: str,
    candidate_short_label: str,
) -> pd.DataFrame:
    if not selected_paths or path_outcomes.empty:
        return pd.DataFrame()
    outcome_lookup = {
        clean_string(row.get("decision_path_id")): row
        for row in path_outcomes.to_dict("records")
        if clean_string(row.get("decision_path_id"))
    }
    rows: list[dict[str, Any]] = []
    for order, path in enumerate(selected_paths, start=1):
        decision_path_id = clean_string(path.get("decision_path_id"))
        outcome = outcome_lookup.get(decision_path_id, {})
        rows.append(
            {
                "candidate_slug": candidate_slug,
                "candidate_short_label": candidate_short_label,
                "decision_path_id": decision_path_id,
                "path_role": clean_string(path.get("path_role")),
                "path_name": clean_string(path.get("path_name")),
                "path_label": clean_string(path.get("path_label")),
                "path_family": clean_string(path.get("path_family")),
                "path_family_label": clean_string(path.get("path_family_label")),
                "timing_shape": clean_string(path.get("timing_shape")),
                "outcome_bias": clean_string(path.get("outcome_bias")),
                "outcome_label": clean_string(outcome.get("outcome_label")) or clean_string(path.get("outcome_label")),
                "display_order": int(path.get("display_order") or order),
                "selection_score": finite_or_none(path.get("selection_score")),
                "selection_reason": clean_string(path.get("selection_reason")),
                "exit_stock_price": finite_or_none(outcome.get("exit_stock_price")),
                "difference_vs_stock": finite_or_none(outcome.get("difference_vs_stock")),
                "outperformance_multiple": finite_or_none(outcome.get("outperformance_multiple")),
            }
        )
    return pd.DataFrame(rows)


def _select_decision_comparison_paths(
    *,
    required_stock_path_to_buy: pd.DataFrame,
    anchor_candidate_slug: str,
    snapshot_date: date,
    target_date: date,
    target_price: float,
    target_horizon_label: str,
    entry_spot: float,
) -> list[dict[str, Any]]:
    path_pool = _build_decision_comparison_path_pool(
        required_stock_path_to_buy=required_stock_path_to_buy,
        anchor_candidate_slug=anchor_candidate_slug,
        snapshot_date=snapshot_date,
        target_date=target_date,
        target_price=target_price,
        target_horizon_label=target_horizon_label,
        entry_spot=entry_spot,
    )
    selected: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for path in path_pool:
        role = clean_string(path.get("path_role"))
        if role in seen_roles:
            continue
        chosen = dict(path)
        chosen["display_order"] = len(selected) + 1
        chosen["selection_score"] = 0.0
        chosen["is_curated_decision_path"] = True
        selected.append(chosen)
        seen_roles.add(role)
        if len(selected) >= 8:
            break
    return selected


def _evaluate_candidate_on_decision_paths(
    spec: dict[str, Any],
    *,
    candidate_slug: str,
    candidate_label: str,
    candidate_short_label: str,
    selected_paths: list[dict[str, Any]],
    target_price: float,
    active_iv_path: dict[str, float],
    comparison_capital: float,
    premium_used: float,
    exit_rule: str,
    target_return_pct: float,
    minimum_outperformance_multiple: float,
    strong_outperformance_multiple: float,
    include_trace_rows: bool = True,
    max_paths: int | None = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strike_value = _single_option_num(spec.get("strike_label"), 0.0)
    representative_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    normalized_exit_rule = clean_string(exit_rule).lower() or "sell_on_thesis_completion"

    paths_to_evaluate = selected_paths if max_paths is None else selected_paths[: int(max_paths)]
    for fallback_display_order, path in enumerate(paths_to_evaluate, start=1):
        path_points = list(path.get("path_points") or [])
        if not path_points:
            continue
        decision_path_id = clean_string(path.get("decision_path_id")) or clean_string(path.get("path_role"))
        path_role = clean_string(path.get("path_role"))
        path_name = clean_string(path.get("path_name"))
        path_label = clean_string(path.get("path_label")) or _humanize_named_path(path_name, kind="stock")
        path_family = clean_string(path.get("path_family"))
        path_family_label = clean_string(path.get("path_family_label"))
        timing_shape = clean_string(path.get("timing_shape"))
        outcome_bias = clean_string(path.get("outcome_bias"))
        path_description = clean_string(path.get("path_description"))
        selection_reason = clean_string(path.get("selection_reason"))
        selection_score = finite_or_none(path.get("selection_score"))
        display_order = int(path.get("display_order") or fallback_display_order)
        path_evaluations: list[dict[str, Any]] = []
        first_cross_strike_day: int | None = None
        for step_index, point in enumerate(path_points):
            requested_days = int(point.get("requested_days") or 0)
            spot = _single_option_num(point.get("spot_price"))
            if first_cross_strike_day is None and strike_value and spot >= strike_value:
                first_cross_strike_day = requested_days
            iv_shift = float(active_iv_path.get(f"{requested_days}d", 0.0))
            evaluation = _single_option_adjusted_evaluation(
                spec,
                spot_price=spot,
                horizon_days=requested_days,
                iv_shift_points=iv_shift,
                comparison_capital=float(comparison_capital),
                premium_used=float(premium_used),
            )
            path_evaluations.append(evaluation)
            if include_trace_rows:
                representative_rows.append(
                    {
                        "candidate_slug": candidate_slug,
                        "candidate_short_label": candidate_short_label,
                        "decision_path_id": decision_path_id,
                        "path_role": path_role,
                        "path_name": path_name,
                        "path_label": path_label,
                        "path_family": path_family,
                        "path_family_label": path_family_label,
                        "timing_shape": timing_shape,
                        "outcome_bias": outcome_bias,
                        "path_description": path_description,
                        "selection_score": selection_score,
                        "selection_reason": selection_reason,
                        "is_curated_decision_path": bool(path.get("is_curated_decision_path", False)),
                        "display_order": display_order,
                        "step_index": int(point.get("step_index") if point.get("step_index") is not None else step_index),
                        "date": clean_string(point.get("date")),
                        "requested_days": requested_days,
                        "spot_price": round(spot, 4),
                        "return_pct": finite_or_none(point.get("return_pct")),
                        "iv_shift_points": round(iv_shift, 4),
                    }
                )
        if not path_evaluations:
            continue

        if normalized_exit_rule == "hold_to_expiry":
            exit_index = len(path_evaluations) - 1
        elif normalized_exit_rule == "sell_at_target_return":
            exit_index = next(
                (
                    idx
                    for idx, evaluation in enumerate(path_evaluations)
                    if _single_option_num(evaluation.get("profit_loss"), 0.0) / max(float(premium_used), 1.0) >= float(target_return_pct)
                ),
                len(path_evaluations) - 1,
            )
        else:
            exit_index = next(
                (
                    idx
                    for idx, point in enumerate(path_points)
                    if _single_option_num(point.get("spot_price"), 0.0) >= float(target_price)
                ),
                len(path_evaluations) - 1,
            )

        exit_eval = path_evaluations[exit_index]
        exit_point = path_points[exit_index]
        profit = _single_option_num(exit_eval.get("profit_loss"), 0.0)
        stock_profit = _single_option_num(exit_eval.get("stock_profit_loss"), 0.0)
        difference = _single_option_num(exit_eval.get("difference_vs_stock"), 0.0)
        outperformance = _single_option_outperformance_multiple(profit, stock_profit)
        outcome_label = _single_option_outcome_label(
            profit_loss=profit,
            stock_profit_loss=stock_profit,
            difference_vs_stock=difference,
            outperformance_multiple=outperformance,
            strong_outperformance_multiple=float(strong_outperformance_multiple),
            minimum_outperformance_multiple=float(minimum_outperformance_multiple),
            clamped_to_expiry=bool(exit_eval.get("clamped_to_expiry")),
            requested_days=int(exit_eval.get("requested_days") or 0),
            effective_days=int(exit_eval.get("effective_days") or 0),
        )
        beats_stock = bool(difference > 0)
        qualifies = bool(
            outperformance is not None
            and outperformance >= float(minimum_outperformance_multiple)
            and difference > 0
        )
        strong_win = bool(
            outperformance is not None
            and outperformance >= float(strong_outperformance_multiple)
            and difference > 0
        )
        outcome_rows.append(
            {
                "candidate_slug": candidate_slug,
                "candidate_label": candidate_label,
                "candidate_short_label": candidate_short_label,
                "decision_path_id": decision_path_id,
                "path_role": path_role,
                "path_name": path_name,
                "path_label": path_label,
                "path_family": path_family,
                "path_family_label": path_family_label,
                "timing_shape": timing_shape,
                "outcome_bias": outcome_bias,
                "path_description": path_description,
                "selection_score": selection_score,
                "selection_reason": selection_reason,
                "is_curated_decision_path": bool(path.get("is_curated_decision_path", False)),
                "display_order": display_order,
                "exit_rule": normalized_exit_rule,
                "exit_date": clean_string(exit_point.get("date")),
                "exit_requested_days": int(exit_eval.get("requested_days") or 0),
                "exit_effective_days": int(exit_eval.get("effective_days") or 0),
                "exit_stock_price": round(_single_option_num(exit_point.get("spot_price"), 0.0), 4),
                "estimated_option_value": finite_or_none(exit_eval.get("estimated_value")),
                "premium_used": round(float(premium_used), 4),
                "profit_loss": round(profit, 4),
                "stock_profit_loss": round(stock_profit, 4),
                "difference_vs_stock": round(difference, 4),
                "outperformance_multiple": round(float(outperformance), 4) if outperformance is not None else None,
                "outperformance_vs_stock_pct": round((difference / max(abs(stock_profit), 1.0)), 6) if stock_profit else None,
                "beats_stock": beats_stock,
                "qualifies_as_winning_path_family": qualifies,
                "qualifies_as_strong_path_family": strong_win,
                "outcome_label": outcome_label,
                "outcome_note": _single_option_path_status_note(outcome_label),
                "clamped_to_expiry": bool(exit_eval.get("clamped_to_expiry")),
                "first_cross_above_strike_day": first_cross_strike_day,
            }
        )
        timing_rows.append(
            {
                "candidate_slug": candidate_slug,
                "candidate_short_label": candidate_short_label,
                "decision_path_id": decision_path_id,
                "path_role": path_role,
                "path_name": path_name,
                "path_label": path_label,
                "path_family": path_family,
                "path_family_label": path_family_label,
                "timing_shape": timing_shape,
                "selection_score": selection_score,
                "selection_reason": selection_reason,
                "is_curated_decision_path": bool(path.get("is_curated_decision_path", False)),
                "first_cross_above_strike_day": first_cross_strike_day,
                "exit_requested_days": int(exit_eval.get("requested_days") or 0),
                "exit_effective_days": int(exit_eval.get("effective_days") or 0),
                "timing_read": (
                    "needs_early_move"
                    if first_cross_strike_day is not None and first_cross_strike_day <= 45 and outcome_label in {"clear_option_win", "wins_but_not_enough"}
                    else "late_move_favors_stock"
                    if outcome_label == "stock_better"
                    else "expiry_or_path_too_narrow"
                    if outcome_label == "fail_too_narrow_or_expiry_issue"
                    else "path_timing_acceptable"
                ),
            }
        )

    return pd.DataFrame(representative_rows), pd.DataFrame(outcome_rows), pd.DataFrame(timing_rows)


def _single_option_edge_path_key(edge_multiple: float) -> str:
    return f"required_path_to_beat_stock_{str(float(edge_multiple)).replace('.', '_')}x"


def _single_option_edge_gap_value(
    spec: dict[str, Any],
    *,
    spot_price: float,
    horizon_days: int,
    iv_shift_points: float,
    comparison_capital: float,
    premium_used: float,
    edge_multiple: float,
) -> tuple[float, dict[str, Any]]:
    evaluation = _single_option_adjusted_evaluation(
        spec,
        spot_price=float(spot_price),
        horizon_days=int(horizon_days),
        iv_shift_points=float(iv_shift_points),
        comparison_capital=float(comparison_capital),
        premium_used=float(premium_used),
    )
    profit = _single_option_num(evaluation.get("profit_loss"), 0.0)
    stock_profit = _single_option_num(evaluation.get("stock_profit_loss"), 0.0)
    required_profit = float(edge_multiple) * stock_profit if stock_profit > 0 else 0.0
    return profit - required_profit, evaluation


def _single_option_required_spot_for_edge(
    spec: dict[str, Any],
    *,
    horizon_days: int,
    iv_shift_points: float,
    comparison_capital: float,
    premium_used: float,
    edge_multiple: float,
    entry_spot: float,
    target_price: float,
    strike_value: float,
    observed_max_spot: float,
) -> tuple[float | None, dict[str, Any] | None, str]:
    lower = max(0.01, float(entry_spot))
    lower_gap, lower_eval = _single_option_edge_gap_value(
        spec,
        spot_price=lower,
        horizon_days=int(horizon_days),
        iv_shift_points=float(iv_shift_points),
        comparison_capital=float(comparison_capital),
        premium_used=float(premium_used),
        edge_multiple=float(edge_multiple),
    )
    if lower_gap >= 0:
        return lower, lower_eval, "already_clears_at_entry_spot"

    upper = max(
        lower * 1.08,
        float(target_price) * 1.15,
        float(strike_value or target_price) * 1.20,
        float(observed_max_spot or target_price) * 1.20,
    )
    upper_eval: dict[str, Any] | None = None
    upper_gap = lower_gap
    for _ in range(14):
        upper_gap, upper_eval = _single_option_edge_gap_value(
            spec,
            spot_price=upper,
            horizon_days=int(horizon_days),
            iv_shift_points=float(iv_shift_points),
            comparison_capital=float(comparison_capital),
            premium_used=float(premium_used),
            edge_multiple=float(edge_multiple),
        )
        if upper_gap >= 0:
            break
        upper *= 1.35
    if upper_gap < 0 or upper_eval is None:
        return None, upper_eval, "unreached_in_search_range"

    lo = lower
    hi = upper
    best_eval = upper_eval
    for _ in range(52):
        mid = (lo + hi) / 2.0
        mid_gap, mid_eval = _single_option_edge_gap_value(
            spec,
            spot_price=mid,
            horizon_days=int(horizon_days),
            iv_shift_points=float(iv_shift_points),
            comparison_capital=float(comparison_capital),
            premium_used=float(premium_used),
            edge_multiple=float(edge_multiple),
        )
        if mid_gap >= 0:
            hi = mid
            best_eval = mid_eval
        else:
            lo = mid
    return hi, best_eval, "solved"


def _single_option_required_edge_path_frame(
    spec: dict[str, Any],
    *,
    candidate_slug: str,
    candidate_short_label: str,
    representative_paths: pd.DataFrame,
    snapshot_date: date,
    target_date: date,
    target_price: float,
    entry_spot: float,
    active_iv_path: dict[str, float],
    comparison_capital: float,
    premium_used: float,
    edge_multiple: float,
) -> pd.DataFrame:
    if representative_paths.empty:
        path_grid = _build_path_grid(snapshot_date, target_date)
        horizons = pd.DataFrame(path_grid)
        horizons["date"] = [
            (snapshot_date + timedelta(days=int(row.get("requested_days") or 0))).isoformat()
            for row in path_grid
        ]
    else:
        horizons = representative_paths[["requested_days", "date"]].drop_duplicates().copy()
    horizons["requested_days"] = pd.to_numeric(horizons.get("requested_days"), errors="coerce").fillna(0).astype(int)
    horizons = horizons.sort_values("requested_days").drop_duplicates(subset=["requested_days"], keep="first")
    if horizons.empty:
        return pd.DataFrame()
    observed_max = _single_option_num(
        pd.to_numeric(representative_paths.get("spot_price", pd.Series(dtype=float)), errors="coerce").max()
        if not representative_paths.empty
        else target_price,
        float(target_price),
    )
    strike_value = _single_option_num(spec.get("strike_label"), float(target_price))
    path_key = _single_option_edge_path_key(edge_multiple)
    edge_label = f"Required path to beat stock {float(edge_multiple):.1f}x"
    rows: list[dict[str, Any]] = []
    for step_index, row in enumerate(horizons.to_dict("records")):
        requested_days = int(row.get("requested_days") or 0)
        iv_shift = float(active_iv_path.get(f"{requested_days}d", 0.0))
        required_spot, evaluation, status = _single_option_required_spot_for_edge(
            spec,
            horizon_days=requested_days,
            iv_shift_points=iv_shift,
            comparison_capital=float(comparison_capital),
            premium_used=float(premium_used),
            edge_multiple=float(edge_multiple),
            entry_spot=float(entry_spot),
            target_price=float(target_price),
            strike_value=float(strike_value),
            observed_max_spot=float(observed_max),
        )
        stock_profit = _single_option_num((evaluation or {}).get("stock_profit_loss"), 0.0)
        profit = _single_option_num((evaluation or {}).get("profit_loss"), 0.0)
        required_profit = float(edge_multiple) * stock_profit if stock_profit > 0 else 0.0
        rows.append(
            {
                "candidate_slug": candidate_slug,
                "candidate_short_label": candidate_short_label,
                "edge_path_name": path_key,
                "edge_label": edge_label,
                "edge_multiple": float(edge_multiple),
                "display_order": 1 if float(edge_multiple) <= 1.5 else 2,
                "step_index": step_index,
                "date": clean_string(row.get("date")) or (snapshot_date + timedelta(days=requested_days)).isoformat(),
                "requested_days": requested_days,
                "required_stock_price": round(float(required_spot), 4) if required_spot is not None else None,
                "spot_price": round(float(required_spot), 4) if required_spot is not None else None,
                "return_pct": (float(required_spot) / float(entry_spot) - 1.0) if required_spot is not None and entry_spot else None,
                "iv_shift_points": round(iv_shift, 4),
                "required_option_profit_loss": round(float(required_profit), 4),
                "estimated_option_profit_loss": round(float(profit), 4) if evaluation is not None else None,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _edge_required_price_at(edge_path: pd.DataFrame, requested_days: int) -> float | None:
    if edge_path.empty:
        return None
    data = edge_path.dropna(subset=["required_stock_price"]).copy()
    if data.empty:
        return None
    data["requested_days"] = pd.to_numeric(data.get("requested_days"), errors="coerce")
    data["required_stock_price"] = pd.to_numeric(data.get("required_stock_price"), errors="coerce")
    data = data.dropna(subset=["requested_days", "required_stock_price"]).sort_values("requested_days")
    if data.empty:
        return None
    return float(np.interp(float(requested_days), data["requested_days"].to_numpy(), data["required_stock_price"].to_numpy()))


def _single_option_edge_gap_outputs(
    *,
    path_outcomes: pd.DataFrame,
    min_edge_path: pd.DataFrame,
    strong_edge_path: pd.DataFrame,
    minimum_outperformance_multiple: float,
    strong_outperformance_multiple: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path_outcomes.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in path_outcomes.to_dict("records"):
        requested_days = int(_single_option_num(row.get("exit_requested_days"), 0.0))
        exit_stock = finite_or_none(row.get("exit_stock_price"))
        profit = _single_option_num(row.get("profit_loss"), 0.0)
        stock_profit = _single_option_num(row.get("stock_profit_loss"), 0.0)
        required_profit = float(minimum_outperformance_multiple) * stock_profit if stock_profit > 0 else 0.0
        strong_required_profit = float(strong_outperformance_multiple) * stock_profit if stock_profit > 0 else 0.0
        min_required_price = _edge_required_price_at(min_edge_path, requested_days)
        strong_required_price = _edge_required_price_at(strong_edge_path, requested_days)
        extra_move = (
            max(0.0, float(min_required_price) - float(exit_stock))
            if min_required_price is not None and exit_stock is not None
            else None
        )
        extra_move_pct = (extra_move / float(exit_stock)) if extra_move is not None and exit_stock else None
        first_cross_day = finite_or_none(row.get("first_cross_above_strike_day"))
        outcome_label = clean_string(row.get("outcome_label"))
        if first_cross_day is None:
            timing_note = "too_low_and_never_crosses_strike"
        elif extra_move is not None and extra_move > 0 and float(first_cross_day) > 45:
            timing_note = "too_low_and_too_late"
        elif extra_move is not None and extra_move > 0:
            timing_note = "needs_more_stock_move"
        elif outcome_label == "stock_better":
            timing_note = "needs_earlier_timing_or_better_entry"
        else:
            timing_note = "clears_or_near_edge"
        rows.append(
            {
                "candidate_slug": clean_string(row.get("candidate_slug")),
                "candidate_short_label": clean_string(row.get("candidate_short_label")),
                "decision_path_id": clean_string(row.get("decision_path_id")),
                "path_role": clean_string(row.get("path_role")),
                "path_name": clean_string(row.get("path_name")),
                "path_label": clean_string(row.get("path_label")),
                "path_family": clean_string(row.get("path_family")),
                "path_family_label": clean_string(row.get("path_family_label")),
                "timing_shape": clean_string(row.get("timing_shape")),
                "outcome_label": outcome_label,
                "display_order": int(_single_option_num(row.get("display_order"), 0.0)),
                "exit_requested_days": requested_days,
                "exit_date": clean_string(row.get("exit_date")),
                "exit_stock_price": exit_stock,
                "required_stock_price_1_5x": min_required_price,
                "required_stock_price_2_0x": strong_required_price,
                "extra_stock_move_needed_1_5x": extra_move,
                "extra_stock_move_needed_pct_1_5x": extra_move_pct,
                "profit_loss": profit,
                "stock_profit_loss": stock_profit,
                "required_profit_loss_1_5x": required_profit,
                "required_profit_loss_2_0x": strong_required_profit,
                "edge_gap_to_1_5x_dollars": profit - required_profit,
                "edge_gap_to_2_0x_dollars": profit - strong_required_profit,
                "outperformance_multiple": finite_or_none(row.get("outperformance_multiple")),
                "edge_gap_to_1_5x_multiple": (
                    float(row.get("outperformance_multiple")) - float(minimum_outperformance_multiple)
                    if finite_or_none(row.get("outperformance_multiple")) is not None
                    else None
                ),
                "timing_gap_note": timing_note,
            }
        )
    gap_by_path = pd.DataFrame(rows)
    if gap_by_path.empty:
        return pd.DataFrame(), pd.DataFrame()
    gap_by_path = gap_by_path.sort_values(
        ["edge_gap_to_1_5x_dollars", "extra_stock_move_needed_1_5x", "display_order"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    gap_by_path["is_closest_to_edge"] = False
    gap_by_path.loc[0, "is_closest_to_edge"] = True
    closest = gap_by_path.head(1).copy()
    closest_row = closest.iloc[0].to_dict()
    extra_move = finite_or_none(closest_row.get("extra_stock_move_needed_1_5x"))
    exit_stock = finite_or_none(closest_row.get("exit_stock_price"))
    if extra_move is not None and extra_move > 0:
        pct_text = f"{(extra_move / exit_stock) * 100:.1f}%" if exit_stock else "n/a"
        annotation = f"Closest miss needs about ${extra_move:,.2f} more stock move ({pct_text}) by this path's exit."
    elif clean_string(closest_row.get("outcome_label")) == "stock_better":
        annotation = "Closest miss needs earlier timing, a better entry, or stronger IV support to clear stock."
    else:
        annotation = "Closest path clears or nearly clears the configured option-over-stock edge."
    closest["annotation_text"] = annotation
    return gap_by_path, closest


def _build_single_option_decision_markdown(
    *,
    ticker: str,
    summary: pd.DataFrame,
    family_counts: pd.DataFrame,
    bullets: pd.DataFrame,
) -> str:
    if summary.empty:
        return ""
    row = summary.iloc[0].to_dict()
    bullet_lines = [
        clean_string(item.get("bullet_text"))
        for item in bullets.sort_values("bullet_order").to_dict("records")
        if clean_string(item.get("bullet_text"))
    ]
    count_row = family_counts.iloc[0].to_dict() if not family_counts.empty else {}
    lines = [
        f"# {ticker} Single-Option Decision View",
        "",
        "## Decision Snapshot",
        "",
        f"- Selected option: `{clean_string(row.get('candidate_short_label')) or clean_string(row.get('candidate_label'))}`",
        f"- Decision status: `{clean_string(row.get('single_option_decision_status'))}`",
        f"- Exit rule: `{clean_string(row.get('exit_rule'))}`",
        f"- Premium used: `${_single_option_num(row.get('premium_used'), 0.0):,.2f}` via `{clean_string(row.get('entry_price_mode'))}`",
        f"- Winning path families: `{int(_single_option_num(count_row.get('qualifying_path_family_count'), 0))}` of `{int(_single_option_num(count_row.get('evaluated_path_family_count'), 0))}`",
        "",
        "## What The View Answers",
        "",
        "In plain terms: what stock paths make one selected call worth buying instead of buying stock?",
        "",
        "This section asks whether the selected call is worth buying instead of buying the stock under a curated set of decision paths. The broad path gallery stays separate as a scenario library; this view keeps only the paths that explain the selected option.",
        "",
        "## Summary Bullets",
        "",
    ]
    if bullet_lines:
        lines.extend(f"- {line}" for line in bullet_lines[:7])
    else:
        lines.append("- No single-option bullets were available for this run.")
    lines.extend(
        [
            "",
            "## Files To Open",
            "",
            "- `charts/single_option_decision_view.png`: curated decision-path chart, bullets, IV sensitivity, and entry sensitivity.",
            "- `tables/single_option_decision_path_selections.csv`: selected decision paths with family, outcome label, score, and reason.",
            "- `tables/single_option_required_path_to_beat_stock_1_5x.csv`: stock path required to clear the option-over-stock threshold.",
            "- `tables/single_option_edge_gap_by_path_family.csv`: closest miss and required-stock gap for each selected path.",
            "- `tables/single_option_path_outcomes.csv`: path-by-path option-vs-stock outcomes.",
            "- `tables/single_option_iv_sensitivity.csv`: low/base/high IV sensitivity for the same selected option.",
            "- `tables/single_option_entry_sensitivity.csv`: cheap/reference/expensive entry sensitivity.",
        ]
    )
    return "\n".join(lines)


def _build_single_option_decision_outputs(
    *,
    ticker: str,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    bullish_action_board: pd.DataFrame,
    required_stock_path_to_buy: pd.DataFrame,
    snapshot_date: date,
    target_price: float,
    target_date: date,
    target_horizon_label: str,
    entry_spot: float,
    active_iv_path_name: str,
    active_iv_path_points: dict[str, float],
    comparison_capital: float,
    candidate_slug: str | None,
    minimum_outperformance_multiple: float,
    strong_outperformance_multiple: float,
    required_winning_path_families: int,
    entry_price_mode: str,
    exit_rule: str,
    target_return_pct: float,
) -> dict[str, Any]:
    outputs = _single_option_empty_outputs()
    normalized_exit_rule = clean_string(exit_rule).lower() or "sell_on_thesis_completion"
    if normalized_exit_rule not in SINGLE_OPTION_EXIT_RULE_CHOICES:
        raise ValueError(f"single_option_exit_rule must be one of {SINGLE_OPTION_EXIT_RULE_CHOICES!r}.")
    normalized_entry_mode = clean_string(entry_price_mode).lower() or "conservative_mid_plus_slippage"
    if normalized_entry_mode not in SINGLE_OPTION_ENTRY_PRICE_MODES:
        raise ValueError(f"entry_price_mode must be one of {SINGLE_OPTION_ENTRY_PRICE_MODES!r}.")
    candidate, spec = _select_single_option_candidate(
        specs=specs,
        candidate_rows=candidate_rows,
        bullish_action_board=bullish_action_board,
        candidate_slug=candidate_slug,
    )
    if candidate is None or spec is None:
        return outputs
    position: StrategyPosition = spec["position"]
    premium_used, premium_source = _single_option_entry_premium(position, mode=normalized_entry_mode)
    candidate_label = clean_string(candidate.get("candidate_label"))
    candidate_short_label = _single_option_candidate_short_label(candidate)
    selected_slug = clean_string(candidate.get("candidate_slug") or spec.get("candidate_slug"))
    expiry_dt = parse_date(candidate.get("expiry_date")) or position.expiry_date
    dte = max((expiry_dt - snapshot_date).days, 0) if expiry_dt else None
    strike_value = _single_option_num(candidate.get("strike_label"), 0.0)
    base_iv = finite_or_none(position.option_legs[0].base_iv if position.option_legs else None)
    breakeven = strike_value + (premium_used / 100.0) if strike_value else None
    max_loss = premium_used
    path_grid = _build_path_grid(snapshot_date, target_date)
    active_iv_horizons = [
        {"label": f"{int(row.get('requested_days') or 0)}d", "requested_days": int(row.get("requested_days") or 0)}
        for row in path_grid
    ]
    active_iv_path = _interpolated_path(
        active_iv_path_points,
        active_iv_horizons,
        default_value=0.0,
    )
    path_pool = _build_decision_comparison_path_pool(
        required_stock_path_to_buy=required_stock_path_to_buy,
        anchor_candidate_slug=selected_slug,
        snapshot_date=snapshot_date,
        target_date=target_date,
        target_price=float(target_price),
        target_horizon_label=target_horizon_label,
        entry_spot=float(entry_spot),
    )
    _, pool_outcomes, _ = _evaluate_candidate_on_decision_paths(
        spec,
        candidate_slug=selected_slug,
        candidate_label=candidate_label,
        candidate_short_label=candidate_short_label,
        selected_paths=path_pool,
        target_price=float(target_price),
        active_iv_path=active_iv_path,
        comparison_capital=float(comparison_capital),
        premium_used=float(premium_used),
        exit_rule=normalized_exit_rule,
        target_return_pct=float(target_return_pct),
        minimum_outperformance_multiple=float(minimum_outperformance_multiple),
        strong_outperformance_multiple=float(strong_outperformance_multiple),
        include_trace_rows=False,
        max_paths=None,
    )
    selected_paths = _select_curated_single_option_decision_paths(path_pool, pool_outcomes)
    representative_paths, path_outcomes, timing_sensitivity = _evaluate_candidate_on_decision_paths(
        spec,
        candidate_slug=selected_slug,
        candidate_label=candidate_label,
        candidate_short_label=candidate_short_label,
        selected_paths=selected_paths,
        target_price=float(target_price),
        active_iv_path=active_iv_path,
        comparison_capital=float(comparison_capital),
        premium_used=float(premium_used),
        exit_rule=normalized_exit_rule,
        target_return_pct=float(target_return_pct),
        minimum_outperformance_multiple=float(minimum_outperformance_multiple),
        strong_outperformance_multiple=float(strong_outperformance_multiple),
        include_trace_rows=True,
    )
    if path_outcomes.empty:
        return outputs
    decision_path_selections = _single_option_decision_path_selection_frame(
        selected_paths=selected_paths,
        path_outcomes=path_outcomes,
        candidate_slug=selected_slug,
        candidate_short_label=candidate_short_label,
    )
    required_edge_1_5x = _single_option_required_edge_path_frame(
        spec,
        candidate_slug=selected_slug,
        candidate_short_label=candidate_short_label,
        representative_paths=representative_paths,
        snapshot_date=snapshot_date,
        target_date=target_date,
        target_price=float(target_price),
        entry_spot=float(entry_spot),
        active_iv_path=active_iv_path,
        comparison_capital=float(comparison_capital),
        premium_used=float(premium_used),
        edge_multiple=float(minimum_outperformance_multiple),
    )
    required_edge_2_0x = _single_option_required_edge_path_frame(
        spec,
        candidate_slug=selected_slug,
        candidate_short_label=candidate_short_label,
        representative_paths=representative_paths,
        snapshot_date=snapshot_date,
        target_date=target_date,
        target_price=float(target_price),
        entry_spot=float(entry_spot),
        active_iv_path=active_iv_path,
        comparison_capital=float(comparison_capital),
        premium_used=float(premium_used),
        edge_multiple=float(strong_outperformance_multiple),
    )
    edge_gap_by_path_family, closest_to_edge = _single_option_edge_gap_outputs(
        path_outcomes=path_outcomes,
        min_edge_path=required_edge_1_5x,
        strong_edge_path=required_edge_2_0x,
        minimum_outperformance_multiple=float(minimum_outperformance_multiple),
        strong_outperformance_multiple=float(strong_outperformance_multiple),
    )
    qualifying_count = int(path_outcomes.get("qualifies_as_winning_path_family", pd.Series(dtype=bool)).fillna(False).sum())
    evaluated_count = int(path_outcomes["decision_path_id"].nunique()) if "decision_path_id" in path_outcomes.columns else int(path_outcomes["path_role"].nunique())
    too_narrow = qualifying_count < int(required_winning_path_families)
    family_counts = pd.DataFrame(
        [
            {
                "candidate_slug": selected_slug,
                "candidate_short_label": candidate_short_label,
                "evaluated_path_family_count": evaluated_count,
                "qualifying_path_family_count": qualifying_count,
                "required_winning_path_families": int(required_winning_path_families),
                "clear_option_win_count": int(path_outcomes["outcome_label"].eq("clear_option_win").sum()),
                "wins_but_not_enough_count": int(path_outcomes["outcome_label"].eq("wins_but_not_enough").sum()),
                "stock_better_count": int(path_outcomes["outcome_label"].eq("stock_better").sum()),
                "fail_too_narrow_or_expiry_issue_count": int(path_outcomes["outcome_label"].eq("fail_too_narrow_or_expiry_issue").sum()),
                "too_narrow_under_representative_paths": bool(too_narrow),
            }
        ]
    )

    iv_rows: list[dict[str, Any]] = []
    terminal_path = selected_paths[0] if selected_paths else {}
    terminal_point = (terminal_path.get("path_points") or [{}])[-1]
    iv_mode_shifts = {"low": -0.10, "base": 0.0, "high": 0.10}
    for order, mode in enumerate(SINGLE_OPTION_DEFAULT_IV_MODES, start=1):
        iv_shift = float(iv_mode_shifts[mode])
        evaluation = _single_option_adjusted_evaluation(
            spec,
            spot_price=_single_option_num(terminal_point.get("spot_price"), target_price),
            horizon_days=int(terminal_point.get("requested_days") or max((target_date - snapshot_date).days, 0)),
            iv_shift_points=iv_shift,
            comparison_capital=float(comparison_capital),
            premium_used=float(premium_used),
        )
        profit = _single_option_num(evaluation.get("profit_loss"), 0.0)
        stock_profit = _single_option_num(evaluation.get("stock_profit_loss"), 0.0)
        difference = _single_option_num(evaluation.get("difference_vs_stock"), 0.0)
        outperformance = _single_option_outperformance_multiple(profit, stock_profit)
        iv_rows.append(
            {
                "candidate_slug": selected_slug,
                "candidate_short_label": candidate_short_label,
                "iv_mode": mode,
                "iv_mode_label": {"low": "Low IV", "base": "Base IV", "high": "High IV"}[mode],
                "display_order": order,
                "iv_shift_points": iv_shift,
                "estimated_option_value": finite_or_none(evaluation.get("estimated_value")),
                "profit_loss": round(profit, 4),
                "stock_profit_loss": round(stock_profit, 4),
                "difference_vs_stock": round(difference, 4),
                "outperformance_multiple": round(float(outperformance), 4) if outperformance is not None else None,
                "sensitivity_note": (
                    "Lower IV hurts this call if value falls versus base."
                    if mode == "low"
                    else "Base IV is the active comparison."
                    if mode == "base"
                    else "Higher IV support can cushion theta but does not fix a bad stock path."
                ),
            }
        )
    iv_sensitivity = pd.DataFrame(iv_rows)

    entry_rows: list[dict[str, Any]] = []
    entry_scenarios = [
        ("cheap_fill", "Cheap fill (-10%)", 0.90),
        ("reference_fill", "Reference fill", 1.00),
        ("expensive_fill", "Expensive fill (+10%)", 1.10),
    ]
    for order, (scenario, label, multiplier) in enumerate(entry_scenarios, start=1):
        adjusted_premium = float(premium_used) * float(multiplier)
        wins = 0
        avg_difference = 0.0
        for path in selected_paths[:8]:
            points = list(path.get("path_points") or [])
            if not points:
                continue
            terminal = points[-1]
            evaluation = _single_option_adjusted_evaluation(
                spec,
                spot_price=_single_option_num(terminal.get("spot_price"), target_price),
                horizon_days=int(terminal.get("requested_days") or 0),
                iv_shift_points=0.0,
                comparison_capital=float(comparison_capital),
                premium_used=adjusted_premium,
            )
            difference = _single_option_num(evaluation.get("difference_vs_stock"), 0.0)
            avg_difference += difference
            if difference > 0:
                wins += 1
        path_count = max(len(selected_paths[:8]), 1)
        entry_rows.append(
            {
                "candidate_slug": selected_slug,
                "candidate_short_label": candidate_short_label,
                "entry_scenario": scenario,
                "entry_scenario_label": label,
                "display_order": order,
                "premium_multiplier": multiplier,
                "premium_used": round(adjusted_premium, 4),
                "path_families_beating_stock": wins,
                "evaluated_path_family_count": path_count,
                "average_difference_vs_stock": round(avg_difference / path_count, 4),
                "entry_read": (
                    "Cheaper entry meaningfully improves the setup."
                    if scenario == "cheap_fill" and wins >= qualifying_count
                    else "Reference entry used in the hero view."
                    if scenario == "reference_fill"
                    else "More expensive fills quickly make the option harder to justify."
                ),
            }
        )
    entry_sensitivity = pd.DataFrame(entry_rows)

    best_outcome = path_outcomes.sort_values("difference_vs_stock", ascending=False).iloc[0].to_dict()
    worst_outcome = path_outcomes.sort_values("difference_vs_stock", ascending=True).iloc[0].to_dict()
    status = "passes_multiple_path_test" if not too_narrow else "too_narrow_under_representative_paths"
    if path_outcomes["outcome_label"].eq("stock_better").sum() >= max(qualifying_count, 1):
        status = "stock_cleaner_under_most_paths"
    summary = pd.DataFrame(
        [
            {
                "ticker": clean_string(ticker).upper(),
                "candidate_slug": selected_slug,
                "candidate_label": candidate_label,
                "candidate_short_label": candidate_short_label,
                "strike_label": clean_string(candidate.get("strike_label")),
                "expiry_date": clean_string(candidate.get("expiry_date")) or (expiry_dt.isoformat() if expiry_dt else ""),
                "premium_used": round(float(premium_used), 4),
                "entry_price_mode": normalized_entry_mode,
                "premium_source": premium_source,
                "base_iv": round(float(base_iv), 4) if base_iv is not None else None,
                "breakeven": round(float(breakeven), 4) if breakeven is not None else None,
                "max_loss": round(float(max_loss), 4),
                "dte": int(dte) if dte is not None else None,
                "exit_rule": normalized_exit_rule,
                "single_option_target_return_pct": float(target_return_pct),
                "minimum_outperformance_multiple": float(minimum_outperformance_multiple),
                "strong_outperformance_multiple": float(strong_outperformance_multiple),
                "required_winning_path_families": int(required_winning_path_families),
                "single_option_decision_status": status,
                "too_narrow_under_representative_paths": bool(too_narrow),
                "best_path_label": clean_string(best_outcome.get("path_label")),
                "best_path_difference_vs_stock": finite_or_none(best_outcome.get("difference_vs_stock")),
                "worst_path_label": clean_string(worst_outcome.get("path_label")),
                "worst_path_difference_vs_stock": finite_or_none(worst_outcome.get("difference_vs_stock")),
                "source_trust_label": clean_string(candidate.get("source_trust_label")),
            }
        ]
    )

    bullet_texts = [
        f"Beats stock in {qualifying_count} of {evaluated_count} representative path families; threshold is {int(required_winning_path_families)}.",
        f"Best path: {clean_string(best_outcome.get('path_label'))} ({_format_gap(best_outcome.get('difference_vs_stock'))} vs stock).",
        f"Weakest path: {clean_string(worst_outcome.get('path_label'))} ({_format_gap(worst_outcome.get('difference_vs_stock'))} vs stock).",
        "Setup is too narrow under the current path test." if too_narrow else "Setup clears the multiple-path test under current assumptions.",
        f"Entry becomes less attractive above about ${float(premium_used) * 1.10:,.2f}.",
        "Low IV scenario should be checked before buying; IV support does not replace stock-path confirmation.",
        "Hero chart compares against long stock, not against zero profit.",
    ]
    bullets = pd.DataFrame(
        [
            {
                "candidate_slug": selected_slug,
                "bullet_order": idx,
                "bullet_type": "decision_read",
                "bullet_text": text,
            }
            for idx, text in enumerate(bullet_texts, start=1)
        ]
    )
    markdown = _build_single_option_decision_markdown(
        ticker=ticker,
        summary=summary,
        family_counts=family_counts,
        bullets=bullets,
    )
    return {
        "single_option_decision_summary": summary,
        "single_option_decision_path_selections": decision_path_selections,
        "single_option_representative_paths": representative_paths,
        "single_option_path_outcomes": path_outcomes,
        "single_option_required_path_to_beat_stock_1_5x": required_edge_1_5x,
        "single_option_required_path_to_beat_stock_2_0x": required_edge_2_0x,
        "single_option_closest_representative_path_to_edge": closest_to_edge,
        "single_option_edge_gap_by_path_family": edge_gap_by_path_family,
        "single_option_path_family_counts": family_counts,
        "single_option_timing_sensitivity": timing_sensitivity,
        "single_option_iv_sensitivity": iv_sensitivity,
        "single_option_entry_sensitivity": entry_sensitivity,
        "single_option_summary_bullets": bullets,
        "single_option_decision_markdown": markdown,
    }


def _chain_overview_empty_outputs() -> dict[str, Any]:
    return {
        "chain_overview_summary": pd.DataFrame(),
        "chain_overview_candidates": pd.DataFrame(),
        "chain_overview_markdown": "",
    }


def _chain_metric_band(score: float, *, positive: bool) -> str:
    clamped = max(0.0, min(float(score), 100.0))
    if positive:
        if clamped >= 78:
            return "High"
        if clamped >= 56:
            return "Moderate"
        return "Low"
    if clamped >= 72:
        return "High"
    if clamped >= 48:
        return "Moderate"
    return "Low"


def _chain_path_metric(frame: pd.DataFrame, path_roles: set[str]) -> float:
    if frame.empty:
        return -9999.0
    data = frame.loc[frame.get("path_role", pd.Series(dtype=str)).astype(str).isin(path_roles)].copy()
    if data.empty:
        return -9999.0
    values = pd.to_numeric(data.get("difference_vs_stock"), errors="coerce").dropna()
    if values.empty:
        return -9999.0
    return float(values.max())


def _build_chain_overview_markdown(
    *,
    ticker: str,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    representative_path_count: int,
    minimum_outperformance_multiple: float,
    strong_outperformance_multiple: float,
    required_winning_path_families: int,
) -> str:
    if summary.empty and candidates.empty:
        return ""
    lines = [
        f"# {ticker} Chain Overview / Compare Options",
        "",
        "## What This Layer Compares",
        "",
        (
            f"This layer compares bullish long calls against long stock across `{representative_path_count}` shared representative path families. "
            "It is a quick decision surface: robust, selective, too narrow, or simply worse than stock."
        ),
        "",
        "## Verdict Rules",
        "",
        f"- Minimum path-family win threshold: beat stock by at least `{minimum_outperformance_multiple:.1f}x` in `{int(required_winning_path_families)}` or more representative path families.",
        f"- Strong outperformance threshold: `{strong_outperformance_multiple:.1f}x` or better in multiple path families without major IV or entry-price fragility.",
        "- `Robust buy candidate`: multi-path win read with acceptable IV, timing, and entry fragility.",
        "- `Selective / thesis-dependent`: interesting, but still needs the right path, timing, or entry conditions.",
        "- `Too narrow`: path-family support is too concentrated or timing/expiry is too precise.",
        "- `Stock better`: stock remains cleaner than the call under representative paths.",
        "",
        "## Top Card Snapshot",
        "",
    ]
    if not summary.empty:
        for row in summary.to_dict("records"):
            lines.append(
                f"- `{clean_string(row.get('card_label'))}`: `{clean_string(row.get('contract_label')) or 'No clear call'}` - "
                f"{clean_string(row.get('headline_metric')) or 'n/a'}; {clean_string(row.get('headline_note')) or clean_string(row.get('explanation_short'))}."
            )
    else:
        lines.append("- No chain overview summary cards were available for this run.")
    lines.extend(["", "## How To Read The Table", ""])
    if not candidates.empty:
        lines.extend(
            [
                "- `Beats Stock (X/Y)` counts path families where the call actually finishes ahead of stock after premium.",
                "- `Strong Wins` counts path families where the call beats stock by the stronger threshold, not just marginally.",
                "- `Robustness` is higher when a call survives more path and IV stress without needing a perfect entry.",
                "- `IV Sensitivity` and `Entry Sensitivity` are higher when the idea breaks easily on lower IV or a worse fill.",
                "- `Best Fit Path Type` shows the path family where the option looked best relative to stock.",
            ]
        )
    else:
        lines.append("- No bullish long-call candidates were available for the chain overview table.")
    lines.extend(
        [
            "",
            "## Files To Open",
            "",
            "- `charts/chain_overview.png`: six summary cards plus verdict distribution.",
            "- `tables/chain_overview_candidates.csv`: compact compare-options table.",
            "- `tables/chain_overview_summary.csv`: frozen card payload for model outputs and publish.",
        ]
    )
    return "\n".join(lines)


def _build_chain_overview_outputs(
    *,
    ticker: str,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    bullish_action_board: pd.DataFrame,
    required_stock_path_to_buy: pd.DataFrame,
    candidate_tradeoff_matrix: pd.DataFrame,
    candidate_robustness_summary: pd.DataFrame,
    premium_sensitivity_summary: pd.DataFrame,
    timing_slip_summary: pd.DataFrame,
    target_stress_summary: pd.DataFrame,
    snapshot_date: date,
    target_price: float,
    target_date: date,
    target_horizon_label: str,
    entry_spot: float,
    active_iv_path_points: dict[str, float],
    comparison_capital: float,
    minimum_outperformance_multiple: float,
    strong_outperformance_multiple: float,
    required_winning_path_families: int,
) -> dict[str, Any]:
    outputs = _chain_overview_empty_outputs()
    spec_lookup = {
        clean_string(spec.get("candidate_slug")): spec
        for spec in specs
        if clean_string(spec.get("strategy_family")) == "long_call"
    }
    if not spec_lookup or candidate_rows.empty:
        return outputs
    long_calls = candidate_rows.loc[candidate_rows.get("strategy_family", pd.Series(dtype=str)).astype(str).eq("long_call")].copy()
    if long_calls.empty:
        return outputs

    ordered_slugs: list[str] = []
    if bullish_action_board is not None and not bullish_action_board.empty:
        for row in bullish_action_board.to_dict("records"):
            slug = clean_string(row.get("candidate_slug"))
            if slug and slug in spec_lookup and slug not in ordered_slugs:
                ordered_slugs.append(slug)
    if not ordered_slugs:
        for row in _sort_candidate_priority(long_calls).to_dict("records"):
            slug = clean_string(row.get("candidate_slug"))
            if slug and slug in spec_lookup and slug not in ordered_slugs:
                ordered_slugs.append(slug)
    if not ordered_slugs:
        return outputs

    anchor_candidate, _ = _select_single_option_candidate(
        specs=specs,
        candidate_rows=long_calls,
        bullish_action_board=bullish_action_board,
        candidate_slug=None,
    )
    anchor_slug = clean_string((anchor_candidate or {}).get("candidate_slug")) or ordered_slugs[0]
    selected_paths = _select_decision_comparison_paths(
        required_stock_path_to_buy=required_stock_path_to_buy,
        anchor_candidate_slug=anchor_slug,
        snapshot_date=snapshot_date,
        target_date=target_date,
        target_price=float(target_price),
        target_horizon_label=target_horizon_label,
        entry_spot=float(entry_spot),
    )
    if not selected_paths:
        return outputs

    path_grid = _build_path_grid(snapshot_date, target_date)
    active_iv_horizons = [
        {"label": f"{int(row.get('requested_days') or 0)}d", "requested_days": int(row.get("requested_days") or 0)}
        for row in path_grid
    ]
    active_iv_path = _interpolated_path(active_iv_path_points, active_iv_horizons, default_value=0.0)

    tradeoff_lookup = {
        clean_string(row.get("candidate_slug")): row
        for row in candidate_tradeoff_matrix.to_dict("records")
    } if candidate_tradeoff_matrix is not None and not candidate_tradeoff_matrix.empty else {}
    robustness_lookup = {
        clean_string(row.get("candidate_slug")): row
        for row in candidate_robustness_summary.to_dict("records")
    } if candidate_robustness_summary is not None and not candidate_robustness_summary.empty else {}
    action_lookup = {
        clean_string(row.get("candidate_slug")): row
        for row in bullish_action_board.to_dict("records")
    } if bullish_action_board is not None and not bullish_action_board.empty else {}
    candidate_lookup = {
        clean_string(row.get("candidate_slug")): row
        for row in long_calls.to_dict("records")
    }

    candidate_records: list[dict[str, Any]] = []
    candidate_path_cache: dict[str, pd.DataFrame] = {}
    for slug in ordered_slugs:
        candidate = candidate_lookup.get(slug)
        spec = spec_lookup.get(slug)
        if candidate is None or spec is None:
            continue
        position: StrategyPosition = spec["position"]
        premium_used, premium_source = _single_option_entry_premium(position, mode="conservative_mid_plus_slippage")
        candidate_short_label = _single_option_candidate_short_label(candidate)
        _, path_outcomes, _ = _evaluate_candidate_on_decision_paths(
            spec,
            candidate_slug=slug,
            candidate_label=clean_string(candidate.get("candidate_label")),
            candidate_short_label=candidate_short_label,
            selected_paths=selected_paths,
            target_price=float(target_price),
            active_iv_path=active_iv_path,
            comparison_capital=float(comparison_capital),
            premium_used=float(premium_used),
            exit_rule="sell_on_thesis_completion",
            target_return_pct=0.50,
            minimum_outperformance_multiple=float(minimum_outperformance_multiple),
            strong_outperformance_multiple=float(strong_outperformance_multiple),
            include_trace_rows=False,
        )
        if path_outcomes.empty:
            continue
        candidate_path_cache[slug] = path_outcomes.copy()

        tradeoff = tradeoff_lookup.get(slug, {})
        robustness = robustness_lookup.get(slug, {})
        action_row = action_lookup.get(slug, {})

        premium_rows = premium_sensitivity_summary.loc[
            premium_sensitivity_summary.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(slug)
        ].copy() if premium_sensitivity_summary is not None and not premium_sensitivity_summary.empty else pd.DataFrame()
        timing_rows = timing_slip_summary.loc[
            timing_slip_summary.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(slug)
        ].copy() if timing_slip_summary is not None and not timing_slip_summary.empty else pd.DataFrame()
        target_rows = target_stress_summary.loc[
            target_stress_summary.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(slug)
        ].copy() if target_stress_summary is not None and not target_stress_summary.empty else pd.DataFrame()

        beats_stock_count = int(path_outcomes.get("beats_stock", pd.Series(dtype=bool)).fillna(False).sum())
        qualifying_count = int(path_outcomes.get("qualifies_as_winning_path_family", pd.Series(dtype=bool)).fillna(False).sum())
        strong_count = int(path_outcomes.get("qualifies_as_strong_path_family", pd.Series(dtype=bool)).fillna(False).sum())
        total_path_family_count = int(path_outcomes.get("path_role", pd.Series(dtype=str)).nunique())
        best_outcome = path_outcomes.sort_values("difference_vs_stock", ascending=False).iloc[0].to_dict()
        worst_outcome = path_outcomes.sort_values("difference_vs_stock", ascending=True).iloc[0].to_dict()
        best_fit_path_type = clean_string(best_outcome.get("path_label")) or "n/a"

        timing_complete = not timing_rows.empty
        if timing_complete:
            timing_rows = timing_rows.copy()
            timing_rows["scenario_order"] = pd.to_numeric(timing_rows.get("scenario_order"), errors="coerce").fillna(99)
            timing_rows["option_vs_stock_edge_pct"] = pd.to_numeric(timing_rows.get("option_vs_stock_edge_pct"), errors="coerce").fillna(0.0)
            base_timing = timing_rows.loc[timing_rows.get("scenario_name").astype(str).eq("base_timing")]
            base_timing_edge = float(base_timing["option_vs_stock_edge_pct"].iloc[0]) if not base_timing.empty else float(timing_rows["option_vs_stock_edge_pct"].iloc[0])
            worst_timing_edge = float(timing_rows["option_vs_stock_edge_pct"].min())
            timing_drop = max(0.0, base_timing_edge - worst_timing_edge)
            timing_breaks = int(timing_rows.get("bucket_transition", pd.Series(dtype=str)).astype(str).isin({"downgrade", "break"}).sum())
            delay_expiry_hits = int(pd.to_numeric(timing_rows.get("target_beyond_expiry_under_delay"), errors="coerce").fillna(0).astype(bool).sum())
            timing_sensitivity_score = min(100.0, timing_drop * 2.6 + timing_breaks * 18.0 + delay_expiry_hits * 18.0)
        else:
            timing_sensitivity_score = 55.0

        premium_complete = not premium_rows.empty
        if premium_complete:
            premium_rows = premium_rows.copy()
            premium_rows["scenario_order"] = pd.to_numeric(premium_rows.get("scenario_order"), errors="coerce").fillna(99)
            premium_rows["option_vs_stock_edge_pct"] = pd.to_numeric(premium_rows.get("option_vs_stock_edge_pct"), errors="coerce").fillna(0.0)
            premium_rows["max_justified_premium_gap"] = pd.to_numeric(premium_rows.get("max_justified_premium_gap"), errors="coerce").fillna(0.0)
            base_entry = premium_rows.loc[premium_rows.get("scenario_name").astype(str).eq("base")]
            base_entry_edge = float(base_entry["option_vs_stock_edge_pct"].iloc[0]) if not base_entry.empty else float(premium_rows["option_vs_stock_edge_pct"].iloc[0])
            cheapest_edge = float(premium_rows["option_vs_stock_edge_pct"].max())
            richest_edge = float(premium_rows["option_vs_stock_edge_pct"].min())
            improvement = max(0.0, cheapest_edge - base_entry_edge)
            deterioration = max(0.0, base_entry_edge - richest_edge)
            premium_transitions = int(premium_rows.get("bucket_transition", pd.Series(dtype=str)).astype(str).isin({"upgrade", "downgrade", "break"}).sum())
            premium_gap_penalty = 10.0 if float(premium_rows["max_justified_premium_gap"].iloc[0]) < 0 else 0.0
            entry_premium_sensitivity_score = min(100.0, improvement * 2.0 + deterioration * 1.5 + premium_transitions * 10.0 + premium_gap_penalty)
        else:
            entry_premium_sensitivity_score = 55.0

        lower_iv_resilience = finite_or_none(tradeoff.get("lower_iv_resilience_score"))
        high_iv_dependency_rate = finite_or_none(tradeoff.get("high_iv_dependency_rate"))
        iv_down_value_change = abs(_single_option_num(candidate.get("iv_down_value_change"), 0.0))
        iv_complete = lower_iv_resilience is not None or high_iv_dependency_rate is not None or iv_down_value_change > 0
        premium_anchor = max(abs(float(premium_used)), abs(_single_option_num(candidate.get("premium_or_entry_cost"), 1.0)), 1.0)
        iv_sensitivity_score = min(
            100.0,
            max(0.0, 100.0 - float(lower_iv_resilience if lower_iv_resilience is not None else 50.0))
            + 35.0 * float(high_iv_dependency_rate if high_iv_dependency_rate is not None else 0.0)
            + 28.0 * min(1.0, iv_down_value_change / premium_anchor),
        )

        robustness_score = float(finite_or_none(tradeoff.get("robustness_score")) or finite_or_none(action_row.get("robustness_score")) or 0.0)
        asymmetry_score = float(
            finite_or_none(tradeoff.get("aggressive_upside_score"))
            or finite_or_none(action_row.get("upside_score"))
            or finite_or_none(tradeoff.get("upside_score"))
            or 0.0
        )

        strong_status = (
            strong_count >= 2
            and qualifying_count >= int(required_winning_path_families)
            and timing_sensitivity_score < 58.0
            and iv_sensitivity_score < 58.0
            and entry_premium_sensitivity_score < 64.0
            and timing_complete
            and premium_complete
            and iv_complete
        )
        worth_buying = (
            qualifying_count >= int(required_winning_path_families)
            and beats_stock_count >= int(required_winning_path_families)
            and timing_sensitivity_score < 70.0
            and iv_sensitivity_score < 72.0
            and entry_premium_sensitivity_score < 76.0
            and timing_complete
            and premium_complete
            and iv_complete
        )
        too_narrow = (
            (qualifying_count > 0 and qualifying_count < int(required_winning_path_families))
            or beats_stock_count == 1
            or timing_sensitivity_score >= 78.0
            or (
                _chain_path_metric(path_outcomes, {"early_rally_path"}) > _chain_path_metric(path_outcomes, {"late_rally_path", "steady_grind_up_path", "recovery_path"}) + 60.0
            )
        )
        stock_better = (
            beats_stock_count == 0
            or (
                qualifying_count == 0
                and clean_string(candidate.get("stock_benchmark_decision")) == "stock_still_better"
            )
            or (
                qualifying_count == 0
                and clean_string(action_row.get("action_bucket")) == "Prefer Stock Instead"
            )
        )
        if strong_status:
            worth_buying_status = "strong"
            final_verdict = CHAIN_OVERVIEW_VERDICT_LABELS["robust_buy_candidate"]
        elif worth_buying:
            worth_buying_status = "worth_buying"
            final_verdict = CHAIN_OVERVIEW_VERDICT_LABELS["robust_buy_candidate"]
        elif stock_better:
            worth_buying_status = "stock_better"
            final_verdict = CHAIN_OVERVIEW_VERDICT_LABELS["stock_better"]
        elif too_narrow:
            worth_buying_status = "too_narrow"
            final_verdict = CHAIN_OVERVIEW_VERDICT_LABELS["too_narrow"]
        else:
            worth_buying_status = "selective"
            final_verdict = CHAIN_OVERVIEW_VERDICT_LABELS["selective_thesis_dependent"]

        reason_parts: list[str] = []
        if final_verdict == CHAIN_OVERVIEW_VERDICT_LABELS["robust_buy_candidate"]:
            reason_parts.append(
                "Best under early breakout and steady grind paths."
                if _chain_path_metric(path_outcomes, {"early_rally_path", "steady_grind_up_path"}) >= _chain_path_metric(path_outcomes, {"late_rally_path", "recovery_path"})
                else "Wins across more than one representative path family."
            )
        elif final_verdict == CHAIN_OVERVIEW_VERDICT_LABELS["stock_better"]:
            reason_parts.append("Stock remains cleaner under representative paths.")
        elif final_verdict == CHAIN_OVERVIEW_VERDICT_LABELS["too_narrow"]:
            reason_parts.append("Needs a narrower path or more precise timing than a robust buy should.")
        else:
            reason_parts.append("Has upside, but still needs the right path, timing, or entry.")
        if iv_sensitivity_score >= 72.0:
            reason_parts.append("Too IV-sensitive for a robust buy read.")
        if entry_premium_sensitivity_score >= 70.0:
            reason_parts.append("Needs a cooler entry premium.")
        if timing_sensitivity_score >= 72.0:
            reason_parts.append("Late rally usually favors stock more than the call.")
        if not reason_parts:
            reason_parts.append(clean_string(action_row.get("headline_reason")) or clean_string(candidate.get("why_this_candidate_wins")))
        explanation_short = clean_string(reason_parts[0]) or "Path-family evidence was limited."
        explanation_detail = " ".join(dict.fromkeys(part for part in reason_parts if clean_string(part)))

        candidate_records.append(
            {
                "candidate_slug": slug,
                "contract": candidate_short_label,
                "candidate_label": clean_string(candidate.get("candidate_label")),
                "premium": round(float(premium_used), 4),
                "premium_source": premium_source,
                "iv": round(float(finite_or_none(position.option_legs[0].base_iv if position.option_legs else None) or 0.0), 4),
                "dte": int(max((position.expiry_date - snapshot_date).days, 0) if position.expiry_date else 0),
                "beats_stock_label": f"{beats_stock_count}/{total_path_family_count}",
                "beats_stock_count": beats_stock_count,
                "qualifying_path_family_count": qualifying_count,
                "total_path_family_count": total_path_family_count,
                "strong_wins": strong_count,
                "strong_outperformance_count": strong_count,
                "timing_sensitivity": _chain_metric_band(timing_sensitivity_score, positive=False),
                "timing_sensitivity_score": round(timing_sensitivity_score, 2),
                "iv_sensitivity": _chain_metric_band(iv_sensitivity_score, positive=False),
                "iv_sensitivity_score": round(iv_sensitivity_score, 2),
                "entry_sensitivity": _chain_metric_band(entry_premium_sensitivity_score, positive=False),
                "entry_premium_sensitivity_score": round(entry_premium_sensitivity_score, 2),
                "expiry_sensitivity_summary": (
                    "Short expiry / precise timing"
                    if bool(candidate.get("target_beyond_expiry")) or bool(candidate.get("weak_horizon_fit")) or timing_sensitivity_score >= 78.0
                    else "Longer expiry is more forgiving"
                    if int(max((position.expiry_date - snapshot_date).days, 0) if position.expiry_date else 0) >= 180
                    else "Moderate expiry sensitivity"
                ),
                "robustness": _chain_metric_band(robustness_score, positive=True),
                "robustness_score": round(robustness_score, 2),
                "asymmetry_score": round(asymmetry_score, 2),
                "worth_buying": bool(worth_buying_status in {"strong", "worth_buying"}),
                "worth_buying_status": worth_buying_status,
                "final_verdict": final_verdict,
                "best_fit_path_type": best_fit_path_type,
                "why_short": explanation_short,
                "why_detail": explanation_detail,
                "source_trust_label": clean_string(candidate.get("source_trust_label")),
                "difference_vs_stock": finite_or_none(candidate.get("difference_vs_stock")),
                "return_on_comparison_capital": finite_or_none(candidate.get("return_on_comparison_capital")),
                "best_path_label": clean_string(best_outcome.get("path_label")),
                "best_path_difference_vs_stock": finite_or_none(best_outcome.get("difference_vs_stock")),
                "worst_path_label": clean_string(worst_outcome.get("path_label")),
                "worst_path_difference_vs_stock": finite_or_none(worst_outcome.get("difference_vs_stock")),
                "minimum_outperformance_multiple": float(minimum_outperformance_multiple),
                "strong_outperformance_multiple": float(strong_outperformance_multiple),
                "required_winning_path_families": int(required_winning_path_families),
                "shared_path_family_count": len(selected_paths),
                "shared_path_anchor_candidate": anchor_slug,
            }
        )

    if not candidate_records:
        return outputs

    candidates = pd.DataFrame(candidate_records)
    verdict_order = {
        CHAIN_OVERVIEW_VERDICT_LABELS["robust_buy_candidate"]: 0,
        CHAIN_OVERVIEW_VERDICT_LABELS["selective_thesis_dependent"]: 1,
        CHAIN_OVERVIEW_VERDICT_LABELS["too_narrow"]: 2,
        CHAIN_OVERVIEW_VERDICT_LABELS["stock_better"]: 3,
    }
    candidates["_verdict_order"] = candidates.get("final_verdict", pd.Series(dtype=str)).map(verdict_order).fillna(99)
    candidates = candidates.sort_values(
        ["_verdict_order", "robustness_score", "asymmetry_score", "beats_stock_count"],
        ascending=[True, False, False, False],
    ).drop(columns=["_verdict_order"], errors="ignore").reset_index(drop=True)

    def card_row(
        *,
        key: str,
        label: str,
        row: dict[str, Any] | None,
        fallback_metric: str,
        fallback_note: str,
    ) -> dict[str, Any]:
        if not row:
            return {
                "card_key": key,
                "card_label": label,
                "candidate_slug": "",
                "contract_label": "No clear call",
                "verdict_badge": CHAIN_OVERVIEW_VERDICT_LABELS["stock_better"],
                "headline_metric": fallback_metric,
                "headline_note": fallback_note,
                "explanation_short": fallback_note,
            }
        return {
            "card_key": key,
            "card_label": label,
            "candidate_slug": clean_string(row.get("candidate_slug")),
            "contract_label": clean_string(row.get("contract")),
            "verdict_badge": clean_string(row.get("final_verdict")),
            "headline_metric": fallback_metric,
            "headline_note": fallback_note,
            "explanation_short": clean_string(row.get("why_short")),
        }

    robust_pool = candidates.loc[candidates.get("final_verdict").astype(str).eq(CHAIN_OVERVIEW_VERDICT_LABELS["robust_buy_candidate"])].copy()
    best_robust = robust_pool.sort_values(["robustness_score", "asymmetry_score"], ascending=[False, False]).iloc[0].to_dict() if not robust_pool.empty else None
    asym_pool = candidates.sort_values(["asymmetry_score", "robustness_score"], ascending=[False, False])
    best_asym = asym_pool.iloc[0].to_dict() if not asym_pool.empty else None
    early_candidate = None
    late_candidate = None
    best_early_score = -9999.0
    best_late_score = -9999.0
    for row in candidates.to_dict("records"):
        outcomes = candidate_path_cache.get(clean_string(row.get("candidate_slug")), pd.DataFrame())
        early_score = _chain_path_metric(outcomes, {"early_rally_path", "earnings_gap_path"})
        late_score = max(
            _chain_path_metric(outcomes, {"late_rally_path", "steady_grind_up_path"}),
            _chain_path_metric(outcomes, {"recovery_path"}),
        )
        if early_score > best_early_score:
            best_early_score = early_score
            early_candidate = row
        if late_score > best_late_score:
            best_late_score = late_score
            late_candidate = row
    iv_sensitive = candidates.sort_values(["iv_sensitivity_score", "asymmetry_score"], ascending=[False, False]).iloc[0].to_dict()
    stock_better_count = int(candidates.get("final_verdict", pd.Series(dtype=str)).astype(str).eq(CHAIN_OVERVIEW_VERDICT_LABELS["stock_better"]).sum())
    summary_rows = [
        card_row(
            key="best_robust_option",
            label="Best Robust Option",
            row=best_robust,
            fallback_metric=f"{0 if best_robust is None else int(best_robust.get('qualifying_path_family_count') or 0)}/{len(selected_paths)} wins",
            fallback_note=(
                "No call cleared the robust multi-path threshold."
                if best_robust is None
                else f"{int(best_robust.get('qualifying_path_family_count') or 0)}/{len(selected_paths)} qualifying path families."
            ),
        ),
        card_row(
            key="best_asymmetric_upside",
            label="Best Asymmetric Upside",
            row=best_asym,
            fallback_metric=f"Asymmetry {int(float(best_asym.get('asymmetry_score') or 0.0))}" if best_asym else "Asymmetry n/a",
            fallback_note=clean_string(best_asym.get("why_short")) if best_asym else "No bullish long call was available.",
        ),
        card_row(
            key="best_early_move_option",
            label="Best Early-Move Option",
            row=early_candidate,
            fallback_metric=_format_gap(best_early_score),
            fallback_note=(
                "Best under the early-breakout / earnings-gap paths."
                if early_candidate
                else "No early-move comparison was available."
            ),
        ),
        card_row(
            key="best_late_move_option",
            label="Best Late-Move Option",
            row=late_candidate,
            fallback_metric=_format_gap(best_late_score),
            fallback_note=(
                "Best under the late-rally / steady-grind / recovery paths."
                if late_candidate
                else "No late-move comparison was available."
            ),
        ),
        card_row(
            key="too_iv_sensitive",
            label="Too IV-Sensitive",
            row=iv_sensitive,
            fallback_metric=f"IV sensitivity {int(float(iv_sensitive.get('iv_sensitivity_score') or 0.0))}" if iv_sensitive else "IV n/a",
            fallback_note=clean_string(iv_sensitive.get("why_short")) if iv_sensitive else "No IV-sensitivity comparison was available.",
        ),
        {
            "card_key": "stock_better_than_these_calls",
            "card_label": "Stock Better Than These Calls",
            "candidate_slug": "long-stock-baseline",
            "contract_label": "Long Stock Baseline",
            "verdict_badge": CHAIN_OVERVIEW_VERDICT_LABELS["stock_better"],
            "headline_metric": f"{stock_better_count}/{len(candidates.index)} calls",
            "headline_note": "Stock stays cleaner than most calls under representative paths." if stock_better_count else "At least one call cleared the stock benchmark often enough to matter.",
            "explanation_short": "Stock remains the explicit benchmark; the call table only shows where options truly justify extra complexity.",
        },
    ]
    summary = pd.DataFrame(summary_rows)
    markdown = _build_chain_overview_markdown(
        ticker=ticker,
        summary=summary,
        candidates=candidates,
        representative_path_count=len(selected_paths),
        minimum_outperformance_multiple=float(minimum_outperformance_multiple),
        strong_outperformance_multiple=float(strong_outperformance_multiple),
        required_winning_path_families=int(required_winning_path_families),
    )
    return {
        "chain_overview_summary": summary,
        "chain_overview_candidates": candidates,
        "chain_overview_markdown": markdown,
    }


def _success_status_label(
    *,
    profit_loss: float | None,
    return_on_comparison_capital: float | None,
    difference_vs_stock: float | None,
    goal_reached: bool,
) -> str:
    if not goal_reached:
        if profit_loss is not None and profit_loss <= -100:
            return "misses_badly"
        return "almost_works"
    if difference_vs_stock is not None and difference_vs_stock < 0:
        return "just_works"
    if return_on_comparison_capital is not None and return_on_comparison_capital >= 0.50:
        return "works_very_well"
    if return_on_comparison_capital is not None and return_on_comparison_capital >= 0.25:
        return "works_well"
    return "just_works"


def _build_path_simulation_outputs(
    *,
    specs: list[dict[str, Any]],
    candidate_rows: pd.DataFrame,
    required_path_summary: pd.DataFrame,
    family_representatives: dict[str, str],
    snapshot_date: date,
    target_date: date,
    target_horizon_days: int,
    target_price: float,
    stock_path_name: str,
    stock_path_points: dict[str, float],
    stock_path_mode: str,
    stock_path_target_end: float,
    iv_path_name: str,
    iv_path_points: dict[str, float],
    iv_path_mode: str,
    goal: str,
    target_option_value: float | None,
    comparison_capital: float,
    simulated_path_count: int,
    representative_selection_mode: str,
    simulation_seed: int | None,
    top_candidate_slug: str,
) -> dict[str, pd.DataFrame]:
    empty_outputs = {
        "stock_path_examples": pd.DataFrame(),
        "iv_path_examples": pd.DataFrame(),
        "path_pair_summary": pd.DataFrame(),
        "option_value_over_path": pd.DataFrame(),
        "compare_vs_stock_over_path": pd.DataFrame(),
        "representative_paths_summary": pd.DataFrame(),
        "strike_comparison_under_path": pd.DataFrame(),
        "expiry_comparison_under_path": pd.DataFrame(),
        "required_vs_assumed_path_summary": pd.DataFrame(),
        "simulation_context": {},
    }
    if not specs or candidate_rows.empty:
        return empty_outputs

    top_candidate_spec = next(
        (spec for spec in specs if clean_string(spec.get("candidate_slug")) == clean_string(top_candidate_slug)),
        specs[0] if specs else None,
    )
    if top_candidate_spec is None:
        return empty_outputs

    latest_date = _terminal_simulation_date(specs, target_date=target_date)
    path_grid = _build_path_grid(snapshot_date, latest_date)
    seed = simulation_seed if simulation_seed is not None else _stable_simulation_seed(
        snapshot_date.isoformat(),
        target_date.isoformat(),
        target_price,
        goal,
        stock_path_name,
        iv_path_name,
        top_candidate_slug,
    )
    rng = np.random.default_rng(seed)
    entry_spot = float(next(iter(stock_path_points.values()), candidate_rows.iloc[0].get("spot_price") or target_price))

    stock_paths = [
        _build_stock_path_from_named_points(
            path_grid,
            named_points=stock_path_points,
            path_id="assumed-stock-path",
            path_name=stock_path_name,
            entry_spot=entry_spot,
        )
    ]
    stock_paths.extend(
        _build_stock_path_pool(
            path_grid,
            entry_spot=entry_spot,
            target_end=float(stock_path_target_end or target_price),
            mode=stock_path_mode,
            simulated_path_count=max(int(simulated_path_count), 4),
            rng=rng,
        )
    )

    required_target_row = required_path_summary.loc[
        (required_path_summary.get("summary_scope") == "candidate")
        & (required_path_summary.get("candidate_slug").astype(str) == clean_string(top_candidate_slug))
        & (required_path_summary.get("goal").astype(str) == clean_string(goal))
        & (required_path_summary.get("iv_variant").astype(str) == clean_string(iv_path_name))
    ].copy()
    required_target_price = finite_or_none(required_target_row.iloc[0].get("required_stock_price_at_target")) if not required_target_row.empty else None
    if required_target_price is not None:
        for index, multiplier in enumerate([0.88, 0.98, 1.04, 1.16], start=1):
            stock_paths.append(
                _build_stock_path_example(
                    path_grid,
                    entry_spot=entry_spot,
                    mode="conditioned",
                    target_end=float(required_target_price) * multiplier,
                    annualized_vol=0.50,
                    rng=rng,
                    path_id=f"conditioned-required-{index:02d}",
                    cross_level=float(required_target_price),
                    cross_behavior="cross_early_then_revert" if index % 2 == 0 else None,
                )
            )
    stock_paths_by_id = {clean_string(path.path_id): path for path in stock_paths if clean_string(path.path_id)}
    stock_paths = list(stock_paths_by_id.values())

    iv_paths = [
        _build_iv_path_from_named_points(
            path_grid,
            named_points=iv_path_points,
            iv_path_id="active-iv-path",
            iv_path_name=iv_path_name,
            base_iv_shift=float(next(iter(iv_path_points.values()), 0.0)),
        )
    ]
    if clean_string(iv_path_mode) in {"presets", "mixed", "noisy"}:
        for name in [
            "flat",
            "iv_up_then_down",
            "iv_down_then_stays_low",
            "earnings_build_then_crush",
            "mean_reversion_lower",
            "mean_reversion_higher",
        ]:
            iv_paths.append(
                _build_iv_path_example(
                    path_grid,
                    base_iv_shift=float(next(iter(iv_path_points.values()), 0.0)),
                    mode=name,
                    rng=rng,
                    iv_path_id=f"preset-{name}",
                )
            )
    if clean_string(iv_path_mode) in {"noisy", "mixed"}:
        iv_paths.append(
            _build_iv_path_example(
                path_grid,
                base_iv_shift=float(next(iter(iv_path_points.values()), 0.0)),
                mode=iv_path_name,
                rng=rng,
                iv_path_id="noisy-active",
                noisy=True,
                stock_path_points=stock_paths[0].path_points if stock_paths else None,
            )
        )
    iv_paths_by_id = {clean_string(path.iv_path_id): path for path in iv_paths if clean_string(path.iv_path_id)}
    iv_paths = list(iv_paths_by_id.values())

    path_pairs = _pair_stock_and_iv_paths(stock_paths, iv_paths)
    candidate_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in candidate_rows.iterrows()
        if clean_string(row.get("candidate_slug"))
    }
    representative_specs = [
        spec
        for spec in specs
        if clean_string(spec.get("candidate_slug")) in {clean_string(top_candidate_slug), *[clean_string(value) for value in family_representatives.values()]}
    ]
    if not representative_specs:
        representative_specs = [top_candidate_spec]

    path_outcomes: dict[str, dict[str, Any]] = {}
    for pair in path_pairs:
        terminal_stock = float(pair.stock_points[min(len(pair.stock_points), len(pair.iv_points)) - 1]["spot_price"])
        terminal_iv = float(pair.iv_points[min(len(pair.stock_points), len(pair.iv_points)) - 1]["iv_shift_points"])
        top_evaluation = _evaluate_at_point(
            top_candidate_spec,
            spot_price=terminal_stock,
            horizon_days=int(target_horizon_days),
            iv_shift_points=terminal_iv,
            comparison_capital=float(comparison_capital),
        )
        family_goal_hits: list[bool] = []
        for spec in representative_specs:
            evaluation = _evaluate_at_point(
                spec,
                spot_price=terminal_stock,
                horizon_days=int(target_horizon_days),
                iv_shift_points=terminal_iv,
                comparison_capital=float(comparison_capital),
            )
            family_goal_hits.append(_goal_reached(evaluation, goal=goal, target_option_value=target_option_value))
        path_outcomes[pair.path_pair_id] = {
            "final_profit_loss": top_evaluation.get("profit_loss"),
            "goal_reached": _goal_reached(top_evaluation, goal=goal, target_option_value=target_option_value),
            "outperformed_stock": float(finite_or_none(top_evaluation.get("difference_vs_stock")) or -1.0) >= 0.0,
            "crossed_key_level": (
                max(float(point.get("spot_price") or 0.0) for point in pair.stock_points) >= float(required_target_price)
                if required_target_price is not None
                else max(float(point.get("spot_price") or 0.0) for point in pair.stock_points) >= float(target_price)
            ),
            "goal_success_rate": round(float(sum(1 for value in family_goal_hits if value)) / float(len(family_goal_hits)), 4) if family_goal_hits else None,
            "final_difference_vs_stock": top_evaluation.get("difference_vs_stock"),
            "terminal_stock_price": terminal_stock,
            "terminal_iv_shift_points": terminal_iv,
        }
    selected_pairs = (
        _select_representative_path_pairs(path_pairs, path_outcomes=path_outcomes)
        if clean_string(representative_selection_mode) == "goal_buckets"
        else []
    )
    if not selected_pairs and path_pairs:
        selected_pairs = [
            {
                "path_pair_id": path_pairs[0].path_pair_id,
                "stock_path_id": path_pairs[0].stock_path_id,
                "iv_path_id": path_pairs[0].iv_path_id,
                "stock_path_name": path_pairs[0].stock_path_name,
                "iv_path_name": path_pairs[0].iv_path_name,
                "representative_bucket": "just_works",
                "selection_reason": "Fallback to the first generated path pair because no representative buckets were available.",
                "final_profit_loss": path_outcomes.get(path_pairs[0].path_pair_id, {}).get("final_profit_loss"),
            }
        ]
    selected_lookup = {clean_string(item.get("path_pair_id")): item for item in selected_pairs if clean_string(item.get("path_pair_id"))}
    selected_stock_lookup = {
        clean_string(item.get("stock_path_id")): item
        for item in selected_pairs
        if clean_string(item.get("stock_path_id"))
    }
    selected_iv_lookup = {
        clean_string(item.get("iv_path_id")): item
        for item in selected_pairs
        if clean_string(item.get("iv_path_id"))
    }

    stock_path_records: list[StockPathExampleRecord] = []
    for stock_path in stock_paths:
        selected_meta = selected_stock_lookup.get(clean_string(stock_path.path_id)) or {}
        for point in stock_path.path_points:
            stock_path_records.append(
                StockPathExampleRecord(
                    path_id=clean_string(stock_path.path_id),
                    path_kind=clean_string(stock_path.path_kind),
                    path_name=clean_string(stock_path.path_name),
                    representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                    selection_reason=clean_string(selected_meta.get("selection_reason")) or "Generated stock path example.",
                    is_representative=bool(selected_meta),
                    date=clean_string(point.get("date")),
                    requested_days=int(point.get("requested_days") or 0),
                    step_index=int(point.get("step_index") or 0),
                    spot_price=float(point.get("spot_price") or 0.0),
                    return_pct=finite_or_none(point.get("return_pct")),
                )
            )
    iv_path_records: list[IVPathExampleRecord] = []
    for iv_path in iv_paths:
        selected_meta = selected_iv_lookup.get(clean_string(iv_path.iv_path_id)) or {}
        for point in iv_path.path_points:
            iv_path_records.append(
                IVPathExampleRecord(
                    iv_path_id=clean_string(iv_path.iv_path_id),
                    iv_path_name=clean_string(iv_path.iv_path_name),
                    representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                    selection_reason=clean_string(selected_meta.get("selection_reason")) or "Generated IV path example.",
                    is_representative=bool(selected_meta),
                    date=clean_string(point.get("date")),
                    requested_days=int(point.get("requested_days") or 0),
                    step_index=int(point.get("step_index") or 0),
                    iv_shift_points=float(point.get("iv_shift_points") or 0.0),
                )
            )
    pair_summary_records: list[PathPairSummaryRecord] = []
    for pair in path_pairs:
        outcome = path_outcomes.get(pair.path_pair_id) or {}
        selected_meta = selected_lookup.get(clean_string(pair.path_pair_id)) or {}
        pair_summary_records.append(
            PathPairSummaryRecord(
                path_pair_id=clean_string(pair.path_pair_id),
                stock_path_id=clean_string(pair.stock_path_id),
                iv_path_id=clean_string(pair.iv_path_id),
                stock_path_name=clean_string(pair.stock_path_name),
                iv_path_name=clean_string(pair.iv_path_name),
                stock_path_kind=clean_string(pair.stock_path_kind),
                representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                selection_reason=clean_string(selected_meta.get("selection_reason")) or "Generated path-pair candidate.",
                is_representative=bool(selected_meta),
                terminal_stock_price=finite_or_none(outcome.get("terminal_stock_price")),
                terminal_iv_shift_points=finite_or_none(outcome.get("terminal_iv_shift_points")),
                final_profit_loss=finite_or_none(outcome.get("final_profit_loss")),
                final_difference_vs_stock=finite_or_none(outcome.get("final_difference_vs_stock")),
                goal_reached=bool(outcome.get("goal_reached")),
                outperformed_stock=bool(outcome.get("outperformed_stock")),
                goal_success_rate=finite_or_none(outcome.get("goal_success_rate")),
            )
        )

    option_value_records: list[ValuationOverPathRecord] = []
    compare_records: list[CompareVsStockOverPathRecord] = []
    selected_pair_objects = [pair for pair in path_pairs if clean_string(pair.path_pair_id) in selected_lookup]
    if not selected_pair_objects and path_pairs:
        selected_pair_objects = [path_pairs[0]]
    for pair in selected_pair_objects:
        selected_meta = selected_lookup.get(clean_string(pair.path_pair_id)) or {}
        point_count = min(len(pair.stock_points), len(pair.iv_points))
        for spec in specs:
            candidate_slug = clean_string(spec.get("candidate_slug"))
            candidate_meta = candidate_lookup.get(candidate_slug, {})
            running_worst: float | None = None
            running_peak: float | None = None
            running_best: float | None = None
            for step_index in range(point_count):
                stock_point = pair.stock_points[step_index]
                iv_point = pair.iv_points[step_index]
                evaluation = _evaluate_at_point(
                    spec,
                    spot_price=float(stock_point.get("spot_price") or 0.0),
                    horizon_days=int(stock_point.get("requested_days") or 0),
                    iv_shift_points=float(iv_point.get("iv_shift_points") or 0.0),
                    comparison_capital=float(comparison_capital),
                )
                profit_loss = finite_or_none(evaluation.get("profit_loss"))
                if profit_loss is not None:
                    running_worst = profit_loss if running_worst is None else min(running_worst, profit_loss)
                    running_peak = profit_loss if running_peak is None else max(running_peak, profit_loss)
                    running_best = profit_loss if running_best is None else max(running_best, profit_loss)
                difference_vs_stock = finite_or_none(evaluation.get("difference_vs_stock"))
                return_on_capital = finite_or_none(evaluation.get("return_on_comparison_capital"))
                goal_reached = _goal_reached(evaluation, goal=goal, target_option_value=target_option_value)
                success_status = _success_status_label(
                    profit_loss=profit_loss,
                    return_on_comparison_capital=return_on_capital,
                    difference_vs_stock=difference_vs_stock,
                    goal_reached=goal_reached,
                )
                return_delta = (
                    round(
                        float(return_on_capital) - float(finite_or_none(evaluation.get("stock_return_on_comparison_capital"))),
                        6,
                    )
                    if return_on_capital is not None and finite_or_none(evaluation.get("stock_return_on_comparison_capital")) is not None
                    else None
                )
                option_value_records.append(
                    ValuationOverPathRecord(
                        path_pair_id=clean_string(pair.path_pair_id),
                        representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                        selection_reason=clean_string(selected_meta.get("selection_reason")) or "Generated representative path pair.",
                        path_scope="representative_path_pair",
                        candidate_slug=candidate_slug,
                        candidate_label=clean_string(spec.get("candidate_label")),
                        strategy_family=clean_string(spec.get("strategy_family")),
                        expiry_date=clean_string(candidate_meta.get("expiry_date")),
                        strike_label=clean_string(candidate_meta.get("strike_label")),
                        date=clean_string(stock_point.get("date")),
                        requested_days=int(stock_point.get("requested_days") or 0),
                        step_index=int(stock_point.get("step_index") or 0),
                        spot_price=float(stock_point.get("spot_price") or 0.0),
                        iv_shift_points=float(iv_point.get("iv_shift_points") or 0.0),
                        modeled_value=finite_or_none(evaluation.get("estimated_value")),
                        profit_loss=profit_loss,
                        return_on_comparison_capital=return_on_capital,
                        stock_modeled_value=finite_or_none(evaluation.get("stock_estimated_value")),
                        stock_profit_loss=finite_or_none(evaluation.get("stock_profit_loss")),
                        stock_return_on_comparison_capital=finite_or_none(evaluation.get("stock_return_on_comparison_capital")),
                        difference_vs_stock=difference_vs_stock,
                        difference_vs_stock_return_pct=return_delta,
                        benchmark_note=clean_string(evaluation.get("benchmark_note")) or _compare_vs_stock_note(
                            strategy_family=clean_string(spec.get("strategy_family")),
                            difference_vs_stock=difference_vs_stock,
                            difference_vs_stock_return_pct=return_delta,
                            clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                            target_beyond_expiry=bool(evaluation.get("target_beyond_expiry")),
                        ),
                        worst_interim_profit_loss_to_date=running_worst,
                        drawdown_from_peak_to_date=(
                            round(float(profit_loss) - float(running_peak), 4)
                            if profit_loss is not None and running_peak is not None
                            else None
                        ),
                        max_favorable_profit_to_date=running_best,
                        success_status=success_status,
                        goal_reached=goal_reached,
                        outperformed_stock=bool((difference_vs_stock or -1.0) >= 0.0),
                        clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                        target_beyond_expiry=bool(evaluation.get("target_beyond_expiry")),
                    )
                )
                compare_records.append(
                    CompareVsStockOverPathRecord(
                        path_pair_id=clean_string(pair.path_pair_id),
                        representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                        selection_reason=clean_string(selected_meta.get("selection_reason")) or "Generated representative path pair.",
                        candidate_slug=candidate_slug,
                        candidate_label=clean_string(spec.get("candidate_label")),
                        strategy_family=clean_string(spec.get("strategy_family")),
                        date=clean_string(stock_point.get("date")),
                        requested_days=int(stock_point.get("requested_days") or 0),
                        step_index=int(stock_point.get("step_index") or 0),
                        strategy_profit_loss=profit_loss,
                        stock_profit_loss=finite_or_none(evaluation.get("stock_profit_loss")),
                        delta_profit_loss_vs_stock=difference_vs_stock,
                        strategy_return_on_comparison_capital=return_on_capital,
                        stock_return_on_comparison_capital=finite_or_none(evaluation.get("stock_return_on_comparison_capital")),
                        delta_return_pct_vs_stock=return_delta,
                        benchmark_note=clean_string(evaluation.get("benchmark_note")) or _compare_vs_stock_note(
                            strategy_family=clean_string(spec.get("strategy_family")),
                            difference_vs_stock=difference_vs_stock,
                            difference_vs_stock_return_pct=return_delta,
                            clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                            target_beyond_expiry=bool(evaluation.get("target_beyond_expiry")),
                        ),
                    )
                )
    option_value_frame = pd.DataFrame([record.__dict__ for record in option_value_records])
    compare_frame = pd.DataFrame([record.__dict__ for record in compare_records])

    terminal_rows = pd.DataFrame()
    if not option_value_frame.empty:
        within_target = option_value_frame.loc[pd.to_numeric(option_value_frame.get("requested_days"), errors="coerce") <= int(target_horizon_days)].copy()
        if within_target.empty:
            within_target = option_value_frame.copy()
        terminal_rows = (
            within_target.sort_values(["path_pair_id", "candidate_slug", "requested_days", "step_index"])
            .groupby(["path_pair_id", "candidate_slug"], dropna=False, as_index=False)
            .tail(1)
        )

    representative_summary_records: list[RepresentativePathSummaryRecord] = []
    for selected in selected_pairs:
        terminal = _terminal_row_for_candidate(
            terminal_rows,
            candidate_slug=clean_string(top_candidate_slug),
            target_horizon_days=int(target_horizon_days),
        )
        pair_rows = terminal_rows.loc[terminal_rows.get("path_pair_id").astype(str) == clean_string(selected.get("path_pair_id"))].copy()
        top_terminal = pair_rows.loc[pair_rows.get("candidate_slug").astype(str) == clean_string(top_candidate_slug)].copy()
        top_row = top_terminal.iloc[0].to_dict() if not top_terminal.empty else {}
        representative_summary_records.append(
            RepresentativePathSummaryRecord(
                path_pair_id=clean_string(selected.get("path_pair_id")),
                stock_path_id=clean_string(selected.get("stock_path_id")),
                iv_path_id=clean_string(selected.get("iv_path_id")),
                stock_path_name=clean_string(selected.get("stock_path_name")),
                iv_path_name=clean_string(selected.get("iv_path_name")),
                representative_bucket=clean_string(selected.get("representative_bucket")),
                selection_reason=clean_string(selected.get("selection_reason")),
                top_candidate_success_status=clean_string(top_row.get("success_status")),
                stock_benchmark_status=(
                    "stock_dominates"
                    if float(finite_or_none(top_row.get("difference_vs_stock")) or -1.0) < 0.0
                    else "options_show_edge"
                ),
                terminal_stock_price=finite_or_none(top_row.get("spot_price")),
                terminal_iv_shift_points=finite_or_none(top_row.get("iv_shift_points")),
                final_profit_loss=finite_or_none(top_row.get("profit_loss")),
                final_difference_vs_stock=finite_or_none(top_row.get("difference_vs_stock")),
            )
        )

    comparison_records: list[PathComparisonRecord] = []
    if not terminal_rows.empty:
        for path_pair_id, group in terminal_rows.groupby("path_pair_id", dropna=False):
            selected_meta = selected_lookup.get(clean_string(path_pair_id)) or {}
            merged_rows: list[dict[str, Any]] = []
            for row in group.to_dict(orient="records"):
                candidate_meta = candidate_lookup.get(clean_string(row.get("candidate_slug")), {})
                merged_rows.append({**candidate_meta, **row})
            merged = pd.DataFrame(merged_rows)
            if merged.empty:
                continue
            for (family, strike_label), strike_group in merged.groupby(["strategy_family", "strike_label"], dropna=False):
                best = strike_group.sort_values(["objective_score", "profit_loss"], ascending=[False, False]).iloc[0]
                comparison_records.append(
                    PathComparisonRecord(
                        comparison_scope="strike",
                        path_pair_id=clean_string(path_pair_id),
                        representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                        selection_reason=clean_string(selected_meta.get("selection_reason")) or "Representative path pair selected for path-first comparison.",
                        strategy_family=clean_string(family),
                        strike_label=clean_string(strike_label),
                        expiry_date=clean_string(best.get("expiry_date")),
                        best_candidate_label=clean_string(best.get("candidate_label")),
                        objective_score=finite_or_none(best.get("objective_score")),
                        profit_loss=finite_or_none(best.get("profit_loss")),
                        return_on_comparison_capital=finite_or_none(best.get("return_on_comparison_capital")),
                        difference_vs_stock=finite_or_none(best.get("difference_vs_stock")),
                        difference_vs_stock_return_pct=finite_or_none(best.get("difference_vs_stock_return_pct")),
                        benchmark_note=clean_string(best.get("benchmark_note")),
                        required_path_difficulty=clean_string(best.get("required_path_difficulty")),
                        timing_risk=clean_string(best.get("timing_risk")),
                        iv_risk=clean_string(best.get("iv_risk")),
                        success_dependency=clean_string(best.get("success_dependency")),
                        source_trust_label=clean_string(best.get("source_trust_label")),
                        source_quality_note=clean_string(best.get("source_quality_note")),
                        weak_horizon_fit=bool(best.get("weak_horizon_fit")),
                        target_beyond_expiry=bool(best.get("target_beyond_expiry")),
                        clamped_to_expiry=bool(best.get("clamped_to_expiry")),
                    )
                )
            for (family, expiry_date), expiry_group in merged.groupby(["strategy_family", "expiry_date"], dropna=False):
                best = expiry_group.sort_values(["objective_score", "profit_loss"], ascending=[False, False]).iloc[0]
                comparison_records.append(
                    PathComparisonRecord(
                        comparison_scope="expiry",
                        path_pair_id=clean_string(path_pair_id),
                        representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                        selection_reason=clean_string(selected_meta.get("selection_reason")) or "Representative path pair selected for path-first comparison.",
                        strategy_family=clean_string(family),
                        strike_label=clean_string(best.get("strike_label")),
                        expiry_date=clean_string(expiry_date),
                        best_candidate_label=clean_string(best.get("candidate_label")),
                        objective_score=finite_or_none(best.get("objective_score")),
                        profit_loss=finite_or_none(best.get("profit_loss")),
                        return_on_comparison_capital=finite_or_none(best.get("return_on_comparison_capital")),
                        difference_vs_stock=finite_or_none(best.get("difference_vs_stock")),
                        difference_vs_stock_return_pct=finite_or_none(best.get("difference_vs_stock_return_pct")),
                        benchmark_note=clean_string(best.get("benchmark_note")),
                        required_path_difficulty=clean_string(best.get("required_path_difficulty")),
                        timing_risk=clean_string(best.get("timing_risk")),
                        iv_risk=clean_string(best.get("iv_risk")),
                        success_dependency=clean_string(best.get("success_dependency")),
                        source_trust_label=clean_string(best.get("source_trust_label")),
                        source_quality_note=clean_string(best.get("source_quality_note")),
                        weak_horizon_fit=bool(best.get("weak_horizon_fit")),
                        target_beyond_expiry=bool(best.get("target_beyond_expiry")),
                        clamped_to_expiry=bool(best.get("clamped_to_expiry")),
                    )
                )

    required_vs_assumed_records: list[RequiredVsAssumedPathSummaryRecord] = []
    representative_candidate_slugs = list(dict.fromkeys([clean_string(top_candidate_slug), *[clean_string(value) for value in family_representatives.values()]]))
    for selected in selected_pairs:
        selected_meta = selected_lookup.get(clean_string(selected.get("path_pair_id"))) or {}
        for candidate_slug in representative_candidate_slugs:
            required_row = required_path_summary.loc[
                (required_path_summary.get("candidate_slug").astype(str) == clean_string(candidate_slug))
                & (required_path_summary.get("goal").astype(str) == clean_string(goal))
                & (required_path_summary.get("iv_variant").astype(str) == clean_string(iv_path_name))
            ].copy()
            if required_row.empty:
                continue
            required = required_row.iloc[0].to_dict()
            terminal = terminal_rows.loc[
                (terminal_rows.get("path_pair_id").astype(str) == clean_string(selected.get("path_pair_id")))
                & (terminal_rows.get("candidate_slug").astype(str) == clean_string(candidate_slug))
            ].copy()
            terminal_row = terminal.iloc[0].to_dict() if not terminal.empty else {}
            terminal_spot = finite_or_none(terminal_row.get("spot_price"))
            required_target = finite_or_none(required.get("required_stock_price_at_target"))
            required_vs_assumed_records.append(
                RequiredVsAssumedPathSummaryRecord(
                    comparison_scope="representative_path_pair",
                    candidate_slug=clean_string(candidate_slug),
                    candidate_label=clean_string(required.get("candidate_label")),
                    strategy_family=clean_string(required.get("strategy_family")),
                    goal=clean_string(goal),
                    assumed_path_name=clean_string(stock_path_name),
                    representative_path_pair_id=clean_string(selected.get("path_pair_id")),
                    representative_bucket=clean_string(selected_meta.get("representative_bucket")) or "supporting_example",
                    first_cleared_horizon=clean_string(required.get("first_cleared_horizon")) or None,
                    required_path_difficulty=clean_string(required.get("required_path_difficulty")),
                    assumed_path_gap_at_target=finite_or_none(required.get("path_gap_at_target")),
                    representative_path_gap_at_target=(
                        round(float(terminal_spot) - float(required_target), 4)
                        if terminal_spot is not None and required_target is not None
                        else None
                    ),
                    representative_terminal_stock_price=terminal_spot,
                    representative_goal_reached=bool(terminal_row.get("goal_reached")),
                )
            )

    return {
        "stock_path_examples": pd.DataFrame([record.__dict__ for record in stock_path_records]),
        "iv_path_examples": pd.DataFrame([record.__dict__ for record in iv_path_records]),
        "path_pair_summary": pd.DataFrame([record.__dict__ for record in pair_summary_records]),
        "option_value_over_path": option_value_frame,
        "compare_vs_stock_over_path": compare_frame,
        "representative_paths_summary": pd.DataFrame([record.__dict__ for record in representative_summary_records]),
        "strike_comparison_under_path": pd.DataFrame([record.__dict__ for record in comparison_records if record.comparison_scope == "strike"]),
        "expiry_comparison_under_path": pd.DataFrame([record.__dict__ for record in comparison_records if record.comparison_scope == "expiry"]),
        "required_vs_assumed_path_summary": pd.DataFrame([record.__dict__ for record in required_vs_assumed_records]),
        "simulation_context": {
            "stock_path_mode": clean_string(stock_path_mode),
            "stock_path_target_end": float(stock_path_target_end),
            "iv_path_mode": clean_string(iv_path_mode),
            "simulated_path_count": int(simulated_path_count),
            "representative_selection_mode": clean_string(representative_selection_mode),
            "simulation_seed": int(seed),
            "selected_path_pair_ids": [clean_string(item.get("path_pair_id")) for item in selected_pairs],
        },
    }


def _path_case_stock_variants(
    *,
    horizon_specs: list[dict[str, Any]],
    target_price: float,
    target_horizon_label: str,
    stock_path_name: str,
    stock_path_points: dict[str, float],
) -> dict[str, dict[str, float]]:
    entry_spot = float(stock_path_points.get("entry", target_price))
    variants: dict[str, dict[str, float]] = {}
    for preset in ["fast_bull", "slow_bull", "flat"]:
        variants[preset] = _interpolated_path(
            _default_stock_path_points(
                preset=preset,
                entry_spot=entry_spot,
                target_price=target_price,
                target_horizon_label=target_horizon_label,
            ),
            horizon_specs,
            default_value=target_price,
        )
    active_name = clean_string(stock_path_name).lower() or "custom_stock_path"
    variants[active_name] = _interpolated_path(stock_path_points, horizon_specs, default_value=target_price)
    return variants


def _path_case_rows(
    specs: list[dict[str, Any]],
    *,
    horizon_specs: list[dict[str, Any]],
    comparison_capital: float,
    target_price: float,
    target_horizon_label: str,
    iv_shift_points: float,
    stock_path_name: str,
    stock_path_points: dict[str, float],
    iv_path_name: str,
    iv_path_points: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stock_path_variants = _path_case_stock_variants(
        horizon_specs=horizon_specs,
        target_price=target_price,
        target_horizon_label=target_horizon_label,
        stock_path_name=stock_path_name,
        stock_path_points=stock_path_points,
    )
    iv_path_variants = _path_case_iv_variants(
        horizon_specs=horizon_specs,
        iv_shift_points=iv_shift_points,
        target_horizon_label=target_horizon_label,
        iv_path_name=iv_path_name,
        iv_path_points=iv_path_points,
    )["path_preset"]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for stock_name, stock_series in stock_path_variants.items():
            for iv_name, iv_series in iv_path_variants.items():
                for horizon in horizon_specs:
                    label = clean_string(horizon["label"]).lower()
                    spot_value = float(stock_series.get(label, target_price))
                    iv_value = float(iv_series.get(label, iv_shift_points))
                    evaluation = _evaluate_at_point(
                        spec,
                        spot_price=spot_value,
                        horizon_days=int(horizon["requested_days"]),
                        iv_shift_points=iv_value,
                        comparison_capital=comparison_capital,
                    )
                    rows.append(
                        {
                            "candidate_slug": spec["candidate_slug"],
                            "candidate_label": spec["candidate_label"],
                            "strategy_family": spec["strategy_family"],
                            "stock_path": stock_name,
                            "iv_path": iv_name,
                            "horizon": label,
                            "requested_days": int(horizon["requested_days"]),
                            "spot_price": spot_value,
                            "iv_shift_points": iv_value,
                            **evaluation,
                        }
                    )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()
    summary_rows: list[dict[str, Any]] = []
    for (candidate_slug, stock_path, iv_path), group in frame.groupby(["candidate_slug", "stock_path", "iv_path"], dropna=False):
        group = group.sort_values("requested_days")
        terminal = group.iloc[-1].to_dict()
        summary_rows.append(
            {
                "candidate_slug": candidate_slug,
                "candidate_label": terminal.get("candidate_label"),
                "strategy_family": terminal.get("strategy_family"),
                "stock_path": stock_path,
                "iv_path": iv_path,
                "final_horizon": terminal.get("horizon"),
                "final_spot_price": terminal.get("spot_price"),
                "final_iv_shift_points": terminal.get("iv_shift_points"),
                "final_estimated_value": terminal.get("estimated_value"),
                "final_profit_loss": terminal.get("profit_loss"),
                "final_return_on_comparison_capital": terminal.get("return_on_comparison_capital"),
                "final_stock_profit_loss": terminal.get("stock_profit_loss"),
                "final_stock_return_on_comparison_capital": terminal.get("stock_return_on_comparison_capital"),
                "final_difference_vs_stock": terminal.get("difference_vs_stock"),
                "final_difference_vs_stock_return_pct": (
                    round(
                        float(terminal.get("return_on_comparison_capital")) - float(terminal.get("stock_return_on_comparison_capital")),
                        6,
                    )
                    if finite_or_none(terminal.get("return_on_comparison_capital")) is not None
                    and finite_or_none(terminal.get("stock_return_on_comparison_capital")) is not None
                    else None
                ),
                "benchmark_note": _compare_vs_stock_note(
                    strategy_family=clean_string(terminal.get("strategy_family")),
                    difference_vs_stock=finite_or_none(terminal.get("difference_vs_stock")),
                    difference_vs_stock_return_pct=(
                        round(
                            float(terminal.get("return_on_comparison_capital")) - float(terminal.get("stock_return_on_comparison_capital")),
                            6,
                        )
                        if finite_or_none(terminal.get("return_on_comparison_capital")) is not None
                        and finite_or_none(terminal.get("stock_return_on_comparison_capital")) is not None
                        else None
                    ),
                    clamped_to_expiry=bool(terminal.get("clamped_to_expiry")),
                    target_beyond_expiry=bool(terminal.get("target_beyond_expiry")),
                ),
                "worst_interim_profit_loss": pd.to_numeric(group["profit_loss"], errors="coerce").min(),
                "best_interim_profit_loss": pd.to_numeric(group["profit_loss"], errors="coerce").max(),
                "path_points": len(group.index),
            }
        )
    return frame, pd.DataFrame(summary_rows)


def _path_case_definitions(
    *,
    entry_spot: float,
    horizon_specs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    target_label = clean_string(horizon_specs[-1]["label"]).lower() if horizon_specs else "target"
    for move in DEFAULT_PATH_CASE_MOVES:
        label = f"{move:+.0%}" if move else "0%"
        endpoint = float(entry_spot * (1.0 + float(move)))
        points = {"entry": float(entry_spot), target_label: endpoint}
        interpolated = _interpolated_path(points, horizon_specs, default_value=float(entry_spot))
        cases[label] = {
            "case_label": label,
            "case_move_pct": float(move),
            "endpoint_price": round(endpoint, 4),
            "path_shape": "linear_to_endpoint",
            "path_points": {key: round(float(value), 4) for key, value in interpolated.items()},
        }
    return cases


def _path_case_defaults(
    *,
    goal: str,
    active_iv_path_name: str,
    iv_shift_points: float,
    default_strategy_family: str,
    default_candidate_within_family: str,
) -> dict[str, Any]:
    return {
        "default_case_label": "0%",
        "default_goal": clean_string(goal) or DEFAULT_GOAL,
        "default_display_mode": "strategy_compare",
        "default_iv_mode": "path_preset" if active_iv_path_name else "point_scenario",
        "default_strategy_family": default_strategy_family,
        "default_candidate_within_family": default_candidate_within_family,
        "default_iv_variant": active_iv_path_name or f"{float(iv_shift_points):+.2f}",
    }


def _path_case_iv_variants(
    *,
    horizon_specs: list[dict[str, Any]],
    iv_shift_points: float,
    target_horizon_label: str,
    iv_path_name: str,
    iv_path_points: dict[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    point_variants = {
        f"{shift:+.2f}": {clean_string(spec["label"]).lower(): float(shift) for spec in horizon_specs}
        for shift in POINT_IV_SCENARIOS
    }
    path_variants: dict[str, dict[str, float]] = {}
    for preset in [
        "flat",
        "iv_up_then_down",
        "iv_down_then_stays_low",
        "earnings_build_then_crush",
        "mean_reversion_lower",
        "mean_reversion_higher",
    ]:
        path_variants[preset] = _interpolated_path(
            _default_iv_path_points(
                preset=preset,
                base_shift=iv_shift_points,
                target_horizon_label=target_horizon_label,
            ),
            horizon_specs,
            default_value=iv_shift_points,
        )
    active_name = clean_string(iv_path_name).lower() or "custom_iv_path"
    path_variants[active_name] = _interpolated_path(iv_path_points, horizon_specs, default_value=iv_shift_points)
    return {
        "point_scenario": point_variants,
        "path_preset": path_variants,
    }


def _required_path_difficulty(
    case_path: dict[str, float],
    required_rows: pd.DataFrame,
) -> str:
    return _shared_required_path_difficulty(case_path, required_rows)


def _path_case_chart_rows(
    *,
    selector_rows: pd.DataFrame,
    required_path_rows: pd.DataFrame,
    horizon_specs: list[dict[str, Any]],
    assumed_path_points: dict[str, float],
    iv_variants: dict[str, dict[str, dict[str, float]]],
    case_definitions: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if selector_rows.empty or required_path_rows.empty:
        return pd.DataFrame()
    assumed_path = _interpolated_path(
        assumed_path_points,
        horizon_specs,
        default_value=float(next(iter(assumed_path_points.values()), 0.0)),
    )
    winner_map = {
        clean_string(row.get("strategy_family")): clean_string(row.get("winning_candidate_slug"))
        for _, row in selector_rows.iterrows()
        if clean_string(row.get("strategy_family")) and clean_string(row.get("winning_candidate_slug"))
    }
    rows: list[dict[str, Any]] = []
    goals = sorted({clean_string(value) for value in required_path_rows.get("goal", pd.Series(dtype=str)).tolist() if clean_string(value)})
    for case_label, case_info in case_definitions.items():
        for goal in goals:
            for iv_mode, variant_map in iv_variants.items():
                for iv_variant in variant_map.keys():
                    for horizon in horizon_specs:
                        horizon_label = clean_string(horizon["label"]).lower()
                        rows.append(
                            {
                                "case_label": case_label,
                                "case_move_pct": case_info.get("case_move_pct"),
                                "display_mode": "strategy_compare",
                                "series_kind": "assumed_path",
                                "series_label": "Assumed Path",
                                "strategy_family": "",
                                "candidate_slug": "",
                                "goal": goal,
                                "iv_mode": iv_mode,
                                "iv_variant": iv_variant,
                                "horizon": horizon_label,
                                "requested_days": int(horizon["requested_days"]),
                                "spot_price": finite_or_none(assumed_path.get(horizon_label)),
                                "unreached": False,
                                "clamped_to_expiry": False,
                                "target_beyond_expiry": False,
                            }
                        )
                    for family, candidate_slug in winner_map.items():
                        family_rows = required_path_rows.loc[
                            (required_path_rows["candidate_slug"] == candidate_slug)
                            & (required_path_rows["goal"] == goal)
                            & (required_path_rows["iv_variant"] == iv_variant)
                        ].copy()
                        if family_rows.empty:
                            continue
                        variant_kind = clean_string(family_rows.iloc[0].get("iv_variant_kind"))
                        expected_kind = "point" if iv_mode == "point_scenario" else "path"
                        if variant_kind != expected_kind:
                            continue
                        for _, row in family_rows.iterrows():
                            rows.append(
                                {
                                    "case_label": case_label,
                                    "case_move_pct": case_info.get("case_move_pct"),
                                    "display_mode": "strategy_compare",
                                    "series_kind": "required_path",
                                    "series_label": family.replace("_", " ").title(),
                                    "strategy_family": family,
                                    "candidate_slug": candidate_slug,
                                    "goal": goal,
                                    "iv_mode": iv_mode,
                                    "iv_variant": iv_variant,
                                    "horizon": clean_string(row.get("horizon")).lower(),
                                    "requested_days": int(row.get("requested_days") or 0),
                                    "spot_price": finite_or_none(row.get("required_stock_price")),
                                    "unreached": bool(row.get("unreached")),
                                    "clamped_to_expiry": bool(row.get("clamped_to_expiry")),
                                    "target_beyond_expiry": bool(row.get("target_beyond_expiry")),
                                }
                            )
                    for family, candidate_slug in winner_map.items():
                        for horizon in horizon_specs:
                            horizon_label = clean_string(horizon["label"]).lower()
                            matching = required_path_rows.loc[
                                (required_path_rows["candidate_slug"] == candidate_slug)
                                & (required_path_rows["goal"] == goal)
                                & (required_path_rows["iv_variant"] == iv_variant)
                                & (required_path_rows["horizon"] == horizon_label)
                            ].copy()
                            if matching.empty:
                                continue
                            variant_kind = clean_string(matching.iloc[0].get("iv_variant_kind"))
                            expected_kind = "point" if iv_mode == "point_scenario" else "path"
                            if variant_kind != expected_kind:
                                continue
                            rows.append(
                                {
                                    "case_label": case_label,
                                    "case_move_pct": case_info.get("case_move_pct"),
                                    "display_mode": "iv_compare",
                                    "series_kind": "required_path",
                                    "series_label": iv_variant,
                                    "strategy_family": family,
                                    "candidate_slug": candidate_slug,
                                    "goal": goal,
                                    "iv_mode": iv_mode,
                                    "iv_variant": iv_variant,
                                    "horizon": horizon_label,
                                    "requested_days": int(matching.iloc[0].get("requested_days") or 0),
                                    "spot_price": finite_or_none(matching.iloc[0].get("required_stock_price")),
                                    "unreached": bool(matching.iloc[0].get("unreached")),
                                    "clamped_to_expiry": bool(matching.iloc[0].get("clamped_to_expiry")),
                                    "target_beyond_expiry": bool(matching.iloc[0].get("target_beyond_expiry")),
                                }
                            )
                        for horizon in horizon_specs:
                            horizon_label = clean_string(horizon["label"]).lower()
                            rows.append(
                                {
                                    "case_label": case_label,
                                    "case_move_pct": case_info.get("case_move_pct"),
                                    "display_mode": "iv_compare",
                                    "series_kind": "assumed_path",
                                    "series_label": "Assumed Path",
                                    "strategy_family": family,
                                    "candidate_slug": candidate_slug,
                                    "goal": goal,
                                    "iv_mode": iv_mode,
                                    "iv_variant": iv_variant,
                                    "horizon": horizon_label,
                                    "requested_days": int(horizon["requested_days"]),
                                    "spot_price": finite_or_none(assumed_path.get(horizon_label)),
                                    "unreached": False,
                                    "clamped_to_expiry": False,
                                    "target_beyond_expiry": False,
                                }
                            )
    return pd.DataFrame(rows)


def _path_case_strategy_rows(
    *,
    specs: list[dict[str, Any]],
    candidates: pd.DataFrame,
    selector_rows: pd.DataFrame,
    required_path_rows: pd.DataFrame,
    horizon_specs: list[dict[str, Any]],
    comparison_capital: float,
    case_definitions: dict[str, dict[str, Any]],
    iv_variants: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    if selector_rows.empty:
        return pd.DataFrame()
    spec_map = {clean_string(spec.get("candidate_slug")): spec for spec in specs if clean_string(spec.get("candidate_slug"))}
    candidate_map = {
        clean_string(row.get("candidate_slug")): row
        for _, row in candidates.iterrows()
        if clean_string(row.get("candidate_slug"))
    }
    selector_map = {
        clean_string(row.get("strategy_family")): row
        for _, row in selector_rows.iterrows()
        if clean_string(row.get("strategy_family"))
    }
    goals = sorted({clean_string(value) for value in required_path_rows.get("goal", pd.Series(dtype=str)).tolist() if clean_string(value)})
    final_horizon = horizon_specs[-1] if horizon_specs else {"label": "target", "requested_days": 0}
    rows: list[dict[str, Any]] = []
    for case_label, case_info in case_definitions.items():
        case_path = case_info.get("path_points") or {}
        final_spot = finite_or_none(case_path.get(clean_string(final_horizon["label"]).lower()))
        for iv_mode, variant_map in iv_variants.items():
            for iv_variant, iv_path in variant_map.items():
                final_iv = finite_or_none(iv_path.get(clean_string(final_horizon["label"]).lower()))
                for family, selector_row in selector_map.items():
                    candidate_slug = clean_string(selector_row.get("winning_candidate_slug"))
                    spec = spec_map.get(candidate_slug)
                    candidate_row = candidate_map.get(candidate_slug)
                    if not spec or final_spot is None:
                        continue
                    evaluation = _evaluate_at_point(
                        spec,
                        spot_price=float(final_spot),
                        horizon_days=int(final_horizon["requested_days"]),
                        iv_shift_points=float(final_iv or 0.0),
                        comparison_capital=float(comparison_capital),
                    )
                    for goal in goals:
                        matching_required = required_path_rows.loc[
                            (required_path_rows["candidate_slug"] == candidate_slug)
                            & (required_path_rows["goal"] == goal)
                            & (required_path_rows["iv_variant"] == iv_variant)
                        ].copy()
                        if matching_required.empty:
                            continue
                        variant_kind = clean_string(matching_required.iloc[0].get("iv_variant_kind"))
                        expected_kind = "point" if iv_mode == "point_scenario" else "path"
                        if variant_kind != expected_kind:
                            continue
                        difficulty = _required_path_difficulty(case_path, matching_required)
                        coverage_flags = clean_string(candidate_row.get("coverage_flags") if candidate_row is not None else selector_row.get("coverage_flags"))
                        confidence_label = clean_string(candidate_row.get("confidence_label") if candidate_row is not None else selector_row.get("confidence_label"))
                        horizon_fit = clean_string(candidate_row.get("horizon_fit_label") if candidate_row is not None else selector_row.get("horizon_fit_label"))
                        difficulty_penalty = {
                            "cleared comfortably": 3.0,
                            "roughly matched": 1.0,
                            "needs more / faster": -1.0,
                            "unreached in sampled range": -4.0,
                        }.get(difficulty, 0.0)
                        rank_score = float(finite_or_none(evaluation.get("profit_loss")) or 0.0) + float(
                            finite_or_none(evaluation.get("difference_vs_stock")) or 0.0
                        ) * 0.20 + difficulty_penalty * 40.0
                        stock_return = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
                        profit_loss_pct = finite_or_none(evaluation.get("return_on_comparison_capital"))
                        difference_vs_stock_return_pct = (
                            round(float(profit_loss_pct) - float(stock_return), 6)
                            if profit_loss_pct is not None and stock_return is not None
                            else None
                        )
                        target_beyond_expiry = bool(candidate_row.get("target_beyond_expiry")) if candidate_row is not None else False
                        timing_risk = clean_string(candidate_row.get("time_sensitivity_summary") if candidate_row is not None else selector_row.get("time_sensitivity_summary"))
                        iv_risk = clean_string(candidate_row.get("iv_sensitivity_summary") if candidate_row is not None else selector_row.get("iv_sensitivity_summary"))
                        success_dependency = (
                            "fast move required before expiry"
                            if target_beyond_expiry
                            else (
                                "needs more / faster path clearance"
                                if difficulty == "needs more / faster"
                                else "eventual target achievement can still work"
                            )
                        )
                        rows.append(
                            {
                                "case_label": case_label,
                                "case_move_pct": case_info.get("case_move_pct"),
                                "goal": goal,
                                "iv_mode": iv_mode,
                                "iv_variant": iv_variant,
                                "strategy_family": family,
                                "strategy_label": clean_string(selector_row.get("strategy_label")) or family.replace("_", " ").title(),
                                "winning_candidate_slug": candidate_slug,
                                "winning_candidate_label": clean_string(selector_row.get("winning_candidate_label")),
                                "relevance_label": clean_string(selector_row.get("relevance_label")),
                                "relevance_bucket": clean_string(selector_row.get("relevance_bucket")),
                                "required_path_difficulty": difficulty,
                                "modeled_value": evaluation.get("estimated_value"),
                                "profit_loss": evaluation.get("profit_loss"),
                                "profit_loss_pct": profit_loss_pct,
                                "stock_profit_loss": evaluation.get("stock_profit_loss"),
                                "stock_return_on_comparison_capital": stock_return,
                                "difference_vs_stock": evaluation.get("difference_vs_stock"),
                                "difference_vs_stock_return_pct": difference_vs_stock_return_pct,
                                "capital_required": candidate_row.get("capital_required") if candidate_row is not None else evaluation.get("capital_required"),
                                "affordable_units": candidate_row.get("affordable_units") if candidate_row is not None else evaluation.get("affordable_units"),
                                "max_loss": candidate_row.get("max_loss") if candidate_row is not None else evaluation.get("max_loss"),
                                "break_even": candidate_row.get("break_even") if candidate_row is not None else evaluation.get("break_even"),
                                "iv_sensitivity_summary": iv_risk,
                                "time_sensitivity_summary": timing_risk,
                                "iv_risk": iv_risk,
                                "timing_risk": timing_risk,
                                "success_dependency": success_dependency,
                                "why_it_wins": clean_string(selector_row.get("why_this_wins")),
                                "why_it_loses": clean_string(selector_row.get("why_this_loses")),
                                "coverage_flags": coverage_flags,
                                "confidence_label": confidence_label,
                                "horizon_fit_label": horizon_fit,
                                "target_beyond_expiry": target_beyond_expiry,
                                "expiry_clamped_estimate": bool(candidate_row.get("expiry_clamped_estimate")) if candidate_row is not None else False,
                                "benchmark_note": _compare_vs_stock_note(
                                    strategy_family=family,
                                    difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                                    difference_vs_stock_return_pct=difference_vs_stock_return_pct,
                                    clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                                    target_beyond_expiry=target_beyond_expiry,
                                ),
                                "case_rank_score": round(rank_score, 4),
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["case_label", "goal", "iv_mode", "iv_variant", "case_rank_score", "relevance_bucket", "strategy_label"],
        ascending=[True, True, True, True, False, True, True],
    ).reset_index(drop=True)
    frame["case_rank"] = frame.groupby(["case_label", "goal", "iv_mode", "iv_variant"]).cumcount() + 1
    return frame


def _path_case_candidate_rankings(
    *,
    specs: list[dict[str, Any]],
    candidates: pd.DataFrame,
    required_path_rows: pd.DataFrame,
    ranked_candidates: pd.DataFrame,
    horizon_specs: list[dict[str, Any]],
    comparison_capital: float,
    case_definitions: dict[str, dict[str, Any]],
    iv_variants: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    if candidates.empty or required_path_rows.empty:
        return pd.DataFrame()
    spec_map = {clean_string(spec.get("candidate_slug")): spec for spec in specs if clean_string(spec.get("candidate_slug"))}
    candidate_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in candidates.iterrows()
        if clean_string(row.get("candidate_slug"))
    }
    shortlisted = ranked_candidates.copy() if not ranked_candidates.empty else candidates.copy()
    candidate_slugs = [
        clean_string(value)
        for value in shortlisted.get("winner_candidate_slug", pd.Series(dtype=str)).tolist()
        if clean_string(value)
    ]
    if not candidate_slugs:
        candidate_slugs = [
            clean_string(value)
            for value in shortlisted.get("candidate_slug", pd.Series(dtype=str)).tolist()
            if clean_string(value)
        ]
    candidate_slugs = list(dict.fromkeys(candidate_slugs))
    if not candidate_slugs:
        candidate_slugs = [
            clean_string(value)
            for value in candidates.get("candidate_slug", pd.Series(dtype=str)).tolist()
            if clean_string(value)
        ]
    if not candidate_slugs:
        return pd.DataFrame()
    goals = sorted({clean_string(value) for value in required_path_rows.get("goal", pd.Series(dtype=str)).tolist() if clean_string(value)})
    final_horizon = horizon_specs[-1] if horizon_specs else {"label": "target", "requested_days": 0}
    rows: list[dict[str, Any]] = []
    for case_label, case_info in case_definitions.items():
        case_path = case_info.get("path_points") or {}
        final_spot = finite_or_none(case_path.get(clean_string(final_horizon["label"]).lower()))
        if final_spot is None:
            continue
        for iv_mode, variant_map in iv_variants.items():
            for iv_variant, iv_path in variant_map.items():
                final_iv = finite_or_none(iv_path.get(clean_string(final_horizon["label"]).lower()))
                expected_kind = "point" if iv_mode == "point_scenario" else "path"
                for candidate_slug in candidate_slugs:
                    spec = spec_map.get(candidate_slug)
                    candidate_row = candidate_lookup.get(candidate_slug)
                    if spec is None or candidate_row is None:
                        continue
                    evaluation = _evaluate_at_point(
                        spec,
                        spot_price=float(final_spot),
                        horizon_days=int(final_horizon["requested_days"]),
                        iv_shift_points=float(final_iv or 0.0),
                        comparison_capital=float(comparison_capital),
                    )
                    for goal in goals:
                        matching_required = required_path_rows.loc[
                            (required_path_rows["candidate_slug"] == candidate_slug)
                            & (required_path_rows["goal"] == goal)
                            & (required_path_rows["iv_variant"] == iv_variant)
                            & (required_path_rows["iv_variant_kind"] == expected_kind)
                        ].copy()
                        if matching_required.empty:
                            continue
                        difficulty = _required_path_difficulty(case_path, matching_required)
                        difficulty_penalty = {
                            "cleared comfortably": 3.0,
                            "roughly matched": 1.0,
                            "needs more / faster": -1.0,
                            "unreached in sampled range": -4.0,
                        }.get(difficulty, 0.0)
                        rank_score = float(finite_or_none(evaluation.get("profit_loss")) or 0.0) + float(
                            finite_or_none(evaluation.get("difference_vs_stock")) or 0.0
                        ) * 0.20 + difficulty_penalty * 40.0
                        stock_return = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
                        profit_loss_pct = finite_or_none(evaluation.get("return_on_comparison_capital"))
                        difference_vs_stock_return_pct = (
                            round(float(profit_loss_pct) - float(stock_return), 6)
                            if profit_loss_pct is not None and stock_return is not None
                            else None
                        )
                        target_beyond_expiry = bool(candidate_row.get("target_beyond_expiry"))
                        timing_risk = clean_string(candidate_row.get("time_sensitivity_summary"))
                        iv_risk = clean_string(candidate_row.get("iv_sensitivity_summary"))
                        success_dependency = (
                            "fast move required before expiry"
                            if target_beyond_expiry
                            else (
                                "needs more / faster path clearance"
                                if difficulty == "needs more / faster"
                                else "eventual target achievement can still work"
                            )
                        )
                        rows.append(
                            {
                                "case_label": case_label,
                                "case_move_pct": case_info.get("case_move_pct"),
                                "goal": goal,
                                "iv_mode": iv_mode,
                                "iv_variant": iv_variant,
                                "candidate_slug": candidate_slug,
                                "candidate_label": clean_string(candidate_row.get("candidate_label")),
                                "strategy_family": clean_string(candidate_row.get("strategy_family")),
                                "strategy_label": clean_string(candidate_row.get("strategy_family")).replace("_", " ").title(),
                                "selection_scope_label": clean_string(candidate_row.get("selection_scope_label")),
                                "required_path_difficulty": difficulty,
                                "modeled_value": evaluation.get("estimated_value"),
                                "profit_loss": evaluation.get("profit_loss"),
                                "profit_loss_pct": profit_loss_pct,
                                "stock_profit_loss": evaluation.get("stock_profit_loss"),
                                "stock_return_on_comparison_capital": stock_return,
                                "difference_vs_stock": evaluation.get("difference_vs_stock"),
                                "difference_vs_stock_return_pct": difference_vs_stock_return_pct,
                                "benchmark_edge": evaluation.get("difference_vs_stock"),
                                "capital_required": candidate_row.get("capital_required"),
                                "affordable_units": candidate_row.get("affordable_units"),
                                "max_loss": candidate_row.get("max_loss"),
                                "break_even": candidate_row.get("break_even"),
                                "iv_sensitivity_summary": iv_risk,
                                "time_sensitivity_summary": timing_risk,
                                "iv_risk": iv_risk,
                                "timing_risk": timing_risk,
                                "success_dependency": success_dependency,
                                "coverage_flags": clean_string(candidate_row.get("coverage_flags")),
                                "confidence_label": clean_string(candidate_row.get("confidence_label")),
                                "horizon_fit_label": clean_string(candidate_row.get("horizon_fit_label")),
                                "target_beyond_expiry": target_beyond_expiry,
                                "expiry_clamped_estimate": bool(candidate_row.get("expiry_clamped_estimate")),
                                "benchmark_note": _compare_vs_stock_note(
                                    strategy_family=clean_string(candidate_row.get("strategy_family")),
                                    difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                                    difference_vs_stock_return_pct=difference_vs_stock_return_pct,
                                    clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                                    target_beyond_expiry=target_beyond_expiry,
                                ),
                                "case_rank_score": round(rank_score, 4),
                            }
                        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["case_label", "goal", "iv_mode", "iv_variant", "case_rank_score", "strategy_family", "candidate_label"],
        ascending=[True, True, True, True, False, True, True],
    ).reset_index(drop=True)
    frame["case_rank"] = frame.groupby(["case_label", "goal", "iv_mode", "iv_variant"]).cumcount() + 1
    return frame


def _calibration_context(candidates: pd.DataFrame) -> dict[str, Any]:
    if candidates.empty:
        return {}
    row = candidates.iloc[0].to_dict()
    notes: list[str] = []
    iv_rank = finite_or_none(row.get("iv_rank"))
    expected_move_pct = finite_or_none(row.get("expected_move_pct"))
    if iv_rank is not None:
        if iv_rank >= 70:
            notes.append("Current IV is elevated versus recent history, so IV mean reversion lower would hurt rich premium structures more.")
        elif iv_rank <= 30:
            notes.append("Current IV sits low versus recent history, so long premium structures may be relying less on future IV collapse.")
    if expected_move_pct is not None:
        notes.append(f"Entry expected move context was about {expected_move_pct * 100:.1f}% when local research metadata matched this snapshot.")
    if clean_string(row.get("nearest_event_date")):
        notes.append(f"Nearest event context points to {clean_string(row.get('nearest_event_type')).replace('_', ' ').title()} on {row.get('nearest_event_date')}.")
    return {
        "iv_rank": iv_rank,
        "iv_percentile": finite_or_none(row.get("iv_percentile")),
        "expected_move_pct": expected_move_pct,
        "nearest_event_date": row.get("nearest_event_date"),
        "nearest_event_type": row.get("nearest_event_type"),
        "notes": notes,
    }


def _rankings(
    candidates: pd.DataFrame,
    path_case_summary: pd.DataFrame,
    *,
    stock_path_name: str,
    iv_path_name: str,
    comparison_capital: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if candidates.empty:
        return pd.DataFrame(columns=["ranking_mode", "winner_candidate", "winner_strategy", "winner_value", "rationale"]), []
    rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []

    def append_card(mode: str, title: str, frame: pd.DataFrame, metric: str, rationale: str, *, ascending: bool = False) -> None:
        if frame.empty or metric not in frame.columns:
            return
        working = frame.dropna(subset=[metric]).copy()
        if working.empty:
            return
        winner = working.sort_values(metric, ascending=ascending).iloc[0]
        row = {
            "ranking_mode": mode,
            "title": title,
            "winner_candidate": winner.get("candidate_label"),
            "winner_strategy": winner.get("strategy_family"),
            "winner_value": winner.get(metric),
            "rationale": rationale,
            "heuristic": True,
        }
        rows.append(row)
        cards.append(row)

    on_time = path_case_summary.loc[
        (path_case_summary.get("stock_path") == stock_path_name)
        & (path_case_summary.get("iv_path") == iv_path_name)
    ].copy()
    slower = path_case_summary.loc[
        (path_case_summary.get("stock_path").isin([stock_path_name, "slow_bull"]))
        & (path_case_summary.get("iv_path") == iv_path_name)
    ].copy()
    iv_fall = path_case_summary.loc[
        path_case_summary.get("iv_path").isin([iv_path_name, "mean_reversion_lower", "iv_down_then_stays_low"])
    ].copy()
    capped = candidates.loc[candidates["strategy_family"].isin(["bull_call_spread", "bear_put_spread", "long_put"])].copy()
    options_only = candidates.loc[candidates["strategy_family"] != "long_stock"].copy()

    append_card(
        "best_target_on_time",
        "Best If Target Is Reached On Time",
        on_time,
        "final_profit_loss",
        "Highest modeled PnL when the chosen stock and IV paths land roughly on schedule.",
    )
    append_card(
        "best_move_slower",
        "Best If Move Is Slower",
        slower,
        "final_profit_loss",
        "Best terminal outcome when the stock path takes longer to work.",
    )
    append_card(
        "best_if_iv_falls",
        "Best If IV Falls",
        iv_fall,
        "final_profit_loss",
        "Best modeled outcome under lower-for-longer / IV compression style assumptions.",
    )
    append_card(
        "best_capped_downside",
        "Best For Capped Downside",
        capped,
        "max_loss",
        "Lowest max loss among capped or defensive structures.",
        ascending=True,
    )
    append_card(
        "best_convexity",
        "Best Convexity",
        options_only,
        "return_on_comparison_capital",
        "Highest return on the normalized budget among option structures when the path works.",
    )
    append_card(
        "best_capital_efficiency",
        f"Best Capital Efficiency For ${comparison_capital:,.0f}",
        candidates,
        "difference_vs_stock",
        "Largest modeled edge versus long stock under the fixed-budget lens.",
    )
    stock_only = candidates.loc[candidates["strategy_family"] == "long_stock"].copy()
    append_card(
        "best_simplest_benchmark",
        "Best Simplest Benchmark",
        stock_only,
        "profit_loss",
        "Long stock remains the cleanest no-expiry baseline and should stay in view as the simplest benchmark.",
    )
    return pd.DataFrame(rows), cards


def _strategy_direction(spot_price: float, target_price: float) -> str:
    if target_price >= spot_price * 1.05:
        return "bullish"
    if target_price <= spot_price * 0.95:
        return "bearish"
    return "neutral"


def _selector_relevance(strategy_family: str, *, direction: str) -> dict[str, Any]:
    family = clean_string(strategy_family).lower()
    if direction == "bullish":
        if family in {"long_stock", "long_call", "bull_call_spread"}:
            return {"label": "high relevance", "bucket": "primary", "rank": 0}
        if family in {"covered_call", "cash_secured_put"}:
            return {"label": "secondary relevance", "bucket": "secondary", "rank": 1}
        return {"label": "defensive / low relevance", "bucket": "lower", "rank": 2}
    if direction == "bearish":
        if family in {"long_put", "bear_put_spread"}:
            return {"label": "high relevance", "bucket": "primary", "rank": 0}
        if family in {"cash_secured_put", "covered_call"}:
            return {"label": "secondary relevance", "bucket": "secondary", "rank": 1}
        return {"label": "defensive / low relevance", "bucket": "lower", "rank": 2}
    if family in {"long_stock", "covered_call", "cash_secured_put"}:
        return {"label": "high relevance", "bucket": "primary", "rank": 0}
    if family in {"long_call", "long_put", "bull_call_spread", "bear_put_spread"}:
        return {"label": "secondary relevance", "bucket": "secondary", "rank": 1}
    return {"label": "defensive / low relevance", "bucket": "lower", "rank": 2}


def _selector_relevance_note(strategy_family: str, *, direction: str) -> str:
    bullish = {"long_stock", "long_call", "bull_call_spread", "covered_call", "cash_secured_put"}
    bearish = {"long_put", "bear_put_spread"}
    family = clean_string(strategy_family).lower()
    if direction == "bullish" and family in bearish:
        return "Less naturally aligned with a bullish target path, so it only ranks well if protection or bearish convexity is the real objective."
    if direction == "bearish" and family in bullish - {"covered_call"}:
        return "Less naturally aligned with a bearish target path unless simplicity, carry, or stock ownership is the real goal."
    if direction == "neutral" and family in {"long_call", "long_put"}:
        return "Needs a meaningful move or IV help, so it is less naturally aligned with a flatter path."
    return "Reasonably aligned with the current path assumptions."


def _selector_warning(row: pd.Series) -> str:
    if not bool(row.get("available", True)):
        return clean_string(row.get("notes")) or "No buildable candidate was available for this family in the local snapshot set."
    if bool(row.get("target_beyond_expiry")):
        return "Target date extends past expiry, so this family is being judged on an expiry-clamped estimate with a weaker timing fit."
    if clean_string(row.get("horizon_fit_label")) == "weak timing match":
        return "Timing fit is weak here because the modeled holding period is shorter than the requested thesis horizon."
    if not bool(row.get("fully_implementable_with_budget", False)):
        return "Needs more than the normalized comparison budget for a whole-unit implementation."
    if finite_or_none(row.get("iv_down_value_change")) is not None and float(row.get("iv_down_value_change")) < -50:
        return "A sharp IV compression would materially hurt this family."
    if finite_or_none(row.get("delayed_move_value_change")) is not None and float(row.get("delayed_move_value_change")) < -50:
        return "A delayed move erodes modeled value noticeably here."
    if clean_string(row.get("warning_or_note")):
        return clean_string(row.get("warning_or_note"))
    if clean_string(row.get("strategy_family")) == "long_stock":
        return "Carries full downside because there is no premium-defined floor."
    return "Check the exact candidate in Path & Contract Explorer before treating this as the final choice."


def _selector_score(
    row: pd.Series,
    *,
    objective_mode: str,
    downside_tolerance: str,
    simplicity_preference: str,
) -> float:
    profit_loss = float(finite_or_none(row.get("profit_loss")) or 0.0)
    return_pct = float(finite_or_none(row.get("return_on_comparison_capital")) or 0.0)
    difference_vs_stock = float(finite_or_none(row.get("difference_vs_stock")) or 0.0)
    max_loss = float(finite_or_none(row.get("max_loss")) or 0.0)
    iv_down_change = float(finite_or_none(row.get("iv_down_value_change")) or 0.0)
    iv_up_change = float(finite_or_none(row.get("iv_up_value_change")) or 0.0)
    delayed_change = float(finite_or_none(row.get("delayed_move_value_change")) or 0.0)
    family = clean_string(row.get("strategy_family")).lower()
    capped_upside = family in {"bull_call_spread", "bear_put_spread", "covered_call", "cash_secured_put"}
    simplicity_score = 2.0 if family == "long_stock" else (1.0 if family in {"covered_call", "cash_secured_put"} else 0.0)
    convexity_score = 2.0 if family in {"long_call", "long_put"} else (1.0 if family in {"bull_call_spread", "bear_put_spread"} else 0.0)
    availability_penalty = 0.0 if bool(row.get("available", True)) else -1_000_000.0
    budget_penalty = -125.0 if not bool(row.get("fully_implementable_with_budget", False)) else 0.0
    coverage_penalty = -40.0 if clean_string(row.get("selection_scope")) == "nearby_snapshot_fallback" else 0.0
    horizon_penalty = 0.0
    if bool(row.get("target_beyond_expiry")):
        horizon_penalty -= 325.0
    elif clean_string(row.get("horizon_fit_label")) == "weak timing match":
        horizon_penalty -= 125.0
    downside_bias = {"low": -0.25, "medium": -0.10, "high": 0.0}.get(clean_string(downside_tolerance).lower(), -0.10)
    simplicity_bias = {"high": 120.0, "medium": 45.0, "low": 0.0}.get(clean_string(simplicity_preference).lower(), 45.0)
    score_map = {
        "max_return_at_target": profit_loss + return_pct * 250.0,
        "outperform_stock": difference_vs_stock + return_pct * 80.0,
        "capital_efficiency": return_pct * 1000.0 + difference_vs_stock * 0.1,
        "downside_control": (-max_loss) + profit_loss * 0.20,
        "robustness_iv_fall": iv_down_change + profit_loss * 0.25 + difference_vs_stock * 0.10,
        "move_takes_time": delayed_change + profit_loss * 0.25,
        "highest_convexity": convexity_score * 400.0 + return_pct * 600.0 + iv_up_change * 0.35,
    }
    base_score = score_map.get(clean_string(objective_mode).lower(), score_map[DEFAULT_OBJECTIVE_MODE])
    score = base_score + availability_penalty + budget_penalty + coverage_penalty + horizon_penalty + max_loss * downside_bias
    if capped_upside:
        score -= 25.0
    if family == "long_stock":
        score += simplicity_score * simplicity_bias
    else:
        score += simplicity_score * (simplicity_bias * 0.35)
    return round(float(score), 4)


def _strategy_selector_rows(
    candidates: pd.DataFrame,
    *,
    requested_families: list[str],
    objective_mode: str,
    downside_tolerance: str,
    simplicity_preference: str,
    spot_price: float,
    target_price: float,
    comparison_capital: float,
) -> pd.DataFrame:
    direction = _strategy_direction(spot_price, target_price)
    rows: list[dict[str, Any]] = []
    for family in requested_families:
        relevance = _selector_relevance(family, direction=direction)
        family_rows = candidates.loc[candidates["strategy_family"] == family].copy()
        if family_rows.empty:
            rows.append(
                {
                    "strategy_family": family,
                    "strategy_label": family.replace("_", " ").title(),
                    "role": STRATEGY_ROLE_MAP.get(family),
                    "relevance_label": relevance["label"],
                    "relevance_bucket": relevance["bucket"],
                    "relevance_rank": relevance["rank"],
                    "relevance_note": _selector_relevance_note(family, direction=direction),
                    "why_this_wins": STRATEGY_WIN_LOSE_MAP.get(family, ("", ""))[0],
                    "why_this_loses": STRATEGY_WIN_LOSE_MAP.get(family, ("", ""))[1],
                    "available": False,
                    "objective_mode": objective_mode,
                    "objective_score": -1_000_000.0,
                    "comparison_capital": comparison_capital,
                    "notes": "No buildable candidate was available for this family in the local candidate set.",
                    "one_line_warning": "No buildable candidate was available for this family in the local candidate set.",
                }
            )
            continue
        family_rows["objective_score"] = family_rows.apply(
            lambda row: _selector_score(
                row,
                objective_mode=objective_mode,
                downside_tolerance=downside_tolerance,
                simplicity_preference=simplicity_preference,
            ),
            axis=1,
        )
        winner = family_rows.sort_values(
            ["objective_score", "profit_loss", "return_on_comparison_capital"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(
            {
                "strategy_family": family,
                "strategy_label": family.replace("_", " ").title(),
                "role": STRATEGY_ROLE_MAP.get(family),
                "relevance_label": relevance["label"],
                "relevance_bucket": relevance["bucket"],
                "relevance_rank": relevance["rank"],
                "relevance_note": _selector_relevance_note(family, direction=direction),
                "why_this_wins": STRATEGY_WIN_LOSE_MAP.get(family, ("", ""))[0],
                "why_this_loses": STRATEGY_WIN_LOSE_MAP.get(family, ("", ""))[1],
                "available": True,
                "objective_mode": objective_mode,
                "objective_score": winner.get("objective_score"),
                "comparison_capital": comparison_capital,
                "winning_candidate_slug": winner.get("candidate_slug"),
                "winning_candidate_label": winner.get("candidate_label"),
                "expiry_date": winner.get("expiry_date"),
                "strike_label": winner.get("strike_label"),
                "winning_candidate_pointer": winner.get("candidate_label"),
                "target_pnl": winner.get("profit_loss"),
                "target_return_pct": winner.get("return_on_comparison_capital"),
                "difference_vs_stock": winner.get("difference_vs_stock"),
                "capital_required": winner.get("capital_required"),
                "max_loss": winner.get("max_loss"),
                "break_even": winner.get("break_even"),
                "affordable_units": winner.get("affordable_units"),
                "fully_implementable_with_budget": winner.get("fully_implementable_with_budget"),
                "iv_sensitivity_summary": winner.get("iv_sensitivity_summary"),
                "time_sensitivity_summary": winner.get("time_sensitivity_summary"),
                "iv_down_value_change": winner.get("iv_down_value_change"),
                "iv_up_value_change": winner.get("iv_up_value_change"),
                "delayed_move_value_change": winner.get("delayed_move_value_change"),
                "target_beyond_expiry": winner.get("target_beyond_expiry"),
                "expiry_clamped_estimate": winner.get("expiry_clamped_estimate"),
                "timing_match_ratio": winner.get("timing_match_ratio"),
                "timing_gap_days": winner.get("timing_gap_days"),
                "horizon_fit_label": winner.get("horizon_fit_label"),
                "coverage_flags": winner.get("coverage_flags"),
                "confidence_label": winner.get("confidence_label"),
                "selection_scope": winner.get("selection_scope"),
                "selection_scope_label": winner.get("selection_scope_label"),
                "warning_or_note": winner.get("warning_or_note"),
                "notes": winner.get("warning_or_note") or winner.get("leg_summary"),
                "one_line_warning": _selector_warning(winner),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["available", "relevance_rank", "objective_score"], ascending=[False, True, False]).reset_index(drop=True)
    frame["objective_rank"] = range(1, len(frame.index) + 1)
    return frame


def _selector_cards(
    selector_rows: pd.DataFrame,
    *,
    objective_mode: str,
    comparison_capital: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if selector_rows.empty:
        return pd.DataFrame(columns=["ranking_mode", "title", "winner_strategy", "rationale", "heuristic"]), []
    available = selector_rows.loc[selector_rows["available"] == True].copy()  # noqa: E712
    if available.empty:
        return pd.DataFrame(columns=["ranking_mode", "title", "winner_strategy", "rationale", "heuristic"]), []
    rows: list[dict[str, Any]] = []
    cards: list[dict[str, Any]] = []

    def _append(
        mode: str,
        title: str,
        frame: pd.DataFrame,
        rationale: str,
        *,
        score_func,
        min_materiality: float,
        min_gap: float,
        ascending: bool = False,
    ) -> None:
        if frame.empty:
            return
        working = frame.copy()
        working["ranking_score"] = working.apply(score_func, axis=1)
        working = working.dropna(subset=["ranking_score"]).copy()
        if working.empty:
            return
        ordered = working.sort_values("ranking_score", ascending=ascending).reset_index(drop=True)
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1] if len(ordered.index) > 1 else None
        winner_score = float(finite_or_none(winner.get("ranking_score")) or 0.0)
        runner_score = float(finite_or_none(runner_up.get("ranking_score")) or 0.0) if runner_up is not None else 0.0
        gap = abs(winner_score - runner_score)
        magnitude = max(abs(winner_score), abs(runner_score), 1.0)
        winner_target_beyond_expiry = bool(winner.get("target_beyond_expiry"))
        winner_horizon_fit = clean_string(winner.get("horizon_fit_label"))
        score_status = "informative"
        reason = rationale
        warning = clean_string(winner.get("one_line_warning"))
        winner_strategy = winner.get("strategy_family")
        winner_strategy_label = winner.get("strategy_label")
        winner_candidate_label = winner.get("winning_candidate_label")
        if abs(winner_score) < min_materiality and gap < min_gap:
            score_status = "no_clear_edge"
        elif gap < min_gap or (gap / magnitude) < 0.08:
            score_status = "weak_differentiation"
        elif winner_target_beyond_expiry and gap < max(min_gap * 2.0, 35.0):
            score_status = "weak_differentiation"
        elif winner_horizon_fit in {"weak timing match", "poor horizon fit"} and gap < max(min_gap * 1.5, 25.0):
            score_status = "weak_differentiation"
        if score_status in LOW_INFORMATION_CARD_STATUSES:
            winner_strategy = None
            winner_strategy_label = "No clear edge"
            winner_candidate_label = "Multiple families compress to similar outcomes"
            reason = f"{rationale} Current local data does not separate the families enough to present a confident family winner."
            warning = "Weak differentiation | heuristic only"
        elif winner_target_beyond_expiry:
            reason = f"{reason} Treat the edge carefully because the target extends beyond expiry and the winning family depends on an expiry-clamped estimate."
        card = {
            "ranking_mode": mode,
            "title": title,
            "winner_strategy": winner_strategy,
            "winner_strategy_label": winner_strategy_label,
            "winner_candidate_label": winner_candidate_label,
            "winner_value": round(winner_score, 4),
            "reason": reason,
            "warning": warning,
            "winning_candidate_slug": winner.get("winning_candidate_slug") if score_status == "informative" else None,
            "heuristic": True,
            "card_status": score_status,
            "is_informative": score_status == "informative",
            "differentiation_gap": round(gap, 4),
            "confidence_label": winner.get("confidence_label"),
            "horizon_fit_label": winner.get("horizon_fit_label"),
            "coverage_flags": winner.get("coverage_flags"),
        }
        rows.append(card)
        cards.append(card)

    capped = available.loc[available["strategy_family"].isin(["long_call", "long_put", "bull_call_spread", "bear_put_spread"])].copy()
    simple = available.loc[available["strategy_family"].isin(["long_stock", "covered_call", "cash_secured_put"])].copy()
    _append(
        "best_overall_current_objective",
        "Best Overall Under Current Objective",
        available,
        f"Best family under the active {objective_mode.replace('_', ' ')} lens.",
        score_func=lambda row: float(finite_or_none(row.get("objective_score")) or 0.0),
        min_materiality=80.0,
        min_gap=60.0,
    )
    _append(
        "best_if_move_is_slower",
        "Best If Move Is Slower",
        available,
        "Best family when the thesis takes longer to arrive and delay matters more.",
        score_func=lambda row: (
            float(finite_or_none(row.get("delayed_move_value_change")) or 0.0)
            + float(finite_or_none(row.get("target_pnl")) or 0.0) * 0.15
            + float(finite_or_none(row.get("difference_vs_stock")) or 0.0) * 0.10
            - (220.0 if bool(row.get("target_beyond_expiry")) else 0.0)
        ),
        min_materiality=20.0,
        min_gap=15.0,
    )
    _append(
        "best_if_iv_falls",
        "Best If IV Falls",
        available,
        "Best family when IV compression becomes the main risk to the trade.",
        score_func=lambda row: (
            float(finite_or_none(row.get("iv_down_value_change")) or 0.0)
            + float(finite_or_none(row.get("target_pnl")) or 0.0) * 0.12
            + float(finite_or_none(row.get("difference_vs_stock")) or 0.0) * 0.10
            - (180.0 if bool(row.get("target_beyond_expiry")) else 0.0)
        ),
        min_materiality=18.0,
        min_gap=12.0,
    )
    _append(
        "best_if_iv_rises",
        "Best If IV Rises",
        available,
        "Best family when higher implied volatility becomes a tailwind.",
        score_func=lambda row: (
            float(finite_or_none(row.get("iv_up_value_change")) or 0.0)
            + float(finite_or_none(row.get("target_pnl")) or 0.0) * 0.10
            + float(finite_or_none(row.get("difference_vs_stock")) or 0.0) * 0.05
            - (160.0 if bool(row.get("target_beyond_expiry")) else 0.0)
        ),
        min_materiality=18.0,
        min_gap=12.0,
    )
    _append(
        "best_for_capped_downside",
        "Best For Capped Downside",
        capped,
        "Lowest max loss among premium-defined or capped-risk structures.",
        score_func=lambda row: (
            -float(finite_or_none(row.get("max_loss")) or 0.0)
            + float(finite_or_none(row.get("target_pnl")) or 0.0) * 0.05
        ),
        min_materiality=5.0,
        min_gap=5.0,
    )
    _append(
        "best_for_capital_efficiency",
        f"Best Capital Efficiency For ${comparison_capital:,.0f}",
        available,
        "Highest return on the normalized budget lens.",
        score_func=lambda row: (
            float(finite_or_none(row.get("target_return_pct")) or 0.0) * 1000.0
            + float(finite_or_none(row.get("difference_vs_stock")) or 0.0) * 0.10
            - (110.0 if bool(row.get("target_beyond_expiry")) else 0.0)
        ),
        min_materiality=20.0,
        min_gap=15.0,
    )
    _append(
        "best_for_convexity",
        "Best For Convexity",
        available.loc[available["strategy_family"].isin(["long_call", "long_put", "bull_call_spread", "bear_put_spread"])],
        "Most leveraged upside or downside response when the path really works.",
        score_func=lambda row: (
            (400.0 if clean_string(row.get("strategy_family")) in {"long_call", "long_put"} else 180.0)
            + float(finite_or_none(row.get("target_return_pct")) or 0.0) * 600.0
            + max(float(finite_or_none(row.get("iv_up_value_change")) or 0.0), 0.0) * 0.15
            - (150.0 if bool(row.get("target_beyond_expiry")) else 0.0)
        ),
        min_materiality=45.0,
        min_gap=30.0,
    )
    _append(
        "best_for_simple_exposure",
        "Best For Simple Exposure",
        simple,
        "Simpler no- or lower-complexity exposure when clarity and fewer moving parts matter more.",
        score_func=lambda row: (
            (650.0 if clean_string(row.get("strategy_family")) == "long_stock" else 420.0)
            + float(finite_or_none(row.get("target_pnl")) or 0.0) * 0.05
            - float(finite_or_none(row.get("max_loss")) or 0.0) * 0.05
        ),
        min_materiality=80.0,
        min_gap=40.0,
    )
    return pd.DataFrame(rows), cards


def _affordability_label(row: pd.Series | dict[str, Any], *, comparison_capital: float) -> str:
    affordable_units = int(finite_or_none(row.get("affordable_units")) or 0)
    fully_implementable = bool(row.get("fully_implementable_with_budget"))
    capital_required = finite_or_none(row.get("capital_required")) or finite_or_none(row.get("unit_capital_required"))
    if fully_implementable:
        if affordable_units > 1:
            return f"fits budget ({affordable_units} whole units)"
        if affordable_units == 1:
            return "fits budget (1 whole unit)"
        return "fits budget"
    if capital_required is None:
        return "budget unclear"
    shortfall = max(float(capital_required) - float(comparison_capital), 0.0)
    if shortfall <= 0:
        return "budget nearly fits"
    return f"needs +${shortfall:,.0f} versus comparison budget"


def _stock_benchmark_decision(
    *,
    strategy_family: str,
    difference_vs_stock: float | None,
    difference_vs_stock_return_pct: float | None,
) -> str:
    family = clean_string(strategy_family).lower()
    pnl_delta = finite_or_none(difference_vs_stock)
    return_delta = finite_or_none(difference_vs_stock_return_pct)
    comparable_delta = pnl_delta if pnl_delta is not None else return_delta
    if family == "long_stock":
        return "long_stock_benchmark"
    if comparable_delta is None:
        return "benchmark_unavailable"
    if abs(float(comparable_delta)) <= 5.0 and (return_delta is None or abs(float(return_delta)) <= 0.01):
        return "tracks_stock_closely"
    return "options_show_edge" if float(comparable_delta) > 0 else "stock_still_better"


def _candidate_relevance_under_thesis(
    row: pd.Series | dict[str, Any],
    *,
    direction: str,
) -> str:
    relevance = _selector_relevance(clean_string(row.get("strategy_family")), direction=direction)
    base = clean_string(relevance.get("label")) or "relevance unclear"
    horizon_fit = clean_string(row.get("horizon_fit_label")).lower()
    if bool(row.get("target_beyond_expiry")):
        return f"{base}; target extends beyond expiry"
    if horizon_fit == "poor horizon fit":
        return f"{base}; poor horizon fit"
    if horizon_fit == "weak timing match":
        return f"{base}; weak timing match"
    return base


def _candidate_win_lose_notes(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    family = clean_string(row.get("strategy_family")).lower()
    wins_base, loses_base = STRATEGY_WIN_LOSE_MAP.get(family, ("Supports the active thesis under some paths.", "Can lose if the active path assumptions fail."))
    benchmark_decision = _stock_benchmark_decision(
        strategy_family=family,
        difference_vs_stock=finite_or_none(row.get("difference_vs_stock")),
        difference_vs_stock_return_pct=finite_or_none(row.get("difference_vs_stock_return_pct")),
    )
    required_path_difficulty = clean_string(row.get("required_path_difficulty")).lower()
    timing_risk = clean_string(row.get("timing_risk") or row.get("time_sensitivity_summary"))
    iv_risk = clean_string(row.get("iv_risk") or row.get("iv_sensitivity_summary"))
    source_trust_label = clean_string(row.get("source_trust_label")).lower()
    source_quality_note = clean_string(row.get("source_quality_note"))

    win_notes = [wins_base]
    lose_notes = [loses_base]

    if benchmark_decision == "options_show_edge":
        win_notes.append("Beats long stock under the active target, path, and comparison-capital assumptions.")
    elif benchmark_decision == "stock_still_better":
        lose_notes.append("Long stock still comes out ahead under the active assumptions.")
    elif benchmark_decision == "tracks_stock_closely":
        lose_notes.append("Outcomes stay compressed versus long stock, so there is little clean edge.")

    if required_path_difficulty == "cleared comfortably":
        win_notes.append("The active assumed stock path clears the required path comfortably.")
    elif required_path_difficulty == "roughly matched":
        win_notes.append("The active assumed stock path is at least close to the required path.")
    elif required_path_difficulty == "needs more / faster":
        lose_notes.append("Needs a bigger or faster move than the active assumed stock path currently provides.")
    elif required_path_difficulty == "unreached in sampled range":
        lose_notes.append("Required path stayed unreached in the sampled stock grid.")

    if bool(row.get("target_beyond_expiry")) or "clamp" in timing_risk.lower():
        lose_notes.append("The thesis depends on getting there before expiry rather than simply getting there eventually.")
    elif "slower move" in timing_risk.lower():
        win_notes.append("Can tolerate a slower move better than many long-premium alternatives.")

    if "high iv dependence" in iv_risk.lower():
        lose_notes.append("Outcome depends heavily on IV staying supportive.")
    elif "moderate iv dependence" in iv_risk.lower():
        lose_notes.append("Outcome still leans on IV not fading too quickly.")
    elif "low iv dependence" in iv_risk.lower() or "holds value better if iv falls" in iv_risk.lower():
        win_notes.append("Outcome is less IV-sensitive than a typical long-premium structure.")

    if family in {"bull_call_spread", "bear_put_spread", "covered_call", "cash_secured_put"}:
        lose_notes.append("Upside is capped once the short strike or assignment profile starts dominating.")
    if not bool(row.get("fully_implementable_with_budget")):
        lose_notes.append("One whole-unit implementation does not fit the normalized comparison budget.")
    if source_trust_label == "fallback_only":
        lose_notes.append(source_quality_note or "Quote support for this expiry is sparse, so pricing confidence is weaker.")
    elif source_trust_label == "quoted_prior_day":
        lose_notes.append(source_quality_note or "Pricing is quoted, but it comes from a prior-day source rather than the requested date.")
    elif source_trust_label == "trusted_quoted":
        win_notes.append("Source coverage is same-day and quoted, so the pricing read is on firmer ground.")

    return " ".join(_dedupe(win_notes)), " ".join(_dedupe(lose_notes))


def _mode_winner_payloads(rankings: pd.DataFrame, *, key_column: str) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    if rankings.empty:
        return payloads
    for _, row in rankings.iterrows():
        mode = clean_string(row.get("ranking_mode"))
        if not mode:
            continue
        payloads[mode] = {
            "winner": clean_string(row.get(key_column)),
            "status": clean_string(row.get("card_status")) or "informative",
            "reason": clean_string(row.get("reason") or row.get("rationale")),
            "warning": clean_string(row.get("warning")),
            "gap": finite_or_none(row.get("differentiation_gap")),
        }
    return payloads


def _family_comparison_rows(
    selector_rows: pd.DataFrame,
    *,
    selector_rankings: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    required_path_summary: pd.DataFrame,
    path_risk_summary: pd.DataFrame,
    iv_path_sensitivity_summary: pd.DataFrame,
    goal: str,
    stock_path_name: str,
    iv_path_name: str,
) -> pd.DataFrame:
    if selector_rows.empty:
        return pd.DataFrame()
    required_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in required_path_summary.loc[
            (required_path_summary.get("summary_scope", pd.Series(dtype=str)) == "family_representative")
            & (required_path_summary.get("goal", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(goal).lower())
            & (required_path_summary.get("iv_variant", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(iv_path_name).lower())
        ].iterrows()
    }
    risk_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in path_risk_summary.loc[
            (path_risk_summary.get("summary_scope", pd.Series(dtype=str)) == "family_representative")
            & (path_risk_summary.get("goal", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(goal).lower())
            & (path_risk_summary.get("stock_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(stock_path_name).lower())
            & (path_risk_summary.get("iv_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(iv_path_name).lower())
        ].iterrows()
    }
    sensitivity_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in iv_path_sensitivity_summary.loc[
            (iv_path_sensitivity_summary.get("summary_scope", pd.Series(dtype=str)) == "family_representative")
            & (iv_path_sensitivity_summary.get("stock_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(stock_path_name).lower())
            & (iv_path_sensitivity_summary.get("active_iv_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(iv_path_name).lower())
        ].iterrows()
    }
    candidate_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in candidate_summary.iterrows()
        if clean_string(row.get("candidate_slug"))
    }
    ranking_payloads = _mode_winner_payloads(selector_rankings, key_column="winner_strategy")
    overall = ranking_payloads.get("best_overall_current_objective", {})
    mode_to_column = {
        "best_overall_current_objective": "best_under_current_objective",
        "best_if_move_is_slower": "best_if_move_is_slower",
        "best_if_iv_falls": "best_if_iv_falls",
        "best_for_capital_efficiency": "best_for_capital_efficiency",
        "best_for_capped_downside": "best_for_capped_downside",
        "best_for_convexity": "best_for_convexity",
        "best_for_simple_exposure": "best_for_simple_exposure",
    }
    rows: list[dict[str, Any]] = []
    for _, selector_row in selector_rows.iterrows():
        family = clean_string(selector_row.get("strategy_family"))
        candidate_slug = clean_string(selector_row.get("winning_candidate_slug"))
        required = required_lookup.get(candidate_slug, {})
        risk = risk_lookup.get(candidate_slug, {})
        sensitivity = sensitivity_lookup.get(candidate_slug, {})
        candidate_meta = candidate_lookup.get(candidate_slug, {})
        benchmark_note = clean_string(risk.get("benchmark_note") or candidate_meta.get("benchmark_note"))
        objective_reason = clean_string(overall.get("reason"))
        if clean_string(overall.get("status")) in LOW_INFORMATION_CARD_STATUSES:
            current_status = clean_string(overall.get("status"))
        elif clean_string(overall.get("winner")) == family:
            current_status = clean_string(overall.get("status")) or "informative"
        else:
            current_status = "not_selected"
        rows.append(
            {
                "strategy_family": family,
                "strategy_label": clean_string(selector_row.get("strategy_label")),
                "winning_candidate_label": clean_string(selector_row.get("winning_candidate_label")),
                "winning_candidate_slug": candidate_slug,
                "objective_rank": selector_row.get("objective_rank"),
                "objective_score": selector_row.get("objective_score"),
                "target_pnl": selector_row.get("target_pnl"),
                "target_return_pct": selector_row.get("target_return_pct"),
                "difference_vs_stock": selector_row.get("difference_vs_stock"),
                "capital_required": selector_row.get("capital_required"),
                "affordability_label": _affordability_label(selector_row, comparison_capital=float(selector_row.get("comparison_capital") or 0.0)),
                "break_even": selector_row.get("break_even"),
                "max_loss": selector_row.get("max_loss"),
                "current_objective_card_status": current_status,
                "current_objective_reason": objective_reason if current_status != "not_selected" else clean_string(selector_row.get("why_this_loses")),
                "current_objective_warning": clean_string(overall.get("warning")) if current_status != "not_selected" else clean_string(selector_row.get("one_line_warning")),
                "best_under_current_objective": clean_string(ranking_payloads.get("best_overall_current_objective", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_overall_current_objective", {}).get("status")) == "informative",
                "best_if_move_is_slower": clean_string(ranking_payloads.get("best_if_move_is_slower", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_if_move_is_slower", {}).get("status")) == "informative",
                "best_if_iv_falls": clean_string(ranking_payloads.get("best_if_iv_falls", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_if_iv_falls", {}).get("status")) == "informative",
                "best_for_capital_efficiency": clean_string(ranking_payloads.get("best_for_capital_efficiency", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_for_capital_efficiency", {}).get("status")) == "informative",
                "best_for_capped_downside": clean_string(ranking_payloads.get("best_for_capped_downside", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_for_capped_downside", {}).get("status")) == "informative",
                "best_for_convexity": clean_string(ranking_payloads.get("best_for_convexity", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_for_convexity", {}).get("status")) == "informative",
                "best_for_simple_exposure": clean_string(ranking_payloads.get("best_for_simple_exposure", {}).get("winner")) == family
                and clean_string(ranking_payloads.get("best_for_simple_exposure", {}).get("status")) == "informative",
                "why_this_wins": clean_string(selector_row.get("why_this_wins")),
                "why_this_loses": clean_string(selector_row.get("why_this_loses")),
                "required_path_difficulty": clean_string(risk.get("required_path_difficulty") or required.get("required_path_difficulty")),
                "path_gap_at_target": finite_or_none(risk.get("path_gap_at_target") or required.get("path_gap_at_target")),
                "first_cleared_horizon": clean_string(risk.get("first_cleared_horizon") or required.get("first_cleared_horizon")),
                "timing_risk": clean_string(risk.get("timing_risk") or selector_row.get("time_sensitivity_summary")),
                "iv_risk": clean_string(risk.get("iv_risk") or sensitivity.get("iv_risk") or selector_row.get("iv_sensitivity_summary")),
                "success_dependency": clean_string(risk.get("success_dependency")),
                "benchmark_edge": finite_or_none(risk.get("benchmark_edge") or selector_row.get("difference_vs_stock")),
                "benchmark_return_edge": finite_or_none(risk.get("benchmark_return_edge")),
                "benchmark_note": benchmark_note,
                "horizon_fit_label": clean_string(selector_row.get("horizon_fit_label")),
                "target_beyond_expiry": bool(selector_row.get("target_beyond_expiry")),
                "confidence_label": clean_string(selector_row.get("confidence_label")),
                "coverage_flags": clean_string(selector_row.get("coverage_flags")),
                "selection_scope_label": clean_string(selector_row.get("selection_scope_label")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["objective_rank", "objective_score"], ascending=[True, False]).reset_index(drop=True)


def _candidate_comparison_rows(
    candidates: pd.DataFrame,
    *,
    required_path_summary: pd.DataFrame,
    path_risk_summary: pd.DataFrame,
    iv_path_sensitivity_summary: pd.DataFrame,
    goal: str,
    stock_path_name: str,
    iv_path_name: str,
    objective_mode: str,
    downside_tolerance: str,
    simplicity_preference: str,
    spot_price: float,
    target_price: float,
    comparison_capital: float,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    required_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in required_path_summary.loc[
            (required_path_summary.get("summary_scope", pd.Series(dtype=str)) == "candidate")
            & (required_path_summary.get("goal", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(goal).lower())
            & (required_path_summary.get("iv_variant", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(iv_path_name).lower())
        ].iterrows()
    }
    risk_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in path_risk_summary.loc[
            (path_risk_summary.get("summary_scope", pd.Series(dtype=str)) == "candidate")
            & (path_risk_summary.get("goal", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(goal).lower())
            & (path_risk_summary.get("stock_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(stock_path_name).lower())
            & (path_risk_summary.get("iv_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(iv_path_name).lower())
        ].iterrows()
    }
    sensitivity_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in iv_path_sensitivity_summary.loc[
            (iv_path_sensitivity_summary.get("summary_scope", pd.Series(dtype=str)) == "candidate")
            & (iv_path_sensitivity_summary.get("stock_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(stock_path_name).lower())
            & (iv_path_sensitivity_summary.get("active_iv_path_name", pd.Series(dtype=str)).astype(str).str.lower() == clean_string(iv_path_name).lower())
        ].iterrows()
    }
    direction = _strategy_direction(float(spot_price), float(target_price))
    frame = candidates.copy()
    frame["objective_score"] = frame.apply(
        lambda row: _selector_score(
            row,
            objective_mode=objective_mode,
            downside_tolerance=downside_tolerance,
            simplicity_preference=simplicity_preference,
        ),
        axis=1,
    )
    frame = frame.sort_values(
        ["objective_score", "difference_vs_stock", "profit_loss", "return_on_comparison_capital"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    frame["active_candidate_rank"] = range(1, len(frame.index) + 1)
    frame["affordability_label"] = frame.apply(
        lambda row: _affordability_label(row, comparison_capital=comparison_capital),
        axis=1,
    )
    frame["relevance_under_thesis"] = frame.apply(
        lambda row: _candidate_relevance_under_thesis(row, direction=direction),
        axis=1,
    )
    frame["weak_horizon_fit"] = frame["horizon_fit_label"].astype(str).str.lower().isin(["weak timing match", "poor horizon fit"])

    for column in [
        "required_path_difficulty",
        "path_gap_at_target",
        "first_cleared_horizon",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "benchmark_edge",
        "benchmark_return_edge",
        "benchmark_note",
    ]:
        frame[column] = None

    for index, row in frame.iterrows():
        candidate_slug = clean_string(row.get("candidate_slug"))
        required = required_lookup.get(candidate_slug, {})
        risk = risk_lookup.get(candidate_slug, {})
        sensitivity = sensitivity_lookup.get(candidate_slug, {})
        frame.at[index, "required_path_difficulty"] = clean_string(risk.get("required_path_difficulty") or required.get("required_path_difficulty"))
        frame.at[index, "path_gap_at_target"] = finite_or_none(risk.get("path_gap_at_target") or required.get("path_gap_at_target"))
        frame.at[index, "first_cleared_horizon"] = clean_string(risk.get("first_cleared_horizon") or required.get("first_cleared_horizon"))
        frame.at[index, "timing_risk"] = clean_string(risk.get("timing_risk") or row.get("time_sensitivity_summary"))
        frame.at[index, "iv_risk"] = clean_string(risk.get("iv_risk") or sensitivity.get("iv_risk") or row.get("iv_sensitivity_summary"))
        frame.at[index, "success_dependency"] = clean_string(risk.get("success_dependency"))
        frame.at[index, "benchmark_edge"] = finite_or_none(risk.get("benchmark_edge") or row.get("difference_vs_stock"))
        frame.at[index, "benchmark_return_edge"] = finite_or_none(risk.get("benchmark_return_edge") or row.get("difference_vs_stock_return_pct"))
        frame.at[index, "benchmark_note"] = clean_string(risk.get("benchmark_note") or row.get("benchmark_note"))

    notes = frame.apply(_candidate_win_lose_notes, axis=1)
    frame["why_this_candidate_wins"] = [item[0] for item in notes]
    frame["why_this_candidate_loses"] = [item[1] for item in notes]
    frame["stock_benchmark_decision"] = frame.apply(
        lambda row: _stock_benchmark_decision(
            strategy_family=clean_string(row.get("strategy_family")),
            difference_vs_stock=finite_or_none(row.get("difference_vs_stock")),
            difference_vs_stock_return_pct=finite_or_none(row.get("difference_vs_stock_return_pct")),
        ),
        axis=1,
    )
    return frame


def _best_group_member(group: pd.DataFrame) -> pd.Series:
    ordered = group.sort_values(
        ["active_candidate_rank", "objective_score", "difference_vs_stock", "profit_loss"],
        ascending=[True, False, False, False],
    )
    return ordered.iloc[0]


def _strike_comparison_rows(candidate_comparison: pd.DataFrame) -> pd.DataFrame:
    if candidate_comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = candidate_comparison.groupby(["strategy_family", "strike_label"], dropna=False)
    for (family, strike_label), group in grouped:
        best = _best_group_member(group)
        rows.append(
            {
                "strategy_family": clean_string(family),
                "strike_label": clean_string(strike_label),
                "best_expiry_date": clean_string(best.get("expiry_date")),
                "best_candidate_label": clean_string(best.get("candidate_label")),
                "active_candidate_rank": best.get("active_candidate_rank"),
                "objective_score": best.get("objective_score"),
                "difference_vs_stock": best.get("difference_vs_stock"),
                "difference_vs_stock_return_pct": best.get("difference_vs_stock_return_pct"),
                "required_path_difficulty": clean_string(best.get("required_path_difficulty")),
                "path_gap_at_target": finite_or_none(best.get("path_gap_at_target")),
                "timing_risk": clean_string(best.get("timing_risk")),
                "iv_risk": clean_string(best.get("iv_risk")),
                "success_dependency": clean_string(best.get("success_dependency")),
                "benchmark_note": clean_string(best.get("benchmark_note")),
                "best_source_quality": clean_string(best.get("source_quality")),
                "best_source_trust_label": clean_string(best.get("source_trust_label")),
                "horizon_fit_label": clean_string(best.get("horizon_fit_label")),
                "weak_horizon_fit": bool(best.get("weak_horizon_fit")),
                "target_beyond_expiry": bool(best.get("target_beyond_expiry")),
                "expiry_count_for_strike": int(group["expiry_date"].astype(str).nunique()),
                "candidate_count_for_strike": int(len(group.index)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["strategy_family", "active_candidate_rank", "strike_label"]).reset_index(drop=True)


def _expiry_fit_note(row: pd.Series | dict[str, Any]) -> str:
    if bool(row.get("target_beyond_expiry")):
        return "Target runs beyond this expiry, so the read is timing-sensitive."
    if bool(row.get("weak_horizon_fit")):
        return "This expiry is available, but timing fit is weaker than the requested thesis horizon."
    return "This expiry lines up reasonably with the active thesis horizon."


def _expiry_comparison_rows(candidate_comparison: pd.DataFrame) -> pd.DataFrame:
    if candidate_comparison.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouped = candidate_comparison.groupby(["strategy_family", "expiry_date"], dropna=False)
    for (family, expiry_date), group in grouped:
        best = _best_group_member(group)
        rows.append(
            {
                "strategy_family": clean_string(family),
                "expiry_date": clean_string(expiry_date),
                "best_strike_label": clean_string(best.get("strike_label")),
                "best_candidate_label": clean_string(best.get("candidate_label")),
                "active_candidate_rank": best.get("active_candidate_rank"),
                "objective_score": best.get("objective_score"),
                "difference_vs_stock": best.get("difference_vs_stock"),
                "difference_vs_stock_return_pct": best.get("difference_vs_stock_return_pct"),
                "required_path_difficulty": clean_string(best.get("required_path_difficulty")),
                "path_gap_at_target": finite_or_none(best.get("path_gap_at_target")),
                "timing_risk": clean_string(best.get("timing_risk")),
                "iv_risk": clean_string(best.get("iv_risk")),
                "success_dependency": clean_string(best.get("success_dependency")),
                "benchmark_note": clean_string(best.get("benchmark_note")),
                "best_source_quality": clean_string(best.get("source_quality")),
                "best_source_trust_label": clean_string(best.get("source_trust_label")),
                "horizon_fit_label": clean_string(best.get("horizon_fit_label")),
                "weak_horizon_fit": bool(best.get("weak_horizon_fit")),
                "target_beyond_expiry": bool(best.get("target_beyond_expiry")),
                "strike_count_for_expiry": int(group["strike_label"].astype(str).nunique()),
                "candidate_count_for_expiry": int(len(group.index)),
                "expiry_fit_note": _expiry_fit_note(best),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["strategy_family", "active_candidate_rank", "expiry_date"]).reset_index(drop=True)


def _build_contract_selection_core(
    *,
    ticker: str,
    snapshot_date: date | str,
    target_price: float,
    target_date: date | str | None = None,
    target_horizon: str | int | float | None = None,
    iv_shift_points: float = 0.0,
    comparison_capital: float = DEFAULT_COMPARISON_CAPITAL,
    strategy_families: Iterable[str] | None = None,
    strike_selection_mode: str = "top_n",
    expiry_selection_mode: str = "auto",
    top_n_strikes: int = DEFAULT_TOP_N_STRIKES,
    data_root: str | Path | None = None,
    goal: str = DEFAULT_GOAL,
    target_option_value: float | None = None,
    objective_mode: str = DEFAULT_OBJECTIVE_MODE,
    downside_tolerance: str = "medium",
    simplicity_preference: str = "medium",
    stock_path_preset: str = "slow_bull",
    stock_path_points: str | None = None,
    stock_path_mode: str = "mixed",
    stock_path_target_end: float | None = None,
    iv_path_preset: str = "flat",
    iv_path_points: str | None = None,
    iv_path_mode: str = "mixed",
    simulated_path_count: int = 18,
    representative_selection_mode: str = "goal_buckets",
    simulation_seed: int | None = None,
    thesis_target_price: float | None = None,
    thesis_target_date: date | str | None = None,
    single_option_candidate_slug: str | None = None,
    minimum_outperformance_multiple: float = 1.5,
    strong_outperformance_multiple: float = 2.0,
    required_winning_path_families: int = 2,
    entry_price_mode: str = "conservative_mid_plus_slippage",
    single_option_exit_rule: str = "sell_on_thesis_completion",
    single_option_target_return_pct: float = 0.50,
) -> ContractSelectionComputation:
    """Build one forward-looking strike, expiry, and path comparison bundle."""

    snapshot = parse_date(snapshot_date)
    if snapshot is None:
        raise ValueError(f"snapshot_date must be a valid date, got: {snapshot_date!r}")
    ticker_label = clean_string(ticker).upper()
    if target_price <= 0:
        raise ValueError("target_price must be positive.")
    if comparison_capital <= 0:
        raise ValueError("comparison_capital must be positive.")
    if goal == "target_option_value" and target_option_value is None:
        raise ValueError("target_option_value must be provided when goal=target_option_value.")
    if clean_string(objective_mode).lower() not in OBJECTIVE_MODE_CHOICES:
        raise ValueError(f"objective_mode must be one of {OBJECTIVE_MODE_CHOICES!r}.")
    if clean_string(downside_tolerance).lower() not in PREFERENCE_BAND_CHOICES:
        raise ValueError(f"downside_tolerance must be one of {PREFERENCE_BAND_CHOICES!r}.")
    if clean_string(simplicity_preference).lower() not in PREFERENCE_BAND_CHOICES:
        raise ValueError(f"simplicity_preference must be one of {PREFERENCE_BAND_CHOICES!r}.")
    if clean_string(stock_path_mode).lower() not in STOCK_PATH_MODE_CHOICES:
        raise ValueError(f"stock_path_mode must be one of {STOCK_PATH_MODE_CHOICES!r}.")
    if clean_string(iv_path_mode).lower() not in IV_PATH_MODE_CHOICES:
        raise ValueError(f"iv_path_mode must be one of {IV_PATH_MODE_CHOICES!r}.")
    if clean_string(representative_selection_mode).lower() not in REPRESENTATIVE_SELECTION_MODE_CHOICES:
        raise ValueError(
            f"representative_selection_mode must be one of {REPRESENTATIVE_SELECTION_MODE_CHOICES!r}."
        )
    if int(simulated_path_count) < 4:
        raise ValueError("simulated_path_count must be at least 4 so the representative buckets stay meaningful.")
    if float(minimum_outperformance_multiple) <= 0:
        raise ValueError("minimum_outperformance_multiple must be positive.")
    if float(strong_outperformance_multiple) < float(minimum_outperformance_multiple):
        raise ValueError("strong_outperformance_multiple must be at least minimum_outperformance_multiple.")
    if int(required_winning_path_families) < 1:
        raise ValueError("required_winning_path_families must be at least 1.")
    if clean_string(entry_price_mode).lower() not in SINGLE_OPTION_ENTRY_PRICE_MODES:
        raise ValueError(f"entry_price_mode must be one of {SINGLE_OPTION_ENTRY_PRICE_MODES!r}.")
    if clean_string(single_option_exit_rule).lower() not in SINGLE_OPTION_EXIT_RULE_CHOICES:
        raise ValueError(f"single_option_exit_rule must be one of {SINGLE_OPTION_EXIT_RULE_CHOICES!r}.")

    families = [clean_string(item).lower() for item in (strategy_families or SUPPORTED_CONTRACT_SELECTION_FAMILIES)]
    families = [item for item in families if item in SUPPORTED_CONTRACT_SELECTION_FAMILIES]
    if "long_stock" not in families:
        families = ["long_stock"] + families

    resolved_target_date, target_horizon_label, requested_days = _target_date_and_horizon(
        snapshot,
        target_date=target_date,
        target_horizon=target_horizon,
    )
    resolved_thesis_target_price = float(thesis_target_price) if thesis_target_price is not None else float(target_price)
    resolved_thesis_target_date = parse_date(thesis_target_date) if thesis_target_date is not None else resolved_target_date
    if resolved_thesis_target_date is None:
        raise ValueError(f"thesis_target_date must be a valid date, got: {thesis_target_date!r}")
    thesis_target_horizon_label = f"{max((resolved_thesis_target_date - snapshot).days, 0)}d"
    market_context = _load_candidate_chains(
        ticker_label,
        snapshot,
        target_date=resolved_target_date,
        data_root=data_root,
        expiry_selection_mode=expiry_selection_mode,
    )
    loaded_chains = [
        (
            item.scope,
            {
                "file_path": item.file_path,
                "storage_location": item.storage_location,
                "source_snapshot_date": item.source_snapshot_date,
                "expiry_date": item.expiry_date,
                "usable_quote_coverage_pct": item.usable_quote_coverage_pct,
                "quote_usable": item.quote_usable,
                "fallback_level": item.fallback_level,
                "source_quality": item.source_quality,
                "source_trust_label": item.source_trust_label,
                "source_quality_note": item.source_quality_note,
            },
            item.chain,
        )
        for item in market_context.resolved_chain_inputs
        if item.chain is not None
    ]
    selection_scope = dict(market_context.selection_scope)
    warnings = list(market_context.warnings)
    if not loaded_chains:
        raise ValueError("No local option-chain snapshots were available to build contract selection.")

    candidate_rows: list[dict[str, Any]] = []
    candidate_specs: list[dict[str, Any]] = []
    all_warnings: list[str] = list(warnings)
    available_expiries: list[str] = [item.expiry_date for item in market_context.resolved_chain_inputs]
    spot_price = float(market_context.spot_price or 0.0) if market_context.spot_price is not None else None
    risk_free_rate = float(market_context.risk_free_rate or 0.0)
    risk_free_rate_source = clean_string(market_context.risk_free_rate_source) or None
    risk_free_rate_series = clean_string(market_context.risk_free_rate_series) or None
    risk_free_rate_matched_date = (
        market_context.risk_free_rate_matched_date.isoformat()
        if market_context.risk_free_rate_matched_date is not None
        else None
    )
    risk_free_rate_note = clean_string(market_context.risk_free_rate_note) or None
    dividend_yield = float(
        ((market_context.research_context.get("dividend_assumption") or {}).get("dividend_yield"))
        or 0.0
    )
    for scope, row, chain in loaded_chains:
        rows, specs, chain_warnings = _discover_candidates_for_chain(
            chain,
            scope=scope,
            target_price=float(target_price),
            target_date=resolved_target_date,
            target_horizon_label=target_horizon_label,
            requested_days=requested_days,
            iv_shift_points=float(iv_shift_points),
            comparison_capital=float(comparison_capital),
            strategy_families=families,
            strike_selection_mode=strike_selection_mode,
            top_n_strikes=top_n_strikes,
            source_snapshot_date=clean_string(row.get("source_snapshot_date")),
            source_expiry_date=clean_string(row.get("expiry_date")) or None,
            source_storage_location=clean_string(row.get("storage_location")).lower(),
            source_snapshot_file=str(row.get("file_path") or ""),
            source_quote_coverage_pct=finite_or_none(row.get("usable_quote_coverage_pct")),
            source_quote_usable=bool(row.get("quote_usable")),
            fallback_level=clean_string(row.get("fallback_level")) or None,
            source_quality=clean_string(row.get("source_quality")) or None,
            source_trust_label=clean_string(row.get("source_trust_label")) or None,
            source_quality_note=clean_string(row.get("source_quality_note")) or None,
        )
        candidate_rows.extend(rows)
        candidate_specs.extend(specs)
        all_warnings.extend(chain_warnings)

    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        raise ValueError("No candidate contracts or structures could be built from the local chain data.")

    candidates = candidates.drop_duplicates(subset=["candidate_label", "strategy_family", "expiry_date", "selection_scope"]).reset_index(drop=True)
    if (candidates["strategy_family"] == "long_stock").sum() > 1:
        stock_rows = candidates.loc[candidates["strategy_family"] == "long_stock"].head(1)
        candidates = pd.concat([stock_rows, candidates.loc[candidates["strategy_family"] != "long_stock"]], ignore_index=True)
    spec_map = {clean_string(spec["candidate_slug"]): spec for spec in candidate_specs}
    candidates = candidates.loc[candidates["candidate_slug"].isin(spec_map.keys())].copy()
    candidates["robustness_score"] = (
        pd.to_numeric(candidates["difference_vs_stock"], errors="coerce").fillna(0.0)
        - pd.to_numeric(candidates["max_loss"], errors="coerce").fillna(0.0) * 0.01
    )
    candidates["objective_score"] = candidates.apply(
        lambda row: _selector_score(
            row,
            objective_mode=clean_string(objective_mode).lower(),
            downside_tolerance=clean_string(downside_tolerance).lower(),
            simplicity_preference=clean_string(simplicity_preference).lower(),
        ),
        axis=1,
    )
    candidates = candidates.sort_values(
        ["objective_score", "difference_vs_stock", "profit_loss", "return_on_comparison_capital"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    candidates["active_candidate_rank"] = range(1, len(candidates.index) + 1)

    horizon_specs = _ordered_horizon_specs(target_horizon_label, requested_days)
    stock_points = _parse_path_points(stock_path_points, allow_entry=float(spot_price or target_price))
    if not stock_points:
        stock_points = _default_stock_path_points(
            preset=stock_path_preset,
            entry_spot=float(spot_price or target_price),
            target_price=float(target_price),
            target_horizon_label=target_horizon_label,
        )
    iv_points = _parse_path_points(iv_path_points, allow_entry=float(iv_shift_points))
    if not iv_points:
        iv_points = _default_iv_path_points(
            preset=iv_path_preset,
            base_shift=float(iv_shift_points),
            target_horizon_label=target_horizon_label,
        )
    active_stock_path_name = (
        "custom_stock_path"
        if clean_string(stock_path_points)
        else (clean_string(stock_path_preset).lower() or "slow_bull")
    )
    active_iv_path_name = (
        "custom_iv_path"
        if clean_string(iv_path_points)
        else (clean_string(iv_path_preset).lower() or "flat")
    )

    ordered_specs = [spec_map[slug] for slug in candidates["candidate_slug"].tolist() if slug in spec_map]
    required_path = _required_path_rows(
        ordered_specs,
        horizon_specs=horizon_specs,
        comparison_capital=float(comparison_capital),
        target_price=float(target_price),
        target_option_value=target_option_value,
        active_iv_path_name=active_iv_path_name,
        active_iv_path_points=iv_points,
    )
    path_case_rows, path_case_summary = _path_case_rows(
        ordered_specs,
        horizon_specs=horizon_specs,
        comparison_capital=float(comparison_capital),
        target_price=float(target_price),
        target_horizon_label=target_horizon_label,
        iv_shift_points=float(iv_shift_points),
        stock_path_name=active_stock_path_name,
        stock_path_points=stock_points,
        iv_path_name=active_iv_path_name,
        iv_path_points=iv_points,
    )
    ranked_candidates, best_cards = _rankings(
        candidates,
        path_case_summary,
        stock_path_name=active_stock_path_name,
        iv_path_name=active_iv_path_name,
        comparison_capital=float(comparison_capital),
    )
    strategy_selector_rows = _strategy_selector_rows(
        candidates,
        requested_families=list(families),
        objective_mode=clean_string(objective_mode).lower(),
        downside_tolerance=clean_string(downside_tolerance).lower(),
        simplicity_preference=clean_string(simplicity_preference).lower(),
        spot_price=float(spot_price or 0.0),
        target_price=float(target_price),
        comparison_capital=float(comparison_capital),
    )
    strategy_selector_rankings, strategy_selector_cards = _selector_cards(
        strategy_selector_rows,
        objective_mode=clean_string(objective_mode).lower(),
        comparison_capital=float(comparison_capital),
    )
    path_case_cases = _path_case_definitions(
        entry_spot=float(spot_price or target_price),
        horizon_specs=horizon_specs,
    )
    path_case_iv_variants = _path_case_iv_variants(
        horizon_specs=horizon_specs,
        iv_shift_points=float(iv_shift_points),
        target_horizon_label=target_horizon_label,
        iv_path_name=active_iv_path_name,
        iv_path_points=iv_points,
    )
    path_case_chart_rows = _path_case_chart_rows(
        selector_rows=strategy_selector_rows,
        required_path_rows=required_path,
        horizon_specs=horizon_specs,
        assumed_path_points=stock_points,
        iv_variants=path_case_iv_variants,
        case_definitions=path_case_cases,
    )
    path_case_strategy_rows = _path_case_strategy_rows(
        specs=ordered_specs,
        candidates=candidates,
        selector_rows=strategy_selector_rows,
        required_path_rows=required_path,
        horizon_specs=horizon_specs,
        comparison_capital=float(comparison_capital),
        case_definitions=path_case_cases,
        iv_variants=path_case_iv_variants,
    )
    path_case_family_rankings = path_case_strategy_rows.copy()
    if not path_case_family_rankings.empty:
        path_case_family_rankings["benchmark_edge"] = path_case_family_rankings["difference_vs_stock"]
    default_strategy_family = clean_string(
        strategy_selector_rows.loc[strategy_selector_rows["available"] == True, "strategy_family"].iloc[0]  # noqa: E712
        if not strategy_selector_rows.loc[strategy_selector_rows["available"] == True].empty  # noqa: E712
        else (strategy_selector_rows.iloc[0].get("strategy_family") if not strategy_selector_rows.empty else "")
    )
    default_candidate_within_family = clean_string(
        strategy_selector_rows.loc[strategy_selector_rows["strategy_family"] == default_strategy_family, "winning_candidate_slug"].iloc[0]
        if default_strategy_family and not strategy_selector_rows.loc[strategy_selector_rows["strategy_family"] == default_strategy_family].empty
        else ""
    )
    default_contract_for_path_explorer = default_candidate_within_family
    if not default_contract_for_path_explorer and not candidates.empty:
        default_contract_for_path_explorer = clean_string(candidates.iloc[0].get("candidate_slug"))
    if default_strategy_family and default_contract_for_path_explorer:
        matching = candidates.loc[
            (candidates["strategy_family"] == default_strategy_family)
            & (candidates["candidate_slug"] == default_contract_for_path_explorer)
        ]
        if matching.empty:
            within_family = candidates.loc[candidates["strategy_family"] == default_strategy_family]
            if not within_family.empty:
                default_contract_for_path_explorer = clean_string(within_family.iloc[0].get("candidate_slug"))
    path_case_candidate_rankings = _path_case_candidate_rankings(
        specs=ordered_specs,
        candidates=candidates,
        required_path_rows=required_path,
        ranked_candidates=ranked_candidates,
        horizon_specs=horizon_specs,
        comparison_capital=float(comparison_capital),
        case_definitions=path_case_cases,
        iv_variants=path_case_iv_variants,
    )
    assumed_stock_path = _interpolated_path(
        stock_points,
        horizon_specs,
        default_value=float(next(iter(stock_points.values()), target_price)),
    )
    assumed_iv_path = _interpolated_path(
        iv_points,
        horizon_specs,
        default_value=float(next(iter(iv_points.values()), iv_shift_points)),
    )
    required_path = _annotate_required_path_rows(
        required_path,
        assumed_stock_path=assumed_stock_path,
    )
    family_representatives = {
        clean_string(row.get("strategy_family")): clean_string(row.get("winning_candidate_slug"))
        for _, row in strategy_selector_rows.iterrows()
        if clean_string(row.get("strategy_family")) and clean_string(row.get("winning_candidate_slug"))
    }
    representative_candidates = [
        {
            "trace_scope": "family_representative",
            "series_label": clean_string(row.get("strategy_label")) or clean_string(row.get("strategy_family")).replace("_", " ").title(),
            "candidate_slug": clean_string(row.get("winning_candidate_slug")),
            "candidate_label": clean_string(row.get("winning_candidate_label")),
            "strategy_family": clean_string(row.get("strategy_family")),
            "spec": next(
                (
                    spec
                    for spec in ordered_specs
                    if clean_string(spec.get("candidate_slug")) == clean_string(row.get("winning_candidate_slug"))
                ),
                None,
            ),
        }
        for _, row in strategy_selector_rows.iterrows()
        if clean_string(row.get("winning_candidate_slug"))
    ]
    top_candidate_row = candidates.iloc[0].to_dict() if not candidates.empty else {}
    top_candidate_spec = next(
        (
            spec
            for spec in ordered_specs
            if clean_string(spec.get("candidate_slug")) == clean_string(top_candidate_row.get("candidate_slug"))
        ),
        None,
    )
    assumed_path_trace_rows = _build_assumed_path_trace_rows(
        representative_candidates=representative_candidates,
        horizon_specs=horizon_specs,
        stock_path=assumed_stock_path,
        iv_path=assumed_iv_path,
        comparison_capital=float(comparison_capital),
        evaluate_point=_evaluate_at_point,
        include_top_candidate={
            "trace_scope": "top_candidate",
            "series_label": "Top Candidate",
            "candidate_slug": clean_string(top_candidate_row.get("candidate_slug")),
            "candidate_label": clean_string(top_candidate_row.get("candidate_label")),
            "strategy_family": clean_string(top_candidate_row.get("strategy_family")),
            "spec": top_candidate_spec,
        }
        if top_candidate_spec is not None
        else None,
    )
    compare_vs_stock_path_rows = _build_compare_vs_stock_path_rows(assumed_path_trace_rows)
    iv_path_trace_rows = _build_iv_path_trace_rows(
        horizon_specs=horizon_specs,
        active_iv_path_name=active_iv_path_name,
        active_iv_path=assumed_iv_path,
        comparison_iv_paths=path_case_iv_variants.get("path_preset") or {},
    )
    required_path_summary = _summarize_required_path_rows(
        required_path,
        assumed_path=assumed_stock_path,
        family_representatives=family_representatives,
    )
    iv_path_sensitivity_summary = _build_iv_path_sensitivity_summary(
        path_case_summary=path_case_summary,
        stock_path_name=active_stock_path_name,
        active_iv_path_name=active_iv_path_name,
        family_representatives=family_representatives,
    )
    path_risk_summary = _build_path_risk_summary(
        required_path_summary=required_path_summary,
        assumed_path_trace_rows=assumed_path_trace_rows,
        compare_vs_stock_path_rows=compare_vs_stock_path_rows,
        candidate_summary=candidates,
        iv_path_sensitivity_summary=iv_path_sensitivity_summary,
        goal=goal,
        stock_path_name=active_stock_path_name,
        iv_path_name=active_iv_path_name,
    )
    family_comparison = _family_comparison_rows(
        strategy_selector_rows,
        selector_rankings=strategy_selector_rankings,
        candidate_summary=candidates,
        required_path_summary=required_path_summary,
        path_risk_summary=path_risk_summary,
        iv_path_sensitivity_summary=iv_path_sensitivity_summary,
        goal=goal,
        stock_path_name=active_stock_path_name,
        iv_path_name=active_iv_path_name,
    )
    candidate_comparison = _candidate_comparison_rows(
        candidates,
        required_path_summary=required_path_summary,
        path_risk_summary=path_risk_summary,
        iv_path_sensitivity_summary=iv_path_sensitivity_summary,
        goal=goal,
        stock_path_name=active_stock_path_name,
        iv_path_name=active_iv_path_name,
        objective_mode=clean_string(objective_mode).lower(),
        downside_tolerance=clean_string(downside_tolerance).lower(),
        simplicity_preference=clean_string(simplicity_preference).lower(),
        spot_price=float(spot_price or 0.0),
        target_price=float(target_price),
        comparison_capital=float(comparison_capital),
    )
    strike_comparison = _strike_comparison_rows(candidate_comparison)
    expiry_comparison = _expiry_comparison_rows(candidate_comparison)
    if not candidate_comparison.empty:
        candidates = candidate_comparison.copy()
    simulation_outputs = _build_path_simulation_outputs(
        specs=ordered_specs,
        candidate_rows=candidates,
        required_path_summary=required_path_summary,
        family_representatives=family_representatives,
        snapshot_date=snapshot,
        target_date=resolved_target_date,
        target_horizon_days=int(requested_days),
        target_price=float(target_price),
        stock_path_name=active_stock_path_name,
        stock_path_points=stock_points,
        stock_path_mode=clean_string(stock_path_mode).lower(),
        stock_path_target_end=float(stock_path_target_end or target_price),
        iv_path_name=active_iv_path_name,
        iv_path_points=iv_points,
        iv_path_mode=clean_string(iv_path_mode).lower(),
        goal=goal,
        target_option_value=target_option_value,
        comparison_capital=float(comparison_capital),
        simulated_path_count=int(simulated_path_count),
        representative_selection_mode=clean_string(representative_selection_mode).lower(),
        simulation_seed=simulation_seed,
        top_candidate_slug=clean_string(top_candidate_row.get("candidate_slug")),
    )
    long_call_view_outputs = _build_assumed_path_long_call_views(
        specs=ordered_specs,
        candidate_rows=candidates,
        snapshot_date=snapshot,
        target_date=resolved_target_date,
        stock_path_name=active_stock_path_name,
        stock_path_points=stock_points,
        iv_path_name=active_iv_path_name,
        iv_path_points=iv_points,
        comparison_capital=float(comparison_capital),
        goal=goal,
        target_option_value=target_option_value,
    )
    path_view_tables, path_centric_focus_paths = _build_path_centric_long_call_views(
        specs=ordered_specs,
        candidate_rows=candidates,
        snapshot_date=snapshot,
        target_date=resolved_target_date,
        target_price=float(target_price),
        target_horizon_label=target_horizon_label,
        entry_spot=float(spot_price or target_price),
        iv_path_name=active_iv_path_name,
        iv_path_points=iv_points,
        comparison_capital=float(comparison_capital),
        goal=goal,
        target_option_value=target_option_value,
        objective_mode=clean_string(objective_mode).lower(),
        downside_tolerance=clean_string(downside_tolerance).lower(),
        simplicity_preference=clean_string(simplicity_preference).lower(),
        active_stock_path_name=active_stock_path_name,
        active_stock_path_points=stock_points,
    )
    gallery_outputs = _build_assumed_path_galleries(
        snapshot_date=snapshot,
        target_date=resolved_target_date,
        target_horizon_label=target_horizon_label,
        target_price=float(target_price),
        entry_spot=float(spot_price or target_price),
        stock_path_name=active_stock_path_name,
        stock_path_points=stock_points,
        iv_path_name=active_iv_path_name,
        iv_path_points=iv_points,
    )
    decision_highlight_outputs = build_decision_highlights(
        ticker=ticker_label,
        candidate_comparison=candidates,
        family_comparison=family_comparison,
        path_view_tables=path_view_tables,
        analysis_trust_level=clean_string(selection_scope.get("analysis_trust_level")),
    )
    entry_justification_outputs = build_entry_justification(
        ticker=ticker_label,
        goal=goal,
        target_price=float(target_price),
        target_date=resolved_target_date,
        active_iv_path_name=active_iv_path_name,
        candidate_comparison=candidates,
        required_path_rows=required_path,
        required_path_summary=required_path_summary,
        action_board_candidates=decision_highlight_outputs.action_board_candidates,
    )
    thesis_mode_outputs = _build_thesis_mode_outputs(
        ticker=ticker_label,
        specs=ordered_specs,
        candidate_rows=candidates,
        snapshot_date=snapshot,
        thesis_target_price=float(resolved_thesis_target_price),
        thesis_target_date=resolved_thesis_target_date,
        thesis_horizon_label=thesis_target_horizon_label,
        entry_spot=float(spot_price or target_price),
        comparison_capital=float(comparison_capital),
        objective_mode=clean_string(objective_mode).lower(),
        downside_tolerance=clean_string(downside_tolerance).lower(),
        simplicity_preference=clean_string(simplicity_preference).lower(),
    )
    stress_test_outputs = _build_stress_test_outputs(
        ticker=ticker_label,
        thesis_target_price=float(resolved_thesis_target_price),
        thesis_target_date=resolved_thesis_target_date,
        snapshot_date=snapshot,
        entry_spot=float(spot_price or target_price),
        comparison_capital=float(comparison_capital),
        thesis_candidate_ranking=thesis_mode_outputs["thesis_candidate_ranking"],
        thesis_stock_vs_option_summary=thesis_mode_outputs["thesis_stock_vs_option_summary"],
        top_candidate_cards=decision_highlight_outputs.top_candidate_cards,
    )
    single_option_outputs = _build_single_option_decision_outputs(
        ticker=ticker_label,
        specs=ordered_specs,
        candidate_rows=candidates,
        bullish_action_board=decision_highlight_outputs.bullish_long_call_action_board,
        required_stock_path_to_buy=entry_justification_outputs.required_stock_path_to_buy,
        snapshot_date=snapshot,
        target_price=float(resolved_thesis_target_price),
        target_date=resolved_thesis_target_date,
        target_horizon_label=thesis_target_horizon_label,
        entry_spot=float(spot_price or target_price),
        active_iv_path_name=active_iv_path_name,
        active_iv_path_points=iv_points,
        comparison_capital=float(comparison_capital),
        candidate_slug=single_option_candidate_slug,
        minimum_outperformance_multiple=float(minimum_outperformance_multiple),
        strong_outperformance_multiple=float(strong_outperformance_multiple),
        required_winning_path_families=int(required_winning_path_families),
        entry_price_mode=clean_string(entry_price_mode).lower(),
        exit_rule=clean_string(single_option_exit_rule).lower(),
        target_return_pct=float(single_option_target_return_pct),
    )
    chain_overview_outputs = _build_chain_overview_outputs(
        ticker=ticker_label,
        specs=ordered_specs,
        candidate_rows=candidates,
        bullish_action_board=decision_highlight_outputs.bullish_long_call_action_board,
        required_stock_path_to_buy=entry_justification_outputs.required_stock_path_to_buy,
        candidate_tradeoff_matrix=decision_highlight_outputs.candidate_tradeoff_matrix,
        candidate_robustness_summary=decision_highlight_outputs.candidate_robustness_summary,
        premium_sensitivity_summary=stress_test_outputs["premium_sensitivity_summary"],
        timing_slip_summary=stress_test_outputs["timing_slip_summary"],
        target_stress_summary=stress_test_outputs["target_stress_summary"],
        snapshot_date=snapshot,
        target_price=float(target_price),
        target_date=resolved_target_date,
        target_horizon_label=target_horizon_label,
        entry_spot=float(spot_price or target_price),
        active_iv_path_points=iv_points,
        comparison_capital=float(comparison_capital),
        minimum_outperformance_multiple=float(minimum_outperformance_multiple),
        strong_outperformance_multiple=float(strong_outperformance_multiple),
        required_winning_path_families=int(required_winning_path_families),
    )
    compare_vs_stock = candidates[
        [
            "candidate_slug",
            "candidate_label",
            "strategy_family",
            "expiry_date",
            "strike_label",
            "comparison_capital",
            "unit_capital_required",
            "affordable_units",
            "fully_implementable_with_budget",
            "estimated_value",
            "profit_loss",
            "return_on_comparison_capital",
            "stock_estimated_value",
            "stock_profit_loss",
            "stock_return_on_comparison_capital",
            "difference_vs_stock",
            "difference_vs_stock_return_pct",
            "benchmark_note",
        ]
    ].copy()
    status = "ok"
    unique_warnings = _dedupe(all_warnings)
    if selection_scope.get("used_nearby_snapshot_fallback"):
        status = "partial"
    if bool(candidates.get("target_beyond_expiry", pd.Series(dtype=bool)).fillna(False).any()):
        status = "partial"
        unique_warnings.append(
            "Target date extends beyond one or more candidate expiries, so affected rows use expiry-clamped estimates and should be treated as weaker timing matches."
        )
    if len(candidates.index) < 4:
        status = "partial"
        unique_warnings.append("Candidate coverage is sparse, so treat the rankings as directional rather than exhaustive.")
    shareability_status = "mostly_self_contained"
    generated_at = _utc_now_iso()
    generated_stamp = generated_at.replace(":", "").replace("-", "").lower()
    generated_suffix = generated_stamp[4:13] if len(generated_stamp) >= 13 else generated_stamp[-9:]
    run_slug = (
        f"{slugify(clean_string(target_horizon_label) or 'target')}"
        f"-{_short_slug_token(clean_string(goal) or DEFAULT_GOAL, limit=3)}"
        f"-{_short_slug_token(active_stock_path_name or 'stock-path', limit=3)}"
        f"-{_short_slug_token(active_iv_path_name or 'iv-path', limit=3)}"
        f"-{generated_suffix}"
    )
    calibration = _calibration_context(candidates)
    selector_notes = list(calibration.get("notes") or [])
    selector_notes.extend(
        [
            "A slower thesis usually favors more time or simpler exposure, because delay erodes many long-premium trades before the stock path arrives.",
            "If IV normalizes lower from an elevated starting point, long-premium trades can lag stock even when direction is broadly right.",
            "Use Strategy Selector to choose the family first, then Path & Contract Explorer to decide the exact strike and expiry.",
        ]
    )
    strategy_selector_context = {
        "objective_mode": clean_string(objective_mode).lower(),
        "downside_tolerance": clean_string(downside_tolerance).lower(),
        "simplicity_preference": clean_string(simplicity_preference).lower(),
        "notes": _dedupe(selector_notes),
    }
    strategy_selector_defaults = {
        "objective_mode": clean_string(objective_mode).lower(),
        "downside_tolerance": clean_string(downside_tolerance).lower(),
        "simplicity_preference": clean_string(simplicity_preference).lower(),
        "default_strategy_family": default_strategy_family,
        "default_candidate_within_family": default_candidate_within_family,
    }
    path_explorer_defaults = {
        "default_strategy_family": default_strategy_family,
        "default_contract_for_path_explorer": default_contract_for_path_explorer,
        "goal": clean_string(goal),
    }
    path_case_defaults = _path_case_defaults(
        goal=goal,
        active_iv_path_name=active_iv_path_name,
        iv_shift_points=float(iv_shift_points),
        default_strategy_family=default_strategy_family,
        default_candidate_within_family=default_candidate_within_family,
    )

    summary_markdown = (
        f"Contract Selection compares local candidate contracts and simple spreads for {ticker_label} "
        f"from the {snapshot.isoformat()} snapshot under a target of {float(target_price):.2f} by {resolved_target_date.isoformat()} "
        f"with an IV shift of {float(iv_shift_points):+.2f}. Required paths stay explicit, while representative simulated paths show how the thesis can miss, almost work, or work under the same valuation engine. Rankings are heuristic and assumption-driven."
    )
    metadata = {
        "report_kind": "contract_selection",
        "ticker": ticker_label,
        "snapshot_date": snapshot.isoformat(),
        "target_price": float(target_price),
        "target_date": resolved_target_date.isoformat(),
        "target_horizon": target_horizon_label,
        "target_horizon_days": int(requested_days),
        "thesis_target_price": float(resolved_thesis_target_price),
        "thesis_target_date": resolved_thesis_target_date.isoformat(),
        "thesis_stock_path_presets": list(THESIS_STOCK_PATH_PRESETS),
        "thesis_iv_path_presets": list(THESIS_IV_PATH_PRESETS),
        "comparison_capital": float(comparison_capital),
        "strategy_families": list(families),
        "selection_scope": selection_scope,
        "status": status,
        "shareability_status": shareability_status,
        "ranking_modes": ranked_candidates["ranking_mode"].tolist() if not ranked_candidates.empty else [],
        "available_expiries": sorted(set(available_expiries)),
        "used_nearby_snapshot_fallback": bool(selection_scope.get("used_nearby_snapshot_fallback")),
        "warnings": unique_warnings,
        "best_candidate_cards": best_cards,
        "spot_price": float(spot_price or 0.0),
        "spot_price_source": clean_string(market_context.spot_price_source) or None,
        "spot_price_matched_date": market_context.spot_price_matched_date.isoformat() if market_context.spot_price_matched_date else None,
        "spot_field_used": clean_string(market_context.spot_price_field_used) or None,
        "spot_used_prior_date": bool(market_context.spot_price_used_prior_date),
        "spot_price_note": clean_string(market_context.spot_price_note) or None,
        "spot_quality_note": clean_string(market_context.spot_quality_note) or None,
        "ibkr_same_day_spot_attempted": bool(market_context.ibkr_same_day_spot_attempted),
        "ibkr_same_day_spot_rejected_reason": clean_string(market_context.ibkr_same_day_spot_rejected_reason) or None,
        "risk_free_rate": risk_free_rate,
        "risk_free_rate_source": risk_free_rate_source,
        "risk_free_rate_series": risk_free_rate_series,
        "risk_free_rate_matched_date": risk_free_rate_matched_date,
        "risk_free_rate_note": risk_free_rate_note,
        "dividend_yield": dividend_yield,
        "research_context": dict(market_context.research_context),
        "research_context_expiry_used": market_context.research_context_expiry_used,
        "iv_shift_points": float(iv_shift_points),
        "goal": goal,
        "target_option_value": target_option_value,
        "objective_mode": clean_string(objective_mode).lower(),
        "downside_tolerance": clean_string(downside_tolerance).lower(),
        "simplicity_preference": clean_string(simplicity_preference).lower(),
        "stock_path_name": active_stock_path_name,
        "stock_path_preset": clean_string(stock_path_preset).lower(),
        "stock_path_points": stock_points,
        "stock_path_mode": clean_string(stock_path_mode).lower(),
        "stock_path_target_end": float(stock_path_target_end or target_price),
        "iv_path_name": active_iv_path_name,
        "iv_path_preset": clean_string(iv_path_preset).lower(),
        "iv_path_points": iv_points,
        "iv_path_mode": clean_string(iv_path_mode).lower(),
        "path_centric_focus_paths": path_centric_focus_paths,
        "simulated_path_count": int(simulated_path_count),
        "representative_selection_mode": clean_string(representative_selection_mode).lower(),
        "simulation_seed": simulation_outputs.get("simulation_context", {}).get("simulation_seed"),
        "simulation_context": simulation_outputs.get("simulation_context", {}),
        "calibration_context": calibration,
        "strategy_selector_defaults": strategy_selector_defaults,
        "strategy_selector_context": strategy_selector_context,
        "strategy_selector_best_cards": strategy_selector_cards,
        "path_explorer_defaults": path_explorer_defaults,
        "path_case_defaults": path_case_defaults,
        "path_case_cases": path_case_cases,
        "source_snapshot_files": list(selection_scope.get("source_snapshot_files") or []),
        "source_snapshot_storage_locations": list(selection_scope.get("source_snapshot_storage_locations") or []),
        "source_snapshot_dates": list(selection_scope.get("source_snapshot_dates") or []),
        "rejected_sparse_same_day_ibkr_files": list(selection_scope.get("rejected_sparse_same_day_ibkr_files") or []),
        "used_full_quoted_ibkr_same_date": bool(selection_scope.get("used_full_quoted_ibkr_same_date")),
        "analysis_trust_level": clean_string(selection_scope.get("analysis_trust_level")) or None,
        "analysis_trust_note": clean_string(selection_scope.get("analysis_trust_note")) or None,
        "decision_highlights": (
            decision_highlight_outputs.decision_highlights.head(12).to_dict(orient="records")
            if not decision_highlight_outputs.decision_highlights.empty
            else []
        ),
        "action_board": (
            decision_highlight_outputs.action_board_candidates.head(16).to_dict(orient="records")
            if not decision_highlight_outputs.action_board_candidates.empty
            else []
        ),
        "bullish_action_board": (
            decision_highlight_outputs.bullish_long_call_action_board.head(16).to_dict(orient="records")
            if not decision_highlight_outputs.bullish_long_call_action_board.empty
            else []
        ),
        "entry_justification": (
            entry_justification_outputs.entry_justification_candidates.head(16).to_dict(orient="records")
            if not entry_justification_outputs.entry_justification_candidates.empty
            else []
        ),
        "thesis_mode": (
            thesis_mode_outputs["thesis_candidate_ranking"].head(12).to_dict(orient="records")
            if not thesis_mode_outputs["thesis_candidate_ranking"].empty
            else []
        ),
        "stress_tests": (
            stress_test_outputs["stress_transition_summary"].head(8).to_dict(orient="records")
            if not stress_test_outputs["stress_transition_summary"].empty
            else []
        ),
        "single_option_decision": (
            single_option_outputs["single_option_decision_summary"].head(1).to_dict(orient="records")
            if not single_option_outputs["single_option_decision_summary"].empty
            else []
        ),
        "chain_overview": (
            chain_overview_outputs["chain_overview_summary"].to_dict(orient="records")
            if not chain_overview_outputs["chain_overview_summary"].empty
            else []
        ),
        "single_option_defaults": {
            "single_option_candidate_slug": clean_string(single_option_candidate_slug) or None,
            "minimum_outperformance_multiple": float(minimum_outperformance_multiple),
            "strong_outperformance_multiple": float(strong_outperformance_multiple),
            "required_winning_path_families": int(required_winning_path_families),
            "entry_price_mode": clean_string(entry_price_mode).lower(),
            "single_option_exit_rule": clean_string(single_option_exit_rule).lower(),
            "single_option_target_return_pct": float(single_option_target_return_pct),
            "iv_base_modes": list(SINGLE_OPTION_DEFAULT_IV_MODES),
            "outcome_labels": list(SINGLE_OPTION_OUTCOME_LABELS),
        },
        "chain_overview_defaults": {
            "minimum_outperformance_multiple": float(minimum_outperformance_multiple),
            "strong_outperformance_multiple": float(strong_outperformance_multiple),
            "required_winning_path_families": int(required_winning_path_families),
            "candidate_scope": "bullish_long_calls_only",
            "stock_benchmark": "long_stock_baseline",
        },
        "trusted_expiry_count": int(selection_scope.get("trusted_expiry_count") or 0),
        "fallback_only_expiry_count": int(selection_scope.get("fallback_only_expiry_count") or 0),
        "default_strategy_family": default_strategy_family,
        "default_candidate_within_family": default_candidate_within_family,
        "default_contract_for_path_explorer": default_contract_for_path_explorer,
        "run_slug": run_slug,
        "generated_at": generated_at,
    }
    return ContractSelectionComputation(
        ticker=ticker_label,
        snapshot_date=snapshot,
        target_price=float(target_price),
        target_date=resolved_target_date,
        target_horizon_label=target_horizon_label,
        target_horizon_days=int(requested_days),
        iv_shift_points=float(iv_shift_points),
        comparison_capital=float(comparison_capital),
        strategy_families=list(families),
        spot_price=float(spot_price or 0.0),
        spot_price_source=clean_string(market_context.spot_price_source) or None,
        spot_price_matched_date=market_context.spot_price_matched_date.isoformat() if market_context.spot_price_matched_date else None,
        spot_price_field_used=clean_string(market_context.spot_price_field_used) or None,
        spot_price_used_prior_date=bool(market_context.spot_price_used_prior_date),
        spot_price_note=clean_string(market_context.spot_price_note) or None,
        spot_quality_note=clean_string(market_context.spot_quality_note) or None,
        ibkr_same_day_spot_attempted=bool(market_context.ibkr_same_day_spot_attempted),
        ibkr_same_day_spot_rejected_reason=clean_string(market_context.ibkr_same_day_spot_rejected_reason) or None,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        available_expiries=sorted(set(available_expiries)),
        selection_scope=selection_scope,
        chain_source_summary=market_context.chain_source_summary.copy(),
        market_context_summary=market_context.market_context_summary.copy(),
        research_context=dict(market_context.research_context),
        status=status,
        shareability_status=shareability_status,
        warnings=unique_warnings,
        goal=goal,
        target_option_value=target_option_value,
        objective_mode=clean_string(objective_mode).lower(),
        downside_tolerance=clean_string(downside_tolerance).lower(),
        simplicity_preference=clean_string(simplicity_preference).lower(),
        stock_path_name=active_stock_path_name,
        iv_path_name=active_iv_path_name,
        stock_path_points=stock_points,
        iv_path_points=iv_points,
        stock_path_mode=clean_string(stock_path_mode).lower(),
        stock_path_target_end=float(stock_path_target_end or target_price),
        iv_path_mode=clean_string(iv_path_mode).lower(),
        simulated_path_count=int(simulated_path_count),
        representative_selection_mode=clean_string(representative_selection_mode).lower(),
        simulation_seed=simulation_outputs.get("simulation_context", {}).get("simulation_seed"),
        run_slug=run_slug,
        generated_at=generated_at,
        candidate_summary=candidates,
        ranked_candidates=ranked_candidates,
        compare_vs_stock=compare_vs_stock,
        required_path_rows=required_path,
        required_path_summary=required_path_summary,
        assumed_path_trace_rows=assumed_path_trace_rows,
        iv_path_trace_rows=iv_path_trace_rows,
        compare_vs_stock_path_rows=compare_vs_stock_path_rows,
        iv_path_sensitivity_summary=iv_path_sensitivity_summary,
        path_risk_summary=path_risk_summary,
        path_case_rows=path_case_rows,
        path_case_summary=path_case_summary,
        path_case_chart_rows=path_case_chart_rows,
        path_case_strategy_rows=path_case_strategy_rows,
        path_case_family_rankings=path_case_family_rankings,
        path_case_candidate_rankings=path_case_candidate_rankings,
        strategy_selector_rows=strategy_selector_rows,
        strategy_selector_rankings=strategy_selector_rankings,
        family_comparison=family_comparison,
        candidate_comparison=candidates,
        strike_comparison=strike_comparison,
        expiry_comparison=expiry_comparison,
        stock_path_library=gallery_outputs["stock_path_library"],
        stock_path_gallery=gallery_outputs["stock_path_gallery"],
        iv_path_gallery=gallery_outputs["iv_path_gallery"],
        stock_path_examples=simulation_outputs["stock_path_examples"],
        iv_path_examples=simulation_outputs["iv_path_examples"],
        path_pair_summary=simulation_outputs["path_pair_summary"],
        option_value_over_path=simulation_outputs["option_value_over_path"],
        compare_vs_stock_over_path=simulation_outputs["compare_vs_stock_over_path"],
        representative_paths_summary=simulation_outputs["representative_paths_summary"],
        strike_comparison_under_path=simulation_outputs["strike_comparison_under_path"],
        expiry_comparison_under_path=simulation_outputs["expiry_comparison_under_path"],
        long_call_value_over_path_strike_view=long_call_view_outputs["long_call_value_over_path_strike_view"],
        long_call_value_over_path_expiry_view=long_call_view_outputs["long_call_value_over_path_expiry_view"],
        long_call_value_over_path_best_of=long_call_view_outputs["long_call_value_over_path_best_of"],
        decision_highlights=decision_highlight_outputs.decision_highlights,
        decision_highlights_explanations=decision_highlight_outputs.decision_highlights_explanations,
        candidate_robustness_summary=decision_highlight_outputs.candidate_robustness_summary,
        candidate_tradeoff_matrix=decision_highlight_outputs.candidate_tradeoff_matrix,
        stock_vs_option_takeaways=decision_highlight_outputs.stock_vs_option_takeaways,
        highlights_score_breakdown=decision_highlight_outputs.highlights_score_breakdown,
        highlights_markdown=decision_highlight_outputs.highlights_markdown,
        action_board_candidates=decision_highlight_outputs.action_board_candidates,
        buy_now_candidates=decision_highlight_outputs.buy_now_candidates,
        watchlist_candidates=decision_highlight_outputs.watchlist_candidates,
        avoid_for_now_candidates=decision_highlight_outputs.avoid_for_now_candidates,
        prefer_stock_instead=decision_highlight_outputs.prefer_stock_instead,
        decision_triggers=decision_highlight_outputs.decision_triggers,
        action_board_score_breakdown=decision_highlight_outputs.action_board_score_breakdown,
        action_board_explanations=decision_highlight_outputs.action_board_explanations,
        action_board_markdown=decision_highlight_outputs.action_board_markdown,
        bullish_long_call_action_board=decision_highlight_outputs.bullish_long_call_action_board,
        bullish_long_call_watchlist=decision_highlight_outputs.bullish_long_call_watchlist,
        bullish_long_call_avoid=decision_highlight_outputs.bullish_long_call_avoid,
        bullish_long_call_triggers=decision_highlight_outputs.bullish_long_call_triggers,
        bullish_long_call_score_breakdown=decision_highlight_outputs.bullish_long_call_score_breakdown,
        other_structures_summary=decision_highlight_outputs.other_structures_summary,
        stock_preference_summary=decision_highlight_outputs.stock_preference_summary,
        bullish_action_board_markdown=decision_highlight_outputs.bullish_action_board_markdown,
        top_candidate_cards=decision_highlight_outputs.top_candidate_cards,
        top_candidate_cards_markdown=decision_highlight_outputs.top_candidate_cards_markdown,
        other_structures_markdown=decision_highlight_outputs.other_structures_markdown,
        entry_justification_candidates=entry_justification_outputs.entry_justification_candidates,
        required_stock_path_to_buy=entry_justification_outputs.required_stock_path_to_buy,
        required_move_summary=entry_justification_outputs.required_move_summary,
        required_move_vs_stock=entry_justification_outputs.required_move_vs_stock,
        required_iv_support_summary=entry_justification_outputs.required_iv_support_summary,
        entry_barrier_summary=entry_justification_outputs.entry_barrier_summary,
        entry_justification_markdown=entry_justification_outputs.entry_justification_markdown,
        thesis_target_price=float(resolved_thesis_target_price),
        thesis_target_date=resolved_thesis_target_date,
        thesis_path_gallery=thesis_mode_outputs["thesis_path_gallery"],
        thesis_iv_gallery=thesis_mode_outputs["thesis_iv_gallery"],
        thesis_mode_candidates=thesis_mode_outputs["thesis_mode_candidates"],
        thesis_path_family_summary=thesis_mode_outputs["thesis_path_family_summary"],
        thesis_iv_family_summary=thesis_mode_outputs["thesis_iv_family_summary"],
        thesis_candidate_ranking=thesis_mode_outputs["thesis_candidate_ranking"],
        max_justified_premium=thesis_mode_outputs["max_justified_premium"],
        current_vs_justified_premium=thesis_mode_outputs["current_vs_justified_premium"],
        thesis_required_move_summary=thesis_mode_outputs["thesis_required_move_summary"],
        thesis_stock_vs_option_summary=thesis_mode_outputs["thesis_stock_vs_option_summary"],
        thesis_mode_markdown=thesis_mode_outputs["thesis_mode_markdown"],
        candidate_stress_grid=stress_test_outputs["candidate_stress_grid"],
        premium_sensitivity_summary=stress_test_outputs["premium_sensitivity_summary"],
        timing_slip_summary=stress_test_outputs["timing_slip_summary"],
        target_stress_summary=stress_test_outputs["target_stress_summary"],
        stress_transition_summary=stress_test_outputs["stress_transition_summary"],
        stress_tests_markdown=stress_test_outputs["stress_tests_markdown"],
        chain_overview_summary=chain_overview_outputs["chain_overview_summary"],
        chain_overview_candidates=chain_overview_outputs["chain_overview_candidates"],
        chain_overview_markdown=chain_overview_outputs["chain_overview_markdown"],
        single_option_decision_summary=single_option_outputs["single_option_decision_summary"],
        single_option_decision_path_selections=single_option_outputs["single_option_decision_path_selections"],
        single_option_representative_paths=single_option_outputs["single_option_representative_paths"],
        single_option_path_outcomes=single_option_outputs["single_option_path_outcomes"],
        single_option_required_path_to_beat_stock_1_5x=single_option_outputs["single_option_required_path_to_beat_stock_1_5x"],
        single_option_required_path_to_beat_stock_2_0x=single_option_outputs["single_option_required_path_to_beat_stock_2_0x"],
        single_option_closest_representative_path_to_edge=single_option_outputs["single_option_closest_representative_path_to_edge"],
        single_option_edge_gap_by_path_family=single_option_outputs["single_option_edge_gap_by_path_family"],
        single_option_path_family_counts=single_option_outputs["single_option_path_family_counts"],
        single_option_timing_sensitivity=single_option_outputs["single_option_timing_sensitivity"],
        single_option_iv_sensitivity=single_option_outputs["single_option_iv_sensitivity"],
        single_option_entry_sensitivity=single_option_outputs["single_option_entry_sensitivity"],
        single_option_summary_bullets=single_option_outputs["single_option_summary_bullets"],
        single_option_decision_markdown=single_option_outputs["single_option_decision_markdown"],
        path_view_tables=path_view_tables,
        required_vs_assumed_path_summary=simulation_outputs["required_vs_assumed_path_summary"],
        strategy_selector_context=strategy_selector_context,
        calibration_context=calibration,
        best_candidate_cards=best_cards,
        strategy_selector_best_cards=strategy_selector_cards,
        summary_markdown=summary_markdown,
        report_metadata=metadata,
    )


def build_contract_selection_analysis(*args, **kwargs) -> ContractSelectionComputation:
    """Build one canonical contract-selection analysis result."""

    return _build_contract_selection_core(*args, **kwargs)
