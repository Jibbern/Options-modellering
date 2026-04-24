"""Canonical analysis bundle writing and publish helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pandas as pd

from .contract_selection import ContractSelectionComputation
from .models import AnalysisBundle, StrategyAnalysisComputation
from .ranking import coverage_payload, unique_warnings
from .replay import HistoricalReplayComputation, collect_local_replay_history
from .scenario import ScenarioDashboardComputation
from ..persistence import write_dataframe_csv, write_json
from ..plots import (
    plot_action_board_overview,
    plot_assumed_path_value_progression,
    plot_buy_watch_avoid_matrix,
    plot_candidate_robustness_vs_upside,
    plot_compare_vs_stock_path,
    plot_compare_vs_stock_over_path,
    plot_conviction_vs_robustness,
    plot_estimated_value_vs_stock,
    plot_family_ranking_overview,
    plot_heatmap,
    plot_highlights_overview,
    plot_iv_path_trace,
    plot_iv_sensitivity,
    plot_iv_robustness_scorecard,
    plot_long_call_delta_over_path_best_of,
    plot_long_call_delta_over_path_expiry_view,
    plot_long_call_delta_over_path_strike_view,
    plot_long_call_iv_expanded_delta,
    plot_long_call_iv_expanded_value,
    plot_long_call_iv_path_delta,
    plot_long_call_iv_path_value,
    plot_long_call_value_over_path_best_of,
    plot_long_call_value_over_path_expiry_view,
    plot_long_call_value_over_path_strike_view,
    plot_path_long_call_compare_vs_stock,
    plot_multi_strategy_estimated_value,
    plot_multi_strategy_payoff,
    plot_path_survival_scorecard,
    plot_iv_path_gallery,
    plot_option_value_over_path,
    plot_path_comparison,
    plot_payoff_at_expiry,
    plot_replay_compare_vs_stock,
    plot_replay_driver_decomposition,
    plot_replay_stock_path,
    plot_replay_strategy_value_path,
    plot_representative_iv_paths,
    plot_representative_stock_paths,
    plot_required_path_strategy_compare,
    plot_required_move_speed_vs_magnitude,
    plot_required_move_vs_stock_chart,
    plot_required_stock_path_to_buy,
    plot_single_option_decision_view,
    plot_stock_path_gallery,
    plot_stock_vs_option_decision_chart,
    plot_stock_vs_option_preference_chart,
    plot_stress_test_overview,
    plot_strike_expiry_entry_barrier_map,
    plot_stock_vs_option_comparison,
    plot_strike_expiry_tradeoff_overview,
    plot_strategy_time_progression,
    plot_premium_sensitivity_chart,
    plot_chain_overview,
    plot_top_candidate_cards,
    plot_top_candidate_stress_cards,
    plot_time_decay,
    plot_timing_slip_chart,
    plot_trigger_map,
    plot_target_stress_chart,
    plot_thesis_candidate_overview,
    plot_thesis_iv_gallery,
    plot_thesis_iv_vs_value,
    plot_thesis_path_gallery,
    plot_thesis_path_vs_value,
    plot_thesis_stock_vs_option,
    plot_current_vs_justified_premium,
    plot_iv_support_requirement_chart,
)
from ..publish.dashboard import generate_dashboard
from ..publish.library import DEFAULT_DASHBOARDS_ROOT, mirror_published_bundle
from ..scenarios import build_case_template, compare_positions, scenario_table
from ..strategies import StrategyPosition
from ..utils import build_stock_grid, clean_string, ensure_directory, finite_or_none, slugify


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_OUTPUT_ROOT = PROJECT_ROOT / "analysis_outputs"
BUNDLE_VERSION = 2
TABLE_COLUMN_ORDERS: dict[str, list[str]] = {
    "summary.csv": [
        "ticker",
        "snapshot_date",
        "target_date",
        "target_price",
        "thesis_target_date",
        "thesis_target_price",
        "goal",
        "comparison_capital",
        "stock_path_name",
        "iv_path_name",
        "spot_price_source",
        "spot_field_used",
        "spot_used_prior_date",
        "spot_quality_note",
        "spot_price_matched_date",
        "risk_free_rate_source",
        "risk_free_rate_series",
        "risk_free_rate_matched_date",
        "risk_free_rate_note",
        "analysis_trust_level",
        "analysis_trust_note",
        "trusted_expiry_count",
        "fallback_only_expiry_count",
        "ibkr_same_day_spot_rejected_reason",
        "source_snapshot_storage_locations",
        "best_family",
        "best_family_candidate",
        "best_candidate",
        "best_candidate_family",
        "best_expiry",
        "best_strike",
        "best_strike_source_trust_label",
        "best_expiry_source_trust_label",
        "best_strike_representative_bucket",
        "best_expiry_representative_bucket",
        "family_edge_status",
        "stock_benchmark_decision",
        "stock_benchmark_note",
        "benchmark_edge",
        "benchmark_return_edge",
        "top_path_risk",
        "timing_risk",
        "iv_risk",
        "required_path_difficulty",
        "required_path_gap_at_target",
        "first_cleared_horizon",
        "iv_sensitivity_note",
        "default_case_label",
        "primary_warning",
    ],
    "decision_highlights.csv": [
        "display_order",
        "highlight_category",
        "highlight_label",
        "selected_candidate_label",
        "selected_family",
        "decision_status",
        "score",
        "source_trust_label",
        "trust_caution",
        "primary_reason",
        "main_warning",
        "decision_tags",
        "difference_vs_stock",
        "return_on_comparison_capital",
        "robustness_score",
        "aggressive_upside_score",
        "balanced_score",
        "time_resilience_score",
        "lower_iv_resilience_score",
        "iv_upside_score",
        "fragility_score",
    ],
    "decision_highlights_explanations.csv": [
        "highlight_category",
        "selected_candidate_label",
        "decision_status",
        "primary_reason",
        "main_warning",
        "score_column",
        "score",
        "robustness_score",
        "aggressive_upside_score",
        "balanced_score",
        "time_resilience_score",
        "lower_iv_resilience_score",
        "iv_upside_score",
        "trust_score",
        "penalty_score",
        "decision_tags",
    ],
    "candidate_robustness_summary.csv": [
        "candidate_slug",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "path_count_tested",
        "iv_view_count",
        "iv_opportunity_count",
        "profitable_iv_path_count",
        "beat_stock_iv_path_count",
        "profitable_iv_path_rate",
        "beat_stock_iv_path_rate",
        "lower_iv_survival_rate",
        "lower_iv_beat_stock_rate",
        "high_iv_dependency_rate",
        "terminal_value_min",
        "terminal_value_max",
        "terminal_value_range",
        "terminal_delta_vs_stock_min",
        "terminal_delta_vs_stock_max",
        "terminal_delta_vs_stock_range",
        "iv_robustness_labels",
        "iv_robustness_notes",
    ],
    "candidate_tradeoff_matrix.csv": [
        "candidate_slug",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "objective_score",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "robustness_score",
        "aggressive_upside_score",
        "balanced_score",
        "time_resilience_score",
        "lower_iv_resilience_score",
        "iv_upside_score",
        "fragility_score",
        "trust_score",
        "penalty_score",
        "profitable_iv_path_rate",
        "beat_stock_iv_path_rate",
        "lower_iv_survival_rate",
        "lower_iv_beat_stock_rate",
        "high_iv_dependency_rate",
        "decision_tags",
        "stock_dominance_note",
    ],
    "stock_vs_option_takeaways.csv": [
        "takeaway_type",
        "status",
        "candidate_label",
        "evidence",
        "difference_vs_stock",
        "note",
    ],
    "highlights_score_breakdown.csv": [
        "candidate_slug",
        "candidate_label",
        "strategy_family",
        "component",
        "component_score",
        "component_note",
    ],
    "action_board_candidates.csv": [
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "source_trust_label",
        "candidate_conviction_score",
        "action_score",
        "robustness_score",
        "upside_score",
        "stock_relative_score",
        "time_decay_risk",
        "iv_dependence_risk",
        "trust_penalty",
        "affordability_status",
        "headline_reason",
        "why_this_is_interesting_now",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "what_has_to_happen",
        "what_would_invalidate",
        "main_warning",
        "difference_vs_stock",
        "return_on_comparison_capital",
        "key_trigger_label",
        "key_trigger_type",
        "key_trigger_value",
        "key_trigger_deadline",
        "decision_tags",
    ],
    "buy_now_candidates.csv": [
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "candidate_conviction_score",
        "action_score",
        "robustness_score",
        "stock_relative_score",
        "headline_reason",
        "why_buy_now",
        "what_would_invalidate",
        "source_trust_label",
    ],
    "watchlist_candidates.csv": [
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "candidate_conviction_score",
        "action_score",
        "robustness_score",
        "stock_relative_score",
        "why_watch_not_buy",
        "why_this_is_interesting_now",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "what_has_to_happen",
        "what_would_invalidate",
        "key_trigger_label",
        "key_trigger_type",
        "key_trigger_value",
        "key_trigger_deadline",
        "source_trust_label",
    ],
    "avoid_for_now_candidates.csv": [
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "action_score",
        "time_decay_risk",
        "iv_dependence_risk",
        "trust_penalty",
        "why_avoid",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "what_would_invalidate",
        "source_trust_label",
    ],
    "prefer_stock_instead.csv": [
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "strategy_family",
        "headline_reason",
        "why_stock_may_be_better",
        "action_score",
        "robustness_score",
        "stock_relative_score",
    ],
    "decision_triggers.csv": [
        "action_bucket",
        "candidate_label",
        "strategy_family",
        "trigger_type_label",
        "key_trigger_type",
        "key_trigger_value",
        "key_trigger_deadline",
        "trigger_direction",
        "urgency",
        "what_has_to_happen",
        "upgrade_rule",
        "what_would_invalidate",
        "invalidate_rule",
        "main_warning",
        "source_trust_label",
        "action_confidence",
    ],
    "action_board_score_breakdown.csv": [
        "action_bucket",
        "candidate_label",
        "strategy_family",
        "component",
        "component_score",
        "component_note",
    ],
    "action_board_explanations.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_label",
        "strategy_family",
        "action_confidence",
        "headline_reason",
        "why_buy_now",
        "why_watch_not_buy",
        "why_avoid",
        "why_stock_may_be_better",
        "why_this_is_interesting_now",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "what_has_to_happen",
        "what_would_invalidate",
        "main_warning",
        "decision_tags",
    ],
    "bullish_long_call_action_board.csv": [
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "headline_reason",
        "why_this_is_interesting_now",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "main_warning",
        "candidate_conviction_score",
        "action_score",
        "robustness_score",
        "upside_score",
        "stock_relative_score",
        "time_decay_risk",
        "iv_dependence_risk",
        "difference_vs_stock",
        "return_on_comparison_capital",
        "key_trigger_label",
        "key_trigger_value",
        "key_trigger_deadline",
    ],
    "bullish_long_call_watchlist.csv": [
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "why_this_is_interesting_now",
        "why_watch_not_buy",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "what_has_to_happen",
        "what_would_invalidate",
        "main_warning",
        "action_score",
        "robustness_score",
        "stock_relative_score",
    ],
    "bullish_long_call_avoid.csv": [
        "action_priority_rank",
        "action_confidence",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "why_avoid",
        "what_is_hurting_this_candidate",
        "main_warning",
        "upgrade_rule",
        "invalidate_rule",
        "what_would_invalidate",
        "action_score",
        "time_decay_risk",
        "iv_dependence_risk",
        "trust_penalty",
    ],
    "bullish_long_call_triggers.csv": [
        "action_bucket",
        "candidate_label",
        "trigger_type_label",
        "key_trigger_type",
        "upgrade_rule",
        "what_has_to_happen",
        "key_trigger_deadline",
        "invalidate_rule",
        "what_would_invalidate",
        "main_warning",
        "source_trust_label",
        "action_confidence",
    ],
    "top_candidate_cards.csv": [
        "card_rank",
        "contract_label",
        "candidate_label",
        "bucket",
        "confidence",
        "why_this_is_interesting",
        "what_hurts_it",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "trust",
        "compare_vs_stock_note",
        "action_score",
    ],
    "chain_overview_summary.csv": [
        "card_key",
        "card_label",
        "contract_label",
        "verdict_badge",
        "headline_metric",
        "headline_note",
        "explanation_short",
        "candidate_slug",
    ],
    "chain_overview_candidates.csv": [
        "contract",
        "premium",
        "iv",
        "dte",
        "beats_stock_label",
        "strong_wins",
        "robustness",
        "iv_sensitivity",
        "entry_sensitivity",
        "best_fit_path_type",
        "final_verdict",
        "why_short",
        "why_detail",
        "source_trust_label",
        "expiry_sensitivity_summary",
        "beats_stock_count",
        "qualifying_path_family_count",
        "total_path_family_count",
        "strong_outperformance_count",
        "robustness_score",
        "asymmetry_score",
        "timing_sensitivity_score",
        "iv_sensitivity_score",
        "entry_premium_sensitivity_score",
        "difference_vs_stock",
        "return_on_comparison_capital",
        "worth_buying",
        "worth_buying_status",
        "best_path_label",
        "best_path_difference_vs_stock",
        "worst_path_label",
        "worst_path_difference_vs_stock",
        "minimum_outperformance_multiple",
        "strong_outperformance_multiple",
        "required_winning_path_families",
        "shared_path_family_count",
        "shared_path_anchor_candidate",
        "candidate_slug",
        "candidate_label",
        "premium_source",
    ],
    "bullish_long_call_score_breakdown.csv": [
        "action_bucket",
        "candidate_label",
        "component",
        "component_score",
        "component_note",
    ],
    "other_structures_summary.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_label",
        "strategy_family",
        "action_confidence",
        "headline_reason",
        "why_this_is_interesting_now",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "main_warning",
        "source_trust_label",
    ],
    "stock_preference_summary.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_label",
        "strategy_family",
        "action_confidence",
        "headline_reason",
        "why_stock_may_be_better",
        "stock_relative_score",
        "difference_vs_stock",
        "robustness_score",
        "main_warning",
        "source_trust_label",
    ],
    "entry_justification_candidates.csv": [
        "entry_display_rank",
        "action_bucket",
        "action_priority_rank",
        "candidate_short_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "required_price_1m",
        "required_price_target",
        "required_move_pct_target",
        "timing_window_days",
        "move_pace_pct_per_month",
        "required_path_difficulty",
        "first_cleared_horizon",
        "requires_fast_move",
        "needs_iv_support",
        "stock_still_better_even_if_path_hits",
        "iv_requirement_label",
        "entry_barrier_score",
        "entry_barrier_label",
        "what_has_to_happen",
        "entry_warning",
        "stock_vs_option_read",
    ],
    "required_stock_path_to_buy.csv": [
        "candidate_slug",
        "candidate_short_label",
        "action_bucket",
        "entry_display_rank",
        "iv_path_label",
        "series_kind",
        "date",
        "requested_days",
        "stock_price",
        "entry_barrier_label",
        "stock_vs_option_read",
    ],
    "required_move_summary.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_short_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "required_path_difficulty",
        "first_cleared_horizon",
        "required_move_pct_1m",
        "required_move_pct_3m",
        "required_move_pct_target",
        "timing_window_days",
        "move_pace_pct_per_month",
        "requires_fast_move",
        "stock_still_better_even_if_path_hits",
        "entry_barrier_score",
        "what_has_to_happen",
        "entry_warning",
        "source_trust_label",
    ],
    "required_move_vs_stock.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_short_label",
        "candidate_label",
        "required_move_pct_target",
        "timing_window_days",
        "assumed_clears_required_at_target",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "stock_relative_score",
        "stock_still_better_even_if_path_hits",
        "stock_vs_option_read",
    ],
    "required_iv_support_summary.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_short_label",
        "candidate_label",
        "required_move_pct_flat_iv",
        "lower_iv_required_move_pct",
        "higher_iv_required_move_pct",
        "lower_iv_move_penalty_pct",
        "higher_iv_move_relief_pct",
        "lower_iv_resilience_score",
        "iv_dependence_risk",
        "iv_requirement_label",
        "iv_requirement_note",
    ],
    "entry_barrier_summary.csv": [
        "action_bucket",
        "action_priority_rank",
        "candidate_short_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "entry_barrier_score",
        "entry_barrier_label",
        "required_move_pct_target",
        "timing_window_days",
        "requires_fast_move",
        "iv_requirement_label",
        "stock_vs_option_read",
        "source_trust_label",
    ],
    "thesis_mode_candidates.csv": [
        "thesis_target_price",
        "thesis_target_date",
        "path_family",
        "path_label",
        "iv_path_name",
        "iv_path_label",
        "candidate_short_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "current_premium",
        "thesis_terminal_value",
        "profit_loss",
        "difference_vs_stock",
        "stock_still_better",
        "stock_relative_justified_premium",
        "thesis_target_beyond_expiry",
    ],
    "thesis_candidate_ranking.csv": [
        "thesis_candidate_rank",
        "candidate_short_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "current_premium",
        "max_justified_premium",
        "premium_gap",
        "premium_gap_pct",
        "entry_attractiveness_status",
        "profitable_scenario_rate",
        "beats_stock_scenario_rate",
        "difference_vs_stock_median",
        "path_sensitivity_label",
        "iv_sensitivity_label",
        "stock_still_better_under_thesis",
        "thesis_target_beyond_expiry",
        "main_reason",
    ],
    "max_justified_premium.csv": [
        "thesis_candidate_rank",
        "candidate_short_label",
        "current_premium",
        "max_justified_premium",
        "premium_gap",
        "premium_gap_pct",
        "entry_attractiveness_status",
        "main_reason",
    ],
    "current_vs_justified_premium.csv": [
        "thesis_candidate_rank",
        "candidate_short_label",
        "current_premium",
        "max_justified_premium",
        "premium_gap",
        "premium_gap_pct",
        "entry_attractiveness_status",
        "main_reason",
    ],
    "thesis_required_move_summary.csv": [
        "candidate_short_label",
        "thesis_target_price",
        "thesis_target_date",
        "days_to_target",
        "required_total_upside_pct",
        "required_monthly_pace_pct",
        "required_timing_window",
        "entry_attractiveness_status",
        "timing_note",
    ],
    "thesis_stock_vs_option_summary.csv": [
        "thesis_candidate_rank",
        "candidate_short_label",
        "entry_attractiveness_status",
        "beats_stock_scenario_rate",
        "difference_vs_stock_median",
        "stock_still_better_under_thesis",
        "main_reason",
    ],
    "candidate_stress_grid.csv": [
        "candidate_rank",
        "candidate_short_label",
        "candidate_slug",
        "metric",
        "Base",
        "Premium -10%",
        "Premium -20%",
        "Premium +10%",
        "Move delayed 2w",
        "Move delayed 1m",
        "Move delayed 2m",
    ],
    "premium_sensitivity_summary.csv": [
        "candidate_short_label",
        "candidate_label",
        "scenario_label",
        "action_bucket",
        "bucket_transition",
        "option_vs_stock_edge_pct",
        "max_justified_premium_gap",
        "scenario_premium",
        "premium_multiplier",
        "main_note",
        "upgrade_rule",
        "main_warning",
        "source_trust_label",
    ],
    "timing_slip_summary.csv": [
        "candidate_short_label",
        "candidate_label",
        "scenario_label",
        "action_bucket",
        "bucket_transition",
        "option_vs_stock_edge_pct",
        "max_justified_premium_gap",
        "delay_days",
        "delayed_target_date",
        "target_beyond_expiry_under_delay",
        "main_note",
        "upgrade_rule",
        "main_warning",
        "source_trust_label",
    ],
    "target_stress_summary.csv": [
        "candidate_short_label",
        "candidate_label",
        "scenario_label",
        "action_bucket",
        "bucket_transition",
        "option_vs_stock_edge_pct",
        "max_justified_premium_gap",
        "target_price",
        "intrinsic_value_at_target",
        "main_note",
        "upgrade_rule",
        "main_warning",
        "source_trust_label",
    ],
    "stress_transition_summary.csv": [
        "stress_rank",
        "candidate_short_label",
        "candidate_label",
        "base_action_bucket",
        "base_option_vs_stock_edge_pct",
        "base_max_justified_premium_gap",
        "best_improving_stress",
        "best_improving_bucket",
        "best_improving_edge_pct",
        "worst_breaking_stress",
        "worst_breaking_bucket",
        "worst_breaking_edge_pct",
        "stress_resilience_score",
        "stress_buy_count",
        "premium_sensitivity_read",
        "timing_sensitivity_read",
        "target_dependency_read",
        "stress_card_note",
        "main_warning",
        "upgrade_rule",
        "source_trust_label",
    ],
    "candidate_summary.csv": [
        "active_candidate_rank",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "target_price",
        "target_date",
        "objective_score",
        "estimated_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "stock_benchmark_decision",
        "benchmark_note",
        "required_path_difficulty",
        "path_gap_at_target",
        "first_cleared_horizon",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "relevance_under_thesis",
        "why_this_candidate_wins",
        "why_this_candidate_loses",
        "break_even",
        "max_loss",
        "max_gain",
        "premium_or_entry_cost",
        "capital_required",
        "affordability_label",
        "budget_note",
        "timing_gap_days",
        "horizon_fit_label",
        "weak_horizon_fit",
        "time_sensitivity_summary",
        "iv_sensitivity_summary",
        "confidence_label",
        "coverage_flags",
        "source_quality",
        "source_trust_label",
        "source_quality_note",
        "warning_or_note",
    ],
    "family_comparison.csv": [
        "strategy_label",
        "strategy_family",
        "winning_candidate_label",
        "objective_rank",
        "objective_score",
        "current_objective_card_status",
        "best_under_current_objective",
        "best_if_move_is_slower",
        "best_if_iv_falls",
        "best_for_capital_efficiency",
        "best_for_capped_downside",
        "best_for_convexity",
        "best_for_simple_exposure",
        "target_pnl",
        "target_return_pct",
        "difference_vs_stock",
        "benchmark_edge",
        "benchmark_return_edge",
        "benchmark_note",
        "required_path_difficulty",
        "path_gap_at_target",
        "first_cleared_horizon",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "affordability_label",
        "break_even",
        "max_loss",
        "current_objective_reason",
        "current_objective_warning",
        "why_this_wins",
        "why_this_loses",
        "horizon_fit_label",
        "target_beyond_expiry",
        "confidence_label",
        "coverage_flags",
        "selection_scope_label",
    ],
    "candidate_comparison.csv": [
        "active_candidate_rank",
        "candidate_label",
        "strategy_family",
        "expiry_date",
        "strike_label",
        "objective_score",
        "estimated_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "stock_benchmark_decision",
        "benchmark_note",
        "required_path_difficulty",
        "path_gap_at_target",
        "first_cleared_horizon",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "relevance_under_thesis",
        "premium_or_entry_cost",
        "capital_required",
        "affordability_label",
        "break_even",
        "max_loss",
        "max_gain",
        "horizon_fit_label",
        "weak_horizon_fit",
        "target_beyond_expiry",
        "confidence_label",
        "coverage_flags",
        "source_quality",
        "source_trust_label",
        "source_quality_note",
        "why_this_candidate_wins",
        "why_this_candidate_loses",
        "warning_or_note",
    ],
    "strike_comparison.csv": [
        "strategy_family",
        "strike_label",
        "best_expiry_date",
        "best_candidate_label",
        "active_candidate_rank",
        "objective_score",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "required_path_difficulty",
        "path_gap_at_target",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "benchmark_note",
        "best_source_quality",
        "best_source_trust_label",
        "horizon_fit_label",
        "weak_horizon_fit",
        "target_beyond_expiry",
        "expiry_count_for_strike",
        "candidate_count_for_strike",
    ],
    "expiry_comparison.csv": [
        "strategy_family",
        "expiry_date",
        "best_strike_label",
        "best_candidate_label",
        "active_candidate_rank",
        "objective_score",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "required_path_difficulty",
        "path_gap_at_target",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "benchmark_note",
        "best_source_quality",
        "best_source_trust_label",
        "horizon_fit_label",
        "weak_horizon_fit",
        "target_beyond_expiry",
        "expiry_fit_note",
        "strike_count_for_expiry",
        "candidate_count_for_expiry",
    ],
    "required_path_summary.csv": [
        "summary_scope",
        "summary_label",
        "strategy_family",
        "goal",
        "iv_variant",
        "first_cleared_horizon",
        "required_stock_price_at_target",
        "assumed_stock_price_at_target",
        "path_gap_at_target",
        "required_path_difficulty",
        "unreached",
        "clamped_to_expiry",
        "target_beyond_expiry",
    ],
    "assumed_path_trace_rows.csv": [
        "trace_scope",
        "series_label",
        "strategy_family",
        "horizon",
        "requested_days",
        "spot_price",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "stock_profit_loss",
        "stock_return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "worst_interim_profit_loss_to_date",
        "drawdown_from_peak_to_date",
    ],
    "iv_path_trace_rows.csv": [
        "trace_scope",
        "iv_path_name",
        "variant_kind",
        "horizon",
        "requested_days",
        "iv_shift_points",
        "delta_from_entry_iv_shift",
    ],
    "compare_vs_stock_path_rows.csv": [
        "trace_scope",
        "series_label",
        "strategy_family",
        "horizon",
        "requested_days",
        "profit_loss",
        "stock_profit_loss",
        "delta_profit_loss_vs_stock",
        "return_on_comparison_capital",
        "stock_return_on_comparison_capital",
        "delta_return_pct_vs_stock",
        "benchmark_note",
    ],
    "iv_path_sensitivity_summary.csv": [
        "summary_scope",
        "summary_label",
        "strategy_family",
        "goal",
        "stock_path_name",
        "iv_path_name",
        "iv_risk",
        "sensitivity_note",
        "pnl_sensitivity_range",
        "benchmark_edge",
        "benchmark_return_edge",
        "benchmark_note",
        "confidence_label",
        "coverage_flags",
    ],
    "path_risk_summary.csv": [
        "summary_scope",
        "summary_label",
        "strategy_family",
        "goal",
        "stock_path_name",
        "iv_path_name",
        "required_path_difficulty",
        "first_cleared_horizon",
        "path_gap_at_target",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "max_downside",
        "worst_interim_profit_loss",
        "worst_drawdown_from_peak",
        "benchmark_edge",
        "benchmark_return_edge",
        "benchmark_note",
        "confidence_label",
        "coverage_flags",
        "target_beyond_expiry",
    ],
    "stock_path_library.csv": [
        "path_name",
        "path_label",
        "path_family",
        "path_family_label",
        "timing_shape",
        "outcome_bias",
        "library_role",
        "display_order",
        "is_active_assumed",
        "path_description",
    ],
    "stock_path_gallery.csv": [
        "path_name",
        "path_label",
        "path_family",
        "path_family_label",
        "timing_shape",
        "outcome_bias",
        "path_description",
        "path_role",
        "display_order",
        "date",
        "requested_days",
        "spot_price",
        "return_pct",
        "is_active_assumed",
    ],
    "single_option_decision_path_selections.csv": [
        "candidate_slug",
        "candidate_short_label",
        "decision_path_id",
        "path_role",
        "path_name",
        "path_label",
        "path_family",
        "path_family_label",
        "timing_shape",
        "outcome_bias",
        "outcome_label",
        "display_order",
        "selection_score",
        "selection_reason",
        "exit_stock_price",
        "difference_vs_stock",
        "outperformance_multiple",
    ],
    "single_option_representative_paths.csv": [
        "candidate_slug",
        "candidate_short_label",
        "decision_path_id",
        "path_role",
        "path_name",
        "path_label",
        "path_family",
        "path_family_label",
        "timing_shape",
        "outcome_bias",
        "display_order",
        "step_index",
        "date",
        "requested_days",
        "spot_price",
        "return_pct",
        "iv_shift_points",
        "selection_score",
        "selection_reason",
        "is_curated_decision_path",
    ],
    "single_option_path_outcomes.csv": [
        "candidate_slug",
        "candidate_label",
        "candidate_short_label",
        "decision_path_id",
        "path_role",
        "path_name",
        "path_label",
        "path_family",
        "path_family_label",
        "timing_shape",
        "outcome_bias",
        "outcome_label",
        "display_order",
        "exit_rule",
        "exit_date",
        "exit_stock_price",
        "profit_loss",
        "stock_profit_loss",
        "difference_vs_stock",
        "outperformance_multiple",
        "outcome_note",
        "selection_score",
        "selection_reason",
        "is_curated_decision_path",
    ],
    "iv_path_gallery.csv": [
        "iv_path_name",
        "iv_path_label",
        "path_role",
        "display_order",
        "date",
        "requested_days",
        "iv_shift_points",
        "is_active_assumed",
    ],
    "stock_path_examples.csv": [
        "path_id",
        "path_kind",
        "path_name",
        "representative_bucket",
        "is_representative",
        "date",
        "requested_days",
        "spot_price",
        "return_pct",
        "selection_reason",
    ],
    "iv_path_examples.csv": [
        "iv_path_id",
        "iv_path_name",
        "representative_bucket",
        "is_representative",
        "date",
        "requested_days",
        "iv_shift_points",
        "selection_reason",
    ],
    "path_pair_summary.csv": [
        "path_pair_id",
        "stock_path_name",
        "iv_path_name",
        "representative_bucket",
        "is_representative",
        "terminal_stock_price",
        "terminal_iv_shift_points",
        "final_profit_loss",
        "final_difference_vs_stock",
        "goal_reached",
        "outperformed_stock",
        "goal_success_rate",
        "selection_reason",
    ],
    "option_value_over_path.csv": [
        "path_pair_id",
        "representative_bucket",
        "candidate_label",
        "strategy_family",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "worst_interim_profit_loss_to_date",
        "drawdown_from_peak_to_date",
        "max_favorable_profit_to_date",
        "success_status",
    ],
    "compare_vs_stock_over_path.csv": [
        "path_pair_id",
        "representative_bucket",
        "candidate_label",
        "strategy_family",
        "date",
        "requested_days",
        "strategy_profit_loss",
        "stock_profit_loss",
        "delta_profit_loss_vs_stock",
        "strategy_return_on_comparison_capital",
        "stock_return_on_comparison_capital",
        "delta_return_pct_vs_stock",
        "benchmark_note",
    ],
    "representative_paths_summary.csv": [
        "path_pair_id",
        "representative_bucket",
        "stock_path_name",
        "iv_path_name",
        "top_candidate_success_status",
        "stock_benchmark_status",
        "terminal_stock_price",
        "terminal_iv_shift_points",
        "final_profit_loss",
        "final_difference_vs_stock",
        "selection_reason",
    ],
    "strike_comparison_under_path.csv": [
        "path_pair_id",
        "representative_bucket",
        "strategy_family",
        "strike_label",
        "best_candidate_label",
        "objective_score",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "required_path_difficulty",
        "timing_risk",
        "iv_risk",
        "success_dependency",
    ],
    "expiry_comparison_under_path.csv": [
        "path_pair_id",
        "representative_bucket",
        "strategy_family",
        "expiry_date",
        "best_candidate_label",
        "objective_score",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "required_path_difficulty",
        "timing_risk",
        "iv_risk",
        "success_dependency",
    ],
    "long_call_value_over_path_strike_view.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "anchor_expiry_date",
        "selection_rank",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "date",
        "requested_days",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "success_status",
        "objective_score",
        "selection_reason",
    ],
    "long_call_value_over_path_expiry_view.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "anchor_strike_label",
        "selection_rank",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "used_strike_fallback",
        "strike_match_mode",
        "fallback_strike_distance",
        "source_trust_label",
        "date",
        "requested_days",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "success_status",
        "objective_score",
        "selection_reason",
    ],
    "long_call_value_over_path_best_of.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "selection_rank",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "used_trust_fallback",
        "source_trust_label",
        "date",
        "requested_days",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "success_status",
        "objective_score",
        "selection_reason",
    ],
    "path_checkpoints.csv": [
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "checkpoint_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "selection_rank",
        "candidate_label",
        "series_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "success_status",
    ],
    "iv_path_value.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "iv_path_label",
        "anchor_contract_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "source_trust_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "iv_effect_note",
        "success_status",
    ],
    "iv_path_delta.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "iv_path_label",
        "anchor_contract_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "source_trust_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "delta_profit_loss_vs_stock",
        "delta_return_pct_vs_stock",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "iv_effect_note",
        "success_status",
    ],
    "iv_checkpoints.csv": [
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "iv_path_label",
        "checkpoint_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "anchor_contract_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "source_trust_label",
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "success_status",
        "iv_effect_note",
    ],
    "iv_expanded_value.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_expanded_family",
        "contract_rank",
        "contract_label",
        "iv_path_name",
        "iv_path_label",
        "chart_include",
        "iv_chart_scope",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "iv_effect_note",
        "success_status",
    ],
    "iv_expanded_delta.csv": [
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_expanded_family",
        "contract_rank",
        "contract_label",
        "iv_path_name",
        "iv_path_label",
        "chart_include",
        "iv_chart_scope",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "delta_profit_loss_vs_stock",
        "delta_return_pct_vs_stock",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "iv_effect_note",
        "iv_robustness_note",
        "success_status",
    ],
    "iv_expanded_checkpoints.csv": [
        "path_scope",
        "stock_path_name",
        "iv_expanded_family",
        "contract_rank",
        "contract_label",
        "iv_path_name",
        "iv_path_label",
        "checkpoint_label",
        "date",
        "requested_days",
        "spot_price",
        "iv_shift_points",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "success_status",
        "iv_effect_note",
    ],
    "iv_robustness_summary.csv": [
        "stock_path_name",
        "iv_expanded_family",
        "contract_rank",
        "contract_label",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "iv_path_count",
        "profitable_iv_path_count",
        "beat_stock_iv_path_count",
        "lower_iv_profitable",
        "lower_iv_beats_stock",
        "high_iv_dependency",
        "terminal_value_min",
        "terminal_value_max",
        "terminal_value_range",
        "terminal_delta_vs_stock_min",
        "terminal_delta_vs_stock_max",
        "terminal_delta_vs_stock_range",
        "best_iv_path",
        "worst_iv_path",
        "iv_robustness_label",
        "iv_robustness_note",
    ],
    "required_vs_assumed_path_summary.csv": [
        "comparison_scope",
        "candidate_slug",
        "candidate_label",
        "strategy_family",
        "goal",
        "assumed_path_name",
        "representative_path_pair_id",
        "representative_bucket",
        "first_cleared_horizon",
        "required_path_difficulty",
        "assumed_path_gap_at_target",
        "representative_path_gap_at_target",
        "representative_terminal_stock_price",
        "representative_goal_reached",
    ],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_text(value: date | str) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _project_relative_path(value: Any) -> str | None:
    text = clean_string(value)
    if not text:
        return None
    path = Path(text)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except (ValueError, OSError, RuntimeError):
        return str(path).replace("\\", "/")


def _analysis_identity(result: Any, analysis_kind: str) -> tuple[str, str, str]:
    if analysis_kind == "strategy":
        strategy = result.strategy if isinstance(result, StrategyAnalysisComputation) else result
        assert isinstance(strategy, StrategyPosition)
        expiry_text = strategy.expiry_date.isoformat() if strategy.expiry_date else "no-expiry"
        return (
            strategy.ticker.upper(),
            strategy.snapshot_date.isoformat(),
            slugify(f"{strategy.name}-{expiry_text}") or "strategy",
        )
    if analysis_kind == "contract_selection":
        assert isinstance(result, ContractSelectionComputation)
        return result.ticker.upper(), result.snapshot_date.isoformat(), clean_string(result.run_slug)
    if analysis_kind == "scenario":
        assert isinstance(result, ScenarioDashboardComputation)
        return result.ticker.upper(), result.snapshot_date.isoformat(), slugify(f"expiry-{result.expiry_date.isoformat()}")
    if analysis_kind == "replay":
        assert isinstance(result, HistoricalReplayComputation)
        return (
            result.ticker.upper(),
            result.snapshot_date.isoformat(),
            slugify(f"{result.strategy_name}-{result.expiry_date.isoformat()}"),
        )
    raise ValueError(f"Unsupported analysis bundle kind: {analysis_kind}")


def _bundle_dir(
    ticker: str,
    snapshot_date: str,
    analysis_kind: str,
    run_slug: str,
    *,
    output_root: str | Path,
) -> Path:
    return Path(output_root) / ticker / f"snapshot_{snapshot_date}" / analysis_kind / run_slug


def _section_path(bundle_dir: Path, section: str, filename: str) -> Path:
    path = bundle_dir / section / filename
    ensure_directory(path.parent)
    return path


def _record_path(file_map: dict[str, dict[str, str]], bundle_dir: Path, path: Path) -> None:
    relative = path.relative_to(bundle_dir).as_posix()
    section = relative.split("/", 1)[0]
    file_map.setdefault(section, {})[path.name] = relative


def _ordered_columns(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    columns = [column for column in preferred if column in frame.columns]
    remainder = [column for column in frame.columns if column not in columns]
    return frame.loc[:, columns + remainder]


def _precision_for_column(column: str) -> int | None:
    lowered = clean_string(column).lower()
    if lowered in {"requested_days", "effective_days", "affordable_units", "coverage_flag_count", "active_candidate_rank", "objective_rank"}:
        return None
    if any(token in lowered for token in ["profit", "loss", "value", "price", "capital", "cost", "premium", "downside", "drawdown", "benchmark_edge"]):
        return 2
    if any(token in lowered for token in ["pct", "ratio", "return", "iv_shift", "expected_move", "sensitivity_range", "delta_from_entry"]):
        return 4
    return 4


def _curate_frame_for_artifact(filename: str, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    curated = frame.copy()
    preferred = TABLE_COLUMN_ORDERS.get(filename)
    if preferred is None:
        lowered = clean_string(filename).lower()
        if lowered.endswith("__compare_vs_stock_path_rows.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("compare_vs_stock_path_rows.csv")
        elif lowered.endswith("__long_call_strike_view.csv") or lowered.endswith("__long_call_strike_value.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("long_call_value_over_path_strike_view.csv")
        elif lowered.endswith("__long_call_expiry_view.csv") or lowered.endswith("__long_call_expiry_value.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("long_call_value_over_path_expiry_view.csv")
        elif lowered.endswith("__long_call_best_of.csv") or lowered.endswith("__long_call_best_of_value.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("long_call_value_over_path_best_of.csv")
        elif lowered.endswith("__long_call_strike_delta.csv") or lowered.endswith("__long_call_expiry_delta.csv") or lowered.endswith("__long_call_best_of_delta.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("compare_vs_stock_path_rows.csv")
        elif lowered.endswith("__path_checkpoints.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("path_checkpoints.csv")
        elif lowered.endswith("__iv_path_value.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("iv_path_value.csv")
        elif lowered.endswith("__iv_path_delta.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("iv_path_delta.csv")
        elif lowered.endswith("__iv_checkpoints.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("iv_checkpoints.csv")
        elif (
            lowered.endswith("__long_call_strike_iv_value.csv")
            or lowered.endswith("__long_call_expiry_iv_value.csv")
            or lowered.endswith("__long_call_best_of_iv_value.csv")
        ):
            preferred = TABLE_COLUMN_ORDERS.get("iv_expanded_value.csv")
        elif (
            lowered.endswith("__long_call_strike_iv_delta.csv")
            or lowered.endswith("__long_call_expiry_iv_delta.csv")
            or lowered.endswith("__long_call_best_of_iv_delta.csv")
        ):
            preferred = TABLE_COLUMN_ORDERS.get("iv_expanded_delta.csv")
        elif (
            lowered.endswith("__long_call_strike_iv_checkpoints.csv")
            or lowered.endswith("__long_call_expiry_iv_checkpoints.csv")
            or lowered.endswith("__long_call_best_of_iv_checkpoints.csv")
        ):
            preferred = TABLE_COLUMN_ORDERS.get("iv_expanded_checkpoints.csv")
        elif lowered.endswith("__iv_robustness_summary.csv"):
            preferred = TABLE_COLUMN_ORDERS.get("iv_robustness_summary.csv")
    if preferred is not None:
        curated = _ordered_columns(curated, preferred)
    for column in curated.columns:
        if pd.api.types.is_bool_dtype(curated[column]):
            continue
        if not pd.api.types.is_numeric_dtype(curated[column]):
            continue
        precision = _precision_for_column(column)
        if precision is None:
            continue
        curated[column] = pd.to_numeric(curated[column], errors="coerce").round(precision)
    return curated


def _write_frame(bundle_dir: Path, file_map: dict[str, dict[str, str]], filename: str, frame: pd.DataFrame) -> Path:
    path = _section_path(bundle_dir, "tables", filename)
    write_dataframe_csv(_curate_frame_for_artifact(filename, frame), path, index=False)
    _record_path(file_map, bundle_dir, path)
    return path


def _write_markdown(bundle_dir: Path, file_map: dict[str, dict[str, str]], filename: str, text: str) -> Path:
    path = _section_path(bundle_dir, "summary", filename)
    path.write_text(text, encoding="utf-8")
    _record_path(file_map, bundle_dir, path)
    return path


def _write_metadata(bundle_dir: Path, file_map: dict[str, dict[str, str]], filename: str, payload: dict[str, Any]) -> Path:
    path = _section_path(bundle_dir, "metadata", filename)
    write_json(payload, path)
    _record_path(file_map, bundle_dir, path)
    return path


def _track_chart(bundle_dir: Path, file_map: dict[str, dict[str, str]], path: Path | None) -> Path | None:
    if path is None:
        return None
    _record_path(file_map, bundle_dir, path)
    return path


def _bundle_sources(report_metadata: dict[str, Any]) -> list[str]:
    metadata = report_metadata.get("metadata") or {}
    candidates = [
        metadata.get("source_snapshot_file"),
        metadata.get("source_snapshot_files"),
        report_metadata.get("source_snapshot_file"),
    ]
    flattened: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, list):
            flattened.extend(item for item in candidate if clean_string(item))
        elif clean_string(candidate):
            flattened.append(str(candidate))
    normalized: list[str] = []
    for item in flattened:
        relative = _project_relative_path(item)
        if relative and relative not in normalized:
            normalized.append(relative)
    return normalized


def _bundle_assumptions(report_metadata: dict[str, Any], analysis_kind: str) -> dict[str, Any]:
    metadata = report_metadata.get("metadata") or {}
    if analysis_kind == "contract_selection":
        keys = [
            "goal",
            "objective_mode",
            "comparison_capital",
            "target_price",
            "target_date",
            "stock_path_name",
            "stock_path_points",
            "iv_path_name",
            "iv_path_points",
        ]
        return {key: metadata.get(key) for key in keys if key in metadata}
    if analysis_kind == "scenario":
        keys = ["comparison_capital", "premium_mode", "featured_focus_strategy"]
        return {key: metadata.get(key) for key in keys if key in metadata}
    if analysis_kind == "replay":
        keys = ["strategy_name", "comparison_capital", "premium_mode"]
        return {key: metadata.get(key) for key in keys if key in metadata}
    if analysis_kind == "strategy":
        strategy_report = report_metadata.get("strategy_report") or {}
        keys = ["strategy", "ticker", "snapshot_date", "expiry_date", "premium_mode"]
        return {key: strategy_report.get(key) for key in keys if key in strategy_report}
    return {}


def _fmt_currency(value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    return f"${float(number):,.2f}"


def _fmt_percent(value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    return f"{float(number) * 100:.1f}%"


def _fmt_scalar(value: Any, *, digits: int = 2) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    return f"{float(number):,.{digits}f}"


def _humanize_benchmark_decision(value: Any) -> str:
    cleaned = clean_string(value)
    mapping = {
        "stock_still_better": "Long stock still looks cleaner",
        "options_show_edge": "Options show edge",
        "long_stock_benchmark": "Long stock is the benchmark",
        "tracks_stock_closely": "Tracks long stock closely",
    }
    return mapping.get(cleaned, cleaned.replace("_", " ").capitalize() if cleaned else "n/a")


def _humanize_label(value: Any) -> str:
    cleaned = clean_string(value)
    return cleaned.replace("_", " ").title() if cleaned else "n/a"


def _markdown_table(rows: list[tuple[str, str]]) -> list[str]:
    return [
        "| Focus | Read |",
        "| --- | --- |",
        *[f"| {left} | {right} |" for left, right in rows],
    ]


def _representative_bucket_priority(value: Any) -> int:
    mapping = {
        "just_works": 0,
        "works_well": 1,
        "almost_works": 2,
        "works_very_well": 3,
        "misses_badly": 4,
    }
    return mapping.get(clean_string(value), len(mapping))


def _sort_path_comparison_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    working = frame.copy()
    working["_bucket_rank"] = working.get("representative_bucket", pd.Series(dtype=str)).map(_representative_bucket_priority)
    sort_columns: list[str] = []
    ascending: list[bool] = []
    for column, direction in [
        ("_bucket_rank", True),
        ("target_beyond_expiry", True),
        ("weak_horizon_fit", True),
        ("objective_score", False),
        ("difference_vs_stock", False),
        ("return_on_comparison_capital", False),
        ("profit_loss", False),
    ]:
        if column in working.columns:
            sort_columns.append(column)
            ascending.append(direction)
    if sort_columns:
        working = working.sort_values(sort_columns, ascending=ascending, na_position="last", kind="mergesort")
    return working.drop(columns=["_bucket_rank"], errors="ignore").reset_index(drop=True)


def _trust_label_read(value: Any) -> str:
    cleaned = clean_string(value)
    mapping = {
        "trusted_quoted": "quoted and trusted",
        "quoted_prior_day": "prior-day quoted fallback",
        "fallback_only": "sparse fallback",
        "structure_only": "structure-only / low trust",
    }
    return mapping.get(cleaned, cleaned.replace("_", " ") if cleaned else "n/a")


def _path_comparison_note(row: dict[str, Any]) -> str:
    notes: list[str] = []
    trust_label = _trust_label_read(row.get("source_trust_label"))
    if trust_label != "n/a":
        notes.append(trust_label)
    quality_note = clean_string(row.get("source_quality_note"))
    if quality_note:
        notes.append(quality_note)
    if bool(row.get("target_beyond_expiry")):
        notes.append("Target runs beyond expiry, so timing fit is weaker.")
    elif bool(row.get("weak_horizon_fit")):
        notes.append("Timing fit is weaker than the requested thesis horizon.")
    benchmark_note = clean_string(row.get("benchmark_note"))
    if benchmark_note:
        notes.append(benchmark_note)
    return " ".join(notes) or "See the same-path comparison table for the detailed trust and timing read."


def _write_strategy_bundle(
    analysis: StrategyAnalysisComputation,
    bundle_dir: Path,
    file_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    strategy = analysis.strategy
    spots = np.asarray(analysis.spot_grid if analysis.spot_grid is not None else build_stock_grid(strategy.entry_spot), dtype=float)

    summary_df = pd.DataFrame([strategy.summary_metrics()])
    _write_frame(bundle_dir, file_map, "summary.csv", summary_df)

    scenarios_df = scenario_table(strategy, spot_grid=spots, horizons=analysis.horizons, iv_shocks=analysis.iv_shocks)
    _write_frame(bundle_dir, file_map, "scenarios.csv", scenarios_df)

    case_spots = build_case_template(strategy.entry_spot)
    case_rows = scenarios_df[scenarios_df["spot_price"].isin([round(value, 4) for value in case_spots.values()])].copy()
    _write_frame(bundle_dir, file_map, "scenario_cases.csv", case_rows)

    _track_chart(bundle_dir, file_map, plot_payoff_at_expiry(strategy, spots, _section_path(bundle_dir, "charts", "payoff_at_expiry.png")))

    before_expiry_horizon = min(max(strategy.days_to_expiry() // 2, 1), 30) if strategy.days_to_expiry() > 1 else 0
    _track_chart(
        bundle_dir,
        file_map,
        plot_estimated_value_vs_stock(
            strategy,
            spots,
            horizon_days=before_expiry_horizon,
            output_path=_section_path(bundle_dir, "charts", "estimated_value_vs_stock.png"),
        ),
    )

    if strategy.option_legs:
        iv_grid = np.linspace(-0.20, 0.20, 17)
        _track_chart(
            bundle_dir,
            file_map,
            plot_iv_sensitivity(
                strategy,
                stock_price=strategy.entry_spot,
                iv_grid=iv_grid,
                horizon_days=before_expiry_horizon,
                output_path=_section_path(bundle_dir, "charts", "iv_sensitivity.png"),
            ),
        )
        horizon_days_grid = np.linspace(0, max(strategy.days_to_expiry(), 1), min(max(strategy.days_to_expiry(), 1), 10) + 1)
        _track_chart(
            bundle_dir,
            file_map,
            plot_time_decay(
                strategy,
                stock_price=strategy.entry_spot,
                horizon_days_grid=horizon_days_grid,
                output_path=_section_path(bundle_dir, "charts", "time_sensitivity.png"),
            ),
        )

    if analysis.comparison_positions:
        compare_df = compare_positions(
            analysis.comparison_positions,
            mode=analysis.comparison_mode,
            spot_grid=spots,
        )
        _write_frame(bundle_dir, file_map, "comparison.csv", compare_df)
        for mode in sorted(compare_df["mode"].unique()):
            _track_chart(
                bundle_dir,
                file_map,
                plot_stock_vs_option_comparison(
                    analysis.comparison_positions,
                    spots,
                    output_path=_section_path(bundle_dir, "charts", f"comparison_{mode}.png"),
                    mode=mode,
                ),
            )

    report_metadata = {
        "generated_at": _utc_now_iso(),
        "report_kind": "strategy",
        "status": "partial" if any("fallback" in str(w).lower() for w in strategy.warnings) else "ok",
        "strategy_report": strategy.report_metadata(),
        "research_context": (strategy.resolved_metadata.get("research_context") or {}),
        "warnings": list(strategy.warnings),
        "comparison_strategies": [position.name for position in analysis.comparison_positions or []],
        "comparison_mode": analysis.comparison_mode,
        "spot_grid_min": float(spots.min()) if len(spots) else None,
        "spot_grid_max": float(spots.max()) if len(spots) else None,
        "generated_files": sorted(path for bucket in file_map.values() for path in bucket.values()),
    }
    _write_metadata(bundle_dir, file_map, "report_metadata.json", report_metadata)

    lines = [
        f"# {strategy.ticker} {strategy.name.replace('_', ' ').title()}",
        "",
        "## Key Metrics",
        "",
    ]
    for key, value in strategy.summary_metrics().items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Resolved Context",
            "",
            f"- spot_price_source: {strategy.resolved_metadata.get('spot_price_source')}",
            f"- spot_price_matched_date: {strategy.resolved_metadata.get('spot_price_matched_date')}",
            f"- risk_free_rate: {strategy.risk_free_rate}",
            f"- risk_free_rate_source: {strategy.resolved_metadata.get('risk_free_rate_source')}",
            f"- risk_free_rate_series: {strategy.resolved_metadata.get('risk_free_rate_series')}",
            f"- risk_free_rate_matched_date: {strategy.resolved_metadata.get('risk_free_rate_matched_date')}",
            f"- dividend_yield: {strategy.dividend_yield}",
        ]
    )
    research_context = report_metadata.get("research_context") or {}
    if research_context:
        lines.extend(["", "## Research Context", ""])
        nearest_event = research_context.get("nearest_event") or {}
        expected_move = research_context.get("expected_move") or {}
        dividend = research_context.get("dividend_assumption") or {}
        options_overview = research_context.get("options_overview") or {}
        if nearest_event:
            lines.append(f"- nearest_event_type: {nearest_event.get('event_type')}")
            lines.append(f"- nearest_event_date: {nearest_event.get('event_date')}")
        if expected_move:
            lines.append(f"- expected_move_matched: {expected_move.get('matched')}")
            lines.append(f"- expected_move_pct: {expected_move.get('expected_move_pct')}")
        if dividend:
            lines.append(f"- dividend_yield_assumption: {dividend.get('dividend_yield')}")
        if options_overview:
            lines.append(f"- options_overview_iv_rank: {options_overview.get('iv_rank')}")
    if strategy.notes or strategy.warnings:
        lines.extend(["", "## Notes", ""])
        for warning in strategy.warnings:
            lines.append(f"- warning: {warning}")
        for note in strategy.notes:
            lines.append(f"- {note}")
    _write_markdown(bundle_dir, file_map, "summary.md", "\n".join(lines))
    return report_metadata


def _write_scenario_bundle(
    result: ScenarioDashboardComputation,
    bundle_dir: Path,
    file_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    _write_frame(bundle_dir, file_map, "summary.csv", result.executive_summary)

    artifact_frames = {
        "strategy_summary.csv": result.strategy_summary,
        "named_scenarios.csv": result.named_scenarios,
        "stock_relative.csv": result.stock_relative,
        "spot_time_grid.csv": result.spot_time_grid,
        "spot_iv_grid.csv": result.spot_iv_grid,
        "forward_quick_scenarios.csv": result.forward_quick_scenarios,
        "forward_spot_time_grid.csv": result.forward_spot_time_grid,
        "forward_spot_iv_grid.csv": result.forward_spot_iv_grid,
        "forward_time_iv_grid.csv": result.forward_time_iv_grid,
        "valuation_explanation.csv": result.valuation_explanation,
    }
    for filename, frame in artifact_frames.items():
        _write_frame(bundle_dir, file_map, filename, frame)

    if not result.spot_time_grid.empty:
        spot_grid = sorted(pd.to_numeric(result.spot_time_grid["spot_price"], errors="coerce").dropna().unique().tolist())
    else:
        spot_grid = build_stock_grid(result.spot_price, points=21).tolist()
    representative = result.scenario_defaults.get("representative_horizon") or {}
    representative_days = int(representative.get("requested_days") or 0)
    iv_defaults = result.scenario_defaults.get("representative_iv_case") or {}
    representative_iv_shift = float(iv_defaults.get("iv_shift") or 0.0)
    comparison_capital = float(result.comparison_capital)
    horizon_specs = list(result.scenario_defaults.get("horizons") or [])
    spot_case_map = result.scenario_defaults.get("spot_cases") or {}
    stock_baseline = next((position for position in result.positions if position.name == "long_stock"), None)

    _track_chart(bundle_dir, file_map, plot_multi_strategy_payoff(result.positions, spot_grid, _section_path(bundle_dir, "charts", "payoff_comparison.png")))
    _track_chart(
        bundle_dir,
        file_map,
        plot_multi_strategy_estimated_value(
            result.positions,
            spot_grid,
            horizon_days=representative_days,
            iv_shift=representative_iv_shift,
            output_path=_section_path(bundle_dir, "charts", "estimated_value_comparison.png"),
        ),
    )
    for horizon in horizon_specs:
        label = str(horizon.get("label") or "")
        if not label:
            continue
        _track_chart(
            bundle_dir,
            file_map,
            plot_multi_strategy_estimated_value(
                result.positions,
                spot_grid,
                horizon_days=int(horizon.get("requested_days") or 0),
                iv_shift=representative_iv_shift,
                output_path=_section_path(bundle_dir, "charts", f"estimated_value_comparison_{slugify(label)}.png"),
            ),
        )

    _track_chart(
        bundle_dir,
        file_map,
        plot_stock_vs_option_comparison(
            result.positions,
            spot_grid,
            output_path=_section_path(bundle_dir, "charts", "stock_vs_strategies_equal_capital.png"),
            mode="equal_capital",
            horizon=representative_days,
            comparison_capital=comparison_capital,
            title=f"{result.ticker} Stock Vs Options Comparison (Normalized To ${comparison_capital:,.0f} Initial Capital)",
        ),
    )
    _track_chart(
        bundle_dir,
        file_map,
        plot_stock_vs_option_comparison(
            result.positions,
            spot_grid,
            output_path=_section_path(bundle_dir, "charts", "stock_vs_strategies_share_equivalent.png"),
            mode="share_equivalent",
            horizon=representative_days,
        ),
    )

    equal_capital = result.stock_relative.loc[result.stock_relative["mode"] == "equal_capital"].copy()
    if not equal_capital.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_heatmap(
                equal_capital,
                x_column="spot_price",
                y_column="strategy",
                value_column="stock_relative_difference",
                output_path=_section_path(bundle_dir, "charts", "compare_vs_stock_matrix.png"),
                title=f"{result.ticker} Relative PnL Vs Long Stock ({representative.get('label', '1m')})",
                x_label="Stock Price",
                y_label="Strategy",
                value_label="PnL Difference Vs Long Stock ($)",
                y_order=[position.name for position in result.positions],
                cmap="BrBG",
                center_zero=True,
            ),
        )

    horizon_labels = list(result.scenario_defaults.get("spot_time_display_order") or ["expiry", "6m", "3m", "1m", "1w", "entry"])
    iv_case_labels = list(result.scenario_defaults.get("spot_iv_display_order") or ["iv_up", "iv_unchanged", "iv_down"])
    for position in result.positions:
        strategy_slug = slugify(position.name)
        _track_chart(bundle_dir, file_map, plot_payoff_at_expiry(position, spot_grid, _section_path(bundle_dir, "charts", f"strategy_payoff_{strategy_slug}.png")))
        _track_chart(
            bundle_dir,
            file_map,
            plot_estimated_value_vs_stock(
                position,
                spot_grid,
                horizon_days=representative_days,
                iv_shift=representative_iv_shift,
                output_path=_section_path(bundle_dir, "charts", f"strategy_estimated_value_{strategy_slug}.png"),
            ),
        )
        comparison_positions = [position]
        if stock_baseline is not None and stock_baseline.name != position.name:
            comparison_positions = [stock_baseline, position]
        _track_chart(
            bundle_dir,
            file_map,
            plot_stock_vs_option_comparison(
                comparison_positions,
                spot_grid,
                output_path=_section_path(bundle_dir, "charts", f"strategy_vs_stock_{strategy_slug}.png"),
                mode="equal_capital",
                horizon=representative_days,
                comparison_capital=comparison_capital,
                title=f"{result.ticker} {position.name.replace('_', ' ').title()} Vs Long Stock (Normalized To ${comparison_capital:,.0f} Initial Capital)",
            ),
        )
        strategy_time = result.spot_time_grid.loc[
            (result.spot_time_grid["strategy"] == position.name)
            & (result.spot_time_grid["iv_case"] == "iv_unchanged")
        ].copy()
        if not strategy_time.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_heatmap(
                    strategy_time,
                    x_column="spot_price",
                    y_column="horizon",
                    value_column="profit_loss",
                    output_path=_section_path(bundle_dir, "charts", f"spot_time_{strategy_slug}.png"),
                    title=f"{result.ticker} {position.name.replace('_', ' ').title()} PnL By Spot x Time",
                    x_label="Stock Price",
                    y_label="Horizon",
                    value_label="Profit / Loss ($)",
                    y_order=horizon_labels,
                    cmap="BrBG",
                    center_zero=True,
                ),
            )
        strategy_iv = result.spot_iv_grid.loc[result.spot_iv_grid["strategy"] == position.name].copy()
        if not strategy_iv.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_heatmap(
                    strategy_iv,
                    x_column="spot_price",
                    y_column="iv_case",
                    value_column="estimated_value",
                    output_path=_section_path(bundle_dir, "charts", f"spot_iv_{strategy_slug}.png"),
                    title=f"{result.ticker} {position.name.replace('_', ' ').title()} Value By Spot x IV",
                    x_label="Stock Price",
                    y_label="IV Scenario",
                    value_label="Estimated Value ($)",
                    y_order=iv_case_labels,
                    cmap="cividis",
                ),
            )
        selected_cases = {
            label: float(case_payload.get("spot_price"))
            for label, case_payload in spot_case_map.items()
            if label in {"bear", "flat", "bull"} and case_payload.get("spot_price") is not None
        }
        if selected_cases:
            _track_chart(
                bundle_dir,
                file_map,
                plot_strategy_time_progression(
                    position,
                    horizon_specs=[(str(item.get("label") or ""), int(item.get("requested_days") or 0)) for item in horizon_specs if item.get("label")],
                    spot_cases=selected_cases,
                    iv_shift=representative_iv_shift,
                    output_path=_section_path(bundle_dir, "charts", f"strategy_time_progression_{strategy_slug}.png"),
                ),
            )

    executive_row = result.executive_summary.iloc[0].to_dict() if not result.executive_summary.empty else {}
    report_metadata = {
        "generated_at": _utc_now_iso(),
        "report_kind": "scenario",
        "analysis_name": "scenario",
        "status": result.status,
        "shareability_status": result.shareability_status,
        "comparison_capital": comparison_capital,
        "capital_sizing_mode": result.capital_sizing_mode,
        "featured_focus_strategy": result.featured_focus_strategy,
        "ticker": result.ticker,
        "snapshot_date": result.snapshot_date.isoformat(),
        "expiry_date": result.expiry_date.isoformat(),
        "warnings": list(result.warnings),
        "available_strategies": list(result.available_strategies),
        "omitted_strategies": list(result.omitted_strategies),
        "scenario_defaults": result.scenario_defaults,
        "decision_hints": result.decision_hints,
        "forward_defaults": result.forward_defaults,
        "replay_defaults": result.replay_defaults,
        "valuation_defaults": result.valuation_defaults,
        "what_matters_most": result.what_matters_most,
        "generated_files": sorted(path for bucket in file_map.values() for path in bucket.values()),
        "metadata": {
            "ticker": result.ticker,
            "snapshot_date": result.snapshot_date.isoformat(),
            "expiry_date": result.expiry_date.isoformat(),
            "spot_price": result.spot_price,
            "spot_source": result.resolved_metadata.get("spot_price_source"),
            "spot_matched_date": result.resolved_metadata.get("spot_price_matched_date"),
            "spot_note": result.resolved_metadata.get("spot_price_note"),
            "risk_free_rate": result.risk_free_rate,
            "risk_free_source": result.resolved_metadata.get("risk_free_rate_source"),
            "risk_free_series": result.resolved_metadata.get("risk_free_rate_series"),
            "risk_free_matched_date": result.resolved_metadata.get("risk_free_rate_matched_date"),
            "risk_free_note": result.resolved_metadata.get("risk_free_rate_note"),
            "dividend_yield": result.dividend_yield,
            "premium_mode": result.premium_mode,
            "source_snapshot_file": _project_relative_path(result.source_snapshot_file),
            "research_context": result.research_context,
            "comparison_capital": comparison_capital,
            "capital_sizing_mode": result.capital_sizing_mode,
            "decision_hints": result.decision_hints,
            "forward_defaults": result.forward_defaults,
            "featured_focus_strategy": result.featured_focus_strategy,
            "replay_defaults": result.replay_defaults,
            "valuation_defaults": result.valuation_defaults,
            "what_matters_most": result.what_matters_most,
            "executive_summary": executive_row,
        },
        "scenario": {
            "ticker": result.ticker,
            "snapshot_date": result.snapshot_date.isoformat(),
            "expiry_date": result.expiry_date.isoformat(),
            "spot_price": result.spot_price,
            "premium_mode": result.premium_mode,
            "risk_free_rate": result.risk_free_rate,
            "dividend_yield": result.dividend_yield,
            "comparison_capital": comparison_capital,
            "capital_sizing_mode": result.capital_sizing_mode,
            "source_snapshot_file": _project_relative_path(result.source_snapshot_file),
            "available_strategies": list(result.available_strategies),
            "omitted_strategies": list(result.omitted_strategies),
            "representative_horizon": representative,
            "representative_iv_case": iv_defaults,
            "executive_summary": executive_row,
            "decision_hints": result.decision_hints,
            "forward_defaults": result.forward_defaults,
            "featured_focus_strategy": result.featured_focus_strategy,
            "replay_defaults": result.replay_defaults,
            "valuation_defaults": result.valuation_defaults,
            "what_matters_most": result.what_matters_most,
            "strategy_cards": result.strategy_summary.to_dict(orient="records"),
        },
        "research_context": result.research_context,
        "related_strategy_details": [
            {
                "strategy": position.name,
                "expiry_date": position.expiry_date.isoformat() if position.expiry_date else result.expiry_date.isoformat(),
            }
            for position in result.positions
            if position.name != "long_stock"
        ],
    }
    _write_metadata(bundle_dir, file_map, "report_metadata.json", report_metadata)

    summary_lines = [
        f"# {result.ticker} Scenario Dashboard ({result.expiry_date.isoformat()})",
        "",
        "## Executive Summary",
        "",
        f"- status: {result.status}",
        f"- shareability_status: {result.shareability_status}",
        f"- comparison_capital: {comparison_capital}",
        f"- capital_sizing_mode: {result.capital_sizing_mode}",
        f"- included_strategies: {', '.join(result.available_strategies)}",
        f"- representative_horizon: {representative.get('label')} ({representative.get('requested_days')}d)",
        f"- featured_focus_strategy: {result.featured_focus_strategy}",
        f"- forward_lab_default_mode: {result.forward_defaults.get('mode')}",
        f"- what_matters_most: {result.what_matters_most}",
        "",
        "## Warnings",
        "",
    ]
    if result.warnings:
        summary_lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        summary_lines.append("- none")
    _write_markdown(bundle_dir, file_map, "summary.md", "\n".join(summary_lines))
    return report_metadata


def _write_contract_selection_bundle(
    result: ContractSelectionComputation,
    bundle_dir: Path,
    file_map: dict[str, dict[str, str]],
) -> dict[str, Any]:
    artifact_frames = {
        "candidate_summary.csv": result.candidate_summary,
        "ranked_candidates.csv": result.ranked_candidates,
        "compare_vs_stock.csv": result.compare_vs_stock,
        "chain_source_summary.csv": result.chain_source_summary,
        "market_context_summary.csv": result.market_context_summary,
        "required_path_rows.csv": result.required_path_rows,
        "required_path_summary.csv": result.required_path_summary,
        "assumed_path_trace_rows.csv": result.assumed_path_trace_rows,
        "iv_path_trace_rows.csv": result.iv_path_trace_rows,
        "compare_vs_stock_path_rows.csv": result.compare_vs_stock_path_rows,
        "iv_path_sensitivity_summary.csv": result.iv_path_sensitivity_summary,
        "path_risk_summary.csv": result.path_risk_summary,
        "stock_path_library.csv": result.stock_path_library,
        "stock_path_gallery.csv": result.stock_path_gallery,
        "iv_path_gallery.csv": result.iv_path_gallery,
        "stock_path_examples.csv": result.stock_path_examples,
        "iv_path_examples.csv": result.iv_path_examples,
        "path_pair_summary.csv": result.path_pair_summary,
        "option_value_over_path.csv": result.option_value_over_path,
        "compare_vs_stock_over_path.csv": result.compare_vs_stock_over_path,
        "representative_paths_summary.csv": result.representative_paths_summary,
        "strike_comparison_under_path.csv": result.strike_comparison_under_path,
        "expiry_comparison_under_path.csv": result.expiry_comparison_under_path,
        "long_call_value_over_path_strike_view.csv": result.long_call_value_over_path_strike_view,
        "long_call_value_over_path_expiry_view.csv": result.long_call_value_over_path_expiry_view,
        "long_call_value_over_path_best_of.csv": result.long_call_value_over_path_best_of,
        "decision_highlights.csv": result.decision_highlights,
        "decision_highlights_explanations.csv": result.decision_highlights_explanations,
        "candidate_robustness_summary.csv": result.candidate_robustness_summary,
        "candidate_tradeoff_matrix.csv": result.candidate_tradeoff_matrix,
        "stock_vs_option_takeaways.csv": result.stock_vs_option_takeaways,
        "highlights_score_breakdown.csv": result.highlights_score_breakdown,
        "action_board_candidates.csv": result.action_board_candidates,
        "buy_now_candidates.csv": result.buy_now_candidates,
        "watchlist_candidates.csv": result.watchlist_candidates,
        "avoid_for_now_candidates.csv": result.avoid_for_now_candidates,
        "prefer_stock_instead.csv": result.prefer_stock_instead,
        "decision_triggers.csv": result.decision_triggers,
        "action_board_score_breakdown.csv": result.action_board_score_breakdown,
        "action_board_explanations.csv": result.action_board_explanations,
        "bullish_long_call_action_board.csv": result.bullish_long_call_action_board,
        "bullish_long_call_watchlist.csv": result.bullish_long_call_watchlist,
        "bullish_long_call_avoid.csv": result.bullish_long_call_avoid,
        "bullish_long_call_triggers.csv": result.bullish_long_call_triggers,
        "bullish_long_call_score_breakdown.csv": result.bullish_long_call_score_breakdown,
        "top_candidate_cards.csv": result.top_candidate_cards,
        "chain_overview_summary.csv": result.chain_overview_summary,
        "chain_overview_candidates.csv": result.chain_overview_candidates,
        "other_structures_summary.csv": result.other_structures_summary,
        "stock_preference_summary.csv": result.stock_preference_summary,
        "entry_justification_candidates.csv": result.entry_justification_candidates,
        "required_stock_path_to_buy.csv": result.required_stock_path_to_buy,
        "required_move_summary.csv": result.required_move_summary,
        "required_move_vs_stock.csv": result.required_move_vs_stock,
        "required_iv_support_summary.csv": result.required_iv_support_summary,
        "entry_barrier_summary.csv": result.entry_barrier_summary,
        "thesis_path_gallery.csv": result.thesis_path_gallery,
        "thesis_iv_gallery.csv": result.thesis_iv_gallery,
        "thesis_mode_candidates.csv": result.thesis_mode_candidates,
        "thesis_path_family_summary.csv": result.thesis_path_family_summary,
        "thesis_iv_family_summary.csv": result.thesis_iv_family_summary,
        "thesis_candidate_ranking.csv": result.thesis_candidate_ranking,
        "max_justified_premium.csv": result.max_justified_premium,
        "current_vs_justified_premium.csv": result.current_vs_justified_premium,
        "thesis_required_move_summary.csv": result.thesis_required_move_summary,
        "thesis_stock_vs_option_summary.csv": result.thesis_stock_vs_option_summary,
        "candidate_stress_grid.csv": result.candidate_stress_grid,
        "premium_sensitivity_summary.csv": result.premium_sensitivity_summary,
        "timing_slip_summary.csv": result.timing_slip_summary,
        "target_stress_summary.csv": result.target_stress_summary,
        "stress_transition_summary.csv": result.stress_transition_summary,
        "single_option_decision_summary.csv": result.single_option_decision_summary,
        "single_option_decision_path_selections.csv": result.single_option_decision_path_selections,
        "single_option_representative_paths.csv": result.single_option_representative_paths,
        "single_option_path_outcomes.csv": result.single_option_path_outcomes,
        "single_option_path_family_counts.csv": result.single_option_path_family_counts,
        "single_option_timing_sensitivity.csv": result.single_option_timing_sensitivity,
        "single_option_iv_sensitivity.csv": result.single_option_iv_sensitivity,
        "single_option_entry_sensitivity.csv": result.single_option_entry_sensitivity,
        "single_option_summary_bullets.csv": result.single_option_summary_bullets,
        "required_vs_assumed_path_summary.csv": result.required_vs_assumed_path_summary,
        "path_case_rows.csv": result.path_case_rows,
        "path_case_summary.csv": result.path_case_summary,
        "path_case_chart_rows.csv": result.path_case_chart_rows,
        "path_case_strategy_rows.csv": result.path_case_strategy_rows,
        "path_case_family_rankings.csv": result.path_case_family_rankings,
        "path_case_candidate_rankings.csv": result.path_case_candidate_rankings,
        "strategy_selector_rows.csv": result.strategy_selector_rows,
        "strategy_selector_rankings.csv": result.strategy_selector_rankings,
        "family_comparison.csv": result.family_comparison,
        "candidate_comparison.csv": result.candidate_comparison,
        "strike_comparison.csv": result.strike_comparison,
        "expiry_comparison.csv": result.expiry_comparison,
    }
    for filename, frame in artifact_frames.items():
        _write_frame(bundle_dir, file_map, filename, frame)
    for filename, frame in sorted((result.path_view_tables or {}).items()):
        _write_frame(bundle_dir, file_map, filename, frame)

    available_families = result.strategy_selector_rows.loc[result.strategy_selector_rows.get("available") == True].copy()  # noqa: E712
    top_family = available_families.iloc[0].to_dict() if not available_families.empty else (
        result.strategy_selector_rows.iloc[0].to_dict() if not result.strategy_selector_rows.empty else {}
    )
    top_candidate = result.candidate_summary.iloc[0].to_dict() if not result.candidate_summary.empty else {}
    family_overview = result.family_comparison.iloc[0].to_dict() if not result.family_comparison.empty else {}
    active_family = clean_string(top_candidate.get("strategy_family")) or clean_string(top_family.get("strategy_family"))
    active_strike_rows = (
        result.strike_comparison_under_path.loc[
            result.strike_comparison_under_path.get("strategy_family").astype(str).str.lower() == active_family.lower()
        ].copy()
        if not result.strike_comparison_under_path.empty
        else pd.DataFrame()
    )
    active_expiry_rows = (
        result.expiry_comparison_under_path.loc[
            result.expiry_comparison_under_path.get("strategy_family").astype(str).str.lower() == active_family.lower()
        ].copy()
        if not result.expiry_comparison_under_path.empty
        else pd.DataFrame()
    )
    if active_strike_rows.empty and not result.strike_comparison.empty:
        active_strike_rows = result.strike_comparison.loc[
            result.strike_comparison.get("strategy_family").astype(str).str.lower() == active_family.lower()
        ].copy()
    if active_expiry_rows.empty and not result.expiry_comparison.empty:
        active_expiry_rows = result.expiry_comparison.loc[
            result.expiry_comparison.get("strategy_family").astype(str).str.lower() == active_family.lower()
        ].copy()
    if not active_strike_rows.empty:
        active_strike_rows = _sort_path_comparison_rows(active_strike_rows)
    if not active_expiry_rows.empty:
        active_expiry_rows = _sort_path_comparison_rows(active_expiry_rows)
    strike_overview = (
        active_strike_rows.iloc[0].to_dict()
        if not active_strike_rows.empty
        else (result.strike_comparison.iloc[0].to_dict() if not result.strike_comparison.empty else {})
    )
    expiry_overview = (
        active_expiry_rows.iloc[0].to_dict()
        if not active_expiry_rows.empty
        else (result.expiry_comparison.iloc[0].to_dict() if not result.expiry_comparison.empty else {})
    )
    top_family_slug = clean_string(top_family.get("winning_candidate_slug"))
    active_required_summary = result.required_path_summary.loc[
        (result.required_path_summary.get("summary_scope") == "family_representative")
        & (result.required_path_summary.get("goal") == clean_string(result.goal))
        & (result.required_path_summary.get("candidate_slug") == top_family_slug)
    ].copy()
    if active_required_summary.empty:
        active_required_summary = result.required_path_summary.loc[
            (result.required_path_summary.get("summary_scope") == "candidate")
            & (result.required_path_summary.get("candidate_slug") == clean_string(top_candidate.get("candidate_slug")))
            & (result.required_path_summary.get("goal") == clean_string(result.goal))
        ].copy()
    active_path_risk = pd.DataFrame()
    if not result.path_risk_summary.empty:
        active_path_risk = result.path_risk_summary.loc[
            (result.path_risk_summary.get("summary_scope") == "family_representative")
            & (result.path_risk_summary.get("candidate_slug") == top_family_slug)
            & (result.path_risk_summary.get("goal") == clean_string(result.goal))
        ].copy()
        if active_path_risk.empty:
            active_path_risk = result.path_risk_summary.loc[
                (result.path_risk_summary.get("summary_scope") == "candidate")
                & (result.path_risk_summary.get("candidate_slug") == clean_string(top_candidate.get("candidate_slug")))
                & (result.path_risk_summary.get("goal") == clean_string(result.goal))
            ].copy()
    active_iv_sensitivity = pd.DataFrame()
    if not result.iv_path_sensitivity_summary.empty:
        active_iv_sensitivity = result.iv_path_sensitivity_summary.loc[
            (result.iv_path_sensitivity_summary.get("summary_scope") == "family_representative")
            & (result.iv_path_sensitivity_summary.get("candidate_slug") == top_family_slug)
        ].copy()
        if active_iv_sensitivity.empty:
            active_iv_sensitivity = result.iv_path_sensitivity_summary.loc[
                (result.iv_path_sensitivity_summary.get("summary_scope") == "candidate")
                & (result.iv_path_sensitivity_summary.get("candidate_slug") == clean_string(top_candidate.get("candidate_slug")))
            ].copy()
    top_path_difficulty = clean_string(active_required_summary.iloc[0].get("required_path_difficulty")) if not active_required_summary.empty else ""
    stock_benchmark_decision = clean_string(top_candidate.get("stock_benchmark_decision")) or (
        "stock_still_better"
        if clean_string(top_candidate.get("strategy_family")) != "long_stock"
        and float(finite_or_none(top_candidate.get("difference_vs_stock")) or 0.0) <= 0.0
        else "options_show_edge"
    )
    top_timing_risk = clean_string(active_path_risk.iloc[0].get("timing_risk")) if not active_path_risk.empty else ""
    top_iv_risk = (
        clean_string(active_path_risk.iloc[0].get("iv_risk"))
        if not active_path_risk.empty
        else clean_string(active_iv_sensitivity.iloc[0].get("iv_risk"))
        if not active_iv_sensitivity.empty
        else ""
    )
    stock_benchmark_note = (
        clean_string(active_path_risk.iloc[0].get("benchmark_note"))
        if not active_path_risk.empty
        else clean_string(top_candidate.get("benchmark_note"))
    )
    benchmark_edge = finite_or_none(active_path_risk.iloc[0].get("benchmark_edge")) if not active_path_risk.empty else finite_or_none(top_candidate.get("difference_vs_stock"))
    benchmark_return_edge = finite_or_none(active_path_risk.iloc[0].get("benchmark_return_edge")) if not active_path_risk.empty else finite_or_none(top_candidate.get("difference_vs_stock_return_pct"))
    required_path_gap = finite_or_none(active_required_summary.iloc[0].get("path_gap_at_target")) if not active_required_summary.empty else None
    first_cleared_horizon = clean_string(active_required_summary.iloc[0].get("first_cleared_horizon")) if not active_required_summary.empty else ""
    iv_sensitivity_note = clean_string(active_iv_sensitivity.iloc[0].get("sensitivity_note")) if not active_iv_sensitivity.empty else ""
    top_path_risk = (
        "timing risk is dominant because the target extends beyond expiry"
        if bool(top_candidate.get("target_beyond_expiry"))
        else (
            top_timing_risk
            or top_iv_risk
            or clean_string(top_path_difficulty)
            or "path risk is modest under the active assumptions"
        )
    )
    decision_snapshot = pd.DataFrame(
        [
            {
                "ticker": result.ticker,
                "snapshot_date": result.snapshot_date.isoformat(),
                "target_date": result.target_date.isoformat(),
                "target_price": result.target_price,
                "thesis_target_date": result.thesis_target_date.isoformat(),
                "thesis_target_price": result.thesis_target_price,
                "goal": result.goal,
                "comparison_capital": result.comparison_capital,
                "stock_path_name": result.stock_path_name,
                "iv_path_name": result.iv_path_name,
                "spot_price_source": clean_string(result.report_metadata.get("spot_price_source")),
                "spot_field_used": clean_string(result.report_metadata.get("spot_field_used")),
                "spot_used_prior_date": bool(result.report_metadata.get("spot_used_prior_date")),
                "spot_quality_note": clean_string(result.report_metadata.get("spot_quality_note")),
                "spot_price_matched_date": clean_string(result.report_metadata.get("spot_price_matched_date")),
                "best_family": clean_string(top_family.get("strategy_label")) or clean_string(top_family.get("strategy_family")),
                "best_family_candidate": clean_string(top_family.get("winning_candidate_label")),
                "best_candidate": clean_string(top_candidate.get("candidate_label")),
                "best_candidate_family": clean_string(top_candidate.get("strategy_family")),
                "best_expiry": clean_string(top_candidate.get("expiry_date")),
                "best_strike": clean_string(top_candidate.get("strike_label")),
                "family_edge_status": clean_string(family_overview.get("current_objective_card_status")),
                "stock_benchmark_decision": stock_benchmark_decision,
                "stock_benchmark_label": _humanize_benchmark_decision(stock_benchmark_decision),
                "stock_benchmark_note": stock_benchmark_note,
                "benchmark_edge": benchmark_edge,
                "benchmark_return_edge": benchmark_return_edge,
                "top_path_risk": top_path_risk,
                "timing_risk": top_timing_risk,
                "iv_risk": top_iv_risk,
                "required_path_difficulty": top_path_difficulty,
                "required_path_gap_at_target": required_path_gap,
                "first_cleared_horizon": first_cleared_horizon,
                "iv_sensitivity_note": iv_sensitivity_note,
                "risk_free_rate_source": clean_string(result.report_metadata.get("risk_free_rate_source")),
                "risk_free_rate_series": clean_string(result.report_metadata.get("risk_free_rate_series")),
                "risk_free_rate_matched_date": clean_string(result.report_metadata.get("risk_free_rate_matched_date")),
                "risk_free_rate_note": clean_string(result.report_metadata.get("risk_free_rate_note")),
                "analysis_trust_level": clean_string(result.report_metadata.get("analysis_trust_level")),
                "analysis_trust_note": clean_string(result.report_metadata.get("analysis_trust_note")),
                "trusted_expiry_count": int(result.report_metadata.get("trusted_expiry_count") or 0),
                "fallback_only_expiry_count": int(result.report_metadata.get("fallback_only_expiry_count") or 0),
                "source_snapshot_storage_locations": " | ".join(result.report_metadata.get("source_snapshot_storage_locations") or []),
                "ibkr_same_day_spot_rejected_reason": clean_string(result.report_metadata.get("ibkr_same_day_spot_rejected_reason")),
                "best_strike_source_trust_label": clean_string(strike_overview.get("source_trust_label") or strike_overview.get("best_source_trust_label")),
                "best_expiry_source_trust_label": clean_string(expiry_overview.get("source_trust_label") or expiry_overview.get("best_source_trust_label")),
                "best_strike_representative_bucket": clean_string(strike_overview.get("representative_bucket")),
                "best_expiry_representative_bucket": clean_string(expiry_overview.get("representative_bucket")),
                "default_case_label": (result.report_metadata.get("path_case_defaults") or {}).get("default_case_label"),
                "primary_warning": result.warnings[0] if result.warnings else "",
            }
        ]
    )
    _write_frame(bundle_dir, file_map, "summary.csv", decision_snapshot)

    if not result.family_comparison.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_family_ranking_overview(
                result.family_comparison,
                output_path=_section_path(bundle_dir, "charts", "family_ranking_overview.png"),
                title=f"{result.ticker} Family Ranking Under Active Assumptions",
            ),
        )
    if not result.decision_highlights.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_highlights_overview(
                result.decision_highlights,
                output_path=_section_path(bundle_dir, "charts", "highlights_overview.png"),
                title=f"{result.ticker} Decision Highlights",
            ),
        )
    if not result.action_board_candidates.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_action_board_overview(
                result.action_board_candidates,
                output_path=_section_path(bundle_dir, "charts", "action_board_overview.png"),
                title=f"{result.ticker} Action Board",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_conviction_vs_robustness(
                result.action_board_candidates,
                output_path=_section_path(bundle_dir, "charts", "conviction_vs_robustness.png"),
                title=f"{result.ticker} Conviction Vs Robustness",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_buy_watch_avoid_matrix(
                result.action_board_candidates,
                output_path=_section_path(bundle_dir, "charts", "buy_watch_avoid_matrix.png"),
                title=f"{result.ticker} Buy / Watch / Avoid Matrix",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_stock_vs_option_preference_chart(
                result.action_board_candidates,
                output_path=_section_path(bundle_dir, "charts", "stock_vs_option_preference_chart.png"),
                title=f"{result.ticker} Stock Vs Option Preference",
            ),
        )
    bullish_chart_frame = pd.concat(
        [frame for frame in [result.bullish_long_call_action_board, result.stock_preference_summary] if frame is not None and not frame.empty],
        ignore_index=True,
    ) if (not result.bullish_long_call_action_board.empty or not result.stock_preference_summary.empty) else pd.DataFrame()
    if not result.bullish_long_call_action_board.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_action_board_overview(
                result.bullish_long_call_action_board,
                output_path=_section_path(bundle_dir, "charts", "bullish_action_board_overview.png"),
                title=f"{result.ticker} Bullish Long-Call Action Board",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_buy_watch_avoid_matrix(
                result.bullish_long_call_action_board,
                output_path=_section_path(bundle_dir, "charts", "bullish_buy_watch_avoid_matrix.png"),
                title=f"{result.ticker} Bullish Long Calls: Buy / Watch / Avoid",
            ),
        )
    if not result.top_candidate_cards.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_top_candidate_cards(
                result.top_candidate_cards,
                output_path=_section_path(bundle_dir, "charts", "top_candidate_cards.png"),
                title=f"{result.ticker} Top Bullish Call Cards",
            ),
        )
    if not result.chain_overview_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_chain_overview(
                summary=result.chain_overview_summary,
                candidates=result.chain_overview_candidates,
                output_path=_section_path(bundle_dir, "charts", "chain_overview.png"),
                title=f"{result.ticker} Chain Overview / Compare Options",
            ),
        )
    if not bullish_chart_frame.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_conviction_vs_robustness(
                bullish_chart_frame,
                output_path=_section_path(bundle_dir, "charts", "bullish_conviction_vs_robustness.png"),
                title=f"{result.ticker} Bullish Calls: Conviction Vs Robustness",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_stock_vs_option_preference_chart(
                bullish_chart_frame,
                output_path=_section_path(bundle_dir, "charts", "stock_vs_option_preference_chart.png"),
                title=f"{result.ticker} Stock Vs Bullish Long-Call Preference",
            ),
        )
    if not result.decision_triggers.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_trigger_map(
                result.decision_triggers,
                output_path=_section_path(bundle_dir, "charts", "trigger_map.png"),
                title=f"{result.ticker} Watchlist Trigger Map",
            ),
        )
    if not result.bullish_long_call_triggers.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_trigger_map(
                result.bullish_long_call_triggers,
                output_path=_section_path(bundle_dir, "charts", "bullish_trigger_map.png"),
                title=f"{result.ticker} Bullish Long-Call Trigger Map",
            ),
        )
    if not result.required_stock_path_to_buy.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_required_stock_path_to_buy(
                result.required_stock_path_to_buy,
                output_path=_section_path(bundle_dir, "charts", "required_stock_path_to_buy.png"),
                title="What Stock Path Is Required To Justify These Calls?",
            ),
        )
    if not result.required_move_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_required_move_speed_vs_magnitude(
                result.required_move_summary,
                output_path=_section_path(bundle_dir, "charts", "required_move_speed_vs_magnitude.png"),
                title=f"{result.ticker} Required Move: Speed Vs Magnitude",
            ),
        )
    if not result.required_move_vs_stock.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_required_move_vs_stock_chart(
                result.required_move_vs_stock,
                output_path=_section_path(bundle_dir, "charts", "required_move_vs_stock_chart.png"),
                title=f"{result.ticker} Required Move Vs Stock",
            ),
        )
    if not result.entry_barrier_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_strike_expiry_entry_barrier_map(
                result.entry_barrier_summary,
                output_path=_section_path(bundle_dir, "charts", "strike_expiry_entry_barrier_map.png"),
                title=f"{result.ticker} Strike / Expiry Entry Barrier Map",
            ),
        )
    if not result.required_iv_support_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_iv_support_requirement_chart(
                result.required_iv_support_summary,
                output_path=_section_path(bundle_dir, "charts", "iv_support_requirement_chart.png"),
                title=f"{result.ticker} IV Support Requirement",
            ),
        )
    if not result.thesis_path_gallery.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_thesis_path_gallery(
                result.thesis_path_gallery,
                output_path=_section_path(bundle_dir, "charts", "thesis_path_gallery.png"),
                title=f"Paths To A ${result.thesis_target_price:,.0f} {result.thesis_target_date.strftime('%b-%Y')} {result.ticker} Thesis",
            ),
        )
    if not result.thesis_iv_gallery.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_thesis_iv_gallery(
                result.thesis_iv_gallery,
                output_path=_section_path(bundle_dir, "charts", "thesis_iv_gallery.png"),
                title=f"{result.ticker} IV Regimes For Thesis Mode",
            ),
        )
    if not result.thesis_candidate_ranking.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_thesis_candidate_overview(
                result.thesis_candidate_ranking,
                output_path=_section_path(bundle_dir, "charts", "thesis_candidate_overview.png"),
                title=f"Which Calls Become Reasonable Under A ${result.thesis_target_price:,.0f} {result.thesis_target_date.strftime('%b-%Y')} Thesis?",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_current_vs_justified_premium(
                result.current_vs_justified_premium,
                output_path=_section_path(bundle_dir, "charts", "current_vs_justified_premium.png"),
                title=f"{result.ticker} Current Premium Vs Thesis-Justified Premium",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_thesis_stock_vs_option(
                result.thesis_stock_vs_option_summary,
                output_path=_section_path(bundle_dir, "charts", "thesis_stock_vs_option.png"),
                title=f"When Stock Still Beats Calls Even If The {result.ticker} Thesis Is Right",
            ),
        )
    if not result.thesis_mode_candidates.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_thesis_path_vs_value(
                result.thesis_mode_candidates,
                output_path=_section_path(bundle_dir, "charts", "thesis_path_vs_value.png"),
                title=f"How The Path To ${result.thesis_target_price:,.0f} Changes Option Value",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_thesis_iv_vs_value(
                result.thesis_mode_candidates,
                output_path=_section_path(bundle_dir, "charts", "thesis_iv_vs_value.png"),
                title=f"How Much IV Support The {result.ticker} Thesis Needs",
            ),
        )
    if not result.stress_transition_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_stress_test_overview(
                result.stress_transition_summary,
                output_path=_section_path(bundle_dir, "charts", "stress_test_overview.png"),
                title=f"{result.ticker} Stress Test Summary For Top Bullish Calls",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_top_candidate_stress_cards(
                result.stress_transition_summary,
                output_path=_section_path(bundle_dir, "charts", "top_candidate_stress_cards.png"),
                title=f"{result.ticker} Top Candidate Stress Cards",
            ),
        )
    if not result.premium_sensitivity_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_premium_sensitivity_chart(
                result.premium_sensitivity_summary,
                output_path=_section_path(bundle_dir, "charts", "premium_sensitivity_chart.png"),
                title="How Sensitive Are These Calls To Entry Price?",
            ),
        )
    if not result.timing_slip_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_timing_slip_chart(
                result.timing_slip_summary,
                output_path=_section_path(bundle_dir, "charts", "timing_slip_chart.png"),
                title="What Happens If The Thesis Arrives Later?",
            ),
        )
    if not result.target_stress_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_target_stress_chart(
                result.target_stress_summary,
                output_path=_section_path(bundle_dir, "charts", "target_stress_chart.png"),
                title="Do These Calls Need The Thesis To Overshoot?",
            ),
        )
    if not result.single_option_decision_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_single_option_decision_view(
                summary=result.single_option_decision_summary,
                representative_paths=result.single_option_representative_paths,
                path_outcomes=result.single_option_path_outcomes,
                iv_sensitivity=result.single_option_iv_sensitivity,
                entry_sensitivity=result.single_option_entry_sensitivity,
                summary_bullets=result.single_option_summary_bullets,
                output_path=_section_path(bundle_dir, "charts", "single_option_decision_view.png"),
                title=f"{result.ticker} Single-Option Decision View",
            ),
        )
    if not result.candidate_tradeoff_matrix.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_candidate_robustness_vs_upside(
                result.candidate_tradeoff_matrix,
                output_path=_section_path(bundle_dir, "charts", "candidate_robustness_vs_upside.png"),
                title=f"{result.ticker} Aggressive Upside Vs Robustness",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_strike_expiry_tradeoff_overview(
                result.candidate_tradeoff_matrix,
                output_path=_section_path(bundle_dir, "charts", "strike_expiry_tradeoff_overview.png"),
                title=f"{result.ticker} Strike And Expiry Trade-Offs",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_stock_vs_option_decision_chart(
                result.candidate_tradeoff_matrix,
                output_path=_section_path(bundle_dir, "charts", "stock_vs_option_decision_chart.png"),
                title=f"{result.ticker} Stock Vs Option Decision Read",
            ),
        )
    if not result.candidate_robustness_summary.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_path_survival_scorecard(
                result.candidate_robustness_summary,
                output_path=_section_path(bundle_dir, "charts", "path_survival_scorecard.png"),
                title=f"{result.ticker} Path Survival Scorecard",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_iv_robustness_scorecard(
                result.candidate_robustness_summary,
                output_path=_section_path(bundle_dir, "charts", "iv_robustness_scorecard.png"),
                title=f"{result.ticker} IV Robustness Scorecard",
            ),
        )

    path_case_defaults = result.report_metadata.get("path_case_defaults") or {}
    required_path_chart_frame = result.path_case_chart_rows.loc[
        (result.path_case_chart_rows.get("case_label") == path_case_defaults.get("default_case_label", "0%"))
        & (result.path_case_chart_rows.get("goal") == path_case_defaults.get("default_goal", result.goal))
        & (result.path_case_chart_rows.get("display_mode") == "strategy_compare")
        & (result.path_case_chart_rows.get("iv_mode") == path_case_defaults.get("default_iv_mode", "path_preset"))
        & (result.path_case_chart_rows.get("iv_variant") == path_case_defaults.get("default_iv_variant", result.iv_path_name))
    ].copy()
    if required_path_chart_frame.empty:
        required_path_chart_frame = result.path_case_chart_rows.loc[
            (result.path_case_chart_rows.get("case_label") == path_case_defaults.get("default_case_label", "0%"))
            & (result.path_case_chart_rows.get("goal") == path_case_defaults.get("default_goal", result.goal))
            & (result.path_case_chart_rows.get("display_mode") == "strategy_compare")
        ].copy()
    if not required_path_chart_frame.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_required_path_strategy_compare(
                required_path_chart_frame,
                output_path=_section_path(bundle_dir, "charts", "required_path_strategy_compare.png"),
                title=f"{result.ticker} Required Path By Strategy Vs Assumed Path",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_required_path_strategy_compare(
                required_path_chart_frame,
                output_path=_section_path(bundle_dir, "charts", "required_path_vs_assumed_path.png"),
                title=f"{result.ticker} Required Path Vs Assumed Path",
            ),
        )
    if not result.assumed_path_trace_rows.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_assumed_path_value_progression(
                result.assumed_path_trace_rows,
                output_path=_section_path(bundle_dir, "charts", "assumed_path_value_progression.png"),
                title=f"{result.ticker} Active Assumed Path Value Progression",
            ),
        )
    if not result.iv_path_trace_rows.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_iv_path_trace(
                result.iv_path_trace_rows,
                output_path=_section_path(bundle_dir, "charts", "iv_path_trace.png"),
                title=f"{result.ticker} Active And Comparison IV Paths",
            ),
        )
    if not result.compare_vs_stock_path_rows.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_compare_vs_stock_path(
                result.compare_vs_stock_path_rows,
                output_path=_section_path(bundle_dir, "charts", "compare_vs_stock_path_delta.png"),
                title=f"{result.ticker} Path Delta Vs Long Stock",
            ),
        )
    if not result.stock_path_gallery.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_stock_path_gallery(
                result.stock_path_gallery,
                output_path=_section_path(bundle_dir, "charts", "stock_path_gallery.png"),
                title=f"{result.ticker} Stock Path Gallery",
            ),
        )
    if not result.iv_path_gallery.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_iv_path_gallery(
                result.iv_path_gallery,
                output_path=_section_path(bundle_dir, "charts", "iv_path_gallery.png"),
                title=f"{result.ticker} IV Path Gallery",
            ),
        )
    if not result.stock_path_examples.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_representative_stock_paths(
                result.stock_path_examples,
                output_path=_section_path(bundle_dir, "charts", "representative_stock_paths.png"),
                title=f"{result.ticker} Representative Stock Paths",
            ),
        )
    if not result.iv_path_examples.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_representative_iv_paths(
                result.iv_path_examples,
                output_path=_section_path(bundle_dir, "charts", "representative_iv_paths.png"),
                title=f"{result.ticker} Representative IV Paths",
            ),
        )
    if not result.option_value_over_path.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_option_value_over_path(
                result.option_value_over_path,
                output_path=_section_path(bundle_dir, "charts", "option_value_over_path.png"),
                title=f"{result.ticker} Option Value Over Representative Path",
            ),
        )
    if not result.compare_vs_stock_over_path.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_compare_vs_stock_over_path(
                result.compare_vs_stock_over_path,
                output_path=_section_path(bundle_dir, "charts", "compare_vs_stock_over_path.png"),
                title=f"{result.ticker} Compare Vs Stock Over Representative Path",
            ),
        )
    if not result.strike_comparison_under_path.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_path_comparison(
                result.strike_comparison_under_path.assign(comparison_scope="strike"),
                output_path=_section_path(bundle_dir, "charts", "strike_comparison_under_same_path.png"),
                title=f"{result.ticker} Strike Comparison Under The Same Path",
                comparison_scope="strike",
            ),
        )
    if not result.expiry_comparison_under_path.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_path_comparison(
                result.expiry_comparison_under_path.assign(comparison_scope="expiry"),
                output_path=_section_path(bundle_dir, "charts", "expiry_comparison_under_same_path.png"),
                title=f"{result.ticker} Expiry Comparison Under The Same Path",
                comparison_scope="expiry",
            ),
        )
    if not result.long_call_value_over_path_strike_view.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_long_call_value_over_path_strike_view(
                result.long_call_value_over_path_strike_view,
                output_path=_section_path(bundle_dir, "charts", "long_call_value_over_path_strike_view.png"),
                title=f"{result.ticker} Long Call Comparison - Same Expiry, Different Strikes",
            ),
        )
    if not result.long_call_value_over_path_expiry_view.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_long_call_value_over_path_expiry_view(
                result.long_call_value_over_path_expiry_view,
                output_path=_section_path(bundle_dir, "charts", "long_call_value_over_path_expiry_view.png"),
                title=f"{result.ticker} Long Call Comparison - Same Strike Concept, Different Expiries",
            ),
        )
    if not result.long_call_value_over_path_best_of.empty:
        _track_chart(
            bundle_dir,
            file_map,
            plot_long_call_value_over_path_best_of(
                result.long_call_value_over_path_best_of,
                output_path=_section_path(bundle_dir, "charts", "long_call_value_over_path_best_of.png"),
                title=f"{result.ticker} Long Call Comparison - Best Curated Set Under Same Path",
            ),
        )
    for path_meta in result.report_metadata.get("path_centric_focus_paths") or []:
        path_name = clean_string(path_meta.get("path_name"))
        path_label = _humanize_label(path_meta.get("path_label") or path_name)
        compare_filename = clean_string(path_meta.get("compare_table"))
        strike_filename = clean_string(path_meta.get("strike_value_table") or path_meta.get("strike_table"))
        strike_delta_filename = clean_string(path_meta.get("strike_delta_table"))
        expiry_filename = clean_string(path_meta.get("expiry_value_table") or path_meta.get("expiry_table"))
        expiry_delta_filename = clean_string(path_meta.get("expiry_delta_table"))
        best_of_filename = clean_string(path_meta.get("best_of_value_table") or path_meta.get("best_of_table"))
        best_of_delta_filename = clean_string(path_meta.get("best_of_delta_table"))
        iv_value_filename = clean_string(path_meta.get("iv_value_table"))
        iv_delta_filename = clean_string(path_meta.get("iv_delta_table"))
        strike_iv_value_filename = clean_string(path_meta.get("strike_iv_value_table"))
        strike_iv_delta_filename = clean_string(path_meta.get("strike_iv_delta_table"))
        expiry_iv_value_filename = clean_string(path_meta.get("expiry_iv_value_table"))
        expiry_iv_delta_filename = clean_string(path_meta.get("expiry_iv_delta_table"))
        best_of_iv_value_filename = clean_string(path_meta.get("best_of_iv_value_table"))
        best_of_iv_delta_filename = clean_string(path_meta.get("best_of_iv_delta_table"))
        compare_frame = result.path_view_tables.get(compare_filename, pd.DataFrame())
        strike_frame = result.path_view_tables.get(strike_filename, pd.DataFrame())
        strike_delta_frame = result.path_view_tables.get(strike_delta_filename, pd.DataFrame())
        expiry_frame = result.path_view_tables.get(expiry_filename, pd.DataFrame())
        expiry_delta_frame = result.path_view_tables.get(expiry_delta_filename, pd.DataFrame())
        best_of_frame = result.path_view_tables.get(best_of_filename, pd.DataFrame())
        best_of_delta_frame = result.path_view_tables.get(best_of_delta_filename, pd.DataFrame())
        iv_value_frame = result.path_view_tables.get(iv_value_filename, pd.DataFrame())
        iv_delta_frame = result.path_view_tables.get(iv_delta_filename, pd.DataFrame())
        strike_iv_value_frame = result.path_view_tables.get(strike_iv_value_filename, pd.DataFrame())
        strike_iv_delta_frame = result.path_view_tables.get(strike_iv_delta_filename, pd.DataFrame())
        expiry_iv_value_frame = result.path_view_tables.get(expiry_iv_value_filename, pd.DataFrame())
        expiry_iv_delta_frame = result.path_view_tables.get(expiry_iv_delta_filename, pd.DataFrame())
        best_of_iv_value_frame = result.path_view_tables.get(best_of_iv_value_filename, pd.DataFrame())
        best_of_iv_delta_frame = result.path_view_tables.get(best_of_iv_delta_filename, pd.DataFrame())
        if not compare_frame.empty:
            lead = compare_frame.sort_values(["selection_rank", "requested_days"]).iloc[0].to_dict()
            _track_chart(
                bundle_dir,
                file_map,
                plot_path_long_call_compare_vs_stock(
                    compare_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("compare_chart"))),
                    title=f"{result.ticker} Long Call - Delta Vs Stock Under Path: {path_label}",
                    subtitle=(
                        f"Fixed: IV path = {_humanize_label(lead.get('iv_path_name'))}; "
                        "curated long-call subset only under this named stock path."
                    ),
                ),
            )
        if not strike_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_value_over_path_strike_view(
                    strike_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("strike_value_chart") or path_meta.get("strike_chart"))),
                    title=f"{result.ticker} Long Call - Strike Ladder Under Path: {path_label}",
                ),
            )
        if not strike_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_delta_over_path_strike_view(
                    strike_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("strike_delta_chart"))),
                    title=f"{result.ticker} Long Call - Strike Delta Vs Stock Under Path: {path_label}",
                ),
            )
        if not expiry_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_value_over_path_expiry_view(
                    expiry_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("expiry_value_chart") or path_meta.get("expiry_chart"))),
                    title=f"{result.ticker} Long Call - Expiry Ladder Under Path: {path_label}",
                ),
            )
        if not expiry_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_delta_over_path_expiry_view(
                    expiry_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("expiry_delta_chart"))),
                    title=f"{result.ticker} Long Call - Expiry Delta Vs Stock Under Path: {path_label}",
                ),
            )
        if not best_of_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_value_over_path_best_of(
                    best_of_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("best_of_value_chart") or path_meta.get("best_of_chart"))),
                    title=f"{result.ticker} Long Call - Best-Of Under Path: {path_label}",
                ),
            )
        if not best_of_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_delta_over_path_best_of(
                    best_of_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("best_of_delta_chart"))),
                    title=f"{result.ticker} Long Call - Best-Of Delta Vs Stock Under Path: {path_label}",
                ),
            )
        if not iv_value_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_path_value(
                    iv_value_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("iv_value_chart"))),
                    title=f"{result.ticker} Long Call - Value Under Fixed Stock Path: {path_label}, Varying IV",
                ),
            )
        if not iv_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_path_delta(
                    iv_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("iv_delta_chart"))),
                    title=f"{result.ticker} Long Call - Delta Vs Stock Under Fixed Stock Path: {path_label}, Varying IV",
                ),
            )
        if not strike_iv_value_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_expanded_value(
                    strike_iv_value_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("strike_iv_value_chart"))),
                    title=f"{result.ticker} Long Call - Strike Ladder IV Sensitivity Under Path: {path_label}",
                ),
            )
        if not strike_iv_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_expanded_delta(
                    strike_iv_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("strike_iv_delta_chart"))),
                    title=f"{result.ticker} Long Call - Strike Ladder IV Delta Vs Stock: {path_label}",
                ),
            )
        if not expiry_iv_value_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_expanded_value(
                    expiry_iv_value_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("expiry_iv_value_chart"))),
                    title=f"{result.ticker} Long Call - Expiry Ladder IV Sensitivity Under Path: {path_label}",
                ),
            )
        if not expiry_iv_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_expanded_delta(
                    expiry_iv_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("expiry_iv_delta_chart"))),
                    title=f"{result.ticker} Long Call - Expiry Ladder IV Delta Vs Stock: {path_label}",
                ),
            )
        if not best_of_iv_value_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_expanded_value(
                    best_of_iv_value_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("best_of_iv_value_chart"))),
                    title=f"{result.ticker} Long Call - Best-Of IV Sensitivity Under Path: {path_label}",
                ),
            )
        if not best_of_iv_delta_frame.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_long_call_iv_expanded_delta(
                    best_of_iv_delta_frame,
                    output_path=_section_path(bundle_dir, "charts", clean_string(path_meta.get("best_of_iv_delta_chart"))),
                    title=f"{result.ticker} Long Call - Best-Of IV Delta Vs Stock: {path_label}",
                ),
            )

    report_metadata = {
        "generated_at": result.generated_at,
        "report_kind": "contract_selection",
        "analysis_name": "contract_selection",
        "status": result.status,
        "shareability_status": result.shareability_status,
        "comparison_capital": result.comparison_capital,
        "warnings": list(result.warnings),
        "generated_files": sorted(path for bucket in file_map.values() for path in bucket.values()),
        "path_centric_focus_paths": list(result.report_metadata.get("path_centric_focus_paths") or []),
        "decision_highlights": result.decision_highlights.head(12).to_dict(orient="records") if not result.decision_highlights.empty else [],
        "action_board": result.action_board_candidates.head(18).to_dict(orient="records") if not result.action_board_candidates.empty else [],
        "bullish_action_board": result.bullish_long_call_action_board.head(18).to_dict(orient="records") if not result.bullish_long_call_action_board.empty else [],
        "stress_tests": result.stress_transition_summary.head(8).to_dict(orient="records") if not result.stress_transition_summary.empty else [],
        "single_option_decision": result.single_option_decision_summary.head(1).to_dict(orient="records") if not result.single_option_decision_summary.empty else [],
        "other_structures_summary": result.other_structures_summary.head(12).to_dict(orient="records") if not result.other_structures_summary.empty else [],
        "metadata": {
            "ticker": result.ticker,
            "snapshot_date": result.snapshot_date.isoformat(),
            "target_price": result.target_price,
            "target_date": result.target_date.isoformat(),
            "target_horizon": result.target_horizon_label,
            "target_horizon_days": result.target_horizon_days,
            "comparison_capital": result.comparison_capital,
            "strategy_families": list(result.strategy_families),
            "selection_scope": result.selection_scope,
            "available_expiries": list(result.available_expiries),
            "used_nearby_snapshot_fallback": bool(result.selection_scope.get("used_nearby_snapshot_fallback")),
            "spot_price": result.spot_price,
            "spot_price_source": result.report_metadata.get("spot_price_source"),
            "spot_price_matched_date": result.report_metadata.get("spot_price_matched_date"),
            "spot_field_used": result.report_metadata.get("spot_field_used"),
            "spot_used_prior_date": bool(result.report_metadata.get("spot_used_prior_date")),
            "spot_price_note": result.report_metadata.get("spot_price_note"),
            "spot_quality_note": result.report_metadata.get("spot_quality_note"),
            "ibkr_same_day_spot_attempted": bool(result.report_metadata.get("ibkr_same_day_spot_attempted")),
            "ibkr_same_day_spot_rejected_reason": result.report_metadata.get("ibkr_same_day_spot_rejected_reason"),
            "risk_free_rate": result.risk_free_rate,
            "risk_free_rate_source": result.report_metadata.get("risk_free_rate_source"),
            "risk_free_rate_series": result.report_metadata.get("risk_free_rate_series"),
            "risk_free_rate_matched_date": result.report_metadata.get("risk_free_rate_matched_date"),
            "risk_free_rate_note": result.report_metadata.get("risk_free_rate_note"),
            "dividend_yield": result.dividend_yield,
            "research_context": result.report_metadata.get("research_context") or {},
            "research_context_expiry_used": result.report_metadata.get("research_context_expiry_used"),
            "expected_move_matched": bool((result.report_metadata.get("research_context") or {}).get("expected_move", {}).get("matched")),
            "nearest_event_type": (result.report_metadata.get("research_context") or {}).get("nearest_event", {}).get("event_type"),
            "iv_shift_points": result.iv_shift_points,
            "goal": result.goal,
            "target_option_value": result.target_option_value,
            "objective_mode": result.objective_mode,
            "downside_tolerance": result.downside_tolerance,
            "simplicity_preference": result.simplicity_preference,
            "stock_path_name": result.stock_path_name,
            "stock_path_points": result.stock_path_points,
            "stock_path_mode": result.stock_path_mode,
            "stock_path_target_end": result.stock_path_target_end,
            "iv_path_name": result.iv_path_name,
            "iv_path_points": result.iv_path_points,
            "iv_path_mode": result.iv_path_mode,
            "simulated_path_count": result.simulated_path_count,
            "representative_selection_mode": result.representative_selection_mode,
            "simulation_seed": result.simulation_seed,
            "simulation_context": (result.report_metadata.get("simulation_context") or {}),
            "calibration_context": result.calibration_context,
            "best_candidate_cards": list(result.best_candidate_cards),
            "strategy_selector_defaults": (result.report_metadata.get("strategy_selector_defaults") or {}),
            "strategy_selector_context": result.strategy_selector_context,
            "strategy_selector_best_cards": list(result.strategy_selector_best_cards),
            "path_explorer_defaults": (result.report_metadata.get("path_explorer_defaults") or {}),
            "path_case_defaults": (result.report_metadata.get("path_case_defaults") or {}),
            "path_case_cases": (result.report_metadata.get("path_case_cases") or {}),
            "source_snapshot_files": list(result.report_metadata.get("source_snapshot_files") or []),
            "source_snapshot_storage_locations": list(result.report_metadata.get("source_snapshot_storage_locations") or []),
            "source_snapshot_dates": list(result.report_metadata.get("source_snapshot_dates") or []),
            "rejected_sparse_same_day_ibkr_files": list(result.report_metadata.get("rejected_sparse_same_day_ibkr_files") or []),
            "used_full_quoted_ibkr_same_date": bool(result.report_metadata.get("used_full_quoted_ibkr_same_date")),
            "analysis_trust_level": result.report_metadata.get("analysis_trust_level"),
            "analysis_trust_note": result.report_metadata.get("analysis_trust_note"),
            "bullish_action_board": result.report_metadata.get("bullish_action_board") or [],
            "trusted_expiry_count": int(result.report_metadata.get("trusted_expiry_count") or 0),
            "fallback_only_expiry_count": int(result.report_metadata.get("fallback_only_expiry_count") or 0),
            "default_strategy_family": result.report_metadata.get("default_strategy_family"),
            "default_candidate_within_family": result.report_metadata.get("default_candidate_within_family"),
            "default_contract_for_path_explorer": result.report_metadata.get("default_contract_for_path_explorer"),
            "path_centric_focus_paths": list(result.report_metadata.get("path_centric_focus_paths") or []),
            "decision_highlights": result.decision_highlights.head(12).to_dict(orient="records") if not result.decision_highlights.empty else [],
            "run_slug": result.run_slug,
        },
        "contract_selection": {
            "ticker": result.ticker,
            "snapshot_date": result.snapshot_date.isoformat(),
            "target_price": result.target_price,
            "target_date": result.target_date.isoformat(),
            "target_horizon": result.target_horizon_label,
            "target_horizon_days": result.target_horizon_days,
            "comparison_capital": result.comparison_capital,
            "strategy_families": list(result.strategy_families),
            "selection_scope": result.selection_scope,
            "available_expiries": list(result.available_expiries),
            "best_candidate_cards": list(result.best_candidate_cards),
            "spot_price": result.spot_price,
            "spot_price_source": result.report_metadata.get("spot_price_source"),
            "spot_price_matched_date": result.report_metadata.get("spot_price_matched_date"),
            "spot_field_used": result.report_metadata.get("spot_field_used"),
            "spot_used_prior_date": bool(result.report_metadata.get("spot_used_prior_date")),
            "spot_price_note": result.report_metadata.get("spot_price_note"),
            "spot_quality_note": result.report_metadata.get("spot_quality_note"),
            "ibkr_same_day_spot_attempted": bool(result.report_metadata.get("ibkr_same_day_spot_attempted")),
            "ibkr_same_day_spot_rejected_reason": result.report_metadata.get("ibkr_same_day_spot_rejected_reason"),
            "risk_free_rate": result.risk_free_rate,
            "risk_free_rate_source": result.report_metadata.get("risk_free_rate_source"),
            "risk_free_rate_series": result.report_metadata.get("risk_free_rate_series"),
            "risk_free_rate_matched_date": result.report_metadata.get("risk_free_rate_matched_date"),
            "risk_free_rate_note": result.report_metadata.get("risk_free_rate_note"),
            "research_context": result.report_metadata.get("research_context") or {},
            "research_context_expiry_used": result.report_metadata.get("research_context_expiry_used"),
            "expected_move_matched": bool((result.report_metadata.get("research_context") or {}).get("expected_move", {}).get("matched")),
            "nearest_event_type": (result.report_metadata.get("research_context") or {}).get("nearest_event", {}).get("event_type"),
            "goal": result.goal,
            "target_option_value": result.target_option_value,
            "objective_mode": result.objective_mode,
            "downside_tolerance": result.downside_tolerance,
            "simplicity_preference": result.simplicity_preference,
            "stock_path_name": result.stock_path_name,
            "stock_path_points": result.stock_path_points,
            "stock_path_mode": result.stock_path_mode,
            "stock_path_target_end": result.stock_path_target_end,
            "iv_path_name": result.iv_path_name,
            "iv_path_points": result.iv_path_points,
            "iv_path_mode": result.iv_path_mode,
            "simulated_path_count": result.simulated_path_count,
            "representative_selection_mode": result.representative_selection_mode,
            "simulation_seed": result.simulation_seed,
            "simulation_context": (result.report_metadata.get("simulation_context") or {}),
            "calibration_context": result.calibration_context,
            "strategy_selector_defaults": (result.report_metadata.get("strategy_selector_defaults") or {}),
            "strategy_selector_context": result.strategy_selector_context,
            "strategy_selector_best_cards": list(result.strategy_selector_best_cards),
            "path_explorer_defaults": (result.report_metadata.get("path_explorer_defaults") or {}),
            "path_case_defaults": (result.report_metadata.get("path_case_defaults") or {}),
            "path_case_cases": (result.report_metadata.get("path_case_cases") or {}),
            "source_snapshot_files": list(result.report_metadata.get("source_snapshot_files") or []),
            "source_snapshot_storage_locations": list(result.report_metadata.get("source_snapshot_storage_locations") or []),
            "source_snapshot_dates": list(result.report_metadata.get("source_snapshot_dates") or []),
            "rejected_sparse_same_day_ibkr_files": list(result.report_metadata.get("rejected_sparse_same_day_ibkr_files") or []),
            "used_full_quoted_ibkr_same_date": bool(result.report_metadata.get("used_full_quoted_ibkr_same_date")),
            "analysis_trust_level": result.report_metadata.get("analysis_trust_level"),
            "analysis_trust_note": result.report_metadata.get("analysis_trust_note"),
            "trusted_expiry_count": int(result.report_metadata.get("trusted_expiry_count") or 0),
            "fallback_only_expiry_count": int(result.report_metadata.get("fallback_only_expiry_count") or 0),
            "run_slug": result.run_slug,
            "default_strategy_family": result.report_metadata.get("default_strategy_family"),
            "default_candidate_within_family": result.report_metadata.get("default_candidate_within_family"),
            "default_contract_for_path_explorer": result.report_metadata.get("default_contract_for_path_explorer"),
            "decision_highlights": result.decision_highlights.head(12).to_dict(orient="records") if not result.decision_highlights.empty else [],
        },
    }
    _write_metadata(bundle_dir, file_map, "report_metadata.json", report_metadata)

    representative_buckets = (
        ", ".join(
            result.representative_paths_summary.get("representative_bucket", pd.Series(dtype=str))
            .astype(str)
            .drop_duplicates()
            .tolist()[:5]
        )
        or "n/a"
    )
    strike_note = _path_comparison_note(strike_overview)
    expiry_note = _path_comparison_note(expiry_overview)
    def _terminal_view_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame()
        return (
            frame.sort_values(["selection_rank", "candidate_slug", "requested_days", "step_index"])
            .groupby("candidate_slug", dropna=False, as_index=False)
            .tail(1)
            .sort_values(["selection_rank", "expiry_date", "strike_label"])
            .reset_index(drop=True)
        )

    long_call_strike_terminal = _terminal_view_rows(result.long_call_value_over_path_strike_view)
    long_call_expiry_terminal = _terminal_view_rows(result.long_call_value_over_path_expiry_view)
    long_call_best_of_terminal = _terminal_view_rows(result.long_call_value_over_path_best_of)
    long_call_strike_overview = long_call_strike_terminal.iloc[0].to_dict() if not long_call_strike_terminal.empty else {}
    long_call_expiry_overview = long_call_expiry_terminal.iloc[0].to_dict() if not long_call_expiry_terminal.empty else {}
    stock_gallery_labels = ", ".join(
        result.stock_path_gallery.sort_values(["display_order", "requested_days"])
        .drop_duplicates(subset=["path_name", "path_label"])
        .get("path_label", pd.Series(dtype=str))
        .astype(str)
        .tolist()[:7]
    ) or "n/a"
    iv_gallery_labels = ", ".join(
        result.iv_path_gallery.sort_values(["display_order", "requested_days"])
        .drop_duplicates(subset=["iv_path_name", "iv_path_label"])
        .get("iv_path_label", pd.Series(dtype=str))
        .astype(str)
        .tolist()[:7]
    ) or "n/a"
    long_call_best_of_labels = ", ".join(
        long_call_best_of_terminal.get("candidate_label", pd.Series(dtype=str)).astype(str).tolist()[:4]
    ) or "n/a"
    path_centric_focus_paths = result.report_metadata.get("path_centric_focus_paths") or []
    path_centric_rows = [
        (
            _humanize_label(path_meta.get("path_label") or path_meta.get("path_name")),
            ", ".join(
                filter(
                    None,
                    [
                        f"charts/{clean_string(path_meta.get('compare_chart'))}" if clean_string(path_meta.get("compare_chart")) else "",
                        f"charts/{clean_string(path_meta.get('strike_value_chart') or path_meta.get('strike_chart'))}" if clean_string(path_meta.get("strike_value_chart") or path_meta.get("strike_chart")) else "",
                        f"charts/{clean_string(path_meta.get('strike_delta_chart'))}" if clean_string(path_meta.get("strike_delta_chart")) else "",
                        f"charts/{clean_string(path_meta.get('expiry_value_chart') or path_meta.get('expiry_chart'))}" if clean_string(path_meta.get("expiry_value_chart") or path_meta.get("expiry_chart")) else "",
                        f"charts/{clean_string(path_meta.get('expiry_delta_chart'))}" if clean_string(path_meta.get("expiry_delta_chart")) else "",
                        f"charts/{clean_string(path_meta.get('best_of_value_chart') or path_meta.get('best_of_chart'))}" if clean_string(path_meta.get("best_of_value_chart") or path_meta.get("best_of_chart")) else "",
                        f"charts/{clean_string(path_meta.get('best_of_delta_chart'))}" if clean_string(path_meta.get("best_of_delta_chart")) else "",
                        f"tables/{clean_string(path_meta.get('checkpoint_table'))}" if clean_string(path_meta.get("checkpoint_table")) else "",
                        f"charts/{clean_string(path_meta.get('iv_value_chart'))}" if clean_string(path_meta.get("iv_value_chart")) else "",
                        f"charts/{clean_string(path_meta.get('iv_delta_chart'))}" if clean_string(path_meta.get("iv_delta_chart")) else "",
                        f"tables/{clean_string(path_meta.get('iv_checkpoint_table'))}" if clean_string(path_meta.get("iv_checkpoint_table")) else "",
                        f"charts/{clean_string(path_meta.get('strike_iv_value_chart'))}" if clean_string(path_meta.get("strike_iv_value_chart")) else "",
                        f"charts/{clean_string(path_meta.get('expiry_iv_value_chart'))}" if clean_string(path_meta.get("expiry_iv_value_chart")) else "",
                        f"charts/{clean_string(path_meta.get('best_of_iv_value_chart'))}" if clean_string(path_meta.get("best_of_iv_value_chart")) else "",
                        f"tables/{clean_string(path_meta.get('iv_robustness_summary_table'))}" if clean_string(path_meta.get("iv_robustness_summary_table")) else "",
                    ],
                )
            ),
        )
        for path_meta in path_centric_focus_paths
    ]
    long_call_expiry_fallback_note = (
        "Some expiry rows use fallback strike matching. Check strike_match_mode in the table to see whether the row used same-moneyness or nearest-numeric matching."
        if bool(long_call_expiry_terminal.get("used_strike_fallback", pd.Series(dtype=bool)).fillna(False).any())
        else "Rows keep the exact same strike concept across expiries."
    )
    highlight_table_rows: list[tuple[str, str]] = []
    if not result.decision_highlights.empty:
        for row in result.decision_highlights.head(6).to_dict("records"):
            category = _humanize_label(row.get("highlight_label") or row.get("highlight_category"))
            candidate = clean_string(row.get("selected_candidate_label")) or "no_clear_edge"
            status = clean_string(row.get("decision_status")) or "n/a"
            warning = clean_string(row.get("main_warning"))
            read = f"`{candidate}` - {status}"
            if warning:
                read += f". {warning}"
            highlight_table_rows.append((category, read))
    if not highlight_table_rows:
        highlight_table_rows = [("Decision Highlights", "No clear highlight edge was available under the current assumptions.")]
    action_table_rows: list[tuple[str, str]] = []
    if not result.action_board_candidates.empty:
        for bucket in ["Buy Now", "Watchlist", "Prefer Stock Instead", "Avoid For Now"]:
            subset = result.action_board_candidates.loc[result.action_board_candidates.get("action_bucket").astype(str).eq(bucket)].copy()
            if subset.empty:
                action_table_rows.append((bucket, "No candidates in this bucket under current assumptions."))
                continue
            lead = subset.sort_values("action_priority_rank").iloc[0]
            action_table_rows.append(
                (
                    bucket,
                    f"`{clean_string(lead.get('candidate_label'))}` - {clean_string(lead.get('headline_reason'))}",
                )
            )
    if not action_table_rows:
        action_table_rows = [("Action Board", "No action-board candidates were available under the current assumptions.")]
    summary_lines = [
        f"# {result.ticker} Contract Selection",
        "",
        (
            f"Path-first contract-selection bundle for {result.ticker} using local chain, spot, risk-free, and metadata inputs. "
            "Read the action board first, then the chain overview compare-options layer, then the entry-justification layer, and then the deeper single-option, trust, and path-centric views."
        ),
        "",
        "## Decision Snapshot",
        "",
        *_markdown_table(
            [
                ("Best Family", str(decision_snapshot.iloc[0]["best_family"])),
                ("Best Candidate", str(decision_snapshot.iloc[0]["best_candidate"])),
                ("Stock Benchmark", _humanize_benchmark_decision(decision_snapshot.iloc[0]["stock_benchmark_decision"])),
                ("Goal", result.goal.replace("_", " ")),
                ("Target Stock Price", _fmt_currency(result.target_price)),
                ("Target Date", result.target_date.isoformat()),
                ("Assumed Stock Path", result.stock_path_name.replace("_", " ")),
                ("Assumed IV Path", result.iv_path_name.replace("_", " ")),
            ]
        ),
        "",
        "## Action Board / Contract Picker",
        "",
        *_markdown_table(
            [
                ("Full Action Board", "summary/action_board.md"),
                ("Primary Bullish Board", "summary/bullish_action_board.md"),
                ("Secondary Structures", "summary/other_structures.md"),
                ("Bullish Long-Call Board", "tables/bullish_long_call_action_board.csv"),
                ("Bullish Watchlist", "tables/bullish_long_call_watchlist.csv"),
                ("Bullish Avoid", "tables/bullish_long_call_avoid.csv"),
                ("Bullish Triggers", "tables/bullish_long_call_triggers.csv"),
                ("Top Candidate Cards", "summary/top_candidate_cards.md"),
                ("Top Candidate Cards Table", "tables/top_candidate_cards.csv"),
                ("Top Candidate Cards Chart", "charts/top_candidate_cards.png"),
                ("Other Structures Table", "tables/other_structures_summary.csv"),
                ("Stock Preference Table", "tables/stock_preference_summary.csv"),
                ("All Action Candidates", "tables/action_board_candidates.csv"),
                ("Decision Triggers", "tables/decision_triggers.csv"),
                *action_table_rows,
            ]
        ),
        "",
        "## Chain Overview / Compare Options",
        "",
        *_markdown_table(
            [
                ("Chain Overview Summary", "summary/chain_overview.md"),
                ("Compare Options Chart", "charts/chain_overview.png"),
                ("Compare Options Cards", "tables/chain_overview_summary.csv"),
                ("Compare Options Table", "tables/chain_overview_candidates.csv"),
                (
                    "How To Read It",
                    "This layer compares bullish long calls against long stock across the same representative path families, then adds timing, IV, and entry sensitivity so you can quickly see which calls are robust, selective, too narrow, or simply worse than stock.",
                ),
            ]
        ),
        "",
        "## Entry Justification / Required Stock Path",
        "",
        *_markdown_table(
            [
                ("Entry Justification Summary", "summary/entry_justification.md"),
                ("Candidate Entry Table", "tables/entry_justification_candidates.csv"),
                ("Required Stock Path Table", "tables/required_stock_path_to_buy.csv"),
                ("Required Move Summary", "tables/required_move_summary.csv"),
                ("Required Move Vs Stock", "tables/required_move_vs_stock.csv"),
                ("IV Support Summary", "tables/required_iv_support_summary.csv"),
                ("Entry Barrier Summary", "tables/entry_barrier_summary.csv"),
                ("Required Stock Path Chart", "charts/required_stock_path_to_buy.png"),
                ("Required Move Speed Vs Magnitude", "charts/required_move_speed_vs_magnitude.png"),
                ("Required Move Vs Stock Chart", "charts/required_move_vs_stock_chart.png"),
                ("Strike / Expiry Barrier Map", "charts/strike_expiry_entry_barrier_map.png"),
                ("IV Support Requirement", "charts/iv_support_requirement_chart.png"),
            ]
        ),
        "",
        "## Thesis / Price Target Mode",
        "",
        *_markdown_table(
            [
                ("Thesis Summary", "summary/thesis_mode.md"),
                ("Thesis Target", f"{_fmt_currency(result.thesis_target_price)} by {result.thesis_target_date.isoformat()}"),
                ("Thesis Candidate Overview", "charts/thesis_candidate_overview.png"),
                ("Current Vs Justified Premium", "charts/current_vs_justified_premium.png"),
                ("Thesis Path Gallery", "charts/thesis_path_gallery.png"),
                ("Thesis IV Gallery", "charts/thesis_iv_gallery.png"),
                ("Path Vs Value", "charts/thesis_path_vs_value.png"),
                ("IV Vs Value", "charts/thesis_iv_vs_value.png"),
                ("Stock Vs Option Under Thesis", "charts/thesis_stock_vs_option.png"),
                ("Candidate Ranking", "tables/thesis_candidate_ranking.csv"),
                ("Current Vs Justified Table", "tables/current_vs_justified_premium.csv"),
                ("Max Justified Premium", "tables/max_justified_premium.csv"),
                ("Required Move Summary", "tables/thesis_required_move_summary.csv"),
                ("Stock Vs Option Summary", "tables/thesis_stock_vs_option_summary.csv"),
            ]
        ),
        "",
        "## Practical Stress Tests",
        "",
        *_markdown_table(
            [
                ("Stress Tests Summary", "summary/stress_tests.md"),
                ("Stress Grid", "tables/candidate_stress_grid.csv"),
                ("Premium Sensitivity", "tables/premium_sensitivity_summary.csv"),
                ("Timing Slip Summary", "tables/timing_slip_summary.csv"),
                ("Target Stress Summary", "tables/target_stress_summary.csv"),
                ("Stress Transition Summary", "tables/stress_transition_summary.csv"),
                ("Stress Overview Chart", "charts/stress_test_overview.png"),
                ("Premium Sensitivity Chart", "charts/premium_sensitivity_chart.png"),
                ("Timing Slip Chart", "charts/timing_slip_chart.png"),
                ("Target Stress Chart", "charts/target_stress_chart.png"),
                ("Top Candidate Stress Cards", "charts/top_candidate_stress_cards.png"),
            ]
        ),
        "",
        "## Single-Option Decision View",
        "",
        *_markdown_table(
            [
                ("Decision Summary", "summary/single_option_decision.md"),
                ("Composite View", "charts/single_option_decision_view.png"),
                ("Selected Option Summary", "tables/single_option_decision_summary.csv"),
                ("Curated Decision Path Selection", "tables/single_option_decision_path_selections.csv"),
                ("Representative Stock Paths", "tables/single_option_representative_paths.csv"),
                ("Path Outcomes Vs Stock", "tables/single_option_path_outcomes.csv"),
                ("Path Family Counts", "tables/single_option_path_family_counts.csv"),
                ("Timing Sensitivity", "tables/single_option_timing_sensitivity.csv"),
                ("IV Sensitivity", "tables/single_option_iv_sensitivity.csv"),
                ("Entry Premium Sensitivity", "tables/single_option_entry_sensitivity.csv"),
                ("Summary Bullets", "tables/single_option_summary_bullets.csv"),
                (
                    "How To Read It",
                    "This view asks what stock paths make one selected call worth buying instead of buying stock. The hero chart is stock-path-only; IV and entry price are isolated below.",
                ),
            ]
        ),
        "",
        "## Decision Highlights",
        "",
        *_markdown_table(
            [
                ("Full Highlights Summary", "summary/highlights.md"),
                ("Highlights Table", "tables/decision_highlights.csv"),
                ("Robustness Table", "tables/candidate_robustness_summary.csv"),
                ("Tradeoff Matrix", "tables/candidate_tradeoff_matrix.csv"),
                *highlight_table_rows,
            ]
        ),
        "",
        "## Market Context / Trust Summary",
        "",
        *_markdown_table(
            [
                ("Analysis Trust Level", str(result.report_metadata.get("analysis_trust_level") or "n/a")),
                ("Trust Note", str(result.report_metadata.get("analysis_trust_note") or "n/a")),
                ("Trusted Expiries", str(result.report_metadata.get("trusted_expiry_count") or 0)),
                ("Sparse/Fallback Expiries", str(result.report_metadata.get("fallback_only_expiry_count") or 0)),
                ("Spot Source", str(result.report_metadata.get("spot_price_source") or "n/a")),
                ("Spot Field", str(result.report_metadata.get("spot_field_used") or "n/a")),
                ("Spot Matched Date", str(result.report_metadata.get("spot_price_matched_date") or "n/a")),
                ("Same-Day IBKR Spot Rejection", str(result.report_metadata.get("ibkr_same_day_spot_rejected_reason") or "n/a")),
                ("Risk-Free Source", str(result.report_metadata.get("risk_free_rate_source") or "n/a")),
                ("Risk-Free Series", str(result.report_metadata.get("risk_free_rate_series") or "n/a")),
                ("Risk-Free Matched Date", str(result.report_metadata.get("risk_free_rate_matched_date") or "n/a")),
                ("Per-Expiry Trust Table", "tables/chain_source_summary.csv"),
                ("Bundle Market Context Table", "tables/market_context_summary.csv"),
            ]
        ),
        "",
        "## Stock Path Gallery",
        "",
        *_markdown_table(
            [
                ("Named Gallery Paths", stock_gallery_labels),
                ("Active Assumed Path", _humanize_label(result.stock_path_name)),
                ("Primary Chart", "charts/stock_path_gallery.png"),
                ("Path Library Metadata", "tables/stock_path_library.csv"),
                ("Supporting Table", "tables/stock_path_gallery.csv"),
                (
                    "How To Read It",
                    "These are deliberate scenario templates, not simulation picks. This is the broad scenario library; single-option pages use a smaller curated decision-path subset.",
                ),
            ]
        ),
        "",
        "## IV Path Gallery",
        "",
        *_markdown_table(
            [
                ("Named IV Regimes", iv_gallery_labels),
                ("Active Assumed IV Path", _humanize_label(result.iv_path_name)),
                ("Primary Chart", "charts/iv_path_gallery.png"),
                ("Supporting Table", "tables/iv_path_gallery.csv"),
                (
                    "How To Read It",
                    "Use this to decide whether the thesis assumes flat IV, mean reversion, IV drag, or event-style build/crush. Keep IV thinking separate from stock-path thinking.",
                ),
            ]
        ),
        "",
        "## Required vs Assumed Path Summary",
        "",
        *_markdown_table(
            [
                ("Required Path Read", str(decision_snapshot.iloc[0]["required_path_difficulty"] or "n/a")),
                ("First Cleared Horizon", str(decision_snapshot.iloc[0]["first_cleared_horizon"] or "not cleared")),
                ("Gap At Target", _fmt_currency(decision_snapshot.iloc[0]["required_path_gap_at_target"])),
                ("Required Path Table", "tables/required_path_summary.csv"),
                ("Required vs Assumed Table", "tables/required_vs_assumed_path_summary.csv"),
                ("How To Read It", "If the assumed path stays above a required-path line, that family clears the goal more easily at that checkpoint."),
            ]
        ),
        "",
        "## Path-Centric Long-Call Scenario Library",
        "",
        *_markdown_table(
            [
                (
                    "How To Use It",
                    "Pick one named stock path first, then read the same-path compare-vs-stock chart, strike ladder, expiry ladder, best-of chart, and checkpoint table for that path.",
                ),
                (
                    "IV Dimension",
                    "Then keep that stock path fixed and read the path-prefixed IV value/delta charts to see whether IV saves, hurts, or crushes the long-call read.",
                ),
                *path_centric_rows,
            ]
        ),
        "",
        "## Same-Path Compare vs Stock",
        "",
        *_markdown_table(
            [
                ("Primary Assumed-Path Chart", "charts/compare_vs_stock_path_delta.png"),
                ("Primary Assumed-Path Table", "tables/compare_vs_stock_path_rows.csv"),
                ("Stock Benchmark Note", str(decision_snapshot.iloc[0]["stock_benchmark_note"] or "n/a")),
                ("Benchmark Edge Vs Stock", _fmt_currency(decision_snapshot.iloc[0]["benchmark_edge"])),
                ("Benchmark Return Edge Vs Stock", _fmt_percent(decision_snapshot.iloc[0]["benchmark_return_edge"])),
                ("Timing Risk", str(decision_snapshot.iloc[0]["timing_risk"] or "n/a")),
                ("IV Risk", str(decision_snapshot.iloc[0]["iv_risk"] or "n/a")),
                (
                    "How To Read It",
                    "This is the primary stock-vs-structure benchmark under the active assumed future. Use the representative compare-vs-stock chart only as secondary context after you understand this one.",
                ),
            ]
        ),
        "",
        "## Long-Call Strike Comparison",
        "",
        *_markdown_table(
            [
                ("Anchor Expiry", str(long_call_strike_overview.get("anchor_expiry_date") or long_call_strike_overview.get("expiry_date") or "n/a")),
                ("Top Strike Read", str(long_call_strike_overview.get("strike_label") or "n/a")),
                ("Strike Moneyness", str(long_call_strike_overview.get("moneyness_bucket") or "n/a")),
                ("Strike Trust", str(long_call_strike_overview.get("source_trust_label") or "n/a")),
                ("Why This View Matters", "Same expiry, same stock path, same IV path. Only strike changes."),
                ("Long-Call Strike Table", "tables/long_call_value_over_path_strike_view.csv"),
                ("Long-Call Strike Chart", "charts/long_call_value_over_path_strike_view.png"),
            ]
        ),
        "",
        "## Long-Call Expiry Comparison",
        "",
        *_markdown_table(
            [
                ("Anchor Strike", str(long_call_expiry_overview.get("anchor_strike_label") or long_call_expiry_overview.get("strike_label") or "n/a")),
                ("Top Expiry Read", str(long_call_expiry_overview.get("expiry_date") or "n/a")),
                ("Expiry Trust", str(long_call_expiry_overview.get("source_trust_label") or "n/a")),
                ("Strike Matching Note", long_call_expiry_fallback_note),
                ("Strike Match Audit", "tables/long_call_value_over_path_expiry_view.csv"),
                ("Why This View Matters", "Same strike concept, same stock path, same IV path. Only expiry changes."),
                ("Long-Call Expiry Table", "tables/long_call_value_over_path_expiry_view.csv"),
                ("Long-Call Expiry Chart", "charts/long_call_value_over_path_expiry_view.png"),
            ]
        ),
        "",
        "## Best-Of Long-Call Comparison",
        "",
        *_markdown_table(
            [
                ("Best-Of Candidate Count", str(int(long_call_best_of_terminal["candidate_slug"].nunique()) if not long_call_best_of_terminal.empty else 0)),
                ("Best-Of Leaders", long_call_best_of_labels),
                ("Selection Logic", "Top long calls by active objective, but with expiry and moneyness diversity constraints."),
                ("Why This View Matters", "Use this when you want the clearest subset of long-call choices under one assumed future."),
                ("Best-Of Table", "tables/long_call_value_over_path_best_of.csv"),
                ("Best-Of Chart", "charts/long_call_value_over_path_best_of.png"),
            ]
        ),
        "",
        "## Representative Path Summary",
        "",
        *_markdown_table(
            [
                (
                    "Selection Mode",
                    f"{result.representative_selection_mode.replace('_', ' ')} with {result.simulated_path_count} generated stock-path examples",
                ),
                ("Representative Buckets", representative_buckets),
                ("Representative Stock Paths", "charts/representative_stock_paths.png"),
                ("Representative IV Paths", "charts/representative_iv_paths.png"),
                ("Representative Path-Pair Table", "tables/path_pair_summary.csv"),
                ("Representative Compare vs Stock", "charts/compare_vs_stock_over_path.png"),
                (
                    "How To Read It",
                    "These are secondary heuristic examples showing how noisy futures can miss, almost work, or work. Use them after the named galleries and the active assumed-path comparisons.",
                ),
            ]
        ),
        "",
        "## Family Selection And Exact Contract Choice",
        "",
        *_markdown_table(
            [
                ("Family Edge Status", str(decision_snapshot.iloc[0]["family_edge_status"] or "informative")),
                ("Why The Family Leads", str(family_overview.get("why_this_wins") or "n/a")),
                ("Why It Can Still Lose", str(family_overview.get("why_this_loses") or "n/a")),
                ("Exact Candidate", str(top_candidate.get("candidate_label") or "n/a")),
                ("Why The Candidate Leads", str(top_candidate.get("why_this_candidate_wins") or "n/a")),
                ("Why It Can Still Lose", str(top_candidate.get("why_this_candidate_loses") or "n/a")),
                ("Broader Strike Table", "tables/strike_comparison_under_path.csv"),
                ("Broader Expiry Table", "tables/expiry_comparison_under_path.csv"),
                ("Representative Value Table", "tables/option_value_over_path.csv"),
            ]
        ),
        "",
        "## Risk Notes",
        "",
        *_markdown_table(
            [
                ("Top Path Risk", str(decision_snapshot.iloc[0]["top_path_risk"] or "n/a")),
                ("Primary Warning", str(decision_snapshot.iloc[0]["primary_warning"] or "n/a")),
                ("Spot Quality Note", str(result.report_metadata.get("spot_quality_note") or "n/a")),
                ("Risk-Free Note", str(result.report_metadata.get("risk_free_rate_note") or "n/a")),
            ]
        ),
    ]
    if result.warnings:
        summary_lines.extend(["", "Additional Warnings:"])
        summary_lines.extend(f"- {warning}" for warning in result.warnings)
    summary_lines.extend(
        [
            "",
            "## How To Read The Core Charts",
            "",
            *_markdown_table(
                [
                    ("`summary/entry_justification.md`", "Read this right after the action board when you want the blunt answer to what the stock actually has to do before a call looks worth buying."),
                    ("`summary/thesis_mode.md`", "Use this when you want to ask: if the stock reaches a specific price by a specific date, which calls become reasonable and what premium is justified?"),
                    ("`thesis_candidate_overview.png`", "Compact thesis-first contract picker showing current premium, justified premium, stock benchmark pressure, and trust."),
                    ("`current_vs_justified_premium.png`", "Green bar = thesis-justified max premium. Grey bar = current premium. If grey is longer, the thesis still asks for a better entry."),
                    ("`thesis_path_gallery.png`", "Same endpoint, different routes. Use this to see why a slow target path can be worse for calls than a fast target path."),
                    ("`thesis_stock_vs_option.png`", "Shows whether calls beat stock under the thesis scenarios, not just whether the stock target is reached."),
                    ("`required_stock_path_to_buy.png`", "Colored line = minimum stock path needed to justify the call. Dashed black = the active assumed path. Use this to judge whether the call asks too much of the stock."),
                    ("`required_move_speed_vs_magnitude.png`", "Lower-left is easier: slower required climb and smaller total upside. Upper-right is demanding: faster pace plus a larger move."),
                    ("`required_move_vs_stock_chart.png`", "Negative bars mean stock still looks cleaner even if the call's required path is broadly met."),
                    ("`strike_expiry_entry_barrier_map.png`", "Compact table visual showing which strike/expiry combinations ask the least or most from the stock path."),
                    ("`iv_support_requirement_chart.png`", "Orange to the right means lower IV makes the call materially harder to justify; blue to the left means friendlier IV helps."),
                    ("`stock_path_gallery.png`", "Start by choosing the stock-path shape you actually want to think with. This is the broad scenario library, not the single-option decision chart."),
                    ("`stock_path_library.csv`", "Stable path-family metadata for the scenario library, including timing shape and outcome bias labels."),
                    ("`single_option_decision_path_selections.csv`", "The 5-8 curated paths selected for the specific option, with outcome label, family label, score, and selection reason."),
                    ("`iv_path_gallery.png`", "Choose the IV regime separately. This is the main IV decision surface, not the representative-IV chart."),
                    ("`required_path_vs_assumed_path.png`", "Start here. The assumed path should clear the required path if the thesis is truly enough."),
                    ("`required_path_strategy_compare.png`", "Compare family-level required paths directly. Lower required paths are easier to satisfy."),
                    ("`<path>__compare_vs_stock_path_delta.png`", "After choosing a named stock path, use its path-prefixed compare-vs-stock chart as the main stock benchmark for that exact future."),
                    ("`<path>__long_call_strike_value.png`", "Value view. Same named stock path, same IV path, same expiry. This isolates strike choice under one explicit future."),
                    ("`<path>__long_call_strike_delta.png`", "Delta-vs-stock view for the same strike ladder. Use it after value to see whether the option is actually beating stock."),
                    ("`<path>__long_call_expiry_value.png`", "Value view. Same named stock path, same IV path, same strike concept. Use it to see when more time helps or still fails."),
                    ("`<path>__long_call_expiry_delta.png`", "Delta-vs-stock view for the same expiry ladder. Use it to separate resilience from stock benchmark edge."),
                    ("`<path>__long_call_best_of_value.png`", "Value view for the curated long-call shortlist under one named stock path."),
                    ("`<path>__long_call_best_of_delta.png`", "Delta-vs-stock view for the same best-of shortlist."),
                    ("`<path>__iv_path_value.png`", "IV-path value view. Same stock path and same anchor long call; only the IV regime changes."),
                    ("`<path>__iv_path_delta.png`", "IV-path delta-vs-stock view. Use it to see whether IV help/crush changes stock-relative edge under the same stock path."),
                    ("`<path>__iv_checkpoints.csv`", "Compact IV checkpoint table explaining whether each IV regime helped, hurt, or left stock cleaner."),
                    ("`<path>__long_call_strike_iv_value.png`", "IV-expanded strike ladder. Same stock path and strike-view contract set; IV regime becomes an explicit comparison axis."),
                    ("`<path>__long_call_strike_iv_delta.png`", "Stock-relative version of the IV-expanded strike ladder."),
                    ("`<path>__long_call_expiry_iv_value.png`", "IV-expanded expiry ladder. Use it to see which expiries survive lower IV or depend on IV support."),
                    ("`<path>__long_call_expiry_iv_delta.png`", "Stock-relative version of the IV-expanded expiry ladder."),
                    ("`<path>__long_call_best_of_iv_value.png`", "IV-expanded best-of view for the curated long-call set."),
                    ("`<path>__long_call_best_of_iv_delta.png`", "Stock-relative version of the IV-expanded best-of view."),
                    ("`<path>__iv_robustness_summary.csv`", "Decision table summarizing which contracts survive lower IV, need IV support, or remain stock-dominated."),
                    ("`compare_vs_stock_path_delta.png`", "This is the primary same-future stock benchmark chart. Above zero means the structure is ahead of stock under the active assumed path."),
                    ("`long_call_value_over_path_strike_view.png`", "Same assumed path, same IV path, same expiry. Use this to isolate strike choice without chart clutter."),
                    ("`long_call_value_over_path_expiry_view.png`", "Same assumed path, same IV path, same strike concept. Use this to see when more time helps or just adds cost."),
                    ("`long_call_value_over_path_best_of.png`", "Curated best-of long calls only. Use this when you want the strongest long-call shortlist under one assumed future."),
                    ("`representative_stock_paths.png`", "Read these only after the named gallery paths. They are selected heuristic examples, not the first scenario-thinking surface."),
                    ("`representative_iv_paths.png`", "Secondary heuristic IV examples that complement, but do not replace, the named IV gallery."),
                    ("`option_value_over_path.png`", "Representative-path option-value comparison. Use it after the assumed-path and long-call views when you want supporting scenario context."),
                    ("`compare_vs_stock_over_path.png`", "Representative-path stock benchmark. Use it as secondary context after the active assumed-path compare-vs-stock chart."),
                    ("`strike_comparison_under_same_path.png`", "Use this when you want the same stock path and IV path, but different strikes."),
                    ("`expiry_comparison_under_same_path.png`", "Use this to see whether extra time rescued the thesis or just added cost."),
                    ("`assumed_path_value_progression.png`", "Read modeled PnL over time for the active assumed path."),
                    ("`iv_path_trace.png`", "The bold line is the active IV assumption; use it when you want the active path plus comparison traces rather than the clean named IV gallery."),
                ]
            ),
        ]
    )
    _write_markdown(bundle_dir, file_map, "summary.md", "\n".join(summary_lines))
    if clean_string(result.highlights_markdown):
        _write_markdown(bundle_dir, file_map, "highlights.md", result.highlights_markdown)
    if clean_string(result.action_board_markdown):
        _write_markdown(bundle_dir, file_map, "action_board.md", result.action_board_markdown)
    if clean_string(result.bullish_action_board_markdown):
        _write_markdown(bundle_dir, file_map, "bullish_action_board.md", result.bullish_action_board_markdown)
    if clean_string(result.other_structures_markdown):
        _write_markdown(bundle_dir, file_map, "other_structures.md", result.other_structures_markdown)
    if clean_string(result.entry_justification_markdown):
        _write_markdown(bundle_dir, file_map, "entry_justification.md", result.entry_justification_markdown)
    if clean_string(result.thesis_mode_markdown):
        _write_markdown(bundle_dir, file_map, "thesis_mode.md", result.thesis_mode_markdown)
    if clean_string(result.stress_tests_markdown):
        _write_markdown(bundle_dir, file_map, "stress_tests.md", result.stress_tests_markdown)
    if clean_string(result.top_candidate_cards_markdown):
        _write_markdown(bundle_dir, file_map, "top_candidate_cards.md", result.top_candidate_cards_markdown)
    if clean_string(result.chain_overview_markdown):
        _write_markdown(bundle_dir, file_map, "chain_overview.md", result.chain_overview_markdown)
    if clean_string(result.single_option_decision_markdown):
        _write_markdown(bundle_dir, file_map, "single_option_decision.md", result.single_option_decision_markdown)
    return report_metadata


def _write_replay_bundle(
    result: HistoricalReplayComputation,
    bundle_dir: Path,
    file_map: dict[str, dict[str, str]],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    _write_frame(bundle_dir, file_map, "summary.csv", result.case_summary)
    _write_frame(bundle_dir, file_map, "case_summary.csv", result.case_summary)
    for filename, frame in {
        "checkpoint_replay.csv": result.checkpoint_replay,
        "expected_move_vs_actual.csv": result.expected_move_vs_actual,
        "driver_decomposition.csv": result.driver_decomposition,
        "compare_vs_stock.csv": result.compare_vs_stock,
    }.items():
        _write_frame(bundle_dir, file_map, filename, frame)

    anchor_label = clean_string(result.case_summary.iloc[0].get("anchor_checkpoint")) if not result.case_summary.empty else ""
    anchor_label = anchor_label or "expiry"

    if not result.checkpoint_replay.empty:
        expected_move_pct = None
        if not result.expected_move_vs_actual.empty:
            expected_move_pct = float(result.expected_move_vs_actual.iloc[0].get("expected_move_pct_at_entry") or 0.0)
        _track_chart(
            bundle_dir,
            file_map,
            plot_replay_stock_path(
                result.checkpoint_replay,
                entry_spot=result.spot_price,
                expected_move_pct=expected_move_pct,
                output_path=_section_path(bundle_dir, "charts", "stock_path_expected_move.png"),
                title=f"{result.ticker} Stock Path Vs Entry Expected Move",
            ),
        )
        _track_chart(
            bundle_dir,
            file_map,
            plot_replay_strategy_value_path(
                result.checkpoint_replay,
                output_path=_section_path(bundle_dir, "charts", "strategy_value_path.png"),
                title=f"{result.ticker} {result.strategy_name.replace('_', ' ').title()} Replay Value Path",
            ),
        )

    if not result.compare_vs_stock.empty:
        for mode, filename, title in [
            ("equal_capital", "strategy_vs_stock_equal_capital.png", f"{result.ticker} {result.strategy_name.replace('_', ' ').title()} Vs Long Stock (${result.comparison_capital:,.0f} normalized)"),
            ("share_equivalent", "strategy_vs_stock_share_equivalent.png", f"{result.ticker} {result.strategy_name.replace('_', ' ').title()} Vs Long Stock (Share-equivalent)"),
        ]:
            subset = result.compare_vs_stock.loc[result.compare_vs_stock["mode"] == mode]
            if subset.empty:
                continue
            _track_chart(
                bundle_dir,
                file_map,
                plot_replay_compare_vs_stock(
                    result.compare_vs_stock,
                    mode=mode,
                    output_path=_section_path(bundle_dir, "charts", filename),
                    title=title,
                ),
            )

    if not result.driver_decomposition.empty and anchor_label:
        anchor_rows = result.driver_decomposition.loc[result.driver_decomposition["checkpoint"] == anchor_label]
        if not anchor_rows.empty:
            _track_chart(
                bundle_dir,
                file_map,
                plot_replay_driver_decomposition(
                    result.driver_decomposition,
                    checkpoint_label=anchor_label,
                    output_path=_section_path(bundle_dir, "charts", f"driver_decomposition_{slugify(anchor_label)}.png"),
                    title=f"{result.ticker} Replay Driver Decomposition ({anchor_label.replace('_', ' ').title()})",
                ),
            )

    metadata = {
        "generated_at": _utc_now_iso(),
        "report_kind": "replay",
        "analysis_name": "replay",
        "status": result.status,
        "shareability_status": result.shareability_status,
        "ticker": result.ticker,
        "snapshot_date": result.snapshot_date.isoformat(),
        "expiry_date": result.expiry_date.isoformat(),
        "strategy_name": result.strategy_name,
        "comparison_capital": result.comparison_capital,
        "valuation_source_rollup": result.valuation_source_rollup,
        "available_checkpoints": list(result.available_checkpoints),
        "what_this_case_shows": result.what_this_case_shows,
        "warnings": list(result.warnings),
        "generated_files": sorted(path for bucket in file_map.values() for path in bucket.values()),
        "metadata": {
            "ticker": result.ticker,
            "snapshot_date": result.snapshot_date.isoformat(),
            "expiry_date": result.expiry_date.isoformat(),
            "strategy_name": result.strategy_name,
            "comparison_capital": result.comparison_capital,
            "premium_mode": result.premium_mode,
            "source_snapshot_file": _project_relative_path(result.source_snapshot_file),
            "spot_price": result.spot_price,
            "spot_source": result.resolved_metadata.get("spot_price_source"),
            "spot_matched_date": result.resolved_metadata.get("spot_price_matched_date"),
            "spot_note": result.resolved_metadata.get("spot_price_note"),
            "risk_free_rate": result.risk_free_rate,
            "risk_free_source": result.resolved_metadata.get("risk_free_rate_source"),
            "risk_free_series": result.resolved_metadata.get("risk_free_rate_series"),
            "risk_free_matched_date": result.resolved_metadata.get("risk_free_rate_matched_date"),
            "risk_free_note": result.resolved_metadata.get("risk_free_rate_note"),
            "dividend_yield": result.dividend_yield,
            "valuation_source_rollup": result.valuation_source_rollup,
            "replay_defaults": result.replay_defaults,
            "research_context": result.research_context,
        },
        "replay": {
            "ticker": result.ticker,
            "snapshot_date": result.snapshot_date.isoformat(),
            "expiry_date": result.expiry_date.isoformat(),
            "strategy_name": result.strategy_name,
            "comparison_capital": result.comparison_capital,
            "available_checkpoints": list(result.available_checkpoints),
            "valuation_source_rollup": result.valuation_source_rollup,
            "what_this_case_shows": result.what_this_case_shows,
            "replay_defaults": result.replay_defaults,
            "entry_spot": result.spot_price,
            "risk_free_rate": result.risk_free_rate,
        },
        "research_context": result.research_context,
    }
    _write_metadata(bundle_dir, file_map, "report_metadata.json", metadata)

    local_history = collect_local_replay_history(
        output_root,
        ticker=result.ticker,
        strategy_name=result.strategy_name,
    )
    if not local_history.empty:
        _write_frame(bundle_dir, file_map, "local_history.csv", local_history)

    summary_lines = [
        f"# {result.ticker} {result.strategy_name.replace('_', ' ').title()} Historical Replay",
        "",
        "## Honesty Note",
        "",
        "- This is a local historical replay / case-study page, not a full options-tape backtest.",
        "- Exact later chain values are used only when the same expiry exists in a later local snapshot.",
        "- Otherwise the replay falls back to modeled estimates and labels that path explicitly.",
        "",
        "## Case Summary",
        "",
        f"- status: {result.status}",
        f"- comparison_capital: {result.comparison_capital}",
        f"- valuation_source_rollup: {result.valuation_source_rollup}",
        f"- what_this_case_shows: {result.what_this_case_shows}",
    ]
    _write_markdown(bundle_dir, file_map, "summary.md", "\n".join(summary_lines))
    return metadata


def write_analysis_bundle(
    result: Any,
    *,
    analysis_kind: str,
    output_root: str | Path = DEFAULT_ANALYSIS_OUTPUT_ROOT,
) -> AnalysisBundle:
    """Write one canonical analysis bundle without rendering HTML."""

    ticker, snapshot_date, run_slug = _analysis_identity(result, analysis_kind)
    bundle_dir = _bundle_dir(ticker, snapshot_date, analysis_kind, run_slug, output_root=output_root)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    for section in ["tables", "charts", "summary", "metadata"]:
        ensure_directory(bundle_dir / section)
    file_map: dict[str, dict[str, str]] = {"tables": {}, "charts": {}, "summary": {}, "metadata": {}}

    if analysis_kind == "strategy":
        report_metadata = _write_strategy_bundle(result, bundle_dir, file_map)
    elif analysis_kind == "scenario":
        report_metadata = _write_scenario_bundle(result, bundle_dir, file_map)
    elif analysis_kind == "contract_selection":
        report_metadata = _write_contract_selection_bundle(result, bundle_dir, file_map)
    elif analysis_kind == "replay":
        report_metadata = _write_replay_bundle(result, bundle_dir, file_map, output_root=output_root)
    else:
        raise ValueError(f"Unsupported analysis bundle kind: {analysis_kind}")

    manifest = {
        "analysis_kind": analysis_kind,
        "bundle_version": BUNDLE_VERSION,
        "ticker": ticker,
        "snapshot_date": snapshot_date,
        "run_slug": run_slug,
        "created_at": report_metadata.get("generated_at"),
        "assumptions": _bundle_assumptions(report_metadata, analysis_kind),
        "warnings": unique_warnings(report_metadata.get("warnings") or []),
        "coverage": coverage_payload(
            report_metadata.get("status"),
            report_metadata.get("shareability_status"),
            report_kind=report_metadata.get("report_kind"),
        ),
        "file_map": file_map,
        "source_references": _bundle_sources(report_metadata),
    }
    manifest_path = write_json(manifest, bundle_dir / "bundle_manifest.json")
    return AnalysisBundle(
        analysis_kind=analysis_kind,
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        ticker=ticker,
        snapshot_date=snapshot_date,
        run_slug=run_slug,
        file_map=file_map,
    )


def resolve_analysis_bundle(
    *,
    bundle: str | Path | None = None,
    ticker: str | None = None,
    snapshot_date: str | None = None,
    analysis_kind: str | None = None,
    run_slug: str | None = None,
    output_root: str | Path = DEFAULT_ANALYSIS_OUTPUT_ROOT,
) -> Path:
    """Resolve a canonical analysis bundle path from a direct path or coordinates."""

    if bundle is not None:
        path = Path(bundle)
        if not path.exists():
            raise FileNotFoundError(f"Analysis bundle not found: {path}")
        return path
    if not (clean_string(ticker) and clean_string(snapshot_date) and clean_string(analysis_kind)):
        raise ValueError("publish-analysis requires either --bundle or ticker/snapshot/analysis-kind coordinates.")
    kind_root = Path(output_root) / clean_string(ticker).upper() / f"snapshot_{snapshot_date}" / clean_string(analysis_kind)
    if not kind_root.exists():
        raise FileNotFoundError(f"Analysis kind folder not found: {kind_root}")
    if clean_string(run_slug):
        path = kind_root / clean_string(run_slug)
        if not path.exists():
            raise FileNotFoundError(f"Analysis bundle not found: {path}")
        return path
    candidates = sorted([path for path in kind_root.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No analysis bundles were found under: {kind_root}")
    return candidates[-1]


def _sanitize_for_published(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_for_published(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_published(item) for item in value]
    text = clean_string(value)
    if text and (":\\" in text or text.startswith("C:/") or text.startswith("C:\\")):
        return _project_relative_path(text)
    return value


PUBLISHED_TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt"}
ABSOLUTE_WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:(?:\\\\|\\/|\\|/)[^\s<>\",|)]+")


def _sanitize_published_text(text: str) -> str:
    def replace_match(match: re.Match[str]) -> str:
        path_text = match.group(0).replace("\\\\", "\\")
        return _project_relative_path(path_text) or match.group(0)

    return ABSOLUTE_WINDOWS_PATH_PATTERN.sub(replace_match, text)


def _copy_publish_artifact(source: Path, target: Path) -> None:
    ensure_directory(target.parent)
    if source.suffix.lower() in PUBLISHED_TEXT_SUFFIXES:
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(source, target)
            return
        target.write_text(_sanitize_published_text(text), encoding="utf-8")
        return
    shutil.copy2(source, target)


def _flatten_bundle_files(bundle_dir: Path, file_map: dict[str, dict[str, str]], publish_dir: Path) -> None:
    for section in file_map.values():
        for _, relative_path in section.items():
            source = bundle_dir / relative_path
            if not source.exists() or source.is_dir():
                continue
            _copy_publish_artifact(source, publish_dir / source.name)


def _latest_compatible_contract_bundle(bundle_dir: Path) -> Path | None:
    contract_root = bundle_dir.parent.parent / "contract_selection"
    if not contract_root.exists():
        return None
    required = {
        "tables/candidate_summary.csv",
        "tables/required_path_rows.csv",
        "tables/strategy_selector_rows.csv",
        "tables/strategy_selector_rankings.csv",
        "tables/path_case_chart_rows.csv",
        "tables/path_case_strategy_rows.csv",
    }
    candidates: list[Path] = []
    for candidate in sorted([path for path in contract_root.iterdir() if path.is_dir()]):
        if all((candidate / rel_path).exists() for rel_path in required):
            candidates.append(candidate)
    return candidates[-1] if candidates else None


def _related_scenario_bundle(bundle_dir: Path) -> Path | None:
    scenario_root = bundle_dir.parent.parent / "scenario"
    if not scenario_root.exists():
        return None
    candidates = sorted([path for path in scenario_root.iterdir() if path.is_dir()])
    return candidates[-1] if candidates else None


def _prepare_related_contract_selection(bundle_dir: Path, publish_dir: Path) -> dict[str, Any] | None:
    related_bundle = _latest_compatible_contract_bundle(bundle_dir)
    if related_bundle is None:
        return None
    related_manifest = _load_json(related_bundle / "bundle_manifest.json")
    target_dir = publish_dir / "_rel" / clean_string(related_manifest.get("run_slug") or related_bundle.name)
    ensure_directory(target_dir)
    copied_files: dict[str, list[str]] = {
        "tables": [],
        "metadata": [],
        "summary": [],
    }
    for relative_path in [
        "tables/candidate_summary.csv",
        "tables/required_path_rows.csv",
        "tables/strategy_selector_rows.csv",
        "tables/strategy_selector_rankings.csv",
        "tables/path_case_chart_rows.csv",
        "tables/path_case_strategy_rows.csv",
        "metadata/report_metadata.json",
        "summary/summary.md",
    ]:
        source = related_bundle / relative_path
        if source.exists():
            target_name = "report_metadata.json" if source.name == "report_metadata.json" else source.name
            _copy_publish_artifact(source, target_dir / target_name)
            section = relative_path.split("/", 1)[0]
            copied_files.setdefault(section, []).append(target_name)
    related_metadata = _load_json(related_bundle / "metadata" / "report_metadata.json")
    return {
        "contract_selection_root": "_rel",
        "related_contract_selection_runs": [
            {
                "run_slug": clean_string(related_manifest.get("run_slug") or related_bundle.name),
                "relative_dir": Path("_rel", clean_string(related_manifest.get("run_slug") or related_bundle.name)).as_posix(),
                "artifacts": copied_files,
                "generated_at": related_metadata.get("generated_at"),
                "status": related_metadata.get("status"),
                "shareability_status": related_metadata.get("shareability_status"),
            }
        ],
    }


def _related_scenario_hrefs(bundle_dir: Path, publish_dir: Path) -> dict[str, Any]:
    scenario_bundle = _related_scenario_bundle(bundle_dir)
    if scenario_bundle is None:
        return {}
    scenario_publish = scenario_bundle / "publish" / "dashboard.html"
    if not scenario_publish.exists():
        return {}
    relative = Path(os.path.relpath(scenario_publish, publish_dir)).as_posix()
    return {
        "scenario_href": relative,
        "strategy_selector_href": relative + "#strategy-selector",
        "scenario_bundle": {
            "analysis_kind": "scenario",
            "run_slug": scenario_bundle.name,
            "dashboard_href": relative,
        },
    }


def publish_analysis_bundle(
    bundle: str | Path,
    *,
    destination: str | Path | None = None,
    dashboards_root: str | Path | None = None,
    publish_dashboards: bool = False,
) -> Path:
    """Render bundle-local frozen HTML from a canonical analysis bundle."""

    bundle_dir = Path(bundle)
    manifest = _load_json(bundle_dir / "bundle_manifest.json")
    report_metadata = _load_json(bundle_dir / "metadata" / "report_metadata.json")
    publish_dir = Path(destination) if destination is not None else bundle_dir / "publish"
    if publish_dir.exists():
        shutil.rmtree(publish_dir)
    ensure_directory(publish_dir)

    _flatten_bundle_files(bundle_dir, manifest.get("file_map") or {}, publish_dir)
    bundle_context: dict[str, Any] = {}
    if manifest.get("analysis_kind") == "scenario":
        bundle_context.update(_prepare_related_contract_selection(bundle_dir, publish_dir) or {})
    if manifest.get("analysis_kind") == "contract_selection":
        bundle_context.update(_related_scenario_hrefs(bundle_dir, publish_dir))
    report_metadata["bundle_file_map"] = manifest.get("file_map") or {}
    if bundle_context:
        report_metadata["bundle_publish_context"] = bundle_context

    sanitized_metadata = _sanitize_for_published(report_metadata)
    write_json(sanitized_metadata, publish_dir / "report_metadata.json")
    dashboard_path = generate_dashboard(publish_dir, destination=publish_dir / "dashboard.html", published=True)
    published_manifest = {
        "analysis_kind": manifest.get("analysis_kind"),
        "ticker": manifest.get("ticker"),
        "snapshot_date": manifest.get("snapshot_date"),
        "run_slug": manifest.get("run_slug"),
        "published_at": _utc_now_iso(),
        "title": clean_string(sanitized_metadata.get("ticker") or manifest.get("ticker"))
        + " "
        + clean_string(sanitized_metadata.get("analysis_name") or manifest.get("analysis_kind")).replace("_", " ").title(),
        "publish_path": {
            "dashboard": "dashboard.html",
        },
        "bundle_file_map": manifest.get("file_map") or {},
        "report_metadata": sanitized_metadata,
    }
    write_json(published_manifest, publish_dir / "published_manifest.json")
    if publish_dashboards:
        mirror_published_bundle(bundle_dir, dashboards_root=dashboards_root or DEFAULT_DASHBOARDS_ROOT)
    return dashboard_path
