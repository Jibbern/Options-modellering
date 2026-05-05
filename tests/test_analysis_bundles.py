import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from options_lab.analysis import (
    build_contract_selection_analysis,
    build_scenario_analysis,
    publish_analysis_bundle,
    write_analysis_bundle,
)
from options_lab.analysis.contract_selection import _path_view_filename
from options_lab.plots import plot_strike_expiry_tradeoff_overview
from options_lab.publish import generate_dashboard


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def test_strike_expiry_tradeoff_plot_writes_empty_state_for_no_long_calls(temp_workspace_root: Path):
    output_path = temp_workspace_root / "strike_expiry_tradeoff_overview.png"

    result = plot_strike_expiry_tradeoff_overview(
        pd.DataFrame(
            [
                {
                    "candidate_label": "Long Stock Baseline",
                    "strategy_family": "long_stock",
                    "balanced_score": 50,
                    "expiry_date": "2026-12-18",
                    "strike_label": "Stock",
                }
            ]
        ),
        output_path=output_path,
        title="PBI Strike And Expiry Trade-Offs",
    )

    assert result == output_path
    assert output_path.exists()


@pytest.mark.slow
def test_contract_selection_analysis_bundle_writes_canonical_artifacts_without_html(temp_analysis_root: Path):
    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
        stock_path_points="entry:16,1m:18,3m:22",
        iv_path_points="entry:0.00,1m:-0.05,3m:-0.10",
    )
    bundle = write_analysis_bundle(
        result,
        analysis_kind="contract_selection",
        output_root=temp_analysis_root,
    )

    manifest = json.loads((bundle.bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    report_metadata = json.loads((bundle.bundle_dir / "metadata" / "report_metadata.json").read_text(encoding="utf-8"))
    with (bundle.bundle_dir / "tables" / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert manifest["analysis_kind"] == "contract_selection"
    assert manifest["ticker"] == "GPRE"
    assert manifest["snapshot_date"] == "2026-04-12"
    assert manifest["run_slug"] == result.run_slug
    assert (bundle.bundle_dir / "tables" / "candidate_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_rows.csv").exists()
    assert (bundle.bundle_dir / "tables" / "goal_required_path_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_paths_by_option.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_family_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_peak_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_exit_ladder.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_entry_sensitivity.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_iv_sensitivity.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_entry_iv_matrix.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_path_sell_hold_summary.csv").exists()
    assert (bundle.bundle_dir / "summary" / "required_path_tables.html").exists()
    assert (bundle.bundle_dir / "summary" / "required_path_tables.md").exists()
    assert (bundle.bundle_dir / "summary" / "required_path_exit_ladder.md").exists()
    assert manifest["file_map"]["tables"]["required_path_entry_sensitivity.csv"] == "tables/required_path_entry_sensitivity.csv"
    assert manifest["file_map"]["tables"]["required_path_iv_sensitivity.csv"] == "tables/required_path_iv_sensitivity.csv"
    assert manifest["file_map"]["tables"]["required_path_entry_iv_matrix.csv"] == "tables/required_path_entry_iv_matrix.csv"
    assert manifest["file_map"]["tables"]["required_path_sell_hold_summary.csv"] == "tables/required_path_sell_hold_summary.csv"
    assert manifest["file_map"]["summary"]["required_path_tables.html"] == "summary/required_path_tables.html"
    assert manifest["file_map"]["summary"]["required_path_tables.md"] == "summary/required_path_tables.md"
    assert (bundle.bundle_dir / "tables" / "chain_source_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "market_context_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "assumed_path_trace_rows.csv").exists()
    assert (bundle.bundle_dir / "tables" / "iv_path_trace_rows.csv").exists()
    assert (bundle.bundle_dir / "tables" / "compare_vs_stock_path_rows.csv").exists()
    assert (bundle.bundle_dir / "tables" / "iv_path_sensitivity_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "path_risk_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "stock_path_examples.csv").exists()
    assert (bundle.bundle_dir / "tables" / "iv_path_examples.csv").exists()
    assert (bundle.bundle_dir / "tables" / "stock_path_library.csv").exists()
    assert (bundle.bundle_dir / "tables" / "stock_path_gallery.csv").exists()
    assert (bundle.bundle_dir / "tables" / "iv_path_gallery.csv").exists()
    assert (bundle.bundle_dir / "tables" / "path_pair_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "option_value_over_path.csv").exists()
    assert (bundle.bundle_dir / "tables" / "compare_vs_stock_over_path.csv").exists()
    assert (bundle.bundle_dir / "tables" / "representative_paths_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "strike_comparison_under_path.csv").exists()
    assert (bundle.bundle_dir / "tables" / "expiry_comparison_under_path.csv").exists()
    assert (bundle.bundle_dir / "tables" / "long_call_value_over_path_strike_view.csv").exists()
    assert (bundle.bundle_dir / "tables" / "long_call_value_over_path_expiry_view.csv").exists()
    assert (bundle.bundle_dir / "tables" / "long_call_value_over_path_best_of.csv").exists()
    assert (bundle.bundle_dir / "tables" / "decision_highlights.csv").exists()
    assert (bundle.bundle_dir / "tables" / "decision_highlights_explanations.csv").exists()
    assert (bundle.bundle_dir / "tables" / "candidate_robustness_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "candidate_tradeoff_matrix.csv").exists()
    assert (bundle.bundle_dir / "tables" / "stock_vs_option_takeaways.csv").exists()
    assert (bundle.bundle_dir / "tables" / "highlights_score_breakdown.csv").exists()
    assert (bundle.bundle_dir / "tables" / "action_board_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "buy_now_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "watchlist_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "avoid_for_now_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "prefer_stock_instead.csv").exists()
    assert (bundle.bundle_dir / "tables" / "decision_triggers.csv").exists()
    assert (bundle.bundle_dir / "tables" / "action_board_score_breakdown.csv").exists()
    assert (bundle.bundle_dir / "tables" / "action_board_explanations.csv").exists()
    assert (bundle.bundle_dir / "tables" / "bullish_long_call_action_board.csv").exists()
    assert (bundle.bundle_dir / "tables" / "bullish_long_call_watchlist.csv").exists()
    assert (bundle.bundle_dir / "tables" / "bullish_long_call_avoid.csv").exists()
    assert (bundle.bundle_dir / "tables" / "bullish_long_call_triggers.csv").exists()
    assert (bundle.bundle_dir / "tables" / "bullish_long_call_score_breakdown.csv").exists()
    assert (bundle.bundle_dir / "tables" / "top_candidate_cards.csv").exists()
    assert (bundle.bundle_dir / "tables" / "other_structures_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "stock_preference_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "entry_justification_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_stock_path_to_buy.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_move_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_move_vs_stock.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_iv_support_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "entry_barrier_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "thesis_mode_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "thesis_path_family_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "thesis_iv_family_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "thesis_candidate_ranking.csv").exists()
    assert (bundle.bundle_dir / "tables" / "max_justified_premium.csv").exists()
    assert (bundle.bundle_dir / "tables" / "current_vs_justified_premium.csv").exists()
    assert (bundle.bundle_dir / "tables" / "thesis_required_move_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "thesis_stock_vs_option_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "candidate_stress_grid.csv").exists()
    assert (bundle.bundle_dir / "tables" / "premium_sensitivity_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "timing_slip_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "target_stress_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "stress_transition_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_decision_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_decision_path_selections.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_representative_paths.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_path_outcomes.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_required_path_to_beat_stock_1_5x.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_required_path_to_beat_stock_2_0x.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_closest_representative_path_to_edge.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_edge_gap_by_path_family.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_path_family_counts.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_timing_sensitivity.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_iv_sensitivity.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_entry_sensitivity.csv").exists()
    assert (bundle.bundle_dir / "tables" / "single_option_summary_bullets.csv").exists()
    assert (bundle.bundle_dir / "tables" / "chain_overview_summary.csv").exists()
    assert (bundle.bundle_dir / "tables" / "chain_overview_candidates.csv").exists()
    assert (bundle.bundle_dir / "tables" / "required_vs_assumed_path_summary.csv").exists()
    required_summary = pd.read_csv(bundle.bundle_dir / "tables" / "required_path_summary.csv")
    required_paths = pd.read_csv(bundle.bundle_dir / "tables" / "required_paths_by_option.csv")
    required_family = pd.read_csv(bundle.bundle_dir / "tables" / "required_path_family_summary.csv")
    required_peaks = pd.read_csv(bundle.bundle_dir / "tables" / "required_path_peak_summary.csv")
    required_exit_ladder = pd.read_csv(bundle.bundle_dir / "tables" / "required_path_exit_ladder.csv")
    assert {"contract_label", "threshold_multiple", "required_move_pct", "verdict"} <= set(required_summary.columns)
    assert {
        "path_family",
        "option_value",
        "option_return_pct",
        "stock_return_pct",
        "clears_threshold",
        "is_checkpoint_marker",
        "display_marker",
        "checkpoint_label",
        "snapshot_date",
        "analysis_horizon_date",
        "chart_horizon_date",
        "path_terminal_date",
        "option_expiry",
        "option_expiry_date",
        "terminal_basis",
        "chart_horizon_basis",
        "valuation_date",
        "effective_days",
        "time_to_expiry_days",
        "intrinsic_value",
        "time_value",
        "shape_template",
        "shape_source_path",
    } <= set(required_paths.columns)
    assert {
        "failure_driver",
        "realism_bucket",
        "clears_count",
        "peak_option_return_pct",
        "peak_date",
    } <= set(required_family.columns)
    assert {"peak_date", "peak_option_return_pct", "stock_price_at_peak"} <= set(required_peaks.columns)
    assert {
        "exit_return_label",
        "exit_return_pct",
        "first_exit_date",
        "stock_price_at_exit",
        "option_return_pct_at_exit",
    } <= set(required_exit_ladder.columns)
    label_strikes = pd.to_numeric(required_summary["contract_label"].astype(str).str.extract(r"^(\d+(?:\.\d+)?)C")[0], errors="coerce")
    actual_strikes = pd.to_numeric(required_summary["strike"], errors="coerce")
    assert actual_strikes[label_strikes.notna()].eq(label_strikes[label_strikes.notna()]).all()
    solved_summary = required_summary.loc[required_summary["status"].astype(str).str.startswith("solved")]
    assert not solved_summary.empty
    assert pd.to_numeric(solved_summary["required_move_pct"], errors="coerce").notna().all()
    solved_paths = required_paths.loc[required_paths["status"].astype(str).str.startswith("solved")]
    assert pd.to_numeric(solved_paths["stock_price"], errors="coerce").notna().any()
    assert pd.to_numeric(solved_paths["option_value"], errors="coerce").notna().any()
    selected_long_dated = solved_paths.loc[
        (solved_paths["contract_label"].astype(str).str.contains("Dec-26"))
        & (solved_paths["path_family"].astype(str).eq("slow_grind"))
        & (pd.to_numeric(solved_paths["threshold_multiple"], errors="coerce").eq(1.5))
    ]
    assert selected_long_dated["days_from_snapshot"].nunique() > 30
    assert pd.to_datetime(selected_long_dated["date"]).max().date().isoformat() == "2026-12-18"
    assert selected_long_dated["chart_horizon_date"].astype(str).eq("2026-12-18").all()
    assert selected_long_dated["chart_horizon_basis"].astype(str).eq("option_expiry").all()
    assert pd.to_numeric(selected_long_dated["time_to_expiry_days"], errors="coerce").min() == 0
    assert pd.to_numeric(required_peaks["peak_option_return_pct"], errors="coerce").notna().any()
    reached_exits = required_exit_ladder.loc[required_exit_ladder["first_exit_date"].notna()]
    assert not reached_exits.empty
    assert pd.to_numeric(reached_exits["stock_price_at_exit"], errors="coerce").notna().all()
    for path_name in [
        "rally_early_then_fade_then_rally_again",
        "range_bound_near_flat",
        "down_first_then_recovery",
        "late_breakout",
        "early_move_above_strike_then_giveback",
        "reaches_target_late_near_expiry",
    ]:
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "compare_vs_stock_path_rows.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_strike_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_strike_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_expiry_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_expiry_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_best_of_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_best_of_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "path_checkpoints.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "iv_path_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "iv_path_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "iv_checkpoints.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_strike_iv_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_strike_iv_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_strike_iv_checkpoints.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_expiry_iv_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_expiry_iv_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_expiry_iv_checkpoints.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_best_of_iv_value.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_best_of_iv_delta.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "long_call_best_of_iv_checkpoints.csv")).exists()
        assert (bundle.bundle_dir / "tables" / _path_view_filename(path_name, "iv_robustness_summary.csv")).exists()
    assert (bundle.bundle_dir / "tables" / "path_case_chart_rows.csv").exists()
    assert (bundle.bundle_dir / "tables" / "path_case_family_rankings.csv").exists()
    assert (bundle.bundle_dir / "tables" / "path_case_candidate_rankings.csv").exists()
    assert (bundle.bundle_dir / "tables" / "family_comparison.csv").exists()
    assert (bundle.bundle_dir / "tables" / "candidate_comparison.csv").exists()
    assert (bundle.bundle_dir / "tables" / "strike_comparison.csv").exists()
    assert (bundle.bundle_dir / "tables" / "expiry_comparison.csv").exists()
    assert not (bundle.bundle_dir / "tables" / "selection_heatmap_rows.csv").exists()
    assert not (bundle.bundle_dir / "tables" / "selection_slice_rows.csv").exists()
    assert (bundle.bundle_dir / "charts" / "required_path_vs_assumed_path.png").exists()
    assert (bundle.bundle_dir / "charts" / "stock_path_gallery.png").exists()
    assert (bundle.bundle_dir / "charts" / "iv_path_gallery.png").exists()
    assert (bundle.bundle_dir / "charts" / "representative_stock_paths.png").exists()
    assert (bundle.bundle_dir / "charts" / "representative_iv_paths.png").exists()
    assert (bundle.bundle_dir / "charts" / "option_value_over_path.png").exists()
    assert (bundle.bundle_dir / "charts" / "compare_vs_stock_over_path.png").exists()
    assert (bundle.bundle_dir / "charts" / "strike_comparison_under_same_path.png").exists()
    assert (bundle.bundle_dir / "charts" / "expiry_comparison_under_same_path.png").exists()
    assert (bundle.bundle_dir / "charts" / "long_call_value_over_path_strike_view.png").exists()
    assert (bundle.bundle_dir / "charts" / "long_call_value_over_path_expiry_view.png").exists()
    assert (bundle.bundle_dir / "charts" / "long_call_value_over_path_best_of.png").exists()
    assert (bundle.bundle_dir / "charts" / "highlights_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "candidate_robustness_vs_upside.png").exists()
    assert (bundle.bundle_dir / "charts" / "path_survival_scorecard.png").exists()
    assert (bundle.bundle_dir / "charts" / "iv_robustness_scorecard.png").exists()
    assert (bundle.bundle_dir / "charts" / "strike_expiry_tradeoff_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "stock_vs_option_decision_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "action_board_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "bullish_action_board_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "conviction_vs_robustness.png").exists()
    assert (bundle.bundle_dir / "charts" / "bullish_conviction_vs_robustness.png").exists()
    assert (bundle.bundle_dir / "charts" / "buy_watch_avoid_matrix.png").exists()
    assert (bundle.bundle_dir / "charts" / "bullish_buy_watch_avoid_matrix.png").exists()
    assert (bundle.bundle_dir / "charts" / "trigger_map.png").exists()
    assert (bundle.bundle_dir / "charts" / "bullish_trigger_map.png").exists()
    assert (bundle.bundle_dir / "charts" / "top_candidate_cards.png").exists()
    assert (bundle.bundle_dir / "charts" / "stock_vs_option_preference_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "required_stock_path_to_buy.png").exists()
    assert (bundle.bundle_dir / "charts" / "required_move_speed_vs_magnitude.png").exists()
    assert (bundle.bundle_dir / "charts" / "required_move_vs_stock_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "strike_expiry_entry_barrier_map.png").exists()
    assert (bundle.bundle_dir / "charts" / "iv_support_requirement_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "thesis_path_gallery.png").exists()
    assert (bundle.bundle_dir / "charts" / "thesis_iv_gallery.png").exists()
    assert (bundle.bundle_dir / "charts" / "thesis_candidate_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "current_vs_justified_premium.png").exists()
    assert (bundle.bundle_dir / "charts" / "thesis_path_vs_value.png").exists()
    assert (bundle.bundle_dir / "charts" / "thesis_iv_vs_value.png").exists()
    assert (bundle.bundle_dir / "charts" / "thesis_stock_vs_option.png").exists()
    assert (bundle.bundle_dir / "charts" / "stress_test_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "premium_sensitivity_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "timing_slip_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "target_stress_chart.png").exists()
    assert (bundle.bundle_dir / "charts" / "top_candidate_stress_cards.png").exists()
    assert (bundle.bundle_dir / "charts" / "chain_overview.png").exists()
    assert (bundle.bundle_dir / "charts" / "single_option_decision_view.png").exists()
    assert (bundle.bundle_dir / "charts" / "required_paths_overview.png").exists()
    assert [path for path in (bundle.bundle_dir / "charts").glob("required_paths_*.png") if path.name != "required_paths_overview.png"]
    for path_name in [
        "rally_early_then_fade_then_rally_again",
        "range_bound_near_flat",
        "down_first_then_recovery",
        "late_breakout",
        "early_move_above_strike_then_giveback",
        "reaches_target_late_near_expiry",
    ]:
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "compare_vs_stock_path_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_strike_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_strike_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_expiry_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_expiry_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_best_of_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_best_of_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "iv_path_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "iv_path_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_strike_iv_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_strike_iv_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_expiry_iv_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_expiry_iv_delta.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_best_of_iv_value.png")).exists()
        assert (bundle.bundle_dir / "charts" / _path_view_filename(path_name, "long_call_best_of_iv_delta.png")).exists()
    assert (bundle.bundle_dir / "charts" / "required_path_strategy_compare.png").exists()
    assert (bundle.bundle_dir / "charts" / "assumed_path_value_progression.png").exists()
    assert (bundle.bundle_dir / "charts" / "iv_path_trace.png").exists()
    assert (bundle.bundle_dir / "charts" / "compare_vs_stock_path_delta.png").exists()
    assert (bundle.bundle_dir / "charts" / "family_ranking_overview.png").exists()
    assert not (bundle.bundle_dir / "charts" / "heatmap_strike_target_price.png").exists()
    assert not (bundle.bundle_dir / "charts" / "slice_strike_target_price.png").exists()
    assert (bundle.bundle_dir / "metadata" / "report_metadata.json").exists()
    assert (bundle.bundle_dir / "summary" / "summary.md").exists()
    assert (bundle.bundle_dir / "summary" / "highlights.md").exists()
    assert (bundle.bundle_dir / "summary" / "action_board.md").exists()
    assert (bundle.bundle_dir / "summary" / "bullish_action_board.md").exists()
    assert (bundle.bundle_dir / "summary" / "other_structures.md").exists()
    assert (bundle.bundle_dir / "summary" / "entry_justification.md").exists()
    assert (bundle.bundle_dir / "summary" / "thesis_mode.md").exists()
    assert (bundle.bundle_dir / "summary" / "stress_tests.md").exists()
    assert (bundle.bundle_dir / "summary" / "top_candidate_cards.md").exists()
    assert (bundle.bundle_dir / "summary" / "chain_overview.md").exists()
    assert (bundle.bundle_dir / "summary" / "single_option_decision.md").exists()
    assert (bundle.bundle_dir / "summary" / "required_path_summary.md").exists()
    assert (bundle.bundle_dir / "summary" / "top_required_path_candidates.md").exists()
    assert report_metadata["report_kind"] == "contract_selection"
    assert "contract_selection" in report_metadata
    assert report_metadata["decision_highlights"]
    assert report_metadata["action_board"]
    assert "contract_selection_lab" not in report_metadata
    assert summary_rows
    summary_row = summary_rows[0]
    assert "risk_free_rate_source" in summary_row
    assert "risk_free_rate_series" in summary_row
    assert "risk_free_rate_matched_date" in summary_row
    assert "risk_free_rate_note" in summary_row
    assert "spot_price_source" in summary_row
    assert "spot_field_used" in summary_row
    assert "spot_used_prior_date" in summary_row
    assert "spot_quality_note" in summary_row
    assert "analysis_trust_level" in summary_row
    assert "trusted_expiry_count" in summary_row
    assert "spot_price_matched_date" in summary_row
    assert "source_snapshot_storage_locations" in summary_row
    assert "source_snapshot_files" not in summary_row
    assert "ibkr_same_day_spot_rejected_reason" in summary_row
    assert "risk_free_rate_source" in report_metadata["metadata"]
    assert "risk_free_rate_series" in report_metadata["metadata"]
    assert "risk_free_rate_matched_date" in report_metadata["metadata"]
    assert "risk_free_rate_note" in report_metadata["metadata"]
    assert "spot_price_source" in report_metadata["metadata"]
    assert "spot_field_used" in report_metadata["metadata"]
    assert "spot_used_prior_date" in report_metadata["metadata"]
    assert "spot_quality_note" in report_metadata["metadata"]
    assert "ibkr_same_day_spot_attempted" in report_metadata["metadata"]
    assert "analysis_trust_level" in report_metadata["metadata"]
    assert "spot_price_matched_date" in report_metadata["metadata"]
    assert report_metadata["metadata"]["source_snapshot_storage_locations"]
    assert report_metadata["metadata"]["source_snapshot_files"]
    assert report_metadata["metadata"]["research_context"]
    summary_text = (bundle.bundle_dir / "summary" / "summary.md").read_text(encoding="utf-8")
    assert not (bundle.bundle_dir / "legacy_report").exists()
    assert "legacy_compatibility" not in manifest
    assert all("legacy_report/" not in rel_path for section in manifest["file_map"].values() for rel_path in section.values())
    assert all("selection_heatmap" not in rel_path for section in manifest["file_map"].values() for rel_path in section.values())
    assert all("selection_slice" not in rel_path for section in manifest["file_map"].values() for rel_path in section.values())
    assert all("heatmap_strike" not in rel_path for section in manifest["file_map"].values() for rel_path in section.values())
    assert all("slice_strike" not in rel_path for section in manifest["file_map"].values() for rel_path in section.values())
    assert "Best Family" in summary_text
    assert "Best Candidate" in summary_text
    assert "Stock Benchmark" in summary_text
    assert "## Action Board / Contract Picker" in summary_text
    assert "summary/action_board.md" in summary_text
    assert "summary/bullish_action_board.md" in summary_text
    assert "summary/other_structures.md" in summary_text
    assert "## Entry Justification / Required Stock Path" in summary_text
    assert "summary/entry_justification.md" in summary_text
    assert "tables/entry_justification_candidates.csv" in summary_text
    assert "tables/required_stock_path_to_buy.csv" in summary_text
    assert "tables/required_move_summary.csv" in summary_text
    assert "tables/required_move_vs_stock.csv" in summary_text
    assert "tables/required_iv_support_summary.csv" in summary_text
    assert "tables/entry_barrier_summary.csv" in summary_text
    assert "charts/required_stock_path_to_buy.png" in summary_text
    assert "charts/required_move_speed_vs_magnitude.png" in summary_text
    assert "charts/required_move_vs_stock_chart.png" in summary_text
    assert "charts/strike_expiry_entry_barrier_map.png" in summary_text
    assert "charts/iv_support_requirement_chart.png" in summary_text
    assert "## Thesis / Price Target Mode" in summary_text
    assert "summary/thesis_mode.md" in summary_text
    assert "charts/thesis_candidate_overview.png" in summary_text
    assert "charts/current_vs_justified_premium.png" in summary_text
    assert "charts/thesis_path_gallery.png" in summary_text
    assert "tables/thesis_candidate_ranking.csv" in summary_text
    assert "tables/current_vs_justified_premium.csv" in summary_text
    assert "## Practical Stress Tests" in summary_text
    assert "summary/stress_tests.md" in summary_text
    assert "tables/candidate_stress_grid.csv" in summary_text
    assert "charts/stress_test_overview.png" in summary_text
    assert "charts/premium_sensitivity_chart.png" in summary_text
    assert "## Chain Overview / Compare Options" in summary_text
    assert "summary/chain_overview.md" in summary_text
    assert "tables/chain_overview_summary.csv" in summary_text
    assert "tables/chain_overview_candidates.csv" in summary_text
    assert "charts/chain_overview.png" in summary_text
    assert "## Single-Option Decision View" in summary_text
    assert "summary/single_option_decision.md" in summary_text
    assert "tables/single_option_decision_path_selections.csv" in summary_text
    assert "tables/single_option_path_outcomes.csv" in summary_text
    assert "tables/single_option_required_path_to_beat_stock_1_5x.csv" in summary_text
    assert "tables/single_option_edge_gap_by_path_family.csv" in summary_text
    assert "charts/single_option_decision_view.png" in summary_text
    assert "tables/action_board_candidates.csv" in summary_text
    assert "tables/decision_triggers.csv" in summary_text
    assert "tables/bullish_long_call_action_board.csv" in summary_text
    assert "tables/bullish_long_call_watchlist.csv" in summary_text
    assert "summary/top_candidate_cards.md" in summary_text
    assert "tables/top_candidate_cards.csv" in summary_text
    assert "charts/top_candidate_cards.png" in summary_text
    assert "tables/other_structures_summary.csv" in summary_text
    assert "## Decision Highlights" in summary_text
    assert "tables/decision_highlights.csv" in summary_text
    assert "summary/highlights.md" in summary_text
    assert "## Market Context / Trust Summary" in summary_text
    assert "## Stock Path Gallery" in summary_text
    assert "tables/stock_path_library.csv" in summary_text
    assert "## IV Path Gallery" in summary_text
    assert "## Path-Centric Long-Call Scenario Library" in summary_text
    assert "## Required vs Assumed Path Summary" in summary_text
    assert "## Same-Path Compare vs Stock" in summary_text
    assert "## Representative Path Summary" in summary_text
    assert "## Family Selection And Exact Contract Choice" in summary_text
    assert "## Long-Call Strike Comparison" in summary_text
    assert "## Long-Call Expiry Comparison" in summary_text
    assert "## Best-Of Long-Call Comparison" in summary_text
    assert "## Risk Notes" in summary_text
    assert "How To Read The Core Charts" in summary_text
    assert "Above zero means the structure is ahead of stock" in summary_text
    assert "summary/entry_justification.md" in summary_text
    assert "required_stock_path_to_buy.png" in summary_text
    assert "required_move_speed_vs_magnitude.png" in summary_text
    assert "required_move_vs_stock_chart.png" in summary_text
    assert "strike_expiry_entry_barrier_map.png" in summary_text
    assert "iv_support_requirement_chart.png" in summary_text
    assert "long_call_value_over_path_strike_view.png" in summary_text
    assert "long_call_value_over_path_expiry_view.png" in summary_text
    assert "long_call_value_over_path_best_of.png" in summary_text
    assert _path_view_filename("late_breakout", "long_call_strike_value.png") in summary_text
    assert _path_view_filename("late_breakout", "long_call_strike_delta.png") in summary_text
    assert _path_view_filename("range_bound_near_flat", "compare_vs_stock_path_delta.png") in summary_text
    assert "stock_path_gallery.png" in summary_text
    assert "stock_path_library.csv" in summary_text
    assert "single_option_decision_path_selections.csv" in summary_text
    assert "single_option_required_path_to_beat_stock_1_5x.csv" in summary_text
    assert "single_option_edge_gap_by_path_family.csv" in summary_text
    assert "iv_path_gallery.png" in summary_text
    assert "compare_vs_stock_path_delta.png" in summary_text
    assert "options_show_edge" not in summary_text
    assert "Source Snapshot Files" not in summary_text
    assert "C:/Users" not in summary_text
    assert "C:\\Users" not in summary_text
    highlights_text = (bundle.bundle_dir / "summary" / "highlights.md").read_text(encoding="utf-8")
    assert "Decision Snapshot" in highlights_text
    assert "What Looks Most Attractive Right Now" in highlights_text
    assert "Where Stock Still Looks Better" in highlights_text
    assert "Calls That Need IV Support" in highlights_text
    assert "Most Robust Calls Across Paths" in highlights_text
    assert "C:/Users" not in highlights_text
    assert "C:\\Users" not in highlights_text
    action_board_text = (bundle.bundle_dir / "summary" / "action_board.md").read_text(encoding="utf-8")
    bullish_action_board_text = (bundle.bundle_dir / "summary" / "bullish_action_board.md").read_text(encoding="utf-8")
    top_candidate_cards_text = (bundle.bundle_dir / "summary" / "top_candidate_cards.md").read_text(encoding="utf-8")
    assert "What Looks Most Actionable Right Now" in action_board_text
    assert "What Belongs On The Watchlist" in action_board_text
    assert "What To Avoid For Now" in action_board_text
    assert "When Stock Is Still Better" in action_board_text
    assert "Key Triggers To Watch" in action_board_text
    assert "Best Bullish Long Calls Right Now" in bullish_action_board_text
    assert "Watchlist: Interesting But Not Buyable Yet" in bullish_action_board_text
    assert "What Seems To Be Hurting Calls" in bullish_action_board_text
    assert "Top Bullish Call Cards" in top_candidate_cards_text
    assert "Upgrade if" in top_candidate_cards_text
    assert "C:/Users" not in action_board_text
    assert "C:\\Users" not in action_board_text
    assert "C:/Users" not in bullish_action_board_text
    assert "C:\\Users" not in bullish_action_board_text
    assert "C:/Users" not in top_candidate_cards_text
    assert "C:\\Users" not in top_candidate_cards_text
    entry_justification_text = (bundle.bundle_dir / "summary" / "entry_justification.md").read_text(encoding="utf-8")
    assert "What Has To Happen For These Calls To Be Worth Buying" in entry_justification_text
    assert "Which Calls Require Too Much" in entry_justification_text
    assert "Which Calls Are More Forgiving" in entry_justification_text
    assert "Which Calls Need Fast Confirmation" in entry_justification_text
    assert "Which Calls Mainly Need Better IV / Better Entry" in entry_justification_text
    assert "When Stock Is Still Better Even If The Path Is \"Right\"" in entry_justification_text
    assert "required_stock_path_to_buy.png" in entry_justification_text
    assert "required_move_speed_vs_magnitude.png" in entry_justification_text
    assert "required_move_vs_stock_chart.png" in entry_justification_text
    assert "C:/Users" not in entry_justification_text
    assert "C:\\Users" not in entry_justification_text
    thesis_mode_text = (bundle.bundle_dir / "summary" / "thesis_mode.md").read_text(encoding="utf-8")
    assert "Thesis Snapshot" in thesis_mode_text
    assert "Which Calls Start To Look Reasonable Under This Thesis" in thesis_mode_text
    assert "Current Premium vs Thesis-Justified Premium" in thesis_mode_text
    assert "When Stock Still Looks Better" in thesis_mode_text
    assert "C:/Users" not in thesis_mode_text
    assert "C:\\Users" not in thesis_mode_text
    single_option_text = (bundle.bundle_dir / "summary" / "single_option_decision.md").read_text(encoding="utf-8")
    chain_overview_text = (bundle.bundle_dir / "summary" / "chain_overview.md").read_text(encoding="utf-8")
    assert "Single-Option Decision View" in single_option_text
    assert "what stock paths make one selected call worth buying instead of buying stock" in single_option_text
    assert "single_option_decision_path_selections.csv" in single_option_text
    assert "single_option_required_path_to_beat_stock_1_5x.csv" in single_option_text
    assert "single_option_edge_gap_by_path_family.csv" in single_option_text
    assert "Chain Overview / Compare Options" in chain_overview_text
    assert "bullish long calls against long stock" in chain_overview_text
    assert "C:/Users" not in chain_overview_text
    assert "C:\\Users" not in chain_overview_text
    assert "C:/Users" not in single_option_text
    assert "C:\\Users" not in single_option_text
    with (bundle.bundle_dir / "tables" / "decision_highlights.csv").open("r", encoding="utf-8", newline="") as handle:
        highlight_rows = list(csv.DictReader(handle))
    assert {row["highlight_category"] for row in highlight_rows} >= {
        "most_robust_call",
        "best_aggressive_upside_call",
        "best_balanced_call",
        "stock_still_best_baseline",
        "requires_iv_support",
    }
    assert any(
        row["decision_status"] in {"weak_differentiation", "no_clear_edge_under_current_assumptions"}
        or "stock" in row.get("main_warning", "").lower()
        for row in highlight_rows
    )
    with (bundle.bundle_dir / "tables" / "action_board_candidates.csv").open("r", encoding="utf-8", newline="") as handle:
        action_rows = list(csv.DictReader(handle))
    assert action_rows
    assert {
        "action_bucket",
        "action_priority_rank",
        "action_confidence",
        "candidate_conviction_score",
        "robustness_score",
        "stock_relative_score",
        "what_has_to_happen",
        "upgrade_rule",
        "what_would_invalidate",
        "invalidate_rule",
    } <= set(action_rows[0])
    assert {row["action_bucket"] for row in action_rows} & {"Buy Now", "Watchlist", "Avoid For Now", "Prefer Stock Instead"}
    with (bundle.bundle_dir / "tables" / "decision_triggers.csv").open("r", encoding="utf-8", newline="") as handle:
        trigger_rows = list(csv.DictReader(handle))
    assert trigger_rows
    assert {"candidate_label", "key_trigger_type", "key_trigger_value", "what_would_invalidate", "upgrade_rule", "invalidate_rule"} <= set(trigger_rows[0])
    assert "trigger_type_label" in trigger_rows[0]
    with (bundle.bundle_dir / "tables" / "top_candidate_cards.csv").open("r", encoding="utf-8", newline="") as handle:
        card_rows = list(csv.DictReader(handle))
    assert card_rows
    assert {"contract_label", "bucket", "upgrade_rule", "invalidate_rule", "compare_vs_stock_note"} <= set(card_rows[0])
    assert summary_text.index("## Action Board / Contract Picker") < summary_text.index("## Decision Highlights")
    assert summary_text.index("## Entry Justification / Required Stock Path") < summary_text.index("## Decision Highlights")
    assert summary_text.index("## Decision Highlights") < summary_text.index("## Market Context / Trust Summary")
    assert summary_text.index("## Stock Path Gallery") < summary_text.index("## IV Path Gallery")
    assert summary_text.index("## IV Path Gallery") < summary_text.index("## Required vs Assumed Path Summary")
    assert summary_text.index("## Same-Path Compare vs Stock") < summary_text.index("## Long-Call Strike Comparison")
    assert summary_text.index("## Long-Call Expiry Comparison") < summary_text.index("## Best-Of Long-Call Comparison")
    assert summary_text.index("## Best-Of Long-Call Comparison") < summary_text.index("## Representative Path Summary")
    path_risk_csv = (bundle.bundle_dir / "tables" / "path_risk_summary.csv").read_text(encoding="utf-8")
    assert "4.999999999999999" not in path_risk_csv


def test_contract_selection_publish_dashboard_reads_edge_artifacts_without_recomputing_analysis():
    source = (PROJECT_ROOT / "options_lab" / "publish" / "dashboard.py").read_text(encoding="utf-8")

    assert "build_contract_selection_analysis" not in source
    assert "single_option_required_path_to_beat_stock_1_5x.csv" in source
    assert "single_option_required_path_to_beat_stock_2_0x.csv" in source
    assert "single_option_closest_representative_path_to_edge.csv" in source
    assert "single_option_edge_gap_by_path_family.csv" in source


def test_publish_analysis_bundle_renders_dashboard_from_canonical_bundle(temp_analysis_root: Path):
    result = build_scenario_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        expiry_date="2026-04-17",
        data_root=DATA_ROOT,
    )
    bundle = write_analysis_bundle(
        result,
        analysis_kind="scenario",
        output_root=temp_analysis_root,
    )
    dashboard_path = publish_analysis_bundle(bundle.bundle_dir)
    html = dashboard_path.read_text(encoding="utf-8")
    published_manifest = json.loads((bundle.bundle_dir / "publish" / "published_manifest.json").read_text(encoding="utf-8"))

    assert dashboard_path.exists()
    assert "Primary Scenario Dashboard" in html
    assert "Path Case Summary" in html
    assert dashboard_path.name == "dashboard.html"
    assert published_manifest["analysis_kind"] == "scenario"
    assert (published_manifest["report_metadata"] or {})["report_kind"] == "scenario"
    assert "publish/dashboard.html" not in html
    assert "C:/Users" not in html
    assert "C:\\Users" not in html
    assert "file:///" not in html


@pytest.mark.slow
def test_contract_selection_publish_uses_bundle_relative_dashboard_links_without_local_path_leaks(temp_analysis_root: Path):
    scenario_result = build_scenario_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        expiry_date="2026-04-17",
        data_root=DATA_ROOT,
    )
    scenario_bundle = write_analysis_bundle(
        scenario_result,
        analysis_kind="scenario",
        output_root=temp_analysis_root,
    )
    publish_analysis_bundle(scenario_bundle.bundle_dir)

    contract_result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
    )
    contract_bundle = write_analysis_bundle(
        contract_result,
        analysis_kind="contract_selection",
        output_root=temp_analysis_root,
    )
    dashboard_path = publish_analysis_bundle(contract_bundle.bundle_dir)
    html = dashboard_path.read_text(encoding="utf-8")
    publish_dir = contract_bundle.bundle_dir / "publish"
    published_summary_md = (publish_dir / "summary.md").read_text(encoding="utf-8")
    published_summary_csv = (publish_dir / "summary.csv").read_text(encoding="utf-8")
    published_candidate_csv = (publish_dir / "candidate_summary.csv").read_text(encoding="utf-8")

    assert "legacy_report" not in html
    assert "C:/Users" not in html
    assert "C:\\Users" not in html
    assert "file:///" not in html
    assert "Decision Snapshot" in html
    assert "Chain Overview / Compare Options" in html
    assert "Chain Overview Candidate Table" in html
    assert "Market Context / Trust Summary" in html
    assert "Required vs Assumed Path" in html
    assert "required_path_tables.html" in html
    assert "Required Path Tables" in html
    assert "Representative Paths" in html
    assert "Option Value Over Path" in html
    assert "Compare vs Stock Over Path" in html
    assert "Same-Path Strike Comparison" in html
    assert "Same-Path Expiry Comparison" in html
    assert "Family / Candidate Highlights" in html
    assert "Warnings / Risk Notes" in html
    assert "Open The Main Explorer" not in html
    assert "What This Wrapper Is For" not in html
    for text in [published_summary_md, published_summary_csv, published_candidate_csv]:
        assert "C:/Users" not in text
        assert "C:\\Users" not in text
        assert "file:///" not in text
    assert "Source Snapshot Files" not in published_summary_md
    assert "Source Snapshot Files" not in published_summary_csv


@pytest.mark.slow
def test_publish_dashboard_ignores_stray_publish_files_and_uses_bundle_file_map(temp_analysis_root: Path):
    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
    )
    bundle = write_analysis_bundle(
        result,
        analysis_kind="contract_selection",
        output_root=temp_analysis_root,
    )
    publish_analysis_bundle(bundle.bundle_dir)

    publish_dir = bundle.bundle_dir / "publish"
    (publish_dir / "zzz_noise.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (publish_dir / "zzz_noise.png").write_bytes(b"not-a-real-png")

    published_report_metadata = json.loads((publish_dir / "report_metadata.json").read_text(encoding="utf-8"))

    assert published_report_metadata["bundle_file_map"]["tables"]["candidate_summary.csv"] == "tables/candidate_summary.csv"
    assert published_report_metadata["bundle_file_map"]["charts"]["family_ranking_overview.png"] == "charts/family_ranking_overview.png"

    dashboard_path = generate_dashboard(publish_dir, destination=publish_dir / "dashboard.html", published=True)
    html = dashboard_path.read_text(encoding="utf-8")

    assert "zzz_noise.csv" not in html
    assert "zzz_noise.png" not in html


@pytest.mark.slow
def test_scenario_publish_carries_explicit_related_contract_run_context(temp_analysis_root: Path):
    contract_result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
    )
    contract_bundle = write_analysis_bundle(
        contract_result,
        analysis_kind="contract_selection",
        output_root=temp_analysis_root,
    )

    scenario_result = build_scenario_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        expiry_date="2026-04-17",
        data_root=DATA_ROOT,
    )
    scenario_bundle = write_analysis_bundle(
        scenario_result,
        analysis_kind="scenario",
        output_root=temp_analysis_root,
    )
    publish_analysis_bundle(scenario_bundle.bundle_dir)

    published_report_metadata = json.loads((scenario_bundle.bundle_dir / "publish" / "report_metadata.json").read_text(encoding="utf-8"))
    related_runs = (published_report_metadata.get("bundle_publish_context") or {}).get("related_contract_selection_runs") or []

    assert related_runs
    assert related_runs[0]["run_slug"] == contract_bundle.run_slug
    assert related_runs[0]["relative_dir"].startswith("_rel/")
    assert "candidate_summary.csv" in related_runs[0]["artifacts"]["tables"]
