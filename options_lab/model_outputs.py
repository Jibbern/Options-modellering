"""Curated product-facing projections of frozen analysis bundles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

from .analysis import DEFAULT_ANALYSIS_OUTPUT_ROOT, resolve_analysis_bundle
from .utils import clean_string, ensure_directory, windows_extended_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_OUTPUT_ROOT = PROJECT_ROOT / "model_outputs"
TEXT_SUFFIXES = {".csv", ".json", ".md", ".html", ".txt"}
ABSOLUTE_WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:(?:\\\\|\\/|\\|/)[^\s<>\",|)]+")
CONTRACT_SELECTION_OVERVIEW_SUMMARIES = {
    "highlights.md",
    "action_board.md",
    "bullish_action_board.md",
    "chain_overview.md",
    "other_structures.md",
    "entry_justification.md",
    "thesis_mode.md",
    "stress_tests.md",
    "top_candidate_cards.md",
    "single_option_decision.md",
}
# Projection allowlists only: model_outputs never computes analysis. New
# product surfaces should be generated in options_lab.analysis and promoted here.
PRIMARY_MODEL_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "summary": (
        "summary.md",
        "highlights.md",
        "action_board.md",
        "bullish_action_board.md",
        "chain_overview.md",
        "other_structures.md",
        "entry_justification.md",
        "thesis_mode.md",
        "stress_tests.md",
        "top_candidate_cards.md",
        "single_option_decision.md",
    ),
    "tables": (
        "summary.csv",
        "chain_source_summary.csv",
        "market_context_summary.csv",
        "stock_path_library.csv",
        "stock_path_gallery.csv",
        "iv_path_gallery.csv",
        "family_comparison.csv",
        "candidate_comparison.csv",
        "strike_comparison_under_path.csv",
        "expiry_comparison_under_path.csv",
        "option_value_over_path.csv",
        "compare_vs_stock_path_rows.csv",
        "compare_vs_stock_over_path.csv",
        "long_call_value_over_path_strike_view.csv",
        "long_call_value_over_path_expiry_view.csv",
        "long_call_value_over_path_best_of.csv",
        "decision_highlights.csv",
        "decision_highlights_explanations.csv",
        "action_board_candidates.csv",
        "buy_now_candidates.csv",
        "watchlist_candidates.csv",
        "avoid_for_now_candidates.csv",
        "prefer_stock_instead.csv",
        "decision_triggers.csv",
        "action_board_score_breakdown.csv",
        "action_board_explanations.csv",
        "bullish_long_call_action_board.csv",
        "bullish_long_call_watchlist.csv",
        "bullish_long_call_avoid.csv",
        "bullish_long_call_triggers.csv",
        "bullish_long_call_score_breakdown.csv",
        "top_candidate_cards.csv",
        "other_structures_summary.csv",
        "stock_preference_summary.csv",
        "entry_justification_candidates.csv",
        "required_stock_path_to_buy.csv",
        "required_move_summary.csv",
        "required_move_vs_stock.csv",
        "required_iv_support_summary.csv",
        "entry_barrier_summary.csv",
        "thesis_path_gallery.csv",
        "thesis_iv_gallery.csv",
        "thesis_mode_candidates.csv",
        "thesis_path_family_summary.csv",
        "thesis_iv_family_summary.csv",
        "thesis_candidate_ranking.csv",
        "max_justified_premium.csv",
        "current_vs_justified_premium.csv",
        "thesis_required_move_summary.csv",
        "thesis_stock_vs_option_summary.csv",
        "candidate_stress_grid.csv",
        "premium_sensitivity_summary.csv",
        "timing_slip_summary.csv",
        "target_stress_summary.csv",
        "stress_transition_summary.csv",
        "chain_overview_summary.csv",
        "chain_overview_candidates.csv",
        "single_option_decision_summary.csv",
        "single_option_decision_path_selections.csv",
        "single_option_representative_paths.csv",
        "single_option_path_outcomes.csv",
        "single_option_required_path_to_beat_stock_1_5x.csv",
        "single_option_required_path_to_beat_stock_2_0x.csv",
        "single_option_closest_representative_path_to_edge.csv",
        "single_option_edge_gap_by_path_family.csv",
        "single_option_path_family_counts.csv",
        "single_option_timing_sensitivity.csv",
        "single_option_iv_sensitivity.csv",
        "single_option_entry_sensitivity.csv",
        "single_option_summary_bullets.csv",
        "candidate_robustness_summary.csv",
        "candidate_tradeoff_matrix.csv",
        "stock_vs_option_takeaways.csv",
        "highlights_score_breakdown.csv",
    ),
    "charts": (
        "highlights_overview.png",
        "action_board_overview.png",
        "bullish_action_board_overview.png",
        "conviction_vs_robustness.png",
        "bullish_conviction_vs_robustness.png",
        "buy_watch_avoid_matrix.png",
        "bullish_buy_watch_avoid_matrix.png",
        "trigger_map.png",
        "bullish_trigger_map.png",
        "top_candidate_cards.png",
        "stock_vs_option_preference_chart.png",
        "required_stock_path_to_buy.png",
        "required_move_speed_vs_magnitude.png",
        "required_move_vs_stock_chart.png",
        "strike_expiry_entry_barrier_map.png",
        "iv_support_requirement_chart.png",
        "thesis_path_gallery.png",
        "thesis_iv_gallery.png",
        "thesis_candidate_overview.png",
        "current_vs_justified_premium.png",
        "thesis_path_vs_value.png",
        "thesis_iv_vs_value.png",
        "thesis_stock_vs_option.png",
        "stress_test_overview.png",
        "premium_sensitivity_chart.png",
        "timing_slip_chart.png",
        "target_stress_chart.png",
        "top_candidate_stress_cards.png",
        "chain_overview.png",
        "single_option_decision_view.png",
        "candidate_robustness_vs_upside.png",
        "path_survival_scorecard.png",
        "iv_robustness_scorecard.png",
        "strike_expiry_tradeoff_overview.png",
        "stock_vs_option_decision_chart.png",
        "stock_path_gallery.png",
        "iv_path_gallery.png",
        "required_path_vs_assumed_path.png",
        "compare_vs_stock_path_delta.png",
        "representative_stock_paths.png",
        "representative_iv_paths.png",
        "option_value_over_path.png",
        "compare_vs_stock_over_path.png",
        "strike_comparison_under_same_path.png",
        "expiry_comparison_under_same_path.png",
        "long_call_value_over_path_strike_view.png",
        "long_call_value_over_path_expiry_view.png",
        "long_call_value_over_path_best_of.png",
    ),
}
PATH_CENTRIC_TABLE_SUFFIXES = (
    "__compare_vs_stock_path_rows.csv",
    "__long_call_strike_value.csv",
    "__long_call_strike_delta.csv",
    "__long_call_expiry_value.csv",
    "__long_call_expiry_delta.csv",
    "__long_call_best_of_value.csv",
    "__long_call_best_of_delta.csv",
    "__path_checkpoints.csv",
    "__iv_path_value.csv",
    "__iv_path_delta.csv",
    "__iv_checkpoints.csv",
    "__long_call_strike_iv_value.csv",
    "__long_call_strike_iv_delta.csv",
    "__long_call_strike_iv_checkpoints.csv",
    "__long_call_expiry_iv_value.csv",
    "__long_call_expiry_iv_delta.csv",
    "__long_call_expiry_iv_checkpoints.csv",
    "__long_call_best_of_iv_value.csv",
    "__long_call_best_of_iv_delta.csv",
    "__long_call_best_of_iv_checkpoints.csv",
    "__iv_robustness_summary.csv",
)
PATH_CENTRIC_CHART_SUFFIXES = (
    "__compare_vs_stock_path_delta.png",
    "__long_call_strike_value.png",
    "__long_call_strike_delta.png",
    "__long_call_expiry_value.png",
    "__long_call_expiry_delta.png",
    "__long_call_best_of_value.png",
    "__long_call_best_of_delta.png",
    "__iv_path_value.png",
    "__iv_path_delta.png",
    "__long_call_strike_iv_value.png",
    "__long_call_strike_iv_delta.png",
    "__long_call_expiry_iv_value.png",
    "__long_call_expiry_iv_delta.png",
    "__long_call_best_of_iv_value.png",
    "__long_call_best_of_iv_delta.png",
)
OVERVIEW_CHARTS = (
    "bullish_action_board_overview.png",
    "bullish_conviction_vs_robustness.png",
    "bullish_buy_watch_avoid_matrix.png",
    "bullish_trigger_map.png",
    "top_candidate_cards.png",
    "action_board_overview.png",
    "conviction_vs_robustness.png",
    "buy_watch_avoid_matrix.png",
    "trigger_map.png",
    "stock_vs_option_preference_chart.png",
    "required_stock_path_to_buy.png",
    "required_move_speed_vs_magnitude.png",
    "required_move_vs_stock_chart.png",
    "strike_expiry_entry_barrier_map.png",
    "iv_support_requirement_chart.png",
    "thesis_candidate_overview.png",
    "current_vs_justified_premium.png",
    "thesis_path_gallery.png",
    "thesis_iv_gallery.png",
    "thesis_path_vs_value.png",
    "thesis_iv_vs_value.png",
    "thesis_stock_vs_option.png",
    "stress_test_overview.png",
    "premium_sensitivity_chart.png",
    "timing_slip_chart.png",
    "target_stress_chart.png",
    "top_candidate_stress_cards.png",
    "highlights_overview.png",
    "candidate_robustness_vs_upside.png",
    "path_survival_scorecard.png",
    "iv_robustness_scorecard.png",
    "strike_expiry_tradeoff_overview.png",
    "stock_vs_option_decision_chart.png",
    "stock_path_gallery.png",
    "iv_path_gallery.png",
    "required_path_vs_assumed_path.png",
    "compare_vs_stock_path_delta.png",
)
ACTION_BOARD_CHARTS = {
    "bullish_action_board_overview.png",
    "bullish_conviction_vs_robustness.png",
    "bullish_buy_watch_avoid_matrix.png",
    "bullish_trigger_map.png",
    "top_candidate_cards.png",
    "action_board_overview.png",
    "conviction_vs_robustness.png",
    "buy_watch_avoid_matrix.png",
    "trigger_map.png",
    "stock_vs_option_preference_chart.png",
}
ENTRY_JUSTIFICATION_CHARTS = {
    "required_stock_path_to_buy.png",
    "required_move_speed_vs_magnitude.png",
    "required_move_vs_stock_chart.png",
    "strike_expiry_entry_barrier_map.png",
    "iv_support_requirement_chart.png",
}
THESIS_MODE_CHARTS = {
    "thesis_candidate_overview.png",
    "current_vs_justified_premium.png",
    "thesis_path_gallery.png",
    "thesis_iv_gallery.png",
    "thesis_path_vs_value.png",
    "thesis_iv_vs_value.png",
    "thesis_stock_vs_option.png",
}
ACTION_BOARD_TABLES = {
    "action_board_candidates.csv",
    "buy_now_candidates.csv",
    "watchlist_candidates.csv",
    "avoid_for_now_candidates.csv",
    "prefer_stock_instead.csv",
    "decision_triggers.csv",
    "action_board_score_breakdown.csv",
    "action_board_explanations.csv",
    "bullish_long_call_action_board.csv",
    "bullish_long_call_watchlist.csv",
    "bullish_long_call_avoid.csv",
    "bullish_long_call_triggers.csv",
    "bullish_long_call_score_breakdown.csv",
    "top_candidate_cards.csv",
    "other_structures_summary.csv",
    "stock_preference_summary.csv",
}
ENTRY_JUSTIFICATION_TABLES = {
    "entry_justification_candidates.csv",
    "required_stock_path_to_buy.csv",
    "required_move_summary.csv",
    "required_move_vs_stock.csv",
    "required_iv_support_summary.csv",
    "entry_barrier_summary.csv",
}
THESIS_MODE_TABLES = {
    "thesis_path_gallery.csv",
    "thesis_iv_gallery.csv",
    "thesis_mode_candidates.csv",
    "thesis_path_family_summary.csv",
    "thesis_iv_family_summary.csv",
    "thesis_candidate_ranking.csv",
    "max_justified_premium.csv",
    "current_vs_justified_premium.csv",
    "thesis_required_move_summary.csv",
    "thesis_stock_vs_option_summary.csv",
}
DECISION_HIGHLIGHT_CHARTS = {
    "highlights_overview.png",
    "candidate_robustness_vs_upside.png",
    "path_survival_scorecard.png",
    "iv_robustness_scorecard.png",
    "strike_expiry_tradeoff_overview.png",
    "stock_vs_option_decision_chart.png",
}
OVERVIEW_TABLES = (
    "bullish_long_call_action_board.csv",
    "bullish_long_call_watchlist.csv",
    "bullish_long_call_avoid.csv",
    "bullish_long_call_triggers.csv",
    "bullish_long_call_score_breakdown.csv",
    "top_candidate_cards.csv",
    "other_structures_summary.csv",
    "stock_preference_summary.csv",
    "action_board_candidates.csv",
    "buy_now_candidates.csv",
    "watchlist_candidates.csv",
    "avoid_for_now_candidates.csv",
    "prefer_stock_instead.csv",
    "decision_triggers.csv",
    "action_board_score_breakdown.csv",
    "action_board_explanations.csv",
    "decision_highlights.csv",
    "decision_highlights_explanations.csv",
    "candidate_robustness_summary.csv",
    "candidate_tradeoff_matrix.csv",
    "stock_vs_option_takeaways.csv",
    "highlights_score_breakdown.csv",
    "stock_path_library.csv",
    "stock_path_gallery.csv",
    "iv_path_gallery.csv",
    "entry_justification_candidates.csv",
    "required_stock_path_to_buy.csv",
    "required_move_summary.csv",
    "required_move_vs_stock.csv",
    "required_iv_support_summary.csv",
    "entry_barrier_summary.csv",
    "thesis_path_gallery.csv",
    "thesis_iv_gallery.csv",
    "thesis_mode_candidates.csv",
    "thesis_path_family_summary.csv",
    "thesis_iv_family_summary.csv",
    "thesis_candidate_ranking.csv",
    "max_justified_premium.csv",
    "current_vs_justified_premium.csv",
    "thesis_required_move_summary.csv",
    "thesis_stock_vs_option_summary.csv",
)
CORE_TABLES = (
    "chain_source_summary.csv",
    "market_context_summary.csv",
    "family_comparison.csv",
    "candidate_comparison.csv",
    "strike_comparison_under_path.csv",
    "expiry_comparison_under_path.csv",
    "compare_vs_stock_path_rows.csv",
    "long_call_value_over_path_strike_view.csv",
    "long_call_value_over_path_expiry_view.csv",
    "long_call_value_over_path_best_of.csv",
)
CORE_VIEW_SUMMARIES = {
    "bullish_action_board.md",
    "chain_overview.md",
    "entry_justification.md",
    "stress_tests.md",
    "top_candidate_cards.md",
    "single_option_decision.md",
}
THESIS_VIEW_SUMMARIES = {
    "thesis_mode.md",
}
SECONDARY_VIEW_SUMMARIES = {
    "action_board.md",
    "highlights.md",
    "other_structures.md",
    "decision_highlights.md",
}
CORE_VIEW_CHARTS = {
    "top_candidate_cards.png",
    "chain_overview.png",
    "current_vs_justified_premium.png",
    "stock_vs_option_preference_chart.png",
    "required_stock_path_to_buy.png",
    "required_move_speed_vs_magnitude.png",
    "stress_test_overview.png",
    "premium_sensitivity_chart.png",
    "timing_slip_chart.png",
    "target_stress_chart.png",
    "top_candidate_stress_cards.png",
    "single_option_decision_view.png",
}
THESIS_VIEW_CHARTS = {
    "thesis_candidate_overview.png",
    "thesis_path_gallery.png",
    "thesis_stock_vs_option.png",
    "current_vs_justified_premium.png",
    "thesis_iv_gallery.png",
}
SECONDARY_VIEW_CHARTS = set(OVERVIEW_CHARTS) - CORE_VIEW_CHARTS - THESIS_VIEW_CHARTS
CORE_VIEW_TABLES = {
    "bullish_long_call_watchlist.csv",
    "bullish_long_call_triggers.csv",
    "chain_overview_summary.csv",
    "chain_overview_candidates.csv",
    "candidate_stress_grid.csv",
    "premium_sensitivity_summary.csv",
    "timing_slip_summary.csv",
    "target_stress_summary.csv",
    "stress_transition_summary.csv",
    "single_option_decision_summary.csv",
    "single_option_decision_path_selections.csv",
    "single_option_representative_paths.csv",
    "single_option_path_outcomes.csv",
    "single_option_required_path_to_beat_stock_1_5x.csv",
    "single_option_required_path_to_beat_stock_2_0x.csv",
    "single_option_closest_representative_path_to_edge.csv",
    "single_option_edge_gap_by_path_family.csv",
    "single_option_path_family_counts.csv",
    "single_option_timing_sensitivity.csv",
    "single_option_iv_sensitivity.csv",
    "single_option_entry_sensitivity.csv",
    "single_option_summary_bullets.csv",
}
THESIS_VIEW_TABLES = {
    "thesis_path_gallery.csv",
    "thesis_iv_gallery.csv",
    "thesis_mode_candidates.csv",
    "thesis_path_family_summary.csv",
    "thesis_iv_family_summary.csv",
    "thesis_candidate_ranking.csv",
    "max_justified_premium.csv",
    "current_vs_justified_premium.csv",
    "thesis_required_move_summary.csv",
    "thesis_stock_vs_option_summary.csv",
}
CURATED_TABLES = set(OVERVIEW_TABLES) | set(CORE_TABLES)
SECONDARY_TABLES = (
    "option_value_over_path.csv",
    "compare_vs_stock_over_path.csv",
)
SECONDARY_CHARTS = (
    "representative_stock_paths.png",
    "representative_iv_paths.png",
    "option_value_over_path.png",
    "compare_vs_stock_over_path.png",
    "strike_comparison_under_same_path.png",
    "expiry_comparison_under_same_path.png",
    "long_call_value_over_path_strike_view.png",
    "long_call_value_over_path_expiry_view.png",
    "long_call_value_over_path_best_of.png",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _workspace_root(bundle_dir: Path, model_root: Path) -> Path:
    try:
        common = os.path.commonpath([str(bundle_dir.resolve()), str(model_root.resolve())])
        return Path(common)
    except (OSError, RuntimeError, ValueError):
        return PROJECT_ROOT


def _relative_path(path: Path, workspace_root: Path) -> str:
    roots = [workspace_root, PROJECT_ROOT]
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError):
        resolved = path
    for root in roots:
        try:
            return str(resolved.relative_to(root.resolve())).replace("\\", "/")
        except (OSError, RuntimeError, ValueError):
            continue
    return resolved.name or str(resolved).replace("\\", "/")


def _sanitize_text(text: str, *, workspace_root: Path) -> str:
    def replace_match(match: re.Match[str]) -> str:
        raw = match.group(0).replace("\\\\", "\\")
        return _relative_path(Path(raw), workspace_root)

    return ABSOLUTE_WINDOWS_PATH_PATTERN.sub(replace_match, text)


def _copy_sanitized(source: Path, target: Path, *, workspace_root: Path) -> None:
    ensure_directory(target.parent)
    if source.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(source, target)
            return
        target.write_text(_sanitize_text(text, workspace_root=workspace_root), encoding="utf-8")
        return
    shutil.copy2(source, target)


def _summary_row(bundle_dir: Path) -> dict[str, str]:
    summary_path = bundle_dir / "tables" / "summary.csv"
    if not summary_path.exists():
        return {}
    try:
        import csv

        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return rows[0] if rows else {}
    except Exception:
        return {}


def _table_rows(bundle_dir: Path, filename: str) -> list[dict[str, str]]:
    table_path = bundle_dir / "tables" / filename
    if not table_path.exists():
        return []
    try:
        import csv

        with table_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _format_expiry_rows(rows: list[dict[str, str]], *, trust_label: str | None = None) -> list[str]:
    filtered = rows
    if trust_label is not None:
        filtered = [row for row in rows if clean_string(row.get("source_trust_label")) == trust_label]
    entries: list[str] = []
    for row in filtered:
        expiry = clean_string(row.get("expiry_date"))
        label = clean_string(row.get("source_trust_label"))
        note = clean_string(row.get("source_quality_note"))
        if not expiry:
            continue
        suffix = f": {label}" if label else ""
        if note:
            suffix += f" ({note})"
        entries.append(f"- `{expiry}`{suffix}")
    return entries


def _humanize_path_name(path_name: str) -> str:
    text = clean_string(path_name).strip().replace("_", " ")
    return text.title() if text else "Scenario Path"


def _path_pack_alias(row: dict[str, Any]) -> str:
    for key in (
        "compare_chart",
        "strike_value_chart",
        "strike_delta_chart",
        "expiry_value_chart",
        "expiry_delta_chart",
        "best_of_value_chart",
        "best_of_delta_chart",
        "checkpoint_table",
        "iv_value_chart",
        "iv_delta_chart",
        "iv_checkpoint_table",
        "strike_iv_value_chart",
        "strike_iv_delta_chart",
        "strike_iv_checkpoint_table",
        "expiry_iv_value_chart",
        "expiry_iv_delta_chart",
        "expiry_iv_checkpoint_table",
        "best_of_iv_value_chart",
        "best_of_iv_delta_chart",
        "best_of_iv_checkpoint_table",
        "iv_robustness_summary_table",
    ):
        value = clean_string(row.get(key))
        if "__" in value:
            return value.split("__", 1)[0]
    return clean_string(row.get("path_name")).lower().replace(" ", "_") or "scenario_path"


def _collect_path_focus_paths(
    *,
    file_map: dict[str, dict[str, str]],
    metadata_paths: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    if metadata_paths:
        focus_rows: list[dict[str, str]] = []
        for row in metadata_paths:
            path_name = clean_string(row.get("path_name"))
            if not path_name:
                continue
            focus_rows.append(
                {
                    "path_name": path_name,
                    "path_label": clean_string(row.get("path_label")) or _humanize_path_name(path_name),
                    "compare_table": clean_string(row.get("compare_table")),
                    "compare_chart": clean_string(row.get("compare_chart")),
                    "strike_value_table": clean_string(row.get("strike_value_table") or row.get("strike_table")),
                    "strike_value_chart": clean_string(row.get("strike_value_chart") or row.get("strike_chart")),
                    "strike_delta_table": clean_string(row.get("strike_delta_table")),
                    "strike_delta_chart": clean_string(row.get("strike_delta_chart")),
                    "expiry_value_table": clean_string(row.get("expiry_value_table") or row.get("expiry_table")),
                    "expiry_value_chart": clean_string(row.get("expiry_value_chart") or row.get("expiry_chart")),
                    "expiry_delta_table": clean_string(row.get("expiry_delta_table")),
                    "expiry_delta_chart": clean_string(row.get("expiry_delta_chart")),
                    "best_of_value_table": clean_string(row.get("best_of_value_table") or row.get("best_of_table")),
                    "best_of_value_chart": clean_string(row.get("best_of_value_chart") or row.get("best_of_chart")),
                    "best_of_delta_table": clean_string(row.get("best_of_delta_table")),
                    "best_of_delta_chart": clean_string(row.get("best_of_delta_chart")),
                    "checkpoint_table": clean_string(row.get("checkpoint_table")),
                    "iv_value_table": clean_string(row.get("iv_value_table")),
                    "iv_value_chart": clean_string(row.get("iv_value_chart")),
                    "iv_delta_table": clean_string(row.get("iv_delta_table")),
                    "iv_delta_chart": clean_string(row.get("iv_delta_chart")),
                    "iv_checkpoint_table": clean_string(row.get("iv_checkpoint_table")),
                    "strike_iv_value_table": clean_string(row.get("strike_iv_value_table")),
                    "strike_iv_value_chart": clean_string(row.get("strike_iv_value_chart")),
                    "strike_iv_delta_table": clean_string(row.get("strike_iv_delta_table")),
                    "strike_iv_delta_chart": clean_string(row.get("strike_iv_delta_chart")),
                    "strike_iv_checkpoint_table": clean_string(row.get("strike_iv_checkpoint_table")),
                    "expiry_iv_value_table": clean_string(row.get("expiry_iv_value_table")),
                    "expiry_iv_value_chart": clean_string(row.get("expiry_iv_value_chart")),
                    "expiry_iv_delta_table": clean_string(row.get("expiry_iv_delta_table")),
                    "expiry_iv_delta_chart": clean_string(row.get("expiry_iv_delta_chart")),
                    "expiry_iv_checkpoint_table": clean_string(row.get("expiry_iv_checkpoint_table")),
                    "best_of_iv_value_table": clean_string(row.get("best_of_iv_value_table")),
                    "best_of_iv_value_chart": clean_string(row.get("best_of_iv_value_chart")),
                    "best_of_iv_delta_table": clean_string(row.get("best_of_iv_delta_table")),
                    "best_of_iv_delta_chart": clean_string(row.get("best_of_iv_delta_chart")),
                    "best_of_iv_checkpoint_table": clean_string(row.get("best_of_iv_checkpoint_table")),
                    "iv_robustness_summary_table": clean_string(row.get("iv_robustness_summary_table")),
                }
            )
        if focus_rows:
            return focus_rows

    tables_map = file_map.get("tables") or {}
    charts_map = file_map.get("charts") or {}
    discovered: dict[str, dict[str, str]] = {}
    for filename in sorted(tables_map):
        for suffix in PATH_CENTRIC_TABLE_SUFFIXES:
            if filename.endswith(suffix):
                path_name = clean_string(filename[: -len(suffix)])
                if not path_name:
                    continue
                discovered.setdefault(
                    path_name,
                    {
                        "path_name": path_name,
                        "path_label": _humanize_path_name(path_name),
                    },
                )
                if suffix == "__compare_vs_stock_path_rows.csv":
                    discovered[path_name]["compare_table"] = filename
                elif suffix == "__long_call_strike_value.csv":
                    discovered[path_name]["strike_value_table"] = filename
                elif suffix == "__long_call_strike_delta.csv":
                    discovered[path_name]["strike_delta_table"] = filename
                elif suffix == "__long_call_expiry_value.csv":
                    discovered[path_name]["expiry_value_table"] = filename
                elif suffix == "__long_call_expiry_delta.csv":
                    discovered[path_name]["expiry_delta_table"] = filename
                elif suffix == "__long_call_best_of_value.csv":
                    discovered[path_name]["best_of_value_table"] = filename
                elif suffix == "__long_call_best_of_delta.csv":
                    discovered[path_name]["best_of_delta_table"] = filename
                elif suffix == "__path_checkpoints.csv":
                    discovered[path_name]["checkpoint_table"] = filename
                elif suffix == "__iv_path_value.csv":
                    discovered[path_name]["iv_value_table"] = filename
                elif suffix == "__iv_path_delta.csv":
                    discovered[path_name]["iv_delta_table"] = filename
                elif suffix == "__iv_checkpoints.csv":
                    discovered[path_name]["iv_checkpoint_table"] = filename
                elif suffix == "__long_call_strike_iv_value.csv":
                    discovered[path_name]["strike_iv_value_table"] = filename
                elif suffix == "__long_call_strike_iv_delta.csv":
                    discovered[path_name]["strike_iv_delta_table"] = filename
                elif suffix == "__long_call_strike_iv_checkpoints.csv":
                    discovered[path_name]["strike_iv_checkpoint_table"] = filename
                elif suffix == "__long_call_expiry_iv_value.csv":
                    discovered[path_name]["expiry_iv_value_table"] = filename
                elif suffix == "__long_call_expiry_iv_delta.csv":
                    discovered[path_name]["expiry_iv_delta_table"] = filename
                elif suffix == "__long_call_expiry_iv_checkpoints.csv":
                    discovered[path_name]["expiry_iv_checkpoint_table"] = filename
                elif suffix == "__long_call_best_of_iv_value.csv":
                    discovered[path_name]["best_of_iv_value_table"] = filename
                elif suffix == "__long_call_best_of_iv_delta.csv":
                    discovered[path_name]["best_of_iv_delta_table"] = filename
                elif suffix == "__long_call_best_of_iv_checkpoints.csv":
                    discovered[path_name]["best_of_iv_checkpoint_table"] = filename
                elif suffix == "__iv_robustness_summary.csv":
                    discovered[path_name]["iv_robustness_summary_table"] = filename
    for filename in sorted(charts_map):
        for suffix in PATH_CENTRIC_CHART_SUFFIXES:
            if filename.endswith(suffix):
                path_name = clean_string(filename[: -len(suffix)])
                if not path_name:
                    continue
                discovered.setdefault(
                    path_name,
                    {
                        "path_name": path_name,
                        "path_label": _humanize_path_name(path_name),
                    },
                )
                if suffix == "__compare_vs_stock_path_delta.png":
                    discovered[path_name]["compare_chart"] = filename
                elif suffix == "__long_call_strike_value.png":
                    discovered[path_name]["strike_value_chart"] = filename
                elif suffix == "__long_call_strike_delta.png":
                    discovered[path_name]["strike_delta_chart"] = filename
                elif suffix == "__long_call_expiry_value.png":
                    discovered[path_name]["expiry_value_chart"] = filename
                elif suffix == "__long_call_expiry_delta.png":
                    discovered[path_name]["expiry_delta_chart"] = filename
                elif suffix == "__long_call_best_of_value.png":
                    discovered[path_name]["best_of_value_chart"] = filename
                elif suffix == "__long_call_best_of_delta.png":
                    discovered[path_name]["best_of_delta_chart"] = filename
                elif suffix == "__iv_path_value.png":
                    discovered[path_name]["iv_value_chart"] = filename
                elif suffix == "__iv_path_delta.png":
                    discovered[path_name]["iv_delta_chart"] = filename
                elif suffix == "__long_call_strike_iv_value.png":
                    discovered[path_name]["strike_iv_value_chart"] = filename
                elif suffix == "__long_call_strike_iv_delta.png":
                    discovered[path_name]["strike_iv_delta_chart"] = filename
                elif suffix == "__long_call_expiry_iv_value.png":
                    discovered[path_name]["expiry_iv_value_chart"] = filename
                elif suffix == "__long_call_expiry_iv_delta.png":
                    discovered[path_name]["expiry_iv_delta_chart"] = filename
                elif suffix == "__long_call_best_of_iv_value.png":
                    discovered[path_name]["best_of_iv_value_chart"] = filename
                elif suffix == "__long_call_best_of_iv_delta.png":
                    discovered[path_name]["best_of_iv_delta_chart"] = filename
    return list(discovered.values())


def _start_here_text(
    *,
    analysis_kind: str,
    source_bundle_path: str,
    summary_row: dict[str, str],
    chain_rows: list[dict[str, str]],
    market_rows: list[dict[str, str]],
    promoted_files: list[str],
    published_dashboard_path: str | None,
    path_focus_paths: list[dict[str, str]],
) -> str:
    promoted_set = set(promoted_files)

    def available(path: str) -> bool:
        return path in promoted_set

    def numbered_available(paths: list[str]) -> list[str]:
        return [path for path in paths if available(path)]

    def bullets_for_available(items: list[tuple[str, str]]) -> list[str]:
        return [f"- `{path}`: {description}" for path, description in items if available(path)]

    if analysis_kind != "contract_selection":
        lines = [
            f"# {clean_string(summary_row.get('ticker') or 'Bundle')} {analysis_kind.replace('_', ' ').title()} Model Output",
            "",
            "Curated analyst-facing projection of a frozen canonical bundle.",
            "",
            f"- Source bundle: `{source_bundle_path}`",
            f"- Analysis kind: `{analysis_kind}`",
        ]
        if published_dashboard_path:
            lines.append(f"- Published dashboard available in source bundle: `{published_dashboard_path}`")
        lines.extend(
            [
                "",
                "## Open First",
                "",
                "1. `summary.md`",
                "2. Inspect the promoted charts in `charts/`",
                "3. Inspect the promoted tables in `tables/`",
                "",
                "## Promoted Files",
                "",
            ]
        )
        lines.extend([f"- `{path}`" for path in promoted_files])
        return "\n".join(lines) + "\n"

    headline = f"# {clean_string(summary_row.get('ticker') or 'Bundle')} {analysis_kind.replace('_', ' ').title()} Model Output"
    lines = [
        headline,
        "",
        "Curated analyst-facing projection of a frozen canonical bundle.",
        "",
        f"- Source bundle: `{source_bundle_path}`",
        f"- Analysis kind: `{analysis_kind}`",
    ]
    if published_dashboard_path:
        lines.append(f"- Published dashboard available in source bundle: `{published_dashboard_path}`")

    open_first_paths = numbered_available(
        [
            "00_core_view/bullish_action_board.md",
            "00_core_view/chain_overview.md",
            "00_core_view/chain_overview.png",
            "00_core_view/entry_justification.md",
            "01_thesis_view/thesis_mode.md",
            "00_core_view/stress_tests.md",
            "00_core_view/single_option_decision.md",
            "00_core_view/top_candidate_cards.png",
            "00_core_view/current_vs_justified_premium.png",
            "00_core_view/stock_vs_option_preference_chart.png",
            "00_core_view/stress_test_overview.png",
            "00_core_view/premium_sensitivity_chart.png",
            "00_core_view/timing_slip_chart.png",
            "00_core_view/target_stress_chart.png",
            "00_core_view/top_candidate_stress_cards.png",
            "00_core_view/single_option_decision_view.png",
            "00_core_view/required_stock_path_to_buy.png",
            "00_core_view/required_move_speed_vs_magnitude.png",
            "00_core_view/bullish_long_call_watchlist.csv",
            "00_core_view/bullish_long_call_triggers.csv",
        ]
    )
    if not open_first_paths:
        open_first_paths = ["summary.md"]

    lines.extend(
        [
            "",
            "## Open First",
            "",
        ]
    )
    lines.extend([f"{idx}. `{path}`" for idx, path in enumerate(open_first_paths, start=1)])
    next_idx = len(open_first_paths) + 1
    if any(path.startswith("01_thesis_view/") for path in promoted_set):
        lines.append(f"{next_idx}. Then use `01_thesis_view/` for deeper target-thesis charts and tables")
        next_idx += 1
    if path_focus_paths:
        lines.append(f"{next_idx}. Then choose a scenario folder under `02_path_packs/` for deeper path analysis")
        next_idx += 1
    lines.append(f"{next_idx}. Then use `03_tables/` and `04_secondary/` only as supporting detail")

    companion_views = [
        path
        for path in [
            "00_core_view/bullish_action_board.md",
            "00_core_view/chain_overview.md",
            "00_core_view/top_candidate_cards.md",
            "04_secondary/bullish_action_board_overview.png",
            "04_secondary/bullish_conviction_vs_robustness.png",
        ]
        if available(path)
    ]
    if companion_views:
        lines.extend(
            [
                "",
                "Bullish-first companion views remain available where the source bundle contains bullish long-call rows.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision Snapshot",
            "",
            f"- Best family: `{clean_string(summary_row.get('best_family') or 'n/a')}`",
            f"- Best candidate: `{clean_string(summary_row.get('best_candidate') or 'n/a')}`",
            f"- Best strike: `{clean_string(summary_row.get('best_strike') or 'n/a')}`",
            f"- Best expiry: `{clean_string(summary_row.get('best_expiry') or 'n/a')}`",
            f"- Stock benchmark note: {clean_string(summary_row.get('stock_benchmark_note') or 'n/a')}",
            "",
            "## Trust Snapshot",
            "",
            f"- Analysis trust level: `{clean_string(summary_row.get('analysis_trust_level') or 'n/a')}`",
            f"- Trusted expiries: `{clean_string(summary_row.get('trusted_expiry_count') or '0')}`",
            f"- Fallback-only expiries: `{clean_string(summary_row.get('fallback_only_expiry_count') or '0')}`",
            f"- Spot: `{clean_string(summary_row.get('spot_price_source') or 'n/a')}` via `{clean_string(summary_row.get('spot_field_used') or 'n/a')}` on `{clean_string(summary_row.get('spot_price_matched_date') or 'n/a')}`",
            f"- Risk-free: `{clean_string(summary_row.get('risk_free_rate_source') or 'n/a')}` / `{clean_string(summary_row.get('risk_free_rate_series') or 'n/a')}` on `{clean_string(summary_row.get('risk_free_rate_matched_date') or 'n/a')}`",
        ]
    )
    if market_rows:
        market = market_rows[0]
        lines.extend(
            [
                f"- Same-day IBKR spot rejection: {clean_string(market.get('ibkr_same_day_spot_rejected_reason') or 'n/a')}",
                f"- Trust note: {clean_string(market.get('analysis_trust_note') or summary_row.get('analysis_trust_note') or 'n/a')}",
            ]
        )
    trusted_lines = _format_expiry_rows(chain_rows, trust_label="trusted_quoted") + _format_expiry_rows(chain_rows, trust_label="quoted_prior_day")
    fallback_lines = _format_expiry_rows(chain_rows, trust_label="fallback_only")
    if trusted_lines:
        lines.extend(["", "## Trusted Expiries", ""] + trusted_lines)
    if fallback_lines:
        lines.extend(["", "## Fallback-Only Expiries", ""] + fallback_lines)
    if path_focus_paths:
        lines.extend(["", "## Path Packs", ""])
        for row in path_focus_paths:
            path_label = clean_string(row.get("path_label")) or _humanize_path_name(clean_string(row.get("path_name")))
            alias = _path_pack_alias(row)
            lines.append(f"- {path_label}: `02_path_packs/{alias}/README.md`")
    inspect_items = bullets_for_available(
        [
            ("00_core_view/bullish_action_board.md", "primary bullish long-call shortlist under the active assumptions"),
            ("00_core_view/chain_overview.md", "compare bullish calls side by side against stock across the same representative path families"),
            ("00_core_view/chain_overview.png", "six top compare-options cards plus verdict distribution"),
            ("00_core_view/chain_overview_candidates.csv", "compact contract-by-contract compare-options table"),
            ("00_core_view/chain_overview_summary.csv", "frozen payload behind the six summary cards"),
            ("00_core_view/entry_justification.md", "what the stock actually has to do before the calls look worth buying"),
            ("00_core_view/single_option_decision.md", "one selected call tested directly against buying stock across curated decision paths"),
            ("00_core_view/single_option_decision_view.png", "hero view showing curated decision paths plus required 1.5x/2.0x option-over-stock edge paths"),
            ("00_core_view/single_option_decision_path_selections.csv", "the 5-8 selected decision paths with family, outcome, score, and reason"),
            ("00_core_view/single_option_required_path_to_beat_stock_1_5x.csv", "minimum stock path required for the selected call to beat stock after the meaningful bullish stock-PnL floor"),
            ("00_core_view/single_option_required_path_to_beat_stock_2_0x.csv", "strong-edge stock path required for the selected call after the meaningful bullish stock-PnL floor"),
            ("00_core_view/single_option_closest_representative_path_to_edge.csv", "closest curated path when the option misses the stock benchmark edge"),
            ("00_core_view/single_option_edge_gap_by_path_family.csv", "path-family gaps showing whether misses need more stock move, earlier timing, IV support, or entry discount"),
            ("00_core_view/top_candidate_cards.png", "card-style first read for contract, why, trigger, trust, and stock note"),
            ("00_core_view/current_vs_justified_premium.png", "current premium versus thesis-justified max premium"),
            ("00_core_view/stock_vs_option_preference_chart.png", "stock versus option preference read"),
            ("00_core_view/required_stock_path_to_buy.png", "required path versus active assumed path for the leading bullish calls"),
            ("00_core_view/required_move_speed_vs_magnitude.png", "how fast and how far the stock needs to move"),
            ("00_core_view/bullish_long_call_watchlist.csv", "bullish calls worth monitoring"),
            ("00_core_view/bullish_long_call_triggers.csv", "richer bullish trigger map"),
            ("01_thesis_view/thesis_mode.md", "thesis-first read for the explicit target price/date and justified-entry premium"),
            ("01_thesis_view/thesis_candidate_overview.png", "compact thesis-mode contract picker"),
            ("01_thesis_view/thesis_path_gallery.png", "multiple routes to the same thesis endpoint"),
            ("01_thesis_view/thesis_stock_vs_option.png", "where stock still beats calls even if the thesis endpoint is reached"),
            ("01_thesis_view/current_vs_justified_premium.png", "premium gap inside the thesis view"),
            ("01_thesis_view/thesis_candidate_ranking.csv", "thesis-mode ranking table"),
            ("01_thesis_view/current_vs_justified_premium.csv", "current-vs-justified premium table"),
            ("04_secondary/bullish_action_board_overview.png", "full bullish scorecard with Why / Warning / Trigger / Trust"),
            ("04_secondary/bullish_buy_watch_avoid_matrix.png", "bullish buy / watch / avoid support read"),
            ("04_secondary/bullish_conviction_vs_robustness.png", "bullish long calls plus stock baseline, separated from other structures"),
            ("04_secondary/bullish_trigger_map.png", "full trigger map for bullish watchlist names"),
            ("03_tables/bullish_long_call_action_board.csv", "bullish long-call board with reasons and triggers"),
            ("03_tables/bullish_long_call_avoid.csv", "bullish calls to keep out of the active shortlist"),
            ("03_tables/top_candidate_cards.csv", "source table behind the card visual"),
            ("03_tables/single_option_path_outcomes.csv", "path-by-path option-vs-stock outcomes behind the single-option hero view"),
            ("03_tables/single_option_edge_gap_by_path_family.csv", "family-by-family required-edge gap behind the single-option hero view"),
            ("03_tables/single_option_iv_sensitivity.csv", "low/base/high IV sensitivity for the selected option"),
            ("03_tables/single_option_entry_sensitivity.csv", "cheap/reference/expensive entry sensitivity for the selected option"),
            ("03_tables/stock_path_library.csv", "broad stock-path library metadata with family and timing labels"),
            ("04_secondary/required_move_vs_stock_chart.png", "whether stock still dominates even after the required path is met"),
            ("04_secondary/strike_expiry_entry_barrier_map.png", "which strike/expiry pairs ask the least or most from the stock path"),
            ("04_secondary/iv_support_requirement_chart.png", "which calls need IV support versus which tolerate lower IV"),
            ("03_tables/required_move_summary.csv", "numeric entry-speed and magnitude read"),
            ("03_tables/required_move_vs_stock.csv", "stock-cleaner versus option-edge read after the required path is achieved"),
            ("03_tables/required_iv_support_summary.csv", "lower-IV penalty versus friendlier-IV relief"),
            ("03_tables/entry_barrier_summary.csv", "compact strike/expiry barrier table"),
            ("04_secondary/other_structures.md", "covered calls / CSPs / spreads as secondary context"),
            ("summary.md", "compact full-bundle summary after the action-board read"),
            ("04_secondary/highlights.md", "broader robustness highlights before the path-pack detail"),
            ("04_secondary/highlights_overview.png", "compact category-to-candidate decision map"),
            ("04_secondary/candidate_robustness_vs_upside.png", "aggressive upside versus robustness, with stock as baseline"),
            ("04_secondary/stock_vs_option_decision_chart.png", "where options beat or lag the long-stock benchmark"),
            ("04_secondary/stock_path_gallery.png", "scenario menu for deliberate named stock paths"),
            ("04_secondary/iv_path_gallery.png", "named IV-regime menu, kept separate from stock paths"),
            ("04_secondary/required_path_vs_assumed_path.png", "required-path sanity check"),
            ("03_tables/chain_source_summary.csv", "per-expiry source, trust label, coverage, and fallback reason"),
            ("03_tables/market_context_summary.csv", "bundle-level spot, rate, and metadata context"),
            ("03_tables/family_comparison.csv", "family-level decision read under current assumptions"),
            ("03_tables/candidate_comparison.csv", "exact candidate ranking and benchmark comparison"),
        ]
    )
    lines.extend(["", "## Inspect Next", ""])
    if inspect_items:
        lines.extend(inspect_items)
    else:
        lines.append("- `model_output_manifest.json`: complete promoted file list")
    if path_focus_paths:
        lines.extend(
            [
                "- `02_path_packs/<path>/iv_path_value.png`: same stock path and same anchor long call, but multiple IV regimes",
                "- `02_path_packs/<path>/iv_path_delta.png`: stock-relative version of the IV-regime comparison",
                "- `02_path_packs/<path>/long_call_strike_iv_value.png`: strike ladder where IV is an explicit axis",
                "- `02_path_packs/<path>/long_call_expiry_iv_value.png`: expiry ladder where IV is an explicit axis",
                "- `02_path_packs/<path>/long_call_best_of_iv_value.png`: best-of long-call set across IV regimes",
                "- `02_path_packs/<path>/iv_robustness_summary.csv`: compact decision read for IV robustness and stock dominance",
            ]
        )
    lines.extend(
        [
            "- `04_secondary/`: representative-path and broad support artifacts, intentionally behind the core and thesis views",
            "",
            "## Full Manifest",
            "",
            "The complete promoted file list is in `model_output_manifest.json`. This guide only lists files that were actually promoted from the source bundle.",
        ]
    )
    return "\n".join(lines) + "\n"


def _selected_targets(
    *,
    bundle_dir: Path,
    file_map: dict[str, dict[str, str]],
    analysis_kind: str,
    path_focus_paths: list[dict[str, str]] | None = None,
) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    copies: list[tuple[Path, Path]] = []
    promoted_files: list[str] = []
    missing_files: list[str] = []

    summary_map = file_map.get("summary") or {}
    tables_map = file_map.get("tables") or {}
    charts_map = file_map.get("charts") or {}

    def add_copy(section_map: dict[str, str], filename: str, target: Path) -> None:
        rel = section_map.get(filename)
        if rel:
            target_text = target.as_posix()
            if target_text not in promoted_files:
                copies.append((bundle_dir / rel, target))
                promoted_files.append(target_text)
        else:
            missing_files.append(filename)

    for filename in PRIMARY_MODEL_ARTIFACTS["summary"]:
        if filename in CONTRACT_SELECTION_OVERVIEW_SUMMARIES:
            continue
        rel = summary_map.get(filename)
        if rel:
            target = Path(filename)
            copies.append((bundle_dir / rel, target))
            promoted_files.append(target.as_posix())
        else:
            missing_files.append(filename)

    if clean_string(analysis_kind) != "contract_selection":
        for filename in PRIMARY_MODEL_ARTIFACTS["tables"]:
            target = Path("summary.csv") if filename == "summary.csv" else Path("tables") / filename
            add_copy(tables_map, filename, target)
        for filename in PRIMARY_MODEL_ARTIFACTS["charts"]:
            add_copy(charts_map, filename, Path("charts") / filename)
        return copies, sorted(promoted_files), sorted(missing_files)

    add_copy(tables_map, "summary.csv", Path("summary.csv"))

    for filename in CORE_VIEW_SUMMARIES:
        add_copy(summary_map, filename, Path("00_core_view") / filename)
    for filename in THESIS_VIEW_SUMMARIES:
        add_copy(summary_map, filename, Path("01_thesis_view") / filename)
    for filename in SECONDARY_VIEW_SUMMARIES:
        add_copy(summary_map, filename, Path("04_secondary") / filename)

    for filename in CORE_VIEW_CHARTS:
        add_copy(charts_map, filename, Path("00_core_view") / filename)
    for filename in THESIS_VIEW_CHARTS:
        add_copy(charts_map, filename, Path("01_thesis_view") / filename)
    for filename in SECONDARY_VIEW_CHARTS:
        add_copy(charts_map, filename, Path("04_secondary") / filename)

    for filename in CORE_VIEW_TABLES:
        add_copy(tables_map, filename, Path("00_core_view") / filename)
    for filename in THESIS_VIEW_TABLES:
        add_copy(tables_map, filename, Path("01_thesis_view") / filename)
    for filename in sorted(CURATED_TABLES - CORE_VIEW_TABLES - THESIS_VIEW_TABLES):
        add_copy(tables_map, filename, Path("03_tables") / filename)
    for filename in SECONDARY_TABLES:
        add_copy(tables_map, filename, Path("04_secondary") / filename)
    for filename in SECONDARY_CHARTS:
        add_copy(charts_map, filename, Path("04_secondary") / filename)

    for row in path_focus_paths or []:
        alias = _path_pack_alias(row)
        pack_dir = Path("02_path_packs") / alias
        path_targets = [
            ("compare_table", "compare_vs_stock_delta.csv", tables_map),
            ("compare_chart", "compare_vs_stock_delta.png", charts_map),
            ("strike_value_table", "long_call_strike_value.csv", tables_map),
            ("strike_value_chart", "long_call_strike_value.png", charts_map),
            ("strike_delta_table", "long_call_strike_delta.csv", tables_map),
            ("strike_delta_chart", "long_call_strike_delta.png", charts_map),
            ("expiry_value_table", "long_call_expiry_value.csv", tables_map),
            ("expiry_value_chart", "long_call_expiry_value.png", charts_map),
            ("expiry_delta_table", "long_call_expiry_delta.csv", tables_map),
            ("expiry_delta_chart", "long_call_expiry_delta.png", charts_map),
            ("best_of_value_table", "long_call_best_of_value.csv", tables_map),
            ("best_of_value_chart", "long_call_best_of_value.png", charts_map),
            ("best_of_delta_table", "long_call_best_of_delta.csv", tables_map),
            ("best_of_delta_chart", "long_call_best_of_delta.png", charts_map),
            ("checkpoint_table", "checkpoints.csv", tables_map),
            ("iv_value_table", "iv_path_value.csv", tables_map),
            ("iv_value_chart", "iv_path_value.png", charts_map),
            ("iv_delta_table", "iv_path_delta.csv", tables_map),
            ("iv_delta_chart", "iv_path_delta.png", charts_map),
            ("iv_checkpoint_table", "iv_checkpoints.csv", tables_map),
            ("iv_robustness_summary_table", "iv_robustness_summary.csv", tables_map),
            ("strike_iv_value_table", "long_call_strike_iv_value.csv", tables_map),
            ("strike_iv_value_chart", "long_call_strike_iv_value.png", charts_map),
            ("strike_iv_delta_table", "long_call_strike_iv_delta.csv", tables_map),
            ("strike_iv_delta_chart", "long_call_strike_iv_delta.png", charts_map),
            ("strike_iv_checkpoint_table", "long_call_strike_iv_checkpoints.csv", tables_map),
            ("expiry_iv_value_table", "long_call_expiry_iv_value.csv", tables_map),
            ("expiry_iv_value_chart", "long_call_expiry_iv_value.png", charts_map),
            ("expiry_iv_delta_table", "long_call_expiry_iv_delta.csv", tables_map),
            ("expiry_iv_delta_chart", "long_call_expiry_iv_delta.png", charts_map),
            ("expiry_iv_checkpoint_table", "long_call_expiry_iv_checkpoints.csv", tables_map),
            ("best_of_iv_value_table", "long_call_best_of_iv_value.csv", tables_map),
            ("best_of_iv_value_chart", "long_call_best_of_iv_value.png", charts_map),
            ("best_of_iv_delta_table", "long_call_best_of_iv_delta.csv", tables_map),
            ("best_of_iv_delta_chart", "long_call_best_of_iv_delta.png", charts_map),
            ("best_of_iv_checkpoint_table", "long_call_best_of_iv_checkpoints.csv", tables_map),
        ]
        for metadata_key, target_name, source_map in path_targets:
            filename = clean_string(row.get(metadata_key))
            if not filename:
                continue
            add_copy(source_map, filename, pack_dir / target_name)

    if not path_focus_paths:
        for filename, rel in sorted(tables_map.items()):
            if any(filename.endswith(suffix) for suffix in PATH_CENTRIC_TABLE_SUFFIXES):
                path_name = filename.split("__", 1)[0]
                target = Path("02_path_packs") / path_name / filename
                if target.as_posix() not in promoted_files:
                    copies.append((bundle_dir / rel, target))
                    promoted_files.append(target.as_posix())
        for filename, rel in sorted(charts_map.items()):
            if any(filename.endswith(suffix) for suffix in PATH_CENTRIC_CHART_SUFFIXES):
                path_name = filename.split("__", 1)[0]
                target = Path("02_path_packs") / path_name / filename
                if target.as_posix() not in promoted_files:
                    copies.append((bundle_dir / rel, target))
                    promoted_files.append(target.as_posix())

    return copies, sorted(promoted_files), sorted(missing_files)


def _archive_entry(
    *,
    source_bundle_path: str,
    promoted_dir: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_bundle_path": source_bundle_path,
        "promoted_dir": promoted_dir,
        "ticker": clean_string(manifest.get("ticker")).upper(),
        "snapshot_date": clean_string(manifest.get("snapshot_date")),
        "analysis_kind": clean_string(manifest.get("analysis_kind")),
        "run_slug": clean_string(manifest.get("run_slug")),
        "updated_at": manifest.get("generated_at"),
        "analysis_trust_level": manifest.get("analysis_trust_level"),
    }


def _path_pack_readme_text(row: dict[str, str]) -> str:
    path_label = clean_string(row.get("path_label")) or _humanize_path_name(clean_string(row.get("path_name")))
    return "\n".join(
        [
            f"# {path_label} Path Pack",
            "",
            "This folder is a curated projection from the frozen contract-selection bundle. It does not recompute analysis.",
            "",
            "## Read In This Order",
            "",
            "1. `compare_vs_stock_delta.png`",
            "2. `long_call_strike_value.png`",
            "3. `long_call_expiry_value.png`",
            "4. `long_call_best_of_value.png`",
            "5. `iv_path_value.png`",
            "6. `iv_path_delta.png`",
            "7. `long_call_strike_iv_value.png`",
            "8. `long_call_expiry_iv_value.png`",
            "9. `long_call_best_of_iv_value.png`",
            "10. `iv_robustness_summary.csv`",
            "11. `long_call_strike_delta.png`",
            "12. `long_call_expiry_delta.png`",
            "13. `long_call_best_of_delta.png`",
            "14. `long_call_strike_iv_delta.png`",
            "15. `long_call_expiry_iv_delta.png`",
            "16. `long_call_best_of_iv_delta.png`",
            "17. `checkpoints.csv`",
            "18. `iv_checkpoints.csv` and the family-specific IV checkpoint CSVs",
            "",
            "Value charts show modeled option value over this stock path. Delta charts show option PnL minus the long-stock benchmark over the same stock path. The single-anchor IV charts isolate pure IV effect; the IV-expanded ladder charts keep the stock path fixed while varying both the selected comparison set and IV regimes in a controlled, chart-curated way.",
        ]
    ) + "\n"


def _update_archive_history(archive_path: Path, entry: dict[str, Any]) -> None:
    payload = {"promotions": []}
    if archive_path.exists():
        payload = _load_json(archive_path)
    promotions = payload.get("promotions") or []
    promotions = [item for item in promotions if item.get("source_bundle_path") != entry["source_bundle_path"]]
    promotions.append(entry)
    payload["promotions"] = promotions
    ensure_directory(archive_path.parent)
    archive_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_model_outputs(
    *,
    bundle: str | Path | None = None,
    ticker: str | None = None,
    snapshot_date: str | None = None,
    analysis_kind: str | None = None,
    run_slug: str | None = None,
    analysis_root: str | Path = DEFAULT_ANALYSIS_OUTPUT_ROOT,
    model_root: str | Path = DEFAULT_MODEL_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Project a frozen analysis bundle into a curated model_outputs folder."""

    bundle_dir = resolve_analysis_bundle(
        bundle=bundle,
        ticker=ticker,
        snapshot_date=snapshot_date,
        analysis_kind=analysis_kind,
        run_slug=run_slug,
        output_root=analysis_root,
    )
    model_root_path = Path(model_root)
    workspace_root = _workspace_root(bundle_dir, model_root_path)

    bundle_manifest = _load_json(bundle_dir / "bundle_manifest.json")
    report_metadata = _load_json(bundle_dir / "metadata" / "report_metadata.json")
    metadata = report_metadata.get("metadata") or {}
    path_focus_paths = _collect_path_focus_paths(
        file_map=bundle_manifest.get("file_map") or {},
        metadata_paths=(report_metadata.get("path_centric_focus_paths") or metadata.get("path_centric_focus_paths") or None),
    )

    resolved_ticker = clean_string(bundle_manifest.get("ticker")).upper()
    resolved_snapshot = clean_string(bundle_manifest.get("snapshot_date"))
    resolved_kind = clean_string(bundle_manifest.get("analysis_kind"))
    resolved_run_slug = clean_string(bundle_manifest.get("run_slug") or bundle_dir.name)
    source_bundle_path = _relative_path(bundle_dir, workspace_root)
    source_publish_dashboard = bundle_dir / "publish" / "dashboard.html"
    published_dashboard_path = _relative_path(source_publish_dashboard, workspace_root) if source_publish_dashboard.exists() else None

    promoted_dir = model_root_path / resolved_ticker / f"snapshot_{resolved_snapshot}" / resolved_kind / resolved_run_slug
    latest_dir = model_root_path / resolved_ticker / "latest"
    archive_path = model_root_path / resolved_ticker / "archive" / "promoted_runs.json"

    if promoted_dir.exists():
        shutil.rmtree(promoted_dir)
    ensure_directory(promoted_dir)

    copies, promoted_files, missing_files = _selected_targets(
        bundle_dir=bundle_dir,
        file_map=bundle_manifest.get("file_map") or {},
        analysis_kind=resolved_kind,
        path_focus_paths=path_focus_paths,
    )
    for source, relative_target in copies:
        if source.exists():
            _copy_sanitized(source, promoted_dir / relative_target, workspace_root=workspace_root)

    if resolved_kind == "contract_selection":
        for row in path_focus_paths:
            alias = _path_pack_alias(row)
            pack_dir = promoted_dir / "02_path_packs" / alias
            if not pack_dir.exists():
                continue
            readme_path = pack_dir / "README.md"
            readme_path.write_text(
                _sanitize_text(_path_pack_readme_text(row), workspace_root=workspace_root),
                encoding="utf-8",
            )
            promoted_files.append((Path("02_path_packs") / alias / "README.md").as_posix())
        promoted_files = sorted(dict.fromkeys(promoted_files))

    summary_row = _summary_row(bundle_dir)
    chain_rows = _table_rows(bundle_dir, "chain_source_summary.csv")
    market_rows = _table_rows(bundle_dir, "market_context_summary.csv")
    start_here_path = promoted_dir / "START_HERE.md"
    start_here_path.write_text(
        _sanitize_text(
            _start_here_text(
                analysis_kind=resolved_kind,
                source_bundle_path=source_bundle_path,
                summary_row=summary_row,
                chain_rows=chain_rows,
                market_rows=market_rows,
                promoted_files=promoted_files,
                published_dashboard_path=published_dashboard_path,
                path_focus_paths=path_focus_paths,
            ),
            workspace_root=workspace_root,
        ),
        encoding="utf-8",
    )
    if resolved_kind == "contract_selection":
        core_start = promoted_dir / "00_core_view" / "START_HERE.md"
        ensure_directory(core_start.parent)
        core_start.write_text(start_here_path.read_text(encoding="utf-8"), encoding="utf-8")
        promoted_files.append("00_core_view/START_HERE.md")
        promoted_files = sorted(dict.fromkeys(promoted_files))

    manifest_payload = {
        "source_bundle_path": source_bundle_path,
        "source_bundle_manifest": _relative_path(bundle_dir / "bundle_manifest.json", workspace_root),
        "source_snapshot_date": resolved_snapshot,
        "analysis_kind": resolved_kind,
        "ticker": resolved_ticker,
        "run_slug": resolved_run_slug,
        "generated_at": report_metadata.get("generated_at"),
        "promoted_files": promoted_files,
        "missing_primary_files": missing_files,
        "analysis_trust_level": clean_string(summary_row.get("analysis_trust_level") or metadata.get("analysis_trust_level")),
        "trusted_expiry_count": int(float(clean_string(summary_row.get("trusted_expiry_count") or metadata.get("trusted_expiry_count") or "0") or 0)),
        "fallback_only_expiry_count": int(float(clean_string(summary_row.get("fallback_only_expiry_count") or metadata.get("fallback_only_expiry_count") or "0") or 0)),
        "spot_source": clean_string(summary_row.get("spot_price_source") or metadata.get("spot_price_source")),
        "spot_field_used": clean_string(summary_row.get("spot_field_used") or metadata.get("spot_field_used")),
        "spot_price_matched_date": clean_string(summary_row.get("spot_price_matched_date") or metadata.get("spot_price_matched_date")),
        "risk_free_source": clean_string(summary_row.get("risk_free_rate_source") or metadata.get("risk_free_rate_source")),
        "risk_free_series": clean_string(summary_row.get("risk_free_rate_series") or metadata.get("risk_free_rate_series")),
        "risk_free_rate_matched_date": clean_string(summary_row.get("risk_free_rate_matched_date") or metadata.get("risk_free_rate_matched_date")),
        "published_dashboard_path": published_dashboard_path,
        "path_centric_focus_paths": path_focus_paths,
    }
    manifest_path = promoted_dir / "model_output_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(windows_extended_path(promoted_dir), windows_extended_path(latest_dir))

    _update_archive_history(
        archive_path,
        _archive_entry(
            source_bundle_path=source_bundle_path,
            promoted_dir=_relative_path(promoted_dir, workspace_root),
            manifest=manifest_payload,
        ),
    )

    return {
        "analysis_kind": resolved_kind,
        "ticker": resolved_ticker,
        "snapshot_date": resolved_snapshot,
        "run_slug": resolved_run_slug,
        "source_bundle_dir": str(bundle_dir),
        "source_bundle_path": source_bundle_path,
        "model_output_dir": str(promoted_dir),
        "latest_dir": str(latest_dir),
        "start_here_path": str(start_here_path),
        "manifest_path": str(manifest_path),
        "promoted_files": promoted_files,
    }
