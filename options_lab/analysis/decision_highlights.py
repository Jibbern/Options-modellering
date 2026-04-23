"""Decision-first highlight layer for contract-selection bundles.

The functions in this module consume already-computed contract-selection tables.
They do not fetch data, reprice contracts, or publish anything.  The goal is to
turn the path/IV evidence into transparent, assumption-relative decision reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..utils import clean_string, finite_or_none


HIGHLIGHT_CATEGORY_ORDER = [
    "most_robust_call",
    "best_aggressive_upside_call",
    "best_balanced_call",
    "best_if_move_is_late",
    "best_if_iv_falls",
    "best_if_iv_rises",
    "best_if_you_want_simplest_exposure",
    "stock_still_best_baseline",
    "most_fragile_call",
    "requires_fast_move",
    "requires_iv_support",
]

HIGHLIGHT_LABELS = {
    "most_robust_call": "Most Robust Call",
    "best_aggressive_upside_call": "Best Aggressive Upside Call",
    "best_balanced_call": "Best Balanced Call",
    "best_if_move_is_late": "Best If Move Is Late",
    "best_if_iv_falls": "Best If IV Falls",
    "best_if_iv_rises": "Best If IV Rises",
    "best_if_you_want_simplest_exposure": "Best Simplest Exposure",
    "stock_still_best_baseline": "Stock Still Best Baseline",
    "most_fragile_call": "Most Fragile Call",
    "requires_fast_move": "Requires Fast Move",
    "requires_iv_support": "Requires IV Support",
}

ACTION_BUCKET_ORDER = {
    "Buy Now": 1,
    "Watchlist": 2,
    "Prefer Stock Instead": 3,
    "Avoid For Now": 4,
}

TRUST_SCORE = {
    "trusted_quoted": 100.0,
    "quoted_prior_day": 82.0,
    "same_day_quoted": 100.0,
    "prior_day_quoted": 82.0,
    "fallback_only": 35.0,
    "structure_only": 20.0,
}


@dataclass
class DecisionHighlightOutputs:
    """Decision-highlight artifact frames and analyst-facing markdown."""

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


def _num(value: Any, default: float = 0.0) -> float:
    numeric = finite_or_none(value)
    if numeric is None:
        return default
    return float(numeric)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_string(value).lower()
    return text in {"true", "1", "yes", "y"}


def _normalize(series: pd.Series, *, default: float = 50.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(default, index=series.index, dtype=float)
    filled = values.fillna(values.median())
    low = float(filled.min())
    high = float(filled.max())
    if abs(high - low) < 1e-9:
        return pd.Series(default, index=series.index, dtype=float)
    return ((filled - low) / (high - low) * 100.0).clip(0.0, 100.0)


def _mean_bool(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.map(_bool).mean())


def _tag_join(tags: list[str]) -> str:
    return " | ".join(dict.fromkeys(clean_string(tag) for tag in tags if clean_string(tag)))


def _candidate_short_label(row: pd.Series | dict[str, Any]) -> str:
    label = clean_string(row.get("candidate_label"))
    if label:
        return label
    family = clean_string(row.get("strategy_family"))
    strike = clean_string(row.get("strike_label"))
    expiry = clean_string(row.get("expiry_date"))
    if family == "long_stock":
        return "Long Stock Baseline"
    if strike or expiry:
        return " ".join(part for part in [family.replace("_", " ").title(), strike, expiry] if part)
    return clean_string(row.get("candidate_slug")) or "Candidate"


def _iv_robustness_frames(path_view_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for filename, frame in sorted((path_view_tables or {}).items()):
        if not clean_string(filename).lower().endswith("__iv_robustness_summary.csv"):
            continue
        if frame is None or frame.empty:
            continue
        working = frame.copy()
        if "source_file" not in working.columns:
            working["source_file"] = filename
        frames.append(working)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_candidate_robustness_summary(
    candidate_comparison: pd.DataFrame,
    path_view_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    base = candidate_comparison.copy() if candidate_comparison is not None else pd.DataFrame()
    if base.empty:
        return pd.DataFrame()

    iv_rows = _iv_robustness_frames(path_view_tables)
    agg_rows: list[dict[str, Any]] = []
    if not iv_rows.empty and "candidate_slug" in iv_rows.columns:
        numeric_cols = [
            "iv_path_count",
            "profitable_iv_path_count",
            "beat_stock_iv_path_count",
            "terminal_value_min",
            "terminal_value_max",
            "terminal_value_range",
            "terminal_delta_vs_stock_min",
            "terminal_delta_vs_stock_max",
            "terminal_delta_vs_stock_range",
        ]
        for col in numeric_cols:
            if col in iv_rows.columns:
                iv_rows[col] = pd.to_numeric(iv_rows[col], errors="coerce")
        for candidate_slug, group in iv_rows.groupby("candidate_slug", dropna=False):
            iv_count = float(group.get("iv_path_count", pd.Series(dtype=float)).fillna(0).sum())
            profit_count = float(group.get("profitable_iv_path_count", pd.Series(dtype=float)).fillna(0).sum())
            beat_count = float(group.get("beat_stock_iv_path_count", pd.Series(dtype=float)).fillna(0).sum())
            agg_rows.append(
                {
                    "candidate_slug": clean_string(candidate_slug),
                    "path_count_tested": int(group.get("stock_path_name", pd.Series(dtype=str)).nunique()),
                    "iv_view_count": int(len(group.index)),
                    "iv_opportunity_count": int(iv_count),
                    "profitable_iv_path_count": int(profit_count),
                    "beat_stock_iv_path_count": int(beat_count),
                    "profitable_iv_path_rate": round(profit_count / iv_count, 4) if iv_count else 0.0,
                    "beat_stock_iv_path_rate": round(beat_count / iv_count, 4) if iv_count else 0.0,
                    "lower_iv_survival_rate": round(_mean_bool(group.get("lower_iv_profitable", pd.Series(dtype=bool))), 4),
                    "lower_iv_beat_stock_rate": round(_mean_bool(group.get("lower_iv_beats_stock", pd.Series(dtype=bool))), 4),
                    "high_iv_dependency_rate": round(_mean_bool(group.get("high_iv_dependency", pd.Series(dtype=bool))), 4),
                    "terminal_value_min": finite_or_none(group.get("terminal_value_min", pd.Series(dtype=float)).min()),
                    "terminal_value_max": finite_or_none(group.get("terminal_value_max", pd.Series(dtype=float)).max()),
                    "terminal_value_range": finite_or_none(group.get("terminal_value_range", pd.Series(dtype=float)).max()),
                    "terminal_delta_vs_stock_min": finite_or_none(group.get("terminal_delta_vs_stock_min", pd.Series(dtype=float)).min()),
                    "terminal_delta_vs_stock_max": finite_or_none(group.get("terminal_delta_vs_stock_max", pd.Series(dtype=float)).max()),
                    "terminal_delta_vs_stock_range": finite_or_none(group.get("terminal_delta_vs_stock_range", pd.Series(dtype=float)).max()),
                    "best_iv_path": clean_string(group.get("best_iv_path", pd.Series(dtype=str)).mode().iloc[0]) if not group.get("best_iv_path", pd.Series(dtype=str)).dropna().empty else "",
                    "worst_iv_path": clean_string(group.get("worst_iv_path", pd.Series(dtype=str)).mode().iloc[0]) if not group.get("worst_iv_path", pd.Series(dtype=str)).dropna().empty else "",
                    "iv_robustness_labels": _tag_join(group.get("iv_robustness_label", pd.Series(dtype=str)).astype(str).tolist()),
                    "iv_robustness_notes": _tag_join(group.get("iv_robustness_note", pd.Series(dtype=str)).astype(str).tolist()),
                }
            )
    robustness = pd.DataFrame(agg_rows)
    merged = base.merge(robustness, on="candidate_slug", how="left") if not robustness.empty else base.copy()
    fill_zero = [
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
        "terminal_value_range",
        "terminal_delta_vs_stock_range",
    ]
    for col in fill_zero:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    is_stock = merged.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_stock")
    focus_count = max(1, int(merged["path_count_tested"].max() or 1))
    merged.loc[is_stock, "path_count_tested"] = focus_count
    merged.loc[is_stock, "profitable_iv_path_rate"] = 1.0
    merged.loc[is_stock, "beat_stock_iv_path_rate"] = 1.0
    merged.loc[is_stock, "lower_iv_survival_rate"] = 1.0
    merged.loc[is_stock, "lower_iv_beat_stock_rate"] = 1.0
    merged.loc[is_stock, "high_iv_dependency_rate"] = 0.0
    merged.loc[is_stock, "iv_robustness_labels"] = "stock_baseline_no_iv_dependency"
    merged.loc[is_stock, "iv_robustness_notes"] = "long stock has no option IV or expiry dependency"
    return merged


def _build_tradeoff_matrix(robustness: pd.DataFrame) -> pd.DataFrame:
    if robustness.empty:
        return pd.DataFrame()
    frame = robustness.copy()
    for col in [
        "objective_score",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "premium_or_entry_cost",
        "capital_required",
        "unit_capital_required",
        "max_loss",
        "delayed_move_value_change",
        "iv_down_value_change",
        "iv_up_value_change",
        "timing_match_ratio",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["source_trust_label"] = frame.get("source_trust_label", "").astype(str)
    frame["trust_score"] = frame["source_trust_label"].str.lower().map(TRUST_SCORE).fillna(55.0)
    frame["upside_score"] = _normalize(frame.get("return_on_comparison_capital", pd.Series(0, index=frame.index)))
    objective_norm = _normalize(frame.get("objective_score", pd.Series(0, index=frame.index)))
    frame["stock_edge_score"] = _normalize(frame.get("difference_vs_stock", pd.Series(0, index=frame.index)))
    frame["iv_robustness_score"] = (
        frame["profitable_iv_path_rate"] * 45.0
        + frame["beat_stock_iv_path_rate"] * 35.0
        + frame["lower_iv_survival_rate"] * 20.0
    ).clip(0.0, 100.0)
    frame["lower_iv_resilience_score"] = (
        frame["lower_iv_survival_rate"] * 65.0
        + frame["lower_iv_beat_stock_rate"] * 35.0
        - frame["high_iv_dependency_rate"] * 20.0
    ).clip(0.0, 100.0)
    terminal_range = _normalize(frame.get("terminal_delta_vs_stock_range", pd.Series(0, index=frame.index)))
    frame["iv_upside_score"] = (
        _normalize(frame.get("terminal_delta_vs_stock_max", pd.Series(0, index=frame.index))) * 0.65
        + terminal_range * 0.25
        + frame["trust_score"] * 0.10
    ).clip(0.0, 100.0)
    weak_horizon = frame.get("weak_horizon_fit", pd.Series(False, index=frame.index)).map(_bool)
    target_beyond = frame.get("target_beyond_expiry", pd.Series(False, index=frame.index)).map(_bool)
    clamped = frame.get("clamped_to_expiry", pd.Series(False, index=frame.index)).map(_bool)
    frame["time_resilience_score"] = (
        100.0
        - weak_horizon.astype(float) * 28.0
        - target_beyond.astype(float) * 32.0
        - clamped.astype(float) * 18.0
        + pd.to_numeric(frame.get("timing_match_ratio", pd.Series(1.0, index=frame.index)), errors="coerce").fillna(1.0).clip(0, 1) * 10.0
    ).clip(0.0, 100.0)
    frame["capital_efficiency_score"] = (
        _normalize(frame.get("return_on_comparison_capital", pd.Series(0, index=frame.index))) * 0.65
        + _normalize(-frame.get("capital_required", pd.Series(0, index=frame.index)).fillna(0)) * 0.35
    ).clip(0.0, 100.0)
    penalties = (
        (100.0 - frame["trust_score"]) * 0.14
        + weak_horizon.astype(float) * 14.0
        + target_beyond.astype(float) * 18.0
        + clamped.astype(float) * 8.0
    )
    frame["penalty_score"] = penalties.clip(0.0, 100.0)
    frame["balanced_score"] = (
        frame["iv_robustness_score"] * 0.30
        + objective_norm * 0.20
        + frame["upside_score"] * 0.15
        + frame["time_resilience_score"] * 0.15
        + frame["trust_score"] * 0.15
        + frame["capital_efficiency_score"] * 0.05
        - frame["penalty_score"] * 0.35
    ).clip(0.0, 100.0)
    frame["aggressive_upside_score"] = (
        frame["upside_score"] * 0.48
        + objective_norm * 0.26
        + frame["iv_upside_score"] * 0.14
        + frame["capital_efficiency_score"] * 0.08
        + frame["trust_score"] * 0.04
        - frame["penalty_score"] * 0.18
    ).clip(0.0, 100.0)
    frame["robustness_score"] = (
        frame["iv_robustness_score"] * 0.52
        + frame["time_resilience_score"] * 0.18
        + frame["trust_score"] * 0.16
        + frame["stock_edge_score"] * 0.14
        - frame["penalty_score"] * 0.22
    ).clip(0.0, 100.0)
    frame["fragility_score"] = (
        100.0
        - frame["robustness_score"]
        + frame["high_iv_dependency_rate"] * 24.0
        + weak_horizon.astype(float) * 16.0
        + target_beyond.astype(float) * 16.0
    ).clip(0.0, 100.0)
    tags: list[str] = []
    tag_rows: list[str] = []
    for _, row in frame.iterrows():
        tags = []
        if clean_string(row.get("strategy_family")) == "long_stock":
            tags.append("stock_dominates_under_current_assumptions")
        if _num(row.get("difference_vs_stock")) <= -50 and clean_string(row.get("strategy_family")) != "long_stock":
            tags.append("premium_too_demanding_under_base_path")
            tags.append("stock_dominates_under_current_assumptions")
        if _num(row.get("aggressive_upside_score")) >= 70:
            tags.append("benefits_from_fast_move")
        if _num(row.get("time_resilience_score")) < 55:
            tags.append("time_decay_sensitive")
            tags.append("suffers_from_flat_to_slow_path")
        if _num(row.get("terminal_delta_vs_stock_range")) >= 75 or _num(row.get("high_iv_dependency_rate")) > 0:
            tags.append("iv_sensitive")
        if clean_string(row.get("source_trust_label")) == "fallback_only":
            tags.append("trust_caution")
        tag_rows.append(_tag_join(tags))
    frame["decision_tags"] = tag_rows
    frame["stock_dominance_note"] = np.where(
        pd.to_numeric(frame.get("difference_vs_stock", 0), errors="coerce").fillna(0) > 0,
        "option shows modeled edge versus stock under active assumptions",
        "stock remains cleaner or ahead under active assumptions",
    )
    preferred = [
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
        "path_count_tested",
        "iv_opportunity_count",
        "profitable_iv_path_rate",
        "beat_stock_iv_path_rate",
        "lower_iv_survival_rate",
        "lower_iv_beat_stock_rate",
        "high_iv_dependency_rate",
        "terminal_delta_vs_stock_range",
        "decision_tags",
        "stock_dominance_note",
        "weak_horizon_fit",
        "target_beyond_expiry",
        "clamped_to_expiry",
    ]
    columns = [col for col in preferred if col in frame.columns] + [col for col in frame.columns if col not in preferred]
    return frame.loc[:, columns]


def _select_candidate(
    frame: pd.DataFrame,
    *,
    category: str,
    score_column: str,
    calls_only: bool = True,
    ascending: bool = False,
) -> dict[str, Any]:
    if frame.empty:
        return _no_edge_row(category, reason="No candidate data was available.")
    pool = frame.copy()
    if calls_only:
        pool = pool.loc[pool.get("strategy_family", "").astype(str).str.lower().eq("long_call")].copy()
    if pool.empty:
        return _no_edge_row(category, reason="No long-call candidate was available for this category.")
    pool[score_column] = pd.to_numeric(pool.get(score_column), errors="coerce").fillna(0.0)
    pool = pool.sort_values([score_column, "trust_score", "objective_score"], ascending=[ascending, False, False])
    selected = pool.iloc[0].to_dict()
    second = pool.iloc[1].to_dict() if len(pool.index) > 1 else {}
    score = _num(selected.get(score_column))
    edge_gap = abs(score - _num(second.get(score_column))) if second else score
    if score < 35:
        edge_status = "no_clear_edge_under_current_assumptions"
    elif edge_gap < 4:
        edge_status = "weak_differentiation"
    else:
        edge_status = "informative_edge"
    return _highlight_from_candidate(selected, category=category, score_column=score_column, edge_status=edge_status)


def _no_edge_row(category: str, *, reason: str) -> dict[str, Any]:
    return {
        "highlight_category": category,
        "highlight_label": HIGHLIGHT_LABELS.get(category, category.replace("_", " ").title()),
        "selected_candidate_slug": "",
        "selected_candidate_label": "no_clear_edge",
        "selected_family": "no_clear_edge",
        "decision_status": "no_clear_edge_under_current_assumptions",
        "score": 0.0,
        "source_trust_label": "",
        "trust_caution": "",
        "primary_reason": reason,
        "main_warning": reason,
        "decision_tags": "no_clear_edge_under_current_assumptions",
        "score_column": "",
    }


def _highlight_from_candidate(
    candidate: dict[str, Any],
    *,
    category: str,
    score_column: str,
    edge_status: str,
) -> dict[str, Any]:
    family = clean_string(candidate.get("strategy_family"))
    diff = _num(candidate.get("difference_vs_stock"))
    tags = clean_string(candidate.get("decision_tags"))
    trust = clean_string(candidate.get("source_trust_label"))
    warning_parts: list[str] = []
    if family != "long_stock" and diff <= 0:
        warning_parts.append("stock still dominates under current assumptions")
    if "requires_iv_support" in tags or _num(candidate.get("high_iv_dependency_rate")) > 0:
        warning_parts.append("requires IV support")
    if _bool(candidate.get("weak_horizon_fit")) or _bool(candidate.get("target_beyond_expiry")):
        warning_parts.append("timing or expiry fit is weak")
    if trust == "fallback_only":
        warning_parts.append("fallback-only quote trust")
    if edge_status != "informative_edge":
        warning_parts.append(edge_status)
    if category == "stock_still_best_baseline" or family == "long_stock":
        status = "stock_cleaner_under_current_assumptions"
    elif diff > 0:
        status = "option_shows_modeled_edge"
    elif edge_status == "no_clear_edge_under_current_assumptions":
        status = edge_status
    else:
        status = "option_interesting_but_stock_still_benchmark"
    reason = _reason_for_category(category, candidate)
    return {
        "highlight_category": category,
        "highlight_label": HIGHLIGHT_LABELS.get(category, category.replace("_", " ").title()),
        "selected_candidate_slug": clean_string(candidate.get("candidate_slug")),
        "selected_candidate_label": _candidate_short_label(candidate),
        "selected_family": family,
        "decision_status": status,
        "score": round(_num(candidate.get(score_column)), 2),
        "source_trust_label": trust,
        "trust_caution": "trust_caution" if trust == "fallback_only" else "",
        "primary_reason": reason,
        "main_warning": "; ".join(warning_parts) if warning_parts else "assumption-relative, not objective mispricing",
        "decision_tags": tags,
        "score_column": score_column,
        "difference_vs_stock": finite_or_none(candidate.get("difference_vs_stock")),
        "return_on_comparison_capital": finite_or_none(candidate.get("return_on_comparison_capital")),
        "robustness_score": finite_or_none(candidate.get("robustness_score")),
        "aggressive_upside_score": finite_or_none(candidate.get("aggressive_upside_score")),
        "balanced_score": finite_or_none(candidate.get("balanced_score")),
        "time_resilience_score": finite_or_none(candidate.get("time_resilience_score")),
        "lower_iv_resilience_score": finite_or_none(candidate.get("lower_iv_resilience_score")),
        "iv_upside_score": finite_or_none(candidate.get("iv_upside_score")),
        "fragility_score": finite_or_none(candidate.get("fragility_score")),
    }


def _reason_for_category(category: str, candidate: dict[str, Any]) -> str:
    label = _candidate_short_label(candidate)
    if category == "most_robust_call":
        return f"{label} has the strongest blend of path survival, IV resilience, timing fit, and data trust among long calls."
    if category == "best_aggressive_upside_call":
        return f"{label} has the highest upside/convexity score, but it should be treated as more assumption-sensitive."
    if category == "best_balanced_call":
        return f"{label} is the best compromise across upside, IV robustness, timing resilience, and quote trust."
    if category == "best_if_move_is_late":
        return f"{label} scores best on delayed-move resilience and expiry fit."
    if category == "best_if_iv_falls":
        return f"{label} is least dependent on friendly IV and holds up best in lower-IV regimes."
    if category == "best_if_iv_rises":
        return f"{label} benefits most from favorable IV support while keeping the same stock path."
    if category == "best_if_you_want_simplest_exposure":
        return "Long stock has no expiry, theta, strike-selection, or IV-regime dependency."
    if category == "stock_still_best_baseline":
        return "The stock baseline remains the cleanest benchmark when option candidates do not beat stock after premium and timing."
    if category == "most_fragile_call":
        return f"{label} has the weakest robustness profile among the highlighted long calls."
    if category == "requires_fast_move":
        return f"{label} has upside but a weaker delayed-move score, so timing matters."
    if category == "requires_iv_support":
        return f"{label} depends most on IV support or elevated IV outcomes."
    return f"{label} is selected under the current assumption-relative score."


def _build_decision_highlights(tradeoff: pd.DataFrame) -> pd.DataFrame:
    if tradeoff.empty:
        return pd.DataFrame([_no_edge_row(category, reason="No tradeoff data was available.") for category in HIGHLIGHT_CATEGORY_ORDER])
    highlights: list[dict[str, Any]] = []
    highlights.append(_select_candidate(tradeoff, category="most_robust_call", score_column="robustness_score"))
    highlights.append(_select_candidate(tradeoff, category="best_aggressive_upside_call", score_column="aggressive_upside_score"))
    highlights.append(_select_candidate(tradeoff, category="best_balanced_call", score_column="balanced_score"))
    highlights.append(_select_candidate(tradeoff, category="best_if_move_is_late", score_column="time_resilience_score"))
    highlights.append(_select_candidate(tradeoff, category="best_if_iv_falls", score_column="lower_iv_resilience_score"))
    highlights.append(_select_candidate(tradeoff, category="best_if_iv_rises", score_column="iv_upside_score"))
    stock_rows = tradeoff.loc[tradeoff.get("strategy_family", "").astype(str).str.lower().eq("long_stock")].copy()
    if not stock_rows.empty:
        stock = stock_rows.sort_values("balanced_score", ascending=False).iloc[0].to_dict()
        highlights.append(
            _highlight_from_candidate(
                stock,
                category="best_if_you_want_simplest_exposure",
                score_column="balanced_score",
                edge_status="informative_edge",
            )
        )
        highlights.append(
            _highlight_from_candidate(
                stock,
                category="stock_still_best_baseline",
                score_column="balanced_score",
                edge_status="informative_edge",
            )
        )
    else:
        highlights.append(_no_edge_row("best_if_you_want_simplest_exposure", reason="No long-stock benchmark was available."))
        highlights.append(_no_edge_row("stock_still_best_baseline", reason="No long-stock benchmark was available."))
    highlights.append(_select_candidate(tradeoff, category="most_fragile_call", score_column="fragility_score"))
    fast_pool = tradeoff.loc[tradeoff.get("strategy_family", "").astype(str).str.lower().eq("long_call")].copy()
    if not fast_pool.empty:
        fast_pool["fast_move_dependency_score"] = (
            fast_pool["aggressive_upside_score"] * 0.65 + (100.0 - fast_pool["time_resilience_score"]) * 0.35
        )
        highlights.append(_select_candidate(fast_pool, category="requires_fast_move", score_column="fast_move_dependency_score"))
        fast_pool["iv_support_dependency_score"] = (
            fast_pool["iv_upside_score"] * 0.55
            + fast_pool["high_iv_dependency_rate"] * 35.0
            + (100.0 - fast_pool["lower_iv_resilience_score"]) * 0.25
        ).clip(0.0, 100.0)
        highlights.append(_select_candidate(fast_pool, category="requires_iv_support", score_column="iv_support_dependency_score"))
    else:
        highlights.append(_no_edge_row("requires_fast_move", reason="No long-call candidate was available."))
        highlights.append(_no_edge_row("requires_iv_support", reason="No long-call candidate was available."))
    frame = pd.DataFrame(highlights)
    frame["display_order"] = frame["highlight_category"].map({name: idx + 1 for idx, name in enumerate(HIGHLIGHT_CATEGORY_ORDER)})
    return frame.sort_values("display_order").reset_index(drop=True)


def _build_explanations(highlights: pd.DataFrame, tradeoff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if highlights.empty:
        return pd.DataFrame()
    trade_lookup = {clean_string(row.get("candidate_slug")): row for row in tradeoff.to_dict("records")}
    for row in highlights.to_dict("records"):
        candidate = trade_lookup.get(clean_string(row.get("selected_candidate_slug")), {})
        rows.append(
            {
                "highlight_category": row.get("highlight_category"),
                "selected_candidate_label": row.get("selected_candidate_label"),
                "decision_status": row.get("decision_status"),
                "primary_reason": row.get("primary_reason"),
                "main_warning": row.get("main_warning"),
                "score_column": row.get("score_column"),
                "score": row.get("score"),
                "robustness_score": finite_or_none(candidate.get("robustness_score")),
                "aggressive_upside_score": finite_or_none(candidate.get("aggressive_upside_score")),
                "balanced_score": finite_or_none(candidate.get("balanced_score")),
                "time_resilience_score": finite_or_none(candidate.get("time_resilience_score")),
                "lower_iv_resilience_score": finite_or_none(candidate.get("lower_iv_resilience_score")),
                "iv_upside_score": finite_or_none(candidate.get("iv_upside_score")),
                "trust_score": finite_or_none(candidate.get("trust_score")),
                "penalty_score": finite_or_none(candidate.get("penalty_score")),
                "decision_tags": row.get("decision_tags"),
            }
        )
    return pd.DataFrame(rows)


def _build_score_breakdown(tradeoff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    component_cols = [
        "robustness_score",
        "aggressive_upside_score",
        "balanced_score",
        "time_resilience_score",
        "lower_iv_resilience_score",
        "iv_upside_score",
        "trust_score",
        "capital_efficiency_score",
        "penalty_score",
        "fragility_score",
    ]
    for row in tradeoff.to_dict("records"):
        for component in component_cols:
            rows.append(
                {
                    "candidate_slug": row.get("candidate_slug"),
                    "candidate_label": row.get("candidate_label"),
                    "strategy_family": row.get("strategy_family"),
                    "component": component,
                    "component_score": finite_or_none(row.get(component)),
                    "component_note": _component_note(component),
                }
            )
    return pd.DataFrame(rows)


def _component_note(component: str) -> str:
    notes = {
        "robustness_score": "Path survival, IV resilience, timing fit, quote trust, and stock-relative edge.",
        "aggressive_upside_score": "Upside potential and convexity, with smaller trust and capital-efficiency adjustments.",
        "balanced_score": "Compromise score across upside, robustness, timing, trust, and penalties.",
        "time_resilience_score": "Penalizes weak horizon fit, target beyond expiry, and expiry-clamped estimates.",
        "lower_iv_resilience_score": "Rewards candidates that keep value or beat stock under lower-IV regimes.",
        "iv_upside_score": "Measures sensitivity to favorable IV regimes.",
        "trust_score": "Quote/source trust score based on local market-context provenance.",
        "capital_efficiency_score": "Return/capital efficiency under the comparison-capital lens.",
        "penalty_score": "Trust, horizon, target-beyond-expiry, and clamp penalties.",
        "fragility_score": "Inverse robustness plus timing/IV-dependency penalties.",
    }
    return notes.get(component, "")


def _build_stock_vs_option_takeaways(highlights: pd.DataFrame, tradeoff: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calls = tradeoff.loc[tradeoff.get("strategy_family", "").astype(str).str.lower().eq("long_call")].copy()
    stock = tradeoff.loc[tradeoff.get("strategy_family", "").astype(str).str.lower().eq("long_stock")].copy()
    best_call = calls.sort_values("balanced_score", ascending=False).iloc[0].to_dict() if not calls.empty else {}
    best_aggr = calls.sort_values("aggressive_upside_score", ascending=False).iloc[0].to_dict() if not calls.empty else {}
    stock_row = stock.iloc[0].to_dict() if not stock.empty else {}
    rows.append(
        {
            "takeaway_type": "stock_vs_best_balanced_call",
            "status": "stock_still_cleaner" if _num(best_call.get("difference_vs_stock")) <= 0 else "option_has_modeled_edge",
            "candidate_label": _candidate_short_label(best_call) if best_call else "no_clear_edge",
            "evidence": "Best balanced call versus long stock under active assumptions.",
            "difference_vs_stock": finite_or_none(best_call.get("difference_vs_stock")),
            "note": clean_string(best_call.get("stock_dominance_note")) or "No balanced call evidence available.",
        }
    )
    rows.append(
        {
            "takeaway_type": "aggressive_call_vs_stock",
            "status": "aggressive_but_stock_benchmark" if _num(best_aggr.get("difference_vs_stock")) <= 0 else "aggressive_option_edge",
            "candidate_label": _candidate_short_label(best_aggr) if best_aggr else "no_clear_edge",
            "evidence": "Highest aggressive-upside long call versus long stock.",
            "difference_vs_stock": finite_or_none(best_aggr.get("difference_vs_stock")),
            "note": "Aggressive upside is useful only if the path and timing arrive strongly enough.",
        }
    )
    rows.append(
        {
            "takeaway_type": "simplest_exposure",
            "status": "stock_baseline_available" if stock_row else "no_stock_baseline",
            "candidate_label": _candidate_short_label(stock_row) if stock_row else "no_clear_edge",
            "evidence": "Long stock removes option premium, IV, expiry, and strike-selection dependencies.",
            "difference_vs_stock": 0.0 if stock_row else None,
            "note": "Use stock when the option edge is not clearly positive after premium and timing.",
        }
    )
    needs_iv = highlights.loc[highlights.get("highlight_category").eq("requires_iv_support")].copy() if not highlights.empty else pd.DataFrame()
    if not needs_iv.empty:
        row = needs_iv.iloc[0]
        rows.append(
            {
                "takeaway_type": "iv_dependency",
                "status": clean_string(row.get("decision_status")),
                "candidate_label": clean_string(row.get("selected_candidate_label")),
                "evidence": clean_string(row.get("primary_reason")),
                "difference_vs_stock": finite_or_none(row.get("difference_vs_stock")),
                "note": clean_string(row.get("main_warning")),
            }
        )
    return pd.DataFrame(rows)


def _markdown_bullets(frame: pd.DataFrame, categories: list[str]) -> list[str]:
    lines: list[str] = []
    if frame.empty:
        return ["- no_clear_edge_under_current_assumptions"]
    subset = frame.loc[frame["highlight_category"].isin(categories)].copy()
    if subset.empty:
        return ["- no_clear_edge_under_current_assumptions"]
    for row in subset.to_dict("records"):
        label = clean_string(row.get("highlight_label"))
        candidate = clean_string(row.get("selected_candidate_label"))
        status = clean_string(row.get("decision_status"))
        warning = clean_string(row.get("main_warning"))
        lines.append(f"- {label}: `{candidate}` - {status}. {warning}")
    return lines


def _build_highlights_markdown(
    *,
    ticker: str,
    highlights: pd.DataFrame,
    takeaways: pd.DataFrame,
    tradeoff: pd.DataFrame,
    analysis_trust_level: str,
) -> str:
    best_balanced = highlights.loc[highlights.get("highlight_category").eq("best_balanced_call")].copy() if not highlights.empty else pd.DataFrame()
    stock_baseline = highlights.loc[highlights.get("highlight_category").eq("stock_still_best_baseline")].copy() if not highlights.empty else pd.DataFrame()
    strongest = clean_string(best_balanced.iloc[0].get("selected_candidate_label")) if not best_balanced.empty else "no_clear_edge"
    stock_read = clean_string(stock_baseline.iloc[0].get("decision_status")) if not stock_baseline.empty else "unknown"
    lines = [
        f"# {clean_string(ticker).upper()} Decision Highlights",
        "",
        "Assumption-driven highlights from the frozen contract-selection bundle. These are not objective recommendations; they summarize what the modeled paths, IV regimes, and local data trust imply.",
        "",
        "## Decision Snapshot",
        "",
        f"- Balanced call read: `{strongest}`",
        f"- Stock baseline read: `{stock_read}`",
        f"- Analysis trust level: `{clean_string(analysis_trust_level) or 'n/a'}`",
        "- If differentiation is weak, the tables say so explicitly instead of forcing a false winner.",
        "",
        "## What Looks Most Attractive Right Now",
        "",
    ]
    lines.extend(_markdown_bullets(highlights, ["best_balanced_call", "most_robust_call", "best_aggressive_upside_call"]))
    lines.extend(["", "## Where Stock Still Looks Better", ""])
    lines.extend(_markdown_bullets(highlights, ["stock_still_best_baseline", "best_if_you_want_simplest_exposure"]))
    if not takeaways.empty:
        for row in takeaways.head(3).to_dict("records"):
            lines.append(f"- {clean_string(row.get('takeaway_type'))}: {clean_string(row.get('status'))}. {clean_string(row.get('note'))}")
    lines.extend(["", "## Calls That Need IV Support", ""])
    lines.extend(_markdown_bullets(highlights, ["requires_iv_support", "best_if_iv_rises"]))
    lines.extend(["", "## Calls That Tolerate a Delayed Move Better", ""])
    lines.extend(_markdown_bullets(highlights, ["best_if_move_is_late"]))
    lines.extend(["", "## Most Robust Calls Across Paths", ""])
    lines.extend(_markdown_bullets(highlights, ["most_robust_call", "best_if_iv_falls"]))
    lines.extend(["", "## High Convexity But Fragile Calls", ""])
    lines.extend(_markdown_bullets(highlights, ["best_aggressive_upside_call", "most_fragile_call", "requires_fast_move"]))
    caution_count = int((tradeoff.get("source_trust_label", pd.Series(dtype=str)).astype(str).str.lower() == "fallback_only").sum()) if not tradeoff.empty else 0
    lines.extend(
        [
            "",
            "## Trust / Data Quality Notes",
            "",
            f"- Fallback-only candidates in tradeoff matrix: `{caution_count}`",
            "- Trust penalties are applied before balanced and robustness scores are used for highlights.",
            "- Exact file provenance remains in bundle metadata and chain-source tables, not in this human-facing summary.",
            "",
            "## How To Read The Overview Charts",
            "",
            "- `highlights_overview.png`: compact category-to-candidate map with status and trust.",
            "- `candidate_robustness_vs_upside.png`: quadrant view of robust versus aggressive candidates, with stock as baseline.",
            "- `path_survival_scorecard.png`: which candidates survive more modeled path/IV opportunities.",
            "- `iv_robustness_scorecard.png`: which candidates survive lower IV or require IV support.",
            "- `strike_expiry_tradeoff_overview.png`: how strike/expiry choices trade upside for resilience.",
            "- `stock_vs_option_decision_chart.png`: whether candidate edge clears the stock benchmark.",
            "",
            "## Next Best Tables To Inspect",
            "",
            "- `decision_highlights.csv` for category winners and cautions.",
            "- `candidate_tradeoff_matrix.csv` for score components and tags.",
            "- `candidate_robustness_summary.csv` for path/IV survival evidence.",
            "- `stock_vs_option_takeaways.csv` for explicit stock-versus-option reads.",
        ]
    )
    return "\n".join(lines) + "\n"


def _money_label(value: Any) -> str:
    numeric = finite_or_none(value)
    if numeric is None:
        return ""
    return f"${float(numeric):,.2f}"


def _date_label(value: Any) -> str:
    text = clean_string(value)
    return text[:10] if text else ""


def _option_or_stock_label(row: dict[str, Any] | pd.Series) -> str:
    family = clean_string(row.get("strategy_family"))
    if family == "long_stock":
        return "Long Stock Baseline"
    return _candidate_short_label(row)


def _humanize_trust_label(value: Any) -> str:
    text = clean_string(value).replace("_", " ").strip()
    return text.title() if text else "Unknown Trust"


def _humanize_trigger_type(value: Any) -> str:
    mapping = {
        "stock_baseline": "Stock Baseline",
        "trust_too_weak_for_action": "Needs Better Trust",
        "quote_quality": "Needs Better Trust",
        "prefer_later_expiry": "Prefer Later Expiry",
        "timing_runway": "Later Expiry / Timing",
        "move_must_start_early": "Move Must Start Early",
        "early_move_needed": "Needs Earlier Move",
        "stock_confirmation": "Stock Confirmation",
        "iv_support": "Needs IV Support",
        "iv_normalization_entry": "Better After IV Cools",
        "premium_below_threshold": "Premium Below Threshold",
        "premium_reset": "Needs Better Entry",
        "premium_entry_improvement": "Needs Cheaper Premium",
        "stock_cleaner_unless_x": "Stock Cleaner Unless Trigger Hits",
        "better_after_event": "Better After Event",
        "post_event_entry": "Better After Event",
        "thesis_confirmation": "Thesis Confirmation",
    }
    key = clean_string(value).lower()
    return mapping.get(key, clean_string(value).replace("_", " ").title() or "Trigger")


def _action_confidence(score: float, trust_score: float, bucket: str) -> str:
    if bucket == "Avoid For Now":
        return "high" if score < 35 or trust_score < 45 else "medium"
    if bucket == "Prefer Stock Instead":
        return "high" if score >= 70 else "medium"
    if score >= 76 and trust_score >= 78:
        return "high"
    if score >= 55 and trust_score >= 60:
        return "medium"
    return "cautious"


def _trigger_profile(row: dict[str, Any]) -> dict[str, str]:
    family = clean_string(row.get("strategy_family"))
    target_date = _date_label(row.get("target_date")) or _date_label(row.get("expiry_date"))
    trust_label = clean_string(row.get("source_trust_label")).lower()
    time_risk = _num(row.get("time_decay_risk"))
    iv_risk = _num(row.get("iv_dependence_risk"))
    diff_vs_stock = _num(row.get("difference_vs_stock"))
    stock_relative_score = _num(row.get("stock_relative_score"))
    conviction_score = _num(row.get("candidate_conviction_score"))
    premium = _money_label(row.get("premium_or_entry_cost"))
    strike = _money_label(row.get("strike_label"))
    break_even = _money_label(row.get("break_even"))
    expiry_date = _date_label(row.get("expiry_date")) or target_date
    event_type = clean_string(row.get("nearest_event_type")).lower()
    confirmation_level = break_even or strike or _money_label(row.get("target_price")) or "the trigger level"

    def profile(
        trigger_type: str,
        trigger_value: str,
        deadline: str,
        invalidate: str,
        *,
        short_trigger: str | None = None,
        upgrade_rule: str | None = None,
    ) -> dict[str, str]:
        return {
            "key_trigger_type": clean_string(trigger_type),
            "key_trigger_label": _humanize_trigger_type(trigger_type),
            "key_trigger_value": clean_string(trigger_value),
            "key_trigger_deadline": clean_string(deadline),
            "main_trigger": clean_string(short_trigger or trigger_value),
            "what_has_to_happen": clean_string(upgrade_rule or trigger_value),
            "upgrade_rule": clean_string(upgrade_rule or trigger_value),
            "what_would_invalidate": clean_string(invalidate),
            "invalidate_rule": clean_string(invalidate),
        }

    if family == "long_stock":
        return profile(
            "stock_baseline",
            "Prefer stock unless an option shows clear modeled stock-relative edge after premium and timing.",
            target_date,
            "Stock remains simplest if option edge stays weak.",
            short_trigger="Prefer stock unless option edge improves",
        )
    if trust_label in {"fallback_only", "structure_only"}:
        return profile(
            "trust_too_weak_for_action",
            "Upgrade only after a quote-usable chain is available for this contract.",
            target_date,
            "Avoid while the chain stays sparse, fallback-heavy, or structure-only.",
            short_trigger="Needs better trust before acting",
        )
    if _bool(row.get("target_beyond_expiry")) or _bool(row.get("weak_horizon_fit")):
        return profile(
            "prefer_later_expiry",
            "Upgrade by moving to a later expiry if the thesis timing is slipping.",
            expiry_date,
            "Avoid if the target timing slips beyond expiry.",
            short_trigger="Prefer later expiry if timing slips",
        )
    if time_risk >= 66:
        return profile(
            "move_must_start_early",
            "Upgrade only if the move starts early enough to outrun theta and timing drag.",
            expiry_date,
            "Avoid if the stock is still waiting late into the contract window.",
            short_trigger="Only works if the move starts early",
        )
    if event_type and iv_risk >= 55:
        return profile(
            "better_after_event",
            f"Upgrade after the {event_type.replace('_', ' ')} if IV cools and the thesis still holds.",
            target_date,
            "Avoid if IV expands further while the stock path still needs to do the heavy lifting.",
            short_trigger="Better after event / IV cool-off",
        )
    if iv_risk >= 55:
        if event_type:
            return profile(
                "better_after_event",
                f"Upgrade after the {event_type.replace('_', ' ')} if IV cools and the thesis still holds.",
                target_date,
                "Avoid if IV stays rich while the stock path only grinds.",
                short_trigger="Better after event / IV cool-off",
            )
        return profile(
            "iv_normalization_entry",
            "Upgrade after IV normalizes and premium cools enough to give the call cleaner carry.",
            target_date,
            "Avoid if IV expands further or mean-reverts lower without a fast enough stock move.",
            short_trigger="Needs lower-IV entry",
        )
    if diff_vs_stock <= 0 and stock_relative_score >= 62 and conviction_score >= 58:
        return profile(
            "stock_confirmation",
            f"Upgrade if stock confirms above {confirmation_level} by {target_date}.",
            target_date,
            f"Avoid if stock stays below {confirmation_level} by {target_date}.",
            short_trigger=f"Needs stock confirmation above {confirmation_level}",
        )
    if premium and diff_vs_stock <= 0 and conviction_score >= 45:
        return profile(
            "premium_below_threshold",
            f"Upgrade if premium cools below {premium} before {target_date}.",
            target_date,
            "Avoid if premium expands further after the stock has already moved.",
            short_trigger=f"Watchlist only unless premium cools below {premium}",
        )
    if diff_vs_stock <= 0:
        return profile(
            "stock_cleaner_unless_x",
            f"Upgrade only if stock confirms above {confirmation_level} and the option still carries enough edge after premium.",
            target_date,
            f"Avoid if stock stays below {confirmation_level} while theta keeps working.",
            short_trigger="Long stock still cleaner unless confirmation improves",
        )
    if _num(row.get("lower_iv_resilience_score")) < 55 or _num(row.get("high_iv_dependency_rate")) > 0:
        return profile(
            "iv_support",
            "Upgrade only if IV stays supportive enough to keep the option edge alive.",
            target_date,
            "Avoid if IV mean-reverts lower without a fast enough stock move.",
            short_trigger="Requires IV support",
        )
    premium = _money_label(row.get("premium_or_entry_cost"))
    if premium:
        return profile(
            "premium_reset",
            f"Upgrade if premium stays below {premium} while the thesis remains intact.",
            target_date,
            "Avoid chasing after stock has already moved and IV has expanded.",
            short_trigger=f"Needs better entry below {premium}",
        )
    return profile(
        "thesis_confirmation",
        "Upgrade only if the thesis confirms cleanly and the option keeps better stock-relative edge.",
        target_date,
        "Avoid if the modeled edge remains compressed.",
        short_trigger="Needs clearer thesis confirmation",
    )


def _action_reasons(row: dict[str, Any], bucket: str) -> dict[str, str]:
    family = clean_string(row.get("strategy_family"))
    label = _option_or_stock_label(row)
    diff = _num(row.get("difference_vs_stock"))
    trust = clean_string(row.get("source_trust_label")) or "unknown_trust"
    time_risk = _num(row.get("time_decay_risk"))
    iv_risk = _num(row.get("iv_dependence_risk"))
    trigger_profile = _trigger_profile(row)

    interesting_parts: list[str] = []
    hurting_parts: list[str] = []

    if family == "long_stock":
        headline = "Stock is the cleanest exposure under current assumptions."
        why_stock = "No expiry, strike, premium, or IV-path dependency."
        interesting_parts.append("Simplest way to express the thesis without option timing or IV risk.")
    else:
        if _num(row.get("aggressive_upside_score")) >= 70:
            interesting_parts.append("High convexity if the stock path breaks in the right direction.")
        if _num(row.get("robustness_score")) >= 60:
            interesting_parts.append("Holds up better than weaker alternatives across path and IV stress.")
        if diff > 0:
            interesting_parts.append("Modeled edge can clear the long-stock benchmark under the active thesis.")
        if not interesting_parts:
            interesting_parts.append("There is still upside interest here, but the edge is narrow under current assumptions.")

        if bucket == "Buy Now":
            headline = f"{label} clears the current action threshold with acceptable trust and stress behavior."
            why_stock = "Stock is still the benchmark, but this option has enough modeled edge to be action-relevant."
        elif bucket == "Watchlist":
            headline = f"{label} is interesting, but it needs a trigger before it is buyable."
            if time_risk >= 55:
                why_stock = "Long stock still cleaner because the option edge is too timing-sensitive right now."
            elif iv_risk >= 55:
                why_stock = "Long stock still cleaner unless IV stays friendly enough to offset premium and decay."
            else:
                why_stock = "Long stock still cleaner until premium, timing, or confirmation improves."
        else:
            headline = f"{label} does not clear the current action threshold."
            if trust.lower() in {"fallback_only", "structure_only"}:
                why_stock = "Long stock still cleaner while this contract remains fallback-heavy."
            elif time_risk >= 55:
                why_stock = "Long stock still cleaner because this contract loses too much if the move is delayed."
            else:
                why_stock = "Long stock or a cleaner candidate is preferable until the contract improves materially."

    if diff <= 0 and family != "long_stock":
        hurting_parts.append("Premium is too demanding versus stock under the current path.")
    if time_risk >= 55:
        hurting_parts.append("Needs an earlier move before theta and timing drag bite too hard.")
    if iv_risk >= 55:
        hurting_parts.append("Benefits from lower-IV entry or more supportive IV than the cautious case.")
    if trust.lower() in {"fallback_only", "structure_only"}:
        hurting_parts.append("Trust is weaker because quotes are sparse or fallback-only.")
    if _bool(row.get("target_beyond_expiry")):
        hurting_parts.append("The thesis timing stretches beyond the contract runway.")
    if _bool(row.get("clamped_to_expiry")):
        hurting_parts.append("The modeled target is effectively clamped to expiry, which reduces flexibility.")
    if not hurting_parts:
        hurting_parts.append("Main assumptions remain the controlling risk.")

    warning = " ".join(dict.fromkeys(hurting_parts[:3]))
    interesting_now = " ".join(interesting_parts)
    trigger_read = clean_string(trigger_profile.get("main_trigger"))
    stock_read = why_stock
    watch_read = (
        f"{warning} Upgrade only if: {clean_string(trigger_profile.get('upgrade_rule'))}"
        if bucket == "Watchlist"
        else ""
    )
    avoid_read = (
        f"{warning} Avoid unless: {clean_string(trigger_profile.get('invalidate_rule'))}"
        if bucket == "Avoid For Now"
        else ""
    )

    return {
        "headline_reason": headline,
        "why_buy_now": headline if bucket == "Buy Now" else "",
        "why_watch_not_buy": watch_read,
        "why_avoid": avoid_read,
        "why_stock_may_be_better": stock_read,
        "why_this_is_interesting_now": interesting_now,
        "what_is_hurting_this_candidate": warning,
        "main_trigger": trigger_read,
        "what_has_to_happen": clean_string(trigger_profile.get("what_has_to_happen")),
        "what_would_invalidate": clean_string(trigger_profile.get("what_would_invalidate")),
        "upgrade_rule": clean_string(trigger_profile.get("upgrade_rule")),
        "invalidate_rule": clean_string(trigger_profile.get("invalidate_rule")),
        "key_trigger_type": clean_string(trigger_profile.get("key_trigger_type")),
        "key_trigger_label": clean_string(trigger_profile.get("key_trigger_label")),
        "key_trigger_value": clean_string(trigger_profile.get("key_trigger_value")),
        "key_trigger_deadline": clean_string(trigger_profile.get("key_trigger_deadline")),
        "main_warning": warning,
    }


def _build_action_board_candidates(tradeoff: pd.DataFrame) -> pd.DataFrame:
    if tradeoff.empty:
        return pd.DataFrame()
    frame = tradeoff.copy()
    for col in [
        "balanced_score",
        "aggressive_upside_score",
        "robustness_score",
        "stock_edge_score",
        "time_resilience_score",
        "lower_iv_resilience_score",
        "trust_score",
        "capital_efficiency_score",
        "penalty_score",
        "difference_vs_stock",
        "return_on_comparison_capital",
        "high_iv_dependency_rate",
    ]:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["time_decay_risk"] = (100.0 - frame["time_resilience_score"]).clip(0.0, 100.0)
    frame["iv_dependence_risk"] = ((100.0 - frame["lower_iv_resilience_score"]) * 0.58 + frame["high_iv_dependency_rate"] * 42.0).clip(0.0, 100.0)
    frame["trust_penalty"] = (100.0 - frame["trust_score"]).clip(0.0, 100.0)
    frame["stock_relative_score"] = frame["stock_edge_score"].clip(0.0, 100.0)
    frame["candidate_conviction_score"] = (
        frame["balanced_score"] * 0.34
        + frame["aggressive_upside_score"] * 0.20
        + frame["stock_relative_score"] * 0.18
        + frame["capital_efficiency_score"] * 0.10
        + frame["trust_score"] * 0.10
        + frame["lower_iv_resilience_score"] * 0.08
        - frame["penalty_score"] * 0.18
    ).clip(0.0, 100.0)
    frame["action_score"] = (
        frame["candidate_conviction_score"] * 0.48
        + frame["robustness_score"] * 0.24
        + frame["stock_relative_score"] * 0.18
        + frame["lower_iv_resilience_score"] * 0.10
        - frame["time_decay_risk"] * 0.12
        - frame["trust_penalty"] * 0.10
    ).clip(0.0, 100.0)
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        family = clean_string(raw.get("strategy_family"))
        trust_score = _num(raw.get("trust_score"))
        action_score = _num(raw.get("action_score"))
        diff = _num(raw.get("difference_vs_stock"))
        weak_horizon = _bool(raw.get("weak_horizon_fit")) or _bool(raw.get("target_beyond_expiry")) or _bool(raw.get("clamped_to_expiry"))
        fallback_trust = clean_string(raw.get("source_trust_label")).lower() in {"fallback_only", "structure_only"}
        if family == "long_stock":
            bucket = "Prefer Stock Instead"
        elif (
            action_score >= 68
            and trust_score >= 70
            and _num(raw.get("robustness_score")) >= 58
            and diff > 0
            and _num(raw.get("time_decay_risk")) < 55
            and _num(raw.get("iv_dependence_risk")) < 70
            and not weak_horizon
            and not fallback_trust
        ):
            bucket = "Buy Now"
        elif (
            fallback_trust
            or action_score < 35
            or (_num(raw.get("time_decay_risk")) >= 72 and weak_horizon)
            or (_num(raw.get("iv_dependence_risk")) >= 82 and diff <= 0)
        ):
            bucket = "Avoid For Now"
        elif action_score >= 43 or _num(raw.get("aggressive_upside_score")) >= 70 or _num(raw.get("balanced_score")) >= 55:
            bucket = "Watchlist"
        else:
            bucket = "Avoid For Now"
        reasons = _action_reasons(raw, bucket)
        row = {
            "action_bucket": bucket,
            "action_priority_rank": 0,
            "action_confidence": _action_confidence(action_score, trust_score, bucket),
            "candidate_slug": clean_string(raw.get("candidate_slug")),
            "candidate_label": _option_or_stock_label(raw),
            "strategy_family": family,
            "expiry_date": clean_string(raw.get("expiry_date")),
            "strike_label": clean_string(raw.get("strike_label")),
            "moneyness_bucket": clean_string(raw.get("moneyness_bucket")),
            "source_trust_label": clean_string(raw.get("source_trust_label")),
            "candidate_conviction_score": round(_num(raw.get("candidate_conviction_score")), 4),
            "action_score": round(action_score, 4),
            "robustness_score": round(_num(raw.get("robustness_score")), 4),
            "upside_score": round(_num(raw.get("aggressive_upside_score")), 4),
            "stock_relative_score": round(_num(raw.get("stock_relative_score")), 4),
            "time_decay_risk": round(_num(raw.get("time_decay_risk")), 4),
            "iv_dependence_risk": round(_num(raw.get("iv_dependence_risk")), 4),
            "trust_penalty": round(_num(raw.get("trust_penalty")), 4),
            "affordability_status": clean_string(raw.get("affordability_label")) or "review_capital_fit",
            "difference_vs_stock": finite_or_none(raw.get("difference_vs_stock")),
            "return_on_comparison_capital": finite_or_none(raw.get("return_on_comparison_capital")),
            "premium_or_entry_cost": finite_or_none(raw.get("premium_or_entry_cost")),
            "capital_required": finite_or_none(raw.get("capital_required")),
            "weak_horizon_fit": _bool(raw.get("weak_horizon_fit")),
            "target_beyond_expiry": _bool(raw.get("target_beyond_expiry")),
            "clamped_to_expiry": _bool(raw.get("clamped_to_expiry")),
            "decision_tags": clean_string(raw.get("decision_tags")),
        }
        row.update(reasons)
        rows.append(row)
    action = pd.DataFrame(rows)
    if action.empty:
        return action
    action["_bucket_order"] = action["action_bucket"].map(ACTION_BUCKET_ORDER).fillna(99)
    action = action.sort_values(["_bucket_order", "action_score", "robustness_score"], ascending=[True, False, False]).reset_index(drop=True)
    action["action_priority_rank"] = action.groupby("action_bucket").cumcount() + 1
    return action.drop(columns=["_bucket_order"])


def _build_decision_triggers(action: pd.DataFrame) -> pd.DataFrame:
    if action.empty:
        return pd.DataFrame()
    trigger_rows = action.loc[action["action_bucket"].isin(["Watchlist", "Avoid For Now"])].copy()
    if trigger_rows.empty:
        trigger_rows = action.loc[action["action_bucket"].eq("Prefer Stock Instead")].copy()
    rows: list[dict[str, Any]] = []
    trigger_rows["_bucket_order"] = trigger_rows["action_bucket"].map(ACTION_BUCKET_ORDER).fillna(99)
    for row in trigger_rows.sort_values(["_bucket_order", "action_priority_rank"]).head(18).to_dict("records"):
        rows.append(
            {
                "action_bucket": row.get("action_bucket"),
                "candidate_label": row.get("candidate_label"),
                "strategy_family": row.get("strategy_family"),
                "key_trigger_type": row.get("key_trigger_type"),
                "trigger_type_label": row.get("key_trigger_label") or _humanize_trigger_type(row.get("key_trigger_type")),
                "key_trigger_value": row.get("key_trigger_value"),
                "key_trigger_deadline": row.get("key_trigger_deadline"),
                "trigger_direction": "must_improve_or_confirm",
                "urgency": "near_term" if _bool(row.get("weak_horizon_fit")) or _bool(row.get("target_beyond_expiry")) else "monitor",
                "what_has_to_happen": row.get("what_has_to_happen"),
                "what_would_invalidate": row.get("what_would_invalidate"),
                "upgrade_rule": row.get("upgrade_rule") or row.get("what_has_to_happen"),
                "invalidate_rule": row.get("invalidate_rule") or row.get("what_would_invalidate"),
                "main_warning": row.get("main_warning"),
                "source_trust_label": row.get("source_trust_label"),
                "action_confidence": row.get("action_confidence"),
            }
        )
    return pd.DataFrame(rows)


def _build_action_board_score_breakdown(action: pd.DataFrame) -> pd.DataFrame:
    if action.empty:
        return pd.DataFrame()
    components = [
        ("candidate_conviction_score", "Blends balanced score, upside, stock-relative edge, trust, IV resilience, and capital fit."),
        ("robustness_score", "Path/IV survival, timing fit, trust, and stock-relative edge."),
        ("upside_score", "Convexity and upside opportunity under the active assumptions."),
        ("stock_relative_score", "Whether the option clears the long-stock benchmark."),
        ("time_decay_risk", "Penalty signal for theta, weak horizon fit, target-beyond-expiry, and clamping."),
        ("iv_dependence_risk", "Penalty signal for needing elevated or friendly IV."),
        ("trust_penalty", "Penalty from sparse, fallback, or lower-trust source quality."),
        ("action_score", "Final transparent action-board score used for bucket ordering."),
    ]
    rows: list[dict[str, Any]] = []
    for row in action.to_dict("records"):
        for component, note in components:
            rows.append(
                {
                    "action_bucket": row.get("action_bucket"),
                    "candidate_label": row.get("candidate_label"),
                    "strategy_family": row.get("strategy_family"),
                    "component": component,
                    "component_score": finite_or_none(row.get(component)),
                    "component_note": note,
                }
            )
    return pd.DataFrame(rows)


def _build_action_board_explanations(action: pd.DataFrame) -> pd.DataFrame:
    if action.empty:
        return pd.DataFrame()
    cols = [
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
        "what_has_to_happen",
        "what_would_invalidate",
        "upgrade_rule",
        "invalidate_rule",
        "main_warning",
        "decision_tags",
    ]
    return action.loc[:, [col for col in cols if col in action.columns]].copy()


def _subset_action_board(action: pd.DataFrame, *, include_families: set[str]) -> pd.DataFrame:
    if action.empty:
        return pd.DataFrame()
    families = {clean_string(value).lower() for value in include_families if clean_string(value)}
    subset = action.loc[action.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().isin(families)].copy()
    if subset.empty:
        return subset
    subset["_bucket_order"] = subset["action_bucket"].map(ACTION_BUCKET_ORDER).fillna(99)
    subset["action_priority_rank"] = pd.to_numeric(subset["action_priority_rank"], errors="coerce").fillna(999)
    subset["action_score"] = pd.to_numeric(subset["action_score"], errors="coerce").fillna(0.0)
    subset = subset.sort_values(["_bucket_order", "action_priority_rank", "action_score"], ascending=[True, True, False]).reset_index(drop=True)
    subset["action_priority_rank"] = subset.groupby("action_bucket").cumcount() + 1
    return subset.drop(columns=["_bucket_order"], errors="ignore")


def _build_stock_preference_summary(action: pd.DataFrame) -> pd.DataFrame:
    if action.empty:
        return pd.DataFrame()
    families = {"long_stock"}
    subset = _subset_action_board(action, include_families=families)
    if subset.empty:
        return subset
    subset["_family_order"] = subset["strategy_family"].astype(str).str.lower().map({"long_stock": 0}).fillna(9)
    subset["action_score"] = pd.to_numeric(subset["action_score"], errors="coerce").fillna(0.0)
    subset = subset.sort_values(["_family_order", "action_priority_rank", "action_score"], ascending=[True, True, False]).drop(columns=["_family_order"]).head(4)
    return subset.reset_index(drop=True)


def _build_other_structures_summary(action: pd.DataFrame) -> pd.DataFrame:
    if action.empty:
        return pd.DataFrame()
    families = {"bull_call_spread", "covered_call", "cash_secured_put", "long_put", "bear_put_spread"}
    subset = _subset_action_board(action, include_families=families)
    if subset.empty:
        return subset
    preferred = [
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
    ]
    return subset.loc[:, [col for col in preferred if col in subset.columns]].copy()


def _short_contract_label(row: dict[str, Any] | pd.Series) -> str:
    family = clean_string(row.get("strategy_family")).lower()
    if family == "long_stock":
        return "Stock"
    expiry = pd.to_datetime(clean_string(row.get("expiry_date")), errors="coerce")
    expiry_label = expiry.strftime("%b-%y") if not pd.isna(expiry) else clean_string(row.get("expiry_date"))
    strike_text = clean_string(row.get("strike_label"))
    strike_numeric = finite_or_none(strike_text)
    if strike_numeric is None:
        strike_label = strike_text
    elif abs(float(strike_numeric) - round(float(strike_numeric))) < 1e-9:
        strike_label = f"{int(round(float(strike_numeric)))}C"
    else:
        strike_label = f"{float(strike_numeric):.2f}".rstrip("0").rstrip(".") + "C"
    if strike_label and expiry_label:
        return f"{strike_label} {expiry_label}"
    return _candidate_short_label(row)


def _compare_vs_stock_note(row: dict[str, Any]) -> str:
    family = clean_string(row.get("strategy_family")).lower()
    if family == "long_stock":
        return "Long stock is already the clean baseline here."
    diff_vs_stock = _num(row.get("difference_vs_stock"))
    why_stock = clean_string(row.get("why_stock_may_be_better"))
    if diff_vs_stock > 0:
        return "Option edge can clear stock if the upgrade rule is met."
    return why_stock or "Long stock still looks cleaner under the base path."


def _build_top_candidate_cards_frame(bullish_action: pd.DataFrame) -> pd.DataFrame:
    if bullish_action.empty:
        return pd.DataFrame()
    data = bullish_action.copy()
    data["_bucket_order"] = data.get("action_bucket", pd.Series(dtype=str)).map(ACTION_BUCKET_ORDER).fillna(99)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999)
    data["action_score"] = pd.to_numeric(data.get("action_score"), errors="coerce").fillna(0.0)
    data = data.sort_values(["_bucket_order", "action_priority_rank", "action_score"], ascending=[True, True, False]).head(5).copy()
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(data.to_dict("records"), start=1):
        rows.append(
            {
                "card_rank": rank,
                "contract_label": _short_contract_label(row),
                "candidate_label": clean_string(row.get("candidate_label")),
                "bucket": clean_string(row.get("action_bucket")),
                "confidence": clean_string(row.get("action_confidence")).title(),
                "why_this_is_interesting": clean_string(row.get("why_this_is_interesting_now")),
                "what_hurts_it": clean_string(row.get("what_is_hurting_this_candidate")),
                "main_trigger": clean_string(row.get("main_trigger")),
                "upgrade_rule": clean_string(row.get("upgrade_rule") or row.get("what_has_to_happen")),
                "invalidate_rule": clean_string(row.get("invalidate_rule") or row.get("what_would_invalidate")),
                "trust": _humanize_trust_label(row.get("source_trust_label")),
                "compare_vs_stock_note": _compare_vs_stock_note(row),
                "action_score": round(_num(row.get("action_score")), 2),
            }
        )
    return pd.DataFrame(rows)


def _build_top_candidate_cards_markdown(*, ticker: str, cards: pd.DataFrame) -> str:
    lines = [
        f"# {clean_string(ticker).upper()} Top Bullish Call Cards",
        "",
        "Compact first-read cards for the most relevant bullish long calls under the active assumptions.",
        "",
    ]
    if cards.empty:
        lines.extend(
            [
                "- No bullish long-call cards were generated.",
                "",
                "Stock may simply be the cleaner choice right now.",
            ]
        )
        return "\n".join(lines) + "\n"
    for row in cards.to_dict("records"):
        lines.extend(
            [
                f"## {clean_string(row.get('contract_label'))} - {clean_string(row.get('bucket'))}",
                "",
                f"- Confidence: `{clean_string(row.get('confidence')) or 'n/a'}`",
                f"- Trust: `{clean_string(row.get('trust')) or 'n/a'}`",
                f"- Interesting because: {clean_string(row.get('why_this_is_interesting'))}",
                f"- Held back by: {clean_string(row.get('what_hurts_it'))}",
                f"- Upgrade if: {clean_string(row.get('upgrade_rule'))}",
                f"- Invalidate if: {clean_string(row.get('invalidate_rule'))}",
                f"- Compare vs stock: {clean_string(row.get('compare_vs_stock_note'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## How To Use These Cards",
            "",
            "- Read these before the wider tables when you want a compact shortlist.",
            "- Then open `summary/bullish_action_board.md` and `summary/entry_justification.md` for the full why/trigger context.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_bullish_action_board_markdown(
    *,
    ticker: str,
    bullish_action: pd.DataFrame,
    bullish_triggers: pd.DataFrame,
    stock_summary: pd.DataFrame,
    analysis_trust_level: str,
) -> str:
    buy_now = bullish_action.loc[bullish_action["action_bucket"].eq("Buy Now")].copy() if not bullish_action.empty else pd.DataFrame()
    watchlist = bullish_action.loc[bullish_action["action_bucket"].eq("Watchlist")].copy() if not bullish_action.empty else pd.DataFrame()
    avoid = bullish_action.loc[bullish_action["action_bucket"].eq("Avoid For Now")].copy() if not bullish_action.empty else pd.DataFrame()
    hurting_lines: list[str] = []
    for row in bullish_action.head(5).to_dict("records"):
        hurting = clean_string(row.get("what_is_hurting_this_candidate"))
        if hurting:
            hurting_lines.append(f"- `{clean_string(row.get('candidate_label'))}`: {hurting}")
    if not hurting_lines:
        hurting_lines = ["- No clear long-call cautions were generated."]
    lines = [
        f"# {clean_string(ticker).upper()} Bullish Long-Call Board",
        "",
        "Primary long-call decision surface from the frozen contract-selection bundle. This is assumption-relative and trust-aware by design.",
        "",
        "## Decision Snapshot",
        "",
        f"- Long-call rows in scope: `{int(len(bullish_action.index)) if not bullish_action.empty else 0}`",
        f"- Buy Now count: `{int(len(buy_now.index)) if not buy_now.empty else 0}`",
        f"- Watchlist count: `{int(len(watchlist.index)) if not watchlist.empty else 0}`",
        f"- Avoid count: `{int(len(avoid.index)) if not avoid.empty else 0}`",
        f"- Analysis trust level: `{clean_string(analysis_trust_level) or 'n/a'}`",
        "",
        "## Best Bullish Long Calls Right Now",
        "",
    ]
    lines.extend(_bucket_lines(bullish_action, "Buy Now", limit=4))
    lines.extend(["", "## Watchlist: Interesting But Not Buyable Yet", ""])
    lines.extend(_bucket_lines(bullish_action, "Watchlist", limit=6))
    lines.extend(["", "## Avoid For Now", ""])
    lines.extend(_bucket_lines(bullish_action, "Avoid For Now", limit=6))
    lines.extend(["", "## What Seems To Be Hurting Calls", ""])
    lines.extend(hurting_lines[:6])
    lines.extend(["", "## When Stock Is Still Better", ""])
    if stock_summary.empty:
        lines.append("- No stock baseline row was available.")
    else:
        for row in stock_summary.loc[stock_summary["strategy_family"].astype(str).str.lower().eq("long_stock")].head(2).to_dict("records"):
            lines.append(f"- `{clean_string(row.get('candidate_label'))}`: {clean_string(row.get('why_stock_may_be_better'))}")
    lines.extend(["", "## Key Triggers To Watch", ""])
    if bullish_triggers.empty:
        lines.append("- No bullish long-call triggers were generated.")
    else:
        for row in bullish_triggers.head(8).to_dict("records"):
            lines.append(
                f"- `{_short_contract_label(row)}` [{clean_string(row.get('trigger_type_label')) or clean_string(row.get('key_trigger_type'))}]: {clean_string(row.get('upgrade_rule') or row.get('what_has_to_happen'))} Invalidate if: {clean_string(row.get('invalidate_rule') or row.get('what_would_invalidate'))}"
            )
    lines.extend(
        [
            "",
            "## Trust / Data Notes",
            "",
            "- Buy Now can legitimately be empty if timing, IV, premium, or trust do not clear the bar.",
            "- Watchlist means the idea is interesting, but the current entry or path assumptions are still too demanding.",
            "- Prefer Stock Instead remains a first-class outcome when the options stay too fragile or too premium-heavy.",
            "",
            "## Best Next Files To Open",
            "",
            "- `summary/top_candidate_cards.md`",
            "- `charts/bullish_action_board_overview.png`",
            "- `charts/top_candidate_cards.png`",
            "- `charts/bullish_buy_watch_avoid_matrix.png`",
            "- `charts/stock_vs_option_preference_chart.png`",
            "- `tables/bullish_long_call_watchlist.csv`",
            "- `tables/bullish_long_call_avoid.csv`",
            "- then the relevant named path pack under `01_path_packs/`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_other_structures_markdown(*, ticker: str, other_structures: pd.DataFrame, stock_summary: pd.DataFrame) -> str:
    lines = [
        f"# {clean_string(ticker).upper()} Other Structures / Secondary Board",
        "",
        "Secondary structure read. Covered calls, cash-secured puts, spreads, and stock-preference notes stay visible, but they do not lead the main bullish long-call surface.",
        "",
        "## Secondary Structures Snapshot",
        "",
    ]
    if other_structures.empty:
        lines.append("- No secondary structures were promoted into the current shortlist.")
    else:
        for row in other_structures.head(8).to_dict("records"):
            lines.append(
                f"- `{clean_string(row.get('candidate_label'))}` [{clean_string(row.get('action_bucket'))}]: {clean_string(row.get('headline_reason'))}"
            )
    lines.extend(["", "## When Stock Still Leads", ""])
    stock_rows = stock_summary.loc[stock_summary.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_stock")].copy() if not stock_summary.empty else pd.DataFrame()
    if stock_rows.empty:
        lines.append("- No stock baseline row was available.")
    else:
        for row in stock_rows.head(2).to_dict("records"):
            lines.append(f"- `{clean_string(row.get('candidate_label'))}`: {clean_string(row.get('why_stock_may_be_better'))}")
    lines.extend(
        [
            "",
            "## How To Use This Section",
            "",
            "- Use this only after the bullish long-call board and stock-vs-option preference chart.",
            "- If a secondary structure still looks interesting, confirm it again in the matching path pack rather than relying on this overview alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def _bucket_lines(frame: pd.DataFrame, bucket: str, *, limit: int = 4) -> list[str]:
    if frame.empty:
        return [f"- No {bucket} rows."]
    subset = frame.loc[frame["action_bucket"].eq(bucket)].sort_values("action_priority_rank").head(limit)
    if subset.empty:
        return [f"- No {bucket} candidates under current assumptions."]
    lines: list[str] = []
    for row in subset.to_dict("records"):
        contract = _short_contract_label(row)
        confidence = clean_string(row.get("action_confidence")) or "n/a"
        interesting = clean_string(row.get("why_this_is_interesting_now")) or clean_string(row.get("headline_reason"))
        hurts = clean_string(row.get("what_is_hurting_this_candidate")) or clean_string(row.get("main_warning"))
        upgrade = clean_string(row.get("upgrade_rule") or row.get("what_has_to_happen") or row.get("main_trigger"))
        invalidate = clean_string(row.get("invalidate_rule") or row.get("what_would_invalidate"))
        if bucket == "Prefer Stock Instead":
            lines.append(
                f"- `{contract}` ({confidence}): {clean_string(row.get('why_stock_may_be_better')) or clean_string(row.get('headline_reason'))}"
            )
            continue
        if bucket == "Buy Now":
            lines.append(f"- `{contract}` ({confidence}): {interesting} Main watch-out: {hurts}")
            continue
        if bucket == "Watchlist":
            lines.append(f"- `{contract}` ({confidence}): {interesting} Held back by: {hurts} Upgrade if: {upgrade}")
            continue
        lines.append(f"- `{contract}` ({confidence}): {hurts} Avoid unless: {invalidate or upgrade}")
    return lines


def _build_action_board_markdown(*, ticker: str, action: pd.DataFrame, triggers: pd.DataFrame, analysis_trust_level: str) -> str:
    lines = [
        f"# {clean_string(ticker).upper()} Action Board",
        "",
        "Assumption-relative contract picker from the frozen contract-selection bundle. Use the bullish long-call board first, then read secondary structures only if you still want alternative implementations of the thesis.",
        "",
        "## What Looks Most Actionable Right Now",
        "",
    ]
    lines.extend(_bucket_lines(action, "Buy Now"))
    lines.extend(["", "## What Belongs On The Watchlist", ""])
    lines.extend(_bucket_lines(action, "Watchlist", limit=6))
    lines.extend(["", "## What To Avoid For Now", ""])
    lines.extend(_bucket_lines(action, "Avoid For Now", limit=6))
    lines.extend(["", "## When Stock Is Still Better", ""])
    lines.extend(_bucket_lines(action, "Prefer Stock Instead", limit=3))
    lines.extend(["", "## Key Triggers To Watch", ""])
    if triggers.empty:
        lines.append("- No explicit watch triggers were generated.")
    else:
        for row in triggers.head(8).to_dict("records"):
            lines.append(
                f"- `{_short_contract_label(row)}`: {clean_string(row.get('upgrade_rule') or row.get('key_trigger_value'))} by `{clean_string(row.get('key_trigger_deadline')) or 'n/a'}`. Invalidate if: {clean_string(row.get('invalidate_rule') or row.get('what_would_invalidate'))}"
            )
    lines.extend(
        [
            "",
            "## Trust / Data Caveats",
            "",
            f"- Analysis trust level: `{clean_string(analysis_trust_level) or 'n/a'}`",
            "- Buy Now requires acceptable quote trust, stock-relative edge, timing fit, and IV/path robustness.",
            "- Watchlist can include interesting convexity that is not yet attractive enough to buy.",
            "- Prefer Stock Instead is first-class when options do not clear premium, timing, IV, and trust hurdles.",
            "- Primary bullish call read: `summary/bullish_action_board.md`",
            "- Secondary structures read: `summary/other_structures.md`",
            "",
            "## How To Use The Path Packs Next",
            "",
            "- Start with `summary/bullish_action_board.md`, then inspect `charts/bullish_action_board_overview.png` and `charts/stock_vs_option_preference_chart.png`.",
            "- If a candidate is Watchlist, open `decision_triggers.csv` and then the matching stock-path pack under `01_path_packs/`.",
            "- Use value charts first, then delta-vs-stock charts, then IV-expanded ladders to see what must go right.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_action_board_outputs(*, ticker: str, tradeoff: pd.DataFrame, analysis_trust_level: str) -> dict[str, Any]:
    action = _build_action_board_candidates(tradeoff)
    triggers = _build_decision_triggers(action)
    breakdown = _build_action_board_score_breakdown(action)
    explanations = _build_action_board_explanations(action)
    bullish_action = _subset_action_board(action, include_families={"long_call"})
    bullish_triggers = _build_decision_triggers(bullish_action)
    bullish_breakdown = _build_action_board_score_breakdown(bullish_action)
    stock_summary = _build_stock_preference_summary(action)
    other_structures = _build_other_structures_summary(action)
    top_candidate_cards = _build_top_candidate_cards_frame(bullish_action)
    return {
        "action_board_candidates": action,
        "buy_now_candidates": action.loc[action.get("action_bucket", pd.Series(dtype=str)).eq("Buy Now")].copy() if not action.empty else pd.DataFrame(),
        "watchlist_candidates": action.loc[action.get("action_bucket", pd.Series(dtype=str)).eq("Watchlist")].copy() if not action.empty else pd.DataFrame(),
        "avoid_for_now_candidates": action.loc[action.get("action_bucket", pd.Series(dtype=str)).eq("Avoid For Now")].copy() if not action.empty else pd.DataFrame(),
        "prefer_stock_instead": action.loc[action.get("action_bucket", pd.Series(dtype=str)).eq("Prefer Stock Instead")].copy() if not action.empty else pd.DataFrame(),
        "decision_triggers": triggers,
        "action_board_score_breakdown": breakdown,
        "action_board_explanations": explanations,
        "action_board_markdown": _build_action_board_markdown(
            ticker=ticker,
            action=action,
            triggers=triggers,
            analysis_trust_level=analysis_trust_level,
        ),
        "bullish_long_call_action_board": bullish_action,
        "bullish_long_call_watchlist": bullish_action.loc[bullish_action.get("action_bucket", pd.Series(dtype=str)).eq("Watchlist")].copy() if not bullish_action.empty else pd.DataFrame(),
        "bullish_long_call_avoid": bullish_action.loc[bullish_action.get("action_bucket", pd.Series(dtype=str)).eq("Avoid For Now")].copy() if not bullish_action.empty else pd.DataFrame(),
        "bullish_long_call_triggers": bullish_triggers,
        "bullish_long_call_score_breakdown": bullish_breakdown,
        "other_structures_summary": other_structures,
        "stock_preference_summary": stock_summary,
        "bullish_action_board_markdown": _build_bullish_action_board_markdown(
            ticker=ticker,
            bullish_action=bullish_action,
            bullish_triggers=bullish_triggers,
            stock_summary=stock_summary,
            analysis_trust_level=analysis_trust_level,
        ),
        "top_candidate_cards": top_candidate_cards,
        "top_candidate_cards_markdown": _build_top_candidate_cards_markdown(
            ticker=ticker,
            cards=top_candidate_cards,
        ),
        "other_structures_markdown": _build_other_structures_markdown(
            ticker=ticker,
            other_structures=other_structures,
            stock_summary=stock_summary,
        ),
    }


def build_decision_highlights(
    *,
    ticker: str,
    candidate_comparison: pd.DataFrame,
    family_comparison: pd.DataFrame | None = None,
    path_view_tables: dict[str, pd.DataFrame] | None = None,
    analysis_trust_level: str | None = None,
) -> DecisionHighlightOutputs:
    """Build transparent decision highlights from frozen analysis tables."""

    robustness = _build_candidate_robustness_summary(candidate_comparison, path_view_tables or {})
    tradeoff = _build_tradeoff_matrix(robustness)
    highlights = _build_decision_highlights(tradeoff)
    explanations = _build_explanations(highlights, tradeoff)
    breakdown = _build_score_breakdown(tradeoff)
    takeaways = _build_stock_vs_option_takeaways(highlights, tradeoff)
    action_outputs = _build_action_board_outputs(
        ticker=ticker,
        tradeoff=tradeoff,
        analysis_trust_level=clean_string(analysis_trust_level),
    )
    markdown = _build_highlights_markdown(
        ticker=ticker,
        highlights=highlights,
        takeaways=takeaways,
        tradeoff=tradeoff,
        analysis_trust_level=clean_string(analysis_trust_level),
    )
    return DecisionHighlightOutputs(
        decision_highlights=highlights,
        decision_highlights_explanations=explanations,
        candidate_robustness_summary=robustness,
        candidate_tradeoff_matrix=tradeoff,
        stock_vs_option_takeaways=takeaways,
        highlights_score_breakdown=breakdown,
        highlights_markdown=markdown,
        action_board_candidates=action_outputs["action_board_candidates"],
        buy_now_candidates=action_outputs["buy_now_candidates"],
        watchlist_candidates=action_outputs["watchlist_candidates"],
        avoid_for_now_candidates=action_outputs["avoid_for_now_candidates"],
        prefer_stock_instead=action_outputs["prefer_stock_instead"],
        decision_triggers=action_outputs["decision_triggers"],
        action_board_score_breakdown=action_outputs["action_board_score_breakdown"],
        action_board_explanations=action_outputs["action_board_explanations"],
        action_board_markdown=action_outputs["action_board_markdown"],
        bullish_long_call_action_board=action_outputs["bullish_long_call_action_board"],
        bullish_long_call_watchlist=action_outputs["bullish_long_call_watchlist"],
        bullish_long_call_avoid=action_outputs["bullish_long_call_avoid"],
        bullish_long_call_triggers=action_outputs["bullish_long_call_triggers"],
        bullish_long_call_score_breakdown=action_outputs["bullish_long_call_score_breakdown"],
        other_structures_summary=action_outputs["other_structures_summary"],
        stock_preference_summary=action_outputs["stock_preference_summary"],
        bullish_action_board_markdown=action_outputs["bullish_action_board_markdown"],
        top_candidate_cards=action_outputs["top_candidate_cards"],
        top_candidate_cards_markdown=action_outputs["top_candidate_cards_markdown"],
        other_structures_markdown=action_outputs["other_structures_markdown"],
    )
