"""Shared path-analysis primitives for analysis-first workflows."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .models import (
    AssumedPathTraceRecord,
    CompareVsStockPathRecord,
    IVPathSensitivitySummaryRecord,
    IVPathTraceRecord,
    PathHorizonSpec,
    PathRiskSummaryRecord,
    RequiredPathSummaryRecord,
)
from ..utils import clean_string, finite_or_none, horizon_to_days


DEFAULT_REQUIRED_GOALS = [
    "itm_1c",
    "break_even",
    "return_25",
    "return_50",
    "outperform_stock",
]


def _stock_path_target_label(target_horizon_label: str) -> str:
    return clean_string(target_horizon_label).lower() or "target"


def stock_path_preset_points(
    *,
    preset: str,
    entry_spot: float,
    target_price: float,
    target_horizon_label: str,
) -> dict[str, float]:
    """Return normalized stock-path preset anchors for assumed and representative paths."""

    target_label = _stock_path_target_label(target_horizon_label)
    midpoint = (float(entry_spot) + float(target_price)) / 2.0
    presets: dict[str, dict[str, float]] = {
        "fast_bull": {"entry": entry_spot, "1w": midpoint, "1m": target_price, target_label: target_price},
        "slow_bull": {"entry": entry_spot, "1w": entry_spot * 1.02, "1m": midpoint, target_label: target_price},
        "flat": {"entry": entry_spot, "1m": entry_spot, target_label: entry_spot},
        "down_then_recover": {"entry": entry_spot, "1w": entry_spot * 0.88, "1m": entry_spot * 0.95, target_label: target_price},
        "gap_up_then_drift": {"entry": entry_spot, "1w": max(entry_spot * 1.12, midpoint), "1m": max(entry_spot * 1.08, midpoint), target_label: target_price},
        "gap_down_then_recover": {"entry": entry_spot, "1w": min(entry_spot * 0.88, midpoint), "1m": entry_spot * 0.94, target_label: target_price},
        "sharp_drop_mean_reversion": {"entry": entry_spot, "1w": entry_spot * 0.78, "1m": entry_spot * 0.90, target_label: midpoint},
        "earnings_gap_up": {"entry": entry_spot, "1w": max(target_price, entry_spot * 1.12), target_label: target_price},
        "earnings_gap_down": {"entry": entry_spot, "1w": min(target_price, entry_spot * 0.88), target_label: target_price},
        "rally_early_then_fade_then_rally_again": {
            "entry": entry_spot,
            "1w": max(entry_spot * 1.14, midpoint),
            "1m": max(entry_spot * 1.05, midpoint * 0.94),
            "3m": max(entry_spot * 1.10, midpoint * 1.02),
            target_label: target_price,
        },
        "range_bound_near_flat": {
            "entry": entry_spot,
            "1w": entry_spot * 1.02,
            "1m": entry_spot * 0.99,
            "3m": entry_spot * 1.01,
            target_label: target_price,
        },
        "down_first_then_recovery": {
            "entry": entry_spot,
            "1w": entry_spot * 0.86,
            "1m": entry_spot * 0.92,
            "3m": midpoint,
            target_label: target_price,
        },
        "late_breakout": {
            "entry": entry_spot,
            "1w": entry_spot * 1.00,
            "1m": entry_spot * 1.02,
            "3m": max(entry_spot * 1.04, midpoint * 0.92),
            target_label: target_price,
        },
        "early_move_above_strike_then_giveback": {
            "entry": entry_spot,
            "1w": max(entry_spot * 1.12, midpoint * 1.02),
            "1m": max(entry_spot * 1.09, midpoint),
            "3m": max(entry_spot * 1.04, midpoint * 0.96),
            target_label: target_price,
        },
        "reaches_target_late_near_expiry": {
            "entry": entry_spot,
            "1w": entry_spot * 1.01,
            "1m": entry_spot * 1.03,
            "3m": max(entry_spot * 1.06, midpoint * 0.94),
            target_label: target_price,
        },
        "plus_20_pct_in_1m": {
            "entry": entry_spot,
            "1w": entry_spot * 1.10,
            "1m": entry_spot * 1.20,
            "3m": entry_spot * 1.18,
            target_label: target_price,
        },
        "plus_30_pct_in_1m": {
            "entry": entry_spot,
            "1w": entry_spot * 1.14,
            "1m": entry_spot * 1.30,
            "3m": entry_spot * 1.27,
            target_label: target_price,
        },
        "plus_20_pct_in_1q": {
            "entry": entry_spot,
            "1w": entry_spot * 1.03,
            "1m": entry_spot * 1.08,
            "3m": entry_spot * 1.20,
            target_label: target_price,
        },
        "plus_30_pct_in_1q": {
            "entry": entry_spot,
            "1w": entry_spot * 1.05,
            "1m": entry_spot * 1.11,
            "3m": entry_spot * 1.30,
            target_label: target_price,
        },
        "quarter_up_then_pullback": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.18, midpoint * 1.08),
            "1m": max(entry_spot * 1.24, midpoint * 1.14),
            "2m": max(entry_spot * 1.10, midpoint * 1.01),
            target_label: target_price,
        },
        "quarter_down_then_next_quarter_recovery": {
            "entry": entry_spot,
            "2w": min(entry_spot * 0.78, midpoint * 0.84),
            "1m": min(entry_spot * 0.84, midpoint * 0.90),
            "2m": max(entry_spot * 0.94, midpoint * 0.98),
            target_label: target_price,
        },
        "two_quarters_down_then_flat_then_recovery": {
            "entry": entry_spot,
            "2w": min(entry_spot * 0.76, midpoint * 0.82),
            "1m": min(entry_spot * 0.80, midpoint * 0.86),
            "2m": min(entry_spot * 0.82, midpoint * 0.88),
            target_label: target_price,
        },
        "high_swing_quarterly_path": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.18, midpoint * 1.08),
            "1m": min(entry_spot * 0.86, midpoint * 0.90),
            "2m": max(entry_spot * 1.08, midpoint * 1.02),
            target_label: target_price,
        },
        "slow_grind_up": {
            "entry": entry_spot,
            "2w": entry_spot * 1.03,
            "1m": entry_spot * 1.08,
            "2m": max(entry_spot * 1.12, midpoint * 0.96),
            target_label: target_price,
        },
        "overshoot_then_mean_revert": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.22, midpoint * 1.12),
            "1m": max(entry_spot * 1.18, midpoint * 1.08),
            "2m": max(entry_spot * 1.08, midpoint * 1.00),
            target_label: target_price,
        },
        "quarter_up_then_hard_pullback": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.24, midpoint * 1.14),
            "1m": max(entry_spot * 1.18, midpoint * 1.08),
            "2m": min(entry_spot * 0.94, midpoint * 0.96),
            target_label: target_price,
        },
        "high_vol_sideways_then_breakout": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.12, midpoint * 1.02),
            "1m": min(entry_spot * 0.92, midpoint * 0.96),
            "2m": entry_spot * 1.02,
            target_label: target_price,
        },
        "earnings_gap_up_then_fade": {
            "entry": entry_spot,
            "1w": max(entry_spot * 1.24, midpoint * 1.14),
            "2w": max(entry_spot * 1.18, midpoint * 1.08),
            "1m": max(entry_spot * 1.10, midpoint),
            "2m": max(entry_spot * 1.02, midpoint * 0.94),
            target_label: target_price,
        },
        "earnings_gap_down_then_recovery": {
            "entry": entry_spot,
            "1w": min(entry_spot * 0.76, midpoint * 0.82),
            "2w": min(entry_spot * 0.84, midpoint * 0.90),
            "1m": entry_spot * 0.92,
            "2m": midpoint,
            target_label: target_price,
        },
        "false_breakout_then_recover": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.14, midpoint * 1.04),
            "1m": min(entry_spot * 0.86, midpoint * 0.90),
            "2m": entry_spot * 1.02,
            target_label: target_price,
        },
        "rally_then_long_range_then_second_leg_up": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.14, midpoint * 1.04),
            "1m": max(entry_spot * 1.08, midpoint * 0.98),
            "2m": max(entry_spot * 1.09, midpoint),
            target_label: target_price,
        },
        "violent_two_sided_quarter": {
            "entry": entry_spot,
            "2w": max(entry_spot * 1.24, midpoint * 1.14),
            "1m": min(entry_spot * 0.78, midpoint * 0.84),
            "2m": max(entry_spot * 1.13, midpoint * 1.04),
            target_label: target_price,
        },
        "slow_bleed_then_capitulation_then_bounce": {
            "entry": entry_spot,
            "2w": entry_spot * 0.94,
            "1m": entry_spot * 0.86,
            "2m": min(entry_spot * 0.74, midpoint * 0.82),
            target_label: target_price,
        },
        "early_breakout_to_target": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.58,
            "1m": entry_spot + (target_price - entry_spot) * 0.88,
            "3m": target_price,
            target_label: target_price,
        },
        "slow_grind_to_target": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.06,
            "1m": entry_spot + (target_price - entry_spot) * 0.18,
            "3m": entry_spot + (target_price - entry_spot) * 0.42,
            "6m": entry_spot + (target_price - entry_spot) * 0.68,
            target_label: target_price,
        },
        "down_then_recover_to_target": {
            "entry": entry_spot,
            "2w": max(entry_spot * 0.88, 0.5),
            "1m": max(entry_spot * 0.82, 0.5),
            "3m": entry_spot + (target_price - entry_spot) * 0.24,
            "6m": entry_spot + (target_price - entry_spot) * 0.64,
            target_label: target_price,
        },
        "rally_retrace_finish_target": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.72,
            "1m": entry_spot + (target_price - entry_spot) * 0.54,
            "3m": entry_spot + (target_price - entry_spot) * 0.40,
            "6m": entry_spot + (target_price - entry_spot) * 0.76,
            target_label: target_price,
        },
        "late_breakout_to_target": {
            "entry": entry_spot,
            "2w": entry_spot * 1.01,
            "1m": entry_spot * 1.02,
            "3m": entry_spot + (target_price - entry_spot) * 0.16,
            "6m": entry_spot + (target_price - entry_spot) * 0.38,
            target_label: target_price,
        },
        "overshoot_then_settle_at_target": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.48,
            "1m": entry_spot + (target_price - entry_spot) * 1.12,
            "3m": entry_spot + (target_price - entry_spot) * 1.18,
            "6m": entry_spot + (target_price - entry_spot) * 0.96,
            target_label: target_price,
        },
        "fast_overshoot_then_sideways": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.80,
            "1m": entry_spot + (target_price - entry_spot) * 1.22,
            "3m": entry_spot + (target_price - entry_spot) * 1.08,
            "6m": entry_spot + (target_price - entry_spot) * 1.02,
            target_label: target_price,
        },
        "weak_start_then_acceleration": {
            "entry": entry_spot,
            "2w": entry_spot * 0.98,
            "1m": entry_spot * 0.96,
            "3m": entry_spot + (target_price - entry_spot) * 0.24,
            "6m": entry_spot + (target_price - entry_spot) * 0.62,
            target_label: target_price,
        },
        "two_stage_bull_run": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.32,
            "1m": entry_spot + (target_price - entry_spot) * 0.48,
            "3m": entry_spot + (target_price - entry_spot) * 0.50,
            "6m": entry_spot + (target_price - entry_spot) * 0.78,
            target_label: target_price,
        },
        "violent_path_to_target": {
            "entry": entry_spot,
            "2w": entry_spot + (target_price - entry_spot) * 0.76,
            "1m": max(entry_spot * 0.88, 0.5),
            "3m": entry_spot + (target_price - entry_spot) * 1.10,
            "6m": entry_spot + (target_price - entry_spot) * 0.72,
            target_label: target_price,
        },
    }
    return presets.get(clean_string(preset).lower(), presets["slow_bull"])


def canonical_horizon_specs(target_horizon_label: str, target_horizon_days: int) -> list[PathHorizonSpec]:
    """Return the canonical ordered horizon grid for path-based analysis."""

    base = [
        PathHorizonSpec(label="entry", requested_days=0),
        PathHorizonSpec(label="1w", requested_days=7),
        PathHorizonSpec(label="1m", requested_days=30),
        PathHorizonSpec(label="3m", requested_days=90),
        PathHorizonSpec(label="6m", requested_days=180),
    ]
    if clean_string(target_horizon_label) not in {item.label for item in base}:
        base.append(
            PathHorizonSpec(
                label=clean_string(target_horizon_label) or "target",
                requested_days=max(int(target_horizon_days), 0),
            )
        )
    deduped: list[PathHorizonSpec] = []
    seen: set[str] = set()
    for item in sorted(base, key=lambda row: (int(row.requested_days), clean_string(row.label))):
        label = clean_string(item.label)
        if not label or label in seen:
            continue
        seen.add(label)
        deduped.append(PathHorizonSpec(label=label, requested_days=max(int(item.requested_days), 0)))
    return deduped


def horizon_spec_dicts(target_horizon_label: str, target_horizon_days: int) -> list[dict[str, Any]]:
    """Compatibility helper returning canonical horizon specs as dictionaries."""

    return [
        {"label": spec.label, "requested_days": spec.requested_days}
        for spec in canonical_horizon_specs(target_horizon_label, target_horizon_days)
    ]


def parse_path_points(raw_points: str | None, *, allow_entry: float | None = None) -> dict[str, float]:
    """Parse comma-separated path points into a normalized mapping."""

    mapping: dict[str, float] = {}
    has_explicit_points = bool(clean_string(raw_points))
    if has_explicit_points:
        for chunk in clean_string(raw_points).split(","):
            if ":" not in chunk:
                raise ValueError(f"Path point must use label:value format, got {chunk!r}")
            label, value = chunk.split(":", 1)
            label_text = clean_string(label).lower()
            if not label_text:
                continue
            try:
                mapping[label_text] = float(value)
            except ValueError as exc:
                raise ValueError(f"Path point value must be numeric, got {value!r}") from exc
    if has_explicit_points and "entry" not in mapping and allow_entry is not None:
        mapping["entry"] = float(allow_entry)
    return mapping


def interpolated_path(
    points: dict[str, float],
    horizon_specs: list[dict[str, Any]],
    *,
    default_value: float,
) -> dict[str, float]:
    """Interpolate named path points onto the canonical horizon grid."""

    def _label_to_days(label: str) -> int:
        cleaned = clean_string(label).lower()
        if cleaned in {"entry", "today", "now"}:
            return 0
        if cleaned.endswith("w"):
            return max(int(round(float(cleaned[:-1]) * 7)), 0)
        return horizon_to_days(cleaned)

    keyed_days: list[int] = []
    keyed_values: list[float] = []
    for label, value in points.items():
        keyed_days.append(_label_to_days(label))
        keyed_values.append(float(value))
    if not keyed_days:
        keyed_days = [0]
        keyed_values = [float(default_value)]
    ordering = np.argsort(np.asarray(keyed_days, dtype=float))
    days_sorted = np.asarray([keyed_days[index] for index in ordering], dtype=float)
    values_sorted = np.asarray([keyed_values[index] for index in ordering], dtype=float)
    result: dict[str, float] = {}
    for spec in horizon_specs:
        requested = int(spec["requested_days"])
        interpolated = float(np.interp(requested, days_sorted, values_sorted))
        result[clean_string(spec["label"]).lower()] = interpolated
    return result


def default_stock_path_points(
    *,
    preset: str,
    entry_spot: float,
    target_price: float,
    target_horizon_label: str,
) -> dict[str, float]:
    """Return normalized default stock path presets."""
    return stock_path_preset_points(
        preset=preset,
        entry_spot=entry_spot,
        target_price=target_price,
        target_horizon_label=target_horizon_label,
    )


def default_iv_path_points(*, preset: str, base_shift: float, target_horizon_label: str) -> dict[str, float]:
    """Return normalized default IV path presets."""

    presets: dict[str, dict[str, float]] = {
        "flat": {"entry": base_shift, target_horizon_label: base_shift},
        "iv_up_then_down": {"entry": base_shift, "1w": base_shift + 0.10, "1m": base_shift + 0.04, target_horizon_label: base_shift},
        "iv_down_then_stays_low": {"entry": base_shift, "1w": base_shift - 0.10, target_horizon_label: base_shift - 0.10},
        "earnings_build_then_crush": {"entry": base_shift, "1w": base_shift + 0.12, "1m": base_shift - 0.12, target_horizon_label: base_shift - 0.08},
        "mean_reversion_lower": {"entry": base_shift, "1m": base_shift - 0.08, target_horizon_label: base_shift - 0.12},
        "mean_reversion_higher": {"entry": base_shift, "1m": base_shift + 0.08, target_horizon_label: base_shift + 0.12},
    }
    return presets.get(clean_string(preset).lower(), presets["flat"])


def required_goals(goal: str, target_option_value: float | None) -> list[str]:
    """Return the ordered required-path goals for one run."""

    goals = list(DEFAULT_REQUIRED_GOALS)
    if goal not in goals and goal != "target_option_value":
        goals.append(goal)
    if goal == "target_option_value" or target_option_value is not None:
        goals.append("target_option_value")
    deduped: list[str] = []
    for item in goals:
        if item not in deduped:
            deduped.append(item)
    return deduped


def required_path_difficulty(case_path: dict[str, float], required_rows: pd.DataFrame) -> str:
    """Describe how hard the required path is relative to a candidate assumed path."""

    if required_rows.empty or bool(required_rows["unreached"].fillna(False).any()):
        return "unreached in sampled range"
    gaps: list[float] = []
    for _, row in required_rows.iterrows():
        label = clean_string(row.get("horizon")).lower()
        required_price = finite_or_none(row.get("required_stock_price"))
        case_price = finite_or_none(case_path.get(label))
        if required_price is None or case_price is None:
            continue
        gaps.append(float(case_price) - float(required_price))
    if not gaps:
        return "unreached in sampled range"
    min_gap = min(gaps)
    if min_gap >= 1.0:
        return "cleared comfortably"
    if min_gap >= -0.5:
        return "roughly matched"
    return "needs more / faster"


def compare_vs_stock_note(
    *,
    strategy_family: str,
    difference_vs_stock: float | None,
    difference_vs_stock_return_pct: float | None,
    clamped_to_expiry: bool,
    target_beyond_expiry: bool,
) -> str:
    """Return a short note explaining why stock or the option structure leads."""

    family = clean_string(strategy_family).lower()
    pnl_delta = finite_or_none(difference_vs_stock)
    return_delta = finite_or_none(difference_vs_stock_return_pct)
    comparable_delta = pnl_delta if pnl_delta is not None else return_delta
    if comparable_delta is None:
        return "Comparison versus stock is unavailable at this checkpoint."
    if target_beyond_expiry or clamped_to_expiry:
        return "Option edge is timing-sensitive here because the modeled horizon is clamped by expiry."
    if abs(float(comparable_delta)) <= 5.0 and (return_delta is None or abs(float(return_delta)) <= 0.01):
        return "Tracks long stock closely under this checkpoint."
    if float(comparable_delta) > 0:
        if family == "long_stock":
            return "This is the long-stock benchmark itself."
        if family in {"long_call", "long_put"}:
            return "Beats long stock here because convexity is paying for the move."
        if family in {"bull_call_spread", "bear_put_spread"}:
            return "Beats long stock here because the spread is monetizing the path efficiently."
        return "Beats long stock here under the active path assumptions."
    if family == "long_stock":
        return "Long stock is the benchmark and remains simpler here."
    if family in {"long_call", "long_put"}:
        return "Long stock is ahead here because time decay or IV pressure offsets the directional move."
    if family in {"bull_call_spread", "bear_put_spread"}:
        return "Long stock is ahead here because the capped structure gives up too much of the move."
    return "Long stock is ahead here under the active path assumptions."


def summarize_required_path_rows(
    required_path_rows: pd.DataFrame,
    *,
    assumed_path: dict[str, float],
    family_representatives: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize required-path rows into decision-friendly candidate and family views."""

    if required_path_rows.empty:
        return pd.DataFrame()

    family_representatives = family_representatives or {}
    representative_lookup = {
        clean_string(candidate_slug): clean_string(family)
        for family, candidate_slug in family_representatives.items()
        if clean_string(candidate_slug)
    }
    records: list[RequiredPathSummaryRecord] = []

    def append_record(group: pd.DataFrame, *, summary_scope: str, summary_label: str) -> None:
        ordered = group.sort_values("requested_days").copy()
        target_row = ordered.iloc[-1]
        unreached = bool(ordered["unreached"].fillna(False).any())
        first_cleared_horizon = None
        for _, row in ordered.iterrows():
            if bool(row.get("unreached")):
                continue
            label = clean_string(row.get("horizon")).lower()
            required_price = finite_or_none(row.get("required_stock_price"))
            assumed_price = finite_or_none(assumed_path.get(label))
            if required_price is None or assumed_price is None:
                continue
            if float(assumed_price) >= float(required_price):
                first_cleared_horizon = clean_string(row.get("horizon"))
                break
        target_label = clean_string(target_row.get("horizon")).lower()
        required_target = finite_or_none(target_row.get("required_stock_price"))
        assumed_target = finite_or_none(assumed_path.get(target_label))
        gap_at_target = (
            round(float(assumed_target) - float(required_target), 4)
            if required_target is not None and assumed_target is not None
            else None
        )
        records.append(
            RequiredPathSummaryRecord(
                summary_scope=summary_scope,
                summary_label=summary_label,
                candidate_slug=clean_string(target_row.get("candidate_slug")),
                candidate_label=clean_string(target_row.get("candidate_label")),
                strategy_family=clean_string(target_row.get("strategy_family")),
                goal=clean_string(target_row.get("goal")),
                iv_variant_kind=clean_string(target_row.get("iv_variant_kind")),
                iv_variant=clean_string(target_row.get("iv_variant")),
                first_cleared_horizon=first_cleared_horizon,
                required_stock_price_at_target=required_target,
                assumed_stock_price_at_target=assumed_target,
                path_gap_at_target=gap_at_target,
                required_path_difficulty=required_path_difficulty(assumed_path, ordered),
                unreached=unreached,
                clamped_to_expiry=bool(ordered["clamped_to_expiry"].fillna(False).any()),
                target_beyond_expiry=bool(ordered["target_beyond_expiry"].fillna(False).any()),
            )
        )

    grouping = ["candidate_slug", "goal", "iv_variant_kind", "iv_variant"]
    for _, group in required_path_rows.groupby(grouping, dropna=False):
        if group.empty:
            continue
        candidate_slug = clean_string(group.iloc[0].get("candidate_slug"))
        append_record(
            group,
            summary_scope="candidate",
            summary_label=clean_string(group.iloc[0].get("candidate_label")) or candidate_slug,
        )
        representative_family = representative_lookup.get(candidate_slug)
        if representative_family:
            append_record(
                group,
                summary_scope="family_representative",
                summary_label=representative_family,
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([record.__dict__ for record in records])


def build_assumed_path_trace_rows(
    *,
    representative_candidates: list[dict[str, Any]],
    horizon_specs: list[dict[str, Any]],
    stock_path: dict[str, float],
    iv_path: dict[str, float],
    comparison_capital: float,
    evaluate_point: Callable[..., dict[str, Any]],
    include_top_candidate: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate the active assumed stock + IV path for family reps and the top candidate."""

    traces: list[AssumedPathTraceRecord] = []
    seen_candidates: set[tuple[str, str]] = set()
    candidate_items = list(representative_candidates)
    if include_top_candidate:
        candidate_items.append(include_top_candidate)

    for candidate in candidate_items:
        spec = candidate.get("spec")
        if spec is None:
            continue
        candidate_slug = clean_string(candidate.get("candidate_slug") or spec.get("candidate_slug"))
        trace_scope = clean_string(candidate.get("trace_scope") or "family_representative")
        dedupe_key = (trace_scope, candidate_slug)
        if not candidate_slug or dedupe_key in seen_candidates:
            continue
        seen_candidates.add(dedupe_key)
        running_worst: float | None = None
        running_peak: float | None = None
        for horizon in horizon_specs:
            label = clean_string(horizon["label"]).lower()
            spot_price = finite_or_none(stock_path.get(label))
            iv_shift_points = finite_or_none(iv_path.get(label))
            if spot_price is None:
                continue
            evaluation = evaluate_point(
                spec,
                spot_price=float(spot_price),
                horizon_days=int(horizon["requested_days"]),
                iv_shift_points=float(iv_shift_points or 0.0),
                comparison_capital=float(comparison_capital),
            )
            profit_loss = finite_or_none(evaluation.get("profit_loss"))
            stock_modeled_value = finite_or_none(evaluation.get("stock_estimated_value"))
            stock_profit_loss = finite_or_none(evaluation.get("stock_profit_loss"))
            stock_return_on_capital = finite_or_none(evaluation.get("stock_return_on_comparison_capital"))
            return_on_capital = finite_or_none(evaluation.get("return_on_comparison_capital"))
            return_delta = (
                round(float(return_on_capital) - float(stock_return_on_capital), 6)
                if return_on_capital is not None and stock_return_on_capital is not None
                else None
            )
            if profit_loss is not None:
                running_worst = profit_loss if running_worst is None else min(running_worst, profit_loss)
                running_peak = profit_loss if running_peak is None else max(running_peak, profit_loss)
            drawdown_to_date = (
                round(float(profit_loss) - float(running_peak), 4)
                if profit_loss is not None and running_peak is not None
                else None
            )
            traces.append(
                AssumedPathTraceRecord(
                    trace_scope=trace_scope,
                    series_label=clean_string(candidate.get("series_label"))
                    or clean_string(candidate.get("candidate_label"))
                    or candidate_slug,
                    candidate_slug=candidate_slug,
                    candidate_label=clean_string(candidate.get("candidate_label") or spec.get("candidate_label")),
                    strategy_family=clean_string(candidate.get("strategy_family") or spec.get("strategy_family")),
                    horizon=label,
                    requested_days=int(horizon["requested_days"]),
                    spot_price=float(spot_price),
                    iv_shift_points=float(iv_shift_points or 0.0),
                    modeled_value=finite_or_none(evaluation.get("estimated_value")),
                    profit_loss=profit_loss,
                    return_on_comparison_capital=return_on_capital,
                    stock_modeled_value=stock_modeled_value,
                    stock_profit_loss=stock_profit_loss,
                    stock_return_on_comparison_capital=stock_return_on_capital,
                    difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                    difference_vs_stock_return_pct=return_delta,
                    benchmark_note=compare_vs_stock_note(
                        strategy_family=clean_string(candidate.get("strategy_family") or spec.get("strategy_family")),
                        difference_vs_stock=finite_or_none(evaluation.get("difference_vs_stock")),
                        difference_vs_stock_return_pct=return_delta,
                        clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                        target_beyond_expiry=bool(evaluation.get("target_beyond_expiry")),
                    ),
                    worst_interim_profit_loss_to_date=running_worst,
                    drawdown_from_peak_to_date=drawdown_to_date,
                    clamped_to_expiry=bool(evaluation.get("clamped_to_expiry")),
                    target_beyond_expiry=bool(evaluation.get("target_beyond_expiry")),
                )
            )
    if not traces:
        return pd.DataFrame()
    return pd.DataFrame([trace.__dict__ for trace in traces])


def build_iv_path_trace_rows(
    *,
    horizon_specs: list[dict[str, Any]],
    active_iv_path_name: str,
    active_iv_path: dict[str, float],
    comparison_iv_paths: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Persist the active IV path alongside comparison presets on the canonical horizon grid."""

    records: list[IVPathTraceRecord] = []
    active_label = clean_string(active_iv_path_name).lower()
    seen_paths: set[str] = set()
    ordered_paths = [(active_label, active_iv_path)] + [
        (clean_string(name).lower(), path)
        for name, path in comparison_iv_paths.items()
        if clean_string(name).lower() != active_label
    ]
    for path_name, path_points in ordered_paths:
        if not path_name or path_name in seen_paths:
            continue
        seen_paths.add(path_name)
        entry_value = finite_or_none(path_points.get("entry"))
        for horizon in horizon_specs:
            label = clean_string(horizon["label"]).lower()
            iv_shift_points = finite_or_none(path_points.get(label))
            if iv_shift_points is None:
                continue
            records.append(
                IVPathTraceRecord(
                    trace_scope="active_assumption" if path_name == active_label else "comparison_preset",
                    iv_path_name=path_name,
                    variant_kind="active" if path_name == active_label else "preset",
                    horizon=label,
                    requested_days=int(horizon["requested_days"]),
                    iv_shift_points=float(iv_shift_points),
                    delta_from_entry_iv_shift=(
                        round(float(iv_shift_points) - float(entry_value), 6)
                        if entry_value is not None
                        else None
                    ),
                )
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([record.__dict__ for record in records])


def build_compare_vs_stock_path_rows(assumed_path_trace_rows: pd.DataFrame) -> pd.DataFrame:
    """Convert assumed-path trace rows into an explicit compare-vs-stock path table."""

    if assumed_path_trace_rows.empty:
        return pd.DataFrame()
    rows: list[CompareVsStockPathRecord] = []
    for _, row in assumed_path_trace_rows.iterrows():
        rows.append(
            CompareVsStockPathRecord(
                trace_scope=clean_string(row.get("trace_scope")),
                series_label=clean_string(row.get("series_label")),
                candidate_slug=clean_string(row.get("candidate_slug")),
                candidate_label=clean_string(row.get("candidate_label")),
                strategy_family=clean_string(row.get("strategy_family")),
                horizon=clean_string(row.get("horizon")),
                requested_days=int(row.get("requested_days") or 0),
                strategy_modeled_value=finite_or_none(row.get("modeled_value")),
                strategy_profit_loss=finite_or_none(row.get("profit_loss")),
                strategy_return_on_comparison_capital=finite_or_none(row.get("return_on_comparison_capital")),
                stock_modeled_value=finite_or_none(row.get("stock_modeled_value")),
                stock_profit_loss=finite_or_none(row.get("stock_profit_loss")),
                stock_return_on_comparison_capital=finite_or_none(row.get("stock_return_on_comparison_capital")),
                delta_profit_loss_vs_stock=finite_or_none(row.get("difference_vs_stock")),
                delta_return_pct_vs_stock=finite_or_none(row.get("difference_vs_stock_return_pct")),
                benchmark_note=clean_string(row.get("benchmark_note")),
                clamped_to_expiry=bool(row.get("clamped_to_expiry")),
                target_beyond_expiry=bool(row.get("target_beyond_expiry")),
            )
        )
    return pd.DataFrame([record.__dict__ for record in rows])


def build_iv_path_sensitivity_summary(
    *,
    path_case_summary: pd.DataFrame,
    stock_path_name: str,
    active_iv_path_name: str,
    family_representatives: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize how outcomes change across IV-path presets for candidates and family representatives."""

    if path_case_summary.empty:
        return pd.DataFrame()
    filtered = path_case_summary.loc[
        path_case_summary.get("stock_path").astype(str).str.lower() == clean_string(stock_path_name).lower()
    ].copy()
    if filtered.empty:
        return pd.DataFrame()
    family_representatives = family_representatives or {}
    representative_lookup = {
        clean_string(candidate_slug): clean_string(family)
        for family, candidate_slug in family_representatives.items()
        if clean_string(candidate_slug)
    }
    active_label = clean_string(active_iv_path_name).lower()
    records: list[IVPathSensitivitySummaryRecord] = []

    def append_record(group: pd.DataFrame, *, summary_scope: str, summary_label: str) -> None:
        ordered = group.sort_values("final_profit_loss", ascending=False).copy()
        best = ordered.iloc[0]
        worst = ordered.iloc[-1]
        active = group.loc[group.get("iv_path").astype(str).str.lower() == active_label].copy()
        active_row = active.iloc[0] if not active.empty else group.iloc[0]
        best_profit = finite_or_none(best.get("final_profit_loss"))
        worst_profit = finite_or_none(worst.get("final_profit_loss"))
        best_diff = finite_or_none(best.get("final_difference_vs_stock"))
        worst_diff = finite_or_none(worst.get("final_difference_vs_stock"))
        best_return = finite_or_none(best.get("final_return_on_comparison_capital"))
        worst_return = finite_or_none(worst.get("final_return_on_comparison_capital"))
        pnl_range = (
            round(float(best_profit) - float(worst_profit), 4)
            if best_profit is not None and worst_profit is not None
            else None
        )
        diff_range = (
            round(float(best_diff) - float(worst_diff), 4)
            if best_diff is not None and worst_diff is not None
            else None
        )
        return_range = (
            round(float(best_return) - float(worst_return), 6)
            if best_return is not None and worst_return is not None
            else None
        )
        if pnl_range is not None and pnl_range >= 250:
            iv_risk = "high iv dependence"
        elif pnl_range is not None and pnl_range >= 100:
            iv_risk = "moderate iv dependence"
        else:
            iv_risk = "low iv dependence"
        sensitivity_note = (
            f"Profit spans about ${pnl_range:,.0f} across IV-path presets."
            if pnl_range is not None
            else "IV-path sensitivity is sparse in the sampled presets."
        )
        records.append(
            IVPathSensitivitySummaryRecord(
                summary_scope=summary_scope,
                summary_label=summary_label,
                candidate_slug=clean_string(active_row.get("candidate_slug")),
                candidate_label=clean_string(active_row.get("candidate_label")),
                strategy_family=clean_string(active_row.get("strategy_family")),
                stock_path_name=clean_string(stock_path_name).lower(),
                active_iv_path_name=active_label,
                best_iv_variant=clean_string(best.get("iv_path")),
                worst_iv_variant=clean_string(worst.get("iv_path")),
                active_profit_loss=finite_or_none(active_row.get("final_profit_loss")),
                active_difference_vs_stock=finite_or_none(active_row.get("final_difference_vs_stock")),
                active_return_on_comparison_capital=finite_or_none(active_row.get("final_return_on_comparison_capital")),
                best_profit_loss=best_profit,
                worst_profit_loss=worst_profit,
                pnl_sensitivity_range=pnl_range,
                best_difference_vs_stock=best_diff,
                worst_difference_vs_stock=worst_diff,
                difference_vs_stock_range=diff_range,
                best_return_on_comparison_capital=best_return,
                worst_return_on_comparison_capital=worst_return,
                return_sensitivity_range=return_range,
                iv_risk=iv_risk,
                sensitivity_note=sensitivity_note,
            )
        )

    grouping = ["candidate_slug"]
    for _, group in filtered.groupby(grouping, dropna=False):
        if group.empty:
            continue
        candidate_slug = clean_string(group.iloc[0].get("candidate_slug"))
        append_record(
            group,
            summary_scope="candidate",
            summary_label=clean_string(group.iloc[0].get("candidate_label")) or candidate_slug,
        )
        representative_family = representative_lookup.get(candidate_slug)
        if representative_family:
            append_record(
                group,
                summary_scope="family_representative",
                summary_label=representative_family,
            )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([record.__dict__ for record in records])


def build_path_risk_summary(
    *,
    required_path_summary: pd.DataFrame,
    assumed_path_trace_rows: pd.DataFrame,
    compare_vs_stock_path_rows: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    iv_path_sensitivity_summary: pd.DataFrame,
    goal: str,
    stock_path_name: str,
    iv_path_name: str,
) -> pd.DataFrame:
    """Summarize path difficulty, timing risk, IV risk, and benchmark edge on disk."""

    if required_path_summary.empty:
        return pd.DataFrame()
    candidate_lookup = {
        clean_string(row.get("candidate_slug")): row.to_dict()
        for _, row in candidate_summary.iterrows()
        if clean_string(row.get("candidate_slug"))
    }
    sensitivity_lookup = {}
    for _, row in iv_path_sensitivity_summary.iterrows():
        sensitivity_lookup[(clean_string(row.get("summary_scope")), clean_string(row.get("candidate_slug")))] = row.to_dict()
    trace_lookup = {}
    for _, row in assumed_path_trace_rows.iterrows():
        key = (clean_string(row.get("trace_scope")), clean_string(row.get("candidate_slug")))
        trace_lookup.setdefault(key, []).append(row.to_dict())
    compare_lookup = {}
    for _, row in compare_vs_stock_path_rows.iterrows():
        key = (clean_string(row.get("trace_scope")), clean_string(row.get("candidate_slug")))
        compare_lookup.setdefault(key, []).append(row.to_dict())

    records: list[PathRiskSummaryRecord] = []
    working = required_path_summary.loc[
        (required_path_summary.get("goal").astype(str).str.lower() == clean_string(goal).lower())
        & (required_path_summary.get("iv_variant").astype(str).str.lower() == clean_string(iv_path_name).lower())
        & (required_path_summary.get("iv_variant_kind").astype(str).str.lower() == "path")
    ].copy()
    for _, row in working.iterrows():
        summary_scope = clean_string(row.get("summary_scope"))
        candidate_slug = clean_string(row.get("candidate_slug"))
        trace_key = ("family_representative" if summary_scope == "family_representative" else "top_candidate", candidate_slug)
        traces = pd.DataFrame(trace_lookup.get(trace_key) or trace_lookup.get(("family_representative", candidate_slug)) or [])
        compares = pd.DataFrame(compare_lookup.get(trace_key) or compare_lookup.get(("family_representative", candidate_slug)) or [])
        candidate_meta = candidate_lookup.get(candidate_slug, {})
        sensitivity = sensitivity_lookup.get((summary_scope, candidate_slug)) or sensitivity_lookup.get(("candidate", candidate_slug)) or {}

        worst_interim = (
            finite_or_none(pd.to_numeric(traces.get("profit_loss"), errors="coerce").min())
            if not traces.empty
            else None
        )
        worst_drawdown = (
            finite_or_none(pd.to_numeric(traces.get("drawdown_from_peak_to_date"), errors="coerce").min())
            if not traces.empty
            else None
        )
        benchmark_edge = (
            finite_or_none(pd.to_numeric(compares.get("delta_profit_loss_vs_stock"), errors="coerce").iloc[-1])
            if not compares.empty
            else None
        )
        benchmark_return_edge = (
            finite_or_none(pd.to_numeric(compares.get("delta_return_pct_vs_stock"), errors="coerce").iloc[-1])
            if not compares.empty
            else None
        )
        benchmark_note = clean_string(compares.iloc[-1].get("benchmark_note")) if not compares.empty else ""
        timing_risk = clean_string(candidate_meta.get("time_sensitivity_summary"))
        if not timing_risk:
            if bool(row.get("target_beyond_expiry")):
                timing_risk = "Timing risk is high because the thesis runs beyond expiry."
            elif clean_string(row.get("required_path_difficulty")) == "needs more / faster":
                timing_risk = "Needs a faster move than the active assumed path."
            else:
                timing_risk = "Timing risk is moderate under the active path."
        iv_risk = clean_string(sensitivity.get("iv_risk")) or clean_string(candidate_meta.get("iv_sensitivity_summary"))
        if not iv_risk:
            iv_risk = "IV-path sensitivity is moderate."
        success_dependency = "eventual target achievement is usually enough"
        first_cleared = clean_string(row.get("first_cleared_horizon"))
        if bool(row.get("target_beyond_expiry")) or "extra month" in timing_risk.lower() or "clamp" in timing_risk.lower():
            success_dependency = "success depends on a faster move before expiry"
        elif first_cleared and first_cleared in {"entry", "1w"}:
            success_dependency = "success improves sharply if the move happens early"
        records.append(
            PathRiskSummaryRecord(
                summary_scope=summary_scope,
                summary_label=clean_string(row.get("summary_label")),
                candidate_slug=candidate_slug,
                candidate_label=clean_string(row.get("candidate_label")),
                strategy_family=clean_string(row.get("strategy_family")),
                goal=clean_string(goal).lower(),
                stock_path_name=clean_string(stock_path_name).lower(),
                iv_path_name=clean_string(iv_path_name).lower(),
                required_path_difficulty=clean_string(row.get("required_path_difficulty")),
                first_cleared_horizon=first_cleared or None,
                path_gap_at_target=finite_or_none(row.get("path_gap_at_target")),
                timing_risk=timing_risk,
                iv_risk=iv_risk,
                success_dependency=success_dependency,
                max_downside=finite_or_none(candidate_meta.get("max_loss")),
                worst_interim_profit_loss=worst_interim,
                worst_drawdown_from_peak=worst_drawdown,
                benchmark_edge=benchmark_edge,
                benchmark_return_edge=benchmark_return_edge,
                benchmark_note=benchmark_note,
                confidence_label=clean_string(candidate_meta.get("confidence_label")),
                coverage_flags=clean_string(candidate_meta.get("coverage_flags")),
                target_beyond_expiry=bool(candidate_meta.get("target_beyond_expiry") or row.get("target_beyond_expiry")),
            )
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame([record.__dict__ for record in records])
