"""Path simulation primitives for analysis-first contract-selection workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import exp, log, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .paths import (
    default_iv_path_points as _default_iv_path_points,
    stock_path_preset_points as _stock_path_preset_points,
)
from ..utils import clean_string


BUSINESS_DAYS_PER_YEAR = 252.0
PATH_CENTRIC_FOCUS_PRESETS = [
    "rally_early_then_fade_then_rally_again",
    "range_bound_near_flat",
    "down_first_then_recovery",
    "late_breakout",
    "early_move_above_strike_then_giveback",
    "reaches_target_late_near_expiry",
]
STOCK_PATH_GALLERY_PRESETS = [
    *PATH_CENTRIC_FOCUS_PRESETS,
    "plus_20_pct_in_1m",
    "plus_30_pct_in_1m",
    "plus_20_pct_in_1q",
    "plus_30_pct_in_1q",
    "quarter_up_then_pullback",
    "quarter_down_then_next_quarter_recovery",
    "two_quarters_down_then_flat_then_recovery",
    "high_swing_quarterly_path",
    "slow_grind_up",
    "overshoot_then_mean_revert",
    "quarter_up_then_hard_pullback",
    "high_vol_sideways_then_breakout",
    "earnings_gap_up_then_fade",
    "earnings_gap_down_then_recovery",
    "false_breakout_then_recover",
    "rally_then_long_range_then_second_leg_up",
    "violent_two_sided_quarter",
    "slow_bleed_then_capitulation_then_bounce",
]
IV_PATH_GALLERY_PRESETS = [
    "flat",
    "mean_reversion_lower",
    "mean_reversion_higher",
    "iv_up_then_down",
    "iv_down_then_stays_low",
    "earnings_build_then_crush",
]
STOCK_PATH_FAMILY_DEFINITIONS: dict[str, dict[str, str]] = {
    "minimum_required_path": {
        "path_family": "minimum_required_path",
        "path_family_label": "Minimum Required Path",
        "timing_shape": "contract_specific_threshold",
        "outcome_bias": "required_threshold",
        "path_description": "Minimum stock path required for the selected option to clear the active decision goal.",
    },
    "early_rally": {
        "path_family": "early_rally",
        "path_family_label": "Early Rally",
        "timing_shape": "front_loaded_upside",
        "outcome_bias": "option_friendly_if_held",
        "path_description": "Upside arrives early enough to help convex options before theta decay dominates.",
    },
    "late_rally": {
        "path_family": "late_rally",
        "path_family_label": "Late Rally",
        "timing_shape": "back_loaded_upside",
        "outcome_bias": "timing_sensitive",
        "path_description": "The endpoint can be bullish, but the move arrives late and can leave calls behind stock.",
    },
    "steady_grind_up": {
        "path_family": "steady_grind_up",
        "path_family_label": "Steady Grind-Up",
        "timing_shape": "smooth_uptrend",
        "outcome_bias": "balanced_bullish",
        "path_description": "A steady upward path that tests whether the option works without a dramatic spike.",
    },
    "false_breakout": {
        "path_family": "false_breakout",
        "path_family_label": "False Breakout",
        "timing_shape": "spike_then_giveback",
        "outcome_bias": "exit_rule_sensitive",
        "path_description": "An early move can briefly help, but giveback tests whether the option needs perfect timing.",
    },
    "recovery": {
        "path_family": "recovery",
        "path_family_label": "Recovery",
        "timing_shape": "down_then_recover",
        "outcome_bias": "drawdown_sensitive",
        "path_description": "Down first, then recovery; useful for seeing whether the contract can survive a bad start.",
    },
    "earnings_gap": {
        "path_family": "earnings_gap",
        "path_family_label": "Earnings Gap",
        "timing_shape": "event_gap_then_follow_through",
        "outcome_bias": "event_sensitive",
        "path_description": "Event-style gap behavior that can separate stock-path effect from timing and IV risk.",
    },
    "quarter_pullback": {
        "path_family": "quarter_pullback",
        "path_family_label": "Quarter-Up Then Pullback",
        "timing_shape": "quarterly_spike_then_pullback",
        "outcome_bias": "profit_taking_sensitive",
        "path_description": "A strong move fades before the target window, testing whether gains must be harvested early.",
    },
    "range_bound": {
        "path_family": "range_bound",
        "path_family_label": "Range-Bound",
        "timing_shape": "sideways_chop",
        "outcome_bias": "stock_or_wait",
        "path_description": "Mostly sideways behavior that exposes premium decay and weak edge cases.",
    },
    "volatile_two_sided": {
        "path_family": "volatile_two_sided",
        "path_family_label": "Volatile Two-Sided",
        "timing_shape": "large_swings",
        "outcome_bias": "timing_and_exit_sensitive",
        "path_description": "Large swings in both directions, useful for stress-testing path dependence.",
    },
    "downside_failure": {
        "path_family": "downside_failure",
        "path_family_label": "Downside Failure",
        "timing_shape": "bleed_lower",
        "outcome_bias": "option_failure",
        "path_description": "A weak path where the bullish option should normally fail or remain too narrow.",
    },
    "overshoot_mean_revert": {
        "path_family": "overshoot_mean_revert",
        "path_family_label": "Overshoot Then Mean-Revert",
        "timing_shape": "overshoot_then_settle",
        "outcome_bias": "exit_rule_sensitive",
        "path_description": "The endpoint can look fine, but the route tests whether early overshoot matters more than finish.",
    },
    "custom_or_active": {
        "path_family": "custom_or_active",
        "path_family_label": "Custom / Active Assumption",
        "timing_shape": "user_defined",
        "outcome_bias": "assumption_specific",
        "path_description": "User-provided or active assumed path, kept separate from the built-in scenario library.",
    },
}

_STOCK_PATH_FAMILY_BY_PRESET = {
    "minimum_required_path": "minimum_required_path",
    "rally_early_then_fade_then_rally_again": "early_rally",
    "early_breakout_to_target": "early_rally",
    "plus_20_pct_in_1m": "early_rally",
    "plus_30_pct_in_1m": "early_rally",
    "plus_20_pct_in_1q": "steady_grind_up",
    "plus_30_pct_in_1q": "steady_grind_up",
    "late_breakout": "late_rally",
    "late_breakout_to_target": "late_rally",
    "reaches_target_late_near_expiry": "late_rally",
    "slow_grind_up": "steady_grind_up",
    "slow_grind_to_target": "steady_grind_up",
    "two_stage_bull_run": "steady_grind_up",
    "rally_then_long_range_then_second_leg_up": "steady_grind_up",
    "false_breakout_then_recover": "false_breakout",
    "early_move_above_strike_then_giveback": "false_breakout",
    "rally_retrace_finish_target": "false_breakout",
    "down_first_then_recovery": "recovery",
    "down_then_recover_to_target": "recovery",
    "earnings_gap_down_then_recovery": "recovery",
    "quarter_down_then_next_quarter_recovery": "recovery",
    "earnings_gap_up_then_fade": "earnings_gap",
    "quarter_up_then_pullback": "quarter_pullback",
    "quarter_up_then_hard_pullback": "quarter_pullback",
    "range_bound_near_flat": "range_bound",
    "high_vol_sideways_then_breakout": "volatile_two_sided",
    "high_swing_quarterly_path": "volatile_two_sided",
    "violent_two_sided_quarter": "volatile_two_sided",
    "violent_path_to_target": "volatile_two_sided",
    "slow_bleed_then_capitulation_then_bounce": "downside_failure",
    "two_quarters_down_then_flat_then_recovery": "downside_failure",
    "overshoot_then_mean_revert": "overshoot_mean_revert",
    "overshoot_then_settle_at_target": "overshoot_mean_revert",
    "fast_overshoot_then_sideways": "overshoot_mean_revert",
    "weak_start_then_acceleration": "recovery",
    "active_assumed_path": "custom_or_active",
}


def stock_path_family_metadata(path_name: str) -> dict[str, str]:
    """Return stable product labels for a named stock-path shape."""

    normalized = clean_string(path_name).lower()
    family_key = _STOCK_PATH_FAMILY_BY_PRESET.get(normalized, "custom_or_active")
    family = dict(STOCK_PATH_FAMILY_DEFINITIONS[family_key])
    family["path_name"] = normalized
    family["path_label"] = humanize_named_path(normalized, kind="stock")
    return family


def build_stock_path_library_rows(*, active_path_name: str | None = None) -> pd.DataFrame:
    """Return one-row-per-path metadata for the full stock-path scenario library."""

    active_name = clean_string(active_path_name).lower()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_path(path_name: str, *, library_role: str, display_order: int, is_active: bool = False) -> None:
        normalized = clean_string(path_name).lower()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        meta = stock_path_family_metadata(normalized)
        rows.append(
            {
                "path_name": normalized,
                "path_label": meta["path_label"],
                "path_family": meta["path_family"],
                "path_family_label": meta["path_family_label"],
                "timing_shape": meta["timing_shape"],
                "outcome_bias": meta["outcome_bias"],
                "library_role": library_role,
                "display_order": int(display_order),
                "is_active_assumed": bool(is_active),
                "path_description": meta["path_description"],
            }
        )

    add_path("minimum_required_path", library_role="contract_specific_decision_path", display_order=0)
    for index, preset_name in enumerate(STOCK_PATH_GALLERY_PRESETS, start=10):
        add_path(preset_name, library_role="broad_scenario_gallery", display_order=index, is_active=preset_name == active_name)
    for index, preset_name in enumerate(
        [
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
        ],
        start=100,
    ):
        add_path(preset_name, library_role="target_endpoint_thesis_path", display_order=index, is_active=preset_name == active_name)
    if active_name and active_name not in seen:
        add_path(active_name, library_role="active_custom_assumption", display_order=999, is_active=True)
    return pd.DataFrame(rows).sort_values(["display_order", "path_name"]).reset_index(drop=True)


@dataclass(frozen=True)
class StockPathExample:
    path_id: str
    path_kind: str
    path_name: str
    path_points: list[dict[str, Any]]
    representative_bucket: str | None = None
    selection_reason: str | None = None
    is_representative: bool = False


@dataclass(frozen=True)
class IVPathExample:
    iv_path_id: str
    iv_path_name: str
    path_points: list[dict[str, Any]]
    representative_bucket: str | None = None
    selection_reason: str | None = None
    is_representative: bool = False


@dataclass(frozen=True)
class PathPairExample:
    path_pair_id: str
    stock_path_id: str
    iv_path_id: str
    stock_path_name: str
    iv_path_name: str
    stock_path_kind: str
    iv_path_kind: str
    stock_points: list[dict[str, Any]] = field(default_factory=list)
    iv_points: list[dict[str, Any]] = field(default_factory=list)


def humanize_named_path(value: str, *, kind: str = "stock") -> str:
    text = clean_string(value)
    if not text:
        return "Active Assumed Path" if kind == "stock" else "Active Assumed IV"
    labels = {
        "rally_early_then_fade_then_rally_again": "Rally Early, Fade, Rally Again",
        "range_bound_near_flat": "Range-Bound, Near Flat",
        "down_first_then_recovery": "Down First, Then Recovery",
        "late_breakout": "Late Breakout",
        "early_move_above_strike_then_giveback": "Early Move Above Strike, Then Giveback",
        "reaches_target_late_near_expiry": "Reaches Target Late Near Expiry",
        "plus_20_pct_in_1m": "+20% Within 1 Month",
        "plus_30_pct_in_1m": "+30% Within 1 Month",
        "plus_20_pct_in_1q": "+20% Within 1 Quarter",
        "plus_30_pct_in_1q": "+30% Within 1 Quarter",
        "quarter_up_then_pullback": "Quarter Up, Then Pullback",
        "quarter_down_then_next_quarter_recovery": "Quarter Down, Next Quarter Recovery",
        "two_quarters_down_then_flat_then_recovery": "Two Quarters Down, Flat, Then Recovery",
        "high_swing_quarterly_path": "High-Swing Quarterly Path",
        "slow_grind_up": "Slow Grind Up",
        "overshoot_then_mean_revert": "Overshoot, Then Mean Revert",
        "quarter_up_then_hard_pullback": "Quarter Up, Then Hard Pullback",
        "high_vol_sideways_then_breakout": "High-Vol Sideways, Then Breakout",
        "earnings_gap_up_then_fade": "Earnings Gap Up, Then Fade",
        "earnings_gap_down_then_recovery": "Earnings Gap Down, Then Recovery",
        "false_breakout_then_recover": "False Breakout, Then Recover",
        "rally_then_long_range_then_second_leg_up": "Rally, Range, Second Leg Up",
        "violent_two_sided_quarter": "Violent Two-Sided Quarter",
        "slow_bleed_then_capitulation_then_bounce": "Slow Bleed, Capitulation, Bounce",
        "early_breakout_to_target": "Early Breakout To Target",
        "slow_grind_to_target": "Slow Grind To Target",
        "down_then_recover_to_target": "Down Then Recover To Target",
        "rally_retrace_finish_target": "Rally, Retrace, Finish At Target",
        "late_breakout_to_target": "Late Breakout To Target",
        "overshoot_then_settle_at_target": "Overshoot, Then Settle At Target",
        "fast_overshoot_then_sideways": "Fast Overshoot, Then Sideways",
        "weak_start_then_acceleration": "Weak Start, Then Acceleration",
        "two_stage_bull_run": "Two-Stage Bull Run",
        "violent_path_to_target": "Violent Path To Target",
        "flat": "Flat",
        "mean_reversion_lower": "Mean Reversion Lower",
        "mean_reversion_higher": "Mean Reversion Higher",
        "iv_up_then_down": "IV Up, Then Down",
        "iv_down_then_stays_low": "IV Down, Then Stays Low",
        "earnings_build_then_crush": "Earnings Build, Then Crush",
        "active_assumed_path": "Active Assumed Path",
        "active_assumed_iv": "Active Assumed IV",
    }
    return labels.get(text, text.replace("_", " ").title())


def stock_path_gallery_named_points(
    *,
    preset: str,
    entry_spot: float,
    target_price: float,
    target_horizon_label: str,
) -> dict[str, float]:
    """Return richer named scenario-gallery anchors, distinct from the active thesis path presets."""

    preset_name = clean_string(preset).lower()
    target_label = clean_string(target_horizon_label).lower() or "target"
    base_move = max(abs(float(target_price) - float(entry_spot)), max(float(entry_spot) * 0.18, 2.75))
    up_small = float(entry_spot) + min(max(base_move * 0.45, 1.4), 2.2)
    up_medium = float(entry_spot) + min(max(base_move * 0.72, 2.6), 3.6)
    up_large = float(entry_spot) + min(max(base_move * 0.95, 3.6), 4.9)
    down_small = max(float(entry_spot) - min(max(base_move * 0.42, 1.4), 2.2), 0.5)
    down_medium = max(float(entry_spot) - min(max(base_move * 0.72, 2.8), 4.0), 0.5)
    down_large = max(float(entry_spot) - min(max(base_move * 0.98, 3.9), 5.2), 0.5)
    near_flat = float(entry_spot) + min(max(base_move * 0.08, 0.08), 0.35)
    mild_up = float(entry_spot) + min(max(base_move * 0.24, 0.55), 1.25)
    moderate_up = float(entry_spot) + min(max(base_move * 0.36, 1.0), 1.85)
    targetish = float(entry_spot) + min(max(base_move * 0.82, 2.9), 4.4)

    presets: dict[str, dict[str, float]] = {
        "rally_early_then_fade_then_rally_again": {
            "entry": float(entry_spot),
            "2w": up_large,
            "1m": up_medium,
            "2m": up_small,
            target_label: targetish,
        },
        "range_bound_near_flat": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) + 0.22,
            "1m": float(entry_spot) - 0.10,
            "2m": float(entry_spot) + 0.16,
            target_label: near_flat,
        },
        "down_first_then_recovery": {
            "entry": float(entry_spot),
            "2w": down_medium,
            "1m": down_small,
            "2m": mild_up,
            target_label: moderate_up,
        },
        "late_breakout": {
            "entry": float(entry_spot),
            "2w": float(entry_spot),
            "1m": float(entry_spot) + 0.25,
            "2m": mild_up,
            target_label: float(target_price),
        },
        "early_move_above_strike_then_giveback": {
            "entry": float(entry_spot),
            "2w": up_large,
            "1m": up_medium,
            "2m": moderate_up,
            target_label: mild_up,
        },
        "reaches_target_late_near_expiry": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) + 0.12,
            "1m": float(entry_spot) + 0.38,
            "2m": moderate_up,
            target_label: float(target_price),
        },
        "plus_20_pct_in_1m": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) * 1.10,
            "1m": float(entry_spot) * 1.20,
            "2m": float(entry_spot) * 1.18,
            target_label: float(target_price),
        },
        "plus_30_pct_in_1m": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) * 1.16,
            "1m": float(entry_spot) * 1.30,
            "2m": float(entry_spot) * 1.27,
            target_label: float(target_price),
        },
        "plus_20_pct_in_1q": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) * 1.04,
            "1m": float(entry_spot) * 1.09,
            "2m": float(entry_spot) * 1.15,
            target_label: float(target_price),
        },
        "plus_30_pct_in_1q": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) * 1.06,
            "1m": float(entry_spot) * 1.12,
            "2m": float(entry_spot) * 1.20,
            target_label: float(target_price),
        },
        "quarter_up_then_pullback": {
            "entry": float(entry_spot),
            "2w": up_large,
            "1m": float(entry_spot) + min(max(base_move * 1.05, 4.0), 5.2),
            "2m": moderate_up,
            target_label: mild_up,
        },
        "quarter_down_then_next_quarter_recovery": {
            "entry": float(entry_spot),
            "2w": down_large,
            "1m": down_medium,
            "2m": down_small,
            target_label: mild_up,
        },
        "two_quarters_down_then_flat_then_recovery": {
            "entry": float(entry_spot),
            "2w": down_large,
            "1m": down_large,
            "2m": down_medium,
            target_label: float(entry_spot) - 0.18,
        },
        "high_swing_quarterly_path": {
            "entry": float(entry_spot),
            "2w": up_large,
            "1m": down_medium,
            "2m": up_medium,
            target_label: float(entry_spot) + min(max(base_move * 0.55, 1.8), 2.9),
        },
        "slow_grind_up": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) + 0.42,
            "1m": mild_up,
            "2m": moderate_up,
            target_label: targetish,
        },
        "overshoot_then_mean_revert": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) + min(max(base_move * 1.05, 4.0), 5.2),
            "1m": float(entry_spot) + min(max(base_move * 0.88, 3.2), 4.4),
            "2m": up_small,
            target_label: float(entry_spot) + min(max(base_move * 0.30, 0.9), 1.7),
        },
        "quarter_up_then_hard_pullback": {
            "entry": float(entry_spot),
            "2w": float(entry_spot) + min(max(base_move * 1.10, 4.1), 5.4),
            "1m": float(entry_spot) + min(max(base_move * 0.92, 3.4), 4.7),
            "2m": down_small,
            target_label: near_flat,
        },
        "high_vol_sideways_then_breakout": {
            "entry": float(entry_spot),
            "2w": up_medium,
            "1m": down_small,
            "2m": float(entry_spot) + min(max(base_move * 0.16, 0.45), 0.95),
            target_label: float(entry_spot) + min(max(base_move * 0.88, 3.2), 4.7),
        },
        "earnings_gap_up_then_fade": {
            "entry": float(entry_spot),
            "1w": float(entry_spot) + min(max(base_move * 1.05, 4.0), 5.3),
            "2w": float(entry_spot) + min(max(base_move * 0.95, 3.5), 4.8),
            "1m": up_medium,
            "2m": mild_up,
            target_label: near_flat,
        },
        "earnings_gap_down_then_recovery": {
            "entry": float(entry_spot),
            "1w": down_large,
            "2w": down_medium,
            "1m": down_small,
            "2m": mild_up,
            target_label: targetish,
        },
        "false_breakout_then_recover": {
            "entry": float(entry_spot),
            "2w": up_medium,
            "1m": down_medium,
            "2m": float(entry_spot) + min(max(base_move * 0.18, 0.45), 1.0),
            target_label: up_small,
        },
        "rally_then_long_range_then_second_leg_up": {
            "entry": float(entry_spot),
            "2w": up_medium,
            "1m": up_small,
            "2m": up_small,
            target_label: targetish,
        },
        "violent_two_sided_quarter": {
            "entry": float(entry_spot),
            "2w": up_large,
            "1m": down_large,
            "2m": up_medium,
            target_label: moderate_up,
        },
        "slow_bleed_then_capitulation_then_bounce": {
            "entry": float(entry_spot),
            "2w": down_small,
            "1m": down_medium,
            "2m": down_large,
            target_label: down_small,
        },
    }
    return dict(presets.get(preset_name, presets["late_breakout"]))


def build_path_grid(snapshot_date: date, end_date: date) -> list[dict[str, Any]]:
    """Return an entry-inclusive business-day grid used by path simulations."""

    start = pd.Timestamp(snapshot_date)
    end = pd.Timestamp(end_date)
    if end < start:
        end = start
    business_days = list(pd.bdate_range(start=start + pd.offsets.BDay(1), end=end))
    dates = [start] + business_days
    return [
        {
            "date": ts.date().isoformat(),
            "requested_days": int((ts - start).days),
            "step_index": index,
            "time_fraction": (index / (len(dates) - 1)) if len(dates) > 1 else 0.0,
        }
        for index, ts in enumerate(dates)
    ]


def _grid_anchor_fraction(grid: list[dict[str, Any]], requested_days: int) -> float:
    if not grid:
        return 0.0
    terminal_days = max(int(grid[-1]["requested_days"]), 1)
    return max(0.0, min(float(requested_days) / float(terminal_days), 1.0))


def build_stock_path_from_named_points(
    grid: list[dict[str, Any]],
    *,
    named_points: dict[str, float],
    path_id: str,
    path_name: str,
    entry_spot: float,
) -> StockPathExample:
    """Build one stock path by interpolating named checkpoint inputs onto the daily grid."""

    anchors: list[tuple[float, float]] = [(0.0, float(entry_spot))]
    for label, value in named_points.items():
        lowered = clean_string(label).lower()
        if lowered == "entry":
            continue
        requested_days = 0
        if lowered.endswith("w"):
            try:
                requested_days = int(round(float(lowered[:-1]) * 7))
            except ValueError:
                requested_days = 0
        elif lowered.endswith("m"):
            try:
                requested_days = int(round(float(lowered[:-1]) * 30))
            except ValueError:
                requested_days = 0
        elif lowered.endswith("d"):
            try:
                requested_days = int(round(float(lowered[:-1])))
            except ValueError:
                requested_days = 0
        else:
            try:
                requested_days = int(round(float(lowered)))
            except ValueError:
                requested_days = 0
        anchors.append((_grid_anchor_fraction(grid, requested_days), float(value)))
    anchors = sorted(anchors, key=lambda item: item[0])
    values = _piecewise_values(grid, anchors, easing="cosine")
    points = [
        {
            **point,
            "spot_price": round(float(value), 4),
            "return_pct": round(float(value / float(entry_spot) - 1.0), 6) if entry_spot else None,
        }
        for point, value in zip(grid, values)
    ]
    return StockPathExample(
        path_id=clean_string(path_id),
        path_kind="assumed",
        path_name=clean_string(path_name) or "assumed_path",
        path_points=points,
    )


def build_iv_path_from_named_points(
    grid: list[dict[str, Any]],
    *,
    named_points: dict[str, float],
    iv_path_id: str,
    iv_path_name: str,
    base_iv_shift: float,
) -> IVPathExample:
    """Build one IV path by interpolating named checkpoint inputs onto the daily grid."""

    anchors: list[tuple[float, float]] = [(0.0, float(base_iv_shift))]
    for label, value in named_points.items():
        lowered = clean_string(label).lower()
        if lowered == "entry":
            continue
        requested_days = 0
        if lowered.endswith("w"):
            try:
                requested_days = int(round(float(lowered[:-1]) * 7))
            except ValueError:
                requested_days = 0
        elif lowered.endswith("m"):
            try:
                requested_days = int(round(float(lowered[:-1]) * 30))
            except ValueError:
                requested_days = 0
        elif lowered.endswith("d"):
            try:
                requested_days = int(round(float(lowered[:-1])))
            except ValueError:
                requested_days = 0
        else:
            try:
                requested_days = int(round(float(lowered)))
            except ValueError:
                requested_days = 0
        anchors.append((_grid_anchor_fraction(grid, requested_days), float(value)))
    anchors = sorted(anchors, key=lambda item: item[0])
    values = _piecewise_values(grid, anchors)
    points = [
        {
            **point,
            "iv_shift_points": round(float(value), 6),
        }
        for point, value in zip(grid, values)
    ]
    return IVPathExample(
        iv_path_id=clean_string(iv_path_id),
        iv_path_name=clean_string(iv_path_name) or "assumed_iv_path",
        path_points=points,
    )


def build_stock_path_gallery_rows(
    grid: list[dict[str, Any]],
    *,
    entry_spot: float,
    target_price: float,
    target_horizon_label: str,
    active_path_name: str,
    active_named_points: dict[str, float],
) -> pd.DataFrame:
    """Build the named stock-path gallery used as the primary scenario-thinking surface."""

    records: list[dict[str, Any]] = []
    active_name = clean_string(active_path_name).lower()
    active_in_gallery = active_name in STOCK_PATH_GALLERY_PRESETS
    for display_order, preset_name in enumerate(STOCK_PATH_GALLERY_PRESETS, start=1):
        metadata = stock_path_family_metadata(preset_name)
        example = build_stock_path_from_named_points(
            grid,
            named_points=stock_path_gallery_named_points(
                preset=preset_name,
                entry_spot=float(entry_spot),
                target_price=float(target_price),
                target_horizon_label=target_horizon_label,
            ),
            path_id=f"gallery-stock-{preset_name}",
            path_name=preset_name,
            entry_spot=float(entry_spot),
        )
        for point in example.path_points:
            records.append(
                {
                    "path_name": preset_name,
                    "path_label": humanize_named_path(preset_name, kind="stock"),
                    "path_family": metadata["path_family"],
                    "path_family_label": metadata["path_family_label"],
                    "timing_shape": metadata["timing_shape"],
                    "outcome_bias": metadata["outcome_bias"],
                    "path_description": metadata["path_description"],
                    "path_role": "gallery_named_path",
                    "display_order": display_order,
                    "date": point.get("date"),
                    "requested_days": point.get("requested_days"),
                    "spot_price": point.get("spot_price"),
                    "return_pct": point.get("return_pct"),
                    "is_active_assumed": preset_name == active_name,
                }
            )
    if not active_in_gallery:
        metadata = stock_path_family_metadata(active_name or "active_assumed_path")
        active_example = build_stock_path_from_named_points(
            grid,
            named_points=active_named_points,
            path_id="gallery-stock-active-assumed",
            path_name=active_name or "active_assumed_path",
            entry_spot=float(entry_spot),
        )
        for point in active_example.path_points:
            records.append(
                {
                    "path_name": active_name or "active_assumed_path",
                    "path_label": humanize_named_path("active_assumed_path", kind="stock"),
                    "path_family": metadata["path_family"],
                    "path_family_label": metadata["path_family_label"],
                    "timing_shape": metadata["timing_shape"],
                    "outcome_bias": metadata["outcome_bias"],
                    "path_description": metadata["path_description"],
                    "path_role": "active_assumed_path",
                    "display_order": len(STOCK_PATH_GALLERY_PRESETS) + 1,
                    "date": point.get("date"),
                    "requested_days": point.get("requested_days"),
                    "spot_price": point.get("spot_price"),
                    "return_pct": point.get("return_pct"),
                    "is_active_assumed": True,
                }
            )
    return pd.DataFrame(records)


def build_iv_path_gallery_rows(
    grid: list[dict[str, Any]],
    *,
    target_horizon_label: str,
    active_iv_path_name: str,
    active_named_points: dict[str, float],
) -> pd.DataFrame:
    """Build the named IV-regime gallery using normalized IV-shift paths."""

    records: list[dict[str, Any]] = []
    active_name = clean_string(active_iv_path_name).lower()
    active_in_gallery = active_name in IV_PATH_GALLERY_PRESETS
    for display_order, preset_name in enumerate(IV_PATH_GALLERY_PRESETS, start=1):
        example = build_iv_path_from_named_points(
            grid,
            named_points=_default_iv_path_points(
                preset=preset_name,
                base_shift=0.0,
                target_horizon_label=target_horizon_label,
            ),
            iv_path_id=f"gallery-iv-{preset_name}",
            iv_path_name=preset_name,
            base_iv_shift=0.0,
        )
        for point in example.path_points:
            records.append(
                {
                    "iv_path_name": preset_name,
                    "iv_path_label": humanize_named_path(preset_name, kind="iv"),
                    "path_role": "gallery_named_path",
                    "display_order": display_order,
                    "date": point.get("date"),
                    "requested_days": point.get("requested_days"),
                    "iv_shift_points": point.get("iv_shift_points"),
                    "is_active_assumed": preset_name == active_name,
                }
            )
    if not active_in_gallery:
        active_example = build_iv_path_from_named_points(
            grid,
            named_points=active_named_points,
            iv_path_id="gallery-iv-active-assumed",
            iv_path_name=active_name or "active_assumed_iv",
            base_iv_shift=float(active_named_points.get("entry", 0.0)),
        )
        for point in active_example.path_points:
            records.append(
                {
                    "iv_path_name": active_name or "active_assumed_iv",
                    "iv_path_label": humanize_named_path("active_assumed_iv", kind="iv"),
                    "path_role": "active_assumed_iv",
                    "display_order": len(IV_PATH_GALLERY_PRESETS) + 1,
                    "date": point.get("date"),
                    "requested_days": point.get("requested_days"),
                    "iv_shift_points": point.get("iv_shift_points"),
                    "is_active_assumed": True,
                }
            )
    return pd.DataFrame(records)


def _piecewise_values(
    grid: list[dict[str, Any]],
    anchors: list[tuple[float, float]],
    *,
    easing: str = "linear",
) -> np.ndarray:
    if not grid:
        return np.asarray([], dtype=float)
    x = np.asarray([float(point[0]) for point in anchors], dtype=float)
    y = np.asarray([float(point[1]) for point in anchors], dtype=float)
    xp = np.asarray([float(point["time_fraction"]) for point in grid], dtype=float)
    if clean_string(easing).lower() != "cosine" or len(x) < 2:
        return np.interp(xp, x, y)
    values = np.empty_like(xp, dtype=float)
    for index, value in enumerate(xp):
        if value <= x[0]:
            values[index] = y[0]
            continue
        if value >= x[-1]:
            values[index] = y[-1]
            continue
        upper = int(np.searchsorted(x, value, side="right"))
        lower = max(0, upper - 1)
        x0 = float(x[lower])
        x1 = float(x[upper])
        y0 = float(y[lower])
        y1 = float(y[upper])
        if x1 <= x0:
            values[index] = y1
            continue
        t = max(0.0, min((float(value) - x0) / (x1 - x0), 1.0))
        eased = 0.5 - 0.5 * np.cos(np.pi * t)
        values[index] = y0 + (y1 - y0) * eased
    return values


def _deterministic_stock_values(
    grid: list[dict[str, Any]],
    *,
    entry_spot: float,
    preset: str,
    target_end: float,
) -> np.ndarray:
    target_label = f"{int(grid[-1]['requested_days'])}d" if grid else "target"
    named_points = _stock_path_preset_points(
        preset=clean_string(preset) or "slow_bull",
        entry_spot=float(entry_spot),
        target_price=float(target_end),
        target_horizon_label=target_label,
    )
    anchors: list[tuple[float, float]] = []
    for label, value in named_points.items():
        lowered = clean_string(label).lower()
        if lowered == "entry":
            requested_days = 0
        elif lowered.endswith("w"):
            requested_days = int(round(float(lowered[:-1]) * 7))
        elif lowered.endswith("m"):
            requested_days = int(round(float(lowered[:-1]) * 30))
        elif lowered.endswith("d"):
            requested_days = int(round(float(lowered[:-1])))
        else:
            requested_days = int(round(float(grid[-1]["requested_days"]))) if grid else 0
        anchors.append((_grid_anchor_fraction(grid, requested_days), float(value)))
    anchors = sorted(anchors, key=lambda item: item[0])
    return _piecewise_values(grid, anchors, easing="cosine")


def _gbm_values(
    grid: list[dict[str, Any]],
    *,
    entry_spot: float,
    annualized_vol: float,
    drift: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not grid:
        return np.asarray([], dtype=float)
    steps = len(grid)
    values = np.empty(steps, dtype=float)
    values[0] = float(entry_spot)
    if steps == 1:
        return values
    dt = 1.0 / BUSINESS_DAYS_PER_YEAR
    shocks = rng.normal(loc=0.0, scale=sqrt(dt), size=steps - 1)
    for index in range(1, steps):
        log_return = (float(drift) - 0.5 * float(annualized_vol) ** 2) * dt + float(annualized_vol) * shocks[index - 1]
        values[index] = max(0.01, values[index - 1] * exp(log_return))
    return values


def _conditioned_values(
    grid: list[dict[str, Any]],
    *,
    entry_spot: float,
    target_end: float,
    annualized_vol: float,
    rng: np.random.Generator,
    cross_level: float | None = None,
    cross_behavior: str | None = None,
) -> np.ndarray:
    if not grid:
        return np.asarray([], dtype=float)
    if len(grid) == 1:
        return np.asarray([float(entry_spot)], dtype=float)
    times = np.asarray([float(point["time_fraction"]) for point in grid], dtype=float)
    total_steps = len(times) - 1
    dt = 1.0 / BUSINESS_DAYS_PER_YEAR
    increments = rng.normal(loc=0.0, scale=sqrt(dt), size=total_steps)
    brownian = np.concatenate([[0.0], np.cumsum(increments)])
    bridge = brownian - times * brownian[-1]
    start_log = log(max(float(entry_spot), 0.01))
    end_log = log(max(float(target_end), 0.01))
    values = np.exp(start_log + times * (end_log - start_log) + float(annualized_vol) * bridge)
    if cross_level is not None and clean_string(cross_behavior).lower() == "cross_early_then_revert":
        desired = log(max(float(cross_level), 0.01))
        early_index = min(max(int(round(total_steps * 0.35)), 1), total_steps - 1)
        gap = desired - log(max(values[early_index], 0.01))
        bump = np.exp(-((times - times[early_index]) ** 2) / max(2.0 * 0.08**2, 1e-6))
        values = np.exp(np.log(np.maximum(values, 0.01)) + gap * bump)
        end_adjustment = log(max(float(target_end), 0.01)) - log(max(values[-1], 0.01))
        values = np.exp(np.log(np.maximum(values, 0.01)) + times * end_adjustment)
    values[0] = float(entry_spot)
    values[-1] = float(target_end)
    return np.maximum(values, 0.01)


def build_stock_path_example(
    grid: list[dict[str, Any]],
    *,
    entry_spot: float,
    mode: str,
    target_end: float,
    annualized_vol: float = 0.55,
    drift: float = 0.18,
    preset: str | None = None,
    rng: np.random.Generator | None = None,
    path_id: str | None = None,
    cross_level: float | None = None,
    cross_behavior: str | None = None,
) -> StockPathExample:
    """Build one deterministic, GBM, or conditioned stock path example."""

    generator = rng or np.random.default_rng(0)
    mode_name = clean_string(mode).lower() or "deterministic"
    if mode_name == "deterministic":
        values = _deterministic_stock_values(
            grid,
            entry_spot=float(entry_spot),
            preset=clean_string(preset) or "slow_bull",
            target_end=float(target_end),
        )
        path_name = clean_string(preset) or "slow_bull"
    elif mode_name == "simulated":
        values = _gbm_values(
            grid,
            entry_spot=float(entry_spot),
            annualized_vol=float(annualized_vol),
            drift=float(drift),
            rng=generator,
        )
        path_name = "gbm"
    elif mode_name == "conditioned":
        values = _conditioned_values(
            grid,
            entry_spot=float(entry_spot),
            target_end=float(target_end),
            annualized_vol=float(annualized_vol),
            rng=generator,
            cross_level=cross_level,
            cross_behavior=cross_behavior,
        )
        path_name = "conditioned_bridge"
    else:
        raise ValueError(f"Unsupported stock path mode: {mode}")
    points = [
        {
            **point,
            "spot_price": round(float(value), 4),
            "return_pct": round(float(value / float(entry_spot) - 1.0), 6) if entry_spot else None,
        }
        for point, value in zip(grid, values)
    ]
    return StockPathExample(
        path_id=clean_string(path_id) or f"{mode_name}-{clean_string(path_name) or 'path'}",
        path_kind=mode_name,
        path_name=clean_string(path_name) or mode_name,
        path_points=points,
    )


def build_stock_path_pool(
    grid: list[dict[str, Any]],
    *,
    entry_spot: float,
    target_end: float,
    mode: str = "mixed",
    simulated_path_count: int = 24,
    annualized_vol: float = 0.55,
    drift: float = 0.18,
    rng: np.random.Generator | None = None,
) -> list[StockPathExample]:
    """Build a moderate pool of deterministic, simulated, and conditioned stock paths."""

    generator = rng or np.random.default_rng(0)
    mode_name = clean_string(mode).lower() or "mixed"
    paths: list[StockPathExample] = []
    if mode_name in {"deterministic", "mixed"}:
        for preset in [
            "flat",
            "slow_bull",
            "fast_bull",
            "down_then_recover",
            "gap_up_then_drift",
            "gap_down_then_recover",
            "rally_early_then_fade_then_rally_again",
            "range_bound_near_flat",
            "down_first_then_recovery",
            "late_breakout",
            "early_move_above_strike_then_giveback",
            "reaches_target_late_near_expiry",
        ]:
            paths.append(
                build_stock_path_example(
                    grid,
                    entry_spot=float(entry_spot),
                    mode="deterministic",
                    preset=preset,
                    target_end=float(target_end),
                    rng=generator,
                    path_id=f"deterministic-{preset}",
                )
            )
    if mode_name in {"simulated", "mixed"}:
        for index in range(max(int(simulated_path_count), 1)):
            paths.append(
                build_stock_path_example(
                    grid,
                    entry_spot=float(entry_spot),
                    mode="simulated",
                    target_end=float(target_end),
                    annualized_vol=float(annualized_vol),
                    drift=float(drift),
                    rng=generator,
                    path_id=f"simulated-gbm-{index + 1:02d}",
                )
            )
    if mode_name in {"conditioned", "mixed"}:
        conditioned_targets = [
            max(float(entry_spot) * 0.85, 0.01),
            float(target_end) * 0.98,
            float(target_end),
            float(target_end) * 1.10,
        ]
        for index, conditioned_target in enumerate(conditioned_targets, start=1):
            paths.append(
                build_stock_path_example(
                    grid,
                    entry_spot=float(entry_spot),
                    mode="conditioned",
                    target_end=float(conditioned_target),
                    annualized_vol=float(annualized_vol),
                    rng=generator,
                    path_id=f"conditioned-{index:02d}",
                    cross_level=float((float(entry_spot) + float(target_end)) / 2.0),
                    cross_behavior="cross_early_then_revert" if index % 2 == 0 else None,
                )
            )
    return paths


def _iv_profile_values(grid: list[dict[str, Any]], *, base_iv_shift: float, mode: str) -> np.ndarray:
    mode_name = clean_string(mode).lower() or "flat"
    anchors = {
        "flat": [(0.0, base_iv_shift), (1.0, base_iv_shift)],
        "mean_reversion_lower": [(0.0, base_iv_shift), (0.4, base_iv_shift - 0.06), (1.0, base_iv_shift - 0.10)],
        "mean_reversion_higher": [(0.0, base_iv_shift), (0.4, base_iv_shift + 0.06), (1.0, base_iv_shift + 0.10)],
        "iv_up_then_down": [(0.0, base_iv_shift), (0.25, base_iv_shift + 0.10), (0.6, base_iv_shift + 0.04), (1.0, base_iv_shift)],
        "iv_down_then_stays_low": [(0.0, base_iv_shift), (0.25, base_iv_shift - 0.10), (1.0, base_iv_shift - 0.10)],
        "earnings_build_then_crush": [(0.0, base_iv_shift), (0.22, base_iv_shift + 0.12), (0.45, base_iv_shift - 0.10), (1.0, base_iv_shift - 0.06)],
    }
    return _piecewise_values(grid, anchors.get(mode_name, anchors["flat"]))


def build_iv_path_example(
    grid: list[dict[str, Any]],
    *,
    base_iv_shift: float,
    mode: str,
    rng: np.random.Generator | None = None,
    iv_path_id: str | None = None,
    noisy: bool = False,
    stock_path_points: list[dict[str, Any]] | None = None,
) -> IVPathExample:
    """Build one IV path example, optionally with light stock-aware bias."""

    generator = rng or np.random.default_rng(0)
    mode_name = clean_string(mode).lower() or "flat"
    values = _iv_profile_values(grid, base_iv_shift=float(base_iv_shift), mode=mode_name)
    if noisy and len(values) > 1:
        values = values + generator.normal(loc=0.0, scale=0.01, size=len(values))
        values[0] = float(base_iv_shift)
    if stock_path_points and len(stock_path_points) == len(values):
        stock_returns = np.asarray([float(point.get("return_pct") or 0.0) for point in stock_path_points], dtype=float)
        stock_changes = np.diff(stock_returns, prepend=stock_returns[0])
        values = values + np.where(stock_changes < -0.03, 0.015, 0.0)
        values = values + np.where(stock_changes > 0.02, -0.008, 0.0)
    points = [
        {
            **point,
            "iv_shift_points": round(float(value), 6),
        }
        for point, value in zip(grid, values)
    ]
    return IVPathExample(
        iv_path_id=clean_string(iv_path_id) or f"iv-{mode_name}",
        iv_path_name=mode_name,
        path_points=points,
    )


def pair_stock_and_iv_paths(
    stock_paths: list[StockPathExample],
    iv_paths: list[IVPathExample],
) -> list[PathPairExample]:
    """Build explicit stock-path / IV-path combinations."""

    pairs: list[PathPairExample] = []
    for stock_path in stock_paths:
        for iv_path in iv_paths:
            pairs.append(
                PathPairExample(
                    path_pair_id=f"{stock_path.path_id}__{iv_path.iv_path_id}",
                    stock_path_id=stock_path.path_id,
                    iv_path_id=iv_path.iv_path_id,
                    stock_path_name=stock_path.path_name,
                    iv_path_name=iv_path.iv_path_name,
                    stock_path_kind=stock_path.path_kind,
                    iv_path_kind="iv_path",
                    stock_points=stock_path.path_points,
                    iv_points=iv_path.path_points,
                )
            )
    return pairs


def _representative_bucket(outcome: dict[str, Any]) -> str:
    profit = float(outcome.get("final_profit_loss") or 0.0)
    goal_reached = bool(outcome.get("goal_reached"))
    outperformed_stock = bool(outcome.get("outperformed_stock"))
    crossed_key_level = bool(outcome.get("crossed_key_level"))
    if not goal_reached and profit <= -100:
        return "misses_badly"
    if not goal_reached:
        return "almost_works" if crossed_key_level else "misses_badly"
    if goal_reached and not outperformed_stock:
        return "just_works"
    if goal_reached and profit < 200:
        return "works_well"
    return "works_very_well"


def select_representative_path_pairs(
    path_pairs: list[PathPairExample],
    *,
    path_outcomes: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select one stable representative path pair per goal-aware outcome bucket."""

    grouped: dict[str, list[tuple[PathPairExample, dict[str, Any]]]] = {}
    for pair in path_pairs:
        outcome = path_outcomes.get(pair.path_pair_id) or {}
        bucket = _representative_bucket(outcome)
        grouped.setdefault(bucket, []).append((pair, outcome))
    selected: list[dict[str, Any]] = []
    for bucket in ["misses_badly", "almost_works", "just_works", "works_well", "works_very_well"]:
        candidates = grouped.get(bucket) or []
        if not candidates:
            continue
        chosen_pair, chosen_outcome = sorted(
            candidates,
            key=lambda item: (
                abs(float(item[1].get("final_profit_loss") or 0.0)),
                item[0].path_pair_id,
            ),
        )[0]
        selected.append(
            {
                "path_pair_id": chosen_pair.path_pair_id,
                "stock_path_id": chosen_pair.stock_path_id,
                "iv_path_id": chosen_pair.iv_path_id,
                "stock_path_name": chosen_pair.stock_path_name,
                "iv_path_name": chosen_pair.iv_path_name,
                "representative_bucket": bucket,
                "selection_reason": (
                    "Selected as the clearest example for this outcome bucket under the active goal."
                ),
                "final_profit_loss": float(chosen_outcome.get("final_profit_loss") or 0.0),
            }
        )
    return selected
