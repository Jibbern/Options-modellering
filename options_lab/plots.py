from __future__ import annotations

import math
from pathlib import Path
import re
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from .scenarios import compare_positions
from .strategies import StrategyPosition
from .utils import clean_string, ensure_directory, finite_or_none, horizon_to_days, windows_extended_path


STRATEGY_VISUAL_SPECS = {
    "long_stock": {"color": "#000000", "marker": "s", "linestyle": "-", "label": "Long Stock"},
    "long_call": {"color": "#E69F00", "marker": "o", "linestyle": "-", "label": "Long Call"},
    "bull_call_spread": {"color": "#56B4E9", "marker": "^", "linestyle": "--", "label": "Bull Call Spread"},
    "long_put": {"color": "#009E73", "marker": "D", "linestyle": "-", "label": "Long Put"},
    "bear_put_spread": {"color": "#0072B2", "marker": "v", "linestyle": "--", "label": "Bear Put Spread"},
    "covered_call": {"color": "#D55E00", "marker": "P", "linestyle": "-.", "label": "Covered Call"},
    "cash_secured_put": {"color": "#CC79A7", "marker": "X", "linestyle": ":", "label": "Cash Secured Put"},
}
STRATEGY_FAMILY_ORDER = list(STRATEGY_VISUAL_SPECS)
SUPPORTING_VISUAL_SPECS = {
    "assumed_path": {"color": "#1A1A1A", "marker": "o", "linestyle": "-", "linewidth": 3.8},
    "stock_baseline": {"color": "#1A1A1A", "marker": None, "linestyle": "--", "linewidth": 2.2},
    "top_candidate": {"color": "#3A3A3A", "marker": "s", "linestyle": (0, (4, 2)), "linewidth": 3.2},
    "comparison_zero": {"color": "#5C5C5C", "marker": None, "linestyle": "--", "linewidth": 1.2},
}
IV_PATH_VISUAL_SPECS = {
    "active_assumption": {"color": "#1A1A1A", "marker": "o", "linestyle": "-", "linewidth": 3.4},
    "flat": {"color": "#6B7280", "marker": "s", "linestyle": "--", "linewidth": 2.2},
    "iv_down_then_stays_low": {"color": "#0072B2", "marker": "v", "linestyle": "-", "linewidth": 2.2},
    "mean_reversion_lower": {"color": "#009E73", "marker": "D", "linestyle": ":", "linewidth": 2.2},
    "earnings_build_then_crush": {"color": "#D55E00", "marker": "P", "linestyle": "-.", "linewidth": 2.2},
    "iv_up_then_down": {"color": "#E69F00", "marker": "^", "linestyle": "-", "linewidth": 2.2},
    "mean_reversion_higher": {"color": "#CC79A7", "marker": "X", "linestyle": ":", "linewidth": 2.2},
}
STOCK_PATH_GALLERY_SPECS = {
    "rally_early_then_fade_then_rally_again": {"color": "#D55E00", "marker": "o", "linestyle": "-", "linewidth": 2.6},
    "range_bound_near_flat": {"color": "#6B7280", "marker": "s", "linestyle": "--", "linewidth": 2.4},
    "down_first_then_recovery": {"color": "#0072B2", "marker": "D", "linestyle": "-", "linewidth": 2.6},
    "late_breakout": {"color": "#009E73", "marker": "^", "linestyle": "-", "linewidth": 2.6},
    "early_move_above_strike_then_giveback": {"color": "#E69F00", "marker": "P", "linestyle": "-.", "linewidth": 2.5},
    "reaches_target_late_near_expiry": {"color": "#CC79A7", "marker": "X", "linestyle": ":", "linewidth": 2.5},
    "plus_20_pct_in_1m": {"color": "#2563EB", "marker": "D", "linestyle": "-", "linewidth": 2.5},
    "plus_30_pct_in_1m": {"color": "#B91C1C", "marker": "v", "linestyle": "-", "linewidth": 2.6},
    "plus_20_pct_in_1q": {"color": "#0F766E", "marker": "h", "linestyle": "--", "linewidth": 2.4},
    "plus_30_pct_in_1q": {"color": "#A16207", "marker": ">", "linestyle": "--", "linewidth": 2.4},
    "quarter_up_then_pullback": {"color": "#8C564B", "marker": "h", "linestyle": "-", "linewidth": 2.4},
    "quarter_down_then_next_quarter_recovery": {"color": "#17BECF", "marker": "v", "linestyle": "--", "linewidth": 2.4},
    "two_quarters_down_then_flat_then_recovery": {"color": "#9467BD", "marker": "<", "linestyle": "-.", "linewidth": 2.4},
    "high_swing_quarterly_path": {"color": "#BCBD22", "marker": ">", "linestyle": "-", "linewidth": 2.4},
    "slow_grind_up": {"color": "#2A9D8F", "marker": "*", "linestyle": ":", "linewidth": 2.4},
    "overshoot_then_mean_revert": {"color": "#F28E2B", "marker": "H", "linestyle": "--", "linewidth": 2.4},
    "quarter_up_then_hard_pullback": {"color": "#B45309", "marker": "8", "linestyle": "-", "linewidth": 2.4},
    "high_vol_sideways_then_breakout": {"color": "#0E7490", "marker": "p", "linestyle": "--", "linewidth": 2.4},
    "earnings_gap_up_then_fade": {"color": "#C2410C", "marker": "P", "linestyle": (0, (5, 1.5)), "linewidth": 2.4},
    "earnings_gap_down_then_recovery": {"color": "#0369A1", "marker": "D", "linestyle": (0, (3, 1.5)), "linewidth": 2.4},
    "false_breakout_then_recover": {"color": "#7C3AED", "marker": "X", "linestyle": "--", "linewidth": 2.4},
    "rally_then_long_range_then_second_leg_up": {"color": "#15803D", "marker": "^", "linestyle": "-.", "linewidth": 2.4},
    "violent_two_sided_quarter": {"color": "#BE123C", "marker": "*", "linestyle": "-", "linewidth": 2.6},
    "slow_bleed_then_capitulation_then_bounce": {"color": "#4338CA", "marker": "v", "linestyle": ":", "linewidth": 2.4},
    "early_breakout_to_target": {"color": "#D55E00", "marker": "o", "linestyle": "-", "linewidth": 2.7},
    "slow_grind_to_target": {"color": "#009E73", "marker": "s", "linestyle": "-", "linewidth": 2.5},
    "down_then_recover_to_target": {"color": "#0072B2", "marker": "D", "linestyle": "--", "linewidth": 2.5},
    "rally_retrace_finish_target": {"color": "#E69F00", "marker": "^", "linestyle": "-.", "linewidth": 2.5},
    "late_breakout_to_target": {"color": "#CC79A7", "marker": "P", "linestyle": ":", "linewidth": 2.6},
    "overshoot_then_settle_at_target": {"color": "#56B4E9", "marker": "X", "linestyle": "-", "linewidth": 2.6},
    "fast_overshoot_then_sideways": {"color": "#BE123C", "marker": ">", "linestyle": "-", "linewidth": 2.6},
    "weak_start_then_acceleration": {"color": "#6B7280", "marker": "v", "linestyle": "--", "linewidth": 2.5},
    "two_stage_bull_run": {"color": "#0F766E", "marker": "h", "linestyle": "-.", "linewidth": 2.5},
    "violent_path_to_target": {"color": "#7C3AED", "marker": "*", "linestyle": "-", "linewidth": 2.6},
    "active_assumed_path": {"color": "#1A1A1A", "marker": "o", "linestyle": "-", "linewidth": 3.4},
}
LONG_CALL_COMPARISON_SPECS = [
    {"color": "#E69F00", "marker": "o", "linestyle": "-", "linewidth": 3.1},
    {"color": "#56B4E9", "marker": "s", "linestyle": "-", "linewidth": 2.6},
    {"color": "#009E73", "marker": "^", "linestyle": "-", "linewidth": 2.6},
    {"color": "#0072B2", "marker": "D", "linestyle": "-", "linewidth": 2.6},
    {"color": "#D55E00", "marker": "P", "linestyle": "-", "linewidth": 2.6},
    {"color": "#CC79A7", "marker": "X", "linestyle": "-", "linewidth": 2.6},
]
SINGLE_OPTION_OUTCOME_SPECS = {
    "clear_option_win": {"color": "#009E73", "marker": "o", "linestyle": "-", "label": "Clear option win"},
    "wins_but_not_enough": {"color": "#E69F00", "marker": "^", "linestyle": "--", "label": "Wins, not enough"},
    "stock_better": {"color": "#D55E00", "marker": "s", "linestyle": "-", "label": "Stock better"},
    "fail_too_narrow_or_expiry_issue": {"color": "#7F7F7F", "marker": "x", "linestyle": ":", "label": "Fail / too narrow"},
}
_FIGURE_FACE = "#FCFBF8"
_AXES_FACE = "#FFFDF9"
_GRID_COLOR = "#D7D4CE"
_SPINE_COLOR = "#C8C3BA"


def _human_label(value: str) -> str:
    return clean_string(value).replace("_", " ").title()


def _style_axes(ax, *, zero_line: bool = False) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(_FIGURE_FACE)
    ax.set_facecolor(_AXES_FACE)
    ax.grid(axis="y", color=_GRID_COLOR, alpha=0.55, linewidth=0.9)
    ax.grid(axis="x", color=_GRID_COLOR, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.tick_params(axis="both", labelsize=10.5)
    if zero_line:
        zero_style = SUPPORTING_VISUAL_SPECS["comparison_zero"]
        ax.axhline(
            0,
            color=str(zero_style["color"]),
            linewidth=float(zero_style["linewidth"]),
            linestyle=zero_style["linestyle"],
            zorder=0,
        )


def _format_money_axis(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"))


def _apply_title(ax, title: str, *, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12, y=1.06)
    if subtitle:
        ax.text(
            0.0,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#57534E",
        )


def _legend_columns(label_count: int, *, maximum: int = 4) -> int:
    if label_count <= 3:
        return label_count
    if label_count <= 6:
        return min(3, label_count)
    return min(maximum, math.ceil(label_count / 2))


def _canonical_sort_key(value: object) -> tuple[int, object]:
    text = clean_string(value)
    if not text:
        return (99, "")
    lowered = text.lower()
    if lowered == "entry":
        return (0, 0)
    if lowered in {"target", "target date"}:
        return (2, 10_000)
    if lowered.endswith("w"):
        try:
            return (1, int(round(float(lowered[:-1]) * 7)))
        except ValueError:
            pass
    try:
        return (1, horizon_to_days(text))
    except Exception:
        pass
    if lowered == "expiry":
        return (2, 100_000)
    iv_order = {
        "iv_down": 0,
        "iv_unchanged": 1,
        "iv_up": 2,
        "iv_down_then_stays_low": 3,
        "mean_reversion_lower": 4,
        "flat": 5,
        "earnings_build_then_crush": 6,
        "iv_up_then_down": 7,
        "mean_reversion_higher": 8,
    }
    if lowered in iv_order:
        return (3, iv_order[lowered])
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return (4, text)
    try:
        return (5, float(text))
    except ValueError:
        return (6, text)


def _sorted_unique(values: list | tuple | pd.Index | np.ndarray) -> list:
    unique = list(dict.fromkeys(list(values)))
    return sorted(unique, key=_canonical_sort_key)


def _ordered_strategy_groups(frame: pd.DataFrame, grouping: list[str]) -> list[tuple[tuple, pd.DataFrame]]:
    grouped = list(frame.groupby(grouping, dropna=False))
    return sorted(
        grouped,
        key=lambda item: (
            STRATEGY_FAMILY_ORDER.index(clean_string(item[0][-1])) if clean_string(item[0][-1]) in STRATEGY_FAMILY_ORDER else len(STRATEGY_FAMILY_ORDER),
            _human_label(clean_string(item[0][0])) if item[0] else "",
        ),
    )


def _dedupe_series_signature(x_values: pd.Series, y_values: pd.Series) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x = tuple(round(float(value), 4) for value in pd.to_numeric(x_values, errors="coerce").fillna(-999999.0).tolist())
    y = tuple(round(float(value), 4) for value in pd.to_numeric(y_values, errors="coerce").fillna(-999999.0).tolist())
    return x, y


def _apply_horizon_ticks(ax, tick_map: pd.DataFrame) -> None:
    if tick_map.empty:
        return
    ordered = tick_map.sort_values("requested_days")
    positions = ordered["requested_days"].tolist()
    labels = ordered["horizon"].tolist()
    crowded = any(abs(float(positions[idx]) - float(positions[idx - 1])) < 10 for idx in range(1, len(positions)))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=22 if crowded else 0, ha="right" if crowded else "center")


def _apply_date_ticks(ax, tick_map: pd.DataFrame, *, label_column: str = "date") -> None:
    if tick_map.empty:
        return
    ordered = tick_map.sort_values("requested_days")
    tick_count = min(5, len(ordered.index))
    indices = np.linspace(0, len(ordered.index) - 1, num=tick_count)
    sampled = ordered.iloc[sorted({int(round(value)) for value in indices})].copy()
    filtered_rows: list[dict[str, Any]] = []
    previous_position: float | None = None
    for row in sampled.to_dict("records"):
        position = float(row.get("requested_days", 0.0) or 0.0)
        if previous_position is not None and position - previous_position < 14:
            continue
        filtered_rows.append(row)
        previous_position = position
    if not filtered_rows:
        filtered_rows = sampled.to_dict("records")

    def _short_tick_label(value: Any) -> str:
        parsed = pd.to_datetime(clean_string(value), errors="coerce")
        if pd.isna(parsed):
            return clean_string(value)
        return parsed.strftime("%b %d")

    labels = [_short_tick_label(row.get(label_column)) for row in filtered_rows]
    positions = [row.get("requested_days") for row in filtered_rows]
    rotation = 16 if len(labels) > 4 else 0
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")


def _finalize(fig, output_path: str | Path) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.tight_layout()
    fig.savefig(windows_extended_path(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _empty_state_chart(
    *,
    output_path: str | Path,
    title: str,
    message: str,
    caption: str | None = None,
    figsize: tuple[float, float] = (11.0, 4.2),
) -> Path:
    """Write a readable chart placeholder for sparse ticker data.

    This keeps bundle generation robust for tickers with thin option chains
    without pretending that missing long-call comparisons are actionable data.
    """

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_FIGURE_FACE)
    ax.set_facecolor(_AXES_FACE)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=14)
    ax.text(
        0.05,
        0.56,
        clean_string(message),
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=12.0,
        color="#292524",
        wrap=True,
    )
    if clean_string(caption):
        ax.text(
            0.05,
            0.18,
            clean_string(caption),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.0,
            color="#57534E",
            wrap=True,
        )
    return _finalize(fig, output_path)


def _place_legend(ax, *, ncol: int = 2, fontsize: int | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = max(1, min(_legend_columns(len(labels), maximum=max(ncol, 4)), len(labels)))
    kwargs: dict[str, object] = {
        "loc": "upper left",
        "bbox_to_anchor": (0.0, -0.18),
        "ncol": ncol,
        "frameon": False,
        "columnspacing": 1.3,
        "handletextpad": 0.6,
        "borderaxespad": 0.0,
    }
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    ax.legend(**kwargs)


def _place_figure_legend(fig, ax, *, ncol: int = 3, fontsize: int = 8, y: float = 0.020) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    columns = max(1, min(_legend_columns(len(labels), maximum=max(ncol, 4)), len(labels)))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=columns,
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.3,
        handletextpad=0.6,
        borderaxespad=0.0,
    )


def _place_figure_caption(fig, caption: str, *, y: float = 0.105) -> None:
    text = clean_string(caption)
    if not text:
        return
    fig.text(
        0.085,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#57534E",
    )


def _finalize_stacked(fig, output_path: str | Path, *, bottom: float = 0.18) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.875, bottom=bottom, hspace=0.08)
    fig.savefig(windows_extended_path(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _create_path_stack(*, figsize: tuple[float, float] = (12.8, 7.3)):
    fig, (main_ax, stock_ax) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.0], "hspace": 0.08},
    )
    _style_axes(main_ax, zero_line=True)
    _style_axes(stock_ax, zero_line=False)
    main_ax.tick_params(axis="x", labelbottom=False)
    return fig, main_ax, stock_ax


def _date_tick_map(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "requested_days" not in frame.columns or "date" not in frame.columns:
        return pd.DataFrame()
    return frame[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")


def _stock_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "spot_price" not in frame.columns:
        return pd.DataFrame()
    working = frame.dropna(subset=["requested_days", "spot_price"]).copy()
    if working.empty:
        return working
    return (
        working[["requested_days", "date", "spot_price"]]
        .drop_duplicates()
        .sort_values("requested_days")
    )


def _plot_stock_context_panel(
    ax,
    frame: pd.DataFrame,
    *,
    strike_refs: list[float] | None = None,
    label: str = "Stock Path",
) -> None:
    stock_frame = _stock_context_frame(frame)
    if stock_frame.empty:
        ax.set_visible(False)
        return
    ax.plot(
        stock_frame["requested_days"],
        stock_frame["spot_price"],
        color="#57534E",
        linewidth=1.7,
        linestyle="-",
        marker="o",
        markersize=3.0,
        alpha=0.88,
        label=label,
        zorder=3,
    )
    for strike in sorted({float(value) for value in (strike_refs or []) if finite_or_none(value) is not None}):
        ax.axhline(
            strike,
            color="#A8A29E",
            linewidth=0.9,
            linestyle=":",
            alpha=0.45,
            zorder=1,
        )
    ax.set_ylabel("Stock ($)")
    _format_money_axis(ax)


def _annotate_path_milestones(ax, frame: pd.DataFrame, *, include_expiry: bool = False) -> None:
    if frame.empty or "requested_days" not in frame.columns:
        return
    x_values = pd.to_numeric(frame.get("requested_days"), errors="coerce").dropna()
    if x_values.empty:
        return
    max_day = int(x_values.max())
    ax.axvline(max_day, color="#78716C", linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.55, zorder=0)
    ax.text(
        max_day,
        0.98,
        "Target",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=7.6,
        color="#57534E",
    )
    if not include_expiry or "expiry_date" not in frame.columns or "date" not in frame.columns:
        return
    date_map = frame[["requested_days", "date"]].drop_duplicates().copy()
    if date_map.empty:
        return
    date_map["_date"] = pd.to_datetime(date_map["date"], errors="coerce")
    expiries = []
    for value in frame["expiry_date"].dropna().unique().tolist():
        parsed = pd.to_datetime(clean_string(value), errors="coerce")
        if not pd.isna(parsed):
            expiries.append(parsed)
    expiries = sorted(expiries)
    for expiry in expiries[:3]:
        if pd.isna(expiry):
            continue
        within = date_map.loc[date_map["_date"] <= expiry].tail(1)
        if within.empty:
            continue
        day = int(within.iloc[-1]["requested_days"])
        if day <= 0 or day >= max_day:
            continue
        ax.axvline(day, color="#A16207", linewidth=0.9, linestyle=":", alpha=0.50, zorder=0)


def _short_expiry_label(value: Any) -> str:
    parsed = pd.to_datetime(clean_string(value), errors="coerce")
    if pd.isna(parsed):
        return clean_string(value)
    return parsed.strftime("%b-%y")


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


def _long_call_plot_label(group: pd.DataFrame, *, view_name: str) -> str:
    lead = group.sort_values(["requested_days", "step_index"]).iloc[0]
    strike = _compact_strike_call_label(lead.get("strike_label"))
    expiry = _short_expiry_label(lead.get("expiry_date"))
    if clean_string(view_name) == "long_call_strike_view":
        return strike
    if clean_string(view_name) == "long_call_expiry_view":
        return expiry
    return f"{strike} {expiry}".strip()
    ax.figure.subplots_adjust(bottom=0.24)


def strategy_visual_spec(strategy_name: str) -> dict[str, object]:
    return dict(
        STRATEGY_VISUAL_SPECS.get(
            strategy_name,
            {"color": "#4C566A", "marker": "o", "linestyle": "-", "label": _human_label(strategy_name)},
        )
    )


def prepare_heatmap_matrix(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    value_column: str,
    x_order: list | None = None,
    y_order: list | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        raise ValueError("Heatmap input frame is empty.")
    data = data.dropna(subset=[x_column, y_column, value_column])
    if data.empty:
        raise ValueError("Heatmap input frame does not contain plottable values.")
    pivot = data.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc="mean")
    if x_order is not None:
        pivot = pivot.reindex(columns=x_order)
    else:
        pivot = pivot.reindex(_sorted_unique(pivot.columns), axis=1)
    if y_order is not None:
        pivot = pivot.reindex(y_order)
    else:
        pivot = pivot.reindex(_sorted_unique(pivot.index))
    return pivot


def _line_style(strategy_name: str, point_count: int) -> dict[str, object]:
    spec = strategy_visual_spec(strategy_name)
    if point_count <= 10:
        markevery: int | None = 1
    else:
        markevery = max(point_count // 8, 2)
    return {
        "color": spec["color"],
        "marker": spec["marker"],
        "linestyle": spec.get("linestyle", "-"),
        "linewidth": 2.6,
        "markersize": 6.0,
        "markerfacecolor": spec["color"],
        "markeredgecolor": "#ffffff",
        "markeredgewidth": 0.9,
        "markevery": markevery,
    }


def plot_payoff_at_expiry(strategy: StrategyPosition, stock_prices, output_path: str | Path) -> Path:
    prices = np.asarray(stock_prices, dtype=float)
    profits = strategy.payoff_at_expiry(prices)
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax, zero_line=True)
    ax.plot(
        prices,
        profits,
        label=strategy_visual_spec(strategy.name).get("label"),
        **_line_style(strategy.name, len(prices)),
    )
    ax.axvline(strategy.entry_spot, color="#888888", linewidth=1, linestyle=":")
    _apply_title(
        ax,
        f"{strategy.ticker} {strategy.name.replace('_', ' ').title()} Payoff At Expiry",
        subtitle="Crossing above zero means the structure is profitable at expiry.",
    )
    ax.set_xlabel("Stock Price At Expiry")
    ax.set_ylabel("Profit / Loss ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=1)
    return _finalize(fig, output_path)


def plot_multi_strategy_payoff(
    positions: list[StrategyPosition],
    stock_prices,
    output_path: str | Path,
) -> Path:
    prices = np.asarray(stock_prices, dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    _style_axes(ax, zero_line=True)
    for position in positions:
        profits = position.payoff_at_expiry(prices)
        ax.plot(
            prices,
            profits,
            label=strategy_visual_spec(position.name).get("label"),
            **_line_style(position.name, len(prices)),
        )
    ax.axvline(positions[0].entry_spot, color="#888888", linewidth=1, linestyle=":")
    _apply_title(
        ax,
        f"{positions[0].ticker} Payoff At Expiry Across Strategies",
        subtitle="Each line uses the same family color and marker as the rest of the project.",
    )
    ax.set_xlabel("Stock Price At Expiry")
    ax.set_ylabel("Profit / Loss ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_estimated_value_vs_stock(
    strategy: StrategyPosition,
    stock_prices,
    *,
    horizon_days: int,
    output_path: str | Path,
    iv_shift: float = 0.0,
    pricing_inputs: dict | None = None,
) -> Path:
    prices = np.asarray(stock_prices, dtype=float)
    valuation_date, clamped = strategy.valuation_date_for_horizon(horizon_days)
    pricing = pricing_inputs or {}
    values = strategy.mark_to_market_value(
        prices,
        valuation_date=valuation_date,
        iv_shift=iv_shift,
        risk_free_rate=pricing.get("risk_free_rate"),
        dividend_yield=pricing.get("dividend_yield"),
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax)
    ax.plot(
        prices,
        values,
        label=strategy_visual_spec(strategy.name).get("label"),
        **_line_style(strategy.name, len(prices)),
    )
    ax.axvline(strategy.entry_spot, color="#888888", linewidth=1, linestyle=":")
    suffix = " (clamped to expiry)" if clamped else ""
    _apply_title(
        ax,
        f"{strategy.ticker} {strategy.name.replace('_', ' ').title()} Estimated Value After {horizon_days} Days{suffix}",
        subtitle="Use this chart when you want mark-to-model value rather than expiry payoff.",
    )
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Estimated Position Value ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=1)
    return _finalize(fig, output_path)


def plot_multi_strategy_estimated_value(
    positions: list[StrategyPosition],
    stock_prices,
    *,
    horizon_days: int,
    output_path: str | Path,
    iv_shift: float = 0.0,
    pricing_inputs: dict | None = None,
) -> Path:
    prices = np.asarray(stock_prices, dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    _style_axes(ax)
    pricing = pricing_inputs or {}
    for position in positions:
        valuation_date, clamped = position.valuation_date_for_horizon(horizon_days)
        values = position.mark_to_market_value(
            prices,
            valuation_date=valuation_date,
            iv_shift=iv_shift,
            risk_free_rate=pricing.get("risk_free_rate"),
            dividend_yield=pricing.get("dividend_yield"),
        )
        suffix = " (clamped)" if clamped and position.option_legs else ""
        ax.plot(
            prices,
            values,
            label=f"{strategy_visual_spec(position.name).get('label')}{suffix}",
            **_line_style(position.name, len(prices)),
        )
    ax.axvline(positions[0].entry_spot, color="#888888", linewidth=1, linestyle=":")
    _apply_title(
        ax,
        f"{positions[0].ticker} Estimated Value After {horizon_days} Days",
        subtitle="Earlier horizons stay on the left; clamped traces are labelled explicitly.",
    )
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Estimated Position Value ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_required_path_strategy_compare(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot representative required paths against the active assumed path."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Required-path comparison input frame is empty.")
    required_rows = data.loc[data["series_kind"] == "required_path"].copy()
    assumed_rows = data.loc[data["series_kind"] == "assumed_path"].copy()
    if required_rows.empty or assumed_rows.empty:
        raise ValueError("Required-path comparison needs both required-path and assumed-path rows.")

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    _style_axes(ax)
    tick_map = (
        data[["requested_days", "horizon"]]
        .dropna(subset=["requested_days"])
        .drop_duplicates()
        .sort_values("requested_days")
    )

    for (series_label, strategy_family), group in _ordered_strategy_groups(required_rows, ["series_label", "strategy_family"]):
        ordered = group.sort_values("requested_days")
        ax.plot(
            ordered["requested_days"],
            ordered["spot_price"],
            label=clean_string(series_label) or strategy_visual_spec(clean_string(strategy_family)).get("label"),
            **_line_style(clean_string(strategy_family), len(ordered.index)),
        )

    assumed = assumed_rows.sort_values("requested_days").drop_duplicates(subset=["requested_days"])
    assumed_style = SUPPORTING_VISUAL_SPECS["assumed_path"]
    ax.plot(
        assumed["requested_days"],
        assumed["spot_price"],
        label="Assumed Path",
        color=str(assumed_style["color"]),
        linewidth=float(assumed_style["linewidth"]),
        linestyle=assumed_style["linestyle"],
        marker=str(assumed_style["marker"]),
        markersize=6.8,
        markerfacecolor=str(assumed_style["color"]),
        markeredgecolor="#ffffff",
        markeredgewidth=0.9,
    )
    if not tick_map.empty:
        _apply_horizon_ticks(ax, tick_map)
    _apply_title(
        ax,
        title,
        subtitle="Lower lines are easier required paths. If the assumed path stays above a line, that goal clears sooner.",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Stock Price ($)")
    _place_legend(ax, ncol=4, fontsize=9)
    return _finalize(fig, output_path)


def plot_assumed_path_value_progression(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    metric_column: str = "profit_loss",
) -> Path:
    """Plot modeled value or PnL progression along the active assumed path."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Assumed-path progression input frame is empty.")
    data = data.dropna(subset=["requested_days", metric_column])
    if data.empty:
        raise ValueError("Assumed-path progression input frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    _style_axes(ax, zero_line=(metric_column == "profit_loss"))
    tick_map = (
        data[["requested_days", "horizon"]]
        .dropna(subset=["requested_days"])
        .drop_duplicates()
        .sort_values("requested_days")
    )

    seen_signatures: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for (trace_scope, series_label, strategy_family), group in _ordered_strategy_groups(
        data,
        ["trace_scope", "series_label", "strategy_family"],
    ):
        ordered = group.sort_values("requested_days")
        style = _line_style(clean_string(strategy_family), len(ordered.index))
        if clean_string(trace_scope) == "top_candidate":
            top_candidate_style = SUPPORTING_VISUAL_SPECS["top_candidate"]
            style["linewidth"] = float(top_candidate_style["linewidth"])
            style["linestyle"] = top_candidate_style["linestyle"]
            style["color"] = top_candidate_style["color"]
            style["marker"] = top_candidate_style["marker"]
            style["markerfacecolor"] = top_candidate_style["color"]
        signature = _dedupe_series_signature(ordered["requested_days"], ordered[metric_column])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ax.plot(
            ordered["requested_days"],
            ordered[metric_column],
            label=clean_string(series_label) or clean_string(strategy_family).replace("_", " ").title(),
            **style,
        )

    if not tick_map.empty:
        _apply_horizon_ticks(ax, tick_map)
    _apply_title(
        ax,
        title,
        subtitle="Solid family lines show the active assumed path. The dashed overlay is the current top candidate when it differs materially.",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Modeled PnL ($)" if metric_column == "profit_loss" else metric_column.replace("_", " ").title())
    _format_money_axis(ax)
    _place_legend(ax, ncol=4, fontsize=9)
    return _finalize(fig, output_path)


def plot_iv_path_trace(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot the active IV path against comparison presets on the canonical horizon grid."""

    data = frame.copy()
    if data.empty:
        raise ValueError("IV-path trace input frame is empty.")
    data = data.dropna(subset=["requested_days", "iv_shift_points"])
    if data.empty:
        raise ValueError("IV-path trace input frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    _style_axes(ax, zero_line=True)
    tick_map = (
        data[["requested_days", "horizon"]]
        .dropna(subset=["requested_days"])
        .drop_duplicates()
        .sort_values("requested_days")
    )
    sort_order = {
        "active_assumption": -1,
        "flat": 0,
        "iv_down_then_stays_low": 1,
        "mean_reversion_lower": 2,
        "earnings_build_then_crush": 3,
        "iv_up_then_down": 4,
        "mean_reversion_higher": 5,
    }
    groups = list(data.groupby(["trace_scope", "iv_path_name"], dropna=False))
    groups.sort(key=lambda item: (sort_order.get(clean_string(item[0][1]), 99), clean_string(item[0][1])))
    for (trace_scope, iv_path_name), group in groups:
        ordered = group.sort_values("requested_days")
        if clean_string(trace_scope) == "active_assumption":
            active_style = IV_PATH_VISUAL_SPECS["active_assumption"]
            ax.plot(
                ordered["requested_days"],
                ordered["iv_shift_points"],
                label=clean_string(iv_path_name) or "active",
                color=str(active_style["color"]),
                linewidth=float(active_style["linewidth"]),
                linestyle=active_style["linestyle"],
                marker=str(active_style["marker"]),
                markersize=6.2,
            )
            continue
        iv_style = IV_PATH_VISUAL_SPECS.get(
            clean_string(iv_path_name),
            {"color": "#6B7280", "marker": "o", "linestyle": "--", "linewidth": 2.0},
        )
        ax.plot(
            ordered["requested_days"],
            ordered["iv_shift_points"],
            label=_human_label(clean_string(iv_path_name)),
            linewidth=float(iv_style["linewidth"]),
            linestyle=iv_style["linestyle"],
            marker=str(iv_style["marker"]),
            color=str(iv_style["color"]),
            alpha=0.92,
        )
    if not tick_map.empty:
        _apply_horizon_ticks(ax, tick_map)
    _apply_title(
        ax,
        title,
        subtitle="Negative values mean IV is below entry. The bold line is the active assumption used in the main path trace.",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel("IV Shift (pts)")
    _place_legend(ax, ncol=4, fontsize=9)
    return _finalize(fig, output_path)


def plot_compare_vs_stock_path(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    metric_column: str = "delta_profit_loss_vs_stock",
) -> Path:
    """Plot path-by-path delta versus the long-stock baseline."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Compare-vs-stock path frame is empty.")
    data = data.dropna(subset=["requested_days", metric_column])
    if data.empty:
        raise ValueError("Compare-vs-stock path frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    _style_axes(ax)
    tick_map = (
        data[["requested_days", "horizon"]]
        .dropna(subset=["requested_days"])
        .drop_duplicates()
        .sort_values("requested_days")
    )
    baseline_style = SUPPORTING_VISUAL_SPECS["stock_baseline"]
    ax.axhline(
        0,
        color=str(baseline_style["color"]),
        linewidth=float(baseline_style["linewidth"]),
        linestyle=baseline_style["linestyle"],
        label="Long Stock Baseline",
        zorder=1,
    )
    seen_signatures: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()
    for (trace_scope, series_label, strategy_family), group in _ordered_strategy_groups(
        data,
        ["trace_scope", "series_label", "strategy_family"],
    ):
        if clean_string(strategy_family) == "long_stock":
            continue
        ordered = group.sort_values("requested_days")
        style = _line_style(clean_string(strategy_family), len(ordered.index))
        if clean_string(trace_scope) == "top_candidate":
            if clean_string(strategy_family) == "long_stock":
                continue
            top_candidate_style = SUPPORTING_VISUAL_SPECS["top_candidate"]
            style["linewidth"] = float(top_candidate_style["linewidth"])
            style["linestyle"] = top_candidate_style["linestyle"]
            style["color"] = top_candidate_style["color"]
            style["marker"] = top_candidate_style["marker"]
            style["markerfacecolor"] = top_candidate_style["color"]
        signature = _dedupe_series_signature(ordered["requested_days"], ordered[metric_column])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        ax.plot(
            ordered["requested_days"],
            ordered[metric_column],
            label=clean_string(series_label) or clean_string(strategy_family).replace("_", " ").title(),
            **style,
        )
    if not tick_map.empty:
        _apply_horizon_ticks(ax, tick_map)
    _apply_title(
        ax,
        title,
        subtitle="Above zero beats long stock at that checkpoint. Below zero means the structure is lagging the stock baseline.",
    )
    ax.set_xlabel("Horizon")
    ax.set_ylabel(
        "PnL Delta vs Long Stock ($)"
        if metric_column == "delta_profit_loss_vs_stock"
        else metric_column.replace("_", " ").title()
    )
    _format_money_axis(ax)
    _place_legend(ax, ncol=4, fontsize=9)
    return _finalize(fig, output_path)


def plot_family_ranking_overview(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot family-level objective scores for the active contract-selection thesis."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Family comparison frame is empty.")
    data = data.dropna(subset=["objective_score"])
    if data.empty:
        raise ValueError("Family comparison frame has no plottable scores.")
    if "objective_rank" in data.columns:
        data = data.sort_values(["objective_rank", "objective_score"], ascending=[True, False])
    else:
        data = data.sort_values("objective_score", ascending=False)
    data = data.reset_index(drop=True)

    fig_height = max(4.8, 1.0 + len(data.index) * 0.72)
    fig, ax = plt.subplots(figsize=(11.4, fig_height))
    _style_axes(ax, zero_line=True)

    y_positions = np.arange(len(data.index))
    labels = [clean_string(value) or _human_label(clean_string(family)) for value, family in zip(data.get("strategy_label", []), data.get("strategy_family", []))]
    colors = [strategy_visual_spec(clean_string(family)).get("color", "#4C566A") for family in data["strategy_family"]]
    bars = ax.barh(
        y_positions,
        pd.to_numeric(data["objective_score"], errors="coerce").fillna(0.0),
        color=colors,
        edgecolor="#F5F2EA",
        linewidth=1.0,
        alpha=0.94,
        zorder=3,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    max_abs_score = max(abs(float(value)) for value in pd.to_numeric(data["objective_score"], errors="coerce").fillna(0.0).tolist()) or 1.0
    ax.set_xlim(
        min(pd.to_numeric(data["objective_score"], errors="coerce").fillna(0.0).min(), 0.0) - max_abs_score * 0.12,
        max(pd.to_numeric(data["objective_score"], errors="coerce").fillna(0.0).max(), 0.0) + max_abs_score * 0.18,
    )
    for bar, (_, row) in zip(bars, data.iterrows()):
        score = float(finite_or_none(row.get("objective_score")) or 0.0)
        x_position = score + (max_abs_score * 0.02 if score >= 0 else -max_abs_score * 0.02)
        ha = "left" if score >= 0 else "right"
        ax.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"{score:,.0f}",
            va="center",
            ha=ha,
            fontsize=9.2,
            color="#292524",
            fontweight="bold",
        )
        if bool(row.get("best_under_current_objective")):
            best_x = ax.get_xlim()[1] - max_abs_score * 0.02
            ax.text(
                best_x,
                bar.get_y() + bar.get_height() / 2,
                "best overall",
                va="center",
                ha="right",
                fontsize=8.9,
                color="#14532D",
                fontweight="bold",
            )

    _apply_title(
        ax,
        title,
        subtitle="Higher scores fit the active objective better. Use family_comparison.csv when the family edge is weak or compressed.",
    )
    ax.set_xlabel("Active Objective Score")
    ax.set_ylabel("Strategy Family")
    return _finalize(fig, output_path)


def _compact_candidate_label(value: Any, *, limit: int = 28) -> str:
    text = clean_string(value)
    if not text:
        return "Candidate"
    text = text.replace("Long Call ", "").replace("Long Stock Baseline", "Stock")
    text = text.replace(" 00:00:00", "")
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _compact_highlight_candidate(value: Any) -> str:
    """Convert verbose analysis labels into chart-sized option labels."""

    text = clean_string(value)
    match = re.match(r"Long Call\s+(\d{4}-\d{2}-\d{2})\s+([0-9.]+)", text, flags=re.IGNORECASE)
    if match:
        expiry = _short_expiry_label(match.group(1))
        strike = _compact_strike_call_label(match.group(2))
        return f"{strike} {expiry}"
    if text.lower() == "long stock baseline":
        return "Stock"
    return _compact_candidate_label(text, limit=22)


def _compact_tradeoff_label(value: Any) -> str:
    """Short label for scatter points without exposing internal candidate slugs."""

    text = clean_string(value)
    if text.lower() == "long stock baseline":
        return "Stock"
    single = re.match(r"(Long Call|Covered Call|Cash Secured Put)\s+(\d{4}-\d{2}-\d{2})\s+([0-9.]+)", text, flags=re.IGNORECASE)
    if single:
        family, expiry, strike = single.groups()
        prefix = {"long call": "", "covered call": "Cov ", "cash secured put": "CSP "}.get(family.lower(), "")
        return f"{prefix}{_compact_strike_call_label(strike)} {_short_expiry_label(expiry)}".strip()
    spread = re.match(r"(Bull Call Spread|Bear Put Spread)\s+(\d{4}-\d{2}-\d{2})\s+([0-9.]+)/([0-9.]+)", text, flags=re.IGNORECASE)
    if spread:
        family, expiry, low, high = spread.groups()
        prefix = "BCS" if family.lower().startswith("bull") else "BPS"
        return f"{prefix} {float(low):g}/{float(high):g} {_short_expiry_label(expiry)}"
    return _compact_candidate_label(text, limit=18)


def _short_highlight_status(value: Any) -> str:
    text = clean_string(value).lower()
    mapping = {
        "option_interesting_but_stock_still_benchmark": "Option; stock benchmark",
        "stock_cleaner_under_current_assumptions": "Stock cleaner",
        "weak_differentiation": "Weak edge",
        "no_clear_edge_under_current_assumptions": "No clear edge",
        "informative_edge": "Informative edge",
    }
    return mapping.get(text, _compact_candidate_label(text.replace("_", " "), limit=27))


def _short_trust_label(value: Any) -> str:
    text = clean_string(value).lower()
    mapping = {
        "trusted_quoted": "Trusted quoted",
        "quoted_prior_day": "Prior-day quoted",
        "fallback_only": "Fallback only",
        "sparse_fallback": "Sparse fallback",
        "structure_only": "Structure only",
    }
    return mapping.get(text, _compact_candidate_label(text.replace("_", " "), limit=18) or "n/a")


def _short_highlight_warning(value: Any) -> str:
    text = clean_string(value)
    if not text:
        return "No major caution"
    lowered = text.lower()
    parts: list[str] = []
    if "stock still dominates" in lowered:
        parts.append("Stock still dominates")
    if "timing or expiry fit is weak" in lowered:
        parts.append("Timing/expiry weak")
    if "weak_differentiation" in lowered or "weak differentiation" in lowered:
        parts.append("Weak edge")
    if "assumption-relative" in lowered:
        parts.append("Assumption-relative")
    if "iv" in lowered and "support" in lowered:
        parts.append("Needs IV support")
    if parts:
        return "; ".join(dict.fromkeys(parts))
    return _compact_candidate_label(text.replace("_", " "), limit=34)


def _top_candidate_rows(frame: pd.DataFrame, *, score_column: str = "balanced_score", limit: int = 10) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    if score_column not in data.columns:
        score_column = "objective_score" if "objective_score" in data.columns else data.columns[0]
    data[score_column] = pd.to_numeric(data[score_column], errors="coerce").fillna(0.0)
    if "strategy_family" in data.columns:
        stock = data.loc[data["strategy_family"].astype(str).str.lower().eq("long_stock")].head(1)
        calls = data.loc[~data.index.isin(stock.index)].sort_values(score_column, ascending=False).head(max(limit - len(stock.index), 1))
        data = pd.concat([stock, calls], ignore_index=True)
    else:
        data = data.sort_values(score_column, ascending=False).head(limit)
    return data.reset_index(drop=True)


def plot_highlights_overview(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render a compact decision-highlight table as a chart."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Decision highlights frame is empty.")
    data = data.sort_values("display_order") if "display_order" in data.columns else data
    data = data.head(11).reset_index(drop=True)
    fig_height = max(6.2, 1.15 + len(data.index) * 0.48)
    fig, ax = plt.subplots(figsize=(12.6, fig_height))
    fig.patch.set_facecolor(_FIGURE_FACE)
    ax.set_facecolor(_AXES_FACE)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15.5, fontweight="bold", pad=14)
    ax.text(
        0.0,
        0.965,
        "Assumption-relative categories. Weak/no-clear edge is shown explicitly instead of forcing a false winner.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#57534E",
        ha="left",
        va="top",
    )
    columns = [
        ("Category", 0.00),
        ("Selected", 0.27),
        ("Status", 0.43),
        ("Trust", 0.63),
        ("Main warning", 0.77),
    ]
    header_y = 0.90
    for label, x0 in columns:
        ax.text(x0, header_y, label, transform=ax.transAxes, fontsize=9.3, fontweight="bold", color="#292524")
    for idx, row in enumerate(data.to_dict("records")):
        y = header_y - 0.055 * (idx + 1)
        if idx % 2 == 0:
            ax.axhspan(y - 0.018, y + 0.030, xmin=0.0, xmax=1.0, color="#F4F1EA", alpha=0.72, zorder=0)
        family = clean_string(row.get("selected_family"))
        color = strategy_visual_spec(family).get("color", "#4C566A")
        ax.text(0.00, y, clean_string(row.get("highlight_label")), transform=ax.transAxes, fontsize=8.6, color="#292524", va="center")
        ax.text(0.27, y, _compact_highlight_candidate(row.get("selected_candidate_label")), transform=ax.transAxes, fontsize=8.8, color=color, va="center", fontweight="bold")
        ax.text(0.43, y, _short_highlight_status(row.get("decision_status")), transform=ax.transAxes, fontsize=8.3, color="#44403C", va="center")
        ax.text(0.63, y, _short_trust_label(row.get("source_trust_label")), transform=ax.transAxes, fontsize=8.3, color="#44403C", va="center")
        ax.text(0.77, y, _short_highlight_warning(row.get("main_warning")), transform=ax.transAxes, fontsize=8.15, color="#57534E", va="center")
    return _finalize(fig, output_path)


def plot_candidate_robustness_vs_upside(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Scatter robustness against aggressive upside with stock marked as baseline."""

    data = _top_candidate_rows(frame, score_column="balanced_score", limit=9)
    if data.empty:
        raise ValueError("Candidate tradeoff frame is empty.")
    fig, ax = plt.subplots(figsize=(11.8, 7.1))
    _style_axes(ax)
    x = pd.to_numeric(data.get("robustness_score"), errors="coerce").fillna(0.0)
    y = pd.to_numeric(data.get("aggressive_upside_score"), errors="coerce").fillna(0.0)
    for idx, row in data.iterrows():
        family = clean_string(row.get("strategy_family"))
        spec = strategy_visual_spec(family)
        is_stock = family == "long_stock"
        x_value = _num_safe(row.get("robustness_score"))
        y_value = _num_safe(row.get("aggressive_upside_score"))
        ax.scatter(
            x_value,
            y_value,
            s=190 if is_stock else 105,
            color=spec.get("color", "#4C566A"),
            marker=spec.get("marker", "o"),
            edgecolor="#FFFFFF",
            linewidth=1.0,
            alpha=0.92,
            label=str(spec.get("label", _human_label(family))),
            zorder=4,
        )
        label_offsets = [(8, 4), (8, 13), (10, -13), (8, -13), (9, 9), (9, 6), (-42, 16), (-42, -12), (-46, 2)]
        offset_x, offset_y = label_offsets[idx % len(label_offsets)]
        ax.annotate(
            _compact_tradeoff_label(row.get("candidate_label")),
            xy=(x_value, y_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            va="center",
            ha="right" if offset_x < 0 else "left",
            fontsize=7.5,
            color="#292524",
        )
    ax.axvline(float(x.median() if len(x.index) else 50), color="#A8A29E", linestyle=":", linewidth=1.0)
    ax.axhline(float(y.median() if len(y.index) else 50), color="#A8A29E", linestyle=":", linewidth=1.0)
    _apply_title(ax, title, subtitle="Upper-right is more balanced: stronger upside and stronger survival/resilience. Stock is shown as the no-option baseline.")
    ax.set_xlabel("Robustness / Resilience Score")
    ax.set_ylabel("Aggressive Upside Score")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", bbox_to_anchor=(0.0, -0.16), ncol=3, frameon=False, fontsize=8.6)
    return _finalize(fig, output_path)


def _num_safe(value: Any) -> float:
    return float(finite_or_none(value) or 0.0)


ACTION_BUCKET_VISUALS = {
    "Buy Now": {"color": "#009E73", "marker": "o", "label": "Buy Now"},
    "Watchlist": {"color": "#E69F00", "marker": "^", "label": "Watchlist"},
    "Avoid For Now": {"color": "#D55E00", "marker": "X", "label": "Avoid"},
    "Prefer Stock Instead": {"color": "#000000", "marker": "s", "label": "Prefer Stock"},
}
ACTION_BUCKET_ORDER = {
    "Buy Now": 0,
    "Watchlist": 1,
    "Prefer Stock Instead": 2,
    "Avoid For Now": 3,
}
CHAIN_OVERVIEW_VERDICT_VISUALS = {
    "Robust buy candidate": {"color": "#0072B2", "marker": "o"},
    "Selective / thesis-dependent": {"color": "#E69F00", "marker": "^"},
    "Too narrow": {"color": "#CC79A7", "marker": "X"},
    "Stock better": {"color": "#D55E00", "marker": "s"},
}


def _action_bucket_visual(bucket: Any) -> dict[str, Any]:
    return ACTION_BUCKET_VISUALS.get(clean_string(bucket), {"color": "#4C566A", "marker": "D", "label": clean_string(bucket) or "Other"})


def _chain_verdict_visual(verdict: Any) -> dict[str, Any]:
    return CHAIN_OVERVIEW_VERDICT_VISUALS.get(clean_string(verdict), {"color": "#7F7F7F", "marker": "D"})


def _sort_action_frame(frame: pd.DataFrame, *, limit: int | None = None) -> pd.DataFrame:
    data = frame.copy()
    data["_bucket_order"] = data.get("action_bucket", pd.Series(dtype=str)).map(ACTION_BUCKET_ORDER).fillna(99)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999)
    data["action_score"] = pd.to_numeric(data.get("action_score"), errors="coerce").fillna(0.0)
    data = data.sort_values(["_bucket_order", "action_priority_rank", "action_score"], ascending=[True, True, False])
    if limit is not None:
        data = data.head(limit)
    return data.drop(columns=["_bucket_order"], errors="ignore")


def _finalize_decision_chart(fig, output_path: str | Path, *, bottom: float = 0.12) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(windows_extended_path(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _action_chart_caption(fig, text: str) -> None:
    if clean_string(text):
        fig.text(0.08, 0.035, clean_string(text), ha="left", va="bottom", fontsize=8.6, color="#57534E")


def _finalize_caption_chart(fig, output_path: str | Path, *, bottom: float = 0.09, top: float = 0.94) -> Path:
    path = Path(output_path)
    ensure_directory(path.parent)
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(windows_extended_path(path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _wrap_chart_text(value: Any, *, width: int) -> str:
    text = clean_string(value)
    if not text:
        return "-"
    wrapped = textwrap.wrap(
        text,
        width=max(8, int(width)),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return "\n".join(wrapped) if wrapped else text


def _wrapped_line_count(value: str) -> int:
    return max(1, clean_string(value).count("\n") + 1)


def _draw_banded_text_table(
    ax,
    *,
    title: str,
    rows: list[dict[str, Any]],
    columns: list[dict[str, Any]],
    caption: str,
    marker_bucket_key: str | None = None,
) -> None:
    fig = ax.figure
    fig.patch.set_facecolor(_FIGURE_FACE)
    ax.set_facecolor(_AXES_FACE)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=15.2, fontweight="bold", pad=14)

    top_y = 0.93
    header_y = top_y
    left_margin = 0.02
    right_margin = 0.985
    for column in columns:
        ax.text(
            float(column["x"]),
            header_y,
            clean_string(column["label"]),
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            color="#292524",
            ha="left",
            va="top",
        )

    wrapped_rows: list[tuple[dict[str, str], float]] = []
    for row in rows:
        wrapped_row: dict[str, str] = {}
        max_lines = 1
        for column in columns:
            key = clean_string(column.get("key"))
            wrapped = _wrap_chart_text(row.get(key), width=int(column.get("wrap", 22)))
            wrapped_row[key] = wrapped
            max_lines = max(max_lines, _wrapped_line_count(wrapped))
        row_units = 0.9 + max_lines * 1.15
        wrapped_rows.append((wrapped_row, row_units))

    total_units = 1.7 + sum(units + 0.28 for _, units in wrapped_rows)
    usable_height = 0.82
    unit_scale = usable_height / max(total_units, 1.0)
    y_top = header_y - 0.05

    for idx, (row, row_units) in enumerate(wrapped_rows):
        row_height = row_units * unit_scale
        y_bottom = y_top - row_height
        if idx % 2 == 0:
            ax.add_patch(
                Rectangle(
                    (left_margin, y_bottom + 0.004),
                    right_margin - left_margin,
                    row_height - 0.008,
                    transform=ax.transAxes,
                    facecolor="#F4F1EA",
                    edgecolor="none",
                    alpha=0.82,
                    zorder=0,
                )
            )
        if marker_bucket_key:
            bucket = clean_string(row.get(marker_bucket_key))
            visual = _action_bucket_visual(bucket)
            marker_x = float(columns[0]["x"]) + 0.006
            marker_y = y_top - 0.020
            ax.scatter(marker_x, marker_y, transform=ax.transAxes, s=70, color=visual["color"], marker=visual["marker"], zorder=3)
        for column in columns:
            key = clean_string(column.get("key"))
            x = float(column["x"])
            color = str(column.get("color", "#44403C"))
            weight = str(column.get("fontweight", "normal"))
            fontsize = float(column.get("fontsize", 8.2))
            bucket_key = clean_string(column.get("bucket_color_from"))
            if bucket_key:
                visual = _action_bucket_visual(row.get(bucket_key))
                color = str(visual["color"])
            ax.text(
                x,
                y_top - 0.012,
                row.get(key, "-"),
                transform=ax.transAxes,
                fontsize=fontsize,
                color=color,
                fontweight=weight,
                va="top",
                ha="left",
                linespacing=1.20,
                clip_on=False,
            )
        y_top = y_bottom - (0.28 * unit_scale)

    _action_chart_caption(fig, caption)


def plot_action_board_overview(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render a compact Buy/Watch/Avoid/Stock scorecard."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Action board frame is empty.")
    rows = []
    available_families = set(data.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower())
    bucket_sequence = ["Buy Now", "Watchlist"]
    if "long_stock" in available_families:
        bucket_sequence.append("Prefer Stock Instead")
    bucket_sequence.append("Avoid For Now")
    for bucket in bucket_sequence:
        subset = data.loc[data.get("action_bucket", pd.Series(dtype=str)).astype(str).eq(bucket)].sort_values("action_priority_rank").head(3)
        if subset.empty:
            if bucket != "Buy Now":
                continue
            rows.append(
                {
                    "bucket": bucket,
                    "candidate": "No candidates cleared this bucket.",
                    "confidence": "",
                    "why": "No row met the current threshold.",
                    "warning": "This empty bucket is honest: there may simply be no actionable contract here.",
                    "trust": "",
                }
            )
        else:
            for row in subset.to_dict("records"):
                rows.append(
                    {
                        "bucket": bucket,
                        "candidate": _compact_tradeoff_label(row.get("candidate_label")),
                        "confidence": clean_string(row.get("action_confidence")).title(),
                        "why": clean_string(row.get("why_this_is_interesting_now")) or clean_string(row.get("headline_reason")),
                        "warning": clean_string(row.get("upgrade_rule")) or clean_string(row.get("main_trigger")) or clean_string(row.get("main_warning")),
                        "trust": _human_label(clean_string(row.get("source_trust_label")) or "unknown_trust"),
                    }
                )
    line_units = 0
    for row in rows:
        line_units += max(
            _wrapped_line_count(_wrap_chart_text(row.get("why"), width=30)),
            _wrapped_line_count(_wrap_chart_text(row.get("warning"), width=30)),
            1,
        )
    fig_height = max(5.8, min(12.8, 1.7 + line_units * 0.48))
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    _draw_banded_text_table(
        ax,
        title=title,
        rows=rows,
        columns=[
            {"label": "Bucket", "key": "bucket", "x": 0.03, "wrap": 16, "fontsize": 8.5, "fontweight": "bold", "bucket_color_from": "bucket"},
            {"label": "Candidate", "key": "candidate", "x": 0.18, "wrap": 24, "fontsize": 8.7, "fontweight": "bold"},
            {"label": "Confidence", "key": "confidence", "x": 0.34, "wrap": 12, "fontsize": 8.2},
            {"label": "Short Why", "key": "why", "x": 0.45, "wrap": 33, "fontsize": 8.0},
            {"label": "Main Warning / Trigger", "key": "warning", "x": 0.70, "wrap": 31, "fontsize": 7.9},
            {"label": "Trust", "key": "trust", "x": 0.91, "wrap": 14, "fontsize": 8.0},
        ],
        caption="Buckets are assumption-relative. Watchlist means interesting, but not yet buyable without a better trigger, entry, IV setup, or timing profile.",
        marker_bucket_key="bucket",
    )
    return _finalize_decision_chart(fig, output_path)


def plot_conviction_vs_robustness(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Scatter action conviction against robustness."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Action board frame is empty.")
    data["candidate_conviction_score"] = pd.to_numeric(data.get("candidate_conviction_score"), errors="coerce").fillna(0.0)
    data["robustness_score"] = pd.to_numeric(data.get("robustness_score"), errors="coerce").fillna(0.0)
    key = _sort_action_frame(data, limit=16)
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    _style_axes(ax)
    for idx, row in key.reset_index(drop=True).iterrows():
        visual = _action_bucket_visual(row.get("action_bucket"))
        x_value = _num_safe(row.get("robustness_score"))
        y_value = _num_safe(row.get("candidate_conviction_score"))
        ax.scatter(
            x_value,
            y_value,
            s=155 if row.get("strategy_family") == "long_stock" else 110,
            color=visual["color"],
            marker=visual["marker"],
            edgecolor="#FFFFFF",
            linewidth=1.0,
            alpha=0.92,
            label=visual["label"],
            zorder=4,
        )
        if idx < 10 or clean_string(row.get("action_bucket")) in {"Buy Now", "Prefer Stock Instead"}:
            offsets = [(8, 7), (8, -10), (-42, 9), (-44, -9), (10, 15)]
            offset_x, offset_y = offsets[idx % len(offsets)]
            ax.annotate(
                _compact_tradeoff_label(row.get("candidate_label")),
                xy=(x_value, y_value),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                fontsize=7.5,
                color="#292524",
                ha="right" if offset_x < 0 else "left",
                va="center",
            )
    ax.axvline(float(key["robustness_score"].median()), color="#A8A29E", linestyle=":", linewidth=1.0)
    ax.axhline(float(key["candidate_conviction_score"].median()), color="#A8A29E", linestyle=":", linewidth=1.0)
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    ax.set_xlabel("Robustness Score")
    ax.set_ylabel("Action Conviction Score")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=4, frameon=False, fontsize=8.4)
    _action_chart_caption(fig, "Upper-right means stronger action interest: more robust and more compelling under the current assumptions.")
    return _finalize_decision_chart(fig, output_path)


def plot_buy_watch_avoid_matrix(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show bucketed candidates with key score bars."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Action board frame is empty.")
    data = _sort_action_frame(data, limit=14)
    labels = [_compact_tradeoff_label(value) for value in data.get("candidate_label", pd.Series(dtype=str))]
    y = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(11.8, max(5.8, len(data.index) * 0.42 + 1.6)))
    _style_axes(ax)
    values = pd.to_numeric(data.get("action_score"), errors="coerce").fillna(0.0)
    colors = [_action_bucket_visual(bucket)["color"] for bucket in data.get("action_bucket", pd.Series(dtype=str))]
    ax.barh(y, values, color=colors, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Action Score")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    for idx, row in enumerate(data.to_dict("records")):
        ax.text(min(98, _num_safe(row.get("action_score")) + 1.2), idx, clean_string(row.get("action_bucket")), va="center", fontsize=7.8, color="#44403C")
    _action_chart_caption(fig, "Bars are ordered by action bucket and priority. Avoid/Watch rows can still be interesting, but they did not clear buy thresholds.")
    return _finalize_decision_chart(fig, output_path)


def plot_trigger_map(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render watchlist/avoid triggers as a compact map."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Decision trigger frame is empty.")
    data = data.head(12).reset_index(drop=True)
    rows = [
        {
            "bucket": clean_string(row.get("action_bucket")),
            "candidate": _compact_tradeoff_label(row.get("candidate_label")),
            "trigger_type": clean_string(row.get("trigger_type_label")) or _human_label(row.get("key_trigger_type")),
            "upgrade_rule": clean_string(row.get("upgrade_rule")) or clean_string(row.get("what_has_to_happen")) or clean_string(row.get("key_trigger_value")),
            "deadline": clean_string(row.get("key_trigger_deadline")) or "n/a",
            "invalidate_if": clean_string(row.get("invalidate_rule")) or clean_string(row.get("what_would_invalidate")),
            "trust": _human_label(clean_string(row.get("source_trust_label")) or "unknown_trust"),
        }
        for row in data.to_dict("records")
    ]
    line_units = 0
    for row in rows:
        line_units += max(
            _wrapped_line_count(_wrap_chart_text(row.get("upgrade_rule"), width=27)),
            _wrapped_line_count(_wrap_chart_text(row.get("invalidate_if"), width=25)),
            1,
        )
    fig_height = max(5.8, min(13.0, 1.8 + line_units * 0.47))
    fig, ax = plt.subplots(figsize=(13.6, fig_height))
    _draw_banded_text_table(
        ax,
        title=title,
        rows=rows,
        columns=[
            {"label": "Candidate", "key": "candidate", "x": 0.02, "wrap": 18, "fontsize": 8.5, "fontweight": "bold"},
            {"label": "Trigger Type", "key": "trigger_type", "x": 0.20, "wrap": 16, "fontsize": 8.1, "fontweight": "bold", "bucket_color_from": "bucket"},
            {"label": "Upgrade Rule", "key": "upgrade_rule", "x": 0.36, "wrap": 27, "fontsize": 7.95},
            {"label": "Deadline", "key": "deadline", "x": 0.62, "wrap": 12, "fontsize": 7.95},
            {"label": "Invalidate If", "key": "invalidate_if", "x": 0.73, "wrap": 25, "fontsize": 7.85},
            {"label": "Trust", "key": "trust", "x": 0.93, "wrap": 14, "fontsize": 7.9},
        ],
        caption="Trigger map is for Watchlist and Avoid rows. Upgrade Rule says what has to improve first; Invalidate If says when the setup should stay off the active shortlist.",
        marker_bucket_key=None,
    )
    return _finalize_decision_chart(fig, output_path)


def plot_top_candidate_cards(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render compact bullish top-candidate cards for the overview surface."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Top candidate cards frame is empty.")
    if "card_rank" in data.columns:
        data["card_rank"] = pd.to_numeric(data.get("card_rank"), errors="coerce").fillna(999)
        data = data.sort_values("card_rank")
    data = data.head(5).reset_index(drop=True)

    wrapped_cards: list[tuple[dict[str, str], float]] = []
    for row in data.to_dict("records"):
        card = {
            "contract": _wrap_chart_text(row.get("contract_label"), width=22),
            "bucket": clean_string(row.get("bucket")),
            "confidence": clean_string(row.get("confidence")),
            "trust": clean_string(row.get("trust")),
            "interesting": _wrap_chart_text(row.get("why_this_is_interesting"), width=72),
            "hurts": _wrap_chart_text(row.get("what_hurts_it"), width=72),
            "upgrade": _wrap_chart_text(row.get("upgrade_rule"), width=72),
            "stock_note": _wrap_chart_text(row.get("compare_vs_stock_note"), width=72),
        }
        body_lines = (
            _wrapped_line_count(card["interesting"])
            + _wrapped_line_count(card["hurts"])
            + _wrapped_line_count(card["upgrade"])
            + _wrapped_line_count(card["stock_note"])
        )
        wrapped_cards.append((card, 0.85 + body_lines * 0.235 + 4 * 0.095))

    card_gap = 0.22
    title_area = 0.70
    total_height = title_area + sum(height for _, height in wrapped_cards) + card_gap * max(len(wrapped_cards) - 1, 0) + 0.20
    fig_height = max(5.8, min(16.0, total_height * 1.02))
    fig, ax = plt.subplots(figsize=(12.8, fig_height))
    fig.patch.set_facecolor(_FIGURE_FACE)
    ax.set_facecolor(_AXES_FACE)
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, total_height)
    ax.text(0.02, total_height - 0.06, title, fontsize=16.0, fontweight="bold", color="#1C1917", va="top", ha="left")

    left = 0.025
    right = 0.975
    top_y = total_height - title_area

    for card, card_height in wrapped_cards:
        y_bottom = top_y - card_height
        visual = _action_bucket_visual(card["bucket"])
        ax.add_patch(
            Rectangle(
                (left, y_bottom),
                right - left,
                card_height,
                facecolor="#FFFDF9",
                edgecolor=str(visual["color"]),
                linewidth=1.2,
                alpha=0.92,
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (left, y_bottom),
                0.012,
                card_height,
                facecolor=str(visual["color"]),
                edgecolor="none",
                alpha=0.95,
                zorder=1,
            )
        )
        header_y = top_y - 0.18
        ax.scatter(left + 0.024, header_y - 0.02, s=78, color=visual["color"], marker=visual["marker"], zorder=3)
        ax.text(left + 0.052, header_y, card["contract"], fontsize=10.4, fontweight="bold", color="#1C1917", va="top", ha="left")
        ax.text(left + 0.36, header_y, card["bucket"], fontsize=8.8, fontweight="bold", color=visual["color"], va="top", ha="left")
        ax.text(right - 0.22, header_y, f"Confidence: {card['confidence'] or 'n/a'}", fontsize=8.2, color="#44403C", va="top", ha="left")
        ax.text(right - 0.22, header_y - 0.26, f"Trust: {card['trust'] or 'n/a'}", fontsize=8.2, color="#44403C", va="top", ha="left")

        label_x = left + 0.052
        text_x = left + 0.165
        line_y = header_y - 0.48
        for prefix, text in [
            ("Interesting:", card["interesting"]),
            ("Held back by:", card["hurts"]),
            ("Upgrade if:", card["upgrade"]),
            ("Vs stock:", card["stock_note"]),
        ]:
            ax.text(label_x, line_y, prefix, fontsize=8.2, fontweight="bold", color="#44403C", va="top", ha="left")
            ax.text(text_x, line_y, text, fontsize=8.1, color="#44403C", va="top", ha="left", linespacing=1.18)
            line_y -= max(_wrapped_line_count(text), 1) * 0.235 + 0.095

        top_y = y_bottom - card_gap

    _action_chart_caption(fig, "Each card is one bullish call. Read top-down: why it is interesting, what hurts it, what would upgrade it, and whether stock still looks cleaner.")
    return _finalize_decision_chart(fig, output_path, bottom=0.10)


def plot_chain_overview(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render compact compare-options cards plus a small verdict distribution."""

    if summary.empty:
        raise ValueError("Chain overview summary frame is empty.")

    cards = summary.copy().head(6).reset_index(drop=True)
    verdict_counts = (
        candidates.get("final_verdict", pd.Series(dtype=str)).astype(str).value_counts().to_dict()
        if candidates is not None and not candidates.empty
        else {}
    )
    shared_path_count = int(pd.to_numeric(candidates.get("shared_path_family_count"), errors="coerce").fillna(0).max()) if candidates is not None and not candidates.empty else 0
    candidate_count = int(len(candidates.index)) if candidates is not None else 0
    threshold_note = ""
    if candidates is not None and not candidates.empty:
        first = candidates.iloc[0]
        threshold_note = (
            f"Shared paths: {int(pd.to_numeric(first.get('shared_path_family_count'), errors='coerce') or 0)} | "
            f"Min win: {float(finite_or_none(first.get('minimum_outperformance_multiple')) or 0):.1f}x in "
            f"{int(pd.to_numeric(first.get('required_winning_path_families'), errors='coerce') or 0)} families"
        )

    fig = plt.figure(figsize=(15.4, 9.6))
    fig.patch.set_facecolor(_FIGURE_FACE)
    grid = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.25, 1.25, 1.0], hspace=0.34, wspace=0.22)
    fig.suptitle(title, x=0.065, y=0.985, ha="left", fontsize=16.2, fontweight="bold", color="#1C1917")

    for idx, row in enumerate(cards.to_dict("records")):
        ax = fig.add_subplot(grid[idx // 3, idx % 3])
        ax.axis("off")
        ax.set_facecolor(_FIGURE_FACE)
        visual = _chain_verdict_visual(row.get("verdict_badge"))
        ax.add_patch(
            Rectangle(
                (0.02, 0.04),
                0.96,
                0.90,
                transform=ax.transAxes,
                facecolor="#FFFDF9",
                edgecolor=visual["color"],
                linewidth=1.25,
                alpha=0.94,
            )
        )
        ax.add_patch(
            Rectangle(
                (0.02, 0.04),
                0.022,
                0.90,
                transform=ax.transAxes,
                facecolor=visual["color"],
                edgecolor="none",
                alpha=0.98,
            )
        )
        ax.scatter([0.08], [0.84], transform=ax.transAxes, s=82, color=visual["color"], marker=visual["marker"])
        ax.text(0.14, 0.88, _wrap_chart_text(row.get("card_label"), width=22), transform=ax.transAxes, fontsize=9.4, fontweight="bold", color="#1C1917", va="top")
        ax.text(0.14, 0.72, _wrap_chart_text(row.get("contract_label"), width=24), transform=ax.transAxes, fontsize=10.6, fontweight="bold", color="#292524", va="top")
        ax.text(0.14, 0.56, clean_string(row.get("verdict_badge")) or "n/a", transform=ax.transAxes, fontsize=8.4, fontweight="bold", color=visual["color"], va="top")
        ax.text(0.14, 0.42, clean_string(row.get("headline_metric")) or "n/a", transform=ax.transAxes, fontsize=8.6, fontweight="bold", color="#44403C", va="top")
        ax.text(
            0.14,
            0.29,
            _wrap_chart_text(row.get("headline_note") or row.get("explanation_short"), width=34),
            transform=ax.transAxes,
            fontsize=8.2,
            color="#44403C",
            va="top",
            linespacing=1.16,
        )

    ax_dist = fig.add_subplot(grid[2, :2])
    _style_axes(ax_dist)
    verdict_order = ["Robust buy candidate", "Selective / thesis-dependent", "Too narrow", "Stock better"]
    y = np.arange(len(verdict_order))
    values = [int(verdict_counts.get(label, 0)) for label in verdict_order]
    colors = [_chain_verdict_visual(label)["color"] for label in verdict_order]
    ax_dist.barh(y, values, color=colors, alpha=0.88)
    ax_dist.set_yticks(y)
    ax_dist.set_yticklabels(verdict_order, fontsize=9.2)
    ax_dist.invert_yaxis()
    ax_dist.set_xlabel("Candidate Count")
    ax_dist.set_title("Verdict Distribution", loc="left", fontsize=12.2, fontweight="bold", pad=8)
    for idx, value in enumerate(values):
        ax_dist.text(value + 0.08, idx, str(value), va="center", fontsize=8.2, color="#44403C")
    ax_dist.set_xlim(0, max(values + [1]) + 1)

    ax_note = fig.add_subplot(grid[2, 2])
    ax_note.axis("off")
    ax_note.set_facecolor(_FIGURE_FACE)
    ax_note.add_patch(
        Rectangle((0.02, 0.02), 0.96, 0.92, transform=ax_note.transAxes, facecolor="#FFFDF9", edgecolor="#DED8CE", linewidth=0.9)
    )
    ax_note.text(0.07, 0.88, "Scope", transform=ax_note.transAxes, fontsize=11.2, fontweight="bold", color="#1C1917", va="top")
    note_lines = [
        f"Bullish calls compared: {candidate_count}",
        f"Shared path families: {shared_path_count}",
        threshold_note or "Thresholds are inherited from the frozen decision defaults.",
        "Benchmark: long stock, not zero P/L.",
    ]
    y_note = 0.70
    for line in note_lines:
        wrapped = _wrap_chart_text(line, width=30)
        ax_note.text(
            0.07,
            y_note,
            wrapped,
            transform=ax_note.transAxes,
            fontsize=8.5,
            color="#44403C",
            va="top",
            linespacing=1.18,
        )
        y_note -= _wrapped_line_count(wrapped) * 0.095 + 0.055

    _action_chart_caption(
        fig,
        "Cards highlight the best robust, asymmetric, early, late, and IV-fragile reads. The comparison stays versus long stock and uses one shared representative path set for every call.",
    )
    return _finalize_caption_chart(fig, output_path, bottom=0.09, top=0.95)


def plot_stock_vs_option_preference_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show where stock is cleaner versus option alternatives."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Action board frame is empty.")
    data = _sort_action_frame(data, limit=12)
    data["stock_relative_score"] = pd.to_numeric(data.get("stock_relative_score"), errors="coerce").fillna(0.0)
    labels = [_compact_tradeoff_label(value) for value in data.get("candidate_label", pd.Series(dtype=str))]
    y = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(11.8, max(5.8, len(data.index) * 0.42 + 1.6)))
    _style_axes(ax)
    colors = [_action_bucket_visual(bucket)["color"] for bucket in data.get("action_bucket", pd.Series(dtype=str))]
    ax.barh(y, data["stock_relative_score"], color=colors, alpha=0.82)
    ax.axvline(50, color="#78716C", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Stock-Relative Score")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    for idx, row in enumerate(data.to_dict("records")):
        note = "stock cleaner" if clean_string(row.get("action_bucket")) == "Prefer Stock Instead" or _num_safe(row.get("difference_vs_stock")) <= 0 else "option edge"
        ax.text(min(98, _num_safe(row.get("stock_relative_score")) + 1.2), idx, note, va="center", fontsize=7.8, color="#44403C")
    _action_chart_caption(fig, "Score above the dashed midpoint means the option is closer to clearing stock; below it, stock remains cleaner.")
    return _finalize_decision_chart(fig, output_path)


def _short_thesis_status(value: Any) -> str:
    mapping = {
        "reasonable_under_thesis": "Reasonable",
        "near_watchlist_under_thesis": "Near / Watch",
        "too_expensive_under_thesis": "Too Expensive",
    }
    text = clean_string(value)
    return mapping.get(text, text.replace("_", " ").title() or "n/a")


def _money_text(value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    return f"${float(number):,.0f}" if abs(float(number)) >= 100 else f"${float(number):,.2f}"


def _edge_pct_text(value: Any) -> str:
    number = finite_or_none(value)
    if number is None:
        return "n/a"
    return f"{float(number):+.0f}%"


def plot_thesis_path_gallery(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show multiple path families to the same thesis endpoint."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Thesis path gallery frame is empty.")
    data["requested_days"] = pd.to_numeric(data.get("requested_days"), errors="coerce").fillna(0).astype(int)
    data["stock_price"] = pd.to_numeric(data.get("stock_price"), errors="coerce")
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    _style_axes(ax)
    for idx, (path_name, group) in enumerate(data.groupby("path_family", sort=False)):
        group = group.sort_values("requested_days")
        spec = STOCK_PATH_GALLERY_SPECS.get(clean_string(path_name), LONG_CALL_COMPARISON_SPECS[idx % len(LONG_CALL_COMPARISON_SPECS)])
        ax.plot(
            group["requested_days"],
            group["stock_price"],
            color=str(spec["color"]),
            marker=spec.get("marker"),
            linestyle=spec.get("linestyle", "-"),
            linewidth=float(spec.get("linewidth", 2.4)),
            markersize=4.0,
            markevery=max(len(group.index) // 5, 1),
            label=clean_string(group.iloc[0].get("path_label")) or _human_label(path_name),
        )
    target = finite_or_none(data.get("target_price", pd.Series(dtype=float)).dropna().iloc[0] if "target_price" in data.columns and not data.get("target_price").dropna().empty else None)
    if target is not None:
        ax.axhline(float(target), color="#1A1A1A", linestyle="--", linewidth=1.2, alpha=0.72, label="Thesis target")
    date_ticks = data[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    _apply_date_ticks(ax, date_ticks)
    _format_money_axis(ax)
    ax.set_title(title, loc="left", fontsize=15.0, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price ($)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8.1)
    _action_chart_caption(fig, "All lines are endpoint-aware thesis paths: same target, different route. Use this before judging option sensitivity.")
    return _finalize_decision_chart(fig, output_path, bottom=0.20)


def plot_thesis_iv_gallery(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show IV regimes used by thesis mode."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Thesis IV gallery frame is empty.")
    data["requested_days"] = pd.to_numeric(data.get("requested_days"), errors="coerce").fillna(0).astype(int)
    data["iv_shift_points"] = pd.to_numeric(data.get("iv_shift_points"), errors="coerce")
    fig, ax = plt.subplots(figsize=(13.2, 6.6))
    _style_axes(ax, zero_line=True)
    for idx, (iv_name, group) in enumerate(data.groupby("iv_path_name", sort=False)):
        group = group.sort_values("requested_days")
        spec = IV_PATH_VISUAL_SPECS.get(clean_string(iv_name), LONG_CALL_COMPARISON_SPECS[idx % len(LONG_CALL_COMPARISON_SPECS)])
        ax.plot(
            group["requested_days"],
            group["iv_shift_points"],
            color=str(spec["color"]),
            marker=spec.get("marker"),
            linestyle=spec.get("linestyle", "-"),
            linewidth=float(spec.get("linewidth", 2.2)),
            markersize=4.2,
            markevery=max(len(group.index) // 5, 1),
            label=clean_string(group.iloc[0].get("iv_path_label")) or _human_label(iv_name),
        )
    date_ticks = data[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    _apply_date_ticks(ax, date_ticks)
    ax.set_title(title, loc="left", fontsize=15.0, fontweight="bold", pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("IV Shift (vol points)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8.2)
    _action_chart_caption(fig, "These IV regimes are applied to the same thesis stock paths. Lower IV can make a correct stock thesis still fail as an option entry.")
    return _finalize_decision_chart(fig, output_path, bottom=0.20)


def plot_thesis_candidate_overview(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render a compact thesis-mode candidate table."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Thesis candidate overview frame is empty.")
    data["thesis_candidate_rank"] = pd.to_numeric(data.get("thesis_candidate_rank"), errors="coerce").fillna(999)
    data = data.sort_values("thesis_candidate_rank").head(8)
    rows = []
    for row in data.to_dict("records"):
        stock_note = "Stock still cleaner" if bool(row.get("stock_still_better_under_thesis")) else "Option can show edge"
        rows.append(
            {
                "candidate": _compact_tradeoff_label(row.get("candidate_label")),
                "status": _short_thesis_status(row.get("entry_attractiveness_status")),
                "current": _money_text(row.get("current_premium")),
                "justified": _money_text(row.get("max_justified_premium")),
                "gap": _money_text(row.get("premium_gap")),
                "why": clean_string(row.get("main_reason")),
                "stock": stock_note,
                "trust": _human_label(row.get("source_trust_label")),
            }
        )
    line_units = sum(max(_wrapped_line_count(_wrap_chart_text(row["why"], width=36)), 1) for row in rows)
    fig, ax = plt.subplots(figsize=(13.4, max(5.8, 2.0 + line_units * 0.50)))
    _draw_banded_text_table(
        ax,
        title=title,
        rows=rows,
        columns=[
            {"label": "Candidate", "key": "candidate", "x": 0.025, "wrap": 16, "fontsize": 8.7, "fontweight": "bold"},
            {"label": "Status", "key": "status", "x": 0.16, "wrap": 14, "fontsize": 8.2, "fontweight": "bold"},
            {"label": "Current", "key": "current", "x": 0.29, "wrap": 9, "fontsize": 8.0},
            {"label": "Justified", "key": "justified", "x": 0.38, "wrap": 9, "fontsize": 8.0},
            {"label": "Gap", "key": "gap", "x": 0.48, "wrap": 9, "fontsize": 8.0},
            {"label": "Main Reason", "key": "why", "x": 0.57, "wrap": 38, "fontsize": 7.9},
            {"label": "Vs Stock", "key": "stock", "x": 0.84, "wrap": 18, "fontsize": 7.9},
            {"label": "Trust", "key": "trust", "x": 0.94, "wrap": 12, "fontsize": 7.7},
        ],
        caption="Justified premium is thesis-relative and conservative: it pressures the option against the stock benchmark across path and IV families.",
    )
    return _finalize_decision_chart(fig, output_path)


def plot_current_vs_justified_premium(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Compare current option premium with thesis-justified premium."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Current vs justified premium frame is empty.")
    data["thesis_candidate_rank"] = pd.to_numeric(data.get("thesis_candidate_rank"), errors="coerce").fillna(999)
    data = data.sort_values("thesis_candidate_rank").head(10)
    labels = [_compact_tradeoff_label(value) for value in data.get("candidate_label", data.get("candidate_short_label", pd.Series(dtype=str)))]
    y = np.arange(len(data.index))
    current = pd.to_numeric(data.get("current_premium"), errors="coerce").fillna(0.0)
    justified = pd.to_numeric(data.get("max_justified_premium"), errors="coerce").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12.2, max(5.6, len(data.index) * 0.50 + 1.8)))
    _style_axes(ax)
    gap_values = pd.to_numeric(data.get("premium_gap"), errors="coerce").fillna(justified - current)
    justified_colors = ["#009E73" if gap >= 0 else "#D55E00" for gap in gap_values]
    ax.barh(y - 0.18, current, height=0.32, color="#6B7280", alpha=0.68, label="Current premium")
    ax.barh(y + 0.18, justified, height=0.32, color=justified_colors, alpha=0.84, label="Thesis-justified max")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Premium / Contract ($)")
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    for idx, (_, row) in enumerate(data.iterrows()):
        gap = finite_or_none(row.get("premium_gap"))
        gap_label = "room" if (gap is not None and gap >= 0) else "over"
        ax.text(
            max(current.iloc[idx], justified.iloc[idx]) + max(float(current.max()), 1.0) * 0.03,
            idx,
            f"{gap_label} {_money_text(abs(gap) if gap is not None else None)}",
            va="center",
            fontsize=7.8,
            color="#44403C",
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=8.6)
    _action_chart_caption(fig, "If the green bar is shorter than current premium, the thesis still does not justify the entry on this conservative read.")
    return _finalize_decision_chart(fig, output_path, bottom=0.16)


def _stress_candidate_order(frame: pd.DataFrame) -> list[str]:
    data = frame.copy()
    if data.empty:
        return []
    if "stress_rank" in data.columns:
        data["stress_rank"] = pd.to_numeric(data.get("stress_rank"), errors="coerce").fillna(999)
        return data.sort_values("stress_rank").get("candidate_short_label", pd.Series(dtype=str)).astype(str).drop_duplicates().tolist()
    if "candidate_rank" in data.columns:
        data["candidate_rank"] = pd.to_numeric(data.get("candidate_rank"), errors="coerce").fillna(999)
        return data.sort_values("candidate_rank").get("candidate_short_label", pd.Series(dtype=str)).astype(str).drop_duplicates().tolist()
    return data.get("candidate_short_label", pd.Series(dtype=str)).astype(str).drop_duplicates().tolist()


def plot_stress_test_overview(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render the top-candidate stress resilience overview."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Stress transition frame is empty.")
    data["stress_rank"] = pd.to_numeric(data.get("stress_rank"), errors="coerce").fillna(999)
    data["stress_resilience_score"] = pd.to_numeric(data.get("stress_resilience_score"), errors="coerce").fillna(0.0)
    data["base_option_vs_stock_edge_pct"] = pd.to_numeric(data.get("base_option_vs_stock_edge_pct"), errors="coerce").fillna(0.0)
    data = data.sort_values("stress_rank").head(5)
    labels = [_compact_tradeoff_label(value) for value in data.get("candidate_short_label", pd.Series(dtype=str))]
    y = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.4, max(5.4, len(data.index) * 0.68 + 2.0)))
    _style_axes(ax)
    colors = [_action_bucket_visual(bucket)["color"] for bucket in data.get("base_action_bucket", pd.Series(dtype=str))]
    ax.barh(y, data["stress_resilience_score"] * 100.0, color=colors, alpha=0.86, edgecolor="#FFFDF9", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Stress Resilience (% scenarios still Buy/Watch)")
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=12)
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.text(
            min(float(row["stress_resilience_score"]) * 100.0 + 2.0, 98.0),
            idx,
            f"{clean_string(row.get('base_action_bucket'))} | edge {_edge_pct_text(row.get('base_option_vs_stock_edge_pct'))}",
            va="center",
            fontsize=8.2,
            color="#292524",
        )
    _action_chart_caption(fig, "Higher resilience means the candidate survives more premium, timing, and target stresses without falling to Avoid/Prefer Stock.")
    return _finalize_decision_chart(fig, output_path, bottom=0.15)


def _plot_stress_line_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    caption: str,
    x_column: str,
    x_label: str,
    scenario_order_column: str = "scenario_order",
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Stress summary frame is empty.")
    data["option_vs_stock_edge_pct"] = pd.to_numeric(data.get("option_vs_stock_edge_pct"), errors="coerce")
    data = data.dropna(subset=["option_vs_stock_edge_pct"])
    if data.empty:
        raise ValueError("Stress summary frame has no plottable edge values.")
    data[scenario_order_column] = pd.to_numeric(data.get(scenario_order_column), errors="coerce").fillna(999)
    order_frame = data.sort_values(scenario_order_column).drop_duplicates(subset=["scenario_label"])
    scenario_labels = order_frame["scenario_label"].astype(str).tolist()
    x_positions = np.arange(len(scenario_labels))
    scenario_to_x = {label: idx for idx, label in enumerate(scenario_labels)}
    fig, ax = plt.subplots(figsize=(12.4, 6.1))
    _style_axes(ax, zero_line=True)
    for idx, label in enumerate(_stress_candidate_order(data)[:5]):
        group = data.loc[data.get("candidate_short_label").astype(str).eq(label)].copy()
        group = group.sort_values(scenario_order_column)
        xs = [scenario_to_x.get(clean_string(value), 0) for value in group.get("scenario_label", pd.Series(dtype=str))]
        spec = LONG_CALL_COMPARISON_SPECS[idx % len(LONG_CALL_COMPARISON_SPECS)]
        ax.plot(
            xs,
            group["option_vs_stock_edge_pct"],
            label=_compact_tradeoff_label(label),
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            linestyle=spec.get("linestyle", "-"),
            linewidth=2.5 if idx == 0 else 2.1,
            markersize=5.4,
            markeredgecolor="#FFFFFF",
            markeredgewidth=0.8,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_wrap_chart_text(label, width=13) for label in scenario_labels], rotation=0, ha="center")
    ax.set_ylabel("Option Edge Vs Stock (%)")
    ax.set_xlabel(x_label)
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=12)
    _place_legend(ax, ncol=3, fontsize=8)
    _action_chart_caption(fig, caption)
    return _finalize_decision_chart(fig, output_path, bottom=0.25)


def plot_premium_sensitivity_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot whether lower entry premium upgrades top calls."""

    return _plot_stress_line_chart(
        frame,
        output_path=output_path,
        title=title,
        caption="This isolates entry price. If the line only turns attractive at -10%/-20%, the idea may be watchlist rather than buy-now.",
        x_column="premium_multiplier",
        x_label="Premium Stress",
    )


def plot_timing_slip_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot how later thesis timing changes option-vs-stock edge."""

    return _plot_stress_line_chart(
        frame,
        output_path=output_path,
        title=title,
        caption="This keeps the broad thesis but delays the move. Steeper declines indicate theta/timing fragility.",
        x_column="delay_days",
        x_label="Timing Stress",
    )


def plot_target_stress_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot whether the call needs the stock to overshoot the thesis."""

    return _plot_stress_line_chart(
        frame,
        output_path=output_path,
        title=title,
        caption="This varies only the endpoint level. If edge appears only in overshoot, the call needs more than the base thesis.",
        x_column="target_price",
        x_label="Target Stress",
    )


def plot_top_candidate_stress_cards(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render compact cards summarizing best/worst stress for each top call."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Stress transition frame is empty.")
    data["stress_rank"] = pd.to_numeric(data.get("stress_rank"), errors="coerce").fillna(999)
    data = data.sort_values("stress_rank").head(5).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in data.to_dict("records"):
        rows.append(
            {
                "candidate": _compact_tradeoff_label(row.get("candidate_short_label") or row.get("candidate_label")),
                "bucket": clean_string(row.get("base_action_bucket")),
                "base": f"{clean_string(row.get('base_action_bucket'))}\n{_edge_pct_text(row.get('base_option_vs_stock_edge_pct'))} edge",
                "best": _wrap_chart_text(
                    f"{clean_string(row.get('best_improving_stress'))}: {clean_string(row.get('best_improving_bucket'))} ({_edge_pct_text(row.get('best_improving_edge_pct'))})",
                    width=28,
                ),
                "worst": _wrap_chart_text(
                    f"{clean_string(row.get('worst_breaking_stress'))}: {clean_string(row.get('worst_breaking_bucket'))} ({_edge_pct_text(row.get('worst_breaking_edge_pct'))})",
                    width=28,
                ),
                "why": _wrap_chart_text(row.get("stress_card_note") or row.get("main_warning"), width=38),
                "trust": _human_label(row.get("source_trust_label")),
            }
        )
    line_units = sum(
        max(
            _wrapped_line_count(row["best"]),
            _wrapped_line_count(row["worst"]),
            _wrapped_line_count(row["why"]),
        )
        for row in rows
    )
    fig, ax = plt.subplots(figsize=(13.2, max(5.8, 2.0 + line_units * 0.52)))
    _draw_banded_text_table(
        ax,
        title=title,
        rows=rows,
        columns=[
            {"label": "Candidate", "key": "candidate", "x": 0.025, "wrap": 14, "fontsize": 8.8, "fontweight": "bold"},
            {"label": "Base", "key": "base", "x": 0.17, "wrap": 15, "fontsize": 8.1, "fontweight": "bold", "bucket_color_from": "bucket"},
            {"label": "Best Improving Stress", "key": "best", "x": 0.31, "wrap": 30, "fontsize": 7.9},
            {"label": "Worst Breaking Stress", "key": "worst", "x": 0.52, "wrap": 30, "fontsize": 7.9},
            {"label": "Decision Read", "key": "why", "x": 0.73, "wrap": 38, "fontsize": 7.8},
            {"label": "Trust", "key": "trust", "x": 0.94, "wrap": 11, "fontsize": 7.6},
        ],
        caption="Each card compares the same candidate across premium, timing, and target stresses. Use it to separate 'bad option' from 'bad entry'.",
        marker_bucket_key="bucket",
    )
    return _finalize_decision_chart(fig, output_path, bottom=0.13)


def plot_single_option_decision_view(
    *,
    summary: pd.DataFrame,
    representative_paths: pd.DataFrame,
    path_outcomes: pd.DataFrame,
    iv_sensitivity: pd.DataFrame,
    entry_sensitivity: pd.DataFrame,
    summary_bullets: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render the single-option path-first decision view from frozen analysis artifacts."""

    if summary.empty or representative_paths.empty or path_outcomes.empty:
        raise ValueError("Single-option decision frames are incomplete.")
    summary_row = summary.iloc[0].to_dict()
    paths = representative_paths.copy()
    outcomes = path_outcomes.copy()
    paths["requested_days"] = pd.to_numeric(paths.get("requested_days"), errors="coerce").fillna(0)
    paths["spot_price"] = pd.to_numeric(paths.get("spot_price"), errors="coerce")
    outcomes = outcomes.drop_duplicates(subset=["path_role"]).copy()
    outcome_lookup = {
        clean_string(row.get("path_role")): row
        for row in outcomes.to_dict("records")
    }
    paths = paths.loc[paths.get("path_role", pd.Series(dtype=str)).astype(str).isin(outcome_lookup.keys())].copy()
    if paths.empty:
        raise ValueError("Single-option decision representative paths have no matching outcomes.")

    fig = plt.figure(figsize=(15.6, 10.6))
    fig.patch.set_facecolor(_FIGURE_FACE)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=3,
        height_ratios=[0.72, 4.25, 1.9],
        width_ratios=[1.35, 1.35, 1.0],
        hspace=0.38,
        wspace=0.28,
    )
    ax_strip = fig.add_subplot(grid[0, :])
    ax_hero = fig.add_subplot(grid[1, :2])
    ax_bullets = fig.add_subplot(grid[1, 2])
    ax_iv = fig.add_subplot(grid[2, 0])
    ax_entry = fig.add_subplot(grid[2, 1])
    ax_note = fig.add_subplot(grid[2, 2])

    ax_strip.axis("off")
    ax_strip.set_facecolor(_FIGURE_FACE)
    premium_value = finite_or_none(summary_row.get("premium_used"))
    base_iv_value = finite_or_none(summary_row.get("base_iv"))
    breakeven_value = finite_or_none(summary_row.get("breakeven"))
    max_loss_value = finite_or_none(summary_row.get("max_loss"))
    strip_items = [
        ("Ticker", summary_row.get("ticker")),
        ("Contract", summary_row.get("candidate_short_label") or summary_row.get("candidate_label")),
        ("Premium Used", f"${float(premium_value or 0):,.2f}"),
        ("IV", f"{float(base_iv_value) * 100:,.1f}%" if base_iv_value is not None else "n/a"),
        ("Breakeven", f"${float(breakeven_value):,.2f}" if breakeven_value is not None else "n/a"),
        ("Max Loss", f"${float(max_loss_value or 0):,.0f}"),
        ("DTE", clean_string(summary_row.get("dte")) or "n/a"),
        ("Exit Rule", _human_label(summary_row.get("exit_rule"))),
    ]
    left = 0.015
    cell_w = 0.122
    for idx, (label, value) in enumerate(strip_items):
        x = left + idx * cell_w
        ax_strip.add_patch(
            Rectangle((x, 0.12), cell_w - 0.008, 0.76, transform=ax_strip.transAxes, facecolor="#F4F1EA", edgecolor="#DED8CE", linewidth=0.8)
        )
        ax_strip.text(x + 0.012, 0.68, label, transform=ax_strip.transAxes, fontsize=7.8, fontweight="bold", color="#57534E", va="center")
        ax_strip.text(x + 0.012, 0.36, clean_string(value), transform=ax_strip.transAxes, fontsize=9.4, fontweight="bold", color="#292524", va="center")

    _style_axes(ax_hero)
    terminal_labels: list[dict[str, object]] = []
    for display_idx, (path_role, group) in enumerate(paths.sort_values(["display_order", "requested_days"]).groupby("path_role", sort=False), start=1):
        outcome = outcome_lookup.get(clean_string(path_role), {})
        outcome_label = clean_string(outcome.get("outcome_label")) or "fail_too_narrow_or_expiry_issue"
        spec = SINGLE_OPTION_OUTCOME_SPECS.get(outcome_label, SINGLE_OPTION_OUTCOME_SPECS["fail_too_narrow_or_expiry_issue"])
        label = clean_string(outcome.get("path_label")) or clean_string(group.iloc[0].get("path_label"))
        ax_hero.plot(
            group["requested_days"],
            group["spot_price"],
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            linestyle=spec["linestyle"],
            linewidth=3.0 if outcome_label == "clear_option_win" else 2.3,
            markersize=5.2,
            markevery=max(1, len(group.index) // 4),
            label=f"{label} - {spec['label']}",
            alpha=0.94,
        )
        terminal = group.sort_values("requested_days").iloc[-1]
        terminal_labels.append(
            {
                "x": float(terminal.get("requested_days") or 0),
                "y": float(terminal.get("spot_price") or 0),
                "label": clean_string(label).replace(" To Target", "").replace(" Path", ""),
                "color": str(spec["color"]),
            }
        )
    y_values = pd.to_numeric(paths["spot_price"], errors="coerce").dropna()
    y_min = float(y_values.min()) if not y_values.empty else 0.0
    y_max = float(y_values.max()) if not y_values.empty else 1.0
    y_span = max(y_max - y_min, 1.0)
    label_gap = y_span * 0.055
    placed_labels: list[tuple[float, float]] = []
    for item in sorted(terminal_labels, key=lambda row: (float(row["x"]), float(row["y"]))):
        base_x = float(item["x"])
        base_y = float(item["y"])
        candidate_offsets = [0.0]
        for idx in range(1, 7):
            candidate_offsets.extend([idx * label_gap, -idx * label_gap])
        label_y = base_y
        for offset in candidate_offsets:
            proposed_y = min(max(base_y + offset, y_min - y_span * 0.08), y_max + y_span * 0.08)
            collision = any(
                abs(proposed_y - used_y) < label_gap and abs(base_x - used_x) < 45.0
                for used_x, used_y in placed_labels
            )
            if not collision:
                label_y = proposed_y
                break
        placed_labels.append((base_x, label_y))
        ax_hero.text(
            base_x + 2.0,
            label_y,
            _wrap_chart_text(clean_string(item["label"]), width=24),
            fontsize=7.4,
            color=str(item["color"]),
            va="center",
            linespacing=1.05,
            clip_on=False,
        )
    ax_hero.set_title("What stock paths make this option worth buying?", loc="left", fontsize=15.6, fontweight="bold", pad=12)
    ax_hero.set_xlabel("Time From Snapshot")
    ax_hero.set_ylabel("Stock Price ($)")
    _format_money_axis(ax_hero)
    _apply_date_ticks(ax_hero, paths[["date", "requested_days"]].drop_duplicates())

    ax_bullets.axis("off")
    ax_bullets.set_facecolor(_AXES_FACE)
    ax_bullets.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, transform=ax_bullets.transAxes, facecolor="#FFFDF9", edgecolor="#DED8CE", linewidth=0.9))
    ax_bullets.text(0.05, 0.94, "Decision Read", transform=ax_bullets.transAxes, fontsize=13.2, fontweight="bold", color="#292524", va="top")
    bullets = []
    if not summary_bullets.empty:
        bullets = [
            clean_string(row.get("bullet_text"))
            for row in summary_bullets.sort_values("bullet_order").to_dict("records")
            if clean_string(row.get("bullet_text"))
        ][:7]
    y = 0.84
    for bullet in bullets:
        wrapped = _wrap_chart_text(bullet, width=34)
        ax_bullets.text(0.06, y, f"- {wrapped}", transform=ax_bullets.transAxes, fontsize=8.8, color="#44403C", va="top", linespacing=1.18)
        y -= 0.055 * max(1, _wrapped_line_count(wrapped)) + 0.030

    def _bar_panel(ax, frame: pd.DataFrame, *, x_label: str, label_col: str, value_col: str, title_text: str) -> None:
        ax.set_facecolor(_AXES_FACE)
        _style_axes(ax, zero_line=True)
        data = frame.copy()
        if data.empty:
            ax.axis("off")
            ax.text(0.03, 0.72, "No sensitivity rows available.", transform=ax.transAxes, fontsize=9.0, color="#57534E")
            return
        data["display_order"] = pd.to_numeric(data.get("display_order"), errors="coerce").fillna(99)
        data[value_col] = pd.to_numeric(data.get(value_col), errors="coerce").fillna(0.0)
        data = data.sort_values("display_order").head(5)
        y_pos = np.arange(len(data.index))
        colors = ["#009E73" if value > 0 else "#D55E00" if value < 0 else "#7F7F7F" for value in data[value_col]]
        ax.barh(y_pos, data[value_col], color=colors, alpha=0.82)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([_wrap_chart_text(value, width=15) for value in data.get(label_col, pd.Series(dtype=str))], fontsize=8.2)
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_title(title_text, loc="left", fontsize=11.8, fontweight="bold", pad=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))

    _bar_panel(
        ax_iv,
        iv_sensitivity,
        x_label="Difference Vs Stock ($)",
        label_col="iv_mode_label",
        value_col="difference_vs_stock",
        title_text="IV Sensitivity",
    )
    _bar_panel(
        ax_entry,
        entry_sensitivity,
        x_label="Avg Difference Vs Stock ($)",
        label_col="entry_scenario_label",
        value_col="average_difference_vs_stock",
        title_text="Entry Premium Sensitivity",
    )

    ax_note.axis("off")
    ax_note.set_facecolor(_FIGURE_FACE)
    outcome_lines = [
        ("Clear option win", "solid circle"),
        ("Wins, not enough", "dashed triangle"),
        ("Stock better", "solid square"),
        ("Fail / too narrow", "dotted x"),
    ]
    ax_note.text(0.02, 0.94, "Outcome Encoding", transform=ax_note.transAxes, fontsize=11.8, fontweight="bold", color="#292524", va="top")
    y_note = 0.78
    for label, marker_text in outcome_lines:
        key = next((name for name, spec in SINGLE_OPTION_OUTCOME_SPECS.items() if spec["label"] == label), "")
        spec = SINGLE_OPTION_OUTCOME_SPECS.get(key, SINGLE_OPTION_OUTCOME_SPECS["fail_too_narrow_or_expiry_issue"])
        ax_note.plot([0.04, 0.12], [y_note, y_note], transform=ax_note.transAxes, color=spec["color"], linestyle=spec["linestyle"], linewidth=2.4)
        ax_note.scatter([0.08], [y_note], transform=ax_note.transAxes, color=spec["color"], marker=spec["marker"], s=48)
        ax_note.text(0.17, y_note, f"{label} ({marker_text})", transform=ax_note.transAxes, fontsize=8.5, color="#44403C", va="center")
        y_note -= 0.14

    caption = (
        "Fixed: selected option, stock benchmark, exit rule, and thresholds are frozen in analysis. "
        "The hero chart shows stock paths only; IV and premium are isolated below so the path question stays readable."
    )
    _action_chart_caption(fig, caption)
    return _finalize_caption_chart(fig, output_path, bottom=0.11, top=0.98)


def plot_thesis_path_vs_value(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show terminal option value by thesis stock path for the top candidate set."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Thesis mode candidate frame is empty.")
    candidates = list(data.get("candidate_short_label", pd.Series(dtype=str)).drop_duplicates().head(3))
    data = data.loc[data.get("candidate_short_label", pd.Series(dtype=str)).isin(candidates)].copy()
    grouped = data.groupby(["path_label", "candidate_short_label"], dropna=False)["thesis_terminal_value"].median().reset_index()
    pivot = grouped.pivot(index="path_label", columns="candidate_short_label", values="thesis_terminal_value").fillna(0.0)
    fig, ax = plt.subplots(figsize=(13.4, max(6.0, len(pivot.index) * 0.46 + 2.0)))
    _style_axes(ax)
    y = np.arange(len(pivot.index))
    width = 0.24 if len(pivot.columns) >= 3 else 0.32
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        spec = LONG_CALL_COMPARISON_SPECS[idx % len(LONG_CALL_COMPARISON_SPECS)]
        ax.barh(y + offset, pivot[column], height=width * 0.88, color=str(spec["color"]), alpha=0.82, label=clean_string(column))
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Median Terminal Option Value ($)")
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(3, len(pivot.columns)), frameon=False, fontsize=8.4)
    _action_chart_caption(fig, "Same endpoint, different route. Faster or earlier target paths can justify more option premium than slow or delayed paths.")
    return _finalize_decision_chart(fig, output_path, bottom=0.16)


def plot_thesis_iv_vs_value(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show terminal option value by IV regime for the top candidate set."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Thesis mode candidate frame is empty.")
    candidates = list(data.get("candidate_short_label", pd.Series(dtype=str)).drop_duplicates().head(3))
    data = data.loc[data.get("candidate_short_label", pd.Series(dtype=str)).isin(candidates)].copy()
    grouped = data.groupby(["iv_path_label", "candidate_short_label"], dropna=False)["thesis_terminal_value"].median().reset_index()
    pivot = grouped.pivot(index="iv_path_label", columns="candidate_short_label", values="thesis_terminal_value").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12.6, max(5.8, len(pivot.index) * 0.52 + 1.8)))
    _style_axes(ax)
    y = np.arange(len(pivot.index))
    width = 0.24 if len(pivot.columns) >= 3 else 0.32
    for idx, column in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2.0) * width
        spec = LONG_CALL_COMPARISON_SPECS[idx % len(LONG_CALL_COMPARISON_SPECS)]
        ax.barh(y + offset, pivot[column], height=width * 0.88, color=str(spec["color"]), alpha=0.82, label=clean_string(column))
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.invert_yaxis()
    ax.set_xlabel("Median Terminal Option Value ($)")
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=min(3, len(pivot.columns)), frameon=False, fontsize=8.4)
    _action_chart_caption(fig, "Same stock thesis, different IV regime. This isolates whether IV support is needed for the entry to make sense.")
    return _finalize_decision_chart(fig, output_path, bottom=0.16)


def plot_thesis_stock_vs_option(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show whether thesis-mode candidates beat the long-stock benchmark."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Thesis stock-vs-option frame is empty.")
    data["thesis_candidate_rank"] = pd.to_numeric(data.get("thesis_candidate_rank"), errors="coerce").fillna(999)
    data = data.sort_values("thesis_candidate_rank").head(10)
    y = np.arange(len(data.index))
    labels = [_compact_tradeoff_label(value) for value in data.get("candidate_short_label", data.get("candidate_label", pd.Series(dtype=str)))]
    values = pd.to_numeric(data.get("difference_vs_stock_median"), errors="coerce").fillna(0.0)
    colors = ["#009E73" if value > 0 else "#D55E00" for value in values]
    fig, ax = plt.subplots(figsize=(12.0, max(5.8, len(data.index) * 0.48 + 1.8)))
    _style_axes(ax, zero_line=True)
    ax.barh(y, values, color=colors, alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Median Difference Vs Long Stock ($)")
    ax.set_title(title, loc="left", fontsize=14.8, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}"))
    span = max(float(np.nanmax(np.abs(values.to_numpy(dtype=float)))) if len(values.index) else 0.0, 1.0)
    label_offset = span * 0.025
    for idx, row in enumerate(data.to_dict("records")):
        note = "stock cleaner" if bool(row.get("stock_still_better_under_thesis")) else "option edge"
        value = float(values.iloc[idx])
        if value >= 0:
            label_x = value - label_offset if value > label_offset * 2 else value + label_offset
            label_align = "right" if value > label_offset * 2 else "left"
        else:
            label_x = value + label_offset
            label_align = "left"
        ax.text(label_x, idx, note, va="center", ha=label_align, fontsize=7.9, color="#44403C", clip_on=True)
    ax.margins(x=0.08)
    _action_chart_caption(fig, "Negative means the option still trails stock in the median thesis scenario even when the target endpoint is reached.")
    return _finalize_decision_chart(fig, output_path)


def plot_required_stock_path_to_buy(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show the minimum stock path versus the active assumed path for a small call set."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Required stock path frame is empty.")
    data["requested_days"] = pd.to_numeric(data.get("requested_days"), errors="coerce").fillna(0).astype(int)
    data["entry_display_rank"] = pd.to_numeric(data.get("entry_display_rank"), errors="coerce").fillna(999).astype(int)
    ordered_candidates = (
        data[["candidate_slug", "candidate_short_label", "entry_display_rank", "action_bucket", "entry_barrier_label"]]
        .drop_duplicates()
        .sort_values(["entry_display_rank", "candidate_short_label"])
        .head(4)
    )
    if ordered_candidates.empty:
        raise ValueError("Required stock path frame is empty.")

    candidate_rows = ordered_candidates.to_dict("records")
    panel_count = len(candidate_rows)
    cols = 2 if panel_count > 1 else 1
    rows = math.ceil(panel_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.8, max(4.9, rows * 3.45)), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).flatten()
    overall_prices = pd.to_numeric(data.get("stock_price"), errors="coerce").dropna()
    lower = float(overall_prices.min()) if not overall_prices.empty else 0.0
    upper = float(overall_prices.max()) if not overall_prices.empty else 1.0
    pad = max((upper - lower) * 0.08, 0.35)
    date_ticks = data[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    legend_handles = []
    legend_labels = []

    for axis, meta in zip(axes_array, candidate_rows):
        subset = data.loc[data.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(clean_string(meta.get("candidate_slug")))].copy()
        subset = subset.sort_values("requested_days")
        required = subset.loc[subset.get("series_kind", pd.Series(dtype=str)).astype(str).eq("required_path")]
        assumed = subset.loc[subset.get("series_kind", pd.Series(dtype=str)).astype(str).eq("assumed_path")]
        visual = _action_bucket_visual(meta.get("action_bucket"))
        _style_axes(axis)
        if not required.empty:
            handle = axis.plot(
                required["requested_days"],
                pd.to_numeric(required.get("stock_price"), errors="coerce"),
                color=visual["color"],
                marker=visual["marker"],
                linewidth=2.8,
                markersize=5.6,
                label="Required path to justify entry",
                zorder=4,
            )[0]
            legend_handles.append(handle)
            legend_labels.append("Required path to justify entry")
        if not assumed.empty:
            handle = axis.plot(
                assumed["requested_days"],
                pd.to_numeric(assumed.get("stock_price"), errors="coerce"),
                color="#3F3F46",
                linestyle="--",
                linewidth=1.9,
                label="Active assumed stock path",
                zorder=3,
            )[0]
            legend_handles.append(handle)
            legend_labels.append("Active assumed stock path")
        if not required.empty and not assumed.empty:
            required_points = required[["requested_days", "stock_price"]].copy()
            assumed_points = assumed[["requested_days", "stock_price"]].copy()
            required_points["stock_price"] = pd.to_numeric(required_points["stock_price"], errors="coerce")
            assumed_points["stock_price"] = pd.to_numeric(assumed_points["stock_price"], errors="coerce")
            merged = required_points.merge(assumed_points, on="requested_days", suffixes=("_required", "_assumed")).dropna()
            if not merged.empty:
                axis.fill_between(
                    merged["requested_days"],
                    merged["stock_price_assumed"],
                    merged["stock_price_required"],
                    color=str(visual["color"]),
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
        barrier_label = clean_string(meta.get("entry_barrier_label")).replace("_", " ").title() or "Entry barrier"
        axis.set_title(
            f"{clean_string(meta.get('candidate_short_label'))} — {barrier_label}",
            loc="left",
            fontsize=11.0,
            fontweight="bold",
            pad=8,
        )
        axis.set_ylim(lower - pad, upper + pad)
        axis.set_ylabel("Stock Price ($)")
        _format_money_axis(axis)
        _apply_date_ticks(axis, date_ticks)

    for axis in axes_array[panel_count:]:
        axis.axis("off")
    for axis in axes_array[max(panel_count - cols, 0):panel_count]:
        axis.set_xlabel("Date")

    fig.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=15.0, fontweight="bold")
    dedup = dict(zip(legend_labels, legend_handles))
    if dedup:
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            ncol=2,
            frameon=False,
            fontsize=8.8,
        )
    active_iv_label = clean_string(data.get("iv_path_label", pd.Series(dtype=str)).dropna().iloc[0] if "iv_path_label" in data.columns and not data.get("iv_path_label").dropna().empty else "")
    caption = (
        f"Fixed: IV path = {active_iv_label or 'active assumed IV'} | "
        "colored path = required stock route | dashed gray = active assumed stock path | shaded gap = hurdle difference"
    )
    _action_chart_caption(fig, caption)
    return _finalize_caption_chart(fig, output_path, bottom=0.13, top=0.93)


def plot_required_move_speed_vs_magnitude(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Scatter required upside magnitude against time available for the move."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Required move summary frame is empty.")
    data["timing_window_days"] = pd.to_numeric(data.get("timing_window_days"), errors="coerce").fillna(0.0)
    data["required_move_pct_target"] = pd.to_numeric(data.get("required_move_pct_target"), errors="coerce").fillna(0.0)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data = data.sort_values(["action_priority_rank", "required_move_pct_target"]).head(12).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.4, 7.1))
    _style_axes(ax)
    max_x = max(float(data["timing_window_days"].max()), 1.0)
    max_y = max(float(data["required_move_pct_target"].max()), 1.0)
    forgiving_x = max_x * 0.45
    demanding_y = max_y * 0.55
    ax.axvspan(0, forgiving_x, ymin=0, ymax=min(demanding_y / (max_y * 1.10), 1.0), color="#56B4E9", alpha=0.07, zorder=0)
    ax.axhspan(demanding_y, max_y * 1.10, color="#D55E00", alpha=0.055, zorder=0)
    ax.text(forgiving_x * 0.10, demanding_y * 0.30, "more forgiving", fontsize=8.4, color="#155E75", fontweight="bold")
    ax.text(max_x * 0.70, max_y * 0.96, "very demanding", fontsize=8.4, color="#9A3412", fontweight="bold")
    seen_buckets: set[str] = set()
    for idx, row in data.iterrows():
        bucket = clean_string(row.get("action_bucket"))
        visual = _action_bucket_visual(bucket)
        label = visual["label"] if bucket not in seen_buckets else None
        seen_buckets.add(bucket)
        edge = "#1A1A1A" if bool(row.get("stock_still_better_even_if_path_hits")) else "#FFFFFF"
        ax.scatter(
            float(row["timing_window_days"]),
            float(row["required_move_pct_target"]),
            s=145,
            color=visual["color"],
            marker=visual["marker"],
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.92,
            label=label,
            zorder=4,
        )
        offset_x, offset_y = [(8, 7), (8, -11), (-38, 10), (-40, -9), (10, 15)][idx % 5]
        ax.annotate(
            clean_string(row.get("candidate_short_label")) or _compact_tradeoff_label(row.get("candidate_label")),
            xy=(float(row["timing_window_days"]), float(row["required_move_pct_target"])),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7.7,
            color="#292524",
            ha="right" if offset_x < 0 else "left",
            va="center",
        )
    if not data.empty:
        ax.axvline(float(data["timing_window_days"].median()), color="#A8A29E", linestyle=":", linewidth=1.0)
        ax.axhline(float(data["required_move_pct_target"].median()), color="#A8A29E", linestyle=":", linewidth=1.0)
    ax.set_xlim(min(0, float(data["timing_window_days"].min()) - max_x * 0.05), max_x * 1.10)
    ax.set_ylim(min(0, float(data["required_move_pct_target"].min()) - max_y * 0.08), max_y * 1.10)
    ax.set_xlabel("Time Window To Clear The Required Path (days)")
    ax.set_ylabel("Required Upside By Key Decision Date (%)")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=4, frameon=False, fontsize=8.4)
    _action_chart_caption(
        fig,
        "Farther right means more time. Higher means a larger required move. Dark outlines mark calls where stock still looks cleaner even if the required path is achieved.",
    )
    return _finalize_decision_chart(fig, output_path)


def plot_required_move_vs_stock_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show whether stock still dominates even if the required move is achieved."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Required move vs stock frame is empty.")
    data["difference_vs_stock"] = pd.to_numeric(data.get("difference_vs_stock"), errors="coerce").fillna(0.0)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data["stock_still_better_even_if_path_hits"] = data.get("stock_still_better_even_if_path_hits", pd.Series(dtype=bool)).fillna(False).astype(bool)
    data = data.sort_values(
        ["stock_still_better_even_if_path_hits", "difference_vs_stock", "action_priority_rank"],
        ascending=[False, True, True],
    ).head(10)
    labels = [clean_string(value) or _compact_tradeoff_label(value) for value in data.get("candidate_short_label", pd.Series(dtype=str))]
    y_pos = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.4, max(5.8, 1.6 + len(data.index) * 0.48)))
    _style_axes(ax, zero_line=True)
    colors = [_action_bucket_visual(bucket)["color"] for bucket in data.get("action_bucket", pd.Series(dtype=str))]
    edges = ["#1A1A1A" if value else "#FFFFFF" for value in data["stock_still_better_even_if_path_hits"]]
    ax.barh(y_pos, data["difference_vs_stock"], color=colors, edgecolor=edges, linewidth=1.2, alpha=0.86)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Modeled Edge Vs Long Stock If The Required Path Is Met ($)")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    _format_money_axis(ax)
    for idx, row in enumerate(data.to_dict("records")):
        note = "stock still better" if bool(row.get("stock_still_better_even_if_path_hits")) or _num_safe(row.get("difference_vs_stock")) <= 0 else "call edge"
        x_value = _num_safe(row.get("difference_vs_stock"))
        ax.text(
            x_value + (0.35 if x_value >= 0 else -0.35),
            idx,
            note,
            ha="left" if x_value >= 0 else "right",
            va="center",
            fontsize=7.8,
            color="#44403C",
        )
    _action_chart_caption(
        fig,
        "Left of zero means stock still looks cleaner even when the required call path is broadly achieved. Black outlines highlight those stock-dominant rows.",
    )
    return _finalize_decision_chart(fig, output_path)


def plot_strike_expiry_entry_barrier_map(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Render a compact barrier map across strike and expiry choices."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Entry barrier summary frame is empty.")
    data["entry_barrier_score"] = pd.to_numeric(data.get("entry_barrier_score"), errors="coerce").fillna(0.0)
    data["required_move_pct_target"] = pd.to_numeric(data.get("required_move_pct_target"), errors="coerce").fillna(0.0)
    data["timing_window_days"] = pd.to_numeric(data.get("timing_window_days"), errors="coerce").fillna(0.0)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data = data.sort_values(["entry_barrier_score", "action_priority_rank"]).head(10).reset_index(drop=True)
    rows = [
        {
            "bucket": clean_string(row.get("action_bucket")),
            "candidate": clean_string(row.get("candidate_short_label")) or _compact_tradeoff_label(row.get("candidate_label")),
            "expiry": _short_expiry_label(row.get("expiry_date")),
            "strike": _compact_strike_call_label(row.get("strike_label")),
            "barrier": clean_string(row.get("entry_barrier_label")).replace("_", " ").title(),
            "move": f"{_num_safe(row.get('required_move_pct_target')):+.1f}%",
            "timing": f"{int(round(_num_safe(row.get('timing_window_days'))))}d",
            "iv_note": clean_string(row.get("iv_requirement_label")).replace("_", " ").title() or "IV secondary",
            "trust": _human_label(clean_string(row.get("source_trust_label")) or "unknown_trust"),
        }
        for row in data.to_dict("records")
    ]
    line_units = sum(max(_wrapped_line_count(_wrap_chart_text(row["candidate"], width=18)), _wrapped_line_count(_wrap_chart_text(row["iv_note"], width=18)), 1) for row in rows)
    fig_height = max(5.6, min(11.8, 1.8 + line_units * 0.36))
    fig, ax = plt.subplots(figsize=(13.6, fig_height))
    _draw_banded_text_table(
        ax,
        title=title,
        rows=rows,
        columns=[
            {"label": "Candidate", "key": "candidate", "x": 0.03, "wrap": 18, "fontsize": 8.5, "fontweight": "bold"},
            {"label": "Expiry", "key": "expiry", "x": 0.24, "wrap": 10, "fontsize": 8.1},
            {"label": "Strike", "key": "strike", "x": 0.33, "wrap": 10, "fontsize": 8.1},
            {"label": "Barrier", "key": "barrier", "x": 0.42, "wrap": 16, "fontsize": 8.1, "fontweight": "bold", "bucket_color_from": "bucket"},
            {"label": "Req. Move", "key": "move", "x": 0.58, "wrap": 10, "fontsize": 8.0},
            {"label": "Timing", "key": "timing", "x": 0.67, "wrap": 10, "fontsize": 8.0},
            {"label": "IV Read", "key": "iv_note", "x": 0.75, "wrap": 18, "fontsize": 7.9},
            {"label": "Trust", "key": "trust", "x": 0.92, "wrap": 12, "fontsize": 7.9},
        ],
        caption="Lower barrier means the call asks less of the stock path. Higher barrier means it needs a bigger, faster, or more IV-friendly move to feel justified.",
        marker_bucket_key="bucket",
    )
    return _finalize_decision_chart(fig, output_path)


def plot_iv_support_requirement_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show which calls get materially harder when IV cools and which improve with friendlier IV."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Required IV support summary frame is empty.")
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data["lower_iv_move_penalty_pct"] = pd.to_numeric(data.get("lower_iv_move_penalty_pct"), errors="coerce").fillna(0.0)
    data["higher_iv_move_relief_pct"] = pd.to_numeric(data.get("higher_iv_move_relief_pct"), errors="coerce").fillna(0.0)
    data = data.sort_values(["lower_iv_move_penalty_pct", "action_priority_rank"], ascending=[False, True]).head(10).reset_index(drop=True)
    y_pos = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.6, max(5.4, 1.3 + len(data.index) * 0.46)))
    _style_axes(ax, zero_line=True)
    ax.barh(
        y_pos - 0.16,
        data["lower_iv_move_penalty_pct"],
        height=0.28,
        color="#D55E00",
        alpha=0.88,
        label="Extra upside needed if IV cools",
    )
    ax.barh(
        y_pos + 0.16,
        -data["higher_iv_move_relief_pct"],
        height=0.28,
        color="#0072B2",
        alpha=0.85,
        label="Upside requirement relieved if IV stays firmer",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([clean_string(value) or _compact_tradeoff_label(value) for value in data.get("candidate_short_label", pd.Series(dtype=str))])
    ax.invert_yaxis()
    ax.set_xlabel("Change In Required Target Move (%) Versus Flat IV")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    for idx, row in enumerate(data.to_dict("records")):
        ax.text(
            max(_num_safe(row.get("lower_iv_move_penalty_pct")), 0.0) + 0.45,
            idx - 0.16,
            clean_string(row.get("iv_requirement_label")).replace("_", " ").title(),
            va="center",
            fontsize=7.7,
            color="#44403C",
        )
    _place_legend(ax, ncol=2, fontsize=8.5)
    _action_chart_caption(
        fig,
        "Orange to the right means the call needs a meaningfully stronger stock move if IV cools. Blue to the left shows how much easier the required move gets if IV stays friendlier.",
    )
    return _finalize_decision_chart(fig, output_path)


def _plot_required_stock_path_to_buy_refined(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Required stock path frame is empty.")
    data["requested_days"] = pd.to_numeric(data.get("requested_days"), errors="coerce").fillna(0).astype(int)
    data["entry_display_rank"] = pd.to_numeric(data.get("entry_display_rank"), errors="coerce").fillna(999).astype(int)
    ordered_candidates = (
        data[["candidate_slug", "candidate_short_label", "entry_display_rank", "action_bucket", "entry_barrier_label"]]
        .drop_duplicates()
        .sort_values(["entry_display_rank", "candidate_short_label"])
        .head(4)
    )
    if ordered_candidates.empty:
        raise ValueError("Required stock path frame is empty.")

    candidate_rows = ordered_candidates.to_dict("records")
    panel_count = len(candidate_rows)
    cols = 2 if panel_count > 1 else 1
    rows = math.ceil(panel_count / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.8, max(4.9, rows * 3.45)), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes).flatten()
    overall_prices = pd.to_numeric(data.get("stock_price"), errors="coerce").dropna()
    lower = float(overall_prices.min()) if not overall_prices.empty else 0.0
    upper = float(overall_prices.max()) if not overall_prices.empty else 1.0
    pad = max((upper - lower) * 0.08, 0.35)
    date_ticks = data[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    legend_handles: list[object] = []
    legend_labels: list[str] = []

    for axis, meta in zip(axes_array, candidate_rows):
        subset = data.loc[data.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(clean_string(meta.get("candidate_slug")))].copy()
        subset = subset.sort_values("requested_days")
        required = subset.loc[subset.get("series_kind", pd.Series(dtype=str)).astype(str).eq("required_path")]
        assumed = subset.loc[subset.get("series_kind", pd.Series(dtype=str)).astype(str).eq("assumed_path")]
        visual = _action_bucket_visual(meta.get("action_bucket"))
        _style_axes(axis)
        if not required.empty:
            handle = axis.plot(
                required["requested_days"],
                pd.to_numeric(required.get("stock_price"), errors="coerce"),
                color=visual["color"],
                marker=visual["marker"],
                linewidth=2.8,
                markersize=5.6,
                label="Required path to justify entry",
                zorder=4,
            )[0]
            legend_handles.append(handle)
            legend_labels.append("Required path to justify entry")
        if not assumed.empty:
            handle = axis.plot(
                assumed["requested_days"],
                pd.to_numeric(assumed.get("stock_price"), errors="coerce"),
                color="#3F3F46",
                linestyle="--",
                linewidth=1.9,
                label="Active assumed stock path",
                zorder=3,
            )[0]
            legend_handles.append(handle)
            legend_labels.append("Active assumed stock path")
        if not required.empty and not assumed.empty:
            required_points = required[["requested_days", "stock_price"]].copy()
            assumed_points = assumed[["requested_days", "stock_price"]].copy()
            required_points["stock_price"] = pd.to_numeric(required_points["stock_price"], errors="coerce")
            assumed_points["stock_price"] = pd.to_numeric(assumed_points["stock_price"], errors="coerce")
            merged = required_points.merge(assumed_points, on="requested_days", suffixes=("_required", "_assumed")).dropna()
            if not merged.empty:
                axis.fill_between(
                    merged["requested_days"],
                    merged["stock_price_assumed"],
                    merged["stock_price_required"],
                    color=str(visual["color"]),
                    alpha=0.08,
                    linewidth=0,
                    zorder=1,
                )
        barrier_label = clean_string(meta.get("entry_barrier_label")).replace("_", " ").title() or "Entry barrier"
        axis.set_title(
            f"{clean_string(meta.get('candidate_short_label'))} - {barrier_label}",
            loc="left",
            fontsize=11.0,
            fontweight="bold",
            pad=8,
        )
        axis.set_ylim(lower - pad, upper + pad)
        axis.set_ylabel("Stock Price ($)")
        _format_money_axis(axis)
        _apply_date_ticks(axis, date_ticks)

    for axis in axes_array[panel_count:]:
        axis.axis("off")
    for axis in axes_array[max(panel_count - cols, 0):panel_count]:
        axis.set_xlabel("Date")

    fig.suptitle(title, x=0.06, y=0.985, ha="left", fontsize=15.0, fontweight="bold")
    dedup = dict(zip(legend_labels, legend_handles))
    if dedup:
        fig.legend(
            dedup.values(),
            dedup.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            ncol=2,
            frameon=False,
            fontsize=8.8,
        )
    active_iv_label = clean_string(data.get("iv_path_label", pd.Series(dtype=str)).dropna().iloc[0] if "iv_path_label" in data.columns and not data.get("iv_path_label").dropna().empty else "")
    caption = (
        f"Fixed: IV path = {active_iv_label or 'active assumed IV'} | "
        "colored path = required stock route | dashed gray = active assumed stock path | shaded gap = hurdle difference"
    )
    _action_chart_caption(fig, caption)
    return _finalize_caption_chart(fig, output_path, bottom=0.13, top=0.93)


def _plot_required_move_speed_vs_magnitude_refined(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Required move summary frame is empty.")
    data["move_pace_pct_per_month"] = pd.to_numeric(data.get("move_pace_pct_per_month"), errors="coerce").fillna(0.0)
    data["required_move_pct_target"] = pd.to_numeric(data.get("required_move_pct_target"), errors="coerce").fillna(0.0)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data = data.sort_values(["action_priority_rank", "required_move_pct_target"]).head(12).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.4, 7.1))
    _style_axes(ax)
    x_values = data["move_pace_pct_per_month"]
    max_x = max(float(x_values.max()), 1.0)
    max_y = max(float(data["required_move_pct_target"].max()), 1.0)
    forgiving_x = max_x * 0.45
    demanding_y = max_y * 0.55
    ax.axvspan(0, forgiving_x, ymin=0, ymax=min(demanding_y / (max_y * 1.10), 1.0), color="#56B4E9", alpha=0.07, zorder=0)
    ax.axhspan(demanding_y, max_y * 1.10, color="#D55E00", alpha=0.055, zorder=0)
    ax.text(forgiving_x * 0.10, demanding_y * 0.30, "more forgiving", fontsize=8.4, color="#155E75", fontweight="bold")
    ax.text(max_x * 0.70, max_y * 0.96, "very demanding", fontsize=8.4, color="#9A3412", fontweight="bold")
    seen_buckets: set[str] = set()
    for idx, row in data.iterrows():
        bucket = clean_string(row.get("action_bucket"))
        visual = _action_bucket_visual(bucket)
        label = visual["label"] if bucket not in seen_buckets else None
        seen_buckets.add(bucket)
        edge = "#1A1A1A" if bool(row.get("stock_still_better_even_if_path_hits")) else "#FFFFFF"
        ax.scatter(
            float(row["move_pace_pct_per_month"]),
            float(row["required_move_pct_target"]),
            s=145,
            color=visual["color"],
            marker=visual["marker"],
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.92,
            label=label,
            zorder=4,
        )
        offset_x, offset_y = [(8, 7), (8, -11), (-38, 10), (-40, -9), (10, 15)][idx % 5]
        ax.annotate(
            clean_string(row.get("candidate_short_label")) or _compact_tradeoff_label(row.get("candidate_label")),
            xy=(float(row["move_pace_pct_per_month"]), float(row["required_move_pct_target"])),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7.7,
            color="#292524",
            ha="right" if offset_x < 0 else "left",
            va="center",
        )
    if not data.empty:
        ax.axvline(float(x_values.median()), color="#A8A29E", linestyle=":", linewidth=1.0)
        ax.axhline(float(data["required_move_pct_target"].median()), color="#A8A29E", linestyle=":", linewidth=1.0)
        x_min = float(x_values.min())
        x_max = float(x_values.max())
        x_pad = max((x_max - x_min) * 0.14, 0.4)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(min(0, float(data["required_move_pct_target"].min()) - max_y * 0.08), max_y * 1.10)
    ax.set_xlabel("Required Pace Of Move (% per month)")
    ax.set_ylabel("Required Upside By Key Decision Date (%)")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), ncol=4, frameon=False, fontsize=8.4)
    _action_chart_caption(
        fig,
        "Farther right means the stock has to climb faster each month. Higher means a larger total upside is still required by the key decision date. Dark outlines mark calls where stock still looks cleaner even if that hurdle is met.",
    )
    return _finalize_decision_chart(fig, output_path)


def _plot_required_move_vs_stock_chart_refined(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Required move vs stock frame is empty.")
    data["difference_vs_stock"] = pd.to_numeric(data.get("difference_vs_stock"), errors="coerce").fillna(0.0)
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data["stock_still_better_even_if_path_hits"] = data.get("stock_still_better_even_if_path_hits", pd.Series(dtype=bool)).fillna(False).astype(bool)
    data = data.sort_values(
        ["stock_still_better_even_if_path_hits", "difference_vs_stock", "action_priority_rank"],
        ascending=[False, True, True],
    ).head(10)
    labels = [clean_string(value) or _compact_tradeoff_label(value) for value in data.get("candidate_short_label", pd.Series(dtype=str))]
    y_pos = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.4, max(5.8, 1.6 + len(data.index) * 0.48)))
    _style_axes(ax, zero_line=True)
    colors = [_action_bucket_visual(bucket)["color"] for bucket in data.get("action_bucket", pd.Series(dtype=str))]
    edges = ["#1A1A1A" if value else "#FFFFFF" for value in data["stock_still_better_even_if_path_hits"]]
    ax.barh(y_pos, data["difference_vs_stock"], color=colors, edgecolor=edges, linewidth=1.2, alpha=0.86)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Modeled Edge Vs Long Stock If The Required Path Is Met ($)")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${value:,.0f}" if abs(value) >= 100 else f"${value:,.2f}"))
    x_min = float(data["difference_vs_stock"].min()) if not data.empty else -1.0
    x_max = float(data["difference_vs_stock"].max()) if not data.empty else 1.0
    x_pad = max((x_max - x_min) * 0.08, 6.0)
    ax.set_xlim(x_min - x_pad, max(x_max + x_pad, x_pad * 0.5))
    note_anchor = max(abs(x_min), abs(x_max), 1.0) * 0.02
    for idx, row in enumerate(data.to_dict("records")):
        note = "stock still better" if bool(row.get("stock_still_better_even_if_path_hits")) or _num_safe(row.get("difference_vs_stock")) <= 0 else "call edge"
        x_value = _num_safe(row.get("difference_vs_stock"))
        ax.text(
            note_anchor if x_value >= 0 else -note_anchor,
            idx,
            note,
            ha="left" if x_value >= 0 else "right",
            va="center",
            fontsize=7.8,
            color="#44403C",
        )
    _action_chart_caption(
        fig,
        "Left of zero means stock still looks cleaner even when the required call path is broadly achieved. Black outlines highlight those stock-dominant rows.",
    )
    return _finalize_decision_chart(fig, output_path)


def _plot_iv_support_requirement_chart_refined(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Required IV support summary frame is empty.")
    data["action_priority_rank"] = pd.to_numeric(data.get("action_priority_rank"), errors="coerce").fillna(999.0)
    data["lower_iv_move_penalty_pct"] = pd.to_numeric(data.get("lower_iv_move_penalty_pct"), errors="coerce").fillna(0.0)
    data["higher_iv_move_relief_pct"] = pd.to_numeric(data.get("higher_iv_move_relief_pct"), errors="coerce").fillna(0.0)
    data = data.sort_values(["lower_iv_move_penalty_pct", "action_priority_rank"], ascending=[False, True]).head(10).reset_index(drop=True)
    y_pos = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.6, max(5.4, 1.3 + len(data.index) * 0.46)))
    _style_axes(ax, zero_line=True)
    ax.barh(
        y_pos - 0.16,
        data["lower_iv_move_penalty_pct"],
        height=0.28,
        color="#D55E00",
        alpha=0.88,
        label="Extra upside needed if IV cools",
    )
    ax.barh(
        y_pos + 0.16,
        -data["higher_iv_move_relief_pct"],
        height=0.28,
        color="#0072B2",
        alpha=0.85,
        label="Upside requirement relieved if IV stays firmer",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([clean_string(value) or _compact_tradeoff_label(value) for value in data.get("candidate_short_label", pd.Series(dtype=str))])
    ax.invert_yaxis()
    ax.set_xlabel("Change In Required Target Move (%) Versus Flat IV")
    ax.set_title(title, loc="left", fontsize=14.6, fontweight="bold", pad=12)
    max_abs = max(
        float(data["lower_iv_move_penalty_pct"].abs().max()) if not data.empty else 0.0,
        float(data["higher_iv_move_relief_pct"].abs().max()) if not data.empty else 0.0,
        0.8,
    )
    ax.set_xlim(-max_abs * 1.35, max_abs * 1.35)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.1f}%"))
    low_sensitivity = (
        float(data["lower_iv_move_penalty_pct"].abs().max()) if not data.empty else 0.0
    ) < 0.15 and (
        float(data["higher_iv_move_relief_pct"].abs().max()) if not data.empty else 0.0
    ) < 0.15
    if low_sensitivity:
        ax.scatter(np.zeros(len(y_pos)), y_pos - 0.16, color="#D55E00", marker="s", s=28, alpha=0.9, zorder=4)
        ax.scatter(np.zeros(len(y_pos)), y_pos + 0.16, color="#0072B2", marker="o", s=24, alpha=0.9, zorder=4)
        ax.text(
            0.01,
            1.02,
            "Most selected calls show very little modeled IV sensitivity in this run.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.4,
            color="#57534E",
        )
    for idx, row in enumerate(data.to_dict("records")):
        penalty = _num_safe(row.get("lower_iv_move_penalty_pct"))
        relief = _num_safe(row.get("higher_iv_move_relief_pct"))
        if penalty > 0.15:
            ax.text(
                penalty + max_abs * 0.05,
                idx - 0.16,
                f"+{penalty:.1f}",
                va="center",
                ha="left",
                fontsize=7.4,
                color="#7C2D12",
            )
        if relief > 0.15:
            ax.text(
                -relief - max_abs * 0.05,
                idx + 0.16,
                f"-{relief:.1f}",
                va="center",
                ha="right",
                fontsize=7.4,
                color="#1D4ED8",
            )
    _place_legend(ax, ncol=2, fontsize=8.5)
    _action_chart_caption(
        fig,
        "Orange to the right means the call needs a meaningfully stronger stock move if IV cools. Blue to the left shows how much easier the required move gets if IV stays friendlier.",
    )
    return _finalize_decision_chart(fig, output_path)


plot_required_stock_path_to_buy = _plot_required_stock_path_to_buy_refined
plot_required_move_speed_vs_magnitude = _plot_required_move_speed_vs_magnitude_refined
plot_required_move_vs_stock_chart = _plot_required_move_vs_stock_chart_refined
plot_iv_support_requirement_chart = _plot_iv_support_requirement_chart_refined


def plot_path_survival_scorecard(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show candidate survival rates over modeled stock paths and IV opportunities."""

    data = _top_candidate_rows(frame, score_column="objective_score", limit=10)
    if data.empty:
        raise ValueError("Candidate robustness frame is empty.")
    data["survival_rate"] = pd.to_numeric(data.get("profitable_iv_path_rate"), errors="coerce").fillna(0.0)
    data["beat_stock_rate"] = pd.to_numeric(data.get("beat_stock_iv_path_rate"), errors="coerce").fillna(0.0)
    data = data.sort_values(["survival_rate", "beat_stock_rate"], ascending=[False, False]).reset_index(drop=True)
    y_pos = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.2, max(5.2, 1.2 + len(data.index) * 0.48)))
    _style_axes(ax)
    ax.barh(y_pos - 0.16, data["survival_rate"] * 100.0, height=0.28, color="#009E73", label="Profitable across IV/path checks", alpha=0.92)
    ax.barh(y_pos + 0.16, data["beat_stock_rate"] * 100.0, height=0.28, color="#0072B2", label="Beats stock across IV/path checks", alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_compact_candidate_label(v, limit=30) for v in data.get("candidate_label", pd.Series(dtype=str))])
    ax.invert_yaxis()
    _apply_title(ax, title, subtitle="Rates are derived from path-centric IV robustness tables; higher is better, but trust still matters.")
    ax.set_xlabel("Share Of Modeled Opportunities (%)")
    ax.set_xlim(0, 105)
    _place_legend(ax, ncol=2, fontsize=8.6)
    return _finalize(fig, output_path)


def plot_iv_robustness_scorecard(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show which candidates survive lower IV and which depend on high IV."""

    data = _top_candidate_rows(frame, score_column="objective_score", limit=10)
    if data.empty:
        raise ValueError("Candidate robustness frame is empty.")
    for col in ["lower_iv_survival_rate", "lower_iv_beat_stock_rate", "high_iv_dependency_rate"]:
        data[col] = pd.to_numeric(data.get(col), errors="coerce").fillna(0.0)
    data = data.sort_values(["lower_iv_survival_rate", "lower_iv_beat_stock_rate"], ascending=[False, False]).reset_index(drop=True)
    y_pos = np.arange(len(data.index))
    fig, ax = plt.subplots(figsize=(12.2, max(5.2, 1.2 + len(data.index) * 0.48)))
    _style_axes(ax)
    ax.barh(y_pos - 0.20, data["lower_iv_survival_rate"] * 100.0, height=0.24, color="#009E73", label="Survives lower IV", alpha=0.92)
    ax.barh(y_pos + 0.02, data["lower_iv_beat_stock_rate"] * 100.0, height=0.24, color="#56B4E9", label="Beats stock under lower IV", alpha=0.90)
    ax.barh(y_pos + 0.24, data["high_iv_dependency_rate"] * 100.0, height=0.24, color="#D55E00", label="Needs high IV support", alpha=0.86)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_compact_candidate_label(v, limit=30) for v in data.get("candidate_label", pd.Series(dtype=str))])
    ax.invert_yaxis()
    _apply_title(ax, title, subtitle="Green means lower-IV resilience. Orange means the call needs friendly IV to look attractive.")
    ax.set_xlabel("Share Of Modeled IV Checks (%)")
    ax.set_xlim(0, 105)
    _place_legend(ax, ncol=3, fontsize=8.4)
    return _finalize(fig, output_path)


def plot_strike_expiry_tradeoff_overview(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Show long-call strike/expiry tradeoffs using balanced score and robustness."""

    data = frame.copy()
    if data.empty:
        return _empty_state_chart(
            output_path=output_path,
            title=title,
            message="No candidate tradeoff rows were available for this ticker and snapshot.",
            caption="This usually means the local option chain is too sparse for the strike/expiry decision view.",
        )
    data = data.loc[data.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_call")].copy()
    if data.empty:
        return _empty_state_chart(
            output_path=output_path,
            title=title,
            message="No bullish long-call rows were available for this ticker under the current local data.",
            caption="The analysis bundle is still valid; use trust/context tables first and treat call-specific views as unavailable.",
        )
    data["strike_numeric"] = pd.to_numeric(data.get("strike_label"), errors="coerce")
    data = data.dropna(subset=["strike_numeric"])
    if data.empty:
        return _empty_state_chart(
            output_path=output_path,
            title=title,
            message="Long-call rows exist, but none have a numeric strike suitable for the strike/expiry chart.",
            caption="Check candidate_tradeoff_matrix.csv for the underlying rows and data-quality notes.",
        )
    data = _top_candidate_rows(data, score_column="balanced_score", limit=14)
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    _style_axes(ax)
    expiries = _sorted_unique(data.get("expiry_date", pd.Series(dtype=str)).astype(str).tolist())
    palette = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#6B7280"]
    expiry_colors = {expiry: palette[idx % len(palette)] for idx, expiry in enumerate(expiries)}
    for expiry, group in data.groupby("expiry_date", dropna=False):
        ordered = group.sort_values("strike_numeric")
        ax.plot(
            ordered["strike_numeric"],
            pd.to_numeric(ordered.get("balanced_score"), errors="coerce").fillna(0.0),
            marker="o",
            linewidth=2.4,
            color=expiry_colors.get(clean_string(expiry), "#4C566A"),
            label=_short_expiry_label(expiry),
        )
    _apply_title(ax, title, subtitle="Higher balanced score means the strike/expiry keeps more upside, IV resilience, timing fit, and trust.")
    ax.set_xlabel("Call Strike")
    ax.set_ylabel("Balanced Decision Score")
    ax.set_ylim(-3, 103)
    _place_legend(ax, ncol=4, fontsize=8.6)
    return _finalize(fig, output_path)


def plot_stock_vs_option_decision_chart(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Bar chart for modeled edge versus long stock."""

    data = _top_candidate_rows(frame, score_column="balanced_score", limit=12)
    if data.empty:
        raise ValueError("Candidate tradeoff frame is empty.")
    data["difference_vs_stock"] = pd.to_numeric(data.get("difference_vs_stock"), errors="coerce").fillna(0.0)
    data = data.sort_values("difference_vs_stock", ascending=True).reset_index(drop=True)
    colors = np.where(data["difference_vs_stock"] >= 0, "#009E73", "#D55E00")
    fig, ax = plt.subplots(figsize=(12.4, max(5.4, 1.2 + len(data.index) * 0.48)))
    _style_axes(ax, zero_line=True)
    y_pos = np.arange(len(data.index))
    bars = ax.barh(y_pos, data["difference_vs_stock"], color=colors, alpha=0.90, edgecolor="#F5F2EA", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([_compact_candidate_label(v, limit=32) for v in data.get("candidate_label", pd.Series(dtype=str))])
    for bar, value in zip(bars, data["difference_vs_stock"].tolist()):
        ax.text(
            value + (4 if value >= 0 else -4),
            bar.get_y() + bar.get_height() / 2,
            f"${value:,.0f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.3,
            color="#292524",
        )
    _apply_title(ax, title, subtitle="Positive means the candidate beats long stock under current modeled assumptions; negative means stock still wins.")
    ax.set_xlabel("Modeled Difference Vs Long Stock ($)")
    _format_money_axis(ax)
    return _finalize(fig, output_path)


def _preferred_path_pair_id(frame: pd.DataFrame) -> str:
    if frame.empty or "path_pair_id" not in frame.columns:
        return ""
    bucket_order = {
        "just_works": 0,
        "works_well": 1,
        "works_very_well": 2,
        "almost_works": 3,
        "misses_badly": 4,
        "supporting_example": 5,
    }
    ordering = frame.copy()
    ordering["bucket_rank"] = ordering.get("representative_bucket").map(
        lambda value: bucket_order.get(clean_string(value), 99)
    )
    ordered = ordering.sort_values(["bucket_rank", "path_pair_id"])
    return clean_string(ordered.iloc[0].get("path_pair_id")) if not ordered.empty else ""


def plot_representative_stock_paths(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot the selected representative stock paths over time."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Representative stock-path frame is empty.")
    representative = data.loc[data.get("is_representative") == True].copy()  # noqa: E712
    if representative.empty:
        representative = data.copy()
    representative = representative.dropna(subset=["requested_days", "spot_price"])
    if representative.empty:
        raise ValueError("Representative stock-path frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    _style_axes(ax)
    tick_map = (
        representative[["requested_days", "date"]]
        .drop_duplicates()
        .sort_values("requested_days")
    )
    bucket_palette = {
        "misses_badly": "#D55E00",
        "almost_works": "#E69F00",
        "just_works": "#009E73",
        "works_well": "#0072B2",
        "works_very_well": "#CC79A7",
    }
    groups = list(representative.groupby(["path_id", "representative_bucket", "path_name"], dropna=False))
    groups.sort(key=lambda item: (_canonical_sort_key(item[0][1]), clean_string(item[0][2])))
    for (_, bucket, path_name), group in groups:
        ordered = group.sort_values("requested_days")
        color = bucket_palette.get(clean_string(bucket), "#4C566A")
        ax.plot(
            ordered["requested_days"],
            ordered["spot_price"],
            label=f"{_human_label(clean_string(bucket))}: {_human_label(clean_string(path_name))}",
            color=color,
            linewidth=2.6,
            linestyle="-",
            marker="o",
            markersize=4.8,
        )
    if not tick_map.empty:
        ax.set_xticks(tick_map["requested_days"].tolist()[:: max(1, len(tick_map.index) // 6)])
        ax.set_xticklabels(tick_map["date"].tolist()[:: max(1, len(tick_map.index) // 6)], rotation=18, ha="right")
    _apply_title(
        ax,
        title,
        subtitle="Representative paths are selected examples, not forecasts. Read them against the required-path charts rather than as standalone truth.",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=3, fontsize=8)
    return _finalize(fig, output_path)


def plot_stock_path_gallery(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot the named stock-path gallery used as the primary scenario surface."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Stock-path gallery frame is empty.")
    data = data.dropna(subset=["requested_days", "spot_price"])
    if data.empty:
        raise ValueError("Stock-path gallery frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(16.8, 8.1))
    _style_axes(ax)
    tick_map = data[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    groups = list(data.groupby(["display_order", "path_name", "path_label", "path_role", "is_active_assumed"], dropna=False))
    groups.sort(key=lambda item: (item[0][0], clean_string(item[0][1])))
    for (_, path_name, path_label, path_role, is_active_assumed), group in groups:
        ordered = group.sort_values("requested_days")
        style_key = clean_string(path_name).lower()
        if clean_string(path_role) == "active_assumed_path":
            style_key = "active_assumed_path"
        style = dict(STOCK_PATH_GALLERY_SPECS.get(style_key, {"color": "#4C566A", "marker": "o", "linestyle": "-", "linewidth": 2.4}))
        linewidth = float(style["linewidth"]) + (0.8 if bool(is_active_assumed) else 0.0)
        label = clean_string(path_label)
        if bool(is_active_assumed) and clean_string(path_role) == "gallery_named_path":
            label = f"{label} (Active)"
        ax.plot(
            ordered["requested_days"],
            ordered["spot_price"],
            label=label,
            color=str(style["color"]),
            linewidth=linewidth,
            linestyle=style["linestyle"],
            marker=str(style["marker"]),
            markersize=4.8 if not bool(is_active_assumed) else 5.2,
            alpha=0.97 if bool(is_active_assumed) else 0.92,
        )
    _apply_date_ticks(ax, tick_map)
    _apply_title(
        ax,
        title,
        subtitle="Named gallery paths are deliberate scenario templates. Use these first to choose the future you want to evaluate before reading the representative-path examples.",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=5, fontsize=7)
    return _finalize(fig, output_path)


def plot_representative_iv_paths(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot the selected representative IV paths over time."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Representative IV-path frame is empty.")
    representative = data.loc[data.get("is_representative") == True].copy()  # noqa: E712
    if representative.empty:
        representative = data.copy()
    representative = representative.dropna(subset=["requested_days", "iv_shift_points"])
    if representative.empty:
        raise ValueError("Representative IV-path frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    _style_axes(ax, zero_line=True)
    tick_map = (
        representative[["requested_days", "date"]]
        .drop_duplicates()
        .sort_values("requested_days")
    )
    groups = list(representative.groupby(["iv_path_id", "representative_bucket", "iv_path_name"], dropna=False))
    groups.sort(key=lambda item: (_canonical_sort_key(item[0][1]), clean_string(item[0][2])))
    for (_, bucket, iv_path_name), group in groups:
        ordered = group.sort_values("requested_days")
        iv_style = IV_PATH_VISUAL_SPECS.get(
            clean_string(iv_path_name),
            {"color": "#6B7280", "marker": "o", "linestyle": "--", "linewidth": 2.0},
        )
        ax.plot(
            ordered["requested_days"],
            ordered["iv_shift_points"],
            label=f"{_human_label(clean_string(bucket))}: {_human_label(clean_string(iv_path_name))}",
            color=str(iv_style["color"]),
            linewidth=float(iv_style["linewidth"]),
            linestyle=iv_style["linestyle"],
            marker=str(iv_style["marker"]),
            markersize=4.8,
            alpha=0.92,
        )
    if not tick_map.empty:
        ax.set_xticks(tick_map["requested_days"].tolist()[:: max(1, len(tick_map.index) // 6)])
        ax.set_xticklabels(tick_map["date"].tolist()[:: max(1, len(tick_map.index) // 6)], rotation=18, ha="right")
    _apply_title(
        ax,
        title,
        subtitle="IV stays conceptually separate from stock. Use this to see whether the same stock path is being helped or hurt by a different IV regime.",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("IV Shift (pts)")
    _place_legend(ax, ncol=3, fontsize=8)
    return _finalize(fig, output_path)


def plot_iv_path_gallery(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot the named IV-regime gallery as a normalized shape surface."""

    data = frame.copy()
    if data.empty:
        raise ValueError("IV-path gallery frame is empty.")
    data = data.dropna(subset=["requested_days", "iv_shift_points"])
    if data.empty:
        raise ValueError("IV-path gallery frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(14.8, 6.4))
    _style_axes(ax, zero_line=True)
    tick_map = data[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    groups = list(data.groupby(["display_order", "iv_path_name", "iv_path_label", "path_role", "is_active_assumed"], dropna=False))
    groups.sort(key=lambda item: (item[0][0], clean_string(item[0][1])))
    for (_, iv_path_name, iv_path_label, path_role, is_active_assumed), group in groups:
        ordered = group.sort_values("requested_days")
        style_key = clean_string(iv_path_name).lower()
        style = dict(
            IV_PATH_VISUAL_SPECS.get(
                "active_assumption" if clean_string(path_role) == "active_assumed_iv" else style_key,
                {"color": "#6B7280", "marker": "o", "linestyle": "-", "linewidth": 2.2},
            )
        )
        linewidth = float(style["linewidth"]) + (0.5 if bool(is_active_assumed) else 0.0)
        label = clean_string(iv_path_label)
        if bool(is_active_assumed) and clean_string(path_role) == "gallery_named_path":
            label = f"{label} (Active)"
        ax.plot(
            ordered["requested_days"],
            ordered["iv_shift_points"],
            label=label,
            color=str(style["color"]),
            linewidth=linewidth,
            linestyle=style["linestyle"],
            marker=str(style["marker"]),
            markersize=4.8 if not bool(is_active_assumed) else 5.1,
            alpha=0.97 if bool(is_active_assumed) else 0.92,
        )
    _apply_date_ticks(ax, tick_map)
    _apply_title(
        ax,
        title,
        subtitle="Named IV gallery paths are normalized regime shapes. Use them to think about IV help or IV drag separately from the stock-path thesis.",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("IV Shift (pts)")
    _place_legend(ax, ncol=3, fontsize=8)
    return _finalize(fig, output_path)


def plot_option_value_over_path(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot modeled PnL over time for multiple candidates under the same representative path pair."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Option-value-over-path frame is empty.")
    active_pair = _preferred_path_pair_id(data)
    if active_pair:
        data = data.loc[data.get("path_pair_id").astype(str) == active_pair].copy()
    data = data.dropna(subset=["requested_days", "profit_loss"])
    if data.empty:
        raise ValueError("Option-value-over-path frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    _style_axes(ax, zero_line=True)
    top_candidates = (
        data.sort_values(["requested_days", "profit_loss"], ascending=[False, False])
        .groupby("candidate_slug", dropna=False, as_index=False)
        .tail(1)
        .sort_values("profit_loss", ascending=False)
        .head(6)
    )
    keep_slugs = set(top_candidates["candidate_slug"].astype(str).tolist())
    working = data.loc[data.get("candidate_slug").astype(str).isin(keep_slugs)].copy()
    for (candidate_slug, label, family), group in _ordered_strategy_groups(
        working,
        ["candidate_slug", "candidate_label", "strategy_family"],
    ):
        ordered = group.sort_values("requested_days")
        ax.plot(
            ordered["requested_days"],
            ordered["profit_loss"],
            label=clean_string(label) or clean_string(candidate_slug),
            **_line_style(clean_string(family), len(ordered.index)),
        )
    tick_map = working[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    if not tick_map.empty:
        ax.set_xticks(tick_map["requested_days"].tolist()[:: max(1, len(tick_map.index) // 6)])
        ax.set_xticklabels(tick_map["date"].tolist()[:: max(1, len(tick_map.index) // 6)], rotation=18, ha="right")
    bucket = clean_string(working.iloc[0].get("representative_bucket"))
    _apply_title(
        ax,
        title,
        subtitle=f"Same stock path, same IV path, different structures. The current chart is anchored to the {_human_label(bucket)} representative path pair.",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Modeled PnL ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=3, fontsize=8)
    return _finalize(fig, output_path)


def plot_compare_vs_stock_over_path(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot path PnL delta versus stock for multiple candidates under the same path pair."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Compare-vs-stock-over-path frame is empty.")
    active_pair = _preferred_path_pair_id(data)
    if active_pair:
        data = data.loc[data.get("path_pair_id").astype(str) == active_pair].copy()
    data = data.dropna(subset=["requested_days", "delta_profit_loss_vs_stock"])
    if data.empty:
        raise ValueError("Compare-vs-stock-over-path frame has no plottable rows.")

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    _style_axes(ax)
    baseline_style = SUPPORTING_VISUAL_SPECS["stock_baseline"]
    ax.axhline(
        0,
        color=str(baseline_style["color"]),
        linewidth=float(baseline_style["linewidth"]),
        linestyle=baseline_style["linestyle"],
        label="Long Stock Baseline",
        zorder=1,
    )
    top_candidates = (
        data.sort_values(["requested_days", "delta_profit_loss_vs_stock"], ascending=[False, False])
        .groupby("candidate_slug", dropna=False, as_index=False)
        .tail(1)
        .sort_values("delta_profit_loss_vs_stock", ascending=False)
        .head(6)
    )
    keep_slugs = set(top_candidates["candidate_slug"].astype(str).tolist())
    working = data.loc[data.get("candidate_slug").astype(str).isin(keep_slugs)].copy()
    for (candidate_slug, label, family), group in _ordered_strategy_groups(
        working,
        ["candidate_slug", "candidate_label", "strategy_family"],
    ):
        if clean_string(family) == "long_stock":
            continue
        ordered = group.sort_values("requested_days")
        ax.plot(
            ordered["requested_days"],
            ordered["delta_profit_loss_vs_stock"],
            label=clean_string(label) or clean_string(candidate_slug),
            **_line_style(clean_string(family), len(ordered.index)),
        )
    tick_map = working[["requested_days", "date"]].drop_duplicates().sort_values("requested_days")
    if not tick_map.empty:
        ax.set_xticks(tick_map["requested_days"].tolist()[:: max(1, len(tick_map.index) // 6)])
        ax.set_xticklabels(tick_map["date"].tolist()[:: max(1, len(tick_map.index) // 6)], rotation=18, ha="right")
    bucket = clean_string(working.iloc[0].get("representative_bucket"))
    _apply_title(
        ax,
        title,
        subtitle=f"Above zero means the structure is ahead of stock along the same {_human_label(bucket)} representative path pair.",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("PnL Delta vs Long Stock ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=3, fontsize=8)
    return _finalize(fig, output_path)


def _long_call_series_style(index: int) -> dict[str, object]:
    spec = LONG_CALL_COMPARISON_SPECS[index % len(LONG_CALL_COMPARISON_SPECS)]
    return {
        "color": spec["color"],
        "linewidth": float(spec["linewidth"]),
        "linestyle": spec["linestyle"],
        "marker": spec["marker"],
        "markersize": 5.2 if index == 0 else 4.8,
        "alpha": 0.98 if index == 0 else 0.95,
    }


def _long_call_view_subtitle(frame: pd.DataFrame, *, view_name: str) -> str:
    working = frame.sort_values(["selection_rank", "requested_days"]).copy()
    lead = working.iloc[0].to_dict() if not working.empty else {}
    stock_path = _human_label(clean_string(lead.get("stock_path_name")))
    iv_path = _human_label(clean_string(lead.get("iv_path_name")))
    if clean_string(view_name) == "long_call_strike_view":
        expiry = _short_expiry_label(lead.get("anchor_expiry_date") or lead.get("expiry_date") or "n/a")
        return (
            f"Fixed: stock path = {stock_path} | IV path = {iv_path} | "
            f"expiry = {expiry}"
        )
    if clean_string(view_name) == "long_call_expiry_view":
        strike_anchor = _compact_strike_call_label(lead.get("anchor_strike_label") or lead.get("strike_label") or "n/a")
        return (
            f"Fixed: stock path = {stock_path} | IV path = {iv_path} | "
            f"strike concept = {strike_anchor}"
        )
    return f"Fixed: stock path = {stock_path} | IV path = {iv_path} | curated long-call subset"


def _sorted_long_call_groups(working: pd.DataFrame, *, view_name: str) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    grouped = list(working.groupby(["candidate_slug", "series_label"], dropna=False, sort=False))

    def key(item: tuple[tuple[str, str], pd.DataFrame]) -> tuple[Any, ...]:
        group = item[1].sort_values(["requested_days", "step_index"])
        lead = group.iloc[0]
        strike_value = finite_or_none(lead.get("strike_label"))
        strike_key = float(strike_value) if strike_value is not None else float("inf")
        expiry_key = clean_string(lead.get("expiry_date"))
        selection_rank = int(lead.get("selection_rank") or 9999)
        if clean_string(view_name) == "long_call_strike_view":
            return (strike_key, selection_rank, expiry_key)
        if clean_string(view_name) == "long_call_expiry_view":
            return (expiry_key, selection_rank, strike_key)
        return (strike_key, expiry_key, selection_rank)

    return sorted(grouped, key=key)


def _plot_long_call_value_over_path(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    subtitle: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Long-call path-comparison frame is empty.")
    data = data.dropna(subset=["requested_days", "modeled_value"])
    if data.empty:
        raise ValueError("Long-call path-comparison frame has no plottable rows.")

    fig, ax, stock_ax = _create_path_stack(figsize=(12.8, 7.4))
    working = data.sort_values(["selection_rank", "requested_days", "candidate_label"]).copy()
    view_name = clean_string(working.iloc[0].get("view_name"))
    strike_refs: list[float] = []
    for index, ((candidate_slug, series_label), group) in enumerate(
        _sorted_long_call_groups(working, view_name=view_name)
    ):
        ordered = group.sort_values("requested_days")
        strike_value = finite_or_none(ordered.iloc[0].get("strike_label")) if not ordered.empty else None
        if strike_value is not None and clean_string(view_name) in {"long_call_strike_view", "long_call_best_of_view"}:
            strike_refs.append(float(strike_value))
        ax.plot(
            ordered["requested_days"],
            ordered["modeled_value"],
            label=_long_call_plot_label(ordered, view_name=view_name) or clean_string(series_label) or clean_string(candidate_slug),
            **_long_call_series_style(index),
        )
    tick_map = _date_tick_map(working)
    _plot_stock_context_panel(stock_ax, working, strike_refs=strike_refs)
    _annotate_path_milestones(ax, working, include_expiry=True)
    _annotate_path_milestones(stock_ax, working, include_expiry=False)
    _apply_date_ticks(stock_ax, tick_map)
    _apply_title(ax, title)
    ax.set_ylabel("Option Value ($)")
    stock_ax.set_xlabel("Date")
    _format_money_axis(ax)
    _place_figure_caption(fig, subtitle)
    _place_figure_legend(fig, ax, ncol=3, fontsize=8, y=0.020)
    return _finalize_stacked(fig, output_path, bottom=0.23)


def _plot_long_call_delta_over_path(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    subtitle: str,
    view_name: str | None = None,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Long-call delta-vs-stock frame is empty.")
    data = data.dropna(subset=["requested_days", "delta_profit_loss_vs_stock"])
    if data.empty:
        raise ValueError("Long-call delta-vs-stock frame has no plottable rows.")

    fig, ax, stock_ax = _create_path_stack(figsize=(12.8, 7.4))
    working = data.sort_values(["selection_rank", "requested_days", "candidate_label"]).copy()
    resolved_view_name = clean_string(view_name or working.iloc[0].get("view_name") or "long_call_best_of_view")
    strike_refs: list[float] = []
    for index, ((candidate_slug, series_label), group) in enumerate(
        _sorted_long_call_groups(working, view_name=resolved_view_name)
    ):
        ordered = group.sort_values("requested_days")
        strike_value = finite_or_none(ordered.iloc[0].get("strike_label")) if not ordered.empty else None
        if strike_value is not None and resolved_view_name in {"long_call_strike_view", "long_call_best_of_view"}:
            strike_refs.append(float(strike_value))
        ax.plot(
            ordered["requested_days"],
            ordered["delta_profit_loss_vs_stock"],
            label=_long_call_plot_label(ordered, view_name=resolved_view_name) or clean_string(series_label) or clean_string(candidate_slug),
            **_long_call_series_style(index),
        )
    tick_map = _date_tick_map(working)
    _plot_stock_context_panel(stock_ax, working, strike_refs=strike_refs)
    _annotate_path_milestones(ax, working, include_expiry=True)
    _annotate_path_milestones(stock_ax, working, include_expiry=False)
    _apply_date_ticks(stock_ax, tick_map)
    _apply_title(ax, title)
    ax.set_ylabel("Delta vs Long Stock ($)")
    stock_ax.set_xlabel("Date")
    _format_money_axis(ax)
    _place_figure_caption(fig, subtitle)
    _place_figure_legend(fig, ax, ncol=3, fontsize=8, y=0.020)
    return _finalize_stacked(fig, output_path, bottom=0.23)


def plot_path_long_call_compare_vs_stock(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    subtitle: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("Path-centric compare-vs-stock frame is empty.")
    data = data.dropna(subset=["requested_days", "delta_profit_loss_vs_stock"])
    if data.empty:
        raise ValueError("Path-centric compare-vs-stock frame has no plottable rows.")

    fig, ax, stock_ax = _create_path_stack(figsize=(12.8, 7.4))
    working = data.sort_values(["selection_rank", "requested_days", "candidate_label"]).copy()
    for index, ((candidate_slug, series_label), group) in enumerate(
        _sorted_long_call_groups(
            working.rename(columns={"view_name": "view_name"}),
            view_name="long_call_best_of_view",
        )
    ):
        ordered = group.sort_values("requested_days")
        ax.plot(
            ordered["requested_days"],
            ordered["delta_profit_loss_vs_stock"],
            label=_long_call_plot_label(ordered, view_name="long_call_best_of_view") or clean_string(series_label) or clean_string(candidate_slug),
            **_long_call_series_style(index),
        )
    tick_map = _date_tick_map(working)
    _plot_stock_context_panel(stock_ax, working)
    _annotate_path_milestones(ax, working)
    _annotate_path_milestones(stock_ax, working)
    _apply_date_ticks(stock_ax, tick_map)
    _apply_title(ax, title)
    ax.set_ylabel("PnL Delta vs Long Stock ($)")
    stock_ax.set_xlabel("Date")
    _format_money_axis(ax)
    _place_figure_caption(fig, subtitle)
    _place_figure_legend(fig, ax, ncol=3, fontsize=8, y=0.020)
    return _finalize_stacked(fig, output_path, bottom=0.23)


def plot_long_call_value_over_path_strike_view(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_long_call_value_over_path(
        frame,
        output_path=output_path,
        title=title,
        subtitle=_long_call_view_subtitle(frame, view_name="long_call_strike_view"),
    )


def plot_long_call_value_over_path_expiry_view(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_long_call_value_over_path(
        frame,
        output_path=output_path,
        title=title,
        subtitle=_long_call_view_subtitle(frame, view_name="long_call_expiry_view"),
    )


def plot_long_call_value_over_path_best_of(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_long_call_value_over_path(
        frame,
        output_path=output_path,
        title=title,
        subtitle=_long_call_view_subtitle(frame, view_name="long_call_best_of_view"),
    )


def plot_long_call_delta_over_path_strike_view(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_long_call_delta_over_path(
        frame,
        output_path=output_path,
        title=title,
        subtitle=_long_call_view_subtitle(frame, view_name="long_call_strike_view"),
        view_name="long_call_strike_view",
    )


def plot_long_call_delta_over_path_expiry_view(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_long_call_delta_over_path(
        frame,
        output_path=output_path,
        title=title,
        subtitle=_long_call_view_subtitle(frame, view_name="long_call_expiry_view"),
        view_name="long_call_expiry_view",
    )


def plot_long_call_delta_over_path_best_of(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_long_call_delta_over_path(
        frame,
        output_path=output_path,
        title=title,
        subtitle=_long_call_view_subtitle(frame, view_name="long_call_best_of_view"),
        view_name="long_call_best_of_view",
    )


def _iv_path_series_style(iv_path_name: str, index: int) -> dict[str, object]:
    spec = dict(
        IV_PATH_VISUAL_SPECS.get(
            clean_string(iv_path_name).lower(),
            LONG_CALL_COMPARISON_SPECS[index % len(LONG_CALL_COMPARISON_SPECS)],
        )
    )
    return {
        "color": spec.get("color", "#6B7280"),
        "linewidth": float(spec.get("linewidth", 2.2)) + (0.25 if clean_string(iv_path_name) == "flat" else 0.0),
        "linestyle": spec.get("linestyle", "-"),
        "marker": spec.get("marker", "o"),
        "markersize": 4.8,
        "alpha": 0.96,
    }


def _sorted_iv_path_groups(working: pd.DataFrame) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    grouped = list(working.groupby(["iv_path_name", "series_label"], dropna=False, sort=False))

    def key(item: tuple[tuple[str, str], pd.DataFrame]) -> tuple[Any, ...]:
        group = item[1].sort_values(["requested_days", "step_index"])
        lead = group.iloc[0]
        display_order = finite_or_none(lead.get("iv_path_display_order"))
        return (
            int(display_order) if display_order is not None else 999,
            _canonical_sort_key(lead.get("iv_path_name")),
        )

    return sorted(grouped, key=key)


def _iv_path_view_caption(frame: pd.DataFrame) -> str:
    working = frame.sort_values(["iv_path_display_order", "requested_days"]).copy()
    lead = working.iloc[0].to_dict() if not working.empty else {}
    stock_path = _human_label(clean_string(lead.get("stock_path_name")))
    contract = clean_string(lead.get("anchor_contract_label")) or _long_call_plot_label(working, view_name="long_call_best_of_view")
    return f"Fixed: stock path = {stock_path} | contract = {contract} | varying IV paths only"


def _plot_iv_path_long_call(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    value_column: str,
    y_label: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("IV-path long-call comparison frame is empty.")
    data = data.dropna(subset=["requested_days", value_column])
    if data.empty:
        raise ValueError("IV-path long-call comparison frame has no plottable rows.")

    fig, ax, stock_ax = _create_path_stack(figsize=(12.8, 7.4))
    working = data.sort_values(["iv_path_display_order", "requested_days", "step_index"]).copy()
    for index, ((iv_path_name, series_label), group) in enumerate(_sorted_iv_path_groups(working)):
        ordered = group.sort_values("requested_days")
        ax.plot(
            ordered["requested_days"],
            ordered[value_column],
            label=clean_string(series_label) or _human_label(clean_string(iv_path_name)),
            **_iv_path_series_style(clean_string(iv_path_name), index),
        )
    tick_map = _date_tick_map(working)
    _plot_stock_context_panel(stock_ax, working, label="Fixed Stock Path")
    _annotate_path_milestones(ax, working, include_expiry=True)
    _annotate_path_milestones(stock_ax, working, include_expiry=False)
    _apply_date_ticks(stock_ax, tick_map)
    _apply_title(ax, title)
    ax.set_ylabel(y_label)
    stock_ax.set_xlabel("Date")
    _format_money_axis(ax)
    _place_figure_caption(fig, _iv_path_view_caption(working))
    _place_figure_legend(fig, ax, ncol=3, fontsize=8, y=0.020)
    return _finalize_stacked(fig, output_path, bottom=0.23)


def plot_long_call_iv_path_value(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_iv_path_long_call(
        frame,
        output_path=output_path,
        title=title,
        value_column="modeled_value",
        y_label="Option Value ($)",
    )


def plot_long_call_iv_path_delta(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_iv_path_long_call(
        frame,
        output_path=output_path,
        title=title,
        value_column="delta_profit_loss_vs_stock",
        y_label="Delta vs Long Stock ($)",
    )


def _iv_expanded_contract_style(contract_rank: int) -> dict[str, object]:
    styles = [
        {"linestyle": "-", "marker": "o", "linewidth": 2.8, "alpha": 0.98},
        {"linestyle": "--", "marker": "s", "linewidth": 2.35, "alpha": 0.92},
        {"linestyle": ":", "marker": "^", "linewidth": 2.35, "alpha": 0.88},
    ]
    return dict(styles[(max(int(contract_rank or 1), 1) - 1) % len(styles)])


def _iv_expanded_series_style(iv_path_name: str, contract_rank: int) -> dict[str, object]:
    base = _iv_path_series_style(iv_path_name, int(contract_rank or 0))
    rank_style = _iv_expanded_contract_style(int(contract_rank or 1))
    base.update(rank_style)
    return base


def _iv_expanded_label(group: pd.DataFrame) -> str:
    lead = group.sort_values(["requested_days", "step_index"]).iloc[0]
    contract = clean_string(lead.get("contract_label")) or _long_call_plot_label(group, view_name="long_call_best_of_view")
    iv_label = clean_string(lead.get("iv_path_label")) or _human_label(clean_string(lead.get("iv_path_name")))
    return f"{contract} / {iv_label}"


def _iv_expanded_caption(frame: pd.DataFrame) -> str:
    working = frame.sort_values(["contract_rank", "iv_path_display_order", "requested_days"]).copy()
    lead = working.iloc[0].to_dict() if not working.empty else {}
    stock_path = _human_label(clean_string(lead.get("stock_path_name")))
    family = _human_label(clean_string(lead.get("iv_expanded_family")))
    contract_count = int(working.loc[working.get("chart_include").astype(bool)]["candidate_slug"].nunique()) if "chart_include" in working.columns and not working.empty else int(working["candidate_slug"].nunique())
    iv_count = int(working.loc[working.get("chart_include").astype(bool)]["iv_path_name"].nunique()) if "chart_include" in working.columns and not working.empty else int(working["iv_path_name"].nunique())
    return (
        f"Fixed: stock path = {stock_path} | scope = {family} ladder | "
        f"plotted core = {contract_count} contracts x {iv_count} IV regimes | full IV set in CSV"
    )


def _plot_iv_expanded_long_call(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    value_column: str,
    y_label: str,
) -> Path:
    data = frame.copy()
    if data.empty:
        raise ValueError("IV-expanded long-call comparison frame is empty.")
    data = data.dropna(subset=["requested_days", value_column])
    if data.empty:
        raise ValueError("IV-expanded long-call comparison frame has no plottable rows.")
    if "chart_include" in data.columns and data["chart_include"].astype(bool).any():
        data = data.loc[data["chart_include"].astype(bool)].copy()

    fig, ax, stock_ax = _create_path_stack(figsize=(13.8, 7.8))
    working = data.sort_values(["contract_rank", "iv_path_display_order", "requested_days", "step_index"]).copy()
    grouped = list(working.groupby(["contract_rank", "candidate_slug", "iv_path_name"], dropna=False, sort=False))

    def key(item: tuple[tuple[Any, ...], pd.DataFrame]) -> tuple[Any, ...]:
        lead = item[1].sort_values(["requested_days", "step_index"]).iloc[0]
        return (
            int(lead.get("contract_rank") or 999),
            int(lead.get("iv_path_display_order") or 999),
            _canonical_sort_key(lead.get("iv_path_name")),
        )

    for _, group in sorted(grouped, key=key):
        ordered = group.sort_values("requested_days")
        lead = ordered.iloc[0]
        ax.plot(
            ordered["requested_days"],
            ordered[value_column],
            label=_iv_expanded_label(ordered),
            **_iv_expanded_series_style(clean_string(lead.get("iv_path_name")), int(lead.get("contract_rank") or 1)),
        )
    tick_map = _date_tick_map(working)
    strike_refs = [
        float(value)
        for value in pd.to_numeric(working.get("strike_label"), errors="coerce").dropna().unique().tolist()
        if np.isfinite(value)
    ][:4]
    _plot_stock_context_panel(stock_ax, working, strike_refs=strike_refs, label="Fixed Stock Path")
    _annotate_path_milestones(ax, working, include_expiry=True)
    _annotate_path_milestones(stock_ax, working, include_expiry=False)
    _apply_date_ticks(stock_ax, tick_map)
    _apply_title(ax, title)
    ax.set_ylabel(y_label)
    stock_ax.set_xlabel("Date")
    _format_money_axis(ax)
    _place_figure_caption(fig, _iv_expanded_caption(working))
    _place_figure_legend(fig, ax, ncol=4, fontsize=7.4, y=0.012)
    return _finalize_stacked(fig, output_path, bottom=0.25)


def plot_long_call_iv_expanded_value(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_iv_expanded_long_call(
        frame,
        output_path=output_path,
        title=title,
        value_column="modeled_value",
        y_label="Option Value ($)",
    )


def plot_long_call_iv_expanded_delta(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    return _plot_iv_expanded_long_call(
        frame,
        output_path=output_path,
        title=title,
        value_column="delta_profit_loss_vs_stock",
        y_label="Delta vs Long Stock ($)",
    )


def plot_path_comparison(
    frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
    comparison_scope: str,
) -> Path:
    """Plot strike or expiry comparison rows under one representative path pair."""

    data = frame.copy()
    if data.empty:
        raise ValueError("Path-comparison frame is empty.")
    working = data.loc[data.get("comparison_scope").astype(str) == clean_string(comparison_scope)].copy()
    if working.empty:
        raise ValueError("Path-comparison frame has no rows for the requested scope.")
    active_pair = _preferred_path_pair_id(working)
    if active_pair:
        working = working.loc[working.get("path_pair_id").astype(str) == active_pair].copy()
    if working.empty:
        raise ValueError("Path-comparison frame has no rows for the selected path pair.")
    label_column = "strike_label" if clean_string(comparison_scope) == "strike" else "expiry_date"
    working = working.sort_values(["objective_score", label_column], ascending=[False, True]).head(8)

    fig_height = max(4.6, 1.0 + len(working.index) * 0.62)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    _style_axes(ax, zero_line=True)
    y_positions = np.arange(len(working.index))
    colors = [strategy_visual_spec(clean_string(family)).get("color", "#4C566A") for family in working["strategy_family"]]
    values = pd.to_numeric(working["profit_loss"], errors="coerce").fillna(0.0)
    bars = ax.barh(y_positions, values, color=colors, edgecolor="#F5F2EA", linewidth=1.0, alpha=0.94, zorder=3)
    labels = [clean_string(value) for value in working[label_column]]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, working.iterrows()):
        ax.text(
            bar.get_width() + (max(abs(values.max()), abs(values.min()), 1.0) * 0.02),
            bar.get_y() + bar.get_height() / 2,
            clean_string(row.get("best_candidate_label")) or clean_string(row.get(label_column)),
            va="center",
            ha="left",
            fontsize=8.9,
            color="#292524",
        )
    _apply_title(
        ax,
        title,
        subtitle=f"Same stock path, same IV path. This isolates how {comparison_scope} choice changes the payoff under one representative future.",
    )
    ax.set_xlabel("Terminal PnL ($)")
    ax.set_ylabel("Strike" if clean_string(comparison_scope) == "strike" else "Expiry")
    _format_money_axis(ax)
    return _finalize(fig, output_path)


def plot_stock_vs_option_comparison(
    positions: list[StrategyPosition],
    stock_prices,
    *,
    output_path: str | Path,
    mode: str = "share_equivalent",
    horizon: str | int = "1m",
    comparison_capital: float | None = None,
    title: str | None = None,
    pricing_inputs: dict | None = None,
) -> Path:
    comparison = compare_positions(
        positions,
        mode=mode,
        spot_grid=stock_prices,
        horizon=horizon,
        pricing_inputs=pricing_inputs,
        comparison_capital=comparison_capital,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax, zero_line=True)
    for strategy_name, frame in comparison.groupby("strategy"):
        ax.plot(
            frame["spot_price"],
            frame["profit_loss"],
            label=strategy_visual_spec(strategy_name).get("label"),
            **_line_style(strategy_name, len(frame.index)),
        )
    ax.axvline(positions[0].entry_spot, color="#888888", linewidth=1, linestyle=":")
    if title is None:
        mode_label = mode.replace("_", " ").title()
        if mode == "equal_capital" and comparison_capital is not None:
            mode_label = f"Normalized To ${comparison_capital:,.0f} Initial Capital"
        title = f"{positions[0].ticker} Stock Vs Options Comparison ({mode_label})"
    _apply_title(ax, title, subtitle="Use the equal-capital view first, then the share-equivalent view as a realism check.")
    ax.set_xlabel("Stock Price")
    ax.set_ylabel("Profit / Loss ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_iv_sensitivity(
    strategy: StrategyPosition,
    *,
    stock_price: float,
    iv_grid,
    horizon_days: int,
    output_path: str | Path,
    pricing_inputs: dict | None = None,
) -> Path:
    valuation_date, _ = strategy.valuation_date_for_horizon(horizon_days)
    pricing = pricing_inputs or {}
    values = [
        float(
            strategy.mark_to_market_value(
                [stock_price],
                valuation_date=valuation_date,
                iv_shift=float(iv_shift),
                risk_free_rate=pricing.get("risk_free_rate"),
                dividend_yield=pricing.get("dividend_yield"),
            )[0]
        )
        for iv_shift in iv_grid
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax)
    ax.plot(iv_grid, values, color="#b45309", linewidth=2.6)
    _apply_title(
        ax,
        f"{strategy.ticker} {strategy.name.replace('_', ' ').title()} IV Sensitivity",
        subtitle="Steeper slopes imply heavier dependence on IV staying supportive.",
    )
    ax.set_xlabel("IV Shock (decimal points)")
    ax.set_ylabel("Estimated Position Value ($)")
    _format_money_axis(ax)
    return _finalize(fig, output_path)


def plot_time_decay(
    strategy: StrategyPosition,
    *,
    stock_price: float,
    horizon_days_grid,
    output_path: str | Path,
    pricing_inputs: dict | None = None,
) -> Path:
    pricing = pricing_inputs or {}
    values = []
    labels = []
    for horizon_days in horizon_days_grid:
        valuation_date, _ = strategy.valuation_date_for_horizon(int(horizon_days))
        value = float(
            strategy.mark_to_market_value(
                [stock_price],
                valuation_date=valuation_date,
                risk_free_rate=pricing.get("risk_free_rate"),
                dividend_yield=pricing.get("dividend_yield"),
            )[0]
        )
        values.append(value)
        labels.append(int(horizon_days))
    fig, ax = plt.subplots(figsize=(9, 5))
    _style_axes(ax)
    ax.plot(labels, values, color="#7c3aed", linewidth=2.4, marker="o", markersize=5.2)
    _apply_title(
        ax,
        f"{strategy.ticker} {strategy.name.replace('_', ' ').title()} Time Sensitivity",
        subtitle="Later checkpoints show how much value survives if the move takes longer than hoped.",
    )
    ax.set_xlabel("Days Since Snapshot")
    ax.set_ylabel("Estimated Position Value ($)")
    _format_money_axis(ax)
    return _finalize(fig, output_path)


def plot_strategy_time_progression(
    strategy: StrategyPosition,
    *,
    horizon_specs,
    spot_cases: dict[str, float],
    output_path: str | Path,
    iv_shift: float = 0.0,
    pricing_inputs: dict | None = None,
) -> Path:
    pricing = pricing_inputs or {}
    labels = [str(label) for label, _ in horizon_specs]
    x_positions = np.arange(len(labels))
    case_order = [label for label in ["bear", "flat", "bull"] if label in spot_cases] or list(spot_cases.keys())[:3]
    case_styles = {
        "bear": {"color": "#b42318", "marker": "o"},
        "flat": {"color": "#475467", "marker": "s"},
        "bull": {"color": "#0f766e", "marker": "^"},
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    _style_axes(ax)
    for case_label in case_order:
        spot_price = float(spot_cases[case_label])
        values = []
        for _, horizon_days in horizon_specs:
            valuation_date, _ = strategy.valuation_date_for_horizon(int(horizon_days))
            value = float(
                strategy.mark_to_market_value(
                    [spot_price],
                    valuation_date=valuation_date,
                    iv_shift=iv_shift,
                    risk_free_rate=pricing.get("risk_free_rate"),
                    dividend_yield=pricing.get("dividend_yield"),
                )[0]
            )
            values.append(value)
        style = case_styles.get(case_label, {"color": "#4C566A", "marker": "o"})
        ax.plot(
            x_positions,
            values,
            label=f"{case_label.replace('_', ' ').title()} @ {spot_price:.2f}",
            color=style["color"],
            marker=style["marker"],
            linewidth=2.4,
            markersize=6.0,
            markeredgecolor="#ffffff",
            markeredgewidth=0.9,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label.replace("_", " ").title() for label in labels])
    _apply_title(
        ax,
        f"{strategy.ticker} {strategy.name.replace('_', ' ').title()} Time Progression",
        subtitle="Bear, flat, and bull checkpoints share one consistent time axis.",
    )
    ax.set_xlabel("Holding Horizon")
    ax.set_ylabel("Estimated Position Value ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_heatmap(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    value_column: str,
    output_path: str | Path,
    title: str,
    x_label: str,
    y_label: str,
    value_label: str,
    x_order: list | None = None,
    y_order: list | None = None,
    cmap: str = "BrBG",
    center_zero: bool = False,
) -> Path:
    pivot = prepare_heatmap_matrix(
        frame,
        x_column=x_column,
        y_column=y_column,
        value_column=value_column,
        x_order=x_order,
        y_order=y_order,
    )
    value_matrix = pivot.to_numpy(dtype=float)
    finite = value_matrix[np.isfinite(value_matrix)]
    fig, ax = plt.subplots(figsize=(10.8, 5.9))
    _style_axes(ax)
    if center_zero and finite.size:
        abs_max = max(abs(float(finite.min())), abs(float(finite.max())))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max) if abs_max > 0 else None
    else:
        norm = None
    image = ax.imshow(value_matrix, aspect="auto", cmap=cmap, norm=norm)
    x_labels = [f"{float(value):.2f}" if isinstance(value, (int, float, np.floating)) else str(value) for value in pivot.columns.tolist()]
    y_labels = [str(value).replace("_", " ").title() for value in pivot.index.tolist()]
    x_positions = np.arange(len(x_labels))
    if len(x_positions) > 10:
        tick_idx = np.linspace(0, len(x_positions) - 1, 9, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([x_labels[index] for index in tick_idx], rotation=0)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=0)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    _apply_title(
        ax,
        title,
        subtitle="Rows and columns use canonical ordering so earlier horizons and lower IV states appear first.",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_label)
    return _finalize(fig, output_path)


def plot_replay_stock_path(
    checkpoint_frame: pd.DataFrame,
    *,
    entry_spot: float,
    expected_move_pct: float | None,
    output_path: str | Path,
    title: str,
) -> Path:
    data = checkpoint_frame.dropna(subset=["matched_stock_date", "matched_stock_close"]).copy()
    if data.empty:
        raise ValueError("Replay stock path needs matched stock rows.")
    data["matched_stock_date"] = pd.to_datetime(data["matched_stock_date"], errors="coerce")
    data = data.dropna(subset=["matched_stock_date"]).sort_values("matched_stock_date")
    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    _style_axes(ax)
    ax.plot(
        data["matched_stock_date"],
        data["matched_stock_close"],
        color="#0F5C57",
        marker="o",
        linewidth=2.6,
        markersize=6.0,
        markeredgecolor="#ffffff",
        markeredgewidth=0.9,
        label="Actual stock path",
    )
    ax.axhline(entry_spot, color="#4C566A", linewidth=1.2, linestyle="--", label="Entry spot")
    if expected_move_pct is not None:
        lower = entry_spot * (1.0 - abs(float(expected_move_pct)))
        upper = entry_spot * (1.0 + abs(float(expected_move_pct)))
        ax.axhspan(lower, upper, color="#E69F00", alpha=0.12, label="Entry expected-move band")
    _apply_title(
        ax,
        title,
        subtitle="The shaded band is the entry expected move; the line shows what the stock actually did through the replay checkpoints.",
    )
    ax.set_xlabel("Checkpoint Date")
    ax.set_ylabel("Stock Price ($)")
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_replay_strategy_value_path(
    checkpoint_frame: pd.DataFrame,
    *,
    output_path: str | Path,
    title: str,
) -> Path:
    data = checkpoint_frame.dropna(subset=["matched_stock_date"]).copy()
    if data.empty:
        raise ValueError("Replay strategy value path needs checkpoint rows.")
    data["matched_stock_date"] = pd.to_datetime(data["matched_stock_date"], errors="coerce")
    data = data.dropna(subset=["matched_stock_date"]).sort_values("matched_stock_date")
    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    _style_axes(ax)
    if "modeled_estimated_value" in data.columns and data["modeled_estimated_value"].notna().any():
        ax.plot(
            data["matched_stock_date"],
            data["modeled_estimated_value"],
            color="#4C566A",
            linestyle="--",
            marker="s",
            linewidth=2.0,
            markersize=5.5,
            label="Modeled value",
        )
    if "selected_value" in data.columns and data["selected_value"].notna().any():
        ax.plot(
            data["matched_stock_date"],
            data["selected_value"],
            color="#0F766E",
            marker="o",
            linewidth=2.8,
            markersize=6.0,
            markeredgecolor="#ffffff",
            markeredgewidth=0.9,
            label="Selected replay value",
        )
    if "exact_observed_value" in data.columns and data["exact_observed_value"].notna().any():
        observed = data.dropna(subset=["exact_observed_value"])
        ax.scatter(
            observed["matched_stock_date"],
            observed["exact_observed_value"],
            color="#0072B2",
            marker="D",
            s=60,
            label="Exact later chain",
            zorder=3,
        )
    _apply_title(
        ax,
        title,
        subtitle="Modeled checkpoints stay separate from exact later-chain observations so the fallback path is visually honest.",
    )
    ax.set_xlabel("Checkpoint Date")
    ax.set_ylabel("Position Value ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_replay_compare_vs_stock(
    compare_frame: pd.DataFrame,
    *,
    mode: str,
    output_path: str | Path,
    title: str,
) -> Path:
    data = compare_frame.loc[compare_frame["mode"] == mode].copy()
    if data.empty:
        raise ValueError("Replay compare-vs-stock plot needs rows for the selected mode.")
    data["matched_stock_date"] = pd.to_datetime(data["matched_stock_date"], errors="coerce")
    data = data.dropna(subset=["matched_stock_date"]).sort_values("matched_stock_date")
    fig, ax = plt.subplots(figsize=(10.4, 5.7))
    _style_axes(ax, zero_line=True)
    ax.plot(
        data["matched_stock_date"],
        data["strategy_profit_loss"],
        color="#0F766E",
        marker="o",
        linewidth=2.8,
        markersize=6.0,
        markeredgecolor="#ffffff",
        markeredgewidth=0.9,
        label="Strategy",
    )
    ax.plot(
        data["matched_stock_date"],
        data["stock_profit_loss"],
        color="#000000",
        marker="s",
        linewidth=2.4,
        markersize=5.8,
        markeredgecolor="#ffffff",
        markeredgewidth=0.9,
        label="Long stock",
    )
    _apply_title(
        ax,
        title,
        subtitle="Use the gap between the two lines to see whether the structure actually earned its complexity versus stock.",
    )
    ax.set_xlabel("Checkpoint Date")
    ax.set_ylabel("Profit / Loss ($)")
    _format_money_axis(ax)
    _place_legend(ax, ncol=2)
    return _finalize(fig, output_path)


def plot_replay_driver_decomposition(
    driver_frame: pd.DataFrame,
    *,
    checkpoint_label: str,
    output_path: str | Path,
    title: str,
) -> Path:
    data = driver_frame.loc[driver_frame["checkpoint"] == checkpoint_label].copy()
    if data.empty:
        raise ValueError("Replay driver decomposition needs rows for the selected checkpoint.")
    fig, ax = plt.subplots(figsize=(9.6, 5.5))
    _style_axes(ax, zero_line=True)
    effects = pd.to_numeric(data["effect"], errors="coerce").fillna(0.0)
    labels = [str(value).replace("_", " ").title() for value in data["component"].tolist()]
    colors = ["#0F766E" if value >= 0 else "#B42318" for value in effects.tolist()]
    ax.bar(labels, effects, color=colors)
    _apply_title(
        ax,
        title,
        subtitle="Positive bars helped the trade; negative bars explain what gave back value.",
    )
    ax.set_xlabel("Driver")
    ax.set_ylabel("Effect ($)")
    _format_money_axis(ax)
    ax.tick_params(axis="x", rotation=15)
    return _finalize(fig, output_path)
