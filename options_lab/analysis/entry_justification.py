"""Entry-justification / required-stock-path decision layer for contract selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .simulation import humanize_named_path
from ..utils import clean_string, finite_or_none


LOWER_IV_VARIANTS = ("mean_reversion_lower", "iv_down_then_stays_low")
HIGHER_IV_VARIANTS = ("mean_reversion_higher", "iv_up_then_down", "earnings_build_then_crush")
ENTRY_CHART_CANDIDATE_LIMIT = 4


@dataclass(frozen=True)
class EntryJustificationOutputs:
    entry_justification_candidates: pd.DataFrame
    required_stock_path_to_buy: pd.DataFrame
    required_move_summary: pd.DataFrame
    required_move_vs_stock: pd.DataFrame
    required_iv_support_summary: pd.DataFrame
    entry_barrier_summary: pd.DataFrame
    entry_justification_markdown: str


def _num(value: Any) -> float:
    return float(finite_or_none(value) or 0.0)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_string(value).lower()
    return text in {"1", "true", "yes", "y"}


def _money_label(value: Any) -> str:
    amount = finite_or_none(value)
    if amount is None:
        return "n/a"
    return f"${float(amount):,.2f}"


def _pct_label(value: Any) -> str:
    amount = finite_or_none(value)
    if amount is None:
        return "n/a"
    return f"{float(amount):+.1f}%"


def _short_expiry_label(value: Any) -> str:
    text = clean_string(value)
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        year = text[2:4]
        month = text[5:7]
        month_names = {
            "01": "Jan",
            "02": "Feb",
            "03": "Mar",
            "04": "Apr",
            "05": "May",
            "06": "Jun",
            "07": "Jul",
            "08": "Aug",
            "09": "Sep",
            "10": "Oct",
            "11": "Nov",
            "12": "Dec",
        }
        return f"{month_names.get(month, month)}-{year}"
    return text or "n/a"


def _short_call_label(row: dict[str, Any]) -> str:
    strike_value = finite_or_none(row.get("strike_label"))
    strike = f"{float(strike_value):g}" if strike_value is not None else clean_string(row.get("strike_label")) or "?"
    expiry = _short_expiry_label(row.get("expiry_date"))
    return f"{strike}C {expiry}"


def _normalize(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if numeric.empty:
        return numeric
    span = float(numeric.max() - numeric.min())
    if span <= 1e-9:
        return pd.Series(50.0, index=numeric.index)
    return ((numeric - float(numeric.min())) / span) * 100.0


def _bucket_order(bucket: Any) -> int:
    mapping = {
        "Buy Now": 0,
        "Watchlist": 1,
        "Prefer Stock Instead": 2,
        "Avoid For Now": 3,
    }
    return mapping.get(clean_string(bucket), 9)


def _select_active_required_rows(
    required_path_rows: pd.DataFrame,
    *,
    goal: str,
    iv_path_name: str,
) -> pd.DataFrame:
    if required_path_rows.empty:
        return pd.DataFrame()
    data = required_path_rows.copy()
    data = data.loc[
        data.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_call")
        & data.get("goal", pd.Series(dtype=str)).astype(str).eq(goal)
    ].copy()
    if data.empty:
        return data
    iv_name = clean_string(iv_path_name).lower()
    preferred = data.loc[
        data.get("iv_variant_kind", pd.Series(dtype=str)).astype(str).str.lower().eq("path")
        & data.get("iv_variant", pd.Series(dtype=str)).astype(str).str.lower().eq(iv_name)
    ].copy()
    if preferred.empty:
        preferred = data.loc[data.get("iv_variant_kind", pd.Series(dtype=str)).astype(str).str.lower().eq("path")].copy()
    if preferred.empty:
        preferred = data.copy()
    preferred["_requested_days"] = pd.to_numeric(preferred.get("requested_days"), errors="coerce").fillna(0).astype(int)
    preferred = (
        preferred.sort_values(["candidate_slug", "_requested_days", "iv_variant"])
        .drop_duplicates(subset=["candidate_slug", "_requested_days"], keep="first")
        .drop(columns=["_requested_days"], errors="ignore")
    )
    return preferred


def _summary_rows_for_candidate(
    required_path_summary: pd.DataFrame,
    *,
    candidate_slug: str,
    goal: str,
) -> pd.DataFrame:
    if required_path_summary.empty:
        return pd.DataFrame()
    data = required_path_summary.copy()
    return data.loc[
        data.get("summary_scope", pd.Series(dtype=str)).astype(str).eq("candidate")
        & data.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_call")
        & data.get("goal", pd.Series(dtype=str)).astype(str).eq(goal)
        & data.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(candidate_slug)
    ].copy()


def _pick_variant_move(summary_rows: pd.DataFrame, variants: tuple[str, ...], *, fallback: float) -> float:
    if summary_rows.empty:
        return float(fallback)
    lower_variants = [clean_string(value).lower() for value in variants]
    data = summary_rows.copy()
    data["iv_variant"] = data.get("iv_variant", pd.Series(dtype=str)).astype(str).str.lower()
    for variant in lower_variants:
        subset = data.loc[data["iv_variant"].eq(variant)].copy()
        if not subset.empty:
            return float(pd.to_numeric(subset.get("required_stock_price_at_target"), errors="coerce").fillna(np.nan).iloc[0])
    return float(fallback)


def _pick_required_row(candidate_rows: pd.DataFrame, *, horizon: str | None = None) -> dict[str, Any]:
    if candidate_rows.empty:
        return {}
    working = candidate_rows.copy()
    working["_requested_days"] = pd.to_numeric(working.get("requested_days"), errors="coerce").fillna(0).astype(int)
    if horizon:
        match = working.loc[working.get("horizon", pd.Series(dtype=str)).astype(str).eq(horizon)].copy()
        if not match.empty:
            return match.sort_values("_requested_days").iloc[0].to_dict()
    return working.sort_values("_requested_days").iloc[-1].to_dict()


def _entry_barrier_label(row: dict[str, Any]) -> str:
    if _bool(row.get("stock_still_better_even_if_path_hits")):
        return "stock still better"
    if _bool(row.get("requires_fast_move")) and _num(row.get("entry_barrier_score")) >= 65:
        return "needs fast move"
    if _bool(row.get("needs_iv_support")) and _num(row.get("entry_barrier_score")) >= 58:
        return "needs IV support"
    if _num(row.get("entry_barrier_score")) >= 72 or _num(row.get("required_move_pct_target")) >= 28:
        return "too demanding"
    return "more forgiving"


def _what_has_to_happen(row: dict[str, Any]) -> str:
    parts: list[str] = []
    if finite_or_none(row.get("required_price_1m")) is not None:
        parts.append(f"about {_money_label(row.get('required_price_1m'))} by 1 month")
    if finite_or_none(row.get("required_price_target")) is not None:
        parts.append(
            f"{_money_label(row.get('required_price_target'))} by {clean_string(row.get('decisive_date')) or clean_string(row.get('target_date'))}"
        )
    sentence = "Needs " + (" and ".join(parts) if parts else "a clearer upside path") + "."
    if _bool(row.get("requires_fast_move")):
        sentence += " The move probably needs to start early enough to outrun theta."
    if _bool(row.get("needs_iv_support")):
        sentence += " A softer or supportive IV path would help keep the premium burden reasonable."
    return sentence


def _entry_warning(row: dict[str, Any]) -> str:
    if _bool(row.get("stock_still_better_even_if_path_hits")):
        return "Stock still looks cleaner even if the required path is achieved."
    if _bool(row.get("target_beyond_expiry")) or _bool(row.get("weak_horizon_fit")):
        return "Target timing runs beyond the clean contract window."
    if _bool(row.get("requires_fast_move")):
        return "A slower move quickly weakens the call because timing and theta matter."
    if _bool(row.get("needs_iv_support")):
        return "Lower IV makes the same directional thesis materially less attractive."
    if _num(row.get("required_move_pct_target")) >= 25:
        return "This strike/expiry asks for a large move versus the available time."
    return "The call is viable only if the stock path and entry stay disciplined."


def _iv_requirement_label(row: dict[str, Any]) -> str:
    lower_penalty = _num(row.get("lower_iv_move_penalty_pct"))
    higher_relief = _num(row.get("higher_iv_move_relief_pct"))
    if lower_penalty >= 5.0 or _num(row.get("iv_dependence_risk")) >= 55:
        return "needs IV support"
    if lower_penalty <= 1.5 and _num(row.get("lower_iv_resilience_score")) >= 65:
        return "handles lower IV"
    if higher_relief >= 4.0:
        return "benefits from higher IV"
    return "IV helpful but secondary"


def _iv_requirement_note(row: dict[str, Any]) -> str:
    flat_move = _num(row.get("required_move_pct_target"))
    lower_move = _num(row.get("lower_iv_required_move_pct"))
    higher_move = _num(row.get("higher_iv_required_move_pct"))
    if lower_move > flat_move + 0.5:
        return f"Lower IV raises the required target move from {_pct_label(flat_move)} to {_pct_label(lower_move)}."
    if higher_move < flat_move - 0.5:
        return f"Friendlier IV cuts the required target move from {_pct_label(flat_move)} to {_pct_label(higher_move)}."
    return "Changing IV regimes does not materially alter the required target path."


def _build_markdown(
    *,
    ticker: str,
    entry_candidates: pd.DataFrame,
    move_summary: pd.DataFrame,
    iv_summary: pd.DataFrame,
    barrier_summary: pd.DataFrame,
) -> str:
    watchlist = entry_candidates.loc[entry_candidates.get("action_bucket").astype(str).eq("Watchlist")].head(5)
    too_demanding = entry_candidates.loc[
        entry_candidates.get("entry_barrier_label").astype(str).isin(["too demanding", "stock still better"])
    ].sort_values(["entry_barrier_score", "action_priority_rank"], ascending=[False, True]).head(5)
    forgiving = barrier_summary.sort_values(["entry_barrier_score", "action_priority_rank"]).head(4)
    fast = entry_candidates.loc[entry_candidates.get("requires_fast_move").fillna(False)].head(4)
    iv_needed = iv_summary.loc[iv_summary.get("iv_requirement_label").astype(str).eq("needs IV support")].head(4)
    stock_better = move_summary.loc[move_summary.get("stock_still_better_even_if_path_hits").fillna(False)].head(4)

    def bullet_rows(frame: pd.DataFrame, text_fn) -> list[str]:
        if frame.empty:
            return ["- No clear examples in this group under current assumptions."]
        return [f"- {text_fn(row)}" for row in frame.to_dict("records")]

    lines = [
        f"# {ticker} Entry Justification / Required Stock Path",
        "",
        "Bullish long-call-first read of what the stock actually has to do before buying the option looks justified under the active assumptions.",
        "",
        "## What Has To Happen For These Calls To Be Worth Buying",
        "",
        *bullet_rows(
            watchlist if not watchlist.empty else forgiving,
            lambda row: f"`{clean_string(row.get('candidate_short_label'))}`: {clean_string(row.get('what_has_to_happen'))}",
        ),
        "",
        "## Which Calls Require Too Much",
        "",
        *bullet_rows(
            too_demanding,
            lambda row: f"`{clean_string(row.get('candidate_short_label'))}`: {clean_string(row.get('entry_warning'))}",
        ),
        "",
        "## Which Calls Are More Forgiving",
        "",
        *bullet_rows(
            forgiving,
            lambda row: f"`{clean_string(row.get('candidate_short_label'))}`: target move {_pct_label(row.get('required_move_pct_target'))} over about {int(_num(row.get('timing_window_days')))} days.",
        ),
        "",
        "## Which Calls Need Fast Confirmation",
        "",
        *bullet_rows(
            fast,
            lambda row: f"`{clean_string(row.get('candidate_short_label'))}`: {clean_string(row.get('speed_requirement_note'))}",
        ),
        "",
        "## Which Calls Mainly Need Better IV / Better Entry",
        "",
        *bullet_rows(
            iv_needed if not iv_needed.empty else watchlist,
            lambda row: f"`{clean_string(row.get('candidate_short_label'))}`: {clean_string(row.get('iv_requirement_note') or row.get('entry_warning'))}",
        ),
        "",
        "## When Stock Is Still Better Even If The Path Is “Right”",
        "",
        *bullet_rows(
            stock_better,
            lambda row: f"`{clean_string(row.get('candidate_short_label'))}`: {clean_string(row.get('stock_vs_option_read'))}",
        ),
        "",
        "## How To Read The Entry Charts",
        "",
        "- `charts/required_stock_path_to_buy.png`: compare each selected call's required stock path with the active assumed path.",
        "- `charts/required_move_speed_vs_magnitude.png`: left/right is required move pace, up/down is total upside still needed by the key decision date.",
        "- `charts/required_move_vs_stock_chart.png`: shows when stock still dominates even if the call's required path is broadly met.",
        "- `charts/strike_expiry_entry_barrier_map.png`: compact barrier read across strikes and expiries.",
        "- `charts/iv_support_requirement_chart.png`: shows which calls become materially harder if IV cools.",
        "",
        "## Best Next Files To Open",
        "",
        "- `charts/required_stock_path_to_buy.png`",
        "- `charts/required_move_speed_vs_magnitude.png`",
        "- `charts/required_move_vs_stock_chart.png`",
        "- `tables/required_move_summary.csv`",
        "- `tables/required_iv_support_summary.csv`",
        "- `tables/entry_barrier_summary.csv`",
    ]
    return (
        "\n".join(lines)
        .replace("â€œRightâ€\x9d", "\"Right\"")
        .replace("â€œRightâ€", "\"Right\"")
        .replace("â€œ", "\"")
        .replace("â€", "\"")
        .replace("“", "\"")
        .replace("”", "\"")
    )


def build_entry_justification(
    *,
    ticker: str,
    goal: str,
    target_price: float,
    target_date: date,
    active_iv_path_name: str,
    candidate_comparison: pd.DataFrame,
    required_path_rows: pd.DataFrame,
    required_path_summary: pd.DataFrame,
    action_board_candidates: pd.DataFrame,
) -> EntryJustificationOutputs:
    if candidate_comparison.empty:
        empty = pd.DataFrame()
        return EntryJustificationOutputs(
            entry_justification_candidates=empty,
            required_stock_path_to_buy=empty,
            required_move_summary=empty,
            required_move_vs_stock=empty,
            required_iv_support_summary=empty,
            entry_barrier_summary=empty,
            entry_justification_markdown="",
        )

    active_required_rows = _select_active_required_rows(required_path_rows, goal=goal, iv_path_name=active_iv_path_name)
    candidate_base = candidate_comparison.loc[
        candidate_comparison.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_call")
    ].copy()
    if candidate_base.empty:
        empty = pd.DataFrame()
        return EntryJustificationOutputs(
            entry_justification_candidates=empty,
            required_stock_path_to_buy=empty,
            required_move_summary=empty,
            required_move_vs_stock=empty,
            required_iv_support_summary=empty,
            entry_barrier_summary=empty,
            entry_justification_markdown="",
        )

    action_long_calls = action_board_candidates.loc[
        action_board_candidates.get("strategy_family", pd.Series(dtype=str)).astype(str).str.lower().eq("long_call")
    ].copy()
    if not action_long_calls.empty:
        action_columns = [
            "candidate_slug",
            "action_bucket",
            "action_priority_rank",
            "action_confidence",
            "action_score",
            "candidate_conviction_score",
            "robustness_score",
            "stock_relative_score",
            "time_decay_risk",
            "iv_dependence_risk",
            "trust_penalty",
            "headline_reason",
            "why_this_is_interesting_now",
            "what_is_hurting_this_candidate",
            "main_trigger",
            "main_warning",
            "source_trust_label",
            "source_quality_note",
            "lower_iv_resilience_score",
            "high_iv_dependency_rate",
        ]
        candidate_base = candidate_base.merge(
            action_long_calls[[col for col in action_columns if col in action_long_calls.columns]].copy(),
            on="candidate_slug",
            how="left",
            suffixes=("", "_action"),
        )

    records: list[dict[str, Any]] = []
    required_path_chart_rows: list[dict[str, Any]] = []

    for row in candidate_base.to_dict("records"):
        slug = clean_string(row.get("candidate_slug"))
        candidate_required = active_required_rows.loc[
            active_required_rows.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(slug)
        ].copy()
        if candidate_required.empty:
            continue
        candidate_required = candidate_required.sort_values("requested_days")
        target_row = _pick_required_row(candidate_required)
        one_week = _pick_required_row(candidate_required, horizon="1w")
        one_month = _pick_required_row(candidate_required, horizon="1m")
        three_month = _pick_required_row(candidate_required, horizon="3m")
        summary_rows = _summary_rows_for_candidate(required_path_summary, candidate_slug=slug, goal=goal)
        required_price_target = _num(target_row.get("required_stock_price"))
        entry_spot = _num(row.get("spot_price"))
        required_move_target_pct = _num(target_row.get("required_move_pct_from_entry")) * 100.0
        required_move_1m_pct = _num(one_month.get("required_move_pct_from_entry")) * 100.0 if one_month else required_move_target_pct
        required_move_3m_pct = _num(three_month.get("required_move_pct_from_entry")) * 100.0 if three_month else required_move_target_pct
        timing_window_days = max(int(_num(target_row.get("requested_days")) or _num(row.get("effective_days")) or _num(row.get("requested_days"))), 1)
        move_pace_pct_per_month = required_move_target_pct / max(timing_window_days / 30.0, 0.35)
        flat_required_price = _pick_variant_move(summary_rows, (clean_string(active_iv_path_name).lower(), "flat"), fallback=required_price_target)
        lower_required_price = _pick_variant_move(summary_rows, LOWER_IV_VARIANTS, fallback=flat_required_price)
        higher_required_price = _pick_variant_move(summary_rows, HIGHER_IV_VARIANTS, fallback=flat_required_price)
        flat_required_move_pct = ((flat_required_price - entry_spot) / entry_spot * 100.0) if entry_spot > 0 else required_move_target_pct
        lower_required_move_pct = ((lower_required_price - entry_spot) / entry_spot * 100.0) if entry_spot > 0 else flat_required_move_pct
        higher_required_move_pct = ((higher_required_price - entry_spot) / entry_spot * 100.0) if entry_spot > 0 else flat_required_move_pct
        lower_penalty = lower_required_move_pct - flat_required_move_pct
        higher_relief = flat_required_move_pct - higher_required_move_pct
        requires_fast_move = (
            _num(row.get("time_decay_risk")) >= 65.0
            or _num(row.get("delayed_move_value_change")) <= -25.0
            or _bool(row.get("target_beyond_expiry"))
            or _bool(row.get("weak_horizon_fit"))
        )
        needs_iv_support = (
            _num(row.get("iv_dependence_risk")) >= 55.0
            or _num(row.get("lower_iv_resilience_score")) < 55.0
            or lower_penalty >= 4.0
            or _num(row.get("high_iv_dependency_rate")) > 0.25
        )
        stock_still_better_even_if_path_hits = bool(target_row.get("assumed_clears_required_path")) and _num(row.get("difference_vs_stock")) <= 0.0
        records.append(
            {
                "candidate_slug": slug,
                "candidate_label": clean_string(row.get("candidate_label")),
                "candidate_short_label": _short_call_label(row),
                "action_bucket": clean_string(row.get("action_bucket")) or "Watchlist",
                "action_priority_rank": int(_num(row.get("action_priority_rank")) or 999),
                "action_confidence": clean_string(row.get("action_confidence")),
                "expiry_date": clean_string(row.get("expiry_date")),
                "strike_label": clean_string(row.get("strike_label")),
                "moneyness_bucket": clean_string(row.get("moneyness_bucket")),
                "source_trust_label": clean_string(row.get("source_trust_label")),
                "goal": clean_string(goal),
                "stock_path_name": clean_string(row.get("stock_path_name")),
                "stock_path_label": humanize_named_path(clean_string(row.get("stock_path_name")), kind="stock"),
                "iv_path_name": clean_string(active_iv_path_name),
                "iv_path_label": humanize_named_path(clean_string(active_iv_path_name), kind="iv"),
                "required_price_1w": finite_or_none(one_week.get("required_stock_price")),
                "required_price_1m": finite_or_none(one_month.get("required_stock_price")),
                "required_price_3m": finite_or_none(three_month.get("required_stock_price")),
                "required_price_target": finite_or_none(target_row.get("required_stock_price")),
                "required_move_pct_1m": round(required_move_1m_pct, 2),
                "required_move_pct_3m": round(required_move_3m_pct, 2),
                "required_move_pct_target": round(required_move_target_pct, 2),
                "timing_window_days": timing_window_days,
                "move_pace_pct_per_month": round(move_pace_pct_per_month, 2),
                "first_cleared_horizon": clean_string(row.get("first_cleared_horizon")),
                "required_path_difficulty": clean_string(row.get("required_path_difficulty")),
                "path_gap_at_target": finite_or_none(row.get("path_gap_at_target")),
                "assumed_clears_required_at_target": bool(target_row.get("assumed_clears_required_path")),
                "time_decay_risk": round(_num(row.get("time_decay_risk")), 2),
                "iv_dependence_risk": round(_num(row.get("iv_dependence_risk")), 2),
                "lower_iv_resilience_score": round(_num(row.get("lower_iv_resilience_score")), 2),
                "required_move_pct_flat_iv": round(flat_required_move_pct, 2),
                "lower_iv_required_move_pct": round(lower_required_move_pct, 2),
                "higher_iv_required_move_pct": round(higher_required_move_pct, 2),
                "lower_iv_move_penalty_pct": round(lower_penalty, 2),
                "higher_iv_move_relief_pct": round(higher_relief, 2),
                "difference_vs_stock": finite_or_none(row.get("difference_vs_stock")),
                "difference_vs_stock_return_pct": finite_or_none(row.get("difference_vs_stock_return_pct")),
                "stock_relative_score": round(_num(row.get("stock_relative_score")), 2),
                "target_beyond_expiry": _bool(row.get("target_beyond_expiry")),
                "weak_horizon_fit": _bool(row.get("weak_horizon_fit")),
                "clamped_to_expiry": _bool(row.get("clamped_to_expiry")),
                "requires_fast_move": requires_fast_move,
                "needs_iv_support": needs_iv_support,
                "stock_still_better_even_if_path_hits": stock_still_better_even_if_path_hits,
                "decisive_date": clean_string(target_row.get("valuation_date")) or target_date.isoformat(),
                "target_date": target_date.isoformat(),
                "headline_reason": clean_string(row.get("headline_reason")),
                "why_this_is_interesting_now": clean_string(row.get("why_this_is_interesting_now")),
                "what_is_hurting_this_candidate": clean_string(row.get("what_is_hurting_this_candidate")),
                "main_trigger": clean_string(row.get("main_trigger")),
                "main_warning": clean_string(row.get("main_warning")),
                "speed_requirement_note": (
                    "Needs earlier confirmation before theta and expiry-clamp risk start dominating."
                    if requires_fast_move
                    else "Longer runway makes this call more tolerant of a slower climb."
                ),
            }
        )

    entry_candidates = pd.DataFrame.from_records(records)
    if entry_candidates.empty:
        empty = pd.DataFrame()
        return EntryJustificationOutputs(
            entry_justification_candidates=empty,
            required_stock_path_to_buy=empty,
            required_move_summary=empty,
            required_move_vs_stock=empty,
            required_iv_support_summary=empty,
            entry_barrier_summary=empty,
            entry_justification_markdown="",
        )

    entry_candidates["entry_barrier_score"] = (
        _normalize(entry_candidates["required_move_pct_target"]) * 0.38
        + _normalize(entry_candidates["move_pace_pct_per_month"]) * 0.20
        + _normalize(entry_candidates["time_decay_risk"]) * 0.15
        + _normalize(entry_candidates["iv_dependence_risk"]) * 0.12
        + _normalize(entry_candidates.get("trust_penalty", pd.Series(0.0, index=entry_candidates.index))) * 0.07
        + entry_candidates["stock_still_better_even_if_path_hits"].astype(float) * 12.0
        + entry_candidates["target_beyond_expiry"].astype(float) * 10.0
        + entry_candidates["weak_horizon_fit"].astype(float) * 7.0
    ).clip(0.0, 100.0)
    entry_candidates["iv_requirement_label"] = entry_candidates.apply(_iv_requirement_label, axis=1)
    entry_candidates["iv_requirement_note"] = entry_candidates.apply(_iv_requirement_note, axis=1)
    entry_candidates["what_has_to_happen"] = entry_candidates.apply(_what_has_to_happen, axis=1)
    entry_candidates["entry_warning"] = entry_candidates.apply(_entry_warning, axis=1)
    entry_candidates["entry_barrier_label"] = entry_candidates.apply(_entry_barrier_label, axis=1)
    entry_candidates["stock_vs_option_read"] = np.where(
        entry_candidates["stock_still_better_even_if_path_hits"],
        "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
        np.where(
            pd.to_numeric(entry_candidates.get("difference_vs_stock"), errors="coerce").fillna(0.0) > 0.0,
            "If the required path is met, the call shows modeled edge versus stock.",
            "The path helps, but stock still keeps the cleaner baseline read.",
        ),
    )
    entry_candidates = entry_candidates.sort_values(
        ["action_bucket", "action_priority_rank", "entry_barrier_score"],
        key=lambda series: series.map(_bucket_order) if series.name == "action_bucket" else series,
    ).reset_index(drop=True)
    entry_candidates["entry_display_rank"] = np.arange(1, len(entry_candidates.index) + 1)

    chart_candidates = entry_candidates.head(ENTRY_CHART_CANDIDATE_LIMIT).copy()
    for row in chart_candidates.to_dict("records"):
        candidate_required = active_required_rows.loc[
            active_required_rows.get("candidate_slug", pd.Series(dtype=str)).astype(str).eq(clean_string(row.get("candidate_slug")))
        ].copy()
        candidate_required = candidate_required.sort_values("requested_days")
        for candidate_row in candidate_required.to_dict("records"):
            required_path_chart_rows.append(
                {
                    "candidate_slug": row["candidate_slug"],
                    "candidate_short_label": row["candidate_short_label"],
                    "action_bucket": row["action_bucket"],
                    "entry_display_rank": row["entry_display_rank"],
                    "iv_path_label": row["iv_path_label"],
                    "series_kind": "required_path",
                    "requested_days": int(_num(candidate_row.get("requested_days"))),
                    "date": clean_string(candidate_row.get("valuation_date")),
                    "stock_price": finite_or_none(candidate_row.get("required_stock_price")),
                    "entry_barrier_label": row["entry_barrier_label"],
                    "stock_vs_option_read": row["stock_vs_option_read"],
                }
            )
            required_path_chart_rows.append(
                {
                    "candidate_slug": row["candidate_slug"],
                    "candidate_short_label": row["candidate_short_label"],
                    "action_bucket": row["action_bucket"],
                    "entry_display_rank": row["entry_display_rank"],
                    "iv_path_label": row["iv_path_label"],
                    "series_kind": "assumed_path",
                    "requested_days": int(_num(candidate_row.get("requested_days"))),
                    "date": clean_string(candidate_row.get("valuation_date")),
                    "stock_price": finite_or_none(candidate_row.get("assumed_stock_price")),
                    "entry_barrier_label": row["entry_barrier_label"],
                    "stock_vs_option_read": row["stock_vs_option_read"],
                }
            )

    required_stock_path_to_buy = pd.DataFrame.from_records(required_path_chart_rows)
    required_move_summary = entry_candidates[
        [
            "candidate_slug",
            "candidate_short_label",
            "candidate_label",
            "action_bucket",
            "action_priority_rank",
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
        ]
    ].copy()
    required_move_vs_stock = entry_candidates[
        [
            "candidate_slug",
            "candidate_short_label",
            "candidate_label",
            "action_bucket",
            "action_priority_rank",
            "required_move_pct_target",
            "timing_window_days",
            "assumed_clears_required_at_target",
            "difference_vs_stock",
            "difference_vs_stock_return_pct",
            "stock_relative_score",
            "stock_still_better_even_if_path_hits",
            "stock_vs_option_read",
        ]
    ].copy()
    required_iv_support_summary = entry_candidates[
        [
            "candidate_slug",
            "candidate_short_label",
            "candidate_label",
            "action_bucket",
            "action_priority_rank",
            "required_move_pct_flat_iv",
            "lower_iv_required_move_pct",
            "higher_iv_required_move_pct",
            "lower_iv_move_penalty_pct",
            "higher_iv_move_relief_pct",
            "lower_iv_resilience_score",
            "iv_dependence_risk",
            "iv_requirement_label",
            "iv_requirement_note",
        ]
    ].copy()
    entry_barrier_summary = entry_candidates[
        [
            "candidate_slug",
            "candidate_short_label",
            "candidate_label",
            "action_bucket",
            "expiry_date",
            "strike_label",
            "moneyness_bucket",
            "action_priority_rank",
            "entry_barrier_score",
            "entry_barrier_label",
            "required_move_pct_target",
            "timing_window_days",
            "requires_fast_move",
            "iv_requirement_label",
            "stock_vs_option_read",
            "source_trust_label",
        ]
    ].copy().sort_values(["entry_barrier_score", "action_priority_rank"])

    markdown = _build_markdown(
        ticker=ticker,
        entry_candidates=entry_candidates,
        move_summary=required_move_vs_stock,
        iv_summary=required_iv_support_summary,
        barrier_summary=entry_barrier_summary,
    )
    return EntryJustificationOutputs(
        entry_justification_candidates=entry_candidates,
        required_stock_path_to_buy=required_stock_path_to_buy,
        required_move_summary=required_move_summary,
        required_move_vs_stock=required_move_vs_stock,
        required_iv_support_summary=required_iv_support_summary,
        entry_barrier_summary=entry_barrier_summary,
        entry_justification_markdown=markdown,
    )
