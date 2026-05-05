import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from matplotlib.figure import Figure

from options_lab.analysis import contract_selection as contract_selection_module
from options_lab.analysis import (
    build_contract_selection_analysis,
    build_scenario_analysis,
    publish_analysis_bundle,
    write_analysis_bundle,
)
from options_lab.analysis.contract_selection import (
    CHAIN_OVERVIEW_CARD_KEYS,
    _build_long_call_required_path_outputs,
    _build_thesis_mode_outputs,
    _discover_candidates_for_chain,
    _long_call_required_spot_for_threshold,
    _path_case_defaults,
    _path_case_iv_variants,
    _path_case_stock_variants,
    _path_view_filename,
    _required_path_realism_bucket,
    _select_curated_single_option_decision_paths,
    _select_long_call_expiry_view_rows,
    _selector_cards,
)
from options_lab.analysis.simulation import (
    build_path_grid,
    build_stock_path_example,
    build_stock_path_library_rows,
    stock_path_family_metadata,
)
from options_lab.io import load_chain
from options_lab import plots as plots_module
from options_lab.plots import plot_option_required_paths, plot_single_option_decision_view


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.fixture(scope="module")
def late_breakout_contract_selection_result():
    return build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
        stock_path_preset="late_breakout",
        iv_path_preset="flat",
    )


def test_path_case_variants_keep_custom_paths_separate_from_builtin_presets():
    horizon_specs = [
        {"label": "entry", "requested_days": 0},
        {"label": "30d", "requested_days": 30},
        {"label": "90d", "requested_days": 90},
    ]

    stock_variants = _path_case_stock_variants(
        horizon_specs=horizon_specs,
        target_price=20.0,
        target_horizon_label="90d",
        stock_path_name="custom_stock_path",
        stock_path_points={"entry": 16.0, "30d": 19.0, "90d": 22.0},
    )
    iv_variants = _path_case_iv_variants(
        horizon_specs=horizon_specs,
        iv_shift_points=0.0,
        target_horizon_label="90d",
        iv_path_name="custom_iv_path",
        iv_path_points={"entry": 0.0, "30d": -0.05, "90d": -0.10},
    )["path_preset"]

    assert stock_variants["custom_stock_path"]["30d"] == 19.0
    assert stock_variants["custom_stock_path"]["90d"] == 22.0
    assert stock_variants["slow_bull"]["30d"] != 19.0
    assert iv_variants["custom_iv_path"]["30d"] == -0.05
    assert iv_variants["custom_iv_path"]["90d"] == -0.10
    assert iv_variants["flat"]["90d"] == 0.0


def test_required_path_chart_renders_unavailable_stock_only_edge(temp_analysis_root: Path):
    output_path = temp_analysis_root / "required_paths_unavailable.png"
    frame = pd.DataFrame(
        [
            {
                "contract_label": "15C Dec-26",
                "threshold_multiple": 1.5,
                "path_id": "contract__1_5x__fast_breakout",
                "path_label": "Fast Breakout (1.5x)",
                "path_family": "fast_breakout",
                "family_display_order": 1,
                "date": "2026-04-12",
                "days_from_snapshot": 0,
                "stock_price": None,
                "option_value": None,
                "option_return_pct": None,
                "stock_return_pct": None,
                "option_vs_stock_multiple": None,
                "clears_threshold": False,
                "realism_bucket": "unavailable",
                "status": "needs_iv_or_entry_support",
                "failure_driver": "entry_premium",
            }
        ]
    )

    result = plot_option_required_paths(
        frame,
        output_path=output_path,
        title="Required Paths For 15C Dec-26 To Beat Stock By 1.5x / 2.0x",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_required_path_chart_caption_labels_holding_period_and_expiry(monkeypatch, temp_analysis_root: Path):
    output_path = temp_analysis_root / "required_paths_holding_period.png"
    figure_text: list[str] = []

    original_figure_text = Figure.text

    def capture_figure_text(self, x, y, s, *args, **kwargs):
        figure_text.append(str(s))
        return original_figure_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(Figure, "text", capture_figure_text)
    frame = pd.DataFrame(
        [
            {
                "contract_label": "15C Dec-26",
                "threshold_multiple": 1.5,
                "strike": 15.0,
                "entry_spot": 15.23,
                "snapshot_date": "2026-04-12",
                "analysis_horizon_date": "2026-07-15",
                "chart_horizon_date": "2026-12-18",
                "path_terminal_date": "2026-12-18",
                "option_expiry": "2026-12-18",
                "option_expiry_date": "2026-12-18",
                "terminal_basis": "option_expiry",
                "chart_horizon_basis": "option_expiry",
                "path_id": "contract__1_5x__slow_grind",
                "path_label": "Slow Grind (1.5x)",
                "path_family": "slow_grind",
                "family_display_order": 1,
                "date": "2026-04-12",
                "days_from_snapshot": 0,
                "stock_price": 15.23,
                "option_value": 341.96,
                "option_return_pct": 0.0,
                "stock_return_pct": 0.0,
                "option_vs_stock_multiple": None,
                "is_checkpoint_marker": True,
                "is_peak_option_return": False,
                "clears_threshold": False,
                "realism_bucket": "plausible",
                "status": "solved_normal_range",
                "failure_driver": "timing",
            },
            {
                "contract_label": "15C Dec-26",
                "threshold_multiple": 2.0,
                "strike": 15.0,
                "entry_spot": 15.23,
                "snapshot_date": "2026-04-12",
                "analysis_horizon_date": "2026-07-15",
                "chart_horizon_date": "2026-12-18",
                "path_terminal_date": "2026-12-18",
                "option_expiry": "2026-12-18",
                "option_expiry_date": "2026-12-18",
                "terminal_basis": "option_expiry",
                "chart_horizon_basis": "option_expiry",
                "path_id": "contract__2_0x__slow_grind",
                "path_label": "Slow Grind (2.0x)",
                "path_family": "slow_grind",
                "family_display_order": 1,
                "date": "2026-12-18",
                "days_from_snapshot": 250,
                "stock_price": 17.31,
                "option_value": 435.0,
                "option_return_pct": 0.2721,
                "stock_return_pct": 0.1366,
                "option_vs_stock_multiple": 1.99,
                "is_checkpoint_marker": True,
                "is_peak_option_return": True,
                "clears_threshold": False,
                "realism_bucket": "plausible",
                "status": "solved_normal_range",
                "failure_driver": "timing",
            },
        ]
    )

    result = plot_option_required_paths(
        frame,
        output_path=output_path,
        title="Required Paths For 15C Dec-26 To Beat Stock By 1.5x / 2.0x",
    )

    assert result == output_path
    assert output_path.exists()
    combined_text = " ".join(figure_text)
    assert "Snapshot: 2026-04-12" in combined_text
    assert "Option expiry: 2026-12-18" in combined_text
    assert "Chart horizon: option expiry" in combined_text
    assert "Threshold rule: option return must be at least 1.5x / 2.0x stock return" in combined_text
    assert "Path terminal: 2026-07-15" not in combined_text


def test_required_path_date_ticks_use_month_day_and_two_week_spacing():
    tick_map = pd.DataFrame(
        {
            "requested_days": list(range(0, 251)),
            "date": [(pd.Timestamp("2026-04-12") + pd.Timedelta(days=day)).date().isoformat() for day in range(0, 251)],
        }
    )

    ticks = plots_module._required_path_date_tick_rows(tick_map)
    labels = [row["label"] for row in ticks]
    positions = [int(row["requested_days"]) for row in ticks]

    assert labels[:3] == ["Apr 12", "Apr 26", "May 10"]
    intervals = [positions[index] - positions[index - 1] for index in range(1, len(positions))]
    assert intervals
    assert min(intervals) >= 14
    assert max(intervals) <= 28


def test_required_path_reference_label_offsets_keep_spot_and_strike_separate():
    spot_offset, strike_offset = plots_module._required_path_reference_label_offsets(15.00, 15.03)

    assert spot_offset[1] > 0
    assert strike_offset[1] < 0
    assert abs(spot_offset[1] - strike_offset[1]) >= 24


def test_required_path_realism_bucket_uses_decimal_return_units():
    assert _required_path_realism_bucket(0.40, "solved_normal_range") == "plausible"
    assert _required_path_realism_bucket(0.75, "solved_aggressive_range") == "aggressive"
    assert _required_path_realism_bucket(2.50, "solved_extreme_range") == "extreme"
    assert _required_path_realism_bucket(3.00, "solved_absurd_range") == "absurd"


def _fake_long_call_spec(*, candidate_slug: str, strike: float, premium: float, expiry: str = "2026-04-17") -> dict:
    position = SimpleNamespace(
        initial_outlay=float(premium),
        expiry_date=pd.Timestamp(expiry).date(),
        option_legs=[SimpleNamespace(strike=float(strike), base_iv=0.55)],
    )
    return {
        "candidate_slug": candidate_slug,
        "candidate_label": f"Long Call {expiry} {strike:.2f}",
        "strategy_family": "long_call",
        "position": position,
        "stock_baseline": SimpleNamespace(),
    }


def _patch_intrinsic_required_path_eval(monkeypatch):
    def fake_adjusted_evaluation(spec, *, spot_price: float, horizon_days: int, comparison_capital: float, premium_used: float, **_kwargs):
        strike = float(spec["position"].option_legs[0].strike)
        estimated_value = max(float(spot_price) - strike, 0.0) * 100.0
        profit = 0.0 if int(horizon_days) <= 0 else estimated_value - float(premium_used)
        stock_profit = (float(spot_price) - 10.0) * 100.0
        return {
            "estimated_value": round(estimated_value, 4),
            "profit_loss": round(profit, 4),
            "stock_profit_loss": round(stock_profit, 4),
            "difference_vs_stock": round(profit - stock_profit, 4),
            "requested_days": int(horizon_days),
            "effective_days": int(horizon_days),
        }

    monkeypatch.setattr(contract_selection_module, "_single_option_adjusted_evaluation", fake_adjusted_evaluation)


def test_long_call_required_path_engine_uses_contract_strike_and_numeric_paths(monkeypatch):
    _patch_intrinsic_required_path_eval(monkeypatch)
    candidate_slug = "long-call-2026-04-17-14-00-exact-snapshot"
    outputs = _build_long_call_required_path_outputs(
        ticker="GPRE",
        specs=[_fake_long_call_spec(candidate_slug=candidate_slug, strike=14.0, premium=300.0)],
        candidate_rows=pd.DataFrame(
            [
                {
                    "candidate_slug": candidate_slug,
                    "candidate_label": "Long Call 2026-04-17 14.00",
                    "candidate_short_label": "14C Apr-26",
                    "strategy_family": "long_call",
                    "primary_strike": 14.0,
                    "strike_label": "14.00",
                    "expiry_date": "2026-04-17",
                    "active_candidate_rank": 1,
                    "objective_score": 50.0,
                }
            ]
        ),
        snapshot_date=pd.Timestamp("2026-04-12").date(),
        target_date=pd.Timestamp("2026-04-17").date(),
        target_horizon_label="5d",
        entry_spot=10.0,
        target_price=20.0,
        active_iv_path_points={"entry": 0.0, "5d": 0.0},
        comparison_capital=1000.0,
        minimum_outperformance_multiple=1.5,
        strong_outperformance_multiple=2.0,
        minimum_edge_stock_return_pct=0.05,
        entry_price_mode="mid",
    )

    summary = outputs["required_path_core_summary"]
    paths = outputs["required_paths_by_option"]
    summary_14c = summary.loc[summary["contract_label"].eq("14C Apr-26")]
    solved_paths = paths.loc[paths["status"].astype(str).str.startswith("solved")]
    day_zero = paths.loc[paths["days_from_snapshot"].eq(0)]

    assert not summary_14c.empty
    assert set(summary_14c["strike"]) == {14.0}
    assert summary_14c["required_move_pct"].notna().all()
    assert summary_14c["status"].astype(str).str.startswith("solved").all()
    assert solved_paths["stock_price"].notna().any()
    assert solved_paths["option_value"].notna().any()
    assert solved_paths["option_return_pct"].notna().any()
    assert day_zero["option_return_pct"].fillna(999).eq(0.0).all()
    assert not day_zero["clears_threshold"].fillna(True).any()


def test_long_call_required_path_engine_writes_absurd_numeric_moves(monkeypatch):
    _patch_intrinsic_required_path_eval(monkeypatch)
    candidate_slug = "long-call-2026-04-17-20-00-exact-snapshot"
    outputs = _build_long_call_required_path_outputs(
        ticker="GPRE",
        specs=[_fake_long_call_spec(candidate_slug=candidate_slug, strike=20.0, premium=300.0)],
        candidate_rows=pd.DataFrame(
            [
                {
                    "candidate_slug": candidate_slug,
                    "candidate_label": "Long Call 2026-04-17 20.00",
                    "candidate_short_label": "20C Apr-26",
                    "strategy_family": "long_call",
                    "primary_strike": 20.0,
                    "strike_label": "20.00",
                    "expiry_date": "2026-04-17",
                    "active_candidate_rank": 1,
                    "objective_score": 50.0,
                }
            ]
        ),
        snapshot_date=pd.Timestamp("2026-04-12").date(),
        target_date=pd.Timestamp("2026-04-17").date(),
        target_horizon_label="5d",
        entry_spot=10.0,
        target_price=20.0,
        active_iv_path_points={"entry": 0.0, "5d": 0.0},
        comparison_capital=1000.0,
        minimum_outperformance_multiple=1.5,
        strong_outperformance_multiple=2.0,
        minimum_edge_stock_return_pct=0.05,
        entry_price_mode="mid",
    )

    summary = outputs["required_path_core_summary"]
    strong = summary.loc[summary["threshold_multiple"].eq(2.0)].iloc[0]
    strong_paths = outputs["required_paths_by_option"].loc[
        outputs["required_paths_by_option"]["threshold_multiple"].eq(2.0)
    ]

    assert strong["status"] == "solved_absurd_range"
    assert strong["realism_bucket"] == "absurd"
    assert float(strong["required_move_pct"]) > 2.5
    assert pd.to_numeric(strong_paths["stock_price"], errors="coerce").notna().any()
    assert pd.to_numeric(strong_paths["option_value"], errors="coerce").notna().any()


def test_long_call_required_path_engine_clamps_path_dates_to_option_expiry(monkeypatch):
    _patch_intrinsic_required_path_eval(monkeypatch)
    candidate_slug = "long-call-2026-04-17-14-00-exact-snapshot"
    outputs = _build_long_call_required_path_outputs(
        ticker="GPRE",
        specs=[_fake_long_call_spec(candidate_slug=candidate_slug, strike=14.0, premium=300.0)],
        candidate_rows=pd.DataFrame(
            [
                {
                    "candidate_slug": candidate_slug,
                    "candidate_label": "Long Call 2026-04-17 14.00",
                    "candidate_short_label": "14C Apr-26",
                    "strategy_family": "long_call",
                    "primary_strike": 14.0,
                    "strike_label": "14.00",
                    "expiry_date": "2026-04-17",
                    "active_candidate_rank": 1,
                    "objective_score": 50.0,
                }
            ]
        ),
        snapshot_date=pd.Timestamp("2026-04-12").date(),
        target_date=pd.Timestamp("2026-07-15").date(),
        target_horizon_label="94d",
        entry_spot=10.0,
        target_price=20.0,
        active_iv_path_points={"entry": 0.0, "94d": 0.0},
        comparison_capital=1000.0,
        minimum_outperformance_multiple=1.5,
        strong_outperformance_multiple=2.0,
        minimum_edge_stock_return_pct=0.05,
        entry_price_mode="mid",
    )

    paths = outputs["required_paths_by_option"]
    summary = outputs["required_path_core_summary"]
    assert pd.to_numeric(paths["days_from_snapshot"], errors="coerce").max() == 5
    assert pd.to_datetime(paths["date"]).max().date() == pd.Timestamp("2026-04-17").date()
    assert set(summary["latest_valid_date"]) <= {"2026-04-17"}


def test_long_call_required_path_engine_writes_smooth_marker_rows_and_peak_summary(monkeypatch):
    _patch_intrinsic_required_path_eval(monkeypatch)
    candidate_slug = "long-call-2026-12-18-15-00-exact-snapshot"
    outputs = _build_long_call_required_path_outputs(
        ticker="GPRE",
        specs=[_fake_long_call_spec(candidate_slug=candidate_slug, strike=15.0, premium=300.0, expiry="2026-12-18")],
        candidate_rows=pd.DataFrame(
            [
                {
                    "candidate_slug": candidate_slug,
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "candidate_short_label": "15C Dec-26",
                    "strategy_family": "long_call",
                    "primary_strike": 15.0,
                    "strike_label": "15.00",
                    "expiry_date": "2026-12-18",
                    "active_candidate_rank": 1,
                    "objective_score": 50.0,
                }
            ]
        ),
        snapshot_date=pd.Timestamp("2026-04-12").date(),
        target_date=pd.Timestamp("2026-07-15").date(),
        target_horizon_label="94d",
        entry_spot=10.0,
        target_price=20.0,
        active_iv_path_points={"entry": 0.0, "94d": 0.0},
        comparison_capital=1000.0,
        minimum_outperformance_multiple=1.5,
        strong_outperformance_multiple=2.0,
        minimum_edge_stock_return_pct=0.05,
        entry_price_mode="mid",
    )

    paths = outputs["required_paths_by_option"]
    peaks = outputs["required_path_peak_summary"]
    exit_ladder = outputs["required_path_exit_ladder"]
    entry_sensitivity = outputs["required_path_entry_sensitivity"]
    iv_sensitivity = outputs["required_path_iv_sensitivity"]
    entry_iv_matrix = outputs["required_path_entry_iv_matrix"]
    sell_hold = outputs["required_path_sell_hold_summary"]
    tables_html = outputs["required_path_tables_html"]
    tables_md = outputs["required_path_tables_markdown"]
    sample_path = paths.loc[
        (paths["threshold_multiple"].eq(1.5))
        & (paths["path_family"].eq("slow_grind"))
    ].copy()

    assert len(sample_path) > 80
    assert sample_path["days_from_snapshot"].max() == 250
    assert {
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
    } <= set(paths.columns)
    assert sample_path["snapshot_date"].eq("2026-04-12").all()
    assert sample_path["analysis_horizon_date"].eq("2026-07-15").all()
    assert sample_path["chart_horizon_date"].eq("2026-12-18").all()
    assert sample_path["path_terminal_date"].eq("2026-12-18").all()
    assert sample_path["option_expiry"].eq("2026-12-18").all()
    assert sample_path["option_expiry_date"].eq("2026-12-18").all()
    assert sample_path["terminal_basis"].eq("option_expiry").all()
    assert sample_path["chart_horizon_basis"].eq("option_expiry").all()
    assert sample_path["shape_template"].eq("slow_grind").all()
    marker_days = set(pd.to_numeric(sample_path.loc[sample_path["is_checkpoint_marker"], "days_from_snapshot"], errors="coerce"))
    assert {0, 7, 14, 30, 250} <= marker_days
    assert any(124 <= day <= 126 for day in marker_days)
    assert any(187 <= day <= 188 for day in marker_days)
    time_to_expiry = pd.to_numeric(sample_path.sort_values("days_from_snapshot")["time_to_expiry_days"], errors="coerce")
    assert time_to_expiry.iloc[0] == 250
    assert time_to_expiry.iloc[-1] == 0
    assert time_to_expiry.is_monotonic_decreasing
    expiry_row = sample_path.sort_values("days_from_snapshot").iloc[-1]
    assert float(expiry_row["option_value"]) == pytest.approx(float(expiry_row["intrinsic_value"]), abs=0.01)
    assert float(expiry_row["time_value"]) == pytest.approx(0.0, abs=0.01)
    family_names = set(paths["path_family"].astype(str))
    assert {
        "fast_breakout",
        "slow_grind",
        "late_acceleration",
        "down_first_recovery",
        "rally_retrace_finish",
        "violent_absurd",
        "expiry_only",
    } <= family_names
    shape_signatures = {}
    for family, family_frame in paths.loc[paths["threshold_multiple"].eq(1.5)].groupby("path_family"):
        ordered = family_frame.sort_values("days_from_snapshot")
        prices = pd.to_numeric(ordered["stock_price"], errors="coerce").dropna()
        if len(prices) < 3:
            continue
        start = float(prices.iloc[0])
        end = float(prices.iloc[-1])
        span = end - start if abs(end - start) > 1e-9 else 1.0
        normalized = tuple(round(float((value - start) / span), 3) for value in prices.iloc[[0, len(prices) // 4, len(prices) // 2, -1]])
        shape_signatures[str(family)] = normalized
    assert len(set(shape_signatures.values())) >= 5
    down_recovery = paths.loc[(paths["threshold_multiple"].eq(1.5)) & (paths["path_family"].eq("down_first_recovery"))]
    rally_retrace = paths.loc[(paths["threshold_multiple"].eq(1.5)) & (paths["path_family"].eq("rally_retrace_finish"))].sort_values("days_from_snapshot")
    assert pd.to_numeric(down_recovery["stock_price"], errors="coerce").min() < 10.0
    assert pd.to_numeric(rally_retrace["stock_price"], errors="coerce").diff().min() < 0
    assert not peaks.empty
    assert {
        "peak_option_return_pct",
        "peak_option_value",
        "peak_date",
        "peak_days_from_snapshot",
        "stock_price_at_peak",
        "option_vs_stock_multiple_at_peak",
        "terminal_option_return_pct",
        "terminal_option_vs_stock_multiple",
    } <= set(peaks.columns)
    solved_peaks = peaks.loc[peaks["status"].astype(str).str.startswith("solved")]
    assert pd.to_numeric(solved_peaks["peak_option_return_pct"], errors="coerce").notna().all()
    assert pd.to_numeric(solved_peaks["stock_price_at_peak"], errors="coerce").notna().all()
    assert (
        pd.to_numeric(solved_peaks["peak_option_return_pct"], errors="coerce")
        >= pd.to_numeric(solved_peaks["terminal_option_return_pct"], errors="coerce")
    ).all()
    assert not exit_ladder.empty
    assert {
        "ticker",
        "strike",
        "expiry",
        "target_absolute_option_return_pct",
        "first_hit_date",
        "stock_price_at_first_hit",
        "option_return_pct_at_first_hit",
        "option_value_at_first_hit",
        "hit_status",
        "exit_return_label",
        "exit_return_pct",
        "first_exit_date",
        "first_exit_days_from_snapshot",
        "stock_price_at_exit",
        "option_return_pct_at_exit",
    } <= set(exit_ladder.columns)
    solved_ladder = exit_ladder.loc[
        exit_ladder["status"].astype(str).str.startswith("solved")
        & exit_ladder["first_exit_date"].notna()
    ]
    assert not solved_ladder.empty
    assert pd.to_numeric(solved_ladder["stock_price_at_exit"], errors="coerce").notna().all()
    assert 3.0 in set(pd.to_numeric(exit_ladder["target_absolute_option_return_pct"], errors="coerce"))

    assert not entry_sensitivity.empty
    assert set(pd.to_numeric(entry_sensitivity["entry_shift_pct"], errors="coerce").round(2)) == {-0.50, -0.25, 0.0, 0.25, 0.50, 1.0}
    assert {
        "ticker",
        "contract_label",
        "strike",
        "expiry",
        "threshold_multiple",
        "path_family",
        "entry_shift_pct",
        "adjusted_entry_premium",
        "required_terminal_stock_price",
        "required_move_pct",
        "realism_bucket",
        "verdict",
    } <= set(entry_sensitivity.columns)
    assert not iv_sensitivity.empty
    assert set(pd.to_numeric(iv_sensitivity["iv_shift_vol_points"], errors="coerce").round(2)) == {-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50}
    assert {
        "ticker",
        "contract_label",
        "strike",
        "expiry",
        "threshold_multiple",
        "path_family",
        "iv_shift_vol_points",
        "adjusted_iv",
        "required_terminal_stock_price",
        "required_move_pct",
        "realism_bucket",
        "verdict",
    } <= set(iv_sensitivity.columns)
    assert not entry_iv_matrix.empty
    assert {"entry_shift_pct", "iv_shift_vol_points", "adjusted_entry_premium", "adjusted_iv"} <= set(entry_iv_matrix.columns)
    assert not sell_hold.empty
    assert {
        "ticker",
        "contract_label",
        "strike",
        "expiry",
        "threshold_multiple",
        "path_family",
        "peak_option_return_pct",
        "option_return_2w_after_peak",
        "option_return_1m_after_peak",
        "expiry_option_return_pct",
        "decay_from_peak_to_expiry_pct",
        "interpretation",
    } <= set(sell_hold.columns)
    assert set(sell_hold["interpretation"].astype(str)) - {""}

    for section in [
        "Contract inputs",
        "Required move summary",
        "Path family comparison",
        "Entry premium sensitivity",
        "IV sensitivity",
        "Entry premium x IV matrix",
        "Sell / hold pressure",
        "Absolute option-return exit ladder",
    ]:
        assert section in tables_html
    assert "1.5x / 2.0x are relative to stock return, not absolute option return." in tables_html
    assert "NaN" not in tables_html
    assert "How to read the tables" in tables_md
    assert "relative to stock return" in tables_md


def test_long_call_required_path_solver_marks_unsolved_only_after_extreme_search(monkeypatch):
    _patch_intrinsic_required_path_eval(monkeypatch)
    spec = _fake_long_call_spec(
        candidate_slug="long-call-2026-04-17-20-00-exact-snapshot",
        strike=20.0,
        premium=1000.0,
    )

    required_spot, evaluation, status = _long_call_required_spot_for_threshold(
        spec,
        horizon_days=5,
        iv_shift_points=0.0,
        comparison_capital=1000.0,
        premium_used=1000.0,
        edge_multiple=1.5,
        entry_spot=10.0,
        minimum_stock_return_floor_pct=0.05,
    )

    assert required_spot is None
    assert status == "unsolved_after_extreme_search"
    assert evaluation["last_extreme_search_stock_price"] > 100.0


def test_path_case_defaults_respect_active_goal_and_iv_variant():
    defaults = _path_case_defaults(
        goal="itm_1c",
        active_iv_path_name="custom_iv_path",
        iv_shift_points=-0.05,
        default_strategy_family="long_call",
        default_candidate_within_family="long-call-15",
    )

    assert defaults["default_goal"] == "itm_1c"
    assert defaults["default_iv_variant"] == "custom_iv_path"
    assert defaults["default_iv_mode"] == "path_preset"


def test_full_quoted_snapshot_candidate_discovery_respects_top_n_strikes():
    chain = load_chain(DATA_ROOT / "GPRE" / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv")

    rows, specs, _warnings = _discover_candidates_for_chain(
        chain,
        scope="exact_snapshot",
        target_price=20.0,
        target_date=pd.Timestamp("2026-07-15").date(),
        target_horizon_label="3m",
        requested_days=94,
        iv_shift_points=0.0,
        comparison_capital=1000.0,
        strategy_families=["long_call"],
        strike_selection_mode="top_n",
        top_n_strikes=1,
        source_snapshot_date="2026-04-12",
        source_expiry_date="2026-04-17",
        source_storage_location="ibkr_full_quoted_snapshot",
        source_snapshot_file=str(chain.source_path),
        source_quote_coverage_pct=1.0,
        source_quote_usable=True,
    )

    assert [row["strategy_family"] for row in rows].count("long_call") == 1
    assert [spec["strategy_family"] for spec in specs].count("long_call") == 1


def test_stock_path_family_metadata_is_stable():
    early = stock_path_family_metadata("early_breakout_to_target")
    gap = stock_path_family_metadata("earnings_gap_up_then_fade")
    library = build_stock_path_library_rows(active_path_name="late_breakout")

    assert early["path_family"] == "early_rally"
    assert early["timing_shape"] == "front_loaded_upside"
    assert gap["path_family"] == "earnings_gap"
    assert gap["outcome_bias"] == "event_sensitive"
    assert not library.empty
    assert {
        "minimum_required_path",
        "early_rally",
        "late_rally",
        "steady_grind_up",
        "false_breakout",
        "recovery",
        "earnings_gap",
        "quarter_pullback",
    } <= set(library["path_family"])
    assert library.loc[library["path_name"].eq("late_breakout"), "is_active_assumed"].any()


def test_curated_single_option_decision_path_selection_prefers_outcome_and_family_diversity():
    path_pool = []
    outcome_rows = []
    cases = [
        ("minimum_required_path", "minimum_required_path", "minimum_required_path", "Minimum Required Path", "contract_specific_threshold", "clear_option_win", 125.0, 2.5),
        ("early_rally_path__early_breakout_to_target", "early_rally_path", "early_breakout_to_target", "Early Rally", "front_loaded_upside", "clear_option_win", 180.0, 3.1),
        ("steady_grind_up_path__slow_grind_to_target", "steady_grind_up_path", "slow_grind_to_target", "Steady Grind-Up", "smooth_uptrend", "wins_but_not_enough", 45.0, 1.2),
        ("late_rally_path__late_breakout_to_target", "late_rally_path", "late_breakout_to_target", "Late Rally", "back_loaded_upside", "stock_better", -35.0, 0.4),
        ("false_breakout_failed_path__false_breakout_then_recover", "false_breakout_failed_path", "false_breakout_then_recover", "False Breakout", "spike_then_giveback", "fail_too_narrow_or_expiry_issue", -120.0, None),
        ("recovery_path__down_then_recover_to_target", "recovery_path", "down_then_recover_to_target", "Recovery", "down_then_recover", "wins_but_not_enough", 20.0, 1.1),
        ("earnings_gap_path__earnings_gap_up_then_fade", "earnings_gap_path", "earnings_gap_up_then_fade", "Earnings Gap", "event_gap_then_follow_through", "stock_better", -10.0, 0.8),
        ("range_bound_near_flat__range_bound_near_flat", "range_bound_near_flat", "range_bound_near_flat", "Range-Bound", "sideways_chop", "fail_too_narrow_or_expiry_issue", -80.0, None),
    ]
    family_by_label = {
        "Minimum Required Path": "minimum_required_path",
        "Early Rally": "early_rally",
        "Steady Grind-Up": "steady_grind_up",
        "Late Rally": "late_rally",
        "False Breakout": "false_breakout",
        "Recovery": "recovery",
        "Earnings Gap": "earnings_gap",
        "Range-Bound": "range_bound",
    }
    for idx, (decision_path_id, role, name, family_label, timing, outcome, difference, multiple) in enumerate(cases, start=1):
        path_pool.append(
            {
                "decision_path_id": decision_path_id,
                "path_role": role,
                "path_name": name,
                "path_label": name.replace("_", " ").title(),
                "path_family": family_by_label[family_label],
                "path_family_label": family_label,
                "timing_shape": timing,
                "outcome_bias": "test",
                "path_description": "Synthetic path.",
                "selection_reason": "Synthetic library candidate.",
                "path_points": [],
            }
        )
        outcome_rows.append(
            {
                "decision_path_id": decision_path_id,
                "path_role": role,
                "path_name": name,
                "path_family": family_by_label[family_label],
                "path_family_label": family_label,
                "timing_shape": timing,
                "outcome_label": outcome,
                "difference_vs_stock": difference,
                "outperformance_multiple": multiple,
                "display_order": idx,
            }
        )

    selected = _select_curated_single_option_decision_paths(path_pool, pd.DataFrame(outcome_rows))
    selected_again = _select_curated_single_option_decision_paths(path_pool, pd.DataFrame(outcome_rows))

    assert 5 <= len(selected) <= 8
    assert [path["decision_path_id"] for path in selected] == [path["decision_path_id"] for path in selected_again]
    assert selected[0]["path_role"] == "minimum_required_path"
    assert set(SINGLE["outcome_label"] for SINGLE in selected) >= {
        "clear_option_win",
        "wins_but_not_enough",
        "stock_better",
        "fail_too_narrow_or_expiry_issue",
    }
    assert [path["display_order"] for path in selected] == list(range(1, len(selected) + 1))
    assert all(path["selection_reason"] for path in selected)
    assert len({path["path_family"] for path in selected}) >= 5


def test_single_option_decision_view_renders_stock_better_paths_with_required_edges(temp_workspace_root: Path):
    output_path = temp_workspace_root / "single_option_decision_view.png"
    summary = pd.DataFrame(
        [
            {
                "ticker": "GPRE",
                "candidate_short_label": "15C Dec-26",
                "premium_used": 320.0,
                "base_iv": 0.55,
                "breakeven": 18.2,
                "max_loss": 320.0,
                "dte": 240,
                "exit_rule": "sell_on_thesis_completion",
            }
        ]
    )
    representative_paths = pd.DataFrame(
        [
            {
                "decision_path_id": path_id,
                "path_label": label,
                "display_order": order,
                "requested_days": days,
                "date": f"2026-{month:02d}-15",
                "spot_price": price,
            }
            for order, (path_id, label, prices) in enumerate(
                [
                    ("late", "Late Rally", [16.0, 17.0, 20.0]),
                    ("steady", "Steady Grind", [16.0, 18.0, 21.0]),
                    ("false", "False Breakout", [16.0, 22.0, 18.5]),
                ],
                start=1,
            )
            for days, month, price in zip([0, 45, 90], [4, 5, 7], prices)
        ]
    )
    path_outcomes = pd.DataFrame(
        [
            {
                "decision_path_id": path_id,
                "path_label": label,
                "outcome_label": "stock_better",
                "display_order": order,
                "difference_vs_stock": -gap,
                "stock_profit_loss": 400.0 + gap,
                "outperformance_multiple": 0.8,
                "qualifies_as_winning_path_family": False,
            }
            for order, (path_id, label, gap) in enumerate(
                [
                    ("late", "Late Rally", 80.0),
                    ("steady", "Steady Grind", 45.0),
                    ("false", "False Breakout", 130.0),
                ],
                start=1,
            )
        ]
    )
    required_edge_paths = pd.DataFrame(
        [
            {
                "edge_path_name": edge_name,
                "edge_label": edge_label,
                "edge_multiple": multiple,
                "display_order": display_order,
                "requested_days": days,
                "date": f"2026-{month:02d}-15",
                "required_stock_price": price,
                "return_pct": (price / 16.0) - 1.0,
                "iv_shift_points": 0.0,
                "required_option_profit_loss": 600.0,
                "status": "solved",
            }
            for display_order, (edge_name, edge_label, multiple, prices) in enumerate(
                [
                    ("required_path_to_beat_stock_1_5x", "Required 1.5x Edge", 1.5, [16.0, 21.0, 25.0]),
                    ("required_path_to_beat_stock_2_0x", "Required 2.0x Strong Edge", 2.0, [16.0, 23.0, 28.0]),
                ],
                start=1,
            )
            for days, month, price in zip([0, 45, 90], [4, 5, 7], prices)
        ]
    )
    closest_edge = pd.DataFrame(
        [
            {
                "decision_path_id": "steady",
                "path_label": "Steady Grind",
                "annotation_text": "Closest miss needs about $4.00 more stock move or earlier timing.",
                "edge_gap_to_1_5x_dollars": -45.0,
            }
        ]
    )
    edge_gap = pd.DataFrame(
        [
            {
                "decision_path_id": "steady",
                "path_label": "Steady Grind",
                "path_family_label": "Steady Grind-Up",
                "timing_shape": "smooth_uptrend",
                "outcome_label": "stock_better",
                "exit_stock_price": 21.0,
                "required_stock_price_1_5x": 25.0,
                "extra_stock_move_needed_1_5x": 4.0,
                "edge_gap_to_1_5x_dollars": -45.0,
                "timing_gap_note": "needs_more_stock_move",
                "is_closest_to_edge": True,
            }
        ]
    )

    result = plot_single_option_decision_view(
        summary=summary,
        representative_paths=representative_paths,
        path_outcomes=path_outcomes,
        required_edge_paths=required_edge_paths,
        edge_gap_by_path_family=edge_gap,
        closest_representative_path_to_edge=closest_edge,
        iv_sensitivity=pd.DataFrame(
            [{"iv_mode_label": "Base IV", "display_order": 1, "difference_vs_stock": -45.0, "sensitivity_note": "Base IV."}]
        ),
        entry_sensitivity=pd.DataFrame(
            [{"entry_scenario_label": "Reference", "display_order": 1, "premium_used": 320.0, "average_difference_vs_stock": -45.0}]
        ),
        summary_bullets=pd.DataFrame([{"bullet_order": 1, "bullet_text": "No representative path beats stock by the required threshold."}]),
        output_path=output_path,
        title="Synthetic single-option decision",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_single_option_entry_profit_is_zero_after_premium_anchor(monkeypatch):
    def fake_evaluate_at_point(*args, **kwargs):
        return {
            "estimated_value": 150.0,
            "stock_profit_loss": 0.0,
            "profit_loss": 50.0,
            "requested_days": kwargs.get("horizon_days", 0),
            "effective_days": kwargs.get("horizon_days", 0),
        }

    monkeypatch.setattr(contract_selection_module, "_evaluate_at_point", fake_evaluate_at_point)

    evaluation = contract_selection_module._single_option_adjusted_evaluation(
        {},
        spot_price=15.23,
        horizon_days=0,
        iv_shift_points=0.0,
        comparison_capital=1000.0,
        premium_used=100.0,
    )

    assert evaluation["profit_loss"] == 0.0
    assert evaluation["difference_vs_stock"] == 0.0
    assert evaluation["entry_premium_anchor_applied"] is True
    assert evaluation["mark_to_market_profit_loss_before_entry_anchor"] == 50.0


def test_single_option_required_edge_does_not_clear_on_zero_stock_profit(monkeypatch):
    def fake_adjusted_evaluation(*args, spot_price: float, **kwargs):
        stock_profit = 0.0 if float(spot_price) <= 10.0 else 100.0
        return {
            "estimated_value": 120.0,
            "profit_loss": 20.0,
            "stock_profit_loss": stock_profit,
            "difference_vs_stock": 20.0 - stock_profit,
            "requested_days": kwargs.get("horizon_days", 30),
            "effective_days": kwargs.get("horizon_days", 30),
        }

    monkeypatch.setattr(contract_selection_module, "_single_option_adjusted_evaluation", fake_adjusted_evaluation)

    required_spot, _evaluation, status = contract_selection_module._single_option_required_spot_for_edge(
        {},
        horizon_days=30,
        iv_shift_points=0.0,
        comparison_capital=1000.0,
        premium_used=100.0,
        edge_multiple=1.5,
        entry_spot=10.0,
        target_price=15.0,
        strike_value=15.0,
        observed_max_spot=16.0,
        minimum_stock_profit_floor=50.0,
    )

    assert required_spot is None
    assert status in {"needs_iv_or_entry_support", "unreached_in_search_range"}


def test_single_option_required_edge_dates_use_snapshot_plus_requested_days(monkeypatch):
    def fake_required_spot(*args, horizon_days: int, entry_spot: float, **kwargs):
        return (
            float(entry_spot) + float(horizon_days) / 10.0,
            {"stock_profit_loss": 100.0, "profit_loss": 200.0},
            "solved",
        )

    monkeypatch.setattr(contract_selection_module, "_single_option_required_spot_for_edge", fake_required_spot)
    snapshot = pd.Timestamp("2026-04-12").date()
    frame = contract_selection_module._single_option_required_edge_path_frame(
        {},
        candidate_slug="long-call",
        candidate_short_label="15C Dec-26",
        representative_paths=pd.DataFrame(
            [
                {"requested_days": 0, "date": "2026-04-19", "spot_price": 15.23},
                {"requested_days": 30, "date": "2026-04-13", "spot_price": 20.0},
                {"requested_days": 60, "date": "2026-04-10", "spot_price": 22.0},
            ]
        ),
        snapshot_date=snapshot,
        target_date=pd.Timestamp("2026-07-15").date(),
        target_price=20.0,
        entry_spot=15.23,
        active_iv_path={},
        comparison_capital=1000.0,
        premium_used=341.96,
        edge_multiple=1.5,
        minimum_stock_profit_floor=50.0,
    )

    assert frame["requested_days"].tolist() == [0, 30, 60]
    assert frame["date"].tolist() == ["2026-04-12", "2026-05-12", "2026-06-11"]


def test_single_option_edge_gap_driver_separates_stock_move_from_timing_iv_entry():
    path_outcomes = pd.DataFrame(
        [
            {
                "decision_path_id": "needs_stock",
                "path_label": "Needs Stock",
                "outcome_label": "stock_better",
                "display_order": 1,
                "exit_requested_days": 30,
                "exit_stock_price": 18.0,
                "profit_loss": 50.0,
                "stock_profit_loss": 150.0,
                "outperformance_multiple": 0.33,
                "first_cross_above_strike_day": 20,
            },
            {
                "decision_path_id": "late_enough_stock",
                "path_label": "Late Enough Stock",
                "outcome_label": "stock_better",
                "display_order": 2,
                "exit_requested_days": 60,
                "exit_stock_price": 24.0,
                "profit_loss": 110.0,
                "stock_profit_loss": 220.0,
                "outperformance_multiple": 0.50,
                "first_cross_above_strike_day": 55,
            },
            {
                "decision_path_id": "flat_tiny_gap",
                "path_label": "Flat Tiny Gap",
                "outcome_label": "fail_too_narrow_or_expiry_issue",
                "display_order": 3,
                "exit_requested_days": 5,
                "exit_stock_price": 15.4,
                "profit_loss": -2.0,
                "stock_profit_loss": 8.0,
                "outperformance_multiple": None,
                "first_cross_above_strike_day": None,
            },
        ]
    )
    min_edge_path = pd.DataFrame(
        [
            {"requested_days": 5, "required_stock_price": None, "status": "no_meaningful_edge_at_this_horizon"},
            {"requested_days": 30, "required_stock_price": 20.0, "status": "solved"},
            {"requested_days": 60, "required_stock_price": 20.0, "status": "solved"},
        ]
    )

    edge_gap, closest = contract_selection_module._single_option_edge_gap_outputs(
        path_outcomes=path_outcomes,
        min_edge_path=min_edge_path,
        strong_edge_path=min_edge_path,
        minimum_outperformance_multiple=1.5,
        strong_outperformance_multiple=2.0,
        minimum_stock_profit_floor=50.0,
    )

    by_id = edge_gap.set_index("decision_path_id")
    assert by_id.loc["needs_stock", "edge_failure_driver"] == "stock_move"
    assert by_id.loc["needs_stock", "extra_stock_move_needed"] == 2.0
    assert by_id.loc["late_enough_stock", "edge_failure_driver"] in {"timing", "iv_or_entry", "timing_iv_or_entry"}
    assert bool(by_id.loc["late_enough_stock", "earlier_timing_needed"]) is True
    assert bool(by_id.loc["late_enough_stock", "iv_support_needed"]) is True
    assert bool(by_id.loc["late_enough_stock", "entry_discount_needed"]) is True
    assert closest.iloc[0]["decision_path_id"] == "late_enough_stock"


def test_single_option_decision_view_renders_when_required_edge_unavailable(temp_workspace_root: Path):
    output_path = temp_workspace_root / "single_option_decision_view_unavailable.png"
    result = plot_single_option_decision_view(
        summary=pd.DataFrame(
            [
                {
                    "ticker": "GPRE",
                    "candidate_short_label": "15C Dec-26",
                    "premium_used": 341.96,
                    "base_iv": 0.55,
                    "breakeven": 18.42,
                    "max_loss": 341.96,
                    "dte": 250,
                    "exit_rule": "sell_on_thesis_completion",
                }
            ]
        ),
        representative_paths=pd.DataFrame(
            [
                {"decision_path_id": "flat", "path_label": "Flat", "display_order": 1, "requested_days": 0, "date": "2026-04-12", "spot_price": 15.23},
                {"decision_path_id": "flat", "path_label": "Flat", "display_order": 1, "requested_days": 30, "date": "2026-05-12", "spot_price": 15.5},
            ]
        ),
        path_outcomes=pd.DataFrame(
            [
                {
                    "decision_path_id": "flat",
                    "path_label": "Flat",
                    "outcome_label": "fail_too_narrow_or_expiry_issue",
                    "qualifies_as_winning_path_family": False,
                    "difference_vs_stock": -20.0,
                    "stock_profit_loss": 10.0,
                }
            ]
        ),
        required_edge_paths=pd.DataFrame(
            [
                {
                    "edge_path_name": "required_path_to_beat_stock_1_5x",
                    "edge_label": "Required 1.5x Edge",
                    "edge_multiple": 1.5,
                    "display_order": 1,
                    "requested_days": 30,
                    "date": "2026-05-12",
                    "required_stock_price": None,
                    "status": "no_meaningful_edge_at_this_horizon",
                }
            ]
        ),
        edge_gap_by_path_family=pd.DataFrame(
            [
                {
                    "decision_path_id": "flat",
                    "path_label": "Flat",
                    "edge_failure_driver": "no_meaningful_stock_edge",
                    "is_closest_to_edge": False,
                }
            ]
        ),
        closest_representative_path_to_edge=pd.DataFrame(),
        iv_sensitivity=pd.DataFrame([{"iv_mode_label": "Base IV", "display_order": 1, "difference_vs_stock": -20.0}]),
        entry_sensitivity=pd.DataFrame([{"entry_scenario_label": "Reference", "display_order": 1, "average_difference_vs_stock": -20.0}]),
        summary_bullets=pd.DataFrame([{"bullet_order": 1, "bullet_text": "No representative path clears the threshold."}]),
        output_path=output_path,
        title="Unavailable edge",
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_thesis_mode_keeps_target_galleries_when_no_long_calls_exist():
    outputs = _build_thesis_mode_outputs(
        ticker="PBI",
        specs=[],
        candidate_rows=pd.DataFrame(
            [
                {
                    "candidate_slug": "long-stock",
                    "strategy_family": "long_stock",
                    "candidate_label": "Long Stock Baseline",
                }
            ]
        ),
        snapshot_date=pd.Timestamp("2026-04-22").date(),
        thesis_target_price=25.0,
        thesis_target_date=pd.Timestamp("2026-12-18").date(),
        thesis_horizon_label="240d",
        entry_spot=16.0,
        comparison_capital=1000.0,
        objective_mode="max_return_at_target",
        downside_tolerance="medium",
        simplicity_preference="medium",
    )

    assert not outputs["thesis_path_gallery"].empty
    assert not outputs["thesis_iv_gallery"].empty
    assert outputs["thesis_mode_candidates"].empty
    assert "Target: `$25.00` by `2026-12-18`" in outputs["thesis_mode_markdown"]
    assert "charts/current_vs_justified_premium.png" not in outputs["thesis_mode_markdown"]


@pytest.mark.slow
def test_contract_selection_analysis_builds_candidates_path_cases_and_selector_outputs():
    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
        stock_path_points="entry:16,1m:18,3m:22",
        iv_path_points="entry:0.00,1m:-0.05,3m:-0.10",
    )

    assert not result.candidate_summary.empty
    assert not result.ranked_candidates.empty
    assert not result.compare_vs_stock.empty
    assert not hasattr(result, "selection_heatmap_rows")
    assert not hasattr(result, "selection_slice_rows")
    assert not result.required_path_rows.empty
    assert not result.required_path_summary.empty
    assert not result.required_path_core_summary.empty
    assert not result.required_paths_by_option.empty
    assert not result.required_path_family_summary.empty
    assert not result.required_path_peak_summary.empty
    assert "What each call needs" in result.required_path_summary_markdown
    assert "Top Required-Path Candidates" in result.top_required_path_candidates_markdown
    assert not result.assumed_path_trace_rows.empty
    assert not result.iv_path_trace_rows.empty
    assert not result.compare_vs_stock_path_rows.empty
    assert not result.iv_path_sensitivity_summary.empty
    assert not result.path_risk_summary.empty
    assert not result.path_case_chart_rows.empty
    assert not result.path_case_strategy_rows.empty
    assert not result.path_case_family_rankings.empty
    assert not result.path_case_candidate_rankings.empty
    assert not result.strategy_selector_rows.empty
    assert not result.strategy_selector_rankings.empty
    assert not result.family_comparison.empty
    assert not result.candidate_comparison.empty
    assert not result.strike_comparison.empty
    assert not result.expiry_comparison.empty
    assert not result.decision_highlights.empty
    assert not result.decision_highlights_explanations.empty
    assert not result.candidate_robustness_summary.empty
    assert not result.candidate_tradeoff_matrix.empty
    assert not result.stock_vs_option_takeaways.empty
    assert not result.highlights_score_breakdown.empty
    assert "Decision Snapshot" in result.highlights_markdown
    assert not result.action_board_candidates.empty
    assert not result.action_board_explanations.empty
    assert not result.action_board_score_breakdown.empty
    assert not result.decision_triggers.empty
    assert not result.bullish_long_call_action_board.empty
    assert not result.bullish_long_call_watchlist.empty
    assert not result.bullish_long_call_triggers.empty
    assert not result.top_candidate_cards.empty
    assert not result.stock_preference_summary.empty
    assert not result.entry_justification_candidates.empty
    assert not result.required_stock_path_to_buy.empty
    assert not result.required_move_summary.empty
    assert not result.required_move_vs_stock.empty
    assert not result.required_iv_support_summary.empty
    assert not result.entry_barrier_summary.empty
    assert result.thesis_target_price == 20.0
    assert not result.thesis_path_gallery.empty
    assert not result.thesis_iv_gallery.empty
    assert not result.thesis_mode_candidates.empty
    assert not result.thesis_candidate_ranking.empty
    assert not result.max_justified_premium.empty
    assert not result.current_vs_justified_premium.empty
    assert not result.thesis_required_move_summary.empty
    assert not result.thesis_stock_vs_option_summary.empty
    assert "Thesis Snapshot" in result.thesis_mode_markdown
    assert not result.candidate_stress_grid.empty
    assert not result.premium_sensitivity_summary.empty
    assert not result.timing_slip_summary.empty
    assert not result.target_stress_summary.empty
    assert not result.stress_transition_summary.empty
    assert "Stress Snapshot" in result.stress_tests_markdown
    assert not result.single_option_decision_summary.empty
    assert not result.single_option_decision_path_selections.empty
    assert not result.single_option_representative_paths.empty
    assert not result.single_option_path_outcomes.empty
    assert not result.single_option_path_family_counts.empty
    assert not result.single_option_timing_sensitivity.empty
    assert not result.single_option_iv_sensitivity.empty
    assert not result.single_option_entry_sensitivity.empty
    assert not result.single_option_summary_bullets.empty
    assert not result.single_option_required_path_to_beat_stock_1_5x.empty
    assert not result.single_option_required_path_to_beat_stock_2_0x.empty
    assert not result.single_option_closest_representative_path_to_edge.empty
    assert not result.single_option_edge_gap_by_path_family.empty
    assert "Single-Option Decision View" in result.single_option_decision_markdown
    assert not result.chain_overview_summary.empty
    assert not result.chain_overview_candidates.empty
    assert "Chain Overview / Compare Options" in result.chain_overview_markdown
    assert set(result.chain_overview_candidates["candidate_slug"]).issubset(
        set(
            result.candidate_summary.loc[
                result.candidate_summary["strategy_family"].astype(str).eq("long_call"),
                "candidate_slug",
            ]
        )
    )
    assert tuple(result.chain_overview_summary["card_key"]) == CHAIN_OVERVIEW_CARD_KEYS
    assert {
        "contract",
        "premium",
        "iv",
        "dte",
        "beats_stock_label",
        "beats_stock_count",
        "strong_wins",
        "strong_outperformance_count",
        "robustness",
        "robustness_score",
        "iv_sensitivity",
        "iv_sensitivity_score",
        "entry_sensitivity",
        "entry_premium_sensitivity_score",
        "best_fit_path_type",
        "worth_buying_status",
        "final_verdict",
        "why_short",
        "why_detail",
    } <= set(result.chain_overview_candidates.columns)
    assert set(result.chain_overview_candidates["final_verdict"]).issubset(
        {
            "Robust buy candidate",
            "Selective / thesis-dependent",
            "Too narrow",
            "Stock better",
        }
    )
    assert result.chain_overview_candidates["minimum_outperformance_multiple"].eq(1.5).all()
    assert result.chain_overview_candidates["strong_outperformance_multiple"].eq(2.0).all()
    assert result.chain_overview_candidates["required_winning_path_families"].eq(2).all()
    assert result.chain_overview_candidates["shared_path_family_count"].between(5, 8).all()
    assert 5 <= result.single_option_decision_path_selections["decision_path_id"].nunique() <= 8
    assert set(result.single_option_decision_path_selections["decision_path_id"]) == set(result.single_option_path_outcomes["decision_path_id"])
    assert set(result.single_option_decision_path_selections["decision_path_id"]) == set(result.single_option_representative_paths["decision_path_id"])
    assert result.single_option_path_outcomes["is_curated_decision_path"].fillna(False).all()
    assert result.single_option_representative_paths["is_curated_decision_path"].fillna(False).all()
    assert result.single_option_decision_path_selections["selection_reason"].astype(str).str.len().gt(0).all()
    assert set(result.single_option_path_outcomes["outcome_label"]).issubset(
        {
            "clear_option_win",
            "wins_but_not_enough",
            "stock_better",
            "fail_too_narrow_or_expiry_issue",
        }
    )
    single_summary = result.single_option_decision_summary.iloc[0]
    assert single_summary["minimum_outperformance_multiple"] == 1.5
    assert single_summary["strong_outperformance_multiple"] == 2.0
    assert single_summary["required_winning_path_families"] == 2
    assert single_summary["entry_price_mode"] == "conservative_mid_plus_slippage"
    assert single_summary["exit_rule"] == "sell_on_thesis_completion"
    assert result.single_option_required_path_to_beat_stock_1_5x["edge_path_name"].eq("required_path_to_beat_stock_1_5x").all()
    assert result.single_option_required_path_to_beat_stock_2_0x["edge_path_name"].eq("required_path_to_beat_stock_2_0x").all()
    assert {
        "decision_path_id",
        "path_family_label",
        "required_stock_price_1_5x",
        "required_stock_price_2_0x",
        "extra_stock_move_needed_1_5x",
        "edge_gap_to_1_5x_dollars",
        "timing_gap_note",
        "is_closest_to_edge",
    } <= set(result.single_option_edge_gap_by_path_family.columns)
    assert result.single_option_edge_gap_by_path_family["decision_path_id"].nunique() == result.single_option_decision_path_selections["decision_path_id"].nunique()
    assert result.single_option_edge_gap_by_path_family["is_closest_to_edge"].fillna(False).sum() == 1
    assert {
        "decision_path_id",
        "annotation_text",
        "edge_gap_to_1_5x_dollars",
    } <= set(result.single_option_closest_representative_path_to_edge.columns)
    assert {
        "difference_vs_stock",
        "stock_profit_loss",
        "outperformance_multiple",
        "qualifies_as_winning_path_family",
        "decision_path_id",
        "path_family",
        "path_family_label",
        "timing_shape",
        "selection_reason",
    } <= set(result.single_option_path_outcomes.columns)
    assert {
        "decision_path_id",
        "path_family",
        "path_family_label",
        "timing_shape",
        "outcome_label",
        "selection_score",
        "selection_reason",
    } <= set(result.single_option_decision_path_selections.columns)
    assert {
        "Base",
        "Premium -10%",
        "Premium -20%",
        "Premium +10%",
        "Move delayed 2w",
        "Move delayed 1m",
        "Move delayed 2m",
    } <= set(result.candidate_stress_grid.columns)
    assert any(column.startswith("Undershoot to") for column in result.candidate_stress_grid.columns)
    assert any(column.startswith("Overshoot to") for column in result.candidate_stress_grid.columns)
    assert {
        "base_action_bucket",
        "best_improving_stress",
        "worst_breaking_stress",
        "stress_resilience_score",
        "premium_sensitivity_read",
        "timing_sensitivity_read",
        "target_dependency_read",
    } <= set(result.stress_transition_summary.columns)
    assert {
        "early_breakout_to_target",
        "slow_grind_to_target",
        "late_breakout_to_target",
    } <= set(result.thesis_path_gallery["path_family"])
    assert {"current_premium", "max_justified_premium", "premium_gap"} <= set(result.thesis_candidate_ranking.columns)
    assert "Best Bullish Long Calls Right Now" in result.bullish_action_board_markdown
    assert "Top Bullish Call Cards" in result.top_candidate_cards_markdown
    assert "Secondary Structures Snapshot" in result.other_structures_markdown
    assert "What Looks Most Actionable Right Now" in result.action_board_markdown
    assert "What Has To Happen For These Calls To Be Worth Buying" in result.entry_justification_markdown
    assert {"Watchlist", "Avoid For Now", "Prefer Stock Instead"} & set(result.action_board_candidates["action_bucket"])
    assert not result.chain_source_summary.empty
    assert not result.market_context_summary.empty
    assert "itm_1c" in set(result.required_path_rows["goal"])
    assert "itm_1c" in set(result.required_path_summary["goal"])
    assert set(result.path_case_chart_rows["series_kind"]) >= {"required_path", "assumed_path"}
    assert set(result.path_case_chart_rows["case_label"]) == {"-20%", "-10%", "0%", "+10%", "+20%"}
    assert result.report_metadata["path_case_defaults"]["default_case_label"] == "0%"
    assert result.report_metadata["path_case_defaults"]["default_goal"] == "break_even"
    assert result.stock_path_name == "custom_stock_path"
    assert result.iv_path_name == "custom_iv_path"
    assert "custom_stock_path" in set(result.path_case_summary["stock_path"])
    assert "custom_iv_path" in set(result.path_case_summary["iv_path"])
    assert result.path_case_rows.loc[
        (result.path_case_rows["stock_path"] == "custom_stock_path")
        & (result.path_case_rows["horizon"] == "3m"),
        "spot_price",
    ].gt(21.9).any()
    assert result.path_case_rows.loc[
        (result.path_case_rows["iv_path"] == "custom_iv_path")
        & (result.path_case_rows["horizon"] == "3m"),
        "iv_shift_points",
    ].lt(-0.09).any()
    assert result.report_metadata["default_strategy_family"]
    assert result.report_metadata["default_contract_for_path_explorer"]
    assert {
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "stock_profit_loss",
        "worst_interim_profit_loss_to_date",
        "drawdown_from_peak_to_date",
    } <= set(result.assumed_path_trace_rows.columns)
    assert {
        "assumed_stock_price",
        "assumed_minus_required_price",
        "assumed_clears_required_path",
        "required_move_pct_from_entry",
        "required_path_note",
    } <= set(result.required_path_rows.columns)
    assert {"required_path_difficulty", "path_gap_at_target", "first_cleared_horizon"} <= set(result.required_path_summary.columns)
    assert {
        "ticker",
        "contract_label",
        "option_type",
        "strike",
        "expiry",
        "dte",
        "entry_premium",
        "entry_iv",
        "snapshot_date",
        "analysis_horizon_date",
        "chart_horizon_date",
        "path_terminal_date",
        "option_expiry_date",
        "terminal_basis",
        "chart_horizon_basis",
        "threshold_multiple",
        "required_terminal_stock_price",
        "required_terminal_stock_return_pct",
        "required_move_pct",
        "earliest_valid_date",
        "latest_valid_date",
        "minimum_required_move_by_expiry",
        "path_count_generated",
        "plausible_path_count",
        "extreme_path_count",
        "status",
        "verdict",
        "concise_explanation",
    } <= set(result.required_path_core_summary.columns)
    assert {
        "contract_label",
        "threshold_multiple",
        "path_id",
        "path_label",
        "path_family",
        "date",
        "days_from_snapshot",
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
        "stock_price",
        "option_value",
        "option_return_pct",
        "stock_return_pct",
        "option_vs_stock_multiple",
        "clears_threshold",
        "is_checkpoint_marker",
        "display_marker",
        "checkpoint_label",
        "is_peak_option_return",
        "realism_bucket",
    } <= set(result.required_paths_by_option.columns)
    assert {
        "contract_label",
        "threshold_multiple",
        "path_family",
        "min_required_move_pct",
        "median_required_move_pct",
        "earliest_clear_date",
        "latest_clear_date",
        "clears_count",
        "realism_bucket",
        "failure_driver",
        "peak_option_return_pct",
        "peak_date",
    } <= set(result.required_path_family_summary.columns)
    assert {
        "contract_label",
        "threshold_multiple",
        "path_family",
        "peak_date",
        "peak_option_return_pct",
        "stock_price_at_peak",
        "terminal_option_return_pct",
    } <= set(result.required_path_peak_summary.columns)
    assert not result.required_path_entry_sensitivity.empty
    assert not result.required_path_iv_sensitivity.empty
    assert not result.required_path_entry_iv_matrix.empty
    assert not result.required_path_sell_hold_summary.empty
    assert {
        "entry_shift_pct",
        "adjusted_entry_premium",
        "required_terminal_stock_price",
        "required_move_pct",
        "realism_bucket",
        "verdict",
    } <= set(result.required_path_entry_sensitivity.columns)
    assert {
        "iv_shift_vol_points",
        "adjusted_iv",
        "required_terminal_stock_price",
        "required_move_pct",
        "realism_bucket",
        "verdict",
    } <= set(result.required_path_iv_sensitivity.columns)
    assert {
        "entry_shift_pct",
        "iv_shift_vol_points",
        "adjusted_entry_premium",
        "adjusted_iv",
        "required_move_pct",
    } <= set(result.required_path_entry_iv_matrix.columns)
    assert {
        "option_return_2w_after_peak",
        "option_return_1m_after_peak",
        "expiry_option_return_pct",
        "decay_from_peak_to_expiry_pct",
        "interpretation",
    } <= set(result.required_path_sell_hold_summary.columns)
    assert "Contract inputs" in result.required_path_tables_html
    assert "Required move summary" in result.required_path_tables_html
    assert "1.5x / 2.0x are relative to stock return, not absolute option return." in result.required_path_tables_html
    assert "NaN" not in result.required_path_tables_html
    assert "How to read the tables" in result.required_path_tables_markdown
    assert set(result.required_path_core_summary["threshold_multiple"]) == {1.5, 2.0}
    assert set(result.required_paths_by_option["path_family"]).issuperset(
        {
            "fast_breakout",
            "slow_grind",
            "late_acceleration",
            "down_first_recovery",
            "rally_retrace_finish",
            "violent_absurd",
            "expiry_only",
        }
    )
    parsed_contract_strikes = pd.to_numeric(
        result.required_path_core_summary["contract_label"].astype(str).str.extract(r"^(\d+(?:\.\d+)?)C")[0],
        errors="coerce",
    )
    actual_contract_strikes = pd.to_numeric(result.required_path_core_summary["strike"], errors="coerce")
    assert actual_contract_strikes[parsed_contract_strikes.notna()].eq(parsed_contract_strikes[parsed_contract_strikes.notna()]).all()
    solved_required = result.required_path_core_summary.loc[
        result.required_path_core_summary["status"].astype(str).str.startswith("solved")
    ]
    assert not solved_required.empty
    assert pd.to_numeric(solved_required["required_move_pct"], errors="coerce").notna().all()
    solved_required_paths = result.required_paths_by_option.loc[
        result.required_paths_by_option["status"].astype(str).str.startswith("solved")
    ]
    assert pd.to_numeric(solved_required_paths["stock_price"], errors="coerce").notna().any()
    assert pd.to_numeric(solved_required_paths["option_value"], errors="coerce").notna().any()
    assert {
        "current_objective_card_status",
        "best_under_current_objective",
        "best_if_move_is_slower",
        "best_if_iv_falls",
        "best_for_capital_efficiency",
        "best_for_capped_downside",
        "best_for_convexity",
        "best_for_simple_exposure",
        "why_this_wins",
        "why_this_loses",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "benchmark_note",
    } <= set(result.family_comparison.columns)
    assert {
        "active_candidate_rank",
        "objective_score",
        "premium_or_entry_cost",
        "affordability_label",
        "required_path_difficulty",
        "path_gap_at_target",
        "timing_risk",
        "iv_risk",
        "success_dependency",
        "relevance_under_thesis",
        "why_this_candidate_wins",
        "why_this_candidate_loses",
        "weak_horizon_fit",
    } <= set(result.candidate_comparison.columns)
    assert {
        "strategy_family",
        "strike_label",
        "best_expiry_date",
        "best_candidate_label",
        "active_candidate_rank",
        "objective_score",
        "required_path_difficulty",
        "timing_risk",
        "iv_risk",
        "weak_horizon_fit",
        "target_beyond_expiry",
    } <= set(result.strike_comparison.columns)
    assert {
        "strategy_family",
        "expiry_date",
        "best_strike_label",
        "best_candidate_label",
        "active_candidate_rank",
        "objective_score",
        "required_path_difficulty",
        "timing_risk",
        "iv_risk",
        "expiry_fit_note",
        "target_beyond_expiry",
    } <= set(result.expiry_comparison.columns)
    assert {
        "case_rank",
        "strategy_family",
        "required_path_difficulty",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "timing_risk",
        "iv_risk",
        "success_dependency",
    } <= set(result.path_case_family_rankings.columns)
    assert {
        "case_rank",
        "candidate_slug",
        "required_path_difficulty",
        "difference_vs_stock_return_pct",
        "benchmark_note",
        "timing_risk",
        "iv_risk",
        "success_dependency",
    } <= set(result.path_case_candidate_rankings.columns)
    assert result.path_case_candidate_rankings["candidate_slug"].nunique() == result.candidate_summary["candidate_slug"].nunique()
    assert {"iv_path_name", "variant_kind", "delta_from_entry_iv_shift"} <= set(result.iv_path_trace_rows.columns)
    assert {"delta_profit_loss_vs_stock", "delta_return_pct_vs_stock", "benchmark_note"} <= set(result.compare_vs_stock_path_rows.columns)
    assert {"summary_scope", "iv_risk", "sensitivity_note", "pnl_sensitivity_range"} <= set(result.iv_path_sensitivity_summary.columns)
    assert {"summary_scope", "timing_risk", "iv_risk", "success_dependency", "worst_drawdown_from_peak"} <= set(result.path_risk_summary.columns)
    assert {
        "source_snapshot_file",
        "storage_location",
        "quote_usable",
        "fallback_level",
        "source_quality",
        "source_trust_label",
        "source_quality_note",
    } <= set(result.chain_source_summary.columns)
    assert {
        "spot_price_source",
        "spot_field_used",
        "spot_used_prior_date",
        "spot_quality_note",
        "ibkr_same_day_spot_attempted",
        "risk_free_rate_source",
        "risk_free_rate_series",
        "expected_move_matched",
        "nearest_event_type",
    } <= set(result.market_context_summary.columns)
    assert {
        "source_quality",
        "source_trust_label",
        "source_quality_note",
    } <= set(result.candidate_comparison.columns)
    assert {
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
    } <= set(result.decision_highlights["highlight_category"])
    assert {
        "robustness_score",
        "aggressive_upside_score",
        "balanced_score",
        "time_resilience_score",
        "lower_iv_resilience_score",
        "iv_upside_score",
        "trust_score",
        "penalty_score",
        "decision_tags",
        "stock_dominance_note",
    } <= set(result.candidate_tradeoff_matrix.columns)
    assert result.decision_highlights["decision_status"].astype(str).str.len().gt(0).all()
    assert result.decision_highlights["decision_status"].astype(str).str.contains(
        "stock_still_benchmark|weak_differentiation|no_clear_edge_under_current_assumptions|informative_edge"
    ).any()
    stock_highlight = result.decision_highlights.loc[
        result.decision_highlights["highlight_category"] == "stock_still_best_baseline"
    ].iloc[0]
    assert stock_highlight["selected_family"] in {"long_stock", "no_clear_edge"}
    assert "stock" in result.stock_vs_option_takeaways["takeaway_type"].str.lower().str.cat(sep=" ")
    assert {
        "action_bucket",
        "action_priority_rank",
        "candidate_conviction_score",
        "stock_relative_score",
        "time_decay_risk",
        "iv_dependence_risk",
        "why_this_is_interesting_now",
        "what_is_hurting_this_candidate",
        "main_trigger",
        "upgrade_rule",
        "invalidate_rule",
        "what_has_to_happen",
        "what_would_invalidate",
    } <= set(result.action_board_candidates.columns)
    assert {
        "key_trigger_type",
        "trigger_type_label",
        "key_trigger_value",
        "key_trigger_deadline",
        "upgrade_rule",
        "invalidate_rule",
    } <= set(result.decision_triggers.columns)
    assert {
        "entry_display_rank",
        "candidate_short_label",
        "required_price_1m",
        "required_price_target",
        "required_move_pct_target",
        "timing_window_days",
        "move_pace_pct_per_month",
        "requires_fast_move",
        "needs_iv_support",
        "stock_still_better_even_if_path_hits",
        "iv_requirement_label",
        "entry_barrier_score",
        "entry_barrier_label",
        "what_has_to_happen",
        "entry_warning",
        "stock_vs_option_read",
    } <= set(result.entry_justification_candidates.columns)
    assert {
        "series_kind",
        "stock_price",
        "entry_barrier_label",
        "stock_vs_option_read",
        "iv_path_label",
    } <= set(result.required_stock_path_to_buy.columns)
    assert {
        "required_move_pct_1m",
        "required_move_pct_3m",
        "required_move_pct_target",
        "timing_window_days",
        "move_pace_pct_per_month",
        "requires_fast_move",
        "stock_still_better_even_if_path_hits",
        "entry_barrier_score",
    } <= set(result.required_move_summary.columns)
    assert {
        "assumed_clears_required_at_target",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "stock_relative_score",
        "stock_still_better_even_if_path_hits",
        "stock_vs_option_read",
    } <= set(result.required_move_vs_stock.columns)
    assert {
        "required_move_pct_flat_iv",
        "lower_iv_required_move_pct",
        "higher_iv_required_move_pct",
        "lower_iv_move_penalty_pct",
        "higher_iv_move_relief_pct",
        "lower_iv_resilience_score",
        "iv_dependence_risk",
        "iv_requirement_label",
        "iv_requirement_note",
    } <= set(result.required_iv_support_summary.columns)
    assert {
        "entry_barrier_score",
        "entry_barrier_label",
        "required_move_pct_target",
        "timing_window_days",
        "iv_requirement_label",
        "stock_vs_option_read",
    } <= set(result.entry_barrier_summary.columns)
    if "Watchlist" in set(result.action_board_candidates["action_bucket"]):
        assert result.decision_triggers.iloc[0]["action_bucket"] == "Watchlist"
    assert set(result.bullish_long_call_action_board["strategy_family"].astype(str).str.lower()) == {"long_call"}
    assert {
        "upgrade_rule",
        "invalidate_rule",
        "why_this_is_watchlist_not_buy",
        "what_is_hurting_this_candidate",
        "why_stock_may_be_better",
    } & set(result.bullish_long_call_action_board.columns)
    assert {
        "contract_label",
        "bucket",
        "why_this_is_interesting",
        "what_hurts_it",
        "upgrade_rule",
        "invalidate_rule",
        "compare_vs_stock_note",
    } <= set(result.top_candidate_cards.columns)
    assert result.top_candidate_cards["upgrade_rule"].astype(str).str.len().gt(0).any()
    assert {"Better After IV Cools", "Needs Earlier Move", "Later Expiry / Timing", "Stock Confirmation", "Needs Better Trust"} & set(
        result.decision_triggers["trigger_type_label"].astype(str)
    )
    expanded_trigger_types = {
        "stock_confirmation",
        "premium_below_threshold",
        "iv_normalization_entry",
        "prefer_later_expiry",
        "move_must_start_early",
        "better_after_event",
        "trust_too_weak_for_action",
        "stock_cleaner_unless_x",
    }
    assert expanded_trigger_types & set(result.decision_triggers["key_trigger_type"].astype(str))
    stock_action = result.action_board_candidates.loc[
        result.action_board_candidates["action_bucket"] == "Prefer Stock Instead"
    ]
    assert not stock_action.empty
    assert {"best_source_quality", "best_source_trust_label"} <= set(result.strike_comparison.columns)
    assert {"best_source_quality", "best_source_trust_label"} <= set(result.expiry_comparison.columns)
    assert result.candidate_comparison["weak_horizon_fit"].isin([True, False]).all()
    assert result.expiry_comparison["target_beyond_expiry"].isin([True, False]).all()
    assert result.report_metadata["spot_price_source"]
    assert "spot_field_used" in result.report_metadata
    assert "spot_used_prior_date" in result.report_metadata
    assert "spot_quality_note" in result.report_metadata
    assert "ibkr_same_day_spot_attempted" in result.report_metadata
    assert "spot_price_matched_date" in result.report_metadata
    assert "research_context" in result.report_metadata
    assert result.report_metadata["research_context"]["options_overview"]


@pytest.mark.slow
def test_contract_selection_analysis_supports_target_option_value_required_path_outputs():
    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        goal="target_option_value",
        target_option_value=250.0,
        data_root=DATA_ROOT,
    )

    assert "target_option_value" in set(result.required_path_rows["goal"])
    assert "target_option_value" in set(result.required_path_summary["goal"])
    assert result.required_path_summary["required_path_difficulty"].notna().any()


@pytest.mark.slow
def test_contract_selection_analysis_generates_simulated_path_outputs():
    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_price=20.0,
        target_date="2026-07-15",
        data_root=DATA_ROOT,
    )

    assert not result.stock_path_examples.empty
    assert not result.iv_path_examples.empty
    assert not result.path_pair_summary.empty
    assert not result.option_value_over_path.empty
    assert not result.compare_vs_stock_over_path.empty
    assert not result.representative_paths_summary.empty
    assert not result.strike_comparison_under_path.empty
    assert not result.expiry_comparison_under_path.empty
    assert not result.required_vs_assumed_path_summary.empty
    assert {
        "path_id",
        "path_kind",
        "path_name",
        "spot_price",
        "requested_days",
        "is_representative",
        "representative_bucket",
    } <= set(result.stock_path_examples.columns)
    assert {
        "iv_path_id",
        "iv_path_name",
        "iv_shift_points",
        "requested_days",
        "is_representative",
    } <= set(result.iv_path_examples.columns)
    assert {
        "path_pair_id",
        "stock_path_id",
        "iv_path_id",
        "representative_bucket",
        "selection_reason",
        "goal_success_rate",
    } <= set(result.path_pair_summary.columns)
    assert {
        "path_pair_id",
        "candidate_slug",
        "strategy_family",
        "date",
        "requested_days",
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "max_favorable_profit_to_date",
        "success_status",
    } <= set(result.option_value_over_path.columns)
    assert {
        "path_pair_id",
        "candidate_slug",
        "delta_profit_loss_vs_stock",
        "delta_return_pct_vs_stock",
        "benchmark_note",
    } <= set(result.compare_vs_stock_over_path.columns)
    assert {
        "path_pair_id",
        "representative_bucket",
        "selection_reason",
        "top_candidate_success_status",
        "stock_benchmark_status",
    } <= set(result.representative_paths_summary.columns)
    assert {
        "path_pair_id",
        "strategy_family",
        "strike_label",
        "best_candidate_label",
        "objective_score",
        "source_trust_label",
        "source_quality_note",
        "weak_horizon_fit",
        "target_beyond_expiry",
    } <= set(result.strike_comparison_under_path.columns)
    assert {
        "path_pair_id",
        "strategy_family",
        "expiry_date",
        "best_candidate_label",
        "objective_score",
        "source_trust_label",
        "source_quality_note",
        "weak_horizon_fit",
        "target_beyond_expiry",
    } <= set(result.expiry_comparison_under_path.columns)
    assert {
        "comparison_scope",
        "candidate_slug",
        "goal",
        "assumed_path_name",
        "required_path_difficulty",
        "representative_path_gap_at_target",
    } <= set(result.required_vs_assumed_path_summary.columns)
    assert set(result.representative_paths_summary["representative_bucket"]) >= {
        "misses_badly",
        "almost_works",
        "just_works",
    }


@pytest.mark.slow
def test_contract_selection_analysis_builds_expanded_path_gallery_and_path_centric_long_call_outputs(
    late_breakout_contract_selection_result,
):
    result = late_breakout_contract_selection_result

    expected_gallery_paths = {
        "rally_early_then_fade_then_rally_again",
        "range_bound_near_flat",
        "down_first_then_recovery",
        "late_breakout",
        "early_move_above_strike_then_giveback",
        "reaches_target_late_near_expiry",
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
        "plus_20_pct_in_1m",
        "plus_30_pct_in_1m",
        "plus_20_pct_in_1q",
        "plus_30_pct_in_1q",
    }
    focus_paths = [
        "rally_early_then_fade_then_rally_again",
        "range_bound_near_flat",
        "down_first_then_recovery",
        "late_breakout",
        "early_move_above_strike_then_giveback",
        "reaches_target_late_near_expiry",
    ]

    assert expected_gallery_paths <= set(result.stock_path_gallery["path_name"])
    assert result.path_view_tables
    for path_name in focus_paths:
        for suffix in [
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
        ]:
            filename = _path_view_filename(path_name, suffix.removeprefix("__"))
            assert filename in result.path_view_tables
            assert not result.path_view_tables[filename].empty

    late_breakout_best_of = result.path_view_tables[_path_view_filename("late_breakout", "long_call_best_of_value.csv")]
    assert set(late_breakout_best_of["stock_path_name"]) == {"late_breakout"}
    assert set(late_breakout_best_of["iv_path_name"]) == {"flat"}
    assert late_breakout_best_of["candidate_slug"].nunique() <= 6

    rally_compare = result.path_view_tables[_path_view_filename("rally_early_then_fade_then_rally_again", "compare_vs_stock_path_rows.csv")]
    assert {
        "stock_path_name",
        "iv_path_name",
        "series_label",
        "delta_profit_loss_vs_stock",
        "delta_return_pct_vs_stock",
        "selection_rank",
    } <= set(rally_compare.columns)
    late_breakout_strike_delta = result.path_view_tables[_path_view_filename("late_breakout", "long_call_strike_delta.csv")]
    assert {
        "delta_profit_loss_vs_stock",
        "delta_return_pct_vs_stock",
        "view_name",
        "stock_path_name",
        "iv_path_name",
    } <= set(late_breakout_strike_delta.columns)
    assert result.strike_comparison_under_path["source_trust_label"].notna().any()
    assert result.expiry_comparison_under_path["source_trust_label"].notna().any()
    late_breakout_checkpoints = result.path_view_tables[_path_view_filename("late_breakout", "path_checkpoints.csv")]
    assert {
        "checkpoint_label",
        "spot_price",
        "iv_shift_points",
        "modeled_value",
        "profit_loss",
        "difference_vs_stock",
        "series_label",
    } <= set(late_breakout_checkpoints.columns)
    assert set(late_breakout_checkpoints["checkpoint_label"]) >= {"entry", "target"}
    late_breakout_iv_value = result.path_view_tables[_path_view_filename("late_breakout", "iv_path_value.csv")]
    late_breakout_iv_delta = result.path_view_tables[_path_view_filename("late_breakout", "iv_path_delta.csv")]
    late_breakout_iv_checkpoints = result.path_view_tables[_path_view_filename("late_breakout", "iv_checkpoints.csv")]
    expected_iv_paths = {
        "flat",
        "mean_reversion_lower",
        "mean_reversion_higher",
        "iv_up_then_down",
        "iv_down_then_stays_low",
        "earnings_build_then_crush",
    }
    assert expected_iv_paths <= set(late_breakout_iv_value["iv_path_name"])
    assert late_breakout_iv_value["candidate_slug"].nunique() == 1
    assert {
        "iv_path_label",
        "anchor_contract_label",
        "terminal_value_vs_flat",
        "terminal_delta_vs_flat",
        "iv_effect_note",
    } <= set(late_breakout_iv_value.columns)
    assert {"delta_profit_loss_vs_stock", "iv_effect_note"} <= set(late_breakout_iv_delta.columns)
    assert {"checkpoint_label", "iv_path_name", "modeled_value", "difference_vs_stock", "iv_effect_note"} <= set(
        late_breakout_iv_checkpoints.columns
    )
    terminal = late_breakout_iv_value.sort_values("requested_days").groupby("iv_path_name", as_index=False).tail(1)
    terminal_values = dict(zip(terminal["iv_path_name"], terminal["modeled_value"]))
    assert terminal_values["mean_reversion_higher"] > terminal_values["flat"] > terminal_values["mean_reversion_lower"]
    late_breakout_strike_iv_value = result.path_view_tables[_path_view_filename("late_breakout", "long_call_strike_iv_value.csv")]
    late_breakout_strike_iv_delta = result.path_view_tables[_path_view_filename("late_breakout", "long_call_strike_iv_delta.csv")]
    late_breakout_expiry_iv_value = result.path_view_tables[_path_view_filename("late_breakout", "long_call_expiry_iv_value.csv")]
    late_breakout_best_of_iv_value = result.path_view_tables[_path_view_filename("late_breakout", "long_call_best_of_iv_value.csv")]
    late_breakout_robustness = result.path_view_tables[_path_view_filename("late_breakout", "iv_robustness_summary.csv")]
    assert expected_iv_paths <= set(late_breakout_strike_iv_value["iv_path_name"])
    assert late_breakout_strike_iv_value["candidate_slug"].nunique() > 1
    assert late_breakout_expiry_iv_value["candidate_slug"].nunique() > 1
    assert late_breakout_best_of_iv_value["candidate_slug"].nunique() > 1
    assert {
        "iv_expanded_family",
        "contract_rank",
        "chart_include",
        "iv_chart_scope",
        "terminal_value_vs_flat",
        "iv_effect_note",
    } <= set(late_breakout_strike_iv_value.columns)
    assert {"delta_profit_loss_vs_stock", "iv_expanded_family", "contract_rank", "chart_include"} <= set(
        late_breakout_strike_iv_delta.columns
    )
    assert bool(late_breakout_strike_iv_value["chart_include"].any())
    assert set(late_breakout_strike_iv_value.loc[late_breakout_strike_iv_value["chart_include"], "iv_path_name"]) <= {
        "flat",
        "mean_reversion_lower",
        "mean_reversion_higher",
        "earnings_build_then_crush",
    }
    assert {
        "iv_expanded_family",
        "beat_stock_iv_path_count",
        "lower_iv_profitable",
        "high_iv_dependency",
        "iv_robustness_label",
        "iv_robustness_note",
    } <= set(late_breakout_robustness.columns)
    assert set(late_breakout_robustness["iv_expanded_family"]) >= {"strike", "expiry", "best_of"}


@pytest.mark.slow
def test_contract_selection_analysis_builds_stock_and_iv_path_galleries(late_breakout_contract_selection_result):
    result = late_breakout_contract_selection_result

    assert not result.stock_path_library.empty
    assert not result.stock_path_gallery.empty
    assert not result.iv_path_gallery.empty
    assert {
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
    } <= set(result.stock_path_library.columns)
    assert {
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
    } <= set(result.stock_path_gallery.columns)
    assert {
        "iv_path_name",
        "iv_path_label",
        "path_role",
        "display_order",
        "date",
        "requested_days",
        "iv_shift_points",
        "is_active_assumed",
    } <= set(result.iv_path_gallery.columns)
    assert set(result.stock_path_gallery["path_name"]) >= {
        "rally_early_then_fade_then_rally_again",
        "range_bound_near_flat",
        "down_first_then_recovery",
        "late_breakout",
        "early_move_above_strike_then_giveback",
        "reaches_target_late_near_expiry",
        "quarter_up_then_hard_pullback",
        "high_vol_sideways_then_breakout",
        "earnings_gap_up_then_fade",
        "earnings_gap_down_then_recovery",
        "false_breakout_then_recover",
        "rally_then_long_range_then_second_leg_up",
        "violent_two_sided_quarter",
        "slow_bleed_then_capitulation_then_bounce",
        "plus_20_pct_in_1m",
        "plus_30_pct_in_1m",
        "plus_20_pct_in_1q",
        "plus_30_pct_in_1q",
    }
    assert {
        "minimum_required_path",
        "early_rally",
        "late_rally",
        "steady_grind_up",
        "false_breakout",
        "recovery",
        "earnings_gap",
        "quarter_pullback",
    } <= set(result.stock_path_library["path_family"])
    assert set(result.iv_path_gallery["iv_path_name"]) >= {
        "flat",
        "mean_reversion_lower",
        "mean_reversion_higher",
        "iv_up_then_down",
        "iv_down_then_stays_low",
        "earnings_build_then_crush",
    }
    assert result.stock_path_gallery.loc[result.stock_path_gallery["is_active_assumed"] == True, "path_name"].eq("late_breakout").any()  # noqa: E712
    assert result.iv_path_gallery.loc[result.iv_path_gallery["is_active_assumed"] == True, "iv_path_name"].eq("flat").any()  # noqa: E712
    assert result.stock_path_gallery["display_order"].nunique() >= 6
    assert result.iv_path_gallery["display_order"].nunique() >= 6


def test_named_stock_path_presets_generate_non_flat_shapes():
    grid = build_path_grid(snapshot_date=pd.Timestamp("2026-04-12").date(), end_date=pd.Timestamp("2026-07-15").date())
    presets = [
        "rally_early_then_fade_then_rally_again",
        "range_bound_near_flat",
        "down_first_then_recovery",
        "late_breakout",
        "early_move_above_strike_then_giveback",
        "reaches_target_late_near_expiry",
        "quarter_up_then_hard_pullback",
        "high_vol_sideways_then_breakout",
        "earnings_gap_up_then_fade",
        "earnings_gap_down_then_recovery",
        "false_breakout_then_recover",
        "rally_then_long_range_then_second_leg_up",
        "violent_two_sided_quarter",
        "slow_bleed_then_capitulation_then_bounce",
        "plus_20_pct_in_1m",
        "plus_30_pct_in_1m",
        "plus_20_pct_in_1q",
        "plus_30_pct_in_1q",
    ]

    for preset in presets:
        path = build_stock_path_example(
            grid,
            entry_spot=16.0,
            mode="deterministic",
            preset=preset,
            target_end=20.0,
        )
        prices = [float(point["spot_price"]) for point in path.path_points]

        assert path.path_name == preset
        assert prices[0] == 16.0
        assert prices[-1] == 20.0
        assert len({round(value, 4) for value in prices[1:-1]}) > 1
        assert any(abs(value - 16.0) > 0.25 for value in prices[1:-1])


@pytest.mark.slow
def test_contract_selection_analysis_builds_assumed_path_long_call_views(late_breakout_contract_selection_result):
    result = late_breakout_contract_selection_result

    assert not result.long_call_value_over_path_strike_view.empty
    assert not result.long_call_value_over_path_expiry_view.empty
    assert not result.long_call_value_over_path_best_of.empty

    expected_columns = {
        "view_name",
        "path_scope",
        "stock_path_name",
        "iv_path_name",
        "candidate_slug",
        "candidate_label",
        "expiry_date",
        "strike_label",
        "moneyness_bucket",
        "source_trust_label",
        "source_quality_note",
        "selection_rank",
        "selection_reason",
        "date",
        "requested_days",
        "modeled_value",
        "profit_loss",
        "return_on_comparison_capital",
        "difference_vs_stock",
        "difference_vs_stock_return_pct",
        "success_status",
        "objective_score",
    }
    assert expected_columns <= set(result.long_call_value_over_path_strike_view.columns)
    assert expected_columns | {"anchor_expiry_date"} <= set(result.long_call_value_over_path_strike_view.columns)
    assert expected_columns | {"anchor_strike_label", "used_strike_fallback", "strike_match_mode", "fallback_strike_distance"} <= set(result.long_call_value_over_path_expiry_view.columns)
    assert expected_columns | {"used_trust_fallback"} <= set(result.long_call_value_over_path_best_of.columns)

    strike_view = result.long_call_value_over_path_strike_view.copy()
    expiry_view = result.long_call_value_over_path_expiry_view.copy()
    best_of_view = result.long_call_value_over_path_best_of.copy()

    assert set(strike_view["path_scope"]) == {"assumed_path"}
    assert set(expiry_view["path_scope"]) == {"assumed_path"}
    assert set(best_of_view["path_scope"]) == {"assumed_path"}
    assert set(strike_view["stock_path_name"]) == {result.stock_path_name}
    assert set(expiry_view["stock_path_name"]) == {result.stock_path_name}
    assert set(best_of_view["stock_path_name"]) == {result.stock_path_name}
    assert set(strike_view["iv_path_name"]) == {result.iv_path_name}
    assert set(expiry_view["iv_path_name"]) == {result.iv_path_name}
    assert set(best_of_view["iv_path_name"]) == {result.iv_path_name}

    strike_terminals = (
        strike_view.sort_values(["candidate_slug", "requested_days"])
        .groupby("candidate_slug", as_index=False)
        .tail(1)
        .sort_values("selection_rank")
    )
    expiry_terminals = (
        expiry_view.sort_values(["candidate_slug", "requested_days"])
        .groupby("candidate_slug", as_index=False)
        .tail(1)
        .sort_values("selection_rank")
    )
    best_of_terminals = (
        best_of_view.sort_values(["candidate_slug", "requested_days"])
        .groupby("candidate_slug", as_index=False)
        .tail(1)
        .sort_values("selection_rank")
    )

    assert strike_terminals["expiry_date"].nunique() == 1
    assert strike_terminals["strike_label"].nunique() > 1
    assert expiry_terminals["expiry_date"].nunique() > 1
    assert expiry_terminals["anchor_strike_label"].nunique() == 1
    assert set(expiry_terminals["strike_match_mode"]) <= {"exact_strike", "same_moneyness", "nearest_numeric"}
    exact_rows = expiry_terminals.loc[expiry_terminals["used_strike_fallback"] == False]  # noqa: E712
    if not exact_rows.empty:
        assert set(exact_rows["strike_match_mode"]) == {"exact_strike"}
    assert best_of_terminals["candidate_slug"].nunique() <= 6
    assert best_of_terminals["expiry_date"].nunique() > 1
    assert best_of_terminals["moneyness_bucket"].nunique() > 1
    assert strike_terminals["selection_rank"].is_monotonic_increasing
    assert expiry_terminals["selection_rank"].is_monotonic_increasing
    assert best_of_terminals["selection_rank"].is_monotonic_increasing


def test_long_call_expiry_view_prefers_exact_then_same_moneyness_then_nearest_numeric():
    long_calls = pd.DataFrame(
        [
            {
                "candidate_slug": "anchor",
                "candidate_label": "15C Apr-2026",
                "expiry_date": "2026-04-17",
                "strike_label": "15.00",
                "primary_strike": 15.0,
                "moneyness_bucket": "near_atm",
                "source_trust_label": "trusted_quoted",
                "active_candidate_rank": 1,
                "objective_score": 90.0,
                "source_quote_coverage_pct": 100.0,
                "source_quote_usable": True,
            },
            {
                "candidate_slug": "exact-later",
                "candidate_label": "15C May-2026",
                "expiry_date": "2026-05-15",
                "strike_label": "15.00",
                "primary_strike": 15.0,
                "moneyness_bucket": "near_atm",
                "source_trust_label": "trusted_quoted",
                "active_candidate_rank": 2,
                "objective_score": 80.0,
                "source_quote_coverage_pct": 100.0,
                "source_quote_usable": True,
            },
            {
                "candidate_slug": "same-bucket-jun",
                "candidate_label": "16C Jun-2026",
                "expiry_date": "2026-06-18",
                "strike_label": "16.00",
                "primary_strike": 16.0,
                "moneyness_bucket": "near_atm",
                "source_trust_label": "trusted_quoted",
                "active_candidate_rank": 3,
                "objective_score": 70.0,
                "source_quote_coverage_pct": 100.0,
                "source_quote_usable": True,
            },
            {
                "candidate_slug": "numeric-sep",
                "candidate_label": "14C Sep-2026",
                "expiry_date": "2026-09-18",
                "strike_label": "14.00",
                "primary_strike": 14.0,
                "moneyness_bucket": "itm",
                "source_trust_label": "trusted_quoted",
                "active_candidate_rank": 4,
                "objective_score": 65.0,
                "source_quote_coverage_pct": 100.0,
                "source_quote_usable": True,
            },
            {
                "candidate_slug": "numeric-sep-alt",
                "candidate_label": "17C Sep-2026",
                "expiry_date": "2026-09-18",
                "strike_label": "17.00",
                "primary_strike": 17.0,
                "moneyness_bucket": "otm",
                "source_trust_label": "trusted_quoted",
                "active_candidate_rank": 5,
                "objective_score": 60.0,
                "source_quote_coverage_pct": 100.0,
                "source_quote_usable": True,
            },
        ]
    )

    selected = _select_long_call_expiry_view_rows(long_calls, anchor_row=long_calls.iloc[0])

    assert not selected.empty
    rows_by_expiry = {row["expiry_date"]: row for _, row in selected.iterrows()}
    assert rows_by_expiry["2026-04-17"]["strike_match_mode"] == "exact_strike"
    assert rows_by_expiry["2026-05-15"]["strike_match_mode"] == "exact_strike"
    assert rows_by_expiry["2026-06-18"]["strike_match_mode"] == "same_moneyness"
    assert bool(rows_by_expiry["2026-06-18"]["used_strike_fallback"]) is True
    assert rows_by_expiry["2026-09-18"]["strike_match_mode"] == "nearest_numeric"
    assert bool(rows_by_expiry["2026-09-18"]["used_strike_fallback"]) is True
    assert rows_by_expiry["2026-09-18"]["strike_label"] == "14.00"
    assert float(rows_by_expiry["2026-09-18"]["fallback_strike_distance"]) == 1.0


def test_selector_cards_emit_no_clear_edge_when_rows_compress():
    rows = pd.DataFrame(
        [
            {
                "strategy_family": "long_call",
                "strategy_label": "Long Call",
                "winning_candidate_label": "Long Call 15",
                "winning_candidate_slug": "long-call-15",
                "available": True,
                "objective_score": 10.0,
                "target_pnl": 0.0,
                "difference_vs_stock": 0.0,
                "target_return_pct": 0.0,
                "return_on_comparison_capital": 0.0,
                "max_loss": 100.0,
                "iv_down_value_change": 0.0,
                "iv_up_value_change": 0.0,
                "delayed_move_value_change": 0.0,
                "one_line_warning": "",
                "target_beyond_expiry": False,
            },
            {
                "strategy_family": "long_stock",
                "strategy_label": "Long Stock",
                "winning_candidate_label": "Long Stock Baseline",
                "winning_candidate_slug": "long-stock",
                "available": True,
                "objective_score": 10.0,
                "target_pnl": 0.0,
                "difference_vs_stock": 0.0,
                "target_return_pct": 0.0,
                "return_on_comparison_capital": 0.0,
                "max_loss": 100.0,
                "iv_down_value_change": 0.0,
                "iv_up_value_change": 0.0,
                "delayed_move_value_change": 0.0,
                "one_line_warning": "",
                "target_beyond_expiry": False,
            },
        ]
    )

    rankings, cards = _selector_cards(rows, objective_mode="max_return_at_target", comparison_capital=1000.0)

    assert not rankings.empty
    assert cards
    overall = rankings.loc[rankings["ranking_mode"] == "best_overall_current_objective"].iloc[0]
    assert overall["card_status"] == "no_clear_edge"
    assert not bool(overall["is_informative"])
    assert pd.isna(overall["winner_strategy"])
    assert "does not separate the families enough" in overall["reason"]


@pytest.mark.slow
def test_scenario_publish_embeds_contract_selection_path_case_data_and_ui_controls(temp_analysis_root: Path):
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
    publish_analysis_bundle(contract_bundle.bundle_dir)

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
    dashboard_path = publish_analysis_bundle(scenario_bundle.bundle_dir)
    html = dashboard_path.read_text(encoding="utf-8")
    published_manifest = json.loads((scenario_bundle.bundle_dir / "publish" / "published_manifest.json").read_text(encoding="utf-8"))

    assert published_manifest["analysis_kind"] == "scenario"
    assert "Strategy Selector" in html
    assert "Path Case Summary" in html
    assert "Path &amp; Contract Explorer" in html or "Path & Contract Explorer" in html
    assert "Required Stock Path by Strategy" in html
    assert "Assumed Path" in html
    assert "Case Outcome" in html
    assert 'data-path-case-root' in html
    assert 'data-path-case-case' in html
    assert 'data-path-case-goal' in html
    assert 'data-path-case-display-mode' in html
    assert 'data-path-case-iv-mode' in html
    assert 'data-path-case-state-note' in html
    assert 'data-path-state-note' in html
    assert 'data-replay-state-summary' in html
    assert "IV Variant" in html
    assert "IV Paths mode keeps" in html
    assert "same effective underlying coverage" in html.lower() or "clamped means" in html.lower()
    assert "C:/Users" not in html
    assert "C:\\Users" not in html
    assert "file:///" not in html
