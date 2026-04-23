import json
from pathlib import Path

import pandas as pd

from options_lab.analysis import (
    build_scenario_analysis,
    build_strategy_analysis,
    publish_analysis_bundle,
    write_analysis_bundle,
)
from options_lab.io import load_chain
from options_lab.plots import prepare_heatmap_matrix, strategy_visual_spec
from options_lab.scenarios import compare_positions, scenario_table
from options_lab.strategies import build_strategy
from options_lab.utils import build_stock_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SAMPLE_FILE = DATA_ROOT / "GPRE" / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"


def test_scenario_table_clamps_long_horizons_to_expiry():
    chain = load_chain(SAMPLE_FILE)
    strategy = build_strategy("long_call", chain)
    table = scenario_table(strategy, spot_grid=[10, 15, 20], horizons=["1m", "3m", "12m"])

    assert set(table["horizon"]) == {"1m", "3m", "12m"}
    assert table["clamped_to_expiry"].all()
    assert table["effective_days"].max() == 5


def test_compare_positions_generates_both_modes():
    chain = load_chain(SAMPLE_FILE)
    positions = [
        build_strategy("long_stock", chain),
        build_strategy("long_call", chain),
        build_strategy("bull_call_spread", chain),
    ]
    comparison = compare_positions(positions, mode="both", spot_grid=[10, 15, 20], comparison_capital=1000.0)

    assert set(comparison["mode"]) == {"share_equivalent", "equal_capital"}
    assert set(comparison["strategy"]) == {"long_stock", "long_call", "bull_call_spread"}
    equal_capital = comparison.loc[comparison["mode"] == "equal_capital"]
    assert (equal_capital["comparison_capital"] == 1000.0).all()
    assert {"unit_capital_required", "affordable_units", "fully_implementable_with_budget", "budget_note", "return_on_comparison_capital"}.issubset(equal_capital.columns)


def test_strategy_analysis_bundle_and_publish_create_bundle_backed_dashboard(temp_analysis_root: Path):
    chain = load_chain(
        SAMPLE_FILE,
        prices_data_root=DATA_ROOT,
        rates_data_root=DATA_ROOT,
        research_data_root=DATA_ROOT,
    )
    long_stock = build_strategy("long_stock", chain)
    long_call = build_strategy("long_call", chain)
    bull_call_spread = build_strategy("bull_call_spread", chain)

    analysis = build_strategy_analysis(
        bull_call_spread,
        spot_grid=build_stock_grid(chain.spot_price or 15.0, points=31),
        comparison_positions=[long_stock, long_call, bull_call_spread],
    )
    bundle = write_analysis_bundle(
        analysis,
        analysis_kind="strategy",
        output_root=temp_analysis_root,
    )
    dashboard_path = publish_analysis_bundle(bundle.bundle_dir)

    manifest = json.loads((bundle.bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((bundle.bundle_dir / "metadata" / "report_metadata.json").read_text(encoding="utf-8"))
    summary_md = (bundle.bundle_dir / "summary" / "summary.md").read_text(encoding="utf-8")
    dashboard_html = dashboard_path.read_text(encoding="utf-8")

    assert manifest["analysis_kind"] == "strategy"
    assert (bundle.bundle_dir / "tables" / "summary.csv").exists()
    assert (bundle.bundle_dir / "charts" / "payoff_at_expiry.png").exists()
    assert payload["strategy_report"]["resolved_metadata"]["spot_price_source"] is not None
    assert "risk_free_rate_source" in payload["strategy_report"]["resolved_metadata"]
    assert "## Research Context" in summary_md
    assert "Bull Call Spread Dashboard" in dashboard_html
    assert "Scenario Table" in dashboard_html
    assert "payoff_at_expiry.png" in dashboard_html
    assert "https://" not in dashboard_html
    assert "http://" not in dashboard_html


def test_multi_strategy_scenario_builder_includes_stock_and_reports_omissions():
    chain = load_chain(SAMPLE_FILE)
    result = build_scenario_analysis(
        chain,
        strategies=["long_stock", "long_call", "bull_call_spread", "unsupported_demo"],
        comparison_capital=1000.0,
    )

    assert result.available_strategies[0] == "long_stock"
    assert "long_call" in result.available_strategies
    assert "bull_call_spread" in result.available_strategies
    assert any(item["strategy"] == "unsupported_demo" for item in result.omitted_strategies)
    assert not result.stock_relative.empty
    assert set(result.stock_relative["mode"]) == {"share_equivalent", "equal_capital"}
    assert set(result.named_scenarios["spot_case"]) >= {"bear", "flat", "bull"}
    assert result.comparison_capital == 1000.0
    assert result.capital_sizing_mode == "hybrid"
    assert {"comparison_capital", "unit_capital_required", "affordable_units", "fully_implementable_with_budget", "budget_note"}.issubset(result.strategy_summary.columns)
    assert result.decision_hints["best_bull_case"]["strategy"] is not None
    assert result.featured_focus_strategy in result.available_strategies
    assert result.replay_defaults["horizon"]
    assert result.replay_defaults["spot_case"]
    assert result.valuation_defaults["strategy"] in result.available_strategies
    assert bool(result.what_matters_most)
    assert result.forward_defaults["mode"] == "spot_time"
    assert not result.forward_quick_scenarios.empty
    assert not result.forward_spot_time_grid.empty
    assert not result.forward_spot_iv_grid.empty
    assert not result.forward_time_iv_grid.empty
    assert not result.valuation_explanation.empty


def test_strategy_visual_specs_are_unique_and_color_safe():
    strategies = [
        "long_stock",
        "long_call",
        "bull_call_spread",
        "long_put",
        "bear_put_spread",
        "covered_call",
        "cash_secured_put",
    ]
    specs = {name: strategy_visual_spec(name) for name in strategies}

    assert len({spec["color"] for spec in specs.values()}) == len(strategies)
    assert len({spec["marker"] for spec in specs.values()}) == len(strategies)
    assert len({spec["linestyle"] for spec in specs.values()}) >= 4
    assert specs["long_stock"]["color"] == "#000000"
    assert specs["long_call"]["marker"] == "o"
    assert specs["covered_call"]["linestyle"] != specs["long_call"]["linestyle"]


def test_prepare_heatmap_matrix_respects_display_order():
    frame = pd.DataFrame(
        [
            {"spot_price": 10.0, "horizon": "entry", "profit_loss": -2.0},
            {"spot_price": 10.0, "horizon": "1w", "profit_loss": -1.0},
            {"spot_price": 10.0, "horizon": "expiry", "profit_loss": 3.0},
            {"spot_price": 10.0, "iv_case": "iv_down", "estimated_value": 1.0},
            {"spot_price": 10.0, "iv_case": "iv_unchanged", "estimated_value": 2.0},
            {"spot_price": 10.0, "iv_case": "iv_up", "estimated_value": 4.0},
        ]
    )

    time_matrix = prepare_heatmap_matrix(
        frame.dropna(subset=["horizon"]),
        x_column="spot_price",
        y_column="horizon",
        value_column="profit_loss",
        y_order=["expiry", "1w", "entry"],
    )
    iv_matrix = prepare_heatmap_matrix(
        frame.dropna(subset=["iv_case"]),
        x_column="spot_price",
        y_column="iv_case",
        value_column="estimated_value",
        y_order=["iv_up", "iv_unchanged", "iv_down"],
    )

    assert list(time_matrix.index) == ["expiry", "1w", "entry"]
    assert list(iv_matrix.index) == ["iv_up", "iv_unchanged", "iv_down"]


def test_prepare_heatmap_matrix_auto_orders_horizons_and_iv_cases_intuitively():
    frame = pd.DataFrame(
        [
            {"spot_price": 10.0, "horizon": "3m", "profit_loss": 2.0},
            {"spot_price": 10.0, "horizon": "entry", "profit_loss": 0.0},
            {"spot_price": 10.0, "horizon": "1w", "profit_loss": 1.0},
            {"spot_price": 10.0, "horizon": "expiry", "profit_loss": 3.0},
            {"spot_price": 10.0, "iv_case": "iv_up", "estimated_value": 4.0},
            {"spot_price": 10.0, "iv_case": "iv_down", "estimated_value": 1.0},
            {"spot_price": 10.0, "iv_case": "iv_unchanged", "estimated_value": 2.0},
        ]
    )

    time_matrix = prepare_heatmap_matrix(
        frame.dropna(subset=["horizon"]),
        x_column="spot_price",
        y_column="horizon",
        value_column="profit_loss",
    )
    iv_matrix = prepare_heatmap_matrix(
        frame.dropna(subset=["iv_case"]),
        x_column="spot_price",
        y_column="iv_case",
        value_column="estimated_value",
    )

    assert list(time_matrix.index) == ["entry", "1w", "3m", "expiry"]
    assert list(iv_matrix.index) == ["iv_down", "iv_unchanged", "iv_up"]


def test_scenario_analysis_publish_creates_primary_html(temp_analysis_root: Path):
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
    metadata = json.loads((bundle.bundle_dir / "metadata" / "report_metadata.json").read_text(encoding="utf-8"))

    assert "Primary Scenario Dashboard" in html
    assert "Compare vs Stock" in html
    assert "Forward Scenario Lab" in html
    assert "Strategy Selector" in html
    assert "Path Case Summary" in html
    assert "Path &amp; Contract Explorer" in html or "Path & Contract Explorer" in html
    assert "Explain Valuation" in html
    assert "Replay / Case View" in html
    assert "What Matters Most Here?" in html
    assert "Decision Snapshot" in html
    assert "Decision Hints" in html
    assert "Strategy Deep Dives" in html
    assert "data-forward-lab-root" in html
    assert "data-path-explorer-root" in html
    assert "data-replay-root" in html
    assert 'id="dashboard-lightbox-html"' in html
    assert metadata["report_kind"] == "scenario"
