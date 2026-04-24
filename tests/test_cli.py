from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_lab.cli import _build_parser, main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SAMPLE_FILE = DATA_ROOT / "GPRE" / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"


def test_cli_help_mentions_canonical_analyze_publish_workflow(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])

    captured = capsys.readouterr()
    assert "analyze-*" in captured.out
    assert "analysis_outputs/" in captured.out
    assert "model_outputs/" in captured.out
    assert "publish-analysis" in captured.out
    assert "build-model-outputs" in captured.out


def test_cli_parser_accepts_inspect_and_metadata_override_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "inspect",
            "--file",
            "data/GPRE/sample.csv",
            "--metadata-file",
            "override.json",
            "--prices-data-root",
            "custom_prices",
            "--rates-data-root",
            "custom_rates",
            "--research-data-root",
            "custom_research",
            "--target-strike",
            "15.0",
        ]
    )

    assert args.command == "inspect"
    assert args.metadata_file == "override.json"
    assert args.prices_data_root == "custom_prices"
    assert args.rates_data_root == "custom_rates"
    assert args.research_data_root == "custom_research"
    assert args.target_strike == 15.0


def test_cli_parser_accepts_canonical_analyze_and_publish_commands():
    parser = _build_parser()
    contract_args = parser.parse_args(
        [
            "analyze-contract-selection",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--target-price",
            "20",
            "--target-date",
            "2026-07-15",
            "--thesis-target-price",
            "30",
            "--thesis-target-date",
            "2026-12-18",
            "--single-option-candidate-slug",
            "long-call-2026-12-18-15-00",
            "--minimum-outperformance-multiple",
            "1.6",
            "--strong-outperformance-multiple",
            "2.1",
            "--required-winning-path-families",
            "3",
            "--entry-price-mode",
            "mid",
            "--single-option-exit-rule",
            "sell_at_target_return",
            "--single-option-target-return-pct",
            "0.75",
            "--goal",
            "itm_1c",
            "--stock-path-mode",
            "mixed",
            "--stock-path-target-end",
            "21",
            "--iv-path-mode",
            "mixed",
            "--simulated-path-count",
            "12",
            "--representative-selection-mode",
            "goal_buckets",
            "--simulation-seed",
            "17",
        ]
    )
    scenario_args = parser.parse_args(
        [
            "analyze-scenario",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--expiry-date",
            "2026-04-17",
        ]
    )
    replay_args = parser.parse_args(
        [
            "analyze-replay",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--expiry-date",
            "2026-04-17",
            "--strategy",
            "long_call",
        ]
    )
    strategy_args = parser.parse_args(
        [
            "analyze-strategy",
            "--file",
            str(SAMPLE_FILE),
            "--strategy",
            "long_call",
        ]
    )
    publish_args = parser.parse_args(
        [
            "publish-analysis",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--analysis-kind",
            "contract_selection",
            "--run-slug",
            "demo-run",
            "--mirror-dashboards",
            "--dashboards-root",
            "Dashboards",
        ]
    )
    model_outputs_args = parser.parse_args(
        [
            "build-model-outputs",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--analysis-kind",
            "contract_selection",
            "--run-slug",
            "demo-run",
            "--model-root",
            "model_outputs",
        ]
    )
    refresh_prices_args = parser.parse_args(
        [
            "refresh-local-prices",
            "--ticker",
            "GPRE",
            "--start",
            "2026-04-01",
            "--end",
            "2026-04-12",
            "--full-refresh",
        ]
    )
    refresh_rates_args = parser.parse_args(
        [
            "refresh-risk-free-rates",
            "--start",
            "2026-04-01",
            "--end",
            "2026-04-12",
            "--series",
            "DGS1MO",
            "--full-refresh",
        ]
    )

    assert contract_args.command == "analyze-contract-selection"
    assert contract_args.goal == "itm_1c"
    assert contract_args.thesis_target_price == 30.0
    assert contract_args.thesis_target_date == "2026-12-18"
    assert contract_args.single_option_candidate_slug == "long-call-2026-12-18-15-00"
    assert contract_args.minimum_outperformance_multiple == 1.6
    assert contract_args.strong_outperformance_multiple == 2.1
    assert contract_args.required_winning_path_families == 3
    assert contract_args.entry_price_mode == "mid"
    assert contract_args.single_option_exit_rule == "sell_at_target_return"
    assert contract_args.single_option_target_return_pct == 0.75
    assert contract_args.stock_path_mode == "mixed"
    assert contract_args.stock_path_target_end == 21.0
    assert contract_args.iv_path_mode == "mixed"
    assert contract_args.simulated_path_count == 12
    assert contract_args.representative_selection_mode == "goal_buckets"
    assert contract_args.simulation_seed == 17
    assert scenario_args.command == "analyze-scenario"
    assert replay_args.command == "analyze-replay"
    assert strategy_args.command == "analyze-strategy"
    assert publish_args.command == "publish-analysis"
    assert publish_args.mirror_dashboards is True
    assert model_outputs_args.command == "build-model-outputs"
    assert model_outputs_args.model_root == "model_outputs"
    assert refresh_prices_args.command == "refresh-local-prices"
    assert refresh_prices_args.full_refresh is True
    assert refresh_rates_args.command == "refresh-risk-free-rates"
    assert refresh_rates_args.series == ["DGS1MO"]


def test_cli_analyze_strategy_maps_target_selection_flags(capsys, temp_analysis_root: Path):
    exit_code = main(
        [
            "analyze-strategy",
            "--file",
            str(SAMPLE_FILE),
            "--strategy",
            "long_call",
            "--target-strike",
            "15",
            "--output-root",
            str(temp_analysis_root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    metadata_path = Path(payload["bundle_dir"]) / "metadata" / "report_metadata.json"
    report_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["analysis_kind"] == "strategy"
    assert report_metadata["strategy_report"]["selection_inputs"]["contract_selector"] == {"target_strike": 15.0}
    assert report_metadata["strategy_report"]["summary"]["primary_strike"] == 15.0


def test_cli_refresh_commands_emit_manifest_backed_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "options_lab.prices.nasdaq_downloader.download_nasdaq_prices",
        lambda **kwargs: {
            "request_window": {"start": kwargs.get("start"), "end": kwargs.get("end")},
            "row_count": 12,
            "latest_date": "2026-04-12",
            "manifest_path": "tmp/prices-manifest.json",
        },
    )
    monkeypatch.setattr(
        "options_lab.rates.fred_downloader.download_fred_rates",
        lambda **kwargs: {
            "requested_series": kwargs.get("series_ids") or [],
            "request_window": {"start": kwargs.get("start"), "end": kwargs.get("end")},
            "merged_row_count": 25,
            "manifest_path": "tmp/rates-manifest.json",
        },
    )

    prices_exit = main(
        [
            "refresh-local-prices",
            "--ticker",
            "GPRE",
            "--start",
            "2026-04-01",
            "--end",
            "2026-04-12",
        ]
    )
    prices_payload = json.loads(capsys.readouterr().out)
    assert prices_exit == 0
    assert prices_payload["command"] == "refresh-local-prices"
    assert prices_payload["row_count"] == 12

    rates_exit = main(
        [
            "refresh-risk-free-rates",
            "--start",
            "2026-04-01",
            "--end",
            "2026-04-12",
            "--series",
            "DGS1MO",
        ]
    )
    rates_payload = json.loads(capsys.readouterr().out)
    assert rates_exit == 0
    assert rates_payload["command"] == "refresh-risk-free-rates"
    assert rates_payload["requested_series"] == ["DGS1MO"]


def test_cli_parser_accepts_ibkr_commands_and_debug_flags():
    parser = _build_parser()
    chain_args = parser.parse_args(
        [
            "fetch-ibkr-chain",
            "--ticker",
            "AAPL",
            "--port",
            "7496",
            "--market-data-mode",
            "delayed",
            "--include-all-exchanges",
            "--debug",
        ]
    )
    option_args = parser.parse_args(
        [
            "fetch-ibkr-options-snapshot",
            "--ticker",
            "AAPL",
            "--expiry",
            "20260619",
            "--right",
            "both",
            "--around-spot",
            "2",
            "--max-contracts",
            "4",
            "--include-all-exchanges",
        ]
    )
    full_chain_args = parser.parse_args(
        [
            "fetch-ibkr-full-chain-snapshot",
            "--ticker",
            "GPRE",
            "--market-data-mode",
            "delayed",
            "--per-expiry-timeout",
            "90",
            "--sparse-retry-wait",
            "4",
            "--include-all-exchanges",
            "--debug",
        ]
    )

    assert chain_args.command == "fetch-ibkr-chain"
    assert chain_args.port == 7496
    assert chain_args.include_all_exchanges is True
    assert chain_args.debug is True
    assert option_args.command == "fetch-ibkr-options-snapshot"
    assert option_args.expiry == ["20260619"]
    assert option_args.around_spot == 2
    assert option_args.max_contracts == 4
    assert full_chain_args.command == "fetch-ibkr-full-chain-snapshot"
    assert full_chain_args.market_data_mode == "delayed"
    assert full_chain_args.per_expiry_timeout == 90.0
    assert full_chain_args.sparse_retry_wait == 4.0
    assert full_chain_args.include_all_exchanges is True
    assert full_chain_args.retry_sparse_quotes_once is True
    assert full_chain_args.debug is True


def test_cli_parser_rejects_unknown_commands():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["definitely-not-a-command"])


@pytest.mark.slow
def test_cli_analyze_and_publish_commands_emit_bundle_backed_json(capsys, temp_analysis_root: Path):
    analyze_exit = main(
        [
            "analyze-contract-selection",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--target-price",
            "20",
            "--target-date",
            "2026-07-15",
            "--output-root",
            str(temp_analysis_root),
            "--data-root",
            str(DATA_ROOT),
        ]
    )
    analyze_payload = json.loads(capsys.readouterr().out)
    assert analyze_exit == 0
    assert analyze_payload["analysis_kind"] == "contract_selection"
    assert Path(analyze_payload["bundle_dir"]).exists()
    assert analyze_payload["top_family"]
    assert analyze_payload["top_candidate"]
    assert analyze_payload["top_expiry"]
    assert analyze_payload["top_strike"]
    assert analyze_payload["family_edge_status"]
    assert analyze_payload["active_goal"] == "break_even"
    assert analyze_payload["stock_path_name"] == "slow_bull"
    assert analyze_payload["iv_path_name"] == "flat"
    assert "required_path_summary.csv" in analyze_payload["key_tables"]
    assert "iv_path_trace_rows.csv" in analyze_payload["key_tables"]
    assert "compare_vs_stock_path_rows.csv" in analyze_payload["key_tables"]
    assert "path_risk_summary.csv" in analyze_payload["key_tables"]
    assert "stock_path_examples.csv" in analyze_payload["key_tables"]
    assert "iv_path_examples.csv" in analyze_payload["key_tables"]
    assert "stock_path_gallery.csv" in analyze_payload["key_tables"]
    assert "iv_path_gallery.csv" in analyze_payload["key_tables"]
    assert "path_pair_summary.csv" in analyze_payload["key_tables"]
    assert "option_value_over_path.csv" in analyze_payload["key_tables"]
    assert "compare_vs_stock_over_path.csv" in analyze_payload["key_tables"]
    assert "representative_paths_summary.csv" in analyze_payload["key_tables"]
    assert "strike_comparison_under_path.csv" in analyze_payload["key_tables"]
    assert "expiry_comparison_under_path.csv" in analyze_payload["key_tables"]
    assert "required_vs_assumed_path_summary.csv" in analyze_payload["key_tables"]
    assert "family_comparison.csv" in analyze_payload["key_tables"]
    assert "candidate_comparison.csv" in analyze_payload["key_tables"]
    assert "strike_comparison.csv" in analyze_payload["key_tables"]
    assert "expiry_comparison.csv" in analyze_payload["key_tables"]
    assert "required_path_vs_assumed_path.png" in analyze_payload["key_charts"]
    assert "stock_path_gallery.png" in analyze_payload["key_charts"]
    assert "iv_path_gallery.png" in analyze_payload["key_charts"]
    assert "representative_stock_paths.png" in analyze_payload["key_charts"]
    assert "representative_iv_paths.png" in analyze_payload["key_charts"]
    assert "option_value_over_path.png" in analyze_payload["key_charts"]
    assert "compare_vs_stock_over_path.png" in analyze_payload["key_charts"]
    assert "strike_comparison_under_same_path.png" in analyze_payload["key_charts"]
    assert "expiry_comparison_under_same_path.png" in analyze_payload["key_charts"]
    assert "required_path_strategy_compare.png" in analyze_payload["key_charts"]
    assert "iv_path_trace.png" in analyze_payload["key_charts"]
    assert "compare_vs_stock_path_delta.png" in analyze_payload["key_charts"]
    assert "family_ranking_overview.png" in analyze_payload["key_charts"]
    assert analyze_payload["top_path_risk"]
    assert analyze_payload["timing_risk"]
    assert analyze_payload["iv_risk"]
    assert analyze_payload["spot_source"]
    assert "spot_matched_date" in analyze_payload
    assert analyze_payload["risk_free_rate_source"]
    assert analyze_payload["source_snapshot_storage_locations"]
    assert analyze_payload["source_snapshot_files"]
    assert "chain_source_summary.csv" in analyze_payload["key_tables"]
    assert "market_context_summary.csv" in analyze_payload["key_tables"]
    assert "expected_move_matched" in analyze_payload
    assert "nearest_event_type" in analyze_payload

    publish_exit = main(
        [
            "publish-analysis",
            "--bundle",
            analyze_payload["bundle_dir"],
        ]
    )
    publish_payload = json.loads(capsys.readouterr().out)
    assert publish_exit == 0
    assert Path(publish_payload["dashboard_path"]).exists()
    assert publish_payload["analysis_kind"] == "contract_selection"


def test_cli_build_model_outputs_projects_existing_bundle_without_recomputing_analysis(
    monkeypatch,
    capsys,
    temp_analysis_root: Path,
    temp_workspace_root: Path,
):
    from tests.test_model_outputs import _create_fake_contract_selection_bundle

    bundle_dir = _create_fake_contract_selection_bundle(
        temp_workspace_root,
        run_slug="cli-demo-run",
    )
    model_root = temp_workspace_root / "model_outputs"

    def _should_not_run(*args, **kwargs):
        raise AssertionError("analysis should not be recomputed when building model outputs")

    monkeypatch.setattr("options_lab.cli.build_contract_selection_analysis", _should_not_run)

    exit_code = main(
        [
            "build-model-outputs",
            "--bundle",
            str(bundle_dir),
            "--model-root",
            str(model_root),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["command"] == "build-model-outputs"
    assert payload["analysis_kind"] == "contract_selection"
    assert payload["ticker"] == "GPRE"
    assert Path(payload["model_output_dir"]).exists()
    assert Path(payload["latest_dir"]).exists()
    assert Path(payload["start_here_path"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert "03_tables/chain_source_summary.csv" in payload["promoted_files"]
    assert "04_secondary/stock_path_gallery.png" in payload["promoted_files"]
    assert "04_secondary/iv_path_gallery.png" in payload["promoted_files"]
    assert "04_secondary/required_path_vs_assumed_path.png" in payload["promoted_files"]
    assert "04_secondary/compare_vs_stock_path_delta.png" in payload["promoted_files"]
    assert any(path.endswith("/long_call_strike_value.png") for path in payload["promoted_files"])
    assert any(path.endswith("/long_call_best_of_delta.png") for path in payload["promoted_files"])
