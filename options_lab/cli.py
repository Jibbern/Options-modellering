"""Canonical command-line entrypoints for Options Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analysis import (
    DEFAULT_ANALYSIS_OUTPUT_ROOT,
    build_contract_selection_analysis,
    build_replay_analysis,
    build_scenario_analysis,
    build_strategy_analysis,
    resolve_analysis_bundle,
    write_analysis_bundle,
)
from .io import load_chain, select_contract
from .persistence import make_json_safe
from .publish import publish_analysis_bundle
from .research_metadata import (
    build_ticker_catalog,
    list_events,
    register_dividends_file,
    register_events_file,
    register_expected_move_file,
    register_notes_file,
    register_options_overview_file,
    resolve_research_context,
)
from .snapshots import available_snapshot_dates, comparison_ready_expiries, list_snapshot_slices
from .strategies import build_strategy
from .utils import build_stock_grid, clean_string


STRATEGY_CHOICES = [
    "long_stock",
    "long_call",
    "long_put",
    "covered_call",
    "cash_secured_put",
    "bull_call_spread",
    "bear_put_spread",
]


def _selection_args(args, *, use_short: bool = False) -> dict[str, float | str]:
    payload: dict[str, float | str] = {}
    strike_value = args.short_target_strike if use_short else args.target_strike
    delta_value = args.short_target_delta if use_short else args.target_delta
    if strike_value is not None:
        payload["target_strike"] = strike_value
    if delta_value is not None:
        payload["target_delta"] = delta_value
    if getattr(args, "pct_otm", None) is not None:
        payload["pct_otm"] = args.pct_otm
    return payload


def _strategy_build_kwargs(args) -> dict[str, Any]:
    selector = _selection_args(args)
    short_selector = _selection_args(args, use_short=True)
    payload: dict[str, Any] = {
        "spot_price": args.spot_price,
        "premium_mode": args.premium_mode,
    }
    if selector:
        payload["contract_selector"] = selector
        payload["long_selector"] = selector
    if short_selector:
        payload["short_selector"] = short_selector
    return payload


def _comparison_positions_for_strategy(chain, args) -> list | None:
    strategies = getattr(args, "strategies", None)
    if not strategies:
        return None
    positions = []
    for name in strategies:
        positions.append(build_strategy(name, chain, spot_price=args.spot_price, premium_mode=args.premium_mode))
    return positions


def _scenario_spot_case_overrides(args) -> dict[str, float]:
    payload: dict[str, float] = {}
    for label in ["far_bear", "bear", "bull", "strong_bull"]:
        value = getattr(args, f"{label}_pct", None)
        if value is not None:
            payload[label] = value
    return payload


def _scenario_iv_case_overrides(args) -> dict[str, float]:
    payload: dict[str, float] = {}
    if getattr(args, "iv_down_points", None) is not None:
        payload["iv_down"] = args.iv_down_points
    if getattr(args, "iv_up_points", None) is not None:
        payload["iv_up"] = args.iv_up_points
    return payload


def _load_metadata_override(path: str | None) -> dict | None:
    if not clean_string(path):
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _print_json(payload) -> None:
    print(json.dumps(make_json_safe(payload), indent=2, allow_nan=False))


def _emit_model_outputs_projection(payload: dict[str, Any]) -> int:
    _print_json(
        {
            "command": "build-model-outputs",
            "analysis_kind": payload.get("analysis_kind"),
            "ticker": payload.get("ticker"),
            "snapshot_date": payload.get("snapshot_date"),
            "run_slug": payload.get("run_slug"),
            "source_bundle_dir": payload.get("source_bundle_dir"),
            "source_bundle_path": payload.get("source_bundle_path"),
            "model_output_dir": payload.get("model_output_dir"),
            "latest_dir": payload.get("latest_dir"),
            "start_here_path": payload.get("start_here_path"),
            "manifest_path": payload.get("manifest_path"),
            "promoted_files": payload.get("promoted_files") or [],
        }
    )
    return 0


def _emit_analysis_bundle(bundle, *, analysis_kind: str) -> int:
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    key_tables = sorted((manifest.get("file_map") or {}).get("tables", {}).keys())
    key_charts = sorted((manifest.get("file_map") or {}).get("charts", {}).keys())
    report_metadata_path = bundle.bundle_dir / "metadata" / "report_metadata.json"
    report_metadata = json.loads(report_metadata_path.read_text(encoding="utf-8")) if report_metadata_path.exists() else {}
    contract_meta = report_metadata.get("metadata") or {}
    research_context = contract_meta.get("research_context") or {}
    default_summary = {}
    summary_csv = bundle.bundle_dir / "tables" / "summary.csv"
    if summary_csv.exists():
        try:
            import csv

            with summary_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            default_summary = rows[0] if rows else {}
        except Exception:
            default_summary = {}
    _print_json(
        {
            "analysis_kind": analysis_kind,
            "ticker": bundle.ticker,
            "snapshot_date": bundle.snapshot_date,
            "run_slug": bundle.run_slug,
            "bundle_dir": str(bundle.bundle_dir),
            "manifest_path": str(bundle.manifest_path),
            "warnings": manifest.get("warnings") or [],
            "key_tables": key_tables,
            "key_charts": key_charts,
            "top_family": default_summary.get("best_family"),
            "top_candidate": default_summary.get("best_candidate"),
            "top_expiry": default_summary.get("best_expiry"),
            "top_strike": default_summary.get("best_strike"),
            "family_edge_status": default_summary.get("family_edge_status"),
            "active_goal": contract_meta.get("goal"),
            "stock_path_name": contract_meta.get("stock_path_name"),
            "stock_path_mode": contract_meta.get("stock_path_mode"),
            "iv_path_name": contract_meta.get("iv_path_name"),
            "iv_path_mode": contract_meta.get("iv_path_mode"),
            "simulated_path_count": contract_meta.get("simulated_path_count"),
            "representative_selection_mode": contract_meta.get("representative_selection_mode"),
            "top_path_risk": default_summary.get("top_path_risk"),
            "timing_risk": default_summary.get("timing_risk"),
            "iv_risk": default_summary.get("iv_risk"),
            "spot_source": contract_meta.get("spot_price_source"),
            "spot_field_used": contract_meta.get("spot_field_used"),
            "spot_matched_date": contract_meta.get("spot_price_matched_date"),
            "spot_quality_note": contract_meta.get("spot_quality_note"),
            "risk_free_rate_source": contract_meta.get("risk_free_rate_source"),
            "risk_free_rate_series": contract_meta.get("risk_free_rate_series"),
            "risk_free_rate_matched_date": contract_meta.get("risk_free_rate_matched_date"),
            "analysis_trust_level": contract_meta.get("analysis_trust_level"),
            "trusted_expiry_count": contract_meta.get("trusted_expiry_count"),
            "fallback_only_expiry_count": contract_meta.get("fallback_only_expiry_count"),
            "source_snapshot_storage_locations": contract_meta.get("source_snapshot_storage_locations") or [],
            "source_snapshot_files": contract_meta.get("source_snapshot_files") or [],
            "expected_move_matched": contract_meta.get("expected_move_matched"),
            "nearest_event_type": contract_meta.get("nearest_event_type")
            or (research_context.get("nearest_event") or {}).get("event_type"),
        }
    )
    return 0


def _ibkr_connection_settings(args):
    from .ibkr import ConnectionSettings

    return ConnectionSettings(host=args.host, port=args.port, client_id=args.client_id)


def _emit_ibkr_failure(
    *,
    ticker: str,
    request_type: str,
    market_data_mode: str,
    connection,
    error: Exception,
    data_root: str | None,
    diagnostics: dict | None = None,
) -> int:
    from .ibkr.store import record_request_failure

    result = record_request_failure(
        ticker,
        request_type=request_type,
        market_data_mode=market_data_mode,
        connection=connection,
        error_message=str(error),
        warnings=[],
        diagnostics=diagnostics,
        data_root=data_root,
    )
    _print_json(
        {
            "status": "failed",
            "ticker": clean_string(ticker).upper(),
            "request_type": request_type,
            "market_data_mode": market_data_mode,
            "error": str(error),
            "failure_stage": diagnostics.get("failure_stage") if diagnostics else None,
            "diagnostics": diagnostics,
            "files": result,
        }
    )
    return 1


def _add_common_snapshot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file", required=True, help="Path to an option-chain CSV snapshot.")
    parser.add_argument("--spot-price", type=float, help="Explicit underlying spot-price override.")
    parser.add_argument("--metadata-file", help="Optional JSON file with metadata overrides.")
    parser.add_argument("--prices-data-root", help="Optional local historical-price data root.")
    parser.add_argument("--rates-data-root", help="Optional local rates data root.")
    parser.add_argument("--research-data-root", help="Optional local research metadata root.")
    parser.add_argument("--premium-mode", default="mid", choices=["mid", "bid", "ask", "last", "conservative"])


def _add_strategy_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-delta", type=float)
    parser.add_argument("--target-strike", type=float)
    parser.add_argument("--pct-otm", type=float)
    parser.add_argument("--short-target-strike", type=float)
    parser.add_argument("--short-target-delta", type=float)


def _add_scenario_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--expiry-date", required=True)
    parser.add_argument("--data-root", help="Override the Options data root.")
    parser.add_argument("--spot-price", type=float)
    parser.add_argument("--premium-mode", default="mid", choices=["mid", "bid", "ask", "last", "conservative"])
    parser.add_argument("--output-root", default=str(DEFAULT_ANALYSIS_OUTPUT_ROOT))
    parser.add_argument("--strategies", nargs="+", default=["long_stock", "long_call", "bull_call_spread", "long_put", "bear_put_spread", "covered_call", "cash_secured_put"], choices=STRATEGY_CHOICES)
    _add_strategy_selection_args(parser)
    parser.add_argument("--far-bear-pct", type=float)
    parser.add_argument("--bear-pct", type=float)
    parser.add_argument("--bull-pct", type=float)
    parser.add_argument("--strong-bull-pct", type=float)
    parser.add_argument("--iv-down-points", type=float)
    parser.add_argument("--iv-up-points", type=float)
    parser.add_argument("--comparison-capital", type=float, default=1000.0)


def _add_contract_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--target-price", required=True, type=float)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-date")
    target_group.add_argument("--target-horizon")
    parser.add_argument("--thesis-target-price", type=float, help="Optional thesis-mode target price, e.g. 30 for a $30 Dec-2026 thesis.")
    parser.add_argument("--thesis-target-date", help="Optional thesis-mode target date, e.g. 2026-12-18.")
    parser.add_argument("--single-option-candidate-slug", help="Optional long-call candidate slug for the single-option decision view.")
    parser.add_argument("--minimum-outperformance-multiple", type=float, default=1.5)
    parser.add_argument("--strong-outperformance-multiple", type=float, default=2.0)
    parser.add_argument("--required-winning-path-families", type=int, default=2)
    parser.add_argument("--minimum-edge-stock-return-pct", type=float, default=0.05)
    parser.add_argument("--entry-price-mode", default="conservative_mid_plus_slippage", choices=["conservative_mid_plus_slippage", "mid", "ask_or_mid"])
    parser.add_argument("--single-option-exit-rule", default="sell_on_thesis_completion", choices=["hold_to_expiry", "sell_at_target_return", "sell_on_thesis_completion"])
    parser.add_argument("--single-option-target-return-pct", type=float, default=0.50)
    parser.add_argument("--iv-shift-points", type=float, default=0.0)
    parser.add_argument("--comparison-capital", type=float, default=1000.0)
    parser.add_argument("--goal", default="break_even", choices=["itm_1c", "break_even", "return_25", "return_50", "outperform_stock", "target_option_value"])
    parser.add_argument("--target-option-value", type=float)
    parser.add_argument("--strategy-families", nargs="+", default=["long_stock", "long_call", "bull_call_spread", "long_put", "bear_put_spread", "covered_call", "cash_secured_put"], choices=STRATEGY_CHOICES)
    parser.add_argument("--objective-mode", default="max_return_at_target", choices=["max_return_at_target", "outperform_stock", "capital_efficiency", "downside_control", "robustness_iv_fall", "move_takes_time", "highest_convexity"])
    parser.add_argument("--downside-tolerance", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--simplicity-preference", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--strike-selection-mode", default="top_n")
    parser.add_argument("--expiry-selection-mode", default="auto")
    parser.add_argument("--stock-path-preset", default="slow_bull")
    parser.add_argument("--stock-path-points")
    parser.add_argument("--stock-path-mode", default="mixed", choices=["deterministic", "simulated", "conditioned", "mixed"])
    parser.add_argument("--stock-path-target-end", type=float)
    parser.add_argument("--iv-path-preset", default="flat")
    parser.add_argument("--iv-path-points")
    parser.add_argument("--iv-path-mode", default="mixed", choices=["active_only", "presets", "mixed", "noisy"])
    parser.add_argument("--simulated-path-count", type=int, default=18)
    parser.add_argument("--representative-selection-mode", default="goal_buckets", choices=["goal_buckets"])
    parser.add_argument("--simulation-seed", type=int)
    parser.add_argument("--data-root", help="Override the Options data root.")
    parser.add_argument("--output-root", default=str(DEFAULT_ANALYSIS_OUTPUT_ROOT))


def _add_replay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--expiry-date", required=True)
    parser.add_argument("--strategy", required=True, choices=STRATEGY_CHOICES)
    parser.add_argument("--data-root", help="Override the Options data root.")
    parser.add_argument("--output-root", default=str(DEFAULT_ANALYSIS_OUTPUT_ROOT))
    parser.add_argument("--premium-mode", default="mid", choices=["mid", "bid", "ask", "last", "conservative"])
    parser.add_argument("--spot-price", type=float)
    parser.add_argument("--comparison-capital", type=float, default=1000.0)
    _add_strategy_selection_args(parser)


def _add_ibkr_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=71)
    parser.add_argument("--market-data-mode", choices=["delayed", "delayed_frozen"], default="delayed")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--data-root")
    parser.add_argument("--debug", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Options Lab is a Python-first options analysis toolkit. "
            "Run analyze-* to build canonical bundles under analysis_outputs/, "
            "promote the curated analyst view into model_outputs/, "
            "then use publish-analysis to render frozen HTML."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one local chain snapshot.")
    _add_common_snapshot_args(inspect_parser)
    inspect_parser.add_argument("--option-type", choices=["call", "put"])
    inspect_parser.add_argument("--expiry")
    _add_strategy_selection_args(inspect_parser)
    inspect_parser.add_argument("--limit", type=int, default=8)

    strategy_parser = subparsers.add_parser("analyze-strategy", help="Run strategy analysis and write a canonical bundle.")
    _add_common_snapshot_args(strategy_parser)
    strategy_parser.add_argument("--output-root", default=str(DEFAULT_ANALYSIS_OUTPUT_ROOT))
    strategy_parser.add_argument("--strategy", required=True, choices=STRATEGY_CHOICES)
    strategy_parser.add_argument("--strategies", nargs="+", choices=STRATEGY_CHOICES)
    _add_strategy_selection_args(strategy_parser)

    scenario_parser = subparsers.add_parser("analyze-scenario", help="Run scenario analysis and write a canonical bundle.")
    _add_scenario_args(scenario_parser)

    contract_parser = subparsers.add_parser("analyze-contract-selection", help="Run contract-selection analysis and write a canonical bundle.")
    _add_contract_selection_args(contract_parser)

    replay_parser = subparsers.add_parser("analyze-replay", help="Run replay analysis and write a canonical bundle.")
    _add_replay_args(replay_parser)

    publish_parser = subparsers.add_parser("publish-analysis", help="Render frozen HTML from an existing canonical bundle.")
    publish_parser.add_argument("--bundle")
    publish_parser.add_argument("--ticker")
    publish_parser.add_argument("--snapshot-date")
    publish_parser.add_argument("--analysis-kind")
    publish_parser.add_argument("--run-slug")
    publish_parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_OUTPUT_ROOT))
    publish_parser.add_argument("--destination")
    publish_parser.add_argument("--mirror-dashboards", action="store_true")
    publish_parser.add_argument("--dashboards-root")

    model_outputs_parser = subparsers.add_parser(
        "build-model-outputs",
        help="Promote a frozen bundle into the curated model_outputs/ analyst view.",
    )
    model_outputs_parser.add_argument("--bundle")
    model_outputs_parser.add_argument("--ticker")
    model_outputs_parser.add_argument("--snapshot-date")
    model_outputs_parser.add_argument("--analysis-kind")
    model_outputs_parser.add_argument("--run-slug")
    model_outputs_parser.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_OUTPUT_ROOT))
    model_outputs_parser.add_argument("--model-root", default=str(Path(__file__).resolve().parents[1] / "model_outputs"))

    refresh_prices = subparsers.add_parser(
        "refresh-local-prices",
        help="Refresh the local Nasdaq-backed historical price store for one ticker.",
    )
    refresh_prices.add_argument("--ticker", required=True)
    refresh_prices.add_argument("--data-root")
    refresh_prices.add_argument("--start")
    refresh_prices.add_argument("--end")
    refresh_prices.add_argument("--full-refresh", action="store_true")

    refresh_rates = subparsers.add_parser(
        "refresh-risk-free-rates",
        help="Refresh the local FRED/Treasury risk-free store.",
    )
    refresh_rates.add_argument("--data-root")
    refresh_rates.add_argument("--start")
    refresh_rates.add_argument("--end")
    refresh_rates.add_argument("--full-refresh", action="store_true")
    refresh_rates.add_argument("--series", action="append")

    import_barchart_options = subparsers.add_parser(
        "import-barchart-options",
        help="Import a manually downloaded Barchart Options Screener CSV into the local chain store.",
    )
    import_barchart_options.add_argument("--ticker", required=True)
    import_barchart_options.add_argument("--csv", required=True)
    import_barchart_options.add_argument("--snapshot-date", required=True)
    import_barchart_options.add_argument("--entry-mode", default="mid", choices=["mid", "ask", "realistic"])
    import_barchart_options.add_argument("--calls-only", action="store_true", default=True)
    import_barchart_options.add_argument("--include-puts", action="store_true")
    import_barchart_options.add_argument("--min-ask", type=float, default=0.0)
    import_barchart_options.add_argument("--min-iv", type=float, default=0.0001)
    import_barchart_options.add_argument("--min-dte", type=int, default=1)
    import_barchart_options.add_argument("--max-dte", type=int, default=900)
    import_barchart_options.add_argument("--min-open-interest", type=int)
    import_barchart_options.add_argument("--allow-zero-volume", action="store_true", default=True)
    import_barchart_options.add_argument("--source", default="barchart_options_screener")
    import_barchart_options.add_argument("--trust-level", default="manually_downloaded_barchart")
    import_barchart_options.add_argument("--data-root")

    import_barchart_prices = subparsers.add_parser(
        "import-barchart-price-history",
        help="Import a manually downloaded Barchart price-history CSV into the local price store.",
    )
    import_barchart_prices.add_argument("--ticker", required=True)
    import_barchart_prices.add_argument("--csv", required=True)
    import_barchart_prices.add_argument("--data-root")

    snapshots_parser = subparsers.add_parser("list-snapshots", help="List discovered local chain snapshots for a ticker.")
    snapshots_parser.add_argument("--ticker", required=True)
    snapshots_parser.add_argument("--data-root")

    metadata_parser = subparsers.add_parser("inspect-metadata", help="Inspect local research metadata and resolved context.")
    metadata_parser.add_argument("--ticker", required=True)
    metadata_parser.add_argument("--snapshot-date")
    metadata_parser.add_argument("--expiry-date")
    metadata_parser.add_argument("--data-root")

    for command, help_text in [
        ("register-expected-move", "Register an expected-move dataset file."),
        ("register-options-overview", "Register an options-overview dataset file."),
        ("register-events", "Register an events dataset file."),
        ("register-dividends", "Register a dividends dataset file."),
        ("register-notes", "Register a notes dataset file."),
    ]:
        sub = subparsers.add_parser(command, help=help_text)
        sub.add_argument("--ticker", required=True)
        sub.add_argument("--file", required=True)
        sub.add_argument("--data-root")

    list_events_parser = subparsers.add_parser("list-events", help="List local research events and nearest-event context.")
    list_events_parser.add_argument("--ticker", required=True)
    list_events_parser.add_argument("--snapshot-date")
    list_events_parser.add_argument("--expiry-date")
    list_events_parser.add_argument("--data-root")

    ibkr_underlying = subparsers.add_parser("fetch-ibkr-underlying", help="Fetch one delayed IBKR underlying quote snapshot.")
    _add_ibkr_common_args(ibkr_underlying)
    ibkr_underlying.add_argument("--exchange")
    ibkr_underlying.add_argument("--primary-exchange")
    ibkr_underlying.add_argument("--currency", default="USD")

    ibkr_chain = subparsers.add_parser("fetch-ibkr-chain", help="Fetch IBKR option-chain metadata under delayed mode.")
    _add_ibkr_common_args(ibkr_chain)
    ibkr_chain.add_argument("--exchange", default="SMART")
    ibkr_chain.add_argument("--currency", default="USD")
    ibkr_chain.add_argument("--include-all-exchanges", action="store_true")

    ibkr_options = subparsers.add_parser("fetch-ibkr-options-snapshot", help="Fetch a delayed IBKR option snapshot slice.")
    _add_ibkr_common_args(ibkr_options)
    ibkr_options.add_argument("--expiry", action="append")
    ibkr_options.add_argument("--right", default="both", choices=["call", "put", "both"])
    ibkr_options.add_argument("--min-strike", type=float)
    ibkr_options.add_argument("--max-strike", type=float)
    ibkr_options.add_argument("--strike", action="append", type=float)
    ibkr_options.add_argument("--around-spot", type=int)
    ibkr_options.add_argument("--max-contracts", type=int)
    ibkr_options.add_argument("--exchange", default="SMART")
    ibkr_options.add_argument("--currency", default="USD")
    ibkr_options.add_argument("--include-all-exchanges", action="store_true")

    ibkr_full_chain = subparsers.add_parser("fetch-ibkr-full-chain-snapshot", help="Fetch one delayed-only full quoted IBKR chain snapshot.")
    _add_ibkr_common_args(ibkr_full_chain)
    ibkr_full_chain.add_argument("--exchange", default="SMART")
    ibkr_full_chain.add_argument("--currency", default="USD")
    ibkr_full_chain.add_argument(
        "--per-expiry-timeout",
        type=float,
        default=90.0,
        help="Maximum wait per expiry batch while delayed quotes populate in TWS/IB Gateway.",
    )
    ibkr_full_chain.add_argument(
        "--no-retry-sparse-quotes",
        action="store_false",
        dest="retry_sparse_quotes_once",
        help="Disable the one-time sparse-quote retry after a short wait.",
    )
    ibkr_full_chain.add_argument(
        "--sparse-retry-wait",
        type=float,
        default=3.0,
        help="Seconds to wait before retrying sparse quotes once inside each expiry batch.",
    )
    ibkr_full_chain.add_argument("--include-all-exchanges", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "publish-analysis":
        try:
            bundle_dir = resolve_analysis_bundle(
                bundle=args.bundle,
                ticker=args.ticker,
                snapshot_date=args.snapshot_date,
                analysis_kind=args.analysis_kind,
                run_slug=args.run_slug,
                output_root=args.analysis_root,
            )
            dashboard_path = publish_analysis_bundle(
                bundle_dir,
                destination=args.destination,
                dashboards_root=args.dashboards_root,
                publish_dashboards=args.mirror_dashboards,
            )
        except (FileNotFoundError, ValueError) as exc:
            _print_json({"error": str(exc)})
            return 1
        _print_json(
            {
                "bundle_dir": str(bundle_dir),
                "analysis_kind": clean_string(args.analysis_kind) or Path(bundle_dir).parent.name,
                "dashboard_path": str(dashboard_path),
                "mirrored_dashboards": bool(args.mirror_dashboards),
            }
        )
        return 0

    if args.command == "build-model-outputs":
        from .model_outputs import build_model_outputs

        try:
            payload = build_model_outputs(
                bundle=args.bundle,
                ticker=args.ticker,
                snapshot_date=args.snapshot_date,
                analysis_kind=args.analysis_kind,
                run_slug=args.run_slug,
                analysis_root=args.analysis_root,
                model_root=args.model_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            _print_json({"error": str(exc)})
            return 1
        return _emit_model_outputs_projection(payload)

    if args.command == "refresh-local-prices":
        from .prices.nasdaq_downloader import download_nasdaq_prices

        manifest = download_nasdaq_prices(
            ticker=args.ticker,
            data_root=args.data_root,
            start=args.start,
            end=args.end,
            full_refresh=args.full_refresh,
        )
        _print_json(
            {
                "command": "refresh-local-prices",
                "ticker": clean_string(args.ticker).upper(),
                "data_root": args.data_root,
                "request_window": manifest.get("request_window"),
                "row_count": manifest.get("row_count"),
                "latest_date": manifest.get("latest_date"),
                "manifest_path": manifest.get("manifest_path"),
            }
        )
        return 0

    if args.command == "refresh-risk-free-rates":
        from .rates.fred_downloader import FredApiKeyError, download_fred_rates

        try:
            manifest = download_fred_rates(
                data_root=args.data_root,
                start=args.start,
                end=args.end,
                series_ids=args.series,
                full_refresh=args.full_refresh,
            )
        except FredApiKeyError as exc:
            _print_json(
                {
                    "command": "refresh-risk-free-rates",
                    "status": "failed",
                    "error": str(exc),
                }
            )
            return 2
        _print_json(
            {
                "command": "refresh-risk-free-rates",
                "data_root": args.data_root,
                "requested_series": manifest.get("requested_series"),
                "request_window": manifest.get("request_window"),
                "merged_row_count": manifest.get("merged_row_count"),
                "manifest_path": manifest.get("manifest_path"),
            }
        )
        return 0

    if args.command == "import-barchart-options":
        from .barchart import import_barchart_options_csv

        try:
            result = import_barchart_options_csv(
                ticker=args.ticker,
                csv_path=args.csv,
                snapshot_date=args.snapshot_date,
                data_root=args.data_root,
                entry_mode=args.entry_mode,
                calls_only=args.calls_only,
                include_puts=args.include_puts,
                min_ask=args.min_ask,
                min_iv=args.min_iv,
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                min_open_interest=args.min_open_interest,
                allow_zero_volume=args.allow_zero_volume,
                source=args.source,
                trust_level=args.trust_level,
            )
        except (FileNotFoundError, ValueError) as exc:
            _print_json({"command": "import-barchart-options", "status": "failed", "error": str(exc)})
            return 1
        _print_json(
            {
                "command": "import-barchart-options",
                "status": "succeeded",
                "ticker": result.ticker,
                "source": result.source,
                "trust_level": result.trust_level,
                "snapshot_date": result.snapshot_date,
                "raw_csv_path": result.raw_csv_path,
                "normalized_output_paths": result.normalized_output_paths,
                "manifest_path": result.manifest_path,
                "rows_raw": result.rows_raw,
                "rows_after_footer_cleanup": result.rows_after_footer_cleanup,
                "rows_for_ticker": result.rows_for_ticker,
                "rows_model_eligible": result.rows_model_eligible,
            }
        )
        return 0

    if args.command == "import-barchart-price-history":
        from .barchart import import_barchart_price_history_csv

        try:
            result = import_barchart_price_history_csv(args.ticker, args.csv, data_root=args.data_root)
        except (FileNotFoundError, ValueError) as exc:
            _print_json({"command": "import-barchart-price-history", "status": "failed", "error": str(exc)})
            return 1
        result["status"] = "succeeded"
        _print_json(result)
        return 0

    if args.command == "fetch-ibkr-underlying":
        from .ibkr.market_data import fetch_underlying_snapshot
        from .ibkr.store import save_underlying_snapshot

        settings = _ibkr_connection_settings(args)
        try:
            snapshot = fetch_underlying_snapshot(
                args.ticker,
                settings=settings,
                market_data_mode=args.market_data_mode,
                exchange=args.exchange,
                primary_exchange=args.primary_exchange,
                currency=args.currency,
                timeout=args.timeout,
            )
            result = save_underlying_snapshot(snapshot, data_root=args.data_root)
            _print_json(
                {
                    "ticker": snapshot.ticker,
                    "market_data_mode": snapshot.market_data_mode,
                    "effective_market_data_type": snapshot.market_data_type_code,
                    "warnings": snapshot.warnings,
                    "missing_fields": snapshot.missing_fields,
                    "resolved_underlying_contract": snapshot.resolved_underlying.to_dict() if args.debug and snapshot.resolved_underlying else None,
                    "files": result,
                }
            )
            return 0
        except Exception as exc:
            return _emit_ibkr_failure(
                ticker=args.ticker,
                request_type="underlying",
                market_data_mode=args.market_data_mode,
                connection=settings,
                error=exc,
                data_root=args.data_root,
            )

    if args.command == "fetch-ibkr-chain":
        from .ibkr.chains import fetch_option_chain
        from .ibkr.store import save_chain_rows

        settings = _ibkr_connection_settings(args)
        try:
            rows = fetch_option_chain(
                args.ticker,
                settings=settings,
                market_data_mode=args.market_data_mode,
                exchange=args.exchange,
                currency=args.currency,
                timeout=args.timeout,
                include_all_exchanges=args.include_all_exchanges,
            )
            if not rows.rows:
                return _emit_ibkr_failure(
                    ticker=args.ticker,
                    request_type="chain",
                    market_data_mode=args.market_data_mode,
                    connection=settings,
                    error=LookupError("No usable chain rows were produced from the IBKR option-parameter response."),
                    data_root=args.data_root,
                    diagnostics=rows.diagnostics.to_dict(),
                )
            result = save_chain_rows(rows, data_root=args.data_root)
            _print_json(
                {
                    "ticker": args.ticker.upper(),
                    "market_data_mode": args.market_data_mode,
                    "row_count": len(rows.rows),
                    "expiries": sorted({row.expiry_date for row in rows.rows}),
                    "resolved_underlying_contract": rows.diagnostics.resolved_underlying.to_dict() if rows.diagnostics.resolved_underlying else None,
                    "raw_opt_param_row_count": rows.diagnostics.row_counts.get("raw_opt_param_rows"),
                    "final_chain_row_count": rows.diagnostics.row_counts.get("final_chain_rows"),
                    "exchanges_seen": rows.diagnostics.raw_exchanges_seen,
                    "trading_classes_seen": rows.diagnostics.raw_trading_classes_seen,
                    "available_expiries": rows.diagnostics.available_expiries[:10],
                    "available_strike_sample": rows.diagnostics.available_strike_sample,
                    "diagnostics": rows.diagnostics.to_dict() if args.debug else None,
                    "files": result,
                }
            )
            return 0
        except Exception as exc:
            return _emit_ibkr_failure(
                ticker=args.ticker,
                request_type="chain",
                market_data_mode=args.market_data_mode,
                connection=settings,
                error=exc,
                data_root=args.data_root,
                diagnostics=getattr(exc, "diagnostics", None),
            )

    if args.command == "fetch-ibkr-options-snapshot":
        from .ibkr.market_data import fetch_option_snapshots
        from .ibkr.store import save_option_snapshot

        settings = _ibkr_connection_settings(args)
        try:
            quotes = fetch_option_snapshots(
                args.ticker,
                settings=settings,
                market_data_mode=args.market_data_mode,
                expiries=args.expiry or None,
                right=args.right,
                min_strike=args.min_strike,
                max_strike=args.max_strike,
                strikes=args.strike or None,
                around_spot=args.around_spot,
                max_contracts=args.max_contracts,
                exchange=args.exchange,
                currency=args.currency,
                timeout=args.timeout,
                include_all_exchanges=args.include_all_exchanges,
            )
            if not quotes.quotes:
                return _emit_ibkr_failure(
                    ticker=args.ticker,
                    request_type="option_snapshot",
                    market_data_mode=args.market_data_mode,
                    connection=settings,
                    error=LookupError("No IBKR option contracts matched the requested filters after diagnostic narrowing."),
                    data_root=args.data_root,
                    diagnostics=quotes.diagnostics.to_dict(),
                )
            result = save_option_snapshot(quotes, data_root=args.data_root)
            _print_json(
                {
                    "ticker": args.ticker.upper(),
                    "market_data_mode": args.market_data_mode,
                    "row_count": len(quotes.quotes),
                    "fields_unavailable": sorted({field for quote in quotes.quotes for field in quote.missing_fields}),
                    "warnings": sorted({warning for quote in quotes.quotes for warning in quote.warnings}),
                    "matched_expiries": quotes.diagnostics.final_selected_expiries,
                    "selected_strikes": quotes.diagnostics.final_selected_strikes,
                    "selected_exchange_count": len(quotes.diagnostics.final_selected_exchanges),
                    "diagnostics": quotes.diagnostics.to_dict() if args.debug else None,
                    "files": result,
                }
            )
            return 0
        except Exception as exc:
            return _emit_ibkr_failure(
                ticker=args.ticker,
                request_type="option_snapshot",
                market_data_mode=args.market_data_mode,
                connection=settings,
                error=exc,
                data_root=args.data_root,
                diagnostics=getattr(exc, "diagnostics", None),
            )

    if args.command == "fetch-ibkr-full-chain-snapshot":
        from .ibkr.market_data import fetch_full_chain_snapshot
        from .ibkr.store import save_chain_rows, save_full_chain_snapshot_run, save_option_snapshot, save_underlying_snapshot

        settings = _ibkr_connection_settings(args)
        try:
            result = fetch_full_chain_snapshot(
                args.ticker,
                settings=settings,
                market_data_mode=args.market_data_mode,
                exchange=args.exchange,
                currency=args.currency,
                timeout=args.timeout,
                per_expiry_timeout=args.per_expiry_timeout,
                retry_sparse_quotes_once=args.retry_sparse_quotes_once,
                sparse_retry_wait_seconds=args.sparse_retry_wait,
                include_all_exchanges=args.include_all_exchanges,
            )
            if not result.option_snapshot.quotes:
                return _emit_ibkr_failure(
                    ticker=args.ticker,
                    request_type="full_chain_snapshot",
                    market_data_mode=args.market_data_mode,
                    connection=settings,
                    error=LookupError("No delayed IBKR option quotes were persisted for the discovered full chain."),
                    data_root=args.data_root,
                    diagnostics=result.option_snapshot.diagnostics.to_dict(),
                )
            underlying_files = save_underlying_snapshot(result.underlying, data_root=args.data_root)
            chain_files = save_chain_rows(result.chain, data_root=args.data_root)
            quote_files = save_option_snapshot(result.option_snapshot, data_root=args.data_root)
            full_chain_files = save_full_chain_snapshot_run(
                args.ticker,
                market_data_mode=args.market_data_mode,
                connection=settings,
                underlying_files=underlying_files,
                chain_files=chain_files,
                option_snapshot_files=quote_files,
                option_snapshot=result.option_snapshot,
                data_root=args.data_root,
            )
            _print_json(
                {
                    "ticker": args.ticker.upper(),
                    "market_data_mode": args.market_data_mode,
                    "snapshot_scope": result.option_snapshot.diagnostics.snapshot_scope,
                    "discovered_expiries": result.option_snapshot.diagnostics.discovered_expiries,
                    "strike_count_by_expiry": result.option_snapshot.diagnostics.strike_count_by_expiry,
                    "attempted_contract_count": result.option_snapshot.diagnostics.attempted_contract_count,
                    "persisted_quote_count": len(result.option_snapshot.quotes),
                    "coverage_summary": (result.option_snapshot.diagnostics.delayed_field_summary or {}).get("coverage_summary"),
                    "per_expiry_timeout": args.per_expiry_timeout,
                    "retry_sparse_quotes_once": args.retry_sparse_quotes_once,
                    "warnings": sorted({warning for quote in result.option_snapshot.quotes for warning in quote.warnings}),
                    "fields_unavailable": sorted({field for quote in result.option_snapshot.quotes for field in quote.missing_fields}),
                    "diagnostics": result.option_snapshot.diagnostics.to_dict() if args.debug else None,
                    "files": {
                        "full_chain_snapshot": full_chain_files,
                        "underlying": underlying_files,
                        "chain": chain_files,
                        "option_snapshot": quote_files,
                    },
                }
            )
            return 0
        except Exception as exc:
            return _emit_ibkr_failure(
                ticker=args.ticker,
                request_type="full_chain_snapshot",
                market_data_mode=args.market_data_mode,
                connection=settings,
                error=exc,
                data_root=args.data_root,
                diagnostics=getattr(exc, "diagnostics", None),
            )

    if args.command == "list-snapshots":
        slices = list_snapshot_slices(args.ticker, data_root=args.data_root)
        expiries_by_snapshot = {
            str(snapshot_date)[:10]: [str(value)[:10] for value in group["expiry_date"].dropna().sort_values().unique().tolist()]
            for snapshot_date, group in slices.groupby("snapshot_date")
        } if not slices.empty else {}
        _print_json(
            {
                "ticker": args.ticker.upper(),
                "available_snapshot_dates": available_snapshot_dates(args.ticker, args.data_root),
                "comparison_ready_expiries": comparison_ready_expiries(args.ticker, args.data_root),
                "expiries_by_snapshot": expiries_by_snapshot,
                "snapshot_slices": slices.assign(
                    snapshot_date=slices["snapshot_date"].astype(str) if "snapshot_date" in slices.columns else [],
                    expiry_date=slices["expiry_date"].astype(str) if "expiry_date" in slices.columns else [],
                ).to_dict(orient="records"),
            }
        )
        return 0

    if args.command == "inspect-metadata":
        resolved_context = (
            resolve_research_context(
                args.ticker,
                snapshot_date=args.snapshot_date,
                expiry_date=args.expiry_date,
                data_root=args.data_root,
            )
            if args.snapshot_date
            else None
        )
        payload = {
            "ticker_catalog": build_ticker_catalog(args.ticker, data_root=args.data_root),
        }
        if resolved_context is not None:
            payload["resolved_context"] = resolved_context
            payload["research_context"] = resolved_context
        _print_json(payload)
        return 0

    if args.command == "register-expected-move":
        _print_json(register_expected_move_file(args.ticker, args.file, data_root=args.data_root))
        return 0
    if args.command == "register-options-overview":
        _print_json(register_options_overview_file(args.ticker, args.file, data_root=args.data_root))
        return 0
    if args.command == "register-events":
        _print_json(register_events_file(args.ticker, args.file, data_root=args.data_root))
        return 0
    if args.command == "register-dividends":
        _print_json(register_dividends_file(args.ticker, args.file, data_root=args.data_root))
        return 0
    if args.command == "register-notes":
        _print_json(register_notes_file(args.ticker, args.file, data_root=args.data_root))
        return 0
    if args.command == "list-events":
        payload = {
            "ticker": args.ticker.upper(),
            "events": list_events(args.ticker, data_root=args.data_root),
            "nearest_event": resolve_research_context(
                args.ticker,
                snapshot_date=args.snapshot_date,
                expiry_date=args.expiry_date,
                data_root=args.data_root,
            ).get("nearest_event", {}) if args.snapshot_date else {},
        }
        _print_json(payload)
        return 0

    if args.command == "analyze-contract-selection":
        try:
            result = build_contract_selection_analysis(
                ticker=args.ticker,
                snapshot_date=args.snapshot_date,
                target_price=args.target_price,
                target_date=args.target_date,
                target_horizon=args.target_horizon,
                iv_shift_points=args.iv_shift_points,
                comparison_capital=args.comparison_capital,
                strategy_families=args.strategy_families,
                strike_selection_mode=args.strike_selection_mode,
                expiry_selection_mode=args.expiry_selection_mode,
                goal=args.goal,
                target_option_value=args.target_option_value,
                objective_mode=args.objective_mode,
                downside_tolerance=args.downside_tolerance,
                simplicity_preference=args.simplicity_preference,
                stock_path_preset=args.stock_path_preset,
                stock_path_points=args.stock_path_points,
                stock_path_mode=args.stock_path_mode,
                stock_path_target_end=args.stock_path_target_end,
                iv_path_preset=args.iv_path_preset,
                iv_path_points=args.iv_path_points,
                iv_path_mode=args.iv_path_mode,
                simulated_path_count=args.simulated_path_count,
                representative_selection_mode=args.representative_selection_mode,
                simulation_seed=args.simulation_seed,
                thesis_target_price=args.thesis_target_price,
                thesis_target_date=args.thesis_target_date,
                single_option_candidate_slug=args.single_option_candidate_slug,
                minimum_outperformance_multiple=args.minimum_outperformance_multiple,
                strong_outperformance_multiple=args.strong_outperformance_multiple,
                required_winning_path_families=args.required_winning_path_families,
                minimum_edge_stock_return_pct=args.minimum_edge_stock_return_pct,
                entry_price_mode=args.entry_price_mode,
                single_option_exit_rule=args.single_option_exit_rule,
                single_option_target_return_pct=args.single_option_target_return_pct,
                data_root=args.data_root,
            )
        except ValueError as exc:
            _print_json({"error": str(exc)})
            return 1
        bundle = write_analysis_bundle(result, analysis_kind="contract_selection", output_root=args.output_root)
        return _emit_analysis_bundle(bundle, analysis_kind="contract_selection")

    if args.command == "analyze-scenario":
        try:
            result = build_scenario_analysis(
                ticker=args.ticker,
                snapshot_date=args.snapshot_date,
                expiry_date=args.expiry_date,
                data_root=args.data_root,
                spot_price=args.spot_price,
                premium_mode=args.premium_mode,
                strategies=args.strategies,
                contract_selector=_selection_args(args),
                long_selector=_selection_args(args),
                short_selector=_selection_args(args, use_short=True),
                spot_case_moves=_scenario_spot_case_overrides(args),
                iv_case_points=_scenario_iv_case_overrides(args),
                comparison_capital=args.comparison_capital,
            )
        except ValueError as exc:
            _print_json({"error": str(exc)})
            return 1
        bundle = write_analysis_bundle(result, analysis_kind="scenario", output_root=args.output_root)
        return _emit_analysis_bundle(bundle, analysis_kind="scenario")

    if args.command == "analyze-replay":
        try:
            result = build_replay_analysis(
                args.ticker,
                snapshot_date=args.snapshot_date,
                expiry_date=args.expiry_date,
                strategy_name=args.strategy,
                data_root=args.data_root,
                premium_mode=args.premium_mode,
                spot_price=args.spot_price,
                comparison_capital=args.comparison_capital,
                contract_selector=_selection_args(args),
                long_selector=_selection_args(args),
                short_selector=_selection_args(args, use_short=True),
            )
        except ValueError as exc:
            _print_json({"error": str(exc)})
            return 1
        bundle = write_analysis_bundle(result, analysis_kind="replay", output_root=args.output_root)
        return _emit_analysis_bundle(bundle, analysis_kind="replay")

    if args.command == "analyze-strategy":
        chain = load_chain(
            args.file,
            metadata_override=_load_metadata_override(args.metadata_file),
            spot_price=args.spot_price,
            prices_data_root=args.prices_data_root,
            rates_data_root=args.rates_data_root,
            research_data_root=args.research_data_root,
        )
        strategy = build_strategy(args.strategy, chain, **_strategy_build_kwargs(args))
        analysis = build_strategy_analysis(
            strategy,
            spot_grid=build_stock_grid(chain.spot_price or strategy.entry_spot).tolist(),
            comparison_positions=_comparison_positions_for_strategy(chain, args) or [],
            comparison_mode="both",
        )
        bundle = write_analysis_bundle(analysis, analysis_kind="strategy", output_root=args.output_root)
        return _emit_analysis_bundle(bundle, analysis_kind="strategy")

    if args.command == "inspect":
        chain = load_chain(
            args.file,
            metadata_override=_load_metadata_override(args.metadata_file),
            spot_price=args.spot_price,
            prices_data_root=args.prices_data_root,
            rates_data_root=args.rates_data_root,
            research_data_root=args.research_data_root,
        )
        frame = chain.contracts.copy()
        if args.option_type:
            frame = frame[frame["option_type"] == args.option_type]
        if getattr(args, "expiry", None):
            frame = frame[frame["expiry_date"].astype(str) == args.expiry]
        preview = frame[["option_type", "strike", "bid", "mid", "ask", "last", "iv", "delta", "open_interest"]].head(args.limit)
        payload = {
            "metadata": chain.metadata.to_dict(),
            "warnings": chain.warnings,
            "preview": preview.to_dict(orient="records"),
        }
        if args.option_type:
            contract = select_contract(
                chain,
                args.option_type,
                expiry=getattr(args, "expiry", None),
                target_delta=args.target_delta,
                target_strike=args.target_strike,
                pct_otm=args.pct_otm,
            )
            payload["selected_contract"] = contract.__dict__
        _print_json(payload)
        return 0

    parser.error("Unsupported command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
