from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from options_lab.analysis import build_contract_selection_analysis
from options_lab.analysis.market_context import resolve_market_context
from options_lab.barchart import (
    import_barchart_options_csv,
    import_barchart_price_history_csv,
)
from options_lab.io import load_chain
from options_lab.research_metadata.catalog import discover_chain_snapshots


def _write_options_screener_fixture(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'Symbol,Type,Price~,Latest,"Exp Date",DTE,Strike,Moneyness,Bid,Ask,Mid,Volume,"Open Int",IV,"IV Rank","IV Pctl",Delta,Gamma,Theta,Vega,Rho,"ITM Prob","OTM Prob","Profit Prob","BE (Bid)","BE (Ask)","BE (Mid)",Links',
                'GPRE,Call,17.50,0.00,12/18/26,227,15,+14.29%,3.20,3.60,3.40,0,"7,403",67.11%,+0.39%,45.00%,0.61,0.05,-0.01,0.03,0.02,61.2%,38.8%,52.0%,18.20,18.60,18.40,https://example.test/call',
                'GPRE,Call,17.50,0.10,2026-12-18,227,20,-14.29%,0.80,1.60,1.20,12,100,72.50%,+1.00%,50.00%,0.34,0.04,-0.02,0.04,0.01,34.0%,66.0%,40.0%,20.80,21.60,21.20,https://example.test/wide',
                'GPRE,Call,17.50,0.05,2026-05-15,10,18,-2.86%,0.15,0.25,0.20,0,0,0.00%,+0.00%,0.00%,0.25,0.02,-0.03,0.01,0.00,25.0%,75.0%,20.0%,18.15,18.25,18.20,https://example.test/invalid-iv',
                'GPRE,Put,17.50,0.30,2026-12-18,227,15,+14.29%,0.35,0.45,0.40,5,200,69.00%,+2.00%,51.00%,-0.31,0.03,-0.01,0.02,-0.01,31.0%,69.0%,30.0%,14.65,14.55,14.60,https://example.test/put',
                '"Downloaded from Barchart.com as of 05-05-2026 07:59am CDT"',
            ]
        ),
        encoding="utf-8",
    )


def _write_price_history_fixture(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Time,Open,High,Low,Latest,Change,%Change,Volume",
                '2026-05-04,17.54,18.09,17.35,18.07,0.31,+1.75%,"1,195,900"',
                '2026-05-01,17.38,18.08,17.38,17.76,0.38,+2.19%,"1,426,400"',
                '"Downloaded from Barchart.com as of 05-05-2026 08:09am CDT"',
            ]
        ),
        encoding="utf-8",
    )


def _write_high_iv_options_fixture(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'Symbol,Type,Price~,Latest,"Exp Date",DTE,Strike,Moneyness,Bid,Ask,Mid,Volume,"Open Int",IV,"IV Rank","IV Pctl",Delta,Gamma,Theta,Vega,Rho,"ITM Prob","OTM Prob","Profit Prob","BE (Bid)","BE (Ask)","BE (Mid)",Links',
                'GPRE,Call,18.07,0.00,05/15/26,10,18,+0.39%,1.10,1.35,1.23,12,"7,403",100.46%,+0.39%,51.00%,0.51,0.05,-0.02,0.04,0.01,51.00%,49.00%,51.00%,19.10,19.35,19.23,https://example.test/high-iv',
                '"Downloaded from Barchart.com as of 05-05-2026 07:59am CDT"',
            ]
        ),
        encoding="utf-8",
    )


def test_barchart_options_import_normalizes_quotes_and_manifest(temp_data_root: Path, temp_workspace_root: Path):
    source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    _write_options_screener_fixture(source)

    result = import_barchart_options_csv(
        ticker="GPRE",
        csv_path=source,
        snapshot_date="2026-05-05",
        data_root=temp_data_root,
    )

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    normalized = pd.read_csv(next(path for path in result.normalized_output_paths if "2026-12-18" in path))

    assert Path(result.raw_csv_path).exists()
    assert source.exists()
    assert manifest["rows_after_footer_cleanup"] == 4
    assert manifest["calls_count"] == 3
    assert manifest["puts_count"] == 1
    assert manifest["rows_model_eligible"] == 2
    assert "2026-12-18" in manifest["expiries"]
    assert normalized["option_type"].eq("call").all()
    assert normalized.loc[normalized["strike"].eq(15.0), "last"].iloc[0] == pytest.approx(0.0)
    assert normalized.loc[normalized["strike"].eq(15.0), "open_interest"].iloc[0] == 7403
    assert normalized.loc[normalized["strike"].eq(15.0), "implied_volatility"].iloc[0] == pytest.approx(0.6711)
    assert normalized.loc[normalized["strike"].eq(15.0), "iv_rank"].iloc[0] == pytest.approx(0.0039)
    assert normalized.loc[normalized["strike"].eq(15.0), "moneyness"].iloc[0] == pytest.approx(0.1429)
    assert normalized.loc[normalized["strike"].eq(15.0), "entry_premium_mid"].iloc[0] == pytest.approx(3.40)
    assert normalized.loc[normalized["strike"].eq(15.0), "entry_premium_ask"].iloc[0] == pytest.approx(3.60)
    assert normalized.loc[normalized["strike"].eq(15.0), "entry_premium_realistic"].iloc[0] == pytest.approx(3.50)
    assert "latest_zero" in normalized.loc[normalized["strike"].eq(15.0), "quality_flags"].iloc[0]
    assert "wide_spread" in normalized.loc[normalized["strike"].eq(20.0), "quality_flags"].iloc[0]


def test_barchart_load_chain_preserves_normalized_decimal_units(temp_data_root: Path, temp_workspace_root: Path):
    source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    _write_high_iv_options_fixture(source)

    result = import_barchart_options_csv(
        ticker="GPRE",
        csv_path=source,
        snapshot_date="2026-05-05",
        data_root=temp_data_root,
    )
    normalized_path = Path(result.normalized_output_paths[0])
    normalized = pd.read_csv(normalized_path)
    chain = load_chain(normalized_path, prices_data_root=temp_data_root, research_data_root=temp_data_root)
    row = chain.contracts.iloc[0]

    assert normalized["implied_volatility"].iloc[0] == pytest.approx(1.0046)
    assert row["iv"] == pytest.approx(1.0046)
    assert row["implied_volatility"] == pytest.approx(1.0046)
    assert row["moneyness"] == pytest.approx(0.0039)
    assert row["moneyness_decimal"] == pytest.approx(0.0039)
    assert row["itm_probability"] == pytest.approx(0.51)
    assert row["profit_probability"] == pytest.approx(0.51)


def test_barchart_options_import_can_include_puts(temp_data_root: Path, temp_workspace_root: Path):
    source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    _write_options_screener_fixture(source)

    result = import_barchart_options_csv(
        ticker="GPRE",
        csv_path=source,
        snapshot_date="2026-05-05",
        data_root=temp_data_root,
        include_puts=True,
    )

    normalized = pd.concat([pd.read_csv(path) for path in result.normalized_output_paths], ignore_index=True)
    assert set(normalized["option_type"]) == {"call", "put"}


def test_barchart_price_history_import_updates_local_price_store(temp_data_root: Path, temp_workspace_root: Path):
    source = temp_workspace_root / "gpre_price-history-05-05-2026.csv"
    _write_price_history_fixture(source)

    result = import_barchart_price_history_csv("GPRE", source, data_root=temp_data_root)
    history = pd.read_csv(result["normalized_csv"])

    assert source.exists()
    assert result["source"] == "barchart_price_history"
    assert history["source"].iloc[-1] == "barchart_price_history"
    assert history["close"].iloc[-1] == pytest.approx(18.07)
    assert int(history["volume"].iloc[-1]) == 1195900


def test_barchart_snapshot_is_discovered_and_loadable(temp_data_root: Path, temp_workspace_root: Path):
    source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    _write_options_screener_fixture(source)
    import_barchart_options_csv("GPRE", source, snapshot_date="2026-05-05", data_root=temp_data_root)

    snapshots = discover_chain_snapshots("GPRE", data_root=temp_data_root, dedupe=False)
    barchart_rows = [row for row in snapshots if row["storage_location"] == "barchart_options_screener"]
    dec_row = next(row for row in barchart_rows if row["expiry_date"] == "2026-12-18")
    chain = load_chain(dec_row["file_path"], prices_data_root=temp_data_root, research_data_root=temp_data_root)

    assert barchart_rows
    assert dec_row["snapshot_date"] == "2026-05-05"
    assert dec_row["quote_usable"] is True
    assert {"spread_pct_of_mid", "quality_flags", "model_eligible"} <= set(chain.contracts.columns)
    assert chain.contracts.loc[chain.contracts["strike"].eq(15.0), "quality_flags"].iloc[0]


def test_market_context_ignores_expired_chain_slices_for_current_snapshot(temp_data_root: Path, temp_workspace_root: Path):
    expired_dir = temp_data_root / "GPRE" / "option_chains"
    expired_dir.mkdir(parents=True, exist_ok=True)
    expired_path = expired_dir / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
    _write_options_screener_fixture(expired_path)

    source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    _write_options_screener_fixture(source)
    import_barchart_options_csv("GPRE", source, snapshot_date="2026-05-05", data_root=temp_data_root)

    context = resolve_market_context(
        ticker="GPRE",
        snapshot_date="2026-05-05",
        target_date="2026-12-18",
        data_root=temp_data_root,
    )

    expiries = pd.to_datetime(context.chain_source_summary["expiry_date"], errors="coerce")
    assert (expiries >= pd.Timestamp("2026-05-05")).all()
    assert "barchart_options_screener" in set(context.chain_source_summary["storage_location"])


def test_required_path_analysis_uses_barchart_source_fields(temp_data_root: Path, temp_workspace_root: Path):
    options_source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    price_source = temp_workspace_root / "gpre_price-history-05-05-2026.csv"
    _write_options_screener_fixture(options_source)
    _write_price_history_fixture(price_source)
    import_barchart_price_history_csv("GPRE", price_source, data_root=temp_data_root)
    import_barchart_options_csv("GPRE", options_source, snapshot_date="2026-05-05", data_root=temp_data_root)

    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-05-05",
        target_price=24.0,
        target_date="2026-12-18",
        data_root=temp_data_root,
        strategy_families=["long_stock", "long_call"],
    )

    assert "barchart_options_screener" in set(result.chain_source_summary["storage_location"])
    assert {
        "option_data_source",
        "entry_price_mode",
        "bid",
        "ask",
        "mid",
        "spread_pct_of_mid",
        "implied_volatility",
        "open_interest",
        "volume",
        "quality_flags",
    } <= set(result.required_path_core_summary.columns)
    assert result.required_path_core_summary["option_data_source"].eq("barchart_options_screener").any()


def test_required_path_analysis_preserves_barchart_iv_and_entry_anchor(temp_data_root: Path, temp_workspace_root: Path):
    options_source = temp_workspace_root / "options-screener-GPRE_2026-05-05.csv"
    price_source = temp_workspace_root / "gpre_price-history-05-05-2026.csv"
    _write_high_iv_options_fixture(options_source)
    _write_price_history_fixture(price_source)
    import_barchart_price_history_csv("GPRE", price_source, data_root=temp_data_root)
    import_barchart_options_csv("GPRE", options_source, snapshot_date="2026-05-05", data_root=temp_data_root)

    result = build_contract_selection_analysis(
        ticker="GPRE",
        snapshot_date="2026-05-05",
        target_price=24.0,
        target_date="2026-05-15",
        data_root=temp_data_root,
        strategy_families=["long_stock", "long_call"],
        entry_price_mode="mid",
    )

    summary = result.required_path_core_summary.loc[
        result.required_path_core_summary["contract_label"].astype(str).eq("18C May-26")
    ]
    assert not summary.empty
    assert summary["entry_iv"].iloc[0] == pytest.approx(1.0046)
    assert summary["implied_volatility"].iloc[0] == pytest.approx(1.0046)
    assert summary["entry_premium"].iloc[0] == pytest.approx(123.0)
    assert summary["entry_premium_per_contract"].iloc[0] == pytest.approx(123.0)
    assert summary["mid_per_share"].iloc[0] == pytest.approx(1.23)
    assert summary["contract_multiplier"].iloc[0] == 100

    iv_sensitivity = result.required_path_iv_sensitivity.loc[
        result.required_path_iv_sensitivity["contract_label"].astype(str).eq("18C May-26")
        & result.required_path_iv_sensitivity["threshold_multiple"].eq(1.5)
    ]
    assert set(pd.to_numeric(iv_sensitivity["iv_shift_vol_points"], errors="coerce").round(2)) >= {-0.50, 0.0, 0.50}
    assert iv_sensitivity.loc[iv_sensitivity["iv_shift_vol_points"].eq(-0.50), "base_iv"].iloc[0] == pytest.approx(1.0046)
    assert iv_sensitivity.loc[iv_sensitivity["iv_shift_vol_points"].eq(-0.50), "adjusted_iv"].iloc[0] == pytest.approx(0.5046)
    assert iv_sensitivity.loc[iv_sensitivity["iv_shift_vol_points"].eq(0.50), "adjusted_iv"].iloc[0] == pytest.approx(1.5046)

    day_zero = result.required_paths_by_option.loc[
        result.required_paths_by_option["contract_label"].astype(str).eq("18C May-26")
        & result.required_paths_by_option["days_from_snapshot"].eq(0)
    ]
    assert not day_zero.empty
    assert pd.to_numeric(day_zero["option_return_pct"], errors="coerce").eq(0.0).all()
    assert pd.to_numeric(day_zero["return_basis_option_value"], errors="coerce").eq(123.0).all()
    assert pd.to_numeric(day_zero["option_value"], errors="coerce").eq(123.0).all()
    assert "modeled_option_value" in day_zero.columns
    assert "Bid/Ask/Mid are per-share option quotes." in result.required_path_tables_html
    assert "Entry premium is per-contract in required-path calculations." in result.required_path_tables_html
