from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from options_lab.analysis import build_strategy_analysis, resolve_market_context, write_analysis_bundle
from options_lab.cli import main
from options_lab.ibkr.models import ConnectionSettings, UnderlyingQuoteSnapshot
from options_lab.ibkr.store import save_underlying_snapshot
from options_lab.io import load_chain
from options_lab.metadata import build_metadata
from options_lab.prices.price_store import save_price_history
from options_lab.rates.rate_store import save_merged_table
from options_lab.research_metadata import (
    build_ticker_catalog,
    get_options_overview,
    register_dividends_file,
    register_events_file,
    register_expected_move_file,
    register_notes_file,
    register_options_overview_file,
)
from options_lab.research_metadata.store import load_dataset_history
from options_lab.snapshots import list_snapshot_slices, snapshot_slices_for_date
from options_lab.strategies import build_strategy
from options_lab.utils import build_stock_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "GPRE"
    / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
)


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_chain_copy(data_root: Path, *, preferred: bool = True, stem_suffix: str = "") -> Path:
    ticker_root = data_root / "GPRE"
    ticker_root.mkdir(parents=True, exist_ok=True)
    filename = f"gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026{stem_suffix}.csv"
    target_dir = ticker_root / "option_chains" if preferred else ticker_root
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    target_path.write_text(SAMPLE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return target_path


def _write_ibkr_chain_copy(
    data_root: Path,
    *,
    snapshot_date: str = "2026-04-12",
    expiry_date: str = "2026-04-17",
    stem_suffix: str = "",
) -> Path:
    ticker_root = data_root / "GPRE" / "ibkr" / "chains" / "normalized"
    ticker_root.mkdir(parents=True, exist_ok=True)
    target_path = ticker_root / f"ibkr_gpre_chain_{snapshot_date}_{expiry_date}{stem_suffix}.csv"
    target_path.write_text(SAMPLE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    target_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "ticker": "GPRE",
                "snapshot_date": snapshot_date,
                "expiry_date": expiry_date,
                "source": "ibkr",
                "spot_price": 15.23,
                "snapshot_scope": "chain_universe",
                "storage_location": "ibkr_chain_universe",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target_path


def _write_ibkr_option_snapshot_copy(
    data_root: Path,
    *,
    snapshot_date: str = "2026-04-12",
    expiry_date: str = "2026-04-17",
    stem_suffix: str = "",
) -> Path:
    ticker_root = data_root / "GPRE" / "ibkr" / "snapshots" / "option_quotes" / "normalized"
    ticker_root.mkdir(parents=True, exist_ok=True)
    target_path = ticker_root / f"ibkr_gpre_option_snapshot_{snapshot_date}_{expiry_date}{stem_suffix}.csv"
    target_path.write_text(SAMPLE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    target_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "ticker": "GPRE",
                "snapshot_date": snapshot_date,
                "snapshot_time": "20:20:00",
                "expiry_date": expiry_date,
                "source": "ibkr",
                "spot_price": 15.23,
                "spot_price_source": "ibkr_delayed",
                "market_data_mode": "delayed",
                "snapshot_scope": "full_chain",
                "storage_location": "ibkr_full_quoted_snapshot",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target_path


def _write_sparse_ibkr_option_snapshot_copy(
    data_root: Path,
    *,
    snapshot_date: str = "2026-04-12",
    expiry_date: str = "2026-04-17",
    stem_suffix: str = "-sparse",
) -> Path:
    target_path = _write_ibkr_option_snapshot_copy(
        data_root,
        snapshot_date=snapshot_date,
        expiry_date=expiry_date,
        stem_suffix=stem_suffix,
    )
    frame = pd.read_csv(target_path, dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    for column in ["Bid", "Mid", "Ask", "Latest", "IV", "Delta", "Volume", "Open Int"]:
        if column in frame.columns:
            frame[column] = ""
    frame.to_csv(target_path, index=False)
    return target_path


def _register_demo_research_context(data_root: Path) -> None:
    expected_move_file = _write_json(
        data_root / "inputs" / "expected_move.json",
        [
            {
                "ticker": "GPRE",
                "snapshot_date": "2026-04-10",
                "expiry_date": "2026-04-17",
                "expected_move_abs": 1.35,
                "expected_move_pct": 0.0887,
                "lower_bound": 13.88,
                "upper_bound": 16.58,
                "implied_volatility": 0.54,
                "source": "synthetic_demo",
                "source_url": "local://demo/expected_move",
                "acquisition_method": "manual_demo",
                "notes": "Synthetic validation data",
                "registered_at": "2026-04-12T10:00:00Z",
            }
        ],
    )
    overview_file = data_root / "inputs" / "options_overview.csv"
    overview_file.parent.mkdir(parents=True, exist_ok=True)
    overview_file.write_text(
        "\n".join(
            [
                "ticker,snapshot_date,implied_volatility,historic_volatility,iv_rank,iv_percentile,iv_hv_ratio,put_call_volume_ratio,put_call_open_interest_ratio,total_call_volume,total_put_volume,total_call_open_interest,total_put_open_interest,earnings_date,source,source_url,acquisition_method,notes,registered_at",
                "GPRE,2026-04-10,0.54,0.41,78,74,1.32,0.89,1.12,12500,11100,98000,109800,2026-05-06,synthetic_demo,local://demo/options_overview,manual_demo,Synthetic validation data,2026-04-12T10:00:00Z",
            ]
        ),
        encoding="utf-8",
    )
    events_file = _write_json(
        data_root / "inputs" / "events.json",
        [
            {
                "ticker": "GPRE",
                "event_date": "2026-04-15",
                "event_time": "AMC",
                "event_type": "earnings",
                "source": "synthetic_demo",
                "source_url": "local://demo/events",
                "acquisition_method": "manual_demo",
                "notes": "Synthetic validation event",
                "registered_at": "2026-04-12T10:00:00Z",
            }
        ],
    )
    dividends_file = _write_json(
        data_root / "inputs" / "dividends.json",
        [
            {
                "ticker": "GPRE",
                "snapshot_date": "2026-04-10",
                "dividend_yield": 0.012,
                "expected_dividend_date": "2026-06-15",
                "source": "synthetic_demo",
                "source_url": "local://demo/dividends",
                "acquisition_method": "manual_demo",
                "notes": "Synthetic dividend assumption",
                "registered_at": "2026-04-12T10:00:00Z",
            }
        ],
    )
    notes_file = _write_json(
        data_root / "inputs" / "notes.json",
        [
            {
                "ticker": "GPRE",
                "note_date": "2026-04-11",
                "category": "thesis",
                "title": "Positioning update",
                "body": "Synthetic analyst note for validation.",
                "source": "synthetic_demo",
                "source_url": "local://demo/notes",
                "acquisition_method": "manual_demo",
                "notes": "Synthetic note",
                "registered_at": "2026-04-12T10:00:00Z",
            }
        ],
    )

    register_expected_move_file("GPRE", expected_move_file, data_root=data_root)
    register_options_overview_file("GPRE", overview_file, data_root=data_root)
    register_events_file("GPRE", events_file, data_root=data_root)
    register_dividends_file("GPRE", dividends_file, data_root=data_root)
    register_notes_file("GPRE", notes_file, data_root=data_root)


def _write_price_history(
    data_root: Path,
    *,
    rows: list[dict[str, object]],
) -> None:
    frame = pd.DataFrame(rows)
    save_price_history("GPRE", frame, data_root=data_root)


def _write_sample_rates_store(data_root: Path) -> None:
    merged = pd.DataFrame(
        [
            {
                "date": "2026-04-09",
                "dgs1mo_percent": 4.20,
                "dgs1mo_decimal": 0.0420,
                "dgs3mo_percent": 4.25,
                "dgs3mo_decimal": 0.0425,
                "dgs6mo_percent": 4.30,
                "dgs6mo_decimal": 0.0430,
                "dgs1_percent": 4.10,
                "dgs1_decimal": 0.0410,
                "downloaded_at": "2026-04-12T12:00:00Z",
            },
            {
                "date": "2026-04-10",
                "dgs1mo_percent": 4.22,
                "dgs1mo_decimal": 0.0422,
                "dgs3mo_percent": 4.27,
                "dgs3mo_decimal": 0.0427,
                "dgs6mo_percent": 4.32,
                "dgs6mo_decimal": 0.0432,
                "dgs1_percent": 4.12,
                "dgs1_decimal": 0.0412,
                "downloaded_at": "2026-04-12T12:00:00Z",
            },
        ]
    )
    save_merged_table(merged, data_root)


def test_register_expected_move_json_dedupes_and_updates_catalog(temp_data_root: Path):
    source_file = _write_json(
        temp_data_root / "inputs" / "expected_move.json",
        [
            {
                "ticker": "GPRE",
                "snapshot_date": "2026-04-10",
                "expiry_date": "2026-04-17",
                "expected_move_abs": 1.10,
                "source": "demo",
                "registered_at": "2026-04-12T09:00:00Z",
            },
            {
                "ticker": "GPRE",
                "snapshot_date": "2026-04-10",
                "expiry_date": "2026-04-17",
                "expected_move_abs": 1.25,
                "source": "demo",
                "registered_at": "2026-04-12T10:00:00Z",
            },
        ],
    )

    result = register_expected_move_file("GPRE", source_file, data_root=temp_data_root)

    history = load_dataset_history("expected_move", "GPRE", data_root=temp_data_root)
    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)
    assert len(history) == 1
    assert history.iloc[0]["expected_move_abs"] == 1.25
    assert result["row_count"] == 1
    assert catalog["datasets"]["expected_move"]["row_count"] == 1
    assert (temp_data_root / "GPRE" / "option_chains").exists()


def test_options_overview_lookup_uses_latest_available_snapshot(temp_data_root: Path):
    overview_file = temp_data_root / "inputs" / "overview.csv"
    overview_file.parent.mkdir(parents=True, exist_ok=True)
    overview_file.write_text(
        "\n".join(
            [
                "ticker,snapshot_date,iv_rank,iv_percentile,source,registered_at",
                "GPRE,2026-04-09,55,60,demo,2026-04-12T08:00:00Z",
                "GPRE,2026-04-10,70,75,demo,2026-04-12T09:00:00Z",
            ]
        ),
        encoding="utf-8",
    )

    register_options_overview_file("GPRE", overview_file, data_root=temp_data_root)
    payload = get_options_overview("GPRE", "2026-04-12", data_root=temp_data_root)

    assert payload["matched"] is True
    assert payload["matched_snapshot_date"] == "2026-04-10"
    assert payload["iv_rank"] == 70
    assert payload["iv_percentile"] == 75


def test_build_metadata_uses_local_research_context_and_dividend_store(temp_data_root: Path):
    chain_path = _write_chain_copy(temp_data_root)
    _register_demo_research_context(temp_data_root)

    metadata = build_metadata(
        chain_path,
        prices_data_root=temp_data_root,
        rates_data_root=temp_data_root / "missing_rates",
        research_data_root=temp_data_root,
    )

    assert metadata.research_context["expected_move"]["matched"] is True
    assert metadata.research_context["options_overview"]["iv_rank"] == 78
    assert metadata.research_context["nearest_event"]["event_type"] == "earnings"
    assert metadata.research_context["nearest_event"]["occurs_before_expiry"] is True
    assert metadata.research_context["dividend_assumption"]["dividend_yield"] == 0.012
    assert metadata.dividend_yield == 0.012
    assert len(metadata.research_context["notes"]) == 1


def test_build_metadata_respects_sidecar_and_override_research_context_precedence(temp_data_root: Path):
    chain_path = _write_chain_copy(temp_data_root)
    _register_demo_research_context(temp_data_root)
    sidecar_path = chain_path.with_suffix(".metadata.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "research_context": {
                    "expected_move": {
                        "matched": True,
                        "expected_move_abs": 8.88,
                        "source": "sidecar_override",
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    from_sidecar = build_metadata(chain_path, research_data_root=temp_data_root)
    from_override = build_metadata(
        chain_path,
        research_data_root=temp_data_root,
        metadata_override={
            "research_context": {
                "expected_move": {
                    "matched": True,
                    "expected_move_abs": 9.99,
                    "source": "explicit_override",
                }
            }
        },
    )

    assert from_sidecar.research_context["expected_move"]["expected_move_abs"] == 8.88
    assert from_sidecar.research_context["expected_move"]["source"] == "sidecar_override"
    assert from_override.research_context["expected_move"]["expected_move_abs"] == 9.99
    assert from_override.research_context["expected_move"]["source"] == "explicit_override"


def test_catalog_discovers_preferred_and_legacy_chain_locations(temp_data_root: Path):
    _write_chain_copy(temp_data_root, preferred=True)
    _write_chain_copy(temp_data_root, preferred=False, stem_suffix="-legacy")

    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)
    locations = {item["storage_location"] for item in catalog["chain_snapshots"]}

    assert "preferred_option_chains" in locations
    assert "legacy_ticker_root" in locations


def test_catalog_prefers_preferred_chain_when_same_file_exists_in_legacy_root(temp_data_root: Path):
    preferred = _write_chain_copy(temp_data_root, preferred=True)
    _write_chain_copy(temp_data_root, preferred=False)

    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)

    assert catalog["chain_snapshot_count"] == 1
    assert catalog["chain_snapshots"][0]["storage_location"] == "preferred_option_chains"
    assert catalog["chain_snapshots"][0]["file_path"].endswith(str(preferred.relative_to(temp_data_root)).replace("\\", "\\"))


def test_catalog_discovers_ibkr_full_quoted_snapshot_location(temp_data_root: Path):
    ibkr_path = _write_ibkr_option_snapshot_copy(temp_data_root)

    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)

    assert catalog["chain_snapshot_count"] == 1
    assert catalog["chain_snapshots"][0]["storage_location"] == "ibkr_full_quoted_snapshot"
    assert catalog["chain_snapshots"][0]["quote_usable"] is True
    assert catalog["chain_snapshots"][0]["usable_quote_coverage_pct"] >= 20.0
    assert catalog["chain_snapshots"][0]["file_path"].endswith(str(ibkr_path.relative_to(temp_data_root)))


def test_catalog_prefers_quote_usable_full_quoted_ibkr_over_preferred_chain_for_same_snapshot_expiry(temp_data_root: Path):
    _write_chain_copy(temp_data_root, preferred=True)
    ibkr = _write_ibkr_option_snapshot_copy(temp_data_root)

    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)
    slices = list_snapshot_slices("GPRE", data_root=temp_data_root)
    same_day = snapshot_slices_for_date("GPRE", "2026-04-12", data_root=temp_data_root)

    assert catalog["chain_snapshot_count"] == 1
    assert catalog["chain_snapshots"][0]["storage_location"] == "ibkr_full_quoted_snapshot"
    assert catalog["chain_snapshots"][0]["file_path"].endswith(str(ibkr.relative_to(temp_data_root)))
    assert catalog["chain_snapshots"][0]["quote_usable"] is True
    assert slices.iloc[0]["storage_location"] == "ibkr_full_quoted_snapshot"
    assert same_day.iloc[0]["storage_location"] == "ibkr_full_quoted_snapshot"


def test_catalog_falls_back_to_preferred_chain_when_same_day_ibkr_slice_is_sparse(temp_data_root: Path):
    preferred = _write_chain_copy(temp_data_root, preferred=True)
    sparse_ibkr = _write_sparse_ibkr_option_snapshot_copy(temp_data_root)

    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)
    slices = list_snapshot_slices("GPRE", data_root=temp_data_root)
    same_day = snapshot_slices_for_date("GPRE", "2026-04-12", data_root=temp_data_root)

    assert catalog["chain_snapshot_count"] == 1
    assert catalog["chain_snapshots"][0]["storage_location"] == "preferred_option_chains"
    assert catalog["chain_snapshots"][0]["file_path"].endswith(str(preferred.relative_to(temp_data_root)))
    assert slices.iloc[0]["storage_location"] == "preferred_option_chains"
    assert same_day.iloc[0]["storage_location"] == "preferred_option_chains"
def test_catalog_does_not_prefer_chain_universe_metadata_over_full_quoted_slice(temp_data_root: Path):
    _write_ibkr_chain_copy(temp_data_root)
    quoted = _write_ibkr_option_snapshot_copy(temp_data_root)

    catalog = build_ticker_catalog("GPRE", data_root=temp_data_root)

    assert catalog["chain_snapshot_count"] == 1
    assert catalog["chain_snapshots"][0]["storage_location"] == "ibkr_full_quoted_snapshot"
    assert catalog["chain_snapshots"][0]["file_path"].endswith(str(quoted.relative_to(temp_data_root)))


def test_resolve_market_context_records_chain_spot_risk_free_and_metadata_provenance(temp_data_root: Path):
    _write_chain_copy(temp_data_root, preferred=True)
    _write_chain_copy(
        temp_data_root,
        preferred=True,
        stem_suffix="-next-expiry",
    )
    next_expiry_target = temp_data_root / "GPRE" / "option_chains" / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026-next-expiry.csv"
    next_expiry_target.rename(
        temp_data_root / "GPRE" / "option_chains" / "gpre-options-exp-2026-05-15-monthly-show-all-stacked-04-13-2026.csv"
    )
    _write_sparse_ibkr_option_snapshot_copy(temp_data_root)
    _register_demo_research_context(temp_data_root)
    save_underlying_snapshot(
        UnderlyingQuoteSnapshot(
            ticker="GPRE",
            snapshot_timestamp=_utc("2026-04-12T19:59:00Z"),
            market_data_mode="delayed",
            market_data_type_code=3,
            bid=15.2,
            ask=15.4,
            last=15.3,
            close=15.25,
            mid=15.3,
            mark=15.3,
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            source="ibkr",
            warnings=[],
            missing_fields=[],
            connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
        ),
        data_root=temp_data_root,
    )

    resolved = resolve_market_context(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_date="2026-07-15",
        data_root=temp_data_root,
    )

    assert resolved.ticker == "GPRE"
    assert resolved.chain_source_summary is not None
    assert not resolved.chain_source_summary.empty
    assert not resolved.market_context_summary.empty
    assert {"source_snapshot_file", "storage_location", "quote_usable", "fallback_level"} <= set(
        resolved.chain_source_summary.columns
    )
    assert {"spot_price_source", "risk_free_rate_source", "expected_move_matched", "nearest_event_type"} <= set(
        resolved.market_context_summary.columns
    )
    assert resolved.spot_price == 15.3
    assert resolved.spot_price_source == "ibkr_delayed"
    assert resolved.spot_price_matched_date.isoformat() == "2026-04-12"
    assert resolved.risk_free_rate_source == "default_fallback"
    assert resolved.research_context["expected_move"]["matched"] is True
    assert resolved.research_context["options_overview"]["iv_rank"] == 78
    assert resolved.research_context["nearest_event"]["event_type"] == "earnings"
    assert any(
        item["storage_location"] == "preferred_option_chains"
        and item["fallback_level"] == "same_day_fallback_from_sparse_ibkr"
        for item in resolved.chain_source_summary.to_dict(orient="records")
    )


def test_resolve_market_context_prefers_same_day_ibkr_spot_and_records_field_priority_and_trust_labels(temp_data_root: Path):
    _write_chain_copy(temp_data_root, preferred=True)
    _register_demo_research_context(temp_data_root)
    _write_sample_rates_store(temp_data_root)
    save_underlying_snapshot(
        UnderlyingQuoteSnapshot(
            ticker="GPRE",
            snapshot_timestamp=_utc("2026-04-12T19:59:00Z"),
            market_data_mode="delayed",
            market_data_type_code=3,
            bid=15.2,
            ask=15.4,
            last=None,
            close=15.25,
            mid=15.31,
            mark=15.30,
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            source="ibkr",
            warnings=["Last trade was unavailable in delayed mode."],
            missing_fields=["last"],
            connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
        ),
        data_root=temp_data_root,
    )

    resolved = resolve_market_context(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_date="2026-04-17",
        data_root=temp_data_root,
    )

    assert resolved.spot_price == 15.31
    assert resolved.spot_price_source == "ibkr_delayed"
    assert resolved.spot_price_matched_date.isoformat() == "2026-04-12"
    assert resolved.spot_price_field_used == "mid"
    assert resolved.spot_price_used_prior_date is False
    assert resolved.ibkr_same_day_spot_attempted is True
    assert resolved.ibkr_same_day_spot_rejected_reason is None
    assert "same-day delayed ibkr spot" in (resolved.spot_price_note or "").lower()
    assert "mid" in (resolved.spot_price_note or "").lower()
    assert resolved.risk_free_rate_source == "fred_local_store"
    assert resolved.risk_free_rate_series == "DGS1MO"
    assert resolved.risk_free_rate_matched_date.isoformat() == "2026-04-10"
    assert {"source_quality", "source_quality_note", "source_trust_label"} <= set(resolved.chain_source_summary.columns)
    assert {"spot_field_used", "spot_used_prior_date", "spot_quality_note", "ibkr_same_day_spot_attempted"} <= set(
        resolved.market_context_summary.columns
    )
    assert set(resolved.chain_source_summary["source_quality"]) == {"same_day_quoted"}


def test_resolve_market_context_falls_back_to_local_price_store_when_same_day_ibkr_spot_is_unusable(temp_data_root: Path):
    _write_chain_copy(temp_data_root, preferred=True)
    _register_demo_research_context(temp_data_root)
    _write_sample_rates_store(temp_data_root)
    _write_price_history(
        temp_data_root,
        rows=[
            {
                "ticker": "GPRE",
                "date": "2026-04-12",
                "open": 15.0,
                "high": 15.2,
                "low": 14.8,
                "close": 15.05,
                "volume": 1500000,
                "adj_close": None,
                "source": "nasdaq_historical_quotes",
                "downloaded_at": "2026-04-12T21:00:00Z",
            }
        ],
    )
    save_underlying_snapshot(
        UnderlyingQuoteSnapshot(
            ticker="GPRE",
            snapshot_timestamp=_utc("2026-04-12T19:59:00Z"),
            market_data_mode="delayed",
            market_data_type_code=3,
            bid=None,
            ask=None,
            last=None,
            close=None,
            mid=None,
            mark=None,
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            source="ibkr",
            warnings=["Delayed same-day quote returned no usable spot fields."],
            missing_fields=["bid", "ask", "last", "mid", "mark", "close"],
            connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
        ),
        data_root=temp_data_root,
    )

    resolved = resolve_market_context(
        ticker="GPRE",
        snapshot_date="2026-04-12",
        target_date="2026-04-17",
        data_root=temp_data_root,
    )

    assert resolved.spot_price == 15.05
    assert resolved.spot_price_source == "nasdaq_historical_quotes"
    assert resolved.spot_price_matched_date.isoformat() == "2026-04-12"
    assert resolved.spot_price_field_used == "close"
    assert resolved.spot_price_used_prior_date is False
    assert resolved.ibkr_same_day_spot_attempted is True
    assert "usable price field" in (resolved.ibkr_same_day_spot_rejected_reason or "").lower()
    assert "historical price" in (resolved.spot_price_note or "").lower()


def test_report_and_cli_integration_surface_research_context(capsys, temp_data_root: Path):
    chain_path = _write_chain_copy(temp_data_root)
    _register_demo_research_context(temp_data_root)
    output_root = temp_data_root / "analysis_outputs"

    inspect_exit = main(
        [
            "inspect-metadata",
            "--ticker",
            "GPRE",
            "--snapshot-date",
            "2026-04-12",
            "--expiry-date",
            "2026-04-17",
            "--data-root",
            str(temp_data_root),
        ]
    )
    inspect_output = json.loads(capsys.readouterr().out)
    assert inspect_exit == 0
    assert inspect_output["research_context"]["nearest_event"]["event_type"] == "earnings"

    chain = load_chain(
        chain_path,
        prices_data_root=temp_data_root,
        rates_data_root=temp_data_root / "missing_rates",
        research_data_root=temp_data_root,
    )
    strategy = build_strategy("long_call", chain)
    analysis = build_strategy_analysis(
        strategy,
        spot_grid=build_stock_grid(chain.spot_price or strategy.entry_spot, points=21),
    )
    bundle = write_analysis_bundle(analysis, analysis_kind="strategy", output_root=output_root)

    report_metadata = json.loads((bundle.bundle_dir / "metadata" / "report_metadata.json").read_text(encoding="utf-8"))
    summary_md = (bundle.bundle_dir / "summary" / "summary.md").read_text(encoding="utf-8")
    assert report_metadata["research_context"]["expected_move"]["matched"] is True
    assert report_metadata["research_context"]["dividend_assumption"]["dividend_yield"] == 0.012
    assert "## Research Context" in summary_md
