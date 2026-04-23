from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from options_lab.io import load_chain
from options_lab.metadata import build_metadata
from options_lab.research_metadata.catalog import discover_chain_snapshots


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ibkr_underlying_store_persists_snapshot_manifest_and_source_notes(temp_data_root: Path):
    from options_lab.ibkr.models import ConnectionSettings, UnderlyingQuoteSnapshot
    from options_lab.ibkr.store import save_underlying_snapshot

    snapshot = UnderlyingQuoteSnapshot(
        ticker="GPRE",
        snapshot_timestamp=_utc("2026-04-12T20:15:00Z"),
        market_data_mode="delayed",
        market_data_type_code=3,
        bid=15.1,
        ask=15.3,
        last=None,
        close=15.0,
        mid=15.2,
        mark=15.2,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        source="ibkr",
        warnings=["Last trade was unavailable in delayed mode."],
        missing_fields=["last"],
        connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
    )

    result = save_underlying_snapshot(snapshot, data_root=temp_data_root)

    normalized_csv = Path(result["normalized_csv"])
    manifest_path = Path(result["manifest_path"])
    source_notes_path = Path(result["source_notes_path"])
    raw_path = Path(result["raw_json"])

    assert normalized_csv.exists()
    assert manifest_path.exists()
    assert source_notes_path.exists()
    assert raw_path.exists()

    frame = pd.read_csv(normalized_csv)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    notes = json.loads(source_notes_path.read_text(encoding="utf-8"))

    assert frame.iloc[0]["ticker"] == "GPRE"
    assert frame.iloc[0]["market_data_mode"] == "delayed"
    assert frame.iloc[0]["source"] == "ibkr"
    assert manifest["market_data_mode"] == "delayed"
    assert manifest["request"]["connection"]["port"] == 7497
    assert notes["policy"]["delayed_only"] is True
    assert "last" in manifest["field_availability"]["missing_fields"]


def test_ibkr_market_data_type_validation_accepts_only_delayed_codes():
    from options_lab.ibkr.models import validate_effective_market_data_type

    assert validate_effective_market_data_type(3) == (True, "accepted")
    assert validate_effective_market_data_type(4) == (True, "accepted")
    assert validate_effective_market_data_type(1)[0] is False
    assert validate_effective_market_data_type(2)[0] is False


def test_ibkr_failure_manifest_records_honest_delayed_error(temp_data_root: Path):
    from options_lab.ibkr.models import ConnectionSettings
    from options_lab.ibkr.store import record_request_failure

    result = record_request_failure(
        "GPRE",
        request_type="underlying",
        market_data_mode="delayed",
        connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
        error_message="Couldn't connect to TWS.",
        warnings=["No delayed session was available."],
        data_root=temp_data_root,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    notes = json.loads(Path(result["source_notes_path"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "failed"
    assert manifest["error"] == "Couldn't connect to TWS."
    assert manifest["market_data_mode"] == "delayed"
    assert notes["last_run"]["status"] == "failed"


def test_ibkr_option_snapshot_saves_chain_compatible_slice_and_sidecar(temp_data_root: Path):
    from options_lab.ibkr.models import (
        ConnectionSettings,
        ContractMatchDiagnostics,
        OptionQuoteSnapshot,
        OptionSnapshotDiagnostics,
        OptionSnapshotFetchResult,
    )
    from options_lab.ibkr.store import save_option_snapshot

    fetched_at = _utc("2026-04-12T20:20:00Z")
    quotes = [
        OptionQuoteSnapshot(
            ticker="GPRE",
            snapshot_timestamp=fetched_at,
            market_data_mode="delayed",
            market_data_type_code=3,
            expiry_date="2026-04-17",
            strike=15.0,
            option_type="call",
            conid=1001,
            local_symbol="GPRE  260417C00015000",
            trading_class="GPRE",
            exchange="SMART",
            bid=0.8,
            ask=1.0,
            last=0.9,
            mid=0.9,
            mark=0.9,
            close=0.75,
            volume=100.0,
            open_interest=250.0,
            implied_volatility=0.42,
            historical_volatility=0.35,
            delta=0.51,
            gamma=0.1,
            theta=-0.03,
            vega=0.08,
            option_price=0.9,
            pv_dividend=0.0,
            under_price=15.2,
            source="ibkr",
            warnings=[],
            missing_fields=[],
            connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
        ),
        OptionQuoteSnapshot(
            ticker="GPRE",
            snapshot_timestamp=fetched_at,
            market_data_mode="delayed",
            market_data_type_code=3,
            expiry_date="2026-04-17",
            strike=15.0,
            option_type="put",
            conid=1002,
            local_symbol="GPRE  260417P00015000",
            trading_class="GPRE",
            exchange="SMART",
            bid=0.7,
            ask=0.95,
            last=0.8,
            mid=0.825,
            mark=0.825,
            close=0.72,
            volume=80.0,
            open_interest=200.0,
            implied_volatility=0.44,
            historical_volatility=0.35,
            delta=-0.49,
            gamma=0.09,
            theta=-0.02,
            vega=0.08,
            option_price=0.825,
            pv_dividend=0.0,
            under_price=15.2,
            source="ibkr",
            warnings=["Open interest was delayed and may lag."],
            missing_fields=[],
            connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
        ),
    ]

    result = save_option_snapshot(
        OptionSnapshotFetchResult(
            quotes=quotes,
            diagnostics=OptionSnapshotDiagnostics(
                contract_match=ContractMatchDiagnostics(
                    requested_expiries=["2026-04-17"],
                    requested_right="both",
                    row_counts={"final_selected_contracts": 2},
                ),
                selected_contract_count=2,
                final_selected_expiries=["2026-04-17"],
                final_selected_strikes=[15.0],
                final_selected_exchanges=["SMART"],
                final_selected_trading_classes=["GPRE"],
                delayed_field_summary={
                    "missing_fields": [],
                    "warnings": ["Open interest was delayed and may lag."],
                    "missing_field_counts": {},
                },
                snapshot_scope="full_chain",
                discovered_expiries=["2026-04-17"],
                strike_count_by_expiry={"2026-04-17": 1},
                attempted_contract_count=2,
            ),
        ),
        data_root=temp_data_root,
    )
    slice_path = Path(result["chain_slice_files"][0])
    sidecar_path = slice_path.with_suffix(".metadata.json")
    manifest_path = Path(result["manifest_path"])

    assert slice_path.exists()
    assert sidecar_path.exists()
    assert manifest_path.exists()

    chain = load_chain(
        slice_path,
        prices_data_root=temp_data_root,
        rates_data_root=temp_data_root,
        research_data_root=temp_data_root,
    )

    assert chain.metadata.ticker == "GPRE"
    assert chain.metadata.expiry_date.isoformat() == "2026-04-17"
    assert chain.metadata.snapshot_date.isoformat() == "2026-04-12"
    assert chain.metadata.spot_price == 15.2
    assert set(chain.contracts["option_type"]) == {"call", "put"}
    assert {"mid", "bid", "ask", "last", "open_interest", "iv", "delta"}.issubset(chain.contracts.columns)

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert sidecar["snapshot_scope"] == "full_chain"
    assert sidecar["storage_location"] == "ibkr_full_quoted_snapshot"
    assert manifest["snapshot_scope"] == "full_chain"
    assert manifest["attempted_contract_count"] == 2
    assert manifest["persisted_quote_count"] == 2
    assert manifest["discovered_expiries"] == ["2026-04-17"]
    assert manifest["strike_count_by_expiry"] == {"2026-04-17": 1}
    assert manifest["coverage_summary"]["contract_count"] == 2
    assert manifest["coverage_summary"]["overall"]["bid"]["available_count"] == 2
    assert manifest["coverage_summary"]["overall"]["ask"]["coverage_pct"] == 100.0
    assert manifest["coverage_summary"]["overall"]["implied_volatility"]["coverage_pct"] == 100.0
    assert manifest["coverage_summary"]["overall"]["volume"]["coverage_pct"] == 100.0
    assert manifest["coverage_summary"]["overall"]["open_interest"]["coverage_pct"] == 100.0


def test_ibkr_full_chain_success_manifest_records_orchestrated_run(temp_data_root: Path):
    from options_lab.ibkr.models import (
        ConnectionSettings,
        ContractMatchDiagnostics,
        OptionQuoteSnapshot,
        OptionSnapshotDiagnostics,
        OptionSnapshotFetchResult,
    )
    from options_lab.ibkr.store import save_full_chain_snapshot_run

    fetched_at = _utc("2026-04-12T20:20:00Z")
    settings = ConnectionSettings(host="127.0.0.1", port=7497, client_id=71)
    option_snapshot = OptionSnapshotFetchResult(
        quotes=[
            OptionQuoteSnapshot(
                ticker="GPRE",
                snapshot_timestamp=fetched_at,
                market_data_mode="delayed",
                market_data_type_code=3,
                expiry_date="2026-04-17",
                strike=15.0,
                option_type="call",
                conid=1001,
                local_symbol="GPRE  260417C00015000",
                trading_class="GPRE",
                exchange="SMART",
                bid=None,
                ask=None,
                last=None,
                mid=0.9,
                mark=0.9,
                close=0.75,
                volume=100.0,
                open_interest=None,
                implied_volatility=None,
                historical_volatility=0.35,
                delta=0.51,
                gamma=0.1,
                theta=-0.03,
                vega=0.08,
                option_price=0.9,
                pv_dividend=0.0,
                under_price=15.2,
                source="ibkr",
                warnings=["Requested market data is not subscribed. Displaying delayed market data."],
                missing_fields=["bid", "ask", "last", "open_interest", "implied_volatility"],
                connection=settings,
            )
        ],
        diagnostics=OptionSnapshotDiagnostics(
            contract_match=ContractMatchDiagnostics(
                requested_expiries=["2026-04-17"],
                requested_right="both",
                row_counts={"final_selected_contracts": 1},
            ),
            snapshot_scope="full_chain",
            discovered_expiries=["2026-04-17"],
            strike_count_by_expiry={"2026-04-17": 1},
            attempted_contract_count=1,
            selected_contract_count=1,
            final_selected_expiries=["2026-04-17"],
            final_selected_strikes=[15.0],
            final_selected_exchanges=["SMART"],
            final_selected_trading_classes=["GPRE"],
            delayed_field_summary={
                "missing_fields": ["bid", "ask", "last", "open_interest", "implied_volatility"],
                "missing_field_counts": {
                    "bid": 1,
                    "ask": 1,
                    "last": 1,
                    "open_interest": 1,
                    "implied_volatility": 1,
                },
                "warnings": ["Requested market data is not subscribed. Displaying delayed market data."],
            },
        ),
    )

    result = save_full_chain_snapshot_run(
        "GPRE",
        market_data_mode="delayed",
        connection=settings,
        underlying_files={"normalized_csv": "data/GPRE/ibkr/snapshots/underlying/normalized/test.csv"},
        chain_files={"normalized_csv": "data/GPRE/ibkr/chains/normalized/test.csv"},
        option_snapshot_files={"normalized_csv": "data/GPRE/ibkr/snapshots/option_quotes/normalized/test.csv"},
        option_snapshot=option_snapshot,
        data_root=temp_data_root,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    notes = json.loads(Path(result["source_notes_path"]).read_text(encoding="utf-8"))

    assert manifest["status"] == "succeeded"
    assert manifest["request_type"] == "full_chain_snapshot"
    assert manifest["snapshot_scope"] == "full_chain"
    assert manifest["attempted_contract_count"] == 1
    assert manifest["persisted_quote_count"] == 1
    assert manifest["field_availability"]["missing_field_counts"]["bid"] == 1
    assert manifest["coverage_summary"]["contract_count"] == 1
    assert manifest["coverage_summary"]["overall"]["bid"]["coverage_pct"] == 0.0
    assert manifest["coverage_summary"]["overall"]["ask"]["coverage_pct"] == 0.0
    assert manifest["coverage_summary"]["overall"]["implied_volatility"]["coverage_pct"] == 0.0
    assert manifest["coverage_summary"]["overall"]["volume"]["coverage_pct"] == 100.0
    assert manifest["coverage_summary"]["overall"]["open_interest"]["coverage_pct"] == 0.0
    assert manifest["files"]["option_snapshot"]["normalized_csv"].endswith("test.csv")
    assert notes["last_run"]["type"] == "full_chain_snapshot"
    assert notes["last_run"]["status"] == "succeeded"


def test_fetch_full_chain_snapshot_orchestrates_underlying_chain_per_expiry_batches_with_retry_controls(monkeypatch):
    from options_lab.ibkr.market_data import fetch_full_chain_snapshot
    from options_lab.ibkr.models import (
        ChainDiscoveryDiagnostics,
        ChainFetchResult,
        ChainRow,
        ConnectionSettings,
        OptionQuoteSnapshot,
        OptionSnapshotFetchResult,
        ResolvedUnderlyingContract,
        UnderlyingQuoteSnapshot,
    )

    settings = ConnectionSettings()
    fetched_at = _utc("2026-04-12T20:20:00Z")
    underlying = UnderlyingQuoteSnapshot(
        ticker="GPRE",
        snapshot_timestamp=_utc("2026-04-12T20:15:00Z"),
        market_data_mode="delayed",
        market_data_type_code=3,
        bid=15.1,
        ask=15.3,
        last=15.2,
        close=15.0,
        mid=15.2,
        mark=15.2,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        source="ibkr",
        warnings=[],
        missing_fields=[],
        connection=settings,
        resolved_underlying=ResolvedUnderlyingContract(
            conid=101,
            symbol="GPRE",
            sec_type="STK",
            currency="USD",
            exchange="SMART",
            primary_exchange="NASDAQ",
            local_symbol="GPRE",
            trading_class="GPRE",
            multiplier="",
        ),
    )
    rows = [
        ChainRow(
            ticker="GPRE",
            underlying_conid=101,
            fetched_at=fetched_at,
            market_data_mode="delayed",
            exchange="SMART",
            trading_class="GPRE",
            multiplier="100",
            expiry_date="2026-04-17",
            strike=15.0,
            option_type="call",
            currency="USD",
            source="ibkr",
            connection=settings,
        ),
        ChainRow(
            ticker="GPRE",
            underlying_conid=101,
            fetched_at=fetched_at,
            market_data_mode="delayed",
            exchange="SMART",
            trading_class="GPRE",
            multiplier="100",
            expiry_date="2026-04-17",
            strike=15.0,
            option_type="put",
            currency="USD",
            source="ibkr",
            connection=settings,
        ),
        ChainRow(
            ticker="GPRE",
            underlying_conid=101,
            fetched_at=fetched_at,
            market_data_mode="delayed",
            exchange="SMART",
            trading_class="GPRE",
            multiplier="100",
            expiry_date="2026-05-15",
            strike=16.0,
            option_type="call",
            currency="USD",
            source="ibkr",
            connection=settings,
        ),
    ]
    chain_result = ChainFetchResult(
        rows=rows,
        diagnostics=ChainDiscoveryDiagnostics(
            requested={"ticker": "GPRE"},
            resolved_underlying=underlying.resolved_underlying,
            available_expiries=["2026-04-17", "2026-05-15"],
            available_strike_count=2,
            available_strike_sample=[15.0, 16.0],
            row_counts={"final_chain_rows": 3},
        ),
    )
    quote_calls: dict[str, object] = {}
    expiry_batches: list[dict[str, object]] = []

    def _fake_underlying(*args, **kwargs):
        quote_calls["underlying"] = kwargs["market_data_mode"]
        return underlying

    def _fake_chain(*args, **kwargs):
        quote_calls["chain"] = kwargs["market_data_mode"]
        return chain_result

    def _fake_collect(*args, **kwargs):
        expiry_batches.append(
            {
                "expiries": sorted({row.expiry_date for row in kwargs["chain_rows"]}),
                "rows_passed": len(kwargs["chain_rows"]),
                "market_data_mode": kwargs["market_data_mode"],
                "snapshot_wait_seconds": kwargs.get("snapshot_wait_seconds"),
                "retry_sparse_quotes_once": kwargs.get("retry_sparse_quotes_once"),
                "sparse_retry_wait_seconds": kwargs.get("sparse_retry_wait_seconds"),
            }
        )
        return OptionSnapshotFetchResult(
            quotes=[
                OptionQuoteSnapshot(
                    ticker="GPRE",
                    snapshot_timestamp=fetched_at,
                    market_data_mode="delayed",
                    market_data_type_code=3,
                    expiry_date=row.expiry_date,
                    strike=row.strike,
                    option_type=row.option_type,
                    conid=1000 + index,
                    local_symbol=f"GPRE-{index}",
                    trading_class=row.trading_class,
                    exchange=row.exchange,
                    bid=0.5,
                    ask=0.7,
                    last=0.6,
                    mid=0.6,
                    mark=0.6,
                    close=0.55,
                    volume=None,
                    open_interest=None,
                    implied_volatility=None,
                    historical_volatility=None,
                    delta=None,
                    gamma=None,
                    theta=None,
                    vega=None,
                    option_price=0.6,
                    pv_dividend=0.0,
                    under_price=15.2,
                    source="ibkr",
                    warnings=["Delayed quote fields were sparse."],
                    missing_fields=["open_interest", "implied_volatility"],
                    connection=settings,
                )
                for index, row in enumerate(kwargs["chain_rows"], start=1)
            ],
            diagnostics=kwargs["diagnostics"],
        )

    monkeypatch.setattr("options_lab.ibkr.market_data.fetch_underlying_snapshot", _fake_underlying)
    monkeypatch.setattr("options_lab.ibkr.market_data.fetch_option_chain", _fake_chain)
    monkeypatch.setattr("options_lab.ibkr.market_data._collect_option_quotes_from_chain_rows", _fake_collect)

    result = fetch_full_chain_snapshot(
        "GPRE",
        settings=settings,
        market_data_mode="delayed",
        exchange="SMART",
        currency="USD",
        per_expiry_timeout=90.0,
        retry_sparse_quotes_once=True,
        sparse_retry_wait_seconds=3.5,
    )

    assert quote_calls["underlying"] == "delayed"
    assert quote_calls["chain"] == "delayed"
    assert len(expiry_batches) == 2
    assert expiry_batches[0]["expiries"] == ["2026-04-17"]
    assert expiry_batches[0]["rows_passed"] == 2
    assert expiry_batches[1]["expiries"] == ["2026-05-15"]
    assert expiry_batches[1]["rows_passed"] == 1
    assert all(batch["market_data_mode"] == "delayed" for batch in expiry_batches)
    assert all(batch["snapshot_wait_seconds"] == 90.0 for batch in expiry_batches)
    assert all(batch["retry_sparse_quotes_once"] is True for batch in expiry_batches)
    assert all(batch["sparse_retry_wait_seconds"] == 3.5 for batch in expiry_batches)
    assert len(result.option_snapshot.quotes) == 3
    assert result.option_snapshot.diagnostics.snapshot_scope == "full_chain"
    assert result.option_snapshot.diagnostics.discovered_expiries == ["2026-04-17", "2026-05-15"]
    assert result.option_snapshot.diagnostics.strike_count_by_expiry == {"2026-04-17": 1, "2026-05-15": 1}
    assert result.option_snapshot.diagnostics.attempted_contract_count == 3


def test_ibkr_snapshot_discovery_and_metadata_spot_resolution_work_locally(temp_data_root: Path):
    from options_lab.ibkr.models import ConnectionSettings, OptionQuoteSnapshot, UnderlyingQuoteSnapshot
    from options_lab.ibkr.store import save_option_snapshot, save_underlying_snapshot

    save_underlying_snapshot(
            UnderlyingQuoteSnapshot(
                ticker="GPRE",
                snapshot_timestamp=_utc("2026-04-11T20:00:00Z"),
                market_data_mode="delayed",
                market_data_type_code=3,
                bid=15.0,
                ask=15.2,
                last=15.1,
                close=15.05,
                mid=15.1,
                mark=15.1,
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
    save_option_snapshot(
            [
                OptionQuoteSnapshot(
                    ticker="GPRE",
                    snapshot_timestamp=_utc("2026-04-12T20:20:00Z"),
                    market_data_mode="delayed",
                    market_data_type_code=3,
                    expiry_date="2026-04-17",
                    strike=15.0,
                    option_type="call",
                    conid=1001,
                    local_symbol="GPRE  260417C00015000",
                    trading_class="GPRE",
                    exchange="SMART",
                    bid=0.8,
                    ask=1.0,
                    last=0.9,
                    mid=0.9,
                    mark=0.9,
                    close=0.75,
                    volume=100.0,
                    open_interest=250.0,
                    implied_volatility=0.42,
                    historical_volatility=0.35,
                    delta=0.51,
                    gamma=0.1,
                    theta=-0.03,
                    vega=0.08,
                    option_price=0.9,
                    pv_dividend=0.0,
                    under_price=15.3,
                    source="ibkr",
                    warnings=[],
                    missing_fields=[],
                    connection=ConnectionSettings(host="127.0.0.1", port=7497, client_id=71),
                )
            ],
            data_root=temp_data_root,
        )

    snapshots = discover_chain_snapshots("GPRE", data_root=temp_data_root)
    metadata = build_metadata(
        Path(snapshots[0]["file_path"]),
        prices_data_root=temp_data_root / "missing_prices",
        rates_data_root=temp_data_root / "missing_rates",
        research_data_root=temp_data_root,
    )

    assert any(item["storage_location"] == "ibkr_chain_snapshot" for item in snapshots)
    assert metadata.spot_price == 15.3
    assert metadata.spot_price_source == "ibkr_delayed"
    assert metadata.spot_price_matched_date.isoformat() == "2026-04-12"


def test_ibkr_get_underlying_spot_supports_same_day_only_and_surfaces_field_used(temp_data_root: Path):
    from options_lab.ibkr.models import ConnectionSettings, UnderlyingQuoteSnapshot
    from options_lab.ibkr.store import get_underlying_spot, save_underlying_snapshot

    save_underlying_snapshot(
        UnderlyingQuoteSnapshot(
            ticker="GPRE",
            snapshot_timestamp=_utc("2026-04-11T20:00:00Z"),
            market_data_mode="delayed",
            market_data_type_code=3,
            bid=15.0,
            ask=15.2,
            last=15.1,
            close=15.05,
            mid=15.1,
            mark=15.1,
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

    match = get_underlying_spot("GPRE", "2026-04-12", data_root=temp_data_root, require_same_day=True)

    assert match.matched_date.isoformat() == "2026-04-12"
    assert match.close_price == 15.31
    assert match.field_used == "mid"
    assert match.used_prior_date is False


def test_fetch_option_chain_reports_exchange_filter_diagnostics(monkeypatch):
    from options_lab.ibkr.chains import fetch_option_chain
    from options_lab.ibkr.models import ConnectionSettings

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request_contract_details(self, contract, *, timeout=None):
            return [
                SimpleNamespace(
                    contract=SimpleNamespace(
                        conId=101,
                        symbol="AAPL",
                        secType="STK",
                        currency="USD",
                        exchange="SMART",
                        primaryExchange="NASDAQ",
                        localSymbol="AAPL",
                        tradingClass="NMS",
                        multiplier="",
                    )
                )
            ]

        def request_option_parameters(self, **kwargs):
            return [
                {
                    "exchange": "BOX",
                    "underlying_conid": 101,
                    "trading_class": "AAPL",
                    "multiplier": "100",
                    "expirations": ["20260417"],
                    "strikes": [180.0, 185.0],
                }
            ]

    monkeypatch.setattr("options_lab.ibkr.chains.DelayedOnlyIbkrSession", _FakeSession)

    result = fetch_option_chain(
        "AAPL",
        settings=ConnectionSettings(),
        market_data_mode="delayed",
        exchange="SMART",
        currency="USD",
    )

    assert result.rows == []
    assert result.diagnostics.failure_stage == "exchange_filter"
    assert result.diagnostics.row_counts["raw_opt_param_rows"] == 1
    assert result.diagnostics.row_counts["after_exchange_filter"] == 0
    assert result.diagnostics.raw_exchanges_seen == ["BOX"]
    assert result.diagnostics.resolved_underlying.symbol == "AAPL"


def test_fetch_option_chain_can_include_all_exchanges_and_normalize_ibkr_expiries(monkeypatch):
    from options_lab.ibkr.chains import fetch_option_chain
    from options_lab.ibkr.models import ConnectionSettings

    class _FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request_contract_details(self, contract, *, timeout=None):
            return [
                SimpleNamespace(
                    contract=SimpleNamespace(
                        conId=202,
                        symbol="AAPL",
                        secType="STK",
                        currency="USD",
                        exchange="SMART",
                        primaryExchange="NASDAQ",
                        localSymbol="AAPL",
                        tradingClass="NMS",
                        multiplier="",
                    )
                )
            ]

        def request_option_parameters(self, **kwargs):
            return [
                {
                    "exchange": "BOX",
                    "underlying_conid": 202,
                    "trading_class": "AAPL",
                    "multiplier": "100",
                    "expirations": ["20260417"],
                    "strikes": [180.0, 185.0],
                }
            ]

    monkeypatch.setattr("options_lab.ibkr.chains.DelayedOnlyIbkrSession", _FakeSession)

    result = fetch_option_chain(
        "AAPL",
        settings=ConnectionSettings(),
        market_data_mode="delayed",
        exchange="SMART",
        currency="USD",
        include_all_exchanges=True,
    )

    assert len(result.rows) == 4
    assert {row.expiry_date for row in result.rows} == {"2026-04-17"}
    assert result.diagnostics.failure_stage is None


def test_ibkr_connection_accepts_tick_option_computation_with_tick_attrib(monkeypatch):
    from options_lab.ibkr.connection import DelayedOnlyIbkrSession
    from options_lab.ibkr.models import ConnectionSettings

    class _FakeWrapper:
        pass

    class _FakeClient:
        def __init__(self, wrapper):
            self.wrapper = wrapper

    monkeypatch.setattr("options_lab.ibkr.connection._ibapi_classes", lambda: (_FakeClient, _FakeWrapper))

    session = DelayedOnlyIbkrSession(ConnectionSettings())
    session._app.tickOptionComputation(1, 13, None, 0.45, 0.51, 0.92, 0.0, 0.08, 0.12, -0.03, 15.2)

    ticks = session._market_ticks[1]
    assert ticks["implied_volatility"] == 0.45
    assert ticks["delta"] == 0.51
    assert ticks["option_price"] == 0.92
    assert ticks["under_price"] == 15.2


def test_ibkr_batch_snapshot_waits_for_real_ticks_after_initial_delayed_warning(monkeypatch):
    import threading
    import time

    from options_lab.ibkr.connection import DelayedOnlyIbkrSession
    from options_lab.ibkr.models import ConnectionSettings

    class _FakeWrapper:
        pass

    class _FakeClient:
        def __init__(self, wrapper):
            self.wrapper = wrapper

        def reqMarketDataType(self, code):
            return None

        def reqMktData(self, reqId, contract, genericTickList, snapshot, regulatorySnapshot, mktDataOptions):
            def _emit_events():
                time.sleep(0.05)
                session._on_error(reqId, 10090, "Requested market data is not subscribed. Displaying delayed market data.")
                time.sleep(0.30)
                session._market_data_type[reqId] = 3
                session._market_ticks[reqId]["implied_volatility"] = 0.40 + (reqId * 0.01)
                session._market_tick_events[reqId].set()

            threading.Thread(target=_emit_events, daemon=True).start()

        def cancelMktData(self, reqId):
            return None

    monkeypatch.setattr("options_lab.ibkr.connection._ibapi_classes", lambda: (_FakeClient, _FakeWrapper))

    session = DelayedOnlyIbkrSession(ConnectionSettings())
    results = session.collect_market_snapshots(
        [SimpleNamespace(symbol="GPRE1"), SimpleNamespace(symbol="GPRE2")],
        market_data_mode="delayed",
        wait_seconds=1.0,
        settle_seconds=0.2,
    )

    assert len(results) == 2
    assert all(result["market_data_type_code"] == 3 for result in results)
    assert all(result["ticks"]["implied_volatility"] > 0.0 for result in results)


def test_select_contract_candidates_with_diagnostics_preserves_both_sides_for_small_mixed_slice():
    from options_lab.ibkr.chains import select_contract_candidates_with_diagnostics
    from options_lab.ibkr.models import ChainRow, ConnectionSettings

    rows = [
        ChainRow(
            ticker="AAPL",
            underlying_conid=101,
            fetched_at=_utc("2026-04-12T20:15:00Z"),
            market_data_mode="delayed",
            exchange="SMART",
            trading_class="AAPL",
            multiplier="100",
            expiry_date="2026-04-17",
            strike=strike,
            option_type=option_type,
            currency="USD",
            source="ibkr",
            connection=ConnectionSettings(),
        )
        for strike in (180.0, 185.0, 190.0)
        for option_type in ("call", "put")
    ]

    selected, diagnostics = select_contract_candidates_with_diagnostics(
        rows,
        expiries=["2026-04-17"],
        right="both",
        max_contracts=4,
    )

    assert len(selected) == 4
    assert [row.strike for row in selected] == [180.0, 180.0, 185.0, 185.0]
    assert [row.option_type for row in selected] == ["call", "put", "call", "put"]
    assert diagnostics.failure_stage is None
    assert diagnostics.row_counts["final_selected"] == 4


def test_fetch_option_snapshots_returns_contract_match_diagnostics_for_expiry_mismatch(monkeypatch):
    from options_lab.ibkr.market_data import fetch_option_snapshots
    from options_lab.ibkr.models import (
        ChainDiscoveryDiagnostics,
        ChainFetchResult,
        ChainRow,
        ConnectionSettings,
        ResolvedUnderlyingContract,
        UnderlyingQuoteSnapshot,
    )

    monkeypatch.setattr(
        "options_lab.ibkr.market_data.fetch_underlying_snapshot",
        lambda *args, **kwargs: UnderlyingQuoteSnapshot(
            ticker="AAPL",
            snapshot_timestamp=_utc("2026-04-12T20:15:00Z"),
            market_data_mode="delayed",
            market_data_type_code=3,
            bid=180.0,
            ask=180.5,
            last=180.2,
            close=179.8,
            mid=180.25,
            mark=180.25,
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            source="ibkr",
            warnings=[],
            missing_fields=[],
            connection=ConnectionSettings(),
        ),
    )
    monkeypatch.setattr(
        "options_lab.ibkr.market_data.fetch_option_chain",
        lambda *args, **kwargs: ChainFetchResult(
            rows=[
                ChainRow(
                    ticker="AAPL",
                    underlying_conid=101,
                    fetched_at=_utc("2026-04-12T20:15:00Z"),
                    market_data_mode="delayed",
                    exchange="SMART",
                    trading_class="AAPL",
                    multiplier="100",
                    expiry_date="2026-04-24",
                    strike=180.0,
                    option_type="call",
                    currency="USD",
                    source="ibkr",
                    connection=ConnectionSettings(),
                )
            ],
            diagnostics=ChainDiscoveryDiagnostics(
                requested={"ticker": "AAPL", "exchange": "SMART", "currency": "USD"},
                resolved_underlying=ResolvedUnderlyingContract(
                    conid=101,
                    symbol="AAPL",
                    sec_type="STK",
                    currency="USD",
                    exchange="SMART",
                    primary_exchange="NASDAQ",
                    local_symbol="AAPL",
                    trading_class="NMS",
                    multiplier=None,
                ),
                raw_option_parameter_rows=[],
                raw_exchanges_seen=["SMART"],
                raw_trading_classes_seen=["AAPL"],
                available_expiries=["2026-04-24"],
                available_strike_count=2,
                available_strike_sample=[180.0, 185.0],
                row_counts={
                    "raw_opt_param_rows": 1,
                    "after_underlying_selection": 1,
                    "after_exchange_filter": 1,
                    "after_trading_class_filter": 1,
                    "after_expiry_strike_normalization": 1,
                    "final_chain_rows": 1,
                },
                selected_exchange="SMART",
                failure_stage=None,
            ),
        ),
    )

    result = fetch_option_snapshots(
        "AAPL",
        settings=ConnectionSettings(),
        market_data_mode="delayed",
        expiries=["2026-04-17"],
        right="both",
        exchange="SMART",
        currency="USD",
    )

    assert result.quotes == []
    assert result.diagnostics.contract_match.requested_expiry_exists is False
    assert result.diagnostics.contract_match.available_expiries == ["2026-04-24"]
    assert result.diagnostics.failure_stage == "expiry_filter"


def test_record_request_failure_persists_failure_stage_and_diagnostics(temp_data_root: Path):
    from options_lab.ibkr.models import ConnectionSettings
    from options_lab.ibkr.store import record_request_failure

    result = record_request_failure(
        "AAPL",
        request_type="chain",
        market_data_mode="delayed",
        connection=ConnectionSettings(host="127.0.0.1", port=7496, client_id=88),
        error_message="Exchange filter removed all rows.",
        warnings=["Requested exchange SMART had no rows."],
        diagnostics={
            "failure_stage": "exchange_filter",
            "raw_exchanges_seen": ["BOX", "CBOE"],
            "row_counts": {"raw_opt_param_rows": 2, "after_exchange_filter": 0},
        },
        data_root=temp_data_root,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["failure_stage"] == "exchange_filter"
    assert manifest["diagnostics"]["raw_exchanges_seen"] == ["BOX", "CBOE"]
    assert manifest["diagnostics"]["row_counts"]["after_exchange_filter"] == 0
