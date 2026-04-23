from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from options_lab.io import load_chain
from options_lab.prices.price_selector import get_underlying_spot
from options_lab.prices.price_store import (
    load_price_history,
    normalize_manual_price_file,
    save_price_history,
)


SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "GPRE"
    / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manual_import_normalizes_and_deduplicates(temp_data_root: Path):
    manual_dir = temp_data_root / "GPRE" / "historical_prices" / "raw" / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    manual_file = manual_dir / "gpre_manual.csv"
    manual_file.write_text(
        "\n".join(
            [
                "Date,Close/Last,Volume,Open,High,Low,Adj Close",
                "04/10/2026,$14.72,\"1,234,567\",$14.50,$14.90,$14.31,$14.72",
                "04/09/2026,$14.10,\"987,654\",$14.00,$14.25,$13.88,$14.10",
                "04/10/2026,$14.80,\"1,111,111\",$14.51,$14.95,$14.30,$14.80",
            ]
        ),
        encoding="utf-8",
    )

    history = normalize_manual_price_file(manual_file, "GPRE", data_root=temp_data_root)

    assert len(history) == 2
    assert history.iloc[-1]["close"] == pytest.approx(14.80)
    assert int(history.iloc[-1]["volume"]) == 1111111
    assert history.iloc[-1]["adj_close"] == pytest.approx(14.80)


def test_selector_falls_back_to_prior_trading_day(temp_data_root: Path):
    frame = pd.DataFrame(
        [
            {
                "ticker": "GPRE",
                "date": "2026-04-09",
                "open": 14.0,
                "high": 14.25,
                "low": 13.88,
                "close": 14.10,
                "volume": 987654,
                "adj_close": None,
                "source": "nasdaq_historical_quotes",
                "downloaded_at": "2026-04-12T10:00:00Z",
            },
            {
                "ticker": "GPRE",
                "date": "2026-04-10",
                "open": 14.5,
                "high": 14.9,
                "low": 14.31,
                "close": 14.72,
                "volume": 1234567,
                "adj_close": None,
                "source": "nasdaq_historical_quotes",
                "downloaded_at": "2026-04-12T10:00:00Z",
            },
        ]
    )
    save_price_history("GPRE", frame, data_root=temp_data_root)

    match = get_underlying_spot("GPRE", "2026-04-12", data_root=temp_data_root)

    assert match.matched_date.isoformat() == "2026-04-10"
    assert match.close_price == pytest.approx(14.72)
    assert match.used_prior_date is True


def test_load_chain_can_use_local_price_store_before_moneyness(temp_data_root: Path):
    frame = pd.DataFrame(
        [
            {
                "ticker": "GPRE",
                "date": "2026-04-10",
                "open": 14.5,
                "high": 14.9,
                "low": 14.31,
                "close": 14.72,
                "volume": 1234567,
                "adj_close": None,
                "source": "nasdaq_historical_quotes",
                "downloaded_at": "2026-04-12T10:00:00Z",
            }
        ]
    )
    save_price_history("GPRE", frame, data_root=temp_data_root)

    chain = load_chain(SAMPLE_FILE, prices_data_root=temp_data_root)

    assert chain.spot_price == pytest.approx(14.72)
    assert chain.metadata.spot_price_matched_date.isoformat() == "2026-04-10"
    assert any("prior trading-day close" in warning for warning in chain.warnings)


def test_load_chain_still_falls_back_to_moneyness_when_no_store_exists(temp_data_root: Path):
    chain = load_chain(SAMPLE_FILE, prices_data_root=temp_data_root)

    assert 15.0 < chain.spot_price < 15.3
    assert any("No local historical price store" in warning for warning in chain.warnings)


def test_load_price_history_returns_empty_frame_when_store_missing(temp_data_root: Path):
    history = load_price_history("GPRE", data_root=temp_data_root)
    assert history.empty
