from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_lab.metadata import build_metadata
from options_lab.rates.rate_selector import get_risk_free_rate, select_series_for_days
from options_lab.rates.rate_store import save_merged_table


SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "GPRE"
    / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
)


def _write_sample_merged_store(data_root: Path) -> None:
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


def test_selector_mapping_by_horizon():
    assert select_series_for_days(30) == "DGS1MO"
    assert select_series_for_days(45) == "DGS1MO"
    assert select_series_for_days(46) == "DGS3MO"
    assert select_series_for_days(135) == "DGS3MO"
    assert select_series_for_days(136) == "DGS6MO"
    assert select_series_for_days(270) == "DGS6MO"
    assert select_series_for_days(271) == "DGS1"


def test_selector_falls_back_to_prior_business_date(temp_data_root: Path):
    _write_sample_merged_store(temp_data_root)

    match = get_risk_free_rate(
        snapshot_date="2026-04-12",
        expiry_date="2026-04-17",
        data_root=temp_data_root,
    )

    assert match.series_used == "DGS1MO"
    assert match.matched_date.isoformat() == "2026-04-10"
    assert match.used_prior_date is True
    assert match.rate_decimal == 0.0422


def test_metadata_prefers_explicit_rate_then_local_store_then_default(temp_data_root: Path):
    _write_sample_merged_store(temp_data_root)

    from_store = build_metadata(SAMPLE_FILE, rates_data_root=temp_data_root)
    assert from_store.risk_free_rate == 0.0422
    assert from_store.risk_free_rate_series == "DGS1MO"
    assert from_store.risk_free_rate_source == "fred_local_store"

    explicit = build_metadata(
        SAMPLE_FILE,
        metadata_override={"risk_free_rate": 0.031, "risk_free_rate_source": "manual_override"},
        rates_data_root=temp_data_root,
    )
    assert explicit.risk_free_rate == 0.031
    assert explicit.risk_free_rate_source == "manual_override"

    fallback = build_metadata(
        SAMPLE_FILE,
        rates_data_root=temp_data_root / "missing_store",
    )
    assert fallback.risk_free_rate == 0.04
    assert fallback.risk_free_rate_source == "default_fallback"
