from __future__ import annotations

from pathlib import Path

from options_lab.metadata import build_metadata


SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "GPRE"
    / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
)


def test_metadata_aliases_and_explicit_spot_override_are_supported():
    metadata = build_metadata(
        SAMPLE_FILE,
        metadata_override={
            "spot_price": 14.95,
            "spot_source": "sidecar_alias",
            "spot_matched_date": "2026-04-12",
            "risk_free_rate": 0.031,
            "risk_free_source": "manual_curve",
            "risk_free_matched_date": "2026-04-10",
            "snapshot_time": "15:59:00",
        },
        spot_price=15.23,
    )

    assert metadata.spot_price == 15.23
    assert metadata.spot_price_source == "sidecar_alias"
    assert metadata.spot_price_matched_date.isoformat() == "2026-04-12"
    assert metadata.risk_free_rate == 0.031
    assert metadata.risk_free_rate_source == "manual_curve"
    assert metadata.risk_free_rate_matched_date.isoformat() == "2026-04-10"
    assert metadata.snapshot_time == "15:59:00"
