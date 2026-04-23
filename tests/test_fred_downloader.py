from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from options_lab.rates.fred_downloader import (
    FredApiKeyError,
    download_fred_rates,
    get_fred_api_key,
    normalize_fred_payload,
)
from options_lab.rates.rate_store import (
    FRED_SERIES,
    build_merged_table,
    empty_series_history,
    get_latest_rates_snapshot,
    load_merged_rates,
    load_series_history,
)

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, url, params, timeout):
        series_id = params["series_id"]
        return _FakeResponse(self.payloads[series_id])


def test_api_key_handling_requires_env(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(FredApiKeyError):
        get_fred_api_key()


def test_normalization_handles_missing_values_and_decimal_conversion():
    payload = {
        "observations": [
            {"date": "2026-04-10", "value": "4.50"},
            {"date": "2026-04-11", "value": "."},
            {"date": "2026-04-12", "value": ""},
        ]
    }
    downloaded_at = datetime(2026, 4, 12, 12, 0, tzinfo=timezone.utc)
    frame = normalize_fred_payload("DGS1MO", payload, downloaded_at=downloaded_at)

    assert frame.loc[0, "rate_percent"] == 4.50
    assert frame.loc[0, "rate_decimal"] == 0.045
    assert pd.isna(frame.loc[1, "rate_percent"])
    assert pd.isna(frame.loc[2, "rate_decimal"])
    assert frame.loc[1, "observation_status"] == "missing"


def test_build_merged_table_contains_expected_columns():
    frame_1m = pd.DataFrame(
        [
            {
                "date": "2026-04-10",
                "series_id": "DGS1MO",
                "rate_percent": 4.30,
                "rate_decimal": 0.043,
                "source": "FRED",
                "downloaded_at": "2026-04-12T10:00:00Z",
                "observation_status": "observed",
            }
        ]
    )
    frame_3m = pd.DataFrame(
        [
            {
                "date": "2026-04-10",
                "series_id": "DGS3MO",
                "rate_percent": 4.35,
                "rate_decimal": 0.0435,
                "source": "FRED",
                "downloaded_at": "2026-04-12T10:00:00Z",
                "observation_status": "observed",
            }
        ]
    )
    merged = build_merged_table(
        {
            "DGS1MO": frame_1m,
            "DGS3MO": frame_3m,
            "DGS6MO": empty_series_history("DGS6MO"),
            "DGS1": empty_series_history("DGS1"),
        }
    )

    expected_columns = {
        "date",
        "dgs1mo_percent",
        "dgs1mo_decimal",
        "dgs3mo_percent",
        "dgs3mo_decimal",
        "dgs6mo_percent",
        "dgs6mo_decimal",
        "dgs1_percent",
        "dgs1_decimal",
        "downloaded_at",
    }
    assert expected_columns.issubset(set(merged.columns))


def test_download_fred_rates_creates_local_store_and_deduplicates(monkeypatch, temp_data_root: Path):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    payloads = {
        "DGS1MO": {
            "observations": [
                {"date": "2026-04-10", "value": "4.30"},
                {"date": "2026-04-10", "value": "4.31"},
                {"date": "2026-04-11", "value": "."},
            ]
        },
        "DGS3MO": {"observations": [{"date": "2026-04-10", "value": "4.35"}]},
        "DGS6MO": {"observations": [{"date": "2026-04-10", "value": "4.40"}]},
        "DGS1": {"observations": [{"date": "2026-04-10", "value": "4.20"}]},
    }
    session = _FakeSession(payloads)

    manifest = download_fred_rates(data_root=temp_data_root, full_refresh=True, session=session)

    assert Path(manifest["manifest_path"]).exists()
    for series_id in FRED_SERIES:
        frame = load_series_history(series_id, temp_data_root)
        assert frame["date"].duplicated().sum() == 0
    merged = load_merged_rates(temp_data_root)
    latest_snapshot = get_latest_rates_snapshot(temp_data_root)

    assert len(load_series_history("DGS1MO", temp_data_root)) == 2
    assert "dgs1mo_decimal" in merged.columns
    assert latest_snapshot["series"]["DGS1MO"]["rate_percent"] == 4.31
