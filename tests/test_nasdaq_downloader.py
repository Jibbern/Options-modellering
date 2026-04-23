from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_lab.prices.nasdaq_downloader import download_nasdaq_prices
from options_lab.prices.price_store import load_price_history

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200, text: str | None = None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("No more fake responses queued.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

def test_download_nasdaq_prices_normalizes_and_writes_outputs(temp_data_root: Path):
    payload = {
        "data": {
            "totalRecords": "3",
            "tradesTable": {
                "rows": [
                    {
                        "date": "04/10/2026",
                        "close/last": "$14.72",
                        "volume": "1,234,567",
                        "open": "$14.50",
                        "high": "$14.90",
                        "low": "$14.31",
                    },
                    {
                        "date": "04/09/2026",
                        "close/last": "$14.10",
                        "volume": "987,654",
                        "open": "$14.00",
                        "high": "$14.25",
                        "low": "$13.88",
                    },
                    {
                        "date": "04/10/2026",
                        "close/last": "$14.80",
                        "volume": "1,111,111",
                        "open": "$14.51",
                        "high": "$14.95",
                        "low": "$14.30",
                    },
                ]
            },
        }
    }
    session = FakeSession([FakeResponse(payload)])

    manifest = download_nasdaq_prices(
        "GPRE",
        data_root=temp_data_root,
        start="2026-04-01",
        end="2026-04-12",
        session=session,
    )

    history = load_price_history("GPRE", temp_data_root)
    assert len(history) == 2
    assert history["date"].is_monotonic_increasing
    assert history.iloc[-1]["close"] == pytest.approx(14.80)
    assert int(history.iloc[-1]["volume"]) == 1111111
    assert set(history.columns) == {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adj_close",
        "source",
        "downloaded_at",
    }
    assert manifest["row_count"] == 2
    assert Path(manifest["manifest_path"]).exists()


def test_download_nasdaq_prices_preserves_raw_error_when_unavailable(temp_data_root: Path):
    payload = {
        "data": None,
        "message": "Data is temporarily unavailable.",
    }
    session = FakeSession([FakeResponse(payload)])

    with pytest.raises(RuntimeError):
        download_nasdaq_prices(
            "GPRE",
            data_root=temp_data_root,
            start="2026-04-01",
            end="2026-04-12",
            session=session,
        )

    raw_files = list((temp_data_root / "GPRE" / "historical_prices" / "raw").glob("*.json"))
    assert raw_files


def test_download_nasdaq_prices_handles_multi_page_results(monkeypatch, temp_data_root: Path):
    monkeypatch.setattr("options_lab.prices.nasdaq_downloader.DEFAULT_LIMIT", 1)
    first_page = {
        "data": {
            "totalRecords": "2",
            "tradesTable": {
                "rows": [
                    {
                        "date": "04/10/2026",
                        "close/last": "$14.72",
                        "volume": "1,234",
                        "open": "$14.50",
                        "high": "$14.90",
                        "low": "$14.31",
                    }
                ]
            },
        }
    }
    second_page = {
        "data": {
            "totalRecords": "2",
            "tradesTable": {
                "rows": [
                    {
                        "date": "04/09/2026",
                        "close/last": "$14.10",
                        "volume": "987",
                        "open": "$14.00",
                        "high": "$14.25",
                        "low": "$13.88",
                    }
                ]
            },
        }
    }
    session = FakeSession([FakeResponse(first_page), FakeResponse(second_page)])

    download_nasdaq_prices(
        "GPRE",
        data_root=temp_data_root,
        start="2026-04-01",
        end="2026-04-12",
        session=session,
    )

    history = load_price_history("GPRE", temp_data_root)
    assert len(history) == 2
    assert session.calls[1]["params"]["offset"] == 1
