from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from options_lab.persistence import write_dataframe_parquet_if_available, write_json
from options_lab.prices.price_store import save_price_history


def test_parquet_helper_skips_cleanly_when_engine_is_unavailable(monkeypatch, temp_analysis_root: Path):
    frame = pd.DataFrame([{"value": 1.0}])

    def _raise_import_error(self, path, index=False):
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_import_error)

    result = write_dataframe_parquet_if_available(frame, temp_analysis_root / "sample.parquet", index=False)

    assert result.written is False
    assert "Skipped Parquet output" in (result.note or "")
    assert not result.path.exists()


def test_price_store_keeps_csv_outputs_when_parquet_is_unavailable(monkeypatch, temp_data_root: Path):
    frame = pd.DataFrame(
        [
            {
                "ticker": "GPRE",
                "date": "2026-04-10",
                "open": 14.5,
                "high": 14.9,
                "low": 14.3,
                "close": 14.72,
                "volume": 1234567,
                "adj_close": None,
                "source": "nasdaq_historical_quotes",
                "downloaded_at": "2026-04-12T10:00:00Z",
            }
        ]
    )

    def _raise_import_error(self, path, index=False):
        raise ImportError("missing parquet engine")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise_import_error)

    saved = save_price_history("GPRE", frame, data_root=temp_data_root)

    assert Path(saved["normalized_csv"]).exists()
    assert Path(saved["merged_csv"]).exists()
    assert saved["normalized_parquet_written"] == "False"
    assert saved["merged_parquet_written"] == "False"
    assert "Skipped Parquet output" in saved["normalized_parquet_note"]


def test_write_json_converts_non_finite_numbers_to_null(temp_analysis_root: Path):
    output_path = temp_analysis_root / "report_metadata.json"
    write_json({"max_gain": float("inf"), "max_loss": float("-inf"), "ok": 1.0}, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert '"max_gain": null' in text
    assert '"max_loss": null' in text
    assert "Infinity" not in text


def test_write_json_handles_nested_pandas_and_numpy_scalars(temp_analysis_root: Path):
    output_path = temp_analysis_root / "nested.json"
    payload = {
        "outer": {
            "note": pd.NA,
            "when": pd.Timestamp("2026-04-12"),
            "values": [np.int64(3), np.float64(1.25), np.float64(np.nan), pd.NaT],
            "flags": {"ok": np.bool_(True), "bad": float("inf")},
        }
    }
    write_json(payload, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert '"note": null' in text
    assert '"when": "2026-04-12T00:00:00"' in text
    assert '"values": [' in text
    assert "<NA>" not in text
    assert "NAType" not in text
    assert "Infinity" not in text
