"""Small persistence helpers for CSV, JSON, and optional Parquet outputs.

CSV and JSON are treated as the reliable baseline for the Options lab.
Parquet is supported when a compatible engine is installed, but the project
should remain usable without it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_directory, windows_extended_path


@dataclass(frozen=True)
class ParquetWriteResult:
    """Result of attempting to write a Parquet file."""

    path: Path
    written: bool
    note: str | None = None


def _is_missing_scalar(value: Any) -> bool:
    """Return True when a scalar should map to JSON null."""

    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (str, bytes, Path, dict, list, tuple, set, pd.Series, pd.DataFrame, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def make_json_safe(value: Any) -> Any:
    """Convert nested Python, pandas, and numpy values into strict JSON-safe data."""

    if is_dataclass(value) and not isinstance(value, type):
        return make_json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return [make_json_safe(record) for record in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [make_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if _is_missing_scalar(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(payload: Any, path: str | Path, *, indent: int = 2) -> Path:
    """Write a JSON payload to disk and return the final path."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    Path(windows_extended_path(output_path)).write_text(
        json.dumps(make_json_safe(payload), indent=indent, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def write_dataframe_csv(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Write a DataFrame to CSV and return the final path."""

    output_path = Path(path)
    ensure_directory(output_path.parent)
    frame.to_csv(windows_extended_path(output_path), index=index)
    return output_path


def write_dataframe_parquet_if_available(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
) -> ParquetWriteResult:
    """Attempt to write a DataFrame to Parquet without making it mandatory.

    Pandas needs an optional engine such as ``pyarrow`` or ``fastparquet`` for
    Parquet support. If no engine is installed, the caller still gets a clean
    result object and can continue with CSV-backed workflows.
    """

    output_path = Path(path)
    ensure_directory(output_path.parent)
    try:
        frame.to_parquet(windows_extended_path(output_path), index=index)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError) as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return ParquetWriteResult(
            path=output_path,
            written=False,
            note=(
                "Skipped Parquet output because no compatible parquet engine was "
                f"available: {exc}"
            ),
        )
    return ParquetWriteResult(path=output_path, written=True, note=None)
