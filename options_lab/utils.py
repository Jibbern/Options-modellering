"""Shared utility helpers and constants for the standalone Options lab."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_MULTIPLIER = 100
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_DIVIDEND_YIELD = 0.0
DEFAULT_HORIZONS = ("1m", "3m", "6m", "12m")
HORIZON_DAYS = {
    "1m": 30,
    "3m": 91,
    "6m": 182,
    "12m": 365,
}


def windows_extended_path(path: str | Path) -> str:
    """Return a Windows extended-length path string when needed.

    This keeps long bundle/chart filenames usable on Windows without changing
    the public artifact naming scheme.
    """

    raw_path = Path(path)
    try:
        normalized = raw_path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        normalized = raw_path
    text = str(normalized)
    if os.name != "nt":
        return text
    if text.startswith("\\\\?\\"):
        return text
    if len(text) < 240:
        return text
    if text.startswith("\\\\"):
        return "\\\\?\\UNC\\" + text.lstrip("\\")
    return "\\\\?\\" + text


def clean_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def normalize_column_name(name: str) -> str:
    text = clean_string(name).lower()
    text = text.replace("%", "pct_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_number(
    value: Any,
    *,
    percent: bool = False,
    unch_as_zero: bool = False,
) -> float | None:
    text = clean_string(value)
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"n/a", "na", "null", "none", "--", "-"}:
        return None
    if lowered == "unch":
        return 0.0 if unch_as_zero else None
    text = text.replace(",", "")
    if text.endswith("%"):
        percent = True
        text = text[:-1]
    try:
        number = float(text)
    except ValueError:
        return None
    if percent:
        number /= 100.0
    return number


def parse_int(value: Any, *, unch_as_zero: bool = False) -> int | None:
    number = parse_number(value, unch_as_zero=unch_as_zero)
    if number is None:
        return None
    return int(round(number))


def parse_date(value: Any) -> date | None:
    """Parse local input values into a plain ``date``.

    ``datetime`` subclasses ``date``, so handle it first to avoid leaking
    time-of-day into downstream horizon/expiry calculations.
    """

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = clean_string(value)
    if not text or text.lower() == "n/a":
        return None
    formats = ("%Y-%m-%d", "%Y%m%d", "%m/%d/%y", "%m/%d/%Y", "%m-%d-%Y")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def years_between(start: date, end: date) -> float:
    return max((end - start).days, 0) / 365.25


def horizon_to_days(value: str | int | float) -> int:
    if isinstance(value, (int, float)):
        return max(int(round(value)), 0)
    text = clean_string(value).lower()
    if text in HORIZON_DAYS:
        return HORIZON_DAYS[text]
    if text.endswith("d"):
        return max(int(float(text[:-1])), 0)
    if text.endswith("m"):
        return max(int(round(float(text[:-1]) * 30.4375)), 0)
    if text.endswith("y"):
        return max(int(round(float(text[:-1]) * 365.25)), 0)
    return max(int(float(text)), 0)


def slugify(value: str) -> str:
    text = clean_string(value).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "output"


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not clean_string(value):
            continue
        return value
    return None


def build_stock_grid(
    spot_price: float,
    *,
    down_move: float = 0.5,
    up_move: float = 0.5,
    points: int = 201,
) -> np.ndarray:
    lower = max(spot_price * (1.0 - down_move), 0.01)
    upper = max(spot_price * (1.0 + up_move), lower + 0.01)
    return np.linspace(lower, upper, points)


def finite_or_none(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None
