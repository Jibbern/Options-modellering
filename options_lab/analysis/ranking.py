"""Shared ranking and summary helpers for the analysis-first layer."""

from __future__ import annotations

from typing import Any

from ..utils import clean_string


def unique_warnings(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return stable de-duplicated warnings."""

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values or []:
        text = clean_string(value)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def coverage_payload(status: str | None, shareability_status: str | None = None, **extra: Any) -> dict[str, Any]:
    """Build a compact bundle coverage/status payload."""

    payload = {
        "status": clean_string(status) or None,
        "shareability_status": clean_string(shareability_status) or None,
    }
    payload.update(extra)
    return payload
