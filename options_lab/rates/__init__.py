"""FRED-backed risk-free rate utilities for the options lab."""

__all__ = [
    "FRED_SERIES",
    "FredApiKeyError",
    "RiskFreeRateMatch",
    "download_fred_rates",
    "get_latest_rates_snapshot",
    "get_risk_free_rate",
    "load_merged_rates",
    "load_series_history",
]


def __getattr__(name: str):
    if name in {"FRED_SERIES", "load_merged_rates", "load_series_history"}:
        from . import rate_store

        return getattr(rate_store, name)
    if name in {"RiskFreeRateMatch", "get_latest_rates_snapshot", "get_risk_free_rate"}:
        from . import rate_selector

        return getattr(rate_selector, name)
    if name in {"FredApiKeyError", "download_fred_rates"}:
        from . import fred_downloader

        return getattr(fred_downloader, name)
    raise AttributeError(name)
