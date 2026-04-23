"""Historical price utilities for the options lab."""

__all__ = [
    "SpotPriceMatch",
    "download_nasdaq_prices",
    "get_latest_price_date",
    "get_underlying_spot",
    "load_price_history",
    "normalize_manual_price_file",
]


def __getattr__(name: str):
    if name in {"download_nasdaq_prices"}:
        from . import nasdaq_downloader

        return getattr(nasdaq_downloader, name)
    if name in {"load_price_history", "normalize_manual_price_file"}:
        from . import price_store

        return getattr(price_store, name)
    if name in {"SpotPriceMatch", "get_latest_price_date", "get_underlying_spot"}:
        from . import price_selector

        return getattr(price_selector, name)
    raise AttributeError(name)
