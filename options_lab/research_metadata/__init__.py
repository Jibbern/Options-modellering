"""Research-metadata helpers for richer local options analysis context."""

from .catalog import build_ticker_catalog, coverage_summary, discover_chain_snapshots, load_ticker_catalog, update_ticker_catalog
from .dividends import get_dividend_assumption, register_dividends_file
from .events import get_nearest_event, list_events, register_events_file
from .expected_move import get_expected_move, register_expected_move_file
from .notes import get_recent_notes, register_notes_file
from .options_overview import get_options_overview, register_options_overview_file


def resolve_research_context(
    ticker: str | None,
    *,
    snapshot_date: str | None = None,
    expiry_date: str | None = None,
    data_root=None,
) -> dict:
    """Resolve the optional local research context around one snapshot."""

    if not ticker or not snapshot_date:
        return {
            "expected_move": {},
            "options_overview": {},
            "nearest_event": {},
            "dividend_assumption": {},
            "notes": [],
            "coverage_summary": {},
        }

    return {
        "expected_move": get_expected_move(
            ticker,
            snapshot_date,
            expiry_date=expiry_date,
            data_root=data_root,
        ),
        "options_overview": get_options_overview(
            ticker,
            snapshot_date,
            data_root=data_root,
        ),
        "nearest_event": get_nearest_event(
            ticker,
            snapshot_date,
            expiry_date=expiry_date,
            data_root=data_root,
        ),
        "dividend_assumption": get_dividend_assumption(
            ticker,
            snapshot_date,
            data_root=data_root,
        ),
        "notes": get_recent_notes(
            ticker,
            snapshot_date,
            data_root=data_root,
        ),
        "coverage_summary": coverage_summary(ticker, data_root),
    }


__all__ = [
    "build_ticker_catalog",
    "coverage_summary",
    "discover_chain_snapshots",
    "get_dividend_assumption",
    "get_expected_move",
    "get_nearest_event",
    "get_options_overview",
    "get_recent_notes",
    "list_events",
    "load_ticker_catalog",
    "register_dividends_file",
    "register_events_file",
    "register_expected_move_file",
    "register_notes_file",
    "register_options_overview_file",
    "resolve_research_context",
    "update_ticker_catalog",
]
