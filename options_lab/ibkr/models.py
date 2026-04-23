"""Typed records for delayed-only IBKR ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..persistence import make_json_safe
from ..utils import clean_string, parse_date

SUPPORTED_MARKET_DATA_MODES = {"delayed": 3, "delayed_frozen": 4}
UNSUPPORTED_MARKET_DATA_MODES = {1: "live", 2: "frozen"}


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_market_data_mode(mode: str) -> str:
    text = clean_string(mode).lower()
    if text not in SUPPORTED_MARKET_DATA_MODES:
        raise ValueError(f"Unsupported IBKR market data mode: {mode}")
    return text


def market_data_mode_code(mode: str) -> int:
    return SUPPORTED_MARKET_DATA_MODES[normalize_market_data_mode(mode)]


def validate_effective_market_data_type(code: int | None) -> tuple[bool, str]:
    if code in SUPPORTED_MARKET_DATA_MODES.values():
        return True, "accepted"
    if code in UNSUPPORTED_MARKET_DATA_MODES:
        return False, (
            "IBKR returned a live/frozen-live market-data type for this request. "
            "The delayed-only workflow does not treat that as success."
        )
    return False, "IBKR did not confirm delayed or delayed-frozen market data for this request."


@dataclass(frozen=True)
class ConnectionSettings:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 71

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True)
class UnderlyingQuoteSnapshot:
    ticker: str
    snapshot_timestamp: datetime
    market_data_mode: str
    market_data_type_code: int | None
    bid: float | None
    ask: float | None
    last: float | None
    close: float | None
    mid: float | None
    mark: float | None
    exchange: str | None
    primary_exchange: str | None
    currency: str | None
    source: str
    warnings: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    connection: ConnectionSettings = field(default_factory=ConnectionSettings)
    resolved_underlying: "ResolvedUnderlyingContract | None" = None

    def to_record(self) -> dict[str, Any]:
        payload = make_json_safe(asdict(self))
        payload["ticker"] = clean_string(self.ticker).upper()
        payload["snapshot_timestamp"] = isoformat_utc(self.snapshot_timestamp)
        payload["market_data_mode"] = normalize_market_data_mode(self.market_data_mode)
        payload["warnings"] = list(self.warnings)
        payload["missing_fields"] = list(self.missing_fields)
        payload.pop("connection", None)
        payload.pop("resolved_underlying", None)
        return payload


@dataclass(frozen=True)
class IbkrSpotMatch:
    ticker: str
    requested_date: date
    matched_timestamp: datetime
    matched_date: date
    close_price: float
    source: str
    market_data_mode: str
    field_used: str | None = None
    used_prior_date: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = make_json_safe(asdict(self))
        payload["requested_date"] = self.requested_date.isoformat()
        payload["matched_timestamp"] = isoformat_utc(self.matched_timestamp)
        payload["matched_date"] = self.matched_date.isoformat()
        return payload


@dataclass(frozen=True)
class ChainRow:
    ticker: str
    underlying_conid: int | None
    fetched_at: datetime
    market_data_mode: str
    exchange: str | None
    trading_class: str | None
    multiplier: str | None
    expiry_date: str | date
    strike: float
    option_type: str
    currency: str | None
    source: str
    warnings: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    connection: ConnectionSettings = field(default_factory=ConnectionSettings)

    def to_record(self) -> dict[str, Any]:
        expiry = parse_date(self.expiry_date)
        payload = make_json_safe(asdict(self))
        payload["ticker"] = clean_string(self.ticker).upper()
        payload["fetched_at"] = isoformat_utc(self.fetched_at)
        payload["market_data_mode"] = normalize_market_data_mode(self.market_data_mode)
        payload["expiry_date"] = expiry.isoformat() if expiry else None
        payload["option_type"] = clean_string(self.option_type).lower()
        payload["warnings"] = list(self.warnings)
        payload["missing_fields"] = list(self.missing_fields)
        payload.pop("connection", None)
        return payload


@dataclass(frozen=True)
class ResolvedUnderlyingContract:
    conid: int | None
    symbol: str | None
    sec_type: str | None
    currency: str | None
    exchange: str | None
    primary_exchange: str | None
    local_symbol: str | None
    trading_class: str | None
    multiplier: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = make_json_safe(asdict(self))
        payload["symbol"] = clean_string(self.symbol).upper() or None
        payload["sec_type"] = clean_string(self.sec_type).upper() or None
        payload["currency"] = clean_string(self.currency).upper() or None
        payload["exchange"] = clean_string(self.exchange).upper() or None
        payload["primary_exchange"] = clean_string(self.primary_exchange).upper() or None
        payload["local_symbol"] = clean_string(self.local_symbol) or None
        payload["trading_class"] = clean_string(self.trading_class).upper() or None
        payload["multiplier"] = clean_string(self.multiplier) or None
        return payload


@dataclass(frozen=True)
class ChainDiscoveryDiagnostics:
    requested: dict[str, Any]
    resolved_underlying: ResolvedUnderlyingContract | None
    raw_option_parameter_rows: list[dict[str, Any]] = field(default_factory=list)
    raw_exchanges_seen: list[str] = field(default_factory=list)
    raw_trading_classes_seen: list[str] = field(default_factory=list)
    available_expiries: list[str] = field(default_factory=list)
    available_strike_count: int = 0
    available_strike_sample: list[float] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    selected_exchange: str | None = None
    failure_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = make_json_safe(asdict(self))
        if self.resolved_underlying is not None:
            payload["resolved_underlying"] = self.resolved_underlying.to_dict()
        payload["requested"] = make_json_safe(self.requested)
        return payload


@dataclass(frozen=True)
class ChainFetchResult:
    rows: list[ChainRow]
    diagnostics: ChainDiscoveryDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [row.to_record() for row in self.rows],
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class ContractMatchDiagnostics:
    requested_expiries: list[str] = field(default_factory=list)
    requested_right: str = "both"
    requested_min_strike: float | None = None
    requested_max_strike: float | None = None
    requested_strikes: list[float] = field(default_factory=list)
    around_spot: int | None = None
    max_contracts: int | None = None
    requested_expiry_exists: bool | None = None
    available_expiries: list[str] = field(default_factory=list)
    nearest_expiries: list[str] = field(default_factory=list)
    available_strike_count: int = 0
    available_strike_sample: list[float] = field(default_factory=list)
    spot_price_used: float | None = None
    around_spot_skipped_reason: str | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    selected_contracts: list[dict[str, Any]] = field(default_factory=list)
    failure_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return make_json_safe(asdict(self))


@dataclass(frozen=True)
class OptionQuoteSnapshot:
    ticker: str
    snapshot_timestamp: datetime
    market_data_mode: str
    market_data_type_code: int | None
    expiry_date: str | date
    strike: float
    option_type: str
    conid: int | None
    local_symbol: str | None
    trading_class: str | None
    exchange: str | None
    bid: float | None
    ask: float | None
    last: float | None
    mid: float | None
    mark: float | None
    close: float | None
    volume: float | None
    open_interest: float | None
    implied_volatility: float | None
    historical_volatility: float | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    option_price: float | None
    pv_dividend: float | None
    under_price: float | None
    source: str
    warnings: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    connection: ConnectionSettings = field(default_factory=ConnectionSettings)
    requested_contract: dict[str, Any] | None = None
    resolved_contract: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        expiry = parse_date(self.expiry_date)
        payload = make_json_safe(asdict(self))
        payload["ticker"] = clean_string(self.ticker).upper()
        payload["snapshot_timestamp"] = isoformat_utc(self.snapshot_timestamp)
        payload["market_data_mode"] = normalize_market_data_mode(self.market_data_mode)
        payload["expiry_date"] = expiry.isoformat() if expiry else None
        payload["option_type"] = clean_string(self.option_type).lower()
        payload["warnings"] = list(self.warnings)
        payload["missing_fields"] = list(self.missing_fields)
        payload.pop("connection", None)
        payload.pop("requested_contract", None)
        payload.pop("resolved_contract", None)
        return payload


@dataclass(frozen=True)
class OptionSnapshotDiagnostics:
    underlying_snapshot: dict[str, Any] | None = None
    resolved_underlying: ResolvedUnderlyingContract | None = None
    chain_diagnostics: ChainDiscoveryDiagnostics | None = None
    contract_match: ContractMatchDiagnostics = field(default_factory=ContractMatchDiagnostics)
    snapshot_scope: str = "filtered_slice"
    discovered_expiries: list[str] = field(default_factory=list)
    strike_count_by_expiry: dict[str, int] = field(default_factory=dict)
    attempted_contract_count: int = 0
    selected_contract_count: int = 0
    final_selected_expiries: list[str] = field(default_factory=list)
    final_selected_strikes: list[float] = field(default_factory=list)
    final_selected_exchanges: list[str] = field(default_factory=list)
    final_selected_trading_classes: list[str] = field(default_factory=list)
    delayed_field_summary: dict[str, Any] = field(default_factory=dict)
    failure_stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = make_json_safe(asdict(self))
        if self.resolved_underlying is not None:
            payload["resolved_underlying"] = self.resolved_underlying.to_dict()
        if self.chain_diagnostics is not None:
            payload["chain_diagnostics"] = self.chain_diagnostics.to_dict()
        return payload


@dataclass(frozen=True)
class OptionSnapshotFetchResult:
    quotes: list[OptionQuoteSnapshot]
    diagnostics: OptionSnapshotDiagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "quotes": [quote.to_record() for quote in self.quotes],
            "diagnostics": self.diagnostics.to_dict(),
        }


@dataclass(frozen=True)
class FullChainSnapshotFetchResult:
    underlying: UnderlyingQuoteSnapshot
    chain: ChainFetchResult
    option_snapshot: OptionSnapshotFetchResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying.to_record(),
            "chain": self.chain.to_dict(),
            "option_snapshot": self.option_snapshot.to_dict(),
        }
