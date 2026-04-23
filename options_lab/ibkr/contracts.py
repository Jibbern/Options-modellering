"""IBKR contract helpers built on the official socket API types."""

from __future__ import annotations

from datetime import date
from typing import Any

from ..utils import clean_string, parse_date
from .models import ResolvedUnderlyingContract


def _contract_class():
    try:
        from ibapi.contract import Contract
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ibapi is required for IBKR ingestion. Install it before running fetch-ibkr commands."
        ) from exc
    return Contract


def build_stock_contract(
    ticker: str,
    *,
    exchange: str = "SMART",
    currency: str = "USD",
    primary_exchange: str | None = None,
):
    contract = _contract_class()()
    contract.symbol = clean_string(ticker).upper()
    contract.secType = "STK"
    contract.exchange = clean_string(exchange).upper() or "SMART"
    contract.currency = clean_string(currency).upper() or "USD"
    if primary_exchange:
        contract.primaryExchange = clean_string(primary_exchange).upper()
    return contract


def build_option_contract(
    ticker: str,
    *,
    expiry_date: date | str,
    strike: float,
    option_type: str,
    exchange: str = "SMART",
    currency: str = "USD",
    trading_class: str | None = None,
    multiplier: str | None = None,
):
    expiry = parse_date(expiry_date)
    if expiry is None:
        raise ValueError(f"expiry_date must be a valid date, got {expiry_date!r}")
    right = clean_string(option_type).upper()
    if right in {"CALL", "C"}:
        right = "C"
    elif right in {"PUT", "P"}:
        right = "P"
    else:
        raise ValueError(f"Unsupported option_type for IBKR contract: {option_type!r}")
    contract = _contract_class()()
    contract.symbol = clean_string(ticker).upper()
    contract.secType = "OPT"
    contract.exchange = clean_string(exchange).upper() or "SMART"
    contract.currency = clean_string(currency).upper() or "USD"
    contract.lastTradeDateOrContractMonth = expiry.strftime("%Y%m%d")
    contract.strike = float(strike)
    contract.right = right
    if trading_class:
        contract.tradingClass = clean_string(trading_class).upper()
    if multiplier:
        contract.multiplier = str(multiplier)
    return contract


def contract_to_dict(contract: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for attribute in (
        "conId",
        "symbol",
        "secType",
        "exchange",
        "primaryExchange",
        "currency",
        "lastTradeDateOrContractMonth",
        "strike",
        "right",
        "localSymbol",
        "tradingClass",
        "multiplier",
    ):
        payload[attribute] = getattr(contract, attribute, None)
    return payload


def resolved_underlying_from_contract(contract: Any) -> ResolvedUnderlyingContract:
    return ResolvedUnderlyingContract(
        conid=getattr(contract, "conId", None),
        symbol=getattr(contract, "symbol", None),
        sec_type=getattr(contract, "secType", None),
        currency=getattr(contract, "currency", None),
        exchange=getattr(contract, "exchange", None),
        primary_exchange=getattr(contract, "primaryExchange", None),
        local_symbol=getattr(contract, "localSymbol", None),
        trading_class=getattr(contract, "tradingClass", None),
        multiplier=getattr(contract, "multiplier", None),
    )
