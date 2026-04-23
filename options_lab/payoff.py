"""Payoff helpers for common option and stock structures at expiry."""

from __future__ import annotations

import numpy as np

from .pricing import intrinsic_value
from .utils import CONTRACT_MULTIPLIER


def _as_array(stock_prices) -> np.ndarray:
    return np.asarray(stock_prices, dtype=float)


def long_stock_profit(stock_prices, entry_spot: float, shares: int = CONTRACT_MULTIPLIER) -> np.ndarray:
    prices = _as_array(stock_prices)
    return (prices - entry_spot) * shares


def long_call_profit(
    stock_prices,
    strike: float,
    premium: float,
    *,
    contracts: int = 1,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    intrinsic = np.maximum(prices - strike, 0.0)
    return (intrinsic - premium) * contracts * multiplier


def long_put_profit(
    stock_prices,
    strike: float,
    premium: float,
    *,
    contracts: int = 1,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    intrinsic = np.maximum(strike - prices, 0.0)
    return (intrinsic - premium) * contracts * multiplier


def covered_call_profit(
    stock_prices,
    entry_spot: float,
    strike: float,
    premium_received: float,
    *,
    shares: int = CONTRACT_MULTIPLIER,
    contracts: int = 1,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    stock_leg = (prices - entry_spot) * shares
    short_call = -(np.maximum(prices - strike, 0.0) - premium_received) * contracts * multiplier
    return stock_leg + short_call


def cash_secured_put_profit(
    stock_prices,
    strike: float,
    premium_received: float,
    *,
    contracts: int = 1,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    intrinsic = np.maximum(strike - prices, 0.0)
    return (premium_received - intrinsic) * contracts * multiplier


def bull_call_spread_profit(
    stock_prices,
    lower_strike: float,
    upper_strike: float,
    net_premium: float,
    *,
    contracts: int = 1,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    spread_intrinsic = np.maximum(prices - lower_strike, 0.0) - np.maximum(prices - upper_strike, 0.0)
    return (spread_intrinsic - net_premium) * contracts * multiplier


def bear_put_spread_profit(
    stock_prices,
    long_strike: float,
    short_strike: float,
    net_premium: float,
    *,
    contracts: int = 1,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    spread_intrinsic = np.maximum(long_strike - prices, 0.0) - np.maximum(short_strike - prices, 0.0)
    return (spread_intrinsic - net_premium) * contracts * multiplier


def option_leg_profit(
    stock_prices,
    *,
    strike: float,
    premium: float,
    option_type: str,
    quantity: int,
    multiplier: int = CONTRACT_MULTIPLIER,
) -> np.ndarray:
    prices = _as_array(stock_prices)
    intrinsic = np.vectorize(lambda s: intrinsic_value(float(s), strike, option_type))(prices)
    return quantity * multiplier * (intrinsic - premium)
