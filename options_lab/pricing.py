"""Lightweight Black-Scholes-style pricing helpers for scenario work."""

from __future__ import annotations

import math


def normal_cdf(value: float) -> float:
    """Pure-Python standard normal cumulative distribution function."""

    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    """Return intrinsic value for a call or put."""

    option = option_type.lower()
    if option == "call":
        return max(spot - strike, 0.0)
    if option == "put":
        return max(strike - spot, 0.0)
    raise ValueError(f"Unsupported option_type={option_type!r}")


def price_option(
    spot: float,
    strike: float,
    time_to_expiry: float,
    iv: float,
    risk_free_rate: float,
    dividend_yield: float,
    option_type: str,
) -> float:
    """Price a European-style option for practical before-expiry scenarios."""

    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")

    option = option_type.lower()
    if time_to_expiry <= 0 or iv <= 0:
        return intrinsic_value(spot, strike, option)

    sigma_sqrt_t = iv * math.sqrt(time_to_expiry)
    if sigma_sqrt_t == 0:
        return intrinsic_value(spot, strike, option)

    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * iv * iv) * time_to_expiry
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * math.exp(-risk_free_rate * time_to_expiry)

    if option == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    if option == "put":
        return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)
    raise ValueError(f"Unsupported option_type={option_type!r}")
