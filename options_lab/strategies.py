"""Strategy construction and mark-to-market logic for the options lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from .io import OptionChain, OptionContract, select_contract
from .pricing import price_option
from .utils import CONTRACT_MULTIPLIER, finite_or_none, years_between


@dataclass(frozen=True)
class PositionLeg:
    """One stock or option leg inside a strategy position."""

    asset_type: str
    quantity: int
    entry_price: float
    option_type: str | None = None
    strike: float | None = None
    expiry_date: date | None = None
    base_iv: float | None = None
    delta: float | None = None
    label: str | None = None
    quote_metadata: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class StrategyPosition:
    """Constructed strategy with enough context for valuation and reporting."""

    name: str
    ticker: str
    snapshot_date: date
    entry_spot: float
    premium_mode: str
    legs: list[PositionLeg]
    risk_free_rate: float
    dividend_yield: float
    summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    resolved_metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    selection_inputs: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def option_legs(self) -> list[PositionLeg]:
        return [leg for leg in self.legs if leg.asset_type == "option"]

    @property
    def expiry_date(self) -> date | None:
        expiries = [leg.expiry_date for leg in self.option_legs if leg.expiry_date is not None]
        return max(expiries) if expiries else None

    @property
    def initial_outlay(self) -> float:
        return float(self.summary.get("initial_outlay", 0.0))

    @property
    def capital_required(self) -> float:
        return float(self.summary.get("capital_required", 0.0))

    def days_to_expiry(self) -> int:
        if self.expiry_date is None:
            return 0
        return max((self.expiry_date - self.snapshot_date).days, 0)

    def payoff_at_expiry(self, stock_prices) -> np.ndarray:
        """Return expiry profit/loss across a stock-price grid."""

        prices = np.asarray(stock_prices, dtype=float)
        total = np.zeros_like(prices, dtype=float)
        for leg in self.legs:
            if leg.asset_type == "stock":
                total += leg.quantity * (prices - leg.entry_price)
                continue
            intrinsic = np.where(
                leg.option_type == "call",
                np.maximum(prices - leg.strike, 0.0),
                np.maximum(leg.strike - prices, 0.0),
            )
            total += leg.quantity * CONTRACT_MULTIPLIER * (intrinsic - leg.entry_price)
        return total

    def mark_to_market_value(
        self,
        stock_prices,
        *,
        valuation_date: date | None = None,
        iv_shift: float = 0.0,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
    ) -> np.ndarray:
        """Estimate position value before expiry using the pricing engine."""

        prices = np.asarray(stock_prices, dtype=float)
        valuation = valuation_date or self.snapshot_date
        rate = self.risk_free_rate if risk_free_rate is None else risk_free_rate
        yield_rate = self.dividend_yield if dividend_yield is None else dividend_yield
        total = np.zeros_like(prices, dtype=float)
        for leg in self.legs:
            if leg.asset_type == "stock":
                total += leg.quantity * prices
                continue
            if leg.expiry_date is None:
                raise ValueError("Option leg is missing expiry_date.")
            time_to_expiry = years_between(valuation, leg.expiry_date)
            sigma = max((leg.base_iv or 0.0) + iv_shift, 1e-6)
            option_values = np.array(
                [
                    price_option(
                        spot=float(spot),
                        strike=float(leg.strike),
                        time_to_expiry=time_to_expiry,
                        iv=sigma,
                        risk_free_rate=rate,
                        dividend_yield=yield_rate,
                        option_type=str(leg.option_type),
                    )
                    for spot in prices
                ]
            )
            total += leg.quantity * CONTRACT_MULTIPLIER * option_values
        return total

    def profit_before_expiry(
        self,
        stock_prices,
        *,
        valuation_date: date | None = None,
        iv_shift: float = 0.0,
        risk_free_rate: float | None = None,
        dividend_yield: float | None = None,
    ) -> np.ndarray:
        """Return mark-to-market profit/loss before expiry."""

        return self.mark_to_market_value(
            stock_prices,
            valuation_date=valuation_date,
            iv_shift=iv_shift,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        ) - self.initial_outlay

    def valuation_date_for_horizon(self, horizon_days: int, *, clamp: bool = True) -> tuple[date, bool]:
        """Return a valuation date for a requested horizon, clamping to expiry when needed."""

        requested = self.snapshot_date + timedelta(days=max(horizon_days, 0))
        expiry_date = self.expiry_date
        if not clamp or expiry_date is None or requested <= expiry_date:
            return requested, False
        return expiry_date, True

    def summary_metrics(self) -> dict[str, Any]:
        """Return summary metrics plus the resolved pricing context used."""

        payload = dict(self.summary)
        research_context = self.resolved_metadata.get("research_context", {}) or {}
        expected_move = research_context.get("expected_move", {}) if isinstance(research_context, dict) else {}
        options_overview = research_context.get("options_overview", {}) if isinstance(research_context, dict) else {}
        nearest_event = research_context.get("nearest_event", {}) if isinstance(research_context, dict) else {}
        dividend_assumption = research_context.get("dividend_assumption", {}) if isinstance(research_context, dict) else {}
        payload["strategy"] = self.name
        payload["ticker"] = self.ticker
        payload["snapshot_date"] = self.snapshot_date.isoformat()
        payload["expiry_date"] = self.expiry_date.isoformat() if self.expiry_date else None
        payload["spot_entry"] = self.entry_spot
        payload["premium_mode"] = self.premium_mode
        payload["spot_price_source"] = self.resolved_metadata.get("spot_price_source")
        payload["spot_price_matched_date"] = self.resolved_metadata.get("spot_price_matched_date")
        payload["risk_free_rate"] = self.risk_free_rate
        payload["risk_free_rate_source"] = self.resolved_metadata.get("risk_free_rate_source")
        payload["risk_free_rate_series"] = self.resolved_metadata.get("risk_free_rate_series")
        payload["risk_free_rate_matched_date"] = self.resolved_metadata.get("risk_free_rate_matched_date")
        payload["dividend_yield"] = self.dividend_yield
        payload["dividend_yield_source"] = dividend_assumption.get("source") if isinstance(dividend_assumption, dict) else None
        payload["expected_move_abs"] = expected_move.get("expected_move_abs") if isinstance(expected_move, dict) else None
        payload["expected_move_pct"] = expected_move.get("expected_move_pct") if isinstance(expected_move, dict) else None
        payload["iv_rank"] = options_overview.get("iv_rank") if isinstance(options_overview, dict) else None
        payload["iv_percentile"] = options_overview.get("iv_percentile") if isinstance(options_overview, dict) else None
        payload["nearest_event_date"] = nearest_event.get("event_date") if isinstance(nearest_event, dict) else None
        payload["nearest_event_type"] = nearest_event.get("event_type") if isinstance(nearest_event, dict) else None
        payload["event_before_expiry"] = nearest_event.get("occurs_before_expiry") if isinstance(nearest_event, dict) else None
        return payload

    def report_metadata(self) -> dict[str, Any]:
        """Return reproducibility metadata for report artifacts."""

        def _serialize_leg(leg: PositionLeg) -> dict[str, Any]:
            return {
                "asset_type": leg.asset_type,
                "quantity": leg.quantity,
                "entry_price": leg.entry_price,
                "option_type": leg.option_type,
                "strike": leg.strike,
                "expiry_date": leg.expiry_date.isoformat() if leg.expiry_date else None,
                "base_iv": leg.base_iv,
                "delta": leg.delta,
                "label": leg.label,
                "quote_metadata": dict(leg.quote_metadata),
            }

        return {
            "strategy": self.name,
            "ticker": self.ticker,
            "snapshot_date": self.snapshot_date.isoformat(),
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "entry_spot": self.entry_spot,
            "premium_mode": self.premium_mode,
            "risk_free_rate": self.risk_free_rate,
            "dividend_yield": self.dividend_yield,
            "source_snapshot_file": self.resolved_metadata.get("source_snapshot_file"),
            "summary": self.summary_metrics(),
            "resolved_metadata": dict(self.resolved_metadata),
            "selection_inputs": dict(self.selection_inputs),
            "warnings": list(self.warnings),
            "notes": list(self.notes),
            "legs": [_serialize_leg(leg) for leg in self.legs],
        }


def _resolve_premium(contract: OptionContract, premium_mode: str, side: str) -> float:
    return contract.premium(mode=premium_mode, side=side)


def _quote_metadata(contract: OptionContract) -> dict[str, Any]:
    raw = dict(contract.raw_row or {})
    payload = {
        "bid": contract.bid,
        "ask": contract.ask,
        "mid": contract.mid,
        "last": contract.last,
        "implied_volatility": contract.iv,
        "delta": contract.delta,
        "volume": contract.volume,
        "open_interest": contract.open_interest,
    }
    for key in [
        "source",
        "trust",
        "entry_price_mode",
        "spread",
        "spread_pct_of_mid",
        "quality_flags",
        "liquidity_bucket",
        "model_eligible",
        "entry_premium_mid",
        "entry_premium_ask",
        "entry_premium_realistic",
        "entry_premium_selected",
        "exit_premium_conservative",
        "iv_rank",
        "iv_percentile",
        "gamma",
        "theta",
        "vega",
        "rho",
        "itm_probability",
        "otm_probability",
        "profit_probability",
        "contract_label",
        "contract_symbol",
    ]:
        if key in raw:
            payload[key] = raw.get(key)
    return payload


def _next_contract_by_strike(
    chain: OptionChain,
    base_contract: OptionContract,
    *,
    direction: str,
) -> OptionContract:
    frame = chain.filter_expiry(base_contract.expiry_date)
    frame = frame[frame["option_type"] == base_contract.option_type]
    if direction == "higher":
        frame = frame[frame["strike"] > base_contract.strike].sort_values("strike")
    else:
        frame = frame[frame["strike"] < base_contract.strike].sort_values("strike", ascending=False)
    if frame.empty:
        raise ValueError(f"No {direction} strike found for {base_contract.option_type} spread leg.")
    return chain.to_contract(frame.iloc[0])


def _ensure_contract(
    chain: OptionChain,
    option_type: str,
    *,
    contract: OptionContract | None = None,
    selector: dict[str, Any] | None = None,
) -> OptionContract:
    if contract is not None:
        return contract
    return select_contract(chain, option_type, **(selector or {}))


def build_strategy(
    name: str,
    chain: OptionChain,
    *,
    spot_price: float | None = None,
    premium_mode: str = "mid",
    contract: OptionContract | None = None,
    long_contract: OptionContract | None = None,
    short_contract: OptionContract | None = None,
    contract_selector: dict[str, Any] | None = None,
    long_selector: dict[str, Any] | None = None,
    short_selector: dict[str, Any] | None = None,
) -> StrategyPosition:
    """Build one supported strategy from a normalized option chain.

    The returned position carries both valuation-ready legs and the resolved
    metadata context that should travel into reports for reproducibility.
    """

    strategy_name = name.lower()
    spot = spot_price or chain.spot_price
    if spot is None:
        raise ValueError("spot_price is required for strategy construction.")
    ticker = chain.ticker or "UNKNOWN"
    snapshot_date = chain.metadata.snapshot_date
    if snapshot_date is None:
        raise ValueError("snapshot_date is missing from the chain metadata.")
    rate = chain.metadata.risk_free_rate
    yield_rate = chain.metadata.dividend_yield
    notes: list[str] = []
    resolved_metadata = chain.metadata.to_dict()
    resolved_metadata["source_snapshot_file"] = str(chain.source_path)
    selection_inputs = {
        "spot_price_override": spot_price,
        "premium_mode": premium_mode,
        "contract_selector": contract_selector or {},
        "long_selector": long_selector or {},
        "short_selector": short_selector or {},
    }

    def _position(legs: list[PositionLeg], summary: dict[str, Any]) -> StrategyPosition:
        return StrategyPosition(
            strategy_name,
            ticker,
            snapshot_date,
            spot,
            premium_mode,
            legs,
            rate,
            yield_rate,
            summary,
            notes=list(notes),
            warnings=list(chain.warnings),
            resolved_metadata=resolved_metadata,
            selection_inputs=selection_inputs,
        )

    if strategy_name == "long_stock":
        legs = [PositionLeg(asset_type="stock", quantity=CONTRACT_MULTIPLIER, entry_price=spot, label="Long 100 shares")]
        summary = {
            "premium_paid": 0.0,
            "premium_received": 0.0,
            "net_premium": 0.0,
            "initial_outlay": spot * CONTRACT_MULTIPLIER,
            "capital_required": spot * CONTRACT_MULTIPLIER,
            "break_even": spot,
            "max_gain": float("inf"),
            "max_loss": spot * CONTRACT_MULTIPLIER,
            "return_on_premium_max": None,
            "return_on_capital_max": None,
        }
        return _position(legs, summary)

    if strategy_name == "long_call":
        contract = _ensure_contract(chain, "call", contract=contract, selector=contract_selector)
        premium = _resolve_premium(contract, premium_mode, "buy")
        legs = [
            PositionLeg(
                asset_type="option",
                quantity=1,
                entry_price=premium,
                option_type="call",
                strike=contract.strike,
                expiry_date=contract.expiry_date,
                base_iv=contract.iv,
                delta=contract.delta,
                label=f"Long call {contract.strike:g}",
                quote_metadata=_quote_metadata(contract),
            )
        ]
        max_loss = premium * CONTRACT_MULTIPLIER
        summary = {
            "premium_paid": max_loss,
            "premium_received": 0.0,
            "net_premium": -max_loss,
            "initial_outlay": max_loss,
            "capital_required": max_loss,
            "break_even": contract.strike + premium,
            "max_gain": float("inf"),
            "max_loss": max_loss,
            "return_on_premium_max": None,
            "return_on_capital_max": None,
            "primary_strike": contract.strike,
        }
        return _position(legs, summary)

    if strategy_name == "long_put":
        contract = _ensure_contract(chain, "put", contract=contract, selector=contract_selector)
        premium = _resolve_premium(contract, premium_mode, "buy")
        max_loss = premium * CONTRACT_MULTIPLIER
        max_gain = max((contract.strike - premium) * CONTRACT_MULTIPLIER, 0.0)
        legs = [
            PositionLeg(
                asset_type="option",
                quantity=1,
                entry_price=premium,
                option_type="put",
                strike=contract.strike,
                expiry_date=contract.expiry_date,
                base_iv=contract.iv,
                delta=contract.delta,
                label=f"Long put {contract.strike:g}",
                quote_metadata=_quote_metadata(contract),
            )
        ]
        summary = {
            "premium_paid": max_loss,
            "premium_received": 0.0,
            "net_premium": -max_loss,
            "initial_outlay": max_loss,
            "capital_required": max_loss,
            "break_even": contract.strike - premium,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "return_on_premium_max": finite_or_none(max_gain / max_loss if max_loss else None),
            "return_on_capital_max": finite_or_none(max_gain / max_loss if max_loss else None),
            "primary_strike": contract.strike,
        }
        return _position(legs, summary)

    if strategy_name == "covered_call":
        short_call = _ensure_contract(chain, "call", contract=contract or short_contract, selector=contract_selector or short_selector)
        premium_received = _resolve_premium(short_call, premium_mode, "sell")
        capital = spot * CONTRACT_MULTIPLIER - premium_received * CONTRACT_MULTIPLIER
        max_gain = (short_call.strike - spot + premium_received) * CONTRACT_MULTIPLIER
        max_loss = capital
        legs = [
            PositionLeg(asset_type="stock", quantity=CONTRACT_MULTIPLIER, entry_price=spot, label="Long 100 shares"),
            PositionLeg(
                asset_type="option",
                quantity=-1,
                entry_price=premium_received,
                option_type="call",
                strike=short_call.strike,
                expiry_date=short_call.expiry_date,
                base_iv=short_call.iv,
                delta=short_call.delta,
                label=f"Short call {short_call.strike:g}",
                quote_metadata=_quote_metadata(short_call),
            ),
        ]
        summary = {
            "premium_paid": 0.0,
            "premium_received": premium_received * CONTRACT_MULTIPLIER,
            "net_premium": premium_received * CONTRACT_MULTIPLIER,
            "initial_outlay": capital,
            "capital_required": capital,
            "break_even": spot - premium_received,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "return_on_premium_max": finite_or_none(max_gain / (premium_received * CONTRACT_MULTIPLIER) if premium_received else None),
            "return_on_capital_max": finite_or_none(max_gain / capital if capital else None),
            "primary_strike": short_call.strike,
        }
        return _position(legs, summary)

    if strategy_name == "cash_secured_put":
        short_put = _ensure_contract(chain, "put", contract=contract or short_contract, selector=contract_selector or short_selector)
        premium_received = _resolve_premium(short_put, premium_mode, "sell")
        collateral = short_put.strike * CONTRACT_MULTIPLIER
        capital = collateral - premium_received * CONTRACT_MULTIPLIER
        max_gain = premium_received * CONTRACT_MULTIPLIER
        max_loss = capital
        legs = [
            PositionLeg(
                asset_type="option",
                quantity=-1,
                entry_price=premium_received,
                option_type="put",
                strike=short_put.strike,
                expiry_date=short_put.expiry_date,
                base_iv=short_put.iv,
                delta=short_put.delta,
                label=f"Short put {short_put.strike:g}",
                quote_metadata=_quote_metadata(short_put),
            )
        ]
        summary = {
            "premium_paid": 0.0,
            "premium_received": max_gain,
            "net_premium": max_gain,
            "initial_outlay": -max_gain,
            "capital_required": collateral,
            "break_even": short_put.strike - premium_received,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "return_on_premium_max": finite_or_none(1.0 if max_gain else None),
            "return_on_capital_max": finite_or_none(max_gain / collateral if collateral else None),
            "primary_strike": short_put.strike,
        }
        return _position(legs, summary)

    if strategy_name == "bull_call_spread":
        long_leg = _ensure_contract(chain, "call", contract=long_contract, selector=long_selector or contract_selector)
        if short_contract is not None or short_selector:
            short_leg = _ensure_contract(chain, "call", contract=short_contract, selector=short_selector)
        else:
            short_leg = _next_contract_by_strike(chain, long_leg, direction="higher")
        if short_leg.strike <= long_leg.strike:
            raise ValueError("Bull call spread requires the short strike to be above the long strike.")
        long_premium = _resolve_premium(long_leg, premium_mode, "buy")
        short_premium = _resolve_premium(short_leg, premium_mode, "sell")
        net_debit = long_premium - short_premium
        width = short_leg.strike - long_leg.strike
        max_loss = net_debit * CONTRACT_MULTIPLIER
        max_gain = (width - net_debit) * CONTRACT_MULTIPLIER
        legs = [
            PositionLeg(
                asset_type="option",
                quantity=1,
                entry_price=long_premium,
                option_type="call",
                strike=long_leg.strike,
                expiry_date=long_leg.expiry_date,
                base_iv=long_leg.iv,
                delta=long_leg.delta,
                label=f"Long call {long_leg.strike:g}",
                quote_metadata=_quote_metadata(long_leg),
            ),
            PositionLeg(
                asset_type="option",
                quantity=-1,
                entry_price=short_premium,
                option_type="call",
                strike=short_leg.strike,
                expiry_date=short_leg.expiry_date,
                base_iv=short_leg.iv,
                delta=short_leg.delta,
                label=f"Short call {short_leg.strike:g}",
                quote_metadata=_quote_metadata(short_leg),
            ),
        ]
        summary = {
            "premium_paid": long_premium * CONTRACT_MULTIPLIER,
            "premium_received": short_premium * CONTRACT_MULTIPLIER,
            "net_premium": -max_loss,
            "initial_outlay": max_loss,
            "capital_required": max_loss,
            "break_even": long_leg.strike + net_debit,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "return_on_premium_max": finite_or_none(max_gain / max_loss if max_loss else None),
            "return_on_capital_max": finite_or_none(max_gain / max_loss if max_loss else None),
            "long_strike": long_leg.strike,
            "short_strike": short_leg.strike,
        }
        return _position(legs, summary)

    if strategy_name == "bear_put_spread":
        long_leg = _ensure_contract(chain, "put", contract=long_contract, selector=long_selector or contract_selector)
        if short_contract is not None or short_selector:
            short_leg = _ensure_contract(chain, "put", contract=short_contract, selector=short_selector)
        else:
            short_leg = _next_contract_by_strike(chain, long_leg, direction="lower")
        if short_leg.strike >= long_leg.strike:
            raise ValueError("Bear put spread requires the short strike to be below the long strike.")
        long_premium = _resolve_premium(long_leg, premium_mode, "buy")
        short_premium = _resolve_premium(short_leg, premium_mode, "sell")
        net_debit = long_premium - short_premium
        width = long_leg.strike - short_leg.strike
        max_loss = net_debit * CONTRACT_MULTIPLIER
        max_gain = (width - net_debit) * CONTRACT_MULTIPLIER
        legs = [
            PositionLeg(
                asset_type="option",
                quantity=1,
                entry_price=long_premium,
                option_type="put",
                strike=long_leg.strike,
                expiry_date=long_leg.expiry_date,
                base_iv=long_leg.iv,
                delta=long_leg.delta,
                label=f"Long put {long_leg.strike:g}",
                quote_metadata=_quote_metadata(long_leg),
            ),
            PositionLeg(
                asset_type="option",
                quantity=-1,
                entry_price=short_premium,
                option_type="put",
                strike=short_leg.strike,
                expiry_date=short_leg.expiry_date,
                base_iv=short_leg.iv,
                delta=short_leg.delta,
                label=f"Short put {short_leg.strike:g}",
                quote_metadata=_quote_metadata(short_leg),
            ),
        ]
        summary = {
            "premium_paid": long_premium * CONTRACT_MULTIPLIER,
            "premium_received": short_premium * CONTRACT_MULTIPLIER,
            "net_premium": -max_loss,
            "initial_outlay": max_loss,
            "capital_required": max_loss,
            "break_even": long_leg.strike - net_debit,
            "max_gain": max_gain,
            "max_loss": max_loss,
            "return_on_premium_max": finite_or_none(max_gain / max_loss if max_loss else None),
            "return_on_capital_max": finite_or_none(max_gain / max_loss if max_loss else None),
            "long_strike": long_leg.strike,
            "short_strike": short_leg.strike,
        }
        return _position(legs, summary)

    raise ValueError(f"Unsupported strategy: {name}")
