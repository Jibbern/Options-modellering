from pathlib import Path
from datetime import date, timedelta

from options_lab.io import load_chain
from options_lab.strategies import PositionLeg, StrategyPosition, build_strategy


SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "GPRE"
    / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
)


def test_long_call_summary_metrics_are_correct():
    chain = load_chain(SAMPLE_FILE)
    strategy = build_strategy("long_call", chain)

    assert strategy.summary["primary_strike"] == 15.0
    assert round(strategy.summary["break_even"], 2) == 15.63
    assert round(strategy.summary["max_loss"], 2) == 63.0
    assert strategy.summary["max_gain"] == float("inf")


def test_bull_call_spread_and_cash_secured_put_metrics():
    chain = load_chain(SAMPLE_FILE)
    bull_spread = build_strategy("bull_call_spread", chain)
    short_put = build_strategy("cash_secured_put", chain)

    assert bull_spread.summary["long_strike"] == 15.0
    assert bull_spread.summary["short_strike"] == 16.0
    assert round(bull_spread.summary["break_even"], 2) == 15.33
    assert round(bull_spread.summary["max_gain"], 2) == 67.0

    assert short_put.summary["primary_strike"] == 15.0
    assert round(short_put.summary["break_even"], 2) == 14.60
    assert round(short_put.summary["max_gain"], 2) == 40.0


def _synthetic_long_call(*, expiry_date: date, strike: float = 20.0, base_iv: float = 0.55) -> StrategyPosition:
    return StrategyPosition(
        name="long_call",
        ticker="GPRE",
        snapshot_date=date(2026, 1, 1),
        entry_spot=15.0,
        premium_mode="mid",
        legs=[
            PositionLeg(
                asset_type="option",
                quantity=1,
                entry_price=1.0,
                option_type="call",
                strike=float(strike),
                expiry_date=expiry_date,
                base_iv=float(base_iv),
            )
        ],
        risk_free_rate=0.04,
        dividend_yield=0.0,
        summary={"initial_outlay": 100.0, "capital_required": 100.0},
    )


def test_long_call_mark_to_market_uses_remaining_time_to_expiry_for_theta_decay():
    position = _synthetic_long_call(expiry_date=date(2026, 7, 1))

    entry_value = float(position.mark_to_market_value([15.0], valuation_date=date(2026, 1, 1), iv_shift=0.0)[0])
    later_value = float(position.mark_to_market_value([15.0], valuation_date=date(2026, 3, 1), iv_shift=0.0)[0])

    assert later_value < entry_value


def test_shorter_expiry_decays_faster_than_longer_expiry_under_flat_stock_and_iv():
    short_call = _synthetic_long_call(expiry_date=date(2026, 4, 1))
    long_call = _synthetic_long_call(expiry_date=date(2027, 1, 1))
    valuation_date = date(2026, 1, 1)
    later_date = valuation_date + timedelta(days=45)

    short_decay = float(short_call.mark_to_market_value([15.0], valuation_date=valuation_date)[0]) - float(
        short_call.mark_to_market_value([15.0], valuation_date=later_date)[0]
    )
    long_decay = float(long_call.mark_to_market_value([15.0], valuation_date=valuation_date)[0]) - float(
        long_call.mark_to_market_value([15.0], valuation_date=later_date)[0]
    )

    assert short_decay > long_decay


def test_out_of_the_money_call_collapses_at_expiry():
    position = _synthetic_long_call(expiry_date=date(2026, 1, 11), strike=20.0)

    expiry_value = float(position.mark_to_market_value([15.0], valuation_date=date(2026, 1, 11), iv_shift=0.0)[0])

    assert expiry_value == 0.0
