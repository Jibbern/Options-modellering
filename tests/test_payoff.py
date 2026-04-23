import numpy as np

from options_lab.payoff import (
    bear_put_spread_profit,
    bull_call_spread_profit,
    cash_secured_put_profit,
    covered_call_profit,
    long_call_profit,
    long_put_profit,
    long_stock_profit,
)


def test_long_stock_profit_math():
    prices = np.array([10.0, 15.0, 20.0])
    result = long_stock_profit(prices, entry_spot=15.0)
    assert result.tolist() == [-500.0, 0.0, 500.0]


def test_option_and_spread_payoffs():
    prices = np.array([10.0, 15.0, 20.0])

    assert long_call_profit(prices, strike=15.0, premium=1.0).tolist() == [-100.0, -100.0, 400.0]
    assert long_put_profit(prices, strike=15.0, premium=1.0).tolist() == [400.0, -100.0, -100.0]
    assert covered_call_profit(prices, entry_spot=15.0, strike=17.0, premium_received=1.0).tolist() == [-400.0, 100.0, 300.0]
    assert cash_secured_put_profit(prices, strike=15.0, premium_received=1.0).tolist() == [-400.0, 100.0, 100.0]
    assert bull_call_spread_profit(prices, lower_strike=15.0, upper_strike=17.0, net_premium=0.5).tolist() == [-50.0, -50.0, 150.0]
    assert bear_put_spread_profit(prices, long_strike=17.0, short_strike=15.0, net_premium=0.5).tolist() == [150.0, 150.0, -50.0]
