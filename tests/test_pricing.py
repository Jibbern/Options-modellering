from options_lab.pricing import intrinsic_value, price_option


def test_intrinsic_value_and_zero_time_pricing():
    assert intrinsic_value(20.0, 15.0, "call") == 5.0
    assert intrinsic_value(20.0, 15.0, "put") == 0.0
    assert price_option(20.0, 15.0, 0.0, 0.4, 0.04, 0.0, "call") == 5.0


def test_black_scholes_sanity_checks():
    at_the_money_call = price_option(100.0, 100.0, 0.5, 0.30, 0.04, 0.0, "call")
    higher_spot_call = price_option(110.0, 100.0, 0.5, 0.30, 0.04, 0.0, "call")
    higher_iv_call = price_option(100.0, 100.0, 0.5, 0.50, 0.04, 0.0, "call")
    at_the_money_put = price_option(100.0, 100.0, 0.5, 0.30, 0.04, 0.0, "put")

    assert at_the_money_call > 0
    assert higher_spot_call > at_the_money_call
    assert higher_iv_call > at_the_money_call
    assert at_the_money_put > 0
