from datetime import date, datetime

from options_lab.utils import parse_date, parse_number


def test_parse_date_normalizes_datetime_to_plain_date():
    parsed = parse_date(datetime(2026, 4, 22, 15, 30))

    assert parsed == date(2026, 4, 22)
    assert type(parsed) is date


def test_parse_date_accepts_common_local_formats():
    assert parse_date("2026-04-22") == date(2026, 4, 22)
    assert parse_date("20260422") == date(2026, 4, 22)
    assert parse_date("04/22/26") == date(2026, 4, 22)


def test_parse_number_handles_percent_and_empty_tokens():
    assert parse_number("12.5%") == 0.125
    assert parse_number("n/a") is None
