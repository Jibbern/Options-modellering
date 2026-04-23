from pathlib import Path

from options_lab.io import load_chain, select_contract


SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "GPRE"
    / "gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_chain_normalizes_sample_data(temp_data_root: Path):
    chain = load_chain(SAMPLE_FILE, prices_data_root=temp_data_root)

    assert chain.metadata.ticker == "GPRE"
    assert chain.metadata.snapshot_date.isoformat() == "2026-04-12"
    assert chain.metadata.expiry_date.isoformat() == "2026-04-17"
    assert len(chain.contracts) == 34
    assert set(chain.contracts["option_type"]) == {"call", "put"}
    assert 15.0 < chain.spot_price < 15.3


def test_loader_handles_percent_strings_unch_and_thousands():
    chain = load_chain(SAMPLE_FILE)

    fifteen_call = chain.contracts[
        (chain.contracts["option_type"] == "call") & (chain.contracts["strike"] == 15.0)
    ].iloc[0]
    sixteen_call = chain.contracts[
        (chain.contracts["option_type"] == "call") & (chain.contracts["strike"] == 16.0)
    ].iloc[0]

    assert abs(fifteen_call["iv"] - 0.5361) < 1e-4
    assert fifteen_call["change"] == 0.54
    assert fifteen_call["pct_change"] == 0.0
    assert sixteen_call["pct_change"] == -0.40
    assert sixteen_call["open_interest"] == 1136


def test_select_contract_helpers_cover_delta_and_otm():
    chain = load_chain(SAMPLE_FILE)

    delta_call = select_contract(chain, "call", target_delta=0.25)
    otm_put = select_contract(chain, "put", pct_otm=0.05)

    assert delta_call.strike == 16.0
    assert otm_put.option_type == "put"
    assert otm_put.strike in {14.0, 15.0}
