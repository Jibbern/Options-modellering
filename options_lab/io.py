"""Option-chain loading, normalization, and contract-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any
import warnings

import pandas as pd

from .metadata import SnapshotMetadata, build_metadata, infer_spot_from_moneyness
from .utils import clean_string, normalize_column_name, parse_date, parse_int, parse_number

COLUMN_ALIASES = {
    "strike": "strike",
    "moneyness": "moneyness",
    "bid": "bid",
    "mid": "mid",
    "ask": "ask",
    "latest": "last",
    "last": "last",
    "change": "change",
    "pct_change": "pct_change",
    "volume": "volume",
    "open_int": "open_interest",
    "open_interest": "open_interest",
    "oi_chg": "oi_change",
    "iv": "iv",
    "delta": "delta",
    "type": "option_type",
    "expiration": "expiry_date",
    "last_trade": "last_trade",
}

EXPECTED_COLUMNS = [
    "strike",
    "moneyness",
    "bid",
    "mid",
    "ask",
    "last",
    "change",
    "pct_change",
    "volume",
    "open_interest",
    "oi_change",
    "iv",
    "delta",
    "option_type",
    "last_trade",
]


@dataclass(frozen=True)
class OptionContract:
    """Normalized option contract record from a snapshot chain."""

    ticker: str
    snapshot_date: date
    expiry_date: date
    option_type: str
    strike: float
    bid: float | None
    mid: float | None
    ask: float | None
    last: float | None
    iv: float | None
    delta: float | None
    volume: int | None
    open_interest: int | None
    moneyness: float | None
    last_trade: date | None
    raw_row: dict[str, Any] = field(default_factory=dict, repr=False)

    def premium(self, mode: str = "mid", side: str | None = None) -> float:
        """Return the working premium for the contract under a quote policy."""

        candidates: list[float | None]
        quote_mode = mode.lower()
        trade_side = (side or "").lower()
        if quote_mode == "conservative":
            if trade_side == "buy":
                candidates = [self.ask, self.mid, self.last, self.bid]
            else:
                candidates = [self.bid, self.mid, self.last, self.ask]
        elif quote_mode in {"buy", "sell"}:
            return self.premium("conservative", side=quote_mode)
        elif quote_mode == "mid":
            candidates = [self.mid, self.last, self.ask, self.bid]
        elif quote_mode == "bid":
            candidates = [self.bid, self.mid, self.last, self.ask]
        elif quote_mode == "ask":
            candidates = [self.ask, self.mid, self.last, self.bid]
        elif quote_mode == "last":
            candidates = [self.last, self.mid, self.ask, self.bid]
        else:
            raise ValueError(f"Unsupported premium mode: {mode}")
        for candidate in candidates:
            if candidate is not None:
                return float(candidate)
        raise ValueError(f"No usable premium found for {self.option_type} {self.strike}")


@dataclass
class OptionChain:
    """Normalized option chain plus resolved snapshot metadata and warnings."""

    source_path: Path
    raw_data: pd.DataFrame
    contracts: pd.DataFrame
    metadata: SnapshotMetadata
    warnings: list[str] = field(default_factory=list)

    @property
    def ticker(self) -> str | None:
        return self.metadata.ticker

    @property
    def spot_price(self) -> float | None:
        return self.metadata.spot_price

    def calls(self) -> pd.DataFrame:
        return self.contracts[self.contracts["option_type"] == "call"].copy()

    def puts(self) -> pd.DataFrame:
        return self.contracts[self.contracts["option_type"] == "put"].copy()

    def filter_expiry(self, expiry: date | str | None) -> pd.DataFrame:
        if expiry is None:
            return self.contracts.copy()
        expiry_date = parse_date(expiry)
        return self.contracts[self.contracts["expiry_date"] == expiry_date].copy()

    def near_money(self, limit: int = 5, spot_price: float | None = None) -> pd.DataFrame:
        spot = spot_price or self.spot_price
        frame = self.contracts.copy()
        if spot is not None:
            frame["distance"] = (frame["strike"] - spot).abs()
        else:
            frame["distance"] = frame["moneyness"].abs().fillna(999.0)
        return frame.sort_values(["distance", "open_interest"], ascending=[True, False]).head(limit)

    def to_contract(self, row: pd.Series | dict[str, Any]) -> OptionContract:
        payload = dict(row)
        raw_payload = dict(payload.get("raw_row", {}) or {})
        raw_payload.update({key: value for key, value in payload.items() if key != "raw_row"})
        return OptionContract(
            ticker=payload["ticker"],
            snapshot_date=payload["snapshot_date"],
            expiry_date=payload["expiry_date"],
            option_type=payload["option_type"],
            strike=float(payload["strike"]),
            bid=payload.get("bid"),
            mid=payload.get("mid"),
            ask=payload.get("ask"),
            last=payload.get("last"),
            iv=payload.get("iv"),
            delta=payload.get("delta"),
            volume=payload.get("volume"),
            open_interest=payload.get("open_interest"),
            moneyness=payload.get("moneyness"),
            last_trade=payload.get("last_trade"),
            raw_row=raw_payload,
        )


def _rename_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    renamed = raw_df.copy()
    renamed.columns = [
        COLUMN_ALIASES.get(normalize_column_name(column), normalize_column_name(column))
        for column in raw_df.columns
    ]
    for column in EXPECTED_COLUMNS:
        if column not in renamed.columns:
            renamed[column] = None
    return renamed


def _normalize_contracts(raw_df: pd.DataFrame, metadata: SnapshotMetadata) -> pd.DataFrame:
    renamed = _rename_columns(raw_df)
    renamed = renamed.apply(lambda column: column.map(clean_string))
    renamed["raw_row"] = raw_df.to_dict(orient="records")

    # Retail exports often include footer rows, blank separator rows, or other
    # non-contract records. A valid option type and strike keeps the filter
    # practical without hardcoding one exact vendor layout.
    contract_mask = renamed["option_type"].str.lower().isin({"call", "put"})
    contract_mask &= renamed["strike"].apply(lambda value: parse_number(value) is not None)
    extra_columns = [
        column for column in renamed.columns if column not in EXPECTED_COLUMNS + ["raw_row"]
    ]
    contracts = renamed.loc[contract_mask, EXPECTED_COLUMNS + extra_columns + ["raw_row"]].copy()

    contracts["ticker"] = metadata.ticker
    contracts["snapshot_date"] = metadata.snapshot_date
    contracts["expiry_date"] = metadata.expiry_date
    contracts["option_type"] = contracts["option_type"].str.lower()
    contracts["strike"] = contracts["strike"].apply(parse_number).astype(float)
    contracts["bid"] = contracts["bid"].apply(parse_number)
    contracts["mid"] = contracts["mid"].apply(parse_number)
    contracts["ask"] = contracts["ask"].apply(parse_number)
    contracts["last"] = contracts["last"].apply(parse_number)
    contracts["change"] = contracts["change"].apply(lambda value: parse_number(value, unch_as_zero=True))
    contracts["pct_change"] = contracts["pct_change"].apply(
        lambda value: parse_number(value, percent=True, unch_as_zero=True)
    )
    contracts["volume"] = contracts["volume"].apply(parse_int)
    contracts["open_interest"] = contracts["open_interest"].apply(parse_int)
    contracts["oi_change"] = contracts["oi_change"].apply(lambda value: parse_int(value, unch_as_zero=True))
    contracts["iv"] = contracts["iv"].apply(lambda value: parse_number(value, percent=True))
    contracts["delta"] = contracts["delta"].apply(parse_number)
    contracts["moneyness"] = contracts["moneyness"].apply(lambda value: parse_number(value, percent=True))
    contracts["last_trade"] = contracts["last_trade"].apply(parse_date)
    contracts["quote_count"] = contracts[["bid", "mid", "ask", "last"]].notna().sum(axis=1)
    contracts["has_quote"] = contracts["quote_count"] > 0
    for column in [
        "underlying_price",
        "spread",
        "spread_pct_of_mid",
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
        "moneyness_decimal",
        "barchart_be_bid",
        "barchart_be_ask",
        "barchart_be_mid",
    ]:
        if column in contracts.columns:
            contracts[column] = contracts[column].apply(parse_number)
    if "model_eligible" in contracts.columns:
        contracts["model_eligible"] = contracts["model_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
    contracts = contracts.sort_values(["option_type", "strike"]).reset_index(drop=True)
    return contracts


def load_chain(
    path: str | Path,
    metadata_override: dict[str, Any] | SnapshotMetadata | None = None,
    spot_price: float | None = None,
    prices_data_root: str | Path | None = None,
    rates_data_root: str | Path | None = None,
    research_data_root: str | Path | None = None,
) -> OptionChain:
    """Load an option-chain CSV snapshot into the internal normalized schema.

    ``prices_data_root``, ``rates_data_root``, and ``research_data_root`` let
    callers point the loader at alternate local stores without changing the
    rest of the workflow.
    """

    source_path = Path(path)
    raw_data = pd.read_csv(
        source_path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8-sig",
    )
    metadata = build_metadata(
        source_path,
        metadata_override=metadata_override,
        spot_price=spot_price,
        prices_data_root=prices_data_root,
        rates_data_root=rates_data_root,
        research_data_root=research_data_root,
    )
    warnings_list: list[str] = []
    contracts = _normalize_contracts(raw_data, metadata)
    if metadata.spot_price is None:
        inferred_spot, message = infer_spot_from_moneyness(contracts)
        if inferred_spot is not None:
            warnings_list.append(message or "Spot price inferred from moneyness.")
            warnings.warn(message, stacklevel=2)
            metadata = replace(metadata, spot_price=round(inferred_spot, 4))
    if metadata.spot_price_note:
        warnings_list.append(metadata.spot_price_note)
    if metadata.snapshot_date is None and not contracts.empty:
        raise ValueError(
            "snapshot_date could not be inferred from the filename or metadata sidecar."
        )
    if metadata.expiry_date is None and not contracts.empty:
        raise ValueError(
            "expiry_date could not be inferred from the filename or metadata sidecar."
        )
    if metadata.risk_free_rate_note:
        warnings_list.append(metadata.risk_free_rate_note)
    contracts["ticker"] = metadata.ticker
    contracts["snapshot_date"] = metadata.snapshot_date
    contracts["expiry_date"] = metadata.expiry_date
    return OptionChain(
        source_path=source_path,
        raw_data=raw_data,
        contracts=contracts,
        metadata=metadata,
        warnings=warnings_list,
    )


def select_contract(
    chain: OptionChain,
    option_type: str,
    *,
    expiry: date | str | None = None,
    target_delta: float | None = None,
    target_strike: float | None = None,
    pct_otm: float | None = None,
) -> OptionContract:
    """Select one working contract using expiry, delta, strike, or OTM filters."""

    option = option_type.lower()
    frame = chain.filter_expiry(expiry)
    frame = frame[frame["option_type"] == option].copy()
    if frame.empty:
        raise ValueError(f"No {option} contracts available for the requested filters.")

    frame["abs_moneyness"] = frame["moneyness"].abs().fillna(999.0)
    frame["liquidity_rank"] = frame["open_interest"].fillna(0) + frame["volume"].fillna(0)
    frame = frame.sort_values(["has_quote", "liquidity_rank"], ascending=[False, False])

    spot = chain.spot_price
    if pct_otm is not None and spot is not None:
        pct = float(pct_otm)
        if option == "call":
            target_strike = spot * (1.0 + pct)
        else:
            target_strike = spot * (1.0 - pct)

    if target_strike is not None:
        frame["distance"] = (frame["strike"] - float(target_strike)).abs()
        chosen = frame.sort_values(
            ["distance", "has_quote", "liquidity_rank"],
            ascending=[True, False, False],
        ).iloc[0]
        return chain.to_contract(chosen)

    if target_delta is not None:
        target = abs(float(target_delta))
        frame = frame.dropna(subset=["delta"])
        if frame.empty:
            raise ValueError("No contracts with valid delta were available.")
        frame["distance"] = (frame["delta"].abs() - target).abs()
        chosen = frame.sort_values(
            ["distance", "has_quote", "liquidity_rank"],
            ascending=[True, False, False],
        ).iloc[0]
        return chain.to_contract(chosen)

    if spot is not None:
        frame["distance"] = (frame["strike"] - spot).abs()
        chosen = frame.sort_values(
            ["distance", "has_quote", "liquidity_rank"],
            ascending=[True, False, False],
        ).iloc[0]
        return chain.to_contract(chosen)

    chosen = frame.sort_values(
        ["abs_moneyness", "has_quote", "liquidity_rank"],
        ascending=[True, False, False],
    ).iloc[0]
    return chain.to_contract(chosen)
