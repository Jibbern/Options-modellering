# Historical Prices

This subsystem maintains the local historical daily-price store used by the options lab for spot resolution and simple historical reference work.

## Purpose

- keep a local daily OHLCV history for tickers such as `GPRE`
- resolve the best available close on or before an options snapshot date
- avoid network access during normal options analysis

## Primary Source

Primary unattended path:

- Nasdaq historical quotes JSON/XHR endpoint

Fallback path:

- manual CSV imports dropped into `data/<TICKER>/historical_prices/raw/manual/`

The local store is the stable interface. The exact downloader transport can change later without forcing the rest of the options lab to change.

## Local Store Layout

```text
data/
  GPRE/
    historical_prices/
      raw/
        manual/
      normalized/
      merged/
      metadata/
```

Important outputs:

- `raw/nasdaq_gpre_historical_YYYYMMDDTHHMMSSZ.json`
- `normalized/gpre_daily_prices.csv`
- `normalized/gpre_daily_prices.parquet` when a Parquet engine is available
- `merged/gpre_daily_prices_merged.csv`
- `metadata/download_manifest.json`
- `metadata/source_notes.json`

## Commands

Full refresh:

```powershell
..\.venv\Scripts\python.exe -m options_lab.prices.nasdaq_downloader --ticker GPRE --full-refresh
```

Selector check:

```powershell
..\.venv\Scripts\python.exe -m options_lab.prices.price_selector --ticker GPRE --snapshot-date 2026-04-12
```

## Spot Fallback Behavior

Spot resolution order in the options lab:

1. explicit `spot_price` override
2. sidecar or metadata-override `spot_price`
3. future explicit underlying field in the chain file, if added later
4. local historical-price selector result
5. near-money moneyness heuristic with a warning

If the snapshot date is a weekend, holiday, or otherwise missing, the selector uses the latest prior available trading day and records the matched date explicitly.

## Manual Import Workflow

If Nasdaq changes behavior or is temporarily unavailable:

1. export a historical CSV manually
2. drop it into `raw/manual/`
3. normalize it through `normalize_manual_price_file(...)`

The normalized canonical schema stays the same regardless of how the raw file arrived.
