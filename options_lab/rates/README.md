# FRED Risk-Free Rates

This subsystem maintains the local nominal U.S. Treasury rate store used by the options lab.

## Purpose

- download the configured Treasury constant-maturity series from the official FRED API
- normalize and merge them into a simple local store
- provide a horizon-aware selector for options modeling
- keep analysis runs local-only after the store has been refreshed

## Required Environment Variable

Set the API key before running the downloader:

```powershell
$env:FRED_API_KEY = "your_fred_api_key"
```

If `FRED_API_KEY` is missing, the downloader fails clearly and does not fabricate data.

## Configured Series

- `DGS1MO` = 1-Month Treasury constant maturity
- `DGS3MO` = 3-Month Treasury constant maturity
- `DGS6MO` = 6-Month Treasury constant maturity
- `DGS1` = 1-Year Treasury constant maturity

## Local Store Layout

```text
data/
  risk_free/
    fred/
      raw/
      normalized/
      merged/
      metadata/
```

Important outputs:

- `raw/DGS1MO_YYYYMMDDTHHMMSSZ.json`
- `normalized/DGS1MO.csv`
- `normalized/DGS1MO.parquet` when a Parquet engine is available
- `merged/fred_treasury_constant_maturity_daily.csv`
- `merged/current_risk_free_snapshot.csv`
- `metadata/download_manifest.json`
- `metadata/latest_rates.json`

CSV and JSON are the guaranteed baseline. Parquet is optional.

## Selector Bucketing

- `<= 45 days` -> `DGS1MO`
- `46-135 days` -> `DGS3MO`
- `136-270 days` -> `DGS6MO`
- `> 270 days` -> `DGS1`

If the exact snapshot date is missing because of weekends or holidays, the selector uses the latest prior available observation and records the matched date.

## Commands

Refresh all series:

```powershell
..\.venv\Scripts\python.exe -m options_lab.rates.fred_downloader --full-refresh
```

Refresh one series:

```powershell
..\.venv\Scripts\python.exe -m options_lab.rates.fred_downloader --series DGS3MO
```

Run a selector check:

```powershell
..\.venv\Scripts\python.exe -m options_lab.rates.rate_selector `
  --snapshot-date 2026-04-12 `
  --expiry-date 2026-04-17
```

## How It Integrates

During normal options analysis:

1. explicit metadata override or sidecar `risk_free_rate`
2. local FRED selector result
3. default fallback `0.04`

The fallback is intentional and visible. If the local store is missing, the options lab does not pretend that live rates were available.
