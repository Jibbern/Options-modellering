# Data Sources

Options Lab stays local-first.

## Canonical data root

`data/` holds source data and normalized local stores.

Typical contents:

- option-chain CSV snapshots
- historical prices
- manually downloaded Barchart Options Screener exports
- rates data
- research metadata
- IBKR delayed snapshots and manifests

## Snapshot discovery

Local chain discovery is handled through `options_lab.snapshots` and `options_lab.research_metadata.catalog`.

Preferred order for chain snapshots:

1. same-date full quoted IBKR slices under `data/<TICKER>/ibkr/snapshots/option_quotes/`, but only when that expiry slice is quote-usable
2. same-date manually downloaded Barchart Options Screener slices under `data/<TICKER>/options/barchart/normalized/`
3. manual `data/<TICKER>/option_chains/` fallback slices
4. legacy flat files directly under `data/<TICKER>/`

IBKR chain-universe metadata under `data/<TICKER>/ibkr/chains/` remains useful provenance, but it does not outrank a usable quoted slice.

The current quote-usability gate for same-day full quoted IBKR precedence is 20% usable-quote coverage per expiry slice. If the same-day IBKR slice is discovered but too sparse, analysis keeps that fact in provenance and falls back to the best local quoted slice instead of silently pretending the sparse file was authoritative.

## Barchart manual CSV imports

Barchart support is local CSV import only:

- no Barchart API credentials
- no Barchart for Excel dependency
- no scraping
- no mutation of the loose source export

Use the Options Screener custom view with these columns when possible:

`Symbol`, `Type`, `Price~`, `Latest`, `Exp Date`, `DTE`, `Strike`, `Moneyness`, `Bid`, `Ask`, `Mid`, `Volume`, `Open Int`, `IV`, `IV Rank`, `IV Pctl`, `Delta`, `Gamma`, `Theta`, `Vega`, `Rho`, `ITM Prob`, `OTM Prob`, `Profit Prob`, `BE (Bid)`, `Links`.

Then import the downloaded local file:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli import-barchart-options `
  --ticker GPRE `
  --csv ".\options-screener-GPRE_2026-05-05.csv" `
  --snapshot-date 2026-05-05 `
  --entry-mode mid
```

PBI uses the same importer:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli import-barchart-options `
  --ticker PBI `
  --csv ".\options-screener-PBI_2026-05-05.csv" `
  --snapshot-date 2026-05-05 `
  --entry-mode mid
```

The importer copies the raw CSV into `data/<TICKER>/options/barchart/raw/`, writes normalized per-expiry slices into `data/<TICKER>/options/barchart/normalized/`, and writes a manifest into `data/<TICKER>/options/barchart/manifests/`. If the raw copy already exists with identical content, the import is idempotent; if the name exists with different content, the importer versions the stored copy.

Normalized Barchart chains keep bid/ask/mid, spread, IV, Greeks, probabilities, volume, open interest, liquidity bucket, quality flags, model eligibility, and explicit provenance. For required-path long-call analysis the default usable entry is mid; ask and realistic entry (`mid + 0.25 * spread`) are preserved for sensitivity and conservative reads. `Latest` is imported for reference only and is not used as the entry premium.

The loose Barchart price-history files can refresh the local historical price store:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli import-barchart-price-history `
  --ticker GPRE `
  --csv ".\gpre_price-history-05-05-2026.csv"
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli import-barchart-price-history `
  --ticker PBI `
  --csv ".\pbi_price-history-05-05-2026.csv"
```

Barchart price-history imports copy the raw file into `data/<TICKER>/historical_prices/raw/manual/` and merge normalized `Time` / `Latest` rows into the existing local price store.

## Spot and rates resolution

Metadata resolution continues to prefer explicit and local sources:

1. explicit spot override
2. sidecar / metadata override
3. future explicit chain-source metadata
4. same-day delayed IBKR underlying spot when a usable field exists
5. local historical-price selector
6. moneyness heuristic fallback only outside the canonical market-context workflow

Canonical analysis uses a stricter spot rule than the older generic metadata helper:

- same-day delayed IBKR spot first
- field priority `last`, then `mid`, then `mark`, then `close`
- if same-day IBKR spot is missing or unusable, fall back to the local historical-price store
- prefer same-day local historical close, otherwise nearest prior local historical close

That spot decision is persisted with:

- `spot_price_source`
- `spot_field_used`
- `spot_price_matched_date`
- `spot_used_prior_date`
- `spot_quality_note`
- `ibkr_same_day_spot_rejected_reason`

Rates remain local-first as well.

## IBKR

IBKR ingestion is delayed-only:

- official `ibapi`
- delayed or delayed-frozen market data only
- no live subscription requirement
- no regulatory snapshot fallback

Saved IBKR data lives under:

```text
data/<TICKER>/ibkr/
  snapshots/
  chains/
  metadata/
```

The recommended current-chain ingest command is `fetch-ibkr-full-chain-snapshot`.

It persists:

- delayed underlying snapshots
- delayed chain-universe discovery
- full quoted option-chain snapshots
- per-expiry analysis-ready CSV slices plus metadata sidecars

Every fetch writes explicit provenance, market-data mode, warnings, missing-field notes, and full-chain coverage metadata.

Risk-free stays local-first:

1. explicit override
2. local FRED / Treasury store
3. existing honest fallback

Contract-selection bundles carry that provenance forward so the bundle states the resolved rate, source, series, matched date, and note explicitly.
They also carry local source-selection provenance through:

- `tables/chain_source_summary.csv`
- `tables/market_context_summary.csv`
- `metadata/report_metadata.json`
- `summary/summary.md`

Human-facing summaries keep that provenance compact:

- `summary/summary.md` and `tables/summary.csv` surface trust rollups, matched dates, and short source labels
- raw file-level provenance remains in `metadata/report_metadata.json` and `tables/chain_source_summary.csv`
- `tables/market_context_summary.csv` is the bundle-level trust rollup for spot, rates, and metadata context

Expiry-level source quality is now explicit too. The canonical classes are:

- `same_day_quoted`
- `same_day_sparse`
- `prior_day_quoted`
- `prior_day_sparse`

Those rows also carry a `source_trust_label` and a plain-language `source_quality_note` so sparse fallback expiries are obvious in ranking outputs rather than hidden inside path charts.

## Research metadata

Research metadata remains optional and local:

- expected move
- events
- dividends
- notes
- options overview

Missing research metadata should degrade analysis honestly rather than fabricate certainty.

When present, local research metadata is now used directly in canonical analysis for:

- expected-move context
- event-aware timing and IV commentary
- options-overview IV context
- dividend assumptions
- notes surfaced as supporting context rather than hidden ranking rules

## Archived journal data

`data/journal/` may still exist on disk as archival user data, but journal is no longer a supported workflow or publish category in the canonical architecture.
