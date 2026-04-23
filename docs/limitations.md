# Limitations

## Pricing Model

- Before-expiry valuation uses European Black-Scholes.
- American early exercise is not modeled.
- Assignment risk, borrow costs, hard-to-borrow constraints, and slippage are not modeled.

## Bundle And Publish Model

- Published dashboards only show what was already written into the analysis bundle.
- If a bundle lacks supporting tables or charts, publishing will not silently recreate them.
- Share-safe publishing may drop or shorten local provenance fields that would leak machine-specific paths.
- Archived `data/journal/` files may remain on disk, but journal is not a supported analysis or publish workflow.

## Snapshot Granularity

- The project is designed around daily snapshots, not intraday updates.
- Scenario tables are only as good as the current chain and supporting local data.

## Data Availability

- If a local historical-price store is missing, spot may fall back to a moneyness heuristic.
- If a local FRED store is missing, the project falls back to `0.04` and records that note.
- If no research metadata exists for a ticker, reports simply carry an empty `research_context`.
- Missing stores are not treated as invisible magic. Warnings and report metadata should make the fallback obvious.

## Research Metadata Scope

- The research-metadata layer is intentionally local-first and manual/semi-manual in v1.5.
- Structured expected-move, options-overview, events, dividend, and notes data can be registered, but they are optional context rather than hard requirements.
- Reference images such as the Barchart dashboard remain provenance aids only. They are not OCR'd and are not treated as authoritative machine-readable inputs.

## Horizon Coverage

- If requested scenario horizons exceed remaining life to expiry, they are clamped to expiry.
- Meaningful 3m, 6m, and 12m examples require longer-dated expiries than the current near-dated sample.

## Sparse Local History

- Replay and historical comparisons need exact or near-exact local snapshot coverage.
- Later checkpoints often fall back to modeled continuation when the same expiry is not available in later local data.
- Sparse-data cases should surface `partial`, `insufficient_data`, clamped-horizon notes, or same-coverage notes rather than pretending differentiation exists.

## Source Stability

- Nasdaq historical quotes are a practical unattended source today, but website behavior can change.
- The architecture intentionally keeps a manual-import path so the local canonical store remains stable even if the download path changes later.
- The optional Barchart dashboard-image downloader is best-effort only and may fail if the page no longer exposes a directly fetchable image URL.
