# Dashboarding

Dashboards are a publish concern, not an analysis concern.

Before opening Dashboards, inspect `model_outputs/<TICKER>/latest/` if you want the clearest chart/table-centric analyst workspace. Dashboards remain secondary presentation over the same frozen bundles.

## Publish contract

`publish-analysis` reads:

- `bundle_manifest.json`
- `tables/`
- `charts/`
- `summary/`
- `metadata/report_metadata.json`

During publish, the copied `report_metadata.json` also carries:

- `bundle_file_map`: the canonical table/chart/summary/metadata contract from the bundle manifest
- `bundle_publish_context`: explicit publish-only links or related-bundle context needed for static navigation

That metadata already includes the trust/provenance fields resolved during analysis, such as:

- spot source, field used, matched date, and same-day rejection note
- risk-free source, series, matched date, and fallback note
- expiry-level source-quality summaries and bundle-level trust rollups

It writes:

- `publish/dashboard.html`
- `publish/published_manifest.json`

inside the existing bundle.

## Primary HTML behavior

The HTML layer is static and bundle-backed:

- no browser-side repricing
- no browser-side reranking
- charts and tables come from the bundle file map first, with directory scanning only as a fallback for local unpublished detail folders
- no dependence on any artifact root outside the published bundle
- no absolute local file links in published pages
- no dependency on journal-specific context or transient staging folders

## Dashboards mirror

`Dashboards/` is optional and secondary. It exists only as a mirror of already-published bundles.

When mirrored, the library is rebuilt from `published_manifest.json` files only. It does not scan legacy output roots.

`model_outputs/` sits between the raw bundle archive and Dashboards:

- `analysis_outputs/`: canonical raw/archive bundles
- `model_outputs/`: curated analyst-facing projection of selected frozen artifacts
- `Dashboards/`: optional HTML presentation mirror

For contract-selection, the curated `latest/` view is grouped for decision reading:

- `00_core_view/`: bullish Action Board first, then chain overview / compare options, entry justification, stress tests, the single-option decision view, and the most important decision charts/tables
- `01_thesis_view/`: explicit target-thesis charts and tables
- `02_path_packs/`: deeper scenario/path packs
- `03_tables/` and `04_secondary/`: supporting tables and lower-priority visuals
- `01_path_packs/<path_alias>/`: value, delta-vs-stock, single-anchor IV-path, IV-expanded strike/expiry/best-of, and robustness-summary outputs for one named stock scenario
- `02_tables/`: source/trust and decision tables
- `03_secondary/`: representative-path support artifacts

## Page roles

The current bundle-backed pages are:

- `scenario`: primary scenario dashboard
- `contract_selection`: primary trust-aware, path-first contract-selection page
- `strategy`: single-strategy detail page
- `replay`: historical learning / case-study page

## Contract-selection publish

Contract-selection publish pages now render the saved bundle artifacts directly instead of acting as a thin wrapper around scenario output.

The primary published reading order is:

- Decision Snapshot
- Chain Overview / Compare Options
- Action Board / Contract Picker
- Entry Justification / Required Stock Path
- Thesis / Price Target Mode
- Single-Option Decision View
- Decision Highlights
- Market Context / Trust Summary
- Stock Path Gallery
- IV Path Gallery
- Required vs Assumed Path
- Path-Centric Compare vs Stock
- Path-Centric Long-Call Strike Ladders
- Path-Centric Long-Call Expiry Ladders
- Path-Centric Best-Of Long-Call Views
- Path-Centric IV-Path Value / Delta Views
- IV-Expanded Strike / Expiry / Best-Of Views
- IV Robustness Summaries
- Path-Centric Checkpoint Tables
- Representative Paths
- Same-Path Strike Comparison
- Same-Path Expiry Comparison
- Family / Candidate Highlights
- Warnings / Risk Notes

That means the main contract-selection page is readable on its own from the frozen bundle. A related scenario page can still be linked when one exists, but it is secondary context rather than the main explorer.

The contract-selection bundle is the canonical source for:

- action-board buckets, watchlist triggers, action-score breakdowns, and stock-preference reads
- entry-justification reads describing what the stock has to do, how quickly, how much IV support matters, and when stock still remains cleaner
- thesis-mode reads describing what a specific target price/date implies for bullish calls, current versus thesis-justified premium, endpoint-aware stock paths, IV sensitivity, and when stock still beats calls even if the target is reached
- decision highlights, robustness summaries, tradeoff matrices, score breakdowns, and stock-vs-option takeaways
- named stock-path and IV-path gallery outputs
- path-prefixed named-scenario long-call comparison outputs
- path-prefixed named-scenario IV-path comparison outputs that keep stock path and contract fixed
- path-prefixed IV-expanded strike, expiry, and best-of outputs that keep stock path fixed while IV varies across the selected long-call set
- path-prefixed IV robustness summaries
- path-prefixed named-scenario checkpoint tables
- representative-family required-path comparisons
- active assumed stock/IV path traces
- representative stock-path and IV-path examples
- assumed-path compare-vs-stock rows and charts
- explicit path-pair valuation tables
- assumed-path long-call strike / expiry / best-of tables and charts
- same-path strike and expiry comparison tables
- family-level path-case rankings
- candidate-level path-case rankings

The dashboard only filters and presents those frozen artifacts. It does not re-rank, reprice, or rebuild trust decisions.

The path-centric charts are saved by the analysis layer with a stacked layout: option value or compare-vs-stock delta on top, compact stock-path context underneath, shared x-axis. Value views and delta-vs-stock views are separate artifact families. IV-path charts reuse the same layout. Some isolate one anchor contract while IV varies; the IV-expanded charts apply IV variation to the strike ladder, expiry ladder, and best-of set without turning publish into a calculation layer. Publish does not rebuild that layout; it only renders the frozen PNGs from the bundle.

Action Board, chain overview, decision highlights, stress tests, and the single-option decision view are also frozen bundle artifacts. Publish may surface them, but bucket selection, compare-options verdicts, trigger generation, category selection, score components, path labels, thresholds, cautions, and stock-vs-option reads are calculated in `options_lab.analysis`, not in the HTML layer.

That means published visuals inherit their quality directly from the saved bundle artifacts:

- consistent strategy-family colors, markers, and line styles
- canonical horizon / IV ordering already baked into the saved PNGs
- chart-reading notes already written in `summary.md`
- summary tables that are pre-rounded and column-ordered for readability
- risk-free and source-snapshot provenance that was already resolved and frozen during analysis
- spot / expiry trust language that was already resolved and frozen during analysis

Contract-selection publish pages no longer depend on legacy selection heatmap/slice tables. They link and render the bundle-written path/simulation artifacts instead.

## Scenario pages

Scenario pages remain useful for snapshot-specific scenario comparison, but they are no longer the primary reading surface for contract-selection. Contract-selection should already be understandable from:

- `summary/summary.md`
- `tables/summary.csv`
- `tables/chain_source_summary.csv`
- `tables/market_context_summary.csv`
- same-path strike/expiry comparison tables
- path-first charts saved in the bundle
