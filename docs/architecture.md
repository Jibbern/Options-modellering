# Architecture

Options Lab is organized around three runtime layers plus one curated outputs layer plus one data layer.

## 1. Data / ingestion

These modules resolve and persist local inputs:

- `options_lab/ibkr/`
- `options_lab/barchart/`
- `options_lab/prices/`
- `options_lab/rates/`
- `options_lab/research_metadata/`
- `options_lab/metadata.py`
- `options_lab/io.py`
- `options_lab/snapshots.py`

Responsibilities:

- discover local option-chain snapshots
- resolve spot, rates, and research context
- refresh local historical prices and local FRED/Treasury rates when requested through thin CLI wrappers
- import manually downloaded Barchart Options Screener and price-history CSVs into normalized local stores without moving the loose source files
- expose quote-usability and coverage on discovered chain slices so source precedence is explicit and testable
- persist delayed-only IBKR underlying, chain, and option snapshots
- persist one-shot delayed-only full quoted chain snapshots for current-chain work
- keep provenance and source notes

## 2. Analysis

`options_lab/analysis/` is the source of truth for decision logic.

Key modules:

- `strategy.py`
- `scenario.py`
- `contract_selection.py`
- `replay.py`
- `paths.py`
- `ranking.py`
- `artifacts.py`
- `market_context.py`
- `models.py`

Responsibilities:

- build strategies and compare them against stock
- resolve canonical local market context across local quoted chains, spot, risk-free, and research metadata
- classify expiry-level source quality and bundle-level trust from that resolved market context
- compute forward scenario tables
- solve required stock paths
- normalize assumed stock paths and IV paths
- generate representative stock paths, simulated GBM paths, and conditioned endpoint paths
- build named stock-path gallery outputs and named IV-regime gallery outputs as first-pass scenario surfaces
- generate separate IV paths and explicit stock-path / IV-path pairings
- evaluate active assumed paths over time
- evaluate selected strategies and contracts over representative path pairs over time
- evaluate focused long-call strike, expiry, and curated best-of subsets under one fixed assumed stock path plus IV path
- persist explicit IV-path traces and IV sensitivity summaries
- persist explicit compare-vs-stock path deltas over time
- persist representative path-pair valuation tables and strike/expiry-under-path comparisons
- persist path-risk summaries covering timing risk, IV risk, and downside shape
- rank path cases at both family and candidate level
- rank strategy families and exact contracts
- compute Action Board / Contract Picker buckets from frozen contract-selection outputs: Buy Now, Watchlist, Avoid For Now, and Prefer Stock Instead
- compute the long-call required-path engine from the same valuation model so each call is judged by what stock path is required to beat owning stock by 1.5x and 2.0x, where the threshold is relative to stock return rather than an absolute option-return target
- compute entry-justification / required-stock-path summaries from the same frozen contract-selection outputs so the product can answer what has to happen before a bullish call looks worth buying
- compute Thesis / Price Target Mode outputs from the same frozen contract-selection engine so the product can answer what a specific target price/date means for bullish calls, max justified premium, path sensitivity, IV sensitivity, and stock-vs-option preference
- compute a Chain Overview / Compare Options layer from the same frozen contract-selection outputs so bullish long calls can be compared side by side against long stock across one shared representative path-family set
- compute the Single-Option Decision View from the same contract-selection valuation/path engine so one selected call can be compared against long stock across 5-8 representative path families, plus low/base/high IV and entry-premium sensitivity
- compute transparent decision highlights and robustness/tradeoff tables from already-computed contract-selection outputs
- compute replay / case-study outputs
- write deterministic analysis bundles

Visual/readability conventions also live here rather than in publish:

- strategy-family color, marker, and line-style mappings
- canonical horizon and IV ordering for saved charts
- bundle-native chart titles, subtitles, and summary notes
- readable table column ordering and numeric rounding for on-disk CSVs

The contract-selection product is path/simulation-first. Legacy strike/expiry heatmap and slice artifacts are not part of the canonical contract-selection bundle; required paths, representative path pairs, valuation-over-path, compare-vs-stock-over-path, and same-path strike/expiry comparisons are.

`analyze-contract-selection` is the primary path-engine entrypoint inside this layer. It owns the canonical required-path, path-case, contract-ranking, same-path strike/expiry comparison, and compare-vs-stock outputs that both on-disk summaries and published contract-selection pages consume directly.

When same-date full quoted IBKR slices exist locally, contract-selection now prefers that richer quoted chain over manual fallback slices only when the slice is quote-usable. The canonical gate is currently 20% usable-quote coverage for that same-day expiry slice. Same-day manually downloaded Barchart Options Screener slices are the next preferred local chain source before older `option_chains/` fallback files. Sparse same-day IBKR slices remain visible in provenance, but they no longer silently override a better local quoted fallback.

The market-context resolver writes that decision through to the bundle so analysis stays auditable on disk:

- `tables/chain_source_summary.csv`
- `tables/market_context_summary.csv`
- `metadata/report_metadata.json`
- `summary/summary.md`

Human-facing summary outputs stay compact and decision-first:

- `summary/summary.md` and `tables/summary.csv` carry trust rollups, matched dates, and short source labels instead of raw file-path dumps
- exact source-file provenance remains available in `metadata/report_metadata.json` and `tables/chain_source_summary.csv`
- same-path strike/expiry comparison tables are the main contract-selection read, with the trust tables providing the supporting evidence

That same resolver now makes the spot and trust rules explicit:

- same-day delayed IBKR spot wins only when a same-day usable field exists
- spot field priority is `last`, then `mid`, then `mark`, then `close`
- if same-day delayed IBKR spot fails, canonical analysis falls back to the local historical-price store
- expiry slices are labeled with source-quality classes such as `same_day_quoted`, `same_day_sparse`, `prior_day_quoted`, and `prior_day_sparse`
- contract-selection summaries carry a bundle-level trust rollup so sparse/fallback expiries are visible before any HTML is opened
- Barchart imports carry `source = barchart_options_screener`, `trust = manually_downloaded_barchart`, `entry_price_mode`, bid/ask/mid, spread, IV, volume, open interest, liquidity, quality flags, and model eligibility through candidate and required-path outputs
- the required-path layer adds execution realism on top of quote metadata: liquidity bucket, fill quality bucket, recommended entry mode, realistic-entry slippage, exit-liquidity risk, execution penalty score, and an execution verdict
- when local Barchart/Nasdaq price history is available, the required-path layer adds historical realism: descriptive forward-return hit rates, percentiles, max seen moves, and historical buckets for each required move; this is not treated as a probability model
- `required_path_candidate_ranking.csv` is the final one-row-per-contract decision layer. It combines required-path score, execution realism, entry/IV fragility, sell/hold pressure, and historical realism into a deterministic rank, final verdict, concise reason, and top risk.

Inside that bundle, the ranking model is intentionally split into two analysis-first layers:

- family selection: choose the strategy family under explicit objective, stock-path, IV-path, timing, and compare-vs-stock assumptions
- exact contract selection: rank exact strike/expiry candidates after folding in affordability, required-path difficulty, timing risk, IV risk, and benchmark edge versus long stock

Path-analysis terminology in the bundle layer:

- required-path engine: the primary long-call product surface; it solves backwards from option-over-stock outperformance and writes `required_path_summary.csv`, `required_path_candidate_ranking.csv`, `required_paths_by_option.csv`, `required_path_family_summary.csv`, `required_path_peak_summary.csv`, `required_path_exit_ladder.csv`, `required_path_execution_realism.csv`, `required_path_historical_realism.csv`, `required_path_entry_sensitivity.csv`, `required_path_iv_sensitivity.csv`, `required_path_entry_iv_matrix.csv`, `required_path_sell_hold_summary.csv`, `required_path_tables.html`, `required_paths_overview.png`, and per-contract `required_paths_<contract_slug>.png`
- per-option required-path chart: the chart horizon defaults to option expiry, while any shorter analysis horizon is only a reference marker; option values are computed along the path with remaining time to expiry declining to intrinsic value at expiry
- required path: the minimum stock path needed to clear a specific goal by each sampled horizon
- assumed path: the active user-selected stock and IV paths used for the main modeled trace
- IV path: the volatility-shift curve sampled alongside the stock path
- simulated / representative path: a generated stock path plus a separate IV path used as an example future under explicit assumptions
- path pair: one stock path plus one IV path plus valuation-over-time for the same strategies/contracts
- compare-vs-stock: explicit long-stock baseline values plus deltas versus that baseline in both PnL and return terms

The deterministic stock-path engine now includes reusable named presets that can drive both the active assumed path and deterministic representative examples, including:

- `rally_early_then_fade_then_rally_again`
- `range_bound_near_flat`
- `down_first_then_recovery`
- `late_breakout`
- `early_move_above_strike_then_giveback`
- `reaches_target_late_near_expiry`
- `quarter_up_then_pullback`
- `quarter_down_then_next_quarter_recovery`
- `two_quarters_down_then_flat_then_recovery`
- `high_swing_quarterly_path`
- `slow_grind_up`
- `overshoot_then_mean_revert`
- `quarter_up_then_hard_pullback`
- `high_vol_sideways_then_breakout`
- `earnings_gap_up_then_fade`
- `earnings_gap_down_then_recovery`
- `false_breakout_then_recover`
- `rally_then_long_range_then_second_leg_up`
- `violent_two_sided_quarter`
- `slow_bleed_then_capitulation_then_bounce`

Those presets are intentionally rendered as eased multi-anchor templates rather than simple straight-line interpolation so they read like plausible path shapes on disk.

The product keeps two path layers explicit:

- gallery paths: named stock-path and IV-path templates for deliberate scenario thinking
- representative paths: heuristic simulation/conditioning outputs used as secondary support

That distinction is reflected directly in the contract-selection bundle:

- `tables/stock_path_gallery.csv` and `charts/stock_path_gallery.png`
- `tables/iv_path_gallery.csv` and `charts/iv_path_gallery.png`
- `tables/stock_path_examples.csv` / `tables/iv_path_examples.csv` plus their representative charts as secondary context

The bundle now also writes a path-centric long-call scenario library for the main named stock paths. Those files are path-prefixed so the reading unit becomes one future at a time, for example:

- `<path-alias>__compare_vs_stock_path_delta.png`
- `<path-alias>__long_call_strike_value.png`
- `<path-alias>__long_call_strike_delta.png`
- `<path-alias>__long_call_expiry_value.png`
- `<path-alias>__long_call_expiry_delta.png`
- `<path-alias>__long_call_best_of_value.png`
- `<path-alias>__long_call_best_of_delta.png`
- `<path-alias>__path_checkpoints.csv`
- `<path-alias>__iv_path_value.png`
- `<path-alias>__iv_path_delta.png`
- `<path-alias>__iv_checkpoints.csv`
- `<path-alias>__long_call_strike_iv_value.png` and `<path-alias>__long_call_strike_iv_delta.png`
- `<path-alias>__long_call_expiry_iv_value.png` and `<path-alias>__long_call_expiry_iv_delta.png`
- `<path-alias>__long_call_best_of_iv_value.png` and `<path-alias>__long_call_best_of_iv_delta.png`
- `<path-alias>__iv_robustness_summary.csv`

That library is analysis-first and bundle-native; `model_outputs/` simply promotes the selected path-prefixed artifacts into the curated reading surface.

The decision-highlights layer also lives in `options_lab.analysis`. It consumes the frozen family/candidate, path-pack, IV-expanded, trust, and compare-vs-stock outputs and writes human-readable but auditable outputs:

- `summary/highlights.md`
- `tables/decision_highlights.csv`
- `tables/decision_highlights_explanations.csv`
- `tables/candidate_robustness_summary.csv`
- `tables/candidate_tradeoff_matrix.csv`
- `tables/stock_vs_option_takeaways.csv`
- `tables/highlights_score_breakdown.csv`
- decision-first overview charts under `charts/`

Those highlights are assumption-relative. They can point to long stock or `no_clear_edge_under_current_assumptions` when options do not clearly beat the stock benchmark.

The Action Board layer sits above the highlights in the same analysis package. It is intentionally not a recommendation engine; it turns the same frozen evidence into a shortlist and trigger map:

- `summary/action_board.md`
- `summary/bullish_action_board.md`
- `summary/other_structures.md`
- `tables/action_board_candidates.csv`
- `tables/buy_now_candidates.csv`
- `tables/watchlist_candidates.csv`
- `tables/avoid_for_now_candidates.csv`
- `tables/prefer_stock_instead.csv`
- `tables/decision_triggers.csv`
- `tables/action_board_score_breakdown.csv`
- `tables/action_board_explanations.csv`
- `tables/bullish_long_call_action_board.csv`
- `tables/bullish_long_call_watchlist.csv`
- `tables/bullish_long_call_avoid.csv`
- `tables/bullish_long_call_triggers.csv`
- `tables/bullish_long_call_score_breakdown.csv`
- `tables/other_structures_summary.csv`
- `tables/stock_preference_summary.csv`
- `charts/bullish_action_board_overview.png`
- `charts/bullish_conviction_vs_robustness.png`
- `charts/bullish_buy_watch_avoid_matrix.png`
- `charts/bullish_trigger_map.png`
- `charts/action_board_overview.png`
- `charts/conviction_vs_robustness.png`
- `charts/buy_watch_avoid_matrix.png`
- `charts/trigger_map.png`
- `charts/stock_vs_option_preference_chart.png`

The bucket logic is transparent and assumption-relative. It can leave Buy Now empty, put convex but fragile calls on Watchlist, move sparse/fallback contracts to Avoid For Now, and promote Prefer Stock Instead when option edge does not clear premium, timing, IV, and trust hurdles.

The Chain Overview / Compare Options layer also lives in analysis. It is a faster compare-many surface than the single-option view and deliberately stays compact:

- `summary/chain_overview.md`
- `tables/chain_overview_summary.csv`
- `tables/chain_overview_candidates.csv`
- `charts/chain_overview.png`

It reuses the same representative path-family set, stock benchmark, and sensitivity evidence already computed upstream. Publish and `model_outputs/` only render those frozen verdicts, counts, and explanations.

The entry-justification layer sits immediately behind the Action Board in the same analysis package. It stays bullish-long-call-first and bundle-native:

- `summary/entry_justification.md`
- `tables/entry_justification_candidates.csv`
- `tables/required_stock_path_to_buy.csv`
- `tables/required_move_summary.csv`
- `tables/required_move_vs_stock.csv`
- `tables/required_iv_support_summary.csv`
- `tables/entry_barrier_summary.csv`
- `charts/required_stock_path_to_buy.png`
- `charts/required_move_speed_vs_magnitude.png`
- `charts/required_move_vs_stock_chart.png`
- `charts/strike_expiry_entry_barrier_map.png`
- `charts/iv_support_requirement_chart.png`

That layer is where the product answers what the stock actually has to do, how quickly, which strikes/expiries ask too much, when IV support matters, and when stock still remains cleaner even if the path is broadly right.

The Thesis / Price Target Mode layer is the explicit endpoint-thesis read. It stays analysis-first and bundle-native:

- `summary/thesis_mode.md`
- `tables/thesis_mode_candidates.csv`
- `tables/thesis_path_family_summary.csv`
- `tables/thesis_iv_family_summary.csv`
- `tables/thesis_candidate_ranking.csv`
- `tables/max_justified_premium.csv`
- `tables/current_vs_justified_premium.csv`
- `tables/thesis_required_move_summary.csv`
- `tables/thesis_stock_vs_option_summary.csv`
- `charts/thesis_path_gallery.png`
- `charts/thesis_iv_gallery.png`
- `charts/thesis_candidate_overview.png`
- `charts/current_vs_justified_premium.png`
- `charts/thesis_path_vs_value.png`
- `charts/thesis_iv_vs_value.png`
- `charts/thesis_stock_vs_option.png`

That layer uses endpoint-aware path families such as `early_breakout_to_target`, `slow_grind_to_target`, `late_breakout_to_target`, and `overshoot_then_settle_at_target` so the same target price/date can be tested through different routes. It is deliberately not a market-neutral fair-value engine; the justified premium is thesis-relative and benchmarked against stock under the same endpoint assumptions.

The path-centric long-call PNGs use the same stacked chart grammar across the product: option value or stock-relative delta in the larger top panel, compact stock path context below, shared dates. Value and delta-vs-stock views are separate chart families so option value/theta never gets confused with stock benchmark edge. The IV-path PNGs use the same grammar in two ways: first to isolate one anchor long call under varying IV, and then to apply those same IV regimes across the strike ladder, expiry ladder, and best-of set. The robustness summary table is the compact decision layer for "survives lower IV", "requires IV support", and "stock still cleaner".

## 3. Publish

`options_lab/publish/` consumes frozen analysis bundles and renders HTML.

Responsibilities:

- render bundle-local `publish/dashboard.html`
- follow the bundle file map and explicit publish-context metadata when choosing which frozen artifacts to render or link
- keep published HTML share-safe
- optionally mirror published bundles into `Dashboards/`
- rebuild the static Dashboards library index from published bundles only

The publish layer does not reprice, rerank, or solve missing analysis on the fly.

For contract-selection bundles specifically, publish is now a direct bundle consumer rather than a scenario wrapper. The primary published reading order mirrors the saved bundle:

- Decision Snapshot
- Action Board / Contract Picker
- Entry Justification / Required Stock Path
- Thesis / Price Target Mode
- Practical Stress Tests
- Single-Option Decision View
- Decision Highlights
- Market Context / Trust Summary
- Required vs Assumed Path
- Stock Path Gallery
- IV Path Gallery
- Path-Centric Compare vs Stock
- Path-Centric Long-Call Strike / Expiry / Best-Of Views
- Representative Paths As Secondary Context
- Family / Candidate Highlights
- Warnings / Risk Notes

Scenario pages remain optional secondary context only.

## 4. Curated outputs

`model_outputs/` is a product-facing projection layer built from already-generated canonical bundles.

Responsibilities:

- select only the primary charts and tables from a frozen bundle
- write a compact `START_HERE.md` for the promoted run
- write a `model_output_manifest.json` that links back to the canonical source bundle
- maintain an obvious `latest/` workspace for each ticker
- keep simple archive bookkeeping for promoted runs
- group contract-selection output into `00_core_view/`, `01_option_required_paths/`, and `99_secondary_or_debug/`

This layer is intentionally not a second analysis engine:

- it does not reprice, rerank, or solve paths
- it does not become a second source of truth
- it only reads `analysis_outputs/` bundles that were already generated by `options_lab.analysis`

For contract-selection, `model_outputs/` is where the user should look first. `analysis_outputs/` remains the raw/archive layer.

The curated reading order for `model_outputs/<TICKER>/latest/` is:

1. `00_core_view/required_paths_overview.png`
2. `00_core_view/required_path_tables.html`
3. `01_option_required_paths/required_path_candidate_ranking.csv`
4. `00_core_view/required_path_summary.md`
5. `00_core_view/required_path_summary.csv`
6. `00_core_view/required_path_tables.md`
7. `00_core_view/top_required_path_candidates.md`
8. `01_option_required_paths/` for per-contract required stock and option-return charts plus execution, sensitivity, and sell/hold CSVs
9. `99_secondary_or_debug/` for supporting Markdown and CSV diagnostics; old fixed-target, single-option, gallery, and path-pack charts stay out of the curated model-output surface

## 5. Runtime directories

The supported directory model is:

- `data/`
- `analysis_outputs/`
- `model_outputs/`
- `Dashboards/`

`analysis_outputs/` is the only canonical analysis artifact root.
`model_outputs/` is a curated analyst-facing view built from those canonical bundles.
Transient folders such as browser caches, staging roots, and test-only artifact trees are not part of the runtime architecture.
