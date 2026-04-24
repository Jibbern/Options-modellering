# Workflows

## Recommended workflow

1. Run an analysis command.
2. Promote the selected bundle into `model_outputs/`.
3. Inspect `model_outputs/<TICKER>/latest/` first.
4. Publish HTML only if you want a frozen presentation layer.

Temporary test roots, browser caches, and staging folders are implementation noise, not part of the supported workflow.

## Path-analysis terms

- required path: the minimum stock path needed for a contract or family to clear a goal such as break-even, +25%, +50%, outperforming stock, or a target option value
- assumed path: the active stock path and IV path selected for the main trace through time
- IV path: the volatility-shift curve sampled across the same horizons as the stock path
- compare-vs-stock: explicit long-stock baseline rows plus delta-versus-stock PnL and return columns in the contract-selection bundle

The active assumed stock path can now use shaped named presets as well as manual points. The current named path presets include:

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
- `plus_20_pct_in_1m`
- `plus_30_pct_in_1m`
- `plus_20_pct_in_1q`
- `plus_30_pct_in_1q`

Thesis / Price Target Mode adds endpoint-aware path families that all reach the same explicit target by the same explicit date:

- `early_breakout_to_target`
- `slow_grind_to_target`
- `down_then_recover_to_target`
- `rally_retrace_finish_target`
- `late_breakout_to_target`
- `overshoot_then_settle_at_target`
- `fast_overshoot_then_sideways`
- `weak_start_then_acceleration`
- `two_stage_bull_run`
- `violent_path_to_target`

## Analysis commands

### IBKR current-chain fetch

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli fetch-ibkr-full-chain-snapshot `
  --ticker GPRE `
  --market-data-mode delayed
```

This is the recommended delayed-only current-chain ingest path. It orchestrates:

- one delayed underlying snapshot
- one delayed chain-universe discovery
- one full quoted option sweep across every discovered expiry, strike, call, and put that IBKR exposes in delayed mode

The saved manifests and sidecars under `data/<TICKER>/ibkr/` carry explicit coverage context such as:

- `snapshot_scope: full_chain`
- discovered expiries
- strike counts by expiry
- attempted contract count
- persisted quote count
- delayed-field availability and missing-field summaries

### Optional local refresh steps

If a run needs fresher local fallback context, refresh the local stores first and then analyze:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli refresh-local-prices `
  --ticker GPRE
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli refresh-risk-free-rates
```

These commands stay inside the same canonical workflow. They refresh local stores only; they do not introduce a second analysis path.

### Contract selection

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-contract-selection `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --target-price 20 `
  --target-date 2026-07-15
```

To ask a thesis-first question such as "what does a $30 Dec-2026 thesis mean for calls?", keep the base analysis target but add explicit thesis inputs:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-contract-selection `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --target-price 20 `
  --target-date 2026-07-15 `
  --thesis-target-price 30 `
  --thesis-target-date 2026-12-18
```

This is the primary Python-first path-thesis workflow. The bundle should be useful before HTML and now carries:

- action-board outputs that convert the current assumptions into Buy Now, Watchlist, Avoid For Now, and Prefer Stock Instead buckets without claiming a universal recommendation
- decision highlights that summarize most-robust, aggressive-upside, balanced, stock-still-wins, delayed-move, and IV-support reads without claiming objective mispricing
- score-breakdown and tradeoff tables that explain why each highlight was selected
- chain-source resolution and market-context summaries
- required-path rows plus required-path summaries
- simulated stock-path examples plus separate IV-path examples
- explicit stock-path / IV-path pair summaries
- valuation-over-path tables for the same candidate set under the same representative path pair
- compare-vs-stock-over-path tables
- representative path selection summaries
- strike-under-path and expiry-under-path comparison tables
- required-versus-assumed-path summary rows
- family-comparison rows that show which strategy family really leads under the active objective
- candidate-comparison rows for the exact strike/expiry ranking under the same assumptions
- strike-comparison and expiry-comparison tables for narrowing exact contract choice
- assumed-path value traces
- IV-path traces plus IV sensitivity summaries
- explicit compare-vs-stock path rows
- path-risk summaries with timing/IV/downside context
- path-case family and candidate rankings
- compare-vs-stock outputs
- path/simulation charts that summarize required paths, representative examples, same-path valuation, and active assumed-path progression

The old contract-selection strike/expiry heatmap and slice bundle artifacts are retired. Use the same-path strike/expiry comparison tables and charts instead; they compare candidates under a concrete stock-path plus IV-path pair rather than a generic grid.

The latest contract-selection bundle also writes a path-centric scenario library for the main named stock paths. `stock_path_library.csv` is the library index: it stores stable family labels, timing-shape labels, outcome-bias labels, and the library role for each named path. `stock_path_gallery.csv` and `stock_path_gallery.png` remain the broad browsing surface.

Each selected library path gets its own same-future chart family:

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
- `<path-alias>__long_call_strike_iv_value.png`
- `<path-alias>__long_call_strike_iv_delta.png`
- `<path-alias>__long_call_strike_iv_checkpoints.csv`
- `<path-alias>__long_call_expiry_iv_value.png`
- `<path-alias>__long_call_expiry_iv_delta.png`
- `<path-alias>__long_call_expiry_iv_checkpoints.csv`
- `<path-alias>__long_call_best_of_iv_value.png`
- `<path-alias>__long_call_best_of_iv_delta.png`
- `<path-alias>__long_call_best_of_iv_checkpoints.csv`
- `<path-alias>__iv_robustness_summary.csv`

Read those files path by path when the real question is: "if the stock takes this route, which long calls win?"

The single-option decision view is deliberately narrower than the gallery. It evaluates the selected call across the decision path pool, then persists only 5-8 curated paths in `single_option_decision_path_selections.csv`. Those rows carry the selected path family, timing shape, outcome label, score, and reason, and only those paths are plotted in `single_option_decision_view.png`.

Each path also gets IV-path packs. The single-anchor IV pack fixes the stock path and anchor long call, then varies IV only. The IV-expanded packs apply the same IV regimes to the strike ladder, expiry ladder, and best-of long-call set. Use the value charts to see whether IV saves or crushes option value, the delta charts to see whether the option still beats stock, checkpoint CSVs for key dates, and `iv_robustness_summary.csv` for the compact decision read: survives lower IV, needs IV support, or stock still dominates.

The path-prefixed charts use a stacked two-panel layout: the option value or stock-relative delta is the larger top panel, and the stock path is a compact context panel underneath on the same x-axis. The lower panel is intentionally smaller so the visual answer remains option-first.

Human-facing summaries are now intentionally compact:

- `summary/summary.md` and `tables/summary.csv` surface trust rollups, matched dates, and decision-first notes
- they do not dump raw absolute source-file paths
- exact file provenance remains available in `metadata/report_metadata.json` and `tables/chain_source_summary.csv`

The Action Board is the first contract-picker read in a contract-selection bundle:

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

Use those files to answer what looks buyable now, what belongs on a watchlist, what should be avoided, and when stock remains the cleaner exposure.

The entry-justification layer is the next read after the Action Board. It stays analysis-first and bundle-native:

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

Use those files to answer what the stock actually has to do before a bullish call looks worth buying, how quickly that move has to happen, whether IV support matters, and when stock still remains the cleaner baseline even if the path is broadly right.

Thesis / Price Target Mode is the explicit target-price layer. It asks: "if the stock reaches X by Y, which calls become reasonable, what route to that target helps or hurts them, and what maximum premium is justified under the thesis?" The canonical artifacts are:

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

Use `current_vs_justified_premium.png` to see whether the current option premium is below or above the thesis-relative max entry price. Use `thesis_stock_vs_option.png` to catch the important case where the target is reached but long stock still remains cleaner than the call.

Two layers stay explicit in the bundle:

- required-path layer: what stock path is required for a candidate or family to work
- path-library layer: broad named scenario gallery plus family/timing/outcome metadata for browsing possible futures
- decision-path layer: a contract-specific 5-8 path subset for the single-option decision chart
- simulated-path layer: example stock paths plus separate IV paths showing how a plausible future could miss, almost work, or work under the same valuation engine

### Curated model outputs

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli build-model-outputs `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --analysis-kind contract_selection
```

This command reads a frozen canonical bundle and writes a curated analyst-facing view under `model_outputs/`.

For contract-selection bundles, the promoted view is grouped by reading job:

- `00_overview/`: Action Board first, then entry justification, thesis mode, single-option decision view, bullish long-call board, decision highlights, stock path gallery, IV path gallery, required-vs-assumed path, and active compare-vs-stock overview
- `01_path_packs/<path_alias>/`: one folder per named scenario path with value views, delta-vs-stock views, single-anchor IV-path views, IV-expanded strike/expiry/best-of views, `iv_robustness_summary.csv`, `checkpoints.csv`, and IV checkpoint tables
- `02_tables/`: trust/source, family, candidate, strike, and expiry decision tables
- `03_secondary/`: representative-path and broader support artifacts

`model_outputs/<TICKER>/latest/` is the default "look here first" workspace. It contains:

- `START_HERE.md`
- `model_output_manifest.json`
- the promoted primary tables and charts

Its reading order is:

1. `00_overview/action_board.md`
2. `00_overview/entry_justification.md`
3. `00_overview/thesis_mode.md`
4. `00_overview/charts/thesis_candidate_overview.png`
5. `00_overview/charts/current_vs_justified_premium.png`
6. `00_overview/charts/thesis_path_gallery.png`
7. `00_overview/charts/thesis_stock_vs_option.png`
8. `00_overview/charts/action_board_overview.png`
9. `00_overview/charts/required_stock_path_to_buy.png`
10. `00_overview/charts/required_move_speed_vs_magnitude.png`
11. `00_overview/charts/stock_vs_option_decision_chart.png`
12. `00_overview/bullish_action_board.md`
13. `00_overview/charts/bullish_action_board_overview.png`
14. `00_overview/charts/bullish_buy_watch_avoid_matrix.png`
15. `00_overview/charts/stock_vs_option_preference_chart.png`
16. `00_overview/tables/bullish_long_call_watchlist.csv` and `00_overview/tables/bullish_long_call_avoid.csv`
17. `02_tables/market_context_summary.csv`
18. `00_overview/charts/single_option_decision_view.png` and `single_option_decision_path_selections.csv`
19. `00_overview/highlights.md` for the broader robustness read
20. `00_overview/stock_path_gallery.png` and `00_overview/iv_path_gallery.png`
21. choose a path under `01_path_packs/`
22. read value charts first: strike, expiry, and best-of
23. read single-anchor IV-path value and delta charts for the same stock path
24. read IV-expanded strike, expiry, and best-of value charts
25. use `iv_robustness_summary.csv` to see what survives lower IV, what needs IV support, and where stock still dominates
26. read delta-vs-stock charts second: compare-vs-stock plus strike, expiry, best-of, and IV-expanded delta views
27. use `checkpoints.csv` and IV checkpoint CSVs for key dates
28. `00_overview/other_structures.md` only after the bullish-first read
29. `03_secondary/` only for supporting representative-path context

No analysis is recomputed during this step. `build-model-outputs` only consumes frozen bundle artifacts.

### Scenario

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-scenario `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --expiry-date 2026-04-17
```

### Replay

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-replay `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --expiry-date 2026-04-17 `
  --strategy long_call
```

### Strategy

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-strategy `
  --file data\GPRE\gpre-options-exp-2026-04-17-monthly-near-the-money-stacked-04-12-2026.csv `
  --strategy long_call
```

## Inspecting a bundle

Each bundle contains:

- `bundle_manifest.json`
- CSV tables under `tables/`
- charts under `charts/`
- markdown summary under `summary/`
- resolved metadata under `metadata/`

These files are intended to be useful even if you never open HTML, but they still represent the raw/archive layer.

If you want the cleanest product-facing read, promote the bundle and inspect:

- `model_outputs/<TICKER>/latest/START_HERE.md`
- `model_outputs/<TICKER>/latest/summary.md`
- `model_outputs/<TICKER>/latest/00_overview/`
- `model_outputs/<TICKER>/latest/01_path_packs/<path_alias>/`
- `model_outputs/<TICKER>/latest/02_tables/`
- `model_outputs/<TICKER>/latest/03_secondary/`

When same-date full quoted IBKR slices exist, they only outrank manual `option_chains/` slices if the same-day expiry slice is quote-usable. The current canonical gate is 20% usable-quote coverage. Sparse same-day IBKR slices are still recorded in provenance but fall back cleanly to the best local quoted slice.

Spot resolution is explicit too:

- same-day delayed IBKR spot first when usable
- field priority `last > mid > mark > close`
- otherwise local historical-price close
- same-day local historical close first, otherwise nearest prior local date

Read the trust/provenance files with that in mind. `chain_source_summary.csv` and `market_context_summary.csv` now expose source-quality classes and trust rollups directly:

- `same_day_quoted`
- `same_day_sparse`
- `prior_day_quoted`
- `prior_day_sparse`

For `contract_selection` bundles, inspect these first:

- `chain_source_summary.csv`
- `market_context_summary.csv`
- `stock_path_gallery.csv`
- `iv_path_gallery.csv`
- `required_path_rows.csv`
- `required_path_summary.csv`
- `stock_path_examples.csv`
- `iv_path_examples.csv`
- `path_pair_summary.csv`
- `option_value_over_path.csv`
- `compare_vs_stock_path_rows.csv`
- `compare_vs_stock_over_path.csv`
- `representative_paths_summary.csv`
- `strike_comparison_under_path.csv`
- `expiry_comparison_under_path.csv`
- `required_vs_assumed_path_summary.csv`
- `family_comparison.csv`
- `candidate_comparison.csv`
- `strike_comparison.csv`
- `expiry_comparison.csv`
- `assumed_path_trace_rows.csv`
- `iv_path_trace_rows.csv`
- `iv_path_sensitivity_summary.csv`
- `path_risk_summary.csv`
- `path_case_family_rankings.csv`
- `path_case_candidate_rankings.csv`

Then inspect these charts and notes:

- `charts/family_ranking_overview.png`: family-level objective scores, including weak-differentiation status
- `charts/stock_path_gallery.png`: named stock-path scenarios used as the first scenario-thinking surface
- `charts/iv_path_gallery.png`: named IV regimes shown separately from representative IV examples
- `charts/required_path_vs_assumed_path.png`: required paths against the current assumed stock path
- `charts/compare_vs_stock_path_delta.png`: the primary same-assumed-path stock benchmark
- `charts/representative_stock_paths.png`: selected heuristic example futures, used as secondary support
- `charts/representative_iv_paths.png`: selected heuristic IV examples, used as secondary support
- `charts/option_value_over_path.png`: representative-path option valuation context
- `charts/compare_vs_stock_over_path.png`: representative-path stock benchmark context
- `01_path_packs/<path>/long_call_strike_value.png`: option value, same path and expiry, multiple strikes only
- `01_path_packs/<path>/long_call_strike_delta.png`: stock-relative delta for the same strike ladder
- `01_path_packs/<path>/long_call_expiry_value.png`: option value, same path and strike concept, multiple expiries only
- `01_path_packs/<path>/long_call_expiry_delta.png`: stock-relative delta for the same expiry ladder
- `01_path_packs/<path>/long_call_best_of_value.png`: curated best-of long calls under one named future
- `01_path_packs/<path>/long_call_best_of_delta.png`: stock-relative delta for the same best-of shortlist
- `01_path_packs/<path>/iv_path_value.png`: same stock path and anchor call, multiple IV regimes only
- `01_path_packs/<path>/iv_path_delta.png`: stock-relative version of that fixed-stock IV-regime comparison
- `01_path_packs/<path>/iv_checkpoints.csv`: compact IV-regime checkpoint table with help/crush/stock-cleaner notes
- `charts/strike_comparison_under_same_path.png`: strike choice under one representative future
- `charts/expiry_comparison_under_same_path.png`: expiry choice under one representative future
- `charts/required_path_strategy_compare.png`: required paths versus the active assumed path
- `charts/assumed_path_value_progression.png`: modeled value/PnL progression along the active assumed path
- `charts/iv_path_trace.png`: active IV assumption versus named comparison presets
- `summary/summary.md`: decision snapshot plus a short chart-reading guide
- `metadata/report_metadata.json`: resolved assumptions, risk-free provenance, and the exact source snapshot files/storage locations used by the run
- trust rollups in those same files: spot field used, same-day IBKR spot rejection notes, `analysis_trust_level`, and trusted-vs-fallback expiry counts

## Publishing

`publish-analysis` consumes the saved bundle contract directly. It follows the bundle file map and explicit publish-context metadata first, so published HTML stays tied to the frozen analysis artifacts instead of ad hoc files in `publish/`.

For contract-selection bundles, the published page is now a self-sufficient path-first read of the saved bundle rather than a thin scenario wrapper. The primary published sections are:

- Decision Snapshot
- Chain Overview / Compare Options
- Single-Option Decision View
- Market Context / Trust Summary
- Stock Path Gallery
- IV Path Gallery
- Required vs Assumed Path
- Same-Path Compare vs Stock
- Long-Call Strike Comparison
- Long-Call Expiry Comparison
- Best-Of Long-Call Comparison
- Representative Paths
- Same-Path Strike Comparison
- Same-Path Expiry Comparison
- Family / Candidate Highlights
- Warnings / Risk Notes

Related scenario links, when present, are secondary only.

Bundle-local publish:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli publish-analysis `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --analysis-kind contract_selection
```

Mirror into `Dashboards/` too:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli publish-analysis `
  --bundle analysis_outputs\GPRE\snapshot_2026-04-12\contract_selection\<run_slug> `
  --mirror-dashboards
```

The supported workflow is just:

1. `analyze-*`
2. `build-model-outputs`
3. inspect `model_outputs/...`
4. optional `publish-analysis`
