# Options Lab

Options Lab is a standalone, local-first options analysis project centered on one canonical workflow:

1. run `analyze-*` in Python
2. promote the bundle into `model_outputs/` for the curated analyst-facing view
3. optionally run `publish-analysis` to render frozen HTML

The HTML layer is not the analysis engine. It only reads precomputed bundle artifacts.

## Canonical directories

- `data/`: local market data, research metadata, and IBKR snapshots
- `analysis_outputs/`: canonical raw/archive analysis bundles
- `model_outputs/`: curated analyst-facing projection of selected frozen bundle artifacts
- `Dashboards/`: optional secondary mirror of already-published bundles
- `options_lab/analysis/`: source of truth for strategy, scenario, contract-selection, path, ranking, and replay logic
- `options_lab/publish/`: bundle-to-HTML rendering and optional Dashboards mirror/index logic
- `options_lab/ibkr/`: delayed-only IBKR ingestion

Transient browser caches, test artifact roots, and staging folders are also not part of the supported workflow.

## Canonical CLI

Analysis commands:

- `fetch-ibkr-full-chain-snapshot` (recommended delayed-only current-chain fetch)
- `refresh-local-prices` (optional local Nasdaq historical-price refresh)
- `refresh-risk-free-rates` (optional local FRED / Treasury refresh)
- `analyze-strategy`
- `analyze-scenario`
- `analyze-contract-selection` (primary path-thesis workflow)
- `analyze-replay`

Publish command:

- `publish-analysis`

Curated output command:

- `build-model-outputs`

## Development

Install the local dependency set with:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` is constrained by `constraints.txt`, which pins the known-good direct dependency versions used by the current test environment.

The default test command runs the fast suite only:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
```

Large end-to-end bundle and publish tests are marked `slow` and are skipped by default. Run them explicitly with:

```powershell
..\.venv\Scripts\python.exe -m pytest -q -m slow
```

Examples:

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli fetch-ibkr-full-chain-snapshot `
  --ticker GPRE `
  --market-data-mode delayed
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli refresh-local-prices `
  --ticker GPRE
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli refresh-risk-free-rates
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-contract-selection `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --target-price 20 `
  --target-date 2026-07-15
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli analyze-contract-selection `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --target-price 20 `
  --target-date 2026-07-15 `
  --thesis-target-price 30 `
  --thesis-target-date 2026-12-18
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli publish-analysis `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --analysis-kind contract_selection
```

```powershell
..\.venv\Scripts\python.exe -m options_lab.cli build-model-outputs `
  --ticker GPRE `
  --snapshot-date 2026-04-12 `
  --analysis-kind contract_selection
```

Bundle layout:

```text
analysis_outputs/
  <TICKER>/
    snapshot_<YYYY-MM-DD>/
      <analysis_kind>/
        <run_slug>/
          bundle_manifest.json
          tables/
          charts/
          summary/
          metadata/
          publish/   # written only after publish-analysis
```

Curated product-facing layout:

```text
model_outputs/
  <TICKER>/
    latest/
      START_HERE.md
      model_output_manifest.json
      summary.md
      summary.csv
      00_overview/
        entry_justification.md
        thesis_mode.md
        bullish_action_board.md
        other_structures.md
        action_board.md
        highlights.md
        tables/
          bullish_long_call_action_board.csv
          bullish_long_call_watchlist.csv
          bullish_long_call_avoid.csv
          bullish_long_call_triggers.csv
          bullish_long_call_score_breakdown.csv
          other_structures_summary.csv
          stock_preference_summary.csv
          action_board_candidates.csv
          buy_now_candidates.csv
          watchlist_candidates.csv
          avoid_for_now_candidates.csv
          prefer_stock_instead.csv
          decision_triggers.csv
          action_board_score_breakdown.csv
          action_board_explanations.csv
          entry_justification_candidates.csv
          required_stock_path_to_buy.csv
          required_move_summary.csv
          required_move_vs_stock.csv
          required_iv_support_summary.csv
          entry_barrier_summary.csv
          thesis_mode_candidates.csv
          thesis_path_family_summary.csv
          thesis_iv_family_summary.csv
          thesis_candidate_ranking.csv
          max_justified_premium.csv
          current_vs_justified_premium.csv
          thesis_required_move_summary.csv
          thesis_stock_vs_option_summary.csv
        decision_highlights.csv
        candidate_robustness_summary.csv
        candidate_tradeoff_matrix.csv
        stock_vs_option_takeaways.csv
        charts/
          required_stock_path_to_buy.png
          required_move_speed_vs_magnitude.png
          required_move_vs_stock_chart.png
          strike_expiry_entry_barrier_map.png
          iv_support_requirement_chart.png
          thesis_path_gallery.png
          thesis_iv_gallery.png
          thesis_candidate_overview.png
          current_vs_justified_premium.png
          thesis_path_vs_value.png
          thesis_iv_vs_value.png
          thesis_stock_vs_option.png
          bullish_action_board_overview.png
          bullish_conviction_vs_robustness.png
          bullish_buy_watch_avoid_matrix.png
          bullish_trigger_map.png
          action_board_overview.png
          conviction_vs_robustness.png
          buy_watch_avoid_matrix.png
          trigger_map.png
          stock_vs_option_preference_chart.png
          highlights_overview.png
          candidate_robustness_vs_upside.png
          path_survival_scorecard.png
          iv_robustness_scorecard.png
          strike_expiry_tradeoff_overview.png
          stock_vs_option_decision_chart.png
      01_path_packs/
        <path_alias>/
          README.md
          compare_vs_stock_delta.png
          long_call_strike_value.png
          long_call_strike_delta.png
          long_call_expiry_value.png
          long_call_expiry_delta.png
          long_call_best_of_value.png
          long_call_best_of_delta.png
          iv_path_value.png
          iv_path_delta.png
          long_call_strike_iv_value.png
          long_call_strike_iv_delta.png
          long_call_expiry_iv_value.png
          long_call_expiry_iv_delta.png
          long_call_best_of_iv_value.png
          long_call_best_of_iv_delta.png
          iv_robustness_summary.csv
          checkpoints.csv
          iv_checkpoints.csv
      02_tables/
      03_secondary/
    snapshot_<YYYY-MM-DD>/
      <analysis_kind>/
        <run_slug>/
          START_HERE.md
          model_output_manifest.json
          summary.md
          summary.csv
          00_overview/
          01_path_packs/
          02_tables/
          03_secondary/
    archive/
      promoted_runs.json
```

If you are deciding where to look first for a ticker, use `model_outputs/<TICKER>/latest/`.

## What stays in Python

The canonical analysis layer owns:

- strategy construction and pricing
- scenario tables and compare-vs-stock outputs
- required stock path solving
- assumed stock-path and IV-path normalization
- strategy-family ranking
- exact contract selection
- path-case rankings and assumed-path traces
- replay / case-study computation
- bundle writing

`analyze-contract-selection` is the main Python-first workflow for path-thesis work. Its bundle is expected to answer:

- what stock path is required for each candidate or family to work
- how the active assumed stock and IV paths evolve over time
- how different IV paths change required paths and terminal outcomes
- how different IV paths change the same long call when the stock path is held fixed
- how different IV paths change the strike ladder, expiry ladder, and curated best-of long-call set while the stock path stays fixed
- which contracts survive lower IV, which need IV support, and where stock still dominates
- which calls are buyable now, watchlist-only, avoid-for-now, or still worse than stock under the active assumptions
- which calls are most robust, most aggressive, most balanced, or too dependent on timing/IV support
- what a specific target-price / target-date thesis implies for bullish calls, including max thesis-justified premium, path sensitivity, IV sensitivity, and where stock still wins
- when stock still beats the option structure under the same assumptions
- which family and exact contract lead under the active path case

The contract-selection bundle is designed to be usable on disk before HTML. The key path-analysis and market-context artifacts now include:

- `tables/chain_source_summary.csv`
- `tables/market_context_summary.csv`
- `tables/decision_highlights.csv`
- `tables/decision_highlights_explanations.csv`
- `tables/candidate_robustness_summary.csv`
- `tables/candidate_tradeoff_matrix.csv`
- `tables/stock_vs_option_takeaways.csv`
- `tables/highlights_score_breakdown.csv`
- `summary/highlights.md`
- `charts/highlights_overview.png`
- `charts/candidate_robustness_vs_upside.png`
- `charts/path_survival_scorecard.png`
- `charts/iv_robustness_scorecard.png`
- `charts/strike_expiry_tradeoff_overview.png`
- `charts/stock_vs_option_decision_chart.png`
- `summary/action_board.md`
- `summary/bullish_action_board.md`
- `summary/other_structures.md`
- `summary/thesis_mode.md`
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
- `charts/bullish_action_board_overview.png`
- `charts/bullish_conviction_vs_robustness.png`
- `charts/bullish_buy_watch_avoid_matrix.png`
- `charts/bullish_trigger_map.png`
- `charts/action_board_overview.png`
- `charts/conviction_vs_robustness.png`
- `charts/buy_watch_avoid_matrix.png`
- `charts/trigger_map.png`
- `charts/stock_vs_option_preference_chart.png`
- `tables/required_path_rows.csv` and `tables/required_path_summary.csv`
- `tables/stock_path_gallery.csv` and `tables/iv_path_gallery.csv`
- `tables/stock_path_examples.csv` and `tables/iv_path_examples.csv`
- `tables/path_pair_summary.csv`
- `tables/option_value_over_path.csv`
- `tables/compare_vs_stock_path_rows.csv`
- `tables/compare_vs_stock_over_path.csv`
- path-prefixed `tables/<path-alias>__iv_path_value.csv`
- path-prefixed `tables/<path-alias>__iv_path_delta.csv`
- path-prefixed `tables/<path-alias>__iv_checkpoints.csv`
- path-prefixed `tables/<path-alias>__long_call_strike_iv_value.csv` and `tables/<path-alias>__long_call_strike_iv_delta.csv`
- path-prefixed `tables/<path-alias>__long_call_expiry_iv_value.csv` and `tables/<path-alias>__long_call_expiry_iv_delta.csv`
- path-prefixed `tables/<path-alias>__long_call_best_of_iv_value.csv` and `tables/<path-alias>__long_call_best_of_iv_delta.csv`
- path-prefixed `tables/<path-alias>__iv_robustness_summary.csv`
- `tables/long_call_value_over_path_strike_view.csv`
- `tables/long_call_value_over_path_expiry_view.csv`
- `tables/long_call_value_over_path_best_of.csv`
- `tables/representative_paths_summary.csv`
- `tables/strike_comparison_under_path.csv`
- `tables/expiry_comparison_under_path.csv`
- `tables/required_vs_assumed_path_summary.csv`
- `tables/family_comparison.csv`
- `tables/candidate_comparison.csv`
- `tables/strike_comparison.csv`
- `tables/expiry_comparison.csv`
- `tables/assumed_path_trace_rows.csv`
- `tables/iv_path_trace_rows.csv`
- `tables/compare_vs_stock_path_rows.csv`
- `tables/iv_path_sensitivity_summary.csv`
- `tables/path_risk_summary.csv`

Contract-selection bundles are intentionally path/simulation-first. They do not write the older strike/expiry heatmap and slice artifact family; same-path valuation, representative-path selection, and required-vs-assumed path outputs are the primary decision artifacts.

The assumed stock-path engine now also accepts named path presets that are deliberately shaped to read like plausible templates rather than straight-line interpolation. The current product-facing presets include:

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

Those same preset names can drive both the active assumed path and deterministic representative/example paths.

The product now separates two different path surfaces deliberately:

- stock path gallery / IV path gallery: named scenario templates used for first-pass thesis thinking
- path library metadata: stable family labels, timing shapes, and outcome-bias labels for the named scenario library
- single-option decision paths: a deterministic 5-8 path subset selected for one specific contract, with outcome labels and selection reasons
- single-option required-edge paths: frozen 1.5x and 2.0x option-over-stock threshold paths that show what stock route the selected call needs before it beats buying stock; near-zero stock P/L does not count as clearing the edge
- representative paths: heuristic examples selected from simulation or conditioning, used as secondary support after the named galleries

The primary same-future stock benchmark is also explicit now:

- `tables/compare_vs_stock_path_rows.csv`
- `charts/compare_vs_stock_path_delta.png`

Use those assumed-path artifacts before the representative `compare_vs_stock_over_path.*` family when the question is "under this exact thesis path, is stock still cleaner?"

The Action Board is the first product read. It does not claim a universal best option; it translates the frozen path/IV evidence into assumption-relative buckets: Buy Now, Watchlist, Avoid For Now, and Prefer Stock Instead. The Chain Overview / Compare Options layer comes immediately after it and compares bullish long calls side by side against long stock across one shared representative path set, so you can quickly see which calls look robust, asymmetric, early-move friendly, late-move forgiving, too IV-sensitive, too narrow, or simply worse than stock. The entry-justification layer then answers the next practical question: what the stock actually has to do, how quickly, and with how much IV help before a bullish call looks worth buying. The single-option decision view narrows the question to one selected call and asks: what stock path is required for this option to beat buying stock by the configured threshold? The hero chart mutes the selected 5-8 representative paths, overlays thick required 1.5x and 2.0x edge paths, and keeps IV and entry-premium sensitivity in compact lower panels so the path question stays readable. The decision-highlights layer remains the broader robustness read behind those layers, with categories such as most robust call, best aggressive upside, best balanced call, stock still wins, needs IV support, and delayed-move risk. If the evidence is weak, outputs explicitly say `weak_differentiation` or `no_clear_edge_under_current_assumptions`.

`build-model-outputs` promotes only the current primary product artifacts from a frozen bundle. It copies the decision-first files into `model_outputs/` and leaves older or noisier secondary files in `analysis_outputs/`.

Human-facing summaries are intentionally compact and decision-first:

- `summary/summary.md` and `tables/summary.csv` surface trust rollups, matched dates, and short source labels only
- exact source file provenance stays in `metadata/report_metadata.json` and `tables/chain_source_summary.csv`
- `tables/chain_source_summary.csv` and `tables/market_context_summary.csv` are the primary on-disk trust evidence when you want to audit which expiries, spot sources, and fallback paths were actually used

Visual conventions now stay consistent across saved bundle charts:

- one fixed color/marker/line-style mapping per strategy family
- a separate neutral style for assumed paths, stock baselines, and top-candidate overlays
- earlier horizons on the left and later horizons on the right
- lower IV cases before higher IV cases where a categorical IV axis is used
- chart notes in `summary/summary.md` explaining how to read the core figures

Definitions:

- required path: the minimum stock price path needed for a candidate or family to clear a goal such as break-even, +25%, +50%, outperforming stock, or a target option value
- assumed path: the user-selected stock path and IV path that drive the main modeled trace over time
- path library: the broad named stock-path gallery plus family/timing/outcome-bias metadata in `stock_path_library.csv`
- decision path: a curated, contract-specific stock path selected for the single-option chart and persisted in `single_option_decision_path_selections.csv`
- required-edge path: a selected-option stock route persisted in `single_option_required_path_to_beat_stock_1_5x.csv` and `single_option_required_path_to_beat_stock_2_0x.csv` that answers how high and how early the stock must move before the option beats stock by the configured edge; entry P/L is anchored to zero and the edge test starts only after stock P/L clears `--minimum-edge-stock-return-pct`
- IV path: the volatility-shift path applied across the same canonical horizons as the stock path
- simulated / representative path: a generated stock path plus a separate IV path used as an explicit example future under the active assumptions; these are not forecasts
- path pair: one stock path plus one IV path plus valuation-over-time for the same candidate set
- compare-vs-stock: explicit long-stock baseline columns plus delta-versus-stock PnL and return fields in the contract-selection tables

`analyze-contract-selection` now supports a stronger path engine without changing the canonical workflow. The most important extra controls are:

- `--stock-path-mode`
- `--stock-path-target-end`
- `--iv-path-mode`
- `--simulated-path-count`
- `--representative-selection-mode`
- `--simulation-seed`
- `--minimum-edge-stock-return-pct`

The default behavior is still analysis-first and bundle-first: required-path outputs are always written, and representative simulated path pairs are added by default as a second layer.

Contract selection is now split clearly inside the bundle:

- family selection: which strategy family leads under the active objective, path, timing, and IV assumptions
- exact contract selection: which exact strike and expiry lead once the family choice is narrowed
- compare-vs-stock: whether the option structure really beats long stock after timing, IV, and capital constraints are applied

The newest contract-selection layer adds three long-call-only same-path views under one fixed assumed stock path plus IV path:

- strike view: same expiry, multiple strikes
- expiry view: same strike concept, multiple expiries
- best-of view: a curated subset of the most relevant long-call candidates

These live alongside the broader family/candidate tables, but they are intended to be the clearest first read when the question is "which long call actually looks best under this assumed future?"

The product now also writes a path-centric scenario library for the core named stock paths. Those artifacts are path-prefixed and let you read one future at a time:

- `stock_path_library.csv`
- `stock_path_gallery.csv`
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

That path-centric library is the primary answer to: "if the stock moves like this, what do the relevant long calls do versus stock?"

The single-option decision view intentionally does not plot the whole library. It first evaluates the selected contract across the decision path pool, then persists a deterministic curated subset in `single_option_decision_path_selections.csv`. That table records the family label, timing shape, outcome label, score, and reason for each representative path shown in `single_option_decision_view.png`. The same analysis also persists `single_option_required_path_to_beat_stock_1_5x.csv`, `single_option_required_path_to_beat_stock_2_0x.csv`, `single_option_closest_representative_path_to_edge.csv`, and `single_option_edge_gap_by_path_family.csv`, so the single-option chart can show the minimum required edge, the stronger required edge, the closest miss, and whether misses are too low, too late, or both without recomputing anything in publish/model-output code.

Each path pack also includes two IV-centric reads. The first holds the named stock path and one anchor long call fixed while varying the IV regime across flat, mean-reversion lower/higher, up-then-down, down-then-low, and earnings build/crush paths. Use `*_iv_path_value` first to isolate pure IV effect, then `*_iv_path_delta` to see whether that same option still beats stock.

The second IV-centric read expands IV into the actual long-call decision views: strike ladder, expiry ladder, and best-of. The CSVs contain the full selected contract set across all six IV regimes. The PNGs stay curated to avoid chart spaghetti: they plot the core contracts and core IV regimes, while `*_iv_robustness_summary.csv` explains which contracts survive lower IV, require IV support, or remain stock-dominated.

The path-pack charts use a stacked layout: the option value or compare-vs-stock delta is the large top panel, while a compact stock-path context panel sits underneath on the same x-axis. The lower stock panel is deliberately secondary; it explains where the stock is without stealing attention from the option behavior.

Long-call strike, expiry, and best-of charts are now split into two deliberate chart families. `*_value` charts show option value over time and make time decay visible; `*_delta` charts show option performance relative to long stock over the same path. PnL and checkpoints remain in CSV tables for audit/detail reads.

The analyst-facing reading order is now:

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
17. market context / trust
18. `00_overview/charts/single_option_decision_view.png` and `00_overview/tables/single_option_decision_path_selections.csv`
19. `00_overview/highlights.md` for broader robustness context
20. stock path gallery and IV path gallery
21. choose a named stock path and open its compare-vs-stock path pack chart
22. inspect that path's long-call strike ladder, expiry ladder, and best-of value charts
23. inspect that path's single-anchor IV-path value and delta-vs-stock charts
24. inspect the IV-expanded strike, expiry, and best-of value charts
25. review `iv_robustness_summary.csv`, checkpoint tables, and IV-checkpoint tables
25. `00_overview/other_structures.md` for secondary structures only after the bullish-first read
26. representative paths as secondary context

## Publishing

`publish-analysis` writes bundle-local HTML under `publish/` inside the bundle. If you want a browseable static library, add the mirror flag and publish into `Dashboards/`.

The publish step is intentionally bundle-driven: it reads the canonical bundle file map plus explicit publish-context metadata, then renders static HTML without recreating the analysis logic.

For `contract_selection`, the published page is now a self-sufficient trust-aware, path-first read of that bundle:

- Decision Snapshot
- Action Board / Contract Picker
- Entry Justification / Required Stock Path
- Thesis / Price Target Mode
- Single-Option Decision View
- Decision Highlights
- Market Context / Trust Summary
- Stock Path Gallery
- IV Path Gallery
- Required vs Assumed Path
- Same-Path Compare vs Stock
- Long-Call Strike Comparison
- Long-Call Expiry Comparison
- Best-Of Long-Call Comparison
- Representative Paths
- Family / Candidate Highlights
- Warnings / Risk Notes

Related scenario pages, when present, are secondary links only. They are not the main contract-selection reading surface anymore.

## Curated inspection

`model_outputs/` is the place to inspect a ticker first.

- `analysis_outputs/` stays the canonical raw/archive layer
- `model_outputs/` is a curated projection of selected frozen artifacts
- `Dashboards/` remains an optional presentation mirror

Each promoted model-output folder includes:

- `START_HERE.md`: compact guide for what to open first
- `model_output_manifest.json`: machine-readable link back to the source bundle
- `00_overview/`: action board first, entry justification, thesis mode, bullish long-call board, highlights, stock/IV galleries, trust, and secondary structures
- `01_path_packs/<path_alias>/`: one folder per scenario path with value charts, delta-vs-stock charts, and checkpoints
- `02_tables/`: trust/source and decision tables
- `03_secondary/`: representative-path and broad support artifacts

The intended reading path is: start with `00_overview/action_board.md`, then `entry_justification.md`, then `thesis_mode.md`, then `thesis_candidate_overview.png`, then `current_vs_justified_premium.png`, then `thesis_path_gallery.png`, then `thesis_stock_vs_option.png`, then the bullish long-call board and path packs. Read value charts first and delta-vs-stock charts second.

`build-model-outputs` never recomputes analysis. It only reads an existing canonical bundle and promotes the current primary files into a clearer workspace.

The published HTML is intended to be share-safe:

- static, local files only
- no browser-side repricing
- no absolute local path leaks
- no dependency on any extra output root beyond the bundle itself

## IBKR

IBKR ingestion remains standalone and delayed-only:

- official `ibapi` socket client
- delayed or delayed-frozen only
- no live data requirement
- no paid snapshot fallback
- explicit manifests and missing-field notes under `data/<TICKER>/ibkr/`
- the recommended current-chain fetch path is `fetch-ibkr-full-chain-snapshot`, which persists:
  - delayed underlying snapshot
  - delayed chain-universe discovery
  - full quoted option-chain slices under `data/<TICKER>/ibkr/snapshots/option_quotes/`
  - sidecars and manifests with `snapshot_scope`, expiry coverage, strike counts, attempted-contract counts, and missing-field summaries

For same-date analysis precedence, full quoted IBKR slices only outrank manual `option_chains/` slices when they are quote-usable. The current canonical gate is 20% usable-quote coverage for that same-day expiry slice. Sparse same-day IBKR slices stay visible in provenance, but analysis falls back cleanly to the best local quoted slice.

Risk-free remains local-first:

- explicit override if provided
- local FRED / Treasury store if available
- existing honest fallback otherwise

Spot resolution inside canonical analysis is now explicit as well:

- same-day delayed IBKR spot first, but only when a same-day usable field exists
- field priority: `last`, then `mid`, then `mark`, then `close`
- if same-day IBKR spot is missing or unusable, fall back to the local historical-price store
- prefer same-day local historical close, otherwise nearest prior local historical close

Contract-selection bundles now carry trust-aware market context directly on disk. The key provenance fields and rollups include:

- spot source, field used, matched date, and whether a prior date was needed
- same-day IBKR spot attempted/rejected notes
- risk-free source, series, matched date, and fallback note
- expiry-by-expiry `source_quality` labels such as `same_day_quoted`, `same_day_sparse`, `prior_day_quoted`, and `prior_day_sparse`
- bundle-level trust rollups such as `analysis_trust_level`, trusted expiry counts, and sparse/fallback expiry counts

That provenance is carried through the contract-selection bundle in `metadata/report_metadata.json`, `tables/summary.csv`, `tables/chain_source_summary.csv`, `tables/market_context_summary.csv`, and `summary/summary.md`.

## Docs

- [Architecture](docs/architecture.md)
- [Workflows](docs/workflows.md)
- [Dashboarding](docs/dashboarding.md)
- [Data sources](docs/data_sources.md)
- [Limitations](docs/limitations.md)
