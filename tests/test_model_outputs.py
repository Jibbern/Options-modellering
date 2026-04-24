from __future__ import annotations

import csv
import json
from importlib import import_module
from pathlib import Path

from options_lab.analysis.contract_selection import _path_view_filename


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_png_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")


def _create_fake_contract_selection_bundle(workspace_root: Path, *, run_slug: str, snapshot_date: str = "2026-04-12") -> Path:
    analysis_root = workspace_root / "analysis_outputs"
    bundle_dir = analysis_root / "GPRE" / f"snapshot_{snapshot_date}" / "contract_selection" / run_slug
    tables_dir = bundle_dir / "tables"
    charts_dir = bundle_dir / "charts"
    summary_dir = bundle_dir / "summary"
    metadata_dir = bundle_dir / "metadata"

    summary_text = "\n".join(
        [
            "# GPRE Contract Selection",
            "",
            "Compact bundle summary for model outputs.",
            "",
            "## Decision Snapshot",
            "",
            "- Best family: Long Stock",
            "- Best candidate: Long Stock Baseline",
        ]
    )
    _write_text(summary_dir / "summary.md", summary_text)
    _write_text(
        summary_dir / "highlights.md",
        "\n".join(
            [
                "# GPRE Decision Highlights",
                "",
                "## Decision Snapshot",
                "",
                "- Balanced call read: `Long Call 2026-12-18 15.00`",
                "",
                "## What Looks Most Attractive Right Now",
                "",
                "- Best Balanced Call: `Long Call 2026-12-18 15.00` - weak_differentiation.",
                "",
                "## Where Stock Still Looks Better",
                "",
                "- Stock Still Best Baseline: `Long Stock Baseline` - informative_edge.",
            ]
        ),
    )
    _write_text(
        summary_dir / "action_board.md",
        "\n".join(
            [
                "# GPRE Action Board",
                "",
                "Assumption-relative contract picker from the frozen bundle.",
                "",
                "## What Looks Most Actionable Right Now",
                "",
                "- No Buy Now candidates under current assumptions.",
                "",
                "## What Belongs On The Watchlist",
                "",
                "- `Long Call 2026-12-18 15.00`: needs stock confirmation.",
                "",
                "## What To Avoid For Now",
                "",
                "- No avoid rows in this fixture.",
                "",
                "## When Stock Is Still Better",
                "",
                "- `Long Stock Baseline`: no expiry or IV dependency.",
                "",
                "## Key Triggers To Watch",
                "",
                "- `Long Call 2026-12-18 15.00`: stock confirmation above $20.",
                "",
                "## Trust / Data Caveats",
                "",
                "- Analysis trust level: `cautious`",
            ]
        ),
    )
    _write_text(
        summary_dir / "bullish_action_board.md",
        "\n".join(
            [
                "# GPRE Bullish Long-Call Board",
                "",
                "## Decision Snapshot",
                "",
                "- Buy Now count: `0`",
                "- Watchlist count: `1`",
                "",
                "## Best Bullish Long Calls Right Now",
                "",
                "- No Buy Now candidates under current assumptions.",
                "",
                "## Watchlist: Interesting But Not Buyable Yet",
                "",
                "- `Long Call 2026-12-18 15.00`: still interesting, but it needs a trigger before it is buyable.",
                "",
                "## Avoid For Now",
                "",
                "- `Long Call 2026-05-15 18.00`: premium too demanding under the current path.",
                "",
                "## When Stock Is Still Better",
                "",
                "- `Long Stock Baseline`: no expiry or IV dependency.",
                "",
                "## Key Triggers To Watch",
                "",
                "- `Long Call 2026-12-18 15.00` [Better After IV Cools]: more attractive after IV cools or premium resets lower.",
            ]
        ),
    )
    _write_text(
        summary_dir / "top_candidate_cards.md",
        "\n".join(
            [
                "# GPRE Top Bullish Call Cards",
                "",
                "Compact first-read cards for the most relevant bullish long calls.",
                "",
                "## Card 1: 15C Dec-26 - Watchlist",
                "",
                "- Interesting: High convexity if the stock path works.",
                "- Hurting it: Premium too demanding under current path.",
                "- Upgrade if: More attractive after IV cools or premium resets lower.",
                "- Invalidate if: Avoid if stock stays below the confirmation level while theta keeps working.",
                "- Compare vs stock: Long stock still cleaner until option edge improves.",
            ]
        ),
    )
    _write_text(
        summary_dir / "other_structures.md",
        "\n".join(
            [
                "# GPRE Other Structures / Secondary Board",
                "",
                "## Secondary Structures Snapshot",
                "",
                "- `Covered Call 2026-12-18 18.00`: secondary income-style structure, not part of the primary bullish long-call board.",
                "",
                "## When Stock Still Leads",
                "",
                "- `Long Stock Baseline`: stock remains the cleanest exposure under current assumptions.",
            ]
        ),
    )
    _write_text(
        summary_dir / "entry_justification.md",
        "\n".join(
            [
                "# GPRE Entry Justification / Required Stock Path",
                "",
                "## What Has To Happen For These Calls To Be Worth Buying",
                "",
                "- `15C Dec-26`: needs about $18.40 by 1 month and $20.10 by the target date.",
                "",
                "## Which Calls Require Too Much",
                "",
                "- `18C May-26`: asks for too much too quickly versus the remaining runway.",
                "",
                "## Which Calls Are More Forgiving",
                "",
                "- `15C Dec-26`: needs a smaller move and more time than the shorter-dated OTM call.",
                "",
                "## Which Calls Need Fast Confirmation",
                "",
                "- `18C May-26`: theta bites quickly if the move slips.",
                "",
                "## Which Calls Mainly Need Better IV / Better Entry",
                "",
                "- `15C Dec-26`: becomes more attractive after IV cools or premium resets lower.",
                "",
                "## When Stock Is Still Better Even If The Path Is \"Right\"",
                "",
                "- `15C Dec-26`: stock may still look cleaner if the move is late or IV normalizes lower.",
                "",
                "## Best Next Files To Open",
                "",
                "- `charts/required_stock_path_to_buy.png`",
                "- `charts/required_move_speed_vs_magnitude.png`",
                "- `charts/required_move_vs_stock_chart.png`",
            ]
        ),
    )
    _write_text(
        summary_dir / "thesis_mode.md",
        "\n".join(
            [
                "# GPRE Thesis Mode",
                "",
                "## Thesis Snapshot",
                "",
                "- Target: `$30.00` by `2026-12-18`",
                "",
                "## What This Target Means For Option Selection",
                "",
                "- Same endpoint, different paths and IV regimes.",
                "",
                "## Which Calls Start To Look Reasonable Under This Thesis",
                "",
                "- `15C Dec-26`: near watchlist under the thesis.",
                "",
                "## Current Premium vs Thesis-Justified Premium",
                "",
                "- Open `charts/current_vs_justified_premium.png`.",
                "",
                "## When Stock Still Looks Better",
                "",
                "- Stock can still be cleaner if the move is slow or IV falls.",
            ]
        ),
    )
    _write_text(
        summary_dir / "stress_tests.md",
        "\n".join(
            [
                "# GPRE Stress Tests",
                "",
                "## Stress Snapshot",
                "",
                "- Thesis target: `$30.00` by `2026-12-18`",
                "",
                "## Which Candidates Are Price-Sensitive?",
                "",
                "- `15C Dec-26`: -20% premium can upgrade the setup.",
                "",
                "## What Breaks If The Move Arrives Later?",
                "",
                "- `15C Dec-26`: weaker if the move is delayed two months.",
                "",
                "## Do Calls Need The Thesis To Overshoot?",
                "",
                "- `15C Dec-26`: overshoot is the strongest stress.",
            ]
        ),
    )
    _write_text(
        summary_dir / "chain_overview.md",
        "\n".join(
            [
                "# GPRE Chain Overview / Compare Options",
                "",
                "## What This Layer Compares",
                "",
                "This layer compares bullish long calls against long stock across shared representative path families.",
                "",
                "## Verdict Rules",
                "",
                "- Robust buy candidate: multi-path win with acceptable fragility.",
                "- Selective / thesis-dependent: still needs the right path or entry.",
                "- Too narrow: path support is too concentrated.",
                "- Stock better: stock remains cleaner under representative paths.",
            ]
        ),
    )
    _write_text(
        summary_dir / "chain_overview.md",
        "\n".join(
            [
                "# GPRE Chain Overview / Compare Options",
                "",
                "## What This Layer Compares",
                "",
                "This layer compares bullish long calls against long stock across shared representative path families.",
                "",
                "## Verdict Rules",
                "",
                "- Robust buy candidate: multi-path win with acceptable fragility.",
                "- Selective / thesis-dependent: still needs the right path or entry.",
                "- Too narrow: path support is too concentrated.",
                "- Stock better: stock remains cleaner under representative paths.",
            ]
        ),
    )
    _write_text(
        summary_dir / "single_option_decision.md",
        "\n".join(
            [
                "# GPRE Single-Option Decision View",
                "",
                "## Decision Snapshot",
                "",
                "- Selected option: `15C Dec-26`",
                "- Decision status: `too_narrow_under_representative_paths`",
                "",
                "## What The View Answers",
                "",
                "This section asks whether one selected call is worth buying instead of buying stock.",
            ]
        ),
    )

    _write_csv(
        tables_dir / "summary.csv",
        [
            {
                "ticker": "GPRE",
                "snapshot_date": snapshot_date,
                "target_date": "2026-07-15",
                "target_price": "20.0",
                "goal": "break_even",
                "comparison_capital": "1000.0",
                "stock_path_name": "slow_bull",
                "iv_path_name": "flat",
                "spot_price_source": "nasdaq_historical_quotes",
                "spot_field_used": "close",
                "spot_used_prior_date": "True",
                "spot_quality_note": "Spot fell back to a prior-date local historical close.",
                "spot_price_matched_date": "2026-04-10",
                "risk_free_rate_source": "fred_local_store",
                "risk_free_rate_series": "DGS3MO",
                "risk_free_rate_matched_date": "2026-04-10",
                "risk_free_rate_note": "Used the latest available prior Treasury observation.",
                "analysis_trust_level": "cautious",
                "analysis_trust_note": "The run mixes quoted expiries with sparse fallback expiries.",
                "trusted_expiry_count": "5",
                "fallback_only_expiry_count": "2",
                "ibkr_same_day_spot_rejected_reason": "No same-day delayed IBKR underlying snapshot for GPRE was available on 2026-04-12.",
                "source_snapshot_storage_locations": "ibkr_full_quoted_snapshot | preferred_option_chains",
                "best_family": "Long Stock",
                "best_family_candidate": "Long Stock Baseline",
                "best_candidate": "Long Stock Baseline",
                "best_candidate_family": "long_stock",
                "best_expiry": "2026-04-17",
                "best_strike": "Stock",
                "best_strike_source_trust_label": "trusted_quoted",
                "best_expiry_source_trust_label": "trusted_quoted",
                "best_strike_representative_bucket": "just_works",
                "best_expiry_representative_bucket": "just_works",
                "family_edge_status": "weak_differentiation",
                "stock_benchmark_decision": "long_stock_benchmark",
                "stock_benchmark_note": "Tracks long stock closely under this checkpoint.",
                "benchmark_edge": "0.0",
                "benchmark_return_edge": "0.0",
                "top_path_risk": "Can tolerate a slower move better than the base case.",
                "timing_risk": "Can tolerate a slower move better than the base case.",
                "iv_risk": "low iv dependence",
                "required_path_difficulty": "roughly matched",
                "required_path_gap_at_target": "4.77",
                "first_cleared_horizon": "entry",
                "iv_sensitivity_note": "Profit spans about $0 across IV-path presets.",
                "default_case_label": "0%",
                "primary_warning": "Used a nearest local chain fallback for expiry 2026-05-15 because no usable same-day slice existed.",
                "stock_benchmark_label": "Long stock is the benchmark",
            }
        ],
    )

    _write_csv(
        tables_dir / "chain_source_summary.csv",
        [
            {
                "ticker": "GPRE",
                "requested_snapshot_date": snapshot_date,
                "expiry_date": "2026-04-17",
                "scope": "exact_snapshot",
                "fallback_level": "exact_same_day_preferred_option_chain",
                "storage_location": "preferred_option_chains",
                "source_snapshot_date": snapshot_date,
                "source_snapshot_file": r"C:\Users\Jibbe\Aktier\Options\data\GPRE\option_chains\gpre-options-exp-2026-04-17.csv",
                "quote_usable": "True",
                "usable_quote_coverage_pct": "100.0",
                "usable_quote_count": "34",
                "contract_count": "34",
                "snapshot_distance_days": "0",
                "source_quality": "same_day_quoted",
                "source_trust_label": "trusted_quoted",
                "source_quality_note": "Same-day quoted source was used for pricing with 100.0% usable quote coverage.",
                "chosen_reason": "Same-day preferred local option_chains slice was used.",
                "rejected_same_day_ibkr_file": "",
                "rejected_same_day_ibkr_coverage_pct": "",
            },
            {
                "ticker": "GPRE",
                "requested_snapshot_date": snapshot_date,
                "expiry_date": "2027-01-15",
                "scope": "nearest_snapshot_fallback",
                "fallback_level": "nearest_sparse_fallback",
                "storage_location": "ibkr_full_quoted_snapshot",
                "source_snapshot_date": "2026-04-19",
                "source_snapshot_file": r"C:\Users\Jibbe\Aktier\Options\data\GPRE\ibkr\snapshots\option_quotes\normalized\ibkr_gpre_options_exp_2027-01-15.csv",
                "quote_usable": "False",
                "usable_quote_coverage_pct": "0.0",
                "usable_quote_count": "0",
                "contract_count": "58",
                "snapshot_distance_days": "7",
                "source_quality": "prior_day_sparse",
                "source_trust_label": "fallback_only",
                "source_quality_note": "Nearest prior sparse slice was used as a last-resort fallback.",
                "chosen_reason": "No usable quoted slice existed for this expiry, so the nearest sparse local slice was selected.",
                "rejected_same_day_ibkr_file": "",
                "rejected_same_day_ibkr_coverage_pct": "",
            },
        ],
    )

    _write_csv(
        tables_dir / "market_context_summary.csv",
        [
            {
                "ticker": "GPRE",
                "requested_snapshot_date": snapshot_date,
                "target_date": "2026-07-15",
                "resolved_expiry_count": "7",
                "analysis_trust_level": "cautious",
                "analysis_trust_note": "The run mixes quoted expiries with sparse fallback expiries.",
                "same_day_quoted_expiry_count": "1",
                "same_day_sparse_expiry_count": "0",
                "prior_day_quoted_expiry_count": "4",
                "prior_day_sparse_expiry_count": "2",
                "trusted_expiry_count": "5",
                "fallback_only_expiry_count": "2",
                "spot_price": "15.23",
                "spot_price_source": "nasdaq_historical_quotes",
                "spot_price_matched_date": "2026-04-10",
                "spot_field_used": "close",
                "spot_used_prior_date": "True",
                "spot_quality_note": "Spot fell back to a prior-date local historical close.",
                "ibkr_same_day_spot_attempted": "True",
                "ibkr_same_day_spot_rejected_reason": "No same-day delayed IBKR underlying snapshot for GPRE was available on 2026-04-12.",
                "risk_free_rate": "0.0369",
                "risk_free_rate_source": "fred_local_store",
                "risk_free_rate_series": "DGS3MO",
                "risk_free_rate_matched_date": "2026-04-10",
                "risk_free_rate_note": "Used the latest available prior Treasury observation.",
                "expected_move_matched": "True",
                "nearest_event_type": "earnings",
            }
        ],
    )

    for filename in [
        "stock_path_library.csv",
        "stock_path_gallery.csv",
        "iv_path_gallery.csv",
        "family_comparison.csv",
        "candidate_comparison.csv",
        "strike_comparison_under_path.csv",
        "expiry_comparison_under_path.csv",
        "option_value_over_path.csv",
        "compare_vs_stock_path_rows.csv",
        "compare_vs_stock_over_path.csv",
        "long_call_value_over_path_strike_view.csv",
        "long_call_value_over_path_expiry_view.csv",
        "long_call_value_over_path_best_of.csv",
    ]:
        _write_csv(
            tables_dir / filename,
            [
                {
                    "path_name": "late_breakout",
                    "path_label": "Late Breakout",
                    "path_role": "gallery_named_path",
                    "display_order": "1",
                    "date": snapshot_date,
                    "requested_days": "0",
                    "spot_price": "15.23",
                    "return_pct": "0.0",
                    "is_active_assumed": "True",
                    "iv_path_name": "flat",
                    "iv_path_label": "Flat",
                    "candidate_label": "Long Stock Baseline",
                    "best_candidate_label": "Long Stock Baseline",
                    "strategy_family": "long_stock",
                    "expiry_date": "2026-04-17",
                    "strike_label": "Stock",
                    "objective_score": "408.95",
                    "difference_vs_stock": "0.0",
                    "delta_profit_loss_vs_stock": "0.0",
                    "timing_risk": "Can tolerate a slower move better than the base case.",
                    "iv_risk": "low iv dependence",
                    "source_trust_label": "trusted_quoted",
                    "weak_horizon_fit": "False",
                    "target_beyond_expiry": "False",
                    "selection_reason": "Selected as the clearest example for this outcome bucket under the active goal.",
                }
            ],
        )

    _write_csv(
        tables_dir / "decision_highlights.csv",
        [
            {
                "display_order": "1",
                "highlight_category": "best_balanced_call",
                "highlight_label": "Best Balanced Call",
                "selected_candidate_slug": "long-call-2026-12-18-15-00",
                "selected_candidate_label": "Long Call 2026-12-18 15.00",
                "selected_family": "long_call",
                "decision_status": "weak_differentiation",
                "score": "72.4",
                "source_trust_label": "quoted_prior_day",
                "trust_caution": "",
                "primary_reason": "Best compromise across upside, timing, IV, and trust.",
                "main_warning": "stock still dominates under current assumptions",
                "decision_tags": "premium_too_demanding_under_base_path",
                "difference_vs_stock": "-120.0",
                "return_on_comparison_capital": "0.18",
                "robustness_score": "62.0",
                "aggressive_upside_score": "81.0",
                "balanced_score": "72.4",
            },
            {
                "display_order": "8",
                "highlight_category": "stock_still_best_baseline",
                "highlight_label": "Stock Still Best Baseline",
                "selected_candidate_slug": "long-stock-baseline",
                "selected_candidate_label": "Long Stock Baseline",
                "selected_family": "long_stock",
                "decision_status": "informative_edge",
                "score": "89.0",
                "source_trust_label": "trusted_quoted",
                "trust_caution": "",
                "primary_reason": "Stock removes IV and expiry dependency.",
                "main_warning": "assumption-relative, not objective mispricing",
                "decision_tags": "stock_dominates_under_current_assumptions",
                "difference_vs_stock": "0.0",
                "return_on_comparison_capital": "0.0",
                "robustness_score": "89.0",
                "aggressive_upside_score": "30.0",
                "balanced_score": "89.0",
            },
        ],
    )
    for filename in [
        "decision_highlights_explanations.csv",
        "candidate_robustness_summary.csv",
        "candidate_tradeoff_matrix.csv",
        "stock_vs_option_takeaways.csv",
        "highlights_score_breakdown.csv",
    ]:
        _write_csv(
            tables_dir / filename,
            [
                {
                    "candidate_slug": "long-call-2026-12-18-15-00",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "strategy_family": "long_call",
                    "highlight_category": "best_balanced_call",
                    "takeaway_type": "stock_vs_best_balanced_call",
                    "status": "stock_still_cleaner",
                    "component": "balanced_score",
                    "component_score": "72.4",
                    "note": "Stock still cleaner under current assumptions.",
                }
            ],
        )

    _write_csv(
        tables_dir / "action_board_candidates.csv",
        [
            {
                "action_bucket": "Watchlist",
                "action_priority_rank": "1",
                "action_confidence": "medium",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "strategy_family": "long_call",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "source_trust_label": "quoted_prior_day",
                "candidate_conviction_score": "63.2",
                "action_score": "58.4",
                "robustness_score": "55.0",
                "upside_score": "77.0",
                "stock_relative_score": "42.0",
                "time_decay_risk": "38.0",
                "iv_dependence_risk": "52.0",
                "trust_penalty": "12.0",
                "affordability_status": "reasonable_premium",
                "difference_vs_stock": "-120.0",
                "headline_reason": "Interesting, but it needs a trigger before it is buyable.",
                "why_this_is_interesting_now": "High convexity if the stock path works.",
                "what_is_hurting_this_candidate": "Premium too demanding under current path. Benefits from lower-IV entry.",
                "main_trigger": "More attractive after IV cools or premium resets lower.",
                "why_buy_now": "",
                "why_watch_not_buy": "Premium too demanding under current path. Trigger before buying: More attractive after IV cools or premium resets lower.",
                "why_avoid": "",
                "why_stock_may_be_better": "Stock may be cleaner until confirmation improves.",
                "what_has_to_happen": "More attractive after IV cools or premium resets lower.",
                "upgrade_rule": "Upgrade if premium cools below the current reference or IV normalizes after the event.",
                "what_would_invalidate": "Avoid if stock stays below the confirmation level while theta keeps working.",
                "invalidate_rule": "Avoid if stock stays below the confirmation level while theta keeps working.",
                "key_trigger_label": "Better After IV Cools",
                "key_trigger_type": "iv_normalization_entry",
                "key_trigger_value": "More attractive after IV cools or premium resets lower.",
                "key_trigger_deadline": "2026-07-15",
                "main_warning": "Premium too demanding under current path. Benefits from lower-IV entry.",
            },
            {
                "action_bucket": "Prefer Stock Instead",
                "action_priority_rank": "1",
                "action_confidence": "high",
                "candidate_slug": "long-stock-baseline",
                "candidate_label": "Long Stock Baseline",
                "strategy_family": "long_stock",
                "expiry_date": "",
                "strike_label": "Stock",
                "source_trust_label": "trusted_quoted",
                "candidate_conviction_score": "80.0",
                "action_score": "74.0",
                "robustness_score": "80.0",
                "upside_score": "30.0",
                "stock_relative_score": "50.0",
                "time_decay_risk": "0.0",
                "iv_dependence_risk": "0.0",
                "trust_penalty": "0.0",
                "affordability_status": "stock_baseline",
                "difference_vs_stock": "0.0",
                "headline_reason": "Stock is the cleanest exposure under current assumptions.",
                "why_this_is_interesting_now": "Simplest way to express the thesis without option timing or IV risk.",
                "what_is_hurting_this_candidate": "Main assumptions remain the controlling risk.",
                "main_trigger": "Prefer stock unless an option shows clear modeled stock-relative edge after premium and timing.",
                "why_buy_now": "",
                "why_watch_not_buy": "",
                "why_avoid": "",
                "why_stock_may_be_better": "No expiry, strike, premium, or IV-path dependency.",
                "what_has_to_happen": "Prefer stock unless an option shows clear modeled stock-relative edge after premium and timing.",
                "upgrade_rule": "Stock remains preferred unless an option shows clear modeled stock-relative edge.",
                "what_would_invalidate": "Stock remains simplest if option edge stays weak.",
                "invalidate_rule": "Stock remains simplest if option edge stays weak.",
                "key_trigger_label": "Stock Baseline",
                "key_trigger_type": "stock_baseline",
                "key_trigger_value": "Prefer stock unless option edge improves.",
                "key_trigger_deadline": "2026-07-15",
                "main_warning": "main assumptions remain the controlling risk",
            },
            {
                "action_bucket": "Watchlist",
                "action_priority_rank": "2",
                "action_confidence": "cautious",
                "candidate_slug": "covered-call-2026-12-18-18-00",
                "candidate_label": "Covered Call 2026-12-18 18.00",
                "strategy_family": "covered_call",
                "expiry_date": "2026-12-18",
                "strike_label": "18.00",
                "source_trust_label": "quoted_prior_day",
                "candidate_conviction_score": "45.0",
                "action_score": "41.0",
                "robustness_score": "49.0",
                "upside_score": "20.0",
                "stock_relative_score": "35.0",
                "time_decay_risk": "28.0",
                "iv_dependence_risk": "15.0",
                "trust_penalty": "12.0",
                "affordability_status": "income_style",
                "difference_vs_stock": "-80.0",
                "headline_reason": "Interesting as secondary yield, but stock still cleaner for upside.",
                "why_this_is_interesting_now": "Income-style way to get paid while waiting.",
                "what_is_hurting_this_candidate": "Caps upside if the bullish thesis works too well.",
                "main_trigger": "Only attractive if the thesis shifts toward slower upside and income matters more.",
                "why_buy_now": "",
                "why_watch_not_buy": "Caps upside too much for the primary bullish thesis.",
                "why_avoid": "",
                "why_stock_may_be_better": "Stock keeps upside cleaner.",
                "what_has_to_happen": "Only attractive if the thesis shifts toward slower upside and income matters more.",
                "upgrade_rule": "Upgrade only if the thesis shifts toward slower upside and income matters more.",
                "what_would_invalidate": "Avoid if you still want uncapped upside.",
                "invalidate_rule": "Avoid if you still want uncapped upside.",
                "key_trigger_label": "Thesis Confirmation",
                "key_trigger_type": "thesis_confirmation",
                "key_trigger_value": "Only attractive if the thesis shifts toward slower upside and income matters more.",
                "key_trigger_deadline": "2026-07-15",
                "main_warning": "Caps upside too much for the primary bullish thesis.",
            },
        ],
    )
    for filename in [
        "watchlist_candidates.csv",
        "prefer_stock_instead.csv",
        "decision_triggers.csv",
        "action_board_score_breakdown.csv",
        "action_board_explanations.csv",
    ]:
        _write_csv(
            tables_dir / filename,
            [
                {
                    "action_bucket": "Watchlist",
                    "action_priority_rank": "1",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "strategy_family": "long_call",
                    "action_confidence": "medium",
                    "trigger_type_label": "Better After IV Cools",
                    "key_trigger_type": "iv_normalization_entry",
                    "key_trigger_value": "More attractive after IV cools or premium resets lower.",
                    "key_trigger_deadline": "2026-07-15",
                    "what_has_to_happen": "More attractive after IV cools or premium resets lower.",
                    "upgrade_rule": "Upgrade if premium cools below the current reference or IV normalizes after the event.",
                    "what_would_invalidate": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "invalidate_rule": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "main_warning": "Premium too demanding under current path. Benefits from lower-IV entry.",
                    "source_trust_label": "quoted_prior_day",
                    "main_trigger": "More attractive after IV cools or premium resets lower.",
                    "why_this_is_interesting_now": "High convexity if the stock path works.",
                    "what_is_hurting_this_candidate": "Premium too demanding under current path. Benefits from lower-IV entry.",
                    "component": "action_score",
                    "component_score": "58.4",
                    "component_note": "Final transparent action-board score used for bucket ordering.",
                    "headline_reason": "Interesting, but it needs a trigger before it is buyable.",
                }
            ],
        )
    for filename in ["buy_now_candidates.csv", "avoid_for_now_candidates.csv"]:
        _write_csv(
            tables_dir / filename,
            [
                {
                    "action_bucket": "Watchlist",
                    "action_priority_rank": "1",
                    "candidate_label": "No rows",
                    "strategy_family": "n/a",
                    "action_confidence": "",
                    "headline_reason": "No candidates in this bucket under current assumptions.",
                }
            ],
        )
    for filename in [
        "bullish_long_call_action_board.csv",
        "bullish_long_call_watchlist.csv",
        "bullish_long_call_avoid.csv",
        "bullish_long_call_triggers.csv",
        "bullish_long_call_score_breakdown.csv",
        "other_structures_summary.csv",
        "stock_preference_summary.csv",
    ]:
        if filename == "bullish_long_call_action_board.csv":
            rows = [
                {
                    "action_bucket": "Watchlist",
                    "action_priority_rank": "1",
                    "action_confidence": "medium",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "expiry_date": "2026-12-18",
                    "strike_label": "15.00",
                    "moneyness_bucket": "atm",
                    "source_trust_label": "quoted_prior_day",
                    "headline_reason": "Interesting, but it needs a trigger before it is buyable.",
                    "why_this_is_interesting_now": "High convexity if the stock path works.",
                    "what_is_hurting_this_candidate": "Premium too demanding under current path. Benefits from lower-IV entry.",
                    "main_trigger": "More attractive after IV cools or premium resets lower.",
                    "upgrade_rule": "Upgrade if premium cools below the current reference or IV normalizes after the event.",
                    "invalidate_rule": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "main_warning": "Premium too demanding under current path. Benefits from lower-IV entry.",
                    "action_score": "58.4",
                    "robustness_score": "55.0",
                    "stock_relative_score": "42.0",
                    "key_trigger_label": "Better After IV Cools",
                    "key_trigger_value": "More attractive after IV cools or premium resets lower.",
                    "key_trigger_deadline": "2026-07-15",
                },
                {
                    "action_bucket": "Avoid For Now",
                    "action_priority_rank": "1",
                    "action_confidence": "cautious",
                    "candidate_label": "Long Call 2026-05-15 18.00",
                    "expiry_date": "2026-05-15",
                    "strike_label": "18.00",
                    "moneyness_bucket": "otm",
                    "source_trust_label": "fallback_only",
                    "headline_reason": "Does not clear the current action threshold.",
                    "why_this_is_interesting_now": "High convexity only if the move arrives very quickly.",
                    "what_is_hurting_this_candidate": "Needs an earlier move before theta and timing drag bite too hard.",
                    "main_trigger": "Prefer a later expiry unless the stock confirms before this contract loses runway.",
                    "upgrade_rule": "Prefer a later expiry unless the stock confirms before this contract loses runway.",
                    "invalidate_rule": "Avoid if the stock is still waiting late into the contract window.",
                    "main_warning": "Needs an earlier move before theta and timing drag bite too hard.",
                    "action_score": "28.0",
                    "robustness_score": "20.0",
                    "stock_relative_score": "18.0",
                    "key_trigger_label": "Later Expiry / Timing",
                    "key_trigger_value": "Prefer a later expiry unless the stock confirms before this contract loses runway.",
                    "key_trigger_deadline": "2026-05-15",
                },
            ]
        elif filename == "bullish_long_call_watchlist.csv":
            rows = [
                {
                    "action_priority_rank": "1",
                    "action_confidence": "medium",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "expiry_date": "2026-12-18",
                    "strike_label": "15.00",
                    "moneyness_bucket": "atm",
                    "source_trust_label": "quoted_prior_day",
                    "why_this_is_interesting_now": "High convexity if the stock path works.",
                    "why_watch_not_buy": "Premium too demanding under current path. Trigger before buying: More attractive after IV cools or premium resets lower.",
                    "main_trigger": "More attractive after IV cools or premium resets lower.",
                    "what_has_to_happen": "More attractive after IV cools or premium resets lower.",
                    "upgrade_rule": "Upgrade if premium cools below the current reference or IV normalizes after the event.",
                    "what_would_invalidate": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "invalidate_rule": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "main_warning": "Premium too demanding under current path. Benefits from lower-IV entry.",
                    "action_score": "58.4",
                    "robustness_score": "55.0",
                    "stock_relative_score": "42.0",
                }
            ]
        elif filename == "bullish_long_call_avoid.csv":
            rows = [
                {
                    "action_priority_rank": "1",
                    "action_confidence": "cautious",
                    "candidate_label": "Long Call 2026-05-15 18.00",
                    "expiry_date": "2026-05-15",
                    "strike_label": "18.00",
                    "moneyness_bucket": "otm",
                    "source_trust_label": "fallback_only",
                    "why_avoid": "Needs an earlier move before theta and timing drag bite too hard. Keep it out of the active shortlist unless the trigger meaningfully improves.",
                    "what_is_hurting_this_candidate": "Needs an earlier move before theta and timing drag bite too hard.",
                    "main_warning": "Needs an earlier move before theta and timing drag bite too hard.",
                    "upgrade_rule": "Prefer a later expiry unless the stock confirms before this contract loses runway.",
                    "what_would_invalidate": "Avoid if the stock is still waiting late into the contract window.",
                    "invalidate_rule": "Avoid if the stock is still waiting late into the contract window.",
                    "action_score": "28.0",
                    "time_decay_risk": "82.0",
                    "iv_dependence_risk": "60.0",
                    "trust_penalty": "65.0",
                }
            ]
        elif filename == "bullish_long_call_triggers.csv":
            rows = [
                {
                    "action_bucket": "Watchlist",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "trigger_type_label": "Better After IV Cools",
                    "key_trigger_type": "iv_normalization_entry",
                    "upgrade_rule": "Upgrade if premium cools below the current reference or IV normalizes after the event.",
                    "what_has_to_happen": "More attractive after IV cools or premium resets lower.",
                    "key_trigger_deadline": "2026-07-15",
                    "invalidate_rule": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "what_would_invalidate": "Avoid if stock stays below the confirmation level while theta keeps working.",
                    "main_warning": "Premium too demanding under current path. Benefits from lower-IV entry.",
                    "source_trust_label": "quoted_prior_day",
                    "action_confidence": "medium",
                }
            ]
        elif filename == "bullish_long_call_score_breakdown.csv":
            rows = [
                {
                    "action_bucket": "Watchlist",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                    "component": "action_score",
                    "component_score": "58.4",
                    "component_note": "Final transparent action-board score used for bucket ordering.",
                }
            ]
        elif filename == "other_structures_summary.csv":
            rows = [
                {
                    "action_bucket": "Watchlist",
                    "action_priority_rank": "1",
                    "candidate_label": "Covered Call 2026-12-18 18.00",
                    "strategy_family": "covered_call",
                    "action_confidence": "cautious",
                    "headline_reason": "Interesting as secondary yield, but stock still cleaner for upside.",
                    "why_this_is_interesting_now": "Income-style way to get paid while waiting.",
                    "what_is_hurting_this_candidate": "Caps upside too much for the primary bullish thesis.",
                    "main_trigger": "Only attractive if the thesis shifts toward slower upside and income matters more.",
                    "main_warning": "Caps upside too much for the primary bullish thesis.",
                    "source_trust_label": "quoted_prior_day",
                }
            ]
        else:
            rows = [
                {
                    "action_bucket": "Prefer Stock Instead",
                    "action_priority_rank": "1",
                    "candidate_label": "Long Stock Baseline",
                    "strategy_family": "long_stock",
                    "action_confidence": "high",
                    "headline_reason": "Stock is the cleanest exposure under current assumptions.",
                    "why_stock_may_be_better": "No expiry, strike, premium, or IV-path dependency.",
                    "stock_relative_score": "50.0",
                    "difference_vs_stock": "0.0",
                    "robustness_score": "80.0",
                    "main_warning": "main assumptions remain the controlling risk",
                    "source_trust_label": "trusted_quoted",
                }
            ]
        _write_csv(tables_dir / filename, rows)

    _write_csv(
        tables_dir / "top_candidate_cards.csv",
        [
            {
                "card_rank": "1",
                "contract_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "bucket": "Watchlist",
                "confidence": "medium",
                "why_this_is_interesting": "High convexity if the stock path works.",
                "what_hurts_it": "Premium too demanding under current path. Benefits from lower-IV entry.",
                "main_trigger": "More attractive after IV cools or premium resets lower.",
                "upgrade_rule": "Upgrade if premium cools below the current reference or IV normalizes after the event.",
                "invalidate_rule": "Avoid if stock stays below the confirmation level while theta keeps working.",
                "trust": "quoted_prior_day",
                "compare_vs_stock_note": "Long stock still cleaner until option edge improves.",
                "action_score": "58.4",
            },
            {
                "card_rank": "2",
                "contract_label": "18C May-26",
                "candidate_label": "Long Call 2026-05-15 18.00",
                "bucket": "Avoid For Now",
                "confidence": "cautious",
                "why_this_is_interesting": "High convexity only if the move arrives very quickly.",
                "what_hurts_it": "Needs an earlier move before theta and timing drag bite too hard.",
                "main_trigger": "Prefer a later expiry unless the stock confirms before this contract loses runway.",
                "upgrade_rule": "Prefer a later expiry unless the stock confirms before this contract loses runway.",
                "invalidate_rule": "Avoid if the stock is still waiting late into the contract window.",
                "trust": "fallback_only",
                "compare_vs_stock_note": "Long stock still cleaner until option edge improves.",
                "action_score": "28.0",
            },
        ],
    )

    _write_csv(
        tables_dir / "entry_justification_candidates.csv",
        [
            {
                "entry_display_rank": "1",
                "action_bucket": "Watchlist",
                "action_priority_rank": "1",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "moneyness_bucket": "atm",
                "source_trust_label": "quoted_prior_day",
                "required_price_1m": "18.40",
                "required_price_target": "20.10",
                "required_move_pct_target": "24.0",
                "timing_window_days": "94",
                "move_pace_pct_per_month": "7.7",
                "required_path_difficulty": "roughly matched",
                "first_cleared_horizon": "target",
                "requires_fast_move": "False",
                "needs_iv_support": "True",
                "stock_still_better_even_if_path_hits": "True",
                "iv_requirement_label": "needs IV support",
                "entry_barrier_score": "54.2",
                "entry_barrier_label": "stock still better",
                "what_has_to_happen": "Needs about $18.40 by 1 month and $20.10 by 2026-07-15.",
                "entry_warning": "Stock still looks cleaner even if the required path is achieved.",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
            {
                "entry_display_rank": "2",
                "action_bucket": "Avoid For Now",
                "action_priority_rank": "1",
                "candidate_short_label": "18C May-26",
                "candidate_label": "Long Call 2026-05-15 18.00",
                "expiry_date": "2026-05-15",
                "strike_label": "18.00",
                "moneyness_bucket": "otm",
                "source_trust_label": "fallback_only",
                "required_price_1m": "21.20",
                "required_price_target": "21.60",
                "required_move_pct_target": "41.8",
                "timing_window_days": "23",
                "move_pace_pct_per_month": "54.5",
                "required_path_difficulty": "needs more / faster",
                "first_cleared_horizon": "not cleared",
                "requires_fast_move": "True",
                "needs_iv_support": "True",
                "stock_still_better_even_if_path_hits": "True",
                "iv_requirement_label": "needs IV support",
                "entry_barrier_score": "88.0",
                "entry_barrier_label": "too demanding",
                "what_has_to_happen": "Needs about $21.20 by 1 month and $21.60 by 2026-05-15. The move probably needs to start early enough to outrun theta.",
                "entry_warning": "A slower move quickly weakens the call because timing and theta matter.",
                "stock_vs_option_read": "The path helps, but stock still keeps the cleaner baseline read.",
            },
        ],
    )
    _write_csv(
        tables_dir / "required_stock_path_to_buy.csv",
        [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "action_bucket": "Watchlist",
                "entry_display_rank": "1",
                "iv_path_label": "Flat",
                "series_kind": "required_path",
                "date": snapshot_date,
                "requested_days": "0",
                "stock_price": "15.23",
                "entry_barrier_label": "stock still better",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "action_bucket": "Watchlist",
                "entry_display_rank": "1",
                "iv_path_label": "Flat",
                "series_kind": "required_path",
                "date": "2026-05-12",
                "requested_days": "30",
                "stock_price": "18.40",
                "entry_barrier_label": "stock still better",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "action_bucket": "Watchlist",
                "entry_display_rank": "1",
                "iv_path_label": "Flat",
                "series_kind": "required_path",
                "date": "2026-07-15",
                "requested_days": "94",
                "stock_price": "20.10",
                "entry_barrier_label": "stock still better",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "action_bucket": "Watchlist",
                "entry_display_rank": "1",
                "iv_path_label": "Flat",
                "series_kind": "assumed_path",
                "date": snapshot_date,
                "requested_days": "0",
                "stock_price": "15.23",
                "entry_barrier_label": "stock still better",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "action_bucket": "Watchlist",
                "entry_display_rank": "1",
                "iv_path_label": "Flat",
                "series_kind": "assumed_path",
                "date": "2026-05-12",
                "requested_days": "30",
                "stock_price": "17.10",
                "entry_barrier_label": "stock still better",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "action_bucket": "Watchlist",
                "entry_display_rank": "1",
                "iv_path_label": "Flat",
                "series_kind": "assumed_path",
                "date": "2026-07-15",
                "requested_days": "94",
                "stock_price": "20.00",
                "entry_barrier_label": "stock still better",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            },
        ],
    )
    for filename, rows in {
        "required_move_summary.csv": [
            {
                "action_bucket": "Watchlist",
                "action_priority_rank": "1",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "moneyness_bucket": "atm",
                "required_path_difficulty": "roughly matched",
                "first_cleared_horizon": "target",
                "required_move_pct_1m": "20.8",
                "required_move_pct_3m": "24.0",
                "required_move_pct_target": "24.0",
                "timing_window_days": "94",
                "move_pace_pct_per_month": "7.7",
                "requires_fast_move": "False",
                "stock_still_better_even_if_path_hits": "True",
                "entry_barrier_score": "54.2",
                "what_has_to_happen": "Needs about $18.40 by 1 month and $20.10 by 2026-07-15.",
                "entry_warning": "Stock still looks cleaner even if the required path is achieved.",
                "source_trust_label": "quoted_prior_day",
            }
        ],
        "required_move_vs_stock.csv": [
            {
                "action_bucket": "Watchlist",
                "action_priority_rank": "1",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "required_move_pct_target": "24.0",
                "timing_window_days": "94",
                "assumed_clears_required_at_target": "False",
                "difference_vs_stock": "-120.0",
                "difference_vs_stock_return_pct": "-12.0",
                "stock_relative_score": "42.0",
                "stock_still_better_even_if_path_hits": "True",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
            }
        ],
        "required_iv_support_summary.csv": [
            {
                "action_bucket": "Watchlist",
                "action_priority_rank": "1",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "required_move_pct_flat_iv": "24.0",
                "lower_iv_required_move_pct": "29.5",
                "higher_iv_required_move_pct": "21.0",
                "lower_iv_move_penalty_pct": "5.5",
                "higher_iv_move_relief_pct": "3.0",
                "lower_iv_resilience_score": "48.0",
                "iv_dependence_risk": "52.0",
                "iv_requirement_label": "needs IV support",
                "iv_requirement_note": "Lower IV raises the required target move from +24.0% to +29.5%.",
            }
        ],
        "entry_barrier_summary.csv": [
            {
                "action_bucket": "Watchlist",
                "action_priority_rank": "1",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "moneyness_bucket": "atm",
                "entry_barrier_score": "54.2",
                "entry_barrier_label": "stock still better",
                "required_move_pct_target": "24.0",
                "timing_window_days": "94",
                "iv_requirement_label": "needs IV support",
                "stock_vs_option_read": "Even if the required path is achieved under the active thesis, stock still looks cleaner after premium.",
                "source_trust_label": "quoted_prior_day",
            }
        ],
    }.items():
        _write_csv(tables_dir / filename, rows)

    for filename, rows in {
        "thesis_path_gallery.csv": [
            {
                "path_family": "early_breakout_to_target",
                "path_label": "Early Breakout To Target",
                "display_order": "1",
                "date": snapshot_date,
                "requested_days": "0",
                "stock_price": "15.23",
                "target_price": "30.0",
                "target_date": "2026-12-18",
            },
            {
                "path_family": "early_breakout_to_target",
                "path_label": "Early Breakout To Target",
                "display_order": "1",
                "date": "2026-12-18",
                "requested_days": "250",
                "stock_price": "30.0",
                "target_price": "30.0",
                "target_date": "2026-12-18",
            },
        ],
        "thesis_iv_gallery.csv": [
            {
                "iv_path_name": "flat",
                "iv_path_label": "Flat",
                "display_order": "1",
                "date": snapshot_date,
                "requested_days": "0",
                "iv_shift_points": "0.0",
                "target_date": "2026-12-18",
            },
            {
                "iv_path_name": "mean_reversion_lower",
                "iv_path_label": "Mean Rev Lower",
                "display_order": "2",
                "date": "2026-12-18",
                "requested_days": "250",
                "iv_shift_points": "-0.10",
                "target_date": "2026-12-18",
            },
        ],
        "thesis_mode_candidates.csv": [
            {
                "thesis_target_price": "30.0",
                "thesis_target_date": "2026-12-18",
                "path_family": "early_breakout_to_target",
                "path_label": "Early Breakout To Target",
                "iv_path_name": "flat",
                "iv_path_label": "Flat",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "candidate_short_label": "15C Dec-26",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "moneyness_bucket": "atm",
                "source_trust_label": "quoted_prior_day",
                "current_premium": "320.0",
                "thesis_terminal_value": "1500.0",
                "profit_loss": "1180.0",
                "difference_vs_stock": "-20.0",
                "stock_still_better": "True",
                "stock_relative_justified_premium": "300.0",
            }
        ],
        "thesis_path_family_summary.csv": [
            {
                "path_family": "early_breakout_to_target",
                "path_label": "Early Breakout To Target",
                "target_price": "30.0",
                "target_date": "2026-12-18",
                "average_candidate_profit_loss": "1180.0",
                "beat_stock_rate": "0.25",
                "best_candidate_slug": "long-call-2026-12-18-15-00",
                "best_candidate_short_label": "15C Dec-26",
                "path_effect_note": "Fast/early target path helps calls more.",
            }
        ],
        "thesis_iv_family_summary.csv": [
            {
                "iv_path_name": "flat",
                "iv_path_label": "Flat",
                "average_candidate_profit_loss": "1180.0",
                "beat_stock_rate": "0.25",
                "iv_effect_note": "Flat IV is the neutral comparison regime.",
            }
        ],
        "thesis_candidate_ranking.csv": [
            {
                "thesis_candidate_rank": "1",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "candidate_short_label": "15C Dec-26",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "moneyness_bucket": "atm",
                "source_trust_label": "quoted_prior_day",
                "current_premium": "320.0",
                "max_justified_premium": "300.0",
                "premium_gap": "-20.0",
                "premium_gap_pct": "-0.0625",
                "entry_attractiveness_status": "near_watchlist_under_thesis",
                "profitable_scenario_rate": "1.0",
                "beats_stock_scenario_rate": "0.25",
                "difference_vs_stock_median": "-20.0",
                "path_sensitivity_label": "path_sensitive",
                "iv_sensitivity_label": "iv_sensitive",
                "stock_still_better_under_thesis": "True",
                "main_reason": "Stock still looks cleaner because option edge is too narrow versus the benchmark.",
            }
        ],
        "max_justified_premium.csv": [
            {
                "thesis_candidate_rank": "1",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "current_premium": "320.0",
                "max_justified_premium": "300.0",
                "premium_gap": "-20.0",
                "premium_gap_pct": "-0.0625",
                "entry_attractiveness_status": "near_watchlist_under_thesis",
                "main_reason": "Stock still looks cleaner because option edge is too narrow versus the benchmark.",
            }
        ],
        "current_vs_justified_premium.csv": [
            {
                "thesis_candidate_rank": "1",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "current_premium": "320.0",
                "max_justified_premium": "300.0",
                "premium_gap": "-20.0",
                "premium_gap_pct": "-0.0625",
                "entry_attractiveness_status": "near_watchlist_under_thesis",
                "main_reason": "Stock still looks cleaner because option edge is too narrow versus the benchmark.",
            }
        ],
        "thesis_required_move_summary.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "thesis_target_price": "30.0",
                "thesis_target_date": "2026-12-18",
                "days_to_target": "250",
                "required_total_upside_pct": "0.97",
                "required_monthly_pace_pct": "0.116",
                "required_timing_window": "fast confirmation preferred",
                "entry_attractiveness_status": "near_watchlist_under_thesis",
                "timing_note": "A slow path can still leave stock cleaner if premium decay absorbs the target move.",
            }
        ],
        "thesis_stock_vs_option_summary.csv": [
            {
                "thesis_candidate_rank": "1",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "entry_attractiveness_status": "near_watchlist_under_thesis",
                "beats_stock_scenario_rate": "0.25",
                "difference_vs_stock_median": "-20.0",
                "stock_still_better_under_thesis": "True",
                "main_reason": "Stock still looks cleaner because option edge is too narrow versus the benchmark.",
            }
        ],
        "candidate_stress_grid.csv": [
            {
                "candidate_rank": "1",
                "candidate_short_label": "15C Dec-26",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "metric": "action bucket",
                "Base": "Watchlist",
                "Premium -10%": "Watchlist",
                "Premium -20%": "Buy Now",
                "Premium +10%": "Prefer Stock Instead",
                "Move delayed 2w": "Watchlist",
                "Move delayed 1m": "Watchlist",
                "Move delayed 2m": "Avoid For Now",
                "Undershoot to 26": "Avoid For Now",
                "Base hit at 30": "Watchlist",
                "Overshoot to 35 then settle": "Buy Now",
            }
        ],
        "premium_sensitivity_summary.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "scenario_name": "premium_minus_20",
                "scenario_label": "Premium -20%",
                "scenario_order": "3",
                "action_bucket": "Buy Now",
                "bucket_transition": "upgrade",
                "option_vs_stock_edge_pct": "4.4",
                "max_justified_premium_gap": "44.0",
                "scenario_premium": "256.0",
                "premium_multiplier": "0.8",
                "main_note": "Entry price is the key blocker; lower premium can upgrade the setup.",
                "source_trust_label": "quoted_prior_day",
            }
        ],
        "timing_slip_summary.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "scenario_name": "delay_2m",
                "scenario_label": "Move delayed 2m",
                "scenario_order": "4",
                "action_bucket": "Avoid For Now",
                "bucket_transition": "weaker",
                "option_vs_stock_edge_pct": "-14.0",
                "max_justified_premium_gap": "-140.0",
                "delay_days": "60",
                "delayed_target_date": "2027-02-16",
                "target_beyond_expiry_under_delay": "True",
                "main_note": "Delayed move weakens the call; theta/timing is a real blocker.",
                "source_trust_label": "quoted_prior_day",
            }
        ],
        "target_stress_summary.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "scenario_name": "overshoot_settle",
                "scenario_label": "Overshoot to 35 then settle",
                "scenario_order": "3",
                "action_bucket": "Buy Now",
                "bucket_transition": "upgrade",
                "option_vs_stock_edge_pct": "18.0",
                "max_justified_premium_gap": "90.0",
                "target_price": "35.0",
                "intrinsic_value_at_target": "2000.0",
                "main_note": "This call needs overshoot/stronger convexity to become more compelling.",
                "source_trust_label": "quoted_prior_day",
            }
        ],
        "stress_transition_summary.csv": [
            {
                "stress_rank": "1",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "candidate_label": "Long Call 2026-12-18 15.00",
                "expiry_date": "2026-12-18",
                "strike_label": "15.00",
                "source_trust_label": "quoted_prior_day",
                "base_action_bucket": "Watchlist",
                "base_option_vs_stock_edge_pct": "-2.0",
                "base_max_justified_premium_gap": "-20.0",
                "best_improving_stress": "Overshoot to 35 then settle",
                "best_improving_bucket": "Buy Now",
                "best_improving_edge_pct": "18.0",
                "worst_breaking_stress": "Move delayed 2m",
                "worst_breaking_bucket": "Avoid For Now",
                "worst_breaking_edge_pct": "-14.0",
                "stress_resilience_score": "0.7",
                "stress_buy_count": "2",
                "premium_sensitivity_read": "entry_price_can_upgrade",
                "timing_sensitivity_read": "breaks_if_delayed",
                "target_dependency_read": "needs_overshoot",
                "stress_card_note": "Looks most interesting if entry premium cools.",
                "main_warning": "Premium too demanding under current path.",
                "upgrade_rule": "Upgrade if premium cools below the current reference.",
            }
        ],
        "single_option_decision_summary.csv": [
            {
                "ticker": "GPRE",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "premium_used": "320.0",
                "entry_price_mode": "conservative_mid_plus_slippage",
                "base_iv": "0.55",
                "breakeven": "18.20",
                "max_loss": "320.0",
                "dte": "250",
                "exit_rule": "sell_on_thesis_completion",
                "single_option_decision_status": "too_narrow_under_representative_paths",
                "required_winning_path_families": "2",
            }
        ],
        "single_option_decision_path_selections.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "decision_path_id": "late_rally_path__late_breakout_to_target",
                "path_role": "late_rally_path",
                "path_name": "late_breakout_to_target",
                "path_label": "Late Breakout To Target",
                "path_family": "late_rally",
                "path_family_label": "Late Rally",
                "timing_shape": "back_loaded_upside",
                "outcome_label": "stock_better",
                "selection_score": "80.0",
                "selection_reason": "Representative stock-better path.",
                "difference_vs_stock": "-35.0",
            }
        ],
        "single_option_representative_paths.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "decision_path_id": "late_rally_path__late_breakout_to_target",
                "path_role": "late_rally_path",
                "path_name": "late_breakout_to_target",
                "path_label": "Late Breakout",
                "path_family": "late_rally",
                "path_family_label": "Late Rally",
                "timing_shape": "back_loaded_upside",
                "display_order": "1",
                "date": "2026-12-18",
                "requested_days": "250",
                "spot_price": "30.0",
            }
        ],
        "single_option_path_outcomes.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "candidate_short_label": "15C Dec-26",
                "decision_path_id": "late_rally_path__late_breakout_to_target",
                "path_role": "late_rally_path",
                "path_name": "late_breakout_to_target",
                "path_label": "Late Breakout",
                "path_family": "late_rally",
                "path_family_label": "Late Rally",
                "timing_shape": "back_loaded_upside",
                "outcome_label": "stock_better",
                "difference_vs_stock": "-35.0",
                "stock_profit_loss": "1200.0",
                "outperformance_multiple": "0.9",
                "outcome_note": "Stock is cleaner on this path after option premium.",
            }
        ],
        "single_option_path_family_counts.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "evaluated_path_family_count": "6",
                "qualifying_path_family_count": "1",
                "required_winning_path_families": "2",
                "too_narrow_under_representative_paths": "True",
            }
        ],
        "single_option_timing_sensitivity.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "path_label": "Late Breakout",
                "timing_read": "late_move_favors_stock",
            }
        ],
        "single_option_iv_sensitivity.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "iv_mode_label": "Low IV",
                "display_order": "1",
                "difference_vs_stock": "-80.0",
                "sensitivity_note": "Lower IV hurts this call if value falls versus base.",
            }
        ],
        "single_option_entry_sensitivity.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "entry_scenario_label": "Cheap fill (-10%)",
                "display_order": "1",
                "premium_used": "288.0",
                "average_difference_vs_stock": "-20.0",
                "entry_read": "Cheaper entry meaningfully improves the setup.",
            }
        ],
        "single_option_summary_bullets.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "bullet_order": "1",
                "bullet_text": "Beats stock in 1 of 6 representative path families.",
            }
        ],
        "chain_overview_summary.csv": [
            {
                "card_key": "best_robust_option",
                "card_label": "Best Robust Option",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "contract_label": "15C Dec-26",
                "verdict_badge": "Selective / thesis-dependent",
                "headline_metric": "2/6 wins",
                "headline_note": "Needs a cooler entry premium.",
                "explanation_short": "Has upside, but still needs the right path, timing, or entry.",
            },
            {
                "card_key": "best_asymmetric_upside",
                "card_label": "Best Asymmetric Upside",
                "candidate_slug": "long-call-2026-12-18-16-00",
                "contract_label": "16C Dec-26",
                "verdict_badge": "Too narrow",
                "headline_metric": "Asymmetry 81",
                "headline_note": "High convexity, but path support is thin.",
                "explanation_short": "Needs a narrower path or more precise timing than a robust buy should.",
            },
            {
                "card_key": "best_early_move_option",
                "card_label": "Best Early-Move Option",
                "candidate_slug": "long-call-2026-12-18-15-00",
                "contract_label": "15C Dec-26",
                "verdict_badge": "Selective / thesis-dependent",
                "headline_metric": "+$48",
                "headline_note": "Best under early breakout and steady grind paths.",
                "explanation_short": "Best under early breakout and steady grind paths.",
            },
            {
                "card_key": "best_late_move_option",
                "card_label": "Best Late-Move Option",
                "candidate_slug": "long-call-2026-12-18-14-00",
                "contract_label": "14C Dec-26",
                "verdict_badge": "Selective / thesis-dependent",
                "headline_metric": "+$22",
                "headline_note": "Longer expiry makes the late move less fragile.",
                "explanation_short": "Has upside, but still needs the right path, timing, or entry.",
            },
            {
                "card_key": "too_iv_sensitive",
                "card_label": "Too IV-Sensitive",
                "candidate_slug": "long-call-2026-05-15-18-00",
                "contract_label": "18C May-26",
                "verdict_badge": "Too narrow",
                "headline_metric": "IV sensitivity 88",
                "headline_note": "Too IV-sensitive for a robust buy read.",
                "explanation_short": "Too IV-sensitive for a robust buy read.",
            },
            {
                "card_key": "stock_better_than_these_calls",
                "card_label": "Stock Better Than These Calls",
                "candidate_slug": "long-stock-baseline",
                "contract_label": "Long Stock Baseline",
                "verdict_badge": "Stock better",
                "headline_metric": "2/3 calls",
                "headline_note": "Stock stays cleaner than most calls under representative paths.",
                "explanation_short": "Stock remains the explicit benchmark; the call table only shows where options truly justify extra complexity.",
            },
        ],
        "chain_overview_candidates.csv": [
            {
                "candidate_slug": "long-call-2026-12-18-15-00",
                "contract": "15C Dec-26",
                "premium": "320.0",
                "iv": "0.55",
                "dte": "250",
                "beats_stock_label": "2/6",
                "beats_stock_count": "2",
                "strong_wins": "1",
                "strong_outperformance_count": "1",
                "robustness": "Moderate",
                "robustness_score": "63.0",
                "iv_sensitivity": "Moderate",
                "iv_sensitivity_score": "58.0",
                "entry_sensitivity": "High",
                "entry_premium_sensitivity_score": "78.0",
                "best_fit_path_type": "Early Breakout",
                "worth_buying_status": "selective",
                "final_verdict": "Selective / thesis-dependent",
                "why_short": "Needs a cooler entry premium.",
                "why_detail": "Has upside, but still needs the right path, timing, or entry. Needs a cooler entry premium.",
                "shared_path_family_count": "6",
                "minimum_outperformance_multiple": "1.5",
                "strong_outperformance_multiple": "2.0",
                "required_winning_path_families": "2",
            },
            {
                "candidate_slug": "long-call-2026-12-18-16-00",
                "contract": "16C Dec-26",
                "premium": "250.0",
                "iv": "0.57",
                "dte": "250",
                "beats_stock_label": "1/6",
                "beats_stock_count": "1",
                "strong_wins": "1",
                "strong_outperformance_count": "1",
                "robustness": "Low",
                "robustness_score": "42.0",
                "iv_sensitivity": "",
                "iv_sensitivity_score": "",
                "entry_sensitivity": "Moderate",
                "entry_premium_sensitivity_score": "64.0",
                "best_fit_path_type": "Early Breakout",
                "worth_buying_status": "too_narrow",
                "final_verdict": "Too narrow",
                "why_short": "Needs a narrower path or more precise timing than a robust buy should.",
                "why_detail": "Needs a narrower path or more precise timing than a robust buy should.",
                "shared_path_family_count": "6",
                "minimum_outperformance_multiple": "1.5",
                "strong_outperformance_multiple": "2.0",
                "required_winning_path_families": "2",
            },
            {
                "candidate_slug": "long-call-2026-05-15-18-00",
                "contract": "18C May-26",
                "premium": "110.0",
                "iv": "0.64",
                "dte": "33",
                "beats_stock_label": "0/6",
                "beats_stock_count": "0",
                "strong_wins": "0",
                "strong_outperformance_count": "0",
                "robustness": "Low",
                "robustness_score": "24.0",
                "iv_sensitivity": "High",
                "iv_sensitivity_score": "88.0",
                "entry_sensitivity": "High",
                "entry_premium_sensitivity_score": "82.0",
                "best_fit_path_type": "Early Breakout",
                "worth_buying_status": "stock_better",
                "final_verdict": "Stock better",
                "why_short": "Stock remains cleaner under representative paths.",
                "why_detail": "Stock remains cleaner under representative paths. Too IV-sensitive for a robust buy read.",
                "shared_path_family_count": "6",
                "minimum_outperformance_multiple": "1.5",
                "strong_outperformance_multiple": "2.0",
                "required_winning_path_families": "2",
            },
        ],
    }.items():
        _write_csv(tables_dir / filename, rows)

    for path_name in [
        "late_breakout",
        "range_bound_near_flat",
    ]:
        for filename in [
            _path_view_filename(path_name, "compare_vs_stock_path_rows.csv"),
            _path_view_filename(path_name, "long_call_strike_value.csv"),
            _path_view_filename(path_name, "long_call_strike_delta.csv"),
            _path_view_filename(path_name, "long_call_expiry_value.csv"),
            _path_view_filename(path_name, "long_call_expiry_delta.csv"),
            _path_view_filename(path_name, "long_call_best_of_value.csv"),
            _path_view_filename(path_name, "long_call_best_of_delta.csv"),
            _path_view_filename(path_name, "path_checkpoints.csv"),
            _path_view_filename(path_name, "iv_path_value.csv"),
            _path_view_filename(path_name, "iv_path_delta.csv"),
            _path_view_filename(path_name, "iv_checkpoints.csv"),
            _path_view_filename(path_name, "long_call_strike_iv_value.csv"),
            _path_view_filename(path_name, "long_call_strike_iv_delta.csv"),
            _path_view_filename(path_name, "long_call_strike_iv_checkpoints.csv"),
            _path_view_filename(path_name, "long_call_expiry_iv_value.csv"),
            _path_view_filename(path_name, "long_call_expiry_iv_delta.csv"),
            _path_view_filename(path_name, "long_call_expiry_iv_checkpoints.csv"),
            _path_view_filename(path_name, "long_call_best_of_iv_value.csv"),
            _path_view_filename(path_name, "long_call_best_of_iv_delta.csv"),
            _path_view_filename(path_name, "long_call_best_of_iv_checkpoints.csv"),
            _path_view_filename(path_name, "iv_robustness_summary.csv"),
        ]:
            _write_csv(
                tables_dir / filename,
                [
                    {
                        "stock_path_name": path_name,
                        "iv_path_name": "flat",
                        "candidate_label": "Long Call 2026-12-18 15.00",
                        "candidate_slug": "long-call-2026-12-18-15-00",
                        "series_label": "15C Dec-2026",
                        "selection_rank": "1",
                        "date": snapshot_date,
                        "requested_days": "0",
                        "profit_loss": "0.0",
                        "modeled_value": "120.0",
                        "delta_profit_loss_vs_stock": "0.0",
                        "delta_return_pct_vs_stock": "0.0",
                        "checkpoint_label": "entry",
                        "spot_price": "15.23",
                        "iv_shift_points": "0.0",
                        "iv_path_label": "Flat",
                        "anchor_contract_label": "15C Dec-2026",
                        "terminal_value_vs_flat": "0.0",
                        "terminal_delta_vs_flat": "0.0",
                        "iv_effect_note": "flat-IV baseline",
                        "iv_expanded_family": "strike",
                        "contract_rank": "1",
                        "contract_label": "15C Dec-2026",
                        "chart_include": "True",
                        "iv_chart_scope": "core_iv_chart",
                        "iv_robustness_label": "survives_lower_iv_but_stock_may_win",
                        "iv_robustness_note": "option value survives lower IV, but stock may still be cleaner",
                        "beat_stock_iv_path_count": "1",
                        "lower_iv_profitable": "True",
                    }
                ],
            )

    _write_csv(
        tables_dir / "path_case_rows.csv",
        [{"label": "noise", "value": "1"}],
    )

    for filename in [
        "action_board_overview.png",
        "bullish_action_board_overview.png",
        "conviction_vs_robustness.png",
        "bullish_conviction_vs_robustness.png",
        "buy_watch_avoid_matrix.png",
        "bullish_buy_watch_avoid_matrix.png",
        "trigger_map.png",
        "bullish_trigger_map.png",
        "top_candidate_cards.png",
        "stock_vs_option_preference_chart.png",
        "required_stock_path_to_buy.png",
        "required_move_speed_vs_magnitude.png",
        "required_move_vs_stock_chart.png",
        "strike_expiry_entry_barrier_map.png",
        "iv_support_requirement_chart.png",
        "thesis_path_gallery.png",
        "thesis_iv_gallery.png",
        "thesis_candidate_overview.png",
        "current_vs_justified_premium.png",
        "thesis_path_vs_value.png",
        "thesis_iv_vs_value.png",
        "thesis_stock_vs_option.png",
        "stress_test_overview.png",
        "premium_sensitivity_chart.png",
        "timing_slip_chart.png",
        "target_stress_chart.png",
        "top_candidate_stress_cards.png",
        "chain_overview.png",
        "single_option_decision_view.png",
        "highlights_overview.png",
        "candidate_robustness_vs_upside.png",
        "path_survival_scorecard.png",
        "iv_robustness_scorecard.png",
        "strike_expiry_tradeoff_overview.png",
        "stock_vs_option_decision_chart.png",
        "stock_path_gallery.png",
        "iv_path_gallery.png",
        "required_path_vs_assumed_path.png",
        "compare_vs_stock_path_delta.png",
        "representative_stock_paths.png",
        "representative_iv_paths.png",
        "option_value_over_path.png",
        "compare_vs_stock_over_path.png",
        "strike_comparison_under_same_path.png",
        "expiry_comparison_under_same_path.png",
        "long_call_value_over_path_strike_view.png",
        "long_call_value_over_path_expiry_view.png",
        "long_call_value_over_path_best_of.png",
    ]:
        _write_png_placeholder(charts_dir / filename)
    for path_name in [
        "late_breakout",
        "range_bound_near_flat",
    ]:
        for filename in [
            _path_view_filename(path_name, "compare_vs_stock_path_delta.png"),
            _path_view_filename(path_name, "long_call_strike_value.png"),
            _path_view_filename(path_name, "long_call_strike_delta.png"),
            _path_view_filename(path_name, "long_call_expiry_value.png"),
            _path_view_filename(path_name, "long_call_expiry_delta.png"),
            _path_view_filename(path_name, "long_call_best_of_value.png"),
            _path_view_filename(path_name, "long_call_best_of_delta.png"),
            _path_view_filename(path_name, "iv_path_value.png"),
            _path_view_filename(path_name, "iv_path_delta.png"),
            _path_view_filename(path_name, "long_call_strike_iv_value.png"),
            _path_view_filename(path_name, "long_call_strike_iv_delta.png"),
            _path_view_filename(path_name, "long_call_expiry_iv_value.png"),
            _path_view_filename(path_name, "long_call_expiry_iv_delta.png"),
            _path_view_filename(path_name, "long_call_best_of_iv_value.png"),
            _path_view_filename(path_name, "long_call_best_of_iv_delta.png"),
        ]:
            _write_png_placeholder(charts_dir / filename)
    _write_png_placeholder(charts_dir / "heatmap_strike_target_price.png")

    report_metadata = {
        "generated_at": "2026-04-20T04:00:00Z",
        "report_kind": "contract_selection",
        "analysis_name": "contract_selection",
        "ticker": "GPRE",
        "snapshot_date": snapshot_date,
        "warnings": [
            "Used a nearest local chain fallback for expiry 2026-05-15 because no usable same-day slice existed."
        ],
        "bullish_action_board": [
            {
                "action_bucket": "Watchlist",
                "candidate_label": "Long Call 2026-12-18 15.00",
            }
        ],
        "other_structures_summary": [
            {
                "action_bucket": "Watchlist",
                "candidate_label": "Covered Call 2026-12-18 18.00",
            }
        ],
        "metadata": {
            "goal": "break_even",
            "stock_path_name": "slow_bull",
            "iv_path_name": "flat",
            "spot_price_source": "nasdaq_historical_quotes",
            "spot_field_used": "close",
            "spot_price_matched_date": "2026-04-10",
            "spot_used_prior_date": True,
            "spot_quality_note": "Spot fell back to a prior-date local historical close.",
            "risk_free_rate_source": "fred_local_store",
            "risk_free_rate_series": "DGS3MO",
            "risk_free_rate_matched_date": "2026-04-10",
            "risk_free_rate_note": "Used the latest available prior Treasury observation.",
            "analysis_trust_level": "cautious",
            "analysis_trust_note": "The run mixes quoted expiries with sparse fallback expiries.",
            "bullish_action_board": [
                {
                    "action_bucket": "Watchlist",
                    "candidate_label": "Long Call 2026-12-18 15.00",
                }
            ],
            "trusted_expiry_count": 5,
            "fallback_only_expiry_count": 2,
            "ibkr_same_day_spot_rejected_reason": "No same-day delayed IBKR underlying snapshot for GPRE was available on 2026-04-12.",
        },
        "contract_selection": {
            "target_date": "2026-07-15",
        },
    }
    _write_text(metadata_dir / "report_metadata.json", json.dumps(report_metadata, indent=2))

    manifest = {
        "analysis_kind": "contract_selection",
        "bundle_version": 2,
        "ticker": "GPRE",
        "snapshot_date": snapshot_date,
        "run_slug": run_slug,
        "created_at": "2026-04-20T04:00:00Z",
        "file_map": {
            "summary": {
                "summary.md": "summary/summary.md",
                "highlights.md": "summary/highlights.md",
                "action_board.md": "summary/action_board.md",
                "bullish_action_board.md": "summary/bullish_action_board.md",
                "other_structures.md": "summary/other_structures.md",
                "entry_justification.md": "summary/entry_justification.md",
                "thesis_mode.md": "summary/thesis_mode.md",
                "stress_tests.md": "summary/stress_tests.md",
                "top_candidate_cards.md": "summary/top_candidate_cards.md",
                "chain_overview.md": "summary/chain_overview.md",
                "single_option_decision.md": "summary/single_option_decision.md",
            },
            "tables": {
                "summary.csv": "tables/summary.csv",
                "decision_highlights.csv": "tables/decision_highlights.csv",
                "decision_highlights_explanations.csv": "tables/decision_highlights_explanations.csv",
                "candidate_robustness_summary.csv": "tables/candidate_robustness_summary.csv",
                "candidate_tradeoff_matrix.csv": "tables/candidate_tradeoff_matrix.csv",
                "stock_vs_option_takeaways.csv": "tables/stock_vs_option_takeaways.csv",
                "highlights_score_breakdown.csv": "tables/highlights_score_breakdown.csv",
                "action_board_candidates.csv": "tables/action_board_candidates.csv",
                "buy_now_candidates.csv": "tables/buy_now_candidates.csv",
                "watchlist_candidates.csv": "tables/watchlist_candidates.csv",
                "avoid_for_now_candidates.csv": "tables/avoid_for_now_candidates.csv",
                "prefer_stock_instead.csv": "tables/prefer_stock_instead.csv",
                "decision_triggers.csv": "tables/decision_triggers.csv",
                "action_board_score_breakdown.csv": "tables/action_board_score_breakdown.csv",
                "action_board_explanations.csv": "tables/action_board_explanations.csv",
                "bullish_long_call_action_board.csv": "tables/bullish_long_call_action_board.csv",
                "bullish_long_call_watchlist.csv": "tables/bullish_long_call_watchlist.csv",
                "bullish_long_call_avoid.csv": "tables/bullish_long_call_avoid.csv",
                "bullish_long_call_triggers.csv": "tables/bullish_long_call_triggers.csv",
                "bullish_long_call_score_breakdown.csv": "tables/bullish_long_call_score_breakdown.csv",
                "top_candidate_cards.csv": "tables/top_candidate_cards.csv",
                "other_structures_summary.csv": "tables/other_structures_summary.csv",
                "stock_preference_summary.csv": "tables/stock_preference_summary.csv",
                "entry_justification_candidates.csv": "tables/entry_justification_candidates.csv",
                "required_stock_path_to_buy.csv": "tables/required_stock_path_to_buy.csv",
                "required_move_summary.csv": "tables/required_move_summary.csv",
                "required_move_vs_stock.csv": "tables/required_move_vs_stock.csv",
                "required_iv_support_summary.csv": "tables/required_iv_support_summary.csv",
                "entry_barrier_summary.csv": "tables/entry_barrier_summary.csv",
                "thesis_path_gallery.csv": "tables/thesis_path_gallery.csv",
                "thesis_iv_gallery.csv": "tables/thesis_iv_gallery.csv",
                "thesis_mode_candidates.csv": "tables/thesis_mode_candidates.csv",
                "thesis_path_family_summary.csv": "tables/thesis_path_family_summary.csv",
                "thesis_iv_family_summary.csv": "tables/thesis_iv_family_summary.csv",
                "thesis_candidate_ranking.csv": "tables/thesis_candidate_ranking.csv",
                "max_justified_premium.csv": "tables/max_justified_premium.csv",
                "current_vs_justified_premium.csv": "tables/current_vs_justified_premium.csv",
                "thesis_required_move_summary.csv": "tables/thesis_required_move_summary.csv",
                "thesis_stock_vs_option_summary.csv": "tables/thesis_stock_vs_option_summary.csv",
                "candidate_stress_grid.csv": "tables/candidate_stress_grid.csv",
                "premium_sensitivity_summary.csv": "tables/premium_sensitivity_summary.csv",
                "timing_slip_summary.csv": "tables/timing_slip_summary.csv",
                "target_stress_summary.csv": "tables/target_stress_summary.csv",
                "stress_transition_summary.csv": "tables/stress_transition_summary.csv",
                "chain_overview_summary.csv": "tables/chain_overview_summary.csv",
                "chain_overview_candidates.csv": "tables/chain_overview_candidates.csv",
                "single_option_decision_summary.csv": "tables/single_option_decision_summary.csv",
                "single_option_decision_path_selections.csv": "tables/single_option_decision_path_selections.csv",
                "single_option_representative_paths.csv": "tables/single_option_representative_paths.csv",
                "single_option_path_outcomes.csv": "tables/single_option_path_outcomes.csv",
                "single_option_path_family_counts.csv": "tables/single_option_path_family_counts.csv",
                "single_option_timing_sensitivity.csv": "tables/single_option_timing_sensitivity.csv",
                "single_option_iv_sensitivity.csv": "tables/single_option_iv_sensitivity.csv",
                "single_option_entry_sensitivity.csv": "tables/single_option_entry_sensitivity.csv",
                "single_option_summary_bullets.csv": "tables/single_option_summary_bullets.csv",
                "chain_source_summary.csv": "tables/chain_source_summary.csv",
                "market_context_summary.csv": "tables/market_context_summary.csv",
                "stock_path_library.csv": "tables/stock_path_library.csv",
                "stock_path_gallery.csv": "tables/stock_path_gallery.csv",
                "iv_path_gallery.csv": "tables/iv_path_gallery.csv",
                "family_comparison.csv": "tables/family_comparison.csv",
                "candidate_comparison.csv": "tables/candidate_comparison.csv",
                "strike_comparison_under_path.csv": "tables/strike_comparison_under_path.csv",
                "expiry_comparison_under_path.csv": "tables/expiry_comparison_under_path.csv",
                "option_value_over_path.csv": "tables/option_value_over_path.csv",
                "compare_vs_stock_path_rows.csv": "tables/compare_vs_stock_path_rows.csv",
                "compare_vs_stock_over_path.csv": "tables/compare_vs_stock_over_path.csv",
                "long_call_value_over_path_strike_view.csv": "tables/long_call_value_over_path_strike_view.csv",
                "long_call_value_over_path_expiry_view.csv": "tables/long_call_value_over_path_expiry_view.csv",
                "long_call_value_over_path_best_of.csv": "tables/long_call_value_over_path_best_of.csv",
                _path_view_filename("late_breakout", "compare_vs_stock_path_rows.csv"): f"tables/{_path_view_filename('late_breakout', 'compare_vs_stock_path_rows.csv')}",
                _path_view_filename("late_breakout", "long_call_strike_value.csv"): f"tables/{_path_view_filename('late_breakout', 'long_call_strike_value.csv')}",
                _path_view_filename("late_breakout", "long_call_strike_delta.csv"): f"tables/{_path_view_filename('late_breakout', 'long_call_strike_delta.csv')}",
                _path_view_filename("late_breakout", "long_call_expiry_value.csv"): f"tables/{_path_view_filename('late_breakout', 'long_call_expiry_value.csv')}",
                _path_view_filename("late_breakout", "long_call_expiry_delta.csv"): f"tables/{_path_view_filename('late_breakout', 'long_call_expiry_delta.csv')}",
                _path_view_filename("late_breakout", "long_call_best_of_value.csv"): f"tables/{_path_view_filename('late_breakout', 'long_call_best_of_value.csv')}",
                _path_view_filename("late_breakout", "long_call_best_of_delta.csv"): f"tables/{_path_view_filename('late_breakout', 'long_call_best_of_delta.csv')}",
                _path_view_filename("late_breakout", "path_checkpoints.csv"): f"tables/{_path_view_filename('late_breakout', 'path_checkpoints.csv')}",
                _path_view_filename("late_breakout", "iv_path_value.csv"): f"tables/{_path_view_filename('late_breakout', 'iv_path_value.csv')}",
                _path_view_filename("late_breakout", "iv_path_delta.csv"): f"tables/{_path_view_filename('late_breakout', 'iv_path_delta.csv')}",
                _path_view_filename("late_breakout", "iv_checkpoints.csv"): f"tables/{_path_view_filename('late_breakout', 'iv_checkpoints.csv')}",
                _path_view_filename("range_bound_near_flat", "compare_vs_stock_path_rows.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'compare_vs_stock_path_rows.csv')}",
                _path_view_filename("range_bound_near_flat", "long_call_strike_value.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'long_call_strike_value.csv')}",
                _path_view_filename("range_bound_near_flat", "long_call_strike_delta.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'long_call_strike_delta.csv')}",
                _path_view_filename("range_bound_near_flat", "long_call_expiry_value.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'long_call_expiry_value.csv')}",
                _path_view_filename("range_bound_near_flat", "long_call_expiry_delta.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'long_call_expiry_delta.csv')}",
                _path_view_filename("range_bound_near_flat", "long_call_best_of_value.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'long_call_best_of_value.csv')}",
                _path_view_filename("range_bound_near_flat", "long_call_best_of_delta.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'long_call_best_of_delta.csv')}",
                _path_view_filename("range_bound_near_flat", "path_checkpoints.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'path_checkpoints.csv')}",
                _path_view_filename("range_bound_near_flat", "iv_path_value.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'iv_path_value.csv')}",
                _path_view_filename("range_bound_near_flat", "iv_path_delta.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'iv_path_delta.csv')}",
                _path_view_filename("range_bound_near_flat", "iv_checkpoints.csv"): f"tables/{_path_view_filename('range_bound_near_flat', 'iv_checkpoints.csv')}",
                "path_case_rows.csv": "tables/path_case_rows.csv",
            },
            "charts": {
                "action_board_overview.png": "charts/action_board_overview.png",
                "bullish_action_board_overview.png": "charts/bullish_action_board_overview.png",
                "conviction_vs_robustness.png": "charts/conviction_vs_robustness.png",
                "bullish_conviction_vs_robustness.png": "charts/bullish_conviction_vs_robustness.png",
                "buy_watch_avoid_matrix.png": "charts/buy_watch_avoid_matrix.png",
                "bullish_buy_watch_avoid_matrix.png": "charts/bullish_buy_watch_avoid_matrix.png",
                "trigger_map.png": "charts/trigger_map.png",
                "bullish_trigger_map.png": "charts/bullish_trigger_map.png",
                "top_candidate_cards.png": "charts/top_candidate_cards.png",
                "stock_vs_option_preference_chart.png": "charts/stock_vs_option_preference_chart.png",
                "required_stock_path_to_buy.png": "charts/required_stock_path_to_buy.png",
                "required_move_speed_vs_magnitude.png": "charts/required_move_speed_vs_magnitude.png",
                "required_move_vs_stock_chart.png": "charts/required_move_vs_stock_chart.png",
                "strike_expiry_entry_barrier_map.png": "charts/strike_expiry_entry_barrier_map.png",
                "iv_support_requirement_chart.png": "charts/iv_support_requirement_chart.png",
                "thesis_path_gallery.png": "charts/thesis_path_gallery.png",
                "thesis_iv_gallery.png": "charts/thesis_iv_gallery.png",
                "thesis_candidate_overview.png": "charts/thesis_candidate_overview.png",
                "current_vs_justified_premium.png": "charts/current_vs_justified_premium.png",
                "thesis_path_vs_value.png": "charts/thesis_path_vs_value.png",
                "thesis_iv_vs_value.png": "charts/thesis_iv_vs_value.png",
                "thesis_stock_vs_option.png": "charts/thesis_stock_vs_option.png",
                "stress_test_overview.png": "charts/stress_test_overview.png",
                "premium_sensitivity_chart.png": "charts/premium_sensitivity_chart.png",
                "timing_slip_chart.png": "charts/timing_slip_chart.png",
                "target_stress_chart.png": "charts/target_stress_chart.png",
                "top_candidate_stress_cards.png": "charts/top_candidate_stress_cards.png",
                "chain_overview.png": "charts/chain_overview.png",
                "single_option_decision_view.png": "charts/single_option_decision_view.png",
                "highlights_overview.png": "charts/highlights_overview.png",
                "candidate_robustness_vs_upside.png": "charts/candidate_robustness_vs_upside.png",
                "path_survival_scorecard.png": "charts/path_survival_scorecard.png",
                "iv_robustness_scorecard.png": "charts/iv_robustness_scorecard.png",
                "strike_expiry_tradeoff_overview.png": "charts/strike_expiry_tradeoff_overview.png",
                "stock_vs_option_decision_chart.png": "charts/stock_vs_option_decision_chart.png",
                "stock_path_gallery.png": "charts/stock_path_gallery.png",
                "iv_path_gallery.png": "charts/iv_path_gallery.png",
                "required_path_vs_assumed_path.png": "charts/required_path_vs_assumed_path.png",
                "compare_vs_stock_path_delta.png": "charts/compare_vs_stock_path_delta.png",
                "representative_stock_paths.png": "charts/representative_stock_paths.png",
                "representative_iv_paths.png": "charts/representative_iv_paths.png",
                "option_value_over_path.png": "charts/option_value_over_path.png",
                "compare_vs_stock_over_path.png": "charts/compare_vs_stock_over_path.png",
                "strike_comparison_under_same_path.png": "charts/strike_comparison_under_same_path.png",
                "expiry_comparison_under_same_path.png": "charts/expiry_comparison_under_same_path.png",
                "long_call_value_over_path_strike_view.png": "charts/long_call_value_over_path_strike_view.png",
                "long_call_value_over_path_expiry_view.png": "charts/long_call_value_over_path_expiry_view.png",
                "long_call_value_over_path_best_of.png": "charts/long_call_value_over_path_best_of.png",
                _path_view_filename("late_breakout", "compare_vs_stock_path_delta.png"): f"charts/{_path_view_filename('late_breakout', 'compare_vs_stock_path_delta.png')}",
                _path_view_filename("late_breakout", "long_call_strike_value.png"): f"charts/{_path_view_filename('late_breakout', 'long_call_strike_value.png')}",
                _path_view_filename("late_breakout", "long_call_strike_delta.png"): f"charts/{_path_view_filename('late_breakout', 'long_call_strike_delta.png')}",
                _path_view_filename("late_breakout", "long_call_expiry_value.png"): f"charts/{_path_view_filename('late_breakout', 'long_call_expiry_value.png')}",
                _path_view_filename("late_breakout", "long_call_expiry_delta.png"): f"charts/{_path_view_filename('late_breakout', 'long_call_expiry_delta.png')}",
                _path_view_filename("late_breakout", "long_call_best_of_value.png"): f"charts/{_path_view_filename('late_breakout', 'long_call_best_of_value.png')}",
                _path_view_filename("late_breakout", "long_call_best_of_delta.png"): f"charts/{_path_view_filename('late_breakout', 'long_call_best_of_delta.png')}",
                _path_view_filename("late_breakout", "iv_path_value.png"): f"charts/{_path_view_filename('late_breakout', 'iv_path_value.png')}",
                _path_view_filename("late_breakout", "iv_path_delta.png"): f"charts/{_path_view_filename('late_breakout', 'iv_path_delta.png')}",
                _path_view_filename("range_bound_near_flat", "compare_vs_stock_path_delta.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'compare_vs_stock_path_delta.png')}",
                _path_view_filename("range_bound_near_flat", "long_call_strike_value.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'long_call_strike_value.png')}",
                _path_view_filename("range_bound_near_flat", "long_call_strike_delta.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'long_call_strike_delta.png')}",
                _path_view_filename("range_bound_near_flat", "long_call_expiry_value.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'long_call_expiry_value.png')}",
                _path_view_filename("range_bound_near_flat", "long_call_expiry_delta.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'long_call_expiry_delta.png')}",
                _path_view_filename("range_bound_near_flat", "long_call_best_of_value.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'long_call_best_of_value.png')}",
                _path_view_filename("range_bound_near_flat", "long_call_best_of_delta.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'long_call_best_of_delta.png')}",
                _path_view_filename("range_bound_near_flat", "iv_path_value.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'iv_path_value.png')}",
                _path_view_filename("range_bound_near_flat", "iv_path_delta.png"): f"charts/{_path_view_filename('range_bound_near_flat', 'iv_path_delta.png')}",
                "heatmap_strike_target_price.png": "charts/heatmap_strike_target_price.png",
            },
            "metadata": {"report_metadata.json": "metadata/report_metadata.json"},
        },
    }
    for path_name in ["late_breakout", "range_bound_near_flat"]:
        for suffix in [
            "long_call_strike_iv_value.csv",
            "long_call_strike_iv_delta.csv",
            "long_call_strike_iv_checkpoints.csv",
            "long_call_expiry_iv_value.csv",
            "long_call_expiry_iv_delta.csv",
            "long_call_expiry_iv_checkpoints.csv",
            "long_call_best_of_iv_value.csv",
            "long_call_best_of_iv_delta.csv",
            "long_call_best_of_iv_checkpoints.csv",
            "iv_robustness_summary.csv",
        ]:
            filename = _path_view_filename(path_name, suffix)
            manifest["file_map"]["tables"][filename] = f"tables/{filename}"
        for suffix in [
            "long_call_strike_iv_value.png",
            "long_call_strike_iv_delta.png",
            "long_call_expiry_iv_value.png",
            "long_call_expiry_iv_delta.png",
            "long_call_best_of_iv_value.png",
            "long_call_best_of_iv_delta.png",
        ]:
            filename = _path_view_filename(path_name, suffix)
            manifest["file_map"]["charts"][filename] = f"charts/{filename}"
    _write_text(bundle_dir / "bundle_manifest.json", json.dumps(manifest, indent=2))
    return bundle_dir


def test_build_model_outputs_projects_primary_contract_selection_artifacts(temp_workspace_root: Path):
    model_outputs = import_module("options_lab.model_outputs")
    bundle_dir = _create_fake_contract_selection_bundle(temp_workspace_root, run_slug="demo-run")
    model_root = temp_workspace_root / "model_outputs"

    result = model_outputs.build_model_outputs(
        bundle=bundle_dir,
        model_root=model_root,
    )

    promoted_dir = model_root / "GPRE" / "snapshot_2026-04-12" / "contract_selection" / "demo-run"
    latest_dir = model_root / "GPRE" / "latest"
    archive_manifest = model_root / "GPRE" / "archive" / "promoted_runs.json"

    assert Path(result["model_output_dir"]) == promoted_dir
    assert Path(result["latest_dir"]) == latest_dir
    assert (promoted_dir / "START_HERE.md").exists()
    assert (promoted_dir / "model_output_manifest.json").exists()
    assert (promoted_dir / "summary.md").exists()
    assert (promoted_dir / "summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "START_HERE.md").exists()
    assert (promoted_dir / "00_core_view" / "bullish_action_board.md").exists()
    assert (promoted_dir / "00_core_view" / "chain_overview.md").exists()
    assert (promoted_dir / "00_core_view" / "entry_justification.md").exists()
    assert (promoted_dir / "00_core_view" / "stress_tests.md").exists()
    assert (promoted_dir / "00_core_view" / "single_option_decision.md").exists()
    assert (promoted_dir / "00_core_view" / "top_candidate_cards.png").exists()
    assert (promoted_dir / "00_core_view" / "chain_overview.png").exists()
    assert (promoted_dir / "00_core_view" / "current_vs_justified_premium.png").exists()
    assert (promoted_dir / "00_core_view" / "stock_vs_option_preference_chart.png").exists()
    assert (promoted_dir / "00_core_view" / "stress_test_overview.png").exists()
    assert (promoted_dir / "00_core_view" / "premium_sensitivity_chart.png").exists()
    assert (promoted_dir / "00_core_view" / "timing_slip_chart.png").exists()
    assert (promoted_dir / "00_core_view" / "target_stress_chart.png").exists()
    assert (promoted_dir / "00_core_view" / "top_candidate_stress_cards.png").exists()
    assert (promoted_dir / "00_core_view" / "single_option_decision_view.png").exists()
    assert (promoted_dir / "00_core_view" / "required_stock_path_to_buy.png").exists()
    assert (promoted_dir / "00_core_view" / "required_move_speed_vs_magnitude.png").exists()
    assert (promoted_dir / "00_core_view" / "bullish_long_call_watchlist.csv").exists()
    assert (promoted_dir / "00_core_view" / "bullish_long_call_triggers.csv").exists()
    assert (promoted_dir / "00_core_view" / "chain_overview_summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "chain_overview_candidates.csv").exists()
    assert (promoted_dir / "00_core_view" / "candidate_stress_grid.csv").exists()
    assert (promoted_dir / "00_core_view" / "premium_sensitivity_summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "timing_slip_summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "target_stress_summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "stress_transition_summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "single_option_decision_summary.csv").exists()
    assert (promoted_dir / "00_core_view" / "single_option_decision_path_selections.csv").exists()
    assert (promoted_dir / "00_core_view" / "single_option_path_outcomes.csv").exists()
    assert (promoted_dir / "00_core_view" / "single_option_iv_sensitivity.csv").exists()
    assert (promoted_dir / "00_core_view" / "single_option_entry_sensitivity.csv").exists()
    assert (promoted_dir / "01_thesis_view" / "thesis_mode.md").exists()
    assert (promoted_dir / "01_thesis_view" / "thesis_candidate_overview.png").exists()
    assert (promoted_dir / "01_thesis_view" / "thesis_path_gallery.png").exists()
    assert (promoted_dir / "01_thesis_view" / "thesis_stock_vs_option.png").exists()
    assert (promoted_dir / "01_thesis_view" / "current_vs_justified_premium.png").exists()
    assert (promoted_dir / "01_thesis_view" / "thesis_candidate_ranking.csv").exists()
    assert (promoted_dir / "03_tables" / "chain_source_summary.csv").exists()
    assert (promoted_dir / "03_tables" / "market_context_summary.csv").exists()
    assert (promoted_dir / "03_tables" / "family_comparison.csv").exists()
    assert (promoted_dir / "03_tables" / "candidate_comparison.csv").exists()
    assert (promoted_dir / "03_tables" / "bullish_long_call_action_board.csv").exists()
    assert (promoted_dir / "03_tables" / "stock_path_library.csv").exists()
    assert (promoted_dir / "03_tables" / "required_move_summary.csv").exists()
    assert (promoted_dir / "04_secondary" / "action_board.md").exists()
    assert (promoted_dir / "04_secondary" / "highlights.md").exists()
    assert (promoted_dir / "04_secondary" / "other_structures.md").exists()
    assert (promoted_dir / "04_secondary" / "bullish_action_board_overview.png").exists()
    assert (promoted_dir / "04_secondary" / "bullish_trigger_map.png").exists()
    assert (promoted_dir / "04_secondary" / "stock_path_gallery.png").exists()
    assert (promoted_dir / "04_secondary" / "iv_path_gallery.png").exists()
    assert (promoted_dir / "04_secondary" / "representative_stock_paths.png").exists()
    assert (promoted_dir / "04_secondary" / "representative_iv_paths.png").exists()
    assert not (promoted_dir / "00_overview").exists()
    late_pack = promoted_dir / "02_path_packs" / "late_breakout"
    range_pack = promoted_dir / "02_path_packs" / "range_bound_flat"
    assert (late_pack / "README.md").exists()
    assert (late_pack / "compare_vs_stock_delta.png").exists()
    assert (late_pack / "long_call_strike_value.png").exists()
    assert (late_pack / "long_call_strike_delta.png").exists()
    assert (late_pack / "long_call_expiry_value.png").exists()
    assert (late_pack / "long_call_expiry_delta.png").exists()
    assert (late_pack / "long_call_best_of_value.png").exists()
    assert (late_pack / "long_call_best_of_delta.png").exists()
    assert (late_pack / "checkpoints.csv").exists()
    assert (late_pack / "iv_path_value.png").exists()
    assert (late_pack / "iv_path_delta.png").exists()
    assert (late_pack / "iv_checkpoints.csv").exists()
    assert (late_pack / "iv_robustness_summary.csv").exists()
    assert (late_pack / "long_call_strike_iv_value.png").exists()
    assert (late_pack / "long_call_strike_iv_delta.png").exists()
    assert (late_pack / "long_call_strike_iv_checkpoints.csv").exists()
    assert (late_pack / "long_call_expiry_iv_value.png").exists()
    assert (late_pack / "long_call_expiry_iv_delta.png").exists()
    assert (late_pack / "long_call_expiry_iv_checkpoints.csv").exists()
    assert (late_pack / "long_call_best_of_iv_value.png").exists()
    assert (late_pack / "long_call_best_of_iv_delta.png").exists()
    assert (late_pack / "long_call_best_of_iv_checkpoints.csv").exists()
    assert (range_pack / "compare_vs_stock_delta.png").exists()
    assert not (promoted_dir / "03_tables" / "path_case_rows.csv").exists()
    assert not (promoted_dir / "04_secondary" / "heatmap_strike_target_price.png").exists()
    assert archive_manifest.exists()

    start_here = (promoted_dir / "START_HERE.md").read_text(encoding="utf-8")
    latest_manifest = json.loads((latest_dir / "model_output_manifest.json").read_text(encoding="utf-8"))
    projection_manifest = json.loads((promoted_dir / "model_output_manifest.json").read_text(encoding="utf-8"))
    chain_source_text = (promoted_dir / "03_tables" / "chain_source_summary.csv").read_text(encoding="utf-8")
    core_start_here = (promoted_dir / "00_core_view" / "START_HERE.md").read_text(encoding="utf-8")
    action_board_text = (promoted_dir / "04_secondary" / "action_board.md").read_text(encoding="utf-8")
    bullish_action_board_text = (promoted_dir / "00_core_view" / "bullish_action_board.md").read_text(encoding="utf-8")
    chain_overview_text = (promoted_dir / "00_core_view" / "chain_overview.md").read_text(encoding="utf-8")
    top_candidate_cards_text = (promoted_dir / "00_core_view" / "top_candidate_cards.md").read_text(encoding="utf-8")
    stress_tests_text = (promoted_dir / "00_core_view" / "stress_tests.md").read_text(encoding="utf-8")
    single_option_text = (promoted_dir / "00_core_view" / "single_option_decision.md").read_text(encoding="utf-8")
    thesis_mode_text = (promoted_dir / "01_thesis_view" / "thesis_mode.md").read_text(encoding="utf-8")

    assert "analysis_outputs/GPRE/snapshot_2026-04-12/contract_selection/demo-run" in start_here
    expected_open_first = [
        "1. `00_core_view/bullish_action_board.md`",
        "2. `00_core_view/chain_overview.md`",
        "3. `00_core_view/chain_overview.png`",
        "4. `00_core_view/entry_justification.md`",
        "5. `01_thesis_view/thesis_mode.md`",
        "6. `00_core_view/stress_tests.md`",
        "7. `00_core_view/single_option_decision.md`",
        "8. `00_core_view/top_candidate_cards.png`",
        "9. `00_core_view/current_vs_justified_premium.png`",
        "10. `00_core_view/stock_vs_option_preference_chart.png`",
        "11. `00_core_view/stress_test_overview.png`",
        "12. `00_core_view/premium_sensitivity_chart.png`",
        "13. `00_core_view/timing_slip_chart.png`",
        "14. `00_core_view/target_stress_chart.png`",
        "15. `00_core_view/top_candidate_stress_cards.png`",
        "16. `00_core_view/single_option_decision_view.png`",
        "17. `00_core_view/required_stock_path_to_buy.png`",
        "18. `00_core_view/required_move_speed_vs_magnitude.png`",
        "19. `00_core_view/bullish_long_call_watchlist.csv`",
        "20. `00_core_view/bullish_long_call_triggers.csv`",
        "21. Then use `01_thesis_view/` for deeper target-thesis charts and tables",
        "22. Then choose a scenario folder under `02_path_packs/` for deeper path analysis",
        "23. Then use `03_tables/` and `04_secondary/` only as supporting detail",
    ]
    assert all(line in start_here for line in expected_open_first)
    assert [start_here.index(line) for line in expected_open_first] == sorted(start_here.index(line) for line in expected_open_first)
    assert "00_core_view/bullish_action_board.md" in start_here.split("## Open First", 1)[1].split("## Decision Snapshot", 1)[0]
    assert "00_core_view/chain_overview.md" in start_here
    assert "01_thesis_view/thesis_mode.md" in start_here
    assert "01_thesis_view/thesis_stock_vs_option.png" in start_here
    assert "04_secondary/highlights.md" in start_here
    assert "04_secondary/highlights_overview.png" in start_here
    assert "04_secondary/candidate_robustness_vs_upside.png" in start_here
    assert "04_secondary/stock_vs_option_decision_chart.png" in start_here
    assert "summary.md" in start_here
    assert "single_option_decision_path_selections.csv" in start_here
    assert "stock_path_library.csv" in start_here
    assert "stock_path_gallery.png" in start_here
    assert "iv_path_gallery.png" in start_here
    assert "required_path_vs_assumed_path.png" in start_here
    assert "02_path_packs/late_breakout/README.md" in start_here
    assert "02_path_packs/range_bound_flat/README.md" in start_here
    assert "03_tables/candidate_comparison.csv" in start_here
    assert "2026-04-17" in start_here
    assert "2027-01-15" in start_here
    assert "C:/Users" not in start_here
    assert "C:\\Users" not in start_here
    assert core_start_here == start_here
    assert "00_core_view/bullish_action_board.md" in core_start_here
    assert "00_core_view/chain_overview.md" in core_start_here
    assert "00_core_view/entry_justification.md" in core_start_here
    assert "01_thesis_view/thesis_mode.md" in core_start_here
    assert "required_stock_path_to_buy.png" in core_start_here
    assert "required_move_speed_vs_magnitude.png" in core_start_here
    assert "04_secondary/required_move_vs_stock_chart.png" in core_start_here
    assert "04_secondary/strike_expiry_entry_barrier_map.png" in core_start_here
    assert "04_secondary/iv_support_requirement_chart.png" in core_start_here
    assert "What Looks Most Actionable Right Now" in action_board_text
    assert "Best Bullish Long Calls Right Now" in bullish_action_board_text
    assert "Watchlist: Interesting But Not Buyable Yet" in bullish_action_board_text
    assert "Key Triggers To Watch" in bullish_action_board_text
    assert "Chain Overview / Compare Options" in chain_overview_text
    assert "Top Bullish Call Cards" in top_candidate_cards_text
    assert "Thesis Snapshot" in thesis_mode_text
    assert "Current Premium vs Thesis-Justified Premium" in thesis_mode_text
    assert "Stress Snapshot" in stress_tests_text
    assert "Which Candidates Are Price-Sensitive?" in stress_tests_text
    assert "Single-Option Decision View" in single_option_text
    assert "Upgrade if" in top_candidate_cards_text
    assert "When Stock Is Still Better" in action_board_text
    assert "C:/Users" not in core_start_here
    assert "C:\\Users" not in core_start_here
    assert "C:/Users" not in action_board_text
    assert "C:\\Users" not in action_board_text
    assert "C:/Users" not in bullish_action_board_text
    assert "C:\\Users" not in bullish_action_board_text
    assert "C:/Users" not in chain_overview_text
    assert "C:\\Users" not in chain_overview_text
    assert "C:/Users" not in top_candidate_cards_text
    assert "C:\\Users" not in top_candidate_cards_text
    assert "C:/Users" not in stress_tests_text
    assert "C:\\Users" not in stress_tests_text
    assert "C:/Users" not in single_option_text
    assert "C:\\Users" not in single_option_text
    assert "C:/Users" not in thesis_mode_text
    assert "C:\\Users" not in thesis_mode_text
    assert "C:/Users" not in chain_source_text
    assert "C:\\Users" not in chain_source_text

    assert projection_manifest["source_bundle_path"] == "analysis_outputs/GPRE/snapshot_2026-04-12/contract_selection/demo-run"
    assert projection_manifest["analysis_kind"] == "contract_selection"
    assert projection_manifest["analysis_trust_level"] == "cautious"
    assert projection_manifest["trusted_expiry_count"] == 5
    assert projection_manifest["fallback_only_expiry_count"] == 2
    assert projection_manifest["spot_source"] == "nasdaq_historical_quotes"
    assert projection_manifest["risk_free_source"] == "fred_local_store"
    assert "03_tables/chain_source_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/START_HERE.md" in projection_manifest["promoted_files"]
    assert "00_core_view/bullish_action_board.md" in projection_manifest["promoted_files"]
    assert "00_core_view/chain_overview.md" in projection_manifest["promoted_files"]
    assert "00_core_view/entry_justification.md" in projection_manifest["promoted_files"]
    assert "00_core_view/stress_tests.md" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_decision.md" in projection_manifest["promoted_files"]
    assert "00_core_view/top_candidate_cards.png" in projection_manifest["promoted_files"]
    assert "00_core_view/chain_overview.png" in projection_manifest["promoted_files"]
    assert "00_core_view/current_vs_justified_premium.png" in projection_manifest["promoted_files"]
    assert "00_core_view/stock_vs_option_preference_chart.png" in projection_manifest["promoted_files"]
    assert "00_core_view/required_stock_path_to_buy.png" in projection_manifest["promoted_files"]
    assert "00_core_view/required_move_speed_vs_magnitude.png" in projection_manifest["promoted_files"]
    assert "00_core_view/stress_test_overview.png" in projection_manifest["promoted_files"]
    assert "00_core_view/premium_sensitivity_chart.png" in projection_manifest["promoted_files"]
    assert "00_core_view/timing_slip_chart.png" in projection_manifest["promoted_files"]
    assert "00_core_view/target_stress_chart.png" in projection_manifest["promoted_files"]
    assert "00_core_view/top_candidate_stress_cards.png" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_decision_view.png" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_decision_path_selections.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/bullish_long_call_watchlist.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/bullish_long_call_triggers.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/chain_overview_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/chain_overview_candidates.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/candidate_stress_grid.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/premium_sensitivity_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/timing_slip_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/target_stress_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/stress_transition_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_decision_summary.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_path_outcomes.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_iv_sensitivity.csv" in projection_manifest["promoted_files"]
    assert "00_core_view/single_option_entry_sensitivity.csv" in projection_manifest["promoted_files"]
    assert "01_thesis_view/thesis_mode.md" in projection_manifest["promoted_files"]
    assert "01_thesis_view/thesis_candidate_overview.png" in projection_manifest["promoted_files"]
    assert "01_thesis_view/thesis_path_gallery.png" in projection_manifest["promoted_files"]
    assert "01_thesis_view/thesis_stock_vs_option.png" in projection_manifest["promoted_files"]
    assert "01_thesis_view/current_vs_justified_premium.png" in projection_manifest["promoted_files"]
    assert "03_tables/bullish_long_call_action_board.csv" in projection_manifest["promoted_files"]
    assert "03_tables/stock_path_library.csv" in projection_manifest["promoted_files"]
    assert "01_thesis_view/current_vs_justified_premium.csv" in projection_manifest["promoted_files"]
    assert "04_secondary/bullish_action_board_overview.png" in projection_manifest["promoted_files"]
    assert "04_secondary/bullish_trigger_map.png" in projection_manifest["promoted_files"]
    assert "04_secondary/highlights.md" in projection_manifest["promoted_files"]
    assert "04_secondary/stock_path_gallery.png" in projection_manifest["promoted_files"]
    assert "04_secondary/iv_path_gallery.png" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/long_call_strike_value.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/checkpoints.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/iv_path_value.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/iv_path_delta.png" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/iv_checkpoints.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/iv_robustness_summary.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/long_call_strike_iv_value.png" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/long_call_expiry_iv_value.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/long_call_best_of_iv_delta.png" in projection_manifest["promoted_files"]
    assert "02_path_packs/range_bound_flat/compare_vs_stock_delta.csv" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/compare_vs_stock_delta.png" in projection_manifest["promoted_files"]
    assert "02_path_packs/late_breakout/long_call_best_of_delta.png" in projection_manifest["promoted_files"]
    assert latest_manifest["source_bundle_path"] == projection_manifest["source_bundle_path"]


def test_build_model_outputs_refreshes_latest_and_records_archive_history(temp_workspace_root: Path):
    model_outputs = import_module("options_lab.model_outputs")
    model_root = temp_workspace_root / "model_outputs"
    first_bundle = _create_fake_contract_selection_bundle(temp_workspace_root, run_slug="demo-run-a")
    second_bundle = _create_fake_contract_selection_bundle(temp_workspace_root, run_slug="demo-run-b")

    model_outputs.build_model_outputs(bundle=first_bundle, model_root=model_root)
    model_outputs.build_model_outputs(bundle=second_bundle, model_root=model_root)

    latest_manifest = json.loads((model_root / "GPRE" / "latest" / "model_output_manifest.json").read_text(encoding="utf-8"))
    archive_payload = json.loads((model_root / "GPRE" / "archive" / "promoted_runs.json").read_text(encoding="utf-8"))

    assert latest_manifest["source_bundle_path"].endswith("demo-run-b")
    assert [entry["source_bundle_path"] for entry in archive_payload["promotions"]] == [
        "analysis_outputs/GPRE/snapshot_2026-04-12/contract_selection/demo-run-a",
        "analysis_outputs/GPRE/snapshot_2026-04-12/contract_selection/demo-run-b",
    ]
