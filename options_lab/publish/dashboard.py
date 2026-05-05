"""Static HTML dashboard rendering for saved report directories.

The dashboard layer is intentionally post-processing oriented: it reads the
existing CSV, JSON, Markdown, and PNG artifacts in one report folder and turns
them into one primary user-facing file, ``dashboard.html``.
"""

from __future__ import annotations

from base64 import b64encode
from datetime import datetime, timezone
from html import escape
import json
import math
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import quote
import warnings

import pandas as pd

from ..persistence import make_json_safe
from ..utils import clean_string, ensure_directory, slugify


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

PREFERRED_PLOT_ORDER = [
    "family_ranking_overview.png",
    "required_path_vs_assumed_path.png",
    "required_path_strategy_compare.png",
    "representative_stock_paths.png",
    "representative_iv_paths.png",
    "option_value_over_path.png",
    "compare_vs_stock_over_path.png",
    "strike_comparison_under_same_path.png",
    "expiry_comparison_under_same_path.png",
    "assumed_path_value_progression.png",
    "compare_vs_stock_path_delta.png",
    "iv_path_trace.png",
    "stock_path_expected_move.png",
    "strategy_value_path.png",
    "strategy_vs_stock_equal_capital.png",
    "strategy_vs_stock_share_equivalent.png",
    "driver_decomposition_expiry.png",
    "driver_decomposition_post_event.png",
    "driver_decomposition_event.png",
    "payoff_comparison.png",
    "estimated_value_comparison.png",
    "stock_vs_strategies_equal_capital.png",
    "stock_vs_strategies_share_equivalent.png",
    "compare_vs_stock_matrix.png",
    "payoff_at_expiry.png",
    "comparison_share_equivalent.png",
    "comparison_equal_capital.png",
    "estimated_value_vs_stock.png",
    "iv_sensitivity.png",
    "time_sensitivity.png",
    "term_structure_atm_iv.png",
    "skew_iv_by_strike.png",
    "skew_iv_by_moneyness.png",
    "skew_iv_by_delta.png",
    "realized_vol_history.png",
    "iv_vs_realized.png",
    "expected_vs_realized_pct.png",
    "event_window_prices.png",
    "event_scenarios_profit_loss.png",
    "event_scenarios_estimated_value.png",
    "trade_review_attribution.png",
    "portfolio_what_if.png",
    "portfolio_expiry_buckets.png",
]

PREFERRED_TABLE_ORDER = [
    "summary.csv",
    "family_comparison.csv",
    "candidate_comparison.csv",
    "strike_comparison.csv",
    "expiry_comparison.csv",
    "representative_paths_summary.csv",
    "path_pair_summary.csv",
    "strike_comparison_under_path.csv",
    "expiry_comparison_under_path.csv",
    "required_vs_assumed_path_summary.csv",
    "required_path_summary.csv",
    "stock_path_examples.csv",
    "iv_path_examples.csv",
    "option_value_over_path.csv",
    "compare_vs_stock_over_path.csv",
    "assumed_path_trace_rows.csv",
    "compare_vs_stock_path_rows.csv",
    "iv_path_trace_rows.csv",
    "iv_path_sensitivity_summary.csv",
    "path_risk_summary.csv",
    "candidate_summary.csv",
    "ranked_candidates.csv",
    "case_summary.csv",
    "checkpoint_replay.csv",
    "expected_move_vs_actual.csv",
    "driver_decomposition.csv",
    "compare_vs_stock.csv",
    "local_history.csv",
    "strategy_summary.csv",
    "named_scenarios.csv",
    "stock_relative.csv",
    "forward_quick_scenarios.csv",
    "forward_spot_time_grid.csv",
    "forward_spot_iv_grid.csv",
    "forward_time_iv_grid.csv",
    "spot_time_grid.csv",
    "spot_iv_grid.csv",
    "valuation_explanation.csv",
    "scenarios.csv",
    "scenario_cases.csv",
    "comparison.csv",
    "term-structure.csv",
    "iv-by-strike.csv",
    "iv-by-moneyness.csv",
    "iv-by-delta.csv",
    "realized-vol-history.csv",
    "expected-vs-realized-history.csv",
    "event-context.csv",
    "event-window.csv",
    "attribution.csv",
    "valuation-context.csv",
    "open-trades.csv",
    "ticker-summary.csv",
    "expiry-buckets.csv",
    "strategy-mix.csv",
    "portfolio-what-if.csv",
]

HIDDEN_FILES = {"dashboard.html"}

DATE_COLUMN_PRIORITY = [
    "published_at",
    "generated_at",
    "as_of_date",
    "review_date",
    "exit_date",
    "entry_date",
    "valuation_date",
    "event_date",
    "matched_date",
    "snapshot_date",
    "date",
]

CHRONOLOGY_SORT_EXCLUDES = {
    "comparison",
    "scenario_cases",
    "leg_details",
    "term_structure",
    "iv_by_strike",
    "iv_by_moneyness",
    "iv_by_delta",
    "strategy_mix",
}

CHRONOLOGY_SORT_HINTS = {
    "history",
    "event",
    "note",
    "review",
    "coverage",
    "open",
    "timeline",
    "published",
}

SCENARIO_TABS_HEAD = """
  <style>
    .scenario-tab-nav {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 18px;
    }
    .scenario-tab-button {
      appearance: none;
      border: 1px solid var(--line);
      background: var(--panel-soft);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    .scenario-tab-button.is-active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }
    .scenario-tab-panel + .scenario-tab-panel {
      margin-top: 18px;
    }
    .scenario-tab-panel {
      scroll-margin-top: 96px;
    }
    .scenario-tab-panel[hidden] {
      display: none !important;
    }
    .scenario-control-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 0 0 14px;
    }
    .scenario-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      font-size: 0.92rem;
      color: var(--muted);
    }
    .scenario-chip strong {
      color: var(--text);
    }
    .strategy-card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }
    .strategy-quick-card {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: var(--panel-soft);
    }
    .strategy-quick-card h3 {
      margin: 0 0 8px;
    }
    .strategy-quick-meta {
      color: var(--muted);
      margin-bottom: 12px;
    }
    .strategy-quick-note {
      margin-top: 12px;
      color: var(--muted);
      font-size: 0.95rem;
    }
    .strategy-card-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .scenario-link-chip,
    .chart-switcher-button {
      appearance: none;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
    }
    .chart-switcher-button.is-active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }
    .decision-hint-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .decision-hint-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px 18px 16px;
      background: var(--panel-soft);
      display: grid;
      gap: 10px;
    }
    .decision-hint-label {
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
      margin-bottom: 6px;
    }
    .decision-hint-value {
      font-weight: 700;
      font-size: 1rem;
      margin-bottom: 4px;
    }
    .decision-hint-detail {
      color: var(--muted);
      font-size: 0.98rem;
      line-height: 1.45;
    }
    .decision-hint-card .scenario-link-chip {
      justify-self: start;
      margin-top: 4px;
    }
    .decision-hint-card.is-emphasized {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(15, 92, 87, 0.12);
    }
    .strategy-deep-dive-group {
      display: grid;
      gap: 16px;
    }
    .strategy-deep-dive {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel-soft);
      overflow: hidden;
    }
    .strategy-deep-dive summary {
      cursor: pointer;
      list-style: none;
      padding: 16px 18px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .strategy-deep-dive summary::-webkit-details-marker {
      display: none;
    }
    .strategy-deep-dive-title {
      font-size: 1rem;
      margin-bottom: 4px;
    }
    .strategy-deep-dive-meta {
      color: var(--muted);
      font-weight: 500;
      font-size: 0.92rem;
    }
    .strategy-deep-dive-body {
      padding: 0 18px 18px;
    }
    .strategy-deep-dive-grid {
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .strategy-deep-dive-sidebar {
      display: grid;
      gap: 12px;
    }
    .strategy-deep-dive-copy {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .chart-switcher-nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }
    .chart-switcher-panel[hidden] {
      display: none !important;
    }
    .deep-dive-figure .plot-card.featured {
      min-height: auto;
    }
    .deep-dive-figure .plot-card.featured img {
      min-height: 340px;
      object-fit: contain;
    }
    .budget-flag {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .budget-flag-fit {
      background: var(--ok-bg);
      color: var(--ok-text);
      border-color: rgba(29, 95, 56, 0.18);
    }
    .budget-flag-tight {
      background: var(--partial-bg);
      color: var(--partial-text);
      border-color: rgba(138, 90, 0, 0.18);
    }
    .scenario-sheet-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 16px;
    }
    .scenario-sheet-grid .panel {
      margin: 0;
    }
    .lead-chart-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }
    .lead-chart-grid .plot-card.featured {
      min-height: 100%;
    }
    .compact-data-table {
      font-size: 0.92rem;
    }
    .inline-table-block + .inline-table-block {
      margin-top: 14px;
    }
    .scenario-top-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
    }
    .scenario-visual-panel {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: var(--panel-soft);
      box-shadow: var(--shadow);
    }
    .scenario-visual-panel h3 {
      margin: 0 0 12px;
    }
    .focus-heatmap-panel .plot-card.featured img {
      min-height: 320px;
      object-fit: contain;
    }
    .replay-root,
    .valuation-root {
      display: grid;
      gap: 16px;
    }
    .replay-controls {
      display: grid;
      gap: 14px;
    }
    .replay-control-group {
      display: grid;
      gap: 8px;
    }
    .replay-label {
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .replay-button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .scenario-link-chip.is-active,
    .chart-switcher-button.is-active,
    .replay-chip.is-active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }
    .replay-slider-row {
      display: grid;
      gap: 10px;
    }
    .replay-slider-row input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }
    .replay-state-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      background: var(--panel-soft);
    }
    .replay-state-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      align-items: start;
    }
    .replay-panel[hidden],
    .valuation-panel[hidden],
    .focus-panel[hidden] {
      display: none !important;
    }
    .replay-summary-title,
    .valuation-panel h3 {
      margin: 0 0 8px;
    }
    .replay-summary-copy,
    .valuation-copy {
      color: var(--muted);
      font-size: 0.95rem;
      margin-bottom: 12px;
    }
    .replay-chart-panel .plot-card.featured img {
      min-height: 340px;
      object-fit: contain;
    }
    .valuation-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }
    .valuation-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      background: var(--panel-soft);
    }
    .valuation-card h4 {
      margin: 0 0 10px;
    }
    .forward-lab-shell {
      display: grid;
      gap: 16px;
    }
    .forward-lab-controls {
      display: grid;
      gap: 14px;
    }
    .forward-lab-control-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .forward-lab-control {
      display: grid;
      gap: 6px;
    }
    .forward-lab-control label,
    .forward-lab-control-title {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .forward-lab-select {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }
    .forward-lab-button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .forward-lab-button {
      appearance: none;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
    }
    .forward-lab-button.is-active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }
    .forward-lab-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .forward-lab-locked-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: var(--panel-soft);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 0.92rem;
    }
    .forward-lab-locked-chip strong {
      color: var(--text);
    }
    .forward-lab-grid {
      display: grid;
      gap: 16px;
    }
    .forward-lab-visual-card {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel-soft);
      padding: 14px;
      box-shadow: var(--shadow);
    }
    .forward-lab-visual-card h3 {
      margin: 0 0 8px;
    }
    .forward-lab-note {
      color: var(--muted);
      font-size: 0.94rem;
      margin-bottom: 12px;
    }
    .forward-lab-svg-wrap {
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #ffffff;
      padding: 8px;
    }
    .forward-lab-svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .forward-lab-detail-grid {
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 16px;
    }
    .forward-lab-summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }
    .forward-lab-summary-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #ffffff;
      padding: 12px 14px;
    }
    .forward-lab-summary-card h4 {
      margin: 0 0 8px;
      font-size: 0.98rem;
    }
    .forward-lab-empty {
      color: var(--muted);
      font-size: 0.95rem;
      padding: 12px 0;
    }
    .status-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 8px 0 2px;
    }
    .status-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--muted);
      font-size: 0.84rem;
      font-weight: 600;
    }
    .status-chip.is-warning {
      background: #fff4e8;
      border-color: #efc79b;
      color: #8b4f16;
    }
    .status-chip.is-muted {
      background: #f6f2eb;
    }
    .forward-lab-fixed-control[hidden] {
      display: none !important;
    }
    .forward-lab-compact .forward-lab-detail-grid {
      grid-template-columns: 1fr;
    }
    .forward-lab-compact .forward-lab-control-grid {
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
    @media (max-width: 960px) {
      .strategy-deep-dive-grid {
        grid-template-columns: 1fr;
      }
      .forward-lab-detail-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
"""

SCENARIO_TABS_SCRIPT = """
  <script>
    (() => {
      const root = document.querySelector("[data-scenario-tabbed-page]");
      if (!root) return;
      const buttons = Array.from(root.querySelectorAll("[data-scenario-tab-target]"));
      const panels = Array.from(root.querySelectorAll("[data-scenario-tab-panel]"));
      if (!buttons.length || !panels.length) return;

      const targetFromHash = () => {
        const hash = (window.location.hash || "").replace(/^#/, "");
        if (!hash) return null;
        const panel = panels.find((candidate) => candidate.id === hash);
        if (panel) {
          return panel.getAttribute("data-scenario-tab-panel");
        }
        const button = buttons.find((candidate) => candidate.getAttribute("data-scenario-tab-target") === hash);
        return button ? button.getAttribute("data-scenario-tab-target") : null;
      };

      const activate = (target, options = {}) => {
        const syncHash = Boolean(options.syncHash);
        buttons.forEach((button) => {
          const active = button.getAttribute("data-scenario-tab-target") === target;
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-selected", active ? "true" : "false");
        });
        panels.forEach((panel) => {
          const active = panel.getAttribute("data-scenario-tab-panel") === target;
          if (active) {
            panel.removeAttribute("hidden");
          } else {
            panel.setAttribute("hidden", "hidden");
          }
        });
        if (syncHash) {
          const activePanel = panels.find((panel) => panel.getAttribute("data-scenario-tab-panel") === target);
          if (activePanel && activePanel.id) {
            try {
              window.history.replaceState(null, "", `#${activePanel.id}`);
            } catch (error) {
              window.location.hash = activePanel.id;
            }
          }
        }
      };

      buttons.forEach((button) => {
        button.addEventListener("click", () => activate(button.getAttribute("data-scenario-tab-target"), { syncHash: true }));
      });
      window.addEventListener("hashchange", () => {
        const target = targetFromHash();
        if (target) activate(target);
      });
      activate(targetFromHash() || "summary");
    })();
    (() => {
      const selectedFamily = new URLSearchParams(window.location.search || "").get("strategy_family") || "";
      if (!selectedFamily) return;
      document.querySelectorAll("[data-strategy-family-card]").forEach((card) => {
        const active = card.getAttribute("data-strategy-family-card") === selectedFamily;
        card.classList.toggle("is-emphasized", active);
      });
    })();
    (() => {
      const switchers = Array.from(document.querySelectorAll("[data-chart-switcher]"));
      switchers.forEach((switcher) => {
        const buttons = Array.from(switcher.querySelectorAll("[data-chart-switch-target]"));
        const panels = Array.from(switcher.querySelectorAll("[data-chart-switch-panel]"));
        if (!buttons.length || !panels.length) return;
        const activate = (target) => {
          buttons.forEach((button) => {
            const active = button.getAttribute("data-chart-switch-target") === target;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
          });
          panels.forEach((panel) => {
            const active = panel.getAttribute("data-chart-switch-panel") === target;
            if (active) {
              panel.removeAttribute("hidden");
            } else {
              panel.setAttribute("hidden", "hidden");
            }
          });
        };
        buttons.forEach((button) => {
          button.addEventListener("click", () => activate(button.getAttribute("data-chart-switch-target")));
        });
        const explicitDefault = switcher.getAttribute("data-chart-switch-default");
        const activeButton = buttons.find((button) => button.classList.contains("is-active"));
        activate(explicitDefault || (activeButton && activeButton.getAttribute("data-chart-switch-target")) || buttons[0].getAttribute("data-chart-switch-target"));
      });
    })();
    (() => {
      const replayRoots = Array.from(document.querySelectorAll("[data-replay-root]"));
      replayRoots.forEach((root) => {
        const spotButtons = Array.from(root.querySelectorAll("[data-replay-spot-case]"));
        const ivButtons = Array.from(root.querySelectorAll("[data-replay-iv-case]"));
        const slider = root.querySelector("[data-replay-horizon-slider]");
        const sliderLabel = root.querySelector("[data-replay-horizon-label]");
        const statusNote = root.querySelector("[data-replay-status-note]");
        const stateSummary = root.querySelector("[data-replay-state-summary]");
        const chartPanels = Array.from(root.querySelectorAll("[data-replay-chart-panel]"));
        const casePanels = Array.from(root.querySelectorAll("[data-replay-case-panel]"));
        const horizons = JSON.parse(root.getAttribute("data-replay-horizons") || "[]");
        if (!slider || !horizons.length) return;
        const state = {
          spotCase: root.getAttribute("data-default-spot-case") || (spotButtons[0] && spotButtons[0].getAttribute("data-replay-spot-case")) || "",
          ivCase: root.getAttribute("data-default-iv-case") || (ivButtons[0] && ivButtons[0].getAttribute("data-replay-iv-case")) || "",
          horizon: root.getAttribute("data-default-horizon") || horizons[0],
        };
        const activate = () => {
          spotButtons.forEach((button) => {
            const active = button.getAttribute("data-replay-spot-case") === state.spotCase;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
          });
          ivButtons.forEach((button) => {
            const active = button.getAttribute("data-replay-iv-case") === state.ivCase;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
          });
          chartPanels.forEach((panel) => {
            const active = panel.getAttribute("data-replay-chart-panel") === state.horizon;
            if (active) {
              panel.removeAttribute("hidden");
            } else {
              panel.setAttribute("hidden", "hidden");
            }
          });
          let activeCasePanel = null;
          casePanels.forEach((panel) => {
            const active =
              panel.getAttribute("data-replay-case-horizon") === state.horizon &&
              panel.getAttribute("data-replay-case-spot") === state.spotCase &&
              panel.getAttribute("data-replay-case-iv") === state.ivCase;
            if (active) {
              panel.removeAttribute("hidden");
              activeCasePanel = panel;
            } else {
              panel.setAttribute("hidden", "hidden");
            }
          });
          if (sliderLabel) {
            sliderLabel.textContent = state.horizon.replace(/_/g, " ");
          }
          if (statusNote) {
            statusNote.textContent = activeCasePanel ? (activeCasePanel.getAttribute("data-replay-status-note") || "") : "";
          }
          if (stateSummary) {
            const flags = activeCasePanel ? (activeCasePanel.getAttribute("data-replay-flags") || "") : "";
            stateSummary.textContent = `Showing ${state.spotCase.replace(/_/g, " ")} | ${state.ivCase.replace(/_/g, " ")} | ${state.horizon.replace(/_/g, " ")}.${flags ? ` Status: ${flags}.` : " Status: distinct coverage available."}`;
          }
        };
        spotButtons.forEach((button) => {
          button.addEventListener("click", () => {
            state.spotCase = button.getAttribute("data-replay-spot-case") || state.spotCase;
            activate();
          });
        });
        ivButtons.forEach((button) => {
          button.addEventListener("click", () => {
            state.ivCase = button.getAttribute("data-replay-iv-case") || state.ivCase;
            activate();
          });
        });
        slider.setAttribute("min", "0");
        slider.setAttribute("max", String(Math.max(horizons.length - 1, 0)));
        const initialIndex = Math.max(horizons.indexOf(state.horizon), 0);
        slider.value = String(initialIndex);
        slider.addEventListener("input", () => {
          const index = Number(slider.value || 0);
          state.horizon = horizons[index] || horizons[0];
          activate();
        });
        activate();
      });
    })();
    (() => {
      const parseJsonScript = (id) => {
        const node = document.getElementById(id);
        if (!node) return null;
        try {
          return JSON.parse(node.textContent || "{}");
        } catch (error) {
          console.warn("Forward scenario lab JSON parse failed", error);
          return null;
        }
      };
      const toTitle = (value) =>
        String(value || "")
          .replace(/_/g, " ")
          .replace(/\\b\\w/g, (match) => match.toUpperCase());
      const formatValue = (value, metric) => {
        if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) {
          return "n/a";
        }
        const numeric = Number(value);
        if (metric === "return_on_comparison_capital" || metric === "stock_return_on_comparison_capital") {
          return `${(numeric * 100).toFixed(1)}%`;
        }
        if (Math.abs(numeric) >= 100) {
          return numeric.toFixed(0);
        }
        if (Math.abs(numeric) >= 10) {
          return numeric.toFixed(1);
        }
        return numeric.toFixed(2);
      };
      const metricMeta = {
        estimated_value: { label: "Modeled Value", palette: "sequential", legend: "Modeled Value" },
        profit_loss: { label: "PnL $", palette: "diverging", legend: "Profit / Loss" },
        return_on_comparison_capital: { label: "PnL %", palette: "diverging", legend: "Return On Comparison Capital" },
        stock_relative_difference: { label: "Outperformance vs Long Stock", palette: "diverging", legend: "Difference Vs Long Stock" },
      };
      const modeMeta = {
        spot_time: {
          label: "Spot x Time",
          dataKey: "spot_time",
          xKey: "spot_price",
          yKey: "horizon",
          fixedKey: "iv_case",
          detailKeys: ["strategy", "horizon", "iv_case", "spot_price", "estimated_value", "profit_loss", "return_on_comparison_capital", "stock_relative_difference"],
          sliceX: "spot_price",
          sliceSeries: "horizon",
          note: "This heatmap moves spot and holding horizon while keeping one IV case fixed.",
        },
        spot_iv: {
          label: "Spot x IV",
          dataKey: "spot_iv",
          xKey: "spot_price",
          yKey: "iv_case",
          fixedKey: "horizon",
          detailKeys: ["strategy", "horizon", "iv_case", "spot_price", "estimated_value", "profit_loss", "return_on_comparison_capital", "stock_relative_difference"],
          sliceX: "spot_price",
          sliceSeries: "iv_case",
          note: "This heatmap moves spot and IV while keeping one holding horizon fixed.",
        },
        time_iv: {
          label: "Time x IV",
          dataKey: "time_iv",
          xKey: "horizon",
          yKey: "iv_case",
          fixedKey: "spot_case",
          detailKeys: ["strategy", "spot_case", "horizon", "iv_case", "spot_price", "estimated_value", "profit_loss", "return_on_comparison_capital", "stock_relative_difference"],
          sliceX: "horizon",
          sliceSeries: "iv_case",
          note: "This heatmap moves time and IV while keeping one named spot case fixed.",
        },
      };
      const paletteColor = (value, minimum, maximum, palette) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
          return "#f5f1e8";
        }
        const numeric = Number(value);
        if (palette === "diverging") {
          const bound = Math.max(Math.abs(minimum), Math.abs(maximum), 0.000001);
          const ratio = Math.max(-1, Math.min(1, numeric / bound));
          if (ratio >= 0) {
            const tint = Math.round(255 - ratio * 90);
            const green = Math.round(240 - ratio * 45);
            return `rgb(${tint}, ${green}, ${Math.round(210 - ratio * 120)})`;
          }
          const scaled = Math.abs(ratio);
          return `rgb(${Math.round(230 - scaled * 35)}, ${Math.round(242 - scaled * 100)}, ${Math.round(230 - scaled * 90)})`;
        }
        const span = Math.max(maximum - minimum, 0.000001);
        const ratio = Math.max(0, Math.min(1, (numeric - minimum) / span));
        return `rgb(${Math.round(250 - ratio * 110)}, ${Math.round(246 - ratio * 90)}, ${Math.round(228 - ratio * 130)})`;
      };
      const buildSvg = (width, height, inner) =>
        `<svg class="forward-lab-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Forward scenario chart">${inner}</svg>`;
      const lineColor = (index) => {
        const colors = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"];
        return colors[index % colors.length];
      };
      const interactiveChartCard = (chartId, label, svg, legendHtml = "", note = "") => {
        return `
          <div class="chart-toolbar chart-toolbar--stacked">
            <div class="chart-toolbar-copy">${note || label}</div>
            <button type="button" class="chart-action" data-lightbox-target="#${chartId}" data-lightbox-caption="${label}">Open larger chart</button>
          </div>
          <div id="${chartId}" class="forward-lab-svg-wrap">${svg}</div>
          ${legendHtml}
        `;
      };
      const seriesLegend = (entries) => {
        if (!entries.length) return "";
        return `<div class="status-chip-row">${entries.map((entry) => (
          `<span class="status-chip"><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:${entry.color};"></span>${entry.label}</span>`
        )).join("")}</div>`;
      };
      const createHeatmapSvg = (rows, state, payload, rootId) => {
        if (!rows.length) {
          return '<div class="forward-lab-empty">No rows matched the current forward scenario selection.</div>';
        }
        const config = modeMeta[state.mode];
        const xOrder = config.xKey === "spot_price"
          ? Array.from(new Set(rows.map((row) => Number(row[config.xKey])))).sort((left, right) => left - right)
          : (payload.orders.horizons_x || []).filter((item) => rows.some((row) => row.horizon === item));
        const yOrderSource = config.yKey === "horizon" ? (payload.orders.horizons_y || []) : (payload.orders.iv_cases_y || []);
        const yOrder = yOrderSource.filter((item) => rows.some((row) => row[config.yKey] === item));
        const metric = state.metric;
        const values = rows.map((row) => Number(row[metric])).filter((value) => !Number.isNaN(value));
        const minimum = values.length ? Math.min(...values) : 0;
        const maximum = values.length ? Math.max(...values) : 0;
        const left = 120;
        const top = 24;
        const cellWidth = 74;
        const cellHeight = 44;
        const width = left + xOrder.length * cellWidth + 24;
        const height = top + yOrder.length * cellHeight + 76;
        const cellMap = new Map();
        rows.forEach((row) => {
          const key = `${row[config.xKey]}|${row[config.yKey]}`;
          cellMap.set(key, row);
        });
        const cells = [];
        xOrder.forEach((xValue, xIndex) => {
          yOrder.forEach((yValue, yIndex) => {
            const row = cellMap.get(`${xValue}|${yValue}`);
            const value = row ? Number(row[metric]) : null;
            const x = left + xIndex * cellWidth;
            const y = top + yIndex * cellHeight;
            const fill = paletteColor(value, minimum, maximum, metricMeta[metric].palette);
            cells.push(`<rect x="${x}" y="${y}" width="${cellWidth - 2}" height="${cellHeight - 2}" rx="6" fill="${fill}" stroke="#d8d2c7"></rect>`);
            cells.push(`<text x="${x + (cellWidth / 2) - 1}" y="${y + 26}" text-anchor="middle" font-size="11" fill="#191714">${value === null ? "n/a" : formatValue(value, metric)}</text>`);
          });
        });
        const xLabels = xOrder.map((value, index) => {
          const label = config.xKey === "spot_price" ? Number(value).toFixed(2) : toTitle(value);
          return `<text x="${left + index * cellWidth + (cellWidth / 2) - 1}" y="${top + yOrder.length * cellHeight + 20}" text-anchor="middle" font-size="12" fill="#6b645b">${label}</text>`;
        }).join("");
        const yLabels = yOrder.map((value, index) =>
          `<text x="${left - 10}" y="${top + index * cellHeight + 26}" text-anchor="end" font-size="12" fill="#6b645b">${toTitle(value)}</text>`
        ).join("");
        const title = `<text x="${left}" y="16" font-size="13" font-weight="700" fill="#191714">${config.label} | ${metricMeta[metric].label}</text>`;
        const chartId = `forward-heatmap-${rootId || "root"}`;
        const svg = buildSvg(width, height, title + cells.join("") + xLabels + yLabels);
        return interactiveChartCard(
          chartId,
          `${config.label} | ${metricMeta[metric].label}`,
          svg,
          "",
          `${config.note} Colors use ${metricMeta[metric].palette === "diverging" ? "a zero-centered diverging scale" : "a sequential modeled-value scale"}.`
        );
      };
      const createSliceSvg = (rows, state, payload, rootId) => {
        if (!rows.length) {
          return '<div class="forward-lab-empty">No slice rows matched the current selection.</div>';
        }
        const config = modeMeta[state.mode];
        const metric = state.metric;
        const xOrder = config.sliceX === "spot_price"
          ? Array.from(new Set(rows.map((row) => Number(row[config.sliceX])))).sort((left, right) => left - right)
          : (payload.orders.horizons_x || []).filter((item) => rows.some((row) => row.horizon === item));
        const seriesOrderSource = config.sliceSeries === "horizon" ? (payload.orders.horizons_y || []) : (payload.orders.iv_cases_y || []);
        const seriesOrder = seriesOrderSource.filter((item) => rows.some((row) => row[config.sliceSeries] === item));
        const values = rows.map((row) => Number(row[metric])).filter((value) => !Number.isNaN(value));
        const minimum = values.length ? Math.min(...values) : 0;
        const maximum = values.length ? Math.max(...values) : 0;
        const left = 64;
        const top = 24;
        const chartWidth = 620;
        const chartHeight = 260;
        const bottom = top + chartHeight;
        const span = Math.max(maximum - minimum, 0.000001);
        const xPosition = (value) => {
          if (config.sliceX === "spot_price") {
            const numericOrder = xOrder.map((item) => Number(item));
            const minX = numericOrder[0];
            const maxX = numericOrder[numericOrder.length - 1];
            const xSpan = Math.max(maxX - minX, 0.000001);
            return left + ((Number(value) - minX) / xSpan) * chartWidth;
          }
          const index = xOrder.indexOf(value);
          const divisor = Math.max(xOrder.length - 1, 1);
          return left + (index / divisor) * chartWidth;
        };
        const yPosition = (value) => bottom - ((Number(value) - minimum) / span) * chartHeight;
        const grid = [];
        for (let step = 0; step <= 4; step += 1) {
          const y = top + (step / 4) * chartHeight;
          grid.push(`<line x1="${left}" y1="${y}" x2="${left + chartWidth}" y2="${y}" stroke="#e6e0d5" stroke-dasharray="3 4"></line>`);
        }
        const lines = [];
        const legendEntries = [];
        seriesOrder.forEach((series, index) => {
          const seriesRows = rows
            .filter((row) => row[config.sliceSeries] === series)
            .sort((leftRow, rightRow) => {
              if (config.sliceX === "spot_price") {
                return Number(leftRow[config.sliceX]) - Number(rightRow[config.sliceX]);
              }
              return (payload.orders.horizons_x || []).indexOf(leftRow[config.sliceX]) - (payload.orders.horizons_x || []).indexOf(rightRow[config.sliceX]);
            });
          const points = seriesRows
            .map((row) => `${xPosition(row[config.sliceX])},${yPosition(row[metric])}`)
            .join(" ");
          const color = lineColor(index);
          legendEntries.push({ label: toTitle(series), color });
          lines.push(`<polyline fill="none" stroke="${color}" stroke-width="3" points="${points}"></polyline>`);
          seriesRows.forEach((row) => {
            lines.push(`<circle cx="${xPosition(row[config.sliceX])}" cy="${yPosition(row[metric])}" r="4" fill="${color}" stroke="#ffffff" stroke-width="1"></circle>`);
          });
        });
        const xLabels = xOrder.map((value) => {
          const label = config.sliceX === "spot_price" ? Number(value).toFixed(2) : toTitle(value);
          return `<text x="${xPosition(value)}" y="${bottom + 20}" text-anchor="middle" font-size="12" fill="#6b645b">${label}</text>`;
        }).join("");
        const svg = buildSvg(
          left + chartWidth + 24,
          bottom + 44,
          `<text x="${left}" y="16" font-size="13" font-weight="700" fill="#191714">Linked Slice Chart | ${metricMeta[metric].label}</text>`
          + grid.join("")
          + `<line x1="${left}" y1="${bottom}" x2="${left + chartWidth}" y2="${bottom}" stroke="#bdb4a6"></line>`
          + `<line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#bdb4a6"></line>`
          + lines.join("")
          + xLabels
        );
        const chartId = `forward-slice-${rootId || "root"}`;
        return interactiveChartCard(
          chartId,
          `Linked Slice Chart | ${metricMeta[metric].label}`,
          svg,
          seriesLegend(legendEntries),
          "The linked slice chart turns the current heatmap selection into a line view so you can read exact path differences more intuitively."
        );
      };
      const renderTable = (columns, rows) => {
        if (!rows.length) {
          return '<div class="forward-lab-empty">No matching rows available.</div>';
        }
        const head = columns.map((column) => `<th>${column.label}</th>`).join("");
        const body = rows.map((row) => (
          `<tr>${columns.map((column) => `<td>${column.render ? column.render(row) : row[column.key]}</td>`).join("")}</tr>`
        )).join("");
        return `<div class="table-wrap compact-data-table"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
      };
      const updateFixedControlVisibility = (root, state) => {
        root.querySelectorAll("[data-forward-fixed-control]").forEach((node) => {
          const target = node.getAttribute("data-forward-fixed-control");
          node.hidden = target !== state.mode;
        });
      };
      const nearestSpotValue = (payload, desiredCase) => {
        const match = (payload.spotCases || []).find((item) => item.value === desiredCase);
        return match ? match.spot_price : null;
      };
      const summarizeQuickRows = (quickRows, state, payload) => {
        const preferredSpotCase = state.fixedSpotCase || "flat";
        const selected = quickRows.filter((row) => row.spot_case === preferredSpotCase);
        const row = selected[0] || quickRows.find((item) => item.spot_case === "flat") || quickRows[0] || null;
        if (!row) {
          return [];
        }
        return [
          {
            title: "Representative Case",
            pairs: [
              ["Spot Case", toTitle(row.spot_case)],
              ["Horizon", toTitle(row.horizon)],
              ["IV Case", toTitle(row.iv_case)],
              ["Modeled Value", formatValue(row.estimated_value, "estimated_value")],
              ["PnL $", formatValue(row.profit_loss, "profit_loss")],
              ["PnL % On Budget", formatValue(row.return_on_comparison_capital, "return_on_comparison_capital")],
            ],
          },
          {
            title: "Vs Long Stock",
            pairs: [
              ["Strategy Normalized PnL", formatValue(row.comparison_profit_loss, "profit_loss")],
              ["Long Stock Normalized PnL", formatValue(row.stock_profit_loss, "profit_loss")],
              ["Difference Vs Stock", formatValue(row.stock_relative_difference, "stock_relative_difference")],
              ["Affordable Units", formatValue(row.affordable_units, "estimated_value")],
              ["Fits Budget", row.fully_implementable_with_budget ? "Yes" : "No"],
            ],
          },
        ];
      };
      const renderSummaryCards = (cards) => {
        if (!cards.length) return '<div class="forward-lab-empty">No compare-vs-stock summary was available.</div>';
        return `<div class="forward-lab-summary-grid">${cards.map((card) => (
          `<article class="forward-lab-summary-card"><h4>${card.title}</h4>${card.pairs.map((pair) => `<div><strong>${pair[0]}:</strong> ${pair[1]}</div>`).join("")}</article>`
        )).join("")}</div>`;
      };
      const roots = Array.from(document.querySelectorAll("[data-forward-lab-root]"));
      roots.forEach((root) => {
        const payload = parseJsonScript(root.getAttribute("data-forward-data-id") || "");
        if (!payload) return;
        const rootId = root.getAttribute("data-forward-root-id") || "root";
        const fixedStrategy = root.getAttribute("data-forward-fixed-strategy") || "";
        const state = {
          strategy: fixedStrategy || payload.defaults.strategy,
          metric: payload.defaults.metric,
          mode: payload.defaults.mode,
          fixedIvCase: payload.defaults.fixed_iv_case,
          fixedHorizon: payload.defaults.fixed_horizon,
          fixedSpotCase: payload.defaults.fixed_spot_case,
        };
        const strategySelect = root.querySelector("[data-forward-strategy]");
        const metricSelect = root.querySelector("[data-forward-metric]");
        const fixedIvSelect = root.querySelector("[data-forward-fixed-iv]");
        const fixedHorizonSelect = root.querySelector("[data-forward-fixed-horizon]");
        const fixedSpotSelect = root.querySelector("[data-forward-fixed-spot]");
        const quickTarget = root.querySelector("[data-forward-quick-table]");
        const heatmapTarget = root.querySelector("[data-forward-heatmap]");
        const sliceTarget = root.querySelector("[data-forward-slice]");
        const compareTarget = root.querySelector("[data-forward-compare]");
        const detailTarget = root.querySelector("[data-forward-detail-table]");
        const noteTarget = root.querySelector("[data-forward-mode-note]");
        if (strategySelect && fixedStrategy) {
          strategySelect.closest(".forward-lab-control").hidden = true;
        }
        const render = () => {
          updateFixedControlVisibility(root, state);
          const config = modeMeta[state.mode];
          const dataset = payload.data[config.dataKey] || [];
          const filtered = dataset.filter((row) => {
            if (row.strategy !== state.strategy) return false;
            if (config.fixedKey === "iv_case" && row.iv_case !== state.fixedIvCase) return false;
            if (config.fixedKey === "horizon" && row.horizon !== state.fixedHorizon) return false;
            if (config.fixedKey === "spot_case" && row.spot_case !== state.fixedSpotCase) return false;
            return true;
          });
          const quickRows = (payload.data.quick || []).filter((row) => row.strategy === state.strategy);
          if (noteTarget) {
            noteTarget.textContent = config.note;
          }
          if (quickTarget) {
            quickTarget.innerHTML = renderTable(
              [
                { label: "Scenario", render: (row) => toTitle(row.spot_case) },
                { label: "Spot Price", render: (row) => formatValue(row.spot_price, "estimated_value") },
                { label: "Horizon", render: (row) => toTitle(row.horizon) },
                { label: "IV Case", render: (row) => toTitle(row.iv_case) },
                { label: "Modeled Value", render: (row) => formatValue(row.estimated_value, "estimated_value") },
                { label: "PnL $", render: (row) => formatValue(row.profit_loss, "profit_loss") },
                { label: "PnL % On Budget", render: (row) => formatValue(row.return_on_comparison_capital, "return_on_comparison_capital") },
                { label: "Vs Long Stock", render: (row) => formatValue(row.stock_relative_difference, "stock_relative_difference") },
              ],
              quickRows.filter((row) => ["bear", "flat", "bull"].includes(row.spot_case))
            );
          }
          if (heatmapTarget) {
            heatmapTarget.innerHTML = createHeatmapSvg(filtered, state, payload, rootId);
          }
          if (sliceTarget) {
            sliceTarget.innerHTML = createSliceSvg(filtered, state, payload, rootId);
          }
          if (compareTarget) {
            compareTarget.innerHTML = renderSummaryCards(summarizeQuickRows(quickRows, state, payload));
          }
          if (detailTarget) {
            const detailRows = filtered.slice().sort((left, right) => {
              if (config.sliceX === "spot_price") {
                return Number(left.spot_price) - Number(right.spot_price);
              }
              const order = payload.orders.horizons_x || [];
              return order.indexOf(left.horizon) - order.indexOf(right.horizon);
            }).slice(0, 18);
            detailTarget.innerHTML = renderTable(
              [
                { label: "Spot Case", render: (row) => row.spot_case ? toTitle(row.spot_case) : "—" },
                { label: "Horizon", render: (row) => toTitle(row.horizon) },
                { label: "IV Case", render: (row) => toTitle(row.iv_case) },
                { label: "Spot Price", render: (row) => formatValue(row.spot_price, "estimated_value") },
                { label: metricMeta[state.metric].label, render: (row) => formatValue(row[state.metric], state.metric) },
              ],
              detailRows
            );
          }
        };
        root.querySelectorAll("[data-forward-mode]").forEach((button) => {
          button.addEventListener("click", () => {
            state.mode = button.getAttribute("data-forward-mode") || state.mode;
            root.querySelectorAll("[data-forward-mode]").forEach((candidate) => {
              const active = candidate.getAttribute("data-forward-mode") === state.mode;
              candidate.classList.toggle("is-active", active);
              candidate.setAttribute("aria-selected", active ? "true" : "false");
            });
            render();
          });
        });
        if (strategySelect) {
          strategySelect.addEventListener("change", () => {
            state.strategy = strategySelect.value || state.strategy;
            render();
          });
        }
        if (metricSelect) {
          metricSelect.addEventListener("change", () => {
            state.metric = metricSelect.value || state.metric;
            render();
          });
        }
        if (fixedIvSelect) {
          fixedIvSelect.addEventListener("change", () => {
            state.fixedIvCase = fixedIvSelect.value || state.fixedIvCase;
            render();
          });
        }
        if (fixedHorizonSelect) {
          fixedHorizonSelect.addEventListener("change", () => {
            state.fixedHorizon = fixedHorizonSelect.value || state.fixedHorizon;
            render();
          });
        }
        if (fixedSpotSelect) {
          fixedSpotSelect.addEventListener("change", () => {
            state.fixedSpotCase = fixedSpotSelect.value || state.fixedSpotCase;
            render();
          });
        }
        if (strategySelect) strategySelect.value = state.strategy;
        if (metricSelect) metricSelect.value = state.metric;
        if (fixedIvSelect) fixedIvSelect.value = state.fixedIvCase;
        if (fixedHorizonSelect) fixedHorizonSelect.value = state.fixedHorizon;
        if (fixedSpotSelect) fixedSpotSelect.value = state.fixedSpotCase;
        root.querySelectorAll("[data-forward-mode]").forEach((button) => {
          const active = button.getAttribute("data-forward-mode") === state.mode;
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-selected", active ? "true" : "false");
        });
        render();
      });
    })();
  </script>
"""

BASE_STYLES = """
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f1ec;
      --panel: #ffffff;
      --panel-soft: #fbfaf7;
      --line: #d8d2c7;
      --line-strong: #bdb4a6;
      --text: #191714;
      --muted: #6b645b;
      --accent: #0f5c57;
      --accent-soft: #e7f3f2;
      --warning-bg: #fff5de;
      --warning-line: #d6b067;
      --warning-text: #6c5118;
      --shadow: 0 14px 40px rgba(26, 24, 21, 0.08);
      --ok-bg: #e8f5ee;
      --ok-text: #1d5f38;
      --partial-bg: #fff3df;
      --partial-text: #8a5a00;
      --insufficient-bg: #fae8e8;
      --insufficient-text: #8b2d2d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top right, rgba(15, 92, 87, 0.06), transparent 28%),
        linear-gradient(180deg, #f0ede6 0%, var(--bg) 100%);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.45;
    }
    a {
      color: var(--accent);
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .page {
      max-width: 1480px;
      margin: 0 auto;
      padding: 26px;
    }
    .hero,
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 22px 24px;
      margin-bottom: 18px;
    }
    .hero {
      padding-bottom: 18px;
    }
    .hero-top {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 16px;
    }
    .eyebrow {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.76rem;
      margin-bottom: 8px;
      font-weight: 600;
    }
    .hero h1 {
      margin: 0 0 8px;
      font-size: 2rem;
      line-height: 1.15;
    }
    .subtitle {
      color: var(--muted);
      margin: 0;
      max-width: 900px;
    }
    .summary-strip {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 8px;
    }
    .summary-cell,
    .metric-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-soft);
      padding: 12px 14px;
      min-height: 92px;
    }
    .summary-label,
    .metric-label {
      color: var(--muted);
      font-size: 0.78rem;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 600;
    }
    .summary-value,
    .metric-value {
      font-size: 1.05rem;
      font-weight: 600;
      word-break: break-word;
    }
    .metric-grid,
    .panel-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }
    .callout {
      border-left: 5px solid var(--accent);
      background: var(--accent-soft);
      border-radius: 14px;
      padding: 14px 16px;
    }
    .callout-title {
      font-weight: 700;
      margin-bottom: 6px;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 8px 14px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
      border: 1px solid transparent;
    }
    .status-ok {
      background: var(--ok-bg);
      color: var(--ok-text);
      border-color: rgba(29, 95, 56, 0.18);
    }
    .status-partial {
      background: var(--partial-bg);
      color: var(--partial-text);
      border-color: rgba(138, 90, 0, 0.18);
    }
    .status-insufficient-data {
      background: var(--insufficient-bg);
      color: var(--insufficient-text);
      border-color: rgba(139, 45, 45, 0.18);
    }
    .warning-panel {
      background: var(--warning-bg);
      border-color: var(--warning-line);
      color: var(--warning-text);
    }
    .section-intro {
      color: var(--muted);
      margin-top: -4px;
      margin-bottom: 14px;
    }
    .kv-wrap {
      overflow-x: auto;
    }
    .kv-table {
      width: 100%;
      border-collapse: collapse;
    }
    .kv-table th,
    .kv-table td {
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }
    .kv-table th {
      width: 260px;
      color: var(--muted);
      font-weight: 600;
    }
    .table-wrap {
      overflow: auto;
      max-height: 520px;
      border: 1px solid var(--line);
      border-radius: 14px;
    }
    .data-table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      font-size: 0.95rem;
    }
    .data-table thead th {
      position: sticky;
      top: 0;
      background: #f8f5ef;
      z-index: 1;
    }
    .data-table th,
    .data-table td {
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    .data-table tbody tr:nth-child(even) {
      background: #fcfaf6;
    }
    .data-table td.numeric,
    .data-table th.numeric {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .summary-section-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 14px;
    }
    .sticky-summary-strip {
      position: sticky;
      top: 12px;
      z-index: 4;
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(10px);
    }
    .summary-block {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel-soft);
      padding: 14px 16px;
    }
    .summary-block h3 {
      margin: 0 0 8px;
      font-size: 0.98rem;
    }
    .shareability-note {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      border-left: 5px solid var(--accent);
      background: var(--accent-soft);
      border-radius: 16px;
      padding: 16px 18px;
    }
    .shareability-copy strong {
      display: block;
      margin-bottom: 6px;
    }
    .meta-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border: 1px solid transparent;
      white-space: nowrap;
    }
    .meta-self-contained {
      background: var(--ok-bg);
      color: var(--ok-text);
      border-color: rgba(29, 95, 56, 0.18);
    }
    .meta-mostly-self-contained {
      background: #eef2f6;
      color: #39536a;
      border-color: rgba(57, 83, 106, 0.18);
    }
    .meta-companion {
      background: var(--partial-bg);
      color: var(--partial-text);
      border-color: rgba(138, 90, 0, 0.18);
    }
    .page-nav {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .section-stack {
      display: grid;
      gap: 18px;
    }
    .related-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .related-card {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel-soft);
      padding: 14px 16px;
    }
    .related-card-title {
      font-weight: 700;
      margin-bottom: 6px;
    }
    .related-card-meta {
      color: var(--muted);
      font-size: 0.92rem;
    }
    .summary-raw details,
    .detail-disclosure {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel-soft);
      padding: 10px 12px;
    }
    .summary-raw summary,
    .detail-disclosure summary {
      cursor: pointer;
      font-weight: 700;
      color: var(--text);
    }
    .plot-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
      gap: 16px;
    }
    .plot-card {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: var(--panel-soft);
    }
    .plot-card img {
      width: 100%;
      display: block;
      background: white;
    }
    .plot-card.featured {
      grid-column: 1 / -1;
    }
    .plot-card.featured img {
      min-height: 440px;
      object-fit: contain;
    }
    .plot-card figcaption {
      padding: 10px 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .chart-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      background: #fffdfa;
    }
    .chart-toolbar.chart-toolbar--stacked {
      align-items: flex-start;
      flex-wrap: wrap;
    }
    .chart-toolbar-copy {
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }
    .chart-action {
      appearance: none;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
    }
    .chart-action:hover {
      border-color: var(--line-strong);
    }
    .lightbox-html {
      width: 100%;
      overflow: auto;
    }
    .lightbox-html .forward-lab-svg-wrap,
    .lightbox-html .path-explorer-chart-wrap {
      max-width: none;
      padding: 12px;
      overflow: auto;
    }
    .lightbox-trigger {
      appearance: none;
      border: 0;
      width: 100%;
      padding: 0;
      margin: 0;
      display: block;
      background: transparent;
      cursor: zoom-in;
      text-align: left;
    }
    .lightbox-modal {
      position: fixed;
      inset: 0;
      background: rgba(19, 23, 26, 0.8);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 26px;
      z-index: 9999;
    }
    .lightbox-modal.is-open {
      display: flex;
    }
    .lightbox-dialog {
      width: min(1500px, 96vw);
      max-height: 94vh;
      background: #ffffff;
      border-radius: 22px;
      overflow: hidden;
      box-shadow: 0 28px 80px rgba(0, 0, 0, 0.28);
      display: flex;
      flex-direction: column;
    }
    .lightbox-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: #f8f5ef;
    }
    .lightbox-title {
      font-weight: 700;
    }
    .lightbox-close {
      appearance: none;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: white;
      font: inherit;
      font-weight: 700;
      padding: 8px 12px;
      cursor: pointer;
    }
    .lightbox-body {
      padding: 18px;
      overflow: auto;
      background: #ffffff;
    }
    .lightbox-body img {
      width: 100%;
      max-height: calc(94vh - 140px);
      object-fit: contain;
      display: block;
      background: white;
    }
    .lightbox-caption {
      color: var(--muted);
      padding: 0 18px 18px;
    }
    .bullet-list {
      margin: 0;
      padding-left: 18px;
    }
    .bullet-list li + li {
      margin-top: 6px;
    }
    .raw-files {
      columns: 2;
      gap: 24px;
    }
    .raw-files li {
      break-inside: avoid;
      margin-bottom: 8px;
    }
    .empty-state {
      color: var(--muted);
      margin: 0;
    }
    h2 {
      margin-top: 0;
      margin-bottom: 8px;
      font-size: 1.22rem;
    }
    h3 {
      margin-top: 18px;
      margin-bottom: 8px;
      font-size: 1rem;
    }
    code {
      background: #f1ede6;
      padding: 1px 6px;
      border-radius: 999px;
    }
    .muted {
      color: var(--muted);
    }
    .nav-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }
    .nav-card {
      display: block;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: var(--panel-soft);
      color: inherit;
    }
    .nav-card:hover {
      border-color: var(--line-strong);
      text-decoration: none;
    }
    .nav-title {
      font-weight: 700;
      margin-bottom: 6px;
    }
    .nav-meta {
      color: var(--muted);
      font-size: 0.92rem;
    }
    @media (max-width: 860px) {
      .hero-top {
        flex-direction: column;
      }
      .raw-files {
        columns: 1;
      }
      .shareability-note {
        flex-direction: column;
      }
      .sticky-summary-strip {
        position: static;
      }
      .plot-card.featured img {
        min-height: 260px;
      }
    }
  </style>
"""

LIGHTBOX_MODAL = """
  <div class="lightbox-modal" id="dashboard-lightbox" aria-hidden="true">
    <div class="lightbox-dialog" role="dialog" aria-modal="true" aria-label="Expanded chart view">
      <div class="lightbox-header">
        <div class="lightbox-title" id="dashboard-lightbox-title">Chart</div>
        <button type="button" class="lightbox-close" data-lightbox-close>Close</button>
      </div>
      <div class="lightbox-body">
        <img id="dashboard-lightbox-image" alt="">
        <div id="dashboard-lightbox-html" class="lightbox-html" hidden="hidden"></div>
      </div>
      <div class="lightbox-caption" id="dashboard-lightbox-caption"></div>
    </div>
  </div>
"""

LIGHTBOX_SCRIPT = """
  <script>
    (function () {
      const modal = document.getElementById("dashboard-lightbox");
      if (!modal) return;
      const image = document.getElementById("dashboard-lightbox-image");
      const htmlTarget = document.getElementById("dashboard-lightbox-html");
      const title = document.getElementById("dashboard-lightbox-title");
      const caption = document.getElementById("dashboard-lightbox-caption");
      const closeButton = modal.querySelector("[data-lightbox-close]");

      function closeModal() {
        modal.classList.remove("is-open");
        modal.setAttribute("aria-hidden", "true");
        image.removeAttribute("src");
        image.setAttribute("hidden", "hidden");
        if (htmlTarget) {
          htmlTarget.innerHTML = "";
          htmlTarget.setAttribute("hidden", "hidden");
        }
      }

      function openImage(src, label) {
        image.setAttribute("src", src);
        image.setAttribute("alt", label);
        image.removeAttribute("hidden");
        if (htmlTarget) {
          htmlTarget.innerHTML = "";
          htmlTarget.setAttribute("hidden", "hidden");
        }
        title.textContent = label;
        caption.textContent = label;
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");
      }

      function openHtml(selector, label) {
        const source = selector ? document.querySelector(selector) : null;
        if (!source || !htmlTarget) return;
        image.removeAttribute("src");
        image.setAttribute("hidden", "hidden");
        htmlTarget.innerHTML = source.innerHTML;
        htmlTarget.removeAttribute("hidden");
        title.textContent = label;
        caption.textContent = label;
        modal.classList.add("is-open");
        modal.setAttribute("aria-hidden", "false");
      }

      document.addEventListener("click", function (event) {
        const imageTrigger = event.target.closest("[data-lightbox-src]");
        if (imageTrigger) {
          const src = imageTrigger.getAttribute("data-lightbox-src");
          if (!src) return;
          const label = imageTrigger.getAttribute("data-lightbox-caption") || "Chart";
          openImage(src, label);
          return;
        }
        const htmlTrigger = event.target.closest("[data-lightbox-target]");
        if (htmlTrigger) {
          const selector = htmlTrigger.getAttribute("data-lightbox-target");
          if (!selector) return;
          const label = htmlTrigger.getAttribute("data-lightbox-caption") || "Chart";
          openHtml(selector, label);
        }
      });

      modal.addEventListener("click", function (event) {
        if (event.target === modal) {
          closeModal();
        }
      });
      if (closeButton) {
        closeButton.addEventListener("click", closeModal);
      }
      document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && modal.classList.contains("is-open")) {
          closeModal();
        }
      });
    })();
  </script>
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug_title(value: str | None) -> str:
    text = clean_string(value).replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "Unknown"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_markdown(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _clean_scalar(value: Any) -> Any:
    cleaned = make_json_safe(value)
    if isinstance(cleaned, str):
        text = cleaned.strip()
        return text or None
    return cleaned


def _is_windows_absolute_path(text: str) -> bool:
    stripped = clean_string(text)
    return bool(re.match(r"^[A-Za-z]:[\\/]", stripped))


def _sanitize_display_text(text: str, *, published: bool) -> str:
    normalized = clean_string(text)
    if not normalized:
        return normalized
    if _is_windows_absolute_path(normalized):
        return Path(normalized).name

    def replace_match(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            return Path(raw).name
        except (TypeError, ValueError):
            return raw

    sanitized = re.sub(r"[A-Za-z]:[\\/][^\s<>\"]+", replace_match, normalized)
    return sanitized


def _format_scalar(value: Any, *, published: bool = False) -> str:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return "N/A"
    if isinstance(cleaned, bool):
        return "Yes" if cleaned else "No"
    if isinstance(cleaned, int):
        return f"{cleaned:,}"
    if isinstance(cleaned, float):
        if not math.isfinite(cleaned):
            return "Unlimited"
        magnitude = abs(cleaned)
        if magnitude >= 1000:
            return f"{cleaned:,.2f}".rstrip("0").rstrip(".")
        if magnitude >= 1:
            return f"{cleaned:,.2f}".rstrip("0").rstrip(".")
        if magnitude == 0:
            return "0"
        return f"{cleaned:.4f}".rstrip("0").rstrip(".")
    return _sanitize_display_text(str(cleaned), published=published)


def _format_percent(value: Any, *, published: bool = False) -> str:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return "N/A"
    try:
        numeric = float(cleaned)
    except (TypeError, ValueError):
        return _sanitize_display_text(str(cleaned), published=published)
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric * 100:,.1f}%".rstrip("0").rstrip(".")


def _format_timestamp(value: Any) -> str:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return "N/A"
    try:
        timestamp = pd.to_datetime(cleaned, errors="raise")
    except (ValueError, TypeError):
        return str(cleaned)
    if pd.isna(timestamp):
        return str(cleaned)
    return timestamp.strftime("%Y-%m-%d %H:%M UTC")


def _relative_href(target_path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(target_path.resolve(), start=base_dir.resolve())).as_posix()


def _existing_relative_href(target_paths: list[Path], base_dir: Path) -> str:
    for candidate in target_paths:
        if candidate.exists():
            return _relative_href(candidate, base_dir)
    return ""


def _published_scenario_candidates(output_path: Path, best_expiry: str) -> list[Path]:
    if not best_expiry:
        return []
    candidates: list[Path] = []
    if output_path.parent.name == "contract-selection":
        candidates.append(output_path.parent.parent / "scenario" / best_expiry / "dashboard.html")
    if (
        output_path.parent.name == "publish"
        and len(output_path.parents) >= 4
        and output_path.parents[2].name == "contract_selection"
    ):
        scenario_root = output_path.parents[3] / "scenario"
        if scenario_root.exists():
            for bundle_dir in sorted(path for path in scenario_root.iterdir() if path.is_dir()):
                expiry_match = bundle_dir.name.endswith(best_expiry)
                if not expiry_match:
                    bundle_manifest = _load_json(bundle_dir / "bundle_manifest.json")
                    bundle_metadata = _load_json(bundle_dir / "metadata" / "report_metadata.json")
                    resolved_expiry = clean_string(
                        (bundle_metadata.get("metadata") or {}).get("expiry_date")
                        or bundle_manifest.get("run_slug")
                    )
                    expiry_match = resolved_expiry.endswith(best_expiry)
                if expiry_match:
                    candidates.append(bundle_dir / "publish" / "dashboard.html")
    return candidates


def _field_key(value: Any) -> str:
    text = clean_string(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _maybe_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_any_dtype(series):
        return pd.Series([pd.NaT] * len(series), index=series.index)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        return pd.Series([pd.NaT] * len(series), index=series.index)


def _date_sort_column(frame: pd.DataFrame) -> str | None:
    if frame.empty:
        return None
    normalized = {_field_key(column): str(column) for column in frame.columns}
    for candidate in DATE_COLUMN_PRIORITY:
        if candidate in normalized:
            column = normalized[candidate]
            parsed = _maybe_datetime_series(frame[column])
            if int(parsed.notna().sum()) >= 2:
                return column
    for column in frame.columns:
        key = _field_key(column)
        if "date" not in key and not key.endswith("_at"):
            continue
        parsed = _maybe_datetime_series(frame[column])
        if int(parsed.notna().sum()) >= 2:
            return str(column)
    return None


def _should_sort_latest_first(frame: pd.DataFrame, *, title: str, table_id: str) -> bool:
    if frame.empty or len(frame.index) < 2:
        return False
    normalized_id = _field_key(table_id)
    normalized_title = _field_key(title)
    if normalized_id in CHRONOLOGY_SORT_EXCLUDES:
        return False
    sort_column = _date_sort_column(frame)
    if sort_column is None:
        return False
    sort_key = _field_key(sort_column)
    if sort_key in {
        "published_at",
        "generated_at",
        "review_date",
        "entry_date",
        "exit_date",
        "valuation_date",
        "event_date",
        "date",
        "matched_date",
        "snapshot_date",
        "as_of_date",
    }:
        return True
    return any(hint in normalized_id or hint in normalized_title for hint in CHRONOLOGY_SORT_HINTS)


def _sort_frame_for_display(frame: pd.DataFrame, *, title: str, table_id: str) -> pd.DataFrame:
    if not _should_sort_latest_first(frame, title=title, table_id=table_id):
        return frame.copy()
    sort_column = _date_sort_column(frame)
    if sort_column is None:
        return frame.copy()
    display = frame.copy()
    parsed = _maybe_datetime_series(display[sort_column])
    display = display.assign(__sort_datetime__=parsed)
    display = display.sort_values(by="__sort_datetime__", ascending=False, na_position="last", kind="mergesort")
    return display.drop(columns=["__sort_datetime__"])


def _format_datetime_value(value: Any) -> str:
    cleaned = _clean_scalar(value)
    if cleaned is None:
        return "N/A"
    try:
        timestamp = pd.to_datetime(cleaned, errors="raise")
    except (ValueError, TypeError):
        return _sanitize_display_text(str(cleaned), published=False)
    if pd.isna(timestamp):
        return _sanitize_display_text(str(cleaned), published=False)
    if getattr(timestamp, "hour", 0) or getattr(timestamp, "minute", 0) or getattr(timestamp, "second", 0):
        return timestamp.strftime("%Y-%m-%d %H:%M")
    return timestamp.strftime("%Y-%m-%d")


def _image_src(path: Path, *, base_dir: Path, embed_images: bool) -> str:
    if embed_images:
        encoded = b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return Path(_relative_href(path, base_dir)).as_posix()


def _render_key_value_rows(items: list[tuple[str, Any]], *, published: bool = False) -> str:
    rows = []
    for label, value in items:
        if _clean_scalar(value) is None:
            continue
        rows.append(
            "<tr>"
            f"<th>{escape(label)}</th>"
            f"<td>{escape(_format_scalar(value, published=published))}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="empty-state">Not available.</p>'
    return '<div class="kv-wrap"><table class="kv-table">' + "".join(rows) + "</table></div>"


def _build_table_html(
    frame: pd.DataFrame,
    *,
    title: str,
    table_id: str,
    published: bool = False,
    table_class: str = "data-table",
) -> str:
    if frame.empty:
        return '<p class="empty-state">No rows available.</p>'
    prepared = _sort_frame_for_display(frame, title=title, table_id=table_id)
    display, numeric_columns = _format_frame(prepared, published=published)
    headers = []
    for original, label in zip(prepared.columns, display.columns):
        classes = "numeric" if original in numeric_columns else ""
        headers.append(f'<th class="{classes}">{escape(label)}</th>')
    body_rows = []
    for _, row in display.iterrows():
        cells = []
        for original, value in zip(prepared.columns, row.tolist()):
            classes = "numeric" if original in numeric_columns else ""
            cells.append(f'<td class="{classes}">{escape(str(value))}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap"><table id="{escape(table_id)}" class="{escape(table_class)}">'
        "<thead><tr>"
        + "".join(headers)
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table></div>"
    )


def _render_inline_dataframe(title: str, frame: pd.DataFrame, *, table_id: str, published: bool = False) -> str:
    return (
        f'<div class="inline-table-block"><h3>{escape(title)}</h3>'
        + _build_table_html(
            frame,
            title=title,
            table_id=table_id,
            published=published,
            table_class="data-table compact-data-table",
        )
        + "</div>"
    )


def _render_metric_cards(items: list[tuple[str, Any]], *, published: bool = False) -> str:
    cards = []
    for label, value in items:
        if _clean_scalar(value) is None:
            continue
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="metric-value">{escape(_format_scalar(value, published=published))}</div>'
            "</div>"
        )
    if not cards:
        return '<p class="empty-state">No key metrics were available.</p>'
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def _render_summary_strip(items: list[tuple[str, Any]], *, published: bool = False) -> str:
    cards = []
    for label, value in items:
        if _clean_scalar(value) is None:
            continue
        cards.append(
            '<div class="summary-cell">'
            f'<div class="summary-label">{escape(label)}</div>'
            f'<div class="summary-value">{escape(_format_scalar(value, published=published))}</div>'
            "</div>"
        )
    if not cards:
        return ""
    return '<div class="summary-strip">' + "".join(cards) + "</div>"


def _href_with_query_params(href: str, **params: str | None) -> str:
    base = clean_string(href)
    if not base:
        return ""
    if "#" in base:
        path, fragment = base.split("#", 1)
    else:
        path, fragment = base, ""
    cleaned = [(key, clean_string(value)) for key, value in params.items() if clean_string(value)]
    if cleaned:
        separator = "&" if "?" in path else "?"
        query_text = "&".join(f"{quote(key)}={quote(value)}" for key, value in cleaned)
        path = f"{path}{separator}{query_text}"
    return f"{path}#{fragment}" if fragment else path


def _render_lightbox_figure(
    *,
    src: str,
    caption: str,
    featured: bool = False,
) -> str:
    classes = "plot-card featured" if featured else "plot-card"
    safe_src = escape(src)
    safe_caption = escape(caption)
    return (
        f'<figure class="{classes}">'
        f'<button type="button" class="lightbox-trigger" data-lightbox-src="{safe_src}" data-lightbox-caption="{safe_caption}">'
        f'<img src="{safe_src}" alt="{safe_caption}">'
        "</button>"
        '<div class="chart-toolbar">'
        f"<figcaption>{safe_caption}</figcaption>"
        f'<button type="button" class="chart-action" data-lightbox-src="{safe_src}" data-lightbox-caption="{safe_caption}">Open larger chart</button>'
        "</div>"
        "</figure>"
    )


def _page_role_label(report_metadata: dict[str, Any]) -> str:
    kind = _report_kind(report_metadata)
    if kind == "contract_selection":
        return "Contract Selection Page"
    if kind == "scenario":
        return "Primary Decision Page"
    if kind == "replay":
        return "Historical Learning Page"
    if kind == "strategy":
        return "Supporting Strategy Detail"
    return "Analysis Detail"


def _scenario_payload(report_metadata: dict[str, Any]) -> dict[str, Any]:
    return report_metadata.get("scenario") or {}


def _contract_selection_payload(report_metadata: dict[str, Any]) -> dict[str, Any]:
    return report_metadata.get("contract_selection") or {}


def _snapshot_contract_selection_root(source_dir: Path, report_metadata: dict[str, Any]) -> Path | None:
    bundle_context = report_metadata.get("bundle_publish_context") or {}
    explicit_root = clean_string(bundle_context.get("contract_selection_root"))
    if explicit_root:
        candidate = (source_dir / explicit_root).resolve()
        if candidate.exists():
            return candidate
    metadata = report_metadata.get("metadata") or {}
    ticker = clean_string(metadata.get("ticker") or report_metadata.get("ticker")).upper()
    snapshot_date = clean_string(metadata.get("snapshot_date") or report_metadata.get("snapshot_date"))
    if not ticker or not snapshot_date:
        return None
    for candidate in [source_dir.resolve()] + list(source_dir.resolve().parents):
        parent = candidate.parent
        expected = parent / "contract-selection"
        if expected.exists():
            return expected
        if candidate.name == "contract-selection":
            return candidate
    root = source_dir.resolve().parents[1] if len(source_dir.resolve().parents) > 1 else None
    if root is None:
        return None
    expected = root / "contract-selection"
    return expected if expected.exists() else None


def _contract_selection_runs_for_snapshot(source_dir: Path, report_metadata: dict[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    required_artifacts = {
        "candidate_summary.csv",
        "required_path_rows.csv",
        "strategy_selector_rows.csv",
        "strategy_selector_rankings.csv",
        "path_case_chart_rows.csv",
        "path_case_strategy_rows.csv",
    }
    bundle_context = report_metadata.get("bundle_publish_context") or {}
    explicit_runs = bundle_context.get("related_contract_selection_runs") or []
    runs: list[tuple[Path, dict[str, Any]]] = []
    for run in explicit_runs:
        relative_dir = clean_string((run or {}).get("relative_dir"))
        if not relative_dir:
            continue
        candidate = (source_dir / relative_dir).resolve()
        metadata_path = candidate / "report_metadata.json"
        if not metadata_path.exists():
            continue
        payload = _load_json(metadata_path)
        if _report_kind(payload) != "contract_selection":
            continue
        payload["_compatible_for_embedding"] = all((candidate / name).exists() for name in required_artifacts)
        runs.append((candidate, payload))
    if runs:
        return runs
    root = _snapshot_contract_selection_root(source_dir, report_metadata)
    if root is None or not root.exists():
        return []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        metadata_path = candidate / "report_metadata.json"
        if not metadata_path.exists():
            continue
        payload = _load_json(metadata_path)
        if _report_kind(payload) != "contract_selection":
            continue
        payload["_compatible_for_embedding"] = all((candidate / name).exists() for name in required_artifacts)
        runs.append((candidate, payload))
    runs.sort(
        key=lambda item: (
            1 if bool(item[1].get("_compatible_for_embedding")) else 0,
            clean_string(item[1].get("generated_at") or (item[1].get("metadata") or {}).get("generated_at")),
        ),
        reverse=True,
    )
    return runs


def _historical_replay_payload(report_metadata: dict[str, Any]) -> dict[str, Any]:
    return report_metadata.get("replay") or {}


def _historical_replay_context_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    payload = _historical_replay_payload(report_metadata)
    metadata = report_metadata.get("metadata") or {}
    return [
        ("Ticker", payload.get("ticker") or report_metadata.get("ticker") or metadata.get("ticker")),
        ("Snapshot Date", payload.get("snapshot_date") or report_metadata.get("snapshot_date") or metadata.get("snapshot_date")),
        ("Expiry Date", payload.get("expiry_date") or report_metadata.get("expiry_date") or metadata.get("expiry_date")),
        ("Strategy", payload.get("strategy_name") or report_metadata.get("strategy_name") or metadata.get("strategy_name")),
        ("Comparison Capital", payload.get("comparison_capital") or report_metadata.get("comparison_capital")),
        ("Spot Price", payload.get("entry_spot") or metadata.get("spot_price")),
        ("Risk-Free Rate", payload.get("risk_free_rate") or metadata.get("risk_free_rate")),
        ("Generated", report_metadata.get("generated_at")),
    ]


def _historical_replay_metric_items(report_metadata: dict[str, Any], summary_df: pd.DataFrame) -> list[tuple[str, Any]]:
    if summary_df.empty:
        return []
    row = summary_df.iloc[0].to_dict()
    preferred = [
        ("Status", row.get("status") or report_metadata.get("status")),
        ("Anchor Checkpoint", row.get("anchor_checkpoint")),
        ("Anchor PnL", row.get("anchor_profit_loss")),
        ("Return On Capital", row.get("anchor_return_on_capital")),
        ("Return On Comparison Capital", row.get("anchor_return_on_comparison_capital")),
        ("Vs Long Stock", row.get("anchor_strategy_minus_stock_pnl")),
        ("Expected Move %", row.get("expected_move_pct_at_entry")),
        ("Actual Move %", row.get("actual_move_pct_anchor")),
    ]
    return [(label, value) for label, value in preferred if _clean_scalar(value) is not None]


def _scenario_strategy_title(strategy_name: str) -> str:
    return clean_string(strategy_name).replace("_", " ").title()


def _scenario_context_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    payload = _scenario_payload(report_metadata)
    metadata = report_metadata.get("metadata") or {}
    representative = payload.get("representative_horizon") or {}
    nearest_event = (_resolve_research_context(report_metadata).get("nearest_event") or {})
    event_label = None
    if nearest_event.get("event_type") or nearest_event.get("event_date"):
        event_bits = [clean_string(nearest_event.get("event_type")).replace("_", " ").title(), clean_string(nearest_event.get("event_date"))]
        event_label = " | ".join(bit for bit in event_bits if bit)
    return [
        ("Ticker", payload.get("ticker") or report_metadata.get("ticker") or metadata.get("ticker")),
        ("Snapshot Date", payload.get("snapshot_date") or report_metadata.get("snapshot_date") or metadata.get("snapshot_date")),
        ("Expiry Date", payload.get("expiry_date") or report_metadata.get("expiry_date") or metadata.get("expiry_date")),
        ("Report Type", "Scenario Dashboard"),
        ("Premium Mode", payload.get("premium_mode") or metadata.get("premium_mode")),
        ("Spot Price", payload.get("spot_price") or metadata.get("spot_price")),
        ("Risk-Free Rate", payload.get("risk_free_rate") or metadata.get("risk_free_rate")),
        ("Comparison Capital", payload.get("comparison_capital") or report_metadata.get("comparison_capital")),
        ("Nearest Event", event_label),
        ("Representative Horizon", representative.get("label")),
        ("Generated", report_metadata.get("generated_at")),
    ]


def _scenario_metric_items(report_metadata: dict[str, Any], summary_df: pd.DataFrame) -> list[tuple[str, Any]]:
    payload = _scenario_payload(report_metadata)
    executive = payload.get("executive_summary") or ((summary_df.iloc[0].to_dict()) if not summary_df.empty else {})
    preferred = [
        ("Comparison Capital", payload.get("comparison_capital") or executive.get("comparison_capital")),
        ("Included Strategies", executive.get("included_strategy_count") or len(payload.get("available_strategies") or [])),
        ("Best Bull Strategy", executive.get("best_bull_strategy")),
        ("Best Bull PnL", executive.get("best_bull_profit_loss")),
        ("Best Flat Strategy", executive.get("best_flat_strategy")),
        ("Best Flat PnL", executive.get("best_flat_profit_loss")),
        ("Best Defensive Strategy", executive.get("best_defensive_strategy")),
        ("Best Defensive PnL", executive.get("best_defensive_profit_loss")),
        ("Best Vs Stock", executive.get("best_equal_capital_relative_strategy")),
        ("Best Vs Stock Diff", executive.get("best_equal_capital_relative_diff")),
    ]
    return [(label, value) for label, value in preferred if _clean_scalar(value) is not None]


def _scenario_what_to_look_at(report_metadata: dict[str, Any], status: str) -> str:
    payload = _scenario_payload(report_metadata)
    available = payload.get("available_strategies") or []
    omitted = payload.get("omitted_strategies") or []
    comparison_capital = payload.get("comparison_capital") or report_metadata.get("comparison_capital")
    if status == "insufficient_data":
        return "Start with the warnings and assumptions first. This scenario dashboard still shows the stock baseline and any buildable structures, but sparse local option coverage limited the comparison set."
    if omitted:
        return (
            "Start with the strategy comparison row cards, then read the omitted-strategy notes so you know which structures "
            "were unavailable locally. The normalized decision view here uses "
            f"${float(comparison_capital):,.0f} of initial capital when that data is available, while still telling you whether one full unit actually fits that budget."
        )
    if "long_stock" in available:
        return (
            "Start with the four main visuals and Compare vs Stock. Then read the grouped decision snapshot to see "
            f"which structure wins under the ${float(comparison_capital):,.0f} normalized view, before drilling into one strategy's deeper chart set and valuation explanation."
        )
    return "Start with the executive charts first, then use the grouped summary, decision hints, and deep dives to compare how spot, time, and IV change the trade-offs."


def _scenario_how_to_read() -> str:
    return (
        "Read this page in three passes. First use the decision snapshot and four featured visuals to see which structures stand out under the normalized $1,000 lens. Then use Compare vs Stock and Replay / Case View to test whether that ranking still holds when spot, time, and IV assumptions change. Finally open one strategy deep dive and Explain Valuation to understand why the modeled value differs from pure payoff before expiry."
    )


def _scenario_strategy_note(strategy_name: str) -> str:
    notes = {
        "long_stock": "Long stock is the clean benchmark: no expiry, no IV exposure, full upside, and full downside dollar risk.",
        "long_call": "Long call keeps upside convexity, but time decay and IV compression can hurt if the move arrives too slowly.",
        "bull_call_spread": "Bull call spreads cap upside, but they often use capital more efficiently than a naked call when you expect a defined bullish path.",
        "long_put": "Long puts express downside directly, but timing matters because a late bearish move can arrive after decay has already done damage.",
        "bear_put_spread": "Bear put spreads cap the payoff, but they can reduce premium spent and make a bearish view easier to fund.",
        "covered_call": "Covered calls monetize upside you are willing to give away, but they still carry most of the stock's downside exposure.",
        "cash_secured_put": "Cash-secured puts get paid to wait, but the structure still needs downside tolerance because assignment risk sits under the premium earned.",
    }
    return notes.get(strategy_name, "Use break-even, max loss, and the deep-dive charts to judge how structure changes the path before expiry.")


def _scenario_budget_badge(row: dict[str, Any], *, published: bool = False) -> str:
    flag = row.get("fully_implementable_with_budget")
    comparison_capital = row.get("comparison_capital")
    if flag is True:
        label = f"Fits ${float(comparison_capital):,.0f} Budget" if _clean_scalar(comparison_capital) is not None else "Fits Budget"
        css_class = "budget-flag budget-flag-fit"
    elif flag is False:
        label = (
            f"Not Fully Implementable At ${float(comparison_capital):,.0f}"
            if _clean_scalar(comparison_capital) is not None
            else "Not Fully Implementable"
        )
        css_class = "budget-flag budget-flag-tight"
    else:
        return ""
    return f'<span class="{escape(css_class)}">{escape(_sanitize_display_text(label, published=published))}</span>'


def _scenario_summary_blocks(
    report_metadata: dict[str, Any],
    strategy_summary: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    published: bool = False,
) -> str:
    payload = _scenario_payload(report_metadata)
    executive = payload.get("executive_summary") or ((summary_df.iloc[0].to_dict()) if not summary_df.empty else {})
    hints = report_metadata.get("decision_hints") or payload.get("decision_hints") or {}
    context_items = [
        ("Comparison Capital", payload.get("comparison_capital") or report_metadata.get("comparison_capital")),
        ("Capital Sizing", payload.get("capital_sizing_mode") or report_metadata.get("capital_sizing_mode")),
        ("Representative Horizon", (payload.get("representative_horizon") or {}).get("label")),
        ("Representative IV Case", (payload.get("representative_iv_case") or {}).get("label")),
    ]
    affordable = strategy_summary.loc[
        pd.to_numeric(strategy_summary.get("affordable_units"), errors="coerce").fillna(-1) >= 1
    ] if not strategy_summary.empty and "affordable_units" in strategy_summary.columns else pd.DataFrame()
    blocks: list[tuple[str, list[tuple[str, Any]]]] = [
        (
            "Cost & Risk",
            [
                ("Comparison Capital", payload.get("comparison_capital") or executive.get("comparison_capital")),
                ("Included Strategies", executive.get("included_strategy_count") or len(payload.get("available_strategies") or [])),
                ("Lowest Max Loss", ((hints.get("lowest_max_loss") or {}).get("strategy"))),
                ("Lowest Max Loss Value", ((hints.get("lowest_max_loss") or {}).get("value"))),
            ],
        ),
        (
            "Upside & Break-Even",
            [
                ("Best Bull Case", executive.get("best_bull_strategy")),
                ("Best Bull PnL", executive.get("best_bull_profit_loss")),
                ("Best Capital Efficiency", ((hints.get("best_capital_efficiency") or {}).get("strategy"))),
                ("Capital Efficiency", ((hints.get("best_capital_efficiency") or {}).get("value"))),
            ],
        ),
        (
            "Scenario Outcomes",
            [
                ("Best Flat Case", executive.get("best_flat_strategy")),
                ("Best Flat PnL", executive.get("best_flat_profit_loss")),
                ("Best Bear Case", executive.get("best_defensive_strategy")),
                ("Bear Case PnL", executive.get("best_defensive_profit_loss")),
            ],
        ),
        (
            "Comparison Context",
            [
                ("Best Vs Stock", executive.get("best_equal_capital_relative_strategy")),
                ("Vs Stock Diff", executive.get("best_equal_capital_relative_diff")),
                ("Strategies That Fit Budget", len(affordable.index) if not affordable.empty else 0),
                ("Featured Focus Strategy", payload.get("featured_focus_strategy")),
                ("Capital Sizing", payload.get("capital_sizing_mode") or report_metadata.get("capital_sizing_mode")),
            ],
        ),
    ]
    rendered = []
    for title, items in blocks:
        visible = [item for item in items if _clean_scalar(item[1]) is not None]
        if not visible:
            continue
        rendered.append(
            '<section class="summary-block">'
            f"<h3>{escape(title)}</h3>"
            + _render_key_value_rows(visible, published=published)
            + "</section>"
        )
    if context_items and not rendered:
        rendered.append(
            '<section class="summary-block"><h3>Comparison Context</h3>'
            + _render_key_value_rows(context_items, published=published)
            + "</section>"
        )
    if not rendered:
        return ""
    return (
        '<section class="panel"><h2>Decision Snapshot</h2>'
        '<p class="section-intro">These grouped blocks keep the main decision frame readable: what it costs, how much can be lost, which path wins, and whether the structure truly fits the working budget.</p>'
        '<div class="summary-section-grid">'
        + "".join(rendered)
        + "</div></section>"
    )


def _render_scenario_decision_hints(report_metadata: dict[str, Any], *, published: bool = False) -> str:
    hints = report_metadata.get("decision_hints") or (_scenario_payload(report_metadata).get("decision_hints") or {})
    labels = [
        ("Best Bull Case", "best_bull_case"),
        ("Best Flat Case", "best_flat_case"),
        ("Best Downside Control", "best_downside_control"),
        ("Best Capital Efficiency", "best_capital_efficiency"),
        ("Lowest Max Loss", "lowest_max_loss"),
        ("Best Vs Stock", "best_vs_stock"),
    ]
    cards = []
    for label, key in labels:
        payload = hints.get(key) or {}
        if _clean_scalar(payload.get("strategy")) is None and _clean_scalar(payload.get("value")) is None:
            continue
        cards.append(
            '<article class="decision-hint-card">'
            f'<div class="decision-hint-label">{escape(label)}</div>'
            f'<div class="decision-hint-value">{escape(_format_scalar(payload.get("strategy"), published=published))}</div>'
            f'<div class="decision-hint-detail">{escape(_format_scalar(payload.get("value"), published=published))}</div>'
            "</article>"
        )
    if not cards:
        return ""
    return (
        '<section class="panel"><h2>Decision Hints</h2>'
        '<p class="section-intro">These hints are intentionally light-touch. They summarize which structure looks strongest under one lens, but you should still confirm the path in the charts below.</p>'
        '<div class="decision-hint-grid">'
        + "".join(cards)
        + "</div></section>"
    )


def _shareability_profile(
    *,
    published: bool,
    embed_images: bool,
    has_supporting_links: bool,
) -> tuple[str, str, str]:
    if published and embed_images and not has_supporting_links:
        return (
            "Fully Self-Contained",
            "This published page embeds its charts and key tables directly into the HTML and does not depend on companion files for the core reading experience.",
            "meta-self-contained",
        )
    if published and embed_images:
        return (
            "Mostly Self-Contained",
            "This published page embeds its main charts and key tables directly into the HTML. It is useful as a standalone file, while related-report and raw-file links still benefit from the rest of the published folder.",
            "meta-mostly-self-contained",
        )
    return (
        "Requires Companion Files For Drill-Down",
        "This page keeps the core narrative visible, but companion files remain important for charts, related-report links, or raw audit artifacts.",
        "meta-companion",
    )


def _render_shareability_note(
    *,
    published: bool,
    embed_images: bool,
    has_supporting_links: bool,
) -> str:
    label, body, badge_class = _shareability_profile(
        published=published,
        embed_images=embed_images,
        has_supporting_links=has_supporting_links,
    )
    return (
        '<section class="panel"><div class="shareability-note">'
        f'<div class="shareability-copy"><strong>Shareability</strong>{escape(body)}</div>'
        f'<span class="meta-badge {escape(badge_class)}">{escape(label)}</span>'
        "</div></section>"
    )


def _render_summary_overview(
    report_metadata: dict[str, Any],
    metric_items: list[tuple[str, Any]],
    top_items: list[tuple[str, Any]],
    summary_df: pd.DataFrame,
    *,
    published: bool = False,
) -> str:
    if not metric_items and not top_items:
        return ""
    context_priority = {
        "Ticker",
        "Snapshot Date",
        "Expiry Date",
        "Strategy",
        "Analysis",
        "Trade Id",
        "Trade Status",
        "Valuation Source",
        "Generated",
        "Spot Price",
        "Risk-Free Rate",
    }
    context_items = [item for item in top_items if item[0] in context_priority][:6]
    primary_items = metric_items[:4]
    exposure_items = metric_items[4:8]
    blocks: list[str] = []
    for title, items in [
        ("Decision Snapshot", primary_items),
        ("Pricing And Exposure", exposure_items),
        ("Context And Status", context_items),
    ]:
        visible = [item for item in items if _clean_scalar(item[1]) is not None]
        if not visible:
            continue
        blocks.append(
            '<section class="summary-block">'
            f"<h3>{escape(title)}</h3>"
            + _render_key_value_rows(visible, published=published)
            + "</section>"
        )
    if not blocks:
        return ""
    raw_summary = ""
    if len(summary_df.index) > 1:
        raw_summary = (
            '<div class="summary-raw" style="margin-top:14px;">'
            '<details class="detail-disclosure"><summary>Show raw summary rows</summary>'
            + _render_dataframe("Summary Rows", summary_df, table_id="summary-rows", published=published)
            + "</details></div>"
        )
    return (
        '<section class="panel"><h2>Summary Overview</h2>'
        '<p class="section-intro">Grouped summary fields make the decision profile easier to scan than a single long row of raw values.</p>'
        '<div class="summary-section-grid">'
        + "".join(blocks)
        + "</div>"
        + raw_summary
        + "</section>"
    )

def _format_frame(frame: pd.DataFrame, *, published: bool = False) -> tuple[pd.DataFrame, set[str]]:
    if frame.empty:
        return frame.copy(), set()
    display = frame.copy()
    numeric_columns: set[str] = set()
    for column in display.columns:
        series = display[column]
        column_key = _field_key(column)
        date_like_column = (
            pd.api.types.is_datetime64_any_dtype(series)
            or column_key in DATE_COLUMN_PRIORITY
            or "date" in column_key
            or column_key.endswith("_at")
            or column_key in {"time", "timestamp", "generated_at", "published_at"}
        )
        parsed_dates = _maybe_datetime_series(series) if date_like_column else pd.Series([pd.NaT] * len(series), index=series.index)
        if date_like_column and int(parsed_dates.notna().sum()) >= max(2, int(series.notna().sum() * 0.6)):
            display[column] = series.map(_format_datetime_value)
            continue
        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.add(column)
        display[column] = series.map(lambda value: _format_scalar(value, published=published))
    display.columns = [_slug_title(str(column)) for column in display.columns]
    return display, numeric_columns


def _render_dataframe(title: str, frame: pd.DataFrame, *, table_id: str, published: bool = False) -> str:
    return (
        f'<section class="panel"><h2>{escape(title)}</h2>'
        + _build_table_html(frame, title=title, table_id=table_id, published=published)
        + "</section>"
    )


def _render_plot_gallery(image_paths: list[Path], *, base_dir: Path, embed_images: bool) -> str:
    if not image_paths:
        return ""
    items = []
    for index, path in enumerate(image_paths):
        caption = _slug_title(path.stem)
        items.append(
            _render_lightbox_figure(
                src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
                caption=caption,
                featured=index == 0,
            )
        )
    return (
        '<section class="panel"><h2>Charts</h2>'
        '<p class="section-intro">Start with the larger lead chart first, then open any preview below in the built-in lightbox when you want a closer read.</p>'
        '<div class="plot-grid">'
        + "".join(items)
        + "</div></section>"
    )


def _bundle_file_map(report_metadata: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    file_map = (report_metadata or {}).get("bundle_file_map") or {}
    if not isinstance(file_map, dict):
        return {}
    return {str(section): payload for section, payload in file_map.items() if isinstance(payload, dict)}


def _bundle_section_paths(
    output_dir: Path,
    report_metadata: dict[str, Any] | None,
    section: str,
    *,
    suffix: str | None = None,
) -> list[Path]:
    section_map = _bundle_file_map(report_metadata).get(section) or {}
    paths: list[Path] = []
    for relative_path in section_map.values():
        relative = Path(clean_string(relative_path))
        if suffix and relative.suffix.lower() != suffix.lower():
            continue
        candidate = output_dir / relative.name
        if candidate.exists() and candidate.is_file():
            paths.append(candidate)
    return paths


def _discover_images(output_dir: Path, report_metadata: dict[str, Any] | None = None) -> list[Path]:
    images = _bundle_section_paths(output_dir, report_metadata, "charts", suffix=".png") or sorted(output_dir.glob("*.png"))
    order_map = {name: index for index, name in enumerate(PREFERRED_PLOT_ORDER)}
    return sorted(
        images,
        key=lambda path: (order_map.get(path.name, len(PREFERRED_PLOT_ORDER)), path.name),
    )


def _discover_images_with_fallback(
    primary_dir: Path,
    fallback_dir: Path | None = None,
    *,
    primary_metadata: dict[str, Any] | None = None,
    fallback_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    by_name: dict[str, Path] = {}
    if fallback_dir is not None and fallback_dir.exists():
        for path in _discover_images(fallback_dir, fallback_metadata):
            by_name[path.name] = path
    if primary_dir.exists():
        for path in _discover_images(primary_dir, primary_metadata):
            by_name[path.name] = path
    order_map = {name: index for index, name in enumerate(PREFERRED_PLOT_ORDER)}
    return sorted(
        by_name.values(),
        key=lambda path: (order_map.get(path.name, len(PREFERRED_PLOT_ORDER)), path.name),
    )


def _csv_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (PREFERRED_TABLE_ORDER.index(path.name), path.name)
    except ValueError:
        return (len(PREFERRED_TABLE_ORDER), path.name)


def _discover_tables(output_dir: Path, report_metadata: dict[str, Any] | None = None) -> list[Path]:
    tables = _bundle_section_paths(output_dir, report_metadata, "tables", suffix=".csv") or [path for path in output_dir.glob("*.csv") if path.is_file()]
    return sorted(tables, key=_csv_sort_key)


def _artifact_files(
    artifact_dir: Path,
    report_dir: Path | None = None,
    *,
    report_metadata: dict[str, Any] | None = None,
) -> list[Path]:
    file_map = _bundle_file_map(report_metadata)
    if artifact_dir.exists() and file_map:
        ordered: list[Path] = []
        seen: set[str] = set()
        for section in ["summary", "tables", "charts", "metadata"]:
            for path in _bundle_section_paths(artifact_dir, report_metadata, section):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                ordered.append(path)
        return ordered
    if artifact_dir.exists():
        return [path for path in sorted(artifact_dir.iterdir()) if path.is_file()]
    if report_dir is not None and report_dir.exists():
        return [path for path in sorted(report_dir.iterdir()) if path.is_file()]
    return []


def _dedupe_warnings(report_metadata: dict[str, Any]) -> list[str]:
    warnings = list(report_metadata.get("warnings") or [])
    strategy_report = report_metadata.get("strategy_report") or {}
    warnings.extend(strategy_report.get("warnings") or [])
    unique: list[str] = []
    for warning in warnings:
        text = clean_string(warning)
        if text and text not in unique:
            unique.append(text)
    return unique


def _report_kind(report_metadata: dict[str, Any]) -> str:
    explicit = clean_string(report_metadata.get("report_kind")).lower()
    if explicit:
        return explicit
    if isinstance(report_metadata.get("scenario"), dict):
        return "scenario"
    if isinstance(report_metadata.get("contract_selection"), dict):
        return "contract_selection"
    if isinstance(report_metadata.get("replay"), dict):
        return "replay"
    if isinstance(report_metadata.get("strategy_report"), dict):
        return "strategy"
    return "analysis"


def _strategy_status(report_metadata: dict[str, Any]) -> str:
    strategy_report = report_metadata.get("strategy_report") or {}
    resolved = strategy_report.get("resolved_metadata") or {}
    warnings = _dedupe_warnings(report_metadata)
    if clean_string(resolved.get("risk_free_rate_source")).lower() == "default_fallback":
        return "partial"
    if "moneyness" in clean_string(resolved.get("spot_price_note")).lower():
        return "partial"
    if any("default fallback" in item.lower() for item in warnings):
        return "partial"
    return "ok"


def _report_status(report_metadata: dict[str, Any]) -> str:
    status = clean_string(report_metadata.get("status")).lower()
    if status:
        return status
    if _report_kind(report_metadata) == "strategy":
        return _strategy_status(report_metadata)
    return "ok"


def _report_generated_at(report_dir: Path, report_metadata: dict[str, Any]) -> str:
    explicit = clean_string(report_metadata.get("generated_at"))
    if explicit:
        return explicit
    metadata_path = report_dir / "report_metadata.json"
    if metadata_path.exists():
        return datetime.fromtimestamp(metadata_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return _utc_now_iso()


def compute_local_coverage(
    ticker: str | None,
    snapshot_date: str | None,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return lightweight local coverage stats for one ticker/snapshot."""

    payload = {
        "chain_snapshot_count": None,
        "expiries_available_for_snapshot": None,
        "historical_prices_available": False,
        "research_context_available": False,
    }
    if not clean_string(ticker):
        return payload
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    try:
        from ..snapshots import list_snapshot_slices
    except ImportError:
        return payload
    try:
        slices = list_snapshot_slices(ticker, data_root=root)
    except (FileNotFoundError, LookupError, ValueError):
        slices = pd.DataFrame()
    if not slices.empty:
        payload["chain_snapshot_count"] = int(slices["snapshot_date"].nunique())
        if snapshot_date:
            try:
                snapshot_ts = pd.Timestamp(snapshot_date).normalize()
                matching = slices.loc[slices["snapshot_date"] == snapshot_ts]
                payload["expiries_available_for_snapshot"] = int(matching["expiry_date"].nunique()) if not matching.empty else 0
            except (ValueError, TypeError):
                payload["expiries_available_for_snapshot"] = None
    history_path = root / clean_string(ticker).upper() / "historical_prices" / "normalized"
    payload["historical_prices_available"] = history_path.exists()
    return payload


def render_html_document(
    title: str,
    body: str,
    *,
    extra_head: str = "",
    extra_body_end: str = "",
) -> str:
    """Render one full standalone HTML document."""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB4bWxucz0naHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmcnIHZpZXdCb3g9JzAgMCAzMiAzMic+PHJlY3Qgd2lkdGg9JzMyJyBoZWlnaHQ9JzMyJyByeD0nOCcgZmlsbD0nIzBmNWM1NycvPjxwYXRoIGQ9J005IDIwaDE0djNIOXptMC0xMWgxNHYzSDl6bTAgNWgxNHYzSDl6JyBmaWxsPSd3aGl0ZScvPjwvc3ZnPg==">
{BASE_STYLES}
{extra_head}
</head>
<body>
  <div class="page">
{body}
  </div>
{LIGHTBOX_MODAL}
{extra_body_end}
{LIGHTBOX_SCRIPT}
</body>
</html>
"""


def _status_badge(status: str) -> str:
    normalized = slugify(clean_string(status) or "ok")
    label = clean_string(status).replace("_", " ").upper() or "OK"
    return f'<span class="status-badge status-{escape(normalized)}">{escape(label)}</span>'


def _strategy_context_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    strategy_report = report_metadata.get("strategy_report") or {}
    summary = strategy_report.get("summary") or {}
    legs = strategy_report.get("legs") or []
    leg_labels = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        label = clean_string(leg.get("label"))
        asset_type = clean_string(leg.get("asset_type"))
        quantity = leg.get("quantity")
        if label:
            leg_labels.append(label)
        elif asset_type == "option":
            option_type = clean_string(leg.get("option_type")).upper()
            strike = leg.get("strike")
            expiry = clean_string(leg.get("expiry_date"))
            leg_labels.append(f"{quantity} {option_type} {strike} exp {expiry}")
        elif asset_type:
            leg_labels.append(f"{quantity} {asset_type}")
    return [
        ("Ticker", strategy_report.get("ticker")),
        ("Snapshot Date", strategy_report.get("snapshot_date")),
        ("Expiry Date", strategy_report.get("expiry_date")),
        ("Strategy", strategy_report.get("strategy")),
        ("Premium Mode", strategy_report.get("premium_mode")),
        ("Legs", " | ".join(leg_labels)),
        ("Generated", report_metadata.get("generated_at")),
        ("Spot Price", summary.get("spot_entry")),
        ("Spot Source", summary.get("spot_price_source")),
        ("Risk-Free Rate", summary.get("risk_free_rate")),
        ("Risk-Free Source", summary.get("risk_free_rate_source")),
    ]


def _generic_analysis_context_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    metadata = report_metadata.get("metadata") or {}
    return [
        ("Ticker", metadata.get("ticker")),
        ("Snapshot Date", metadata.get("snapshot_date")),
        ("Expiry Date", metadata.get("expiry_date")),
        ("Analysis", report_metadata.get("analysis_name")),
        ("Generated", report_metadata.get("generated_at")),
        ("Risk-Free Rate", metadata.get("risk_free_rate")),
        ("Risk-Free Source", metadata.get("risk_free_rate_source") or metadata.get("risk_free_source")),
        ("Valuation Source", metadata.get("valuation_source")),
        ("Status", report_metadata.get("status")),
    ]


def _strategy_metric_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    summary = (report_metadata.get("strategy_report") or {}).get("summary") or {}
    priority = [
        "break_even",
        "max_gain",
        "max_loss",
        "net_debit",
        "net_credit",
        "premium_paid",
        "premium_received",
        "capital_required",
        "return_on_premium",
        "return_on_capital",
        "days_to_expiry",
    ]
    items: list[tuple[str, Any]] = []
    for key in priority:
        if key in summary:
            items.append((_slug_title(key), summary.get(key)))
    if len(items) < 8:
        for key, value in summary.items():
            if key in priority:
                continue
            if _clean_scalar(value) is None:
                continue
            items.append((_slug_title(key), value))
            if len(items) >= 10:
                break
    return items


def _generic_analysis_metric_items(report_metadata: dict[str, Any], summary_df: pd.DataFrame) -> list[tuple[str, Any]]:
    if summary_df.empty:
        return []
    row = make_json_safe(summary_df.iloc[0].to_dict())
    preferred = [
        "status",
        "atm_iv",
        "mean_iv",
        "median_iv",
        "rv_20d",
        "rv_30d",
        "expected_move_pct",
        "realized_move_pct",
        "profit_loss",
        "estimated_value",
        "open_trade_count",
        "capital_required_total",
        "premium_at_risk_total",
        "entry_delta_proxy_total",
    ]
    items: list[tuple[str, Any]] = []
    for key in preferred:
        if key in row and _clean_scalar(row.get(key)) is not None:
            items.append((_slug_title(key), row.get(key)))
    if len(items) < 8:
        for key, value in row.items():
            if key in preferred or _clean_scalar(value) is None:
                continue
            items.append((_slug_title(key), value))
            if len(items) >= 10:
                break
    return items


def _render_leg_table(report_metadata: dict[str, Any], *, published: bool = False) -> str:
    strategy_report = report_metadata.get("strategy_report") or {}
    legs = strategy_report.get("legs") or []
    if not legs:
        return ""
    frame = pd.DataFrame(legs)
    return _render_dataframe("Leg Details", frame, table_id="leg-details", published=published)


def _resolve_research_context(report_metadata: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report_metadata.get("research_context"), dict):
        return report_metadata.get("research_context") or {}
    strategy_report = report_metadata.get("strategy_report") or {}
    resolved = strategy_report.get("resolved_metadata") or {}
    if isinstance(resolved.get("research_context"), dict):
        return resolved.get("research_context") or {}
    metadata = report_metadata.get("metadata") or {}
    if isinstance(metadata.get("research_context"), dict):
        return metadata.get("research_context") or {}
    frozen_entry = metadata.get("frozen_entry_context") or report_metadata.get("frozen_entry_context") or {}
    if isinstance(frozen_entry.get("research_context_at_entry"), dict):
        return frozen_entry.get("research_context_at_entry") or {}
    return {}


def _research_availability(research_context: dict[str, Any]) -> list[tuple[str, Any]]:
    expected_move = research_context.get("expected_move") or {}
    options_overview = research_context.get("options_overview") or {}
    nearest_event = research_context.get("nearest_event") or {}
    dividend = research_context.get("dividend_assumption") or {}
    notes = research_context.get("notes") or []
    return [
        ("Expected Move Available", "Yes" if expected_move.get("matched") else "No"),
        ("Options Overview Available", "Yes" if options_overview.get("matched") else "No"),
        ("Nearest Event Available", "Yes" if nearest_event.get("matched") else "No"),
        ("Dividend Assumption Available", "Yes" if dividend.get("matched") else "No"),
        ("Research Notes", len(notes)),
    ]


def _render_research_context(report_metadata: dict[str, Any], *, published: bool = False) -> str:
    research_context = _resolve_research_context(report_metadata)
    if not research_context:
        return ""
    blocks: list[str] = []
    expected_move = research_context.get("expected_move") or {}
    if expected_move:
        blocks.append(
            _render_key_value_rows(
                [
                    ("Matched", expected_move.get("matched")),
                    ("Snapshot Date", expected_move.get("matched_snapshot_date") or expected_move.get("snapshot_date")),
                    ("Expiry Date", expected_move.get("expiry_date")),
                    ("Expected Move Abs", expected_move.get("expected_move_abs")),
                    ("Expected Move Pct", expected_move.get("expected_move_pct")),
                    ("Lower Bound", expected_move.get("lower_bound")),
                    ("Upper Bound", expected_move.get("upper_bound")),
                    ("Implied Volatility", expected_move.get("implied_volatility")),
                    ("Source", expected_move.get("source")),
                    ("Notes", expected_move.get("notes")),
                ],
                published=published,
            )
        )
    options_overview = research_context.get("options_overview") or {}
    if options_overview:
        blocks.append(
            _render_key_value_rows(
                [
                    ("Matched", options_overview.get("matched")),
                    ("Snapshot Date", options_overview.get("matched_snapshot_date") or options_overview.get("snapshot_date")),
                    ("IV Rank", options_overview.get("iv_rank")),
                    ("IV Percentile", options_overview.get("iv_percentile")),
                    ("Historic Volatility", options_overview.get("historic_volatility")),
                    ("Put/Call Volume Ratio", options_overview.get("put_call_volume_ratio")),
                    ("Put/Call OI Ratio", options_overview.get("put_call_open_interest_ratio")),
                    ("Source", options_overview.get("source")),
                    ("Notes", options_overview.get("notes")),
                ],
                published=published,
            )
        )
    nearest_event = research_context.get("nearest_event") or {}
    if nearest_event:
        blocks.append(
            _render_key_value_rows(
                [
                    ("Matched", nearest_event.get("matched")),
                    ("Event Type", nearest_event.get("event_type")),
                    ("Event Date", nearest_event.get("event_date")),
                    ("Event Time", nearest_event.get("event_time")),
                    ("Days To Event", nearest_event.get("days_to_event")),
                    ("Occurs Before Expiry", nearest_event.get("occurs_before_expiry")),
                    ("Source", nearest_event.get("source")),
                    ("Notes", nearest_event.get("notes")),
                ],
                published=published,
            )
        )
    dividend_assumption = research_context.get("dividend_assumption") or {}
    if dividend_assumption:
        blocks.append(
            _render_key_value_rows(
                [
                    ("Matched", dividend_assumption.get("matched")),
                    ("Snapshot Date", dividend_assumption.get("matched_snapshot_date") or dividend_assumption.get("snapshot_date")),
                    ("Dividend Yield", dividend_assumption.get("dividend_yield")),
                    ("Expected Dividend Date", dividend_assumption.get("expected_dividend_date")),
                    ("Source", dividend_assumption.get("source")),
                    ("Notes", dividend_assumption.get("notes")),
                ],
                published=published,
            )
        )
    notes = research_context.get("notes") or []
    if notes:
        notes_frame = pd.DataFrame(notes[:5])
        blocks.append(_render_dataframe("Research Notes", notes_frame, table_id="research-notes", published=published))
    if not blocks:
        return ""
    return (
        '<section class="panel"><h2>Research Context</h2>'
        '<p class="section-intro">Local expected-move, volatility, event, dividend, and note context resolved for this report.</p>'
        + "".join(blocks)
        + "</section>"
    )


def _render_assumptions(report_metadata: dict[str, Any], *, published: bool = False) -> str:
    kind = _report_kind(report_metadata)
    if kind == "strategy":
        strategy_report = report_metadata.get("strategy_report") or {}
        resolved = strategy_report.get("resolved_metadata") or {}
        items = [
            ("Spot Price", strategy_report.get("entry_spot")),
            ("Spot Source", resolved.get("spot_price_source")),
            ("Spot Matched Date", resolved.get("spot_price_matched_date")),
            ("Spot Resolution Note", resolved.get("spot_price_note")),
            ("Risk-Free Rate", strategy_report.get("risk_free_rate")),
            ("Risk-Free Source", resolved.get("risk_free_rate_source")),
            ("Risk-Free Series", resolved.get("risk_free_rate_series")),
            ("Risk-Free Matched Date", resolved.get("risk_free_rate_matched_date")),
            ("Dividend Yield", strategy_report.get("dividend_yield")),
            ("Premium Mode", strategy_report.get("premium_mode")),
            ("Snapshot File", strategy_report.get("source_snapshot_file")),
        ]
    else:
        metadata = report_metadata.get("metadata") or {}
        items = [
            ("Ticker", metadata.get("ticker")),
            ("Snapshot Date", metadata.get("snapshot_date")),
            ("Expiry Date", metadata.get("expiry_date")),
            ("Valuation Source", metadata.get("valuation_source")),
            ("Compare Dates", ", ".join(metadata.get("compare_dates") or [])),
            ("As Of Date", metadata.get("as_of_date")),
        ]
    rows = _render_key_value_rows(items, published=published)
    if not rows:
        return ""
    return (
        '<section class="panel"><h2>Inputs And Assumptions</h2>'
        '<p class="section-intro">Resolved pricing inputs, matched dates, and source notes used when the report was built.</p>'
        + rows
        + "</section>"
    )
def _render_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{escape(warning)}</li>" for warning in warnings)
    return (
        '<section class="panel warning-panel"><h2>Warnings And Fallback Notes</h2>'
        '<ul class="bullet-list">'
        + items
        + "</ul></section>"
    )


def _render_available_files(files: list[Path], *, base_dir: Path) -> str:
    visible = [path for path in files if path.name not in HIDDEN_FILES]
    if not visible:
        return ""
    rows = []
    for path in visible:
        href = _relative_href(path, base_dir=base_dir)
        rows.append(
            f'<tr><th>{escape(path.name)}</th><td><a href="{escape(href)}">{escape(path.name)}</a></td></tr>'
        )
    return (
        '<section class="panel"><h2>Raw Files</h2>'
        '<p class="section-intro">The raw audit layer remains available alongside this dashboard.</p>'
        '<div class="kv-wrap"><table class="kv-table"><tbody>'
        + "".join(rows)
        + "</tbody></table></div></section>"
    )


def _find_dashboards_root(start_path: Path) -> Path | None:
    current = start_path.resolve()
    for candidate in [current] + list(current.parents):
        if (candidate / "library_manifest.json").exists():
            return candidate
    return None


def _load_library_records(dashboards_root: Path | None) -> list[dict[str, Any]]:
    if dashboards_root is None:
        return []
    manifest_path = dashboards_root / "library_manifest.json"
    if not manifest_path.exists():
        return []
    payload = _load_json(manifest_path)
    return list(payload.get("records") or [])


def _current_published_record(
    artifact_dir: Path,
    dashboards_root: Path | None,
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if dashboards_root is None:
        return None
    try:
        relative_dir = artifact_dir.resolve().relative_to(dashboards_root.resolve()).as_posix()
    except ValueError:
        return None
    for record in records:
        if clean_string(record.get("published_dir")).replace("\\", "/") == relative_dir:
            return record
    return None


def _record_title(record: dict[str, Any]) -> str:
    category = clean_string(record.get("category"))
    if category == "contract_selection":
        return "Contract Selection"
    if category == "scenario":
        expiry = clean_string(record.get("expiry_date"))
        return f"Primary Scenario Dashboard {expiry}".strip()
    if category == "replay":
        strategy = clean_string(record.get("strategy_name")).replace("-", " ").title() or "Historical Replay"
        if record.get("history_mode"):
            return f"{strategy} Replay History"
        expiry = clean_string(record.get("expiry_date"))
        return f"{strategy} Historical Replay {expiry}".strip()
    if category == "strategy":
        return f"{clean_string(record.get('strategy_name') or 'strategy').replace('-', ' ').title()} Strategy"
    return clean_string(record.get("analysis_name") or "analysis").replace("-", " ").title()


def _record_role_label(record: dict[str, Any]) -> str:
    category = clean_string(record.get("category"))
    if category == "contract_selection":
        return "Contract Selection Page"
    if category == "scenario":
        return "Primary Decision Page"
    if category == "replay":
        return "Historical Learning Page"
    if category == "strategy":
        return "Supporting Strategy Detail"
    return "Analysis Detail"


def _related_report_items(
    report_metadata: dict[str, Any],
    *,
    artifact_dir: Path,
) -> list[tuple[str, str, str]]:
    dashboards_root = _find_dashboards_root(artifact_dir)
    records = _load_library_records(dashboards_root)
    current = _current_published_record(artifact_dir, dashboards_root, records)
    if dashboards_root is None or current is None:
        return []

    identity = clean_string(current.get("identity"))
    ticker = clean_string(current.get("ticker") or (report_metadata.get("metadata") or {}).get("ticker")).upper()
    snapshot_date = clean_string(current.get("snapshot_date") or (report_metadata.get("metadata") or {}).get("snapshot_date") or (report_metadata.get("strategy_report") or {}).get("snapshot_date"))
    category = clean_string(current.get("category"))
    item_dir = artifact_dir
    sibling_records = [record for record in records if clean_string(record.get("identity")) != identity]

    seen: set[str] = set()
    items: list[tuple[str, str, str]] = []

    def add_item(target: Path, label: str, meta: str) -> None:
        href = _relative_href(target, item_dir)
        key = f"{href}|{label}"
        if key in seen:
            return
        seen.add(key)
        items.append((href, label, meta))

    if dashboards_root is not None:
        add_item(dashboards_root / "index.html", "Dashboards library", "Main published library entry point")
        add_item(dashboards_root / "all_dashboards.html", "All dashboards", "Flat browse across all published dashboards")
    if ticker:
        add_item(dashboards_root / ticker / "index.html", f"{ticker} dashboard index", "Ticker-level published index")
    if ticker and snapshot_date:
        anchor = {"contract_selection": "summary", "scenario": "summary", "replay": "summary", "strategy": "strategies"}.get(category, "summary")
        add_item(
            dashboards_root / ticker / snapshot_date / "index.html",
            "Snapshot Hub",
            f"Jump back into the Snapshot Hub#{anchor} for this date",
        )

    if ticker and snapshot_date and category != "contract_selection":
        contract_records = [
            record
            for record in sibling_records
            if clean_string(record.get("ticker")).upper() == ticker
            and clean_string(record.get("snapshot_date")) == snapshot_date
            and clean_string(record.get("category")) == "contract_selection"
        ]
        for record in contract_records[:1]:
            add_item(
                dashboards_root / clean_string(record.get("published_dashboard")),
                _record_title(record),
                "Strike / expiry choice page for the same snapshot",
            )

    if ticker and snapshot_date and category != "scenario":
        scenario_records = [
            record
            for record in sibling_records
            if clean_string(record.get("ticker")).upper() == ticker
            and clean_string(record.get("snapshot_date")) == snapshot_date
            and clean_string(record.get("category")) == "scenario"
        ]
        scenario_records = sorted(
            scenario_records,
            key=lambda record: (
                clean_string(record.get("expiry_date")),
                clean_string(record.get("published_at") or record.get("generated_at")),
            ),
            reverse=True,
        )
        for record in scenario_records[:2]:
            add_item(
                dashboards_root / clean_string(record.get("published_dashboard")),
                _record_title(record),
                "Primary Decision Page for the same snapshot",
            )

    if ticker and snapshot_date and category != "strategy":
        strategy_records = [
            record
            for record in sibling_records
            if clean_string(record.get("ticker")).upper() == ticker
            and clean_string(record.get("snapshot_date")) == snapshot_date
            and clean_string(record.get("category")) == "strategy"
        ]
        strategy_records = sorted(
            strategy_records,
            key=lambda record: (
                clean_string(record.get("expiry_date")),
                _record_title(record),
            ),
            reverse=True,
        )
        for record in strategy_records[:3]:
            add_item(
                dashboards_root / clean_string(record.get("published_dashboard")),
                _record_title(record),
                "Supporting strategy detail for the same snapshot",
            )

    if ticker and snapshot_date and category != "replay":
        replay_records = [
            record
            for record in sibling_records
            if clean_string(record.get("ticker")).upper() == ticker
            and clean_string(record.get("snapshot_date")) == snapshot_date
            and clean_string(record.get("category")) == "replay"
            and not bool(record.get("history_mode"))
        ]
        replay_records = sorted(
            replay_records,
            key=lambda record: (
                clean_string(record.get("expiry_date")),
                _record_title(record),
            ),
            reverse=True,
        )
        for record in replay_records[:3]:
            add_item(
                dashboards_root / clean_string(record.get("published_dashboard")),
                _record_title(record),
                "Historical learning page for the same snapshot",
            )

    return items


def _render_related_reports(report_metadata: dict[str, Any], *, artifact_dir: Path) -> str:
    items = _related_report_items(report_metadata, artifact_dir=artifact_dir)
    if not items:
        return ""
    cards = []
    for href, label, meta in items:
        cards.append(
            '<article class="related-card">'
            f'<div class="related-card-title"><a href="{escape(href)}">{escape(label)}</a></div>'
            f'<div class="related-card-meta">{escape(meta)}</div>'
            "</article>"
        )
    return (
        '<section class="panel"><h2>Related Reports</h2>'
        '<p class="section-intro">Use these links to move between the Snapshot Hub, the Primary Scenario Dashboard, and supporting detail dashboards without guessing how they fit together.</p>'
        '<div class="related-grid">'
        + "".join(cards)
        + "</div></section>"
    )


def _render_strategy_tables(report_dir: Path, *, published: bool = False) -> str:
    sections: list[str] = []
    scenarios = _load_csv(report_dir / "scenarios.csv")
    if scenarios is not None:
        sections.append(_render_dataframe("Scenario Table", scenarios, table_id="scenario-table", published=published))
    scenario_cases = _load_csv(report_dir / "scenario_cases.csv")
    if scenario_cases is not None:
        sections.append(_render_dataframe("Scenario Cases", scenario_cases, table_id="scenario-cases", published=published))
    comparison = _load_csv(report_dir / "comparison.csv")
    if comparison is not None:
        sections.append(_render_dataframe("Comparison", comparison, table_id="comparison", published=published))
    return "".join(sections)


def _render_generic_analysis_tables(report_dir: Path, *, report_metadata: dict[str, Any] | None = None, published: bool = False) -> str:
    sections: list[str] = []
    for path in _discover_tables(report_dir, report_metadata):
        if path.name == "summary.csv":
            continue
        frame = _load_csv(path)
        if frame is None:
            continue
        sections.append(_render_dataframe(_slug_title(path.stem), frame, table_id=slugify(path.stem), published=published))
    return "".join(sections)


def _page_title(report_metadata: dict[str, Any]) -> str:
    kind = _report_kind(report_metadata)
    if kind == "contract_selection":
        payload = _contract_selection_payload(report_metadata)
        metadata = report_metadata.get("metadata") or {}
        ticker = clean_string(payload.get("ticker") or metadata.get("ticker")).upper()
        snapshot_date = clean_string(payload.get("snapshot_date") or metadata.get("snapshot_date"))
        if ticker and snapshot_date:
            return f"{ticker} Contract Selection {snapshot_date}"
        if ticker:
            return f"{ticker} Contract Selection"
        return "Contract Selection"
    if kind == "scenario":
        payload = _scenario_payload(report_metadata)
        ticker = clean_string(payload.get("ticker") or report_metadata.get("ticker") or (report_metadata.get("metadata") or {}).get("ticker")).upper()
        expiry = clean_string(payload.get("expiry_date") or report_metadata.get("expiry_date") or (report_metadata.get("metadata") or {}).get("expiry_date"))
        if ticker and expiry:
            return f"{ticker} Scenario Dashboard {expiry}"
        if ticker:
            return f"{ticker} Scenario Dashboard"
        return "Scenario Dashboard"
    if kind == "replay":
        payload = _historical_replay_payload(report_metadata)
        ticker = clean_string(payload.get("ticker") or report_metadata.get("ticker") or (report_metadata.get("metadata") or {}).get("ticker")).upper()
        strategy = clean_string(payload.get("strategy_name") or report_metadata.get("strategy_name") or (report_metadata.get("metadata") or {}).get("strategy_name")).replace("_", " ").title()
        if (report_metadata.get("metadata") or {}).get("history_mode"):
            if ticker and strategy:
                return f"{ticker} {strategy} Replay History"
            if strategy:
                return f"{strategy} Replay History"
            return "Historical Replay History"
        expiry = clean_string(payload.get("expiry_date") or report_metadata.get("expiry_date") or (report_metadata.get("metadata") or {}).get("expiry_date"))
        if ticker and strategy and expiry:
            return f"{ticker} {strategy} Historical Replay {expiry}"
        if ticker and strategy:
            return f"{ticker} {strategy} Historical Replay"
        return "Historical Replay"
    if kind == "strategy":
        strategy_report = report_metadata.get("strategy_report") or {}
        strategy = clean_string(strategy_report.get("strategy")).replace("_", " ").title()
        ticker = clean_string(strategy_report.get("ticker")).upper()
        return f"{ticker} {strategy} Dashboard".strip()
    ticker = clean_string((report_metadata.get("metadata") or {}).get("ticker")).upper()
    analysis = clean_string(report_metadata.get("analysis_name")).replace("_", " ").title()
    if ticker and analysis:
        return f"{ticker} {analysis} Dashboard"
    if analysis:
        return f"{analysis} Dashboard"
    return "Options Lab Dashboard"


def _top_summary_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    kind = _report_kind(report_metadata)
    if kind == "strategy":
        return _strategy_context_items(report_metadata)
    if kind == "replay":
        return _historical_replay_context_items(report_metadata)
    return _generic_analysis_context_items(report_metadata)


def _provenance_items(report_metadata: dict[str, Any]) -> list[tuple[str, Any]]:
    kind = _report_kind(report_metadata)
    research_context = _resolve_research_context(report_metadata)
    if kind == "replay":
        payload = _historical_replay_payload(report_metadata)
        metadata = report_metadata.get("metadata") or {}
        return [
            ("Spot Price", metadata.get("spot_price")),
            ("Spot Source", metadata.get("spot_source")),
            ("Spot Matched Date", metadata.get("spot_matched_date")),
            ("Spot Resolution", metadata.get("spot_note")),
            ("Risk-Free Rate", metadata.get("risk_free_rate")),
            ("Risk-Free Source", metadata.get("risk_free_source")),
            ("Risk-Free Series", metadata.get("risk_free_series")),
            ("Risk-Free Matched Date", metadata.get("risk_free_matched_date")),
            ("Valuation Source Rollup", report_metadata.get("valuation_source_rollup")),
        ]
    if kind == "strategy":
        strategy_report = report_metadata.get("strategy_report") or {}
        resolved = strategy_report.get("resolved_metadata") or {}
        return [
            ("Spot Price", strategy_report.get("entry_spot")),
            ("Spot Source", resolved.get("spot_price_source")),
            ("Spot Matched Date", resolved.get("spot_price_matched_date")),
            ("Spot Resolution", resolved.get("spot_price_note")),
            ("Risk-Free Rate", strategy_report.get("risk_free_rate")),
            ("Risk-Free Source", resolved.get("risk_free_rate_source")),
            ("Risk-Free Matched Date", resolved.get("risk_free_rate_matched_date")),
            ("Research Metadata", "Yes" if research_context else "No"),
            ("Explicit Override Used", "Yes" if resolved.get("spot_price_source") == "override" else "No"),
        ]
    metadata = report_metadata.get("metadata") or {}
    frozen_entry = metadata.get("frozen_entry_context") or {}
    return [
        ("Valuation Source", metadata.get("valuation_source")),
        ("Spot At Entry", frozen_entry.get("spot_at_entry")),
        ("Spot Source", frozen_entry.get("spot_source")),
        ("Spot Matched Date", frozen_entry.get("spot_matched_date")),
        ("Risk-Free Rate", frozen_entry.get("risk_free_rate_at_entry") or metadata.get("risk_free_rate")),
        ("Risk-Free Source", frozen_entry.get("risk_free_source") or metadata.get("risk_free_rate_source") or metadata.get("risk_free_source")),
        ("Risk-Free Matched Date", frozen_entry.get("risk_free_matched_date") or metadata.get("risk_free_rate_matched_date")),
        ("Risk-Free Series", metadata.get("risk_free_rate_series") or metadata.get("risk_free_series")),
        ("Risk-Free Note", metadata.get("risk_free_rate_note") or metadata.get("risk_free_note")),
        ("Source Snapshot Storage", ", ".join(metadata.get("source_snapshot_storage_locations") or [])),
        ("Source Snapshot Files", ", ".join(metadata.get("source_snapshot_files") or [])),
        ("Research Metadata", "Yes" if research_context else "No"),
        ("Historical Prices Used", "Yes" if metadata.get("price_history_used") else "No"),
    ]


def _coverage_items(report_metadata: dict[str, Any], coverage: dict[str, Any]) -> list[tuple[str, Any]]:
    ticker = None
    snapshot_date = None
    kind = _report_kind(report_metadata)
    if kind == "strategy":
        strategy_report = report_metadata.get("strategy_report") or {}
        ticker = strategy_report.get("ticker")
        snapshot_date = strategy_report.get("snapshot_date")
    elif kind == "replay":
        payload = _historical_replay_payload(report_metadata)
        metadata = report_metadata.get("metadata") or {}
        ticker = payload.get("ticker") or metadata.get("ticker")
        snapshot_date = payload.get("snapshot_date") or metadata.get("snapshot_date")
    else:
        metadata = report_metadata.get("metadata") or {}
        ticker = metadata.get("ticker")
        snapshot_date = metadata.get("snapshot_date")
    research_context = _resolve_research_context(report_metadata)
    resolved = (report_metadata.get("strategy_report") or {}).get("resolved_metadata") or {}
    return [
        ("Ticker", ticker),
        ("Snapshot Date", snapshot_date),
        ("Chain Snapshots Available", coverage.get("chain_snapshot_count")),
        ("Expiries For Snapshot", coverage.get("expiries_available_for_snapshot")),
        ("Expected Move Available", "Yes" if (research_context.get("expected_move") or {}).get("matched") else "No"),
        ("Options Overview Available", "Yes" if (research_context.get("options_overview") or {}).get("matched") else "No"),
        ("Nearest Event Available", "Yes" if (research_context.get("nearest_event") or {}).get("matched") else "No"),
        ("FRED Local Store Used", "Yes" if clean_string(resolved.get("risk_free_rate_source")).lower() == "fred_local_store" else "No"),
        ("Historical Prices Used", "Yes" if coverage.get("historical_prices_available") else "No"),
    ]


def _what_to_look_at(report_metadata: dict[str, Any], status: str) -> str:
    kind = _report_kind(report_metadata)
    analysis_name = clean_string(report_metadata.get("analysis_name"))
    if kind == "replay":
        metadata = report_metadata.get("metadata") or {}
        if metadata.get("history_mode"):
            return "Read the latest cases first. This page summarizes local historical replay cases for one strategy, but it is still a sparse learning layer rather than a full backtest. Pay most attention to valuation source quality, expected move versus actual move, and whether the strategy beat long stock for honest reasons."
        return "Start with valuation source and checkpoint quality before reading the outcome. Then compare expected move versus actual move, stock versus strategy, and the driver table to see whether direction, timing, IV, or structure explained the result."
    if kind == "strategy":
        if status == "partial":
            return "Focus first on break-even, max loss, and premium or capital required. Then read the payoff chart to see how spot movement helps or hurts, and remember that time decay and IV changes can still move value before expiry. This page uses fallback inputs, so check the warnings before comparing trades."
        return "Focus first on break-even, max loss, max gain, and premium or capital required. Then use the payoff chart to see how spot movement changes the setup, and remember that before expiry both implied volatility and time decay can move the option value even if spot does not."
    if analysis_name == "term_structure":
        return "Focus on how ATM implied volatility changes across expiries. A steeper curve means the market is pricing different volatility further out in time. If this page is partial, the real limitation is sparse local expiry coverage rather than the chart itself."
    if analysis_name == "skew":
        return "Focus on ATM IV first, then compare downside versus upside wing pricing. Skew tells you whether puts or calls are carrying richer implied volatility, which matters for how expensive protection or upside optionality looks."
    if analysis_name == "realized_vol":
        return "Start with the latest realized-vol windows at the top, then use the history chart to judge whether realized volatility is expanding or compressing. This helps you compare what actually happened in the stock with what options are currently pricing."
    if analysis_name == "iv_vs_realized":
        return "Focus on the spread between ATM implied volatility and realized-vol windows. A large positive gap usually means options are pricing more movement than the stock has recently delivered; a narrow gap can mean options are less richly priced."
    if analysis_name == "event_scenarios":
        return "Focus on the pre-event, through-event, and post-event cases first. The most important assumption on this page is the post-event IV change, because earnings-style setups often lose value from IV crush even when the stock move is directionally right."
    if analysis_name == "trade_review":
        return "Focus on valuation source, total PnL, and the attribution block. Start by asking whether the exit value was observed or modeled, then see how much of the outcome came from stock move, time decay, IV change, and structure residual rather than one headline return figure."
    if analysis_name == "portfolio_summary":
        return "Start with premium at risk, expiry buckets, and the what-if shock table. This is a light portfolio view, so treat it as a practical concentration and timing read rather than a full correlation-aware risk engine."
    if analysis_name == "event_analysis":
        return "Focus on the nearest-event timing and the realized move around the event window before digging into the raw event rows. The main question is whether the stock historically moved enough around the event to justify the implied move now."
    if analysis_name == "expected_vs_realized":
        return "Focus on expected move versus realized move percentage first. This tells you whether the option market has recently been overpricing or underpricing realized movement. Missing rows usually mean the local expected-move history is still sparse."
    if status == "insufficient_data":
        return "This page is limited by missing local inputs. Read the warnings panel first to see which data would unlock a fuller analysis."
    return "Start with the summary metrics, then the plots, then the detailed tables and raw metadata."


def _render_callout(title: str, body: str) -> str:
    if not clean_string(body):
        return ""
    return (
        '<section class="panel"><div class="callout">'
        f'<div class="callout-title">{escape(title)}</div>'
        f"<div>{escape(body)}</div>"
        "</div></section>"
    )


def _render_provenance_and_quality(report_metadata: dict[str, Any], coverage: dict[str, Any], *, published: bool = False) -> str:
    provenance = _render_key_value_rows(_provenance_items(report_metadata), published=published)
    quality = _render_key_value_rows(_coverage_items(report_metadata, coverage), published=published)
    research = _render_key_value_rows(_research_availability(_resolve_research_context(report_metadata)), published=published)
    blocks = []
    if provenance:
        blocks.append(f'<div class="panel">{ "<h2>Data Provenance And Freshness</h2>" + provenance }</div>')
    if quality:
        blocks.append(f'<div class="panel">{ "<h2>Coverage And Quality</h2>" + quality }</div>')
    if research:
        blocks.append(f'<div class="panel">{ "<h2>Research Availability</h2>" + research }</div>')
    if not blocks:
        return ""
    return '<section class="panel-grid">' + "".join(blocks) + "</section>"


def _render_raw_notes(markdown_text: str, *, published: bool = False) -> str:
    text = clean_string(markdown_text)
    if not text:
        return ""
    text = _sanitize_display_text(text, published=published)
    return (
        '<section class="panel"><h2>Raw Notes</h2>'
        '<p class="section-intro">This section preserves the plain Markdown notes written alongside the report.</p>'
        f'<pre class="raw-notes">{escape(text)}</pre>'
        "</section>"
    )


def _scenario_lead_images(image_paths: list[Path]) -> list[Path]:
    preferred = [
        "payoff_comparison.png",
        "estimated_value_comparison.png",
        "stock_vs_strategies_equal_capital.png",
        "compare_vs_stock_matrix.png",
        "stock_vs_strategies_share_equivalent.png",
    ]
    order = {name: index for index, name in enumerate(preferred)}
    matching = [path for path in image_paths if path.name in preferred]
    return sorted(matching, key=lambda path: (order.get(path.name, len(preferred)), path.name))


def _scenario_strategy_preview(strategy_name: str, image_paths: list[Path], *, prefix: str) -> Path | None:
    target = f"{prefix}_{slugify(strategy_name)}.png"
    for path in image_paths:
        if path.name == target:
            return path
    return None


def _scenario_strategy_chart_map(strategy_name: str, image_paths: list[Path]) -> dict[str, Path]:
    strategy_slug = slugify(strategy_name)
    candidates = {
        "payoff": f"strategy_payoff_{strategy_slug}.png",
        "estimated-value": f"strategy_estimated_value_{strategy_slug}.png",
        "spot-time": f"spot_time_{strategy_slug}.png",
        "spot-iv": f"spot_iv_{strategy_slug}.png",
        "vs-stock": f"strategy_vs_stock_{strategy_slug}.png",
        "time-progression": f"strategy_time_progression_{strategy_slug}.png",
    }
    resolved: dict[str, Path] = {}
    for key, filename in candidates.items():
        for path in image_paths:
            if path.name == filename:
                resolved[key] = path
                break
    return resolved


def _scenario_card_preview(strategy_name: str, image_paths: list[Path]) -> Path | None:
    chart_map = _scenario_strategy_chart_map(strategy_name, image_paths)
    for key in ["payoff", "estimated-value", "spot-time", "spot-iv", "vs-stock", "time-progression"]:
        if key in chart_map:
            return chart_map[key]
    return None


def _scenario_strategy_cards(
    strategy_summary: pd.DataFrame,
    image_paths: list[Path],
    *,
    base_dir: Path,
    embed_images: bool,
    published: bool,
) -> str:
    if strategy_summary.empty:
        return '<p class="empty-state">No strategy rows were available.</p>'
    cards: list[str] = []
    display_columns = [
        ("One Unit Cost", "unit_capital_required"),
        ("Normalized Budget", "comparison_capital"),
        ("Affordable Units", "affordable_units"),
        ("Max Loss", "max_loss"),
        ("Max Gain", "max_gain"),
        ("Break-Even", "break_even"),
        ("Bear @ $1k", "bear_comparison_profit_loss"),
        ("Base @ $1k", "base_comparison_profit_loss"),
        ("Bull @ $1k", "bull_comparison_profit_loss"),
        ("Base Return On $1k", "selected_return_on_comparison_capital"),
    ]
    for row in strategy_summary.to_dict(orient="records"):
        strategy_name = clean_string(row.get("strategy"))
        preview = _scenario_card_preview(strategy_name, image_paths)
        deep_dive_id = f"deep-dive-{slugify(strategy_name)}"
        cards.append(
            '<article class="strategy-quick-card">'
            f"<h3>{escape(strategy_name.replace('_', ' ').title())}</h3>"
            f'<div class="strategy-quick-meta">{escape(_format_scalar(row.get("leg_summary"), published=published))}</div>'
            + _scenario_budget_badge(row, published=published)
            + _render_key_value_rows(
                [(label, row.get(column)) for label, column in display_columns],
                published=published,
            )
            + (
                _render_lightbox_figure(
                    src=_image_src(preview, base_dir=base_dir, embed_images=embed_images),
                    caption=f"{strategy_name.replace('_', ' ').title()} preview",
                )
                if preview is not None
                else ""
            )
            + f'<div class="strategy-quick-note">{escape(_scenario_strategy_note(strategy_name))}</div>'
            + (
                f'<div class="strategy-quick-note">{escape(clean_string(row.get("warning_or_note")))}</div>'
                if clean_string(row.get("warning_or_note"))
                else ""
            )
            + (
                '<div class="strategy-card-actions">'
                f'<a class="scenario-link-chip" href="#{escape(deep_dive_id)}">Open deep dive</a>'
                "</div>"
            )
            + "</article>"
        )
    return '<div class="strategy-card-grid">' + "".join(cards) + "</div>"


def _scenario_deep_dive_links(
    strategy_name: str,
    *,
    artifact_dir: Path,
) -> list[tuple[str, str]]:
    title = strategy_name.replace("_", " ").title()
    links = []
    for href, label, _ in _related_report_items({}, artifact_dir=artifact_dir):
        lower_label = clean_string(label).lower()
        if "snapshot hub" in lower_label:
            links.append((href, "Open Snapshot Hub"))
        elif clean_string(strategy_name).replace("_", " ") in lower_label or clean_string(strategy_name).replace("_", "-") in lower_label:
            links.append((href, f"Open {title} detail"))
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for href, label in links:
        key = f"{href}|{label}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((href, label))
    return unique[:2]


def _render_scenario_strategy_deep_dives(
    report_metadata: dict[str, Any],
    strategy_summary: pd.DataFrame,
    image_paths: list[Path],
    *,
    base_dir: Path,
    artifact_dir: Path,
    data_script_id: str,
    forward_quick: pd.DataFrame,
    forward_spot_time: pd.DataFrame,
    forward_spot_iv: pd.DataFrame,
    forward_time_iv: pd.DataFrame,
    embed_images: bool,
    published: bool,
) -> str:
    if strategy_summary.empty:
        return ""
    blocks = []
    for row in strategy_summary.to_dict(orient="records"):
        strategy_name = clean_string(row.get("strategy"))
        title = strategy_name.replace("_", " ").title()
        chart_map = _scenario_strategy_chart_map(strategy_name, image_paths)
        if not chart_map:
            continue
        nav = []
        panels = []
        for key, label in [
            ("payoff", "Payoff"),
            ("estimated-value", "Estimated Value"),
            ("spot-time", "Spot x Time"),
            ("spot-iv", "Spot x IV"),
            ("vs-stock", "Vs Stock"),
            ("time-progression", "Time Progression"),
        ]:
            path = chart_map.get(key)
            if path is None:
                continue
            panel_index = len(nav)
            panel_id = f"{slugify(strategy_name)}-{key}"
            nav.append(
                f'<button type="button" class="chart-switcher-button" data-chart-switch-target="{escape(panel_id)}" aria-selected="false">{escape(label)}</button>'
            )
            panels.append(
                f'<div class="chart-switcher-panel deep-dive-figure" data-chart-switch-panel="{escape(panel_id)}"{" hidden" if panel_index else ""}>'
                + _render_lightbox_figure(
                    src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
                    caption=f"{title} {label}",
                    featured=True,
                )
                + "</div>"
            )
        related_links = _scenario_deep_dive_links(strategy_name, artifact_dir=artifact_dir)
        link_html = (
            '<div class="strategy-card-actions">'
            + "".join(
                f'<a class="scenario-link-chip" href="{escape(href)}">{escape(label)}</a>'
                for href, label in related_links
            )
            + "</div>"
            if related_links
            else ""
        )
        summary_pairs = [
            ("One Unit Cost", row.get("unit_capital_required")),
            ("Affordable Units", row.get("affordable_units")),
            ("Break-Even", row.get("break_even")),
            ("Max Loss", row.get("max_loss")),
            ("Bull @ $1k", row.get("bull_comparison_profit_loss")),
            ("Base Return On $1k", row.get("selected_return_on_comparison_capital")),
        ]
        forward_subsection = (
            '<div class="strategy-deep-dive-copy"><strong>Forward Scenario View</strong> Keep the strategy fixed here and reuse the forward lab controls to inspect spot, time, and IV paths without leaving the page.</div>'
            + _render_forward_lab_component(
                report_metadata,
                data_script_id=data_script_id,
                quick_frame=forward_quick,
                spot_time_frame=forward_spot_time,
                spot_iv_frame=forward_spot_iv,
                time_iv_frame=forward_time_iv,
                published=published,
                fixed_strategy=strategy_name,
                compact=True,
            )
        )
        blocks.append(
            f'<details class="strategy-deep-dive" id="deep-dive-{escape(slugify(strategy_name))}">'
            '<summary>'
            '<div>'
            f'<div class="strategy-deep-dive-title">{escape(title)}</div>'
            f'<div class="strategy-deep-dive-meta">{escape(_format_scalar(row.get("leg_summary"), published=published))}</div>'
            "</div>"
            + _scenario_budget_badge(row, published=published)
            + "</summary>"
            '<div class="strategy-deep-dive-body">'
            '<div class="strategy-deep-dive-grid">'
            '<div class="strategy-deep-dive-sidebar">'
            + _render_key_value_rows(summary_pairs, published=published)
            + f'<div class="strategy-deep-dive-copy">{escape(_scenario_strategy_note(strategy_name))}</div>'
            + (
                f'<div class="strategy-deep-dive-copy">{escape(clean_string(row.get("budget_note")))}</div>'
                if clean_string(row.get("budget_note"))
                else ""
            )
            + (
                f'<div class="strategy-deep-dive-copy">{escape(clean_string(row.get("warning_or_note")))}</div>'
                if clean_string(row.get("warning_or_note"))
                else ""
            )
            + link_html
            + "</div>"
            + (
                '<div data-chart-switcher>'
                + '<div class="chart-switcher-nav">' + "".join(nav) + "</div>"
                + "".join(panels)
                + "</div>"
            )
            + forward_subsection
            + "</div></div></details>"
        )
    if not blocks:
        return ""
    return (
        '<section class="panel"><h2>Strategy Deep Dives</h2>'
        '<p class="section-intro">The overview stays compact above. Open one structure here when you want the fuller payoff, valuation, heatmap, and stock-comparison chart set without leaving the page.</p>'
        '<div class="strategy-deep-dive-group">'
        + "".join(blocks)
        + "</div></section>"
    )


def _scenario_focus_strip(items: list[tuple[str, Any]], *, published: bool = False) -> str:
    chips = []
    for label, value in items:
        if _clean_scalar(value) is None:
            continue
        chips.append(
            '<div class="scenario-chip">'
            f"<strong>{escape(label)}</strong>"
            f"<span>{escape(_format_scalar(value, published=published))}</span>"
            "</div>"
        )
    if not chips:
        return ""
    return '<div class="scenario-control-row">' + "".join(chips) + "</div>"


def _scenario_assumptions_table(report_metadata: dict[str, Any], *, published: bool = False) -> str:
    metadata = report_metadata.get("metadata") or {}
    scenario_defaults = report_metadata.get("scenario_defaults") or {}
    payload = _scenario_payload(report_metadata)
    items = [
        ("Spot Price", payload.get("spot_price") or metadata.get("spot_price")),
        ("Spot Source", metadata.get("spot_source")),
        ("Spot Matched Date", metadata.get("spot_matched_date")),
        ("Spot Resolution Note", metadata.get("spot_note")),
        ("Risk-Free Rate", payload.get("risk_free_rate") or metadata.get("risk_free_rate")),
        ("Risk-Free Source", metadata.get("risk_free_source")),
        ("Risk-Free Series", metadata.get("risk_free_series")),
        ("Risk-Free Matched Date", metadata.get("risk_free_matched_date")),
        ("Dividend Yield", payload.get("dividend_yield") or metadata.get("dividend_yield")),
        ("Premium Mode", payload.get("premium_mode") or metadata.get("premium_mode")),
        ("Comparison Capital", payload.get("comparison_capital") or report_metadata.get("comparison_capital") or scenario_defaults.get("comparison_capital")),
        ("Capital Sizing Mode", payload.get("capital_sizing_mode") or report_metadata.get("capital_sizing_mode") or scenario_defaults.get("capital_sizing_mode")),
        ("Snapshot File", metadata.get("source_snapshot_file")),
        ("Representative Horizon", ((payload.get("representative_horizon") or {}).get("label"))),
        ("Representative IV Case", ((payload.get("representative_iv_case") or {}).get("label"))),
        ("Shareability Status", report_metadata.get("shareability_status")),
    ]
    table = _render_key_value_rows(items, published=published)
    if not table:
        return ""
    return (
        '<section class="panel"><h2>Assumptions And Provenance</h2>'
        '<p class="section-intro">These are the resolved local inputs and defaults behind the scenario comparisons above.</p>'
        + table
        + "</section>"
    )


def _scenario_omissions(omitted: list[dict[str, Any]], *, published: bool = False) -> str:
    if not omitted:
        return ""
    frame = pd.DataFrame(omitted)
    return _render_dataframe("Omitted Strategies", frame, table_id="omitted-strategies", published=published)


def _scenario_image_section(
    title: str,
    image_paths: list[Path],
    *,
    base_dir: Path,
    embed_images: bool,
) -> str:
    if not image_paths:
        return ""
    figures = [
        _render_lightbox_figure(
            src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
            caption=_slug_title(path.stem),
            featured=index == 0,
        )
        for index, path in enumerate(image_paths)
    ]
    return (
        f'<section class="panel"><h2>{escape(title)}</h2>'
        '<div class="lead-chart-grid">'
        + "".join(figures)
        + "</div></section>"
    )


def _scenario_image_by_name(image_paths: list[Path], filename: str) -> Path | None:
    for path in image_paths:
        if path.name == filename:
            return path
    return None


def _scenario_compact_case_frame(frame: pd.DataFrame, *, comparison_capital: Any) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    sort_columns = [column for column in ["equal_capital_profit_loss", "profit_loss", "stock_relative_difference"] if column in working.columns]
    if sort_columns:
        working[sort_columns[0]] = pd.to_numeric(working[sort_columns[0]], errors="coerce")
        working = working.sort_values(by=sort_columns[0], ascending=False, na_position="last", kind="mergesort")
    table = pd.DataFrame()
    if "strategy" in working.columns:
        table["Strategy"] = working["strategy"].map(_scenario_strategy_title)
    if "spot_case" in working.columns:
        table["Spot Case"] = working["spot_case"].map(lambda value: clean_string(value).replace("_", " ").title())
    if "horizon" in working.columns:
        table["Horizon"] = working["horizon"]
    if "iv_case" in working.columns:
        table["IV Case"] = working["iv_case"].map(lambda value: clean_string(value).replace("_", " ").title())
    if "spot_price" in working.columns:
        table["Spot Price"] = working["spot_price"]
    if "estimated_value" in working.columns:
        table["Estimated Value"] = working["estimated_value"]
    if "profit_loss" in working.columns:
        table["PnL"] = working["profit_loss"]
    if "equal_capital_profit_loss" in working.columns:
        table[f"PnL @ ${float(comparison_capital):,.0f}"] = working["equal_capital_profit_loss"]
    if "return_on_comparison_capital" in working.columns:
        table[f"Return On ${float(comparison_capital):,.0f}"] = working["return_on_comparison_capital"].map(_format_percent)
    if "stock_relative_difference" in working.columns:
        table["Vs Long Stock"] = working["stock_relative_difference"]
    if "unit_capital_required" in working.columns:
        table["Unit Capital Required"] = working["unit_capital_required"]
    if "affordable_units" in working.columns:
        table["Affordable Units"] = working["affordable_units"]
    if "fully_implementable_with_budget" in working.columns:
        table["Fits Budget"] = working["fully_implementable_with_budget"]
    if "valuation_date" in working.columns:
        table["Valuation Date"] = working["valuation_date"]
    if "clamped_to_expiry" in working.columns:
        table["Clamped To Expiry"] = working["clamped_to_expiry"]
    return table


def _forward_lab_record_columns() -> list[str]:
    return [
        "strategy",
        "spot_case",
        "horizon",
        "iv_case",
        "spot_price",
        "valuation_date",
        "requested_days",
        "effective_days",
        "clamped_to_expiry",
        "estimated_value",
        "profit_loss",
        "comparison_estimated_value",
        "comparison_profit_loss",
        "return_on_comparison_capital",
        "comparison_capital",
        "unit_capital_required",
        "affordable_units",
        "fully_implementable_with_budget",
        "budget_note",
        "stock_estimated_value",
        "stock_profit_loss",
        "stock_return_on_comparison_capital",
        "stock_relative_difference",
    ]


def _forward_lab_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    keep_columns = [column for column in _forward_lab_record_columns() if column in frame.columns]
    return make_json_safe(frame[keep_columns].to_dict(orient="records"))


def _forward_lab_filtered_frame(
    frame: pd.DataFrame,
    *,
    strategy_name: str,
    mode: str,
    fixed_iv_case: str,
    fixed_horizon: str,
    fixed_spot_case: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.loc[frame.get("strategy") == strategy_name].copy()
    if mode == "spot_time":
        return working.loc[working.get("iv_case") == fixed_iv_case].copy()
    if mode == "spot_iv":
        return working.loc[working.get("horizon") == fixed_horizon].copy()
    if mode == "time_iv":
        return working.loc[working.get("spot_case") == fixed_spot_case].copy()
    return working


def _forward_lab_default_compare_summary(
    quick_frame: pd.DataFrame,
    *,
    comparison_capital: float,
    fixed_spot_case: str,
    published: bool,
) -> str:
    if quick_frame.empty:
        return '<div class="forward-lab-empty">No representative compare-vs-stock rows were available.</div>'
    selected = quick_frame.loc[quick_frame.get("spot_case") == fixed_spot_case].copy()
    if selected.empty:
        selected = quick_frame.loc[quick_frame.get("spot_case") == "flat"].copy()
    if selected.empty:
        selected = quick_frame.head(1).copy()
    row = selected.iloc[0].to_dict()
    cards = [
        (
            "Representative Case",
            [
                ("Spot Case", row.get("spot_case")),
                ("Horizon", row.get("horizon")),
                ("IV Case", row.get("iv_case")),
                ("Modeled Value", row.get("estimated_value")),
                ("PnL", row.get("profit_loss")),
                (f"Return On ${float(comparison_capital):,.0f}", _format_percent(row.get("return_on_comparison_capital"))),
            ],
        ),
        (
            "Vs Long Stock",
            [
                (f"Strategy PnL @ ${float(comparison_capital):,.0f}", row.get("comparison_profit_loss")),
                ("Long Stock PnL", row.get("stock_profit_loss")),
                ("Difference Vs Stock", row.get("stock_relative_difference")),
                ("Affordable Units", row.get("affordable_units")),
                ("Fits Budget", row.get("fully_implementable_with_budget")),
            ],
        ),
    ]
    return (
        '<div class="forward-lab-summary-grid">'
        + "".join(
            '<article class="forward-lab-summary-card"><h4>'
            + escape(title)
            + "</h4>"
            + _render_key_value_rows(pairs, published=published)
            + "</article>"
            for title, pairs in cards
        )
        + "</div>"
    )


def _forward_lab_json_payload(
    report_metadata: dict[str, Any],
    *,
    quick_frame: pd.DataFrame,
    spot_time_frame: pd.DataFrame,
    spot_iv_frame: pd.DataFrame,
    time_iv_frame: pd.DataFrame,
) -> str:
    payload = _scenario_payload(report_metadata)
    scenario_defaults = payload.get("scenario_defaults") or report_metadata.get("scenario_defaults") or {}
    forward_defaults = payload.get("forward_defaults") or report_metadata.get("forward_defaults") or {}
    strategies = payload.get("available_strategies") or []
    horizon_desc = scenario_defaults.get("spot_time_display_order") or ["entry", "1w", "1m", "3m", "6m", "expiry"]
    horizon_asc = list(horizon_desc)
    iv_order = scenario_defaults.get("spot_iv_display_order") or ["iv_down", "iv_unchanged", "iv_up"]
    json_payload = {
        "defaults": {
            "strategy": clean_string(forward_defaults.get("strategy")) or (clean_string(strategies[0]) if strategies else ""),
            "metric": clean_string(forward_defaults.get("metric")) or "profit_loss",
            "mode": clean_string(forward_defaults.get("mode")) or "spot_time",
            "fixed_iv_case": clean_string(forward_defaults.get("fixed_iv_case")) or "iv_unchanged",
            "fixed_horizon": clean_string(forward_defaults.get("fixed_horizon")) or "expiry",
            "fixed_spot_case": clean_string(forward_defaults.get("fixed_spot_case")) or "flat",
            "comparison_capital_label": clean_string(forward_defaults.get("comparison_capital_label")) or "$1,000",
        },
        "strategies": [
            {"value": clean_string(strategy_name), "label": _scenario_strategy_title(strategy_name)}
            for strategy_name in strategies
            if clean_string(strategy_name)
        ],
        "metrics": [
            {"value": "estimated_value", "label": "Modeled Value"},
            {"value": "profit_loss", "label": "PnL $"},
            {"value": "return_on_comparison_capital", "label": "PnL %"},
            {"value": "stock_relative_difference", "label": "Outperformance vs Long Stock"},
        ],
        "modes": [
            {"value": "spot_time", "label": "Spot x Time"},
            {"value": "spot_iv", "label": "Spot x IV"},
            {"value": "time_iv", "label": "Time x IV"},
        ],
        "ivCases": [
            {"value": clean_string(value), "label": clean_string(value).replace("_", " ").title()}
            for value in scenario_defaults.get("spot_iv_display_order") or ["iv_down", "iv_unchanged", "iv_up"]
        ],
        "horizons": [
            {"value": clean_string(item.get("label")), "label": clean_string(item.get("label")).replace("_", " ").title()}
            for item in scenario_defaults.get("horizons") or []
            if clean_string(item.get("label"))
        ],
        "spotCases": [
            {
                "value": clean_string(label),
                "label": clean_string(label).replace("_", " ").title(),
                "spot_price": case_payload.get("spot_price"),
            }
            for label, case_payload in (scenario_defaults.get("spot_cases") or {}).items()
            if clean_string(label)
        ],
        "orders": {
            "horizons_x": horizon_asc,
            "horizons_y": horizon_desc,
            "iv_cases_y": iv_order,
        },
        "data": {
            "quick": _forward_lab_records(quick_frame),
            "spot_time": _forward_lab_records(spot_time_frame),
            "spot_iv": _forward_lab_records(spot_iv_frame),
            "time_iv": _forward_lab_records(time_iv_frame),
        },
    }
    json_text = json.dumps(make_json_safe(json_payload), separators=(",", ":"))
    return (
        json_text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _render_forward_lab_component(
    report_metadata: dict[str, Any],
    *,
    data_script_id: str,
    quick_frame: pd.DataFrame,
    spot_time_frame: pd.DataFrame,
    spot_iv_frame: pd.DataFrame,
    time_iv_frame: pd.DataFrame,
    published: bool,
    fixed_strategy: str | None = None,
    compact: bool = False,
) -> str:
    payload = _scenario_payload(report_metadata)
    defaults = payload.get("forward_defaults") or report_metadata.get("forward_defaults") or {}
    strategies = payload.get("available_strategies") or []
    default_strategy = clean_string(fixed_strategy) or clean_string(defaults.get("strategy")) or (clean_string(strategies[0]) if strategies else "")
    comparison_capital = float(payload.get("comparison_capital") or report_metadata.get("comparison_capital") or 1000.0)
    metric_default = clean_string(defaults.get("metric")) or "profit_loss"
    mode_default = clean_string(defaults.get("mode")) or "spot_time"
    fixed_iv = clean_string(defaults.get("fixed_iv_case")) or "iv_unchanged"
    fixed_horizon = clean_string(defaults.get("fixed_horizon")) or "expiry"
    fixed_spot_case = clean_string(defaults.get("fixed_spot_case")) or "flat"
    component_key = slugify(clean_string(fixed_strategy) or default_strategy or f"forward-{data_script_id}")
    strategy_select_id = f"forward-strategy-select-{component_key}"
    metric_select_id = f"forward-metric-select-{component_key}"
    fixed_iv_select_id = f"forward-fixed-iv-{component_key}"
    fixed_horizon_select_id = f"forward-fixed-horizon-{component_key}"
    fixed_spot_select_id = f"forward-fixed-spot-{component_key}"
    default_detail = _forward_lab_filtered_frame(
        {
            "spot_time": spot_time_frame,
            "spot_iv": spot_iv_frame,
            "time_iv": time_iv_frame,
        }.get(mode_default, spot_time_frame),
        strategy_name=default_strategy,
        mode=mode_default,
        fixed_iv_case=fixed_iv,
        fixed_horizon=fixed_horizon,
        fixed_spot_case=fixed_spot_case,
    )
    default_detail_table = default_detail.loc[
        :,
        [
            column
            for column in ["spot_case", "horizon", "iv_case", "spot_price", metric_default]
            if column in default_detail.columns
        ],
    ].copy()
    if not default_detail_table.empty:
        default_detail_table = default_detail_table.rename(
            columns={
                "spot_case": "Spot Case",
                "horizon": "Horizon",
                "iv_case": "IV Case",
                "spot_price": "Spot Price",
                metric_default: {
                    "estimated_value": "Modeled Value",
                    "profit_loss": "PnL $",
                    "return_on_comparison_capital": "PnL %",
                    "stock_relative_difference": "Outperformance vs Long Stock",
                }.get(metric_default, metric_default),
            }
        )
    default_quick = quick_frame.loc[quick_frame.get("strategy") == default_strategy].copy() if not quick_frame.empty else pd.DataFrame()
    quick_table = _scenario_compact_case_frame(default_quick, comparison_capital=comparison_capital)
    strategy_options = "".join(
        f'<option value="{escape(clean_string(strategy_name))}"{" selected" if clean_string(strategy_name) == default_strategy else ""}>{escape(_scenario_strategy_title(strategy_name))}</option>'
        for strategy_name in strategies
        if clean_string(strategy_name)
    )
    # Build selectors from scenario defaults directly to keep the markup visible without JS execution.
    scenario_defaults = payload.get("scenario_defaults") or report_metadata.get("scenario_defaults") or {}
    iv_select_options = "".join(
        f'<option value="{escape(clean_string(value))}"{" selected" if clean_string(value) == fixed_iv else ""}>{escape(clean_string(value).replace("_", " ").title())}</option>'
        for value in scenario_defaults.get("spot_iv_display_order") or ["iv_down", "iv_unchanged", "iv_up"]
    )
    horizon_select_options = "".join(
        f'<option value="{escape(clean_string(item.get("label")))}"{" selected" if clean_string(item.get("label")) == fixed_horizon else ""}>{escape(clean_string(item.get("label")).replace("_", " ").title())}</option>'
        for item in scenario_defaults.get("horizons") or []
        if clean_string(item.get("label"))
    )
    spot_case_options = "".join(
        f'<option value="{escape(clean_string(label))}"{" selected" if clean_string(label) == fixed_spot_case else ""}>{escape(clean_string(label).replace("_", " ").title())}</option>'
        for label in (scenario_defaults.get("spot_cases") or {}).keys()
        if clean_string(label)
    )
    mode_buttons = "".join(
        f'<button type="button" class="forward-lab-button{" is-active" if mode == mode_default else ""}" data-forward-mode="{escape(mode)}" aria-selected="false">{escape(label)}</button>'
        for mode, label in [("spot_time", "Spot x Time"), ("spot_iv", "Spot x IV"), ("time_iv", "Time x IV")]
    )
    metric_options = "".join(
        f'<option value="{escape(value)}"{" selected" if value == metric_default else ""}>{escape(label)}</option>'
        for value, label in [
            ("estimated_value", "Modeled Value"),
            ("profit_loss", "PnL $"),
            ("return_on_comparison_capital", "PnL %"),
            ("stock_relative_difference", "Outperformance vs Long Stock"),
        ]
    )
    root_classes = "forward-lab-shell forward-lab-compact" if compact else "forward-lab-shell"
    return (
        f'<div class="{root_classes}" data-forward-lab-root data-forward-root-id="{escape(component_key)}" data-forward-data-id="{escape(data_script_id)}"'
        + (f' data-forward-fixed-strategy="{escape(default_strategy)}"' if clean_string(fixed_strategy) else "")
        + ">"
        + '<div class="forward-lab-controls">'
        + '<div class="forward-lab-chip-row">'
        + f'<div class="forward-lab-locked-chip"><strong>Comparison Capital</strong><span>{escape(_format_scalar(comparison_capital, published=published))}</span></div>'
        + f'<div class="forward-lab-locked-chip"><strong>Forward Lens</strong><span>{escape(payload.get("capital_sizing_mode") or "hybrid")}</span></div>'
        + "</div>"
        + '<div class="forward-lab-control-grid">'
        + f'<div class="forward-lab-control"><label for="{escape(strategy_select_id)}">Strategy</label>'
        + f'<select id="{escape(strategy_select_id)}" class="forward-lab-select" data-forward-strategy>{strategy_options}</select></div>'
        + f'<div class="forward-lab-control"><label for="{escape(metric_select_id)}">Metric</label>'
        + f'<select id="{escape(metric_select_id)}" class="forward-lab-select" data-forward-metric>{metric_options}</select></div>'
        + '<div class="forward-lab-control"><div class="forward-lab-control-title">Heatmap Mode</div>'
        + f'<div class="forward-lab-button-row">{mode_buttons}</div></div>'
        + f'<div class="forward-lab-control forward-lab-fixed-control" data-forward-fixed-control="spot_time"><label for="{escape(fixed_iv_select_id)}">Fixed IV Case</label>'
        + f'<select id="{escape(fixed_iv_select_id)}" class="forward-lab-select" data-forward-fixed-iv>{iv_select_options}</select></div>'
        + f'<div class="forward-lab-control forward-lab-fixed-control" data-forward-fixed-control="spot_iv" hidden="hidden"><label for="{escape(fixed_horizon_select_id)}">Fixed Horizon</label>'
        + f'<select id="{escape(fixed_horizon_select_id)}" class="forward-lab-select" data-forward-fixed-horizon>{horizon_select_options}</select></div>'
        + f'<div class="forward-lab-control forward-lab-fixed-control" data-forward-fixed-control="time_iv" hidden="hidden"><label for="{escape(fixed_spot_select_id)}">Fixed Spot Case</label>'
        + f'<select id="{escape(fixed_spot_select_id)}" class="forward-lab-select" data-forward-fixed-spot>{spot_case_options}</select></div>'
        + "</div></div>"
        + '<div class="forward-lab-grid">'
        + '<section class="forward-lab-visual-card"><h3>Bear / Base / Bull Quick Scenario Table</h3><div class="forward-lab-note">Representative horizon, IV unchanged, selected strategy. Use this as the fastest practical decision read before switching heatmap modes.</div>'
        + f'<div data-forward-quick-table>{_build_table_html(quick_table, title="Forward Quick Scenarios", table_id=f"{slugify(default_strategy)}-forward-quick", published=published, table_class="data-table compact-data-table")}</div>'
        + "</section>"
        + '<section class="forward-lab-visual-card"><h3>Forward Scenario Controls</h3><div class="forward-lab-note" data-forward-mode-note>This heatmap moves spot and holding horizon while keeping one IV case fixed.</div></section>'
        + '<div class="forward-lab-detail-grid">'
        + '<section class="forward-lab-visual-card"><h3>Main Heatmap</h3><div data-forward-heatmap class="forward-lab-empty">Interactive heatmap loads from embedded forward scenario data.</div></section>'
        + '<section class="forward-lab-visual-card"><h3>Linked Slice Chart</h3><div data-forward-slice class="forward-lab-empty">Interactive slice chart loads from embedded forward scenario data.</div></section>'
        + "</div>"
        + '<section class="forward-lab-visual-card"><h3>Representative Compare vs Stock</h3><div class="forward-lab-note">Long stock stays visible here so the selected structure is always compared with the baseline under the same report-level capital framing.</div>'
        + f'<div data-forward-compare>{_forward_lab_default_compare_summary(default_quick, comparison_capital=comparison_capital, fixed_spot_case=fixed_spot_case, published=published)}</div></section>'
        + '<section class="forward-lab-visual-card"><h3>Current Slice Detail Table</h3><div class="forward-lab-note">This lower detail table follows the active mode and fixed-axis selection so you can inspect the exact rows behind the heatmap.</div>'
        + f'<div data-forward-detail-table>{_build_table_html(default_detail_table, title="Forward Slice Detail", table_id=f"{slugify(default_strategy)}-forward-detail", published=published, table_class="data-table compact-data-table")}</div></section>'
        + "</div></div>"
    )


def _scenario_focus_heatmap_panel(
    report_metadata: dict[str, Any],
    image_paths: list[Path],
    *,
    base_dir: Path,
    embed_images: bool,
) -> str:
    payload = _scenario_payload(report_metadata)
    available = payload.get("available_strategies") or []
    focus_strategy = clean_string(payload.get("featured_focus_strategy"))
    panels: list[str] = []
    buttons: list[str] = []
    for strategy_name in available:
        path = _scenario_strategy_preview(strategy_name, image_paths, prefix="spot_time")
        if path is None:
            continue
        panel_id = f"focus-{slugify(strategy_name)}"
        buttons.append(
            f'<button type="button" class="chart-switcher-button{" is-active" if clean_string(strategy_name) == focus_strategy else ""}" '
            f'data-chart-switch-target="{escape(panel_id)}" aria-selected="false">{escape(_scenario_strategy_title(strategy_name))}</button>'
        )
        panels.append(
            f'<div class="chart-switcher-panel focus-panel" data-chart-switch-panel="{escape(panel_id)}"'
            + ("" if clean_string(strategy_name) == focus_strategy else ' hidden="hidden"')
            + ">"
            + _render_lightbox_figure(
                src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
                caption=f"{_scenario_strategy_title(strategy_name)} Spot x Time",
                featured=True,
            )
            + f'<div class="strategy-quick-note">Use this heatmap to see how {_scenario_strategy_title(strategy_name)} changes when timing matters as much as direction.</div>'
            + "</div>"
        )
    if not panels:
        return '<div class="scenario-visual-panel"><h3>Featured Spot x Time Heatmap</h3><p class="empty-state">No focus heatmap available.</p></div>'
    return (
        '<div class="scenario-visual-panel">'
        "<h3>Featured Spot x Time Heatmap</h3>"
        f'<div data-chart-switcher data-chart-switch-default="focus-{escape(slugify(focus_strategy))}" class="focus-heatmap-panel">'
        + '<div class="chart-switcher-nav">' + "".join(buttons) + "</div>"
        + "".join(panels)
        + "</div></div>"
    )


def _render_scenario_summary_visuals(
    report_metadata: dict[str, Any],
    image_paths: list[Path],
    *,
    base_dir: Path,
    embed_images: bool,
) -> str:
    payload = _scenario_payload(report_metadata)
    representative = payload.get("representative_horizon") or {}
    representative_label = clean_string(representative.get("label")) or "representative"
    estimated_path = _scenario_image_by_name(image_paths, f"estimated_value_comparison_{slugify(representative_label)}.png")
    if estimated_path is None:
        estimated_path = _scenario_image_by_name(image_paths, "estimated_value_comparison.png")
    visual_specs = [
        ("Payoff Comparison", _scenario_image_by_name(image_paths, "payoff_comparison.png"), "How all included strategies pay off if held to expiry."),
        (
            f"Estimated Value ({representative_label.replace('_', ' ').title()})",
            estimated_path,
            "How the modeled position values compare before expiry at the default holding horizon.",
        ),
        (
            "Compare vs Stock ($1,000 Normalized)",
            _scenario_image_by_name(image_paths, "stock_vs_strategies_equal_capital.png"),
            "The clean first comparison against simply buying the stock under the same starting budget.",
        ),
    ]
    cards: list[str] = []
    for title, path, description in visual_specs:
        if path is None:
            cards.append(
                f'<div class="scenario-visual-panel"><h3>{escape(title)}</h3><p class="empty-state">{escape(description)}</p></div>'
            )
            continue
        cards.append(
            '<div class="scenario-visual-panel">'
            f"<h3>{escape(title)}</h3>"
            + _render_lightbox_figure(
                src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
                caption=title,
                featured=True,
            )
            + f'<div class="strategy-quick-note">{escape(description)}</div>'
            + "</div>"
        )
    cards.append(
        _scenario_focus_heatmap_panel(
            report_metadata,
            image_paths,
            base_dir=base_dir,
            embed_images=embed_images,
        )
    )
    return (
        '<section class="panel"><h2>Main Control-Panel Visuals</h2>'
        '<p class="section-intro">These four visuals are the fastest way to understand payoff shape, before-expiry value, stock-relative trade-offs, and timing risk before you touch the deeper controls.</p>'
        '<div class="scenario-top-grid">'
        + "".join(cards[:4])
        + "</div></section>"
    )


def _render_scenario_compare_vs_stock(
    report_metadata: dict[str, Any],
    named_scenarios: pd.DataFrame,
    stock_relative: pd.DataFrame,
    image_paths: list[Path],
    *,
    base_dir: Path,
    embed_images: bool,
    published: bool,
) -> str:
    payload = _scenario_payload(report_metadata)
    comparison_capital = payload.get("comparison_capital") or report_metadata.get("comparison_capital") or 1000.0
    representative = payload.get("representative_horizon") or {}
    representative_iv = payload.get("representative_iv_case") or {}
    spot_cases = payload.get("scenario_defaults", {}).get("spot_cases") or report_metadata.get("scenario_defaults", {}).get("spot_cases") or {}
    equal_capital = named_scenarios.loc[
        (named_scenarios.get("horizon") == representative.get("label"))
        & (named_scenarios.get("iv_case") == representative_iv.get("label"))
        & (named_scenarios.get("spot_case").isin(["bear", "flat", "bull"]))
    ].copy() if not named_scenarios.empty else pd.DataFrame()
    share_equivalent = stock_relative.loc[
        (stock_relative.get("mode") == "share_equivalent")
        & (stock_relative.get("spot_price").isin([case.get("spot_price") for case in spot_cases.values() if isinstance(case, dict)]))
    ].copy() if not stock_relative.empty else pd.DataFrame()
    if not share_equivalent.empty:
        spot_case_lookup = {
            float(case_payload.get("spot_price")): label
            for label, case_payload in spot_cases.items()
            if isinstance(case_payload, dict) and case_payload.get("spot_price") is not None
        }
        share_equivalent["spot_case"] = share_equivalent["spot_price"].map(lambda value: spot_case_lookup.get(float(value)))
    equal_chart = _scenario_image_by_name(image_paths, "stock_vs_strategies_equal_capital.png")
    secondary_images = [
        path
        for path in [
            _scenario_image_by_name(image_paths, "compare_vs_stock_matrix.png"),
            _scenario_image_by_name(image_paths, "stock_vs_strategies_share_equivalent.png"),
        ]
        if path is not None
    ]
    equal_table = _scenario_compact_case_frame(equal_capital, comparison_capital=comparison_capital)
    share_table = _scenario_compact_case_frame(share_equivalent, comparison_capital=comparison_capital)
    return (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="compare-vs-stock" id="compare-vs-stock">'
        '<h2>Compare vs Stock</h2>'
        '<p class="section-intro">Long stock remains the benchmark. Start with the normalized equal-capital read, then use share-equivalent as the reality check for one contract versus 100 shares.</p>'
        + _render_callout(
            "What To Look At",
            f'The main question here is not just "which line is highest?" but whether the extra upside or downside protection is worth the cost and funding friction. The first table keeps everything on the same ${float(comparison_capital):,.0f} budget, while the secondary read shows the raw one-unit relationship.',
        )
        + (
            '<div class="scenario-sheet-grid"><section class="panel"><h3>Equal-Capital Chart</h3>'
            + (
                _render_lightbox_figure(
                    src=_image_src(equal_chart, base_dir=base_dir, embed_images=embed_images),
                    caption=f"Compare vs Stock ({float(comparison_capital):,.0f} Normalized)",
                    featured=True,
                )
                if equal_chart is not None
                else '<p class="empty-state">No equal-capital comparison chart available.</p>'
            )
            + "</section>"
            + _render_inline_dataframe(
                f"Equal-Capital Cases (${float(comparison_capital):,.0f} Normalized)",
                equal_table,
                table_id="compare-vs-stock-equal-capital",
                published=published,
            )
            + "</div>"
        )
        + (
            '<div style="height:14px"></div><div class="scenario-sheet-grid">'
            + "".join(
                '<section class="panel"><h3>'
                + escape(_slug_title(path.stem))
                + "</h3>"
                + _render_lightbox_figure(
                    src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
                    caption=_slug_title(path.stem),
                    featured=True,
                )
                + "</section>"
                for path in secondary_images
            )
            + _render_inline_dataframe(
                "Share-Equivalent Cases",
                share_table,
                table_id="compare-vs-stock-share-equivalent",
                published=published,
            )
            + "</div>"
            if secondary_images or not share_table.empty
            else ""
        )
        + "</section>"
    )


def _render_scenario_valuation_explanation(
    report_metadata: dict[str, Any],
    valuation_explanation: pd.DataFrame,
    *,
    published: bool,
) -> str:
    payload = _scenario_payload(report_metadata)
    defaults = payload.get("valuation_defaults") or report_metadata.get("valuation_defaults") or {}
    strategies = payload.get("available_strategies") or []
    if valuation_explanation.empty or not strategies:
        return (
            '<section class="panel scenario-tab-panel" data-scenario-tab-panel="explain-valuation">'
            '<h2>Explain Valuation</h2><p class="empty-state">No valuation explanation rows available.</p></section>'
        )
    buttons: list[str] = []
    panels: list[str] = []
    default_strategy = clean_string(defaults.get("strategy")) or clean_string(strategies[0])
    for strategy_name in strategies:
        strategy_rows = valuation_explanation.loc[
            (valuation_explanation.get("strategy") == strategy_name)
            & (valuation_explanation.get("horizon") == defaults.get("horizon"))
            & (valuation_explanation.get("iv_case") == defaults.get("iv_case"))
        ].copy()
        if strategy_rows.empty:
            strategy_rows = valuation_explanation.loc[valuation_explanation.get("strategy") == strategy_name].copy()
        if strategy_rows.empty:
            continue
        strategy_rows["spot_case_sort"] = strategy_rows["spot_case"].map(
            {"far_bear": 0, "bear": 1, "flat": 2, "bull": 3, "strong_bull": 4}
        ).fillna(99)
        strategy_rows = strategy_rows.sort_values(by=["spot_case_sort", "valuation_date"], kind="mergesort").drop(columns=["spot_case_sort"])
        selected = strategy_rows.loc[strategy_rows["spot_case"] == defaults.get("spot_case")].copy()
        selected_row = (selected.iloc[0] if not selected.empty else strategy_rows.iloc[0]).to_dict()
        strategy_title = _scenario_strategy_title(strategy_name)
        option_extrinsic = _clean_scalar(selected_row.get("option_extrinsic_value"))
        if isinstance(option_extrinsic, (int, float)) and abs(float(option_extrinsic)) > 0.01:
            valuation_copy = (
                f"{strategy_title} still carries extrinsic value at the selected horizon, which means modeled value remains different from pure intrinsic payoff. "
                "That gap is the market still charging for time and uncertainty."
            )
        else:
            valuation_copy = (
                f"{strategy_title} is already close to pure intrinsic behavior at the selected horizon. "
                "When expiry is near or the horizon is clamped, modeled value and expiry payoff converge."
            )
        summary_items = [
            ("Selected Spot Case", selected_row.get("spot_case")),
            ("Selected Horizon", selected_row.get("horizon")),
            ("Selected IV Case", selected_row.get("iv_case")),
            ("Valuation Date", selected_row.get("valuation_date")),
            ("Modeled Value", selected_row.get("modeled_value")),
            ("Profit / Loss Now", selected_row.get("profit_loss_now")),
            ("Payoff At Expiry (Same Spot)", selected_row.get("payoff_at_expiry_same_spot")),
            ("Entry Delta Estimate", selected_row.get("entry_delta_estimate")),
        ]
        breakdown_items = [
            ("Stock Leg Value", selected_row.get("stock_leg_value")),
            ("Option Intrinsic Value", selected_row.get("option_intrinsic_value")),
            ("Option Extrinsic Value", selected_row.get("option_extrinsic_value")),
            ("Clamped To Expiry", selected_row.get("clamped_to_expiry")),
        ]
        table = strategy_rows.loc[
            strategy_rows["spot_case"].isin(["bear", "flat", "bull"]),
            [
                "spot_case",
                "modeled_value",
                "profit_loss_now",
                "payoff_at_expiry_same_spot",
                "option_intrinsic_value",
                "option_extrinsic_value",
                "valuation_date",
            ],
        ].copy()
        table = table.rename(
            columns={
                "spot_case": "Spot Case",
                "modeled_value": "Modeled Value",
                "profit_loss_now": "Profit / Loss Now",
                "payoff_at_expiry_same_spot": "Payoff At Expiry (Same Spot)",
                "option_intrinsic_value": "Option Intrinsic",
                "option_extrinsic_value": "Option Extrinsic",
                "valuation_date": "Valuation Date",
            }
        )
        panel_id = f"valuation-{slugify(strategy_name)}"
        buttons.append(
            f'<button type="button" class="chart-switcher-button{" is-active" if clean_string(strategy_name) == default_strategy else ""}" '
            f'data-chart-switch-target="{escape(panel_id)}" aria-selected="false">{escape(strategy_title)}</button>'
        )
        panels.append(
            f'<div class="chart-switcher-panel valuation-panel" data-chart-switch-panel="{escape(panel_id)}"'
            + ("" if clean_string(strategy_name) == default_strategy else ' hidden="hidden"')
            + ">"
            + f'<h3>{escape(strategy_title)}</h3>'
            + f'<div class="valuation-copy">{escape(valuation_copy)}</div>'
            + '<div class="valuation-grid">'
            + '<article class="valuation-card"><h4>Modeled Value vs Payoff</h4>'
            + _render_key_value_rows(summary_items, published=published)
            + "</article>"
            + '<article class="valuation-card"><h4>Intrinsic vs Extrinsic</h4>'
            + _render_key_value_rows(breakdown_items, published=published)
            + "</article>"
            + "</div>"
            + _render_inline_dataframe(
                "Bear / Flat / Bull Valuation Breakdown",
                table,
                table_id=f"{panel_id}-cases",
                published=published,
            )
            + "</div>"
        )
    return (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="explain-valuation" id="explain-valuation">'
        '<h2>Explain Valuation</h2>'
        '<p class="section-intro">This section explains why modeled value before expiry can differ from simple payoff. Intrinsic value is what the option is worth if exercised immediately, while extrinsic value is what remains because time and volatility still matter. Use it together with the Forward Scenario Lab when you want to see the same logic expressed across spot, time, and IV paths.</p>'
        + _render_callout(
            "What To Look At",
            "If modeled value is well above intrinsic value, the market is still paying for time and uncertainty. As expiry approaches that extrinsic cushion shrinks, so a trade can lose value even when the payoff-at-expiry chart still looks attractive. Then jump to the Forward Scenario Lab to see how that same sensitivity changes across the full scenario grid.",
        )
        + f'<div data-chart-switcher data-chart-switch-default="valuation-{escape(slugify(default_strategy))}" class="valuation-root"><div class="chart-switcher-nav">'
        + "".join(buttons)
        + "</div>"
        + "".join(panels)
        + "</div></section>"
    )


def _render_scenario_replay_case_view(
    report_metadata: dict[str, Any],
    named_scenarios: pd.DataFrame,
    image_paths: list[Path],
    *,
    base_dir: Path,
    embed_images: bool,
    published: bool,
) -> str:
    payload = _scenario_payload(report_metadata)
    defaults = payload.get("replay_defaults") or report_metadata.get("replay_defaults") or {}
    scenario_defaults = payload.get("scenario_defaults") or report_metadata.get("scenario_defaults") or {}
    horizons = [clean_string(item.get("label")) for item in scenario_defaults.get("horizons") or [] if clean_string(item.get("label"))]
    if not horizons:
        return (
            '<section class="panel scenario-tab-panel" data-scenario-tab-panel="replay-case-view">'
            '<h2>Replay / Case View</h2><p class="empty-state">No replay horizons available.</p></section>'
        )
    comparison_capital = payload.get("comparison_capital") or report_metadata.get("comparison_capital") or 1000.0
    ordered_spot_cases = [label for label in ["far_bear", "bear", "flat", "bull", "strong_bull"] if label in (scenario_defaults.get("spot_cases") or {})]
    ordered_iv_cases = [label for label in ["iv_down", "iv_unchanged", "iv_up"] if label in (scenario_defaults.get("iv_cases") or {})]
    chart_panels: list[str] = []
    for horizon in horizons:
        image = _scenario_image_by_name(image_paths, f"estimated_value_comparison_{slugify(horizon)}.png") or _scenario_image_by_name(image_paths, "estimated_value_comparison.png")
        chart_panels.append(
            f'<div class="replay-chart-panel" data-replay-chart-panel="{escape(horizon)}" hidden="hidden">'
            + (
                _render_lightbox_figure(
                    src=_image_src(image, base_dir=base_dir, embed_images=embed_images),
                    caption=f"Estimated Value Comparison ({horizon.replace('_', ' ').title()})",
                    featured=True,
                )
                if image is not None
                else '<p class="empty-state">No chart available for this horizon.</p>'
            )
            + "</div>"
        )
    case_panels: list[str] = []
    for horizon in horizons:
        for spot_case in ordered_spot_cases:
            for iv_case in ordered_iv_cases:
                case_frame = named_scenarios.loc[
                    (named_scenarios.get("horizon") == horizon)
                    & (named_scenarios.get("spot_case") == spot_case)
                    & (named_scenarios.get("iv_case") == iv_case)
                ].copy() if not named_scenarios.empty else pd.DataFrame()
                if not case_frame.empty:
                    peer_frame = named_scenarios.loc[
                        (named_scenarios.get("horizon") == horizon)
                        & (named_scenarios.get("spot_case") == spot_case)
                    ].copy() if not named_scenarios.empty else pd.DataFrame()
                    ranked = case_frame.copy()
                    ranked["equal_capital_profit_loss"] = pd.to_numeric(ranked.get("equal_capital_profit_loss"), errors="coerce")
                    ranked = ranked.sort_values(by="equal_capital_profit_loss", ascending=False, na_position="last", kind="mergesort")
                    best_row = ranked.iloc[0].to_dict()
                    stock_rank = ranked.loc[ranked["strategy"] != "long_stock"].copy()
                    stock_rank["stock_relative_difference"] = pd.to_numeric(stock_rank.get("stock_relative_difference"), errors="coerce")
                    stock_rank = stock_rank.sort_values(by="stock_relative_difference", ascending=False, na_position="last", kind="mergesort")
                    best_vs_stock = stock_rank.iloc[0].to_dict() if not stock_rank.empty else {}
                    valuation_dates = ", ".join(_dedupe_warnings({"warnings": case_frame.get("valuation_date", pd.Series(dtype=str)).astype(str).tolist()}))
                    clamped = bool(case_frame.get("clamped_to_expiry", pd.Series(dtype=bool)).fillna(False).any())
                    shared_coverage = bool(peer_frame.get("valuation_date", pd.Series(dtype=str)).astype(str).nunique() == 1) if not peer_frame.empty else False
                    peer_best = (
                        peer_frame.groupby("iv_case")["equal_capital_profit_loss"].max().dropna()
                        if not peer_frame.empty and "equal_capital_profit_loss" in peer_frame.columns
                        else pd.Series(dtype=float)
                    )
                    compressed_across_iv = bool((peer_best.max() - peer_best.min()) < 10.0) if len(peer_best.index) > 1 else False
                    replay_flags: list[str] = []
                    replay_notes: list[str] = []
                    if clamped:
                        replay_flags.append("clamped to expiry")
                        replay_notes.append("Clamped means the requested holding horizon extended beyond expiry, so later rows reuse the expiry valuation date instead of a true later checkpoint.")
                    if shared_coverage:
                        replay_flags.append("same effective underlying coverage")
                        replay_notes.append("All IV cases for this horizon / spot case share the same effective valuation date, so the underlier path coverage itself is not changing across those toggles.")
                    if compressed_across_iv:
                        replay_flags.append("weak differentiation")
                        replay_notes.append("Current IV-case switching does not materially change the outcome here, so treat the difference as thin rather than decisive.")
                    if not clamped and shared_coverage:
                        replay_notes.append("Modeled continuation is being compared off the same saved checkpoint date rather than new exact later chain points.")
                    summary_items = [
                        ("Best Strategy", _scenario_strategy_title(best_row.get("strategy"))),
                        (f"Best PnL @ ${float(comparison_capital):,.0f}", best_row.get("equal_capital_profit_loss")),
                        ("Best Vs Stock", _scenario_strategy_title(best_vs_stock.get("strategy")) if best_vs_stock else None),
                        ("Vs Stock Diff", best_vs_stock.get("stock_relative_difference") if best_vs_stock else None),
                        ("Valuation Date", valuation_dates),
                        ("Any Clamped Rows", clamped),
                    ]
                    copy = (
                        f"{_scenario_strategy_title(best_row.get('strategy'))} leads this {spot_case.replace('_', ' ')} / {iv_case.replace('_', ' ')} case under the ${float(comparison_capital):,.0f} normalized view. "
                        "Use the table to see whether the winner still fits the budget and whether the edge versus long stock is meaningful."
                    )
                    if replay_notes:
                        copy += " " + " ".join(replay_notes[:2])
                    display_frame = _scenario_compact_case_frame(case_frame, comparison_capital=comparison_capital)
                else:
                    summary_items = [("Status", "No rows available")]
                    copy = "No rows were available for this case. Sparse local option coverage or omitted strategies likely limited the replay view here."
                    display_frame = pd.DataFrame()
                    replay_flags = ["partial coverage"]
                    replay_notes = ["No additional exact checkpoint data was available for this case."]
                case_panels.append(
                    f'<div class="replay-panel" data-replay-case-panel data-replay-case-horizon="{escape(horizon)}" '
                    f'data-replay-case-spot="{escape(spot_case)}" data-replay-case-iv="{escape(iv_case)}" '
                    f'data-replay-flags="{escape("; ".join(replay_flags))}" '
                    f'data-replay-status-note="{escape(" ".join(replay_notes))}" hidden="hidden">'
                    '<div class="replay-state-grid">'
                    '<article class="replay-state-card">'
                    f'<h3 class="replay-summary-title">{escape(spot_case.replace("_", " ").title())} / {escape(iv_case.replace("_", " ").title())} / {escape(horizon.replace("_", " ").title())}</h3>'
                    f'<div class="replay-summary-copy">{escape(copy)}</div>'
                    + ('<div class="status-chip-row">' + "".join(
                        f'<span class="{"status-chip is-warning" if "clamped" in clean_string(flag) or "partial" in clean_string(flag) or "weak" in clean_string(flag) else "status-chip is-muted"}">{escape(flag)}</span>'
                        for flag in replay_flags
                    ) + "</div>" if replay_flags else "")
                    + _render_key_value_rows(summary_items, published=published)
                    + "</article>"
                    + _render_inline_dataframe(
                        "Selected Case Table",
                        display_frame,
                        table_id=f"replay-{slugify(horizon)}-{slugify(spot_case)}-{slugify(iv_case)}",
                        published=published,
                    )
                    + "</div></div>"
                )
    return (
        f'<section class="panel scenario-tab-panel" data-scenario-tab-panel="replay-case-view" id="replay-case-view">'
        '<h2>Replay / Case View</h2>'
        '<p class="section-intro">Use this control panel to replay one named spot case, one IV case, and one holding horizon at a time. Nothing is re-priced in the browser; the page is only switching between pre-rendered scenario states.</p>'
        + _render_callout(
            "What To Look At",
            "Change one dimension at a time. Move the horizon slider first to see how modeled values evolve, then switch spot and IV cases to test whether the same strategy still wins when the path changes. Clamped means the requested horizon ran past expiry, so the replay is reusing the expiry valuation point instead of a truly later option checkpoint.",
        )
        + _render_callout(
            "Plain-Language Notes",
            "Clamped means the requested horizon ran beyond expiry. Same effective underlying coverage means different spot or IV toggles are still reusing the same saved underlier checkpoint. Modeled continuation reused means the page is extending off the same checkpoint rather than showing a new exact later chain. Treat Replay as a learning tool, not a certainty engine.",
        )
        + f'<div class="replay-root" data-replay-root data-default-spot-case="{escape(clean_string(defaults.get("spot_case")))}" '
        + f'data-default-iv-case="{escape(clean_string(defaults.get("iv_case")))}" data-default-horizon="{escape(clean_string(defaults.get("horizon")))}" '
        + f"data-replay-horizons='{escape(json.dumps(horizons))}'>"
        + '<div class="replay-controls">'
        + '<div class="replay-control-group"><div class="replay-label">Spot Case</div><div class="replay-button-row">'
        + "".join(
            f'<button type="button" class="scenario-link-chip replay-chip" data-replay-spot-case="{escape(label)}" aria-selected="false">{escape(label.replace("_", " ").title())}</button>'
            for label in ordered_spot_cases
        )
        + "</div></div>"
        + '<div class="replay-control-group"><div class="replay-label">IV Case</div><div class="replay-button-row">'
        + "".join(
            f'<button type="button" class="scenario-link-chip replay-chip" data-replay-iv-case="{escape(label)}" aria-selected="false">{escape(label.replace("_", " ").title())}</button>'
            for label in ordered_iv_cases
        )
        + "</div></div>"
        + '<div class="replay-control-group replay-slider-row"><div class="replay-label">Holding Horizon</div>'
        + '<input type="range" data-replay-horizon-slider step="1">'
        + '<div class="strategy-quick-note">Selected horizon: <strong data-replay-horizon-label></strong></div>'
        + "</div></div>"
        + '<div class="strategy-quick-note" data-replay-state-summary></div>'
        + '<div class="strategy-quick-note" data-replay-status-note></div>'
        + '<div class="replay-root">'
        + "".join(chart_panels)
        + "".join(case_panels)
        + "</div></div></section>"
    )

def _render_scenario_body(
    report_dir: Path,
    artifact_dir: Path,
    destination: Path,
    report_metadata: dict[str, Any],
    summary_df: pd.DataFrame | None,
    *,
    embed_images: bool,
    published: bool,
) -> tuple[str, str, str]:
    payload = _scenario_payload(report_metadata)
    summary_df = summary_df if summary_df is not None else pd.DataFrame()
    strategy_summary = _load_csv(artifact_dir / "strategy_summary.csv")
    named_scenarios = _load_csv(artifact_dir / "named_scenarios.csv")
    stock_relative = _load_csv(artifact_dir / "stock_relative.csv")
    spot_time_grid = _load_csv(artifact_dir / "spot_time_grid.csv")
    spot_iv_grid = _load_csv(artifact_dir / "spot_iv_grid.csv")
    forward_quick_scenarios = _load_csv(artifact_dir / "forward_quick_scenarios.csv")
    forward_spot_time_grid = _load_csv(artifact_dir / "forward_spot_time_grid.csv")
    forward_spot_iv_grid = _load_csv(artifact_dir / "forward_spot_iv_grid.csv")
    forward_time_iv_grid = _load_csv(artifact_dir / "forward_time_iv_grid.csv")
    valuation_explanation = _load_csv(artifact_dir / "valuation_explanation.csv")
    contract_runs = _contract_selection_runs_for_snapshot(report_dir, report_metadata)
    latest_contract_run_dir, latest_contract_metadata = contract_runs[0] if contract_runs else (None, {})
    explorer_candidate_summary = _load_csv(latest_contract_run_dir / "candidate_summary.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_required_path = _load_csv(latest_contract_run_dir / "required_path_rows.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_path_summary = _load_csv(latest_contract_run_dir / "path_case_summary.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_path_case_chart = _load_csv(latest_contract_run_dir / "path_case_chart_rows.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_path_case_strategy = _load_csv(latest_contract_run_dir / "path_case_strategy_rows.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_selector_rows = _load_csv(latest_contract_run_dir / "strategy_selector_rows.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_selector_rankings = _load_csv(latest_contract_run_dir / "strategy_selector_rankings.csv") if latest_contract_run_dir else pd.DataFrame()
    explorer_images = _discover_images(latest_contract_run_dir, latest_contract_metadata) if latest_contract_run_dir else []
    status = _report_status(report_metadata)
    warnings = _dedupe_warnings(report_metadata)
    images = _discover_images_with_fallback(
        artifact_dir,
        report_dir if artifact_dir != report_dir else None,
        primary_metadata=report_metadata,
    )
    base_dir = destination.parent if published else artifact_dir
    related_items = _related_report_items(report_metadata, artifact_dir=artifact_dir)
    shareability_note = _render_shareability_note(
        published=published,
        embed_images=embed_images,
        has_supporting_links=bool(
            related_items or [path for path in _artifact_files(artifact_dir, report_dir, report_metadata=report_metadata) if path.name not in HIDDEN_FILES]
        ),
    )
    top_strip = _render_summary_strip(_scenario_context_items(report_metadata)[:9], published=published)
    representative = payload.get("representative_horizon") or {}
    representative_iv = payload.get("representative_iv_case") or {}
    valuation_defaults = payload.get("valuation_defaults") or report_metadata.get("valuation_defaults") or {}
    comparison_capital = payload.get("comparison_capital") or report_metadata.get("comparison_capital")
    comparison_capital_value = (
        float(comparison_capital)
        if isinstance(_clean_scalar(comparison_capital), (int, float))
        else 1000.0
    )
    forward_data_script_id = f"forward-scenario-data-{slugify(clean_string(payload.get('ticker')) or report_dir.name)}"
    selected_cases = named_scenarios.copy()
    if not selected_cases.empty:
        selected_cases = selected_cases.loc[
            (selected_cases["horizon"] == representative.get("label"))
            & (selected_cases["iv_case"] == representative_iv.get("label"))
            & (selected_cases["spot_case"].isin(["bear", "flat", "bull"]))
        ].copy()
    selected_cases_display = _scenario_compact_case_frame(
        selected_cases,
        comparison_capital=comparison_capital_value,
    ) if not selected_cases.empty else pd.DataFrame()

    hero = (
        '<section class="hero">'
        '<div class="hero-top">'
        '<div>'
        '<div class="eyebrow">Primary Scenario Dashboard</div>'
        f"<h1>{escape(_page_title(report_metadata))}</h1>"
        '<p class="subtitle">This is the main options control panel for one ticker, snapshot, and expiry. Use it as the primary decision page to compare stock and multiple option structures across payoff, time, IV, and a normalized $1,000 budget lens using saved local artifacts only.</p>'
        "</div>"
        + _status_badge(status)
        + "</div></section>"
    )
    tabs = (
        '<section class="panel"><div class="scenario-tab-nav">'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="summary" aria-selected="true">Summary</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="compare-vs-stock" aria-selected="false">Compare vs Stock</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="forward-scenario-lab" aria-selected="false">Forward Scenario Lab</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="strategy-selector" aria-selected="false">Strategy Selector</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="path-case-summary" aria-selected="false">Path Case Summary</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="path-contract-explorer" aria-selected="false">Path &amp; Contract Explorer</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="strategy-deep-dives" aria-selected="false">Strategy Deep Dives</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="explain-valuation" aria-selected="false">Explain Valuation</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="replay-case-view" aria-selected="false">Replay / Case View</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="assumptions" aria-selected="false">Assumptions</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="raw-details" aria-selected="false">Raw Details</button>'
        "</div></section>"
    )
    decision_strip = _scenario_focus_strip(
        [
            ("Spot", payload.get("spot_price")),
            ("Rate", payload.get("risk_free_rate")),
            ("Comparison Capital", comparison_capital_value),
            ("Representative Horizon", representative.get("label")),
            ("Representative IV", representative_iv.get("label")),
            ("Featured Focus", payload.get("featured_focus_strategy")),
            ("Strategies", len(payload.get("available_strategies") or [])),
        ],
        published=published,
    )

    summary_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="summary" id="summary">'
        '<h2>Summary</h2>'
        f'<p class="section-intro">{escape(_scenario_how_to_read())}</p>'
        + decision_strip
        + _scenario_summary_blocks(report_metadata, strategy_summary, summary_df, published=published)
        + _render_scenario_summary_visuals(
            report_metadata,
            images,
            base_dir=base_dir,
            embed_images=embed_images,
        )
        + _render_callout(
            "What Matters Most Here?",
            clean_string(payload.get("what_matters_most"))
            or clean_string(report_metadata.get("what_matters_most"))
            or _scenario_what_to_look_at(report_metadata, status),
        )
        + _render_scenario_decision_hints(report_metadata, published=published)
        + (
            _render_inline_dataframe(
                "Bull / Base / Bear Decision Table",
                selected_cases_display,
                table_id="scenario-decision-table",
                published=published,
            )
            if not selected_cases_display.empty
            else ""
        )
        + _render_related_reports(report_metadata, artifact_dir=artifact_dir)
        + "</section>"
    )

    strategies_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="strategy-deep-dives" id="strategy-deep-dives">'
        '<h2>Strategy Comparison</h2>'
        '<p class="section-intro">Compare the structures quickly here first, then open one deep dive when you want the full chart set for a single strategy without losing the overview-first workflow.</p>'
        + _render_callout(
            "What To Look At",
            "Break-even and max loss define the shape, but the more practical read is whether the strategy both fits the working budget and still wins under the bull, base, and bear cases that matter for this expiry.",
        )
        + _scenario_strategy_cards(strategy_summary, images, base_dir=base_dir, embed_images=embed_images, published=published)
        + _render_inline_dataframe(
            "Strategy Comparison Table",
            strategy_summary,
            table_id="strategy-summary",
            published=published,
        )
        + _render_scenario_strategy_deep_dives(
            report_metadata,
            strategy_summary,
            images,
            base_dir=base_dir,
            artifact_dir=artifact_dir,
            data_script_id=forward_data_script_id,
            forward_quick=forward_quick_scenarios,
            forward_spot_time=forward_spot_time_grid,
            forward_spot_iv=forward_spot_iv_grid,
            forward_time_iv=forward_time_iv_grid,
            embed_images=embed_images,
            published=published,
        )
        + "</section>"
    )
    forward_data_script = (
        f'<script type="application/json" id="{escape(forward_data_script_id)}">'
        + _forward_lab_json_payload(
            report_metadata,
            quick_frame=forward_quick_scenarios,
            spot_time_frame=forward_spot_time_grid,
            spot_iv_frame=forward_spot_iv_grid,
            time_iv_frame=forward_time_iv_grid,
        )
        + "</script>"
    )
    forward_lab_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="forward-scenario-lab" id="forward-scenario-lab">'
        '<h2>Forward Scenario Lab</h2>'
        '<p class="section-intro">This is the main forward-looking control surface inside the Primary Scenario Dashboard. It keeps the report-level $1,000 comparison-capital framing locked, then lets you inspect how spot, time, and IV interact for one strategy at a time without re-pricing anything in the browser.</p>'
        + _render_callout(
            "What To Look At",
            "Start with the Bear / Base / Bull quick table for the selected strategy. Then choose one heatmap mode, note which variable is fixed, and use the linked slice chart below it to turn the heatmap into a more intuitive line read. Compare the same case against long stock before deciding whether the options structure is actually better for this path.",
        )
        + forward_data_script
        + _render_forward_lab_component(
            report_metadata,
            data_script_id=forward_data_script_id,
            quick_frame=forward_quick_scenarios,
            spot_time_frame=forward_spot_time_grid,
            spot_iv_frame=forward_spot_iv_grid,
            time_iv_frame=forward_time_iv_grid,
            published=published,
        )
        + "</section>"
    )
    explorer_data_script_id = f"path-contract-data-{slugify(clean_string(payload.get('ticker')) or report_dir.name)}"
    explorer_data_script = (
        f'<script type="application/json" id="{escape(explorer_data_script_id)}">'
        + _path_contract_json_payload(
            latest_contract_metadata,
            candidate_summary=explorer_candidate_summary,
            strategy_selector_rows=explorer_selector_rows,
            required_path_rows=explorer_required_path,
        )
        + "</script>"
    ) if latest_contract_run_dir and not explorer_required_path.empty else ""
    strategy_selector_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="strategy-selector" id="strategy-selector">'
        '<h2>Strategy Selector</h2>'
        '<p class="section-intro">Use this tab to decide which strategy family best fits your target price, timing, IV path, budget, and objective before you choose the exact strike or expiry. Every winner here is conditional on those assumptions.</p>'
        + _render_callout(
            "What To Look At",
            "Read the current objective and assumptions first. Then use the best-given-your-assumptions cards to see which family wins under delayed moves, IV compression, capital efficiency, or simpler exposure. Only after that should you open Path & Contract Explorer to choose the exact contract.",
        )
        + (
            _render_strategy_selector_component(
                latest_contract_metadata,
                selector_rows=explorer_selector_rows,
                selector_rankings=explorer_selector_rankings,
                published=published,
            )
            if latest_contract_run_dir and not explorer_selector_rows.empty
            else '<p class="empty-state">No Strategy Selector data was available for this snapshot yet. Run `analyze-contract-selection` for this snapshot, then republish the scenario bundle to embed the latest family-ranking data.</p>'
        )
        + "</section>"
    )
    path_case_data_script_id = f"path-case-data-{slugify(clean_string(payload.get('ticker')) or report_dir.name)}"
    path_case_data_script = (
        f'<script type="application/json" id="{escape(path_case_data_script_id)}">'
        + _path_case_summary_json_payload(
            latest_contract_metadata,
            chart_rows=explorer_path_case_chart,
            strategy_rows=explorer_path_case_strategy,
        )
        + "</script>"
    ) if latest_contract_run_dir and not explorer_path_case_chart.empty and not explorer_path_case_strategy.empty else ""
    path_case_summary_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="path-case-summary" id="path-case-summary">'
        '<h2>Path Case Summary</h2>'
        '<p class="section-intro">This tab bridges family choice and exact contract choice. Use Strategy Selector to choose the family. Use Path Case Summary to see whether your thesis path actually clears that family&#39;s required path. Then use Path &amp; Contract Explorer to choose the exact contract once the family/path fit is clear.</p>'
        + (
            path_case_data_script
            + _render_path_case_summary_component(
                latest_contract_metadata,
                data_script_id=path_case_data_script_id,
                chart_rows=explorer_path_case_chart,
                strategy_rows=explorer_path_case_strategy,
                published=published,
            )
            if latest_contract_run_dir and not explorer_path_case_chart.empty and not explorer_path_case_strategy.empty
            else '<p class="empty-state">No Path Case Summary data was available for this snapshot yet. Run `analyze-contract-selection` for this snapshot, then republish the scenario bundle to embed the latest path-case layer.</p>'
        )
        + "</section>"
    )
    path_explorer_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="path-contract-explorer" id="path-contract-explorer">'
        '<h2>Path &amp; Contract Explorer</h2>'
        '<p class="section-intro">Use this tab when the question is not just what one chosen structure does, but which strike, expiry, or simple structure is best given your specific stock path, timing, IV path, and $1,000 budget assumptions. The Required Stock Path Chart and bundle-written path tables are the decision center.</p>'
        + _render_callout(
            "What To Look At",
            "Start with the ranked cards, then use the Required Stock Path Chart to see how much and how fast the stock must move for one candidate to become worth it. Use the path-case and same-path comparison tables to cross-check expiry choice, IV change, or delayed timing.",
        )
        + (
            explorer_data_script
            + _render_path_contract_explorer_component(
                latest_contract_metadata,
                data_script_id=explorer_data_script_id,
                candidate_summary=explorer_candidate_summary,
                path_case_summary=explorer_path_summary,
                calibration_context=((latest_contract_metadata.get("metadata") or {}).get("calibration_context") or {}),
                published=published,
            )
            if latest_contract_run_dir and not explorer_candidate_summary.empty
            else '<p class="empty-state">No contract-selection analysis bundle was available for this snapshot yet. Run `analyze-contract-selection`, then republish the scenario bundle to embed the latest Path &amp; Contract Explorer data.</p>'
        )
        + (
            '<div class="focus-strip"><strong>Deep-Dive Links</strong> Use the strategy deep dives below for current-expiry structures, then the supporting detail pages for the full drill-down.</div>'
            '<p><a class="scenario-link-chip" href="#strategy-deep-dives">Open Strategy Deep Dives</a> <a class="scenario-link-chip" href="#compare-vs-stock">Compare Vs Stock</a></p>'
            if latest_contract_run_dir
            else ""
        )
        + "</section>"
    )

    compare_section = _render_scenario_compare_vs_stock(
        report_metadata,
        named_scenarios,
        stock_relative,
        images,
        base_dir=base_dir,
        embed_images=embed_images,
        published=published,
    )
    valuation_section = _render_scenario_valuation_explanation(
        report_metadata,
        valuation_explanation,
        published=published,
    )
    replay_section = _render_scenario_replay_case_view(
        report_metadata,
        named_scenarios,
        images,
        base_dir=base_dir,
        embed_images=embed_images,
        published=published,
    )

    assumptions_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="assumptions" id="assumptions">'
        '<h2>Assumptions</h2>'
        '<p class="section-intro">Assumptions, local data provenance, research context, and omitted-structure reasons live here so the decision sections above stay clean and easy to scan.</p>'
        + _scenario_assumptions_table(report_metadata, published=published)
        + _render_research_context(report_metadata, published=published)
        + _scenario_omissions(report_metadata.get("omitted_strategies") or [], published=published)
        + "</section>"
    )

    raw_detail_sections = [
        _render_dataframe("Strategy Summary", strategy_summary, table_id="strategy-summary-raw", published=published),
        _render_dataframe("Named Scenarios", named_scenarios, table_id="named-scenarios", published=published),
        _render_dataframe("Stock Relative", stock_relative, table_id="stock-relative", published=published),
        _render_dataframe("Forward Quick Scenarios", forward_quick_scenarios, table_id="forward-quick-scenarios", published=published),
        _render_dataframe("Forward Spot x Time Grid", forward_spot_time_grid, table_id="forward-spot-time-grid", published=published),
        _render_dataframe("Forward Spot x IV Grid", forward_spot_iv_grid, table_id="forward-spot-iv-grid", published=published),
        _render_dataframe("Forward Time x IV Grid", forward_time_iv_grid, table_id="forward-time-iv-grid", published=published),
        _render_dataframe("Spot x Time Grid", spot_time_grid, table_id="spot-time-grid", published=published),
        _render_dataframe("Spot x IV Grid", spot_iv_grid, table_id="spot-iv-grid", published=published),
        _render_dataframe("Valuation Explanation", valuation_explanation, table_id="valuation-explanation", published=published),
        _render_raw_notes(_load_markdown(artifact_dir / "summary.md"), published=published),
        _render_available_files(_artifact_files(artifact_dir, artifact_dir, report_metadata=report_metadata), base_dir=base_dir),
    ]
    raw_details_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="raw-details" id="raw-details">'
        '<h2>Raw Details</h2>'
        '<p class="section-intro">These saved tables remain available as the audit layer, but they sit below the decision-first sections on purpose.</p>'
        + "".join(section for section in raw_detail_sections if clean_string(section))
        + "</section>"
    )

    body = (
        '<div data-scenario-tabbed-page>'
        + hero
        + shareability_note
        + (f'<section class="panel sticky-summary-strip">{top_strip}</section>' if top_strip else "")
        + _render_warnings(warnings)
        + tabs
        + summary_section
        + compare_section
        + forward_lab_section
        + strategy_selector_section
        + path_case_summary_section
        + path_explorer_section
        + strategies_section
        + valuation_section
        + replay_section
        + assumptions_section
        + raw_details_section
        + "</div>"
    )
    return body, SCENARIO_TABS_HEAD + CONTRACT_SELECTION_HEAD + PATH_CASE_SUMMARY_HEAD + PATH_CONTRACT_EXPLORER_HEAD, SCENARIO_TABS_SCRIPT + PATH_CASE_SUMMARY_SCRIPT + PATH_CONTRACT_EXPLORER_SCRIPT


CONTRACT_SELECTION_HEAD = """
  <style>
    .contract-lab-controls { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:14px; }
    .contract-lab-button { appearance:none; border:1px solid var(--line); background:#ffffff; border-radius:999px; padding:9px 14px; font:inherit; font-weight:600; cursor:pointer; }
    .contract-lab-button.is-active { background:var(--accent-soft); border-color:var(--accent); color:var(--accent); }
    .contract-lab-panel[hidden] { display:none !important; }
    .rank-card-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:14px; }
    .rank-card { border:1px solid var(--line); border-radius:18px; background:var(--panel-soft); padding:18px 18px 16px; display:grid; gap:10px; }
    .rank-card h3 { margin:0; font-size:1.02rem; line-height:1.35; }
    .rank-card .winner { font-weight:700; font-size:1.08rem; }
    .rank-card p { margin:0; line-height:1.45; }
    .rank-card .scenario-link-chip { justify-self:start; margin-top:4px; }
  </style>
"""


CONTRACT_SELECTION_BODY_END = """
  <script>
    document.querySelectorAll("[data-contract-lab-root]").forEach((root) => {
      const buttons = Array.from(root.querySelectorAll("[data-contract-lab-target]"));
      const panels = Array.from(root.querySelectorAll("[data-contract-lab-panel]"));
      const activate = (target) => {
        buttons.forEach((button) => {
          const active = button.getAttribute("data-contract-lab-target") === target;
          button.classList.toggle("is-active", active);
          button.setAttribute("aria-selected", active ? "true" : "false");
        });
        panels.forEach((panel) => { panel.hidden = panel.getAttribute("data-contract-lab-panel") !== target; });
      };
      buttons.forEach((button) => button.addEventListener("click", () => activate(button.getAttribute("data-contract-lab-target"))));
      if (buttons.length) activate(buttons[0].getAttribute("data-contract-lab-target"));
    });
  </script>
"""


PATH_CONTRACT_EXPLORER_HEAD = """
  <style>
    .path-explorer-root { display:grid; gap:16px; min-width:0; }
    .path-explorer-controls { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; align-items:end; min-width:0; }
    .path-explorer-chart-wrap { border:1px solid var(--line); border-radius:18px; background:#fffdf8; padding:14px; overflow:auto; max-width:100%; }
    .path-explorer-note { color:var(--muted); font-size:0.92rem; }
    .strategy-selector-warning { color: var(--muted); font-size: 0.92rem; }
    .path-explorer-root .panel,
    .path-explorer-root .table-wrap { min-width: 0; }
    #path-contract-candidate-table tbody tr.is-selected { background:#fff7e6; }
    #path-contract-candidate-table tbody tr.is-dimmed { opacity:0.58; }
  </style>
"""


PATH_CONTRACT_EXPLORER_SCRIPT = """
  <script>
    (() => {
      const parseJsonScript = (id) => {
        const node = document.getElementById(id);
        if (!node) return null;
        try {
          return JSON.parse(node.textContent || "{}");
        } catch (error) {
          console.warn("Path explorer JSON parse failed", error);
          return null;
        }
      };
      const toTitle = (value) => String(value || "").replace(/_/g, " ").replace(/\\b\\w/g, (m) => m.toUpperCase());
      const query = new URLSearchParams(window.location.search || "");
      const readSeed = (name) => query.get(name) || "";
      const formatValue = (value, kind = "number") => {
        if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "n/a";
        const numeric = Number(value);
        if (kind === "currency" || kind === "number") {
          return Math.abs(numeric) >= 100 ? numeric.toFixed(0) : numeric.toFixed(2);
        }
        return numeric.toFixed(2);
      };
      const hexToRgb = (value) => {
        const cleaned = String(value || "").replace("#", "");
        return {
          r: parseInt(cleaned.slice(0, 2), 16),
          g: parseInt(cleaned.slice(2, 4), 16),
          b: parseInt(cleaned.slice(4, 6), 16),
        };
      };
      const lerp = (start, end, amount) => Math.round(start + (end - start) * amount);
      const blendHex = (start, end, amount) => {
        const from = hexToRgb(start);
        const to = hexToRgb(end);
        return `rgb(${lerp(from.r, to.r, amount)}, ${lerp(from.g, to.g, amount)}, ${lerp(from.b, to.b, amount)})`;
      };
      const paletteColor = (value, minimum, maximum, sequential = false) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) return "#f5f1e8";
        const numeric = Number(value);
        if (!sequential) {
          const bound = Math.max(Math.abs(minimum), Math.abs(maximum), 0.000001);
          const ratio = Math.max(-1, Math.min(1, numeric / bound));
          if (ratio >= 0) return blendHex("#f7f7f7", "#b2182b", ratio);
          return blendHex("#2166ac", "#f7f7f7", 1 - Math.abs(ratio));
        }
        const span = Math.max(maximum - minimum, 0.000001);
        const ratio = Math.max(0, Math.min(1, (numeric - minimum) / span));
        return blendHex("#f7fbff", "#084594", ratio);
      };
      const buildSvg = (width, height, inner, label) =>
        `<svg class="forward-lab-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}">${inner}</svg>`;
      const lineColor = (index) => ["#000000", "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"][index % 7];
      const horizonLabel = (payload, value) => {
        const lookup = (payload.orders && payload.orders.horizon_display) || payload.horizon_display || {};
        return lookup[value] || toTitle(value);
      };
      const orderedHorizonLabels = (payload, rows) => {
        const explicit = (payload.orders && payload.orders.horizons_x) || [];
        const present = new Set(rows.map((row) => row.horizon));
        const ordered = explicit.filter((label) => present.has(label));
        if (ordered.length) return ordered;
        return Array.from(
          new Map(
            rows
              .slice()
              .sort((left, right) => Number(left.requested_days || 0) - Number(right.requested_days || 0))
              .map((row) => [row.horizon, true])
          ).keys()
        ).filter(Boolean);
      };
      const chartShell = (id, title, note, svg, legend = "") => `
        <div class="chart-toolbar chart-toolbar--stacked">
          <div class="chart-toolbar-copy">${note}</div>
          <button type="button" class="chart-action" data-lightbox-target="#${id}" data-lightbox-caption="${title}">Open larger chart</button>
        </div>
        <div id="${id}" class="path-explorer-chart-wrap">${svg}</div>
        ${legend}
      `;
      const chipLegend = (entries) => {
        if (!entries.length) return "";
        return `<div class="status-chip-row">${entries.map((entry) => (
          `<span class="status-chip"><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:${entry.color};"></span>${entry.label}</span>`
        )).join("")}</div>`;
      };
      const roots = Array.from(document.querySelectorAll("[data-path-explorer-root]"));
      roots.forEach((root) => {
        const payload = parseJsonScript(root.getAttribute("data-path-data-id") || "");
        if (!payload) return;
        const familySelect = root.querySelector("[data-path-family]");
        const candidateSelect = root.querySelector("[data-path-candidate]");
        const goalSelect = root.querySelector("[data-path-goal]");
        const chartTarget = root.querySelector("[data-path-required-chart]");
        const noteTarget = root.querySelector("[data-path-chart-note]");
        const stateNoteTarget = root.querySelector("[data-path-state-note]");
        const rows = Array.isArray(payload.required_path_rows) ? payload.required_path_rows : [];
        const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
        const families = Array.isArray(payload.families) ? payload.families : [];
        const goals = Array.isArray(payload.goals) ? payload.goals : [];
        if (!candidateSelect || !goalSelect || !chartTarget || !rows.length) return;
        families.forEach((family) => {
          if (!familySelect) return;
          const option = document.createElement("option");
          option.value = family.strategy_family || "";
          option.textContent = family.strategy_label || family.strategy_family || "";
          familySelect.appendChild(option);
        });
        goals.forEach((goal) => {
          const option = document.createElement("option");
          option.value = goal;
          option.textContent = toTitle(goal);
          goalSelect.appendChild(option);
        });
        const state = {
          family: readSeed("strategy_family") || payload.defaults.family || (families[0] && families[0].strategy_family) || "",
          candidateSlug: readSeed("candidate") || payload.defaults.candidate_slug || (candidates[0] && candidates[0].candidate_slug) || "",
          goal: readSeed("goal") || payload.defaults.goal || goals[0] || "break_even",
        };
        const populateCandidates = () => {
          const options = candidates.filter((candidate) => !state.family || candidate.strategy_family === state.family);
          candidateSelect.innerHTML = "";
          options.forEach((candidate) => {
            const option = document.createElement("option");
            option.value = candidate.candidate_slug || "";
            option.textContent = candidate.candidate_label || candidate.candidate_slug || "";
            candidateSelect.appendChild(option);
          });
          if (!options.some((candidate) => candidate.candidate_slug === state.candidateSlug)) {
            state.candidateSlug = (options[0] && options[0].candidate_slug) || "";
          }
        };
        const selectedCandidate = () => candidates.find((item) => item.candidate_slug === state.candidateSlug);
        const renderStateNote = () => {
          if (!stateNoteTarget) return;
          const candidate = selectedCandidate();
          const warnings = [
            candidate && candidate.selection_scope_label,
            candidate && candidate.horizon_fit_label,
            candidate && candidate.confidence_label,
            candidate && candidate.target_beyond_expiry ? "Target beyond expiry" : "",
          ].filter(Boolean);
          stateNoteTarget.textContent = `Showing ${toTitle(state.family)} | ${candidate ? candidate.candidate_label : "selected candidate"} | ${toTitle(state.goal)}${warnings.length ? ` | ${warnings.join(" | ")}` : ""}`;
        };
        const updateCandidateTable = () => {
          const table = root.querySelector("#path-contract-candidate-table");
          if (!table) return;
          const rows = Array.from(table.querySelectorAll("tbody tr"));
          const candidate = selectedCandidate();
          rows.forEach((row) => {
            const cells = Array.from(row.querySelectorAll("td"));
            const rowCandidate = (cells[0] && cells[0].textContent || "").trim();
            const rowFamily = (cells[1] && cells[1].textContent || "").trim().toLowerCase();
            const familyMatches = !state.family || rowFamily === toTitle(state.family).toLowerCase();
            row.classList.toggle("is-selected", Boolean(candidate) && rowCandidate === candidate.candidate_label);
            row.classList.toggle("is-dimmed", !familyMatches);
          });
        };
        const renderRequiredPathChart = () => {
          const filtered = rows.filter((row) => row.candidate_slug === state.candidateSlug && row.goal === state.goal && !row.unreached);
          if (!filtered.length) {
            chartTarget.innerHTML = '<div class="path-explorer-chart-wrap"><div class="path-explorer-note">No reachable required-stock path rows were available for this candidate / goal combination.</div></div>';
            return;
          }
          const horizons = orderedHorizonLabels(payload, filtered);
          const variants = Array.from(new Set(filtered.map((row) => row.iv_variant)));
          const values = filtered.map((row) => Number(row.required_stock_price)).filter((value) => !Number.isNaN(value));
          const minY = Math.min.apply(null, values);
          const maxY = Math.max.apply(null, values);
          const left = 74;
          const top = 28;
          const chartWidth = Math.max(560, horizons.length * 100);
          const chartHeight = 252;
          const width = left + chartWidth + 32;
          const height = 372;
          const bottom = top + chartHeight;
          const xPos = (label) => {
            const idx = horizons.indexOf(label);
            const divisor = Math.max(horizons.length - 1, 1);
            return left + (idx / divisor) * chartWidth;
          };
          const yPos = (value) => {
            const span = Math.max(maxY - minY, 0.000001);
            return bottom - ((Number(value) - minY) / span) * chartHeight;
          };
          const lines = [];
          const legend = [];
          variants.forEach((variant, index) => {
            const series = filtered
              .filter((row) => row.iv_variant === variant)
              .sort((a, b) => Number(a.requested_days || 0) - Number(b.requested_days || 0));
            const points = series.map((row) => `${xPos(row.horizon)},${yPos(row.required_stock_price)}`).join(" ");
            const color = lineColor(index);
            legend.push({ label: toTitle(variant), color });
            lines.push(`<polyline fill="none" stroke="${color}" stroke-width="3" points="${points}"></polyline>`);
            series.forEach((row) => {
              lines.push(`<circle cx="${xPos(row.horizon)}" cy="${yPos(row.required_stock_price)}" r="4" fill="${color}" stroke="#ffffff" stroke-width="1"></circle>`);
            });
          });
          const xLabels = horizons.map((label) => `<text x="${xPos(label)}" y="${bottom + 22}" text-anchor="middle" font-size="12" fill="#6b645b">${horizonLabel(payload, label)}</text>`).join("");
          const yLabels = [0, 0.25, 0.5, 0.75, 1].map((step) => {
            const value = minY + (maxY - minY) * step;
            const y = bottom - step * chartHeight;
            return `<text x="${left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#6b645b">${value.toFixed(2)}</text><line x1="${left}" y1="${y}" x2="${left + chartWidth}" y2="${y}" stroke="#ebe4d8" stroke-dasharray="4 4"></line>`;
          }).join("");
          const candidate = candidates.find((item) => item.candidate_slug === state.candidateSlug);
          const clamped = filtered.some((row) => Boolean(row.clamped_to_expiry));
          if (noteTarget) {
            noteTarget.textContent = `Required stock price over time for ${candidate ? candidate.candidate_label : "the selected candidate"} under ${toTitle(state.goal)}. Each line is a different IV scenario or IV path, and the target slot uses ${payload.target_horizon_label || "the active target horizon"} when it differs from standard 1w / 1m / 3m / 6m labels.${clamped ? " Some later horizons are clamped to expiry for this contract." : ""}`;
          }
          const svg = buildSvg(
            width,
            height,
            `<text x="${left}" y="16" font-size="13" font-weight="700" fill="#191714">Required Stock Path Chart | ${candidate ? candidate.candidate_label : ""} | ${toTitle(state.goal)}</text><line x1="${left}" y1="${bottom}" x2="${left + chartWidth}" y2="${bottom}" stroke="#bdb4a6"></line><line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#bdb4a6"></line>${yLabels}${lines.join("")}${xLabels}`,
            "Required stock path chart"
          );
          chartTarget.innerHTML = chartShell(
            `path-required-${root.getAttribute("data-path-data-id") || "root"}`,
            "Required Stock Path Chart",
            clamped
              ? "This required-path read includes expiry-clamped horizons because the requested target extends beyond this contract's expiry."
              : "This chart is the main explorer visual. Read it before drilling into exact-candidate path tables.",
            svg,
            chipLegend(legend)
          );
        };
        const render = () => {
          if (familySelect) {
            familySelect.value = state.family;
          }
          populateCandidates();
          candidateSelect.value = state.candidateSlug;
          goalSelect.value = state.goal;
          renderStateNote();
          updateCandidateTable();
          renderRequiredPathChart();
        };
        if (familySelect) {
          familySelect.addEventListener("change", () => {
            state.family = familySelect.value;
            render();
          });
        }
        candidateSelect.addEventListener("change", () => { state.candidateSlug = candidateSelect.value; render(); });
        goalSelect.addEventListener("change", () => { state.goal = goalSelect.value; render(); });
        render();
      });
      document.querySelectorAll("[data-path-family-jump]").forEach((node) => {
        node.addEventListener("click", () => {
          const family = node.getAttribute("data-path-family-jump") || "";
          const root = document.querySelector("[data-path-explorer-root]");
          if (!root) return;
          const familySelect = root.querySelector("[data-path-family]");
          if (!familySelect) return;
          familySelect.value = family;
          familySelect.dispatchEvent(new Event("change", { bubbles: true }));
        });
      });
    })();
  </script>
"""


PATH_CASE_SUMMARY_HEAD = """
  <style>
    .path-case-root { display:grid; gap:16px; min-width:0; }
    .path-case-controls { display:grid; gap:14px; }
    .path-case-pill-row { display:flex; flex-wrap:wrap; gap:8px; }
    .path-case-pill {
      appearance:none;
      border:1px solid var(--line);
      background:#ffffff;
      color:var(--text);
      border-radius:999px;
      padding:10px 14px;
      font:inherit;
      font-weight:700;
      cursor:pointer;
    }
    .path-case-pill.is-active {
      background: var(--accent-soft);
      border-color: var(--accent);
      color: var(--accent);
    }
    .path-case-control-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:12px; }
    .path-case-chart-wrap { border:1px solid var(--line); border-radius:18px; background:#fffdf8; padding:14px; overflow:auto; max-width:100%; }
    .path-case-assumption-strip { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:12px; }
    .path-case-assumption-card {
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px 16px;
      background:var(--panel-soft);
    }
    .path-case-assumption-card h4 { margin:0 0 8px; font-size:0.95rem; }
    .path-case-assumption-value { font-size:1.02rem; font-weight:700; margin-bottom:6px; }
    .path-case-assumption-copy { color:var(--muted); font-size:0.92rem; line-height:1.45; }
    .path-case-table-wrap { overflow:auto; max-width:100%; }
    .path-case-table { width:100%; border-collapse:collapse; min-width:1120px; font-size:0.96rem; }
    .path-case-table th {
      position:sticky; top:0; z-index:1; background:#f8f5ef;
      text-align:left; padding:10px 12px; border-bottom:1px solid var(--line);
    }
    .path-case-table td { padding:10px 12px; border-bottom:1px solid #ece6dc; vertical-align:top; }
    .path-case-table td.numeric, .path-case-table th.numeric { text-align:right; }
    .path-case-table td strong { display:block; }
    .path-case-table tbody tr.is-emphasis { background:#fff7e6; }
    .path-case-mini-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }
    .path-case-mini-card { border:1px solid var(--line); border-radius:16px; background:var(--panel-soft); padding:14px; }
  </style>
"""


PATH_CASE_SUMMARY_SCRIPT = """
  <script>
    (() => {
      const parseJsonScript = (id) => {
        const node = document.getElementById(id);
        if (!node) return null;
        try {
          return JSON.parse(node.textContent || "{}");
        } catch (error) {
          console.warn("Path case summary JSON parse failed", error);
          return null;
        }
      };
      const titleCase = (value) => String(value || "").replace(/_/g, " ").replace(/\\b\\w/g, (m) => m.toUpperCase());
      const query = new URLSearchParams(window.location.search || "");
      const formatNumber = (value, kind = "number") => {
        if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "n/a";
        const numeric = Number(value);
        if (kind === "pct") return `${(numeric * 100).toFixed(1)}%`;
        if (kind === "currency") return `$${Math.abs(numeric) >= 100 ? numeric.toFixed(0) : numeric.toFixed(2)}`;
        return Math.abs(numeric) >= 100 ? numeric.toFixed(0) : numeric.toFixed(2);
      };
      const buildSvg = (width, height, inner, label) => `<svg class="forward-lab-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}">${inner}</svg>`;
      const familyColor = (family) => ({
        long_stock: "#000000",
        long_call: "#E69F00",
        bull_call_spread: "#56B4E9",
        long_put: "#009E73",
        bear_put_spread: "#0072B2",
        covered_call: "#D55E00",
        cash_secured_put: "#CC79A7",
      }[family] || "#6b645b");
      const ivColor = (index) => ["#0f5c57", "#5f7c8a", "#9a6324", "#6c5ce7", "#b03060", "#008080", "#7f8c8d"][index % 7];
      const legend = (entries) => {
        if (!entries.length) return "";
        return `<div class="status-chip-row">${entries.map((entry) => `<span class="status-chip"><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:${entry.color};"></span>${entry.label}</span>`).join("")}</div>`;
      };
      const chartShell = (id, title, note, svg, legendHtml = "") => `
        <div class="chart-toolbar chart-toolbar--stacked">
          <div class="chart-toolbar-copy">${note}</div>
          <button type="button" class="chart-action" data-lightbox-target="#${id}" data-lightbox-caption="${title}">Open larger chart</button>
        </div>
        <div id="${id}" class="path-case-chart-wrap">${svg}</div>
        ${legendHtml}
      `;
      const horizonLabel = (payload, label) => {
        const lookup = (payload.orders && payload.orders.horizon_display) || payload.horizon_display || {};
        return lookup[label] || titleCase(label);
      };
      const orderedHorizonLabels = (payload, rows) => {
        const explicit = (payload.orders && payload.orders.horizons_x) || payload.horizon_order || [];
        const present = new Set(rows.map((row) => String(row.horizon || "")));
        const ordered = explicit.filter((label) => present.has(label));
        if (ordered.length) return ordered;
        return Array.from(
          new Map(
            rows
              .slice()
              .sort((left, right) => Number(left.requested_days || 0) - Number(right.requested_days || 0))
              .map((row) => [String(row.horizon || ""), true])
          ).keys()
        ).filter(Boolean);
      };
      const roots = Array.from(document.querySelectorAll("[data-path-case-root]"));
      roots.forEach((root) => {
        const payload = parseJsonScript(root.getAttribute("data-path-case-data-id") || "");
        if (!payload) return;
        const rows = Array.isArray(payload.chart_rows) ? payload.chart_rows : [];
        const strategyRows = Array.isArray(payload.strategy_rows) ? payload.strategy_rows : [];
        if (!rows.length && !strategyRows.length) return;
        const caseButtons = Array.from(root.querySelectorAll("[data-path-case-case]"));
        const goalSelect = root.querySelector("[data-path-case-goal]");
        const displayModeSelect = root.querySelector("[data-path-case-display-mode]");
        const ivModeSelect = root.querySelector("[data-path-case-iv-mode]");
        const ivVariantSelect = root.querySelector("[data-path-case-iv-variant]");
        const familySelect = root.querySelector("[data-path-case-family]");
        const chartTarget = root.querySelector("[data-path-case-chart]");
        const noteTarget = root.querySelector("[data-path-case-note]");
        const assumptionsTarget = root.querySelector("[data-path-case-assumptions]");
        const tableTarget = root.querySelector("[data-path-case-table]");
        const modeNoteTarget = root.querySelector("[data-path-case-mode-note]");
        const stateNoteTarget = root.querySelector("[data-path-case-state-note]");
        const tableNoteTarget = root.querySelector("[data-path-case-table-note]");
        const defaults = payload.defaults || {};
        const state = {
          caseLabel: query.get("path_case") || defaults.case_label || "0%",
          goal: query.get("goal") || defaults.goal || "break_even",
          displayMode: query.get("path_case_display_mode") || defaults.display_mode || "strategy_compare",
          ivMode: query.get("path_case_iv_mode") || defaults.iv_mode || "path_preset",
          ivVariant: defaults.iv_variant || "",
          family: query.get("strategy_family") || defaults.strategy_family || "",
        };
        const setOptions = (node, values, formatter) => {
          if (!node) return;
          node.innerHTML = "";
          values.forEach((value) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = formatter ? formatter(value) : value;
            node.appendChild(option);
          });
        };
        setOptions(goalSelect, payload.goals || [], (value) => titleCase(value));
        setOptions(displayModeSelect, payload.display_modes || [], (value) => value === "iv_compare" ? "IV Paths" : "Strategies");
        setOptions(ivModeSelect, payload.iv_modes || [], (value) => value === "point_scenario" ? "Point Scenario" : "IV Path");
        setOptions(familySelect, payload.families || [], (value) => titleCase(value));
        if ((payload.families || []).length && !(payload.families || []).includes(state.family)) {
          state.family = payload.families[0];
        }
        const refreshIvVariants = () => {
          const variants = (payload.iv_variants && payload.iv_variants[state.ivMode]) || [];
          setOptions(ivVariantSelect, variants, (value) => state.ivMode === "point_scenario" ? value : titleCase(value));
          if (!variants.includes(state.ivVariant)) {
            state.ivVariant = variants[0] || "";
          }
          if (ivVariantSelect) ivVariantSelect.value = state.ivVariant;
        };
        const selectedFamilyLabel = () => titleCase(state.family);
        const selectedIvLabel = () => state.ivMode === "point_scenario" ? state.ivVariant : titleCase(state.ivVariant);
        const renderStateNote = () => {
          if (!stateNoteTarget) return;
          stateNoteTarget.textContent = state.displayMode === "iv_compare"
            ? `Showing ${state.caseLabel} | ${titleCase(state.goal)} | IV Paths mode | Family: ${selectedFamilyLabel()} | Comparing ${state.ivMode === "point_scenario" ? "point IV scenarios" : "IV paths"} with ${selectedIvLabel()} highlighted.`
            : `Showing ${state.caseLabel} | ${titleCase(state.goal)} | Strategies mode | IV assumption: ${selectedIvLabel()} | ${selectedFamilyLabel()} stays highlighted so you can compare it against the rest.`;
        };
        const updateCaseButtons = () => {
          caseButtons.forEach((button) => {
            const active = button.getAttribute("data-path-case-case") === state.caseLabel;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-pressed", active ? "true" : "false");
          });
        };
        const renderAssumptions = () => {
          if (!assumptionsTarget) return;
          const caseInfo = (payload.case_definitions || {})[state.caseLabel] || {};
          const displayLabel = state.displayMode === "iv_compare" ? "IV Paths" : "Strategies";
          assumptionsTarget.innerHTML = `
            <div class="path-case-assumption-card">
              <h4>Required Path</h4>
              <div class="path-case-assumption-value">${titleCase(state.goal)}</div>
              <div class="path-case-assumption-copy">Colored lines show the stock level each strategy or IV variant needs over time to clear the selected goal.</div>
            </div>
            <div class="path-case-assumption-card">
              <h4>Assumed Path</h4>
              <div class="path-case-assumption-value">Configured thesis path</div>
              <div class="path-case-assumption-copy">The thicker dark line is the user-configured thesis path from the current contract-selection run.</div>
            </div>
            <div class="path-case-assumption-card">
              <h4>Case Outcome</h4>
              <div class="path-case-assumption-value">${state.caseLabel}</div>
              <div class="path-case-assumption-copy">Standardized comparison case ending near ${formatNumber(caseInfo.endpoint_price, "currency")} over the active target horizon.</div>
            </div>
            <div class="path-case-assumption-card">
              <h4>Display Mode</h4>
              <div class="path-case-assumption-value">${displayLabel}</div>
              <div class="path-case-assumption-copy">${state.displayMode === "iv_compare" ? `One family (${selectedFamilyLabel()}) across multiple IV assumptions.` : `Multiple strategy families under one IV assumption (${selectedIvLabel()}).`}</div>
            </div>
          `;
        };
        const renderChart = () => {
          if (!chartTarget) return;
          const filtered = rows.filter((row) => {
            if (row.case_label !== state.caseLabel || row.goal !== state.goal || row.display_mode !== state.displayMode || row.iv_mode !== state.ivMode) {
              return false;
            }
            if (state.displayMode === "strategy_compare") {
              return row.iv_variant === state.ivVariant;
            }
            if (!state.family) return false;
            return row.strategy_family === state.family;
          });
          if (!filtered.length) {
            chartTarget.innerHTML = '<div class="forward-lab-empty">No path-case rows matched the current selection.</div>';
            return;
          }
          const horizons = orderedHorizonLabels(payload, filtered);
          const requiredSeries = filtered.filter((row) => row.series_kind === "required_path");
          const seriesLabels = Array.from(new Set(requiredSeries.map((row) => row.series_label)));
          const numeric = filtered.map((row) => Number(row.spot_price)).filter((value) => !Number.isNaN(value));
          const minY = Math.min(...numeric);
          const maxY = Math.max(...numeric);
          const left = 84;
          const top = 32;
          const chartWidth = Math.max(560, horizons.length * 110);
          const chartHeight = 286;
          const width = left + chartWidth + 36;
          const height = 432;
          const bottom = top + chartHeight;
          const xPos = (label) => {
            const idx = horizons.indexOf(label);
            const divisor = Math.max(horizons.length - 1, 1);
            return left + (idx / divisor) * chartWidth;
          };
          const yPos = (value) => {
            const span = Math.max(maxY - minY, 0.000001);
            return bottom - ((Number(value) - minY) / span) * chartHeight;
          };
          const grid = [0, 0.25, 0.5, 0.75, 1].map((step) => {
            const y = bottom - step * chartHeight;
            const value = minY + (maxY - minY) * step;
            return `<text x="${left - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#6b645b">${value.toFixed(2)}</text><line x1="${left}" y1="${y}" x2="${left + chartWidth}" y2="${y}" stroke="#ebe4d8" stroke-dasharray="4 4"></line>`;
          }).join("");
          const shapes = [];
          const legendEntries = [];
          const assumed = Array.from(
            new Map(
              filtered
                .filter((row) => row.series_kind === "assumed_path")
                .sort((a, b) => Number(a.requested_days || 0) - Number(b.requested_days || 0))
                .map((row) => [row.horizon, row])
            ).values()
          );
          if (assumed.length) {
            const points = assumed.map((row) => `${xPos(row.horizon)},${yPos(row.spot_price)}`).join(" ");
            shapes.push(`<polyline fill="none" stroke="#191714" stroke-width="5" points="${points}"></polyline>`);
            assumed.forEach((row) => shapes.push(`<circle cx="${xPos(row.horizon)}" cy="${yPos(row.spot_price)}" r="4.5" fill="#191714" stroke="#ffffff" stroke-width="1"></circle>`));
            legendEntries.push({ label: "Assumed Path", color: "#191714" });
          }
          seriesLabels
            .slice()
            .sort((leftLabel, rightLabel) => {
              if (state.displayMode === "strategy_compare") {
                const leftSelected = leftLabel === selectedFamilyLabel();
                const rightSelected = rightLabel === selectedFamilyLabel();
                if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
              } else {
                const leftSelected = leftLabel === state.ivVariant;
                const rightSelected = rightLabel === state.ivVariant;
                if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
              }
              return String(leftLabel).localeCompare(String(rightLabel));
            })
            .forEach((label, index) => {
            const series = filtered.filter((row) => row.series_kind === "required_path" && row.series_label === label).sort((a, b) => Number(a.requested_days || 0) - Number(b.requested_days || 0));
            if (!series.length) return;
            const family = series[0].strategy_family || "";
            const color = state.displayMode === "iv_compare" ? ivColor(index) : familyColor(family);
            const emphasized = state.displayMode === "iv_compare" ? label === state.ivVariant : family === state.family;
            const opacity = emphasized ? 1 : 0.55;
            const strokeWidth = emphasized ? 4 : 2.6;
            const points = series.filter((row) => row.spot_price !== null && row.spot_price !== undefined && !Number.isNaN(Number(row.spot_price))).map((row) => `${xPos(row.horizon)},${yPos(row.spot_price)}`).join(" ");
            if (!points) return;
            shapes.push(`<polyline fill="none" stroke="${color}" stroke-width="${strokeWidth}" opacity="${opacity}" points="${points}"></polyline>`);
            series.forEach((row) => {
              if (row.spot_price === null || row.spot_price === undefined || Number.isNaN(Number(row.spot_price))) return;
              shapes.push(`<circle cx="${xPos(row.horizon)}" cy="${yPos(row.spot_price)}" r="${emphasized ? 4.4 : 3.6}" fill="${color}" opacity="${opacity}" stroke="#ffffff" stroke-width="1"></circle>`);
            });
            legendEntries.push({ label: emphasized ? `${label} (highlighted)` : label, color });
          });
          const xLabels = horizons.map((label) => `<text x="${xPos(label)}" y="${bottom + 24}" text-anchor="middle" font-size="12" fill="#6b645b">${horizonLabel(payload, label)}</text>`).join("");
          const title = state.displayMode === "iv_compare"
            ? `Required Stock Path by IV Path | ${selectedFamilyLabel()}`
            : "Required Stock Path by Strategy";
          if (noteTarget) {
            noteTarget.textContent = state.displayMode === "iv_compare"
              ? `Thick line = your configured assumed path. Colored lines show how different IV assumptions change the required stock path for ${selectedFamilyLabel()}.`
              : "Thick line = your configured assumed path. Colored lines show what each strategy family requires over time to clear the selected goal, with the selected family highlighted.";
          }
          if (modeNoteTarget) {
            modeNoteTarget.textContent = state.displayMode === "iv_compare"
              ? `IV Paths mode keeps ${selectedFamilyLabel()} fixed and compares ${state.ivMode === "point_scenario" ? "point IV scenarios" : "IV paths"}. The table below also switches to IV-variant comparisons for this family.`
              : `Strategies mode keeps one IV assumption fixed (${selectedIvLabel()}) and compares the representative family winners while keeping ${selectedFamilyLabel()} visually prominent.`;
          }
          const svg = buildSvg(
            width,
            height,
            `<text x="${left}" y="16" font-size="13" font-weight="700" fill="#191714">${title}</text><line x1="${left}" y1="${bottom}" x2="${left + chartWidth}" y2="${bottom}" stroke="#bdb4a6"></line><line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#bdb4a6"></line>${grid}${shapes.join("")}${xLabels}`,
            title
          );
          chartTarget.innerHTML = chartShell(
            `path-case-chart-${root.getAttribute("data-path-case-data-id") || "root"}`,
            title,
            "Read the thick assumed path against the thinner required-path lines. If the assumed path stays above a required path, that strategy or IV path is more likely to work under these assumptions.",
            svg,
            legend(legendEntries)
          );
        };
        const renderTable = () => {
          if (!tableTarget) return;
          let filtered = strategyRows.filter((row) =>
            row.case_label === state.caseLabel &&
            row.goal === state.goal &&
            row.iv_mode === state.ivMode
          );
          if (!filtered.length) {
            tableTarget.innerHTML = '<div class="forward-lab-empty">No ranked case rows matched the current selection.</div>';
            return;
          }
          if (state.displayMode === "iv_compare") {
            filtered = filtered
              .filter((row) => row.strategy_family === state.family)
              .slice()
              .sort((left, right) => Number(right.case_rank_score || 0) - Number(left.case_rank_score || 0));
            if (tableNoteTarget) {
              tableNoteTarget.textContent = `This IV-variant table keeps ${selectedFamilyLabel()} fixed and compares how each ${state.ivMode === "point_scenario" ? "point IV scenario" : "IV path"} changes the case outcome.`;
            }
            const body = filtered.map((row) => `
              <tr class="${row.iv_variant === state.ivVariant ? "is-emphasis" : ""}">
                <td><strong>${state.ivMode === "point_scenario" ? row.iv_variant : titleCase(row.iv_variant)}</strong><span class="strategy-selector-warning">${row.winning_candidate_label || ""}</span></td>
                <td>${row.required_path_difficulty || ""}</td>
                <td class="numeric">${formatNumber(row.modeled_value, "currency")}</td>
                <td class="numeric">${formatNumber(row.profit_loss, "currency")}</td>
                <td class="numeric">${formatNumber(row.profit_loss_pct, "pct")}</td>
                <td class="numeric">${formatNumber(row.difference_vs_stock, "currency")}</td>
                <td class="numeric">${formatNumber(row.break_even, "currency")}</td>
                <td>${[row.coverage_flags, row.horizon_fit_label, row.confidence_label].filter(Boolean).join(" | ")}</td>
                <td>${row.iv_sensitivity_summary || ""}<br><span class="strategy-selector-warning">${row.time_sensitivity_summary || ""}</span></td>
                <td>${row.why_it_wins || ""}<br><span class="strategy-selector-warning">${row.why_it_loses || ""}</span></td>
              </tr>
            `).join("");
            tableTarget.innerHTML = `
              <div class="path-case-table-wrap">
                <table class="path-case-table" id="path-case-table">
                  <thead>
                    <tr>
                      <th>IV Variant</th>
                      <th>Required Path Difficulty</th>
                      <th class="numeric">Modeled Value</th>
                      <th class="numeric">PnL $</th>
                      <th class="numeric">PnL %</th>
                      <th class="numeric">Vs Stock</th>
                      <th class="numeric">Break-even</th>
                      <th>Coverage / Timing</th>
                      <th>IV / Timing Sensitivity</th>
                      <th>Why It Wins / Loses</th>
                    </tr>
                  </thead>
                  <tbody>${body}</tbody>
                </table>
              </div>
            `;
            return;
          }
          filtered = filtered
            .filter((row) => row.iv_variant === state.ivVariant)
            .slice()
            .sort((left, right) => {
              const leftSelected = left.strategy_family === state.family;
              const rightSelected = right.strategy_family === state.family;
              if (leftSelected !== rightSelected) return leftSelected ? -1 : 1;
              return Number(left.case_rank || 0) - Number(right.case_rank || 0);
            });
          if (tableNoteTarget) {
            tableNoteTarget.textContent = `This family-ranked table keeps the IV assumption fixed at ${selectedIvLabel()} and sorts strategies by the current ${state.caseLabel} case outcome, with ${selectedFamilyLabel()} emphasized.`;
          }
          const body = filtered.map((row) => `
            <tr class="${row.strategy_family === state.family ? "is-emphasis" : ""}">
              <td><strong>${row.strategy_label || titleCase(row.strategy_family)}</strong><span class="strategy-selector-warning">${row.winning_candidate_label || ""}</span></td>
              <td>${row.relevance_label || ""}</td>
              <td>${row.required_path_difficulty || ""}</td>
              <td class="numeric">${formatNumber(row.modeled_value, "currency")}</td>
              <td class="numeric">${formatNumber(row.profit_loss, "currency")}</td>
              <td class="numeric">${formatNumber(row.profit_loss_pct, "pct")}</td>
              <td class="numeric">${formatNumber(row.difference_vs_stock, "currency")}</td>
              <td class="numeric">${formatNumber(row.capital_required, "currency")}</td>
              <td class="numeric">${formatNumber(row.affordable_units)}</td>
              <td class="numeric">${formatNumber(row.max_loss, "currency")}</td>
              <td class="numeric">${formatNumber(row.break_even, "currency")}</td>
              <td>${row.iv_sensitivity_summary || ""}</td>
              <td>${row.time_sensitivity_summary || ""}</td>
              <td>${row.why_it_wins || ""}<br><span class="strategy-selector-warning">${row.why_it_loses || ""}</span></td>
            </tr>
          `).join("");
          tableTarget.innerHTML = `
            <div class="path-case-table-wrap">
              <table class="path-case-table" id="path-case-table">
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Relevance</th>
                    <th>Required Path Difficulty</th>
                    <th class="numeric">Modeled Value</th>
                    <th class="numeric">PnL $</th>
                    <th class="numeric">PnL %</th>
                    <th class="numeric">Vs Stock</th>
                    <th class="numeric">Capital</th>
                    <th class="numeric">Affordable Units</th>
                    <th class="numeric">Max Loss</th>
                    <th class="numeric">Break-even</th>
                    <th>IV Sensitivity</th>
                    <th>Timing Sensitivity</th>
                    <th>Why It Wins / Loses</th>
                  </tr>
                </thead>
                <tbody>${body}</tbody>
              </table>
            </div>
          `;
        };
        const render = () => {
          if (goalSelect) goalSelect.value = state.goal;
          if (displayModeSelect) displayModeSelect.value = state.displayMode;
          if (ivModeSelect) ivModeSelect.value = state.ivMode;
          if (familySelect) familySelect.value = state.family;
          refreshIvVariants();
          updateCaseButtons();
          renderStateNote();
          renderAssumptions();
          renderChart();
          renderTable();
        };
        caseButtons.forEach((button) => button.addEventListener("click", () => {
          state.caseLabel = button.getAttribute("data-path-case-case") || state.caseLabel;
          render();
        }));
        if (goalSelect) goalSelect.addEventListener("change", () => { state.goal = goalSelect.value || state.goal; render(); });
        if (displayModeSelect) displayModeSelect.addEventListener("change", () => { state.displayMode = displayModeSelect.value || state.displayMode; render(); });
        if (ivModeSelect) ivModeSelect.addEventListener("change", () => { state.ivMode = ivModeSelect.value || state.ivMode; refreshIvVariants(); render(); });
        if (ivVariantSelect) ivVariantSelect.addEventListener("change", () => { state.ivVariant = ivVariantSelect.value || state.ivVariant; render(); });
        if (familySelect) familySelect.addEventListener("change", () => { state.family = familySelect.value || state.family; render(); });
        render();
      });
    })();
  </script>
"""


def _render_contract_selection_rank_cards(report_metadata: dict[str, Any], *, published: bool = False) -> str:
    payload = _contract_selection_payload(report_metadata)
    cards = payload.get("best_candidate_cards") or []
    if not cards:
        return '<p class="empty-state">No ranking cards were available.</p>'
    blocks = []
    for card in cards:
        blocks.append(
            '<article class="rank-card">'
            f"<h3>{escape(clean_string(card.get('title')))}</h3>"
            f'<div class="winner">{escape(clean_string(card.get("winner_candidate")))}</div>'
            f'<div class="meta-line">{escape(clean_string(card.get("winner_strategy")).replace("_", " ").title())}</div>'
            f'<div class="meta-line">{escape(_format_scalar(card.get("winner_value"), published=published))}</div>'
            f'<p>{escape(clean_string(card.get("rationale")))}</p>'
            "</article>"
        )
    return '<div class="rank-card-grid">' + "".join(blocks) + "</div>"


def _render_contract_selection_rank_cards_from_payload(payload: dict[str, Any], *, published: bool = False) -> str:
    return _render_contract_selection_rank_cards({"contract_selection": payload}, published=published)


def _render_strategy_selector_rank_cards(
    rows: list[dict[str, Any]],
    *,
    published: bool = False,
    link_base: str = "",
    link_anchor: str = "path-contract-explorer",
    link_label: str = "Open this family in Path & Contract Explorer",
) -> str:
    if not rows:
        return '<p class="empty-state">No Strategy Selector ranking cards were available.</p>'
    blocks = []
    for row in rows:
        status = clean_string(row.get("card_status"))
        informative = bool(row.get("is_informative", True))
        strategy_label = clean_string(row.get("winner_strategy_label") or row.get("winner_strategy")).replace("_", " ").title()
        if not strategy_label:
            strategy_label = "No clear edge"
        warning = clean_string(row.get("warning"))
        winner_strategy = clean_string(row.get("winner_strategy"))
        deep_link = (
            (
                f'<a class="scenario-link-chip" href="{escape(_href_with_query_params(link_base or f"#{link_anchor}", strategy_family=winner_strategy))}"'
                + (f' data-path-family-jump="{escape(winner_strategy)}"' if not link_base else "")
                + f">{escape(link_label)}</a>"
            )
            if winner_strategy and informative
            else ""
        )
        status_line = ""
        if status == "weak_differentiation":
            status_line = '<div class="meta-line">Insufficient differentiation under current assumptions</div>'
        elif status == "no_clear_edge":
            status_line = '<div class="meta-line">No clear edge under current data</div>'
        coverage_line = ""
        coverage_flags = clean_string(row.get("coverage_flags"))
        if coverage_flags:
            coverage_line = f'<div class="meta-line">{escape(coverage_flags)}</div>'
        blocks.append(
            '<article class="rank-card">'
            f"<h3>{escape(clean_string(row.get('title')))}</h3>"
            f'<div class="winner">{escape(strategy_label)}</div>'
            + status_line
            + coverage_line
            + f'<div class="meta-line">{escape(clean_string(row.get("winner_candidate_label")) or "Best available candidate under current assumptions")}</div>'
            f'<p>{escape(clean_string(row.get("reason")))}</p>'
            + (f'<p class="strategy-selector-warning"><strong>Warning:</strong> {escape(warning)}</p>' if warning else "")
            + deep_link
            + "</article>"
        )
    return '<div class="rank-card-grid">' + "".join(blocks) + "</div>"


def _render_strategy_selector_component(
    report_metadata: dict[str, Any],
    *,
    selector_rows: pd.DataFrame,
    selector_rankings: pd.DataFrame,
    published: bool = False,
) -> str:
    payload = _contract_selection_payload(report_metadata)
    selector_context = payload.get("strategy_selector_context") or {}
    selector_cards = payload.get("strategy_selector_best_cards") or selector_rankings.to_dict(orient="records")
    selector_defaults = payload.get("strategy_selector_defaults") or {}
    default_family = clean_string(selector_defaults.get("default_strategy_family") or payload.get("default_strategy_family"))
    if default_family and "strategy_family" in selector_rows.columns:
        default_family_row = selector_rows.loc[selector_rows["strategy_family"] == default_family].head(1)
    else:
        default_family_row = pd.DataFrame()
    default_family_label = (
        clean_string(default_family_row.iloc[0].get("strategy_label"))
        if not default_family_row.empty
        else default_family.replace("_", " ").title()
    )
    coverage_value = "Nearby snapshot fallback" if bool((payload.get("selection_scope") or {}).get("used_nearby_snapshot_fallback")) else "Exact coverage"
    timing_value = (
        "Target beyond expiry"
        if bool(selector_rows.get("target_beyond_expiry", pd.Series(dtype=bool)).fillna(False).any())
        else "Within available expiries"
    )
    differentiation_value = (
        "Some categories are compressed"
        if any(clean_string(card.get("card_status")) in {"weak_differentiation", "no_clear_edge"} for card in selector_cards)
        else "Distinct family edges available"
    )
    selector_strip = _render_summary_strip(
        [
            ("Focus", "Strategy family choice"),
            ("Coverage", coverage_value),
            ("Timing Fit", timing_value),
            ("Differentiation", differentiation_value),
            ("Default Family", default_family_label),
            ("Ranking", "Heuristic only"),
        ],
        published=published,
    )
    inputs = _render_key_value_rows(
        [
            ("Target Stock Price", payload.get("target_price")),
            ("Target Date", payload.get("target_date")),
            ("Comparison Capital", payload.get("comparison_capital")),
            ("Objective Mode", clean_string(selector_context.get("objective_mode")).replace("_", " ").title()),
            ("Downside Tolerance", clean_string(selector_context.get("downside_tolerance")).title()),
            ("Simplicity Preference", clean_string(selector_context.get("simplicity_preference")).title()),
            ("IV Path", clean_string(payload.get("iv_path_name")).replace("_", " ").title()),
            ("Stock Path", clean_string(payload.get("stock_path_name")).replace("_", " ").title()),
        ],
        published=published,
    )
    ranking_columns = [
        column
        for column in [
            "strategy_label",
            "relevance_label",
            "role",
            "target_pnl",
            "target_return_pct",
            "difference_vs_stock",
            "capital_required",
            "max_loss",
            "break_even",
            "confidence_label",
            "horizon_fit_label",
            "iv_sensitivity_summary",
            "time_sensitivity_summary",
            "objective_score",
            "winning_candidate_pointer",
            "notes",
        ]
        if column in selector_rows.columns
    ]
    if not selector_rows.empty and "relevance_bucket" in selector_rows.columns:
        primary_rows = selector_rows.loc[selector_rows["relevance_bucket"] != "lower"].copy()
        lower_rows = selector_rows.loc[selector_rows["relevance_bucket"] == "lower"].copy()
    else:
        primary_rows = selector_rows.copy() if not selector_rows.empty else pd.DataFrame()
        lower_rows = pd.DataFrame()
    family_table = primary_rows[ranking_columns].copy() if not primary_rows.empty else pd.DataFrame()
    lower_family_table = lower_rows[ranking_columns].copy() if not lower_rows.empty else pd.DataFrame()
    why_cards = []
    if not selector_rows.empty:
        for _, row in selector_rows.iterrows():
            strategy_family = clean_string(row.get("strategy_family"))
            strategy_label = clean_string(row.get("strategy_label") or strategy_family).replace("_", " ").title()
            jump_link = (
                f'<a class="scenario-link-chip" href="#path-contract-explorer" data-path-family-jump="{escape(strategy_family)}">Explore {escape(strategy_label)} contracts</a>'
                if strategy_family and bool(row.get("available", True))
                else ""
            )
            why_cards.append(
                f'<article class="decision-hint-card" data-strategy-family-card="{escape(strategy_family)}">'
                f'<div class="decision-hint-label">{escape(strategy_label)}</div>'
                f'<div class="decision-hint-detail"><strong>Fit:</strong> {escape(clean_string(row.get("relevance_label")) or clean_string(row.get("relevance_note")) or "")}</div>'
                f'<div class="decision-hint-detail"><strong>Wins:</strong> {escape(clean_string(row.get("why_this_wins")) or clean_string(row.get("role")) or "")}</div>'
                f'<div class="decision-hint-detail"><strong>Loses:</strong> {escape(clean_string(row.get("why_this_loses")) or clean_string(row.get("relevance_note")) or "")}</div>'
                f'<div class="decision-hint-detail"><strong>Vs Stock:</strong> {escape(_format_scalar(row.get("difference_vs_stock"), published=published))} modeled difference under the active target.</div>'
                + (f'<div class="decision-hint-detail"><strong>Warning:</strong> {escape(clean_string(row.get("one_line_warning")))}</div>' if clean_string(row.get("one_line_warning")) else "")
                + jump_link
                + "</article>"
            )
    assumption_notes = selector_context.get("notes") or []
    assumption_block = (
        '<section class="panel"><h3>How To Think About Assumptions</h3><ul class="bullet-list">'
        + "".join(f"<li>{escape(clean_string(note))}</li>" for note in assumption_notes if clean_string(note))
        + "</ul></section>"
        if assumption_notes
        else ""
    )
    return (
        '<div data-strategy-selector-root>'
        + '<section class="panel"><h3>What This Is</h3><p class="section-intro">Strategy Selector is the family-choice layer. It ranks stock, calls, spreads, puts, covered calls, and cash-secured puts under your current thesis. It is not another contract table. Use it to choose the right trade family first, then move into Path &amp; Contract Explorer for the exact strike and expiry choice.</p></section>'
        + (f'<section class="panel sticky-summary-strip">{selector_strip}</section>' if selector_strip else "")
        + '<section class="panel"><h3>Best Given Your Assumptions</h3><p class="section-intro">These are family-level calls only. Each card is conditional, heuristic, and tied to the active target, timing, IV path, and budget assumptions.</p>'
        + _render_strategy_selector_rank_cards(selector_cards, published=published)
        + "</section>"
        + _render_inline_dataframe("Strategy Ranking Table", family_table, table_id="strategy-selector-table", published=published)
        + (
            '<details class="panel"><summary><strong>Lower-Relevance Families</strong> <span class="path-explorer-note">Defensive or lower-priority families under the active thesis.</span></summary>'
            + _render_inline_dataframe("Lower-Relevance Families", lower_family_table, table_id="strategy-selector-lower-relevance", published=published)
            + "</details>"
            if not lower_family_table.empty
            else ""
        )
        + ('<section class="panel"><h3>Why This Wins / Loses</h3><div class="decision-hint-grid">' + "".join(why_cards) + "</div></section>" if why_cards else "")
        + assumption_block
        + '<section class="panel"><h3>Inputs Summary</h3>'
        + inputs
        + "</section>"
        + '<section class="panel"><h3>Next Step</h3><p class="section-intro">Use Strategy Selector to choose the family, then switch to Path &amp; Contract Explorer for the exact strike and expiry. Compare the winner against long stock before committing to premium decay or capped upside trade-offs.</p><p><a class="scenario-link-chip" href="#path-contract-explorer">Open Path &amp; Contract Explorer</a> <a class="scenario-link-chip" href="#compare-vs-stock">Jump To Compare Vs Stock</a> <a class="scenario-link-chip" href="#strategy-deep-dives">Open Strategy Deep Dives</a></p></section>'
        + "</div>"
    )


def _render_contract_selection_run_history(
    runs: list[tuple[Path, dict[str, Any]]],
    *,
    current_run_slug: str,
    published: bool = False,
) -> str:
    if not runs:
        return ""
    rows: list[dict[str, Any]] = []
    for run_dir, metadata in runs:
        payload = _contract_selection_payload(metadata)
        meta = metadata.get("metadata") or {}
        rows.append(
            {
                "run_slug": payload.get("run_slug") or meta.get("run_slug") or run_dir.name,
                "generated_at": metadata.get("generated_at"),
                "goal": payload.get("goal") or meta.get("goal"),
                "stock_path": payload.get("stock_path_name") or meta.get("stock_path_name"),
                "iv_path": payload.get("iv_path_name") or meta.get("iv_path_name"),
                "target_price": payload.get("target_price") or meta.get("target_price"),
                "target_date": payload.get("target_date") or meta.get("target_date"),
                "is_current": (payload.get("run_slug") or meta.get("run_slug") or run_dir.name) == current_run_slug,
                "compatible_for_embedding": bool(metadata.get("_compatible_for_embedding")),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return ""
    return _render_inline_dataframe(
        "Latest Explorer Runs",
        frame,
        table_id="contract-selection-run-history",
        published=published,
    )


_CANONICAL_HORIZON_DAY_ORDER = {
    "entry": 0,
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "target": 365,
    "expiry": 9999,
}


def _horizon_display_label(label: str, *, target_horizon_label: str = "") -> str:
    cleaned = clean_string(label).lower()
    target_clean = clean_string(target_horizon_label).lower()
    special = {
        "entry": "Entry",
        "1w": "1W",
        "1m": "1M",
        "3m": "3M",
        "6m": "6M",
        "target": "Target",
        "expiry": "Expiry",
    }
    if cleaned in special:
        return special[cleaned]
    if cleaned and target_clean and cleaned == target_clean:
        return f"{clean_string(target_horizon_label)} target"
    return clean_string(label)


def _dynamic_horizon_orders(*frames: pd.DataFrame, target_horizon_label: str = "") -> dict[str, Any]:
    discovered: dict[str, int] = {}
    for frame in frames:
        if frame.empty or "horizon" not in frame.columns:
            continue
        requested_series = frame.get("requested_days", pd.Series([None] * len(frame)))
        for horizon, requested_days in zip(frame["horizon"].tolist(), requested_series.tolist()):
            label = clean_string(horizon).lower()
            if not label:
                continue
            fallback_days = _CANONICAL_HORIZON_DAY_ORDER.get(label, 365)
            try:
                requested = int(float(requested_days)) if requested_days not in {None, ""} and not pd.isna(requested_days) else fallback_days
            except (TypeError, ValueError):
                requested = fallback_days
            current = discovered.get(label)
            if current is None or requested < current:
                discovered[label] = requested
    if not discovered:
        fallback = ["entry", "1w", "1m", "3m", "6m"]
        target_clean = clean_string(target_horizon_label).lower()
        if target_clean and target_clean not in fallback:
            fallback.append(target_clean)
        discovered = {label: _CANONICAL_HORIZON_DAY_ORDER.get(label, 365) for label in fallback}
    ordered = sorted(
        discovered.items(),
        key=lambda item: (
            item[1],
            _CANONICAL_HORIZON_DAY_ORDER.get(item[0], item[1]),
            item[0],
        ),
    )
    horizons_x = [label for label, _ in ordered]
    return {
        "horizons_x": horizons_x,
        "horizons_y": list(reversed(horizons_x)),
        "iv_cases_y": ["iv_down", "iv_unchanged", "iv_up"],
        "target_horizon_label": clean_string(target_horizon_label),
        "horizon_display": {
            label: _horizon_display_label(label, target_horizon_label=target_horizon_label)
            for label in horizons_x
        },
    }


def _path_contract_json_payload(
    metadata: dict[str, Any],
    *,
    candidate_summary: pd.DataFrame,
    strategy_selector_rows: pd.DataFrame,
    required_path_rows: pd.DataFrame,
) -> str:
    payload = _contract_selection_payload(metadata)
    target_horizon_label = clean_string(payload.get("target_horizon") or metadata.get("target_horizon"))
    selector_defaults = payload.get("strategy_selector_defaults") or {}
    path_defaults = payload.get("path_explorer_defaults") or {}
    default_family = clean_string(
        path_defaults.get("default_strategy_family")
        or selector_defaults.get("default_strategy_family")
        or payload.get("default_strategy_family")
    )
    default_candidate = clean_string(
        path_defaults.get("default_contract_for_path_explorer")
        or payload.get("default_contract_for_path_explorer")
        or payload.get("default_candidate_within_family")
    )
    if not default_candidate and default_family and not candidate_summary.empty and "strategy_family" in candidate_summary.columns:
        within_family = candidate_summary.loc[candidate_summary["strategy_family"] == default_family]
        if not within_family.empty:
            default_candidate = clean_string(within_family.iloc[0].get("candidate_slug"))
    defaults = {
        "candidate_slug": default_candidate
        or clean_string(candidate_summary.get("candidate_slug", pd.Series(dtype=str)).iloc[0] if not candidate_summary.empty else ""),
        "goal": clean_string(payload.get("goal") or "break_even"),
        "family": default_family
        or clean_string(strategy_selector_rows.get("strategy_family", pd.Series(dtype=str)).iloc[0] if not strategy_selector_rows.empty else ""),
    }
    candidates = []
    if not candidate_summary.empty:
        seen: set[str] = set()
        families: list[dict[str, str]] = []
        seen_families: set[str] = set()
        for _, row in candidate_summary.iterrows():
            slug = clean_string(row.get("candidate_slug"))
            if not slug or slug in seen:
                continue
            seen.add(slug)
            family = clean_string(row.get("strategy_family"))
            if family and family not in seen_families:
                seen_families.add(family)
                families.append({"strategy_family": family, "strategy_label": family.replace("_", " ").title()})
            candidates.append(
                {
                    "candidate_slug": slug,
                    "candidate_label": clean_string(row.get("candidate_label")),
                    "strategy_family": clean_string(row.get("strategy_family")),
                    "strategy_label": clean_string(row.get("strategy_family")).replace("_", " ").title(),
                    "expiry_date": clean_string(row.get("expiry_date")),
                    "strike_label": clean_string(row.get("strike_label")),
                    "selection_scope_label": clean_string(row.get("selection_scope_label")),
                    "coverage_flags": clean_string(row.get("coverage_flags")),
                    "confidence_label": clean_string(row.get("confidence_label")),
                    "horizon_fit_label": clean_string(row.get("horizon_fit_label")),
                    "target_beyond_expiry": bool(row.get("target_beyond_expiry")),
                    "premium_or_entry_cost": row.get("premium_or_entry_cost"),
                    "break_even": row.get("break_even"),
                    "max_loss": row.get("max_loss"),
                    "max_gain": row.get("max_gain"),
                    "unit_capital_required": row.get("unit_capital_required"),
                    "affordable_units": row.get("affordable_units"),
                    "estimated_value": row.get("estimated_value"),
                    "profit_loss": row.get("profit_loss"),
                    "return_on_comparison_capital": row.get("return_on_comparison_capital"),
                    "difference_vs_stock": row.get("difference_vs_stock"),
                    "iv_sensitivity_summary": clean_string(row.get("iv_sensitivity_summary")),
                    "time_sensitivity_summary": clean_string(row.get("time_sensitivity_summary")),
                    "warning_or_note": clean_string(row.get("warning_or_note")),
                }
            )
    else:
        families = []
    data = required_path_rows.copy()
    if not data.empty:
        data = data[
            [
                column
                for column in [
                    "candidate_slug",
                    "candidate_label",
                    "strategy_family",
                    "goal",
                    "iv_variant_kind",
                    "iv_variant",
                    "horizon",
                    "requested_days",
                    "required_stock_price",
                    "required_stock_price_label",
                    "unreached",
                    "clamped_to_expiry",
                    "valuation_date",
                    "iv_shift_points",
                ]
                if column in data.columns
            ]
        ].copy()
    goals = []
    if not data.empty and "goal" in data.columns:
        for item in data["goal"].astype(str).tolist():
            if item not in goals:
                goals.append(item)
    horizon_orders = _dynamic_horizon_orders(required_path_rows, target_horizon_label=target_horizon_label)
    json_payload = {
            "defaults": defaults,
            "required_path_rows": make_json_safe(data.to_dict(orient="records")),
            "goals": goals,
            "candidates": candidates,
            "families": families,
            "orders": horizon_orders,
            "target_horizon_label": horizon_orders.get("target_horizon_label"),
            "horizon_display": horizon_orders.get("horizon_display") or {},
        }
    json_text = json.dumps(make_json_safe(json_payload), ensure_ascii=False, separators=(",", ":"))
    return (
        json_text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _render_path_contract_explorer_component(
    report_metadata: dict[str, Any],
    *,
    data_script_id: str,
    candidate_summary: pd.DataFrame,
    path_case_summary: pd.DataFrame,
    calibration_context: dict[str, Any],
    published: bool = False,
) -> str:
    payload = _contract_selection_payload(report_metadata)
    path_defaults = payload.get("path_explorer_defaults") or {}
    default_family = clean_string(path_defaults.get("default_strategy_family") or payload.get("default_strategy_family"))
    default_contract = clean_string(path_defaults.get("default_contract_for_path_explorer") or payload.get("default_contract_for_path_explorer"))
    if default_contract and "candidate_slug" in candidate_summary.columns:
        default_candidate_row = candidate_summary.loc[candidate_summary["candidate_slug"] == default_contract].head(1)
    else:
        default_candidate_row = pd.DataFrame()
    coverage_value = (
        clean_string(default_candidate_row.iloc[0].get("selection_scope_label"))
        if not default_candidate_row.empty
        else ("Nearby snapshot fallback" if bool((payload.get("selection_scope") or {}).get("used_nearby_snapshot_fallback")) else "Exact coverage")
    )
    timing_value = (
        "Target beyond expiry"
        if not default_candidate_row.empty and bool(default_candidate_row.iloc[0].get("target_beyond_expiry"))
        else clean_string(default_candidate_row.iloc[0].get("horizon_fit_label")) if not default_candidate_row.empty else "See selected contract"
    )
    confidence_value = clean_string(default_candidate_row.iloc[0].get("confidence_label")) if not default_candidate_row.empty else "heuristic only"
    default_contract_label = clean_string(default_candidate_row.iloc[0].get("candidate_label")) if not default_candidate_row.empty else default_contract
    status_flags = []
    for label, warning in [
        (coverage_value, coverage_value.lower() != "exact coverage"),
        (timing_value, "target beyond expiry" in timing_value.lower() or "weak" in timing_value.lower() or "poor" in timing_value.lower()),
        (confidence_value or "heuristic only", (confidence_value or "").lower() not in {"", "exact coverage"}),
        ("Heuristic only", True),
    ]:
        text = clean_string(label)
        if not text:
            continue
        css = "status-chip is-warning" if warning else "status-chip"
        status_flags.append(f'<span class="{css}">{escape(text)}</span>')
    explorer_strip = _render_summary_strip(
        [
            ("Focus", "Exact contract choice"),
            ("Default Family", default_family.replace("_", " ").title()),
            ("Default Contract", default_contract_label),
            ("Coverage", coverage_value),
            ("Timing Fit", timing_value),
            ("Confidence", confidence_value or "heuristic only"),
        ],
        published=published,
    )
    candidate_table = candidate_summary[
        [
            column
            for column in [
                "candidate_label",
                "strategy_family",
                "selection_scope_label",
                "expiry_date",
                "strike_label",
                "coverage_flags",
                "confidence_label",
                "horizon_fit_label",
                "target_beyond_expiry",
                "premium_or_entry_cost",
                "break_even",
                "max_loss",
                "max_gain",
                "unit_capital_required",
                "affordable_units",
                "fully_implementable_with_budget",
                "estimated_value",
                "profit_loss",
                "return_on_comparison_capital",
                "difference_vs_stock",
                "iv_sensitivity_summary",
                "time_sensitivity_summary",
                "warning_or_note",
            ]
            if column in candidate_summary.columns
        ]
    ].head(18) if not candidate_summary.empty else pd.DataFrame()
    path_summary = path_case_summary[
        [
            column
            for column in [
                "candidate_label",
                "stock_path",
                "iv_path",
                "final_horizon",
                "final_spot_price",
                "final_profit_loss",
                "final_return_on_comparison_capital",
                "final_difference_vs_stock",
                "worst_interim_profit_loss",
            ]
            if column in path_case_summary.columns
        ]
    ].head(12) if not path_case_summary.empty else pd.DataFrame()
    calibration_lines = []
    for note in calibration_context.get("notes") or []:
        calibration_lines.append(f"<li>{escape(note)}</li>")
    rank_cards_html = _render_contract_selection_rank_cards_from_payload(payload, published=published)
    return (
        f'<div class="path-explorer-root" data-path-explorer-root data-path-data-id="{escape(data_script_id)}">'
        + (f'<section class="panel"><h3>What This Is</h3><p class="section-intro">Path &amp; Contract Explorer is the exact-contract layer. It compares strikes, expiries, and structures at the contract level, with Required Stock Path Chart as the main visual and the candidate table as the decision center.</p></section><section class="panel sticky-summary-strip">{explorer_strip}</section>' if explorer_strip else "")
        + ('<div class="status-chip-row">' + "".join(status_flags) + "</div>" if status_flags else "")
        + '<section class="panel"><h3>Controls</h3><p class="section-intro">These selectors switch between precomputed contract-level views. Nothing is re-priced in the browser.</p><div class="path-explorer-controls">'
        '<label class="forward-lab-control"><span class="forward-lab-label">Strategy Family</span><select data-path-family></select></label>'
        '<label class="forward-lab-control"><span class="forward-lab-label">Candidate</span><select data-path-candidate></select></label>'
        '<label class="forward-lab-control"><span class="forward-lab-label">Goal</span><select data-path-goal></select></label>'
        f'<div class="forward-lab-control"><span class="forward-lab-label">Comparison Capital</span><div class="scenario-link-chip is-active">${escape(_format_scalar(payload.get("comparison_capital") or report_metadata.get("comparison_capital"), published=published))} locked</div></div>'
        '</div><div class="strategy-quick-note" data-path-state-note></div></section>'
        + _render_inline_dataframe(
            "Candidate Comparison Table",
            candidate_table,
            table_id="path-contract-candidate-table",
            published=published,
        )
        + '<section class="panel"><h3>Required Stock Path Chart</h3><p class="section-intro" data-path-chart-note>This chart shows what stock level is required over time for the selected goal under different IV scenarios or IV paths.</p><div data-path-required-chart></div></section>'
        + (
            '<section class="panel"><h3>Assumption Support</h3><ul class="bullet-list">' + "".join(calibration_lines) + "</ul>"
            + _render_inline_dataframe(
                "Path Case Summary",
                path_summary,
                table_id="path-contract-path-summary",
                published=published,
            )
            + "</section>"
            if calibration_lines
            else _render_inline_dataframe(
                "Path Case Summary",
                path_summary,
                table_id="path-contract-path-summary",
                published=published,
            )
        )
        + '<section class="panel"><h3>Contract-Level Rankings</h3><p class="section-intro">These are exact-candidate rankings only. They are meant to complement the candidate table, not replace it.</p>'
        + rank_cards_html
        + '</section>'
        + '<section class="panel"><h3>Deep-Dive Links</h3><p class="section-intro">Use the strategy deep dives for the chosen family and then compare the resulting structure back against long stock.</p><p><a class="scenario-link-chip" href="#strategy-deep-dives">Open Strategy Deep Dives</a> <a class="scenario-link-chip" href="#compare-vs-stock">Jump To Compare Vs Stock</a></p></section>'
        + "</div>"
    )


def _path_case_summary_json_payload(
    metadata: dict[str, Any],
    *,
    chart_rows: pd.DataFrame,
    strategy_rows: pd.DataFrame,
) -> str:
    payload = _contract_selection_payload(metadata)
    target_horizon_label = clean_string(payload.get("target_horizon") or metadata.get("target_horizon"))
    path_defaults = payload.get("path_case_defaults") or {}
    selector_defaults = payload.get("strategy_selector_defaults") or {}
    default_strategy_family = clean_string(
        path_defaults.get("default_strategy_family")
        or selector_defaults.get("default_strategy_family")
        or payload.get("default_strategy_family")
    )
    display_modes = []
    for value in chart_rows.get("display_mode", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in display_modes:
            display_modes.append(value)
    iv_modes = []
    for value in chart_rows.get("iv_mode", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in iv_modes:
            iv_modes.append(value)
    iv_variants: dict[str, list[str]] = {}
    if not chart_rows.empty:
        for iv_mode, group in chart_rows.groupby("iv_mode", dropna=False):
            values: list[str] = []
            for value in group.get("iv_variant", pd.Series(dtype=str)).astype(str).tolist():
                if value and value not in values:
                    values.append(value)
            iv_variants[clean_string(iv_mode)] = values
    goals = []
    for value in chart_rows.get("goal", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in goals:
            goals.append(value)
    families = []
    for value in strategy_rows.get("strategy_family", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in families:
            families.append(value)
    horizon_orders = _dynamic_horizon_orders(chart_rows, target_horizon_label=target_horizon_label)
    json_payload = {
        "defaults": {
            "case_label": clean_string(path_defaults.get("default_case_label") or "0%"),
            "goal": clean_string(path_defaults.get("default_goal") or "break_even"),
            "display_mode": clean_string(path_defaults.get("default_display_mode") or "strategy_compare"),
            "iv_mode": clean_string(path_defaults.get("default_iv_mode") or "path_preset"),
            "iv_variant": clean_string(path_defaults.get("default_iv_variant") or ""),
            "strategy_family": default_strategy_family,
            "candidate_slug": clean_string(path_defaults.get("default_candidate_within_family") or payload.get("default_candidate_within_family")),
        },
        "case_definitions": make_json_safe(payload.get("path_case_cases") or {}),
        "chart_rows": make_json_safe(chart_rows.to_dict(orient="records")) if not chart_rows.empty else [],
        "strategy_rows": make_json_safe(strategy_rows.to_dict(orient="records")) if not strategy_rows.empty else [],
        "goals": goals,
        "display_modes": display_modes or ["strategy_compare", "iv_compare"],
        "iv_modes": iv_modes or ["point_scenario", "path_preset"],
        "iv_variants": iv_variants,
        "families": families,
        "orders": horizon_orders,
        "horizon_order": horizon_orders.get("horizons_x") or ["entry", "1w", "1m", "3m", "6m", "target"],
        "target_horizon_label": horizon_orders.get("target_horizon_label"),
        "horizon_display": horizon_orders.get("horizon_display") or {},
    }
    json_text = json.dumps(make_json_safe(json_payload), ensure_ascii=False, separators=(",", ":"))
    return (
        json_text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _render_path_case_summary_component(
    report_metadata: dict[str, Any],
    *,
    data_script_id: str,
    chart_rows: pd.DataFrame,
    strategy_rows: pd.DataFrame,
    published: bool = False,
) -> str:
    payload = _contract_selection_payload(report_metadata)
    defaults = payload.get("path_case_defaults") or {}
    case_labels = ["-20%", "-10%", "0%", "+10%", "+20%"]
    pills = []
    default_case = clean_string(defaults.get("default_case_label") or "0%")
    for label in case_labels:
        active = label == default_case
        pills.append(
            f'<button type="button" class="path-case-pill{" is-active" if active else ""}" data-path-case-case="{escape(label)}" aria-pressed="{"true" if active else "false"}">{escape(label)}</button>'
        )
    goal_values = []
    for value in chart_rows.get("goal", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in goal_values:
            goal_values.append(value)
    iv_modes = []
    for value in chart_rows.get("iv_mode", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in iv_modes:
            iv_modes.append(value)
    families = []
    for value in strategy_rows.get("strategy_family", pd.Series(dtype=str)).astype(str).tolist():
        if value and value not in families:
            families.append(value)
    return (
        f'<div class="path-case-root" data-path-case-root data-path-case-data-id="{escape(data_script_id)}">'
        + '<section class="panel"><h3>What To Look At</h3><p class="section-intro">Use Strategy Selector to choose the family first. Then use Path Case Summary to see whether your thesis path actually clears that family&#39;s required path. The thick line is your Assumed Path, thinner colored lines are Required Path thresholds, and the ranked table below shows the current Case Outcome under the selected percent case and IV assumption.</p></section>'
        + '<section class="panel"><h3>Case Controls</h3><div class="path-case-controls">'
        + f'<div class="path-case-pill-row">{"".join(pills)}</div>'
        + '<div class="path-case-control-grid">'
        + '<label class="forward-lab-control"><span class="forward-lab-label">Goal</span><select class="forward-lab-select" data-path-case-goal>'
        + "".join(
            f'<option value="{escape(value)}"{" selected" if value == clean_string(defaults.get("default_goal") or "break_even") else ""}>{escape(value.replace("_", " ").title())}</option>'
            for value in goal_values
        )
        + '</select></label>'
        + '<label class="forward-lab-control"><span class="forward-lab-label">Display Mode</span><select class="forward-lab-select" data-path-case-display-mode>'
        + '<option value="strategy_compare">Strategies</option><option value="iv_compare">IV Paths</option>'
        + '</select></label>'
        + '<label class="forward-lab-control"><span class="forward-lab-label">IV Mode</span><select class="forward-lab-select" data-path-case-iv-mode>'
        + "".join(
            f'<option value="{escape(value)}">{"Point Scenario" if value == "point_scenario" else "IV Path"}</option>'
            for value in (iv_modes or ["point_scenario", "path_preset"])
        )
        + '</select></label>'
        + '<label class="forward-lab-control"><span class="forward-lab-label">IV Variant</span><select class="forward-lab-select" data-path-case-iv-variant></select></label>'
        + '<label class="forward-lab-control"><span class="forward-lab-label">Strategy Family</span><select class="forward-lab-select" data-path-case-family>'
        + "".join(
            f'<option value="{escape(value)}">{escape(value.replace("_", " ").title())}</option>'
            for value in families
        )
        + '</select></label>'
        + '</div><div class="strategy-quick-note" data-path-case-state-note></div></div></section>'
        + '<section class="panel"><h3>Required Stock Path by Strategy</h3><p class="section-intro" data-path-case-note>The thick line is your configured assumed stock path. If it stays above a strategy&#39;s required path, that strategy clears the selected goal more easily under the current IV assumptions.</p><div data-path-case-chart></div><p class="path-explorer-note" data-path-case-mode-note></p></section>'
        + '<section class="panel"><h3>Case Assumptions</h3><div class="path-case-assumption-strip" data-path-case-assumptions></div></section>'
        + '<section class="panel"><h3>Ranked Case Table</h3><p class="section-intro" data-path-case-table-note>This table shows the current Case Outcome under the selected percent path case. It is sorted by the precomputed case ranking and keeps long stock visible as the baseline.</p><div data-path-case-table></div></section>'
        + '<section class="panel"><h3>Supporting Mini-Visuals</h3><div class="path-case-mini-grid"><article class="path-case-mini-card"><h4>Bridge Layer</h4><p class="section-intro">Strategy Selector chooses the family. Path Case Summary checks whether your thesis path clears that family&#39;s required path. Path &amp; Contract Explorer then chooses the exact strike and expiry.</p></article><article class="path-case-mini-card"><h4>How IV Changes The Read</h4><p class="section-intro">Switch to IV Paths mode to hold one family fixed and see whether a falling or rising IV path makes the required stock path easier or harder.</p></article></div></section>'
        + "</div>"
    )


def _render_contract_selection_body(
    report_dir: Path,
    artifact_dir: Path,
    output_path: Path,
    report_metadata: dict[str, Any],
    summary_df: pd.DataFrame,
    *,
    embed_images: bool,
    published: bool,
) -> tuple[str, str, str]:
    summary_df = summary_df if summary_df is not None else pd.DataFrame()
    base_dir = artifact_dir
    payload = _contract_selection_payload(report_metadata)
    metadata = report_metadata.get("metadata") or {}
    bundle_context = report_metadata.get("bundle_publish_context") or {}
    warnings = _dedupe_warnings(report_metadata)
    summary_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    candidate_summary = _load_csv(artifact_dir / "candidate_summary.csv")
    candidate_comparison = _load_csv(artifact_dir / "candidate_comparison.csv")
    family_comparison = _load_csv(artifact_dir / "family_comparison.csv")
    ranked = _load_csv(artifact_dir / "ranked_candidates.csv")
    strategy_selector_rows = _load_csv(artifact_dir / "strategy_selector_rows.csv")
    chain_source_summary = _load_csv(artifact_dir / "chain_source_summary.csv")
    market_context_summary = _load_csv(artifact_dir / "market_context_summary.csv")
    required_vs_assumed = _load_csv(artifact_dir / "required_vs_assumed_path_summary.csv")
    representative_paths_summary = _load_csv(artifact_dir / "representative_paths_summary.csv")
    path_pair_summary = _load_csv(artifact_dir / "path_pair_summary.csv")
    option_value_over_path = _load_csv(artifact_dir / "option_value_over_path.csv")
    compare_vs_stock_over_path = _load_csv(artifact_dir / "compare_vs_stock_over_path.csv")
    strike_comparison_under_path = _load_csv(artifact_dir / "strike_comparison_under_path.csv")
    expiry_comparison_under_path = _load_csv(artifact_dir / "expiry_comparison_under_path.csv")
    path_risk_summary = _load_csv(artifact_dir / "path_risk_summary.csv")
    stock_path_library = _load_csv(artifact_dir / "stock_path_library.csv")
    required_path_summary = _load_csv(artifact_dir / "required_path_summary.csv")
    required_paths_by_option = _load_csv(artifact_dir / "required_paths_by_option.csv")
    required_path_family_summary = _load_csv(artifact_dir / "required_path_family_summary.csv")
    required_path_peak_summary = _load_csv(artifact_dir / "required_path_peak_summary.csv")
    required_path_exit_ladder = _load_csv(artifact_dir / "required_path_exit_ladder.csv")
    required_path_entry_sensitivity = _load_csv(artifact_dir / "required_path_entry_sensitivity.csv")
    required_path_iv_sensitivity = _load_csv(artifact_dir / "required_path_iv_sensitivity.csv")
    required_path_entry_iv_matrix = _load_csv(artifact_dir / "required_path_entry_iv_matrix.csv")
    required_path_sell_hold_summary = _load_csv(artifact_dir / "required_path_sell_hold_summary.csv")
    required_path_markdown = _load_markdown(artifact_dir / "required_path_summary.md")
    required_path_exit_ladder_markdown = _load_markdown(artifact_dir / "required_path_exit_ladder.md")
    required_path_tables_markdown = _load_markdown(artifact_dir / "required_path_tables.md")
    required_path_tables_html_path = artifact_dir / "required_path_tables.html"
    top_required_path_markdown = _load_markdown(artifact_dir / "top_required_path_candidates.md")
    chain_overview_summary = _load_csv(artifact_dir / "chain_overview_summary.csv")
    chain_overview_candidates = _load_csv(artifact_dir / "chain_overview_candidates.csv")
    chain_overview_markdown = _load_markdown(artifact_dir / "chain_overview.md")
    single_option_summary = _load_csv(artifact_dir / "single_option_decision_summary.csv")
    single_option_decision_paths = _load_csv(artifact_dir / "single_option_decision_path_selections.csv")
    single_option_path_outcomes = _load_csv(artifact_dir / "single_option_path_outcomes.csv")
    single_option_required_edge_1_5x = _load_csv(artifact_dir / "single_option_required_path_to_beat_stock_1_5x.csv")
    single_option_required_edge_2_0x = _load_csv(artifact_dir / "single_option_required_path_to_beat_stock_2_0x.csv")
    single_option_closest_edge = _load_csv(artifact_dir / "single_option_closest_representative_path_to_edge.csv")
    single_option_edge_gap_by_family = _load_csv(artifact_dir / "single_option_edge_gap_by_path_family.csv")
    single_option_iv_sensitivity = _load_csv(artifact_dir / "single_option_iv_sensitivity.csv")
    single_option_entry_sensitivity = _load_csv(artifact_dir / "single_option_entry_sensitivity.csv")
    single_option_markdown = _load_markdown(artifact_dir / "single_option_decision.md")
    images = _discover_images(artifact_dir, report_metadata)

    def chart_by_name(filename: str) -> Path | None:
        for path in images:
            if path.name == filename:
                return path
        return None

    def charts_by_prefix(prefix: str) -> list[str]:
        return sorted(path.name for path in images if path.name.startswith(prefix))

    def subset_frame(frame: pd.DataFrame, columns: list[str], *, limit: int | None = None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        selected = frame[[column for column in columns if column in frame.columns]].copy()
        return selected.head(limit).reset_index(drop=True) if limit is not None else selected.reset_index(drop=True)

    def terminal_rows(frame: pd.DataFrame, *, limit: int = 18) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        working = frame.copy()
        for column in ["requested_days", "step_index"]:
            if column in working.columns:
                working[column] = pd.to_numeric(working[column], errors="coerce")
        sort_columns = [column for column in ["path_pair_id", "candidate_slug", "requested_days", "step_index"] if column in working.columns]
        if sort_columns:
            working = working.sort_values(sort_columns, na_position="last", kind="mergesort")
        if {"path_pair_id", "candidate_slug"} <= set(working.columns):
            working = working.groupby(["path_pair_id", "candidate_slug"], dropna=False, as_index=False).tail(1)
        return working.head(limit).reset_index(drop=True)

    def render_chart_section(title: str, intro: str, filenames: list[str]) -> str:
        figures = []
        for index, name in enumerate(filenames):
            path = chart_by_name(name)
            if path is None:
                continue
            figures.append(
                _render_lightbox_figure(
                    src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
                    caption=_slug_title(path.stem),
                    featured=index == 0,
                )
            )
        if not figures:
            return (
                f'<section class="panel"><h2>{escape(title)}</h2>'
                f'<p class="section-intro">{escape(intro)}</p>'
                '<p class="empty-state">No saved chart was available for this section.</p></section>'
            )
        return (
            f'<section class="panel"><h2>{escape(title)}</h2>'
            f'<p class="section-intro">{escape(intro)}</p>'
            '<div class="lead-chart-grid">'
            + "".join(figures)
            + "</div></section>"
        )

    def render_markdown_panel(title: str, markdown_text: str) -> str:
        text = clean_string(markdown_text)
        if not text:
            return ""
        text = _sanitize_display_text(text, published=published)
        return (
            f'<section class="panel"><h2>{escape(title)}</h2>'
            '<p class="section-intro">Frozen Markdown generated upstream in analysis.</p>'
            f'<pre class="raw-notes">{escape(text)}</pre></section>'
        )

    def render_required_path_tables_workbook() -> str:
        if not required_path_tables_html_path.exists():
            return ""
        href = Path(_relative_href(required_path_tables_html_path, base_dir)).as_posix()
        return (
            '<section class="panel"><h2>Required Path Tables</h2>'
            '<p class="section-intro">Spreadsheet-style frozen workbook for required moves, entry premium sensitivity, IV sensitivity, sell/hold pressure, and absolute option-return exit ladders.</p>'
            f'<p><a class="scenario-link-chip" href="{escape(href)}">Open required_path_tables.html</a></p>'
            f'<iframe title="Required Path Tables" src="{escape(href)}" style="width:100%; min-height:760px; border:1px solid #c8d7e6; border-radius:6px; background:#fff;"></iframe>'
            "</section>"
        )

    def render_chain_overview_cards(frame: pd.DataFrame) -> str:
        if frame is None or frame.empty:
            return ""
        blocks: list[str] = []
        for row in frame.to_dict("records"):
            blocks.append(
                '<section class="summary-block">'
                f"<h3>{escape(clean_string(row.get('card_label')) or 'Compare Options')}</h3>"
                + _render_key_value_rows(
                    [
                        ("Contract", clean_string(row.get("contract_label")) or "No clear call"),
                        ("Verdict", clean_string(row.get("verdict_badge")) or "n/a"),
                        ("Headline", clean_string(row.get("headline_metric")) or "n/a"),
                        ("Why", clean_string(row.get("headline_note")) or clean_string(row.get("explanation_short")) or "n/a"),
                    ],
                    published=published,
                )
                + "</section>"
            )
        return (
            '<section class="panel"><h2>Chain Overview Summary Cards</h2>'
            '<p class="section-intro">These six frozen cards keep the compare-options layer fast to scan before you open the deeper single-option or path-pack views.</p>'
            '<div class="summary-section-grid">'
            + "".join(blocks)
            + "</div></section>"
        )

    top_candidate_slugs: list[str] = []
    if ranked is not None and not ranked.empty and "candidate_slug" in ranked.columns:
        top_candidate_slugs = ranked["candidate_slug"].astype(str).drop_duplicates().tolist()[:6]
    if not top_candidate_slugs and candidate_summary is not None and not candidate_summary.empty and "candidate_slug" in candidate_summary.columns:
        top_candidate_slugs = candidate_summary["candidate_slug"].astype(str).drop_duplicates().tolist()[:6]

    option_value_preview = option_value_over_path.copy() if option_value_over_path is not None else pd.DataFrame()
    compare_vs_stock_preview = compare_vs_stock_over_path.copy() if compare_vs_stock_over_path is not None else pd.DataFrame()
    if top_candidate_slugs:
        if not option_value_preview.empty and "candidate_slug" in option_value_preview.columns:
            option_value_preview = option_value_preview.loc[option_value_preview["candidate_slug"].astype(str).isin(top_candidate_slugs)].copy()
        if not compare_vs_stock_preview.empty and "candidate_slug" in compare_vs_stock_preview.columns:
            compare_vs_stock_preview = compare_vs_stock_preview.loc[compare_vs_stock_preview["candidate_slug"].astype(str).isin(top_candidate_slugs)].copy()
    option_value_preview = terminal_rows(option_value_preview)
    compare_vs_stock_preview = terminal_rows(compare_vs_stock_preview)

    chain_source_table = subset_frame(
        chain_source_summary,
        [
            "expiry_date",
            "storage_location",
            "source_quality",
            "source_trust_label",
            "source_snapshot_date",
            "quote_usable",
            "usable_quote_coverage_pct",
            "fallback_level",
            "chosen_reason",
        ],
        limit=20,
    )
    market_context_table = subset_frame(
        market_context_summary,
        [
            "analysis_trust_level",
            "analysis_trust_note",
            "trusted_expiry_count",
            "fallback_only_expiry_count",
            "spot_price_source",
            "spot_field_used",
            "spot_price_matched_date",
            "ibkr_same_day_spot_rejected_reason",
            "risk_free_rate_source",
            "risk_free_rate_series",
            "risk_free_rate_matched_date",
        ],
        limit=1,
    )
    required_vs_assumed_table = subset_frame(
        required_vs_assumed,
        [
            "candidate_label",
            "strategy_family",
            "representative_bucket",
            "required_path_difficulty",
            "assumed_path_gap_at_target",
            "representative_path_gap_at_target",
            "representative_goal_reached",
        ],
        limit=18,
    )
    representative_table = subset_frame(
        representative_paths_summary,
        [
            "path_pair_id",
            "representative_bucket",
            "selection_reason",
            "top_candidate_success_status",
            "stock_benchmark_status",
            "terminal_stock_price",
            "terminal_iv_shift_points",
            "final_difference_vs_stock",
        ],
        limit=12,
    )
    path_pair_table = subset_frame(
        path_pair_summary,
        [
            "path_pair_id",
            "stock_path_name",
            "iv_path_name",
            "representative_bucket",
            "selection_reason",
            "goal_success_rate",
        ],
        limit=12,
    )
    option_value_table = subset_frame(
        option_value_preview,
        [
            "path_pair_id",
            "representative_bucket",
            "candidate_label",
            "strategy_family",
            "date",
            "spot_price",
            "modeled_value",
            "profit_loss",
            "difference_vs_stock",
            "success_status",
        ],
    )
    compare_vs_stock_table = subset_frame(
        compare_vs_stock_preview,
        [
            "path_pair_id",
            "representative_bucket",
            "candidate_label",
            "strategy_family",
            "date",
            "delta_profit_loss_vs_stock",
            "delta_return_pct_vs_stock",
            "benchmark_note",
        ],
    )
    strike_table = subset_frame(
        strike_comparison_under_path,
        [
            "path_pair_id",
            "representative_bucket",
            "strategy_family",
            "strike_label",
            "expiry_date",
            "best_candidate_label",
            "objective_score",
            "difference_vs_stock",
            "timing_risk",
            "iv_risk",
            "source_trust_label",
            "weak_horizon_fit",
            "target_beyond_expiry",
        ],
        limit=18,
    )
    expiry_table = subset_frame(
        expiry_comparison_under_path,
        [
            "path_pair_id",
            "representative_bucket",
            "strategy_family",
            "expiry_date",
            "strike_label",
            "best_candidate_label",
            "objective_score",
            "difference_vs_stock",
            "timing_risk",
            "iv_risk",
            "source_trust_label",
            "weak_horizon_fit",
            "target_beyond_expiry",
        ],
        limit=18,
    )
    family_preview = subset_frame(
        family_comparison,
        [
            "strategy_label",
            "current_objective_card_status",
            "winning_candidate_label",
            "timing_risk",
            "iv_risk",
            "benchmark_note",
            "why_this_wins",
        ],
        limit=8,
    )
    candidate_preview = subset_frame(
        candidate_comparison,
        [
            "candidate_label",
            "strategy_family",
            "expiry_date",
            "strike_label",
            "objective_score",
            "difference_vs_stock",
            "timing_risk",
            "iv_risk",
            "source_trust_label",
            "weak_horizon_fit",
            "target_beyond_expiry",
        ],
        limit=10,
    )
    risk_table = subset_frame(
        path_risk_summary,
        [
            "summary_scope",
            "candidate_label",
            "required_path_difficulty",
            "timing_risk",
            "iv_risk",
            "success_dependency",
            "worst_drawdown_from_peak",
        ],
        limit=12,
    )
    required_path_summary_table = subset_frame(
        required_path_summary,
        [
            "contract_label",
            "threshold_multiple",
            "strike",
            "expiry",
            "chart_horizon_date",
            "entry_premium",
            "required_terminal_stock_price",
            "required_move_pct",
            "earliest_valid_date",
            "status",
            "verdict",
            "concise_explanation",
        ],
        limit=18,
    )
    required_paths_by_option_table = subset_frame(
        required_paths_by_option,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "date",
            "time_to_expiry_days",
            "stock_price",
            "option_value",
            "intrinsic_value",
            "time_value",
            "option_return_pct",
            "stock_return_pct",
            "option_vs_stock_multiple",
            "is_checkpoint_marker",
            "is_peak_option_return",
            "clears_threshold",
            "realism_bucket",
        ],
        limit=24,
    )
    required_path_family_table = subset_frame(
        required_path_family_summary,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "min_required_move_pct",
            "median_required_move_pct",
            "earliest_clear_date",
            "latest_clear_date",
            "clears_count",
            "realism_bucket",
            "failure_driver",
            "peak_option_return_pct",
            "peak_date",
        ],
        limit=24,
    )
    required_path_peak_table = subset_frame(
        required_path_peak_summary,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "peak_date",
            "peak_option_return_pct",
            "peak_option_value",
            "stock_price_at_peak",
            "option_vs_stock_multiple_at_peak",
            "terminal_option_return_pct",
            "terminal_option_vs_stock_multiple",
        ],
        limit=24,
    )
    required_path_exit_ladder_table = subset_frame(
        required_path_exit_ladder,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "exit_return_label",
            "first_exit_date",
            "stock_price_at_exit",
            "option_return_pct_at_exit",
            "stock_return_pct_at_exit",
            "option_vs_stock_multiple_at_exit",
        ],
        limit=24,
    )
    required_path_entry_sensitivity_table = subset_frame(
        required_path_entry_sensitivity,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "entry_shift_pct",
            "adjusted_entry_premium",
            "required_terminal_stock_price",
            "required_move_pct",
            "realism_bucket",
            "verdict",
        ],
        limit=24,
    )
    required_path_iv_sensitivity_table = subset_frame(
        required_path_iv_sensitivity,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "iv_shift_vol_points",
            "adjusted_iv",
            "required_terminal_stock_price",
            "required_move_pct",
            "realism_bucket",
            "verdict",
        ],
        limit=24,
    )
    required_path_entry_iv_matrix_table = subset_frame(
        required_path_entry_iv_matrix,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "entry_shift_pct",
            "iv_shift_vol_points",
            "adjusted_entry_premium",
            "adjusted_iv",
            "required_move_pct",
            "realism_bucket",
        ],
        limit=24,
    )
    required_path_sell_hold_table = subset_frame(
        required_path_sell_hold_summary,
        [
            "contract_label",
            "threshold_multiple",
            "path_family",
            "peak_option_return_pct",
            "peak_date",
            "stock_price_at_peak",
            "option_return_2w_after_peak",
            "option_return_1m_after_peak",
            "expiry_option_return_pct",
            "decay_from_peak_to_expiry_pct",
            "interpretation",
        ],
        limit=24,
    )
    single_option_summary_table = subset_frame(
        single_option_summary,
        [
            "candidate_short_label",
            "premium_used",
            "base_iv",
            "breakeven",
            "max_loss",
            "dte",
            "exit_rule",
            "single_option_decision_status",
            "minimum_edge_stock_return_pct",
            "minimum_edge_stock_profit_floor",
            "required_winning_path_families",
        ],
        limit=1,
    )
    single_option_decision_paths_table = subset_frame(
        single_option_decision_paths,
        [
            "path_label",
            "path_family_label",
            "timing_shape",
            "outcome_label",
            "selection_score",
            "selection_reason",
            "difference_vs_stock",
            "outperformance_multiple",
        ],
        limit=8,
    )
    single_option_outcomes_table = subset_frame(
        single_option_path_outcomes,
        [
            "path_label",
            "outcome_label",
            "exit_stock_price",
            "profit_loss",
            "stock_profit_loss",
            "difference_vs_stock",
            "outperformance_multiple",
            "outcome_note",
        ],
        limit=8,
    )
    single_option_required_edge_table = subset_frame(
        pd.concat([single_option_required_edge_1_5x, single_option_required_edge_2_0x], ignore_index=True)
        if (single_option_required_edge_1_5x is not None or single_option_required_edge_2_0x is not None)
        else pd.DataFrame(),
        [
            "edge_label",
            "requested_days",
            "date",
            "required_stock_price",
            "return_pct",
            "iv_shift_points",
            "required_option_profit_loss",
            "stock_profit_loss_at_required_edge",
            "status",
        ],
        limit=12,
    )
    single_option_closest_edge_table = subset_frame(
        single_option_closest_edge,
        [
            "path_label",
            "path_family_label",
            "timing_shape",
            "outcome_label",
            "exit_stock_price",
            "required_stock_price_1_5x",
            "extra_stock_move_needed",
            "edge_gap_to_1_5x_dollars",
            "edge_failure_driver",
            "earlier_timing_needed",
            "iv_support_needed",
            "entry_discount_needed",
            "timing_gap_note",
            "annotation_text",
        ],
        limit=1,
    )
    single_option_edge_gap_table = subset_frame(
        single_option_edge_gap_by_family,
        [
            "path_label",
            "path_family_label",
            "timing_shape",
            "outcome_label",
            "exit_stock_price",
            "required_stock_price_1_5x",
            "extra_stock_move_needed",
            "edge_gap_to_1_5x_dollars",
            "edge_gap_to_1_5x_normalized",
            "edge_failure_driver",
            "timing_gap_note",
            "is_closest_to_edge",
        ],
        limit=8,
    )
    stock_path_library_table = subset_frame(
        stock_path_library,
        [
            "path_label",
            "path_family_label",
            "timing_shape",
            "outcome_bias",
            "library_role",
            "path_description",
        ],
        limit=32,
    )
    single_option_iv_table = subset_frame(
        single_option_iv_sensitivity,
        ["iv_mode_label", "iv_shift_points", "estimated_option_value", "difference_vs_stock", "sensitivity_note"],
        limit=3,
    )
    single_option_entry_table = subset_frame(
        single_option_entry_sensitivity,
        ["entry_scenario_label", "premium_used", "path_families_beating_stock", "average_difference_vs_stock", "entry_read"],
        limit=3,
    )
    chain_overview_table = subset_frame(
        chain_overview_candidates,
        [
            "contract",
            "premium",
            "iv",
            "dte",
            "beats_stock_label",
            "strong_wins",
            "robustness",
            "iv_sensitivity",
            "entry_sensitivity",
            "best_fit_path_type",
            "final_verdict",
            "why_short",
            "why_detail",
        ],
        limit=16,
    )

    preferred_contract = clean_string(payload.get("default_contract_for_path_explorer") or payload.get("default_candidate_within_family"))
    preferred_family = clean_string(payload.get("default_strategy_family"))
    best_expiry = clean_string(summary_row.get("best_expiry"))
    scenario_href_base = clean_string(bundle_context.get("scenario_href"))
    strategy_selector_href = clean_string(bundle_context.get("strategy_selector_href"))
    scenario_href = ""
    if best_expiry:
        if scenario_href_base:
            scenario_href = scenario_href_base + "#path-contract-explorer"
        elif published:
            scenario_href_base = _existing_relative_href(
                _published_scenario_candidates(output_path, best_expiry),
                output_path.parent,
            )
        if scenario_href_base:
            scenario_href = scenario_href_base + "#path-contract-explorer"
    if not strategy_selector_href:
        strategy_selector_href = scenario_href.replace("#path-contract-explorer", "#strategy-selector") if scenario_href else ""
    related_links: list[str] = []
    if strategy_selector_href:
        related_links.append(
            f'<a class="scenario-link-chip" href="{escape(_href_with_query_params(strategy_selector_href, strategy_family=preferred_family) if preferred_family else strategy_selector_href)}">Open related scenario Strategy Selector</a>'
        )
    if scenario_href:
        related_links.append(
            f'<a class="scenario-link-chip" href="{escape(_href_with_query_params(scenario_href, strategy_family=preferred_family, candidate=preferred_contract) if preferred_contract else scenario_href)}">Open related scenario contract explorer</a>'
        )

    top_strip = _render_summary_strip(
        [
            ("Ticker", payload.get("ticker") or metadata.get("ticker")),
            ("Snapshot Date", payload.get("snapshot_date") or metadata.get("snapshot_date")),
            ("Trust Level", summary_row.get("analysis_trust_level")),
            ("Spot Source", summary_row.get("spot_price_source")),
            ("Best Family", summary_row.get("best_family")),
            ("Best Candidate", summary_row.get("best_candidate")),
            ("Best Strike", summary_row.get("best_strike")),
            ("Best Expiry", summary_row.get("best_expiry")),
        ],
        published=published,
    )

    body = (
        '<div data-contract-selection-page>'
        '<section class="hero"><div class="hero-top"><div><div class="eyebrow">Contract Selection</div>'
        f'<h1>{escape(clean_string(payload.get("ticker") or metadata.get("ticker")))} Contract Selection</h1>'
        '<p class="subtitle">This page reads the frozen contract-selection bundle directly. The core question is what each long call needs from the stock path before it beats owning stock by the configured outperformance hurdles.</p>'
        '</div></div></section>'
        + _render_shareability_note(published=published, embed_images=embed_images, has_supporting_links=True)
        + (f'<section class="panel sticky-summary-strip">{top_strip}</section>' if top_strip else "")
        + '<section class="panel"><h2>Decision Snapshot</h2><p class="section-intro">Use this section for the fastest read on the active thesis, benchmark choice, and current trust balance in the bundle.</p>'
        + _render_metric_cards(
            [
                ("Best Family", summary_row.get("best_family")),
                ("Best Candidate", summary_row.get("best_candidate")),
                ("Trusted Expiries", summary_row.get("trusted_expiry_count")),
                ("Sparse/Fallback Expiries", summary_row.get("fallback_only_expiry_count")),
                ("Target Price", summary_row.get("target_price")),
                ("Benchmark Edge vs Stock", summary_row.get("benchmark_edge")),
            ],
            published=published,
        )
        + _render_key_value_rows(
            [
                ("Goal", summary_row.get("goal")),
                ("Target Date", summary_row.get("target_date")),
                ("Assumed Stock Path", summary_row.get("stock_path_name")),
                ("Assumed IV Path", summary_row.get("iv_path_name")),
                ("Stock Benchmark", summary_row.get("stock_benchmark_label")),
                ("Top Path Risk", summary_row.get("top_path_risk")),
            ],
            published=published,
        )
        + "</section>"
        + render_chart_section(
            "Required-Path Engine",
            "This is the primary product view: each long call is solved backwards from the 1.5x and 2.0x option-over-stock thresholds. Those thresholds are relative to stock return, not absolute option return.",
            ["required_paths_overview.png"],
        )
        + render_required_path_tables_workbook()
        + render_markdown_panel("Required Path Summary", required_path_markdown)
        + render_markdown_panel("Required Path Tables Notes", required_path_tables_markdown)
        + render_markdown_panel("Top Required-Path Candidates", top_required_path_markdown)
        + _render_inline_dataframe("Required Path Summary Table", required_path_summary_table, table_id="required-path-summary", published=published)
        + render_chart_section(
            "Per-Option Required Paths",
            "Each chart shows the stock path the option requires over time, plus the option-return path that the stock move implies. These charts are frozen artifacts, not dashboard recomputation.",
            [name for name in charts_by_prefix("required_paths_") if name != "required_paths_overview.png"][:6],
        )
        + _render_inline_dataframe("Required Paths By Option", required_paths_by_option_table, table_id="required-paths-by-option", published=published)
        + _render_inline_dataframe("Required Path Family Summary", required_path_family_table, table_id="required-path-family-summary", published=published)
        + _render_inline_dataframe("Required Path Peak Summary", required_path_peak_table, table_id="required-path-peak-summary", published=published)
        + _render_inline_dataframe("Required Path Entry Premium Sensitivity", required_path_entry_sensitivity_table, table_id="required-path-entry-sensitivity", published=published)
        + _render_inline_dataframe("Required Path IV Sensitivity", required_path_iv_sensitivity_table, table_id="required-path-iv-sensitivity", published=published)
        + _render_inline_dataframe("Required Path Entry x IV Matrix", required_path_entry_iv_matrix_table, table_id="required-path-entry-iv-matrix", published=published)
        + _render_inline_dataframe("Required Path Sell / Hold Summary", required_path_sell_hold_table, table_id="required-path-sell-hold-summary", published=published)
        + render_markdown_panel("Required Path Exit Ladder Notes", required_path_exit_ladder_markdown)
        + _render_inline_dataframe("Required Path Exit Ladder", required_path_exit_ladder_table, table_id="required-path-exit-ladder", published=published)
        + '<section class="panel"><h2>Chain Overview / Compare Options</h2><p class="section-intro">Supporting tables and notes only. Legacy comparison charts are intentionally omitted so the page stays centered on the required-path engine.</p></section>'
        + render_markdown_panel("Chain Overview Notes", chain_overview_markdown)
        + render_chain_overview_cards(chain_overview_summary)
        + _render_inline_dataframe("Chain Overview Candidate Table", chain_overview_table, table_id="chain-overview-candidates", published=published)
        + '<section class="panel"><h2>Market Context / Trust Summary</h2><p class="section-intro">These tables show which local market data actually drove the run, how trustworthy each expiry is, and why the chosen spot and risk-free inputs were accepted.</p>'
        + _render_key_value_rows(
            [
                ("Analysis Trust Level", summary_row.get("analysis_trust_level")),
                ("Trust Note", summary_row.get("analysis_trust_note")),
                ("Spot Source", summary_row.get("spot_price_source")),
                ("Spot Field", summary_row.get("spot_field_used")),
                ("Spot Matched Date", summary_row.get("spot_price_matched_date")),
                ("Same-Day IBKR Spot Rejection", summary_row.get("ibkr_same_day_spot_rejected_reason")),
                ("Risk-Free Source", summary_row.get("risk_free_rate_source")),
                ("Risk-Free Series", summary_row.get("risk_free_rate_series")),
                ("Risk-Free Matched Date", summary_row.get("risk_free_rate_matched_date")),
            ],
            published=published,
        )
        + _render_inline_dataframe("Expiry Trust Table", chain_source_table, table_id="contract-chain-source", published=published)
        + _render_inline_dataframe("Bundle Market Context", market_context_table, table_id="contract-market-context", published=published)
        + "</section>"
        + _render_inline_dataframe("Required vs Assumed Path Summary", required_vs_assumed_table, table_id="required-vs-assumed", published=published)
        + _render_inline_dataframe("Stock Path Library Metadata", stock_path_library_table, table_id="stock-path-library", published=published)
        + _render_inline_dataframe("Representative Paths Summary", representative_table, table_id="representative-paths", published=published)
        + _render_inline_dataframe("Path Pair Summary", path_pair_table, table_id="path-pair-summary", published=published)
        + _render_inline_dataframe("Option Value Over Path", option_value_table, table_id="option-value-over-path", published=published)
        + _render_inline_dataframe("Compare vs Stock Over Path", compare_vs_stock_table, table_id="compare-vs-stock-over-path", published=published)
        + _render_inline_dataframe("Same-Path Strike Comparison", strike_table, table_id="same-path-strike", published=published)
        + _render_inline_dataframe("Same-Path Expiry Comparison", expiry_table, table_id="same-path-expiry", published=published)
        + '<section class="panel"><h2>Family / Candidate Highlights</h2><p class="section-intro">These are supporting highlight layers. Use them after the same-path tables, not before them.</p>'
        + _render_strategy_selector_rank_cards(
            (payload.get("strategy_selector_best_cards") or []),
            published=published,
            link_base="",
            link_anchor="family-candidate-highlights",
            link_label="Supporting family highlight",
        )
        + (_render_inline_dataframe("Family Comparison", family_preview, table_id="family-comparison", published=published) if not family_preview.empty else '<p class="empty-state">No family comparison rows were available for this run.</p>')
        + (_render_inline_dataframe("Candidate Comparison", candidate_preview, table_id="candidate-comparison", published=published) if not candidate_preview.empty else '<p class="empty-state">No candidate comparison rows were available for this run.</p>')
        + "</section>"
        + '<section class="panel"><h2>Warnings / Risk Notes</h2><p class="section-intro">Read this section before treating a high-ranked candidate as fully trustworthy. Sparse expiries and target-beyond-expiry timing issues stay visible here on purpose.</p>'
        + _render_warnings(warnings)
        + (_render_inline_dataframe("Path Risk Summary", risk_table, table_id="path-risk-summary", published=published) if not risk_table.empty else "")
        + (_render_callout("Optional Related Scenario Pages", "Scenario dashboards are secondary companions now. Use them only when you want an additional snapshot-level page.") if related_links else "")
        + (f'<p>{" ".join(related_links)}</p>' if related_links else "")
        + "</section>"
        + "</div>"
    )
    return body, CONTRACT_SELECTION_HEAD, CONTRACT_SELECTION_BODY_END


def _render_replay_plot_section(
    title: str,
    filenames: list[str],
    *,
    images: list[Path],
    base_dir: Path,
    embed_images: bool,
) -> str:
    matches = [path for name in filenames for path in images if path.name == name]
    if not matches:
        return f'<section class="panel"><h3>{escape(title)}</h3><p class="empty-state">No saved chart was available for this section.</p></section>'
    figures = [
        _render_lightbox_figure(
            src=_image_src(path, base_dir=base_dir, embed_images=embed_images),
            caption=_slug_title(path.stem),
            featured=index == 0,
        )
        for index, path in enumerate(matches)
    ]
    return (
        f'<section class="panel"><h3>{escape(title)}</h3>'
        '<div class="lead-chart-grid">'
        + "".join(figures)
        + "</div></section>"
    )


def _render_historical_replay_history_body(
    report_dir: Path,
    artifact_dir: Path,
    destination: Path,
    report_metadata: dict[str, Any],
    summary_df: pd.DataFrame | None,
    *,
    embed_images: bool,
    published: bool,
) -> tuple[str, str, str]:
    summary_df = summary_df if summary_df is not None else pd.DataFrame()
    case_history = _load_csv(artifact_dir / "case_history.csv")
    expected_history = _load_csv(artifact_dir / "expected_move_history.csv")
    compare_history = _load_csv(artifact_dir / "compare_vs_stock_history.csv")
    valuation_counts = _load_csv(artifact_dir / "valuation_source_counts.csv")
    base_dir = destination.parent if published else artifact_dir
    status = _report_status(report_metadata)
    warnings = _dedupe_warnings(report_metadata)
    top_strip = _render_summary_strip(_historical_replay_context_items(report_metadata), published=published)
    shareability_note = _render_shareability_note(
        published=published,
        embed_images=embed_images,
        has_supporting_links=bool(_related_report_items(report_metadata, artifact_dir=artifact_dir) or [path for path in _artifact_files(artifact_dir, report_dir, report_metadata=report_metadata) if path.name not in HIDDEN_FILES]),
    )
    summary_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="summary">'
        '<h2>Summary</h2>'
        '<p class="section-intro">This is a local replay history rollup, not a production backtest. Use it to compare what local cases taught you about timing, expected move, and stock-versus-structure trade-offs.</p>'
        + _render_callout(
            "What To Learn",
            clean_string(report_metadata.get("what_this_case_shows"))
            or "Read valuation-source quality first, then compare expected move versus actual move and strategy versus stock across the latest local replay cases.",
        )
        + _render_summary_overview(
            report_metadata,
            _historical_replay_metric_items(report_metadata, summary_df),
            _historical_replay_context_items(report_metadata),
            summary_df,
            published=published,
        )
        + _render_inline_dataframe("Latest Replay Cases", case_history, table_id="replay-case-history", published=published)
        + _render_related_reports(report_metadata, artifact_dir=artifact_dir)
        + "</section>"
    )
    expected_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="expected-vs-actual">'
        '<h2>Expected vs Actual</h2>'
        '<p class="section-intro">Use this table to see whether realized move tended to exceed or fall short of the entry expected-move framing across local cases.</p>'
        + _render_inline_dataframe(
            "Expected Move vs Actual Across Cases",
            expected_history,
            table_id="replay-expected-history",
            published=published,
        )
        + "</section>"
    )
    drivers_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="drivers">'
        '<h2>Drivers</h2>'
        '<p class="section-intro">These counts summarize how often later valuation was exact versus modeled. Sparse exact coverage should lower confidence in any simple historical conclusion.</p>'
        + _render_inline_dataframe(
            "Valuation Source Counts",
            valuation_counts,
            table_id="replay-valuation-source-counts",
            published=published,
        )
        + "</section>"
    )
    compare_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="compare-vs-stock">'
        '<h2>Compare vs Stock</h2>'
        '<p class="section-intro">This history rollup keeps the comparison careful: did the strategy beat or lag long stock across the local cases you actually have?</p>'
        + _render_inline_dataframe(
            "Strategy vs Stock Across Cases",
            compare_history,
            table_id="replay-compare-history",
            published=published,
        )
        + "</section>"
    )
    assumptions_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="assumptions">'
        '<h2>Assumptions</h2>'
        '<p class="section-intro">Sparse local history, missing later chains, and modeled checkpoints all belong in the interpretation layer. This page is intentionally explicit about those limits.</p>'
        + _render_provenance_and_quality(report_metadata, compute_local_coverage((report_metadata.get("metadata") or {}).get("ticker"), (report_metadata.get("metadata") or {}).get("snapshot_date")), published=published)
        + _render_assumptions(report_metadata, published=published)
        + _render_raw_notes(_load_markdown(artifact_dir / "summary.md"), published=published)
        + _render_available_files(_artifact_files(artifact_dir, artifact_dir, report_metadata=report_metadata), base_dir=base_dir)
        + "</section>"
    )
    body = (
        '<div data-scenario-tabbed-page>'
        '<section class="hero">'
        '<div class="hero-top"><div>'
        '<div class="eyebrow">Historical Replay History</div>'
        f"<h1>{escape(_page_title(report_metadata))}</h1>"
        '<p class="subtitle">This page rolls up multiple local replay cases for one ticker and strategy. Treat it as a learning aid built from the cases you actually have, not a claim of repeatable historical edge.</p>'
        "</div>"
        + _status_badge(status)
        + "</div></section>"
        + shareability_note
        + (f'<section class="panel sticky-summary-strip">{top_strip}</section>' if top_strip else "")
        + _render_warnings(warnings)
        + '<section class="panel"><div class="scenario-tab-nav">'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="summary" aria-selected="true">Summary</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="expected-vs-actual" aria-selected="false">Expected vs Actual</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="drivers" aria-selected="false">Drivers</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="compare-vs-stock" aria-selected="false">Compare vs Stock</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="assumptions" aria-selected="false">Assumptions</button>'
        "</div></section>"
        + summary_section
        + expected_section
        + drivers_section
        + compare_section
        + assumptions_section
        + "</div>"
    )
    return body, SCENARIO_TABS_HEAD, SCENARIO_TABS_SCRIPT


def _render_historical_replay_body(
    report_dir: Path,
    artifact_dir: Path,
    destination: Path,
    report_metadata: dict[str, Any],
    summary_df: pd.DataFrame | None,
    *,
    embed_images: bool,
    published: bool,
) -> tuple[str, str, str]:
    metadata = report_metadata.get("metadata") or {}
    if metadata.get("history_mode"):
        return _render_historical_replay_history_body(
            report_dir,
            artifact_dir,
            destination,
            report_metadata,
            summary_df,
            embed_images=embed_images,
            published=published,
        )

    summary_df = summary_df if summary_df is not None else pd.DataFrame()
    checkpoint_replay = _load_csv(artifact_dir / "checkpoint_replay.csv")
    expected_move_vs_actual = _load_csv(artifact_dir / "expected_move_vs_actual.csv")
    driver_decomposition = _load_csv(artifact_dir / "driver_decomposition.csv")
    compare_vs_stock = _load_csv(artifact_dir / "compare_vs_stock.csv")
    local_history = _load_csv(artifact_dir / "local_history.csv")
    images = _discover_images(artifact_dir, report_metadata)
    base_dir = destination.parent if published else artifact_dir
    status = _report_status(report_metadata)
    warnings = _dedupe_warnings(report_metadata)
    payload = _historical_replay_payload(report_metadata)
    shareability_note = _render_shareability_note(
        published=published,
        embed_images=embed_images,
        has_supporting_links=bool(_related_report_items(report_metadata, artifact_dir=artifact_dir) or [path for path in _artifact_files(artifact_dir, report_dir, report_metadata=report_metadata) if path.name not in HIDDEN_FILES]),
    )
    top_strip = _render_summary_strip(_historical_replay_context_items(report_metadata), published=published)
    summary_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="summary">'
        '<h2>Summary</h2>'
        '<p class="section-intro">This is a historical replay / case-study page. It replays one strategy idea through the local data you actually have and labels whether each checkpoint came from an exact later chain, a modeled estimate, or missing data.</p>'
        + _render_callout(
            "Honesty Note",
            "This page is for learning from one local case, not for claiming a statistically robust backtest. Exact later chain values are used only when the same expiry appears in a later local snapshot. Otherwise the replay falls back to a modeled estimate and says so explicitly.",
        )
        + _render_callout(
            "What This Case Shows",
            clean_string(report_metadata.get("what_this_case_shows")) or clean_string(payload.get("what_this_case_shows")) or _what_to_look_at(report_metadata, status),
        )
        + _render_summary_overview(
            report_metadata,
            _historical_replay_metric_items(report_metadata, summary_df),
            _historical_replay_context_items(report_metadata),
            summary_df,
            published=published,
        )
        + _render_related_reports(report_metadata, artifact_dir=artifact_dir)
        + "</section>"
    )
    outcome_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="outcome-path">'
        '<h2>Outcome Path</h2>'
        '<p class="section-intro">Start here when you want to see what the stock actually did and how the position value changed along the replay checkpoints.</p>'
        + _render_replay_plot_section(
            "Stock Path And Replay Value",
            ["stock_path_expected_move.png", "strategy_value_path.png"],
            images=images,
            base_dir=base_dir,
            embed_images=embed_images,
        )
        + _render_inline_dataframe(
            "Checkpoint Replay",
            checkpoint_replay,
            table_id="checkpoint-replay",
            published=published,
        )
        + "</section>"
    )
    expected_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="expected-vs-actual">'
        '<h2>Expected vs Actual</h2>'
        '<p class="section-intro">This section compares the expected move available at entry with what the stock actually delivered by each checkpoint or event window.</p>'
        + _render_callout(
            "What To Learn",
            "If realized movement stayed inside the entry expected move, the structure may simply not have gotten enough help from direction. If realized movement beat the implied move but the trade still struggled, timing, IV, or structure may be the better explanation.",
        )
        + _render_inline_dataframe(
            "Expected Move vs Actual",
            expected_move_vs_actual,
            table_id="expected-vs-actual",
            published=published,
        )
        + "</section>"
    )
    drivers_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="drivers">'
        '<h2>Drivers</h2>'
        '<p class="section-intro">This decomposition is informative rather than exact. It shows how much of the path looked like spot move, time decay, IV change, rate change, or structure residual in the modeled sequence.</p>'
        + _render_replay_plot_section(
            "Driver Decomposition",
            ["driver_decomposition_expiry.png", "driver_decomposition_post_event.png", "driver_decomposition_event.png"],
            images=images,
            base_dir=base_dir,
            embed_images=embed_images,
        )
        + _render_inline_dataframe(
            "Driver Decomposition",
            driver_decomposition,
            table_id="driver-decomposition",
            published=published,
        )
        + "</section>"
    )
    compare_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="compare-vs-stock">'
        '<h2>Compare vs Stock</h2>'
        '<p class="section-intro">Long stock stays the benchmark. Use the normalized $1,000 view first, then the share-equivalent read as a realism check.</p>'
        + _render_callout(
            "What To Learn",
            "A strategy can still be directionally right but lag long stock if time decay, IV compression, or a capped structure gave back too much of the move.",
        )
        + _render_replay_plot_section(
            "Strategy vs Stock",
            ["strategy_vs_stock_equal_capital.png", "strategy_vs_stock_share_equivalent.png"],
            images=images,
            base_dir=base_dir,
            embed_images=embed_images,
        )
        + _render_inline_dataframe(
            "Compare vs Stock",
            compare_vs_stock,
            table_id="replay-compare-vs-stock",
            published=published,
        )
        + "</section>"
    )
    assumptions_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="assumptions">'
        '<h2>Assumptions</h2>'
        '<p class="section-intro">Resolved local inputs, matched dates, research context, and provenance notes live here so the replay conclusions above stay honest.</p>'
        + _render_provenance_and_quality(
            report_metadata,
            compute_local_coverage((metadata or {}).get("ticker"), (metadata or {}).get("snapshot_date")),
            published=published,
        )
        + _render_assumptions(report_metadata, published=published)
        + _render_research_context(report_metadata, published=published)
        + _render_raw_notes(_load_markdown(artifact_dir / "summary.md"), published=published)
        + _render_available_files(_artifact_files(artifact_dir, report_dir, report_metadata=report_metadata), base_dir=base_dir)
        + "</section>"
    )
    local_history_section = (
        '<section class="panel scenario-tab-panel" data-scenario-tab-panel="local-history">'
        '<h2>Local History</h2>'
        '<p class="section-intro">When multiple local replay cases exist for the same strategy, this table lets you compare them latest first without overstating the completeness of the history.</p>'
        + _render_inline_dataframe(
            "Local Replay History",
            local_history,
            table_id="replay-local-history",
            published=published,
        )
        + "</section>"
    )
    body = (
        '<div data-scenario-tabbed-page>'
        '<section class="hero">'
        '<div class="hero-top"><div>'
        '<div class="eyebrow">Historical Replay / Case Study</div>'
        f"<h1>{escape(_page_title(report_metadata))}</h1>"
        '<p class="subtitle">This page replays one strategy idea using local historical prices, local chain coverage when it exists, and explicit modeled fallbacks when it does not. It is built to teach what actually drove the result, not to claim false certainty.</p>'
        "</div>"
        + _status_badge(status)
        + "</div></section>"
        + shareability_note
        + (f'<section class="panel sticky-summary-strip">{top_strip}</section>' if top_strip else "")
        + _render_warnings(warnings)
        + '<section class="panel"><div class="scenario-tab-nav">'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="summary" aria-selected="true">Summary</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="outcome-path" aria-selected="false">Outcome Path</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="expected-vs-actual" aria-selected="false">Expected vs Actual</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="drivers" aria-selected="false">Drivers</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="compare-vs-stock" aria-selected="false">Compare vs Stock</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="assumptions" aria-selected="false">Assumptions</button>'
        '<button type="button" class="scenario-tab-button" data-scenario-tab-target="local-history" aria-selected="false">Local History</button>'
        "</div></section>"
        + summary_section
        + outcome_section
        + expected_section
        + drivers_section
        + compare_section
        + assumptions_section
        + local_history_section
        + "</div>"
    )
    return body, SCENARIO_TABS_HEAD, SCENARIO_TABS_SCRIPT


def _render_dashboard_body(
    report_dir: Path,
    artifact_dir: Path,
    destination: Path,
    report_metadata: dict[str, Any],
    summary_df: pd.DataFrame | None,
    *,
    embed_images: bool,
    published: bool,
) -> tuple[str, str, str]:
    if _report_kind(report_metadata) == "contract_selection":
        return _render_contract_selection_body(
            report_dir,
            artifact_dir,
            destination,
            report_metadata,
            summary_df,
            embed_images=embed_images,
            published=published,
        )
    if _report_kind(report_metadata) == "scenario":
        return _render_scenario_body(
            report_dir,
            artifact_dir,
            destination,
            report_metadata,
            summary_df,
            embed_images=embed_images,
            published=published,
        )
    if _report_kind(report_metadata) == "replay":
        return _render_historical_replay_body(
            report_dir,
            artifact_dir,
            destination,
            report_metadata,
            summary_df,
            embed_images=embed_images,
            published=published,
        )
    title = _page_title(report_metadata)
    summary_df = summary_df if summary_df is not None else pd.DataFrame()
    status = _report_status(report_metadata)
    warnings = _dedupe_warnings(report_metadata)
    top_items = _top_summary_items(report_metadata)
    metric_items = _strategy_metric_items(report_metadata) if _report_kind(report_metadata) == "strategy" else _generic_analysis_metric_items(report_metadata, summary_df)
    images = _discover_images(artifact_dir, report_metadata)
    summary_md = _load_markdown(artifact_dir / "summary.md")
    coverage = compute_local_coverage(
        (_strategy_context_items(report_metadata) if _report_kind(report_metadata) == "strategy" else _generic_analysis_context_items(report_metadata))[0][1] if top_items else None,
        (report_metadata.get("strategy_report") or {}).get("snapshot_date") or (report_metadata.get("metadata") or {}).get("snapshot_date"),
    )
    base_dir = destination.parent if published else artifact_dir
    related_items = _related_report_items(report_metadata, artifact_dir=artifact_dir) if published else []
    shareability_note = _render_shareability_note(
        published=published,
        embed_images=embed_images,
        has_supporting_links=bool(related_items or [path for path in _artifact_files(artifact_dir, report_dir, report_metadata=report_metadata) if path.name not in HIDDEN_FILES]),
    )
    summary_strip = _render_summary_strip(top_items[:8], published=published)
    hero = (
        '<section class="hero">'
        '<div class="hero-top">'
        '<div>'
        f'<div class="eyebrow">{escape(_page_role_label(report_metadata))}</div>'
        f"<h1>{escape(title)}</h1>"
        f'<p class="subtitle">Static local report dashboard built from saved artifacts in {escape(_sanitize_display_text(str(report_dir.name), published=published))}. Open the Snapshot Hub first when you want the clearest overview for one date, then use this page for focused drill-down.</p>'
        "</div>"
        + _status_badge(status)
        + "</div>"
        + "</section>"
    )
    sections = [
        hero,
        shareability_note,
    ]
    if summary_strip:
        sections.append(f'<section class="panel sticky-summary-strip">{summary_strip}</section>')
    if metric_items:
        sections.append(
            '<section class="panel"><h2>Key Metrics</h2>'
            '<p class="section-intro">This compact block surfaces the most decision-useful metrics first.</p>'
            + _render_metric_cards(metric_items, published=published)
            + "</section>"
        )
    sections.append(_render_plot_gallery(images, base_dir=base_dir, embed_images=embed_images))
    sections.append(_render_callout("What To Look At", _what_to_look_at(report_metadata, status)))
    sections.append(_render_related_reports(report_metadata, artifact_dir=artifact_dir))
    sections.append(_render_warnings(warnings))
    sections.append(_render_summary_overview(report_metadata, metric_items, top_items, summary_df, published=published))
    sections.append(_render_leg_table(report_metadata, published=published))
    if _report_kind(report_metadata) == "strategy":
        sections.append(_render_strategy_tables(artifact_dir, published=published))
    else:
        sections.append(_render_generic_analysis_tables(artifact_dir, report_metadata=report_metadata, published=published))
    sections.append(_render_provenance_and_quality(report_metadata, coverage, published=published))
    sections.append(_render_assumptions(report_metadata, published=published))
    sections.append(_render_research_context(report_metadata, published=published))
    sections.append(_render_raw_notes(summary_md, published=published))
    sections.append(_render_available_files(_artifact_files(artifact_dir, report_dir, report_metadata=report_metadata), base_dir=base_dir))
    return "\n".join(section for section in sections if clean_string(section)), "", ""


def generate_dashboard(
    report_dir: str | Path,
    *,
    destination: str | Path | None = None,
    embed_images: bool = False,
    published: bool = False,
) -> Path:
    """Generate one static dashboard for an existing saved report folder."""

    source_dir = Path(report_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Report directory not found: {source_dir}")
    artifact_dir = Path(destination).parent if destination is not None and published else source_dir
    metadata_path = artifact_dir / "report_metadata.json"
    if not metadata_path.exists():
        metadata_path = source_dir / "report_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing report metadata: {metadata_path}")
    report_metadata = _load_json(metadata_path)
    output_path = Path(destination) if destination is not None else source_dir / "dashboard.html"
    ensure_directory(output_path.parent)
    summary_df = _load_csv(artifact_dir / "summary.csv")
    body, extra_head, extra_body_end = _render_dashboard_body(
        source_dir,
        artifact_dir,
        output_path,
        report_metadata,
        summary_df,
        embed_images=embed_images,
        published=published,
    )
    output_path.write_text(
        render_html_document(
            _page_title(report_metadata),
            body,
            extra_head=extra_head,
            extra_body_end=extra_body_end,
        ),
        encoding="utf-8",
    )
    return output_path
