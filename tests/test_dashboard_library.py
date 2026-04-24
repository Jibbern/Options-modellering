from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_lab.analysis import (
    build_contract_selection_analysis,
    build_replay_analysis,
    build_scenario_analysis,
    publish_analysis_bundle,
    write_analysis_bundle,
)
from options_lab.publish import mirror_published_bundle, rebuild_dashboard_library


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.mark.slow
def test_dashboards_library_is_rebuilt_only_from_bundle_publishes(temp_workspace_root: Path):
    analysis_root = temp_workspace_root / "analysis_outputs"
    dashboards_root = temp_workspace_root / "Dashboards"

    scenario_bundle = write_analysis_bundle(
        build_scenario_analysis(
            ticker="GPRE",
            snapshot_date="2026-04-12",
            expiry_date="2026-04-17",
            data_root=DATA_ROOT,
        ),
        analysis_kind="scenario",
        output_root=analysis_root,
    )
    publish_analysis_bundle(scenario_bundle.bundle_dir)
    mirror_published_bundle(scenario_bundle.bundle_dir, dashboards_root=dashboards_root)

    contract_bundle = write_analysis_bundle(
        build_contract_selection_analysis(
            ticker="GPRE",
            snapshot_date="2026-04-12",
            target_price=20.0,
            target_date="2026-07-15",
            data_root=DATA_ROOT,
        ),
        analysis_kind="contract_selection",
        output_root=analysis_root,
    )
    publish_analysis_bundle(contract_bundle.bundle_dir)
    mirror_published_bundle(contract_bundle.bundle_dir, dashboards_root=dashboards_root)

    replay_bundle = write_analysis_bundle(
        build_replay_analysis(
            "GPRE",
            snapshot_date="2026-04-12",
            expiry_date="2026-04-17",
            strategy_name="long_call",
            data_root=DATA_ROOT,
        ),
        analysis_kind="replay",
        output_root=analysis_root,
    )
    publish_analysis_bundle(replay_bundle.bundle_dir)
    mirror_published_bundle(replay_bundle.bundle_dir, dashboards_root=dashboards_root)

    records = rebuild_dashboard_library(dashboards_root=dashboards_root)
    library_manifest = json.loads((dashboards_root / "library_manifest.json").read_text(encoding="utf-8"))
    root_index = (dashboards_root / "index.html").read_text(encoding="utf-8")
    all_dashboards = (dashboards_root / "all_dashboards.html").read_text(encoding="utf-8")
    ticker_index = (dashboards_root / "GPRE" / "index.html").read_text(encoding="utf-8")
    snapshot_index = (dashboards_root / "GPRE" / "2026-04-12" / "index.html").read_text(encoding="utf-8")
    contract_html = (dashboards_root / "GPRE" / "2026-04-12" / "contract-selection" / "dashboard.html").read_text(encoding="utf-8")
    scenario_html = (dashboards_root / "GPRE" / "2026-04-12" / "scenario" / "2026-04-17" / "dashboard.html").read_text(encoding="utf-8")

    assert {record["analysis_kind"] for record in records} >= {"scenario", "contract_selection", "replay"}
    assert {record["analysis_kind"] for record in library_manifest["records"]} >= {"scenario", "contract_selection", "replay"}
    assert "Options Lab Dashboard Library" in root_index
    assert "secondary mirror of already-published analysis bundles" in root_index
    assert "analytics" not in root_index.lower()
    assert "journal" not in root_index.lower()
    assert "All Published Dashboards" in all_dashboards
    assert "GPRE Dashboard Index" in ticker_index
    assert "Bundle-backed published dashboards available for this snapshot." in snapshot_index
    assert "Decision Snapshot" in contract_html
    assert "Market Context / Trust Summary" in contract_html
    assert "Required vs Assumed Path" in contract_html
    assert "Representative Paths" in contract_html
    assert "Option Value Over Path" in contract_html
    assert "Compare vs Stock Over Path" in contract_html
    assert "Same-Path Strike Comparison" in contract_html
    assert "Same-Path Expiry Comparison" in contract_html
    assert "Family / Candidate Highlights" in contract_html
    assert "Warnings / Risk Notes" in contract_html
    assert "Open The Main Explorer" not in contract_html
    assert "What This Wrapper Is For" not in contract_html
    assert "publish/dashboard.html" not in contract_html
    assert "Primary Scenario Dashboard" in scenario_html
    assert "Path Case Summary" in scenario_html
    for text in [root_index, all_dashboards, ticker_index, snapshot_index, contract_html, scenario_html]:
        assert "C:/Users" not in text
        assert "C:\\Users" not in text
        assert "file:///" not in text
