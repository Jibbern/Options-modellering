import json
from pathlib import Path

from options_lab.analysis import build_replay_analysis, publish_analysis_bundle, write_analysis_bundle
from options_lab.publish import mirror_published_bundle, rebuild_dashboard_library


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def test_replay_analysis_build_uses_exact_then_modeled_fallbacks():
    result = build_replay_analysis(
        "GPRE",
        snapshot_date="2026-04-12",
        expiry_date="2026-04-17",
        strategy_name="long_call",
        data_root=DATA_ROOT,
    )

    checkpoints = result.checkpoint_replay.set_index("checkpoint")

    assert result.status == "partial"
    assert "event" in result.available_checkpoints
    assert checkpoints.loc["entry", "valuation_source"] == "exact_local_later_chain"
    assert checkpoints.loc["1w", "valuation_source"] == "approximate_model_estimate"
    assert bool(checkpoints.loc["1m", "clamped_to_expiry"]) is True
    assert not result.expected_move_vs_actual.empty
    assert not result.driver_decomposition.empty
    assert set(result.compare_vs_stock["mode"]) == {"equal_capital", "share_equivalent"}
    summary_row = result.case_summary.iloc[0]
    assert summary_row["exact_chain_checkpoint_count"] >= 1
    assert summary_row["modeled_checkpoint_count"] >= 1
    assert bool(summary_row["what_this_case_shows"])


def test_replay_bundle_publish_and_dashboards_mirror_stay_share_safe(temp_workspace_root: Path):
    result = build_replay_analysis(
        "GPRE",
        snapshot_date="2026-04-12",
        expiry_date="2026-04-17",
        strategy_name="long_call",
        data_root=DATA_ROOT,
    )
    bundle = write_analysis_bundle(
        result,
        analysis_kind="replay",
        output_root=temp_workspace_root / "analysis_outputs",
    )
    bundle_report_metadata = json.loads((bundle.bundle_dir / "metadata" / "report_metadata.json").read_text(encoding="utf-8"))
    dashboard_path = publish_analysis_bundle(bundle.bundle_dir)
    dashboards_root = temp_workspace_root / "Dashboards"
    mirrored_dir = mirror_published_bundle(bundle.bundle_dir, dashboards_root=dashboards_root)
    records = rebuild_dashboard_library(dashboards_root=dashboards_root)

    html = dashboard_path.read_text(encoding="utf-8")
    mirrored_html = (mirrored_dir / "dashboard.html").read_text(encoding="utf-8")
    library_manifest = json.loads((dashboards_root / "library_manifest.json").read_text(encoding="utf-8"))

    assert dashboard_path.exists()
    assert bundle_report_metadata["report_kind"] == "replay"
    assert "replay" in bundle_report_metadata
    assert "historical_replay" not in bundle_report_metadata
    assert "Historical Replay / Case Study" in html
    assert "Honesty Note" in html
    assert "Expected vs Actual" in html
    assert "Drivers" in html
    assert "Compare vs Stock" in html
    assert "Local History" in html
    assert any(record["analysis_kind"] == "replay" for record in records)
    assert any(record["analysis_kind"] == "replay" for record in library_manifest["records"])
    for text in [html, mirrored_html]:
        assert "C:/Users" not in text
        assert "C:\\Users" not in text
        assert "file:///" not in text
