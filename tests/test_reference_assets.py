from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_lab.reference_assets.barchart_dashboard import (
    best_effort_attempt_path,
    list_dashboard_artifacts,
    register_dashboard_image,
    register_existing_dashboard_images,
    download_dashboard_image_best_effort,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
class _FakeResponse:
    def __init__(self, *, text: str = "", content: bytes = b"", status_code: int = 200, headers: dict | None = None):
        self.text = text
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse | Exception]):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        if not self.responses:
            raise AssertionError("No more fake responses queued.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_register_existing_dashboard_images_moves_top_level_file_and_writes_manifest(temp_data_root: Path):
    loose_dir = temp_data_root / "GPRE"
    loose_dir.mkdir(parents=True, exist_ok=True)
    loose_file = loose_dir / "GPRE-options-summary.png"
    loose_file.write_bytes(b"fake-png")

    artifacts = register_existing_dashboard_images("GPRE", data_root=temp_data_root)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    destination = Path(artifact.local_path)
    assert destination.exists()
    assert not loose_file.exists()
    assert "raw\\manual" in str(destination) or "raw/manual" in str(destination)
    assert artifact.snapshot_date is None

    manifest_items = list_dashboard_artifacts("GPRE", data_root=temp_data_root)
    assert len(manifest_items) == 1
    assert manifest_items[0]["artifact_type"] == "options_data_dashboard_image"
    assert Path(temp_data_root / "GPRE" / "options_metadata" / "barchart" / "options_data_dashboard" / "metadata" / "dashboard_manifest.json").exists()


def test_register_dashboard_image_uses_standard_name_when_date_is_known(temp_data_root: Path):
    source_file = temp_data_root / "source.png"
    source_file.write_bytes(b"fake-png")

    artifact = register_dashboard_image(
        source_file,
        "GPRE",
        data_root=temp_data_root,
        snapshot_date="2026-04-10",
        acquisition_method="manual",
        move=False,
    )

    assert Path(artifact.local_path).name == "gpre_options_data_dashboard_2026-04-10.png"
    assert artifact.snapshot_date.isoformat() == "2026-04-10"


def test_best_effort_downloader_records_failure_debug_info_when_no_image_url_is_found(temp_data_root: Path):
    session = _FakeSession(
        [
            _FakeResponse(
                text="<html><head><title>Options Data</title></head><body>No dashboard image here</body></html>",
                status_code=200,
                headers={"content-type": "text/html"},
            )
        ]
    )

    with pytest.raises(RuntimeError):
        download_dashboard_image_best_effort("GPRE", data_root=temp_data_root, session=session)

    attempt_file = best_effort_attempt_path("GPRE", temp_data_root)
    assert attempt_file.exists()
    payload = json.loads(attempt_file.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert "No direct dashboard image URL" in payload["error"]
    assert Path(payload["debug_html_path"]).exists()


def test_best_effort_downloader_can_register_a_discovered_image_url(temp_data_root: Path):
    session = _FakeSession(
        [
            _FakeResponse(
                text='<html><head><meta property="og:image" content="https://example.com/gpre-options-dashboard.png"></head></html>',
                status_code=200,
                headers={"content-type": "text/html"},
            ),
            _FakeResponse(
                content=b"\x89PNG\r\n\x1a\nfake",
                status_code=200,
                headers={"content-type": "image/png"},
            ),
        ]
    )

    artifact = download_dashboard_image_best_effort(
        "GPRE",
        data_root=temp_data_root,
        snapshot_date="2026-04-10",
        session=session,
    )

    assert Path(artifact.local_path).exists()
    assert artifact.acquisition_method == "automated_best_effort"
    assert artifact.source_url == "https://example.com/gpre-options-dashboard.png"
    manifest_items = list_dashboard_artifacts("GPRE", data_root=temp_data_root)
    assert len(manifest_items) == 1
