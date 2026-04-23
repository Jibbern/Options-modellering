"""Helpers for optional Barchart Options Data Dashboard image assets.

These dashboard images are reference context only. They are not authoritative
machine-readable market data and are intentionally kept separate from the core
chain, pricing, and scenario flows.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..persistence import write_json
from ..utils import clean_string, ensure_directory, parse_date

SOURCE = "barchart"
ARTIFACT_TYPE = "options_data_dashboard_image"
BARCHART_OPTIONS_DATA_URL = "https://www.barchart.com/stocks/quotes/{ticker}/options-data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MANIFEST_NAME = "dashboard_manifest.json"
SOURCE_NOTES_NAME = "source_notes.json"
BEST_EFFORT_ATTEMPT_NAME = "last_download_attempt.json"


@dataclass(frozen=True)
class BarchartDashboardArtifact:
    """Registered dashboard-image metadata for one ticker artifact."""

    ticker: str
    snapshot_date: date | None
    source: str
    source_url: str | None
    artifact_type: str
    local_path: str
    downloaded_at: str
    acquisition_method: str
    notes: str | None = None
    original_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["snapshot_date"] = self.snapshot_date.isoformat() if self.snapshot_date else None
        return payload


def options_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return options_root() / "data"


def dashboard_root(ticker: str, data_root: str | Path | None = None) -> Path:
    base = Path(data_root) if data_root is not None else default_data_root()
    return (
        base
        / clean_string(ticker).upper()
        / "options_metadata"
        / "barchart"
        / "options_data_dashboard"
    )


def ensure_dashboard_structure(
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Path]:
    """Create and return the folder structure for Barchart dashboard assets."""

    root = dashboard_root(ticker, data_root)
    raw = ensure_directory(root / "raw")
    return {
        "root": ensure_directory(root),
        "raw": raw,
        "raw_manual": ensure_directory(raw / "manual"),
        "raw_downloaded": ensure_directory(raw / "downloaded"),
        "normalized": ensure_directory(root / "normalized"),
        "metadata": ensure_directory(root / "metadata"),
    }


def manifest_path(ticker: str, data_root: str | Path | None = None) -> Path:
    return dashboard_root(ticker, data_root) / "metadata" / MANIFEST_NAME


def source_notes_path(ticker: str, data_root: str | Path | None = None) -> Path:
    return dashboard_root(ticker, data_root) / "metadata" / SOURCE_NOTES_NAME


def best_effort_attempt_path(ticker: str, data_root: str | Path | None = None) -> Path:
    return dashboard_root(ticker, data_root) / "metadata" / BEST_EFFORT_ATTEMPT_NAME


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _isoformat_utc(value: datetime | None = None) -> str:
    timestamp = value or _utc_now()
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_slug(value: datetime | None = None) -> str:
    timestamp = value or _utc_now()
    return timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _infer_snapshot_date_from_filename(path: str | Path) -> date | None:
    filename = Path(path).name
    patterns = (
        r"(?P<date>\d{4}-\d{2}-\d{2})",
        r"(?P<date>\d{2}-\d{2}-\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, filename)
        if not match:
            continue
        parsed = parse_date(match.group("date"))
        if parsed is not None:
            return parsed
    return None


def _build_standard_filename(
    ticker: str,
    snapshot_date: date | None,
    suffix: str,
    *,
    fallback_name: str,
) -> str:
    if snapshot_date is None:
        return fallback_name
    return f"{clean_string(ticker).lower()}_options_data_dashboard_{snapshot_date.isoformat()}{suffix.lower()}"


def _load_manifest(ticker: str, data_root: str | Path | None = None) -> dict[str, Any]:
    path = manifest_path(ticker, data_root)
    if not path.exists():
        root = dashboard_root(ticker, data_root)
        return {
            "generated_at": _isoformat_utc(),
            "ticker": clean_string(ticker).upper(),
            "source": SOURCE,
            "artifact_type": ARTIFACT_TYPE,
            "root": str(root),
            "artifacts": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(
    ticker: str,
    manifest: dict[str, Any],
    data_root: str | Path | None = None,
) -> Path:
    ensure_dashboard_structure(ticker, data_root)
    manifest["generated_at"] = _isoformat_utc()
    return write_json(manifest, manifest_path(ticker, data_root))


def _write_source_notes(ticker: str, data_root: str | Path | None = None) -> Path:
    ensure_dashboard_structure(ticker, data_root)
    notes = {
        "ticker": clean_string(ticker).upper(),
        "source": SOURCE,
        "artifact_type": ARTIFACT_TYPE,
        "notes": [
            "Options Data Dashboard images are optional reference assets only.",
            "They are not parsed as machine-readable pricing inputs.",
            "Manual registration is the stable primary path.",
            "The automated downloader is best-effort and may stop working if Barchart changes page behavior or blocks requests.",
        ],
    }
    return write_json(notes, source_notes_path(ticker, data_root))


def _write_sidecar_metadata(
    artifact: BarchartDashboardArtifact,
    ticker: str,
    data_root: str | Path | None = None,
) -> Path:
    metadata_dir = ensure_dashboard_structure(ticker, data_root)["metadata"]
    local_name = Path(artifact.local_path).stem
    return write_json(artifact.to_dict(), metadata_dir / f"{local_name}.json")


def _upsert_artifact(
    artifact: BarchartDashboardArtifact,
    ticker: str,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    manifest = _load_manifest(ticker, data_root)
    artifacts = [item for item in manifest.get("artifacts", []) if item.get("local_path") != artifact.local_path]
    artifacts.append(artifact.to_dict())
    artifacts.sort(key=lambda item: (item.get("snapshot_date") or "", item.get("local_path") or ""))
    manifest["artifacts"] = artifacts
    manifest["source_notes"] = str(_write_source_notes(ticker, data_root))
    manifest["artifact_count"] = len(artifacts)
    _write_sidecar_metadata(artifact, ticker, data_root)
    _write_manifest(ticker, manifest, data_root)
    return manifest


def list_dashboard_artifacts(
    ticker: str,
    data_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return all known dashboard artifacts for a ticker from the local manifest."""

    manifest = _load_manifest(ticker, data_root)
    return list(manifest.get("artifacts", []))


def find_existing_dashboard_candidates(
    ticker: str,
    data_root: str | Path | None = None,
) -> list[Path]:
    """Find loose top-level dashboard images that should live in the metadata area."""

    base = (Path(data_root) if data_root is not None else default_data_root()) / clean_string(ticker).upper()
    if not base.exists():
        return []
    candidates: list[Path] = []
    for path in base.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = clean_string(path.stem).lower()
        if "option" in stem and ("summary" in stem or "dashboard" in stem):
            candidates.append(path)
    return sorted(candidates)


def register_dashboard_image(
    path: str | Path,
    ticker: str,
    *,
    data_root: str | Path | None = None,
    snapshot_date: date | str | None = None,
    source_url: str | None = None,
    acquisition_method: str = "manual",
    notes: str | None = None,
    move: bool = False,
) -> BarchartDashboardArtifact:
    """Register one dashboard image in the local reference-asset store."""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Dashboard image was not found: {source_path}")
    if source_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported dashboard image type: {source_path.suffix}")

    structure = ensure_dashboard_structure(ticker, data_root)
    resolved_snapshot_date = parse_date(snapshot_date) if snapshot_date is not None else _infer_snapshot_date_from_filename(source_path)
    destination_dir = structure["raw_downloaded"] if acquisition_method == "automated_best_effort" else structure["raw_manual"]
    fallback_name = source_path.name
    target_name = _build_standard_filename(
        ticker,
        resolved_snapshot_date,
        source_path.suffix,
        fallback_name=fallback_name,
    )
    destination_path = destination_dir / target_name

    note_parts: list[str] = []
    if notes:
        note_parts.append(clean_string(notes))
    if resolved_snapshot_date is None and target_name == fallback_name:
        note_parts.append(
            "Snapshot date could not be inferred reliably from the filename, so the original filename was preserved."
        )

    if source_path.resolve() != destination_path.resolve():
        ensure_directory(destination_path.parent)
        if move:
            shutil.move(str(source_path), str(destination_path))
        else:
            shutil.copy2(source_path, destination_path)

    downloaded_at = _isoformat_utc(datetime.fromtimestamp(destination_path.stat().st_mtime, tz=timezone.utc))
    artifact = BarchartDashboardArtifact(
        ticker=clean_string(ticker).upper(),
        snapshot_date=resolved_snapshot_date,
        source=SOURCE,
        source_url=source_url or BARCHART_OPTIONS_DATA_URL.format(ticker=clean_string(ticker).upper()),
        artifact_type=ARTIFACT_TYPE,
        local_path=str(destination_path),
        downloaded_at=downloaded_at,
        acquisition_method=acquisition_method,
        notes=" ".join(note_parts) or None,
        original_filename=source_path.name,
    )
    _upsert_artifact(artifact, ticker, data_root)
    return artifact


def register_existing_dashboard_images(
    ticker: str,
    data_root: str | Path | None = None,
) -> list[BarchartDashboardArtifact]:
    """Move and register loose top-level dashboard images for a ticker."""

    artifacts: list[BarchartDashboardArtifact] = []
    for candidate in find_existing_dashboard_candidates(ticker, data_root):
        artifacts.append(
            register_dashboard_image(
                candidate,
                ticker,
                data_root=data_root,
                acquisition_method="manual",
                move=True,
            )
        )
    return artifacts


def _build_barchart_headers(ticker: str) -> dict[str, str]:
    ticker_lower = clean_string(ticker).lower()
    return {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "accept-language": "en-US,en;q=0.9",
        "referer": f"https://www.barchart.com/stocks/quotes/{ticker_lower}",
        "user-agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }


def _extract_candidate_image_urls(html: str, ticker: str) -> list[str]:
    candidates: list[str] = []
    ticker_lower = clean_string(ticker).lower()
    patterns = (
        r'content="(?P<url>https?://[^"]+\.png[^"]*)"',
        r'"(?P<url>https?://[^"]+\.png[^"]*)"',
        r"'(?P<url>https?://[^']+\.png[^']*)'",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE):
            url = clean_string(match.group("url"))
            if not url:
                continue
            lowered = url.lower()
            if "logo" in lowered or "icon" in lowered or "throbber" in lowered or "barchart-og" in lowered:
                continue
            if not any(keyword in lowered for keyword in (ticker_lower, "dashboard", "options-data", "options", "summary")):
                continue
            candidates.append(url)
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def download_dashboard_image_best_effort(
    ticker: str,
    *,
    data_root: str | Path | None = None,
    snapshot_date: date | str | None = None,
    session: requests.Session | None = None,
    timeout: int = 30,
) -> BarchartDashboardArtifact:
    """Try to fetch a Barchart dashboard image without making the project depend on it.

    This function inspects the public page HTML for a directly fetchable image URL.
    If no such URL is discoverable, it fails clearly and records the attempt
    metadata for debugging. Manual registration remains the stable workflow.
    """

    ticker_upper = clean_string(ticker).upper()
    structure = ensure_dashboard_structure(ticker_upper, data_root)
    page_url = BARCHART_OPTIONS_DATA_URL.format(ticker=ticker_upper)
    active_session = session or requests.Session()
    attempt_payload: dict[str, Any] = {
        "ticker": ticker_upper,
        "page_url": page_url,
        "attempted_at": _isoformat_utc(),
        "method": "best_effort_page_inspection",
        "success": False,
    }

    try:
        page_response = active_session.get(
            page_url,
            headers=_build_barchart_headers(ticker_upper),
            timeout=timeout,
        )
        page_response.raise_for_status()
    except requests.RequestException as exc:
        attempt_payload["error"] = f"Page request failed: {exc}"
        write_json(attempt_payload, best_effort_attempt_path(ticker_upper, data_root))
        raise RuntimeError(
            f"Could not load the Barchart Options Data Dashboard page for {ticker_upper}: {exc}"
        ) from exc

    html = page_response.text
    debug_html_path = structure["metadata"] / f"{ticker_upper.lower()}_options_data_dashboard_page_{_timestamp_slug()}.html"
    debug_html_path.write_text(html, encoding="utf-8")
    attempt_payload["debug_html_path"] = str(debug_html_path)

    candidate_urls = _extract_candidate_image_urls(html, ticker_upper)
    attempt_payload["candidate_image_urls"] = candidate_urls
    if not candidate_urls:
        attempt_payload["error"] = (
            "No direct dashboard image URL could be discovered from the page HTML. "
            "Manual registration remains the stable workflow."
        )
        write_json(attempt_payload, best_effort_attempt_path(ticker_upper, data_root))
        raise RuntimeError(attempt_payload["error"])

    image_url = candidate_urls[0]
    attempt_payload["selected_image_url"] = image_url
    try:
        image_response = active_session.get(
            image_url,
            headers={"user-agent": _build_barchart_headers(ticker_upper)["user-agent"]},
            timeout=timeout,
        )
        image_response.raise_for_status()
    except requests.RequestException as exc:
        attempt_payload["error"] = f"Image request failed: {exc}"
        write_json(attempt_payload, best_effort_attempt_path(ticker_upper, data_root))
        raise RuntimeError(
            f"Found a candidate image URL for {ticker_upper}, but fetching it failed: {exc}"
        ) from exc

    content_type = clean_string(image_response.headers.get("content-type")).lower()
    if not content_type.startswith("image/"):
        attempt_payload["error"] = (
            f"Candidate URL did not return an image content-type: {content_type or 'unknown'}"
        )
        write_json(attempt_payload, best_effort_attempt_path(ticker_upper, data_root))
        raise RuntimeError(attempt_payload["error"])

    suffix = ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        suffix = ".jpg"
    destination = structure["raw_downloaded"] / _build_standard_filename(
        ticker_upper,
        parse_date(snapshot_date),
        suffix,
        fallback_name=f"{ticker_upper.lower()}_options_data_dashboard_{_timestamp_slug()}{suffix}",
    )
    destination.write_bytes(image_response.content)
    attempt_payload["saved_image_path"] = str(destination)
    attempt_payload["success"] = True
    write_json(attempt_payload, best_effort_attempt_path(ticker_upper, data_root))

    return register_dashboard_image(
        destination,
        ticker_upper,
        data_root=data_root,
        snapshot_date=snapshot_date,
        source_url=image_url,
        acquisition_method="automated_best_effort",
        notes="Downloaded via best-effort page inspection from the Barchart options-data page.",
        move=False,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Barchart dashboard reference-asset management."""

    parser = argparse.ArgumentParser(
        description=(
            "Manage optional Barchart Options Data Dashboard images as reference assets. "
            "These images are not core pricing input."
        )
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--data-root", help="Override the Options data root.")
    parser.add_argument("--snapshot-date", help="Optional snapshot date for the registered image.")
    parser.add_argument("--source-path", help="Register one specific local image path.")
    parser.add_argument("--register-existing", action="store_true", help="Move and register loose top-level dashboard images for the ticker.")
    parser.add_argument("--download", action="store_true", help="Try the optional best-effort Barchart downloader.")
    parser.add_argument("--list", action="store_true", help="List registered dashboard artifacts for the ticker.")
    parser.add_argument("--notes", help="Optional note to attach when registering a local image.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Barchart dashboard reference-asset CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.register_existing:
        artifacts = register_existing_dashboard_images(args.ticker, data_root=args.data_root)
        print(json.dumps([artifact.to_dict() for artifact in artifacts], indent=2))
        return 0

    if args.source_path:
        artifact = register_dashboard_image(
            args.source_path,
            args.ticker,
            data_root=args.data_root,
            snapshot_date=args.snapshot_date,
            notes=args.notes,
            acquisition_method="manual",
            move=False,
        )
        print(json.dumps(artifact.to_dict(), indent=2))
        return 0

    if args.download:
        artifact = download_dashboard_image_best_effort(
            args.ticker,
            data_root=args.data_root,
            snapshot_date=args.snapshot_date,
        )
        print(json.dumps(artifact.to_dict(), indent=2))
        return 0

    if args.list:
        print(json.dumps(list_dashboard_artifacts(args.ticker, data_root=args.data_root), indent=2))
        return 0

    parser.error("Choose one action: --register-existing, --source-path, --download, or --list.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
