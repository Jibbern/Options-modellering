"""Secondary Dashboards mirror and library builders for published bundles."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import shutil
from typing import Any

from .dashboard import render_html_document
from ..persistence import write_json
from ..utils import clean_string, ensure_directory, slugify, windows_extended_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DASHBOARDS_ROOT = PROJECT_ROOT / "Dashboards"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_report_metadata(bundle_dir: Path) -> dict[str, Any]:
    return _load_json(bundle_dir / "metadata" / "report_metadata.json")


def _bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    return _load_json(bundle_dir / "bundle_manifest.json")


def _published_manifest(publish_dir: Path) -> dict[str, Any]:
    return _load_json(publish_dir / "published_manifest.json")


def _scenario_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("scenario") or {}


def _contract_selection_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("contract_selection") or {}


def _replay_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return metadata.get("replay") or {}


def _mirror_target_dir(bundle_dir: Path, published_manifest: dict[str, Any]) -> Path:
    ticker = clean_string(published_manifest.get("ticker")).upper()
    snapshot_date = clean_string(published_manifest.get("snapshot_date"))
    analysis_kind = clean_string(published_manifest.get("analysis_kind"))
    run_slug = clean_string(published_manifest.get("run_slug"))
    metadata = published_manifest.get("report_metadata") or {}
    publish_path = published_manifest.get("publish_path") or {}
    if analysis_kind == "scenario":
        expiry = clean_string(_scenario_metadata(metadata).get("expiry_date") or metadata.get("expiry_date") or run_slug.replace("expiry-", ""))
        return DEFAULT_DASHBOARDS_ROOT / ticker / snapshot_date / "scenario" / expiry
    if analysis_kind == "contract_selection":
        return DEFAULT_DASHBOARDS_ROOT / ticker / snapshot_date / "contract-selection"
    if analysis_kind == "replay":
        replay = _replay_metadata(metadata)
        expiry = clean_string(replay.get("expiry_date") or metadata.get("expiry_date"))
        strategy = slugify(replay.get("strategy_name") or metadata.get("strategy_name") or run_slug)
        return DEFAULT_DASHBOARDS_ROOT / ticker / snapshot_date / "replay" / expiry / strategy
    if analysis_kind == "strategy":
        strategy = slugify((metadata.get("strategy_report") or {}).get("strategy") or run_slug)
        expiry = clean_string((metadata.get("strategy_report") or {}).get("expiry_date") or metadata.get("expiry_date") or "no-expiry")
        return DEFAULT_DASHBOARDS_ROOT / ticker / snapshot_date / "strategy" / expiry / strategy
    fallback = slugify(analysis_kind or "analysis")
    if publish_path:
        return DEFAULT_DASHBOARDS_ROOT / ticker / snapshot_date / fallback / slugify(run_slug or fallback)
    return DEFAULT_DASHBOARDS_ROOT / ticker / snapshot_date / fallback / slugify(run_slug or fallback)


def _rewrite_mirrored_publish_links(target_dir: Path, published_manifest: dict[str, Any]) -> None:
    replacements: list[tuple[re.Pattern[str], str]] = []
    analysis_kind = clean_string(published_manifest.get("analysis_kind"))
    if analysis_kind == "contract_selection":
        replacements.append(
            (
                re.compile(r"(?:\.\./)+scenario/expiry-([0-9]{4}-[0-9]{2}-[0-9]{2})/publish/dashboard\.html"),
                r"../scenario/\1/dashboard.html",
            )
        )
    elif analysis_kind == "scenario":
        replacements.append(
            (
                re.compile(r"(?:\.\./)+contract_selection/[^\"?#]*/publish/dashboard\.html"),
                r"../../contract-selection/dashboard.html",
            )
        )
    if not replacements:
        return
    for path in [target_dir / "dashboard.html", target_dir / "report_metadata.json", target_dir / "published_manifest.json"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for pattern, replacement in replacements:
            updated = pattern.sub(replacement, updated)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def mirror_published_bundle(
    bundle_dir: str | Path,
    *,
    dashboards_root: str | Path = DEFAULT_DASHBOARDS_ROOT,
) -> Path:
    """Mirror one bundle-local publish/ directory into Dashboards/."""

    bundle_path = Path(bundle_dir)
    publish_dir = bundle_path / "publish"
    if not (publish_dir / "dashboard.html").exists():
        raise FileNotFoundError(f"Bundle publish directory is missing dashboard.html: {publish_dir}")
    published_manifest = _published_manifest(publish_dir)
    target_root = Path(dashboards_root)
    target_dir = _mirror_target_dir(bundle_path, published_manifest)
    relative_target = target_dir.relative_to(DEFAULT_DASHBOARDS_ROOT) if target_dir.is_absolute() and DEFAULT_DASHBOARDS_ROOT in target_dir.parents else None
    if dashboards_root != DEFAULT_DASHBOARDS_ROOT and relative_target is not None:
        target_dir = Path(dashboards_root) / relative_target
    if target_dir.exists():
        shutil.rmtree(windows_extended_path(target_dir))
    shutil.copytree(windows_extended_path(publish_dir), windows_extended_path(target_dir))
    _rewrite_mirrored_publish_links(target_dir, published_manifest)
    manifest_path = target_dir / "published_manifest.json"
    payload = _load_json(manifest_path)
    payload["mirrored_to_dashboards"] = str(target_dir)
    write_json(payload, manifest_path)
    rebuild_dashboard_library(dashboards_root=target_root)
    return target_dir


def _published_records(dashboards_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for manifest_path in dashboards_root.rglob("published_manifest.json"):
        payload = _load_json(manifest_path)
        dashboard_path = manifest_path.parent / "dashboard.html"
        if not dashboard_path.exists():
            continue
        record = {
            "analysis_kind": clean_string(payload.get("analysis_kind")),
            "ticker": clean_string(payload.get("ticker")).upper(),
            "snapshot_date": clean_string(payload.get("snapshot_date")),
            "run_slug": clean_string(payload.get("run_slug")),
            "title": clean_string(payload.get("title")) or clean_string(payload.get("analysis_kind")).replace("_", " ").title(),
            "published_at": clean_string(payload.get("published_at")),
            "published_dir": str(manifest_path.parent.relative_to(dashboards_root)).replace("\\", "/"),
            "published_dashboard": str(dashboard_path.relative_to(dashboards_root)).replace("\\", "/"),
        }
        metadata = payload.get("report_metadata") or {}
        if record["analysis_kind"] == "scenario":
            record["expiry_date"] = clean_string(_scenario_metadata(metadata).get("expiry_date") or metadata.get("expiry_date"))
        elif record["analysis_kind"] == "contract_selection":
            record["target_date"] = clean_string(_contract_selection_metadata(metadata).get("target_date") or metadata.get("target_date"))
        elif record["analysis_kind"] == "replay":
            replay = _replay_metadata(metadata)
            record["expiry_date"] = clean_string(replay.get("expiry_date") or metadata.get("expiry_date"))
            record["strategy_name"] = clean_string(replay.get("strategy_name") or metadata.get("strategy_name"))
        elif record["analysis_kind"] == "strategy":
            strategy = metadata.get("strategy_report") or {}
            record["expiry_date"] = clean_string(strategy.get("expiry_date"))
            record["strategy_name"] = clean_string(strategy.get("strategy"))
        records.append(record)
    records.sort(key=lambda item: (item["ticker"], item["snapshot_date"], item["analysis_kind"], item.get("published_at") or "", item["run_slug"]))
    return records


def _link(href: str, label: str, detail: str | None = None) -> str:
    detail_html = f'<div class="detail">{detail}</div>' if detail else ""
    return f'<li><a href="{href}">{label}</a>{detail_html}</li>'


def _page_section(title: str, intro: str, items: list[str]) -> str:
    if not items:
        return ""
    return (
        f'<section class="panel"><h2>{title}</h2><p class="section-intro">{intro}</p><ul class="bullet-list">'
        + "".join(items)
        + "</ul></section>"
    )


def _write_simple_page(path: Path, title: str, intro: str, sections: list[str]) -> None:
    ensure_directory(path.parent)
    body = (
        '<section class="hero"><div class="hero-top"><div>'
        f'<div class="eyebrow">Published Dashboard Library</div><h1>{title}</h1>'
        f'<p class="subtitle">{intro}</p></div></div></section>'
        + "".join(section for section in sections if section)
    )
    path.write_text(render_html_document(title, body), encoding="utf-8")


def rebuild_dashboard_library(
    *,
    dashboards_root: str | Path = DEFAULT_DASHBOARDS_ROOT,
) -> list[dict[str, Any]]:
    """Rebuild simple Dashboards indexes from mirrored published bundles only."""

    root = ensure_directory(dashboards_root)
    records = _published_records(root)
    write_json({"records": records}, root / "library_manifest.json")

    flat_items = [
        _link(record["published_dashboard"], record["title"], f'{record["ticker"]} {record["snapshot_date"]} {record["analysis_kind"]}')
        for record in records
    ]
    _write_simple_page(
        root / "all_dashboards.html",
        "All Published Dashboards",
        "Flat browse across bundle-backed HTML publishes mirrored into Dashboards/.",
        [_page_section("Dashboards", "Every mirrored publish currently available.", flat_items)],
    )

    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_snapshot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_ticker[record["ticker"]].append(record)
        by_snapshot[(record["ticker"], record["snapshot_date"])].append(record)

    ticker_items = [
        _link(f'{ticker}/index.html', ticker, f'{len(items)} published dashboard{"s" if len(items) != 1 else ""}')
        for ticker, items in sorted(by_ticker.items())
    ]
    _write_simple_page(
        root / "index.html",
        "Options Lab Dashboard Library",
        "Dashboards/ is a secondary mirror of already-published analysis bundles.",
        [
            _page_section("Tickers", "Open one ticker to browse snapshot-specific publishes.", ticker_items),
            _page_section("Everything", "Use the flat list when you want one long browseable view.", [_link("all_dashboards.html", "Open all dashboards")]),
        ],
    )

    for ticker, ticker_records in by_ticker.items():
        snapshot_items = defaultdict(list)
        for record in ticker_records:
            snapshot_items[record["snapshot_date"]].append(record)
        sections = []
        for snapshot_date, snapshot_records in sorted(snapshot_items.items()):
            sections.append(
                _page_section(
                    snapshot_date,
                    "Published bundle-backed dashboards for this snapshot date.",
                    [
                        _link(
                            f'{snapshot_date}/{record["published_dashboard"].split("/", 2)[-1]}',
                            record["title"],
                            record["analysis_kind"],
                        )
                        for record in snapshot_records
                    ],
                )
            )
            snapshot_dir = root / ticker / snapshot_date
            summary_items = [
                _link(
                    record["published_dashboard"].split("/", 2)[-1],
                    record["title"],
                    record["analysis_kind"],
                )
                for record in snapshot_records
            ]
            _write_simple_page(
                snapshot_dir / "index.html",
                f"{ticker} {snapshot_date}",
                "Bundle-backed published dashboards available for this snapshot.",
                [_page_section("Published Pages", "Use the primary scenario page first when it is available.", summary_items)],
            )
        _write_simple_page(
            root / ticker / "index.html",
            f"{ticker} Dashboard Index",
            "Snapshot-specific publish index built only from mirrored bundle publishes.",
            sections,
        )
    return records
