"""Publish frozen HTML from canonical analysis bundles."""

from .dashboard import generate_dashboard, render_html_document
from .library import DEFAULT_DASHBOARDS_ROOT, mirror_published_bundle, rebuild_dashboard_library


def publish_analysis_bundle(*args, **kwargs):
    from ..analysis.artifacts import publish_analysis_bundle as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "DEFAULT_DASHBOARDS_ROOT",
    "generate_dashboard",
    "mirror_published_bundle",
    "publish_analysis_bundle",
    "rebuild_dashboard_library",
    "render_html_document",
]
