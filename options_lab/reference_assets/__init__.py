"""Reference-asset helpers for optional non-core metadata artifacts."""

__all__ = [
    "BarchartDashboardArtifact",
    "download_dashboard_image_best_effort",
    "list_dashboard_artifacts",
    "register_dashboard_image",
    "register_existing_dashboard_images",
]


def __getattr__(name: str):
    if name in {
        "BarchartDashboardArtifact",
        "download_dashboard_image_best_effort",
        "list_dashboard_artifacts",
        "register_dashboard_image",
        "register_existing_dashboard_images",
    }:
        from . import barchart_dashboard

        return getattr(barchart_dashboard, name)
    raise AttributeError(name)
