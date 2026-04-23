"""Canonical Python-first analysis entrypoints and bundle helpers.

Imports are intentionally lazy here so legacy compatibility modules can depend
on lightweight submodules such as ``options_lab.analysis.paths`` without
triggering bundle/report imports during module initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .artifacts import DEFAULT_ANALYSIS_OUTPUT_ROOT
    from .models import AnalysisBundle, StrategyAnalysisComputation


def build_contract_selection_analysis(*args, **kwargs):
    from .contract_selection import build_contract_selection_analysis as _impl

    return _impl(*args, **kwargs)


def build_scenario_analysis(*args, **kwargs):
    from .scenario import build_scenario_analysis as _impl

    return _impl(*args, **kwargs)


def build_replay_analysis(*args, **kwargs):
    from .replay import build_replay_analysis as _impl

    return _impl(*args, **kwargs)


def build_strategy_analysis(*args, **kwargs):
    from .strategy import build_strategy_analysis as _impl

    return _impl(*args, **kwargs)


def resolve_market_context(*args, **kwargs):
    from .market_context import resolve_market_context as _impl

    return _impl(*args, **kwargs)


def write_analysis_bundle(*args, **kwargs):
    from .artifacts import write_analysis_bundle as _impl

    return _impl(*args, **kwargs)


def publish_analysis_bundle(*args, **kwargs):
    from .artifacts import publish_analysis_bundle as _impl

    return _impl(*args, **kwargs)


def resolve_analysis_bundle(*args, **kwargs):
    from .artifacts import resolve_analysis_bundle as _impl

    return _impl(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in {"AnalysisBundle", "StrategyAnalysisComputation"}:
        from . import models

        return getattr(models, name)
    if name == "DEFAULT_ANALYSIS_OUTPUT_ROOT":
        from .artifacts import DEFAULT_ANALYSIS_OUTPUT_ROOT

        return DEFAULT_ANALYSIS_OUTPUT_ROOT
    raise AttributeError(name)


__all__ = [
    "AnalysisBundle",
    "DEFAULT_ANALYSIS_OUTPUT_ROOT",
    "StrategyAnalysisComputation",
    "build_contract_selection_analysis",
    "build_replay_analysis",
    "build_scenario_analysis",
    "build_strategy_analysis",
    "publish_analysis_bundle",
    "resolve_market_context",
    "resolve_analysis_bundle",
    "write_analysis_bundle",
]
