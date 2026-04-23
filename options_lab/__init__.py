"""Options Lab canonical Python-first analysis toolkit."""

from .analysis import (
    DEFAULT_ANALYSIS_OUTPUT_ROOT,
    AnalysisBundle,
    build_contract_selection_analysis,
    build_replay_analysis,
    build_scenario_analysis,
    build_strategy_analysis,
    publish_analysis_bundle,
    resolve_analysis_bundle,
    write_analysis_bundle,
)
from .io import OptionChain, OptionContract, load_chain, select_contract
from .pricing import price_option
from .publish import DEFAULT_DASHBOARDS_ROOT, mirror_published_bundle, rebuild_dashboard_library
from .scenarios import compare_positions, scenario_table
from .strategies import StrategyPosition, build_strategy

__all__ = [
    "AnalysisBundle",
    "DEFAULT_ANALYSIS_OUTPUT_ROOT",
    "DEFAULT_DASHBOARDS_ROOT",
    "OptionChain",
    "OptionContract",
    "StrategyPosition",
    "build_contract_selection_analysis",
    "build_replay_analysis",
    "build_scenario_analysis",
    "build_strategy",
    "build_strategy_analysis",
    "compare_positions",
    "load_chain",
    "mirror_published_bundle",
    "price_option",
    "publish_analysis_bundle",
    "rebuild_dashboard_library",
    "resolve_analysis_bundle",
    "scenario_table",
    "select_contract",
    "write_analysis_bundle",
]
