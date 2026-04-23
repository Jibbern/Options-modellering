from __future__ import annotations

from datetime import date

import numpy as np

from options_lab.analysis.simulation import (
    build_iv_path_example,
    build_path_grid,
    build_stock_path_example,
    build_stock_path_pool,
    pair_stock_and_iv_paths,
    select_representative_path_pairs,
)


def test_simulation_builds_deterministic_gbm_and_conditioned_stock_paths():
    grid = build_path_grid(date(2026, 4, 12), date(2026, 7, 15))
    rng = np.random.default_rng(7)

    deterministic = build_stock_path_example(
        grid,
        entry_spot=16.0,
        mode="deterministic",
        preset="slow_bull",
        target_end=20.0,
        rng=rng,
        path_id="deterministic-slow-bull",
    )
    gbm = build_stock_path_example(
        grid,
        entry_spot=16.0,
        mode="simulated",
        target_end=20.0,
        annualized_vol=0.55,
        drift=0.18,
        rng=rng,
        path_id="gbm-demo",
    )
    conditioned = build_stock_path_example(
        grid,
        entry_spot=16.0,
        mode="conditioned",
        target_end=20.0,
        annualized_vol=0.50,
        drift=0.18,
        cross_level=18.0,
        cross_behavior="cross_early_then_revert",
        rng=rng,
        path_id="conditioned-demo",
    )

    assert deterministic.path_kind == "deterministic"
    assert deterministic.path_points[0]["spot_price"] == 16.0
    assert deterministic.path_points[-1]["spot_price"] > deterministic.path_points[0]["spot_price"]

    assert gbm.path_kind == "simulated"
    assert len(gbm.path_points) == len(grid)
    assert gbm.path_points[-1]["spot_price"] > 0

    assert conditioned.path_kind == "conditioned"
    assert abs(conditioned.path_points[-1]["spot_price"] - 20.0) < 1.0
    assert max(point["spot_price"] for point in conditioned.path_points) >= 18.0


def test_simulation_builds_independent_iv_paths_and_selects_representative_pairs():
    grid = build_path_grid(date(2026, 4, 12), date(2026, 7, 15))
    rng = np.random.default_rng(11)
    stock_paths = build_stock_path_pool(
        grid,
        entry_spot=16.0,
        target_end=20.0,
        mode="mixed",
        simulated_path_count=10,
        rng=rng,
    )
    iv_paths = [
        build_iv_path_example(
            grid,
            base_iv_shift=0.0,
            mode="flat",
            rng=rng,
            iv_path_id="iv-flat",
        ),
        build_iv_path_example(
            grid,
            base_iv_shift=0.0,
            mode="earnings_build_then_crush",
            rng=rng,
            iv_path_id="iv-event",
        ),
        build_iv_path_example(
            grid,
            base_iv_shift=0.0,
            mode="mean_reversion_lower",
            rng=rng,
            iv_path_id="iv-lower",
        ),
    ]
    path_pairs = pair_stock_and_iv_paths(stock_paths[:6], iv_paths)
    selected = select_representative_path_pairs(
        path_pairs,
        path_outcomes={
            pair.path_pair_id: {
                "final_profit_loss": -200.0 + index * 120.0,
                "goal_reached": index >= 2,
                "outperformed_stock": index >= 3,
                "crossed_key_level": index >= 1,
            }
            for index, pair in enumerate(path_pairs)
        },
    )

    assert len(path_pairs) == 18
    assert {path.path_kind for path in stock_paths} >= {"deterministic", "simulated", "conditioned"}
    assert iv_paths[0].path_points[0]["iv_shift_points"] == 0.0
    assert iv_paths[1].path_points[-1]["iv_shift_points"] < iv_paths[1].path_points[1]["iv_shift_points"]
    assert selected
    assert {item["representative_bucket"] for item in selected} >= {
        "misses_badly",
        "almost_works",
        "just_works",
        "works_well",
    }
