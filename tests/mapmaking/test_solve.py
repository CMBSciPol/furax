import json
import math
from pathlib import Path

import jax.numpy as jnp

import furax.linalg
from furax.mapmaking._solve import _solver_stats
from furax.mapmaking.results import write_json


def test_solver_stats_keep_non_finite_residuals(tmp_path: Path):
    result = furax.linalg.CGResult(
        solution=jnp.zeros(2),
        residuals=jnp.array([1.0, jnp.nan, jnp.inf, 0.0]),
        num_steps=jnp.array(2),
        converged=jnp.array(False),
    )
    stats = _solver_stats(result, {'max_steps': 4}, previous_steps=3)
    assert stats['residuals'][0] == 1.0
    assert math.isnan(stats['residuals'][1])
    assert math.isinf(stats['residuals'][2])
    assert len(stats['residuals']) == 3
    assert stats['num_steps'] == 5
    assert stats['resumed_after'] == 3

    # the stats file reads back with the same values
    write_json(tmp_path / 'solver_stats.json', stats)
    loaded = json.loads((tmp_path / 'solver_stats.json').read_text())
    assert math.isnan(loaded['residuals'][1]) and math.isinf(loaded['residuals'][2])
