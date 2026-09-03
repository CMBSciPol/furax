"""Multi-process runs of the multi-observation mapmaker.

Every test spawns real OS processes with ``jax.distributed.initialize`` on the CPU backend: the
cross-process paths (global mesh, all-gathered probe, per-process slot blocks) are otherwise
unreachable, since a single process is their degenerate case. The children run the mapmaker on
synthetic observations of unequal length, with explicit templates so the amplitude scatter is
exercised too, and compare against a single-process run of the same configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

N_SAMPLES = (512, 640, 1536, 768, 1024)  # five observations, no two of a kind
N_DETS = 4
N_HARMONICS = 2
PORT_BASE = 54500


def _run_mapmaker(out_path: Path, *, max_buckets: int) -> None:
    """Map the synthetic observations and save the products to ``out_path``."""
    import jax

    from furax.mapmaking import MultiObservationMapMaker
    from furax.mapmaking.config import (
        HealpixConfig,
        HWPSynchronousConfig,
        LandscapeConfig,
        MapMakingConfig,
        PointingConfig,
        TemplatesConfig,
        WeightingConfig,
        WeightingMode,
    )
    from tests.mapmaking.helpers import FakeLazyObservation

    observations = [
        FakeLazyObservation(seed=i, n_dets=N_DETS, n_samples=n) for i, n in enumerate(N_SAMPLES)
    ]
    # Unit weights: a fitted noise level depends on the padding, which differs between layouts.
    config = MapMakingConfig(
        pointing=PointingConfig(on_the_fly=True),
        landscape=LandscapeConfig(stokes='IQU', healpix=HealpixConfig(nside=8)),
        weighting=WeightingConfig(mode=WeightingMode.IDENTITY),
        templates=TemplatesConfig(hwp_synchronous=HWPSynchronousConfig(N_HARMONICS, explicit=True)),
        hits_cut=0.0,
        cond_cut=0.0,
        max_buckets=max_buckets,
    )
    results = MultiObservationMapMaker(observations, config=config).run()
    if jax.process_index() == 0:
        assert results.template_amplitudes is not None
        np.savez(
            out_path,
            map=np.asarray(results.map.data),
            hit_map=np.asarray(results.hit_map),
            icov=np.asarray(results.icov),
            amplitudes=np.asarray(results.template_amplitudes['hwp_synchronous']),
        )


def _child(proc_id: int, n_proc: int, port: int, n_local: int, out: str, max_buckets: int) -> None:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_local}'
    import jax

    jax.config.update('jax_enable_x64', True)
    jax.distributed.initialize(
        coordinator_address=f'localhost:{port}', num_processes=n_proc, process_id=proc_id
    )
    _run_mapmaker(Path(out), max_buckets=max_buckets)


def _spawn(n_proc: int, n_local: int, port: int, out: Path, max_buckets: int) -> None:
    root = Path(__file__).resolve().parents[2]  # the children import `tests.mapmaking.helpers`
    env = {**os.environ, 'PYTHONPATH': str(root)}
    args = [str(n_proc), str(port), str(n_local), str(out), str(max_buckets)]
    procs = [
        subprocess.Popen([sys.executable, __file__, str(i), *args], cwd=root, env=env)
        for i in range(n_proc)
    ]
    codes = [p.wait(timeout=600) for p in procs]
    assert codes == [0] * n_proc, f'child exit codes {codes}'


@pytest.fixture(scope='module')
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, np.ndarray]:
    """The single-process products, one bucket, computed in a child so the backend is CPU."""
    out = tmp_path_factory.mktemp('reference') / 'single.npz'
    _spawn(1, 1, PORT_BASE, out, max_buckets=1)
    return dict(np.load(out))


@pytest.mark.distributed
@pytest.mark.parametrize(
    'n_proc, n_local, max_buckets',
    [(2, 1, 1), (2, 2, 3), (3, 1, 2)],
    ids=['2proc-1dev-1bucket', '2proc-2dev-3buckets', '3proc-1dev-2buckets'],
)
def test_multi_process_matches_single_process(
    n_proc: int,
    n_local: int,
    max_buckets: int,
    reference: dict[str, np.ndarray],
    tmp_path: Path,
) -> None:
    out = tmp_path / 'multi.npz'
    _spawn(n_proc, n_local, PORT_BASE + 10 * n_proc + n_local, out, max_buckets)
    result = dict(np.load(out))
    np.testing.assert_array_equal(result['hit_map'], reference['hit_map'])
    np.testing.assert_allclose(result['icov'], reference['icov'], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result['map'], reference['map'], rtol=1e-8, atol=1e-10)
    # per-observation amplitudes come back in observation order whatever the layout
    assert result['amplitudes'].shape == (len(N_SAMPLES), N_DETS, 2 * N_HARMONICS)
    np.testing.assert_allclose(result['amplitudes'], reference['amplitudes'], rtol=1e-8, atol=1e-10)


if __name__ == '__main__':
    proc_id, n_proc, port, n_local, out, max_buckets = sys.argv[1:7]
    _child(int(proc_id), int(n_proc), int(port), int(n_local), out, int(max_buckets))
