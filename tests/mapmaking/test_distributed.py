"""Multi-process tests for the cross-process sky reduction.

``cross_process_sum`` is a no-op whenever ``jax.process_count() == 1``, so its real behaviour is
unreachable from an ordinary test session however many *devices* that session provides. These
tests therefore spawn real OS processes joined by ``jax.distributed.initialize``, which is also
the only way to exercise the heterogeneous per-process shapes the per-process execution exists to
allow.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

NPIX = 5
# deliberately unequal: no global padding is involved, each process owns what it owns
SEGMENT_SIZES = [3, 7, 2, 9]


def _expected_total(n_proc: int) -> np.ndarray:
    total = np.zeros(NPIX, np.float64)
    for proc_id, size in enumerate(SEGMENT_SIZES[:n_proc]):
        total += (np.arange(size * NPIX, dtype=np.float64).reshape(size, NPIX) + proc_id).sum(0)
    return total


def _child(proc_id: int, n_proc: int, port: int, n_local: int) -> int:
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_local}'
    import jax
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)
    jax.distributed.initialize(
        coordinator_address=f'localhost:{port}', num_processes=n_proc, process_id=proc_id
    )
    from furax.mapmaking._distributed import cross_process_sum

    size = SEGMENT_SIZES[proc_id]
    segment = jnp.arange(size * NPIX, dtype=jnp.float64).reshape(size, NPIX) + proc_id
    partial = jax.device_put(segment.sum(axis=0), jax.local_devices()[0])

    total = cross_process_sum({'map': partial, 'hits': partial * 2})
    expected = _expected_total(n_proc)
    ok = np.allclose(np.asarray(total['map']), expected) and np.allclose(
        np.asarray(total['hits']), 2 * expected
    )
    return 0 if ok else 1


@pytest.mark.distributed
@pytest.mark.parametrize('n_proc, n_local', [(3, 1), (2, 2)])
def test_cross_process_sum_over_heterogeneous_segments(n_proc: int, n_local: int) -> None:
    port = 54400 + 10 * n_proc + n_local
    env = {**os.environ, 'FURAX_TEST_CHILD': '1'}
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, str(i), str(n_proc), str(port), str(n_local)], env=env
        )
        for i in range(n_proc)
    ]
    codes = [p.wait(timeout=180) for p in procs]
    assert codes == [0] * n_proc, f'child exit codes {codes}'


if __name__ == '__main__':  # spawned child, not collected by pytest
    sys.exit(_child(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])))
