import logging
import os
from pathlib import Path

import pytest

# Disable JAX GPU memory pre-allocation so the allocator can free memory between
# tests instead of holding 75% of GPU memory for the entire session.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# Do not import JAX here (directly or transitively): `JAX_PLATFORMS` and the device
# count are snapshotted at `import jax`, which must happen after pytest_configure has
# set them. Hence JAX-touching imports are deferred into fixtures and test modules.


def pytest_configure(config: pytest.Config) -> None:
    """Provision several CPU devices for a ``-m distributed`` selection.

    Must run before any ``import jax``: the platform and device count are fixed at backend init.
    """
    markexpr = str(getattr(config.option, 'markexpr', '') or '')
    running_distributed = 'distributed' in markexpr and 'not distributed' not in markexpr
    if not running_distributed:
        return
    if 'SLURM_JOB_ID' in os.environ:
        import jax

        jax.distributed.initialize()
    else:
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=8')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Drop the marked tests that the current session cannot run.

    ``distributed`` tests need several devices, and are skipped so the summary names the selection
    that picks them up.

    ``insubprocess`` tests re-exec themselves through a hook that replaces the whole run protocol,
    which pytest-xdist reads as a worker crash and turns into an INTERNALERROR. That same hook
    fires before a skip mark would be honoured, so under xdist they have to leave the item list
    outright; `pytest -m insubprocess` runs them in a serial session.
    """
    import jax

    if jax.device_count() < 2:
        skip = pytest.mark.skip('distributed test; run with `pytest -m distributed`')
        for item in items:
            if item.get_closest_marker('distributed'):
                item.add_marker(skip)

    # the controller carries `dist`, each worker carries `workerinput`; both collect, and their
    # collections have to agree, so the condition must hold on either side
    under_xdist = getattr(config.option, 'dist', 'no') != 'no' or hasattr(config, 'workerinput')
    if not under_xdist:
        return
    deselected = [item for item in items if item.get_closest_marker('insubprocess')]
    if not deselected:
        return
    items[:] = [item for item in items if item.get_closest_marker('insubprocess') is None]
    config.hook.pytest_deselected(items=deselected)


@pytest.fixture(scope='session', autouse=True)
def enable_x64() -> None:
    import jax

    jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='session', autouse=True)
def compilation_cache() -> None:
    """Persist XLA executables across runs, so a re-run pays for tracing but not compilation.

    Most of the suite's wall time is XLA compilation of small kernels, and the same kernels come
    back run after run. JAX keys each entry on the HLO and on the jaxlib/backend version, so a
    stale entry is a miss, not a wrong answer. The defaults skip anything that compiles in under a
    second or is under a few kilobytes, which is nearly every kernel here, hence the thresholds.

    Set ``FURAX_TEST_NO_COMPILATION_CACHE`` to run against cold compilation instead.
    """
    import jax

    if os.environ.get('FURAX_TEST_NO_COMPILATION_CACHE'):
        return
    cache_dir = os.environ.get('FURAX_TEST_COMPILATION_CACHE_DIR')
    jax.config.update(
        'jax_compilation_cache_dir',
        cache_dir or str(Path(__file__).parents[1] / '.pytest_cache' / 'jax'),
    )
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)


@pytest.fixture(scope='session', autouse=True)
def silence_furax_logger() -> None:
    logging.getLogger('furax-mapmaking').setLevel(logging.WARNING)
