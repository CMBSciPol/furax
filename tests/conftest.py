import logging
import os

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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``distributed``-marked tests unless several devices are available."""
    import jax

    if jax.device_count() >= 2:
        return
    skip = pytest.mark.skip(reason='distributed test; run with `pytest -m distributed`')
    for item in items:
        if item.get_closest_marker('distributed'):
            item.add_marker(skip)


@pytest.fixture(scope='session', autouse=True)
def enable_x64() -> None:
    import jax

    jax.config.update('jax_enable_x64', True)


@pytest.fixture(scope='session', autouse=True)
def silence_furax_logger() -> None:
    logging.getLogger('furax-mapmaking').setLevel(logging.WARNING)
