import subprocess
import sys

# Imports every module of the packages, and exits with the first one that starts the JAX backend
_SCRIPT = """
import importlib
import pkgutil

from jax._src import xla_bridge

for name in ('furax', 'so_mapmaking'):
    package = importlib.import_module(name)
    modules = pkgutil.walk_packages(package.__path__, f'{name}.')
    for module in [name, *(module.name for module in modules)]:
        importlib.import_module(module)
        if xla_bridge.backends_are_initialized():
            raise SystemExit(f'importing {module} starts the JAX backend')
"""


def test_import_does_not_start_jax_backend() -> None:
    """Importing furax leaves the JAX backend to the caller to start.

    Before it starts, JAX can still be configured, e.g. by `jax.distributed.initialize`, and
    the process forked without risk of a deadlock. The test session has started it already, so
    the imports run in a fresh interpreter.
    """
    command = [sys.executable, '-c', _SCRIPT]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
