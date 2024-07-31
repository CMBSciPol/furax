import contextvars
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import jax.numpy as jnp
import lineax as lx
import yaml
from jax import Array
from jaxtyping import Int8


def default_solver_callback(solution: lx.Solution) -> Int8[Array, '']:
    num_steps = solution.stats['num_steps']
    ok = num_steps < solution.stats['max_steps']
    if ok:
        print(f'Converged in {num_steps} iterations')
    else:
        print(f'Did not converge in {num_steps} iterations')
    return jnp.array(0, dtype=jnp.int8)


@dataclass(frozen=True)
class ConfigState:
    solver: lx.AbstractLinearSolver = lx.CG(rtol=1e-6, atol=1e-6, max_steps=500)
    solver_throw: bool = False
    solver_options: dict[str, Any] = field(default_factory=dict)
    solver_callback: Callable[[lx.Solution], Int8[Array, '']] = default_solver_callback

    def tree_flatten(self):  # type: ignore[no-untyped-def]
        return (), asdict(self)

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # type: ignore[no-untyped-def]
        return cls(**aux_data)


_config_var = contextvars.ContextVar('config', default=ConfigState())


class Config:
    def __init__(self, **kwargs):
        config = _config_var.get()
        self._instance = replace(config, **kwargs)

    def __str__(self):
        return yaml.dump(self._instance, indent=4)

    def __enter__(self) -> ConfigState:
        self.token = _config_var.set(self._instance)
        return self._instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        _config_var.reset(self.token)

    @classmethod
    def instance(cls):
        return _config_var.get()
