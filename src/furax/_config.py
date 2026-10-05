import contextvars
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from types import TracebackType
from typing import TYPE_CHECKING, Any

import lineax as lx
import yaml

if TYPE_CHECKING:
    from furax.linalg import CGResult, CGSolver

__all__ = ['Config']


def default_solver_callback(solution: 'lx.Solution | CGResult') -> None:
    pass


def verbose_solver_callback(solution: 'lx.Solution | CGResult') -> None:
    if isinstance(solution, lx.Solution):
        num_steps = solution.stats['num_steps']
        max_steps = solution.stats['max_steps']
    else:
        num_steps = solution.num_steps
        max_steps = solution.residuals.shape[0]
    ok = num_steps < max_steps
    if ok:
        print(f'Converged in {num_steps} iterations')
    else:
        print(f'Did not converge in {num_steps} iterations')


def default_solver() -> lx.AbstractLinearSolver[Any]:
    return lx.CG(rtol=1e-6, atol=1e-6, max_steps=500)


@dataclass(frozen=True)
class ConfigState:
    solver: 'lx.AbstractLinearSolver[Any] | CGSolver' = field(default_factory=default_solver)
    solver_throw: bool = False
    solver_options: dict[str, Any] = field(default_factory=dict)
    solver_callback: 'Callable[[lx.Solution | CGResult], None]' = default_solver_callback

    def tree_flatten(self):
        return (), asdict(self)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


_config_var = contextvars.ContextVar('config', default=ConfigState())  # noqa: B039 (ConfigState is frozen)


class Config:
    def __init__(self, **kwargs: Any) -> None:
        config = _config_var.get()
        self._instance = replace(config, **kwargs)

    def __str__(self) -> str:
        return yaml.dump(self._instance, indent=4)

    def __enter__(self) -> ConfigState:
        self.token = _config_var.set(self._instance)
        return self._instance

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _config_var.reset(self.token)

    @classmethod
    def instance(cls) -> ConfigState:
        return _config_var.get()
