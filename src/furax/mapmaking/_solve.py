from dataclasses import dataclass, replace
from logging import Logger
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils as mhu
from jaxtyping import Array, PyTree

import furax.tree
from furax import AbstractLinearOperator
from furax.linalg import CGResult, LowRankTerms, cg, lanczos_tr
from furax.linalg._lanczos import LanczosWhich

from ._checkpoint import CHECKPOINT_FILES, SolveCheckpoint
from .config import SolverConfig, TwoLevelConfig
from .layout import SlotPlacement
from .preconditioner import make_two_level_preconditioner


class SystemSolution(NamedTuple):
    """The solution of a [`MapMakingSystem`][furax.mapmaking._solve.MapMakingSystem]."""

    x: PyTree[Array]
    """The estimate of the system unknowns."""
    injected_sky: PyTree[Array] | None
    """The estimate of the injected sky, if solved for."""
    solver_stats: dict[str, Any]
    """The solver statistics."""


@dataclass(frozen=True)
class MapMakingSystem:
    """The normal system handed to the solver."""

    A: AbstractLinearOperator
    """The normal operator, acting on the unknowns."""
    b: PyTree
    """The right-hand side, in the same unknowns."""
    preconditioner: AbstractLinearOperator
    """Approximate inverse of `A`."""
    has_amplitudes: bool
    """Whether the unknowns include explicit-template amplitudes."""
    injected_b: PyTree | None = None
    """The right-hand side for the injected simulated sky, if any."""

    def solve(
        self,
        config: SolverConfig,
        placement: SlotPlacement,
        logger: Logger,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
    ) -> SystemSolution:
        """Solve the system, and the injected sky's if it has one, with the conjugate gradient.

        Args:
            config: The solver options.
            placement: The placement of the data on the devices.
            logger: Where to report progress.
            checkpoint_dir: Directory to save the iterates in every `config.checkpoint_every`
                steps. `None` for no checkpoints.
            resume: Start from the checkpoints in `checkpoint_dir`, if any.
        """
        checkpoints = [
            None if checkpoint_dir is None else SolveCheckpoint(checkpoint_dir / file)
            for file in CHECKPOINT_FILES
        ]
        converged: list[SolveCheckpoint] = []

        result, stats = _solve(
            self, self.b, 'main', checkpoints[0], config, placement, logger, resume
        )
        if checkpoints[0] is not None and result.converged:
            converged.append(checkpoints[0])
        injected = None
        if self.injected_b is not None:
            # the same normal system, with the simulated sky as the right-hand side
            response, stats['injected'] = _solve(
                self,
                self.injected_b,
                'injected',
                checkpoints[1],
                config,
                placement,
                logger,
                resume,
            )
            if checkpoints[1] is not None and response.converged:
                converged.append(checkpoints[1])
            injected = response.solution if not self.has_amplitudes else response.solution[0]

        # A converged solve keeps its checkpoint until the other is done too, so that a run killed
        # in between does not redo it.
        if jax.process_index() == 0:
            for checkpoint in converged:
                checkpoint.remove()
        return SystemSolution(result.solution, injected, stats)

    def deflate(self, config: TwoLevelConfig, logger: Logger) -> 'MapMakingSystem':
        """The system with a two-level preconditioner deflating some modes of the current one.

        Args:
            config: The options of the eigenpairs to deflate.
            logger: Where to report the eigenvalues found.
        """
        preconditioned = config.spectrum == 'preconditioned'
        # The spectrum is positive, so its smallest and largest eigenvalues are its algebraic ends.
        lanczos_which: dict[str, LanczosWhich] = {'small': 'SA', 'large': 'LA', 'both': 'BE'}
        result = lanczos_tr(
            self.A,
            key=jax.random.key(config.seed),
            k=config.rank,
            m=config.krylov_size,
            which=lanczos_which[config.which],
            max_restarts=config.max_restarts,
            tol=config.tol,
            preconditioner=self.preconditioner if preconditioned else None,
        )
        theta = result.eigenvalues
        logger.info(
            f'Deflating {config.rank} eigenpairs ({config.which}) of the '
            f'{config.spectrum} spectrum, eigenvalues in '
            f'[{float(theta[0]):.6e}, {float(theta[-1]):.6e}], '
            f'max residual bound {float(jnp.max(result.residual_norms)):.3e}'
        )
        M = make_two_level_preconditioner(
            self.preconditioner,
            LowRankTerms(theta, result.eigenvectors),
            preconditioned=preconditioned,
        )
        return replace(self, preconditioner=M)


def _solve(
    system: MapMakingSystem,
    b: PyTree[Array],
    name: str,
    checkpoint: SolveCheckpoint | None,
    config: SolverConfig,
    placement: SlotPlacement,
    logger: Logger,
    resume: bool,
) -> tuple[CGResult, dict[str, Any]]:
    """Solve the normal system for the right-hand side `b`, checkpointing if requested."""
    options = config.options
    rank = jax.process_index()

    def log_iteration(step: Array, r_norm: Array) -> None:
        if rank == 0:  # log from rank 0 only
            logger.info(f'CG step={int(step)} residual={float(r_norm):.6e}')

    x0, previous_steps = None, 0
    if resume and checkpoint is not None:
        x0, previous_steps = _resume(checkpoint, b, name, logger)
    # max_steps bounds the steps over every run that resumes the solve
    cg_options: dict[str, Any] = dict(options)
    cg_options['max_steps'] = max(options['max_steps'] - previous_steps, 0)
    if checkpoint is not None and config.checkpoint_every:
        cg_options |= {
            'checkpoint_every': config.checkpoint_every,
            'checkpoint_callback': checkpoint.save,
        }
    result = cg(
        system.A,
        b,
        x0,
        preconditioner=system.preconditioner,
        iteration_callback=log_iteration if config.verbose else None,
        **cg_options,
    )
    jax.effects_barrier()  # the last checkpoint is written
    stats = _solver_stats(result, options, previous_steps)
    # Residual history is incomplete for a resumed solve
    b_norm = float(furax.tree.norm(b))
    _log_solve(name, stats, b_norm, logger)
    if checkpoint is not None and config.checkpoint_every:
        # The last iterate, to resume with more steps
        x = placement.gather(result.solution)
        if rank == 0:
            checkpoint.save(result.num_steps, x)
    return result, stats


def _resume(
    checkpoint: SolveCheckpoint, b: PyTree[Array], name: str, logger: Logger
) -> tuple[PyTree[Array] | None, int]:
    """The saved iterate and its number of steps, or `(None, 0)` if there is nothing to resume."""
    saved = checkpoint.load(b)
    if saved is None and checkpoint.path.exists():
        logger.warning(f'ignoring {checkpoint.path} (incompatible structure)')
    # Every process must start from the same iterate, or the collectives diverge
    steps = np.array(-1 if saved is None else saved[1])
    if jax.process_count() > 1:
        per_process = mhu.process_allgather(steps).tolist()
        if len(set(per_process)) > 1:
            logger.warning(
                f'inconsistent number of steps in the {name} checkpoint across processes '
                f'({per_process}), starting from zero'
            )
            saved = None
    if saved is None:
        return None, 0
    x0, previous_steps = saved
    checkpoint.resume(previous_steps)
    logger.info(f'resuming the {name} solve after step {previous_steps}')
    return x0, previous_steps


def _log_solve(name: str, stats: dict[str, Any], b_norm: float, logger: Logger) -> None:
    """Log the outcome of a CG solve, as a warning if it did not reach its tolerance.

    The residual is relative to `b_norm`, the norm of the right-hand side: the initial residual of
    a resumed solve is not.
    """
    options = stats['options']
    residual = stats['residuals'][-1]
    relative = residual / b_norm if b_norm else float('nan')
    outcome = f'{stats["num_steps"]} steps, relative residual {relative:.2e}'
    if stats['converged']:
        logger.info(f'{name} solve converged ({outcome})')
    elif options['rtol'] == options['atol'] == 0:
        # no tolerance: the solve runs max_steps steps by design
        logger.info(f'{name} solve finished ({outcome})')
    else:
        logger.warning(
            f'{name} solve did not converge ({outcome}, '
            f'rtol={options["rtol"]}, atol={options["atol"]})'
        )


def _solver_stats(result: CGResult, options: dict[str, Any], previous_steps: int) -> dict[str, Any]:
    """The statistics of a CG solve: its convergence, residuals and options, as JSON values.

    `num_steps` counts the steps of every run that resumed the solve, `previous_steps` of them
    before this one, while `residuals` holds the residual norm of this run only, from step
    `previous_steps`.
    """
    n_steps = int(result.num_steps)
    # residuals[0] is the initial residual, residuals[i + 1] the one after step i
    residuals = np.asarray(result.residuals)[: n_steps + 1]
    stats = {
        'num_steps': previous_steps + n_steps,
        'converged': bool(result.converged),
        'residuals': [float(r) for r in residuals],
        'options': options,
    }
    if previous_steps:
        stats['resumed_after'] = previous_steps
    return stats
